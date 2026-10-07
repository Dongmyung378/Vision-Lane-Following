"""Persist a bounded, synchronized stationary RGB episode with real readbacks.

This D4 smoke capture creates no driving labels and never selects waypoints.
CARLA is imported by CollectionSession only when a real session is entered.
"""

from __future__ import annotations

import datetime as dt
import math
import queue
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from .config import ConfigError, collection_config_from_dict
from .episode import EpisodeError, EpisodeWriter, validate_episode
from .session import CollectionSession, SessionError, fields_match
from scripts.carla_rgb_probe import ProbeFailure, matching_image, png_from_rgb, rgb_from_bgra


TIME_TOLERANCE = 1e-6
MOUNT_TOLERANCE = 1e-3


class CaptureError(RuntimeError):
    """A stable public failure code, without raw simulator or path messages."""

    def __init__(self, code: str, stage: str = "capture"):
        super().__init__(code)
        self.code, self.stage = code, stage


class BoundedSensorQueue:
    """The sensor callback never blocks; any overflow invalidates the capture."""

    def __init__(self, capacity: int):
        if type(capacity) is not int or not 1 <= capacity <= 64:
            raise CaptureError("invalid-sensor-queue-capacity")
        self.images: queue.Queue = queue.Queue(maxsize=capacity)
        self.overflow = threading.Event()

    def receive(self, image: Any) -> None:
        try:
            self.images.put_nowait(image)
        except queue.Full:
            self.overflow.set()

    def matching(self, frame: int, timeout: float) -> Any:
        if self.overflow.is_set():
            raise CaptureError("camera-queue-overflow")
        image = matching_image(self.images, frame, timeout)
        if self.overflow.is_set():
            raise CaptureError("camera-queue-overflow")
        return image


def _finite_number(value: Any, code: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise CaptureError(code)
    return float(value)


def _vector(vector: Any) -> dict[str, float]:
    return {name: _finite_number(getattr(vector, name), "invalid-vehicle-kinematics") for name in ("x", "y", "z")}


def _matrix(matrix: Any) -> list[list[float]]:
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise CaptureError("invalid-transform-matrix")
    return [[_finite_number(value, "invalid-transform-matrix") for value in row] for row in matrix]


def _transform(transform: Any) -> dict[str, Any]:
    return {
        "location": {name: _finite_number(getattr(transform.location, name), "invalid-transform") for name in ("x", "y", "z")},
        "rotation": {name: _finite_number(getattr(transform.rotation, name), "invalid-transform") for name in ("pitch", "yaw", "roll")},
        "matrix": _matrix(transform.get_matrix()),
    }


def _mount_matrix(cfg: Any) -> list[list[float]]:
    location, rotation = cfg.camera.mount.location, cfg.camera.mount.rotation
    pitch, yaw, roll = [math.radians(getattr(rotation, name)) for name in ("pitch", "yaw", "roll")]
    cp, sp, cy, sy, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw), math.cos(roll), math.sin(roll)
    return [[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, location.x],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, location.y],
            [sp, -cp * sr, cp * cr, location.z], [0.0, 0.0, 0.0, 1.0]]


def _calibration(cfg: Any, matrix: Any) -> dict[str, Any]:
    intrinsics = cfg.camera.intrinsics()
    return {"intrinsics": intrinsics,
            "K": [[intrinsics["fx"], 0.0, intrinsics["cx"]],
                  [0.0, intrinsics["fy"], intrinsics["cy"]], [0.0, 0.0, 1.0]],
            "width": cfg.camera.width, "height": cfg.camera.height,
            "distortion": {"lens_k": cfg.camera.lens_k, "lens_kcube": cfg.camera.lens_kcube},
            "requested_mount_matrix": _matrix(matrix),
            "coordinate_convention": "carla-x-forward-y-right-z-up", "rotation_units": "degrees"}


def _tick(session: Any, cfg: Any, images: BoundedSensorQueue, previous: dict | None) -> tuple[Any, Any, dict]:
    before = time.perf_counter()
    frame = session.world.tick(cfg.connection.timeout_seconds)
    if type(frame) is not int or frame < 0:
        raise CaptureError("invalid-world-frame")
    snapshot = session.world.get_snapshot()
    if type(snapshot.frame) is not int or snapshot.frame != frame:
        raise CaptureError("snapshot-frame-mismatch")
    image = images.matching(frame, cfg.connection.timeout_seconds)
    if type(image.frame) is not int or image.frame != frame:
        raise CaptureError("sensor-frame-mismatch")
    if image.width != cfg.camera.width or image.height != cfg.camera.height:
        raise CaptureError("image-dimensions-mismatch")
    fov = _finite_number(image.fov, "invalid-image-fov")
    if not math.isclose(fov, cfg.camera.fov, rel_tol=0, abs_tol=TIME_TOLERANCE):
        raise CaptureError("image-fov-mismatch")
    timestamp = _finite_number(snapshot.timestamp.elapsed_seconds, "invalid-snapshot-timestamp")
    sensor_timestamp = _finite_number(image.timestamp, "invalid-sensor-timestamp")
    delta = _finite_number(snapshot.timestamp.delta_seconds, "invalid-world-timestep")
    if timestamp < 0 or sensor_timestamp < 0 or not math.isclose(sensor_timestamp, timestamp, rel_tol=0, abs_tol=TIME_TOLERANCE):
        raise CaptureError("sensor-snapshot-time-mismatch")
    if not math.isclose(delta, cfg.world.fixed_delta_seconds, rel_tol=0, abs_tol=TIME_TOLERANCE):
        raise CaptureError("world-timestep-mismatch")
    if previous is not None:
        if frame != previous["frame"] + 1:
            raise CaptureError("world-frame-not-consecutive")
        if not math.isclose(timestamp - previous["timestamp"], cfg.world.fixed_delta_seconds, rel_tol=0, abs_tol=TIME_TOLERANCE):
            raise CaptureError("simulation-time-not-consecutive")
    return image, snapshot, {
        "schema_version": 1, "frame": frame, "world_frame": frame,
        "snapshot_frame": snapshot.frame, "sensor_frame": image.frame,
        "timestamp": timestamp, "snapshot_timestamp": timestamp,
        "sensor_timestamp": sensor_timestamp, "delta_seconds": delta,
        "tick_to_rgb_ms": (time.perf_counter() - before) * 1000,
    }


def _record(session: Any, cfg: Any, image: Any, snapshot: Any, row: dict, index: int) -> tuple[dict, bytes]:
    actor = snapshot.find(session.vehicle.id)
    if actor is None:
        raise CaptureError("vehicle-snapshot-missing")
    vehicle_transform = actor.get_transform()
    inverse = _matrix(vehicle_transform.get_inverse_matrix())
    camera_transform = _transform(image.transform)
    requested = _matrix(session.camera_mount_transform.get_matrix())
    camera_matrix = camera_transform["matrix"]
    relative = [[sum(inverse[r][k] * camera_matrix[k][c] for k in range(4)) for c in range(4)] for r in range(4)]
    residuals = [abs(relative[r][c] - requested[r][c]) for r in range(4) for c in range(4)]
    if any(not math.isfinite(value) for row in relative for value in row) or any(
            not math.isfinite(value) for value in residuals):
        raise CaptureError("invalid-camera-mount-matrix")
    mount_error = max(residuals)
    if mount_error > MOUNT_TOLERANCE:
        raise CaptureError("sensor-vehicle-mount-mismatch")
    # CARLA get_control() is a last-tick client-cache read. No live transform RPC
    # is used: vehicle pose/kinematics belong to this immutable snapshot.
    cached_control = session.vehicle.get_control()
    control = {name: getattr(cached_control, name) for name in cfg.vehicle.control.to_dict()}
    if not fields_match(cfg.vehicle.control.to_dict(), control, exact_keys=True):
        raise CaptureError("vehicle-control-readback-mismatch")
    rgb, stats = rgb_from_bgra(bytes(image.raw_data), image.width, image.height)
    row.update({
        "index": index, "image": {"width": image.width, "height": image.height},
        "vehicle": {"actor_id": session.vehicle.id, "transform": _transform(vehicle_transform),
                    "velocity": _vector(actor.get_velocity()), "angular_velocity": _vector(actor.get_angular_velocity()),
                    "acceleration": _vector(actor.get_acceleration()), "control": control,
                    "control_source": "client-cache-at-world-frame"},
        "camera": {"actor_id": session.camera.id, "transform": camera_transform},
        "mount_matrix_max_abs_error": mount_error, "rgb": stats,
        "label_state": "NOT_GENERATED", "label": None,
    })
    return row, png_from_rgb(rgb, image.width, image.height)


def capture_episode(cfg: Any, destination: str | Path, metadata: Mapping[str, Any],
                    carla_module: Any = None, session_factory: Any = CollectionSession) -> dict[str, Any]:
    """PASS requires full capture, restored session, and fresh on-disk validation.

    An existing destination is refused. A failed or interrupted owned episode
    remains FAILED on disk and cannot be used as a completed dataset episode.
    """
    report: dict[str, Any] = {"schema_version": 1, "status": "FAIL", "scope": "stationary-D4-smoke",
                              "frames_written": 0, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    stage = "configuration"
    writer = session = None
    captured = False
    error_record = None
    started = time.perf_counter()
    try:
        cfg = collection_config_from_dict(cfg.to_dict())
        values = dict(metadata)
        if values.get("town") != cfg.map or values.get("control_mode") != "stationary_D4_smoke" or values.get("labels_state") != "not-generated-D6":
            raise CaptureError("invalid-stationary-episode-metadata", "metadata")
        values.update(schema_version=1, expected_frames=cfg.verification.frames,
                      collection_config=cfg.to_dict(), configuration_sha256=cfg.sha256,
                      camera_calibration=_calibration(cfg, _mount_matrix(cfg)),
                      applied_manifest={"applied_verified": False})
        stage = "output-directory"
        destination = Path(destination)
        if destination.exists() or destination.is_symlink():
            raise CaptureError("destination-already-exists", stage)
        writer = EpisodeWriter(destination, values)
        stage = "session-setup"
        session = session_factory(cfg, carla_module=carla_module)
        with session:
            writer.update_metadata({"applied_manifest": session.manifest,
                                    "camera_calibration": _calibration(cfg, session.camera_mount_transform.get_matrix())})
            stage = "rgb-capture"
            images = BoundedSensorQueue(cfg.verification.queue_capacity)
            session.camera.listen(images.receive)
            previous = None
            for _ in range(cfg.verification.warmup_frames):
                _, _, previous = _tick(session, cfg, images, previous)
            if previous is not None:
                session.verify_applied()
            capture_started = time.perf_counter()
            for index in range(cfg.verification.frames):
                image, snapshot, row = _tick(session, cfg, images, previous)
                if previous is None:
                    session.verify_applied()
                row, png = _record(session, cfg, image, snapshot, row, index)
                writer.append(row, png)
                report["frames_written"] += 1
                previous = row
            if images.overflow.is_set():
                raise CaptureError("camera-queue-overflow")
            session.verify_applied()
            if session.manifest.get("config_sha256") != cfg.sha256 or not session.manifest.get("applied_verified"):
                raise CaptureError("applied-configuration-unverified")
            writer.update_metadata({"applied_manifest": session.manifest})
            report["metrics"] = {"captured_frames": report["frames_written"],
                                 "capture_wall_seconds": time.perf_counter() - capture_started,
                                 "simulation_advance_seconds": cfg.verification.frames * cfg.world.fixed_delta_seconds,
                                 "timing_scope": "capture-loop-including-validation-png-and-disk-write-excluding-warmup"}
            captured = True
            stage = "cleanup"
        report["cleanup"] = session.cleanup_report
        if session.cleanup_report.get("status") != "PASS":
            raise CaptureError("session-cleanup-incomplete", "cleanup")
        stage = "episode-finalize"
        writer.finalize(True, session.cleanup_report)
        stage = "disk-validation"
        validation = validate_episode(Path(destination))
        report["validation"] = validation
        if validation.get("status") != "PASS":
            raise CaptureError("episode-disk-validation-failed", "disk-validation")
        report["status"] = "PASS"
    except BaseException as error:
        if isinstance(error, (SessionError, CaptureError)):
            code, failure_stage = error.code, error.stage
        elif isinstance(error, EpisodeError):
            code, failure_stage = error.code, stage
        elif isinstance(error, ConfigError):
            code, failure_stage = "invalid-configuration", stage
        elif isinstance(error, ProbeFailure):
            code, failure_stage = str(error), stage
        elif isinstance(error, (KeyboardInterrupt, SystemExit)):
            code, failure_stage = "capture-interrupted", stage
        else:
            code, failure_stage = "capture-operation-failed", stage
        error_record = {"code": code, "stage": failure_stage, "exception_type": type(error).__name__}
        report["error"] = error_record
        report["capture_succeeded"] = captured
        if session is not None:
            report["cleanup"] = session.cleanup_report
        if writer is not None:
            try:
                writer.finalize(False, report.get("cleanup", {"status": "NOT_RUN"}), error_record)
            except Exception as final_error:
                report["finalization_error"] = {"code": "failure-finalization-failed", "exception_type": type(final_error).__name__}
    report["elapsed_seconds"] = time.perf_counter() - started
    report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return report
