"""Verify the D3 config on an existing CARLA world, then restore that world.

This short stationary check produces calibration and RGB evidence, not a
dataset, a driving benchmark, or a claim of deterministic simulation physics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.config import CollectionConfig, ConfigError, load_collection_config
from carla_tools.session import CollectionSession, SessionError
from scripts.carla_rgb_probe import ProbeFailure, gpu_sample, matching_image, png_from_rgb, rgb_from_bgra

REPORT_NAME = "collection_report.json"
MOUNT_TOLERANCE = 0.001
TIME_TOLERANCE = 0.000001


class VerificationError(RuntimeError):
    """A stable public verification error code."""


def relative_matrix_error(vehicle_transform: Any, camera_transform: Any,
                          requested_matrix: list[list[float]]) -> float:
    """Compare rigid extrinsics by matrix composition, never Euler subtraction."""
    inverse = vehicle_transform.get_inverse_matrix()
    camera = camera_transform.get_matrix()
    if any(not math.isfinite(value) for matrix in (inverse, camera, requested_matrix)
           for row in matrix for value in row):
        raise VerificationError("invalid-camera-mount-matrix")
    actual = [[sum(inverse[row][k] * camera[k][col] for k in range(4))
               for col in range(4)] for row in range(4)]
    residuals = [abs(actual[row][col] - requested_matrix[row][col])
                 for row in range(4) for col in range(4)]
    if any(not math.isfinite(value) for value in residuals):
        raise VerificationError("invalid-camera-mount-matrix")
    return max(residuals)


def _tick_and_image(session: Any, cfg: CollectionConfig, images: queue.Queue,
                    overflow: threading.Event, previous: dict | None) -> tuple[Any, Any, dict]:
    before = time.perf_counter()
    frame = session.world.tick(cfg.connection.timeout_seconds)
    snapshot = session.world.get_snapshot()
    if snapshot.frame != frame:
        raise VerificationError("snapshot-frame-mismatch")
    image = matching_image(images, frame, cfg.connection.timeout_seconds)
    if overflow.is_set():
        raise VerificationError("camera-queue-overflow")
    if image.width != cfg.camera.width or image.height != cfg.camera.height:
        raise VerificationError("image-dimensions-mismatch")
    if not math.isfinite(image.fov) or not math.isclose(image.fov, cfg.camera.fov, abs_tol=TIME_TOLERANCE):
        raise VerificationError("image-fov-mismatch")
    timestamp = snapshot.timestamp
    if (not math.isfinite(timestamp.elapsed_seconds) or not math.isfinite(image.timestamp)
            or not math.isclose(image.timestamp, timestamp.elapsed_seconds, rel_tol=0, abs_tol=TIME_TOLERANCE)):
        raise VerificationError("sensor-snapshot-time-mismatch")
    if not math.isclose(timestamp.delta_seconds, cfg.world.fixed_delta_seconds,
                        rel_tol=0, abs_tol=TIME_TOLERANCE):
        raise VerificationError("world-timestep-mismatch")
    if previous is not None:
        if frame != previous["world_frame"] + 1:
            raise VerificationError("world-frame-not-consecutive")
        if not math.isclose(timestamp.elapsed_seconds - previous["simulation_seconds"],
                            cfg.world.fixed_delta_seconds, rel_tol=0, abs_tol=TIME_TOLERANCE):
            raise VerificationError("simulation-time-not-consecutive")
    return image, snapshot, {
        "world_frame": frame, "snapshot_frame": snapshot.frame, "sensor_frame": image.frame,
        "simulation_seconds": timestamp.elapsed_seconds, "sensor_seconds": image.timestamp,
        "delta_seconds": timestamp.delta_seconds,
        "tick_to_rgb_ms": (time.perf_counter() - before) * 1000,
    }


def _capture(session: Any, cfg: CollectionConfig, folder: Path, run_number: int, result: dict) -> dict:
    images: queue.Queue = queue.Queue(maxsize=cfg.verification.queue_capacity)
    overflow = threading.Event()

    def receive(image: Any) -> None:
        try:
            images.put_nowait(image)
        except queue.Full:
            overflow.set()

    session.camera.listen(receive)
    previous = None
    for _ in range(cfg.verification.warmup_frames):
        _, _, previous = _tick_and_image(session, cfg, images, overflow, previous)
    # Actor control/pose are client-cached and require a tick to be readable.
    # When warmup=0 this happens after the first measured tick below.
    if previous is not None:
        session.verify_applied()
    start = time.perf_counter()
    final_rgb = b""
    for index in range(cfg.verification.frames):
        image, snapshot, record = _tick_and_image(session, cfg, images, overflow, previous)
        if previous is None:
            session.verify_applied()
        previous = record
        actor_snapshot = snapshot.find(session.vehicle.id)
        if actor_snapshot is None:
            raise VerificationError("vehicle-snapshot-missing")
        mount_error = relative_matrix_error(actor_snapshot.get_transform(), image.transform,
                                             session.camera_mount_transform.get_matrix())
        if mount_error > MOUNT_TOLERANCE:
            raise VerificationError("sensor-vehicle-mount-mismatch")
        final_rgb, image_stats = rgb_from_bgra(bytes(image.raw_data), image.width, image.height)
        velocity = actor_snapshot.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        if not math.isfinite(speed):
            raise VerificationError("invalid-vehicle-speed")
        record.update({"index": index, "mount_matrix_max_abs_error": mount_error,
                       "vehicle_speed_mps": speed, "rgb": image_stats})
        result["frames"].append(record)
    duration = time.perf_counter() - start
    session.verify_applied()
    preview_name = f"collection_preview_run_{run_number:02d}.png"
    png = png_from_rgb(final_rgb, cfg.camera.width, cfg.camera.height)
    (folder / preview_name).write_bytes(png)
    latencies = [row["tick_to_rgb_ms"] for row in result["frames"]]
    result.update({
        "status": "PASS", "applied": session.manifest,
        "preview_png_written": True, "preview_png": preview_name,
        "preview_sha256": hashlib.sha256(png).hexdigest(),
        "metrics": {
            "warmup_frames": cfg.verification.warmup_frames,
            "captured_frames": len(result["frames"]),
            "simulation_duration_seconds": cfg.verification.frames * cfg.world.fixed_delta_seconds,
            "capture_wall_seconds": duration,
            "capture_fps": len(result["frames"]) / duration if duration else None,
            "tick_to_rgb_mean_ms": sum(latencies) / len(latencies),
            "tick_to_rgb_max_ms": max(latencies),
            "mount_matrix_max_abs_error": max(row["mount_matrix_max_abs_error"] for row in result["frames"]),
            "vehicle_speed_max_mps": max(row["vehicle_speed_mps"] for row in result["frames"]),
            "scope": "stationary-rgb-loop-including-validation-excluding-warmup-png-gpu",
        },
    })
    return result


def initialization_signature(manifest: dict) -> str:
    """Hash verified initialization values, excluding IDs, time and settling pose."""
    if not manifest.get("applied_verified"):
        raise VerificationError("applied-configuration-unverified")
    vehicle, camera = manifest["vehicle"], manifest["camera"]
    if not vehicle.get("control_verified") or not camera.get("mount_verified"):
        raise VerificationError("applied-calibration-unverified")
    values = {key: manifest[key] for key in ("carla_version", "map", "versions", "world_settings", "weather")}
    values["vehicle"] = {key: vehicle[key] for key in (
        "blueprint", "attributes", "spawn_point_index", "spawn_point_count", "spawn_transform", "control_applied")}
    values["camera"] = {key: camera[key] for key in (
        "blueprint", "attributes", "attachment_type", "mount_requested", "requested_mount_matrix", "intrinsics")}
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_verification(cfg: CollectionConfig, output_dir: Path, runs: int = 2,
                     carla_module: Any = None, session_factory: Any = CollectionSession) -> dict:
    """PASS requires every capture AND every session cleanup to succeed."""
    report: dict[str, Any] = {
        "schema_version": 1, "status": "FAIL", "scope": "d3-collection-configuration-and-calibration",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "configuration": cfg.to_dict(), "configuration_sha256": cfg.sha256,
        "camera_intrinsics_pixels": cfg.camera.intrinsics(), "runs_requested": runs, "runs": [],
        "world_initialization": "reuse-current-town01-without-reload",
        "reproducibility_scope": "same-config-fixed-spawn-calibration-and-observed-initialization",
        "physics_determinism_verified": False,
        "gpu_memory_scope": "whole-device-snapshots-not-process-or-peak",
        "tolerances": {"timestamp_seconds": TIME_TOLERANCE, "mount_matrix_max_abs": MOUNT_TOLERANCE},
    }
    stage = "output-directory"
    session = None
    current = None
    try:
        if type(runs) is not int or not 1 <= runs <= 10:
            raise VerificationError("invalid-runs-count")
        output_dir.mkdir(parents=True, exist_ok=True)
        # Only these managed files are removed; no directory cleanup is used.
        for index in range(1, 11):
            (output_dir / f"collection_preview_run_{index:02d}.png").unlink(missing_ok=True)
        report["gpu_before"] = gpu_sample()
        for index in range(1, runs + 1):
            stage = "session-setup"
            session = session_factory(cfg, carla_module=carla_module)
            current = {"run": index, "status": "FAIL", "frames": [], "preview_png_written": False}
            report["runs"].append(current)
            with session:
                stage = "rgb-capture"
                _capture(session, cfg, output_dir, index, current)
                if session.manifest["config_sha256"] != cfg.sha256:
                    raise VerificationError("configuration-hash-mismatch")
                current["initialization_sha256"] = initialization_signature(session.manifest)
                stage = "cleanup"
            current["cleanup"] = session.cleanup_report
            if session.cleanup_report.get("status") != "PASS":
                raise VerificationError("session-cleanup-incomplete")
        report["gpu_after"] = gpu_sample()
        stage = "repeat-check"
        if len({run["initialization_sha256"] for run in report["runs"]}) != 1:
            raise VerificationError("initialization-repeat-mismatch")
        report["status"] = "PASS"
        report["configuration_repeats_verified"] = runs >= 2 and len(report["runs"]) == runs
    except Exception as error:
        if current is not None:
            if stage != "repeat-check":
                current["status"] = "FAIL"
            if session is not None:
                current["cleanup"] = session.cleanup_report
        if isinstance(error, SessionError):
            code = error.code
            stage = error.stage
        elif isinstance(error, (VerificationError, ProbeFailure)):
            code = str(error)
        else:
            code = "unexpected-error"
        report["error"] = {"stage": stage, "code": code, "type": type(error).__name__}
        if isinstance(error, SessionError) and error.error_type:
            report["error"]["cause_type"] = error.error_type
    report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return report


def write_report(path: Path, report: dict) -> None:
    """Replace the report atomically, with finite JSON and no partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "collection_d3.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "local" / "collection-d3")
    parser.add_argument("--runs", type=int, choices=range(1, 11), default=2)
    parser.add_argument("--validate-only", action="store_true", help="Validate JSON and calibration without CARLA or RPC.")
    args = parser.parse_args(argv)
    try:
        cfg = load_collection_config(args.config)
    except ConfigError:
        failure = {"status": "FAIL", "error": {"stage": "configuration", "code": "invalid-configuration"}}
        if not args.validate_only:
            try:
                write_report(args.output_dir / REPORT_NAME, failure)
            except (OSError, ValueError):
                failure["report_written"] = False
        print(json.dumps(failure))
        return 1
    if args.validate_only:
        print(json.dumps({"status": "CONFIG_VALIDATED", "simulator_verified": False,
                          "configuration_sha256": cfg.sha256,
                          "camera_intrinsics_pixels": cfg.camera.intrinsics()}, indent=2))
        return 0
    try:
        write_report(args.output_dir / REPORT_NAME, {
            "schema_version": 1, "status": "RUNNING", "configuration_sha256": cfg.sha256,
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        })
    except (OSError, ValueError):
        print(json.dumps({"status": "FAIL", "error": {"stage": "report-output", "code": "report-write-failed"}}))
        return 1
    report = run_verification(cfg, args.output_dir, args.runs)
    try:
        write_report(args.output_dir / REPORT_NAME, report)
    except (OSError, ValueError):
        print(json.dumps({"status": "FAIL", "error": {"stage": "report-output", "code": "report-write-failed"}}))
        return 1
    print(json.dumps({"status": report["status"], "runs_completed": sum(
        run["status"] == "PASS" for run in report["runs"]),
        "configuration_sha256": cfg.sha256, "error": report.get("error")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
