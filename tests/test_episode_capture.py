"""Capture timing/ownership/failure contracts without a CARLA installation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from carla_tools.capture import BoundedSensorQueue, CaptureError, _mount_matrix, _record, capture_episode
from carla_tools.config import CollectionConfig
from carla_tools.session import SessionError


def cleanup(status="PASS"):
    return {"status": status, "sensor_stopped": True, "settings_restored": True,
            "weather_restored": True, "owned_actors_absent": status == "PASS"}


def metadata():
    return {"dataset_version": "d4-smoke-v1", "episode_id": "episode-001",
            "route_id": "town01-fixed-0", "scene_group_id": "town01-scene-0", "town": "Town01",
            "perturbation": {"lateral_m": 0, "yaw_deg": 0, "provenance": "stationary-no-perturbation"},
            "control_mode": "stationary_D4_smoke", "labels_state": "not-generated-D6",
            "route_contract": {"fixture": "metadata-passthrough-only"}}


class Transform:
    def __init__(self, matrix=None, location=None, rotation=None):
        self.matrix = matrix or [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        self.location = location or SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.rotation = rotation or SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0)

    def get_matrix(self):
        return copy.deepcopy(self.matrix)

    def get_inverse_matrix(self):
        matrix = self.get_matrix()
        result = [[matrix[c][r] for c in range(3)] + [0.0] for r in range(3)] + [[0.0, 0.0, 0.0, 1.0]]
        for row in range(3):
            result[row][3] = -sum(result[row][column] * matrix[column][3] for column in range(3))
        return result


class FixtureWriter:
    """Only a capture fixture; production writer integrity has separate tests."""

    def __init__(self, destination, values):
        self.destination = destination
        self.destination.mkdir(parents=True, exist_ok=False)
        (self.destination / "images").mkdir()
        self.metadata = dict(values, status="RUNNING", frame_count=0)
        self.rows = []
        self._write()

    def _write(self):
        (self.destination / "episode.json").write_text(json.dumps(self.metadata, allow_nan=False), encoding="utf-8")

    def update_metadata(self, values):
        self.metadata.update(copy.deepcopy(values))
        self._write()

    def append(self, row, png):
        row = copy.deepcopy(row)
        image_path = f"images/{row['frame']:08d}.png"
        (self.destination / image_path).write_bytes(png)
        row["image"].update(path=image_path, sha256=hashlib.sha256(png).hexdigest())
        with (self.destination / "frames.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        self.rows.append(row)
        self.metadata["frame_count"] = len(self.rows)
        return row

    def finalize(self, capture_succeeded, restoration, error=None):
        self.metadata["status"] = "COMPLETE" if capture_succeeded and restoration["status"] == "PASS" else "FAILED"
        self.metadata["cleanup"] = restoration
        if error is not None:
            self.metadata["error"] = error
        self._write()
        return copy.deepcopy(self.metadata)


def fixture_validate(destination):
    values = json.loads((destination / "episode.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (destination / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    valid = values["status"] == "COMPLETE" and len(rows) == values["expected_frames"]
    for row in rows:
        png = (destination / row["image"]["path"]).read_bytes()
        valid &= png.startswith(b"\x89PNG\r\n\x1a\n") and hashlib.sha256(png).hexdigest() == row["image"]["sha256"]
    return {"status": "PASS" if valid else "FAIL", "frame_count": len(rows), "errors": []}


class Session:
    def __init__(self, cfg, carla_module=None):
        self.cfg = cfg
        self.fault = carla_module or ""
        mount = cfg.camera.mount
        self.camera_mount_transform = Transform(_mount_matrix(cfg), SimpleNamespace(**mount.location.to_dict()),
                                                SimpleNamespace(**mount.rotation.to_dict()))
        self.camera = SimpleNamespace(id=11, listen=self.listen)
        self.vehicle = SimpleNamespace(id=10, get_control=lambda: SimpleNamespace(**cfg.vehicle.control.to_dict()),
                                       get_transform=Mock(side_effect=AssertionError("Live transform access is forbidden")))
        self.world = SimpleNamespace(tick=self.tick, get_snapshot=lambda: self.snapshot)
        self.frame, self.ticks, self.verify_calls = 100, 0, 0
        self.callback = None
        self.manifest = {"config_sha256": cfg.sha256, "applied_verified": False}
        self.cleanup_report = {"status": "NOT_RUN"}

    def __enter__(self):
        if self.fault == "setup":
            self.cleanup_report = cleanup()
            raise SessionError("fixture-setup-refused", "world-preflight")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup_report = cleanup("FAIL" if self.fault == "cleanup" else "PASS")
        if self.fault == "cleanup" and exc_type is None:
            raise SessionError("session-cleanup-failed", "cleanup")
        return False

    def listen(self, callback):
        self.callback = callback

    def verify_applied(self):
        self.verify_calls += 1
        if self.fault == "final-verification" and self.verify_calls == 2:
            raise SessionError("fixture-configuration-changed", "applied-readback")
        self.manifest["applied_verified"] = True

    def tick(self, timeout):
        self.ticks += 1
        bad = self.ticks == 3
        if self.fault == "interrupt" and bad:
            raise KeyboardInterrupt()
        if self.fault == "raw-error" and bad:
            raise RuntimeError("private-person/secret-token")
        self.frame += 2 if self.fault == "frame-gap" and bad else 1
        timestamp = self.frame * self.cfg.world.fixed_delta_seconds
        actor = SimpleNamespace(get_transform=lambda: Transform(),
                                get_velocity=lambda: SimpleNamespace(x=math.nan if self.fault == "kinematics" and bad else 0.0, y=0.0, z=0.0),
                                get_angular_velocity=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
                                get_acceleration=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0))
        self.snapshot = SimpleNamespace(frame=self.frame + (1 if self.fault == "snapshot-frame" and bad else 0),
                                        timestamp=SimpleNamespace(elapsed_seconds=timestamp,
                                                                  delta_seconds=0.1 if self.fault == "delta" and bad else self.cfg.world.fixed_delta_seconds),
                                        find=lambda actor_id: None if self.fault == "vehicle-missing" and bad else actor)
        transform = self.camera_mount_transform
        if self.fault == "mount" and bad:
            transform = Transform()
        count = self.cfg.camera.width * self.cfg.camera.height
        raw = (b"\x10\x20\x30\xff" + b"\x11\x21\x31\xff") * (count // 2)
        if self.fault == "constant-rgb" and bad:
            raw = b"\x00\x00\x00\xff" * count
        image = SimpleNamespace(frame=self.frame + (1 if self.fault == "sensor-ahead" and bad else 0),
                                timestamp=math.nan if self.fault == "nonfinite-time" and bad else timestamp + (0.1 if self.fault == "sensor-time" and bad else 0),
                                width=self.cfg.camera.width + (1 if self.fault == "dimensions" and bad else 0),
                                height=self.cfg.camera.height,
                                fov=80.0 if self.fault == "fov" and bad else self.cfg.camera.fov,
                                transform=transform, raw_data=raw)
        if self.fault != "timeout":
            self.callback(image)
        if self.fault == "overflow":
            self.callback(image)
        return self.frame


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        base = CollectionConfig()
        self.cfg = replace(base, camera=replace(base.camera, width=16, height=16),
                           verification=replace(base.verification, warmup_frames=1, frames=3, queue_capacity=1),
                           connection=replace(base.connection, timeout_seconds=0.01))
        self.writer_patch = patch("carla_tools.capture.EpisodeWriter", FixtureWriter)
        self.validator_patch = patch("carla_tools.capture.validate_episode", side_effect=fixture_validate)
        self.writer_patch.start()
        self.validator = self.validator_patch.start()
        self.addCleanup(self.writer_patch.stop)
        self.addCleanup(self.validator_patch.stop)
        self.addCleanup(self.temporary.cleanup)

    def run_capture(self, fault="", cfg=None, name="episode"):
        return capture_episode(cfg or self.cfg, self.root / name, metadata(), fault, Session)

    def stored(self, name="episode"):
        return json.loads((self.root / name / "episode.json").read_text(encoding="utf-8"))

    def test_success_persists_all_rgb_frames_and_same_frame_state(self):
        instances = []

        def factory(cfg, carla_module=None):
            session = Session(cfg, carla_module)
            instances.append(session)
            return session

        report = capture_episode(self.cfg, self.root / "episode", metadata(), session_factory=factory)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["frames_written"], 3)
        self.assertEqual(self.stored()["status"], "COMPLETE")
        self.assertEqual(self.stored()["route_contract"], metadata()["route_contract"])
        self.assertTrue(self.stored()["applied_manifest"]["applied_verified"])
        self.validator.assert_called_once()
        rows = [json.loads(line) for line in (self.root / "episode" / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["frame"] for row in rows], [102, 103, 104])
        for row in rows:
            self.assertEqual({row[key] for key in ("frame", "world_frame", "snapshot_frame", "sensor_frame")}, {row["frame"]})
            self.assertEqual(row["label_state"], "NOT_GENERATED")
            self.assertIsNone(row["label"])
            self.assertEqual(row["vehicle"]["velocity"], {"x": 0.0, "y": 0.0, "z": 0.0})
            self.assertEqual(row["vehicle"]["control"], self.cfg.vehicle.control.to_dict())
            self.assertLess(row["mount_matrix_max_abs_error"], 1e-3)
        self.assertEqual(instances[0].ticks, 4)
        instances[0].vehicle.get_transform.assert_not_called()

    def test_existing_output_is_refused_without_overwrite_or_session(self):
        folder = self.root / "episode"
        folder.mkdir()
        sentinel = folder / "keep.txt"
        sentinel.write_text("untouched", encoding="utf-8")
        factory = Mock(side_effect=AssertionError("No simulator connection allowed"))
        report = capture_episode(self.cfg, folder, metadata(), session_factory=factory)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"]["stage"], "output-directory")
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "untouched")
        self.assertEqual(list(folder.iterdir()), [sentinel])
        factory.assert_not_called()

    def test_preexisting_empty_destination_is_also_refused(self):
        folder = self.root / "episode"
        folder.mkdir()
        report = self.run_capture()
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"]["code"], "destination-already-exists")
        self.assertEqual(list(folder.iterdir()), [])

    def test_frame_time_rgb_geometry_and_kinematics_failures_leave_partial_episode_failed(self):
        faults = ("frame-gap", "snapshot-frame", "sensor-ahead", "sensor-time", "delta", "nonfinite-time",
                  "vehicle-missing", "mount", "constant-rgb", "kinematics", "dimensions", "fov", "raw-error")
        for fault in faults:
            with self.subTest(fault=fault):
                report = self.run_capture(fault, name=fault)
                self.assertEqual(report["status"], "FAIL")
                self.assertEqual(report["frames_written"], 1)
                self.assertEqual(self.stored(fault)["status"], "FAILED")
                self.assertEqual(report["cleanup"]["status"], "PASS")
                self.assertNotIn("secret-token", json.dumps(report))
        self.validator.assert_not_called()

    def test_cleanup_failure_never_marks_full_capture_complete(self):
        report = self.run_capture("cleanup")
        self.assertEqual(report["frames_written"], 3)
        self.assertTrue(report["capture_succeeded"])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(self.stored()["status"], "FAILED")
        self.validator.assert_not_called()

    def test_setup_failure_is_recorded_without_frames(self):
        report = self.run_capture("setup")
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"]["code"], "fixture-setup-refused")
        self.assertEqual(report["frames_written"], 0)
        self.assertEqual(self.stored()["status"], "FAILED")

    def test_interruption_records_failure_and_cleanup(self):
        report = self.run_capture("interrupt")
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"]["code"], "capture-interrupted")
        self.assertEqual(report["frames_written"], 1)
        self.assertEqual(report["cleanup"]["status"], "PASS")
        self.assertEqual(self.stored()["status"], "FAILED")

    def test_timeout_and_nonblocking_overflow_fail_before_any_frame_is_written(self):
        for fault, code in (("timeout", "rgb-frame-timeout"), ("overflow", "camera-queue-overflow")):
            with self.subTest(fault=fault):
                report = self.run_capture(fault, name=fault)
                self.assertEqual(report["error"]["code"], code)
                self.assertEqual(report["frames_written"], 0)
                self.assertEqual(self.stored(fault)["status"], "FAILED")

    def test_changed_applied_readback_at_end_is_failed_despite_full_frame_count(self):
        report = self.run_capture("final-verification")
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["frames_written"], 3)
        self.assertEqual(self.stored()["status"], "FAILED")
        self.validator.assert_not_called()

    def test_disk_validation_failure_downgrades_complete_episode(self):
        self.validator.side_effect = None
        self.validator.return_value = {"status": "FAIL", "errors": [{"code": "tampered-image"}]}
        report = self.run_capture()
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"]["code"], "episode-disk-validation-failed")
        self.assertEqual(self.stored()["status"], "FAILED")

    def test_frame_count_bound_is_enforced_before_output_or_simulator_access(self):
        cfg = replace(self.cfg, verification=replace(self.cfg.verification, frames=1001))
        report = self.run_capture(cfg=cfg)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"]["stage"], "configuration")
        self.assertFalse((self.root / "episode").exists())

    def test_queue_discards_old_startup_delivery_and_refuses_any_overflow(self):
        images = BoundedSensorQueue(2)
        images.receive(SimpleNamespace(frame=1))
        expected = SimpleNamespace(frame=2)
        images.receive(expected)
        self.assertIs(images.matching(2, 0.01), expected)
        images.receive(expected)
        images.receive(expected)
        images.receive(expected)
        with self.assertRaisesRegex(CaptureError, "camera-queue-overflow"):
            images.matching(2, 0.01)

    def test_real_writer_and_disk_validator_accept_capture_and_detect_image_tampering(self):
        self.writer_patch.stop()
        self.validator_patch.stop()
        from carla_tools.episode import validate_episode

        report = self.run_capture()
        self.assertEqual(report["status"], "PASS", report.get("error"))
        self.assertEqual(report["validation"]["frame_count"], 3)
        self.assertEqual(validate_episode(self.root / "episode")["status"], "PASS")
        image = next((self.root / "episode" / "images").glob("*.png"))
        image.write_bytes(image.read_bytes() + b"tamper")
        invalid = validate_episode(self.root / "episode")
        self.assertEqual(invalid["status"], "FAIL")
        self.assertIn("image-hash-mismatch", {item["code"] for item in invalid["errors"]})

    def test_finite_matrix_inputs_cannot_hide_nan_from_overflow_in_later_residuals(self):
        session = Session(self.cfg)
        large = 1e308
        vehicle_matrix = [[1.0, 0.0, 0.0, 0.0], [0.0, large, large, 0.0],
                          [0.0, -large, -large, 0.0], [0.0, 0.0, 0.0, 1.0]]
        camera_matrix = _mount_matrix(self.cfg)
        camera_matrix[1] = [large, large, large, large]
        camera_matrix[2] = [large, large, large, large]
        self.assertTrue(all(math.isfinite(value) for matrix in (vehicle_matrix, camera_matrix) for row in matrix for value in row))
        actor = SimpleNamespace(get_transform=lambda: Transform(vehicle_matrix))
        snapshot = SimpleNamespace(find=lambda actor_id: actor)
        image = SimpleNamespace(transform=Transform(camera_matrix))
        # The first residual is zero; later inf-inf terms are NaN. Python max()
        # alone can ignore those later NaNs and falsely return a finite zero.
        with self.assertRaisesRegex(CaptureError, "invalid-camera-mount-matrix"):
            _record(session, self.cfg, image, snapshot, {}, 0)


if __name__ == "__main__":
    unittest.main()
