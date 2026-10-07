"""Check D3 evidence boundaries, same-frame geometry, and failure cleanup."""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from carla_tools.config import CollectionConfig, collection_config_from_dict
from scripts import verify_collection as verify


IDENTITY = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]


class Transform:
    def __init__(self, matrix=IDENTITY, inverse=IDENTITY):
        self.matrix, self.inverse = matrix, inverse

    def get_matrix(self):
        return self.matrix

    def get_inverse_matrix(self):
        return self.inverse


def config():
    data = CollectionConfig().to_dict()
    data["camera"].update(width=16, height=16)
    data["verification"].update(warmup_frames=1, frames=2, queue_capacity=1)
    return collection_config_from_dict(data)


class FakeSession:
    def __init__(self, cfg, carla_module=None):
        self.cfg = cfg
        self.cleanup_report = {"status": "NOT_RUN"}
        self.camera_mount_transform = Transform()
        self.camera = SimpleNamespace(listen=self.listen)
        self.vehicle = SimpleNamespace(id=7)
        self.frame = 0
        self.cleanup_ok = True
        self.image_change = self.snapshot_change = lambda value: None
        self.burst = False
        self.world = SimpleNamespace(tick=self.tick, get_snapshot=self.get_snapshot)
        self.manifest = {
            "config_sha256": cfg.sha256, "applied_verified": True,
            "carla_version": "0.9.16", "map": "Town01", "versions": {"client": "0.9.16", "server": "0.9.16"},
            "world_settings": cfg.world.to_dict(), "weather": {"preset": "ClearNoon"},
            "vehicle": {"actor_id": 7, "blueprint": cfg.vehicle.blueprint, "attributes": dict(cfg.vehicle.attributes),
                        "spawn_point_index": 0, "spawn_point_count": 255, "spawn_transform": {},
                        "control_applied": cfg.vehicle.control.to_dict(), "control_verified": True},
            "camera": {"actor_id": 8, "blueprint": cfg.camera.blueprint, "attributes": {},
                       "attachment_type": "Rigid", "mount_requested": cfg.camera.mount.to_dict(),
                       "requested_mount_matrix": IDENTITY, "intrinsics": cfg.camera.intrinsics(), "mount_verified": True},
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup_report = {"status": "PASS" if self.cleanup_ok else "FAIL", "owned_actors_absent": self.cleanup_ok}

    def listen(self, callback):
        self.callback = callback

    def tick(self, timeout):
        self.frame += 1
        image = SimpleNamespace(frame=self.frame, timestamp=self.frame * .05,
                                width=self.cfg.camera.width, height=self.cfg.camera.height, fov=self.cfg.camera.fov,
                                transform=Transform(), raw_data=bytes([1, 2, 3, 255, 4, 5, 6, 255] * 128))
        self.image_change(image)
        self.callback(image)
        if self.burst:
            self.callback(image)
        return self.frame

    def get_snapshot(self):
        actor = SimpleNamespace(get_transform=lambda: Transform(),
                                get_velocity=lambda: SimpleNamespace(x=0., y=0., z=0.))
        snapshot = SimpleNamespace(frame=self.frame,
                                   timestamp=SimpleNamespace(elapsed_seconds=self.frame * .05, delta_seconds=.05),
                                   find=lambda actor_id: actor)
        self.snapshot_change(snapshot)
        return snapshot

    def verify_applied(self):
        return self.manifest


class GeometryTests(unittest.TestCase):
    def test_nonfinite_entries_after_first_matrix_cell_are_rejected(self):
        for invalid in (float("nan"), float("inf"), float("-inf")):
            for target in ("inverse", "camera", "requested"):
                with self.subTest(value=invalid, target=target):
                    matrix = copy.deepcopy(IDENTITY)
                    matrix[2][3] = invalid
                    inverse = matrix if target == "inverse" else IDENTITY
                    camera = matrix if target == "camera" else IDENTITY
                    requested = matrix if target == "requested" else IDENTITY
                    with self.assertRaisesRegex(verify.VerificationError, "invalid-camera-mount-matrix"):
                        verify.relative_matrix_error(Transform(inverse=inverse), Transform(camera), requested)

    def test_rotated_vehicle_frame_uses_matrix_composition(self):
        # A vehicle at (10,20) rotated +90 degrees; camera is 1m forward in that
        # vehicle, at (10,21), and follows its rotation. Relative mount is x=1.
        inverse = [[0., 1., 0., -20.], [-1., 0., 0., 10.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
        world_camera = [[0., -1., 0., 10.], [1., 0., 0., 21.], [0., 0., 1., 2.], [0., 0., 0., 1.]]
        mount = copy.deepcopy(IDENTITY)
        mount[0][3], mount[2][3] = 1., 2.
        self.assertEqual(verify.relative_matrix_error(Transform(inverse=inverse), Transform(world_camera), mount), 0.)

    def test_signatures_ignore_actor_ids_but_detect_actual_calibration_changes(self):
        first = FakeSession(config()).manifest
        other = copy.deepcopy(first)
        other["vehicle"]["actor_id"] = 900
        other["camera"]["actor_id"] = 901
        self.assertEqual(verify.initialization_signature(first), verify.initialization_signature(other))
        other["camera"]["intrinsics"]["fx"] += 1
        self.assertNotEqual(verify.initialization_signature(first), verify.initialization_signature(other))


class EvidenceTests(unittest.TestCase):
    def run_check(self, session=None, runs=1, factory=None):
        cfg = config()
        session = session or FakeSession(cfg)
        factory = factory or (lambda *args, **kwargs: session)
        with tempfile.TemporaryDirectory() as folder, patch.object(verify, "gpu_sample", return_value={"status": "unavailable"}):
            report = verify.run_verification(cfg, Path(folder), runs, session_factory=factory)
        return report, session

    def test_two_successful_runs_have_same_verified_initialization(self):
        report, _ = self.run_check(runs=2, factory=FakeSession)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["configuration_repeats_verified"])
        self.assertFalse(report["physics_determinism_verified"])
        for run in report["runs"]:
            self.assertEqual(len(run["frames"]), 2)
            self.assertEqual(run["cleanup"]["status"], "PASS")
            self.assertTrue(run["preview_png_written"])

    def test_unverified_applied_settings_cannot_pass(self):
        session = FakeSession(config())
        session.manifest["applied_verified"] = False
        report, _ = self.run_check(session)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"]["code"], "applied-configuration-unverified")

    def test_single_run_does_not_claim_repeat_verification(self):
        report, _ = self.run_check()
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["configuration_repeats_verified"])

    def test_rendered_rgb_cannot_pass_with_incomplete_cleanup(self):
        session = FakeSession(config())
        session.cleanup_ok = False
        report, _ = self.run_check(session)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["runs"][0]["metrics"]["captured_frames"], 2)

    def test_frame_timestamp_dimensions_fov_and_mount_must_match(self):
        changes = [
            ("image", lambda value: setattr(value, "timestamp", 7.), "sensor-snapshot-time-mismatch"),
            ("image", lambda value: setattr(value, "width", 20), "image-dimensions-mismatch"),
            ("image", lambda value: setattr(value, "fov", 70.), "image-fov-mismatch"),
            ("snapshot", lambda value: setattr(value, "frame", 99), "snapshot-frame-mismatch"),
            ("snapshot", lambda value: setattr(value.timestamp, "delta_seconds", .1), "world-timestep-mismatch"),
            ("snapshot", lambda value: setattr(value, "find", lambda _: None), "vehicle-snapshot-missing"),
        ]
        for target, change, code in changes:
            with self.subTest(code=code):
                session = FakeSession(config())
                setattr(session, f"{target}_change", change)
                report, _ = self.run_check(session)
                self.assertEqual(report["status"], "FAIL")
                self.assertEqual(report["error"]["code"], code)
                self.assertEqual(session.cleanup_report["status"], "PASS")
        session = FakeSession(config())
        matrix = copy.deepcopy(IDENTITY)
        matrix[0][3] = .02
        session.image_change = lambda value: setattr(value, "transform", Transform(matrix))
        report, _ = self.run_check(session)
        self.assertEqual(report["error"]["code"], "sensor-vehicle-mount-mismatch")

    def test_bounded_callback_overflow_is_failure_without_blocking(self):
        session = FakeSession(config())
        session.burst = True
        report, _ = self.run_check(session)
        self.assertEqual(report["error"]["code"], "camera-queue-overflow")

    def test_repeat_detects_changed_actual_spawn_not_only_same_input_config(self):
        sessions = []

        def factory(cfg, **kwargs):
            session = FakeSession(cfg)
            if sessions:
                session.manifest["vehicle"]["spawn_transform"] = {"x": 1.}
            sessions.append(session)
            return session

        report, _ = self.run_check(runs=2, factory=factory)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"]["code"], "initialization-repeat-mismatch")

    def test_capture_failure_removes_stale_previews_and_redacts_rpc_text(self):
        def factory(*args, **kwargs):
            raise RuntimeError("private/path/access-token")

        with tempfile.TemporaryDirectory() as folder, patch.object(verify, "gpu_sample", return_value={}):
            path = Path(folder) / "collection_preview_run_01.png"
            path.write_bytes(b"old successful image")
            report = verify.run_verification(config(), Path(folder), session_factory=factory)
            self.assertFalse(path.exists())
        self.assertEqual(report["status"], "FAIL")
        self.assertNotIn("access-token", json.dumps(report))

    def test_offline_cli_never_runs_simulator_and_uses_distinct_status(self):
        output = io.StringIO()
        with patch.object(verify, "run_verification") as run, redirect_stdout(output):
            code = verify.main(["--validate-only"])
        run.assert_not_called()
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "CONFIG_VALIDATED")

    def test_invalid_config_overwrites_a_previous_pass_report(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            path = Path(folder)
            (path / verify.REPORT_NAME).write_text('{"status":"PASS"}')
            (path / "invalid.json").write_text('{"schema_version":false}')
            code = verify.main(["--config", str(path / "invalid.json"), "--output-dir", folder])
            self.assertEqual(json.loads((path / verify.REPORT_NAME).read_text())["status"], "FAIL")
        self.assertEqual(code, 1)

    def test_interrupted_rerun_cannot_leave_a_previous_pass_report(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / verify.REPORT_NAME
            path.write_text('{"status":"PASS"}')
            with patch.object(verify, "run_verification", side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt):
                    verify.main(["--output-dir", folder])
            self.assertEqual(json.loads(path.read_text())["status"], "RUNNING")

    def test_unwritable_report_stops_before_simulator_use(self):
        output = io.StringIO()
        with patch.object(verify, "write_report", side_effect=OSError("private/path/token")):
            with patch.object(verify, "run_verification") as run, redirect_stdout(output):
                code = verify.main([])
        self.assertEqual(code, 1)
        run.assert_not_called()
        self.assertNotIn("private/path", output.getvalue())


if __name__ == "__main__":
    unittest.main()
