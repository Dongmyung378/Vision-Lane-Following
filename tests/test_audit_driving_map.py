"""Public CPU map audit: independent corner geometry and immutable sources."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import audit_driving_map as audit


def fixture(directory):
    native = directory / "driving"
    xml = directory / "Town01.xodr"
    xml.write_text("<OpenDRIVE>CPU fixture, never a native simulator</OpenDRIVE>", encoding="utf-8")
    xml_sha = audit.digest(xml.read_bytes())
    pose = {"location": {"x": 10., "y": 20., "z": .1},
            "rotation": {"yaw": 90., "pitch": 34., "roll": 12.}}
    bbox = {"location": {"x": 1., "y": 2., "z": .75},
            "extent": {"x": 2., "y": 1., "z": 3.}}
    for case in audit.CASES:
        for condition in audit.CONDITIONS:
            episode_id = f"{case}__{condition}"
            episode = native / "episodes" / episode_id
            episode.mkdir(parents=True)
            rows = [{"frame": frame, "timestamp": frame * .05, "vehicle": {"transform": pose}}
                    for frame in range(100, 104)]
            trace_bytes = ("\n".join(json.dumps(row) for row in rows[:2]) + "\n").encode()
            (episode / "steps.jsonl").write_bytes(trace_bytes)
            metadata = {"episode_id": episode_id, "status": "COMPLETE", "geometry_case": case,
                        "condition": {"name": condition}, "bounding_box": bbox, "stop_observations": rows[2:],
                        "trace_sha256": audit.digest(trace_bytes), "route_contract": {
                            "town": "Town01", "carla_version": "0.9.16", "opendrive_sha256": xml_sha,
                            "centerline_anchor_sequence": [{"road_id": 7, "section_id": 2, "lane_id": -1}]}}
            (episode / "episode.json").write_text(json.dumps(metadata), encoding="utf-8")
    return native, xml, xml_sha, pose, bbox


def fake_api(query=None):
    waypoint = SimpleNamespace(road_id=7, section_id=2, lane_id=-1, lane_type="Driving")
    map_object = SimpleNamespace(get_waypoint=Mock(side_effect=query or (lambda *args, **kwargs: waypoint)))
    api = SimpleNamespace(Map=Mock(return_value=map_object), Location=lambda **kwargs: SimpleNamespace(**kwargs),
                          LaneType=SimpleNamespace(Driving="Driving"),
                          Client=Mock(side_effect=AssertionError("Client, RPC and server use are forbidden")))
    return api, map_object


class AuditDrivingMapTests(unittest.TestCase):
    def test_four_native_sensor_corners_preserve_unrotated_center_and_ignore_pitch_roll(self):
        pose = {"location": {"x": 10., "y": 20., "z": .1},
                "rotation": {"yaw": 90., "pitch": 34., "roll": 12.}}
        bbox = {"location": {"x": 1., "y": 2., "z": .75},
                "extent": {"x": 2., "y": 1., "z": 3.}}
        corners = audit.sensor_corners(pose, bbox)
        # These expected world points are computed independently from the
        # native rule: center=(11,22,.85), yaw90 rotates +/-2,+/-1 extents.
        expected = ((10., 24.), (10., 20.), (12., 24.), (12., 20.))
        self.assertEqual([corner["name"] for corner in corners],
                         ["front_right", "rear_right", "front_left", "rear_left"])
        for corner, (x, y) in zip(corners, expected):
            self.assertAlmostEqual(corner["world_m"]["x"], x, places=5)
            self.assertAlmostEqual(corner["world_m"]["y"], y, places=5)
            self.assertAlmostEqual(corner["world_m"]["z"], .85, places=6)

    def test_all_fifteen_episodes_main_and_stop_poses_use_cpu_map_and_preserve_source_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            native, xml, xml_sha, _, _ = fixture(Path(temporary))
            api, native_map = fake_api()
            with patch.object(audit, "PINNED_SHA256", xml_sha):
                result = audit.audit(native, xodr_path=xml, carla_module=api)
            self.assertEqual(result["status"], "PASS", result)
            self.assertEqual(result["episode_count"], 15)
            self.assertEqual((result["main_pose_count"], result["stop_pose_count"]), (30, 30))
            self.assertEqual(result["corner_count"], 240)
            self.assertEqual(result["accepted_corner_count"], 240)
            self.assertIs(result["sources_unchanged"], True)
            self.assertEqual(len(result["sources"]), 31)
            self.assertTrue(all(source["unchanged"] for source in result["sources"]))
            self.assertTrue(all(not Path(source["source"]).is_absolute() for source in result["sources"]))
            api.Map.assert_called_once_with("Town01", xml.read_text(encoding="utf-8"))
            api.Client.assert_not_called()
            self.assertEqual(native_map.get_waypoint.call_count, 480)
            for call in native_map.get_waypoint.call_args_list:
                self.assertEqual(call.kwargs, {"project_to_road": False, "lane_type": "Driving"})

    def test_adjacent_driving_lane_and_height_sensitive_queries_cannot_pass(self):
        for mode in ("adjacent", "height"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                native, xml, xml_sha, _, _ = fixture(Path(temporary))
                def query(point, **kwargs):
                    if mode == "height" and point.z > 0:
                        return None
                    return SimpleNamespace(road_id=8 if mode == "adjacent" else 7,
                                           section_id=2, lane_id=-1, lane_type="Driving")
                api, _ = fake_api(query)
                with patch.object(audit, "PINNED_SHA256", xml_sha):
                    result = audit.audit(native, xodr_path=xml, carla_module=api)
                self.assertEqual(result["status"], "FAIL")
                self.assertEqual(result["bad_corner_count"], 240)
                bad = result["first_bad_corner"]
                self.assertEqual((bad["frame"], bad["phase"], bad["corner_index"]), (100, "main", 0))
                self.assertEqual(bad["code"], "sensor-corner-outside-original-route-lanes" if mode == "adjacent"
                                 else "sensor-corner-outside-driving-lane")
                self.assertEqual(result["height_sensitive_corner_count"], 0 if mode == "adjacent" else 240)

    def test_wrong_metadata_xodr_hash_is_rejected_before_map_construction(self):
        with tempfile.TemporaryDirectory() as temporary:
            native, xml, xml_sha, _, _ = fixture(Path(temporary))
            path = next((native / "episodes").iterdir()) / "episode.json"
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata["route_contract"]["opendrive_sha256"] = "0" * 64
            path.write_text(json.dumps(metadata), encoding="utf-8")
            api, _ = fake_api()
            with patch.object(audit, "PINNED_SHA256", xml_sha):
                result = audit.audit(native, xodr_path=xml, carla_module=api)
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("episode-xodr-hash-mismatch", [error["code"] for error in result["errors"]])
            api.Map.assert_not_called()

    def test_source_mutation_during_map_queries_prevents_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            native, xml, xml_sha, _, _ = fixture(Path(temporary))
            trace = next((native / "episodes").iterdir()) / "steps.jsonl"
            changed = False
            def query(*args, **kwargs):
                nonlocal changed
                if not changed:
                    trace.write_bytes(trace.read_bytes() + b"\n")
                    changed = True
                return SimpleNamespace(road_id=7, section_id=2, lane_id=-1, lane_type="Driving")
            api, _ = fake_api(query)
            with patch.object(audit, "PINNED_SHA256", xml_sha):
                result = audit.audit(native, xodr_path=xml, carla_module=api)
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["bad_corner_count"], 0)
            self.assertIs(result["sources_unchanged"], False)
            self.assertIn("audit-source-snapshot-not-unchanged", [error["code"] for error in result["errors"]])

    def test_cli_refuses_existing_or_source_internal_output_without_auditing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native = root / "driving"
            native.mkdir()
            existing = root / "old.json"
            existing.write_text('{"status":"PASS","old":true}', encoding="utf-8")
            old = existing.read_bytes()
            for output in (existing, native / "new.json"):
                with self.subTest(output=output.name), patch.object(audit, "audit") as operation, contextlib.redirect_stdout(io.StringIO()) as stdout:
                    code = audit.main(["--native", str(native), "--output", str(output)])
                self.assertEqual(code, 1)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "FAIL")
                operation.assert_not_called()
            self.assertEqual(existing.read_bytes(), old)
            self.assertFalse((native / "new.json").exists())

    def test_cli_reserves_fail_then_sanitizes_interruption_without_leaving_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native = root / "driving"
            native.mkdir()
            output = root / "new.json"
            def interrupted(*args, **kwargs):
                self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "FAIL")
                raise KeyboardInterrupt("secret-token private-environment-location")
            with patch.object(audit, "audit", side_effect=interrupted), contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = audit.main(["--native", str(native), "--output", str(output)])
            self.assertEqual(code, 1)
            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "FAIL")
            self.assertEqual(persisted["error"]["exception_type"], "KeyboardInterrupt")
            self.assertNotIn("secret-token", output.read_text(encoding="utf-8") + stdout.getvalue())

    def test_cli_rejects_linked_output_ancestors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native = root / "driving"
            native.mkdir()
            linked = root / "linked"
            real = root / "real"
            real.mkdir()
            try:
                linked.symlink_to(real, target_is_directory=True)
            except OSError:
                # Windows may not grant symlink creation. Exercise ancestor
                # detection with an API stand-in instead of skipping the guard.
                original = Path.is_symlink
                with patch.object(Path, "is_symlink", lambda path: path == linked or original(path)):
                    self.assertTrue(audit.linked_path(linked / "new.json"))
                return
            with patch.object(audit, "audit") as operation, contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = audit.main(["--native", str(native), "--output", str(linked / "new.json")])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(stdout.getvalue())["error"]["code"], "audit-output-path-link-refused")
            operation.assert_not_called()
            self.assertFalse((real / "new.json").exists())

    def test_help_works_with_carla_import_blocked(self):
        source = "import runpy,sys;sys.modules['carla']=None;sys.argv=['audit_driving_map.py','--help'];runpy.run_path('scripts/audit_driving_map.py',run_name='__main__')"
        result = subprocess.run([sys.executable, "-I", "-c", source], cwd=audit.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--xodr", result.stdout)


if __name__ == "__main__":
    unittest.main()
