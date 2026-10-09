"""Replay actual shared-engine CPU artifacts for every approved initial offset.

The stand-in does not simulate steering dynamics. A valid yaw-induced driving
failure remains a correctly recorded episode rather than a claimed recovery.
"""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from carla_tools.baseline_driving import run_baseline_trial as original_trial
from carla_tools.baseline_validation import validate_baseline_episode
from carla_tools.driving_route import FixedRouteTracker
from carla_tools.pilot import load_pilot_config
import test_baseline_native as native_fixture
from test_driving_native import NativeStandInWorld


class PilotNativeConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.config = load_pilot_config()
        helper = native_fixture.BaselineNativeTests()
        for condition in cls.config.conditions:
            def conditioned(route, case, _condition, destination, collection, *, selected=condition, **kwargs):
                return original_trial(route, case, selected, destination, collection, **kwargs)
            with patch("test_baseline_native.run_baseline_trial", side_effect=conditioned):
                report = helper.run_case(NativeStandInWorld(), cls.root / condition["name"],
                                         method="privileged_reference", full_resolution=True)
            if report["status"] != "PASS":
                raise AssertionError(report)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_all_five_approved_initial_conditions_pass_independent_source_replay(self):
        for condition in self.config.conditions:
            with self.subTest(condition=condition["name"]):
                gate = validate_baseline_episode(self.root / condition["name"], allowed_conditions=self.config.conditions)
                self.assertEqual(gate["status"], "PASS", gate)
                self.assertTrue(gate["evaluation_metrics"]["integrity_valid"])
                if "yaw" in condition["name"]:
                    self.assertFalse(gate["evaluation_metrics"]["driving_success"])
                    self.assertEqual(gate["evaluation_metrics"]["termination_reason"], "lane_departure")

    def test_default_baseline_protocol_still_refuses_recovery_condition(self):
        gate = validate_baseline_episode(self.root / "right_offset")
        self.assertEqual(gate["status"], "FAIL")
        self.assertEqual(gate["errors"][0]["code"], "baseline-condition-invalid")

    def test_approved_condition_metadata_must_match_actual_initial_pose(self):
        for original_name, replacement_name in (("left_offset", "right_offset"), ("left_yaw", "right_yaw")):
            with self.subTest(condition=original_name):
                path = self.root / original_name / "episode.json"
                saved = path.read_bytes()
                try:
                    metadata = json.loads(saved)
                    metadata["condition"] = copy.deepcopy(next(condition for condition in self.config.conditions
                                                                if condition["name"] == replacement_name))
                    path.write_text(json.dumps(metadata), encoding="utf-8")
                    gate = validate_baseline_episode(path.parent, allowed_conditions=self.config.conditions)
                    self.assertEqual(gate["status"], "FAIL")
                    self.assertEqual(gate["errors"][0]["code"], "baseline-initial-reference-or-condition-mismatch")
                finally:
                    path.write_bytes(saved)

    def test_recomputed_gt_label_cannot_hide_unrequested_initial_pose(self):
        directory = self.root / "left_offset"
        metadata_path, trace_path = directory / "episode.json", directory / "steps.jsonl"
        metadata_saved, trace_saved = metadata_path.read_bytes(), trace_path.read_bytes()
        try:
            metadata = json.loads(metadata_saved)
            rows = [json.loads(line) for line in trace_saved.decode().splitlines()]
            first = rows[0]
            pose = first["vehicle"]["transform"]
            pose["location"]["y"] -= .4
            pose["matrix"][1][3] = pose["location"]["y"]
            first["tracking"] = FixedRouteTracker(metadata["route_contract"], max_speed_mps=12.,
                                                   minimum_dt_seconds=.05).observe(pose, first["frame"], first["timestamp"])
            trace = ("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n").encode()
            trace_path.write_bytes(trace)
            metadata["trace_sha256"] = hashlib.sha256(trace).hexdigest()
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            gate = validate_baseline_episode(directory, allowed_conditions=self.config.conditions)
            self.assertEqual(gate["status"], "FAIL")
            self.assertEqual(gate["errors"][0]["code"], "baseline-initial-reference-or-condition-mismatch")
        finally:
            metadata_path.write_bytes(metadata_saved)
            trace_path.write_bytes(trace_saved)


if __name__ == "__main__":
    unittest.main()
