"""Native stand-ins check RGB/control separation, failure accounting and cleanup."""
from __future__ import annotations

import json
import copy
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from carla_tools.baseline_driving import run_baseline_trial
from carla_tools.config import CollectionConfig
from carla_tools.contracts import route_sha256
from test_driving_native import ListeningSensor, NativeStandInWorld, native_module
from test_collection_session import Transform
from test_driving_route import dense_route


class RGBPrediction:
    def __init__(self, status="PASS"):
        self.status, self.calls = status, []

    def predict(self, rgb, *, width, height):
        self.calls.append((rgb, width, height))
        return {"status": self.status, "waypoints_ego_m": [[x, .2] for x in (3., 6., 9., 12., 15.)]
                if self.status == "PASS" else None,
                "diagnostics": {"method": "independent-test-rgb", "failure": None if self.status == "PASS" else "no-markings"},
                "timing_ms": 0.0}


class BaselineNativeTests(unittest.TestCase):
    def run_case(self, world, destination, *, predictor=None, method="B0", full_resolution=False):
        config = CollectionConfig()
        config = replace(config, verification=replace(config.verification, warmup_frames=2))
        if not full_resolution:
            config = replace(config, camera=replace(config.camera, width=16, height=16))
        route = dense_route()
        for anchor in route["centerline_anchor_sequence"]:
            anchor["world_rotation"] = {"pitch": 0., "yaw": 0., "roll": 0.}
        route["start"] = copy.deepcopy(route["centerline_anchor_sequence"][0])
        route["route_sha256"] = route_sha256(route)
        original_transform = ListeningSensor.get_transform

        def sensor_transform(sensor):
            value = original_transform(sensor)
            matrix = value.get_matrix()
            return Transform(location=SimpleNamespace(x=matrix[0][3], y=matrix[1][3], z=matrix[2][3]),
                             rotation=SimpleNamespace(pitch=math.degrees(math.asin(matrix[2][0])),
                                                      yaw=math.degrees(math.atan2(matrix[1][0], matrix[0][0])),
                                                      roll=math.degrees(math.atan2(-matrix[2][1], matrix[2][2]))), matrix=matrix)

        with patch("carla_tools.driving.verify_geometry_route_in_world", return_value=None), \
                patch.object(ListeningSensor, "get_transform", sensor_transform):
            return run_baseline_trial(route, "straight", {"name": "center", "lateral_m": 0., "yaw_deg": 0.},
                                      destination, config, carla_module=native_module(world), predictor=predictor, method=method)

    def test_rgb_prediction_controls_steering_and_every_frame_has_source_image(self):
        world, predictor = NativeStandInWorld(), RGBPrediction()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trial"
            report = self.run_case(world, directory, predictor=predictor)
            self.assertEqual(report["status"], "PASS", report)
            metadata = json.loads((directory / "episode.json").read_text())
            rows = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), len(predictor.calls))
            self.assertGreater(len(rows), 2)
            self.assertEqual(metadata["termination_reason"], "route_completed")
            self.assertTrue(metadata["evaluation_metrics"]["driving_success"])
            self.assertTrue(metadata["cleanup"]["owned_actors_absent"])
            self.assertFalse(metadata["perception_uses_route"])
            self.assertFalse(metadata["perception_uses_vehicle_pose"])
            self.assertGreater(rows[0]["command"]["steer"], 0)
            for row, (rgb, width, height) in zip(rows, predictor.calls):
                self.assertIs(type(rgb), bytes)
                self.assertEqual(len(rgb), width * height * 3)
                self.assertIsNotNone(row["image"])
                self.assertEqual(row["perception"]["waypoints_ego_m"], [[x, .2] for x in (3., 6., 9., 12., 15.)])
                self.assertNotEqual(row["perception"]["waypoints_ego_m"], row["tracking"]["label"]["waypoints_ego_m"])
                self.assertEqual(row["command_source_frame"], row["frame"])
                self.assertEqual(row["command_target_frame"], None if row["terminal"] else row["frame"] + 1)
                self.assertEqual(row["evaluation_events"], [])
            self.assertEqual(set(world.actors), {1})

    def test_detector_failure_keeps_metric_row_and_stops_without_gt_fallback(self):
        world, predictor = NativeStandInWorld(), RGBPrediction("FAIL")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trial"
            report = self.run_case(world, directory, predictor=predictor)
            self.assertEqual(report["status"], "PASS", report)
            metadata = json.loads((directory / "episode.json").read_text())
            rows = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertIsNone(rows[0]["command"])
            self.assertTrue(rows[0]["terminal"])
            self.assertTrue(rows[0]["technical_failure"])
            self.assertEqual(metadata["status"], "COMPLETE")
            self.assertEqual(metadata["termination_reason"], "technical_failure")
            self.assertFalse(metadata["evaluation_metrics"]["driving_success"])
            self.assertTrue(metadata["evaluation_metrics"]["integrity_valid"])
            self.assertEqual(metadata["evaluation_metrics"]["completion_pct"], 0)
            self.assertTrue(metadata["stop_observations"])
            self.assertEqual(set(world.actors), {1})

    def test_reference_and_rgb_share_controller_evaluator_and_initial_conditions(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            for method in ("B0", "privileged_reference"):
                report = self.run_case(NativeStandInWorld(), destination / method, predictor=RGBPrediction(), method=method)
                self.assertEqual(report["status"], "PASS", report)
            rgb = json.loads((destination / "B0/episode.json").read_text())
            gt = json.loads((destination / "privileged_reference/episode.json").read_text())
            for key in ("controller_config_sha256", "evaluation_config_sha256", "condition", "initialization", "stop_control"):
                # Final steering differs because the two waypoint inputs differ.
                if key != "stop_control":
                    self.assertEqual(rgb[key], gt[key])
            self.assertTrue(gt["perception_uses_route"])
            self.assertEqual(gt["method"], "privileged_reference")

    def test_failed_control_readback_invalidates_runtime_and_still_cleans_owned_actors(self):
        world = NativeStandInWorld(acknowledge_control=False)
        with tempfile.TemporaryDirectory() as temporary:
            report = self.run_case(world, Path(temporary) / "trial", predictor=RGBPrediction())
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["error"]["code"], "driving-control-acknowledgement-mismatch")
            self.assertEqual(set(world.actors), {1})

    def test_actual_shared_engine_output_passes_independent_rgb_failure_replay(self):
        from carla_tools.baseline_validation import validate_baseline_episode
        world = NativeStandInWorld()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "straight__center"
            report = self.run_case(world, directory, full_resolution=True)
            self.assertEqual(report["status"], "PASS", report)
            gate = validate_baseline_episode(directory)
            self.assertEqual(gate["status"], "PASS", gate)
            self.assertFalse(gate["evaluation_metrics"]["driving_success"])
            self.assertEqual(gate["evaluation_metrics"]["termination_reason"], "technical_failure")

    def test_predictor_exception_preserves_current_frame_and_completes_braking(self):
        class BrokenPrediction:
            def predict(self, rgb, *, width, height):
                raise RuntimeError("injected numeric backend failure")

        world = NativeStandInWorld()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trial"
            report = self.run_case(world, directory, predictor=BrokenPrediction())
            self.assertEqual(report["status"], "PASS", report)
            metadata = json.loads((directory / "episode.json").read_text())
            row = json.loads((directory / "steps.jsonl").read_text().strip())
            self.assertEqual(row["perception"]["error"], "baseline-predictor-exception")
            self.assertIsNone(row["command"])
            self.assertTrue(row["terminal"])
            self.assertEqual(metadata["termination_reason"], "technical_failure")
            self.assertGreaterEqual(len(metadata["stop_observations"]), 12)
            self.assertTrue(metadata["cleanup"]["owned_actors_absent"])

    def test_malformed_finite_pass_is_recorded_and_braked_before_controller_step(self):
        class MalformedPrediction:
            def predict(self, rgb, *, width, height):
                return {"status": "PASS", "diagnostics": {}, "timing_ms": 1.}

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trial"
            report = self.run_case(NativeStandInWorld(), directory, predictor=MalformedPrediction())
            self.assertEqual(report["status"], "PASS", report)
            metadata = json.loads((directory / "episode.json").read_text())
            row = json.loads((directory / "steps.jsonl").read_text().strip())
            self.assertIsNone(row["command"])
            self.assertEqual(row["perception"]["status"], "FAIL")
            self.assertEqual(metadata["termination_reason"], "technical_failure")
            self.assertGreaterEqual(len(metadata["stop_observations"]), 12)


if __name__ == "__main__":
    unittest.main()
