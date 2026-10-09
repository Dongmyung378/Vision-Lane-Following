"""Independent branch fixtures verify the stateless RGB-only inference policy."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from carla_tools.learning_data import load_learning_config
from carla_tools.rgb_predictor import BaselineAssistedWaypointPredictor


def points(y=0.0):
    return [[float(x), y] for x in (3, 6, 9, 12, 15)]


def passed(values=None):
    return {"status": "PASS", "waypoints_ego_m": points() if values is None else values, "error": None}


def failed():
    return {"status": "FAIL", "waypoints_ego_m": None, "error": "fixture-detection-failed"}


class RGBPredictorTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((360, 640, 3), dtype=np.uint8)
        self.baseline, self.learned = Mock(), Mock()
        self.baseline.predict.return_value = passed(points(0.1))
        self.learned.predict_rgb.return_value = passed(points(0.2))
        self.predictor = BaselineAssistedWaypointPredictor(self.learned, baseline=self.baseline)

    def test_valid_baseline_skips_learned_inference(self):
        result = self.predictor.predict_rgb(self.rgb)
        self.assertEqual(result["waypoints_ego_m"], points(0.1))
        self.assertEqual(result["diagnostics"]["selected_predictor"], "B0")
        self.assertFalse(result["diagnostics"]["fallback_used"])
        self.learned.predict_rgb.assert_not_called()

    def test_failed_baseline_uses_same_original_rgb_only(self):
        self.baseline.predict.return_value = failed()
        result = self.predictor.predict_rgb(self.rgb)
        self.assertEqual(result["diagnostics"]["selected_predictor"], "ResNet18")
        self.assertEqual(result["waypoints_ego_m"], points(0.2))
        baseline_args, baseline_kwargs = self.baseline.predict.call_args
        self.assertEqual(len(baseline_args), 1)
        self.assertIs(baseline_args[0], self.rgb)
        self.assertEqual(baseline_kwargs, {"width": 640, "height": 360})
        args, kwargs = self.learned.predict_rgb.call_args
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], self.rgb)
        self.assertEqual(kwargs, {})

    def test_invalid_baseline_pass_falls_back_by_geometry_contract(self):
        for invalid in ([[3.0, 0.0]]*5, points()[:4], points(float("nan")), [[-3.0, 0.0]]+points()[1:]):
            with self.subTest(points=invalid):
                self.baseline.predict.return_value = passed(invalid)
                result = self.predictor.predict_rgb(self.rgb)
                self.assertEqual(result["status"], "PASS")
                self.assertEqual(result["diagnostics"]["selected_predictor"], "ResNet18")

    def test_invalid_learned_pass_is_explicit_failure(self):
        self.baseline.predict.return_value = failed()
        self.learned.predict_rgb.return_value = passed([[3.0, 0.0]]*5)
        result = self.predictor.predict_rgb(self.rgb)
        self.assertEqual(result["status"], "FAIL")
        self.assertIsNone(result["waypoints_ego_m"])
        self.assertIsNone(result["diagnostics"]["selected_predictor"])

    def test_both_fail_after_success_do_not_reuse_a_previous_path(self):
        self.assertEqual(self.predictor.predict_rgb(self.rgb)["status"], "PASS")
        self.baseline.predict.return_value = failed()
        self.learned.predict_rgb.return_value = failed()
        result = self.predictor.predict_rgb(self.rgb)
        self.assertEqual(result["status"], "FAIL")
        self.assertIsNone(result["waypoints_ego_m"])
        self.assertIsNone(result["diagnostics"]["selected_predictor"])
        self.assertTrue(result["error"])

    def test_repeated_frame_is_independent_of_an_intervening_frame(self):
        self.baseline.predict.return_value = failed()
        self.learned.predict_rgb.side_effect = lambda rgb: passed(points(float(rgb[0, 0, 0])/100))
        other = self.rgb.copy()
        other[0, 0, 0] = 30
        first = self.predictor.predict_rgb(self.rgb)
        middle = self.predictor.predict_rgb(other)
        repeated = self.predictor.predict_rgb(self.rgb)
        self.assertEqual(first["waypoints_ego_m"], repeated["waypoints_ego_m"])
        self.assertNotEqual(first["waypoints_ego_m"], middle["waypoints_ego_m"])
        self.assertEqual(self.learned.predict_rgb.call_count, 3)

    def test_baseline_exception_falls_back_without_stale_state(self):
        self.baseline.predict.side_effect = ValueError("fixture-invalid-baseline")
        result = self.predictor.predict_rgb(self.rgb)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["diagnostics"]["selected_predictor"], "ResNet18")

    def test_invalid_rgb_never_reaches_either_predictor(self):
        for invalid in (self.rgb[:180], self.rgb.astype(np.float32), self.rgb[:, :, 0], None):
            with self.subTest(rgb=type(invalid).__name__):
                result = self.predictor.predict_rgb(invalid)
                self.assertEqual(result["status"], "FAIL")
                self.assertIsNone(result["waypoints_ego_m"])
        self.baseline.predict.assert_not_called()
        self.learned.predict_rgb.assert_not_called()

    def test_invalid_dimensions_never_reach_predictors(self):
        result = self.predictor.predict(self.rgb, width=320, height=180)
        self.assertEqual(result["status"], "FAIL")
        self.baseline.predict.assert_not_called()
        self.learned.predict_rgb.assert_not_called()

    def test_constructor_requires_actual_callable_interfaces(self):
        with self.assertRaises(TypeError):
            BaselineAssistedWaypointPredictor(object(), baseline=self.baseline)
        with self.assertRaises(TypeError):
            BaselineAssistedWaypointPredictor(self.learned, baseline=object())


class AssistedEvaluationTests(unittest.TestCase):
    def setUp(self):
        import cv2
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.samples = []
        for index, (split, case) in enumerate((("train", "straight"), ("dev", "straight"),
                                              ("dev", "left_curve"), ("dev", "right_curve"))):
            rgb = np.full((360, 640, 3), 20+index, dtype=np.uint8)
            path = self.root / f"image-{index}.png"
            cv2.imwrite(str(path), rgb)
            import hashlib
            self.samples.append({"sample_id": str(index), "episode_id": "ep-"+split,
                "route_id": "route-"+split, "scene_group_id": "scene-"+split, "split": split,
                "case": case, "condition": "center", "frame": index+10, "timestamp": 0.5+index*0.05,
                "image": path.name, "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "image_width": 640, "image_height": 360, "waypoints_ego_m": points(), "speed_mps": 3.0})
        self.manifest = self.root / "pilot_manifest.json"
        self.manifest.write_text(json.dumps({"schema_version": 1, "status": "COMPLETE",
            "scope": "moving-rgb-waypoint-pilot", "town": "Town01", "reserved_final_town": "Town02",
            "sample_count": 4, "split_counts": {"train": 1, "dev": 3}, "samples": self.samples}))
        self.checkpoint = self.root / "checkpoint.pt"
        self.checkpoint.write_bytes(b"explicit-mocked-unit-checkpoint")
        source = Path(__file__).resolve().parents[1] / "scripts/evaluate_rgb_predictor.py"
        spec = importlib.util.spec_from_file_location("evaluate_rgb_predictor_fixture", source)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.learned, self.basic = Mock(), Mock()
        self.learned.predict_rgb.return_value = passed(points(0.2))
        self.basic.predict.side_effect = [passed(points(0.1)), failed(), failed()]

    def tearDown(self):
        self.temp.cleanup()

    def evaluate(self, name="evaluation"):
        with patch.object(self.module, "load_checkpoint", return_value=(self.learned, load_learning_config(), {})), \
                patch.object(self.module, "OpenCVLanePredictor", return_value=self.basic):
            return self.module.evaluate_rgb_predictor(self.manifest, self.checkpoint, self.root/name)

    def test_actual_calls_account_for_branches_groups_and_failures(self):
        self.learned.predict_rgb.side_effect = [passed(points(0.2)), failed()]
        result = self.evaluate()
        self.assertEqual(result["branch_counts"], {"B0": 1, "ResNet18": 1, "failed": 1})
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["metrics"][0]["coverage"], 2/3)
        self.assertEqual(result["metrics"][0]["failed_prediction_count"], 1)
        self.assertAlmostEqual(result["metrics"][0]["far_mean_m"], 0.15)
        self.assertFalse(result["performance_target_passed"])
        self.assertEqual(len(result["metrics"]), 7)
        self.assertEqual(result["baseline_attempt_count"], 3)
        self.assertEqual(result["learned_attempt_count"], 2)
        self.assertIn("B0", result["latency_scope"])
        self.assertEqual(len(result["source_sha256"]), 6)
        self.assertEqual(len(result["predictor_source_sha256"]), 2)
        self.assertTrue(result["actual_inference_performed"])
        self.assertTrue(result["inputs_and_sources_unchanged"])

    def test_rgb_byte_tampering_is_rejected_before_prediction(self):
        (self.root / self.samples[1]["image"]).write_bytes(b"tampered-source")
        with self.assertRaisesRegex(ValueError, "image-sha256-mismatch"):
            self.evaluate()
        self.basic.predict.assert_not_called()

    def test_checkpoint_changed_during_inference_is_rejected(self):
        def mutate(rgb, **kwargs):
            self.checkpoint.write_bytes(b"changed-checkpoint")
            return passed()
        self.basic.predict.side_effect = mutate
        with self.assertRaisesRegex(ValueError, "input-or-source-changed"):
            self.evaluate()

    def test_existing_output_is_preserved(self):
        target = self.root / "existing"
        target.mkdir()
        marker = target / "keep.txt"
        marker.write_text("preserve")
        with self.assertRaisesRegex(ValueError, "output-directory-already-exists"):
            self.evaluate("existing")
        self.assertEqual(marker.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
