"""Synthetic pinhole scenes verify the independent RGB-only lane predictor."""

import hashlib
import contextlib
import io
import json
import math
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

from carla_tools.baseline import (
    BaselineConfig, BaselineError, OpenCVLanePredictor, baseline_config_from_dict,
    load_baseline_config, nominal_ground_to_pixels, overlay_baseline,
)

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None


def scene(curvature=0.0, offset=0.0, yaw_deg=0.0, kind="yellow", lane_width=4.0, shadow=False, outliers=False):
    """Rasterise physical paint polygons; no predictor geometry is reused.

    Projection is independently derived from a CARLA fixed-camera rigid matrix,
    not the implementation's inverse-ground function.
    """
    image = np.full((360, 640, 3), 50, dtype=np.uint8)
    image[:110] = (120, 150, 180)
    yaw = math.radians(yaw_deg)
    rotate = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    pitch = math.radians(-12)
    rotation = np.array([[math.cos(pitch), 0, -math.sin(pitch)], [0, 1, 0], [math.sin(pitch), 0, math.cos(pitch)]])

    def center(s):
        if curvature:
            return np.array([math.sin(curvature*s)/curvature, (1-math.cos(curvature*s))/curvature]), curvature*s
        return np.array([s, 0.0]), 0.0

    def pixel(point):
        ego = rotate @ (point - np.array([0.0, offset]))
        vector = rotation.T @ np.array([ego[0]-1.5, ego[1], -2.4])
        if vector[0] <= 0.05:
            return None
        return np.array([320*vector[1]/vector[0]+320, -320*vector[2]/vector[0]+180])

    sides = [-1] if kind == "yellow" else [-1, 1]
    for side in sides:
        for start in np.arange(2.0, 39.0, 8.0 if kind == "yellow" else 2.0):
            length = 3.0 if kind == "yellow" else 2.05
            stripe = []
            for sign, s_values in ((-1, np.linspace(start, start+length, 20)), (1, np.linspace(start+length, start, 20))):
                for s in s_values:
                    point, angle = center(s)
                    normal = np.array([-math.sin(angle), math.cos(angle)])
                    uv = pixel(point + (side*lane_width/2 + sign*0.08)*normal)
                    if uv is not None:
                        stripe.append(uv)
            if len(stripe) >= 4:
                cv2.fillPoly(image, [np.rint(stripe).astype(np.int32)], (255, 185, 0) if kind == "yellow" else (230, 230, 230))
    if outliers:
        rng = np.random.default_rng(43)
        for x, y in rng.integers([0, 145], [640, 360], size=(70, 2)):
            cv2.circle(image, (int(x), int(y)), 1, (255, 200, 0), -1)
    if shadow:
        image[215:275] = (image[215:275].astype(float)*0.45).astype(np.uint8)
    expected = []
    for s in (3, 6, 9, 12, 15):
        point, _ = center(s)
        expected.append((rotate @ (point-np.array([0.0, offset]))).tolist())
    return image, np.array(expected)


class BaselineConfigTests(unittest.TestCase):
    def test_defaults_strict_roundtrip_and_hash(self):
        config = load_baseline_config()
        self.assertEqual(config, BaselineConfig())
        self.assertEqual(baseline_config_from_dict(config.to_dict()), config)
        self.assertEqual(len(config.sha256), 64)
        self.assertEqual(config.sha256, hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest())
        with self.assertRaises(FrozenInstanceError):
            config.camera_x_m = 9

    def test_unknown_missing_duplicate_and_nonfinite_rejected(self):
        defaults = BaselineConfig().to_dict()
        for changed in ({**defaults, "actual_vehicle_pose": {}}, {key: value for key, value in defaults.items() if key != "fov_deg"}):
            with self.assertRaises(BaselineError):
                baseline_config_from_dict(changed)
        for key, value in (("camera_z_m", 2.5), ("camera_pitch_deg", -10), ("image_width", 1280),
                           ("nominal_lane_width_m", float("nan")), ("maximum_lateral_m", float("inf")),
                           ("minimum_marking_pixels", True), ("maximum_curvature_per_m", True),
                           ("yellow_min_hue", 50), ("schema_version", True)):
            with self.subTest(key=key, value=value), self.assertRaises(BaselineError):
                baseline_config_from_dict({**defaults, key: value})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"config.json"
            path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_baseline_config(path)


@unittest.skipIf(cv2 is None, "Install requirements-baseline.txt for the RGB detector tests")
class BaselinePredictionTests(unittest.TestCase):
    def setUp(self):
        self.predictor = OpenCVLanePredictor()

    def assert_prediction(self, image, expected, tolerance=0.48):
        result = self.predictor.predict(image)
        self.assertEqual(result["status"], "PASS", result)
        predicted = np.asarray(result["waypoints_ego_m"])
        self.assertEqual(predicted.shape, (5, 2))
        self.assertLess(float(np.max(np.linalg.norm(predicted-expected, axis=1))), tolerance, result)
        return result

    def test_straight_center_offsets_and_yaws(self):
        for offset, yaw in ((0, 0), (-0.3, 0), (0.3, 0), (0, -5), (0, 5)):
            with self.subTest(offset=offset, yaw=yaw):
                image, expected = scene(offset=offset, yaw_deg=yaw)
                result = self.assert_prediction(image, expected)
                self.assertEqual(result["diagnostics"]["lane_width_source"], "fixed-nominal-prior")
                self.assertEqual(result["diagnostics"]["inferred_boundary"], "right")

    def test_left_right_curve_arc_lengths_and_sign(self):
        for curvature in (-0.04, 0.04, -0.06, 0.06):
            with self.subTest(curvature=curvature):
                image, expected = scene(curvature=curvature)
                result = self.assert_prediction(image, expected, tolerance=0.7)
                points = np.asarray(result["waypoints_ego_m"])
                self.assertEqual(np.sign(points[-1, 1]), np.sign(curvature))
                self.assertLess(points[-1, 0], 15.1)

    def test_shadow_and_compact_yellow_outliers_do_not_change_lane(self):
        for shadow, outliers in ((True, False), (False, True), (True, True)):
            with self.subTest(shadow=shadow, outliers=outliers):
                image, expected = scene(shadow=shadow, outliers=outliers)
                self.assert_prediction(image, expected)

    def test_paired_white_and_lane_width_rejection(self):
        image, expected = scene(kind="white")
        result = self.assert_prediction(image, expected)
        self.assertEqual(result["diagnostics"]["boundary_mode"], "paired-white")
        self.assertIsNone(result["diagnostics"]["inferred_boundary"])
        for width in (1.5, 6.0):
            image, _ = scene(kind="white", lane_width=width)
            result = self.predictor.predict(image)
            self.assertEqual(result["status"], "FAIL", result)

    def test_empty_road_and_noisy_colour_have_no_gt_fallback(self):
        images = [np.full((360, 640, 3), value, dtype=np.uint8) for value in (0, 50, 200)]
        rng = np.random.default_rng(14)
        images.append(rng.integers(0, 256, size=(360, 640, 3), dtype=np.uint8))
        for image in images:
            result = self.predictor.predict(image)
            self.assertEqual(result["status"], "FAIL", result)
            self.assertIsNone(result["waypoints_ego_m"])

    def test_oncoming_lane_yellow_on_right_is_refused(self):
        image, _ = scene()
        image = np.ascontiguousarray(image[:, ::-1])
        result = self.predictor.predict(image)
        self.assertEqual(result["status"], "FAIL", result)
        self.assertIsNone(result["waypoints_ego_m"])

    def test_invalid_rgb_inputs_fail_without_throwing(self):
        for rgb, width, height in ((b"", 640, 360), (b"x"*691200, True, 360),
                                   (np.zeros((360, 640, 4), dtype=np.uint8), 640, 360),
                                   (np.zeros((360, 640, 3), dtype=float), 640, 360), (None, 640, 360)):
            result = self.predictor.predict(rgb, width, height)
            self.assertEqual(result["status"], "FAIL", result)

    def test_stateless_bytes_and_array_replay_exact_except_timing(self):
        image, expected = scene(curvature=0.04)
        first = self.assert_prediction(image, expected, tolerance=0.7)
        self.predictor.predict(np.zeros_like(image))
        second = self.predictor.predict(image.tobytes())
        first.pop("timing_ms")
        second.pop("timing_ms")
        self.assertEqual(first, second)
        self.assertEqual(self.predictor.config, BaselineConfig())

    def test_overlay_uses_nominal_projection_and_preserves_input(self):
        image, _ = scene()
        prior = image.copy()
        result = self.predictor.predict(image)
        overlay = overlay_baseline(image, result)
        self.assertEqual(overlay.shape, image.shape)
        self.assertTrue(np.array_equal(image, prior))
        self.assertFalse(np.array_equal(image, overlay))
        uv, front = nominal_ground_to_pixels(result["waypoints_ego_m"], self.predictor.config)
        self.assertTrue(front.all())
        self.assertTrue(np.isfinite(uv).all())

    def test_cli_local_only_fresh_outputs_and_existing_source_preserved(self):
        from scripts import predict_baseline as cli
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, _ = scene()
            source = root / "source.png"
            cv2.imwrite(str(source), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            source_before = source.read_bytes()
            output = root / "results/local/prediction.json"
            overlay = root / "results/local/overlay.png"
            arguments = ["--image", str(source), "--output", str(output), "--overlay", str(overlay)]
            with patch.object(cli, "ROOT", root), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(arguments), 0)
                self.assertEqual(json.loads(output.read_text())["status"], "PASS")
                records_before = output.read_bytes(), overlay.read_bytes()
                with self.assertRaises(SystemExit) as raised:
                    cli.main(arguments)
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual((output.read_bytes(), overlay.read_bytes()), records_before)
                with self.assertRaises(SystemExit):
                    cli.main(["--image", str(source), "--output", str(root/"public.json"), "--overlay", str(overlay)])
            self.assertEqual(source.read_bytes(), source_before)


if __name__ == "__main__":
    unittest.main()
