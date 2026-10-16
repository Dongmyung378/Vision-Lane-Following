"""CPU optical scenes exercise blind-zone alignment and safe fallbacks."""
import copy
from dataclasses import replace
import unittest

import cv2
import numpy as np

from carla_tools.baseline import load_baseline_config, nominal_ground_to_pixels
from carla_tools.learning_data import validate_waypoint_points
from carla_tools.near_correction import RGBNearCorrection
from carla_tools.path_correction import PathCorrectionConfig, PathCorrectedPredictor, RGBPathCorrection


def paint(*, start=9., end=12., offset=0., slope=0., color=(230, 190, 10)):
    image = np.full((360, 640, 3), 45, np.uint8)
    x = np.linspace(start, end, 500)
    boundary = np.column_stack((x, -2. + offset + slope * (x - start)))
    uv, valid = nominal_ground_to_pixels(boundary, load_baseline_config())
    cv2.polylines(image, [np.round(uv[valid]).astype(np.int32)], False, color, 3)
    return image


class PathCorrectionTests(unittest.TestCase):
    def test_further_visible_dash_corrects_unsupported_near_path_in_both_directions(self):
        image = paint()
        corrector = RGBPathCorrection()
        for error in (-.6, .6):
            points = [[float(x), error] for x in (3, 6, 9, 12, 15)]
            original = copy.deepcopy(points)
            local, local_diagnostic = RGBNearCorrection().apply(image, points)
            self.assertEqual(local, points)
            self.assertFalse(local_diagnostic['available'])
            corrected, diagnostic = corrector.apply(image, points)
            self.assertTrue(diagnostic['available'])
            self.assertEqual(diagnostic['method'], 'paint-normal-and-heading-alignment-with-arc-resampling')
            self.assertLess(abs(corrected[0][1]), abs(error))
            self.assertLess(abs(corrected[1][1]), abs(error))
            self.assertEqual(corrected[2:], points[2:])
            self.assertEqual(points, original)
            validate_waypoint_points(corrected)
            again, repeated = corrector.apply(image, points)
            self.assertEqual(again, corrected)
            self.assertEqual(repeated, diagnostic)

    def test_observed_heading_uses_arc_length_and_preserves_far_predictions(self):
        points = [[float(x), .6] for x in (3, 6, 9, 12, 15)]
        corrected, diagnostic = RGBPathCorrection().apply(paint(slope=.1), points)
        self.assertTrue(diagnostic['available'])
        self.assertGreater(abs(diagnostic['heading_change_rad']), .03)
        self.assertLess(abs(diagnostic['heading_change_rad']), .3)
        self.assertNotEqual(corrected[0][0], points[0][0])
        self.assertEqual(corrected[2:], points[2:])
        self.assertLess(abs(corrected[0][1]), abs(points[0][1]))
        validate_waypoint_points(corrected)

    def test_available_local_evidence_retains_the_existing_near_result(self):
        points = [[float(x), .6] for x in (3, 6, 9, 12, 15)]
        image = paint(start=3.5, end=7.)
        expected, _ = RGBNearCorrection().apply(image, points)
        corrected, diagnostic = RGBPathCorrection().apply(image, points)
        self.assertEqual(corrected, expected)
        self.assertEqual(diagnostic['method'], 'local-near-boundary')

    def test_absent_unsupported_and_geometrically_inconsistent_evidence_falls_back(self):
        points = [[float(x), .6] for x in (3, 6, 9, 12, 15)]
        corrector = RGBPathCorrection()
        for image in (np.full((360, 640, 3), 45, np.uint8), paint(color=(255, 255, 255)),
                      paint(offset=4.), paint(slope=.8)):
            corrected, diagnostic = corrector.apply(image, points)
            self.assertEqual(corrected, points)
            self.assertFalse(diagnostic['available'])
        invalid = copy.deepcopy(points)
        invalid[1] = [2., .6]
        with self.assertRaises(ValueError):
            corrector.apply(paint(), invalid)

    def test_config_rejects_nonfinite_bool_and_unbounded_transforms(self):
        for change in ({'blend': float('nan')}, {'blend': 1.1}, {'heading_gain': 1.1}, {'minimum_pixels': True},
                       {'maximum_heading_change_rad': .6}, {'maximum_near_change_m': 4.},
                       {'minimum_forward_m': 30.}, {'curve_segment_samples': 4}):
            with self.assertRaises(ValueError):
                replace(PathCorrectionConfig(), **change)
        self.assertEqual(PathCorrectionConfig().sha256, PathCorrectionConfig().sha256)

    def test_wrapper_has_fresh_failures_and_does_not_mutate_underlying_identity(self):
        class Predictor:
            identity = {'variant': 'fixture', 'device': 'cpu', 'checkpoint_sha256': 'fixture-only'}
            def __init__(self):
                self.fail = False
            def predict(self, rgb, *, width, height):
                return {'status': 'FAIL' if self.fail else 'PASS',
                        'waypoints_ego_m': None if self.fail else [[float(x), .6] for x in (3, 6, 9, 12, 15)],
                        'diagnostics': {}, 'error': 'fixture-failure' if self.fail else None, 'timing_ms': 0.}
        raw = Predictor()
        wrapper = PathCorrectedPredictor(raw)
        self.assertEqual(raw.identity['variant'], 'fixture')
        self.assertEqual(wrapper.identity['variant'], 'fixture_path_near')
        identity = wrapper.identity
        identity['correction_config']['blend'] = 0.
        self.assertEqual(wrapper.identity['correction_config']['blend'], .6)
        self.assertEqual(wrapper.predict(paint())['status'], 'PASS')
        raw.fail = True
        failed = wrapper.predict(paint())
        self.assertEqual(failed['status'], 'FAIL')
        self.assertIsNone(failed['waypoints_ego_m'])
        self.assertNotIn('path_correction', failed['diagnostics'])


if __name__ == '__main__':
    unittest.main()
