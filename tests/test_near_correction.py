"""Optical CPU fixtures, independent of any private driving result."""
import copy
from dataclasses import replace
import unittest

import cv2
import numpy as np

from carla_tools.baseline import load_baseline_config, nominal_ground_to_pixels
from carla_tools.near_correction import NearCorrectionConfig, RGBNearCorrection, NearCorrectedPredictor


def painted_scene(offset=0., slope=0.):
    image = np.full((360, 640, 3), 45, np.uint8)
    x = np.linspace(3.5, 11., 500)
    boundary = np.column_stack((x, offset - 2. + slope * (x - 3.)))
    uv, valid = nominal_ground_to_pixels(boundary, load_baseline_config())
    cv2.polylines(image, [np.round(uv[valid]).astype(np.int32)], False, (230, 190, 10), 4)
    return image


class NearCorrectionTests(unittest.TestCase):
    def test_observed_boundary_corrects_both_lateral_signs_and_preserves_far_points(self):
        correction = RGBNearCorrection()
        for error in (-.8, .8):
            points = [[float(x), error] for x in (3, 6, 9, 12, 15)]
            unchanged = copy.deepcopy(points)
            out, diagnostics = correction.apply(painted_scene(), points)
            self.assertTrue(diagnostics['available'])
            self.assertLess(abs(out[0][1]), abs(error))
            self.assertLess(abs(out[1][1]), abs(error))
            self.assertEqual([p[0] for p in out], [p[0] for p in points])
            self.assertEqual(out[2:], points[2:])
            self.assertEqual(points, unchanged)
            again, duplicate = correction.apply(painted_scene(), points)
            self.assertEqual(again, out)
            self.assertEqual(duplicate, diagnostics)

    def test_no_markings_and_unsupported_colour_leave_fresh_model_points(self):
        correction = RGBNearCorrection()
        points = [[float(x), .4] for x in (3, 6, 9, 12, 15)]
        image = painted_scene()
        image[image[:, :, 0] > 100] = (255, 255, 255)
        for rgb in (np.full((360, 640, 3), 45, np.uint8), image):
            out, diagnostics = correction.apply(rgb, points)
            self.assertEqual(out, points)
            self.assertFalse(diagnostics['available'])

    def test_invalid_config_or_frame_never_becomes_a_path(self):
        with self.assertRaises(ValueError):
            replace(NearCorrectionConfig(), blend=1.1)
        with self.assertRaises(ValueError):
            replace(NearCorrectionConfig(), minimum_span_m=float('nan'))
        with self.assertRaises(ValueError):
            replace(NearCorrectionConfig(), minimum_forward_m=20.)
        correction = RGBNearCorrection()
        with self.assertRaises(ValueError):
            correction.apply(np.zeros((360, 640, 3)), [[x, 0.] for x in (3, 6, 9, 12, 15)])

    def test_wrapper_accepts_only_rgb_and_has_no_retained_success_after_failure(self):
        class FixturePredictor:
            identity = {'variant': 'fixture', 'device': 'cpu', 'checkpoint_sha256': 'fixture-only'}
            def __init__(self): self.fail = False; self.calls = []
            def predict(self, rgb, *, width, height):
                self.calls.append((rgb, width, height))
                return {'status': 'FAIL' if self.fail else 'PASS',
                    'waypoints_ego_m': None if self.fail else [[float(x), .8] for x in (3, 6, 9, 12, 15)],
                    'diagnostics': {}, 'error': 'fixture-failure' if self.fail else None, 'timing_ms': 0.}
        model = FixturePredictor()
        wrapper = NearCorrectedPredictor(model)
        image = painted_scene()
        self.assertEqual(wrapper.predict(image)['status'], 'PASS')
        model.fail = True
        failed = wrapper.predict(image)
        self.assertEqual(failed['status'], 'FAIL')
        self.assertIsNone(failed['waypoints_ego_m'])
        self.assertNotIn('near_correction', failed['diagnostics'])
        self.assertEqual(wrapper.identity['variant'], 'fixture_near')
        self.assertTrue(all(call[0] is image for call in model.calls))


if __name__ == '__main__': unittest.main()
