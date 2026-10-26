"""CPU geometry, optical evidence and timed runtime checks for turn confidence."""
import copy
from dataclasses import FrozenInstanceError, replace
import inspect
import json
import math
import unittest

import numpy as np
import torch

from carla_tools import adaptive_path_correction as original
from carla_tools.guarded_path_correction import guard_path_proposal
from carla_tools.pipeline_latency import STAGES, validate_pipeline_timing
from carla_tools.sharp_turn_path_correction import (SharpTurnConfig, SharpTurnRGBPathCorrection,
    SharpTurnPathPredictor, sharp_turn_guard_path_proposal)
import test_adaptive_path_correction as fixtures

SETTINGS = ((.3, .7), (.5, .7))


def path(severe=False):
    return ([[3., .8], [4.8, 2.7], [6.3, 5.], [8., 8.], [9.5, 11.]] if severe else
        [[3., .8], [5., 1.1], [7., 3.], [9., 5.], [11., 7.]])


def proposal(raw):
    result = copy.deepcopy(raw)
    for point in result[:2]:
        point[1] -= .2
    return result


class SharpTurnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_strict_frozen_configuration_and_no_privileged_inputs(self):
        for start, end in SETTINGS:
            config = SharpTurnConfig(fade_start_rad=start, fade_end_rad=end)
            self.assertEqual(SharpTurnConfig.from_dict(config.to_dict()), config)
            self.assertEqual(SharpTurnConfig.fromdict(config.to_dict()), config)
            with self.assertRaises(FrozenInstanceError):
                config.fade_start_rad = .1
        for bad in (True, float("nan"), .7):
            with self.assertRaises(ValueError):
                SharpTurnConfig(fade_start_rad=bad)
        with self.assertRaises(ValueError):
            SharpTurnConfig.from_dict(dict(SharpTurnConfig().to_dict(), route=[]))
        self.assertEqual(set(inspect.signature(SharpTurnRGBPathCorrection.apply).parameters),
            {"self", "rgb", "points", "width", "height"})
        for key in ("ground_truth", "route", "pose", "speed", "previous_path"):
            with self.subTest(key=key), self.assertRaises(TypeError):
                sharp_turn_guard_path_proposal(path(), path(), 1.1,
                    fade_start_rad=.3, fade_end_rad=.7, **{key: []})

    def test_normal_turn_preserves_correction_without_unconditional_quality_suppression(self):
        raw = path()
        _, _, old = original.adaptive_guard_path_proposal(raw, proposal(raw), 1.1)
        self.assertLess(old["quality"]["geometry_confidence"], 1.)
        for start, end in SETTINGS:
            _, guard, detail = sharp_turn_guard_path_proposal(raw, proposal(raw), 1.1,
                fade_start_rad=start, fade_end_rad=end)
            self.assertEqual(detail["quality"]["sharp_turn_severity"], 0.)
            self.assertEqual(detail["quality"]["geometry_confidence"], 1.)
            self.assertGreaterEqual(detail["common_scale"], old["common_scale"])
            self.assertTrue(guard["original_output_bounds_verified"])

    def test_severe_turn_retains_original_quality_and_selected_path_exactly(self):
        raw = path(True)
        old, old_guard, old_detail = original.adaptive_guard_path_proposal(raw, proposal(raw), 1.1)
        self.assertLess(old_detail["quality"]["geometry_confidence"], 1.)
        for start, end in SETTINGS:
            selected, guard, detail = sharp_turn_guard_path_proposal(raw, proposal(raw), 1.1,
                fade_start_rad=start, fade_end_rad=end)
            self.assertEqual(detail["quality"]["sharp_turn_severity"], 1.)
            self.assertEqual(detail["quality"]["geometry_confidence"], old_detail["quality"]["geometry_confidence"])
            self.assertEqual(selected, old)
            for key in ("budget_projection_scale", "outer_disagreement_confidence", "near_point_change_m"):
                self.assertEqual(guard[key], old_guard[key])

    def test_left_right_mirror_symmetry(self):
        for severe in (False, True):
            raw = path(severe)
            mirrored = lambda points: [[x, -y] for x, y in points]
            for start, end in SETTINGS:
                a, _, da = sharp_turn_guard_path_proposal(raw, proposal(raw), 1.1,
                    fade_start_rad=start, fade_end_rad=end)
                b, _, db = sharp_turn_guard_path_proposal(mirrored(raw), mirrored(proposal(raw)), 1.1,
                    fade_start_rad=start, fade_end_rad=end)
                self.assertEqual(mirrored(a), b)
                self.assertEqual(da["quality"]["geometry_confidence"], db["quality"]["geometry_confidence"])

    def test_output_budgets_shape_finiteness_and_distant_point_immutability(self):
        for start, end in SETTINGS:
            for displacement in (.1, .499999, .5, .500001, .6, .74999, .8):
                raw = path()
                before = copy.deepcopy(raw)
                proposed = copy.deepcopy(raw)
                for point in proposed[:2]:
                    point[1] += displacement
                selected, _, detail = sharp_turn_guard_path_proposal(raw, proposed, 1.1,
                    fade_start_rad=start, fade_end_rad=end)
                self.assertEqual(raw, before)
                self.assertEqual(selected[2:], raw[2:])
                self.assertTrue(all(math.isfinite(v) for point in selected for v in point))
                self.assertTrue(guard_path_proposal(raw, selected)[1]["accepted"])
                self.assertLessEqual(detail["shape_change"]["maximum_segment_heading_change_rad"], .12)
                self.assertLessEqual(detail["shape_change"]["maximum_signed_curvature_change_per_m"], .04)

    def test_half_metre_proposal_boundary_is_continuous(self):
        for start, end in SETTINGS:
            outputs = []
            for shift in (.499999, .5, .500001):
                raw = fixtures.straight()
                proposed = copy.deepcopy(raw)
                proposed[0][1], proposed[1][1] = shift, .2
                selected, _, _ = sharp_turn_guard_path_proposal(raw, proposed, 3.,
                    fade_start_rad=start, fade_end_rad=end)
                outputs.append(np.asarray(selected))
            self.assertLess(float(np.max(np.abs(outputs[0] - outputs[-1]))), .00001)

    def test_heading_severity_transition_has_no_endpoint_jump(self):
        for start, end in SETTINGS:
            for boundary in (start, end):
                confidence = []
                for heading in (boundary - 1e-7, boundary, boundary + 1e-7):
                    raw = [[3., .8], [3. + 2 * math.cos(heading), .8 + 2 * math.sin(heading)],
                        [7., 4.], [9., 6.], [11., 8.]]
                    _, _, detail = sharp_turn_guard_path_proposal(raw, proposal(raw), 1.1,
                        fade_start_rad=start, fade_end_rad=end)
                    confidence.append(detail["quality"]["geometry_confidence"])
                self.assertLess(max(confidence) - min(confidence), .00001)

    def test_synthetic_visible_dash_corrects_near_path_and_empty_image_preserves_raw(self):
        for start, end in SETTINGS:
            corrector = SharpTurnRGBPathCorrection(config=SharpTurnConfig(fade_start_rad=start, fade_end_rad=end))
            raw = fixtures.straight(.3)
            before = copy.deepcopy(raw)
            selected, detail = corrector.apply(fixtures.paint(start=9., end=12.), raw)
            self.assertLess(abs(selected[0][1]), abs(raw[0][1]))
            self.assertEqual(selected[2:], raw[2:])
            self.assertEqual(raw, before)
            json.dumps(detail, allow_nan=False)
            unchanged, no_evidence = corrector.apply(np.zeros((360, 640, 3), np.uint8), raw)
            self.assertEqual(unchanged, raw)
            self.assertIsNot(unchanged, raw)
            self.assertFalse(no_evidence["supported_path_correction"]["accepted"])

    def test_original_globals_unchanged_and_replaced_configuration_refused(self):
        quality = original.proposal_quality
        raw = path()
        sharp_turn_guard_path_proposal(raw, proposal(raw), 1.1, fade_start_rad=.3, fade_end_rad=.7)
        self.assertIs(original.proposal_quality, quality)
        self.assertIs(original.adaptive_guard_path_proposal.__globals__["proposal_quality"], quality)
        corrector = SharpTurnRGBPathCorrection()
        corrector.config = replace(corrector.config, fade_start_rad=.5)
        with self.assertRaises(ValueError):
            corrector.apply(fixtures.paint(), fixtures.straight())

    def test_one_forward_original_timing_stages_and_three_diagnostic_keys(self):
        for start, end in SETTINGS:
            base = fixtures.AdaptiveRuntimeTests().base()
            predictor = SharpTurnPathPredictor(base, SharpTurnConfig(fade_start_rad=start, fade_end_rad=end))
            forwards = []
            hook = base.raw.model.register_forward_hook(lambda *_: forwards.append(1))
            self.addCleanup(hook.remove)
            timer = predictor.timer()
            timer.measure("bgra_conversion_ms", lambda: None)
            result = predictor.predict_profiled(np.zeros((360, 640, 3), np.uint8), timer)
            timer.measure("controller_ms", lambda: None)
            timing = validate_pipeline_timing(timer.finish())
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(len(forwards), 1)
            self.assertEqual(timing["executed_stages"], list(STAGES))
            self.assertEqual(predictor.correction, "supported_path")
            for key in ("path_correction", "guarded_path_correction", "supported_path_correction"):
                self.assertIn(key, result["diagnostics"])
            changed = predictor.identity
            changed["sharp_turn_config"]["fade_start_rad"] = .1
            self.assertEqual(predictor.identity["sharp_turn_config"], predictor.config.to_dict())

    def test_invalid_frame_does_not_retain_previous_success(self):
        predictor = SharpTurnPathPredictor(fixtures.AdaptiveRuntimeTests().base())
        self.assertEqual(predictor.predict(np.zeros((360, 640, 3), np.uint8))["status"], "PASS")
        failed = predictor.predict(np.zeros((360, 640, 3), float))
        self.assertEqual(failed["status"], "FAIL")
        self.assertIsNone(failed["waypoints_ego_m"])
        self.assertNotIn("supported_path_correction", failed["diagnostics"])


if __name__ == "__main__":
    unittest.main()
