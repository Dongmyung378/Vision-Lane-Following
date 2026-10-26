"""CPU optical and geometry checks for an independently bounded RGB proposal."""
import copy
from dataclasses import FrozenInstanceError, replace
import inspect
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from carla_tools.guarded_path_correction import GuardedRGBPathCorrection
from carla_tools.learning_data import load_learning_config
from carla_tools.path_correction import PathCorrectionConfig, RGBPathCorrection
from carla_tools.pipeline_latency import PipelineFrameTimer, STAGES, validate_pipeline_timing
from carla_tools.supported_path_correction import (SupportedPathConfig, SupportedPathPredictor,
    SupportedRGBPathCorrection, support_guard_path_proposal, supported_config_from_dict)
from test_path_correction import paint
from test_recovery_training import TinyWaypointModel


def points(y=0.):
    return [[float(x), y] for x in (3, 6, 9, 12, 15)]


class SupportedConfigurationTests(unittest.TestCase):
    def test_config_is_strict_frozen_finite_and_has_no_privileged_inputs(self):
        config = SupportedPathConfig()
        self.assertEqual(supported_config_from_dict(config.to_dict()), config)
        self.assertEqual(config.sha256, SupportedPathConfig().sha256)
        with self.assertRaises(FrozenInstanceError):
            config.maximum_extrapolation_m = 3.
        for change in ({"maximum_extrapolation_m": 3.}, {"maximum_extrapolation_m": True},
                {"maximum_curvature_change_per_m": float("nan")}, {"scale_bisection_steps": True},
                {"scale_bisection_steps": 2}, {"GT": []}, {"version": "other"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                supported_config_from_dict(dict(config.to_dict(), **change))
        self.assertEqual(list(inspect.signature(SupportedRGBPathCorrection.apply).parameters),
            ["self", "rgb", "points", "width", "height"])
        for keyword in ("ground_truth", "pose", "previous_path", "route"):
            with self.subTest(keyword=keyword), self.assertRaises(TypeError):
                support_guard_path_proposal(points(), points(), [3., 7.], **{keyword: []})


class SupportedGeometryTests(unittest.TestCase):
    def test_observed_support_reduces_extrapolation_continuously_and_keeps_far_points(self):
        raw = points()
        proposed = points(); proposed[0][1] = proposed[1][1] = .1
        snapshot = copy.deepcopy((raw, proposed))
        selected, legacy, detail = support_guard_path_proposal(raw, proposed, [3.5, 7.])
        self.assertTrue(legacy["accepted"])
        self.assertEqual(detail["support_weights"], [.5, 1.])
        self.assertEqual(detail["shape_scale"], 1.)
        self.assertAlmostEqual(selected[0][1], .05)
        self.assertAlmostEqual(selected[1][1], .1)
        self.assertEqual(selected[2:], raw[2:])
        self.assertEqual((raw, proposed), snapshot)
        unsupported, _, detail = support_guard_path_proposal(raw, proposed, [8., 11.])
        self.assertEqual(unsupported, raw)
        self.assertEqual(detail["support_weights"], [0., 0.])
        self.assertFalse(detail["accepted"])

    def test_inside_support_preserves_a_small_consistent_proposal_exactly(self):
        raw = points()
        proposed = points(); proposed[0][1] = proposed[1][1] = .1
        selected, _, detail = support_guard_path_proposal(raw, proposed, [2., 7.])
        self.assertEqual(selected, proposed)
        self.assertTrue(detail["accepted"])
        self.assertEqual(detail["shape_scale"], 1.)

    def test_legacy_disagreement_guard_cannot_be_bypassed_by_support_attenuation(self):
        raw = points()
        proposed = points(); proposed[0][1] = proposed[1][1] = .8
        selected, legacy, detail = support_guard_path_proposal(raw, proposed, [3.5, 7.])
        self.assertFalse(legacy["accepted"])
        self.assertEqual(selected, raw)
        self.assertEqual(detail["rejection_reasons"], ["legacy-disagreement-guard-rejected"])

    def test_opposing_near_proposal_is_attenuated_to_both_shape_limits(self):
        raw = points()
        proposed = points(); proposed[0][1] = .4; proposed[1][1] = -.4
        selected, legacy, detail = support_guard_path_proposal(raw, proposed, [2., 7.])
        self.assertTrue(legacy["accepted"])
        self.assertTrue(detail["accepted"])
        self.assertGreater(detail["shape_scale"], 0.)
        self.assertLess(detail["shape_scale"], 1.)
        self.assertLessEqual(detail["shape_change"]["maximum_segment_heading_change_rad"], .12)
        self.assertLessEqual(detail["shape_change"]["maximum_signed_curvature_change_per_m"], .04)
        self.assertEqual(selected[2:], raw[2:])

    def test_millimetre_model_perturbation_does_not_add_a_discrete_scale_jump(self):
        outputs = []
        for perturbation in (-.001, 0., .001):
            raw = points(); raw[0][1] += perturbation
            proposed = copy.deepcopy(raw); proposed[0][1] += .4; proposed[1][1] -= .4
            selected, _, detail = support_guard_path_proposal(raw, proposed, [2., 7.])
            self.assertTrue(0 < detail["shape_scale"] < 1)
            outputs.append(np.asarray(selected))
        self.assertLess(float(np.max(np.abs(outputs[0] - outputs[-1]))), .005)

    def test_missing_support_preserves_fresh_raw_and_invalid_support_or_points_are_rejected(self):
        raw, proposed = points(), points(); proposed[0][1] = .1
        selected, _, detail = support_guard_path_proposal(raw, proposed, None)
        self.assertEqual(selected, raw)
        self.assertIsNot(selected, raw)
        self.assertFalse(detail["accepted"])
        for support in ([3., 3.], [-1., 7.], [float("nan"), 7.], [True, 7.], [3.], "3,7"):
            with self.subTest(support=support), self.assertRaises(ValueError):
                support_guard_path_proposal(raw, proposed, support)
        far = points(); far[2][1] = .01
        with self.assertRaises(ValueError):
            support_guard_path_proposal(raw, far, [3., 7.])
        invalid = points(); invalid[0][1] = float("inf")
        with self.assertRaises(ValueError):
            support_guard_path_proposal(raw, invalid, [3., 7.])


class SupportedOpticalTests(unittest.TestCase):
    def test_observed_local_boundary_support_is_measured_and_output_is_serializable(self):
        base = RGBPathCorrection()
        correction = SupportedRGBPathCorrection(base)
        raw = points(.3)
        selected, detail = correction.apply(paint(start=3.5, end=7.), raw)
        self.assertEqual(detail["path_correction"]["method"], "local-near-boundary")
        support = detail["supported_path_correction"]
        self.assertEqual(support["observed_support"]["source"], "observed-local-polynomial-center-support")
        self.assertTrue(support["accepted"])
        self.assertEqual(selected[0], raw[0])
        self.assertEqual(support["support_weights"][0], 0.)
        self.assertLess(abs(selected[1][1]), .3)
        self.assertEqual(selected[2:], raw[2:])
        json.dumps(detail, allow_nan=False)
        repeated, duplicate = correction.apply(paint(start=3.5, end=7.), raw)
        self.assertEqual(repeated, selected)
        self.assertEqual(duplicate, detail)

    def test_far_observed_dash_is_reidentified_but_does_not_justify_near_extrapolation(self):
        image, raw = paint(start=9., end=12.), points(.4)
        legacy, detail = GuardedRGBPathCorrection(RGBPathCorrection()).apply(image, raw)
        self.assertNotEqual(legacy, raw)
        selected, new_detail = SupportedRGBPathCorrection().apply(image, raw)
        support = new_detail["supported_path_correction"]
        self.assertEqual(new_detail["path_correction"], detail["path_correction"])
        self.assertEqual(support["observed_support"]["source"],
            "reidentified-observed-RGB-component-center-support")
        self.assertGreater(support["center_support_forward_m"][0], 8.)
        self.assertEqual(support["support_weights"], [0., 0.])
        self.assertEqual(selected, raw)

    def test_absent_or_unreidentified_paint_does_not_become_an_observed_span(self):
        correction, raw = SupportedRGBPathCorrection(), points(.2)
        image = np.full((360, 640, 3), 45, np.uint8)
        selected, detail = correction.apply(image, raw)
        self.assertEqual(selected, raw)
        self.assertFalse(detail["supported_path_correction"]["accepted"])
        proposal = copy.deepcopy(raw); proposal[0][1] -= .1
        fake = {"available": True, "method": "paint-normal-and-heading-alignment-with-arc-resampling",
            "observed_paint_center_ego_m": [9., -2.]}
        with patch.object(correction.base, "apply", return_value=(proposal, fake)):
            selected, detail = correction.apply(image, raw)
        self.assertEqual(selected, raw)
        self.assertEqual(detail["supported_path_correction"]["observed_support"]["reason"],
            "selected-component-not-uniquely-reidentified")

    def test_changed_base_config_and_input_mutation_are_refused(self):
        correction = SupportedRGBPathCorrection()
        correction.base.config = replace(PathCorrectionConfig(), blend=.5)
        with self.assertRaises(ValueError):
            correction.apply(paint(), points())
        correction = SupportedRGBPathCorrection()
        def mutate(_rgb, raw, **_kwargs):
            raw[0][1] = .1
            return raw, {"available": False}
        with patch.object(correction.base, "apply", side_effect=mutate), self.assertRaises(ValueError):
            correction.apply(paint(), points())


class SupportedRuntimeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def base(self):
        torch.manual_seed(3)
        raw = SimpleNamespace(model=TinyWaypointModel().eval(), config=load_learning_config(),
            identity={"variant": "hybrid_geometry", "model_version": "synthetic-model-version"})
        corrector = RGBPathCorrection()
        identity = {"variant": "hybrid_geometry_path_near", "model_version": "synthetic-model-version",
            "checkpoint_sha256": "1" * 64, "correction_config_sha256": corrector.config.sha256}
        return SimpleNamespace(raw=raw, corrector=corrector, correction="path", identity=identity,
            timer=lambda: PipelineFrameTimer(device="cpu"))

    def test_one_CPU_forward_and_proposal_with_all_guard_work_in_original_timing_stage(self):
        base = self.base()
        identity, corrector = copy.deepcopy(base.identity), base.corrector
        wrapped = SupportedPathPredictor(base)
        calls, corrections = [], []
        handle = base.raw.model.register_forward_hook(lambda *_args: calls.append(1))
        self.addCleanup(handle.remove)
        def original_proposal(_rgb, raw, **_kwargs):
            corrections.append(1)
            return copy.deepcopy(raw), {"available": False, "method": "unchanged-learned-path"}
        timer = wrapped.timer()
        timer.measure("bgra_conversion_ms", lambda: None)
        with patch.object(corrector, "apply", side_effect=original_proposal):
            prediction = wrapped.predict_profiled(np.zeros((360, 640, 3), np.uint8), timer)
        timer.measure("controller_ms", lambda: None)
        timing = validate_pipeline_timing(timer.finish())
        self.assertEqual(prediction["status"], "PASS")
        self.assertEqual((len(calls), len(corrections)), (1, 1))
        self.assertEqual(timing["executed_stages"], list(STAGES))
        self.assertEqual(prediction["waypoints_ego_m"], prediction["learned_waypoints_ego_m"])
        self.assertIn("supported_path_correction", prediction["diagnostics"])
        self.assertIn("guarded_path_correction", prediction["diagnostics"])
        self.assertEqual(base.identity, identity)
        self.assertIs(base.corrector, corrector)
        changed = wrapped.identity; changed["supported_config"]["maximum_extrapolation_m"] = .2
        self.assertEqual(wrapped.identity["supported_config"], SupportedPathConfig().to_dict())
        json.dumps(prediction, allow_nan=False)

    def test_failure_has_no_retained_success_or_supported_diagnostic(self):
        wrapped = SupportedPathPredictor(self.base())
        self.assertEqual(wrapped.predict(np.zeros((360, 640, 3), np.uint8))["status"], "PASS")
        failed = wrapped.predict(np.zeros((360, 640, 3), float))
        self.assertEqual(failed["status"], "FAIL")
        self.assertIsNone(failed["waypoints_ego_m"])
        self.assertNotIn("supported_path_correction", failed["diagnostics"])


if __name__ == "__main__":
    unittest.main()
