"""CPU geometry, optical evidence and runtime checks for adaptive RGB fusion."""
import copy
from dataclasses import FrozenInstanceError, replace
import inspect
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from carla_tools.adaptive_path_correction import (AdaptivePathConfig, AdaptivePathPredictor,
    AdaptiveRGBPathCorrection, adaptive_config_from_dict, adaptive_guard_path_proposal)
from carla_tools.guarded_path_correction import guard_path_proposal
from carla_tools.learning_data import load_learning_config
from carla_tools.path_correction import PathCorrectionConfig, RGBPathCorrection
from carla_tools.pipeline_latency import PipelineFrameTimer, STAGES, validate_pipeline_timing
from test_path_correction import paint
from test_recovery_training import TinyWaypointModel


def straight(y=0.):
    return [[float(x), y] for x in (3, 6, 9, 12, 15)]


def curve():
    return [[3., 1.], [5., 3.2], [6.5, 5.8], [7.8, 8.4], [9.1, 11.1]]


class AdaptiveConfigurationTests(unittest.TestCase):
    def test_strict_frozen_configuration_and_RGB_only_signature(self):
        cfg = AdaptivePathConfig()
        self.assertEqual(adaptive_config_from_dict(cfg.to_dict()), cfg)
        self.assertEqual(cfg.sha256, AdaptivePathConfig().sha256)
        with self.assertRaises(FrozenInstanceError):
            cfg.trusted_span_m = 8.
        for change in ({"maximum_first_point_change_m": .6}, {"maximum_near_mean_change_m": .6},
                {"trusted_span_m": True}, {"trusted_span_m": float("nan")}, {"scale_bisection_steps": True},
                {"version": "unknown"}, {"schema_version": True}, {"route": []}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                adaptive_config_from_dict(dict(cfg.to_dict(), **change))
        self.assertEqual(list(inspect.signature(AdaptiveRGBPathCorrection.apply).parameters),
            ["self", "rgb", "points", "width", "height"])
        for key in ("ground_truth", "route", "pose", "speed", "previous_path"):
            with self.subTest(key=key), self.assertRaises(TypeError):
                adaptive_guard_path_proposal(straight(), straight(), 3., **{key: []})


class AdaptiveGeometryTests(unittest.TestCase):
    def test_budget_boundary_is_continuous_on_both_sides(self):
        selected = []
        for shift in (.499999, .5, .500001):
            raw, proposal = straight(), straight()
            proposal[0][1] = shift; proposal[1][1] = .2
            points, decision, detail = adaptive_guard_path_proposal(raw, proposal, 3.)
            self.assertTrue(decision["original_output_bounds_verified"])
            self.assertTrue(detail["accepted"])
            self.assertTrue(guard_path_proposal(raw, points)[1]["accepted"])
            selected.append(np.asarray(points))
        self.assertLess(float(np.max(np.abs(selected[0] - selected[-1]))), .00001)

    def test_original_output_budgets_finiteness_immutability_and_far_points(self):
        for dx in np.linspace(-.4, .7, 8):
            for dy in np.linspace(-.6, .6, 7):
                raw = straight(.1); proposed = copy.deepcopy(raw)
                proposed[0][0] += float(dx); proposed[0][1] += float(dy)
                proposed[1][1] -= float(dy) / 2
                before = copy.deepcopy((raw, proposed))
                selected, decision, _ = adaptive_guard_path_proposal(raw, proposed, 2.8)
                self.assertEqual((raw, proposed), before)
                self.assertEqual(selected[2:], raw[2:])
                self.assertTrue(np.isfinite(selected).all())
                self.assertTrue(guard_path_proposal(raw, selected)[1]["accepted"])
                self.assertLessEqual(decision["near_point_change_m"][0], .5)
                self.assertLessEqual(decision["near_mean_change_m"], .5)

    def test_small_trusted_proposal_preserves_current_correction_and_common_blend(self):
        raw = straight(); proposed = straight(); proposed[0][1] = proposed[1][1] = .1
        selected, _, detail = adaptive_guard_path_proposal(raw, proposed, 3.)
        self.assertEqual(selected, proposed)
        self.assertEqual(detail["common_scale"], 1.)
        self.assertEqual(detail["quality"]["geometry_confidence"], 1.)
        proposal = curve(); raw = copy.deepcopy(proposal)
        proposal[0][0] += .3; proposal[0][1] += .12; proposal[1][1] += .15
        selected, _, detail = adaptive_guard_path_proposal(raw, proposal, 1.8)
        for index in range(2):
            for axis in range(2):
                self.assertAlmostEqual(selected[index][axis] - raw[index][axis],
                    detail["common_scale"] * (proposal[index][axis] - raw[index][axis]), places=12)

    def test_short_observation_reduces_turn_weakening_even_with_positive_lateral_shift(self):
        raw, proposal = curve(), curve()
        proposal[0][0] += .3; proposal[0][1] += .12
        short, _, a = adaptive_guard_path_proposal(raw, proposal, 1.)
        long, _, b = adaptive_guard_path_proposal(raw, proposal, 3.)
        self.assertGreater(a["quality"]["raw_signed_near_curvature_per_m"], .04)
        self.assertGreater(a["quality"]["maximum_turn_weakening_per_m"], .015)
        self.assertEqual(a["quality"]["geometry_confidence"], 0.)
        self.assertEqual(short, raw)
        self.assertEqual(b["quality"]["geometry_confidence"], 1.)
        self.assertGreater(np.linalg.norm(np.asarray(long) - raw), 0.)

    def test_local_lateral_turn_weakening_is_reduced_and_turn_strengthening_is_retained(self):
        raw, proposal = curve(), curve(); proposal[0][1] -= .2
        _, _, weak = adaptive_guard_path_proposal(raw, proposal, 1.)
        proposal = curve(); proposal[0][1] += .1
        _, _, strong = adaptive_guard_path_proposal(raw, proposal, 1.)
        self.assertLess(weak["quality"]["geometry_confidence"], .1)
        self.assertEqual(strong["quality"]["geometry_confidence"], 1.)

    def test_left_right_mirror_uses_the_same_geometric_risk(self):
        raw, proposal = curve(), curve(); proposal[0][0] += .25; proposal[0][1] += .08
        positive, _, a = adaptive_guard_path_proposal(raw, proposal, 1.7)
        mirror = lambda points: [[x, -y] for x, y in points]
        negative, _, b = adaptive_guard_path_proposal(mirror(raw), mirror(proposal), 1.7)
        self.assertEqual(negative, mirror(positive))
        self.assertAlmostEqual(a["quality"]["geometry_confidence"], b["quality"]["geometry_confidence"])

    def test_short_span_does_not_suppress_a_straight_recovery(self):
        raw, proposal = straight(.3), straight(.3)
        proposal[0][1] = proposal[1][1] = .15
        selected, _, detail = adaptive_guard_path_proposal(raw, proposal, .6, .08)
        self.assertEqual(detail["quality"]["geometry_confidence"], 1.)
        self.assertEqual(selected, proposal)

    def test_bad_evidence_or_distant_modification_is_rejected(self):
        for span in (None, 0., -1., True, float("inf")):
            with self.subTest(span=span), self.assertRaises(ValueError):
                adaptive_guard_path_proposal(straight(), straight(), span)
        for residual in (-1., True, float("nan")):
            with self.subTest(residual=residual), self.assertRaises(ValueError):
                adaptive_guard_path_proposal(straight(), straight(), 3., residual)
        proposal = straight(); proposal[-1][1] = .1
        with self.assertRaises(ValueError):
            adaptive_guard_path_proposal(straight(), proposal, 3.)


class AdaptiveOpticalTests(unittest.TestCase):
    def test_distant_visible_straight_dash_retains_useful_blind_zone_alignment(self):
        raw = straight(.3)
        selected, detail = AdaptiveRGBPathCorrection().apply(paint(start=9., end=12.), raw)
        adaptive = detail["supported_path_correction"]
        self.assertEqual(detail["path_correction"]["method"], "paint-normal-and-heading-alignment-with-arc-resampling")
        self.assertGreater(adaptive["center_support_forward_m"][0] - raw[0][0], 4.)
        self.assertEqual(adaptive["quality"]["geometry_confidence"], 1.)
        self.assertLess(abs(selected[0][1]), abs(raw[0][1]))
        self.assertEqual(selected[2:], raw[2:])
        json.dumps(detail, allow_nan=False)

    def test_no_evidence_returns_fresh_raw_without_mutation(self):
        raw = curve(); before = copy.deepcopy(raw)
        selected, detail = AdaptiveRGBPathCorrection().apply(np.zeros((360, 640, 3), np.uint8), raw)
        self.assertEqual(selected, raw); self.assertIsNot(selected, raw)
        self.assertEqual(raw, before)
        self.assertFalse(detail["supported_path_correction"]["accepted"])
        self.assertEqual(detail["supported_path_correction"]["rejection_reasons"], ["observed-support-unavailable"])

    def test_changed_base_configuration_or_mutating_proposal_is_refused(self):
        c = AdaptiveRGBPathCorrection(); c.base.config = replace(PathCorrectionConfig(), blend=.5)
        with self.assertRaises(ValueError):c.apply(paint(), straight())
        c = AdaptiveRGBPathCorrection()
        def bad(_rgb, points, **kwargs):
            points[0][1] = .1
            return points, {"available": False}
        with patch.object(c.base, "apply", side_effect=bad), self.assertRaises(ValueError):
            c.apply(paint(), straight())


class AdaptiveRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads(); torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.old_threads)

    def base(self):
        torch.manual_seed(3)
        raw = SimpleNamespace(model=TinyWaypointModel().eval(), config=load_learning_config(),
            identity={"variant": "hybrid_geometry", "model_version": "synthetic-model-version"})
        corrector = RGBPathCorrection()
        identity = {"variant": "hybrid_geometry_path_near", "model_version": "synthetic-model-version",
            "checkpoint_sha256": "1" * 64, "correction_config_sha256": corrector.config.sha256}
        return SimpleNamespace(raw=raw, corrector=corrector, correction="path", identity=identity,
            timer=lambda: PipelineFrameTimer(device="cpu"))

    def test_exactly_one_forward_and_proposal_in_existing_timing_stage(self):
        base = self.base(); original_identity = copy.deepcopy(base.identity)
        predictor = AdaptivePathPredictor(base); forwards, proposals = [], []
        hook = base.raw.model.register_forward_hook(lambda *_args:forwards.append(1)); self.addCleanup(hook.remove)
        def proposal(_rgb, raw, **kwargs):
            proposals.append(1)
            return copy.deepcopy(raw), {"available": False, "method": "unchanged-learned-path"}
        timer = predictor.timer(); timer.measure("bgra_conversion_ms", lambda:None)
        with patch.object(base.corrector, "apply", side_effect=proposal):
            result = predictor.predict_profiled(np.zeros((360, 640, 3), np.uint8), timer)
        timer.measure("controller_ms", lambda:None)
        timing = validate_pipeline_timing(timer.finish())
        self.assertEqual(result["status"], "PASS"); self.assertEqual((len(forwards), len(proposals)), (1, 1))
        self.assertEqual(timing["executed_stages"], list(STAGES))
        self.assertEqual(base.identity, original_identity)
        self.assertEqual(predictor.correction, "supported_path")
        for key in ("path_correction", "guarded_path_correction", "supported_path_correction"):
            self.assertIn(key, result["diagnostics"])
        mutated = predictor.identity; mutated["adaptive_config"]["trusted_span_m"] = 5.
        self.assertEqual(predictor.identity["adaptive_config"], AdaptivePathConfig().to_dict())

    def test_failed_frame_does_not_retain_an_earlier_success(self):
        predictor = AdaptivePathPredictor(self.base())
        self.assertEqual(predictor.predict(np.zeros((360, 640, 3), np.uint8))["status"], "PASS")
        failed = predictor.predict(np.zeros((360, 640, 3), float))
        self.assertEqual(failed["status"], "FAIL"); self.assertIsNone(failed["waypoints_ego_m"])
        self.assertNotIn("supported_path_correction", failed["diagnostics"])


if __name__ == "__main__":unittest.main()
