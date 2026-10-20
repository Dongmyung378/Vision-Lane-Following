"""CPU-only guard checks: current RGB paths, exact provenance and serial timing."""
import copy
from dataclasses import FrozenInstanceError
import inspect
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from carla_tools.guarded_path_correction import (GuardConfig, GuardedPathConfig, GuardedPathPredictor,
    GuardedRGBPathCorrection, build_guarded_predictor, guard_config_from_dict, guard_path_proposal,
    guarded_preprocessing_config, path_preprocessing_config, validate_preprocessing_config,
    validate_recorded_guarded_correction, wrap_preprocessing)
from carla_tools.learning_data import load_learning_config
from carla_tools.path_correction import RGBPathCorrection
from carla_tools.pipeline_latency import PipelineFrameTimer, STAGES, validate_pipeline_timing
from test_recovery_training import TinyWaypointModel


def points():
    return [[3. * index, 0.] for index in range(1, 6)]


class GuardConfigurationTests(unittest.TestCase):
    def test_strict_config_roundtrip_and_frozen_no_GTruth_inputs(self):
        config = GuardConfig()
        self.assertEqual(config, GuardedPathConfig())
        self.assertEqual(guard_config_from_dict(config.to_dict()), config)
        self.assertEqual(config.maximum_first_point_change_m, .5)
        with self.assertRaises(FrozenInstanceError):
            config.maximum_first_point_change_m = .2
        for change in ({"maximum_first_point_change_m": float("nan")}, {"maximum_near_mean_change_m": True},
                       {"maximum_first_point_change_m": .01}, {"version": "other"}, {"schema_version": True}, {"GT": []}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                guard_config_from_dict(dict(config.to_dict(), **change))
        self.assertEqual(list(inspect.signature(guard_path_proposal).parameters), ["learned_points", "proposed_points", "config"])
        with self.assertRaises(TypeError):
            guard_path_proposal(points(), points(), ground_truth=points())
        with self.assertRaises(TypeError):
            guard_path_proposal(points(), points(), previous_path=points())
        for value in ({}, False, 0, "default"):
            with self.subTest(config=value), self.assertRaises(ValueError):
                guard_path_proposal(points(), points(), value)

    def test_preprocessing_source_configs_are_strict_and_copied(self):
        for value in (path_preprocessing_config(), guarded_preprocessing_config()):
            self.assertEqual(validate_preprocessing_config(value), value)
            self.assertIsNot(validate_preprocessing_config(value), value)
            changed = dict(value, extra=True)
            with self.assertRaises(ValueError):
                validate_preprocessing_config(changed)
            changed = dict(value, correction_config_sha256="0" * 64)
            with self.assertRaises(ValueError):
                validate_preprocessing_config(changed)
        changed = guarded_preprocessing_config(); changed["guard_config_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_preprocessing_config(changed)


class PureGuardTests(unittest.TestCase):
    def test_large_current_proposal_returns_current_raw_and_small_proposal_is_preserved(self):
        raw, large = points(), points()
        large[0][1], large[1][1] = -.9, -.7
        selected, decision = guard_path_proposal(raw, large)
        self.assertEqual(selected, raw)
        self.assertIsNot(selected, raw)
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["selected_source"], "current-learned-RGB-path")
        self.assertEqual(set(decision["rejection_reasons"]), {"first-point-disagreement", "near-mean-disagreement"})
        small = points(); small[0][1], small[1][1] = .1, .2
        selected, decision = guard_path_proposal(raw, small)
        self.assertTrue(decision["accepted"])
        self.assertEqual(selected, small)
        self.assertEqual(selected[2:], raw[2:])
        self.assertEqual(raw, points())

    def test_second_point_mean_rule_boundaries_and_invalid_paths(self):
        raw, proposal = points(), points()
        proposal[1][1] = 1.1
        selected, decision = guard_path_proposal(raw, proposal)
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["rejection_reasons"], ["near-mean-disagreement"])
        proposal = points(); proposal[0][1] = .5
        selected, decision = guard_path_proposal(raw, proposal)
        self.assertTrue(decision["accepted"])
        proposal = points(); proposal[2][1] += 1e-9
        selected, _ = guard_path_proposal(raw, proposal)
        self.assertEqual(selected[2:], raw[2:])
        for mutation in ("far", "nan", "shape", "unordered"):
            proposal = points()
            if mutation == "far": proposal[2][1] = .01
            elif mutation == "nan": proposal[0][0] = float("nan")
            elif mutation == "shape": proposal.pop()
            else: proposal[1] = [2., 0.]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                guard_path_proposal(raw, proposal)


class GuardedRuntimeCPUIntegrationTests(unittest.TestCase):
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

    def test_one_actual_CPU_forward_original_correction_guard_same_interval_and_base_unchanged(self):
        base = self.base()
        identity, corrector = copy.deepcopy(base.identity), base.corrector
        wrapped = build_guarded_predictor(base)
        self.assertIs(wrapped.base, base)
        self.assertIs(wrapped.raw_predictor, base.raw)
        self.assertIs(wrapped.corrector.base, corrector)
        self.assertEqual(wrapped.identity["correction_config_sha256"], identity["correction_config_sha256"])
        self.assertEqual(wrapped.identity["guard_config_sha256"], wrapped.config.sha256)
        changed = wrapped.identity; changed["guard_config"]["maximum_first_point_change_m"] = .1
        self.assertEqual(wrapped.identity["guard_config"], wrapped.config.to_dict())
        forward_calls, correction_calls = [], []
        handle = base.raw.model.register_forward_hook(lambda *_args: forward_calls.append(1))
        self.addCleanup(handle.remove)
        def proposal(_rgb, learned, **kwargs):
            correction_calls.append(1)
            result = copy.deepcopy(learned)
            result[0][1] -= .9; result[1][1] -= .7
            return result, {"version": "synthetic-legacy", "config_sha256": corrector.config.sha256}
        image = np.zeros((360, 640, 3), np.uint8)
        timer = wrapped.timer()
        timer.measure("bgra_conversion_ms", lambda: None)
        with patch.object(corrector, "apply", side_effect=proposal):
            prediction = wrapped.predict_profiled(image, timer)
            timer.measure("controller_ms", lambda: None)
            timing = validate_pipeline_timing(timer.finish())
            self.assertEqual(timing["executed_stages"], list(STAGES))
            self.assertEqual((len(forward_calls), len(correction_calls)), (1, 1))
            self.assertEqual(prediction["status"], "PASS")
            self.assertEqual(prediction["waypoints_ego_m"], prediction["learned_waypoints_ego_m"])
            self.assertFalse(prediction["diagnostics"]["guarded_path_correction"]["accepted"])
            self.assertEqual(prediction["diagnostics"]["path_correction"]["version"], "synthetic-legacy")
            self.assertGreater(timing["stages_ms"]["near_correction_ms"], 0.)
            self.assertEqual(validate_recorded_guarded_correction(wrapped, image, prediction)["status"], "PASS")
            self.assertEqual(len(forward_calls), 1)  # Recorded-raw replay performs no inference.
            for key in ("waypoints_ego_m", "guarded_path_correction", "path_correction"):
                changed = copy.deepcopy(prediction)
                if key == "waypoints_ego_m": changed[key][0][1] += .01
                else: changed["diagnostics"][key]["version"] = "tampered"
                with self.subTest(tampering=key), self.assertRaisesRegex(ValueError, "diagnostic-mismatch"):
                    validate_recorded_guarded_correction(wrapped, image, changed)
        self.assertEqual(base.identity, identity)
        self.assertIs(base.corrector, corrector)

    def test_failed_forward_is_retained_without_calling_correction_and_config_substitution_refused(self):
        base = self.base(); wrapped = GuardedPathPredictor(base)
        with patch.object(base.raw.model, "forward", side_effect=ValueError("synthetic-forward-failure")), \
             patch.object(base.corrector, "apply", side_effect=AssertionError("must-not-run")):
            prediction = wrapped.predict(np.zeros((360, 640, 3), np.uint8))
            self.assertEqual(prediction["status"], "FAIL")
            self.assertIsNone(prediction["waypoints_ego_m"])
            self.assertIsNone(prediction["learned_waypoints_ego_m"])
        with self.assertRaises(ValueError):
            GuardedPathPredictor(SimpleNamespace(correction="near"))
        with self.assertRaises(ValueError):
            validate_recorded_guarded_correction(wrapped, np.zeros((360, 640, 3), np.uint8), None)
        self.assertIs(wrap_preprocessing(base, path_preprocessing_config()), base)
        self.assertIsInstance(wrap_preprocessing(base, guarded_preprocessing_config()), GuardedPathPredictor)
        before = base.corrector.config
        from dataclasses import replace
        base.corrector.config = replace(before, blend=.5)
        with self.assertRaisesRegex(ValueError, "base-config-changed"):
            wrapped.corrector.apply(np.zeros((360, 640, 3), np.uint8), points())


if __name__ == "__main__":
    unittest.main()
