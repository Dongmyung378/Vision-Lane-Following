"""Synthetic geometry, optical evidence and single-forward runtime contracts."""
import copy
from dataclasses import FrozenInstanceError, replace
import inspect
import json
import math
import unittest

import numpy as np
import torch

from carla_tools import path_correction
from carla_tools.centering_path_correction import (CenteringPathConfig, CenteringRGBPathCorrection,
    CenteringRGBProposal, CenteringPathPredictor, blindzone_arc_anchors,
    curvature_aware_learned_curve, longitudinal_confidence)
from carla_tools.guarded_path_correction import guard_path_proposal
from carla_tools.pipeline_latency import STAGES, validate_pipeline_timing
import test_adaptive_path_correction as fixtures


def circle(curvature, offset=(0., 0.)):
    return [[math.sin(curvature * s) / curvature + offset[0],
        (1. - math.cos(curvature * s)) / curvature + offset[1]] for s in (3., 6., 9., 12., 15.)]


class CenteringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_frozen_config_strict_roundtrip_and_no_privileged_inputs(self):
        cfg = CenteringPathConfig()
        self.assertEqual(CenteringPathConfig.fromdict(cfg.to_dict()), cfg)
        with self.assertRaises(FrozenInstanceError):
            cfg.version = "different"
        for bad in ({}, dict(cfg.to_dict(), route=[]), dict(cfg.to_dict(), version="different")):
            with self.assertRaises(ValueError):
                CenteringPathConfig.from_dict(bad)
        self.assertEqual(set(inspect.signature(CenteringRGBPathCorrection.apply).parameters),
            {"self", "rgb", "points", "width", "height"})
        for name in ("pose", "route", "speed", "ground_truth", "previous_path"):
            with self.assertRaises(TypeError):
                CenteringRGBPathCorrection().apply(np.zeros((360,640,3),np.uint8),fixtures.straight(),**{name:[]})

    def test_arc_continuation_preserves_circle_and_translation(self):
        for k in (-.08, .08):
            points = circle(k, (.7, -.4))
            anchors, detail = blindzone_arc_anchors(points)
            expected = [[math.sin(k*s)/k+.7,(1-math.cos(k*s))/k-.4] for s in (-3.,0.,3.,6.,9.,12.,15.)]
            np.testing.assert_allclose(anchors, expected, atol=1e-12)
            self.assertAlmostEqual(detail["menger_curvature_per_m"], k)
            self.assertFalse(detail["fixed_horizon_norm_assumption_used"])
            np.testing.assert_allclose(anchors[2:], points, atol=0)

    def test_straight_limit_and_mirror_are_finite_and_continuous(self):
        straight = np.array(fixtures.straight(.4))
        anchors, _ = blindzone_arc_anchors(straight.tolist())
        np.testing.assert_allclose(anchors[:2], [[-3.,.4],[0.,.4]], atol=1e-12)
        for k in (-1e-8,1e-8):
            points = [[s, .4 + .5*k*s*s] for s in (3.,6.,9.,12.,15.)]
            curved, _ = blindzone_arc_anchors(points)
            self.assertLess(float(np.max(np.abs(curved-anchors))), 2e-6)
        points = circle(.08)
        mirror = [[x,-y] for x,y in points]
        a,_=curvature_aware_learned_curve(points,80)
        b,_=curvature_aware_learned_curve(mirror,80)
        np.testing.assert_allclose(a*np.array([1.,-1.]),b,atol=1e-12)

    def test_degenerate_and_reversing_paths_fail_safely(self):
        for points in ([[3.,0.]]*5, [[3.,0.],[6.,0.],[4.,.1],[8.,.2],[11.,.3]]):
            with self.assertRaises(ValueError):
                blindzone_arc_anchors(points)
            with self.assertRaises(ValueError):
                CenteringRGBPathCorrection().apply(np.zeros((360,640,3),np.uint8),points)

    def test_longitudinal_confidence_is_symmetric_with_smooth_endpoints(self):
        cfg=CenteringPathConfig()
        def path(heading):return [[3.,.3],[3.+3.*math.cos(heading),.3+3.*math.sin(heading)],[8.,3.],[10.,5.],[12.,7.]]
        self.assertEqual(longitudinal_confidence(path(.2),cfg),1.)
        self.assertEqual(longitudinal_confidence(path(.8),cfg),0.)
        for heading in (.5,.6,.7):
            self.assertAlmostEqual(longitudinal_confidence(path(heading),cfg),longitudinal_confidence(path(-heading),cfg))
            values=[longitudinal_confidence(path(heading+d),cfg) for d in (-1e-7,0,1e-7)]
            self.assertLess(max(values)-min(values),2e-6)

    def test_synthetic_optical_evidence_correction_and_output_bounds(self):
        corrector=CenteringRGBPathCorrection()
        raw=fixtures.straight(.3);before=copy.deepcopy(raw)
        selected,detail=corrector.apply(fixtures.paint(start=9.,end=12.),raw)
        self.assertLess(abs(selected[0][1]),abs(raw[0][1]))
        self.assertEqual(raw,before)
        self.assertEqual(selected[2:],raw[2:])
        self.assertTrue(guard_path_proposal(raw,selected)[1]["accepted"])
        shape=detail["supported_path_correction"]["shape_change"]
        self.assertLessEqual(shape["maximum_segment_heading_change_rad"],.12)
        self.assertLessEqual(shape["maximum_signed_curvature_change_per_m"],.04)
        json.dumps(detail,allow_nan=False)
        empty,no_evidence=corrector.apply(np.zeros((360,640,3),np.uint8),raw)
        self.assertEqual(empty,raw)
        self.assertIsNot(empty,raw)
        self.assertFalse(no_evidence["supported_path_correction"]["accepted"])

    def test_original_curve_globals_and_replaced_config_are_protected(self):
        original=path_correction._learned_curve
        CenteringRGBProposal().apply(fixtures.paint(),fixtures.straight(.2))
        self.assertIs(path_correction._learned_curve,original)
        self.assertIs(path_correction.RGBPathCorrection.apply.__globals__["_learned_curve"],original)
        corrector=CenteringRGBPathCorrection()
        corrector.config=replace(corrector.config,sharp_turn_config=replace(corrector.config.sharp_turn_config,fade_start_rad=.3))
        with self.assertRaises(ValueError):
            corrector.apply(fixtures.paint(),fixtures.straight())

    def test_invalid_projected_order_returns_the_original_path(self):
        raw=[[3.,1.],[3.5,4.],[5.,7.],[6.,10.],[7.,13.]]
        proposed=[[.1,5.],[5.,4.],*copy.deepcopy(raw[2:])]
        corrector=CenteringRGBProposal()
        corrector._operation=lambda *a,**k:(copy.deepcopy(proposed),{
            "available":True,"method":"paint-normal-and-heading-alignment-with-arc-resampling"})
        selected,detail=corrector.apply(np.zeros((360,640,3),np.uint8),raw)
        self.assertEqual(selected,raw)
        self.assertIsNot(selected,raw)
        self.assertFalse(detail["available"])
        self.assertEqual(detail["rejection"],"centering-longitudinal-projection-invalid-path")

    def test_one_forward_original_timing_and_three_diagnostic_keys(self):
        base=fixtures.AdaptiveRuntimeTests().base()
        predictor=CenteringPathPredictor(base)
        forwards=[]
        hook=base.raw.model.register_forward_hook(lambda *_:forwards.append(1))
        self.addCleanup(hook.remove)
        timer=predictor.timer();timer.measure("bgra_conversion_ms",lambda:None)
        result=predictor.predict_profiled(np.zeros((360,640,3),np.uint8),timer)
        timer.measure("controller_ms",lambda:None)
        timing=validate_pipeline_timing(timer.finish())
        self.assertEqual(len(forwards),1)
        self.assertEqual(result["status"],"PASS")
        self.assertEqual(timing["executed_stages"],list(STAGES))
        self.assertEqual(predictor.correction,"supported_path")
        for name in ("path_correction","guarded_path_correction","supported_path_correction"):
            self.assertIn(name,result["diagnostics"])
        identity=predictor.identity;identity["centering_config"]["version"]="changed"
        self.assertEqual(predictor.identity["centering_config"],predictor.config.to_dict())

    def test_failure_does_not_retain_previous_prediction(self):
        predictor=CenteringPathPredictor(fixtures.AdaptiveRuntimeTests().base())
        self.assertEqual(predictor.predict(np.zeros((360,640,3),np.uint8))["status"],"PASS")
        failed=predictor.predict(np.zeros((360,640,3),float))
        self.assertEqual(failed["status"],"FAIL")
        self.assertIsNone(failed["waypoints_ego_m"])
        self.assertNotIn("supported_path_correction",failed["diagnostics"])


if __name__ == "__main__":
    unittest.main()
