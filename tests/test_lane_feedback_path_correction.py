"""Episode state, optical fallback and bounded feedback runtime contracts."""
import copy
from dataclasses import FrozenInstanceError
import inspect
import unittest
from unittest.mock import patch

import numpy as np
import torch

from carla_tools.guarded_path_correction import guard_path_proposal
from carla_tools.lane_feedback_path_correction import (LaneFeedbackConfig,
    LaneFeedbackPathPredictor, LaneFeedbackRGBPathCorrection)
from carla_tools.lane_pose_observer import LanePoseObserverConfig
from carla_tools.sharp_turn_path_correction import SharpTurnConfig, SharpTurnRGBPathCorrection, _guard
from carla_tools.supported_path_correction import _shape_change
from carla_tools.pipeline_latency import STAGES, validate_pipeline_timing
import test_adaptive_path_correction as fixtures


class FakeObserver:
    """A raw-independent RGB observer with a pure raw-dependent estimate."""
    def __init__(self, *, confidence=1., shift=.1, available=True):
        self.config=LanePoseObserverConfig()
        self.confidence,self.shift,self.available=confidence,shift,available
        self.reset()

    def reset(self):
        self.observed=[]

    def observe(self, rgb, *, width=640, height=360):
        self.observed.append(bytes(rgb))

    def estimate(self, points):
        proposal=copy.deepcopy(points)
        for index in range(2):proposal[index][1]+=self.shift
        return {'available':self.available,'confidence':self.confidence,
            'observation_index':len(self.observed),'proposed_waypoints_ego_m':proposal,
            'observed_boundary_min_x_m':2.,'observed_boundary_max_x_m':6.,
            'normal_fit_residual_p95_m':.01}


class LaneFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads();torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def image(self):
        return np.zeros((360,640,3),np.uint8)

    def corrector(self, **kwargs):
        observer=FakeObserver(**kwargs)
        return LaneFeedbackRGBPathCorrection(observer=observer),observer

    def apply_frame(self, corrector, image, points):
        corrector.begin_frame()
        return corrector.apply(image,points)

    def test_configuration_is_strict_frozen_and_has_no_privileged_inputs(self):
        cfg=LaneFeedbackConfig()
        self.assertEqual(LaneFeedbackConfig.from_dict(cfg.to_dict()),cfg)
        self.assertEqual(LaneFeedbackConfig.fromdict(cfg.to_dict()).sha256,cfg.sha256)
        self.assertEqual(cfg.sharp_turn_config,SharpTurnConfig(fade_start_rad=.5,fade_end_rad=.7))
        with self.assertRaises(FrozenInstanceError):cfg.version='other'
        for bad in ({},dict(cfg.to_dict(),route=[]),dict(cfg.to_dict(),version='other')):
            with self.subTest(bad=bad),self.assertRaises(ValueError):LaneFeedbackConfig.from_dict(bad)
        self.assertEqual(list(inspect.signature(LaneFeedbackRGBPathCorrection.apply).parameters),
            ['self','rgb','points','width','height'])
        for key in ('route','pose','speed','timestamp','ground_truth'):
            with self.subTest(key=key),self.assertRaises(TypeError):
                self.corrector()[0].apply(self.image(),fixtures.straight(),**{key:[]})

    def test_same_pixels_in_distinct_frames_advance_but_replay_does_not(self):
        corrector,observer=self.corrector()
        points=fixtures.straight()
        first,detail=self.apply_frame(corrector,self.image(),points)
        repeated,replay_detail=corrector.apply(self.image(),points)
        self.assertEqual(first,repeated);self.assertEqual(detail,replay_detail)
        self.assertEqual(len(observer.observed),1)
        self.apply_frame(corrector,self.image(),points)
        self.assertEqual(len(observer.observed),2)
        self.assertEqual(detail['supported_path_correction']['lane_feedback']['observation_count'],1)

    def test_recorded_raw_reentry_recomputes_points_with_identical_RGB_state(self):
        corrector,observer=self.corrector()
        raw=fixtures.straight(.2)
        selected,detail=self.apply_frame(corrector,self.image(),raw)
        other=fixtures.straight(.21)
        second,detail2=corrector.apply(self.image(),other)
        self.assertEqual(len(observer.observed),1)
        self.assertNotEqual(selected,second)
        self.assertAlmostEqual(second[0][1]-selected[0][1],.01)
        self.assertEqual(second[2:],other[2:])
        self.assertEqual(detail['supported_path_correction']['lane_feedback']['observed_rgb_sha256'],
            detail2['supported_path_correction']['lane_feedback']['observed_rgb_sha256'])

    def test_new_image_without_frame_boundary_is_rejected_and_reset_is_reproducible(self):
        corrector,observer=self.corrector()
        with self.assertRaises(ValueError):corrector.apply(self.image(),fixtures.straight())
        first=self.apply_frame(corrector,self.image(),fixtures.straight())
        changed=self.image();changed[0,0,0]=1
        with self.assertRaises(ValueError):corrector.apply(changed,fixtures.straight())
        corrector.reset()
        self.assertEqual(len(observer.observed),0)
        self.assertEqual(first,self.apply_frame(corrector,self.image(),fixtures.straight()))

    def test_unavailable_or_zero_confidence_is_exact_original_optical_fallback(self):
        rgb=fixtures.paint(start=9.,end=12.)
        points=fixtures.straight(.3)
        expected,original=SharpTurnRGBPathCorrection(config=SharpTurnConfig(fade_start_rad=.5,fade_end_rad=.7)).apply(rgb,points)
        for change in ({'confidence':0.},{'available':False}):
            corrector,_=self.corrector(**change)
            actual,detail=self.apply_frame(corrector,rgb,points)
            self.assertEqual(actual,expected)
            self.assertEqual(detail['path_correction'],original['path_correction'])
            self.assertEqual(detail['guarded_path_correction'],original['guarded_path_correction'])
            support=copy.deepcopy(detail['supported_path_correction']);support.pop('lane_feedback')
            self.assertEqual(support,original['supported_path_correction'])

    def test_continuous_observation_confidence_blends_proposals_without_new_gain(self):
        points=fixtures.straight()
        outputs=[]
        for confidence in (0.,.25,.5,.75,1.):
            corrector,_=self.corrector(confidence=confidence,shift=.2)
            output,_=self.apply_frame(corrector,self.image(),points)
            outputs.append(output)
            self.assertAlmostEqual(output[0][1],confidence*.2)
            self.assertEqual(output[2:],points[2:])
        self.assertEqual(outputs[0],points)
        self.assertEqual(outputs[-1][0],[3.,.2])

    def test_full_temporal_proposal_has_no_longitudinal_phase_change(self):
        corrector,_=self.corrector(confidence=1.,shift=.1)
        raw=fixtures.straight(.1)
        proposal=copy.deepcopy(raw)
        proposal[0][0]+=.1;proposal[1][0]+=.1
        baseline,guard,detail=_guard(raw,proposal,3.,.01,corrector.config.sharp_turn_config)
        _,old=corrector._fallback.apply(self.image(),raw)
        old['guarded_path_correction']=guard
        old['supported_path_correction']=dict(detail,proposed_waypoints_ego_m=proposal,
            center_support_forward_m=[2.,5.])
        with patch.object(corrector._fallback,'apply',return_value=(baseline,old)):
            selected,_=self.apply_frame(corrector,self.image(),raw)
        self.assertEqual([p[0] for p in selected],[p[0] for p in raw])

    def test_disagreement_boundary_is_continuous_and_original_guards_hold(self):
        outputs=[]
        for shift in (.499999,.5,.500001,1.2):
            corrector,_=self.corrector(shift=shift)
            raw=fixtures.straight()
            before=copy.deepcopy(raw)
            selected,_=self.apply_frame(corrector,self.image(),raw)
            self.assertEqual(raw,before);self.assertEqual(selected[2:],raw[2:])
            self.assertTrue(guard_path_proposal(raw,selected)[1]['accepted'])
            shape=_shape_change(raw,selected)
            self.assertLessEqual(shape['maximum_segment_heading_change_rad'],.12)
            self.assertLessEqual(shape['maximum_signed_curvature_change_per_m'],.04)
            outputs.append(np.asarray(selected))
        self.assertLess(np.max(np.abs(outputs[0]-outputs[2])),1e-5)
        self.assertTrue(np.isfinite(outputs[-1]).all())

    def test_bad_observation_cannot_change_distant_points_or_confidence_domain(self):
        for confidence in (float('nan'),-1.,1.1,True):
            corrector,_=self.corrector(confidence=confidence)
            with self.assertRaises(ValueError):self.apply_frame(corrector,self.image(),fixtures.straight())
        corrector,observer=self.corrector()
        original=observer.estimate
        def bad(points):
            value=original(points);value['proposed_waypoints_ego_m'][-1][1]+=.01
            return value
        observer.estimate=bad
        with self.assertRaises(ValueError):self.apply_frame(corrector,self.image(),fixtures.straight())

    def test_single_timed_forward_reset_and_failed_frame_cleanup(self):
        base=fixtures.AdaptiveRuntimeTests().base()
        predictor=LaneFeedbackPathPredictor(base)
        observer=FakeObserver(available=False)
        predictor.corrector=LaneFeedbackRGBPathCorrection(base.corrector,predictor.config,observer=observer)
        calls=[]
        handle=base.raw.model.register_forward_hook(lambda *_:calls.append(1));self.addCleanup(handle.remove)
        timer=predictor.timer();timer.measure('bgra_conversion_ms',lambda:None)
        first=predictor.predict_profiled(self.image(),timer)
        timer.measure('controller_ms',lambda:None)
        timing=validate_pipeline_timing(timer.finish())
        self.assertEqual(first['status'],'PASS');self.assertEqual(len(calls),1)
        self.assertEqual(len(observer.observed),1);self.assertEqual(timing['executed_stages'],list(STAGES))
        self.assertTrue(predictor.identity['requires_episode_reset'])
        self.assertEqual(set(first['diagnostics']) & {'path_correction','guarded_path_correction','supported_path_correction'},
            {'path_correction','guarded_path_correction','supported_path_correction'})
        failed=predictor.predict(np.zeros((360,640,3),float))
        self.assertEqual(failed['status'],'FAIL');self.assertIsNone(failed['waypoints_ego_m'])
        self.assertNotIn('supported_path_correction',failed['diagnostics'])
        self.assertFalse(predictor.corrector._pending_frame)
        self.assertEqual(predictor.predict(self.image())['status'],'PASS')
        predictor.reset();self.assertEqual(len(observer.observed),0)
        identity=predictor.identity;identity['requires_episode_reset']=False
        self.assertTrue(predictor.identity['requires_episode_reset'])


if __name__=='__main__':unittest.main()
