"""Small deterministic API safety checks; no model, simulator or GPU."""
import copy
import math
from types import SimpleNamespace
import unittest
from dataclasses import replace

from carla_tools.traffic_policy import TrafficPolicy,TrafficPolicyConfig,TrafficEvidenceBridge
from carla_tools.traffic_perception import RGBObservation,TrafficPerceptionConfig
from carla_tools.traffic_behavior import TrafficSupervisor,TrafficConfig


def calibration():
    c,s=math.cos(math.radians(-12)),math.sin(math.radians(-12))
    return {'width':640,'height':360,'intrinsics':{'fx':320.,'fy':320.,'cx':320.,'cy':180.},
            'camera_to_actor_matrix':[[c,0.,-s,1.5],[0.,1.,0.,0.],[s,0.,c,2.4],[0.,0.,0.,1.]]}


class PathPredictor:
    def __init__(self):self.failed=False
    def timer(self):return SimpleNamespace(finish=lambda:{})
    def predict_profiled(self,*args,**kwargs):
        return {'status':'FAIL' if self.failed else 'PASS','waypoints_ego_m':None if self.failed else [[x,0.] for x in (3.,6.,9.,12.,15.)]}


class Perception:
    def __init__(self):
        self.calibration=calibration();self.config=TrafficPerceptionConfig(front_bumper_x_m=2.4,ego_half_width_m=.95)
        self.signal='none';self.bad_age=False
    def reset(self):pass
    def process(self,o):
        return {'frame':o.frame+int(self.bad_age),'timestamp':o.timestamp,'objects':[],'signals':[],
                'signal':self.signal,'signal_confidence':0.,'stop_line_distance_m':None,
                'stop_line_confidence':0.,'stop_line':{},'stop_required':False,
                'stop_sign_id':None,'perception_valid':True,'GT_inputs':False}


class Controller:
    def reset(self):self.calls=0
    def step(self,*args,**kwargs):
        self.calls+=1
        return {'steer':.1,'throttle':.5,'brake':0.,'hand_brake':False,'reverse':False,
                'diagnostics':{'effective_target_speed_mps':6.}}


def observation(frame=0):return RGBObservation(bytes(640*360*3),640,360,frame,frame*.05,0.)


class CleanPolicyTests(unittest.TestCase):
    def test_rejects_untyped_input_before_controller_execution(self):
        lane=Controller();policy=TrafficPolicy(PathPredictor(),Perception(),lane)
        with self.assertRaises(ValueError):policy.command({'rgb':b'', 'world':object()})
        self.assertEqual(lane.calls,0)

    def test_reset_discards_previous_unknown_signal_hold_and_controller_state(self):
        traffic=Perception();lane=Controller();policy=TrafficPolicy(PathPredictor(),traffic,lane)
        traffic.signal='unknown';command,_=policy.command(observation())
        self.assertEqual(command['brake'],1.)
        traffic.signal='none';policy.reset();command,_=policy.command(observation())
        fresh,_=TrafficPolicy(PathPredictor(),Perception(),Controller()).command(observation())
        self.assertEqual(command,fresh);self.assertGreater(command['throttle'],0.)
        self.assertEqual(lane.calls,1)

    def test_bad_observation_age_rejected_before_controller_execution(self):
        traffic=Perception();traffic.bad_age=True;lane=Controller()
        policy=TrafficPolicy(PathPredictor(),traffic,lane)
        with self.assertRaises(ValueError):policy.command(observation())
        self.assertEqual(lane.calls,0)

    def test_failed_lane_path_requests_full_stop_and_rejects_unbound_speed_envelope(self):
        predictor=PathPredictor();predictor.failed=True;lane=Controller()
        command,diagnostic=TrafficPolicy(predictor,Perception(),lane).command(observation())
        self.assertEqual(lane.calls,0);self.assertEqual(command['throttle'],0.);self.assertEqual(command['brake'],1.)
        self.assertFalse(diagnostic['traffic_perception']['objects'])
        with self.assertRaises(ValueError):TrafficPolicyConfig(maximum_speed_mps=10.)

    def test_default_bridge_rejects_mismatched_perception_assumptions(self):
        for field,value in (('bbox_footpoint_uncertainty_px',4.),('camera_height_uncertainty_m',.2),
                            ('camera_pitch_uncertainty_deg',3.),('path_margin_m',.5),
                            ('minimum_signal_confidence',.8),('track_max_age_s',1.)):
            with self.subTest(field=field):
                traffic=Perception();traffic.config=replace(traffic.config,**{field:value})
                with self.assertRaisesRegex(ValueError,field):
                    TrafficPolicy(PathPredictor(),traffic,Controller())
        traffic=Perception();traffic.config=replace(traffic.config,score_threshold=.5,
            front_bumper_x_m=2.6,ego_half_width_m=1.1)
        policy=TrafficPolicy(PathPredictor(),traffic,Controller())
        self.assertEqual(policy.bridge.range.front,2.6)
        self.assertEqual(policy.bridge.range.half,1.1)

    def test_default_bridge_rejects_mismatched_control_model_but_custom_bridge_is_explicit(self):
        for field,value in (('comfortable_deceleration_mps2',1.),('object_standstill_gap_m',3.),
                            ('reaction_time_seconds',.5),('minimum_confidence',.7),
                            ('served_stop_reset_distance_m',8.)):
            with self.subTest(field=field):
                traffic=Perception();supervisor=TrafficSupervisor(replace(TrafficConfig(),**{field:value}))
                with self.assertRaisesRegex(ValueError,field):
                    TrafficPolicy(PathPredictor(),traffic,Controller(),supervisor=supervisor)
                custom=TrafficEvidenceBridge(nominal_calibration=traffic.calibration)
                policy=TrafficPolicy(PathPredictor(),traffic,Controller(),supervisor=supervisor,bridge=custom)
                self.assertIs(policy.bridge,custom)

if __name__=='__main__':unittest.main()
