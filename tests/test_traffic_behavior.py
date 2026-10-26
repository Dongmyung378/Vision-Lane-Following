"""Traffic safety decisions from typed RGB estimates; no simulator or model."""
import copy
from dataclasses import FrozenInstanceError, replace
import math
import unittest

from carla_tools.controller import ControllerConfig, PurePursuitController, VehicleGeometry
from carla_tools.traffic_behavior import (SpeedPI, TrafficConfig, TrafficError,
    TrafficObservation, TrafficSupervisor, TrackedObject)


def lane(target=25/3.6,steer=.23):
    return {'steer':steer,'throttle':.1,'brake':0.,'hand_brake':False,'reverse':False,
            'diagnostics':{'target_speed_mps':target,'untouched_lane_witness':[1,2,3]}}


def obj(kind='pedestrian',distance=6.,closing=None,in_path=True):
    return TrackedObject('rgb-track-1',kind,distance,0.,.5,closing,in_path,.9)


class BehaviorTests(unittest.TestCase):
    def test_nominal_PI_matches_existing_PP_over_varying_motion(self):
        geometry=VehicleGeometry(2.8,-1.4,1.4,70.,((0.,1.),(100.,1.)),front_track_m=1.6)
        pp=PurePursuitController(ControllerConfig(),geometry);traffic=TrafficSupervisor()
        for i in range(120):
            speed=(i%21)*.3
            points=[[x,.003*(i%7)*x*x] for x in (3.,6.,9.,12.,15.)]
            original=pp.step(points,speed);snapshot=copy.deepcopy(original)
            actual=traffic.step(TrafficObservation(i,i*.05),speed,original,timestamp=i*.05)
            self.assertEqual(actual['steer'],original['steer'])
            self.assertAlmostEqual(actual['throttle'],original['throttle'],places=14)
            self.assertAlmostEqual(actual['brake'],original['brake'],places=14)
            self.assertEqual(original,snapshot)

    def test_in_path_pedestrian_emergency_and_clear_hysteresis(self):
        traffic=TrafficSupervisor();original=lane()
        first=traffic.step(TrafficObservation(0,0.,(obj(distance=4.),)),3.,original,timestamp=0.)
        self.assertEqual(first['brake'],1.)
        self.assertEqual(first['steer'],original['steer'])
        self.assertTrue(first['diagnostics']['traffic_supervisor']['emergency_brake'])
        for i in range(1,11):
            actual=traffic.step(TrafficObservation(i,i*.05),0.,original,timestamp=i*.05)
            self.assertEqual(actual['throttle'],0.)
        actual=traffic.step(TrafficObservation(11,.55),0.,original,timestamp=.55)
        self.assertGreater(actual['throttle'],0.)

    def test_roadside_object_does_not_stop_and_vehicle_follow_reduces_speed(self):
        a=TrafficSupervisor().step(TrafficObservation(0,0.,(obj(in_path=False),)),3.,lane(),timestamp=0.)
        self.assertEqual(a['diagnostics']['traffic_supervisor']['state'],'cruise')
        b=TrafficSupervisor().step(TrafficObservation(0,0.,(obj('vehicle',8.,1.),)),3.,lane(),timestamp=0.)
        self.assertEqual(b['diagnostics']['traffic_supervisor']['state'],'follow_vehicle')
        self.assertLess(b['diagnostics']['traffic_supervisor']['target_speed_mps'],3.)

    def test_unknown_closing_speed_is_not_confident_zero(self):
        unknown=TrafficSupervisor().step(TrafficObservation(0,0.,(obj('vehicle',6.,None),)),4.,lane(),timestamp=0.)
        known=TrafficSupervisor().step(TrafficObservation(0,0.,(obj('vehicle',6.,0.),)),4.,lane(),timestamp=0.)
        self.assertTrue(unknown['diagnostics']['traffic_supervisor']['emergency_brake'])
        self.assertFalse(known['diagnostics']['traffic_supervisor']['emergency_brake'])

    def test_same_gap_pulling_away_lead_uses_signed_relative_velocity(self):
        stationary=TrafficSupervisor().step(TrafficObservation(0,0.,(obj('vehicle',3.,0.),)),0.,lane(),timestamp=0.)
        departing=TrafficSupervisor().step(TrafficObservation(0,0.,(obj('vehicle',3.,-1.),)),0.,lane(),timestamp=0.)
        self.assertGreater(departing['diagnostics']['traffic_supervisor']['target_speed_mps'],
                           stationary['diagnostics']['traffic_supervisor']['target_speed_mps'])
        self.assertGreater(departing['throttle'],stationary['throttle'])
        self.assertFalse(departing['diagnostics']['traffic_supervisor']['emergency_brake'])

    def test_red_stop_green_confirmed_then_resume(self):
        traffic=TrafficSupervisor()
        stopped=TrafficObservation(0,0.,signal='red',signal_confidence=.9,stop_line_distance_m=1.,stop_line_confidence=.9)
        actual=traffic.step(stopped,0.,lane(),timestamp=0.)
        self.assertEqual(actual['brake'],1.)
        for i in range(1,18):
            obs=replace(stopped,frame=i,timestamp=i*.05,signal='green')
            actual=traffic.step(obs,0.,lane(),timestamp=i*.05)
            if i<17:self.assertEqual(actual['throttle'],0.)
        self.assertGreater(actual['throttle'],0.)

    def test_green_or_unregulated_line_alone_does_not_require_stop(self):
        for signal in ('none','green'):
            obs=TrafficObservation(0,0.,signal=signal,signal_confidence=.9,stop_line_distance_m=.5,stop_line_confidence=.9)
            out=TrafficSupervisor().step(obs,2.,lane(),timestamp=0.)
            self.assertEqual(out['diagnostics']['traffic_supervisor']['state'],'cruise')

    def test_mandatory_full_stop_hold_and_served_id(self):
        traffic=TrafficSupervisor()
        for i in range(40):
            obs=TrafficObservation(i,i*.05,stop_line_distance_m=1.,stop_line_confidence=.9,
                                   stop_required=True,stop_sign_id='RGB-sign-A')
            out=traffic.step(obs,0.,lane(),timestamp=i*.05)
            if i<=20:self.assertEqual(out['throttle'],0.)
        self.assertGreater(out['throttle'],0.)
        self.assertTrue(out['diagnostics']['traffic_supervisor']['mandatory_stop_served'])
        new=replace(obs,frame=40,timestamp=2.,stop_sign_id='RGB-sign-B')
        out=traffic.step(new,0.,lane(),timestamp=2.)
        self.assertEqual(out['throttle'],0.)
        self.assertTrue(out['diagnostics']['traffic_supervisor']['mandatory_stop_pending'])

    def test_full_stop_far_from_line_does_not_serve_stop_sign(self):
        traffic=TrafficSupervisor()
        for i in range(30):
            out=traffic.step(TrafficObservation(i,i*.05,stop_required=True,
                stop_line_distance_m=10.,stop_line_confidence=.9),0.,lane(),timestamp=i*.05)
        self.assertFalse(out['diagnostics']['traffic_supervisor']['mandatory_stop_served'])

    def test_stale_invalid_and_unknown_signal_remain_conservative(self):
        traffic=TrafficSupervisor()
        first=TrafficObservation(0,0.)
        traffic.step(first,2.,lane(),timestamp=0.)
        repeated=traffic.step(first,2.,lane(),timestamp=.05)
        self.assertEqual(repeated['diagnostics']['traffic_supervisor']['state'],'uncertain')
        for obs in (TrafficObservation(0,0.,perception_valid=False),
                    TrafficObservation(0,0.,signal='unknown'),
                    TrafficObservation(0,0.,(obj(distance=None),))):
            out=TrafficSupervisor().step(obs,0.,lane(),timestamp=0.)
            self.assertEqual(out['throttle'],0.)
            self.assertEqual(out['brake'],1.)

    def test_immutable_typed_contract_rejects_GT_and_bad_numbers(self):
        with self.assertRaises(TypeError):TrafficObservation(0,0.,actor_transform={})
        with self.assertRaises(FrozenInstanceError):TrafficObservation(0,0.).frame=2
        with self.assertRaises(TrafficError):TrafficObservation(0,0.,objects=[obj()])
        with self.assertRaises(TrafficError):TrafficObservation(0,math.nan)
        with self.assertRaises(TrafficError):TrafficSupervisor().step({'frame':0},0.,lane(),timestamp=0.)


    def test_one_sided_distance_bound_is_separate_from_nominal_unknown(self):
        far=TrackedObject('far','vehicle',None,None,60.,None,True,.9,3.,20.)
        moving=TrafficSupervisor().step(TrafficObservation(0,0.,(far,)),3.,lane(3.),timestamp=0.)
        self.assertEqual(moving['diagnostics']['effective_target_speed_mps'],3.)
        close=replace(far,conservative_nearest_distance_m=.2)
        stopped=TrafficSupervisor().step(TrafficObservation(0,0.,(close,)),3.,lane(3.),timestamp=0.)
        self.assertTrue(stopped['diagnostics']['traffic_supervisor']['emergency_brake'])
        self.assertEqual(stopped['brake'],1.)
        with self.assertRaises(TrafficError):replace(far,conservative_nearest_distance_m=float('nan'))

if __name__=='__main__':unittest.main()
