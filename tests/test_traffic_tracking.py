"""Synthetic CPU tests for vehicle velocity filtering; no model or simulator."""
import copy
import math
import unittest

from carla_tools.traffic_tracking import AdaptiveVehicleTracker
from carla_tools.traffic_perception import (
    CausalObjectTracker,TrafficPerceptionConfig,TrafficPerception,RGBObservation,
)
from carla_tools.traffic_behavior import TrackedObject,TrafficObservation,TrafficSupervisor


def detection(distance,label='car',box=None):
    return {'label':label,'kind':'pedestrian' if label=='person' else 'vehicle',
            'confidence':.9,'bbox_xyxy':box or [280.,130.,360.,210.],'distance_m':distance}


def sequence(tracker,distances,label='car'):
    for i,distance in enumerate(distances):
        result=tracker.update([detection(distance,label)],i*.05)[0]
    return result


def supervised(item,nearest):
    tracked=TrackedObject(item['track_id'],item['kind'],item['distance_m'],0.,item['distance_m']-nearest,
                          item['closing_speed_mps'],True,.9)
    lane={'steer':.1,'throttle':.3,'brake':0.,'hand_brake':False,'reverse':False,
          'diagnostics':{'effective_target_speed_mps':3.}}
    return TrafficSupervisor().step(TrafficObservation(0,0.,(tracked,)),3.,lane,timestamp=0.)


class AdaptiveTrackingTests(unittest.TestCase):
    def test_sudden_lead_stop_retains_full_TTC_emergency_brake(self):
        item=sequence(AdaptiveVehicleTracker(TrafficPerceptionConfig()),(7.,7.,6.85))
        self.assertAlmostEqual(item['closing_speed_mps'],3.)
        command=supervised(item,4.2)
        self.assertTrue(command['diagnostics']['traffic_supervisor']['emergency_brake'])
        self.assertEqual(command['brake'],1.)

    def test_old_positive_outlier_is_reduced_when_latest_measurement_reverses(self):
        cfg=TrafficPerceptionConfig()
        old=sequence(CausalObjectTracker(cfg),(10.,9.7,9.74))
        new=sequence(AdaptiveVehicleTracker(cfg),(10.,9.7,9.74))
        self.assertAlmostEqual(old['closing_speed_mps'],6.)
        self.assertAlmostEqual(new['closing_speed_mps'],2.6)
        self.assertEqual(old['track_id'],new['track_id'])

    def test_pedestrian_tracking_retains_original_even_sample_behavior(self):
        cfg=TrafficPerceptionConfig();a=CausalObjectTracker(cfg);b=AdaptiveVehicleTracker(cfg)
        for i,d in enumerate((10.,9.7,9.74,9.8,9.7,9.9)):
            self.assertEqual(a.update([detection(d,'person')],i*.05),b.update([detection(d,'person')],i*.05))

    def test_boxes_IDs_classes_and_history_are_unchanged(self):
        cfg=TrafficPerceptionConfig();a=CausalObjectTracker(cfg);b=AdaptiveVehicleTracker(cfg)
        for i,d in enumerate((10.,9.7,9.74,9.8,9.7,9.9)):
            objects=[detection(d,'car'),detection(d+.3,'truck',[282.,132.,362.,212.])]
            old=a.update(objects,i*.05);new=b.update(objects,i*.05)
            for x,y in zip(old,new):
                self.assertEqual({k:v for k,v in x.items() if k!='closing_speed_mps'},
                                 {k:v for k,v in y.items() if k!='closing_speed_mps'})
            self.assertEqual(a.tracks,b.tracks)

    def test_multiple_close_detections_are_preserved_and_input_is_immutable(self):
        tracker=AdaptiveVehicleTracker(TrafficPerceptionConfig())
        objects=[detection(8.),detection(8.,'truck',[281.,131.,361.,211.]),
                 detection(4.,'person',[305.,100.,335.,240.])]
        original=copy.deepcopy(objects);result=tracker.update(objects,0.)
        self.assertEqual(len(result),3);self.assertEqual(len({r['track_id'] for r in result}),3)
        self.assertEqual(objects,original)

    def test_latest_known_approach_is_not_hidden_by_two_sample_filter(self):
        for previous in (-10.,-2.,0.,2.,10.):
            for latest in (1.,2.,3.,5.,10.):
                item=sequence(AdaptiveVehicleTracker(TrafficPerceptionConfig()),
                              (20.+.05*(previous+latest),20.+.05*latest,20.))
                self.assertGreaterEqual(item['closing_speed_mps'],latest-1e-10)
                self.assertTrue(supervised(item,1.4*latest)['diagnostics']['traffic_supervisor']['emergency_brake'])

    def test_reset_starts_new_unknown_velocity_history(self):
        tracker=AdaptiveVehicleTracker(TrafficPerceptionConfig())
        sequence(tracker,(10.,9.9,9.8));tracker.reset()
        first=tracker.update([detection(5.)],0.)
        fresh=AdaptiveVehicleTracker(TrafficPerceptionConfig()).update([detection(5.)],0.)
        self.assertEqual(first,fresh);self.assertIsNone(first[0]['closing_speed_mps'])

    def test_malformed_detector_output_is_rejected_before_tracking(self):
        c,s=math.cos(math.radians(-12.)),math.sin(math.radians(-12.))
        calibration={'width':640,'height':360,'intrinsics':{'fx':320.,'fy':320.,'cx':320.,'cy':180.},
                     'camera_to_actor_matrix':[[c,0.,-s,1.5],[0.,1.,0.,0.],[s,0.,c,2.4],[0.,0.,0.,1.]]}
        class MalformedBackend:
            def predict(self,observation):return [{'label_id':3,'score':.9,'bbox_xyxy':[float('nan'),0.,10.,10.]}]
        cfg=TrafficPerceptionConfig(stop_line_detection=False)
        perception=TrafficPerception(MalformedBackend(),calibration,cfg)
        perception.tracker=AdaptiveVehicleTracker(cfg)
        with self.assertRaises(ValueError):
            perception.process(RGBObservation(bytes(640*360*3),640,360,0,0.,0.))
        self.assertEqual(perception.tracker.sequence,0)

if __name__=='__main__':unittest.main()
