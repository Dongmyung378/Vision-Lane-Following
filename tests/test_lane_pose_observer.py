"""Camera-only motion, observed-support and replay-state safety checks."""
from dataclasses import replace
import copy
import math
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from carla_tools.baseline import nominal_ground_to_pixels
from carla_tools.lane_pose_observer import LanePoseObserverConfig, RGBTemporalLanePoseObserver


RAW=[[3.,.2],[6.,.2],[9.,.2],[12.,.2],[15.,.2]]


def road_image():
    rng=np.random.default_rng(1234)
    image=np.full((360,640,3),35,dtype=np.uint8)
    for _ in range(500):
        x=int(rng.integers(70,570));y=int(rng.integers(230,355))
        value=int(rng.integers(50,150))
        cv2.rectangle(image,(x,y),(x+3,y+3),(value,value,value),-1)
    return image


def ground_homography():
    s,c=math.sin(math.radians(-12.)),math.cos(math.radians(-12.))
    return np.array([[320*c,320.,-320*c*1.5-320*s*2.4],
        [320*s+180*c,0.,-(320*s+180*c)*1.5+320*c*2.4-180*s*2.4],
        [c,0.,-c*1.5-s*2.4]])


class LanePoseObserverTests(unittest.TestCase):
    def memory_observer(self,points):
        observer=RGBTemporalLanePoseObserver();observer.index=0
        observer.memory=[{'index':0,'points':np.asarray(points,float),'variance_m2':np.full(len(points),.02**2)}]
        return observer

    def test_metric_motion_recovers_ground_se2(self):
        old=road_image();angle=-.01
        transform=np.array([[math.cos(angle),-math.sin(angle),-.12],
            [math.sin(angle),math.cos(angle),-.01],[0.,0.,1.]])
        h=ground_homography();current=cv2.warpPerspective(old,h@transform@np.linalg.inv(h),(640,360))
        observer=RGBTemporalLanePoseObserver()
        observer.observe(old);motion=observer.observe(current)
        self.assertEqual(motion['status'],'PASS',motion)
        np.testing.assert_allclose(motion['translation_m'],transform[:2,2],atol=.025)
        self.assertAlmostEqual(motion['yaw_rad'],angle,delta=.003)

    def test_spatial_hull_interpolates_past_and_future_observed_paint(self):
        x=np.r_[np.arange(-2.,-.5,.08),np.arange(4.,6.,.08)]
        observer=self.memory_observer(np.column_stack((x,np.full(len(x),-2.))))
        result=observer.estimate(RAW)
        self.assertTrue(result['available'],result)
        self.assertIs(type(result['confidence']),float)
        self.assertFalse(result['blindzone_extrapolation_used'])
        self.assertAlmostEqual(result['center_y_at_reference_m'],0.,places=9)
        self.assertLess(result['observed_center_min_x_m'],3.)
        self.assertGreater(result['observed_center_max_x_m'],3.)

    def test_ahead_only_paint_never_extrapolates_into_blind_zone(self):
        x=np.arange(6.,8.,.08)
        result=self.memory_observer(np.column_stack((x,np.full(len(x),-2.)))).estimate(RAW)
        self.assertFalse(result['available'])
        self.assertEqual(result['confidence'],0.)

    def test_estimate_is_pure_and_preserves_longitudinal_and_far_points(self):
        x=np.arange(2.,7.,.08);observer=self.memory_observer(np.column_stack((x,np.full(len(x),-2.))))
        state=copy.deepcopy(observer.memory);first=observer.estimate(RAW);second=observer.estimate(RAW)
        self.assertEqual(first,second)
        np.testing.assert_array_equal(observer.memory[0]['points'],state[0]['points'])
        np.testing.assert_array_equal(observer.memory[0]['variance_m2'],state[0]['variance_m2'])
        self.assertEqual(first['proposed_waypoints_ego_m'][2:],RAW[2:])
        self.assertEqual([p[0] for p in first['proposed_waypoints_ego_m']],[p[0] for p in RAW])
        self.assertEqual(observer.index,0)

    def test_failed_motion_discards_old_memory(self):
        observer=RGBTemporalLanePoseObserver();observer.observe(road_image())
        observer.memory=[{'index':0,'points':np.array([[3.,-2.]]),'variance_m2':np.array([.001])}]
        result=observer.observe(np.zeros((360,640,3),np.uint8))
        self.assertEqual(result['status'],'FAIL')
        self.assertEqual(observer.memory,[])
        self.assertFalse(observer.estimate(RAW)['available'])
        self.assertEqual(observer.history_resets,1)

    def test_identical_new_frames_advance_and_reset_clears_observations(self):
        observer=RGBTemporalLanePoseObserver();image=road_image()
        observer.observe(image);observer.observe(image)
        self.assertEqual(observer.index,1)
        observer.reset()
        self.assertEqual(observer.index,-1);self.assertIsNone(observer.previous_gray);self.assertEqual(observer.memory,[])

    def test_config_roundtrip_and_invalid_physical_bounds(self):
        config=LanePoseObserverConfig()
        self.assertEqual(config,LanePoseObserverConfig.from_dict(config.to_dict()))
        for value in (True,float('nan'),0):
            with self.assertRaises(ValueError):replace(config,motion_inlier_threshold_m=value)
        with self.assertRaises(ValueError):replace(config,history_frames=41)
        with self.assertRaises(ValueError):LanePoseObserverConfig.from_dict({**config.to_dict(),'route':'forbidden'})

    def test_weak_last_past_paint_cannot_create_high_confidence_hull_jump(self):
        past=np.column_stack((np.arange(-1.9,-1.4,.04),np.full(13,-3.49)))
        future_x=np.arange(3.3,6.,.04)
        points=np.vstack((past,np.column_stack((future_x,np.full(len(future_x),-2.)))))
        observer=self.memory_observer(points)
        estimates=[]
        for offset in (-.005,0.,.005):
            raw=[[p[0],p[1]-.2+offset] for p in RAW]
            estimates.append(observer.estimate(raw))
        self.assertLess(max(r['confidence'] for r in estimates),.01)
        self.assertEqual(observer.index,0)

    def test_reference_confidence_fades_at_measured_hull_edge(self):
        x=np.arange(2.4,3.04,.02)
        observer=self.memory_observer(np.column_stack((x,np.full(len(x),-2.))))
        result=observer.estimate(RAW)
        self.assertLess(result['confidence'],.03)

    def test_motion_failure_forces_fallback_even_with_valid_current_paint(self):
        observer=RGBTemporalLanePoseObserver();observer.observe(road_image())
        image=road_image();x=np.linspace(4.2,10.,400)
        boundary=np.column_stack((x,1.5*x-7.10555127546399))
        pixels,visible=nominal_ground_to_pixels(boundary,observer.baseline)
        cv2.polylines(image,[np.rint(pixels[visible]).astype(np.int32)],False,(255,200,0),4)
        with patch.object(observer,'_motion',return_value={'status':'FAIL','reason':'synthetic-motion-rejection'}):
            observer.observe(image)
        self.assertGreater(len(observer.current_paint),12)
        raw=[[3.,1.],[6.,5.5],[9.,10.],[12.,14.5],[15.,19.]]
        supported=self.memory_observer(observer.current_paint)
        self.assertTrue(supported.estimate(raw)['available'])
        result=observer.estimate(raw)
        self.assertFalse(result['available']);self.assertEqual(result['confidence'],0.)
        self.assertEqual(result['rejection'],'current-ground-motion-failed')
        self.assertEqual(observer.history_resets,1)


if __name__=='__main__':unittest.main()
