"""Synthetic CPU pixels and geometry; no downloads, model inference or CARLA."""
from dataclasses import replace
import math
import unittest
from carla_tools.traffic_perception import (
    RGBObservation,TrafficPerceptionConfig,TrafficPerception,ground_estimate,
    validate_nominal_calibration,signal_color,CausalObjectTracker,detect_stop_line,
    to_traffic_observation,
)

def calibration():
    angle=math.radians(-12.);c,s=math.cos(angle),math.sin(angle)
    # CARLA pitch: positive local x tilts upward.
    return {'width':640,'height':360,'intrinsics':{'fx':320.,'fy':320.,'cx':320.,'cy':180.},
        'camera_to_actor_matrix':[[c,0.,-s,1.5],[0.,1.,0.,0.],[s,0.,c,2.4],[0.,0.,0.,1.]]}

def project_ground(x,y=0.):
    m=calibration()['camera_to_actor_matrix'];d=[x-m[0][3],y,-m[2][3]]
    local=[sum(m[r][axis]*d[r] for r in range(3)) for axis in range(3)]
    return 320.+320.*local[1]/local[0],180.-320.*local[2]/local[0]

class Backend:
    def __init__(self,rows):self.rows=rows;self.inputs=[]
    def predict(self,observation):self.inputs.append(observation);return self.rows

class TrafficPerceptionTests(unittest.TestCase):
    def test_RGB_input_rejects_extra_GT_nan_and_bad_buffer(self):
        value=RGBObservation(bytes(640*360*3),640,360,1,0.,0.)
        with self.assertRaises(TypeError):RGBObservation(value.rgb,640,360,1,0.,0.,actor=object())
        with self.assertRaises(ValueError):replace(value,timestamp=float('nan'))
        with self.assertRaises(ValueError):replace(value,frame=True)
        with self.assertRaises(ValueError):replace(value,rgb=b'bad')

    def test_ground_footpoint_static_projection_and_front_bumper_units(self):
        c=calibration();config=TrafficPerceptionConfig()
        for x,y in [(4.,0.),(8.,.5),(12.,-1.),(25.,0.)]:
            u,v=project_ground(x,y);ground=ground_estimate(u,v,c,config)
            self.assertTrue(ground['valid']);self.assertAlmostEqual(ground['distance_m'],x-2.3,places=8)
            self.assertAlmostEqual(ground['lateral_y_m'],y,places=8)
            self.assertGreaterEqual(ground['distance_uncertainty_m'],.25)
        self.assertFalse(ground_estimate(320.,0.,c,config)['valid'])
        self.assertEqual(ground_estimate(320.,300.,c,config,truncated=True)['reason'],'image_border_truncation')
        reflected=calibration();reflected['camera_to_actor_matrix'][1][1]=-1.
        with self.assertRaises(ValueError):validate_nominal_calibration(reflected)
        with self.assertRaises(ValueError):validate_nominal_calibration(dict(c,actor_world_pose={}))

    def test_COCO_filter_RGB_distance_and_outside_corridor(self):
        _,bottom=project_ground(10.)
        backend=Backend([{'label_id':3,'score':.8,'bbox_xyxy':[270.,120.,370.,bottom]},
            {'label_id':1,'score':.9,'bbox_xyxy':[490.,140.,530.,bottom]},
            {'label_id':18,'score':.99,'bbox_xyxy':[10.,20.,40.,80.]},
            {'label_id':8,'score':.1,'bbox_xyxy':[10.,20.,40.,80.]}])
        perception=TrafficPerception(backend,calibration(),TrafficPerceptionConfig(stop_line_detection=False))
        image=RGBObservation(bytes(640*360*3),640,360,1,.05,0.)
        result=perception.process(image)
        self.assertEqual(len(result['objects']),2);self.assertTrue(result['objects'][1]['in_path'])
        self.assertFalse(result['objects'][0]['in_path']);self.assertFalse(result['GT_inputs'])
        self.assertEqual(backend.inputs,[image]);self.assertEqual(result['stop_line']['status'],'not_observed')
        with self.assertRaises(ValueError):perception.process(image)
        perception.reset();perception.process(image)

    def test_tracker_uses_only_past_images_and_unknown_initial_velocity(self):
        tracker=CausalObjectTracker(TrafficPerceptionConfig())
        def obj(distance):return {'label':'car','confidence':.9,'bbox_xyxy':[10.,10.,50.,50.],'distance_m':distance}
        first=tracker.update([obj(10.)],0.)[0]
        second=tracker.update([obj(9.8)],.1)[0];third=tracker.update([obj(9.6)],.2)[0]
        self.assertEqual(first['track_id'],third['track_id']);self.assertIsNone(first['closing_speed_mps'])
        self.assertIsNone(second['closing_speed_mps']);self.assertAlmostEqual(third['closing_speed_mps'],2.)
        fourth=tracker.update([obj(9.)],2.)[0];self.assertNotEqual(first['track_id'],fourth['track_id'])

    def test_signal_color_is_RGB_bbox_only_and_ambiguous_unknown(self):
        red=bytes([255,0,0])*100
        result=signal_color(red,10,10,[0,0,10,10]);self.assertEqual(result['color'],'red')
        self.assertEqual(signal_color(bytes(300),10,10,[0,0,10,10])['color'],'unknown')
        mixed=bytes([255,0,0])*50+bytes([0,255,0])*50
        self.assertEqual(signal_color(mixed,10,10,[0,0,10,10])['color'],'unknown')

    def test_actual_paint_candidate_and_absence_not_virtual_GT_stopline(self):
        c=calibration();cfg=TrafficPerceptionConfig()
        image=bytearray(640*360*3)
        observation=RGBObservation(bytes(image),640,360,1,0.,0.)
        self.assertEqual(detect_stop_line(observation,c,cfg)['status'],'not_observed')
        for y in range(275,282):
            for x in range(160,480):image[(y*640+x)*3:(y*640+x)*3+3]=b'\xff\xff\xff'
        result=detect_stop_line(replace(observation,rgb=bytes(image)),c,cfg)
        self.assertEqual(result['status'],'paint_bar_candidate');self.assertFalse(result['legal_stopline_verified'])
        for y in range(245,252):
            for x in range(160,480):image[(y*640+x)*3:(y*640+x)*3+3]=b'\xff\xff\xff'
        self.assertEqual(detect_stop_line(replace(observation,rgb=bytes(image)),c,cfg)['status'],'ambiguous_paint_stripes')

    def test_typed_behavior_adapter_preserves_unknown_distance_and_RGB_sign_identity(self):
        from carla_tools.traffic_behavior import TrafficObservation,TrackedObject
        # A central bottom-truncated pedestrian is retained with unknown range,
        # rather than silently disappearing from the safety observation.
        backend=Backend([{'label_id':1,'score':.9,'bbox_xyxy':[300.,100.,340.,360.]},
            {'label_id':13,'score':.9,'bbox_xyxy':[410.,60.,450.,100.]}])
        perception=TrafficPerception(backend,calibration(),TrafficPerceptionConfig(stop_line_detection=False))
        obs=RGBObservation(bytes(640*360*3),640,360,1,.05,0.)
        raw=perception.process(obs);typed=to_traffic_observation(raw)
        self.assertIs(type(typed),TrafficObservation);self.assertIs(type(typed.objects[0]),TrackedObject)
        self.assertIsNone(typed.objects[0].distance_m);self.assertTrue(typed.objects[0].in_path)
        self.assertGreater(typed.objects[0].uncertainty_m,0.);self.assertTrue(typed.stop_required)
        second=to_traffic_observation(perception.process(replace(obs,frame=2,timestamp=.1)))
        self.assertEqual(typed.stop_sign_id,second.stop_sign_id)
        with self.assertRaises(TypeError):replace(typed,GT_actor_id=10)

    def test_explicit_lower_bound_adapter_keeps_unknown_nominal_range(self):
        backend=Backend([{'label_id':3,'score':.9,'bbox_xyxy':[300.,100.,340.,360.]}])
        perception=TrafficPerception(backend,calibration(),TrafficPerceptionConfig(stop_line_detection=False))
        raw=perception.process(RGBObservation(bytes(640*360*3),640,360,1,.05,0.))
        raw['objects'][0]['conservative_nearest_distance_m']=15.
        typed=to_traffic_observation(raw).objects[0]
        self.assertIsNone(typed.distance_m)
        self.assertEqual(typed.uncertainty_m,60.)
        self.assertEqual(typed.conservative_nearest_distance_m,15.)

if __name__=='__main__':unittest.main()
