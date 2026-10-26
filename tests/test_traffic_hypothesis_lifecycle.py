"""Safety preservation and an explicit tentative-belief counterexample."""
from types import SimpleNamespace
import copy
import math
import unittest
from carla_tools.traffic_hypothesis import HypothesisMemoryBridge as TentativePedestrianLifecycleBridge
from carla_tools.traffic_behavior import TrafficSupervisor
from carla_tools.traffic_perception import to_traffic_observation

def calibration():
    c = math.cos(math.radians(12.0))
    s = math.sin(math.radians(12.0))
    return {'width': 640, 'height': 360, 'intrinsics': {'fx': 320.0, 'fy': 320.0, 'cx': 320.0, 'cy': 180.0}, 'camera_to_actor_matrix': [[c, 0.0, s, 1.5], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 2.4], [0.0, 0.0, 0.0, 1.0]]}

def person(*, confidence=0.65, distance=12.67, uncertainty=5.84, valid=True, closing=None, box=None):
    return {'track_id': 'person', 'label': 'person', 'kind': 'pedestrian', 'confidence': confidence, 'bbox_xyxy': box or [288.0, 136.0, 306.0, 169.0], 'distance_m': distance, 'lateral_y_m': -0.96, 'distance_uncertainty_m': uncertainty, 'lateral_uncertainty_m': 0.6, 'ground_valid': valid, 'closing_speed_mps': closing, 'in_path': True}

def run_step(b, index, items=(), *, occluded=False, valid=True, grey=75, speed=0.0):
    detections = [{'label': o['label'], 'confidence': o['confidence'], 'bbox_xyxy': o['bbox_xyxy']} for o in items]
    if occluded:
        detections.append({'label': 'truck', 'confidence': 0.9, 'bbox_xyxy': [270.0, 130.0, 350.0, 215.0]})
    raw = {'frame': index, 'timestamp': index * 0.05, 'objects': copy.deepcopy(list(items)), 'detections': detections, 'GT_inputs': False, 'perception_valid': valid, 'signal': 'none', 'signal_confidence': 0.0, 'stop_required': False, 'stop_line_distance_m': None, 'stop_line_confidence': 0.0, 'stop_line': {}}
    observation = SimpleNamespace(frame=index, timestamp=index * 0.05, ego_speed_mps=speed, width=640, height=360, rgb=bytes([grey]) * 640 * 360 * 3)
    return b.update(raw, {'status': 'PASS', 'waypoints_ego_m': [[3.0, 0.0], [40.0, 0.0]]}, observation)

class LifecycleTests(unittest.TestCase):

    def make(self):
        return TentativePedestrianLifecycleBridge(nominal_calibration=calibration())

    def weak_pair(self, b):
        run_step(b, 0, [person()], speed=3.0)
        run_step(b, 1, [person()], speed=3.0)
        for i in range(2, 11):
            run_step(b, i, occluded=True, speed=3.0)
        self.assertIn('person', b.tracks)
        self.assertTrue(b.tracks['person']['threat_relevant'])

    def test_four_positive_current_unoccluded_road_negatives_retire_weak_far_hypothesis(self):
        b = self.make()
        self.weak_pair(b)
        for i in range(11, 14):
            _, notes = run_step(b, i)
            self.assertFalse(notes['person_clearance']['retired_tentative_hypotheses'])
        _, notes = run_step(b, 14)
        self.assertEqual(notes['person_clearance']['retired_tentative_hypotheses'][0]['track_id'], 'person')
        self.assertNotIn('person', b.tracks)

    def test_two_frame_far_person_behind_vehicle_keeps_memory(self):
        b = self.make()
        self.weak_pair(b)
        for i in range(11, 27):
            result, notes = run_step(b, i, occluded=True)
        self.assertIn('person', notes['unresolved_lost_tracks'])
        self.assertFalse(result['perception_valid'])

    def test_actual_near_person_immediate_emergency_then_occlusion_retains_hold(self):
        b = self.make()
        p = person(distance=3.0, uncertainty=0.5, closing=3.0, box=[280.0, 170.0, 330.0, 250.0])
        traffic, _ = run_step(b, 0, [p], speed=3.0)
        c = TrafficSupervisor().step(to_traffic_observation(traffic), 3.0, {'throttle': 0.5, 'brake': 0.0, 'steer': 0.0, 'hand_brake': False, 'reverse': False, 'diagnostics': {'effective_target_speed_mps': 3.0}}, timestamp=0.0)
        b.acknowledge(c)
        self.assertEqual(c['brake'], 1.0)
        for i in range(1, 16):
            result, notes = run_step(b, i, occluded=True)
        self.assertIn('person', notes['unresolved_lost_tracks'])

    def test_high_confidence_confirmed_and_unknown_ground_persons_never_retire(self):
        for mode in ('high', 'confirmed', 'unknown'):
            b = self.make()
            p = person(confidence=0.9) if mode == 'high' else person(valid=False, distance=None, uncertainty=None) if mode == 'unknown' else person()
            count = 3 if mode == 'confirmed' else 1
            for i in range(count):
                run_step(b, i, [p], speed=3.0)
            for i in range(count, 11):
                run_step(b, i, occluded=True, speed=3.0)
            for i in range(11, 27):
                _, notes = run_step(b, i)
            self.assertIn('person', b.tracks)
            self.assertFalse(notes['person_clearance']['retired_tentative_hypotheses'])

    def test_current_single_far_person_TTC_emergency_locks_safety_memory(self):
        b = self.make()
        traffic, _ = run_step(b, 0, [person(uncertainty=1.0, closing=10.0)])
        c = TrafficSupervisor().step(to_traffic_observation(traffic), 0.0, {'throttle': 0.5, 'brake': 0.0, 'steer': 0.0, 'hand_brake': False, 'reverse': False, 'diagnostics': {'effective_target_speed_mps': 3.0}}, timestamp=0.0)
        self.assertTrue(c['diagnostics']['traffic_supervisor']['emergency_brake'])
        b.acknowledge(c)
        for i in range(1, 16):
            _, notes = run_step(b, i)
        self.assertIn('person', b.tracks)
        self.assertFalse(notes['person_clearance']['retired_tentative_hypotheses'])

    def test_low_light_invalid_perception_and_unstable_camera_do_not_accumulate_negative_evidence(self):
        for mode in ('lowlight', 'invalid', 'unstable'):
            b = self.make()
            self.weak_pair(b)
            for i in range(11, 27):
                _, notes = run_step(b, i, grey=10 if mode == 'lowlight' else (75 if i % 2 else 150) if mode == 'unstable' else 75, valid=mode != 'invalid')
            self.assertIn('person', b.tracks)
            self.assertFalse(notes['person_clearance']['retired_tentative_hypotheses'])

    def test_calibration_change_blocks_certificate(self):
        b = self.make()
        self.weak_pair(b)
        b.calibration['intrinsics']['cx'] += 8.0
        for i in range(11, 27):
            _, notes = run_step(b, i)
        self.assertIn('person', b.tracks)
        self.assertFalse(notes['person_clearance']['retired_tentative_hypotheses'])

    def test_stationary_camera_jumps_once_then_remains_stable_still_retains_hazard(self):
        b = self.make()
        self.weak_pair(b)
        run_step(b, 11)
        for i in range(12, 28):
            _, notes = run_step(b, i, grey=150)
        self.assertIn('person', b.tracks)
        self.assertTrue(notes['person_clearance']['low_speed_camera_jump_latched'])
        self.assertFalse(notes['person_clearance']['retired_tentative_hypotheses'])

    def test_explicit_counterexample_real_brief_far_person_can_leave_old_patch_and_lose_belief(self):
        b = self.make()
        run_step(b, 0, [person()], speed=3.0)
        for i in range(1, 11):
            run_step(b, i, occluded=True, speed=3.0)
        for i in range(11, 16):
            _, notes = run_step(b, i)
        self.assertNotIn('person', b.tracks)
        self.assertFalse(notes['person_clearance']['safety_guarantee_claimed'])
if __name__ == '__main__':
    unittest.main()
