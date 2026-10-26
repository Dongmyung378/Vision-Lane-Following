"""Synthetic counterexamples; no images, model, simulator or GT policy input."""
from types import SimpleNamespace
import copy
import unittest
from carla_tools.traffic_hypothesis import DuplicateEvidenceBridge as MemoryEvidenceBridge, duplicate_support
from carla_tools.traffic_perception import to_traffic_observation
from carla_tools.traffic_behavior import TrafficSupervisor

def obj(identifier='car', label='car', distance=12.0, uncertainty=1.0, closing=0.0, box=None):
    return {'track_id': identifier, 'label': label, 'kind': 'pedestrian' if label == 'person' else 'vehicle', 'confidence': 0.9, 'bbox_xyxy': box or [280.0, 130.0, 350.0, 190.0], 'distance_m': distance, 'distance_uncertainty_m': uncertainty, 'lateral_y_m': 0.0, 'lateral_uncertainty_m': 0.1, 'ground_valid': True, 'closing_speed_mps': closing, 'in_path': True}

def bridge():
    result = MemoryEvidenceBridge.__new__(MemoryEvidenceBridge)
    result.front = 2.4
    result.half = 0.95
    result.reset()
    return result

def step(instance, index, objects, speed=3.0):
    observation = SimpleNamespace(frame=index, timestamp=index * 0.05, ego_speed_mps=speed, width=640, height=360)
    instance.observation = observation
    raw = {'frame': index, 'timestamp': index * 0.05, 'objects': copy.deepcopy(objects), 'GT_inputs': False, 'perception_valid': True, 'signal': 'none', 'signal_confidence': 0.0, 'stop_required': False, 'stop_line_distance_m': None, 'stop_line_confidence': 0.0, 'stop_line': {}}
    snapshot = copy.deepcopy(raw)
    result, notes = instance._memory_update(raw, {'status': 'PASS', 'waypoints_ego_m': [[3.0, 0.0], [40.0, 0.0]]}, observation)
    assert raw == snapshot
    return (result, notes)

class SafetyTests(unittest.TestCase):

    def establish(self, b, a=None, c=None):
        for i in range(3):
            step(b, i, [a or obj(), c or obj('truck', 'truck')])

    def test_repeated_far_duplicate_requires_fresh_observed_representative(self):
        b = bridge()
        self.establish(b)
        result, notes = step(b, 3, [obj('truck', 'truck')])
        self.assertEqual(notes['identity_memory']['resolved_duplicate_identities'][0]['obsolete_track_id'], 'car')
        self.assertNotIn('car', b.tracks)
        self.assertEqual([o['track_id'] for o in result['objects']], ['truck'])

    def test_empty_observations_never_release_even_certified_pair(self):
        b = bridge()
        self.establish(b)
        _, notes = step(b, 3, [])
        self.assertFalse(notes['identity_memory']['resolved_duplicate_identities'])
        self.assertIn('car', b.tracks)
        self.assertIn('truck', b.tracks)

    def test_two_distinct_vehicle_footpoints_are_not_merged(self):
        b = bridge()
        a = obj()
        c = obj('truck', 'truck', box=[280.0, 130.0, 350.0, 194.0])
        self.assertFalse(duplicate_support(a, c))
        self.establish(b, a, c)
        _, notes = step(b, 3, [c])
        self.assertFalse(notes['identity_memory']['resolved_duplicate_identities'])
        self.assertIn('car', b.tracks)

    def test_close_overlapped_vehicle_and_lost_close_hazard_stay_fail_closed(self):
        b = bridge()
        a = obj(distance=3.0)
        c = obj('truck', 'truck', distance=3.0)
        self.establish(b, a, c)
        for i in range(3, 16):
            result, notes = step(b, i, [c])
        self.assertIn('car', notes['unresolved_lost_tracks'])
        self.assertFalse(result['perception_valid'])

    def test_single_support_and_unknown_or_approaching_representative_cannot_release(self):
        for closing in (None, 8.0):
            b = bridge()
            self.establish(b)
            _, notes = step(b, 3, [obj('truck', 'truck', closing=closing)])
            self.assertFalse(notes['identity_memory']['resolved_duplicate_identities'])
            self.assertIn('car', b.tracks)
        b = bridge()
        step(b, 0, [obj(), obj('truck', 'truck')])
        _, notes = step(b, 1, [obj('truck', 'truck')])
        self.assertFalse(notes['identity_memory']['resolved_duplicate_identities'])

    def test_real_pedestrian_occluded_by_vehicle_still_immediately_brakes_and_latches(self):
        b = bridge()
        person = obj('person', 'person', distance=3.0, uncertainty=0.5, closing=3.0)
        car = obj(distance=3.0, uncertainty=0.5, closing=0.0)
        result, notes = step(b, 0, [person, car])
        command = TrafficSupervisor().step(to_traffic_observation(result), 3.0, {'throttle': 0.5, 'brake': 0.0, 'steer': 0.1, 'hand_brake': False, 'reverse': False, 'diagnostics': {'effective_target_speed_mps': 3.0}}, timestamp=0.0)
        self.assertEqual(command['brake'], 1.0)
        self.assertTrue(command['diagnostics']['traffic_supervisor']['emergency_brake'])
        for i in range(1, 13):
            result, notes = step(b, i, [car])
        self.assertIn('person', notes['unresolved_lost_tracks'])
        self.assertFalse(result['perception_valid'])
        self.assertEqual(notes['identity_memory']['pedestrian_evidence']['person']['status'], 'tentative')

    def test_malformed_and_cross_kind_boxes_never_supply_duplicate_certificate(self):
        a = obj()
        c = obj('truck', 'truck')
        c['bbox_xyxy'] = [float('nan'), 130.0, 350.0, 190.0]
        self.assertFalse(duplicate_support(a, c))
        self.assertFalse(duplicate_support(a, obj('person', 'person')))

    def test_old_fast_TTC_threat_is_retained_when_matching_truck_claims_no_approach(self):
        b = bridge()
        self.establish(b, obj(closing=8.0), obj('truck', 'truck'))
        _, notes = step(b, 3, [obj('truck', 'truck')])
        self.assertFalse(notes['identity_memory']['resolved_duplicate_identities'])
        self.assertIn('car', b.tracks)

    def test_reset_drops_evidence_and_does_not_change_current_boxes_or_ids(self):
        b = bridge()
        items = [obj(), obj('truck', 'truck')]
        result, _ = step(b, 0, items)
        self.assertEqual(result['objects'], items)
        b.reset()
        self.assertFalse(b.tracks)
        self.assertFalse(b.duplicate_evidence)
        self.assertFalse(b.person_evidence)
if __name__ == '__main__':
    unittest.main()
