import copy
import unittest
from types import SimpleNamespace
from carla_tools.traffic_signal_confirmation import SignalConfirmationAdapter as SignalEvidenceAdapter
from carla_tools.traffic_behavior import TrafficSupervisor
from carla_tools.traffic_perception import to_traffic_observation

def signal(color='red', score=0.9, box=(300.0, 50.0, 310.0, 80.0)):
    known = color != 'unknown'
    s = {'color': color, 'confidence': 1.0 if known else 0.0, 'counts': {c: 16 if c == color else 0 for c in ('red', 'yellow', 'green')}, 'colored_pixels': 16 if known else 0, 'bbox_xyxy': list(box), 'label_id': 10}
    detection = {'label_id': 10, 'confidence': score, 'bbox_xyxy': list(box)}
    return (s, detection)

def raw(i, heads=(), valid=True):
    selected = next((s for c in ('red', 'yellow', 'unknown', 'green') for s, d in heads if s['color'] == c), None)
    return {'GT_inputs': False, 'frame': i, 'timestamp': i * 0.05, 'signals': [s for s, d in heads], 'detections': [d for s, d in heads], 'signal': selected['color'] if selected else 'none', 'signal_confidence': selected['confidence'] if selected else 0.0, 'objects': [], 'stop_line_distance_m': None, 'stop_line_confidence': 0.0, 'stop_line': {}, 'stop_required': False, 'perception_valid': valid}
LANE = {'steer': 0.12, 'throttle': 0.3, 'brake': 0.0, 'hand_brake': False, 'reverse': False, 'diagnostics': {'effective_target_speed_mps': 3.0}}

class SignalCandidateTests(unittest.TestCase):

    def setUp(self):
        self.candidate = SignalEvidenceAdapter()
        self.supervisor = TrafficSupervisor()

    def step(self, i, heads=(), valid=True, speed=0.0, rgb=None):
        source = raw(i, heads, valid)
        frozen = copy.deepcopy(source)
        result, notes = self.candidate.update(source, SimpleNamespace(frame=i, timestamp=i * 0.05, width=640, height=360, rgb=rgb))
        self.assertEqual(source, frozen)
        command = self.supervisor.step(to_traffic_observation(result), speed, LANE, timestamp=i * 0.05)
        self.assertEqual(command['steer'], 0.12)
        return (command, notes)

    def test_one_frame_stop_colour_never_permanently_latches_across_score_range(self):
        for score in (0.5, 0.541, 0.7, 0.99):
            self.setUp()
            command, notes = self.step(0, [signal(score=score, box=(151.0, 106.0, 159.0, 116.0))], speed=3.0)
            self.assertEqual(command['diagnostics']['effective_target_speed_mps'], 0.0)
            self.assertGreater(command['brake'], 0.0)
            for i in range(1, 31):
                command, notes = self.step(i)
            self.assertFalse(command['diagnostics']['traffic_supervisor']['red_latched'])
            self.assertGreater(command['throttle'], 0.0)

    def test_detector_score_is_not_replaced_by_perfect_colour_confidence(self):
        _, notes = self.step(0, [signal(score=0.541)])
        e = notes['confirmation_evidence'][0]
        self.assertEqual(e['detector_score_recovered_from_original_detection'], 0.541)
        self.assertEqual(e['joint_detector_colour_score'], 0.541)

    def test_real_red_immediately_brakes_then_confirmed_occlusion_never_releases(self):
        for i in range(3):
            command, notes = self.step(i, [signal()], speed=3.0 if i == 0 else 0.0)
            self.assertEqual(command['diagnostics']['effective_target_speed_mps'], 0.0)
        self.assertTrue(notes['confirmation_evidence'][0]['temporal_confirmed_stop'])
        for i in range(3, 101):
            command, notes = self.step(i)
        self.assertEqual(command['throttle'], 0.0)
        self.assertTrue(command['diagnostics']['traffic_supervisor']['red_latched'])

    def test_real_yellow_immediately_brakes_and_keeps_confirmed_stop(self):
        for i in range(3):
            command, notes = self.step(i, [signal('yellow')], speed=3.0 if i == 0 else 0.0)
            self.assertEqual(command['diagnostics']['effective_target_speed_mps'], 0.0)
        for i in range(3, 41):
            command, notes = self.step(i)
        self.assertEqual(command['throttle'], 0.0)

    def test_another_head_green_cannot_release_confirmed_red(self):
        for i in range(3):
            self.step(i, [signal()])
        for i in range(3, 61):
            command, notes = self.step(i, [signal('green', box=(500.0, 50.0, 510.0, 80.0))])
        self.assertEqual(command['throttle'], 0.0)
        self.assertTrue(command['diagnostics']['traffic_supervisor']['red_latched'])

    def test_same_head_confirmed_green_can_release(self):
        for i in range(3):
            self.step(i, [signal()])
        for i in range(3, 41):
            command, notes = self.step(i, [signal('green')])
        self.assertGreater(command['throttle'], 0.0)

    def test_head_switch_does_not_pool_unconfirmed_observations(self):
        self.step(0, [signal()])
        self.step(1, [signal()])
        _, notes = self.step(2, [signal(box=(500.0, 50.0, 510.0, 80.0))])
        self.assertFalse(notes['confirmation_evidence'][0]['temporal_confirmed_stop'])
        self.assertEqual(notes['confirmation_evidence'][0]['consecutive_stop_observations'], 1)

    def test_unknown_or_opposite_backface_without_lamp_colour_is_never_green(self):
        for i in range(31):
            command, notes = self.step(i, [signal('unknown', score=0.99)])
            self.assertEqual(command['throttle'], 0.0)
            self.assertNotEqual(notes['effective_signal'], 'green')
            self.assertFalse(notes['confirmation_evidence'][0]['temporal_confirmed_stop'])

    def test_existing_object_uncertainty_is_never_cleared(self):
        for i in range(31):
            command, notes = self.step(i, valid=False)
        self.assertEqual(command['throttle'], 0.0)

    def test_missing_original_detector_binding_cannot_confirm_stop_colour(self):
        s, d = signal()
        source = raw(0, [(s, d)])
        source['detections'] = []
        result, notes = self.candidate.update(source, SimpleNamespace(frame=0, timestamp=0.0, width=640, height=360))
        self.assertIsNone(notes['confirmation_evidence'][0]['detector_score_recovered_from_original_detection'])
        self.assertFalse(notes['confirmation_evidence'][0]['temporal_confirmed_stop'])

    def test_prior_unknown_and_green_head_then_one_red_survives_occlusion(self):
        for i, color in enumerate(('unknown', 'green', 'unknown')):
            self.step(i, [signal(color)])
        command, notes = self.step(3, [signal(score=0.65)])
        self.assertTrue(notes['confirmation_evidence'][0]['temporal_confirmed_stop'])
        self.assertEqual(notes['confirmation_evidence'][0]['confirmed_stop_basis'], 'persistent-head-before-stop-colour')
        for i in range(4, 81):
            command, notes = self.step(i)
        self.assertEqual(command['throttle'], 0.0)
        self.assertTrue(command['diagnostics']['traffic_supervisor']['red_latched'])

    def test_new_strong_independent_RGB_head_one_red_survives_occlusion(self):
        image = bytearray([20, 20, 20]) * (640 * 360)
        for y in range(53, 60):
            for x in range(303, 310):
                if (x - 306) ** 2 + (y - 56) ** 2 <= 9:
                    at = (y * 640 + x) * 3
                    image[at:at + 3] = bytes([255, 0, 0])
        command, notes = self.step(0, [signal(score=0.95, box=(300.0, 50.0, 312.0, 86.0))], speed=3.0, rgb=bytes(image))
        self.assertTrue(notes['confirmation_evidence'][0]['independent_RGB_head_shape']['supported'])
        self.assertTrue(notes['confirmation_evidence'][0]['temporal_confirmed_stop'])
        self.assertTrue(command['diagnostics']['traffic_supervisor']['red_latched'])
        for i in range(1, 81):
            command, notes = self.step(i)
        self.assertEqual(command['throttle'], 0.0)
        self.assertTrue(command['diagnostics']['traffic_supervisor']['red_latched'])

    def test_high_NN_score_without_independent_pixels_does_not_fabricate_head_shape(self):
        _, notes = self.step(0, [signal(score=0.99)])
        self.assertFalse(notes['confirmation_evidence'][0]['independent_RGB_head_shape']['supported'])
        self.assertFalse(notes['confirmation_evidence'][0]['temporal_confirmed_stop'])
if __name__ == '__main__':
    unittest.main()
