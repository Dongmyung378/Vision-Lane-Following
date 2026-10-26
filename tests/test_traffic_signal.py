"""Behavioral counterexamples, not implementation mirror assertions."""
from dataclasses import replace
import copy
import unittest

from carla_tools.traffic_behavior import TrafficObservation, TrafficSupervisor
from carla_tools.traffic_perception import RGBObservation, to_traffic_observation
from carla_tools.traffic_signal import SignalEvidenceAdapter


def head(color, box=(300., 40., 320., 80.)):
    valid = color != 'unknown'
    return {'color': color, 'confidence': .9 if valid else 0.,
            'colored_pixels': 10 if valid else 0,
            'counts': {k: 10 if k == color else 0 for k in ('red', 'yellow', 'green')},
            'bbox_xyxy': list(box)}


def raw(i, signals=(), valid=True):
    chosen = next((s for color in ('red', 'yellow', 'unknown', 'green') for s in signals if s['color'] == color), None)
    return {'GT_inputs': False, 'frame': i, 'timestamp': i*.05, 'objects': [], 'signals': list(signals),
            'signal': chosen['color'] if chosen else 'none', 'signal_confidence': chosen['confidence'] if chosen else 0.,
            'stop_line_distance_m': None, 'stop_line_confidence': 0., 'stop_line': {},
            'stop_required': False, 'perception_valid': valid}


def lane():
    return {'steer': .12, 'throttle': .3, 'brake': 0., 'hand_brake': False, 'reverse': False,
            'diagnostics': {'effective_target_speed_mps': 3.}}


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.adapter = SignalEvidenceAdapter()
        self.supervisor = TrafficSupervisor()

    def step(self, i, signals=(), valid=True):
        source = raw(i, signals, valid)
        frozen = copy.deepcopy(source)
        observation = RGBObservation(bytes(640*360*3), 640, 360, i, i*.05, 0.)
        filtered, notes = self.adapter.update(source, observation)
        self.assertEqual(source, frozen)
        command = self.supervisor.step(to_traffic_observation(filtered), 0., lane(), timestamp=i*.05)
        self.assertEqual(command['steer'], .12)
        return command, notes

    def test_transient_colorless_box_stops_then_resumes_without_permanent_red(self):
        command, _ = self.step(0, [head('unknown')])
        self.assertEqual(command['brake'], 1.)
        self.assertFalse(command['diagnostics']['traffic_supervisor']['red_latched'])
        for i in range(1, 23):
            command, _ = self.step(i)
        self.assertGreater(command['throttle'], 0.)

    def test_continuously_unreadable_signal_never_times_out_to_driving(self):
        for i in range(121):
            command, notes = self.step(i, [head('unknown')])
            self.assertEqual(command['throttle'], 0.)
            self.assertTrue(notes['unknown_currently_visible'])

    def test_confirmed_red_disappearance_never_releases_latch(self):
        self.step(0, [head('red')])
        for i in range(1, 121):
            command, _ = self.step(i)
            self.assertEqual(command['throttle'], 0.)
            self.assertTrue(command['diagnostics']['traffic_supervisor']['red_latched'])

    def test_green_on_other_head_cannot_release_red(self):
        self.step(0, [head('red')])
        for i in range(1, 31):
            command, notes = self.step(i, [head('green', (500., 40., 520., 80.))])
            self.assertEqual(command['throttle'], 0.)
            self.assertFalse(notes['associated_green_observed'])

    def test_associated_confirmed_green_releases_after_existing_hysteresis(self):
        self.step(0, [head('red')])
        for i in range(1, 20):
            command, _ = self.step(i, [head('green')])
        self.assertGreater(command['throttle'], 0.)
        self.assertFalse(command['diagnostics']['traffic_supervisor']['red_latched'])

    def test_unknown_after_confirmed_red_cannot_erase_evidence(self):
        self.step(0, [head('red')])
        self.step(1, [head('unknown')])
        for i in range(2, 42):
            command, _ = self.step(i)
        self.assertEqual(command['throttle'], 0.)
        self.assertTrue(command['diagnostics']['traffic_supervisor']['red_latched'])

    def test_simultaneous_unknown_and_red_preserves_red_across_invalidity(self):
        self.step(0, [head('red'), head('unknown', (500., 40., 520., 80.))])
        for i in range(1, 32):
            command, _ = self.step(i)
        self.assertEqual(command['throttle'], 0.)
        self.assertTrue(command['diagnostics']['traffic_supervisor']['red_latched'])

    def test_existing_object_perception_invalidity_is_never_cleared(self):
        for i in range(31):
            command, _ = self.step(i, valid=False)
        self.assertEqual(command['throttle'], 0.)

    def test_two_confirmed_heads_need_both_to_show_green(self):
        self.step(0, [head('red'), head('red', (500., 40., 520., 80.))])
        for i in range(1, 31):
            command, _ = self.step(i, [head('green')])
        self.assertEqual(command['throttle'], 0.)


if __name__ == '__main__':
    unittest.main()
