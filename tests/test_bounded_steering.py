import math
import random
import unittest

from carla_tools.bounded_steering import BoundedSteeringConfig, BoundedSteeringFeedback


class BoundedSteeringTests(unittest.TestCase):
    def test_cap_precedes_one_filter_and_history_uses_actual_output(self):
        control = BoundedSteeringFeedback(BoundedSteeringConfig(gain=1.,
            correction_limit=.01, smoothing_alpha=.5, slew_per_second=1.))
        first = control.step(.1, .9, enabled=True)
        self.assertAlmostEqual(first.bounded_request_correction, .01)
        self.assertAlmostEqual(first.filtered_steer, .055)
        self.assertAlmostEqual(first.steer, .05)
        second = control.step(.1, .9, enabled=True)
        self.assertAlmostEqual(second.previous_steer, .05)
        self.assertAlmostEqual(second.steer, .08)

    def test_feedback_and_actuator_bounds_over_changing_inputs(self):
        cfg = BoundedSteeringConfig(smoothing_alpha=.23, slew_per_second=.7)
        baseline = BoundedSteeringFeedback(cfg)
        candidate = BoundedSteeringFeedback(cfg)
        rng = random.Random(17)
        previous = 0.
        for index in range(1000):
            nominal, proposal, dt = rng.uniform(-1., 1.), rng.uniform(-1., 1.), rng.uniform(.01, .1)
            base = baseline.step(nominal, dt_seconds=dt)
            result = candidate.step(nominal, proposal, enabled=index % 3 != 0, dt_seconds=dt)
            self.assertLessEqual(abs(result.bounded_request_correction), cfg.correction_limit)
            self.assertLessEqual(abs(result.steer - base.steer), cfg.correction_limit + 1e-12)
            self.assertLessEqual(abs(result.steer - previous), cfg.slew_per_second * dt + 1e-12)
            self.assertLessEqual(abs(result.steer), 1.)
            previous = result.steer

    def test_unusable_proposal_falls_back_without_requiring_a_prediction(self):
        control = BoundedSteeringFeedback()
        result = control.step(.05, math.nan, enabled=False)
        self.assertEqual(result.steer, .05)
        self.assertEqual(result.bounded_request_correction, 0.)

    def test_rejected_input_does_not_advance_output_history(self):
        control = BoundedSteeringFeedback()
        control.step(.05)
        for arguments in ((math.nan, None, False, .05), (.1, math.inf, True, .05),
                          (.1, None, False, 0.), (1.1, None, False, .05)):
            with self.assertRaises(ValueError):
                control.step(arguments[0], arguments[1], enabled=arguments[2], dt_seconds=arguments[3])
        self.assertEqual(control.step(.05).previous_steer, .05)
        control.reset()
        self.assertEqual(control.step(0.).previous_steer, 0.)

    def test_configuration_rejects_invalid_limits(self):
        for invalid in (False, 0, {}, []):
            with self.assertRaises(ValueError):
                BoundedSteeringFeedback(invalid)
        for kwargs in ({"gain": 2.}, {"correction_limit": -.1}, {"maximum_steer": 0.},
                       {"smoothing_alpha": 0.}, {"slew_per_second": math.inf}, {"gain": True}):
            with self.assertRaises(ValueError):
                BoundedSteeringConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
