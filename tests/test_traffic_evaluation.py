"""Six functional contracts and explicit adversarial failure witnesses."""
from dataclasses import replace
import unittest

from carla_tools.traffic_evaluation import (SCENARIOS, TrafficEvaluationConfig,
    TrafficEvaluationError, TrafficEvaluator, TrafficGroundTruth)


def scene(name):
    rows=[]
    for i in range(110):
        if name=='clear_road':
            speed=1. if i<100 else 0.;progress=min(5.,i*.05)
        elif i<20:speed=1.;progress=i*.05
        elif i<60:speed=0.;progress=1.
        elif i<87:speed=3.;progress=min(5.,1.+(i-60)*.15)
        else:speed=0.;progress=5.
        line=(1. if i<60 else 1.-(i-60)*.15) if name in ('red_to_green','mandatory_stop') else None
        rows.append(TrafficGroundTruth(i,i*.05,speed,progress,.2,
            minimum_object_clearance_m=2. if name!='clear_road' else None,
            front_bumper_line_distance_m=line,
            signal_state=('red' if i<60 else 'green') if name=='red_to_green' else 'none',
            stimulus_present=name!='clear_road' and i<60,stimulus_visible=name!='clear_road' and i<60,
            hazard_cleared=name!='clear_road' and i>=60,
            expected_wait=name!='clear_road' and 20<=i<60,
            phase='stopping' if progress>=5. else 'driving',lateral_error_m=100.))
    # Clear road still has a real >=.5s terminal stop.
    if name=='clear_road':
        for i in range(110,114):rows.append(replace(rows[-1],frame=i,timestamp=i*.05))
    return rows


def evaluate(name,rows):
    evaluator=TrafficEvaluator(TrafficEvaluationConfig(name,5.))
    for row in rows:evaluator.update(row)
    return evaluator.finalize()


class EvaluationTests(unittest.TestCase):
    def test_all_six_positive_synthetic_scenarios(self):
        for name in SCENARIOS:
            with self.subTest(name=name):
                result=evaluate(name,scene(name))
                self.assertTrue(result['functional_success'],result)
                self.assertFalse(result['CTE_used_as_acceptance_gate'])
                self.assertEqual(result['lateral_abs_mean_m'],100.)

    def test_missing_or_invisible_hazard_cannot_pass(self):
        for field in ('stimulus_present','stimulus_visible'):
            data=[replace(row,**{field:False}) for row in scene('crossing_pedestrian')]
            result=evaluate('crossing_pedestrian',data)
            self.assertFalse(result['functional_success'])
            self.assertFalse(result['scenario_valid'])

    def test_never_clearing_stimulus_cannot_pass_even_if_route_finishes(self):
        data=[replace(row,hazard_cleared=False) for row in scene('leading_vehicle')]
        result=evaluate('leading_vehicle',data)
        self.assertTrue(result['route_completed'])
        self.assertFalse(result['functional_success'])
        self.assertIn('required_stimulus_never_cleared',result['scenario_errors'])

    def test_permanently_stopped_zero_collision_cannot_pass(self):
        data=[replace(row,speed_mps=0.,route_progress_m=0.,expected_wait=True) for row in scene('red_to_green')]
        result=evaluate('red_to_green',data)
        self.assertEqual(result['collision_count'],0)
        self.assertFalse(result['functional_success'])
        self.assertIn('route_not_completed',result['failure_reasons'])
        self.assertIn('restart_not_observed',result['failure_reasons'])

    def test_red_front_bumper_crossing_is_rule_violation(self):
        data=scene('red_to_green');data[30]=replace(data[30],front_bumper_line_distance_m=-.1)
        result=evaluate('red_to_green',data)
        self.assertIn(30,result['red_line_crossing_frames'])
        self.assertFalse(result['functional_success'])

    def test_missing_rule_or_object_GT_is_not_a_success(self):
        data=scene('red_to_green');data[30]=replace(data[30],front_bumper_line_distance_m=None)
        result=evaluate('red_to_green',data)
        self.assertFalse(result['functional_success'])
        self.assertIn('required_same_frame_stop_line_GT_missing',result['scenario_errors'])
        data=scene('crossing_pedestrian');data[30]=replace(data[30],minimum_object_clearance_m=None)
        self.assertFalse(evaluate('crossing_pedestrian',data)['functional_success'])

    def test_rolling_mandatory_stop_is_rejected(self):
        data=[replace(row,speed_mps=.2) if 20<=i<60 else row for i,row in enumerate(scene('mandatory_stop'))]
        result=evaluate('mandatory_stop',data)
        self.assertIn('mandatory_full_stop_not_observed',result['failure_reasons'])
        self.assertTrue(result['mandatory_line_violation_frames'])

    def test_wait_is_legal_but_failure_to_restart_is_not(self):
        data=scene('crossing_pedestrian')
        late=[replace(row,speed_mps=0.,route_progress_m=1.) if 60<=i else row for i,row in enumerate(data[:105])]
        for i in range(105,150):
            progress=min(5.,1.+(i-105)*.15)
            late.append(replace(data[-1],frame=i,timestamp=i*.05,route_progress_m=progress,
                speed_mps=3. if progress<5. else 0.,phase='driving' if progress<5. else 'stopping'))
        result=evaluate('crossing_pedestrian',late)
        self.assertIn('restart_deadline_exceeded',result['failure_reasons'])
        self.assertFalse(result['functional_success'])
        normal=evaluate('crossing_pedestrian',data)
        self.assertGreater(normal['legal_wait_seconds'],1.9)
        self.assertTrue(normal['functional_success'])

    def test_unnecessary_stop_body_violation_collision_and_frame_corruption(self):
        data=scene('clear_road')
        for i in range(20,75):data[i]=replace(data[i],speed_mps=0.)
        self.assertIn('unnecessary_or_permanent_stop',evaluate('clear_road',data)['failure_reasons'])
        for changes,reason in (({'body_lane_margin_m':-.01},'body_outside_lane'),
                               ({'collision_events':1},'collision'),({'lane_invasion_events':1},'lane_invasion')):
            data=scene('clear_road');data[30]=replace(data[30],**changes)
            self.assertIn(reason,evaluate('clear_road',data)['failure_reasons'])
        evaluator=TrafficEvaluator(TrafficEvaluationConfig('clear_road',5.));evaluator.update(data[0])
        with self.assertRaises(TrafficEvaluationError):evaluator.update(data[2])


if __name__=='__main__':unittest.main()
