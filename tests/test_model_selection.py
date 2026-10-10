"""Meaningful selection cases for absolute budgets and valid DEV comparisons."""
import copy
import math
import unittest

from carla_tools.model_selection import load_selection_policy, select_waypoint_model


def report(name, far, near=(.2,.2,.2)):
    groups = []
    for case,n,f in zip(('left_curve','right_curve','straight'),near,far):
        groups.append(dict(case=case,condition='all',sample_count=10,predicted_count=10,failed_prediction_count=0,
            coverage=1.,near_mean_m=n,far_mean_m=f,far_p95_m=2.,mean_euclidean_m=(2*n+3*f)/5))
    groups.insert(0,dict(case='all',condition='all',sample_count=30,predicted_count=30,failed_prediction_count=0,
        coverage=1.,near_mean_m=sum(near)/3,far_mean_m=sum(far)/3,far_p95_m=2.,
        mean_euclidean_m=(2*sum(near)/3+3*sum(far)/3)/5))
    return dict(status='PASS',split='dev',variant=name,fixture_injection=False,final_town_used=False,
        manifest_sha256='a'*64,config_sha256='b'*64,checkpoint_sha256='c'*64,sample_count=30,
        latency_ms={'p95':12.},metrics=groups)


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.reference=report('resnet',(.8,.8,.05))
        self.candidate=report('hybrid_geometry',(.3,.3,.1))

    def choose(self):
        return select_waypoint_model(self.reference,{'hybrid_geometry':self.candidate})

    def test_better_average_and_valid_case_budgets_allow_worse_straight_and_tail(self):
        self.candidate['metrics'][-1]['far_p95_m']=3.
        self.assertEqual(self.choose()['selected_candidate'],'hybrid_geometry')
        self.assertFalse(self.choose()['closed_loop_verified'])

    def test_one_case_over_budget_is_rejected_despite_better_overall_average(self):
        self.candidate=report('hybrid_geometry',(1.01,.1,.1))
        decision=self.choose()
        self.assertLess(decision['reference_mean_m'],.5)
        self.assertEqual(decision['selected_candidate'],'resnet')
        self.assertIn('left_curve:far_mean_m',decision['failed_criteria']['hybrid_geometry'])
        self.candidate=report('hybrid_geometry',(.2,.2,.1),(.31,.1,.1))
        self.assertIn('left_curve:near_mean_m',self.choose()['failed_criteria']['hybrid_geometry'])

    def test_coverage_and_forward_budget_are_independent_requirements(self):
        for index in (0,1):
            row=self.candidate['metrics'][index]
            row['predicted_count']-=1
            row['failed_prediction_count']+=1
            row['coverage']=row['predicted_count']/row['sample_count']
        total=self.candidate['metrics'][0]['predicted_count']
        for metric in ('near_mean_m','far_mean_m','mean_euclidean_m'):
            self.candidate['metrics'][0][metric]=sum(row[metric]*row['predicted_count'] for row in self.candidate['metrics'][1:])/total
        self.assertEqual(self.choose()['selected_candidate'],'resnet')
        self.candidate=report('hybrid_geometry',(.3,.3,.1))
        self.candidate['latency_ms']['p95']=50.001
        self.assertIn('forward-latency',self.choose()['failed_criteria']['hybrid_geometry'])

    def test_equal_average_is_not_an_improvement_and_limits_are_inclusive(self):
        self.candidate=report('hybrid_geometry',(.8,.8,.05))
        self.assertEqual(self.choose()['selected_candidate'],'resnet')
        self.candidate=report('hybrid_geometry',(.3,.3,.1),(.3,.3,.3))
        self.candidate['latency_ms']['p95']=50.
        self.assertEqual(self.choose()['selected_candidate'],'hybrid_geometry')

    def test_wrong_dataset_missing_case_and_false_coverage_are_rejected(self):
        for mutate in (lambda x:x.update(manifest_sha256='d'*64),
                       lambda x:x['metrics'].pop(),lambda x:x['metrics'][0].update(coverage=.99)):
            self.candidate=report('hybrid_geometry',(.3,.3,.1))
            mutate(self.candidate)
            with self.assertRaises(ValueError):
                self.choose()

    def test_nonfinite_error_and_inconsistent_overall_mean_are_rejected(self):
        for value in (math.nan,math.inf,-.1):
            self.candidate['metrics'][1]['near_mean_m']=value
            with self.assertRaises(ValueError):
                self.choose()
        self.candidate=report('hybrid_geometry',(.3,.3,.1))
        self.candidate['metrics'][0]['mean_euclidean_m']=0.
        with self.assertRaisesRegex(ValueError,'inconsistent-overall-mean'):
            self.choose()
        self.candidate=report('hybrid_geometry',(.3,.3,.1))
        self.candidate['metrics'][0]['far_mean_m']=0.
        self.candidate['metrics'][0]['mean_euclidean_m']=.08
        with self.assertRaisesRegex(ValueError,'inconsistent-case-weighted-mean'):
            self.choose()

    def test_ranking_uses_named_overall_group_and_does_not_mutate_reports(self):
        faster=report('hybrid_mirror',(.4,.4,.1))
        snapshot=copy.deepcopy(self.candidate)
        self.candidate['metrics'].reverse()
        decision=select_waypoint_model(self.reference,{'hybrid_mirror':faster,'hybrid_geometry':self.candidate})
        self.assertEqual(decision['selected_candidate'],'hybrid_geometry')
        self.assertEqual(self.candidate['metrics'],list(reversed(snapshot['metrics'])))
        self.assertEqual(decision['policy'],load_selection_policy())


if __name__ == '__main__':
    unittest.main()
