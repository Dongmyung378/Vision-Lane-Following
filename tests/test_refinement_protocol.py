import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np
from PIL import Image

from carla_tools.baseline import load_baseline_config
from carla_tools.driving import route_initial_location
from carla_tools.driving_media import export_driving_media
from carla_tools.evaluation import load_evaluation_config
from carla_tools.refinement_driving import NativeBaselinePredictor, NativeCheckpointPredictor, RefinementTrialPolicy
from carla_tools.refinement_validation import validate_refinement_episode
from scripts.run_refinement_pipeline import trial_matrix, summarize_attempts


def routes():
    return [{'case': case, 'route': {'route_id': case}} for case in ('straight', 'left_curve', 'right_curve')]


class RefinementProtocolTests(unittest.TestCase):
    def test_cuda_tensor_device_identity_is_canonical_without_running_a_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            path.write_bytes(b'fixture-weights')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            model = Mock()
            model.parameters.return_value = iter([SimpleNamespace(device='cuda:0')])
            state = dict(variant='hybrid_geometry', checkpoint_role='selected-refined-model', epoch=1,
                model_contract={'model_version': 'fixture'}, model_contract_sha256='fixture', source_sha256={}, manifest_sha256='fixture')
            with patch('carla_tools.refinement_driving.load_refined_checkpoint', return_value=(model, Mock(), state)):
                predictor = NativeCheckpointPredictor(path, kind='refined', expected_sha256=digest, device='cpu')
            self.assertEqual(predictor.identity['device'], 'cuda')

    def test_full_paired_matrix_and_original_route_ids(self):
        plan = trial_matrix(routes(), ['B0', 'resnet', 'hybrid_geometry'])
        self.assertEqual(len(plan), 81)
        self.assertEqual(len({p['episode'] for p in plan}), 81)
        states = {(p['route_id'], p['initial_reference_m'], p['condition']['name']) for p in plan}
        self.assertEqual(len(states), 27)
        for state in states:
            matching = [p for p in plan if (p['route_id'], p['initial_reference_m'], p['condition']['name']) == state]
            self.assertEqual({p['variant'] for p in matching}, {'B0', 'resnet', 'hybrid_geometry'})
            self.assertEqual({p['condition']['yaw_deg'] for p in matching}, {0.})
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            trial_matrix([routes()[0]] * 3, ['B0', 'resnet', 'hybrid_geometry'])

    def test_failure_denominator_and_unexecuted_trials_are_distinct(self):
        plan = trial_matrix(routes(), ['B0', 'resnet', 'hybrid_geometry'])
        episodes = []
        for trial in plan:
            if trial['variant'] == 'hybrid_geometry':
                episodes.append(dict(variant=trial['variant'], episode=trial['episode'],
                    gate={'status': 'PASS', 'metrics': {'driving_success': len(episodes) == 0}}))
        result = summarize_attempts(plan, episodes)
        self.assertAlmostEqual(result['hybrid_geometry']['full_protocol_driving_success_rate'], 1/27)
        self.assertEqual(result['hybrid_geometry']['driving_failed_episodes'], 26)
        self.assertEqual(result['B0']['unexecuted_episodes'], 27)
        self.assertIsNone(result['B0']['full_protocol_driving_success_rate'])
        episodes[-1]['gate'] = {'status': 'FAIL'}
        partial = summarize_attempts(plan, episodes)['hybrid_geometry']
        self.assertEqual(partial['operation_or_integrity_failures'], 1)
        self.assertIsNone(partial['full_protocol_driving_success_rate'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            summarize_attempts(plan, [episodes[0], episodes[0]])

    def test_start_policy_is_frozen_and_verified_from_external_protocol(self):
        predictor = Mock(identity={'variant': 'hybrid_geometry'})
        policy = RefinementTrialPolicy(load_baseline_config(), load_evaluation_config(), {'route_id': 'original'},
            {'name': 'center', 'lateral_m': 0., 'yaw_deg': 0.}, predictor=predictor,
            method='hybrid_geometry', initial_reference_m=4.)
        metadata = {}
        policy.configure_metadata(metadata)
        self.assertEqual(metadata['requested_initial_reference_m'], 4.)
        self.assertEqual(metadata['prediction_inputs'], 'rgb-only')
        self.assertFalse(metadata['perception_uses_route'])
        with self.assertRaisesRegex(ValueError, 'frozen-protocol'):
            RefinementTrialPolicy(load_baseline_config(), load_evaluation_config(), {}, {}, predictor=predictor,
                method='hybrid_geometry', initial_reference_m=3.)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'episode.json').write_text(json.dumps({'requested_initial_reference_m': 0.}))
            Path(directory, 'steps.jsonl').write_text('{}\n')
            result = validate_refinement_episode(directory, predictor, expected_initial_reference_m=4.)
            self.assertEqual(result['status'], 'FAIL')
            self.assertIn('refinement-native-requested-start-mismatch', result['errors'])

    def test_start_interpolates_original_route_without_reanchoring(self):
        route = {'route_length_m': 10., 'centerline_anchor_sequence': [
            {'arc_distance_m': 0., 'world_location': {'x': 100., 'y': 50., 'z': 0.}},
            {'arc_distance_m': 10., 'world_location': {'x': 110., 'y': 50., 'z': 0.}}]}
        # Use the contract's original world coordinates, not an origin at (0,0).
        point, yaw = route_initial_location(route, 4.)
        self.assertEqual(point, {'x': 104., 'y': 50., 'z': 0.})
        self.assertEqual(yaw, 0.)
        self.assertEqual(route['centerline_anchor_sequence'][0]['world_location']['x'], 100.)

    def test_baseline_keeps_fixed_nominal_calibration_separate_from_measured_pose(self):
        predictor = NativeBaselinePredictor(load_baseline_config())
        identity = predictor.identity
        self.assertEqual(identity['variant'], 'B0')
        self.assertEqual(identity['device'], 'cpu')
        identity['variant'] = 'changed'
        self.assertEqual(predictor.identity['variant'], 'B0')
        policy = RefinementTrialPolicy(load_baseline_config(), load_evaluation_config(), {'route_id': 'original'},
            {'name': 'center'}, predictor=predictor, method='B0', initial_reference_m=2.)
        self.assertEqual(policy.prediction_inputs, 'rgb-and-fixed-nominal-calibration')

    def test_cpu_video_and_controls_keep_terminal_failure_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'images').mkdir()
            points = [[3.*i, 0.] for i in range(1, 6)]
            rows = []
            for i in range(3):
                frame = 101+i
                path = root/'images'/f'{frame:08d}.png'
                Image.fromarray(np.full((360, 640, 3), i*80, dtype=np.uint8)).save(path)
                rows.append({'frame': frame, 'timestamp': 5.+i*.05,
                    'image': {'path': f'images/{frame:08d}.png', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
                    'speed_mps': 2., 'tracking': {'label': {'waypoints_ego_m': points}, 'reference_arc_m': i*.1,
                        'lateral_error_m': -.3, 'heading_error_deg': 0.},
                    'perception': {'status': 'PASS' if i < 2 else 'FAIL', 'waypoints_ego_m': points if i < 2 else None},
                    'command': {'throttle': .1, 'steer': .2, 'brake': 0.} if i < 2 else None,
                    'termination_reason': None if i < 2 else 'technical_failure', 'latency_ms': {'pipeline_ms': 12.}})
            (root/'episode.json').write_text(json.dumps({'method': 'hybrid_geometry', 'evaluation_metrics':
                {'driving_success': False, 'termination_reason': 'technical_failure'}}))
            (root/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            original = (root/'steps.jsonl').read_bytes()
            result = export_driving_media(root)
            self.assertEqual(result['frame_count'], 3)
            self.assertEqual(result['last_frame'], 103)
            self.assertFalse(result['driving_success'])
            with (root/'controls.csv').open(newline='') as stream:
                controls = list(csv.DictReader(stream))
            self.assertEqual([r['frame'] for r in controls], ['101', '102', '103'])
            self.assertEqual(controls[-1]['prediction_status'], 'FAIL')
            self.assertEqual(controls[-1]['termination_reason'], 'technical_failure')
            self.assertEqual((root/'steps.jsonl').read_bytes(), original)
            with self.assertRaisesRegex(ValueError, 'already-exists'):
                export_driving_media(root)


if __name__ == '__main__':
    unittest.main()
