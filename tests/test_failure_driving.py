import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from carla_tools.failure_driving import SCOPE, REFERENCE_IDENTITY, source_hashes, validate_failure_episode
from carla_tools.failure_analysis import analyze_failure_episode
from scripts.run_failure_pipeline import failure_trial_matrix
from scripts.prepare_kaggle_failure_driving import prepare_failure_driving, EXTRA_SOURCES
from scripts.prepare_kaggle_week1 import NOTEBOOK_ID, NOTEBOOK_TITLE

try:
    from tests.test_baseline_validation import write_baseline, save_sources
except ModuleNotFoundError:
    from test_baseline_validation import write_baseline, save_sources


class FailureDrivingTests(unittest.TestCase):
    def test_paired_matrix_is_frozen_and_covers_all_states_without_final_town(self):
        routes = [{'case': case, 'route': {'route_id': case}} for case in ('straight', 'left_curve', 'right_curve')]
        trials = failure_trial_matrix(routes)
        self.assertEqual(len(trials), 81)
        self.assertEqual(len({t['episode'] for t in trials}), 81)
        states = {(t['route_id'], t['initial_reference_m'], t['condition']['name']) for t in trials}
        self.assertEqual(len(states), 27)
        for state in states:
            arms = {t['variant'] for t in trials if (t['route_id'], t['initial_reference_m'], t['condition']['name']) == state}
            self.assertEqual(arms, {'privileged_reference', 'hybrid_geometry', 'hybrid_geometry_near'})
        with self.assertRaises(ValueError): failure_trial_matrix([routes[0]] * 3)
        with self.assertRaises(ValueError): failure_trial_matrix(routes, 'unknown')

    def test_privileged_reference_replays_labels_controls_and_metrics_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'reference'
            metadata, rows = write_baseline(root, method='privileged_reference')
            metadata.update(scope=SCOPE, native_model=REFERENCE_IDENTITY, native_source_sha256=source_hashes(), requested_initial_reference_m=0.)
            save_sources(root, metadata, rows)
            gate = validate_failure_episode(root, None, expected_initial_reference_m=0.)
            self.assertEqual(gate['status'], 'PASS', gate)
            analysis = analyze_failure_episode(root, integrity_verified=True)
            self.assertEqual(analysis['classification'], 'clean-completion')
            self.assertTrue(analysis['labels_reconstructed'])
            self.assertEqual(analysis['near_error_m']['maximum'], 0.)
            self.assertFalse(analysis['counterfactual_is_new_driving'])
            rows[0]['tracking']['label']['waypoints_ego_m'][0][1] += .5
            save_sources(root, metadata, rows)
            self.assertEqual(validate_failure_episode(root, None, expected_initial_reference_m=0.)['status'], 'FAIL')
            with self.assertRaisesRegex(ValueError, 'label-or-coordinate'):
                analyze_failure_episode(root, integrity_verified=True)

    def test_invalid_perception_is_classified_and_external_integrity_proof_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'failed'
            write_baseline(root, fail_prediction=True)
            with self.assertRaisesRegex(ValueError, 'integrity-proof'):
                analyze_failure_episode(root)
            analysis = analyze_failure_episode(root, integrity_verified=True)
            self.assertEqual(analysis['classification'], 'perception-invalid-path')
            self.assertEqual(analysis['perception_status_counts'], {'FAIL': 1})
            self.assertEqual(analysis['near_error_m']['count'], 0)
            (root / 'steps.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'trace-changed'):
                analyze_failure_episode(root, integrity_verified=True)

    def test_bundle_reuses_exact_existing_identity_and_embeds_exact_remediation_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / 'metadata.json'
            metadata.write_text(json.dumps(dict(id=NOTEBOOK_ID, title=NOTEBOOK_TITLE, is_private=True)))
            routes = root / 'routes.json'
            routes.write_text(json.dumps({'split': 'dev', 'final_town_used': False, 'routes': [{}, {}, {}]}))
            bundle = prepare_failure_driving(metadata, root / 'bundle', dev_routes=routes,
                candidate_sha256='a' * 64, candidate_bytes=100, reference_archive_sha256='b' * 64,
                reference_checkpoint_sha256='c' * 64, reference_version=1,
                candidate_archive_sha256='d' * 64, candidate_version=2)
            submission = json.loads((root / 'bundle/kernel-metadata.json').read_text())
            self.assertEqual((submission['id'], submission['title']), (NOTEBOOK_ID, NOTEBOOK_TITLE))
            self.assertFalse(bundle['creates_new_notebook'])
            self.assertFalse(bundle['creates_dataset'])
            code = (root / 'bundle/native.py').read_text()
            self.assertEqual(hashlib.sha256(code.encode()).hexdigest(), bundle['code_sha256'])
            namespace = {}
            selected = []
            for node in ast.parse(code).body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'_project_payload', 'PROJECT_SOURCES'} for t in node.targets):
                    selected.append(node)
                if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name)
                        and node.value.func.value.id == 'PROJECT_SOURCES' and node.value.func.attr == 'update'):
                    selected.append(node)
            isolated = 'import base64,lzma,json,hashlib\n' + '\n'.join(ast.unparse(n) for n in selected)
            exec(compile(isolated, 'fixture-review', 'exec'), namespace)
            self.assertTrue(set(EXTRA_SOURCES) <= namespace['PROJECT_SOURCES'].keys())
            for name, digest in bundle['source_sha256'].items():
                self.assertEqual(hashlib.sha256(namespace['PROJECT_SOURCES'][name].encode()).hexdigest(), digest)
            self.assertIn('scripts/run_failure_pipeline.py', code)
            self.assertEqual(bundle['evaluated_methods'], ['privileged_reference', 'hybrid_geometry', 'hybrid_geometry_near'])


if __name__ == '__main__': unittest.main()
