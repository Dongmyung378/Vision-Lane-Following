import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import zipfile
import numpy as np

from carla_tools.baseline import load_baseline_config
from carla_tools.evaluation import load_evaluation_config
from carla_tools.refinement_driving import RefinementTrialPolicy
from carla_tools.refinement_validation import _RecordedReplay
from scripts.prepare_kaggle_refinement_driving import unpack_reference, prepare_native, NOTEBOOK_ID, NOTEBOOK_TITLE


class RefinementDrivingTests(unittest.TestCase):
    def test_predictor_gets_rgb_and_controller_gets_only_points_speed_and_clock(self):
        points = [[3., 0.], [6., 0.], [9., 0.], [12., 0.], [15., 0.]]
        predictor = Mock()
        predictor.identity = {'variant': 'hybrid_geometry'}
        predictor.predict.return_value = dict(status='PASS', waypoints_ego_m=points, diagnostics={}, timing_ms=1.)
        controller = Mock()
        controller.geometry.rear_axle_x_m, controller.geometry.rear_axle_y_m = -1., 0.
        policy = RefinementTrialPolicy(load_baseline_config(), load_evaluation_config(), {'sensitive_route': True},
            {'name': 'center', 'lateral_m': 0., 'yaw_deg': 0.}, predictor=predictor, method='hybrid_geometry')
        pixels = np.random.default_rng(42).integers(0, 256, (360, 640, 4), dtype=np.uint8)
        image = Mock(width=640, height=360, raw_data=pixels.tobytes())
        command, perception, latency = policy.command(image, 2., controller)
        predictor.predict.assert_called_once()
        self.assertEqual(set(predictor.predict.call_args.kwargs), {'width', 'height'})
        self.assertEqual(len(predictor.predict.call_args.args), 1)
        controller.step.assert_called_once_with(points, 2., dt_seconds=.05)
        self.assertAlmostEqual(latency['pipeline_ms'], sum(latency[k] for k in ('preprocess_ms', 'inference_ms', 'controller_ms')))

    def test_recorded_replay_rejects_changed_rgb_and_extra_frame(self):
        rgb = b'original-image-bytes'
        perception = {'status': 'PASS', 'waypoints_ego_m': [[3., 0.]], 'timing_ms': 1.}
        replay = _RecordedReplay([{'rgb_sha256': hashlib.sha256(rgb).hexdigest(), 'perception': perception}])
        with self.assertRaises(Exception):
            replay.predict(b'changed', width=640, height=360)
        self.assertEqual(replay.index, 0)
        result = replay.predict(rgb, width=640, height=360)
        result['status'] = 'mutated'
        self.assertEqual(perception['status'], 'PASS')
        with self.assertRaises(Exception):
            replay.predict(rgb, width=640, height=360)

    def test_reference_archive_hash_and_member_checkpoint_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = b'fixed-reference-weights'
            expected = hashlib.sha256(checkpoint).hexdigest()
            archive = root / 'reference.zip'
            with zipfile.ZipFile(archive, 'x') as stream:
                stream.writestr('experiment/resnet/checkpoint.pt', checkpoint)
                stream.writestr('experiment/resnet/training_report.json', json.dumps(dict(status='PASS', variant='resnet', checkpoint_sha256=expected)))
                stream.writestr('../../unrelated', b'never-extracted')
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            paths = unpack_reference(archive, root / 'verified', digest, expected)
            self.assertEqual(paths[0].read_bytes(), checkpoint)
            self.assertFalse((root / 'unrelated').exists())
            with self.assertRaisesRegex(ValueError, 'archive-hash'):
                unpack_reference(archive, root / 'bad', 'a' * 64, expected)
            self.assertFalse((root / 'bad').exists())

    def test_native_bundle_keeps_existing_id_and_uses_existing_training_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, routes = root / 'metadata.json', root / 'routes.json'
            metadata.write_text(json.dumps(dict(id=NOTEBOOK_ID, title=NOTEBOOK_TITLE, is_private=True)))
            routes.write_text(json.dumps(dict(split='dev', final_town_used=False, routes=[{}, {}, {}])))
            output = root / 'bundle'
            result = prepare_native(metadata, output, candidate_sha256='a'*64, candidate_bytes=100,
                dev_routes=routes, reference_archive_sha256='b'*64, reference_checkpoint_sha256='c'*64)
            submitted = json.loads((output / 'kernel-metadata.json').read_text())
            self.assertEqual(submitted['id'], NOTEBOOK_ID)
            self.assertEqual(submitted['title'], NOTEBOOK_TITLE)
            self.assertEqual(submitted['kernel_sources'], ['dongmyungpark/vlf-carla-vulkan-preflight-20261005'])
            code = (output / 'native.py').read_bytes()
            self.assertEqual(hashlib.sha256(code).hexdigest(), result['code_sha256'])
            # Inspect transport assignments on Windows without importing Linux
            # pwd/grp or executing the simulator/bootstrap function bodies.
            namespace = {}
            selected = {'_project_payload', 'PROJECT_SOURCES', 'DEV_ROUTES_BYTES', 'REFERENCE_HANDLE'}
            assignments = [node for node in ast.parse(code.decode()).body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id in selected for t in node.targets)]
            isolated = 'import base64,zlib,lzma,json,hashlib\n' + '\n'.join(ast.unparse(node) for node in assignments)
            exec(compile(isolated, 'review-transport', 'exec'), namespace)
            self.assertEqual(namespace['REFERENCE_HANDLE'], 'dongmyungpark/vlf-carla-vulkan-preflight-20261005/versions/16')
            self.assertEqual(namespace['DEV_ROUTES_BYTES'], routes.read_bytes())
            for name, digest in result['source_sha256'].items():
                self.assertEqual(hashlib.sha256(namespace['PROJECT_SOURCES'][name].encode()).hexdigest(), digest)
            archived = root / 'archived-bundle'
            pinned = prepare_native(metadata, archived, candidate_sha256='a'*64, candidate_bytes=100,
                dev_routes=routes, reference_archive_sha256='b'*64, reference_checkpoint_sha256='c'*64,
                candidate_archive_sha256='d'*64, candidate_version=18)
            archived_metadata = json.loads((archived / 'kernel-metadata.json').read_text())
            self.assertEqual(archived_metadata['kernel_sources'], [])
            self.assertEqual(pinned['candidate_transport'], 'hash-pinned-saved-version-download')
            archived_code = (archived / 'native.py').read_text()
            self.assertIn('selected_checkpoint = None', archived_code)
            self.assertIn("CANDIDATE_VERSION=18", archived_code)
            self.assertIn('native-archived-candidate-size-mismatch', archived_code)
            compile(archived_code, 'archived-native', 'exec')


if __name__ == '__main__':
    unittest.main()
