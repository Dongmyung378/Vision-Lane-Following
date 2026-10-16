import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from carla_tools.driving_demo import CASES, build_driving_demo
from carla_tools.driving_media import export_driving_media


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root, *, method='hybrid<script>', learned=True, straight_frames=42):
    records = []
    for case_index, case in enumerate(CASES):
        directory = root / 'episodes' / f'case{case_index}'
        images = directory / 'images'
        images.mkdir(parents=True)
        rows, image_hashes = [], {}
        frames = straight_frames if case == 'straight' else 5
        completed = case != 'right_curve'
        margin = .3 if case == 'straight' else -.1
        reason = 'route_completed' if completed else 'lane_departure'
        for index in range(frames):
            frame = 10 + index
            image_path = f'images/{frame:08d}.png'
            # Distinct frame colors make preservation of complete episodes
            # meaningful, including the final failed frame.
            rgb = np.full((360, 640, 3), 50 + index, dtype=np.uint8)
            assert cv2.imwrite(str(directory / image_path), rgb)
            digest = _sha(directory / image_path)
            image_hashes[image_path] = digest
            terminal = index == frames - 1
            rows.append({'frame': frame, 'timestamp': index * .05, 'terminal': terminal,
                'image': {'path': image_path, 'sha256': digest}, 'speed_mps': 3.,
                'body_lane_margin_m': margin,
                'tracking': {'reference_arc_m': index * .3, 'lateral_error_m': 0. if case == 'straight' else .9,
                    'heading_error_deg': 0., 'label': {'waypoints_ego_m': [[3., 0.], [6., 0.], [9., 0.], [12., 0.], [15., 0.]]}},
                'perception': {'status': 'PASS', 'waypoints_ego_m': [[3., 0.], [6., 0.], [9., 0.], [12., 0.], [15., 0.]]},
                'command': {'throttle': .3, 'steer': 0., 'brake': 0.},
                'termination_reason': reason if terminal else None, 'latency_ms': {'pipeline_ms': 12.}})
        trace = directory / 'steps.jsonl'
        trace.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        metrics = {'driving_success': completed, 'termination_reason': reason, 'collision_count': 0,
            'lane_invasion_count': 0 if case == 'straight' else 1, 'minimum_body_lane_margin_m': margin}
        model = {'variant': method}
        if learned:
            model['checkpoint_sha256'] = 'a' * 64
        metadata = {'status': 'COMPLETE', 'method': method, 'geometry_case': case,
            'requested_initial_reference_m': 0., 'condition': {'name': 'center'}, 'step_count': len(rows),
            'trace_sha256': _sha(trace), 'evaluation_metrics': metrics, 'native_model': model,
            'perception_uses_route': not learned, 'perception_uses_vehicle_pose': not learned,
            'prediction_inputs': 'rgb-only' if learned else 'privileged-original-route-and-same-frame-pose'}
        metadata_path = directory / 'episode.json'
        metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
        media = export_driving_media(directory)
        gate = {'status': 'PASS', 'closed_loop_executed': True, 'frame_count': len(rows), 'metrics': metrics,
            'native': {'status': 'PASS', 'source': {'metadata_sha256': _sha(metadata_path),
                'trace_sha256': _sha(trace), 'image_sha256': image_hashes}}}
        records.append({'variant': method, 'case': case, 'condition': 'center', 'initial_reference_m': 0.,
            'episode': directory.relative_to(root).as_posix(), 'gate': gate, 'media': media})
    return records


class DrivingDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A fresh clone does not contain ignored local artifact directories.
        (Path(__file__).resolve().parents[1] / 'results/local').mkdir(parents=True, exist_ok=True)

    def test_fixed_selection_preserves_full_success_and_failed_episodes_and_escapes_labels(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'results/local') as temporary:
            root = Path(temporary)
            records = _fixture(root / 'native')
            # Selection order does not depend on record order or outcomes.
            manifest = build_driving_demo(root / 'native', list(reversed(records)), root / 'demo',
                                          latency_report={'pipeline_p95_ms': 14.})
            self.assertTrue(manifest['m1_gate_pass'])
            self.assertEqual(manifest['selected_episode_count'], 3)
            self.assertEqual([x['case'] for x in manifest['episodes']], list(CASES))
            self.assertEqual([x['outcome'] for x in manifest['episodes']],
                ['clean_completed', 'goal_completed_with_lane_or_collision_issue', 'driving_failure'])
            self.assertFalse(manifest['failed_episodes_truncated'])
            for item in manifest['episodes']:
                for name, identity in item['files'].items():
                    target = root / 'demo' / identity['path']
                    self.assertEqual(_sha(target), identity['sha256'])
                    self.assertEqual(target.read_bytes(), (root / 'native' / item['source_episode'] / name).read_bytes())
            page = (root / 'demo/index.html').read_text(encoding='utf-8')
            self.assertIn('hybrid&lt;script&gt;', page)
            self.assertNotIn('hybrid<script>', page)
            self.assertIn('lane_departure', page)
            self.assertIn('does not establish real-time', page)
            self.assertEqual(_sha(root / 'demo/index.html'), manifest['html_sha256'])
            self.assertEqual(json.loads((root / 'demo/manifest.json').read_text(encoding='utf-8')), manifest)

    def test_missing_duplicate_unverified_and_traversal_records_reject_before_output_creation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'results/local') as temporary:
            root = Path(temporary)
            records = _fixture(root / 'native', straight_frames=5)
            bad_sets = [records[:-1], records + [records[0]]]
            unverified = copy.deepcopy(records)
            unverified[0]['gate']['status'] = 'FAIL'
            bad_sets.append(unverified)
            escaped = copy.deepcopy(records)
            escaped[0]['episode'] = '../outside'
            bad_sets.append(escaped)
            for index, bad in enumerate(bad_sets):
                output = root / f'bad{index}'
                with self.assertRaises(ValueError):
                    build_driving_demo(root / 'native', bad, output)
                self.assertFalse(output.exists())

    def test_gate_source_or_video_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'results/local') as temporary:
            root = Path(temporary)
            records = _fixture(root / 'native', straight_frames=5)
            first = root / 'native' / records[0]['episode']
            video = first / 'driving.mp4'
            original = video.read_bytes()
            video.write_bytes(original + b'tampered')
            with self.assertRaisesRegex(ValueError, 'media-source-hash-mismatch'):
                build_driving_demo(root / 'native', records, root / 'bad-video')
            self.assertFalse((root / 'bad-video').exists())
            video.write_bytes(original)
            (first / 'episode.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'gate-source-hash-mismatch'):
                build_driving_demo(root / 'native', records, root / 'bad-metadata')

    def test_reference_or_short_clean_route_does_not_pass_learned_gate(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'results/local') as temporary:
            root = Path(temporary)
            for learned, frames, name in ((False, 42, 'reference'), (True, 5, 'short')):
                records = _fixture(root / name, learned=learned, straight_frames=frames)
                manifest = build_driving_demo(root / name, records, root / (name + '-demo'))
                self.assertFalse(manifest['m1_gate_pass'])
                self.assertEqual(manifest['qualifying_routes'], [])

    def test_existing_output_and_inside_native_output_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'results/local') as temporary:
            root = Path(temporary)
            records = _fixture(root / 'native', straight_frames=5)
            (root / 'existing').mkdir()
            with self.assertRaisesRegex(ValueError, 'output-already-exists'):
                build_driving_demo(root / 'native', records, root / 'existing')
            with self.assertRaisesRegex(ValueError, 'output-inside-native-source'):
                build_driving_demo(root / 'native', records, root / 'native/new-demo')


if __name__ == '__main__':
    unittest.main()
