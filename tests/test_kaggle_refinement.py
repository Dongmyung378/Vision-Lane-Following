import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_kaggle_refinement import prepare_refinement, EXTRA_SOURCES
from scripts.prepare_kaggle_spatial import NOTEBOOK_ID, NOTEBOOK_TITLE


class RefinementBundleTests(unittest.TestCase):
    def test_existing_identity_source_hashes_and_unchanged_title(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / 'metadata.json'
            metadata.write_text(json.dumps(dict(id=NOTEBOOK_ID, title=NOTEBOOK_TITLE, is_private=True)))
            reference = root / 'reference.json'
            reference.write_text('{}')
            output = root / 'bundle'
            bundle = prepare_refinement(metadata, output, reference_path=reference,
                pilot_archive_sha256='a' * 64, pilot_manifest_sha256='b' * 64, pilot_download_version=17)
            result = json.loads((output / 'kernel-metadata.json').read_text())
            self.assertEqual(result['id'], NOTEBOOK_ID)
            self.assertEqual(result['title'], NOTEBOOK_TITLE)
            self.assertFalse(bundle['creates_new_notebook'])
            self.assertFalse(bundle['creates_dataset'])
            code = (output / result['code_file']).read_bytes()
            self.assertEqual(hashlib.sha256(code).hexdigest(), bundle['code_sha256'])
            self.assertTrue(set(EXTRA_SOURCES) <= set(bundle['source_sha256']))
            self.assertIn('carla_tools/model_selection.py',bundle['source_sha256'])
            self.assertIn('scripts/select_waypoint_model.py',code.decode())
            self.assertIn('current_selection',code.decode())
            namespace = {}
            exec(compile(code.decode().rsplit('raise SystemExit(main_spatial())', 1)[0], 'review', 'exec'), namespace)
            for name, digest in bundle['source_sha256'].items():
                self.assertEqual(hashlib.sha256(namespace['PROJECT_SOURCES'][name].encode()).hexdigest(), digest)
            self.assertEqual(namespace['FROZEN_BASELINE'], '{}')

    def test_wrong_notebook_identity_is_refused_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / 'metadata.json'
            metadata.write_text(json.dumps(dict(id='different/notebook', title=NOTEBOOK_TITLE, is_private=True)))
            with self.assertRaisesRegex(ValueError, 'existing-private'):
                prepare_refinement(metadata, root / 'bundle', reference_path=root / 'absent',
                    pilot_archive_sha256='a' * 64, pilot_manifest_sha256='b' * 64, pilot_download_version=17)
            self.assertFalse((root / 'bundle').exists())


if __name__ == '__main__':
    unittest.main()
