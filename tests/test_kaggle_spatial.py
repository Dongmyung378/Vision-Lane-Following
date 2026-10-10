"""Existing notebook identity, immutable transport and actual GPU proof checks."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import prepare_kaggle_spatial as packager
from scripts import kaggle_spatial_bootstrap as bootstrap


class KaggleSpatialTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metadata = self.root / "metadata.json"
        self.metadata.write_text(json.dumps({"id": packager.NOTEBOOK_ID, "title": packager.NOTEBOOK_TITLE,
                                             "is_private": True, "api_key": "SECRET-SENTINEL"}))

    def prepare(self, **kwargs):
        return packager.prepare(self.metadata, self.root / "bundle", pilot_archive_sha256="a" * 64,
                                pilot_manifest_sha256="b" * 64, pilot_download_version=17, **kwargs)

    def test_exact_existing_private_t4_notebook_and_original_source_bytes(self):
        report = self.prepare()
        metadata = json.loads((self.root / "bundle/kernel-metadata.json").read_text())
        code = (self.root / "bundle/spatial_training.py").read_bytes()
        self.assertEqual(metadata["id"], packager.NOTEBOOK_ID)
        self.assertEqual(metadata["title"], packager.NOTEBOOK_TITLE)
        self.assertEqual(metadata["kernel_sources"], [packager.PILOT_NOTEBOOK_ID])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertTrue(metadata["is_private"])
        self.assertNotIn("api_key", metadata)
        self.assertNotIn(b"SECRET-SENTINEL", code)
        self.assertNotIn(b"\r\n", code)
        self.assertFalse(report["creates_new_notebook"])
        self.assertFalse(report["creates_dataset"])
        self.assertFalse(report["local_gpu_used"])
        self.assertEqual(hashlib.sha256(code).hexdigest(), report["code_sha256"])
        namespace = {}
        reviewed = code.decode().rsplit("raise SystemExit(main_spatial())", 1)[0]
        exec(compile(reviewed, "reviewed_bundle", "exec"), namespace)
        for name, digest in report["source_sha256"].items():
            self.assertEqual(namespace["PROJECT_SOURCES"][name].encode(), (packager.ROOT / name).read_bytes())
            self.assertEqual(hashlib.sha256(namespace["PROJECT_SOURCES"][name].encode()).hexdigest(), digest)
        self.assertEqual(namespace["MAX_SECONDS"], 3600)
        self.assertEqual(namespace["TRAINING_SECONDS"], 2700)

    def test_new_or_public_notebook_and_overwrite_are_refused(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "new-local-output"):
            self.prepare()
        self.metadata.write_text(json.dumps({"id": "owner/new-notebook", "title": packager.NOTEBOOK_TITLE,
                                             "is_private": True}))
        with self.assertRaisesRegex(ValueError, "existing-private"):
            self.prepare()

    def test_unbounded_runs_and_unfrozen_pilot_are_refused(self):
        for kwargs in ({"maximum_seconds": 3601}, {"training_seconds": 3500}, {"max_epochs": 0},
                       {"max_epochs": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.prepare(**kwargs)

    def test_reused_reference_bytes_are_frozen_separately_from_public_sources(self):
        reference=self.root/"reference.json"
        reference.write_bytes(b'{"training":{"status":"PASS"}}\r\n')
        report=self.prepare(reused_baseline_path=reference)
        code=(self.root/"bundle/spatial_training.py").read_text()
        namespace={}
        exec(compile(code.rsplit("raise SystemExit(main_spatial())",1)[0],"prepared","exec"),namespace)
        self.assertEqual(namespace["FROZEN_BASELINE"].encode(),reference.read_bytes())
        self.assertEqual(report["reused_baseline_sha256"],hashlib.sha256(reference.read_bytes()).hexdigest())
        self.assertNotIn("reference.json",report["source_sha256"])

    def test_completed_artifacts_are_bound_to_original_bytes(self):
        experiment = self.root / "experiment"
        (experiment / "hybrid/evaluation").mkdir(parents=True)
        checkpoint = experiment / "hybrid/checkpoint.pt"
        checkpoint.write_bytes(b"real-checkpoint-fixture")
        predictions = experiment / "hybrid/evaluation/predictions.jsonl"
        predictions.write_text('{"sample_id":"test"}\n')
        report = bootstrap.preserve_spatial_artifacts(self.root)
        self.assertEqual(report["source_sha256"]["experiment/hybrid/checkpoint.pt"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        with self.assertRaises(FileExistsError):
            bootstrap.preserve_spatial_artifacts(self.root)

    def test_missing_arm_or_unverified_reports_cannot_claim_gpu_completion(self):
        with self.assertRaisesRegex(bootstrap.TrainingBootstrapError, "three-arm-proof"):
            bootstrap.validate_spatial_report({"status": "PASS", "fixture_injection": False,
                                               "final_town_used": False}, 1665, 1305)


if __name__ == "__main__":
    unittest.main()
