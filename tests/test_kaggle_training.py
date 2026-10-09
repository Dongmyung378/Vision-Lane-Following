"""Training packages retain notebook identity and guard attached pilot bytes."""

from __future__ import annotations

import hashlib
import json
import copy
import tempfile
import unittest
import zipfile
from unittest.mock import patch
import contextlib
import io
import sys
import types
from pathlib import Path

from scripts.prepare_kaggle_training import (NOTEBOOK_ID, NOTEBOOK_TITLE, PILOT_NOTEBOOK_ID,
                                            PILOT_DATASET_ID, SOURCES, prepare)
from scripts.kaggle_training_bootstrap import (TrainingBootstrapError, safe_extract_pilot,
                                             export_runtime_sources, validate_resume_proof,
                                             resolve_frozen_pilot, cloud_download_helper_source)


class KaggleTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metadata = self.root / "metadata.json"
        self.metadata.write_text(json.dumps({"id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE,
                                             "api_key": "DO-NOT-COPY-THIS-SENTINEL"}))

    def prepare(self, **kwargs):
        return prepare(self.metadata, self.root / "bundle", pilot_archive_sha256="a" * 64,
                       pilot_manifest_sha256="b" * 64, **kwargs)

    def test_existing_identity_sources_and_credentials_filtered(self):
        report = self.prepare()
        metadata = json.loads((self.root / "bundle/kernel-metadata.json").read_text())
        self.assertEqual(metadata["id"], NOTEBOOK_ID)
        self.assertEqual(metadata["title"], NOTEBOOK_TITLE)
        self.assertEqual(metadata["kernel_sources"], [PILOT_NOTEBOOK_ID])
        self.assertTrue(metadata["enable_gpu"])
        self.assertNotIn("DO-NOT-COPY-THIS-SENTINEL", (self.root / "bundle/training.py").read_text())
        self.assertNotIn("api_key", metadata)
        self.assertFalse(report["creates_new_notebook"])
        self.assertEqual(set(report["source_sha256"]), set(SOURCES))

    def test_refuses_wrong_existing_identity(self):
        self.metadata.write_text(json.dumps({"id": "owner/new-experiment", "title": NOTEBOOK_TITLE}))
        with self.assertRaisesRegex(ValueError, "id-or-title-mismatch"):
            self.prepare()
        self.assertFalse((self.root / "bundle").exists())

    def test_private_dataset_transport_preserves_identity_and_frozen_hashes(self):
        self.metadata.write_text(json.dumps({"id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE,
                                             "dataset_sources": ["owner/existing-source"]}))
        report = self.prepare(pilot_dataset=PILOT_DATASET_ID)
        metadata = json.loads((self.root / "bundle/kernel-metadata.json").read_text())
        self.assertEqual(metadata["id"], NOTEBOOK_ID)
        self.assertEqual(metadata["title"], NOTEBOOK_TITLE)
        self.assertEqual(metadata["dataset_sources"], ["owner/existing-source", PILOT_DATASET_ID])
        self.assertEqual(metadata["kernel_sources"], [])
        self.assertEqual(report["pilot_archive_sha256"], "a" * 64)
        self.assertEqual(report["pilot_manifest_sha256"], "b" * 64)
        self.assertEqual(report["pilot_transport"], "private-dataset")
        self.assertFalse(report["creates_new_notebook"])

    def test_private_dataset_transport_rejects_unapproved_identifier(self):
        with self.assertRaisesRegex(ValueError, "pilot-dataset-not-approved-stable-id"):
            self.prepare(pilot_dataset="owner/new-experiment-dataset")
        self.assertFalse((self.root / "bundle").exists())

    def test_existing_dataset_attachment_is_not_duplicated(self):
        self.metadata.write_text(json.dumps({"id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE,
                                             "dataset_sources": [PILOT_DATASET_ID]}))
        self.prepare(pilot_dataset=PILOT_DATASET_ID)
        metadata = json.loads((self.root / "bundle/kernel-metadata.json").read_text())
        self.assertEqual(metadata["dataset_sources"], [PILOT_DATASET_ID])

    def test_collection_version_pin_preserves_notebook_identity_and_hashes(self):
        report = self.prepare(pilot_kernel_version=14)
        metadata = json.loads((self.root / "bundle/kernel-metadata.json").read_text())
        self.assertEqual(metadata["id"], NOTEBOOK_ID)
        self.assertEqual(metadata["title"], NOTEBOOK_TITLE)
        self.assertEqual(metadata["kernel_sources"], [PILOT_NOTEBOOK_ID + "/14"])
        self.assertEqual(metadata["dataset_sources"], [])
        self.assertEqual(report["pilot_kernel_version"], 14)
        self.assertEqual(report["pilot_archive_sha256"], "a" * 64)
        self.assertEqual(report["pilot_manifest_sha256"], "b" * 64)
        self.assertFalse(report["creates_new_notebook"])

    def test_collection_version_must_be_positive_integer(self):
        for version in (0, -1, "14", True, 1.5):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "positive-integer"):
                self.prepare(pilot_kernel_version=version)
        self.assertFalse((self.root / "bundle").exists())

    def test_collection_version_and_dataset_cannot_be_combined(self):
        with self.assertRaisesRegex(ValueError, "mutually-exclusive"):
            self.prepare(pilot_dataset=PILOT_DATASET_ID, pilot_kernel_version=14)
        self.assertFalse((self.root / "bundle").exists())

    def test_requires_immutable_pilot_hashes(self):
        with self.assertRaisesRegex(ValueError, "frozen-pilot-hashes"):
            prepare(self.metadata, self.root / "bundle", pilot_archive_sha256="", pilot_manifest_sha256="b" * 64)

    def test_refuses_existing_package(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "already-exists"):
            self.prepare()

    def test_safe_archive_and_manifest(self):
        archive = self.root / "pilot.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("pilot/pilot_manifest.json", '{"status":"COMPLETE"}')
        manifest = safe_extract_pilot(archive, self.root / "extracted")
        self.assertEqual(json.loads(manifest.read_text())["status"], "COMPLETE")

    def test_path_escape_rejected_before_writes(self):
        archive = self.root / "pilot.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("../escape.json", "unsafe")
            output.writestr("pilot/pilot_manifest.json", "{}")
        with self.assertRaisesRegex(TrainingBootstrapError, "unsafe-member"):
            safe_extract_pilot(archive, self.root / "extracted")
        self.assertFalse((self.root / "escape.json").exists())

    def test_duplicate_member_rejected(self):
        archive = self.root / "pilot.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("pilot/pilot_manifest.json", "{}")
            output.writestr("pilot/duplicate", "first")
            output.writestr("pilot/duplicate", "second")
        with self.assertRaisesRegex(TrainingBootstrapError, "unsafe-member"):
            safe_extract_pilot(archive, self.root / "extracted")

    def test_export_runtime_actual_bytes_and_hash(self):
        project, output = self.root / "project", self.root / "out"
        project.mkdir()
        output.mkdir()
        source = "example = 1\n"
        (project / "example.py").write_text(source, newline="\n")
        result = export_runtime_sources(project, output, {"example.py": source})
        self.assertEqual(result["source_sha256"]["example.py"], hashlib.sha256(source.encode()).hexdigest())
        with zipfile.ZipFile(output / "training_runtime_sources.zip") as archive:
            self.assertEqual(archive.read("example.py"), source.encode())

    def test_runtime_changed_source_refused(self):
        project, output = self.root / "project", self.root / "out"
        project.mkdir()
        output.mkdir()
        (project / "example.py").write_text("changed = 2\n")
        with self.assertRaisesRegex(TrainingBootstrapError, "source-mismatch"):
            export_runtime_sources(project, output, {"example.py": "example = 1\n"})

    def resume_reports(self):
        initial = {"status": "PASS", "completed_epochs": 2, "history": [{"epoch": 1}, {"epoch": 2}]}
        train = {"status": "PASS", "completed_epochs": 3,
                 "history": initial["history"] + [{"epoch": 3}],
                 "runtime": {"device": "cuda", "amp_enabled": True},
                 "resume": {"start_epoch": 2, "checkpoint_sha256": "a" * 64,
                            "optimizer_state_restored": True, "scaler_state_restored": True,
                            "rng_state_restored": True, "sampler_state_restored": True}}
        return initial, train

    def test_actual_gpu_resume_proof_requires_all_states_and_history(self):
        initial, train = self.resume_reports()
        report = validate_resume_proof(initial, train, "a" * 64)
        self.assertTrue(report["history_prefix_preserved"])
        self.assertEqual(report["start_epoch"], 2)
        for key in ("optimizer_state_restored", "scaler_state_restored", "rng_state_restored", "sampler_state_restored"):
            changed = copy.deepcopy(train)
            changed["resume"][key] = False
            with self.subTest(key=key), self.assertRaises(TrainingBootstrapError):
                validate_resume_proof(initial, changed, "a" * 64)

    def test_cpu_resume_cannot_claim_actual_gpu_proof(self):
        initial, train = self.resume_reports()
        train["runtime"]["device"] = "cpu"
        with self.assertRaises(TrainingBootstrapError):
            validate_resume_proof(initial, train, "a" * 64)

    def test_resume_different_checkpoint_or_history_rejected(self):
        initial, train = self.resume_reports()
        with self.assertRaises(TrainingBootstrapError):
            validate_resume_proof(initial, train, "b" * 64)
        train["history"][0] = {"epoch": 99}
        with self.assertRaises(TrainingBootstrapError):
            validate_resume_proof(initial, train, "a" * 64)

    def test_mounted_exact_archive_precedes_explicit_cloud_fallback(self):
        inputs = self.root / "input"
        inputs.mkdir()
        archive = inputs / "pilot_artifacts.zip"
        archive.write_bytes(b"exact-cloud-bytes")
        sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        calls = []
        result, transport = resolve_frozen_pilot(inputs, sha, download_version=16,
                                                 cloud_download=lambda version: calls.append(version))
        self.assertEqual(result, archive)
        self.assertEqual(transport, "mounted-output")
        self.assertEqual(calls, [])

    def test_cloud_fallback_requires_explicit_version_and_exact_archive_hash(self):
        inputs = self.root / "input"
        inputs.mkdir()
        archive = self.root / "pilot_artifacts.zip"
        archive.write_bytes(b"exact-existing-output")
        sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        calls = []
        def download(version):
            calls.append(version)
            return archive
        result, transport = resolve_frozen_pilot(inputs, sha, download_version=16, cloud_download=download)
        self.assertEqual(calls, [16])
        self.assertEqual(result, archive)
        self.assertEqual(transport, "kagglehub-existing-notebook-output")
        with self.assertRaisesRegex(TrainingBootstrapError, "sha256-mismatch"):
            resolve_frozen_pilot(inputs, "0" * 64, download_version=16, cloud_download=download)
        with self.assertRaisesRegex(TrainingBootstrapError, "not-mounted"):
            resolve_frozen_pilot(inputs, sha, cloud_download=download)
        self.assertEqual(calls, [16, 16])

    def test_cloud_helper_suppresses_library_messages_and_sensitive_errors(self):
        def denied(*args, **kwargs):
            print("SECRET-TOKEN-AND-SIGNED-URL")
            raise RuntimeError("SECRET-TOKEN-AND-SIGNED-URL")
        fake = types.SimpleNamespace(notebook_output_download=denied, __version__="test")
        source = cloud_download_helper_source(16, self.root, self.root / "download")
        self.assertIn(PILOT_NOTEBOOK_ID + "/versions/16", source)
        output = io.StringIO()
        with patch.dict(sys.modules, {"kagglehub": fake}), patch.object(sys, "path", list(sys.path)), \
                patch("logging.disable"), contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit):
                exec(compile(source, "cloud_helper", "exec"), {})
        self.assertNotIn("SECRET-TOKEN-AND-SIGNED-URL", output.getvalue())
        self.assertEqual(json.loads(output.getvalue()), {"status": "FAIL", "exception_type": "RuntimeError"})

    def test_cloud_version_option_is_explicit_and_fail_closed(self):
        report = self.prepare(pilot_download_version=16)
        self.assertEqual(report["pilot_download_version"], 16)
        self.assertIn("PILOT_DOWNLOAD_VERSION = 16", (self.root / "bundle/training.py").read_text())
        for version in (0, -1, "16", True, 1.5):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "positive-integer"):
                prepare(self.metadata, self.root / "invalid", pilot_archive_sha256="a" * 64,
                        pilot_manifest_sha256="b" * 64, pilot_download_version=version)
        with self.assertRaisesRegex(ValueError, "cloud-and-kernel-version-mismatch"):
            prepare(self.metadata, self.root / "invalid", pilot_archive_sha256="a" * 64,
                    pilot_manifest_sha256="b" * 64, pilot_download_version=16, pilot_kernel_version=15)


if __name__ == "__main__":
    unittest.main()
