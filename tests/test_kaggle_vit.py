"""ViT packages preserve notebook identity, input bytes and remote GPU bounds."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import sys
import types
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import prepare_kaggle_vit as packager
from scripts import kaggle_vit_bootstrap as bootstrap


class KaggleViTTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metadata = self.root / "metadata.json"
        self.metadata.write_text(json.dumps({
            "id": packager.NOTEBOOK_ID, "title": packager.NOTEBOOK_TITLE,
            "is_private": True, "api_key": "PRIVATE-AUTH-SENTINEL",
            "dataset_sources": ["owner/already-attached-data"],
        }))
        self.fixture = self.root / "public"
        (self.fixture / "scripts").mkdir(parents=True)
        self.source = b"example = 42\r\n"
        (self.fixture / "example.py").write_bytes(self.source)
        actual_root = packager.ROOT
        for name in ("kaggle_training_bootstrap.py", "kaggle_vit_bootstrap.py"):
            (self.fixture / "scripts" / name).write_bytes((actual_root / "scripts" / name).read_bytes())

    def prepare(self, **kwargs):
        with patch.object(packager, "ROOT", self.fixture), patch.object(packager, "SOURCES", ("example.py",)):
            return packager.prepare(self.metadata, self.root / "bundle",
                                    pilot_archive_sha256="a" * 64,
                                    pilot_manifest_sha256="b" * 64,
                                    pilot_download_version=17, **kwargs)

    def test_reuses_private_notebook_and_exact_existing_output_without_upload(self):
        result = self.prepare()
        directory = self.root / "bundle"
        metadata = json.loads((directory / "kernel-metadata.json").read_text())
        self.assertEqual(metadata["id"], packager.NOTEBOOK_ID)
        self.assertEqual(metadata["title"], packager.NOTEBOOK_TITLE)
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(metadata["dataset_sources"], ["owner/already-attached-data"])
        self.assertEqual(metadata["kernel_sources"], [packager.PILOT_NOTEBOOK_ID])
        self.assertTrue(metadata["is_private"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertFalse(result["submitted"])
        self.assertFalse(result["creates_new_notebook"])
        self.assertFalse(result["creates_dataset"])
        self.assertFalse(result["local_gpu_used"])
        self.assertEqual(result["pilot_download_version"], 17)
        self.assertEqual(result["kernel_sources"], [packager.PILOT_NOTEBOOK_ID])
        self.assertNotIn("PRIVATE-AUTH-SENTINEL", (directory / "vit_training.py").read_text())
        self.assertNotIn("api_key", metadata)

    def test_embedded_source_preserves_crlf_and_raw_sha256(self):
        result = self.prepare()
        code = (self.root / "bundle/vit_training.py").read_text(encoding="utf-8")
        # Remove only the final execution expression. Importing/materializing
        # the bundle for review must not start a GPU job or network operation.
        source = code.rsplit("raise SystemExit(main_vit())", 1)[0]
        namespace = {}
        with patch.object(bootstrap, "main_vit", side_effect=AssertionError("must-not-run")):
            exec(compile(source, "prepared_vit", "exec"), namespace)
        self.assertEqual(namespace["PROJECT_SOURCES"]["example.py"].encode("utf-8"), self.source)
        self.assertEqual(result["source_sha256"]["example.py"], hashlib.sha256(self.source).hexdigest())
        self.assertEqual(namespace["MAX_SECONDS"], 2400)
        self.assertEqual(namespace["TRAINING_SECONDS"], 1200)
        self.assertEqual(namespace["PILOT_DOWNLOAD_VERSION"], 17)
        self.assertEqual(namespace["PRETRAINED_SHA256"], bootstrap.PRETRAINED_SHA256)
        self.assertEqual(hashlib.sha256(code.encode("utf-8")).hexdigest(), result["code_sha256"])
        self.assertNotIn(b"\r\n", (self.root / "bundle/vit_training.py").read_bytes())

    def test_private_existing_identity_is_required(self):
        for field, value in (("id", "owner/new-notebook"), ("title", "new title"), ("is_private", False)):
            metadata = {"id": packager.NOTEBOOK_ID, "title": packager.NOTEBOOK_TITLE, "is_private": True}
            metadata[field] = value
            self.metadata.write_text(json.dumps(metadata))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "existing-private"):
                self.prepare()
        self.assertFalse((self.root / "bundle").exists())

    def test_mutable_or_invalid_pilot_input_is_rejected(self):
        with patch.object(packager, "ROOT", self.fixture), patch.object(packager, "SOURCES", ("example.py",)):
            for value in ("", "A" * 64, "g" * 64, None):
                with self.subTest(hash=value), self.assertRaisesRegex(ValueError, "frozen-pilot-hashes"):
                    packager.prepare(self.metadata, self.root / "bundle", pilot_archive_sha256=value,
                                     pilot_manifest_sha256="b" * 64, pilot_download_version=17)
            for value in (None, True, "17", 0, -1):
                with self.subTest(version=value), self.assertRaisesRegex(ValueError, "positive-integer"):
                    packager.prepare(self.metadata, self.root / "bundle", pilot_archive_sha256="a" * 64,
                                     pilot_manifest_sha256="b" * 64, pilot_download_version=value)

    def test_time_and_epoch_limits_fail_closed(self):
        for kwargs in ({"maximum_seconds": 2401}, {"maximum_seconds": True},
                       {"maximum_seconds": 600}, {"training_seconds": 2400},
                       {"training_seconds": 0}, {"training_seconds": True},
                       {"max_epochs": 0}, {"max_epochs": 65}, {"max_epochs": True}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "limit-invalid"):
                self.prepare(**kwargs)
        self.assertFalse((self.root / "bundle").exists())

    def test_package_refuses_overwrite(self):
        self.prepare()
        original = (self.root / "bundle/vit_training.py").read_bytes()
        with self.assertRaisesRegex(ValueError, "already-exists"):
            self.prepare()
        self.assertEqual((self.root / "bundle/vit_training.py").read_bytes(), original)

    def test_existing_attachment_fields_are_validated(self):
        metadata = json.loads(self.metadata.read_text())
        metadata["model_sources"] = ["owner/model with space"]
        self.metadata.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "source-list-invalid"):
            self.prepare()
        self.assertFalse((self.root / "bundle").exists())

    def test_actual_fp16_t4_proof_is_required(self):
        valid = {"cuda_available": True, "gpu": "Tesla T4", "precision": "float16",
                 "backward_verified": True, "torch": "2.7.1+cu126", "torchvision": "0.22.1+cu126",
                 "timm": "1.0.20"}
        self.assertEqual(bootstrap.validate_cuda_smoke(valid), valid)
        for field, value in (("gpu", "RTX 3060"), ("cuda_available", False),
                             ("precision", "bfloat16"), ("backward_verified", False),
                             ("timm", "unversioned"), ("torch", "2.8.0")):
            changed = dict(valid, **{field: value})
            with self.subTest(field=field), self.assertRaisesRegex(bootstrap.TrainingBootstrapError, "t4-fp16"):
                bootstrap.validate_cuda_smoke(changed)

    def reports(self):
        provenance = {"pretrained": True, "hash_verified": True,
                      "sha256": bootstrap.PRETRAINED_SHA256, "url": bootstrap.PRETRAINED_URL}
        train = {"status": "PASS", "completed_epochs": 32,
                 "runtime": {"device": "cuda", "gpu": "Tesla T4", "encoder_amp_fp16": True, "head_precision": "float32"},
                 "training_protocol": "frozen-feature-head-only", "fixture_injection": False, "final_town_used": False,
                 "pretrained_provenance": provenance, "pretrained_revision": bootstrap.PRETRAINED_REVISION,
                 "manifest_sha256": "b" * 64, "source_sha256": {"source.py": "c" * 64}, "checkpoint_sha256": "d" * 64,
                 "feature_cache": {split: {"source": "RGB-only", "encoder_frozen": True, "encoder_amp_fp16": True}
                                   for split in ("train", "dev")},
                 "overfit_check": {"status": "PASS", "encoder_frozen": True, "head_reset_to_fresh_initialization": True}}
        evaluation = {"status": "PASS", "split": "dev", "final_town_used": False,
                      "sample_count": 1305, "manifest_sha256": "b" * 64,
                      "performance_target_passed": False, "metrics": [{"far_mean_m": 1.5}],
                      "fixture_injection": False, "pretrained_revision": bootstrap.PRETRAINED_REVISION,
                      "pretrained_provenance": provenance, "source_sha256": train["source_sha256"], "checkpoint_sha256": "d" * 64,
                      "cached_feature_prediction_check": {"checked": True}}
        return train, evaluation, {"split_counts": {"train": 1665, "dev": 1305}}

    def test_accuracy_failure_is_retained_as_valid_completed_experiment(self):
        train, evaluation, pilot = self.reports()
        with patch.object(bootstrap, "PILOT_MANIFEST_SHA256", "b" * 64):
            report = bootstrap.validate_experiment_reports(train, evaluation, pilot)
        self.assertFalse(report["performance_target_passed"])
        self.assertEqual(report["sample_count"], 1305)

    def test_partial_eval_or_final_town_cannot_claim_valid_experiment(self):
        train, evaluation, pilot = self.reports()
        for field, value in (("sample_count", 1304), ("final_town_used", True),
                             ("split", "train"), ("manifest_sha256", "a" * 64),
                             ("status", "FAIL"), ("performance_target_passed", None)):
            changed = dict(evaluation, **{field: value})
            with patch.object(bootstrap, "PILOT_MANIFEST_SHA256", "b" * 64), self.subTest(field=field), \
                    self.assertRaises(bootstrap.TrainingBootstrapError):
                bootstrap.validate_experiment_reports(train, changed, pilot)

    def test_synthetic_cpu_or_different_model_provenance_cannot_claim_gpu_result(self):
        import copy
        train, evaluation, pilot = self.reports()
        modifications = [
            ("runtime", "device", "cpu"), ("runtime", "gpu", "RTX 3060"),
            ("runtime", "encoder_amp_fp16", False), ("pretrained_provenance", "hash_verified", False),
            ("pretrained_provenance", "sha256", "e" * 64),
            ("overfit_check", "head_reset_to_fresh_initialization", False),
        ]
        for section, field, value in modifications:
            changed = copy.deepcopy(train)
            changed[section][field] = value
            with patch.object(bootstrap, "PILOT_MANIFEST_SHA256", "b" * 64), self.subTest(field=field), \
                    self.assertRaises(bootstrap.TrainingBootstrapError):
                bootstrap.validate_experiment_reports(changed, evaluation, pilot)
        for split in ("train", "dev"):
            changed = copy.deepcopy(train)
            changed["feature_cache"][split]["encoder_frozen"] = False
            with patch.object(bootstrap, "PILOT_MANIFEST_SHA256", "b" * 64), self.subTest(split=split), \
                    self.assertRaisesRegex(bootstrap.TrainingBootstrapError, "feature-cache"):
                bootstrap.validate_experiment_reports(changed, evaluation, pilot)

    def test_weight_download_is_exact_and_errors_are_sanitized(self):
        payload = b"approved-weight-fixture"
        class Response:
            status = 200
            def __enter__(self):
                self.stream = io.BytesIO(payload)
                return self
            def __exit__(self, *args):
                return False
            def read(self, count):
                return self.stream.read(count)
        with patch.object(bootstrap, "PRETRAINED_BYTES", len(payload)), \
                patch.object(bootstrap, "PRETRAINED_SHA256", hashlib.sha256(payload).hexdigest()):
            source = bootstrap.pretrained_download_helper_source(self.root / "weights.bin")
        output = io.StringIO()
        with patch("urllib.request.urlopen", return_value=Response()), redirect_stdout(output):
            exec(compile(source, "weights", "exec"), {})
        self.assertEqual((self.root / "weights.bin").read_bytes(), payload)
        self.assertEqual(json.loads(output.getvalue())["status"], "PASS")
        self.assertFalse((self.root / "weights.bin.partial").exists())
        failed = bootstrap.pretrained_download_helper_source(self.root / "failed.bin")
        output = io.StringIO()
        with patch("urllib.request.urlopen", side_effect=RuntimeError("SECRET-SIGNED-URL")), redirect_stdout(output):
            with self.assertRaises(SystemExit):
                exec(compile(failed, "weights", "exec"), {})
        self.assertNotIn("SECRET-SIGNED-URL", output.getvalue())
        self.assertEqual(json.loads(output.getvalue()), {"status": "FAIL", "exception_type": "RuntimeError"})

    def test_weight_hash_mismatch_never_promotes_file(self):
        class Response:
            status = 200
            def __enter__(self):
                self.stream = io.BytesIO(b"changed")
                return self
            def __exit__(self, *args):
                return False
            def read(self, count):
                return self.stream.read(count)
        source = bootstrap.pretrained_download_helper_source(self.root / "weights.bin")
        with patch("urllib.request.urlopen", return_value=Response()), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                exec(compile(source, "weights", "exec"), {})
        self.assertFalse((self.root / "weights.bin").exists())
        self.assertFalse((self.root / "weights.bin.partial").exists())

    def test_existing_mount_accepts_only_exact_archive_hash(self):
        inputs = self.root / "input"
        inputs.mkdir()
        archive = inputs / "pilot_artifacts.zip"
        archive.write_bytes(b"verified-existing-v17-output")
        sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        with patch.object(bootstrap, "vit_cloud_download_helper_source", side_effect=AssertionError("no-cloud")):
            path, transport = bootstrap.resolve_frozen_pilot(inputs, sha, download_version=17,
                                                            cloud_download=lambda version: self.fail("mount should be used"))
        self.assertEqual(path, archive)
        self.assertEqual(transport, "mounted-output")
        archive.write_bytes(b"older-or-modified-output")
        with self.assertRaisesRegex(bootstrap.TrainingBootstrapError, "not-mounted"):
            bootstrap.resolve_frozen_pilot(inputs, sha)

    def test_cloud_backend_diagnostic_is_allowlisted_and_never_leaks_urls(self):
        class BackendError(RuntimeError):
            error_code = 403
            response = types.SimpleNamespace(status_code=403)
        def denied(*args, **kwargs):
            print("SECRET-TOKEN-SIGNED-URL")
            raise BackendError("Permission denied SECRET-TOKEN-SIGNED-URL")
        source = bootstrap.vit_cloud_download_helper_source(17, self.root, self.root / "download")
        self.assertIn(packager.PILOT_NOTEBOOK_ID + "/versions/17", source)
        fake = types.SimpleNamespace(notebook_output_download=denied, __version__="test")
        output = io.StringIO()
        with patch.dict(sys.modules, {"kagglehub": fake}), patch.object(sys, "path", list(sys.path)), \
                patch("logging.disable"), redirect_stdout(output):
            with self.assertRaises(SystemExit):
                exec(compile(source, "cloud_helper", "exec"), {})
        self.assertNotIn("SECRET-TOKEN-SIGNED-URL", output.getvalue())
        self.assertEqual(json.loads(output.getvalue()), {
            "status": "FAIL", "exception_type": "BackendError", "http_status": 403,
            "backend_error_code": 403, "reason": "access-denied",
        })

    def test_artifact_exports_predictions_and_checkpoint_but_excludes_caches(self):
        experiment = self.root / "experiment"
        (experiment / "evaluation").mkdir(parents=True)
        (experiment / "feature_cache").mkdir()
        (experiment / "checkpoint.pt").write_bytes(b"trained-head-and-encoder")
        (experiment / "training_report.json").write_text('{"status":"PASS"}')
        (experiment / "evaluation/predictions.jsonl").write_text('{"sample_id":"one"}\n')
        (experiment / "feature_cache/train.pt").write_bytes(b"do-not-export-features")
        report = bootstrap.preserve_vit_artifacts(self.root)
        with zipfile.ZipFile(self.root / "training_artifacts.zip") as archive:
            self.assertEqual(set(archive.namelist()), {
                "experiment/checkpoint.pt", "experiment/training_report.json",
                "experiment/evaluation/predictions.jsonl",
            })
            for name in archive.namelist():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), report["source_sha256"][name])
        self.assertFalse(report["feature_caches_exported"])
        with self.assertRaisesRegex(bootstrap.TrainingBootstrapError, "already-exists"):
            bootstrap.preserve_vit_artifacts(self.root)


if __name__ == "__main__":
    unittest.main()
