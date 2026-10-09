"""Same-session handoff starts training only after immutable native cleanup."""

from __future__ import annotations

import ast
import base64
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path

from scripts.prepare_kaggle_pilot_training import prepare, run_training_after_pilot
from scripts.prepare_kaggle_week1 import NOTEBOOK_ID, NOTEBOOK_TITLE
from scripts.prepare_kaggle_training import SOURCES as TRAINING_SOURCES
from scripts.kaggle_training_bootstrap import resolve_local_pilot_archive, TrainingBootstrapError


class PilotTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "pilot_artifacts.zip"
        (self.root / "pilot").mkdir()
        self.manifest = self.root / "pilot/pilot_manifest.json"
        self.manifest.write_text('{"status":"COMPLETE","scope":"moving-rgb-waypoint-pilot"}', encoding="utf-8")
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("pilot/pilot_manifest.json", self.manifest.read_bytes())
        self.archive_sha = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.manifest_sha = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.write_report("kaggle_carla_bootstrap.json", {
            "status": "PASS", "server_stopped": True, "pilot_verified": True,
            "pilot": {"gate": {"status": "PASS", "manifest_sha256": self.manifest_sha}}})
        self.write_report("pilot_archive_report.json", {
            "status": "PASS", "archive_sha256": self.archive_sha,
            "archive_bytes": self.archive.stat().st_size, "manifest_sha256": self.manifest_sha})
        self.sources = {"example.py": "value = 1\n"}
        self.bootstrap = (
            "import json\nfrom pathlib import Path\nPROJECT_SOURCES = {}\n"
            "PILOT_ARCHIVE_SHA256 = ''\nPILOT_MANIFEST_SHA256 = ''\n"
            "PILOT_DOWNLOAD_VERSION = 123\nMAX_SECONDS = 1\n"
            "def main(local_archive=None):\n"
            " assert local_archive.name == 'pilot_artifacts.zip'\n"
            " assert PROJECT_SOURCES == {'example.py':'value = 1\\n'}\n"
            " assert PILOT_DOWNLOAD_VERSION is None and MAX_SECONDS == 5400\n"
            " output=local_archive.parent\n"
            " (output/'training_called').write_text('called')\n"
            " (output/'kaggle_training_bootstrap.json').write_text(json.dumps({"
            "'status':'PASS','training_process_stopped':True,"
            "'pilot_archive_sha256':PILOT_ARCHIVE_SHA256,'pilot_manifest_sha256':PILOT_MANIFEST_SHA256}))\n"
            " return 0\n")

    def write_report(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def run_handoff(self, code=0, bootstrap=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_training_after_pilot(code, self.root, bootstrap or self.bootstrap, self.sources)

    def test_pass_handoff_uses_dynamic_hashes_and_isolated_namespace(self):
        self.assertEqual(self.run_handoff(), 0)
        report = json.loads((self.root / "combined_pipeline_report.json").read_text())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["pilot_archive_sha256"], self.archive_sha)
        self.assertEqual(report["pilot_manifest_sha256"], self.manifest_sha)
        self.assertEqual(report["training_source_sha256"]["example.py"],
                         hashlib.sha256(self.sources["example.py"].encode()).hexdigest())
        self.assertTrue((self.root / "training_called").exists())

    def test_native_failure_cannot_start_training(self):
        self.assertEqual(self.run_handoff(code=1), 1)
        self.assertFalse((self.root / "training_called").exists())

    def test_native_status_cleanup_and_gate_failures_cannot_start_training(self):
        path = self.root / "kaggle_carla_bootstrap.json"
        original = json.loads(path.read_text())
        for field in ("status", "server_stopped", "pilot_verified", "gate"):
            changed = json.loads(json.dumps(original))
            if field == "status":
                changed[field] = "FAIL"
            elif field == "gate":
                changed["pilot"]["gate"]["status"] = "FAIL"
            else:
                changed[field] = False
            self.write_report(path.name, changed)
            with self.subTest(field=field):
                self.assertEqual(self.run_handoff(), 1)
                self.assertFalse((self.root / "training_called").exists())

    def test_failed_archive_report_refuses_training(self):
        report = json.loads((self.root / "pilot_archive_report.json").read_text())
        report["status"] = "FAIL"
        self.write_report("pilot_archive_report.json", report)
        self.assertEqual(self.run_handoff(), 1)
        self.assertFalse((self.root / "training_called").exists())

    def test_changed_archive_or_manifest_refuses_training(self):
        data = self.archive.read_bytes()
        self.archive.write_bytes(data + b"changed")
        self.assertEqual(self.run_handoff(), 1)
        self.archive.write_bytes(data)
        self.manifest.write_text('{"status":"CHANGED"}', encoding="utf-8")
        self.assertEqual(self.run_handoff(), 1)
        self.assertFalse((self.root / "training_called").exists())

    def test_archived_manifest_is_checked_against_native_file(self):
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("pilot/pilot_manifest.json", "different")
        artifact = json.loads((self.root / "pilot_archive_report.json").read_text())
        artifact.update(archive_sha256=hashlib.sha256(self.archive.read_bytes()).hexdigest(),
                        archive_bytes=self.archive.stat().st_size)
        self.write_report("pilot_archive_report.json", artifact)
        self.assertEqual(self.run_handoff(), 1)
        self.assertFalse((self.root / "training_called").exists())

    def test_training_failure_is_reported_without_sensitive_exception_message(self):
        source = "def main(local_archive=None):\n raise RuntimeError('SECRET-SIGNED-URL')\n"
        self.assertEqual(self.run_handoff(bootstrap=source), 1)
        report_text = (self.root / "combined_pipeline_report.json").read_text()
        self.assertNotIn("SECRET-SIGNED-URL", report_text)
        self.assertEqual(json.loads(report_text)["code"], "combined-pipeline-failed")

    def test_local_archive_requires_unchanged_bytes_and_exact_owned_location(self):
        self.assertEqual(resolve_local_pilot_archive(self.archive, self.archive_sha, self.root), self.archive)
        with self.assertRaisesRegex(TrainingBootstrapError, "sha256-mismatch"):
            resolve_local_pilot_archive(self.archive, "0" * 64, self.root)
        with self.assertRaisesRegex(TrainingBootstrapError, "location-invalid"):
            resolve_local_pilot_archive(self.archive, self.archive_sha, self.root / "other")

    def test_combined_package_keeps_existing_identity_and_two_source_allowlists(self):
        metadata_path = self.root / "metadata.json"
        metadata_path.write_text(json.dumps({"id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE,
                                             "api_key": "DO-NOT-COPY-SECRET"}), encoding="utf-8")
        output = self.root / "package"
        report = prepare(metadata_path, output)
        metadata = json.loads((output / "kernel-metadata.json").read_text())
        code = (output / "week1.py").read_text(encoding="utf-8")
        self.assertEqual(metadata["id"], NOTEBOOK_ID)
        self.assertEqual(metadata["title"], NOTEBOOK_TITLE)
        self.assertTrue(metadata["is_private"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(report["workload"], "moving-rgb-waypoint-pilot-and-training")
        self.assertEqual(report["pilot_hash_origin"], "verified-same-session-pilot-report")
        self.assertEqual(len(report["source_sha256"]), 42)
        self.assertEqual(set(report["training_source_sha256"]), set(TRAINING_SOURCES))
        self.assertFalse(report["creates_new_notebook"])
        self.assertFalse(report["submitted"])
        self.assertNotIn("DO-NOT-COPY-SECRET", code)
        self.assertLess(len(code.encode()), 1_000_000)
        compile(code, "week1.py", "exec")
        literals = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(code).body
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in ("TRAINING_SOURCES_PACKED", "TRAINING_PAYLOAD_SHA256")}
        payload = zlib.decompress(base64.b64decode(literals["TRAINING_SOURCES_PACKED"], validate=True))
        self.assertEqual(hashlib.sha256(payload).hexdigest(), literals["TRAINING_PAYLOAD_SHA256"])
        sources = json.loads(payload)
        self.assertEqual({name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()},
                         report["training_source_sha256"])
        with self.assertRaisesRegex(ValueError, "already-exists"):
            prepare(metadata_path, output)


if __name__ == "__main__":
    unittest.main()
