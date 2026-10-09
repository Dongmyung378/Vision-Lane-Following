"""CPU-only source packaging checks; these tests never submit a Kaggle job."""

from __future__ import annotations

import ast
import base64
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from scripts import prepare_kaggle_probe as legacy
from scripts import prepare_kaggle_week1 as packager


def literal_assignments(source: str) -> dict:
    """Inspect payload constants without executing Linux/bootstrap code."""
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = value
    if "PROJECT_SOURCES_PACKED" in values:
        payload = zlib.decompress(base64.b64decode(values["PROJECT_SOURCES_PACKED"], validate=True))
        if hashlib.sha256(payload).hexdigest() != values["PROJECT_PAYLOAD_SHA256"]:
            raise ValueError("embedded-project-transport-hash-mismatch")
        values["PROJECT_SOURCES"] = json.loads(payload.decode("utf-8"))
    return values


class KaggleWeek1BundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metadata_path = self.root / "existing-notebook.json"
        self.metadata = {
            "id": "dongmyungpark/vlf-carla-rgb-smoke-20261005",
            "title": "VLF CARLA RGB smoke 20261005",
            "dataset_sources": ["public-owner/public-dataset"],
            "kernel_sources": [], "competition_sources": [], "model_sources": [],
        }
        self.write_metadata()
        self.output = self.root / "prepared"

    def write_metadata(self):
        self.metadata_path.write_text(json.dumps(self.metadata), encoding="utf-8")

    def prepare(self):
        return packager.prepare(self.metadata_path, self.output)

    def fake_project(self):
        project = self.root / "public-project"
        for relative in packager.SOURCES:
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("PUBLIC_VALUE = 'public-source'\n" if relative.endswith(".py") else "{}\n",
                            encoding="utf-8")
        bootstrap = project / "scripts/kaggle_carla_bootstrap.py"
        bootstrap.write_text("WORKLOAD = 'rgb'\nPROJECT_SOURCES = {}\nPROBE_SOURCE = ''\n"
                             "VULKAN_CHILD_SOURCE = ''\ndef main():\n    return 0\n", encoding="utf-8")
        (project / "scripts/kaggle_vulkan_probe.py").write_text(
            "def vulkan_devices():\n    return {'status': 'not_verified'}\n", encoding="utf-8")
        return project

    def test_real_bundle_compiles_and_hashes_match_exact_embedded_source_bytes(self):
        manifest = self.prepare()
        code = (self.output / "week1.py").read_text(encoding="utf-8")
        compile(code, "week1.py", "exec")
        constants = literal_assignments(code)
        self.assertEqual(constants["WORKLOAD"], "week1")
        sources = constants["PROJECT_SOURCES"]
        self.assertEqual(set(sources), set(packager.SOURCES))
        self.assertEqual(constants["PROBE_SOURCE"], sources["scripts/carla_rgb_probe.py"])
        compile(constants["VULKAN_CHILD_SOURCE"], "vulkan-child.py", "exec")
        self.assertEqual(manifest["code_sha256"], hashlib.sha256(code.encode("utf-8")).hexdigest())
        for relative, source in sources.items():
            self.assertEqual(manifest["source_sha256"][relative], hashlib.sha256(source.encode("utf-8")).hexdigest())
            if relative.endswith(".py"):
                compile(source, relative, "exec")
            else:
                json.loads(source)
        persisted = json.loads((self.output / "bundle_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted, manifest)
        self.assertFalse(manifest["submitted"])
        self.assertFalse(manifest["creates_new_notebook"])

    def test_bundle_preserves_exact_existing_id_and_title_and_forces_private_gpu_submission(self):
        self.metadata.update(is_private=False, enable_gpu=False, enable_internet=False,
                             code_file="stale.py", title=packager.NOTEBOOK_TITLE)
        self.write_metadata()
        self.prepare()
        metadata = json.loads((self.output / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["id"], "dongmyungpark/vlf-carla-rgb-smoke-20261005")
        self.assertEqual(metadata["title"], "VLF CARLA RGB smoke 20261005")
        self.assertEqual(metadata["code_file"], "week1.py")
        self.assertIs(metadata["is_private"], True)
        self.assertIs(metadata["enable_gpu"], True)
        self.assertIs(metadata["enable_internet"], True)
        self.assertIs(metadata["enable_tpu"], False)
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        for key in ("dataset_sources", "kernel_sources", "competition_sources", "model_sources"):
            self.assertEqual(metadata[key], self.metadata[key])

    def test_changed_owner_slug_date_or_title_is_refused_without_outputs(self):
        for field, value in (("id", "different-owner/vlf-carla-rgb-smoke-20261005"),
                             ("id", "dongmyungpark/vlf-carla-rgb-smoke-20261008"),
                             ("id", "dongmyungpark/vlf-carla-rgb-smoke-random"),
                             ("title", "New D6 experiment")):
            with self.subTest(field=field, value=value):
                original = self.metadata[field]
                self.metadata[field] = value
                self.write_metadata()
                with self.assertRaisesRegex(ValueError, "existing-notebook-id-or-title-mismatch"):
                    self.prepare()
                self.assertFalse(self.output.exists())
                self.metadata[field] = original

    def test_arbitrary_metadata_credentials_and_environment_do_not_enter_any_artifact(self):
        sentinel = "TEST_PRIVATE_SENTINEL_MUST_NOT_BE_PACKAGED"
        self.metadata.update(token=sentinel, password=sentinel, credentials={"secret": sentinel},
                             local_identity=sentinel, environment={"KAGGLE_KEY": sentinel})
        self.write_metadata()
        with patch.dict(os.environ, {"KAGGLE_KEY": sentinel, "KAGGLE_USERNAME": sentinel}):
            self.prepare()
        for path in self.output.iterdir():
            self.assertNotIn(sentinel, path.read_text(encoding="utf-8"))
        metadata = json.loads((self.output / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(set(metadata), {"id", "title", "code_file", "language", "kernel_type", "is_private",
                                         "enable_gpu", "enable_tpu", "enable_internet", "machine_shape",
                                         "dataset_sources", "kernel_sources", "competition_sources", "model_sources"})

    def test_explicit_public_allowlist_excludes_unlisted_credentials_logs_and_local_files(self):
        project = self.fake_project()
        sentinel = "TEST_UNLISTED_PRIVATE_CONTENT"
        for relative in ("kaggle.json", ".env", "results/local/private.log", "scripts/extra_private.py"):
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(sentinel, encoding="utf-8")
        with patch.object(packager, "ROOT", project):
            self.prepare()
        code = (self.output / "week1.py").read_text(encoding="utf-8")
        self.assertNotIn(sentinel, code)
        embedded = literal_assignments(code)["PROJECT_SOURCES"]
        for relative in embedded:
            path = PurePosixPath(relative)
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)
            self.assertIn(path.parts[0], ("carla_tools", "scripts", "configs"))
            self.assertIn(path.suffix, (".py", ".json"))

    def test_malformed_source_lists_are_refused_before_output_creation(self):
        project = self.fake_project()
        invalid_values = ({"secret": "TEST_PRIVATE_SENTINEL"}, "owner/slug", None, False,
                          [42], [{"secret": "TEST_PRIVATE_SENTINEL"}], [""], ["   "],
                          [" owner/slug"], ["owner/slug\nprivate"])
        for key in ("dataset_sources", "kernel_sources", "competition_sources", "model_sources"):
            original = self.metadata[key]
            for value in invalid_values:
                with self.subTest(key=key, value=value), patch.object(packager, "ROOT", project):
                    self.metadata[key] = value
                    self.write_metadata()
                    with self.assertRaisesRegex(ValueError, "existing-notebook-source-list-invalid"):
                        self.prepare()
                    self.assertFalse(self.output.exists())
            self.metadata[key] = original

    def test_missing_optional_source_lists_default_to_empty_lists(self):
        for key in ("dataset_sources", "kernel_sources", "competition_sources", "model_sources"):
            self.metadata.pop(key)
        self.write_metadata()
        self.prepare()
        metadata = json.loads((self.output / "kernel-metadata.json").read_text(encoding="utf-8"))
        for key in ("dataset_sources", "kernel_sources", "competition_sources", "model_sources"):
            self.assertEqual(metadata[key], [])

    def test_preparing_package_never_runs_processes_or_contacts_network(self):
        with patch("subprocess.run", side_effect=AssertionError("No process allowed")), \
                patch("subprocess.Popen", side_effect=AssertionError("No server allowed")), \
                patch("urllib.request.urlopen", side_effect=AssertionError("No network allowed")):
            self.prepare()
        self.assertTrue((self.output / "week1.py").is_file())

    def test_invalid_public_python_is_rejected_before_output_creation(self):
        project = self.fake_project()
        (project / "carla_tools/config.py").write_text("def invalid(:\n", encoding="utf-8")
        with patch.object(packager, "ROOT", project), self.assertRaises(SyntaxError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_invalid_public_json_is_rejected_before_output_creation(self):
        project = self.fake_project()
        (project / "configs/collection_d4.json").write_text("{broken-json", encoding="utf-8")
        with patch.object(packager, "ROOT", project), self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_existing_output_directory_is_preserved_even_if_empty(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "output-directory-already-exists"):
            self.prepare()
        self.assertEqual(list(self.output.iterdir()), [])
        sentinel = self.output / "existing.txt"
        sentinel.write_text("preserved", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "output-directory-already-exists"):
            self.prepare()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserved")
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_materialized_bundle_has_closed_imports_without_carla_or_installed_project(self):
        self.prepare()
        constants = literal_assignments((self.output / "week1.py").read_text(encoding="utf-8"))
        project = self.root / "isolated-project"
        for relative, source in constants["PROJECT_SOURCES"].items():
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        empty_directory = self.root / "empty-cwd"
        empty_directory.mkdir()
        # -I excludes the active checkout and PYTHONPATH; --help imports the
        # complete workload graph but must not connect to CARLA or create data.
        entrypoint = project / "scripts/run_week1_pipeline.py"
        isolated_launcher = ("import runpy,sys; sys.modules['carla']=None; "
                             "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')")
        result = subprocess.run([sys.executable, "-I", "-c", isolated_launcher, str(entrypoint), "--help"],
                                cwd=empty_directory, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output-dir", result.stdout)
        self.assertEqual(list(empty_directory.iterdir()), [])
        self.assertFalse((project / "results").exists())

    def test_legacy_rgb_packager_keeps_original_notebook_and_bootstrap_defaults(self):
        output = self.root / "legacy-rgb"
        with patch.object(sys, "argv", ["prepare_kaggle_probe.py", "--owner", "dongmyungpark",
                                        "--kind", "carla", "--output-dir", str(output)]), \
                contextlib.redirect_stdout(io.StringIO()):
            legacy.main()
        metadata = json.loads((output / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["id"], packager.NOTEBOOK_ID)
        self.assertEqual(metadata["title"], packager.NOTEBOOK_TITLE)
        code = (output / "probe.py").read_text(encoding="utf-8")
        compile(code, "legacy-probe.py", "exec")
        constants = literal_assignments(code)
        self.assertEqual(constants["WORKLOAD"], "rgb")
        self.assertEqual(constants["PROJECT_SOURCES"], {})
        self.assertTrue(constants["PROBE_SOURCE"])
        self.assertTrue(constants["VULKAN_CHILD_SOURCE"])

    def test_cli_mismatch_failure_is_sanitized_and_nonzero(self):
        self.metadata["id"] = "private-identity/TEST_PRIVATE_SENTINEL"
        self.write_metadata()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = packager.main(["--existing-metadata", str(self.metadata_path), "--output-dir", str(self.output)])
        self.assertEqual(code, 1)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "FAIL")
        self.assertNotIn("TEST_PRIVATE_SENTINEL", output.getvalue())
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
