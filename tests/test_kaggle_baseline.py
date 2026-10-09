"""Offline packaging, pinned-runtime, and failure-output checks for the baseline."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import prepare_kaggle_baseline as baseline_cli
from scripts import prepare_kaggle_week1 as packager
from test_kaggle_driving import bootstrap_tree, native_branch
from test_kaggle_week1 import literal_assignments


def runtime_helpers():
    source, _main, failure = bootstrap_tree()
    namespace = {"Path": Path, "hashlib": hashlib, "json": json, "zipfile": zipfile,
                 "BootstrapFailure": failure}
    helpers = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
               and node.name in ("export_baseline_runtime_sources", "preserve_baseline_failure_artifacts")]
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "baseline-bootstrap-helpers.py", "exec"), namespace)
    return namespace


class KaggleBaselineBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.values = {"id": packager.NOTEBOOK_ID, "title": packager.NOTEBOOK_TITLE}
        self.metadata = self.root / "existing.json"
        self.metadata.write_text(json.dumps(self.values), encoding="utf-8")
        self.output = self.root / "prepared"

    def prepare(self):
        return packager.prepare(self.metadata, self.output, workload="baseline")

    def fake_project(self):
        project = self.root / "public-project"
        for relative in packager.SOURCES + packager.DRIVING_SOURCES + packager.BASELINE_SOURCES:
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative.endswith(".py"):
                contents = "PUBLIC_VALUE = 1\n"
            elif relative.endswith(".json"):
                contents = "{}\n"
            else:
                contents = "numpy==2.2.6\nopencv-python-headless==4.12.0.88\n"
            path.write_text(contents, encoding="utf-8")
        (project / "scripts/kaggle_carla_bootstrap.py").write_text(
            "WORKLOAD = 'rgb'\nPROJECT_SOURCES = {}\nPROBE_SOURCE = ''\nVULKAN_CHILD_SOURCE = ''\n"
            "def main():\n    return 0\n", encoding="utf-8")
        (project / "scripts/kaggle_vulkan_probe.py").write_text(
            "def vulkan_devices():\n    return {'status': 'not_verified'}\n", encoding="utf-8")
        return project

    def test_real_bundle_has_complete_public_sources_and_exact_id_title_private_t4(self):
        manifest = self.prepare()
        metadata = json.loads((self.output / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["id"], packager.NOTEBOOK_ID)
        self.assertEqual(metadata["title"], packager.NOTEBOOK_TITLE)
        self.assertTrue(metadata["is_private"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        code = (self.output / metadata["code_file"]).read_text(encoding="utf-8")
        compile(code, "baseline-bundle.py", "exec")
        constants = literal_assignments(code)
        self.assertEqual(constants["WORKLOAD"], "baseline")
        self.assertEqual(set(constants["PROJECT_SOURCES"]),
                         set(packager.SOURCES + packager.DRIVING_SOURCES + packager.BASELINE_SOURCES))
        self.assertEqual(manifest["workload"], "camera-only-opencv-baseline")
        self.assertFalse(manifest["submitted"])
        self.assertFalse(manifest["creates_new_notebook"])
        self.assertEqual(manifest["code_sha256"], hashlib.sha256(code.encode("utf-8")).hexdigest())
        for relative, source in constants["PROJECT_SOURCES"].items():
            self.assertEqual(manifest["source_sha256"][relative], hashlib.sha256(source.encode("utf-8")).hexdigest())
            self.assertFalse(relative.startswith(("results/", "docs/", ".env", "kaggle.json")))

    def test_arbitrary_private_metadata_environment_and_unlisted_sources_are_excluded(self):
        sentinel = "PRIVATE_TEST_SENTINEL_MUST_NOT_BE_EMBEDDED"
        self.values.update(token=sentinel, environment={"KAGGLE_KEY": sentinel}, report=sentinel)
        self.metadata.write_text(json.dumps(self.values), encoding="utf-8")
        project = self.fake_project()
        for relative in (".env", "kaggle.json", "results/local/private.txt", "docs/local/progress.md", "scripts/private.py"):
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(sentinel, encoding="utf-8")
        with patch.object(packager, "ROOT", project), patch.dict(os.environ, {"KAGGLE_KEY": sentinel}), \
                patch("subprocess.run", side_effect=AssertionError("No process allowed")), \
                patch("subprocess.Popen", side_effect=AssertionError("No server allowed")), \
                patch("urllib.request.urlopen", side_effect=AssertionError("No network allowed")):
            self.prepare()
        for path in self.output.iterdir():
            self.assertNotIn(sentinel, path.read_text(encoding="utf-8"))

    def test_changed_existing_identity_and_bad_dependency_pins_leave_no_outputs(self):
        project = self.fake_project()
        for key, value in (("id", "another-owner/new-notebook"), ("title", "New camera baseline")):
            with self.subTest(key=key), patch.object(packager, "ROOT", project):
                self.metadata.write_text(json.dumps({**self.values, key: value}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "existing-notebook-id-or-title-mismatch"):
                    self.prepare()
                self.assertFalse(self.output.exists())
        self.metadata.write_text(json.dumps(self.values), encoding="utf-8")
        (project / "requirements-baseline.txt").write_text("numpy\nopencv-python-headless\n", encoding="utf-8")
        with patch.object(packager, "ROOT", project), self.assertRaisesRegex(ValueError, "baseline-runtime-dependency-pins-mismatch"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_cli_sanitizes_invalid_metadata_and_selects_baseline_without_submission(self):
        project = self.fake_project()
        with patch.object(packager, "ROOT", project), contextlib.redirect_stdout(io.StringIO()) as output:
            code = baseline_cli.main(["--existing-metadata", str(self.metadata), "--output-dir", str(self.output)])
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(output.getvalue())["submitted"])
        self.assertEqual(json.loads(output.getvalue())["workload"], "camera-only-opencv-baseline")
        self.metadata.write_text(json.dumps({"id": "PRIVATE_SENTINEL", "title": "wrong"}), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = baseline_cli.main(["--existing-metadata", str(self.metadata), "--output-dir", str(self.root / "invalid")])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output.getvalue())["code"], "baseline-package-preparation-failed")
        self.assertNotIn("PRIVATE_SENTINEL", output.getvalue())

    def test_materialized_cli_dependencies_are_closed_without_carla_or_repository_on_path(self):
        self.prepare()
        sources = literal_assignments((self.output / "week1.py").read_text(encoding="utf-8"))["PROJECT_SOURCES"]
        project = self.root / "materialized"
        for relative, contents in sources.items():
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
        empty = self.root / "empty-cwd"
        empty.mkdir()
        launcher = ("import runpy,sys; sys.modules['carla']=None; sys.argv=sys.argv[1:]; "
                    "runpy.run_path(sys.argv[0],run_name='__main__')")
        result = subprocess.run([sys.executable, "-I", "-c", launcher,
                                 str(project / "scripts/run_baseline_pipeline.py"), "--help"],
                                cwd=empty, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output-dir", result.stdout)
        self.assertEqual(list(empty.iterdir()), [])
        self.assertFalse((project / "results").exists())


class KaggleBaselineBootstrapTests(unittest.TestCase):
    def test_runtime_install_is_hash_pinned_and_checked_in_actual_client_interpreter(self):
        branch, failure = native_branch("camera-baseline-runtime-install")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = {"opencv_package": "4.12.0.88", "opencv": "4.12.0", "numpy": "2.2.6", "opencv_cuda_used": False}
            command = Mock(return_value=SimpleNamespace(stdout=json.dumps(expected) + "\n"))
            constants = literal_assignments((packager.ROOT / "scripts/kaggle_carla_bootstrap.py").read_text(encoding="utf-8"))
            namespace = {"WORKLOAD": "baseline", "scratch": root, "stage": Mock(), "command": command,
                         "uv": "uv", "python": "carla-python312", "env": {}, "json": json, "report": {},
                         "BASELINE_DEPENDENCIES": constants["BASELINE_DEPENDENCIES"], "BootstrapFailure": failure}
            exec(branch, namespace)
            requirements = (root / "baseline-runtime-requirements.txt").read_text(encoding="utf-8")
            self.assertIn("numpy==2.2.6 --hash=sha256:", requirements)
            self.assertIn("opencv-python-headless==4.12.0.88 --hash=sha256:", requirements)
            install = command.call_args_list[0].args[0]
            self.assertIn("--require-hashes", install)
            self.assertIn("--only-binary", install)
            self.assertEqual(install[install.index("--python") + 1], "carla-python312")
            smoke = command.call_args_list[1].args[0]
            self.assertEqual(smoke[:2], ["carla-python312", "-c"])
            compile(smoke[2], "real-client-runtime-smoke.py", "exec")
            self.assertEqual(namespace["report"]["baseline_runtime"], expected)
            self.assertIn("m.version('opencv-python-headless')=='4.12.0.88'", smoke[2])

    def test_week1_and_driving_workloads_do_not_install_baseline_dependencies(self):
        branch, failure = native_branch("camera-baseline-runtime-install")
        for workload in ("rgb", "week1", "driving"):
            command = Mock()
            namespace = {"WORKLOAD": workload, "command": command, "BootstrapFailure": failure}
            exec(branch, namespace)
            command.assert_not_called()

    def test_baseline_branch_accepts_integrity_pass_and_refuses_pipeline_or_gate_failure(self):
        branch, failure = native_branch("embedded-project-required")
        for pipeline, gate in (("PASS", "PASS"), ("FAIL", "PASS"), ("PASS", "FAIL")):
            with self.subTest(pipeline=pipeline, gate=gate), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output, scratch = root / "output", root / "scratch"
                output.mkdir()
                scratch.mkdir()
                directory = output / "baseline"
                directory.mkdir()
                evidence = {"status": pipeline, "episode_count": 3, "frame_count": 100,
                            "gate": {"status": gate}, "completion_rate": 0.0}
                (directory / "baseline_report.json").write_text(json.dumps(evidence), encoding="utf-8")
                command = Mock()
                helpers = runtime_helpers()
                namespace = {"WORKLOAD": "baseline", "PROJECT_SOURCES": {"scripts/public.py": "VALUE=1\n"},
                             "scratch": scratch, "output": output, "python": "client312", "env": {},
                             "command": command, "stage": Mock(), "json": json, "report": {},
                             "BootstrapFailure": failure, "export_baseline_runtime_sources": helpers["export_baseline_runtime_sources"]}
                if pipeline == gate == "PASS":
                    exec(branch, namespace)
                    self.assertTrue(namespace["report"]["baseline_verified"])
                    self.assertEqual(namespace["report"]["baseline"]["episode_count"], 3)
                    self.assertTrue((output / "baseline_runtime_sources.zip").is_file())
                else:
                    with self.assertRaisesRegex(failure, "baseline-evidence-not-pass"):
                        exec(branch, namespace)
                    self.assertNotIn("baseline_verified", namespace["report"])
                self.assertEqual(command.call_args.args[:3], (
                    ["client312", str(scratch / "project/scripts/run_baseline_pipeline.py"), "--output-dir", str(directory)],
                    "carla_baseline", 600))

    def test_actual_runtime_source_hashes_match_exported_bytes_and_tampering_is_refused(self):
        helpers = runtime_helpers()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, output = root / "project", root / "output"
            project.mkdir()
            output.mkdir()
            sources = {"a.py": "A=1\n", "nested/b.json": "{}\n"}
            for relative, source in sources.items():
                path = project / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8", newline="\n")
            manifest = helpers["export_baseline_runtime_sources"](project, output, sources)
            persisted = json.loads((output / "baseline_runtime_source_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest, persisted)
            with zipfile.ZipFile(output / "baseline_runtime_sources.zip") as archive:
                self.assertEqual(set(archive.namelist()), set(sources))
                for relative in sources:
                    self.assertEqual(hashlib.sha256(archive.read(relative)).hexdigest(), manifest["source_sha256"][relative])
            (project / "a.py").write_text("A='changed'\n", encoding="utf-8")
            other = root / "other-output"
            other.mkdir()
            with self.assertRaisesRegex(helpers["BootstrapFailure"], "baseline-runtime-source-hash-mismatch"):
                helpers["export_baseline_runtime_sources"](project, other, sources)

    def test_partial_failure_artifacts_are_archived_and_an_existing_archive_is_preserved(self):
        helpers = runtime_helpers()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            directory = output / "baseline/episodes/partial"
            directory.mkdir(parents=True)
            path = directory / "trace.jsonl"
            path.write_bytes(b'{"stage":"before-failure"}\n')
            preserve = helpers["preserve_baseline_failure_artifacts"]
            preserve(output)
            destination = output / "baseline_artifacts.zip"
            before = destination.read_bytes()
            with zipfile.ZipFile(destination) as archive:
                self.assertEqual(archive.read("baseline/episodes/partial/trace.jsonl"), path.read_bytes())
            path.write_bytes(b"changed after preserved archive")
            preserve(output)
            self.assertEqual(destination.read_bytes(), before)

    def test_actual_finally_stops_owned_server_and_archives_partial_baseline_on_failure(self):
        _source, main, _failure = bootstrap_tree()
        guarded = next(node for node in main.body if isinstance(node, ast.Try))
        cleanup = compile(ast.Module(body=guarded.finalbody, type_ignores=[]), "baseline-owned-server-cleanup.py", "exec")
        helpers = runtime_helpers()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            partial = output / "baseline/partial.json"
            partial.parent.mkdir()
            partial.write_text("{}\n", encoding="utf-8")
            owned = SimpleNamespace(pid=12345, wait=Mock(), poll=lambda: 0)
            kill = Mock()
            report = {"status": "FAIL", "error": "pipeline-timeout"}
            namespace = {"signal": SimpleNamespace(alarm=Mock(), signal=Mock(), SIGTERM=signal.SIGTERM, SIGKILL=signal.SIGTERM, SIGALRM=14),
                         "process": owned, "os": SimpleNamespace(killpg=kill), "subprocess": subprocess,
                         "previous_handler": None, "report": report, "WORKLOAD": "baseline", "output": output,
                         "preserve_baseline_failure_artifacts": helpers["preserve_baseline_failure_artifacts"],
                         "time": SimpleNamespace(monotonic=lambda: 10.0), "started": 2.0, "json": json, "print": Mock()}
            exec(cleanup, namespace)
            kill.assert_called_once_with(12345, signal.SIGTERM)
            owned.wait.assert_called_once_with(timeout=10)
            self.assertTrue(report["server_stopped"])
            self.assertEqual(report["status"], "FAIL")
            self.assertTrue((output / "baseline_artifacts.zip").is_file())
            self.assertEqual(json.loads((output / "kaggle_carla_bootstrap.json").read_text(encoding="utf-8")), report)


if __name__ == "__main__":
    unittest.main()
