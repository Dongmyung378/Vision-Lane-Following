"""Existing-notebook driving bundles and CPU-only production bootstrap branches."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import prepare_kaggle_driving as driving_cli
from scripts import prepare_kaggle_week1 as packager
from test_kaggle_week1 import literal_assignments


DRIVING_SOURCES = {
    "carla_tools/controller.py", "carla_tools/driving_route.py", "carla_tools/driving.py",
    "carla_tools/driving_validation.py", "scripts/run_driving_pipeline.py", "scripts/verify_driving.py",
    "configs/controller_d8.json", "configs/driving_validation_d10.json", "configs/routes_d6_geometry.json",
}


def bootstrap_tree():
    source = (packager.ROOT / "scripts/kaggle_carla_bootstrap.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    failure = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BootstrapFailure")
    namespace = {}
    exec(compile(ast.Module(body=[failure], type_ignores=[]), "bootstrap-error.py", "exec"), namespace)
    return source, main, namespace["BootstrapFailure"]


def native_branch(fragment):
    """Execute the actual branch AST without importing pwd or starting Linux."""
    source, main, failure = bootstrap_tree()
    node = next(node for node in ast.walk(main) if isinstance(node, ast.If)
                and "WORKLOAD" in ast.unparse(node.test)
                and fragment in (ast.get_source_segment(source, node) or ""))
    return compile(ast.Module(body=[node], type_ignores=[]), "bootstrap-native-branch.py", "exec"), failure


class KaggleDrivingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metadata = self.root / "existing.json"
        self.values = {"id": packager.NOTEBOOK_ID, "title": packager.NOTEBOOK_TITLE}
        self.metadata.write_text(json.dumps(self.values), encoding="utf-8")
        self.output = self.root / "bundle"

    def prepare(self):
        return packager.prepare(self.metadata, self.output, workload="driving")

    def test_same_id_title_private_t4_exact_public_sources_hashes_and_compile(self):
        manifest = self.prepare()
        submission = json.loads((self.output / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(submission["id"], "dongmyungpark/vlf-carla-rgb-smoke-20261005")
        self.assertEqual(submission["title"], "VLF CARLA RGB smoke 20261005")
        self.assertIs(submission["is_private"], True)
        self.assertIs(submission["enable_gpu"], True)
        self.assertIs(submission["enable_tpu"], False)
        self.assertIs(submission["enable_internet"], True)
        self.assertEqual(submission["machine_shape"], "NvidiaTeslaT4")
        code = (self.output / submission["code_file"]).read_text(encoding="utf-8")
        compile(code, "prepared-driving.py", "exec")
        embedded = literal_assignments(code)
        self.assertEqual(embedded["WORKLOAD"], "driving")
        self.assertEqual(set(embedded["PROJECT_SOURCES"]), set(packager.SOURCES) | DRIVING_SOURCES)
        self.assertEqual(manifest["workload"], "D8-D10-privileged-driving")
        self.assertFalse(manifest["submitted"])
        self.assertFalse(manifest["creates_new_notebook"])
        self.assertEqual(hashlib.sha256(code.encode("utf-8")).hexdigest(), manifest["code_sha256"])
        for relative, source in embedded["PROJECT_SOURCES"].items():
            path = PurePosixPath(relative)
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)
            self.assertIn(path.parts[0], {"carla_tools", "scripts", "configs"})
            self.assertEqual(hashlib.sha256(source.encode("utf-8")).hexdigest(), manifest["source_sha256"][relative])
            if relative.endswith(".py"):
                compile(source, relative, "exec")
            else:
                json.loads(source)

    def test_authentication_and_local_environment_are_excluded(self):
        sentinel = "TEST_CREDENTIAL_SENTINEL_MUST_NOT_LEAVE_METADATA"
        self.values.update(token=sentinel, kaggle_key=sentinel, environment={"KAGGLE_KEY": sentinel})
        self.metadata.write_text(json.dumps(self.values), encoding="utf-8")
        with patch.dict(os.environ, {"KAGGLE_KEY": sentinel, "KAGGLE_USERNAME": sentinel}), \
                patch("subprocess.run", side_effect=AssertionError("Preparation must not run processes")), \
                patch("urllib.request.urlopen", side_effect=AssertionError("Preparation must not contact network")):
            self.prepare()
        for path in self.output.iterdir():
            self.assertNotIn(sentinel, path.read_text(encoding="utf-8"))
        sources = literal_assignments((self.output / "week1.py").read_text(encoding="utf-8"))["PROJECT_SOURCES"]
        self.assertFalse(any("results/" in path or "kaggle.json" in path or path.startswith(".env") for path in sources))

    def test_changed_notebook_or_malformed_metadata_refuses_before_output(self):
        variants = ["{malformed-json", json.dumps({"id": packager.NOTEBOOK_ID}),
                    json.dumps({**self.values, "id": "other-owner/new-notebook"}),
                    json.dumps({**self.values, "title": "D10 new experiment"}),
                    json.dumps({**self.values, "dataset_sources": {"secret": "private-token"}})]
        for contents in variants:
            with self.subTest(contents=contents):
                self.metadata.write_text(contents, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def assert_non_object_cli_metadata_is_refused(self, cli, failure_code):
        sentinel = "TEST_PRIVATE_ROOT_METADATA_MUST_NOT_BE_PRINTED"
        for value in (None, [sentinel], sentinel):
            with self.subTest(root_type=type(value).__name__):
                self.metadata.write_text(json.dumps(value), encoding="utf-8")
                output, errors = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors), \
                        patch("subprocess.run", side_effect=AssertionError("No process may run")), \
                        patch("subprocess.Popen", side_effect=AssertionError("No server may start")), \
                        patch("urllib.request.urlopen", side_effect=AssertionError("No network may be contacted")):
                    result = cli(["--existing-metadata", str(self.metadata), "--output-dir", str(self.output)])
                self.assertEqual(result, 1)
                self.assertEqual(json.loads(output.getvalue()), {
                    "status": "FAIL", "code": failure_code, "exception_type": "ValueError"})
                self.assertEqual(errors.getvalue(), "")
                self.assertNotIn(sentinel, output.getvalue())
                self.assertFalse(self.output.exists())

    def test_week1_cli_rejects_null_list_string_metadata_with_sanitized_failure(self):
        self.assert_non_object_cli_metadata_is_refused(packager.main, "week1-package-preparation-failed")

    def test_driving_cli_rejects_null_list_string_metadata_with_sanitized_failure(self):
        self.assert_non_object_cli_metadata_is_refused(driving_cli.main, "driving-package-preparation-failed")

    def test_existing_bundle_is_preserved_and_cli_error_is_sanitized(self):
        self.output.mkdir()
        sentinel = self.output / "preserved.txt"
        sentinel.write_bytes(b"existing-public-artifact")
        with self.assertRaisesRegex(ValueError, "output-directory-already-exists"):
            self.prepare()
        self.assertEqual(sentinel.read_bytes(), b"existing-public-artifact")
        self.values["id"] = "private-owner/private-token"
        self.metadata.write_text(json.dumps(self.values), encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = driving_cli.main(["--existing-metadata", str(self.metadata), "--output-dir", str(self.output)])
        self.assertEqual(result, 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "FAIL")
        self.assertNotIn("private-token", output.getvalue())
        self.assertEqual(sentinel.read_bytes(), b"existing-public-artifact")

    def test_driving_cli_selects_driving_workload_without_submission(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = driving_cli.main(["--existing-metadata", str(self.metadata), "--output-dir", str(self.output)])
        self.assertEqual(result, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["workload"], "D8-D10-privileged-driving")
        self.assertFalse(report["creates_new_notebook"])
        manifest = json.loads((self.output / "bundle_manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["submitted"])

    def test_materialized_bundle_has_closed_import_and_runtime_config_dependencies(self):
        self.prepare()
        sources = literal_assignments((self.output / "week1.py").read_text(encoding="utf-8"))["PROJECT_SOURCES"]
        project = self.root / "materialized"
        for relative, source in sources.items():
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        cwd = self.root / "isolated-cwd"
        cwd.mkdir()
        launcher = ("import runpy,sys; sys.modules['carla']=None; sys.argv=sys.argv[1:]; "
                    "runpy.run_path(sys.argv[0],run_name='__main__')")
        for entrypoint in ("run_driving_pipeline.py", "verify_driving.py"):
            result = subprocess.run([sys.executable, "-I", "-c", launcher,
                                     str(project / "scripts" / entrypoint), "--help"],
                                    cwd=cwd, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--", result.stdout)
        code = ("import json,sys; sys.modules['carla']=None; sys.path.insert(0,sys.argv[1]); "
                "from carla_tools.controller import load_controller_config; "
                "from carla_tools.driving import load_validation_config; "
                "from carla_tools.geometry_survey import load_geometry_catalog; "
                "limits=load_validation_config(); catalog=load_geometry_catalog(sys.argv[1]+'/configs/routes_d6_geometry.json'); "
                "print(json.dumps({'speed':load_controller_config().target_speed_kmh, "
                "'cases':sorted(x['case'] for x in catalog['geometry_cases']), 'conditions':len(limits['initial_conditions'])}))")
        result = subprocess.run([sys.executable, "-I", "-c", code, str(project)], cwd=cwd,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"speed": 25.0,
                         "cases": ["left_curve", "right_curve", "straight"], "conditions": 5})
        self.assertEqual(list(cwd.iterdir()), [])
        self.assertFalse((project / "results").exists())


class KaggleDrivingBootstrapTests(unittest.TestCase):
    def test_driving_branch_materializes_public_sources_runs_gate_and_requires_nested_pass(self):
        branch, failure = native_branch("embedded-project-required")
        for pipeline_status, gate_status in (("PASS", "PASS"), ("FAIL", "PASS"), ("PASS", "FAIL")):
            with self.subTest(pipeline=pipeline_status, gate=gate_status), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output, scratch = root / "output", root / "scratch"
                output.mkdir()
                scratch.mkdir()
                artifact = output / "driving"
                artifact.mkdir()
                evidence = {"status": pipeline_status, "episode_count": 15, "frame_count": 1900,
                            "gate": {"status": gate_status}}
                (artifact / "driving_report.json").write_text(json.dumps(evidence), encoding="utf-8")
                command = Mock()
                report = {}
                namespace = {"WORKLOAD": "driving", "PROJECT_SOURCES": {"scripts/public.py": "VALUE=1\n"},
                             "scratch": scratch, "output": output, "python": "isolated-python", "env": {"PUBLIC": "1"},
                             "command": command, "stage": Mock(), "json": json, "report": report,
                             "BootstrapFailure": failure}
                if pipeline_status == gate_status == "PASS":
                    exec(branch, namespace)
                    self.assertTrue(report["driving_verified"])
                    self.assertEqual(report["driving"], evidence)
                else:
                    with self.assertRaisesRegex(failure, "driving-evidence-not-pass"):
                        exec(branch, namespace)
                    self.assertNotIn("driving_verified", report)
                arguments = command.call_args.args
                self.assertEqual(arguments[0], ["isolated-python", str(scratch / "project/scripts/run_driving_pipeline.py"),
                                               "--output-dir", str(artifact)])
                self.assertEqual(arguments[1:3], ("carla_driving", 600))
                self.assertEqual((scratch / "project/scripts/public.py").read_text(encoding="utf-8"), "VALUE=1\n")

    def test_missing_sources_or_escaping_paths_never_run_a_workload(self):
        branch, failure = native_branch("embedded-project-required")
        for sources, code in (({}, "embedded-project-required"), ({"../escape.py": "secret"}, "embedded-project-path-invalid")):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                command = Mock()
                namespace = {"WORKLOAD": "driving", "PROJECT_SOURCES": sources, "scratch": root,
                             "command": command, "stage": Mock(), "BootstrapFailure": failure}
                with self.assertRaisesRegex(failure, code):
                    exec(branch, namespace)
                command.assert_not_called()
                self.assertFalse((root / "escape.py").exists())

    def test_only_verified_official_startup_map_is_pinned_and_ambiguity_is_preserved(self):
        branch, failure = native_branch("GameDefaultMap")
        official = "/Game/Carla/Maps/Town10HD_Opt.Town10HD_Opt"
        desired = "/Game/Carla/Maps/Town01.Town01"
        for original, passes in ((official, True), (desired, True), ("/Game/Carla/Maps/Town02.Town02", False),
                                 (official + "\nGameDefaultMap=" + official, False)):
            with self.subTest(original=original), tempfile.TemporaryDirectory() as temporary:
                server = Path(temporary)
                ini = server / "CarlaUE4/Config/DefaultEngine.ini"
                ini.parent.mkdir(parents=True)
                contents = "[EngineSettings]\nGameDefaultMap=" + original + "\nOtherSetting=preserved\n"
                ini.write_text(contents, encoding="utf-8")
                report = {}
                namespace = {"WORKLOAD": "driving", "server": server, "re": re,
                             "report": report, "BootstrapFailure": failure}
                if passes:
                    exec(branch, namespace)
                    self.assertEqual(report["startup_map"], "Town01")
                    self.assertIn("GameDefaultMap=" + desired, ini.read_text(encoding="utf-8"))
                    self.assertIn("OtherSetting=preserved", ini.read_text(encoding="utf-8"))
                else:
                    with self.assertRaisesRegex(failure, "package-startup-map-ambiguous"):
                        exec(branch, namespace)
                    self.assertEqual(ini.read_text(encoding="utf-8"), contents)
                    self.assertNotIn("startup_map", report)

    def test_rpc_readiness_requires_successful_version_and_map_calls_before_workload(self):
        source, main, failure = bootstrap_tree()
        parent = next(node for node in ast.walk(main) if hasattr(node, "body") and isinstance(node.body, list)
                      and any(isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "ready_until"
                                                                  for target in item.targets) for item in node.body))
        index = next(index for index, item in enumerate(parent.body) if isinstance(item, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "ready_until" for target in item.targets))
        branch = compile(ast.Module(body=parent.body[index:index + 3], type_ignores=[]), "bootstrap-rpc.py", "exec")
        for outcome in ("ready", "rpc-failure", "server-exited"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as temporary:
                run = Mock(return_value=SimpleNamespace(returncode=0 if outcome == "ready" else 1,
                                                        stdout="RPC_READY" if outcome == "ready" else "", stderr=""))
                namespace = {"time": SimpleNamespace(monotonic=Mock(side_effect=[0.0, 181.0]), sleep=Mock()),
                             "remaining": lambda seconds: seconds, "process": SimpleNamespace(poll=lambda: 0 if outcome == "server-exited" else None),
                             "subprocess": SimpleNamespace(run=run), "python": "isolated-python",
                             "output": Path(temporary), "env": {}, "BootstrapFailure": failure}
                if outcome == "ready":
                    exec(branch, namespace)
                    rpc_code = run.call_args.args[0][2]
                    self.assertIn("get_server_version()=='0.9.16'", rpc_code)
                    self.assertIn("c.get_world().get_map()", rpc_code)
                    namespace["time"].sleep.assert_not_called()
                else:
                    expected = "server-exited-before-rpc-ready" if outcome == "server-exited" else "server-rpc-ready-timeout"
                    with self.assertRaisesRegex(failure, expected):
                        exec(branch, namespace)
                if outcome == "server-exited":
                    run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
