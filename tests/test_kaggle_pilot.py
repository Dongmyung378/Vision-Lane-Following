"""Pilot runtime transport and failure artifacts retain native source proof."""
from __future__ import annotations

import contextlib
import ast
import base64
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from scripts import prepare_kaggle_week1 as packager
from scripts.prepare_kaggle_pilot import main
from test_kaggle_baseline import runtime_helpers
from test_kaggle_driving import native_branch, bootstrap_tree
from test_kaggle_week1 import literal_assignments


class KagglePilotTests(unittest.TestCase):
    def test_linux_bootstrap_admits_pilot_and_rejects_unsupported_or_missing_sources(self):
        source, main_node, failure = bootstrap_tree()
        guard = next(node for node in ast.walk(main_node) if isinstance(node, ast.If)
                     and "linux-and-embedded-probes-required" in (ast.get_source_segment(source, node) or "")
                     and "sys.platform" in ast.unparse(node.test))
        code = compile(ast.Module(body=[guard], type_ignores=[]), "bootstrap-workload-preflight", "exec")
        namespace = {"sys": SimpleNamespace(platform="linux"), "PROBE_SOURCE": "rgb probe",
                     "VULKAN_CHILD_SOURCE": "Vulkan probe", "WORKLOAD": "pilot", "BootstrapFailure": failure}
        exec(code, namespace)
        for replacement in ({"WORKLOAD": "unsupported"}, {"PROBE_SOURCE": ""},
                            {"VULKAN_CHILD_SOURCE": ""}, {"sys": SimpleNamespace(platform="win32")}):
            with self.subTest(replacement=replacement), self.assertRaisesRegex(failure, "linux-and-embedded-probes-required"):
                exec(code, {**namespace, **replacement})

    def test_frozen_existing_notebook_bundle_has_complete_sources_without_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "existing.json"
            metadata.write_text(json.dumps({"id": packager.NOTEBOOK_ID, "title": packager.NOTEBOOK_TITLE}))
            output = root / "bundle"
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(main(["--existing-metadata", str(metadata), "--output-dir", str(output)]), 0)
            result = json.loads(stream.getvalue())
            self.assertEqual(result["notebook_id"], packager.NOTEBOOK_ID)
            self.assertEqual(result["title"], packager.NOTEBOOK_TITLE)
            self.assertFalse(result["submitted"])
            self.assertFalse(result["creates_new_notebook"])
            code = (output / "week1.py").read_text()
            constants = literal_assignments(code)
            payload = zlib.decompress(base64.b64decode(constants["PROJECT_SOURCES_PACKED"], validate=True))
            self.assertEqual(hashlib.sha256(payload).hexdigest(), constants["PROJECT_PAYLOAD_SHA256"])
            sources = json.loads(payload)
            self.assertLess(len(code.encode("utf-8")), 1_000_000)
            expected = set(packager.SOURCES + packager.DRIVING_SOURCES + packager.BASELINE_SOURCES + packager.PILOT_SOURCES)
            self.assertEqual(set(sources), expected)
            self.assertNotIn("AGENTS.md", sources)
            self.assertNotIn("docs/ROADMAP.md", sources)
            self.assertNotIn("carla_tools/waypoint_model.py", sources)
            for relative, source in sources.items():
                if relative.endswith(".py"):
                    compile(source, relative, "exec")
                self.assertEqual(source, (packager.ROOT / relative).read_text(encoding="utf-8"))

    def test_wrong_existing_notebook_cannot_create_replacement_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "existing.json"
            metadata.write_text(json.dumps({"id": "someone/new-notebook", "title": packager.NOTEBOOK_TITLE}))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--existing-metadata", str(metadata), "--output-dir", str(root / "out")]), 1)
            self.assertFalse((root / "out").exists())

    def test_pilot_artifact_export_preserves_runtime_bytes_and_failure_output(self):
        helper = runtime_helpers()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, output = root / "project", root / "output"
            project.mkdir()
            output.mkdir()
            (project / "public.py").write_bytes(b"VALUE = 7\n")
            report = helper["export_baseline_runtime_sources"](project, output, {"public.py": "VALUE = 7\n"}, artifact="pilot")
            with zipfile.ZipFile(output / "pilot_runtime_sources.zip") as archive:
                self.assertEqual(archive.read("public.py"), b"VALUE = 7\n")
            self.assertEqual(json.loads((output / "pilot_runtime_source_manifest.json").read_text()), report)
            (output / "pilot").mkdir()
            (output / "pilot" / "pilot_report.json").write_bytes(b'{"status":"FAIL"}\n')
            helper["preserve_baseline_failure_artifacts"](output, artifact="pilot")
            with zipfile.ZipFile(output / "pilot_artifacts.zip") as archive:
                self.assertEqual(archive.read("pilot/pilot_report.json"), b'{"status":"FAIL"}\n')

    def test_native_pilot_branch_checks_full_gate_and_has_bounded_collection_command(self):
        branch, failure = native_branch("embedded-project-required")
        for status in ("PASS", "FAIL"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output, scratch = root / "output", root / "scratch"
                output.mkdir()
                scratch.mkdir()
                (output / "pilot").mkdir()
                # The real pilot pipeline keeps episode_count in its nested
                # independent gate and calls the source count native_frame_count.
                evidence = {"status": "PASS", "native_frame_count": 3425,
                            "sample_count": 2970, "route_count": 7,
                            "split_counts": {"train": 1665, "dev": 1305},
                            "gate": {"status": status, "episode_count": 35}}
                (output / "pilot" / "pilot_report.json").write_text(json.dumps(evidence))
                command = Mock()
                namespace = {"WORKLOAD": "pilot", "PROJECT_SOURCES": {"public.py": "VALUE = 7\n"},
                             "scratch": scratch, "output": output, "python": "client312", "env": {},
                             "command": command, "stage": Mock(), "json": json, "report": {},
                             "BootstrapFailure": failure,
                             "export_baseline_runtime_sources": runtime_helpers()["export_baseline_runtime_sources"]}
                if status == "PASS":
                    exec(branch, namespace)
                    self.assertTrue(namespace["report"]["pilot_verified"])
                    self.assertEqual(namespace["report"]["pilot"]["episode_count"], 35)
                    self.assertEqual(namespace["report"]["pilot"]["frame_count"], 3425)
                    self.assertEqual(namespace["report"]["pilot"]["sample_count"], 2970)
                    self.assertEqual(namespace["report"]["pilot"]["split_counts"], {"train": 1665, "dev": 1305})
                    self.assertEqual(command.call_args.args[2], 1800)
                    self.assertIn("run_pilot_pipeline.py", command.call_args.args[0][1])
                else:
                    with self.assertRaisesRegex(failure, "pilot-evidence-not-pass"):
                        exec(branch, namespace)


if __name__ == "__main__":
    unittest.main()
