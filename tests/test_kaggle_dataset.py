"""Existing-notebook identity, complete source transport and scratch capture."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import Mock
import unittest
import zlib
import zipfile

from scripts import prepare_kaggle_dataset as packager
from test_dataset_pipeline import catalog_fixture
from test_kaggle_week1 import literal_assignments


class KaggleDatasetTests(unittest.TestCase):
    def prepare(self, root, **overrides):
        metadata = root / "existing.json"
        metadata.write_text(json.dumps({"id": packager.NOTEBOOK_ID, "title": packager.NOTEBOOK_TITLE,
                                      "is_private": True, **overrides}))
        catalog = root / "catalog.json"
        catalog.write_text(json.dumps(catalog_fixture()))
        return packager.prepare_dataset(metadata, root / "bundle", route_catalog_path=catalog)

    def test_frozen_transport_keeps_identity_and_all_native_source_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self.prepare(root)
            code = (root / "bundle/native.py").read_text()
            constants = literal_assignments(code)
            sources = json.loads(zlib.decompress(base64.b64decode(constants["PROJECT_SOURCES_PACKED"], validate=True)))
            tree = ast.parse(code)
            update = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "PROJECT_SOURCES")
            payload = next(node.value for node in ast.walk(update) if isinstance(node, ast.Constant) and isinstance(node.value, str))
            sources.update(json.loads(zlib.decompress(base64.b64decode(payload, validate=True))))
            self.assertEqual(set(sources), set(bundle["source_sha256"]))
            for name, source in sources.items():
                self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), bundle["source_sha256"][name])
                self.assertEqual(source, (packager.ROOT / name).read_text(encoding="utf-8"))
            from carla_tools.dataset_v1 import SOURCE_MODULES
            self.assertTrue(set(SOURCE_MODULES) <= set(sources))
            self.assertEqual(bundle["expected_episodes"], 700)
            self.assertEqual(bundle["maximum_seconds"], 18000)
            self.assertEqual(bundle["pre_submit_checks"]["status"], "PASS")
            self.assertEqual(hashlib.sha256(code.encode()).hexdigest(), bundle["code_sha256"])
            self.assertFalse(bundle["submitted"])
            self.assertFalse(bundle["creates_new_notebook"])
            self.assertLess(len(code.encode()), 1_000_000)
            submission = json.loads((root / "bundle/kernel-metadata.json").read_text())
            self.assertEqual(submission["id"], packager.NOTEBOOK_ID)
            self.assertEqual(submission["title"], packager.NOTEBOOK_TITLE)
            self.assertTrue(submission["is_private"])
            self.assertEqual(submission["code_file"], "native.py")
            self.assertNotIn("AGENTS.md", sources)
            self.assertFalse(any(name.startswith(("results/", "docs/local/")) for name in sources))

    def test_wrong_identity_or_public_metadata_cannot_create_a_new_notebook_bundle(self):
        for invalid in ({"id": "other/new"}, {"title": "another title"}, {"is_private": False}, {"is_private": None}):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with self.assertRaisesRegex(ValueError, "existing-private-notebook-identity"):
                    self.prepare(root, **invalid)
                self.assertFalse((root / "bundle").exists())

    def test_generated_bootstrap_collects_in_scratch_and_retains_failed_scratch_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare(root)
            code = (root / "bundle/native.py").read_text()
            self.assertIn('"--output-dir", str(scratch / "dataset")', code)
            self.assertIn('"--artifact-dir", str(output)', code)
            self.assertIn('"--frozen-route-catalog", str(frozen_routes)', code)
            self.assertIn('"--runtime-sources", str(output / "dataset_runtime_source_manifest.json")', code)
            self.assertIn('preserve_baseline_failure_artifacts(output, artifact="dataset", source_directory=scratch / "dataset", owned_root=scratch)', code)
            self.assertIn('16200 if pilot else 600', code)
            self.assertIn('maximum_seconds = 18000 if WORKLOAD == "dataset"', code)
            self.assertIn('WORKLOAD = \'dataset\'', code)
            self.assertIn('"carla_dataset" if pilot', code)

    def test_actual_native_branch_requires_dataset_gate_and_exports_runtime_sources(self):
        for status in ("PASS", "FAIL"):
            with self.subTest(gate_status=status), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.prepare(root)
                code = (root / "bundle/native.py").read_text()
                tree = ast.parse(code)
                functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
                failure = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BootstrapFailure")
                namespace = {"Path": Path, "hashlib": hashlib, "json": json, "zipfile": zipfile}
                exec(compile(ast.Module(body=[failure, functions["export_baseline_runtime_sources"]], type_ignores=[]),
                             "dataset-source-export", "exec"), namespace)
                main = functions["main"]
                branch = next(node for node in ast.walk(main) if isinstance(node, ast.If)
                    and "WORKLOAD" in ast.unparse(node.test)
                    and "embedded-project-required" in (ast.get_source_segment(code, node) or ""))
                scratch, output = root / "scratch", root / "working"
                scratch.mkdir()
                output.mkdir()
                evidence = {"status": "PASS", "gate": {"status": status, "episode_count": 700},
                    "native_frame_count": 55000, "sample_count": 50000, "split_counts": {"train": 30000, "dev": 20000}, "route_count": 7}
                (output / "dataset_report.json").write_text(json.dumps(evidence))
                command = Mock()
                namespace.update(WORKLOAD="dataset", PROJECT_SOURCES={"public.py": "VALUE = 7\n"},
                    FROZEN_DATASET_ROUTES=literal_assignments(code)["FROZEN_DATASET_ROUTES"],
                    scratch=scratch, output=output, python="python312", env={}, command=command, stage=Mock(), report={},
                    base64=base64, zlib=zlib)
                compiled = compile(ast.Module(body=[branch], type_ignores=[]), "dataset-native-branch", "exec")
                if status == "PASS":
                    exec(compiled, namespace)
                    self.assertTrue(namespace["report"]["dataset_verified"])
                    self.assertEqual(namespace["report"]["dataset"]["frame_count"], 55000)
                    arguments, label, timeout, _env = command.call_args.args
                    self.assertEqual(arguments[1], str(scratch / "project/scripts/run_dataset_pipeline.py"))
                    self.assertEqual(arguments[arguments.index("--output-dir") + 1], str(scratch / "dataset"))
                    self.assertEqual(label, "carla_dataset")
                    self.assertEqual(timeout, 16200)
                else:
                    with self.assertRaisesRegex(namespace["BootstrapFailure"], "dataset-evidence-not-pass"):
                        exec(compiled, namespace)
                with zipfile.ZipFile(output / "dataset_runtime_sources.zip") as archive:
                    self.assertEqual(archive.read("public.py"), b"VALUE = 7\n")

    def retention_helper(self, root):
        self.prepare(root)
        tree = ast.parse((root / "bundle/native.py").read_text())
        failure = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BootstrapFailure")
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "preserve_baseline_failure_artifacts")
        namespace = {"Path": Path}
        exec(compile(ast.Module(body=[failure, helper], type_ignores=[]), "dataset-retention-parent", "exec"), namespace)
        return namespace["preserve_baseline_failure_artifacts"], namespace["BootstrapFailure"]

    def native_retention_fixture(self, root):
        owner, output = root / "owned-scratch", root / "working"
        native = owner / "dataset"
        native.mkdir(parents=True)
        output.mkdir()
        sources = {"images/frame.png": b"original-native-rgb", "steps.jsonl": b'{"frame":1}\n'}
        for name, payload in sources.items():
            path = native / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        return owner, output, native, sources

    def test_parent_recovers_a_truncated_zip_and_retains_failure_facts_and_all_raw_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            helper, _failure = self.retention_helper(root)
            owner, output, native, sources = self.native_retention_fixture(root)
            path = output / "dataset_artifacts.zip"
            with zipfile.ZipFile(path, "x") as archive:
                archive.writestr("dataset/images/frame.png", sources["images/frame.png"])
            broken = path.read_bytes()[:-25]
            path.write_bytes(broken)
            report = helper(output, artifact="dataset", source_directory=native, owned_root=owner)
            self.assertEqual(report["status"], "RECOVERED_COMPLETE")
            self.assertEqual(report["dataset_quality_status"], "NOT_EVALUATED")
            self.assertEqual(report["incomplete_reason"], "archive-unreadable-or-truncated")
            self.assertEqual(report["incomplete_archive_sha256"], hashlib.sha256(broken).hexdigest())
            retained = owner / report["retained_incomplete_archive_in_owned_scratch"]
            self.assertEqual(retained.read_bytes(), broken)
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(set(archive.namelist()), {"dataset/" + name for name in sources})
                for name, payload in sources.items():
                    self.assertEqual(archive.read("dataset/" + name), payload)
                    self.assertEqual((native / name).read_bytes(), payload)
            evidence = json.loads((output / "dataset_archive_retention.json").read_text())
            self.assertEqual(evidence["events"][-1], report)

    def test_valid_prefix_zip_or_same_size_stale_bytes_are_recovered_against_full_source_inventory(self):
        for mismatch in ("prefix", "stale"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                helper, _failure = self.retention_helper(root)
                owner, output, native, sources = self.native_retention_fixture(root)
                path = output / "dataset_artifacts.zip"
                with zipfile.ZipFile(path, "x") as archive:
                    for index, (name, payload) in enumerate(sources.items()):
                        if mismatch == "prefix" and index > 0:
                            continue
                        archive.writestr("dataset/" + name, b"x" * len(payload) if mismatch == "stale" else payload)
                report = helper(output, artifact="dataset", source_directory=native, owned_root=owner)
                self.assertEqual(report["status"], "RECOVERED_COMPLETE")
                self.assertEqual(report["incomplete_reason"], "archive-inventory-mismatch" if mismatch == "prefix"
                                 else "archive-source-bytes-mismatch")
                with zipfile.ZipFile(path) as archive:
                    self.assertEqual({name.removeprefix("dataset/"): archive.read(name) for name in archive.namelist()}, sources)

    def test_complete_valid_archive_is_left_byte_for_byte_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            helper, _failure = self.retention_helper(root)
            owner, output, native, sources = self.native_retention_fixture(root)
            path = output / "dataset_artifacts.zip"
            with zipfile.ZipFile(path, "x", zipfile.ZIP_DEFLATED) as archive:
                for name, payload in sources.items():
                    archive.writestr("dataset/" + name, payload)
            original, before = path.read_bytes(), path.stat().st_mtime_ns
            report = helper(output, artifact="dataset", source_directory=native, owned_root=owner)
            self.assertEqual(report["status"], "PRESERVED_COMPLETE")
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(path.stat().st_mtime_ns, before)
            self.assertEqual(list(owner.glob("dataset_incomplete_artifacts_*.zip")), [])

    def test_parent_retention_rejects_source_outside_the_explicit_owned_scratch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            helper, failure = self.retention_helper(root)
            owner, output, _native, _sources = self.native_retention_fixture(root)
            outside = root / "unowned/dataset"
            outside.mkdir(parents=True)
            marker = outside / "keep.txt"
            marker.write_bytes(b"keep")
            with self.assertRaisesRegex(failure, "directory-invalid"):
                helper(output, artifact="dataset", source_directory=outside, owned_root=owner)
            self.assertEqual(marker.read_bytes(), b"keep")
            self.assertFalse((output / "dataset_artifacts.zip").exists())


if __name__ == "__main__":
    unittest.main()
