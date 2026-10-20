"""CPU-only freeze closure, path isolation and write-once regressions."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from carla_tools.contracts import canonical_sha256
from carla_tools.evaluation_freeze import (
    EvaluationFreezeError, build_freeze, freeze_id, public_source_paths,
    verify_freeze, write_freeze)
from carla_tools.final_evaluation import build_manifest
from carla_tools.selected_pipeline import pipeline_id
from scripts.freeze_final_evaluation import main

try:
    from tests.test_final_evaluation import routes_fixture, selection_fixture
except ModuleNotFoundError:
    from test_final_evaluation import routes_fixture, selection_fixture


PROJECT = Path(__file__).resolve().parents[1]


class EvaluationFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.selection = selection_fixture()
        for name in {*self.selection["source_sha256"], "carla_tools/evaluation_freeze.py",
                     "carla_tools/final_evaluation.py", "carla_tools/contracts.py", "scripts/freeze_final_evaluation.py"}:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / name, target)
        self.store = self.root / "private"
        self.store.mkdir()
        self.checkpoint = self.store / "checkpoint.pt"
        # The freeze binds bytes without deserializing a model. CPU tensor
        # loading is covered by the independent selected-runtime tests.
        self.checkpoint.write_bytes(b"synthetic-checkpoint-binding-not-trained-weights")
        self.selection["model"]["checkpoint_sha256"] = self.digest(self.checkpoint)
        self.selection["pipeline_id"] = pipeline_id(self.selection)
        self.selection_path = self.store / "pipeline.json"
        self.save(self.selection_path, self.selection)
        self.review_path = self.store / "review.json"
        self.save(self.review_path, {"status": "PASS", "scope": "synthetic bounded review"})
        self.routes = routes_fixture()
        self.routes_path = self.store / "routes.json"
        self.save(self.routes_path, self.routes)
        self.artifacts, bindings = [], []
        for name, roles in (("dataset", ["dataset", "split"]), ("selection", ["selection"]),
                            ("topology", ["topology"]), ("budget", ["budget"])):
            path = self.store / (name + ".json")
            self.save(path, {"evidence": name, "fixture_only": True})
            self.artifacts.append(path)
            bindings.append({"path": path.relative_to(self.root).as_posix(),
                             "sha256": self.digest(path), "roles": roles})
        self.manifest = build_manifest(self.routes, self.selection, bindings)
        self.manifest_path = self.store / "manifest.json"
        self.save(self.manifest_path, self.manifest)
        for name, content in (("README.md", "Public usage\n"), ("requirements-cpu.txt", "numpy\n"),
                              ("configs/runtime.json", '{"fixed_delta_seconds":0.05}\n'),
                              ("docs/usage.md", "Evaluation method\n"),
                              ("docs/local/private-record.md", "Private record\n"),
                              ("docs/ROADMAP.md", "Private planning\n"),
                              ("docs/EVALUATION_SPEC_D5.md", "Historical private record\n")):
            path = self.root / name
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_text(content, encoding="utf-8")
        self.output = self.store / "freeze.json"

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def save(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    def build(self, **kwargs):
        return build_freeze(self.root, self.manifest_path, self.routes_path, self.selection_path,
            kwargs.pop("artifact_paths", self.artifacts), kwargs.pop("source_paths", None),
            development_review_path=kwargs.pop("development_review_path", self.review_path), **kwargs)

    def frozen(self):
        value = self.build()
        write_freeze(self.output, value)
        return value

    def rewrite(self, value):
        value["freeze_id"] = freeze_id(value)
        self.save(self.output, value)

    def test_actual_manifest_ninety_trials_and_complete_selected_binding_roundtrip(self):
        value = self.frozen()
        observed = verify_freeze(self.output, self.root, expected_freeze_id=value["freeze_id"])
        self.assertEqual(observed["status"], "PASS")
        self.assertTrue(observed["expected_identity_checked"])
        self.assertEqual(observed["selected_pipeline_id"], self.selection["pipeline_id"])
        self.assertEqual(observed["verified_inputs"]["checkpoint"]["sha256"], self.digest(self.checkpoint))
        self.assertEqual(observed["verified_inputs"]["manifest"]["sha256"], self.digest(self.manifest_path))
        self.assertFalse(observed["GPU_used"])
        self.assertFalse(observed["heldout_observation_executed"])
        self.assertEqual(len(self.manifest["trials"]), 90)
        dataset = next(item for item in value["files"] if item["path"] == "private/dataset.json")
        self.assertEqual(dataset["roles"], ["artifact", "manifest:dataset", "manifest:split"])

    def test_write_once_preserves_original_even_for_another_valid_freeze(self):
        value = self.frozen()
        before = self.output.read_bytes()
        another = copy.deepcopy(value)
        another["created_utc"] += "-different"
        another["freeze_id"] = freeze_id(another)
        with self.assertRaisesRegex(EvaluationFreezeError, "already-exists"):
            write_freeze(self.output, another)
        self.assertEqual(self.output.read_bytes(), before)
        self.assertEqual(list(self.store.glob("*.pending")), [])

    def test_code_tamper_rejected(self):
        self.frozen()
        path = self.root / "carla_tools/controller.py"
        path.write_bytes(path.read_bytes() + b"\n# changed\n")
        with self.assertRaises(ValueError):
            verify_freeze(self.output, self.root)

    def test_config_tamper_rejected(self):
        self.frozen()
        (self.root / "configs/runtime.json").write_text('{"fixed_delta_seconds":0.1}')
        with self.assertRaisesRegex(EvaluationFreezeError, "bound-file-mismatch"):
            verify_freeze(self.output, self.root)

    def test_checkpoint_tamper_and_missing_file_rejected(self):
        self.frozen()
        original = self.checkpoint.read_bytes()
        self.checkpoint.write_bytes(b"another-checkpoint")
        with self.assertRaises(ValueError):
            verify_freeze(self.output, self.root)
        self.checkpoint.write_bytes(original)
        self.artifacts[0].unlink()
        with self.assertRaisesRegex(EvaluationFreezeError, "input-not-file"):
            verify_freeze(self.output, self.root)

    def test_selected_identity_and_controller_changes_rejected(self):
        self.frozen()
        self.selection["controller_config"]["base"]["target_speed_kmh"] = 35.
        self.save(self.selection_path, self.selection)
        with self.assertRaises(ValueError):
            verify_freeze(self.output, self.root)

    def test_selected_checkpoint_declared_hash_must_match_actual_bytes(self):
        self.selection["model"]["checkpoint_sha256"] = "f" * 64
        self.selection["pipeline_id"] = pipeline_id(self.selection)
        self.save(self.selection_path, self.selection)
        self.manifest = build_manifest(self.routes, self.selection, self.manifest["artifact_bindings"])
        self.save(self.manifest_path, self.manifest)
        with self.assertRaisesRegex(EvaluationFreezeError, "selected-checkpoint-mismatch"):
            self.build()

    def test_source_and_artifact_omission_or_duplicate_rejected_at_build(self):
        sources = public_source_paths(self.root)
        for kwargs in ({"source_paths": sources[:-1]}, {"source_paths": sources + [sources[0]]},
                       {"artifact_paths": self.artifacts[:-1]}, {"artifact_paths": self.artifacts * 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build(**kwargs)

    def test_required_provenance_role_cannot_be_omitted(self):
        self.manifest["artifact_bindings"][0]["roles"] = ["dataset"]
        self.save(self.manifest_path, self.manifest)
        with self.assertRaises(ValueError):
            self.build()

    def test_metadata_artifact_tamper_rejected_before_freezing(self):
        self.artifacts[0].write_text('{"fixture_changed":true}')
        with self.assertRaisesRegex(EvaluationFreezeError, "declared-artifact-hash-mismatch"):
            self.build()

    def test_new_unexpected_source_or_removed_source_rejected(self):
        self.frozen()
        new_source = self.root / "carla_tools/unexpected_module.py"
        new_source.write_text("CHANGED = True\n")
        with self.assertRaisesRegex(EvaluationFreezeError, "inventory-mismatch"):
            verify_freeze(self.output, self.root)
        new_source.unlink()
        (self.root / "docs/usage.md").unlink()
        with self.assertRaisesRegex(EvaluationFreezeError, "inventory-mismatch"):
            verify_freeze(self.output, self.root)

    def test_recomputed_self_hash_does_not_hide_missing_core_or_file_bindings(self):
        value = self.frozen()
        original = copy.deepcopy(value)
        for modification in ("checkpoint_input", "files_entry", "required_source", "duplicate_entry", "role"):
            changed = copy.deepcopy(original)
            if modification == "checkpoint_input":
                changed["inputs"].pop("checkpoint")
            elif modification == "files_entry":
                changed["files"].pop()
            elif modification == "required_source":
                changed["required_source_paths"].pop()
            elif modification == "duplicate_entry":
                changed["files"].append(copy.deepcopy(changed["files"][0]))
            else:
                changed["files"][0]["roles"] = []
            self.rewrite(changed)
            with self.subTest(modification=modification), self.assertRaises(ValueError):
                verify_freeze(self.output, self.root)

    def test_retained_external_id_rejects_altered_freeze_even_with_recomputed_self_hash(self):
        value = self.frozen()
        expected = value["freeze_id"]
        value["created_utc"] += "-different"
        self.rewrite(value)
        # Metadata editing can be internally consistent, hence the independent ID.
        self.assertEqual(verify_freeze(self.output, self.root)["status"], "PASS")
        with self.assertRaisesRegex(EvaluationFreezeError, "unexpected-identity"):
            verify_freeze(self.output, self.root, expected_freeze_id=expected)

    def test_path_traversal_absolute_drive_alternate_stream_and_case_alias_rejected(self):
        for name in ("../outside", "/outside", "C:/outside", "private/../checkpoint.pt", "private\\file",
                     "private/checkpoint.pt:secret", "private/NUL.txt", "private/file. "):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.build(artifact_paths=[*self.artifacts, name])
        names = [path.relative_to(self.root).as_posix() for path in self.artifacts]
        with self.assertRaisesRegex(EvaluationFreezeError, "duplicate-path"):
            self.build(artifact_paths=names + [names[0].upper()])

    def test_checkpoint_path_traversal_rejected_before_any_model_read(self):
        self.selection["model"]["checkpoint_path"] = "../outside.pt"
        self.selection["pipeline_id"] = pipeline_id(self.selection)
        self.save(self.selection_path, self.selection)
        self.manifest = build_manifest(self.routes, self.selection, self.manifest["artifact_bindings"])
        self.save(self.manifest_path, self.manifest)
        with self.assertRaisesRegex(EvaluationFreezeError, "unsafe-relative"):
            self.build()

    def test_symlink_file_and_parent_are_rejected(self):
        original = self.artifacts[0]
        outside = self.root / "outside.json"
        outside.write_bytes(original.read_bytes())
        original.unlink()
        try:
            original.symlink_to(outside)
        except OSError:
            # Windows may not grant link creation. Exercise the same OS link
            # response without requiring developer mode or elevated privilege.
            original.write_bytes(outside.read_bytes())
            native = Path.is_symlink
            for linked in (original, original.parent):
                with patch.object(Path, "is_symlink", autospec=True,
                        side_effect=lambda p: p == linked or native(p)):
                    with self.assertRaisesRegex(EvaluationFreezeError, "link-forbidden"):
                        self.build()
        else:
            with self.assertRaisesRegex(EvaluationFreezeError, "link-forbidden"):
                self.build()

    def test_concurrent_publication_preserves_other_writer_bytes(self):
        import carla_tools.evaluation_freeze as module
        value = self.build()
        name = "rename" if module.os.name == "nt" else "link"
        original = getattr(module.os, name)

        def another_writer(source, destination):
            Path(destination).write_bytes(b"other-writer-complete-artifact")
            return original(source, destination)

        with patch.object(module.os, name, side_effect=another_writer):
            with self.assertRaisesRegex(EvaluationFreezeError, "already-exists"):
                write_freeze(self.output, value)
        self.assertEqual(self.output.read_bytes(), b"other-writer-complete-artifact")

    def test_nonPASS_review_duplicate_JSON_and_observed_manifest_rejected(self):
        for encoded in ('{"status":"FAIL"}', '{"status":"PASS","status":"PASS"}'):
            self.review_path.write_text(encoded, encoding="utf-8")
            with self.subTest(encoded=encoded), self.assertRaises(ValueError):
                self.build()
        self.save(self.review_path, {"status": "PASS"})
        self.manifest["final_results_observed"] = True
        self.save(self.manifest_path, self.manifest)
        with self.assertRaises(ValueError):
            self.build()

    def test_raw_JSON_byte_change_rejected_despite_equal_parsed_semantics(self):
        self.frozen()
        self.manifest_path.write_text(json.dumps(self.manifest, indent=4), encoding="utf-8")
        with self.assertRaisesRegex(EvaluationFreezeError, "bound-file-mismatch"):
            verify_freeze(self.output, self.root)

    def test_private_records_and_caches_excluded_public_docs_and_requirements_bound(self):
        value = self.frozen()
        self.assertNotIn("docs/local/private-record.md", value["source_paths"])
        self.assertNotIn("docs/ROADMAP.md", value["source_paths"])
        self.assertNotIn("docs/EVALUATION_SPEC_D5.md", value["source_paths"])
        self.assertIn("docs/usage.md", value["source_paths"])
        self.assertIn("requirements-cpu.txt", value["source_paths"])
        cache = self.root / "carla_tools/__pycache__/ignored.pyc"
        cache.parent.mkdir()
        cache.write_bytes(b"generated")
        (self.root / "docs/local/private-record.md").write_text("Another private record")
        self.assertEqual(verify_freeze(self.output, self.root)["status"], "PASS")

    def test_changed_input_during_verification_rejected(self):
        self.frozen()
        import carla_tools.evaluation_freeze as module
        original = module._read
        changed = False

        def mutate_after_read(root, relative, **kwargs):
            nonlocal changed
            result = original(root, relative, **kwargs)
            if relative == "private/checkpoint.pt" and not changed:
                changed = True
                self.manifest_path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
            return result

        with patch.object(module, "_read", side_effect=mutate_after_read):
            with self.assertRaisesRegex(EvaluationFreezeError, "changed-during|bound-file-mismatch"):
                verify_freeze(self.output, self.root)

    def test_supplied_project_runtime_cannot_differ_from_executing_validator_sources(self):
        path = self.root / "carla_tools/evaluation_freeze.py"
        path.write_bytes(path.read_bytes() + b"\n# another verifier\n")
        with self.assertRaisesRegex(EvaluationFreezeError, "execution-source-mismatch"):
            self.build()

    def test_CLI_create_and_verify_use_same_contract_and_refuse_overwrite(self):
        arguments = ["create", "--project-root", str(self.root), "--manifest", str(self.manifest_path),
            "--routes", str(self.routes_path), "--selection", str(self.selection_path),
            "--development-review", str(self.review_path), "--output", str(self.output)]
        for artifact in self.artifacts:
            arguments.extend(["--artifact", str(artifact)])
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            self.assertEqual(main(arguments), 0)
        frozen = json.loads(captured.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            self.assertEqual(main(["verify", "--project-root", str(self.root), "--freeze", str(self.output),
                "--expected-freeze-id", frozen["freeze_id"]]), 0)
        self.assertEqual(json.loads(captured.getvalue())["status"], "PASS")
        with self.assertRaisesRegex(EvaluationFreezeError, "already-exists"):
            main(arguments)


if __name__ == "__main__":
    unittest.main()
