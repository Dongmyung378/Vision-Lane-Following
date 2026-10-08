"""CPU-only pipeline failure boundaries and QA writes confined to owned copies."""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from carla_tools import week1
from carla_tools.labels import validate_labels
from scripts import run_week1_pipeline as pipeline
from test_week1 import make_case


class Week1PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def fake_pipeline(self, destination, *, captured=None, gate=None, capture_error=None):
        """Mock simulator-only stages; retain the actual exclusive report writer."""
        carla = SimpleNamespace(Client=Mock())
        client = carla.Client.return_value
        client.get_client_version.return_value = "0.9.16"
        client.get_server_version.return_value = "0.9.16"
        catalog = {"status": "SURVEY_VERIFIED", "candidate_counts": {}, "spawn_points_scanned": 3,
                   "classification_policy": {}}
        captured = captured if captured is not None else {
            "status": "PASS", "cases": [{"case": case, "episode_directory": case}
                                         for case in ("straight", "left_curve", "right_curve")]}
        gate = gate if gate is not None else {"status": "PASS", "episode_count": 3, "frame_count": 6}
        with ExitStack() as stack:
            mocks = {}
            values = {"survey_geometry_catalog": catalog, "load_geometry_catalog": catalog,
                      "gpu_sample": {"status": "unavailable"}, "collect_geometry_cases": captured,
                      "generate_labels": {"status": "PASS", "source": {}, "metrics": {}, "labels_sha256": "a" * 64},
                      "render_episode_overlay": {"status": "PASS", "visible_points": 4, "all_points_in_front": True},
                      "verify_week1_cases": gate, "negative_checks": {"status": "PASS", "count": 9},
                      "check_connection": {"status": "PASS"}}
            for name, value in values.items():
                mocks[name] = stack.enter_context(patch.object(pipeline, name, return_value=value))
            if capture_error is not None:
                mocks["collect_geometry_cases"].side_effect = capture_error
            importer = stack.enter_context(patch.object(pipeline.importlib, "import_module", return_value=carla))
            stack.enter_context(redirect_stdout(io.StringIO()))
            report = pipeline.run_pipeline(destination)
        return report, mocks, importer

    def test_existing_output_is_refused_without_import_or_changed_pass_evidence(self):
        destination = self.directory / "existing"
        destination.mkdir()
        sentinel = destination / "week1_report.json"
        original = b'{"status":"PASS","old":true}\n'
        sentinel.write_bytes(original)
        with patch.object(pipeline.importlib, "import_module") as importer:
            report = pipeline.run_pipeline(destination)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"]["code"], "pipeline-output-already-exists")
        importer.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), original)
        self.assertEqual(list(destination.iterdir()), [sentinel])

    def test_failed_capture_cannot_reach_labels_gate_negative_qa_or_rpc(self):
        destination = self.directory / "capture-failure"
        report, mocks, _ = self.fake_pipeline(destination, captured={"status": "FAIL", "cases": []})
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"]["code"], "geometry-capture-failed")
        for name in ("generate_labels", "render_episode_overlay", "verify_week1_cases", "negative_checks", "check_connection"):
            mocks[name].assert_not_called()
        self.assertEqual(json.loads((destination / "week1_report.json").read_text())["status"], "FAIL")

    def test_failed_gate_preserves_failure_and_stops_negative_qa_and_rpc(self):
        destination = self.directory / "gate-failure"
        report, mocks, _ = self.fake_pipeline(destination, gate={"status": "FAIL", "errors": [{"code": "fixture"}]})
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"]["code"], "week1-gate-failed")
        self.assertEqual(report["gate"]["status"], "FAIL")
        mocks["negative_checks"].assert_not_called()
        mocks["check_connection"].assert_not_called()
        self.assertEqual(json.loads((destination / "week1_report.json").read_text())["status"], "FAIL")

    def test_capture_interruption_leaves_failed_report_and_no_later_stage_claims(self):
        destination = self.directory / "interrupted"
        report, mocks, _ = self.fake_pipeline(destination, capture_error=KeyboardInterrupt())
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"]["exception_type"], "KeyboardInterrupt")
        self.assertEqual(json.loads((destination / "week1_report.json").read_text())["status"], "FAIL")
        mocks["generate_labels"].assert_not_called()
        self.assertNotIn("gate", report)
        self.assertNotIn("episode_count", report)

    def test_native_negative_checks_use_actual_fixtures_and_leave_original_bytes_unchanged(self):
        case = make_case(self.directory, "left_curve", 1)
        before = {str(path.relative_to(case["episode"])): path.read_bytes()
                  for path in case["episode"].rglob("*") if path.is_file()}
        label_bytes = case["labels"].read_bytes()
        qa_parent = self.directory / "qa"
        qa_parent.mkdir()
        report = pipeline.negative_checks(case["episode"], case["labels"], qa_parent)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["count"], 9)
        self.assertTrue(all(check["observed_status"] == "FAIL" for check in report["checks"]))
        self.assertFalse(report["source_files_modified"])
        after = {str(path.relative_to(case["episode"])): path.read_bytes()
                 for path in case["episode"].rglob("*") if path.is_file()}
        self.assertEqual(after, before)
        self.assertEqual(case["labels"].read_bytes(), label_bytes)
        self.assertEqual(list(qa_parent.iterdir()), [])

    def test_invalid_negative_qa_source_stops_before_copy_or_mutation(self):
        with patch.object(pipeline, "validate_labels", return_value={"status": "FAIL", "errors": []}), \
                patch.object(pipeline.shutil, "copytree") as clone:
            with self.assertRaisesRegex(pipeline.PipelineError, "negative-QA-source-invalid"):
                pipeline.negative_checks(self.directory / "missing-raw", self.directory / "missing-labels", self.directory)
        clone.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_changed_source_absolute_image_path_cannot_write_outside_qa_clone(self):
        case = make_case(self.directory, "straight", 0)
        rows_path = case["episode"] / "frames.jsonl"
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
        original_image = case["episode"] / rows[0]["image"]["path"]
        victim = self.directory / "victim.png"
        victim_bytes = original_image.read_bytes()
        victim.write_bytes(victim_bytes)
        injected, attempted_writes = False, []
        original_write_bytes = Path.write_bytes

        def validate_then_change_path(episode, labels):
            nonlocal injected
            checked = validate_labels(episode, labels)
            if not injected and Path(labels) == case["labels"] and checked["status"] == "PASS":
                injected = True
                rows[0]["image"]["path"] = str(victim.resolve())
                rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            return checked

        def never_write_victim(path, data):
            if path.resolve() == victim.resolve():
                attempted_writes.append(path)
                raise AssertionError("Outside-clone write attempted; temporary victim protected.")
            return original_write_bytes(path, data)

        with patch.object(pipeline, "validate_labels", side_effect=validate_then_change_path), \
                patch.object(Path, "write_bytes", never_write_victim):
            with self.assertRaisesRegex(pipeline.PipelineError, "negative-QA-image-outside-owned-clone"):
                pipeline.negative_checks(case["episode"], case["labels"], self.directory)
        self.assertTrue(injected)
        self.assertEqual(attempted_writes, [])
        self.assertEqual(victim.read_bytes(), victim_bytes)

    def test_audit_source_changed_after_initial_validation_cannot_report_stale_pass(self):
        case = make_case(self.directory, "straight", 0)
        metadata_path = case["episode"] / "episode.json"
        changed = False

        def validate_then_change_metadata(episode, labels):
            nonlocal changed
            checked = validate_labels(episode, labels)
            if not changed and checked["status"] == "PASS":
                changed = True
                metadata = json.loads(metadata_path.read_text())
                metadata["dataset_version"] = "changed-during-audit"
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            return checked

        with patch.object(week1, "validate_labels", side_effect=validate_then_change_metadata):
            report = week1.audit_labeled_episode(case["episode"], case["labels"])
        self.assertTrue(changed)
        self.assertEqual(report["status"], "FAIL", "Audit used a stale validated identity after the source metadata changed.")
        self.assertTrue(report["errors"])
        self.assertNotIn("checks", report)


if __name__ == "__main__":
    unittest.main()
