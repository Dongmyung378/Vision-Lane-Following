"""Protect previous artifacts and retain pre-frame failures in metric exports."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from carla_tools.evaluation import load_evaluation_config, write_metric_csv
from scripts.run_baseline_pipeline import _attempt_metric, run_pipeline


class BaselinePipelineTests(unittest.TestCase):
    def test_existing_output_refused_before_simulator_import_and_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "output"
            directory.mkdir()
            sentinel = directory / "baseline_report.json"
            sentinel.write_bytes(b"previous original evidence")
            with patch("scripts.run_baseline_pipeline.importlib.import_module") as imported:
                report = run_pipeline(directory)
            imported.assert_not_called()
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["error"]["code"], "baseline-pipeline-output-already-exists")
            self.assertEqual(sentinel.read_bytes(), b"previous original evidence")
            self.assertFalse((directory.parent / "baseline_artifacts.zip").exists())

    def test_missing_frame_bootstrap_failure_stays_in_csv_with_unknown_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "failed-native"
            directory.mkdir()
            original = {"episode_id": "failed-native", "status": "FAILED", "termination_reason": "technical_failure"}
            source = directory / "episode.json"
            source.write_text(json.dumps(original), encoding="utf-8")
            before = source.read_bytes()
            metric = _attempt_metric(directory, "B0", {"route_id": "route-example"}, load_evaluation_config())
            self.assertEqual(metric["termination_reason"], "technical_failure")
            self.assertIsNone(metric["completion_pct"])
            self.assertFalse(metric["integrity_valid"])
            self.assertEqual(metric["method"], "B0")
            self.assertEqual(metric["route_id"], "route-example")
            self.assertEqual(source.read_bytes(), before)
            aggregate = write_metric_csv([metric], Path(temporary) / "failure.csv")
            self.assertEqual(aggregate["episode_count"], 1)
            self.assertEqual(aggregate["groups"][0]["failed_episodes"], 1)
            self.assertIn("N/A", (Path(temporary) / "failure.csv").read_text())

    def test_corrupt_trace_cannot_reuse_cached_success_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            metadata = {"episode_id": "damaged", "status": "COMPLETE", "evaluation_route_length_m": 19.,
                        "evaluation_start_reference_m": 0., "condition": "center", "termination_reason": "route_completed",
                        "evaluation_metrics": {"integrity_valid": True, "driving_success": True, "sample_count": 999}}
            (directory / "episode.json").write_text(json.dumps(metadata), encoding="utf-8")
            damaged = b'{"frame":1,"timestamp":'
            (directory / "steps.jsonl").write_bytes(damaged)
            metric = _attempt_metric(directory, "B0", {"route_id": "route-example"}, load_evaluation_config())
            self.assertFalse(metric["integrity_valid"])
            self.assertFalse(metric["driving_success"])
            self.assertEqual(metric["sample_count"], 0)
            self.assertEqual(metric["termination_reason"], "technical_failure")
            self.assertEqual((directory / "steps.jsonl").read_bytes(), damaged)


if __name__ == "__main__":
    unittest.main()
