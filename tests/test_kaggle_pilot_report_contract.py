"""Check the bootstrap against the actual producer's persisted report schema.

Native collection, sensor replay and the RPC endpoint are replaced at their
unit boundaries. The production pilot pipeline builds the report and archive;
the production bootstrap branch consumes those same persisted report bytes.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from carla_tools.pilot import load_pilot_config, select_pilot_routes
from scripts.run_pilot_pipeline import run_pipeline
from test_kaggle_baseline import runtime_helpers
from test_kaggle_driving import native_branch
from test_pilot import candidate_fixture


class KagglePilotReportContractTests(unittest.TestCase):
    def produce_pilot(self, output, *, reference_success=True):
        selected = select_pilot_routes(candidate_fixture(), load_pilot_config())
        carla = SimpleNamespace(Client=Mock(return_value=SimpleNamespace(
            set_timeout=Mock(), get_client_version=lambda: "0.9.16", get_server_version=lambda: "0.9.16",
            get_world=lambda: SimpleNamespace(get_map=lambda: object()))))

        def collect(route, case, condition, directory, collection, *, carla_module, method):
            self.assertEqual(method, "privileged_reference")
            directory.mkdir(parents=True)
            (directory / "episode.json").write_text(json.dumps({"evaluation_metrics": {"driving_success": True}}))
            return {"status": "PASS", "step_count": 90, "case": case, "condition": condition["name"]}

        def manifest(directory, routes, episodes, config):
            self.assertEqual(len(routes), 7)
            self.assertEqual(len(episodes), 35)
            path = directory / "pilot_manifest.json"
            path.write_text('{"fixture":"report-contract"}')
            return path

        gate = {"status": "PASS", "sample_count": 2000, "episode_count": 35,
                "split_counts": {"train": 1400, "dev": 600}, "manifest_sha256": "a" * 64}
        baseline = {"status": "PASS", "gate": {"status": "PASS", "episode_metrics": [
            {"method": "privileged_reference", "driving_success": reference_success} for _ in range(3)] + [
            {"method": "B0", "driving_success": False} for _ in range(3)]}}
        with patch("scripts.run_pilot_pipeline.survey_pilot_routes", return_value={"routes": selected}), \
                patch("scripts.run_pilot_pipeline.run_baseline_trial", side_effect=collect), \
                patch("scripts.run_pilot_pipeline.build_pilot_manifest", side_effect=manifest), \
                patch("scripts.run_pilot_pipeline.validate_pilot_manifest", return_value=gate), \
                patch("scripts.run_pilot_pipeline.run_baseline_pipeline", return_value=baseline), \
                patch("scripts.run_pilot_pipeline.check_connection", return_value={"status": "PASS"}), \
                patch("builtins.print"), \
                patch("scripts.run_pilot_pipeline.importlib.import_module", return_value=carla):
            returned = run_pipeline(output / "pilot")
        persisted = json.loads((output / "pilot" / "pilot_report.json").read_text())
        return returned, persisted

    def bootstrap(self, output, workload):
        branch, failure = native_branch("embedded-project-required")
        scratch = output.parent / ("scratch-" + workload)
        scratch.mkdir()
        namespace = {"WORKLOAD": workload, "PROJECT_SOURCES": {"public.py": "VALUE = 7\n"},
                     "scratch": scratch, "output": output, "python": "client312", "env": {},
                     "command": Mock(), "stage": Mock(), "json": json, "report": {},
                     "BootstrapFailure": failure,
                     "export_baseline_runtime_sources": runtime_helpers()["export_baseline_runtime_sources"]}
        exec(branch, namespace)
        return namespace

    def test_actual_persisted_pilot_schema_is_consumed_without_legacy_count_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            output.mkdir()
            returned, persisted = self.produce_pilot(output)
            self.assertEqual(returned["status"], "PASS", returned)
            self.assertEqual(persisted["status"], "PASS", persisted)
            self.assertNotIn("episode_count", persisted)
            self.assertNotIn("frame_count", persisted)
            self.assertNotIn("archive_sha256", persisted)
            self.assertIn("archive_sha256", returned)
            self.assertEqual(persisted["native_frame_count"], 3150)
            namespace = self.bootstrap(output, "pilot")
            summary = namespace["report"]["pilot"]
            self.assertTrue(namespace["report"]["pilot_verified"])
            self.assertEqual(summary["episode_count"], 35)
            self.assertEqual(summary["frame_count"], 3150)
            self.assertEqual(summary["sample_count"], 2000)
            self.assertEqual(summary["split_counts"], {"train": 1400, "dev": 600})
            self.assertEqual(summary["route_count"], 7)
            self.assertEqual(namespace["command"].call_args.args[2], 1800)

    def test_failed_development_reference_blocks_a_valid_collection_source_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            output.mkdir()
            returned, persisted = self.produce_pilot(output, reference_success=False)
            self.assertEqual(returned["status"], "FAIL")
            self.assertEqual(persisted["status"], "FAIL")
            self.assertEqual(persisted["gate"]["status"], "PASS")
            self.assertEqual(persisted["error"]["code"], "pilot-development-reference-driving-failed")
            self.assertNotIn("connection_after", persisted)
            with self.assertRaisesRegex(RuntimeError, "pilot-evidence-not-pass"):
                self.bootstrap(output, "pilot")

    def test_existing_workloads_retain_legacy_count_fields(self):
        for workload, episodes in (("week1", 3), ("driving", 15), ("baseline", 6)):
            with self.subTest(workload=workload), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "output"
                directory = output / workload
                directory.mkdir(parents=True)
                evidence = {"status": "PASS", "episode_count": episodes, "frame_count": 300, "gate": {"status": "PASS"}}
                (directory / (workload + "_report.json")).write_text(json.dumps(evidence))
                namespace = self.bootstrap(output, workload)
                self.assertEqual(namespace["report"][workload], evidence)
                self.assertTrue(namespace["report"][workload + "_verified"])


if __name__ == "__main__":
    unittest.main()
