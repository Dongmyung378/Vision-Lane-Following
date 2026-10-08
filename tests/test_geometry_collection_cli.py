"""Three-case collection orchestration without CARLA or server processes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from carla_tools.config import CollectionConfig
from carla_tools.geometry_survey import survey_geometry_catalog
from scripts import run_geometry_collection as runner

try:
    from tests.test_geometry_survey import CARLA, Map
except ModuleNotFoundError:
    from test_geometry_survey import CARLA, Map


class GeometryCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.world_map = Map()
        self.catalog = survey_geometry_catalog(self.world_map, CARLA)
        self.catalog_path = self.root / "catalog.json"
        self.catalog_path.write_text(json.dumps(self.catalog), encoding="utf-8")
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(CollectionConfig().to_dict()), encoding="utf-8")
        self.output = self.root / "dataset-d6-geometry-v1"
        self.world = SimpleNamespace(get_map=Mock(return_value=self.world_map),
                                     tick=Mock(side_effect=AssertionError("Collector must never tick the world")))
        self.client = SimpleNamespace(set_timeout=Mock(), get_client_version=Mock(return_value="0.9.16"),
                                      get_server_version=Mock(return_value="0.9.16"), get_world=Mock(return_value=self.world))
        self.carla = SimpleNamespace(Client=Mock(return_value=self.client), LaneType=CARLA.LaneType)
        self.captured = []

    def capture(self, cfg, destination, metadata, carla_module=None):
        self.captured.append((cfg, destination, metadata, carla_module))
        destination.mkdir()
        return {"status": "PASS", "frames_written": cfg.verification.frames,
                "cleanup": {"status": "PASS"}, "validation": {"status": "PASS"}}

    def run_collection(self, capture_function=None, **kwargs):
        return runner.collect_geometry_cases(self.catalog_path, self.output, self.config_path,
                                              carla_module=self.carla,
                                              capture_function=capture_function or self.capture, **kwargs)

    def test_three_source_verified_cases_use_exact_spawn_indices_and_stationary_metadata(self):
        report = self.run_collection()
        self.assertEqual(report["status"], "PASS", report.get("error"))
        self.assertTrue(report["report_written"])
        self.assertEqual([case["case"] for case in report["cases"]], ["straight", "left_curve", "right_curve"])
        self.assertEqual([cfg.vehicle.spawn_point_index for cfg, _, _, _ in self.captured], [0, 2, 1])
        self.assertEqual(self.client.get_world.call_count, 3)
        self.assertEqual(self.client.get_server_version.call_count, 3)
        self.world.tick.assert_not_called()
        for cfg, destination, metadata, module in self.captured:
            self.assertEqual(cfg.verification.frames, 60)
            self.assertEqual(cfg.vehicle.control.to_dict(), CollectionConfig().vehicle.control.to_dict())
            self.assertEqual(metadata["control_mode"], "stationary_D4_smoke")
            self.assertEqual(metadata["labels_state"], "not-generated-D6")
            self.assertEqual(metadata["perturbation"]["provenance"], "stationary-no-perturbation")
            self.assertIs(module, self.carla)
            self.assertEqual(metadata["route_contract"]["spawn_point_index"], cfg.vehicle.spawn_point_index)
            self.assertTrue((self.output / f"{destination.name}_capture_report.json").is_file())
        summary = json.loads((self.output / runner.REPORT_NAME).read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "PASS")
        self.assertFalse(summary["training_split_created"])
        self.assertFalse(summary["labels_generated"])
        self.assertFalse(summary["server_lifecycle_owned"])

    def test_preexisting_case_or_report_is_refused_before_client_without_overwrite(self):
        for existing in ("geometry_left_curve_01", runner.REPORT_NAME):
            with self.subTest(existing=existing):
                self.output.mkdir(exist_ok=True)
                sentinel = self.output / existing
                sentinel.write_text("preserved", encoding="utf-8")
                report = self.run_collection()
                self.assertEqual(report["status"], "FAIL")
                self.assertEqual(report["error"]["code"], "geometry-output-already-exists")
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserved")
                self.assertFalse(report["report_written"])
                self.carla.Client.assert_not_called()
                sentinel.unlink()

    def test_live_source_change_refuses_capture_and_persists_failure_report(self):
        self.world_map.to_opendrive = lambda: "changed-map-source"
        report = self.run_collection()
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"]["code"], "live-geometry-route-contract-mismatch")
        self.assertEqual(self.captured, [])
        self.assertTrue(report["report_written"])
        self.world.tick.assert_not_called()

    def test_later_live_source_change_preserves_first_success_and_stops_remaining_cases(self):
        def capture_and_change(*args, **kwargs):
            value = self.capture(*args, **kwargs)
            self.world_map.to_opendrive = lambda: "changed-after-first-episode"
            return value

        report = self.run_collection(capture_and_change)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(len(self.captured), 1)
        self.assertEqual([case["status"] for case in report["cases"]], ["PASS", "FAIL", "NOT_RUN"])

    def test_version_mismatch_refuses_capture(self):
        self.client.get_server_version.return_value = "0.9.15"
        report = self.run_collection()
        self.assertEqual(report["error"]["code"], "carla-version-mismatch")
        self.assertEqual(self.captured, [])
        self.client.get_world.assert_not_called()

    def test_capture_failure_stops_remaining_cases_and_keeps_all_completed_evidence(self):
        def failure(cfg, destination, metadata, **kwargs):
            self.captured.append((cfg, destination, metadata, kwargs))
            destination.mkdir()
            return {"status": "FAIL", "frames_written": 4, "cleanup": {"status": "PASS"},
                    "validation": {"status": "FAIL"}, "error": {"code": "rgb-frame-timeout"}}

        report = self.run_collection(failure)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(len(self.captured), 1)
        self.assertEqual(report["cases"][0]["error"]["code"], "rgb-frame-timeout")
        stored = json.loads((self.output / report["cases"][0]["capture_report"]).read_text(encoding="utf-8"))
        self.assertEqual(stored["status"], "FAIL")

    def test_pass_flag_without_cleanup_disk_validation_or_full_frame_count_is_refused(self):
        for missing in ("cleanup", "validation", "count"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                self.output = Path(directory) / "geometry"

                def incomplete(cfg, destination, metadata, **kwargs):
                    value = self.capture(cfg, destination, metadata, **kwargs)
                    if missing == "count":
                        value["frames_written"] = 59
                    else:
                        value[missing]["status"] = "FAIL"
                    return value

                report = self.run_collection(incomplete)
                self.assertEqual(report["status"], "FAIL")
                self.assertEqual(report["error"]["code"], "geometry-case-capture-failed")

    def test_rpc_exception_and_interruption_are_sanitized_and_never_pass(self):
        for error, expected in ((RuntimeError("private-person/secret-token"), "geometry-collection-operation-failed"),
                                (KeyboardInterrupt(), "geometry-collection-interrupted")):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                self.output = Path(directory) / "geometry"
                self.client.get_world.side_effect = error
                report = self.run_collection()
                self.assertEqual(report["status"], "FAIL")
                self.assertEqual(report["error"]["code"], expected)
                self.assertNotIn("secret-token", json.dumps(report))
                self.assertTrue(report["report_written"])

    def test_boolean_frame_count_cannot_prove_single_frame_capture(self):
        def invalid_count(cfg, destination, metadata, **kwargs):
            value = self.capture(cfg, destination, metadata, **kwargs)
            value["frames_written"] = True
            return value

        report = self.run_collection(invalid_count, frames=1)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"]["code"], "geometry-case-capture-failed")

    def test_invalid_frame_count_or_config_has_no_runtime_import_or_output(self):
        with patch.object(runner.importlib, "import_module", side_effect=AssertionError("Runtime import forbidden")):
            report = runner.collect_geometry_cases(self.catalog_path, self.output, self.config_path, frames=1001)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"]["code"], "invalid-capture-frame-count")
        self.assertFalse(self.output.exists())

    def test_validate_only_cli_never_imports_carla_or_creates_output(self):
        with patch("builtins.print") as printed, patch.object(runner.importlib, "import_module", side_effect=AssertionError("Runtime import forbidden")):
            code = runner.main(["--catalog", str(self.catalog_path), "--config", str(self.config_path),
                                "--output-root", str(self.output), "--validate-only"])
        self.assertEqual(code, 0)
        value = json.loads(printed.call_args.args[0])
        self.assertEqual(value["status"], "CONFIG_VALIDATED")
        self.assertFalse(value["simulator_verified"])
        self.assertFalse(self.output.exists())

    def test_cli_prints_compact_summary_and_returns_nonzero_on_failure(self):
        failure = {"status": "FAIL", "cases": [], "report_written": True, "error": {"code": "fixture-failure"}}
        with patch.object(runner, "collect_geometry_cases", return_value=failure), patch("builtins.print") as printed:
            code = runner.main(["--catalog", "unused.json"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(printed.call_args.args[0])["error"]["code"], "fixture-failure")


if __name__ == "__main__":
    unittest.main()
