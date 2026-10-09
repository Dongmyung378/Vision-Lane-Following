"""Pilot CLI publication boundaries and preflight checks."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_pilot_pipeline import run_pipeline
from scripts.verify_pilot import main


class PilotPipelineTests(unittest.TestCase):
    def test_existing_output_refuses_before_importing_carla_or_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "pilot"
            destination.mkdir()
            marker = destination / "kept.txt"
            marker.write_text("keep")
            with patch("scripts.run_pilot_pipeline.importlib.import_module", side_effect=AssertionError("CARLA must not import")):
                report = run_pipeline(destination)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["error"]["code"], "pilot-output-already-exists")
            self.assertEqual(marker.read_text(), "keep")
            self.assertFalse((destination / "pilot_report.json").exists())

    def test_bootstrap_failure_preserves_failed_attempt_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "pilot"
            with patch("scripts.run_pilot_pipeline.importlib.import_module", side_effect=ImportError("missing CARLA")):
                report = run_pipeline(destination)
            self.assertEqual(report["status"], "FAIL")
            self.assertTrue((destination / "pilot_report.json").is_file())
            self.assertTrue((destination.parent / "pilot_artifacts.zip").is_file())
            self.assertFalse((destination / "pilot_manifest.json").exists())

    def test_validator_refuses_output_inside_sources_and_existing_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sources" / "pilot_manifest.json"
            source.parent.mkdir()
            source.write_text("{}")
            existing = root / "gate.json"
            existing.write_text("kept")
            for output in (source.parent / "gate.json", existing):
                with self.subTest(output=output), self.assertRaises(SystemExit):
                    main(["--manifest", str(source), "--output", str(output)])
            self.assertEqual(existing.read_text(), "kept")

    def test_invalid_source_writes_explicit_failed_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sources" / "pilot_manifest.json"
            source.parent.mkdir()
            source.write_text("{}")
            output = root / "gate.json"
            self.assertEqual(main(["--manifest", str(source), "--output", str(output)]), 1)
            self.assertEqual(json.loads(output.read_text())["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
