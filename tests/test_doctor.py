"""Meaningful failure-path and privacy checks for the dependency-free D1 audit."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("doctor", ROOT / "scripts" / "doctor.py")
assert SPEC is not None and SPEC.loader is not None
doctor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = doctor
SPEC.loader.exec_module(doctor)


class CommandTests(unittest.TestCase):
    def test_absent_command_is_unknown_and_keeps_no_error_text(self):
        with patch.object(doctor.subprocess, "run", side_effect=FileNotFoundError("C:/Users/private-person/token")):
            result = doctor.run_command(["missing"])
        self.assertEqual(result.reason, "not-found")
        self.assertEqual(result.stdout, "")

    def test_failure_discards_stdout_and_stderr(self):
        failure = subprocess.CompletedProcess(["git"], 1, "private-output", "secret-token")
        with patch.object(doctor.subprocess, "run", return_value=failure):
            result = doctor.run_command(["git"])
        self.assertEqual(result, doctor.CommandResult(reason="failed"))

    def test_timeout_is_bounded_and_discards_partial_output(self):
        with patch.object(doctor.subprocess, "run", side_effect=subprocess.TimeoutExpired("probe", 8, output="private-output")) as run:
            result = doctor.run_command(["probe"])
        self.assertEqual(result.reason, "timeout")
        self.assertEqual(run.call_args.kwargs["timeout"], 8)


class ParsingTests(unittest.TestCase):
    def test_gpu_csv_handles_quoted_commas_and_multiple_devices(self):
        devices = doctor.parse_gpu_csv('"NVIDIA, Test GPU", 6144, 616.92\nNVIDIA Second, 8192, 616.92\n')
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0]["name"], "NVIDIA, Test GPU")
        self.assertEqual(devices[0]["memory_mib"], 6144)

    def test_malformed_gpu_cannot_become_verified_capacity(self):
        with patch.object(doctor, "run_command", return_value=doctor.CommandResult("NVIDIA Test, N/A, 616.92")):
            report = doctor.probe_gpus()
        self.assertEqual(report["probe_status"], "invalid-response")
        self.assertEqual(report["devices"], [])

    def test_unknown_ram_is_preserved_and_unselected_identity_fields_are_dropped(self):
        payload = {"cpu_name": "Intel Test", "memory_bytes": None, "username": "private-person", "serial": "private-serial"}
        with patch.object(doctor, "run_command", return_value=doctor.CommandResult(json.dumps(payload))):
            report = doctor.probe_windows_hardware()
        self.assertIsNone(report["memory_bytes"])
        self.assertNotIn("private", json.dumps(report))

    def test_path_shaped_hardware_labels_are_rejected(self):
        self.assertIsNone(doctor.safe_label("C:\\Users\\private-person"))

    def test_optional_simulator_inspection_does_not_claim_server_version(self):
        with tempfile.TemporaryDirectory(prefix="private-person-") as folder:
            root = Path(folder)
            (root / "CarlaUE4.exe").touch()
            dist = root / "PythonAPI" / "carla" / "dist"
            dist.mkdir(parents=True)
            (dist / "carla-0.9.16-cp312-cp312-win_amd64.whl").touch()
            (dist / "private-person-token.whl").touch()
            report = doctor.probe_simulator(root)
        self.assertTrue(report["executable_found"])
        self.assertFalse(report["server_version_verified"])
        self.assertEqual(report["windows_wheels"], [{"version": "0.9.16", "python": "cp312", "abi": "cp312"}])
        self.assertNotIn("private-person", json.dumps(report))


class ReportTests(unittest.TestCase):
    def missing_report(self, root):
        with patch.object(doctor, "run_command", return_value=doctor.CommandResult(reason="not-found")), patch.object(doctor.metadata, "version", side_effect=doctor.metadata.PackageNotFoundError), patch.object(doctor.platform, "system", return_value="Windows"):
            return doctor.build_report(project_root=root)

    def test_missing_tools_complete_audit_without_claiming_d2_readiness(self):
        with tempfile.TemporaryDirectory(prefix="private-person-") as folder:
            report = self.missing_report(Path(folder))
        self.assertTrue(report["audit"]["d1_complete"])
        self.assertFalse(report["audit"]["d2_preflight_ready"])
        self.assertFalse(report["audit"]["d2_connection_verified"])
        self.assertIsNone(report["memory"]["total_gib"])
        self.assertNotIn("private-person", json.dumps(report))
        checks = {check["id"]: check["status"] for check in report["checks"]}
        self.assertEqual(checks["ram"], "warn")
        self.assertEqual(checks["carla-api"], "deferred")

    def test_package_inspection_reads_metadata_without_importing_packages(self):
        with patch.object(doctor.metadata, "version", return_value="0.9.16") as version:
            report = doctor.package_version("carla")
        version.assert_called_once_with("carla")
        self.assertEqual(report, {"status": "installed-metadata", "version": "0.9.16"})

    def test_non_string_package_metadata_is_unknown(self):
        with patch.object(doctor.metadata, "version", return_value=None):
            self.assertEqual(doctor.package_version("carla"), {"status": "unknown", "version": None})

    def test_six_gib_gpu_is_warning_and_does_not_block_package_preflight(self):
        gpu = {"probe_status": "ok", "devices": [{"name": "NVIDIA Test", "memory_mib": 6144, "driver_version": "616.92"}]}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "CarlaUE4.exe").touch()
            with patch.object(doctor, "run_command", return_value=doctor.CommandResult(reason="not-found")), patch.object(doctor.metadata, "version", return_value="0.9.16"), patch.object(doctor.platform, "system", return_value="Windows"), patch.object(doctor.platform, "machine", return_value="AMD64"), patch.object(doctor.sys, "version_info", (3, 12, 7)), patch.object(doctor, "probe_gpus", return_value=gpu):
                report = doctor.build_report(project_root=root, carla_root=root)
        checks = {check["id"]: check["status"] for check in report["checks"]}
        self.assertEqual(checks["gpu"], "warn")
        self.assertTrue(report["audit"]["d2_preflight_ready"])
        self.assertFalse(report["audit"]["d2_connection_verified"])

    def test_arm64_is_not_reported_as_compatible_with_windows_x64_wheel(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(doctor.platform, "machine", return_value="ARM64"):
            report = self.missing_report(Path(folder))
        self.assertFalse(report["python"]["carla_wheel_supported"])

    def test_incompatible_explicit_wheels_prevent_d2_preflight_ready(self):
        for filename in ("carla-0.9.15-cp312-cp312-win_amd64.whl", "carla-0.9.16-cp311-cp311-win_amd64.whl"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "CarlaUE4.exe").touch()
                dist = root / "PythonAPI" / "carla" / "dist"
                dist.mkdir(parents=True)
                (dist / filename).touch()
                with patch.object(doctor, "run_command", return_value=doctor.CommandResult(reason="not-found")), patch.object(doctor.metadata, "version", return_value="0.9.16"), patch.object(doctor.platform, "system", return_value="Windows"), patch.object(doctor.platform, "machine", return_value="AMD64"), patch.object(doctor.sys, "version_info", (3, 12, 7)):
                    report = doctor.build_report(project_root=root, carla_root=root)
                self.assertTrue(report["simulator"]["wheel_files_found"])
                self.assertFalse(report["simulator"]["compatible_windows_wheel_found"])
                self.assertFalse(report["audit"]["d2_preflight_ready"])
                checks = {check["id"]: check["status"] for check in report["checks"]}
                self.assertEqual(checks["simulator-wheels"], "fail")

    def test_explicit_output_creates_missing_parent_and_exports_json(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "parent with spaces" / "report.json"
            report = self.missing_report(Path(folder))
            doctor.write_report(report, destination)
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), report)

    def test_default_exit_succeeds_and_strict_exit_reports_pending_preflight(self):
        with tempfile.TemporaryDirectory() as folder:
            report = self.missing_report(Path(folder))
        with patch.object(doctor, "build_report", return_value=report), redirect_stdout(io.StringIO()):
            self.assertEqual(doctor.main([]), 0)
            self.assertEqual(doctor.main(["--strict"]), 1)

    def test_output_failure_does_not_print_sensitive_exception(self):
        with tempfile.TemporaryDirectory() as folder:
            report = self.missing_report(Path(folder))
        stderr = io.StringIO()
        with patch.object(doctor, "build_report", return_value=report), patch.object(doctor, "write_report", side_effect=PermissionError("C:/Users/private-person/token")), redirect_stderr(stderr):
            self.assertEqual(doctor.main(["--output", "report.json"]), 2)
        self.assertIn("Could not write", stderr.getvalue())
        self.assertNotIn("private-person", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
