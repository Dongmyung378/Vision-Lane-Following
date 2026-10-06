"""Read-only connection guarantees and failure handling without CARLA installed."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_carla_connection", ROOT / "scripts" / "check_carla_connection.py")
assert SPEC is not None and SPEC.loader is not None
connection = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = connection
SPEC.loader.exec_module(connection)


SUPPORTED_RUNTIME = {
    "implementation": "CPython", "python_version": "3.12.7", "python_major": 3,
    "python_minor": 12, "architecture": "x86_64", "pointer_bits": 64,
}


def snapshot(frame, elapsed, delta=0.05):
    return SimpleNamespace(frame=frame, timestamp=SimpleNamespace(elapsed_seconds=elapsed, delta_seconds=delta))


class ReadOnlyWorld:
    def __init__(self, ticks=None, synchronous=False, map_name="/Game/Carla/Maps/Town01"):
        self.initial = snapshot(10, 0.5)
        self.pending = list(ticks if ticks is not None else [snapshot(11, 0.55), snapshot(12, 0.60), snapshot(13, 0.65)])
        self.settings = SimpleNamespace(synchronous_mode=synchronous, no_rendering_mode=False, fixed_delta_seconds=None)
        self.map_name = map_name
        self.map_reads = 0
        self.wait_timeouts = []
        self.mutations = []

    def get_map(self):
        self.map_reads += 1
        return SimpleNamespace(name=self.map_name)

    def get_snapshot(self):
        return self.initial

    def get_settings(self):
        return self.settings

    def wait_for_tick(self, timeout):
        self.wait_timeouts.append(timeout)
        return self.pending.pop(0) if self.pending else None

    def tick(self, *args):
        self.mutations.append("tick")
        raise AssertionError("Read-only probe must not advance the world")

    def apply_settings(self, *args):
        self.mutations.append("apply_settings")
        raise AssertionError("Read-only probe must not change settings")

    def spawn_actor(self, *args):
        self.mutations.append("spawn_actor")
        raise AssertionError("D2 connection check must not spawn actors")


def fake_carla(world, client_version="0.9.16", server_version="0.9.16"):
    client = SimpleNamespace(
        set_timeout=Mock(), get_client_version=lambda: client_version,
        get_server_version=lambda: server_version, get_world=lambda: world,
        load_world=Mock(side_effect=AssertionError("Read-only check must not load a world")),
    )
    return SimpleNamespace(Client=Mock(return_value=client)), client


class ConnectionTests(unittest.TestCase):
    def setUp(self):
        self.runtime_patch = patch.object(connection, "runtime_info", return_value=dict(SUPPORTED_RUNTIME))
        self.package_patch = patch.object(connection.metadata, "version", return_value="0.9.16")
        self.runtime = self.runtime_patch.start()
        self.package = self.package_patch.start()
        self.addCleanup(self.runtime_patch.stop)
        self.addCleanup(self.package_patch.stop)

    def test_real_api_contract_requires_three_increasing_snapshots_without_mutation(self):
        world = ReadOnlyWorld()
        carla, client = fake_carla(world)
        report = connection.check_connection(connection.ConnectionConfig(), carla)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["observed_ticks"], 3)
        self.assertEqual([tick["frame"] for tick in report["ticks"]], [11, 12, 13])
        self.assertEqual(world.wait_timeouts, [15.0, 15.0, 15.0])
        client.set_timeout.assert_called_once_with(15.0)
        client.load_world.assert_not_called()
        self.assertEqual(world.map_reads, 1)
        self.assertEqual(world.mutations, [])

    def test_async_frame_gaps_are_valid_when_frame_and_time_both_increase(self):
        world = ReadOnlyWorld([snapshot(12, 0.6), snapshot(15, 0.75), snapshot(20, 1.0)])
        carla, _ = fake_carla(world)
        report = connection.check_connection(connection.ConnectionConfig(), carla)
        self.assertEqual(report["status"], "PASS")
        self.assertAlmostEqual(report["observed_simulation_seconds"], 0.5)

    def test_synchronous_world_is_pending_and_is_never_ticked_or_reconfigured(self):
        world = ReadOnlyWorld(synchronous=True)
        carla, client = fake_carla(world)
        report = connection.check_connection(connection.ConnectionConfig(), carla)
        self.assertEqual(report["status"], "PENDING")
        self.assertEqual(report["error"]["code"], "external-synchronous-world")
        self.assertEqual(report["observed_ticks"], 0)
        self.assertEqual(world.wait_timeouts, [])
        self.assertEqual(world.mutations, [])
        client.load_world.assert_not_called()
        self.assertIn("elapsed_seconds", report)

    def test_unsupported_python_or_architecture_stops_before_package_or_server_access(self):
        for changes in ({"python_minor": 13}, {"python_minor": 9}, {"pointer_bits": 32},
                        {"architecture": "unsupported"}, {"implementation": "PyPy"}):
            with self.subTest(changes=changes):
                self.runtime.return_value = dict(SUPPORTED_RUNTIME, **changes)
                carla, _ = fake_carla(ReadOnlyWorld())
                report = connection.check_connection(connection.ConnectionConfig(), carla)
                self.assertEqual(report["status"], "FAIL")
                self.assertEqual(report["error"]["stage"], "python-runtime")
                carla.Client.assert_not_called()
        self.package.assert_not_called()

    def test_missing_package_does_not_attempt_to_import_carla(self):
        self.package.side_effect = connection.metadata.PackageNotFoundError("private/path/secret")
        with patch.object(connection.importlib, "import_module") as loader:
            report = connection.check_connection(connection.ConnectionConfig())
        self.assertEqual(report["error"]["code"], "carla-package-not-installed")
        loader.assert_not_called()
        self.assertNotIn("secret", json.dumps(report))

    def test_package_version_mismatch_stops_before_server_access(self):
        self.package.return_value = "0.9.15"
        carla, _ = fake_carla(ReadOnlyWorld())
        report = connection.check_connection(connection.ConnectionConfig(), carla)
        self.assertEqual(report["error"]["code"], "installed-carla-version-mismatch")
        carla.Client.assert_not_called()

    def test_client_or_server_version_mismatch_cannot_pass(self):
        for client_version, server_version in (("0.9.15", "0.9.16"), ("0.9.16", "0.9.15"), ("0.9.16", "0.9.16-dirty")):
            with self.subTest(client=client_version, server=server_version):
                world = ReadOnlyWorld()
                carla, _ = fake_carla(world, client_version, server_version)
                report = connection.check_connection(connection.ConnectionConfig(), carla)
                self.assertEqual(report["status"], "FAIL")
                self.assertEqual(report["error"]["code"], "carla-package-api-server-version-mismatch")
                self.assertEqual(world.map_reads, 0)

    def test_missing_runtime_module_is_fail_and_keeps_no_exception_text(self):
        with patch.object(connection.importlib, "import_module", side_effect=ImportError("private-person/access-token")):
            report = connection.check_connection(connection.ConnectionConfig())
        self.assertEqual(report["error"]["stage"], "carla-import")
        self.assertNotIn("access-token", json.dumps(report))

    def test_rpc_failure_reports_stage_and_type_without_private_host_or_error_text(self):
        carla, client = fake_carla(ReadOnlyWorld())
        client.get_server_version = Mock(side_effect=RuntimeError("private-hostname private-person secret-token"))
        report = connection.check_connection(connection.ConnectionConfig(host="private-hostname"), carla)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"], {"stage": "server-version", "code": "operation-failed", "exception_type": "RuntimeError"})
        self.assertNotIn("private-hostname", json.dumps(report))
        self.assertNotIn("secret-token", json.dumps(report))

    def test_private_map_paths_and_malformed_versions_are_not_exported(self):
        world = ReadOnlyWorld(map_name="C:/Users/private-person/secret-token/custom_map")
        carla, _ = fake_carla(world)
        report = connection.check_connection(connection.ConnectionConfig(), carla)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["map_name"], "custom-map")
        self.assertNotIn("private-person", json.dumps(report))
        self.package.return_value = "private-person/secret-token"
        report = connection.check_connection(connection.ConnectionConfig(), carla)
        self.assertEqual(report["package_version"], "invalid-response")
        self.assertNotIn("secret-token", json.dumps(report))
        self.package.return_value = "0.9.16"
        carla, _ = fake_carla(world, server_version="0.9.16-private-person")
        report = connection.check_connection(connection.ConnectionConfig(), carla)
        self.assertEqual(report["server_version"], "invalid-response")
        self.assertNotIn("private-person", json.dumps(report))

    def test_nonincreasing_frame_or_time_fails_and_keeps_only_verified_ticks(self):
        for bad_snapshot, code in ((snapshot(11, 0.7), "server-frame-not-increasing"),
                                   (snapshot(12, 0.55), "simulation-time-not-increasing")):
            with self.subTest(code=code):
                world = ReadOnlyWorld([snapshot(11, 0.55), bad_snapshot])
                carla, _ = fake_carla(world)
                report = connection.check_connection(connection.ConnectionConfig(), carla)
                self.assertEqual(report["status"], "FAIL")
                self.assertEqual(report["observed_ticks"], 1)
                self.assertEqual(report["error"]["code"], code)

    def test_missing_wait_snapshot_is_fail_with_bounded_wait(self):
        world = ReadOnlyWorld(ticks=[])
        carla, _ = fake_carla(world)
        report = connection.check_connection(connection.ConnectionConfig(timeout=0.2), carla)
        self.assertEqual(report["error"]["code"], "snapshot-missing")
        self.assertEqual(world.wait_timeouts, [0.2])
        self.assertEqual(world.mutations, [])

    def test_invalid_snapshot_fields_cannot_pass(self):
        for observed in (snapshot(True, 0.55), snapshot(11, float("nan")), snapshot(11, 0.55, 0), snapshot(11, 0.55, float("inf"))):
            with self.subTest(observed=observed):
                carla, _ = fake_carla(ReadOnlyWorld([observed]))
                report = connection.check_connection(connection.ConnectionConfig(), carla)
                self.assertEqual(report["status"], "FAIL")
                self.assertEqual(report["observed_ticks"], 0)

    def test_invalid_configuration_fails_before_runtime_or_network(self):
        for config in (connection.ConnectionConfig(ticks=0), connection.ConnectionConfig(ticks=21),
                       connection.ConnectionConfig(timeout=float("nan")), connection.ConnectionConfig(port=0)):
            with self.subTest(config=config):
                carla, _ = fake_carla(ReadOnlyWorld())
                report = connection.check_connection(config, carla)
                self.assertEqual(report["error"]["code"], "invalid-configuration")
                carla.Client.assert_not_called()
        self.runtime.assert_not_called()

    def test_main_writes_failed_report_and_returns_nonzero(self):
        self.package.side_effect = connection.metadata.PackageNotFoundError()
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            output = Path(folder) / "nested" / "connection.json"
            code = connection.main(["--output", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"]["code"], "carla-package-not-installed")

    def test_main_pending_status_is_nonzero(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(connection, "check_connection", return_value={"status": "PENDING"}), redirect_stdout(io.StringIO()):
            code = connection.main(["--output", str(Path(folder) / "connection.json")])
        self.assertEqual(code, 1)

    def test_nonfinite_cli_timeout_still_writes_standard_json_failure(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            output = Path(folder) / "connection.json"
            code = connection.main(["--timeout", "nan", "--output", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"), parse_constant=lambda value: self.fail(value))
        self.assertEqual(code, 1)
        self.assertEqual(report["error"]["code"], "invalid-configuration")
        self.assertIsNone(report["rpc_timeout_seconds"])

    def test_report_write_failure_cannot_return_success_or_export_error_text(self):
        with patch.object(connection, "check_connection", return_value={"status": "PASS"}), patch.object(Path, "write_text", side_effect=OSError("private-person/secret-token")), patch.object(Path, "mkdir"), redirect_stdout(io.StringIO()) as output:
            code = connection.main(["--output", "unused.json"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output.getvalue())["error"]["stage"], "report-write")
        self.assertNotIn("secret-token", output.getvalue())


if __name__ == "__main__":
    unittest.main()
