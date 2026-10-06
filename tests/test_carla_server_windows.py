"""Exercise Windows lifecycle refusal paths without starting a CARLA server."""

from __future__ import annotations

import json
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell.exe") if platform.system() == "Windows" else None


@unittest.skipUnless(POWERSHELL, "Windows PowerShell is required for lifecycle integration checks.")
class WindowsCarlaLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory(prefix="vlf-carlalifecycle-")
        self.addCleanup(self.fixture.cleanup)
        self.root = Path(self.fixture.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        self.script = scripts / "carla_server.ps1"
        shutil.copyfile(ROOT / "scripts" / "carla_server.ps1", self.script)
        config = self.root / "config"
        config.mkdir()
        shutil.copyfile(ROOT / "config" / "environment.json", config / "environment.json")
        self.package = self.root / "carla" / "CARLA_0.9.16"
        shipping = self.package / "CarlaUE4" / "Binaries" / "Win64"
        shipping.mkdir(parents=True)
        self.launcher = self.package / "CarlaUE4.exe"
        self.launcher.touch()
        (shipping / "CarlaUE4-Win64-Shipping.exe").touch()
        self.state_path = self.root / "results" / "local" / "carla_server_state.json"

    def run_lifecycle(self, action):
        return subprocess.run(
            [
                POWERSHELL,
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(self.script),
                "-Action",
                action,
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

    def write_state(self, process_number, creation_stamp):
        self.state_path.parent.mkdir(parents=True)
        state = {
            "schemaVersion": 1,
            "projectRoot": str(self.root),
            "packageRoot": str(self.package),
            "processes": [
                {
                    "processId": process_number,
                    "parentProcessId": None,
                    "executablePath": str(self.launcher),
                    "creationTimeUtc": creation_stamp,
                }
            ],
        }
        self.state_path.write_text(json.dumps(state), encoding="utf-8")

    def assert_message(self, expected, result):
        # Windows PowerShell wraps even a word at the host's error-stream width.
        message = result.stdout + result.stderr
        self.assertIn("".join(expected.split()), "".join(message.split()))

    def process_creation_stamp(self, process_number):
        # Only this test's own integer PID is interpolated into the CIM filter.
        command = (
            f"$ownedTestProcess = Get-CimInstance Win32_Process -Filter 'ProcessId={process_number}' -OperationTimeoutSec 3; "
            "if ($null -eq $ownedTestProcess) { exit 1 }; "
            "$ownedTestProcess.CreationDate.ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ss.fffZ')"
        )
        result = subprocess.run(
            [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        stamp = result.stdout.strip()
        self.assertRegex(stamp, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
        return stamp

    def test_start_refuses_occupied_port_before_launching_dummy_executable(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                listener.bind(("127.0.0.1", 2000))
            except OSError:
                self.skipTest("Port 2000 is already in use; this test will not disturb its owner.")
            listener.listen(1)
            result = self.run_lifecycle("Start")
        self.assertEqual(result.returncode, 1)
        self.assert_message("Start refused: port(s) 2000 are already occupied", result)
        self.assert_message("No unowned process will be stopped", result)
        self.assertFalse(self.state_path.exists())

    def test_stop_refuses_unowned_python_process_and_preserves_state(self):
        watcher = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            # Use its genuine creation time, so the executable path alone must
            # prevent a fake CARLA ownership record from terminating Python.
            stamp = self.process_creation_stamp(watcher.pid)
            self.write_state(watcher.pid, stamp)
            original_state = self.state_path.read_bytes()
            result = self.run_lifecycle("Stop")
            self.assertEqual(result.returncode, 1)
            self.assert_message("Stop refused: a recorded PID has a different path or creation time", result)
            self.assert_message("No processes were terminated", result)
            self.assertIsNone(watcher.poll(), "The unowned watcher was terminated.")
            self.assertEqual(self.state_path.read_bytes(), original_state)
        finally:
            # This Popen object owns the watcher handle; no PID search or broad
            # process-name termination is used to clean up the test fixture.
            if watcher.poll() is None:
                watcher.terminate()
                try:
                    watcher.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    watcher.kill()
                    watcher.wait(timeout=5)

    def test_stop_cleans_stale_nonexistent_pid_then_is_idempotent(self):
        self.write_state(2_147_483_647, "2000-01-01T00:00:00.000Z")
        result = self.run_lifecycle("Stop")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.state_path.exists())
        second = self.run_lifecycle("Stop")
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertIn("already stopped (no owned state)", second.stdout)


if __name__ == "__main__":
    unittest.main()
