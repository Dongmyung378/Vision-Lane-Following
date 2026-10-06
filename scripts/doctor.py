"""Audit the D1 workstation without installing packages or connecting to CARLA.

Uses only Python's standard library. Exported reports intentionally omit paths,
user/host names, environment variables, device identifiers, and command errors.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CARLA_VERSION = "0.9.16"
COMMAND_TIMEOUT_SECONDS = 8
GIB = 1024**3
WINDOWS_WHEEL = re.compile(
    r"^carla-(?P<version>\d+\.\d+\.\d+)-(?P<python>cp\d+)-"
    r"(?P<abi>cp\d+)-win_amd64\.whl$"
)


@dataclass(frozen=True)
class CommandResult:
    stdout: str = ""
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.reason is None


def run_command(arguments: Sequence[str]) -> CommandResult:
    """Bound every local probe and retain no raw diagnostics or command paths."""
    try:
        result = subprocess.run(
            list(arguments),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        return CommandResult(reason="not-found")
    except subprocess.TimeoutExpired:
        return CommandResult(reason="timeout")
    except OSError:
        return CommandResult(reason="unavailable")
    if result.returncode:
        return CommandResult(reason="failed")
    return CommandResult(stdout=result.stdout)


def positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return result if result > 0 else None


def safe_label(value: Any) -> str | None:
    """Hardware labels are useful; arbitrary paths or multiline text are not."""
    if not isinstance(value, str):
        return None
    label = value.strip()
    if not label or len(label) > 160 or any(ord(char) < 32 for char in label):
        return None
    if "\\" in label or "/" in label:
        return None
    return label


def probe_windows_hardware() -> dict[str, Any]:
    # Select only the fields needed for capacity planning; no serial/host data.
    command = (
        "$ErrorActionPreference = 'Stop'; "
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
        "$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1; "
        "$computer = Get-CimInstance Win32_ComputerSystem; "
        "$osInfo = Get-CimInstance Win32_OperatingSystem; "
        "[pscustomobject]@{cpu_name=$cpu.Name; "
        "physical_cores=$cpu.NumberOfCores; "
        "logical_processors=$cpu.NumberOfLogicalProcessors; "
        "memory_bytes=$computer.TotalPhysicalMemory; "
        "os_caption=$osInfo.Caption} | ConvertTo-Json -Compress"
    )
    result = run_command(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command]
    )
    if not result.ok:
        return {"probe_status": result.reason}
    try:
        values = json.loads(result.stdout.lstrip("\ufeff"))
        if not isinstance(values, dict):
            raise ValueError
    except (ValueError, TypeError):
        return {"probe_status": "invalid-response"}
    return {
        "probe_status": "ok",
        "cpu_name": safe_label(values.get("cpu_name")),
        "physical_cores": positive_int(values.get("physical_cores")),
        "logical_processors": positive_int(values.get("logical_processors")),
        "memory_bytes": positive_int(values.get("memory_bytes")),
        "os_caption": safe_label(values.get("os_caption")),
    }


def parse_gpu_csv(output: str) -> list[dict[str, Any]]:
    """nvidia-smi CSV supports quoted model names containing commas."""
    devices: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(output), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 3:
            raise ValueError("Unexpected GPU response shape.")
        name = safe_label(row[0])
        memory_mib = positive_int(row[1].strip())
        driver = row[2].strip()
        if not name or memory_mib is None or not re.fullmatch(r"\d+(?:\.\d+)+", driver):
            raise ValueError("Unexpected GPU response value.")
        devices.append(
            {"name": name, "memory_mib": memory_mib, "driver_version": driver}
        )
    return devices


def probe_gpus() -> dict[str, Any]:
    result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not result.ok:
        return {"probe_status": result.reason, "devices": []}
    try:
        devices = parse_gpu_csv(result.stdout)
    except (ValueError, csv.Error):
        return {"probe_status": "invalid-response", "devices": []}
    return {"probe_status": "ok" if devices else "empty-response", "devices": devices}


def probe_git() -> dict[str, Any]:
    result = run_command(["git", "--version"])
    if not result.ok:
        return {"probe_status": result.reason, "version": None}
    match = re.fullmatch(r"git version (\d+\.\d+(?:\.[0-9A-Za-z]+)*)\s*", result.stdout)
    return {
        "probe_status": "ok" if match else "invalid-response",
        "version": match.group(1) if match else None,
    }


def package_version(name: str) -> dict[str, Any]:
    # metadata.version does not import native extensions, initialize CUDA, or
    # establish a simulator connection. Version presence is not an import test.
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        return {"status": "not-installed", "version": None}
    except (OSError, ValueError):
        return {"status": "unknown", "version": None}
    if not isinstance(version, str) or not re.fullmatch(r"[0-9A-Za-z.+!_-]{1,80}", version):
        return {"status": "unknown", "version": None}
    return {"status": "installed-metadata", "version": version}


def probe_simulator(carla_root: Path | None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "candidate_version": CARLA_VERSION,
        "explicit_root_provided": carla_root is not None,
        "executable_found": False,
        "wheel_files_found": False,
        "windows_wheels": [],
        "compatible_windows_wheel_found": False,
        "server_version_verified": False,
    }
    if carla_root is None:
        report["probe_status"] = "deferred"
        return report
    try:
        report["executable_found"] = (carla_root / "CarlaUE4.exe").is_file()
        for wheel in sorted((carla_root / "PythonAPI" / "carla" / "dist").glob("*.whl")):
            if not wheel.is_file():
                continue
            report["wheel_files_found"] = True
            match = WINDOWS_WHEEL.fullmatch(wheel.name)
            if match:
                report["windows_wheels"].append(match.groupdict())
        python_tag = f"cp{sys.version_info[0]}{sys.version_info[1]}"
        report["compatible_windows_wheel_found"] = any(
            wheel["version"] == CARLA_VERSION
            and wheel["python"] == python_tag
            and wheel["abi"] == python_tag
            for wheel in report["windows_wheels"]
        )
        report["probe_status"] = "checked" if carla_root.is_dir() else "not-found"
    except OSError:
        report["probe_status"] = "unavailable"
    return report


def build_report(
    project_root: Path = PROJECT_ROOT, carla_root: Path | None = None
) -> dict[str, Any]:
    system = platform.system()
    hardware = probe_windows_hardware() if system == "Windows" else {}
    ram = hardware.get("memory_bytes")
    bits = struct.calcsize("P") * 8
    architecture = {
        "amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64",
        "aarch64": "arm64", "x86": "x86", "i386": "x86", "i686": "x86",
    }.get(platform.machine().lower(), "unknown")
    python_supported = (
        system in {"Windows", "Linux"}
        and platform.python_implementation() == "CPython"
        and (3, 10) <= sys.version_info[:2] <= (3, 12)
        and bits == 64
        and architecture == "x86_64"
    )
    try:
        usage = shutil.disk_usage(project_root)
        disk = {
            "probe_status": "ok",
            "total_gib": round(usage.total / GIB, 2),
            "free_gib": round(usage.free / GIB, 2),
        }
    except OSError:
        disk = {"probe_status": "unavailable", "total_gib": None, "free_gib": None}
    report: dict[str, Any] = {
        "schema_version": 1,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "architecture_bits": bits,
            "architecture": architecture,
            "carla_wheel_supported": python_supported,
            "virtual_environment": sys.prefix != sys.base_prefix,
        },
        "os": {
            "system": system,
            "release": platform.release(),
            "version": platform.version(),
            "caption": hardware.get("os_caption"),
        },
        "cpu": {
            "name": hardware.get("cpu_name"),
            "physical_cores": hardware.get("physical_cores"),
            "logical_processors": hardware.get("logical_processors") or os.cpu_count(),
            "probe_status": hardware.get("probe_status", "not-probed"),
        },
        "memory": {"total_gib": round(ram / GIB, 2) if ram else None},
        "gpus": probe_gpus(),
        "disk": disk,
        "git": probe_git(),
        "packages": {name: package_version(name) for name in ("carla", "torch")},
        "simulator": probe_simulator(carla_root),
    }
    checks: list[dict[str, str]] = []

    def check(name: str, status: str, detail: str) -> None:
        checks.append({"id": name, "status": status, "detail": detail})

    check("python", "pass" if python_supported else "fail", "CARLA 0.9.16 wheels target CPython 3.10-3.12, x86-64.")
    check("ram", "pass" if ram else "warn", "Physical memory measured." if ram else "Physical memory could not be measured; check manually.")
    devices = report["gpus"]["devices"]
    adequate_vram = any(device["memory_mib"] >= 8192 for device in devices)
    check("gpu", "pass" if adequate_vram else "warn", "Measured NVIDIA VRAM meets the 8 GiB recommendation; graphics performance remains untested." if adequate_vram else "8 GiB dedicated VRAM is recommended; missing or smaller VRAM requires a D2 runtime check.")
    enough_disk = disk["free_gib"] is not None and disk["free_gib"] >= 20
    check("disk", "pass" if enough_disk else "warn", "At least 20 GiB free on the project volume; allow extra space for datasets." if enough_disk else "Project-volume free space is unknown or below the approximate 20 GB simulator footprint.")
    check("git", "pass" if report["git"]["version"] else "warn", "Git version measured." if report["git"]["version"] else "Git could not be measured.")
    carla = report["packages"]["carla"]
    api_ready = carla["version"] == CARLA_VERSION
    api_status = "pass" if api_ready else "fail" if carla["version"] else "deferred"
    check("carla-api", api_status, "Matching CARLA API package metadata found; import and connection require the separate runtime probe." if api_ready else "Install or match CARLA 0.9.16 Python API.")
    executable_found = report["simulator"]["executable_found"]
    check("simulator", "pass" if executable_found else "deferred", "CarlaUE4.exe exists at the explicit package root; runtime/version verification requires the separate connection probe." if executable_found else "Provide an extracted Windows CARLA package root.")
    has_wheels = report["simulator"]["wheel_files_found"]
    matching_wheel = report["simulator"]["compatible_windows_wheel_found"]
    wheels_ready = not has_wheels or matching_wheel
    check("simulator-wheels", "pass" if matching_wheel else "fail" if has_wheels else "deferred", "A package wheel filename matches CARLA 0.9.16 and this CPython tag; runtime remains untested." if matching_wheel else "Package wheel files do not match this Windows Python/CARLA candidate." if has_wheels else "No package wheel files were verified; server version remains unknown.")
    check("torch", "info", "PyTorch metadata recorded if present; training/CUDA execution is not tested on D1.")
    report["checks"] = checks
    report["audit"] = {
        "d1_complete": True,
        "d2_preflight_ready": system == "Windows" and python_supported and api_ready and executable_found and wheels_ready,
        "d2_connection_verified": False,
        "measurement_limits": [
            "Unavailable hardware probes are recorded as unknown; no capacity is inferred.",
            "GPU performance, CARLA server/API compatibility, sensors, and CUDA execution are not verified by this static audit.",
            "Simulator version is a candidate; executable presence does not verify its version.",
        ],
    }
    return report


def write_report(report: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write sanitized JSON to this explicit file; parent directories are created.")
    parser.add_argument("--carla-root", type=Path, help="Inspect only this extracted Windows package root. No auto-discovery or server connection.")
    parser.add_argument("--strict", action="store_true", help="Return 1 unless D2 package preflight is ready; D2 runtime checks remain separate.")
    args = parser.parse_args(argv)
    report = build_report(carla_root=args.carla_root)
    if args.output:
        try:
            write_report(report, args.output)
        except (OSError, ValueError):
            print("ERROR: Could not write the report. Check the output location and permissions.", file=sys.stderr)
            return 2
    print("Environment audit complete. Missing probes and dependencies are recorded explicitly.")
    for item in report["checks"]:
        print(f"[{item['status'].upper()}] {item['id']}: {item['detail']}")
    print("D2 package preflight: " + ("ready" if report["audit"]["d2_preflight_ready"] else "pending"))
    print("CARLA server connection: not verified by this audit; run scripts/check_carla_connection.py.")
    if args.output:
        print("Sanitized JSON report written.")
    return 1 if args.strict and not report["audit"]["d2_preflight_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
