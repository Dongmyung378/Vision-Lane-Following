"""Verify D2's pinned CARLA connection without changing the running simulator.

The check reads the active world and observes asynchronous server ticks. It
never loads a map, advances a synchronous world, changes settings, or creates
actors. CARLA is imported only when the check runs; unit tests need no CARLA.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import platform
import re
import struct
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any


CARLA_VERSION = "0.9.16"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "local" / "carla_connection.json"


class ConnectionFailure(RuntimeError):
    """A fixed public error code; raw runtime errors are never exported."""


@dataclass(frozen=True)
class ConnectionConfig:
    host: str = "127.0.0.1"
    port: int = 2000
    timeout: float = 15.0
    ticks: int = 3


def runtime_info() -> dict[str, Any]:
    machine = platform.machine().lower()
    return {
        "implementation": platform.python_implementation(),
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "python_major": sys.version_info.major,
        "python_minor": sys.version_info.minor,
        "architecture": "x86_64" if machine in ("amd64", "x86_64", "x64") else "unsupported",
        "pointer_bits": struct.calcsize("P") * 8,
    }


def safe_version(value: Any) -> str:
    if isinstance(value, str) and len(value) <= 32 and re.fullmatch(r"\d+\.\d+\.\d+", value):
        return value
    return "invalid-response"


def snapshot_record(snapshot: Any, *, require_positive_delta: bool = False) -> dict[str, Any]:
    if snapshot is None:
        raise ConnectionFailure("snapshot-missing")
    frame = snapshot.frame
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ConnectionFailure("invalid-snapshot-frame")
    elapsed = snapshot.timestamp.elapsed_seconds
    delta = snapshot.timestamp.delta_seconds
    for value in (elapsed, delta):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ConnectionFailure("invalid-snapshot-timestamp")
    if require_positive_delta and delta <= 0:
        raise ConnectionFailure("nonpositive-simulation-step")
    return {"frame": frame, "elapsed_seconds": float(elapsed), "delta_seconds": float(delta)}


def check_connection(config: ConnectionConfig, carla_module: Any = None) -> dict[str, Any]:
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 1, "status": "FAIL", "scope": "carla-read-only-connection",
        "expected_version": CARLA_VERSION,
        "ticks_requested": config.ticks if isinstance(config.ticks, int) and not isinstance(config.ticks, bool) else None,
        "rpc_timeout_seconds": config.timeout if isinstance(config.timeout, (int, float)) and not isinstance(config.timeout, bool) and math.isfinite(config.timeout) else None,
        "ticks": [], "observed_ticks": 0,
    }
    stage = "configuration"
    try:
        if (not isinstance(config.host, str) or not config.host.strip()
                or isinstance(config.port, bool) or not isinstance(config.port, int) or not 1 <= config.port <= 65535
                or isinstance(config.ticks, bool) or not isinstance(config.ticks, int) or not 1 <= config.ticks <= 20
                or isinstance(config.timeout, bool) or not isinstance(config.timeout, (int, float))
                or not math.isfinite(config.timeout) or not 0 < config.timeout <= 120):
            raise ConnectionFailure("invalid-configuration")
        stage = "python-runtime"
        runtime = runtime_info()
        report["runtime"] = runtime
        if (runtime["implementation"] != "CPython" or runtime["python_major"] != 3
                or not 10 <= runtime["python_minor"] <= 12
                or runtime["architecture"] != "x86_64" or runtime["pointer_bits"] != 64):
            raise ConnectionFailure("requires-cpython310-through312-x64")
        stage = "package-version"
        try:
            package_version = metadata.version("carla")
        except metadata.PackageNotFoundError:
            raise ConnectionFailure("carla-package-not-installed") from None
        report["package_version"] = safe_version(package_version)
        if package_version != CARLA_VERSION:
            raise ConnectionFailure("installed-carla-version-mismatch")
        stage = "carla-import"
        carla = carla_module if carla_module is not None else importlib.import_module("carla")
        stage = "client-creation"
        client = carla.Client(config.host, config.port)
        client.set_timeout(config.timeout)
        stage = "client-version"
        client_version = client.get_client_version()
        report["client_version"] = safe_version(client_version)
        stage = "server-version"
        server_version = client.get_server_version()
        report["server_version"] = safe_version(server_version)
        stage = "version-check"
        if client_version != CARLA_VERSION or server_version != CARLA_VERSION or client_version != package_version:
            raise ConnectionFailure("carla-package-api-server-version-mismatch")
        report["versions_verified"] = True
        stage = "world-read"
        world = client.get_world()
        if world is None:
            raise ConnectionFailure("world-missing")
        stage = "map-read"
        map_name = world.get_map().name
        if not isinstance(map_name, str) or not map_name.strip():
            raise ConnectionFailure("invalid-map-name")
        name = map_name.replace("\\", "/").rsplit("/", 1)[-1]
        # Preserve standard CARLA map names, not arbitrary private custom paths.
        report["map_name"] = name if re.fullmatch(r"Town\d{2}(?:HD)?(?:_Opt)?", name) else "custom-map"
        report["map_read_verified"] = True
        stage = "snapshot-read"
        initial = snapshot_record(world.get_snapshot())
        report["initial_snapshot"] = initial
        stage = "world-settings-read"
        settings = world.get_settings()
        fixed_delta = settings.fixed_delta_seconds
        if fixed_delta is not None and (isinstance(fixed_delta, bool) or not isinstance(fixed_delta, (int, float))
                                        or not math.isfinite(fixed_delta) or fixed_delta < 0):
            raise ConnectionFailure("invalid-world-timestep")
        if not isinstance(settings.synchronous_mode, bool) or not isinstance(settings.no_rendering_mode, bool):
            raise ConnectionFailure("invalid-world-settings")
        report["world_settings"] = {
            "synchronous_mode": settings.synchronous_mode,
            "no_rendering_mode": settings.no_rendering_mode,
            "fixed_delta_seconds": fixed_delta,
        }
        if settings.synchronous_mode:
            report["status"] = "PENDING"
            report["error"] = {"stage": "tick-observation", "code": "external-synchronous-world", "exception_type": None}
            report["next_action"] = "Run this read-only check against an asynchronous server; preserve the current world's owner and settings."
            return report
        stage = "tick-observation"
        previous = initial
        for _ in range(config.ticks):
            observed = snapshot_record(world.wait_for_tick(config.timeout), require_positive_delta=True)
            if observed["frame"] <= previous["frame"]:
                raise ConnectionFailure("server-frame-not-increasing")
            if observed["elapsed_seconds"] <= previous["elapsed_seconds"]:
                raise ConnectionFailure("simulation-time-not-increasing")
            report["ticks"].append(observed)
            report["observed_ticks"] = len(report["ticks"])
            previous = observed
        report["observed_simulation_seconds"] = previous["elapsed_seconds"] - initial["elapsed_seconds"]
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = {
            "stage": stage, "code": str(error) if isinstance(error, ConnectionFailure) else "operation-failed",
            "exception_type": type(error).__name__,
        }
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
    return report


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--ticks", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(arguments)
    report = check_connection(ConnectionConfig(args.host, args.port, args.timeout, args.ticks))
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError:
        report["status"] = "FAIL"
        report["error"] = {"stage": "report-write", "code": "report-write-failed", "exception_type": "OSError"}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
