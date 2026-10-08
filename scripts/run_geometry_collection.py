"""Collect three source-pinned stationary geometry QA episodes on CARLA.

The server must already be running. Importing this harness does not import
CARLA, launch processes, or create files. Raw captures remain unlabelled D4
episodes; D6 labels and D7 overlays are separate subsequent artifacts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.config import ConfigError, collection_config_from_dict, load_collection_config
from carla_tools.contracts import ContractError, validate_identifier
from carla_tools.geometry_survey import CASES, load_geometry_catalog, verify_geometry_route_in_world
from carla_tools.routes import RouteError


REPORT_NAME = "geometry_collection_report.json"


class GeometryCollectionError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _publish_json(path: Path, value: dict) -> None:
    """Publish complete finite JSON exclusively, preserving previous results."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pending", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _plans(catalog_path: str | Path, config_path: str | Path, frames: int, dataset_version: str) -> list[dict]:
    validate_identifier(dataset_version)
    if type(frames) is not int or not 1 <= frames <= 1000:
        raise GeometryCollectionError("invalid-capture-frame-count")
    catalog = load_geometry_catalog(catalog_path)
    base = load_collection_config(config_path)
    plans = []
    for item in catalog["geometry_cases"]:
        route = item["route"]
        values = base.to_dict()
        values["vehicle"]["spawn_point_index"] = route["spawn_point_index"]
        values["verification"]["frames"] = frames
        cfg = collection_config_from_dict(values)
        if route["town"] != cfg.map or route["carla_version"] != cfg.carla_version:
            raise GeometryCollectionError("geometry-configuration-identity-mismatch")
        episode_id = f"geometry_{item['case']}_01"
        plans.append({"case": item["case"], "episode_id": episode_id, "config": cfg, "route": route,
                      "metadata": {"dataset_version": dataset_version, "episode_id": episode_id,
                                   "route_id": route["route_id"], "scene_group_id": route["scene_group_id"],
                                   "town": cfg.map, "route_contract": route,
                                   "perturbation": {"lateral_m": 0.0, "yaw_deg": 0.0,
                                                    "provenance": "stationary-no-perturbation"},
                                   "control_mode": "stationary_D4_smoke", "labels_state": "not-generated-D6",
                                   "geometry_qa": {"case": item["case"], "evidence": item["geometry"],
                                                   "scope": "D6-static-geometry-QA-not-training-split"}}})
    if [plan["case"] for plan in plans] != list(CASES):
        raise GeometryCollectionError("geometry-cases-incomplete")
    return plans


def collect_geometry_cases(catalog_path: str | Path, output_root: str | Path, config_path: str | Path, *,
                           carla_module: Any = None, capture_function: Any = None,
                           frames: int = 60, dataset_version: str = "d6-geometry-v1") -> dict:
    """PASS only after all three live-route checks and complete captures pass.

    Existing managed episode/report paths are refused before connecting. The
    function never starts or stops a server; capture_episode owns its actors and
    restores the asynchronous world before the next case starts.
    """
    report: dict[str, Any] = {"schema_version": 1, "status": "FAIL", "scope": "D6-static-geometry-QA-not-training-split",
                              "dataset_version": dataset_version, "frames_per_case": frames, "cases": [],
                              "labels_generated": False, "training_split_created": False,
                              "server_lifecycle_owned": False, "report_written": False,
                              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    stage, report_owned, current = "configuration", False, None
    started = time.perf_counter()
    output_root = Path(output_root)
    try:
        plans = _plans(catalog_path, config_path, frames, dataset_version)
        report["cases"] = [{"case": plan["case"], "episode_id": plan["episode_id"], "status": "NOT_RUN",
                            "episode_directory": plan["episode_id"], "route_sha256": plan["route"]["route_sha256"],
                            "configuration_sha256": plan["config"].sha256,
                            "capture_report": f"{plan['episode_id']}_capture_report.json"} for plan in plans]
        stage = "output-preflight"
        if output_root.is_symlink() or getattr(output_root, "is_junction", lambda: False)():
            raise GeometryCollectionError("output-root-link-refused")
        paths = [output_root / REPORT_NAME]
        for plan in plans:
            paths.extend([output_root / plan["episode_id"], output_root / f"{plan['episode_id']}_capture_report.json"])
        if any(path.exists() or path.is_symlink() for path in paths):
            raise GeometryCollectionError("geometry-output-already-exists")
        output_root.mkdir(parents=True, exist_ok=True)
        report_owned = True
        stage = "carla-import"
        carla = carla_module if carla_module is not None else importlib.import_module("carla")
        if capture_function is None:
            from carla_tools.capture import capture_episode
            capture_function = capture_episode
        client = carla.Client(plans[0]["config"].connection.host, plans[0]["config"].connection.port)
        client.set_timeout(plans[0]["config"].connection.timeout_seconds)
        for plan, current in zip(plans, report["cases"]):
            stage = "live-route-validation"
            cfg = plan["config"]
            current["status"] = "FAIL"
            if client.get_client_version() != cfg.carla_version or client.get_server_version() != cfg.carla_version:
                raise GeometryCollectionError("carla-version-mismatch")
            verify_geometry_route_in_world(plan["route"], client.get_world().get_map(), carla)
            current["live_route_verified"] = True
            stage = "episode-capture"
            captured = capture_function(cfg, output_root / plan["episode_id"], plan["metadata"], carla_module=carla)
            if type(captured) is not dict:
                raise GeometryCollectionError("capture-report-invalid")
            _publish_json(output_root / current["capture_report"], captured)
            current["frames_written"] = captured.get("frames_written", 0)
            current["cleanup_status"] = captured.get("cleanup", {}).get("status")
            current["disk_validation_status"] = captured.get("validation", {}).get("status")
            if (captured.get("status") != "PASS" or type(captured.get("frames_written")) is not int or
                    captured.get("frames_written") != frames or
                    current["cleanup_status"] != "PASS" or current["disk_validation_status"] != "PASS"):
                current["error"] = captured.get("error", {"code": "capture-or-cleanup-or-validation-incomplete"})
                raise GeometryCollectionError("geometry-case-capture-failed")
            current["status"] = "PASS"
        if len(report["cases"]) != 3 or not all(case["status"] == "PASS" for case in report["cases"]):
            raise GeometryCollectionError("geometry-cases-incomplete")
        report["status"] = "PASS"
    except BaseException as error:
        code = error.code if isinstance(error, (GeometryCollectionError, RouteError, ContractError)) else (
            "invalid-configuration" if isinstance(error, ConfigError) else
            "geometry-collection-interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "geometry-collection-operation-failed")
        report["error"] = {"stage": stage, "code": code, "exception_type": type(error).__name__}
        if current is not None and current["status"] != "PASS":
            current["status"] = "FAIL"
    report["elapsed_seconds"] = time.perf_counter() - started
    report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    if report_owned:
        try:
            report["report_written"] = True
            _publish_json(output_root / REPORT_NAME, report)
        except Exception as error:
            report["status"] = "FAIL"
            report["report_written"] = False
            report["report_error"] = {"code": "geometry-report-write-failed", "exception_type": type(error).__name__}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "collection_d4.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "local" / "dataset-d6-geometry-v1")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--dataset-version", default="d6-geometry-v1")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        try:
            plans = _plans(args.catalog, args.config, args.frames, args.dataset_version)
            print(json.dumps({"status": "CONFIG_VALIDATED", "simulator_verified": False,
                              "cases": [plan["case"] for plan in plans], "frames_per_case": args.frames}))
            return 0
        except (ConfigError, ContractError, RouteError, GeometryCollectionError):
            print(json.dumps({"status": "FAIL", "code": "geometry-configuration-invalid"}))
            return 1
    report = collect_geometry_cases(args.catalog, args.output_root, args.config,
                                    frames=args.frames, dataset_version=args.dataset_version)
    print(json.dumps({"status": report["status"], "cases": report["cases"],
                      "report_written": report["report_written"], "error": report.get("error")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
