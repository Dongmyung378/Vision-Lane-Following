"""Run RGB baseline and privileged reference with one shared driving protocol."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import sys
import time
import zipfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.baseline import load_baseline_config
from carla_tools.baseline_driving import SCOPE, run_baseline_trial
from carla_tools.config import load_collection_config
from carla_tools.contracts import read_strict_json, strict_json_loads
from carla_tools.controller import load_controller_config
from carla_tools.driving import DrivingError
from carla_tools.evaluation import evaluate_episode, load_evaluation_config, write_metric_csv
from carla_tools.geometry_survey import load_geometry_catalog
from scripts.check_carla_connection import ConnectionConfig, check_connection
from scripts.verify_collection import write_report


def _attempt_metric(directory, method, route, evaluation_config):
    try:
        metadata = read_strict_json(directory / "episode.json")
    except Exception:
        metadata = {"episode_id": directory.name, "evaluation_route_length_m": None,
                    "evaluation_config": evaluation_config.to_dict(), "evaluation_config_sha256": evaluation_config.sha256}
    if not isinstance(metadata, dict):
        metadata = {"episode_id": directory.name, "evaluation_route_length_m": None}
    rows = []
    trace_failed = False
    try:
        for line in (directory / "steps.jsonl").read_text(encoding="utf-8").splitlines():
            rows.append(strict_json_loads(line))
    except Exception:
        rows.append(None)
        trace_failed = True
    metadata.update(method=method, route_id=route["route_id"])
    if trace_failed or metadata.get("status") != "COMPLETE":
        metadata["termination_reason"] = "technical_failure"
    return evaluate_episode(metadata, rows, metadata.get("events", []), config=evaluation_config)


def run_pipeline(output_dir):
    from carla_tools.baseline_validation import validate_baseline_manifest
    directory = Path(output_dir)
    started, owned = time.perf_counter(), False
    report = {"schema_version": 1, "status": "FAIL", "scope": SCOPE, "stage": "output-preflight",
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "cases": [], "episode_count": 0, "frame_count": 0}
    manifest = {"schema_version": 1, "scope": SCOPE, "episodes": []}
    metrics = []
    try:
        if directory.exists() or directory.is_symlink():
            raise DrivingError("baseline-pipeline-output-already-exists")
        directory.mkdir(parents=True, exist_ok=False)
        owned = True
        catalog = load_geometry_catalog(ROOT / "configs/routes_d6_geometry.json")
        collection = load_collection_config(ROOT / "configs/collection_d4.json")
        baseline, controller = load_baseline_config(), load_controller_config()
        evaluation = replace(load_evaluation_config(), timeout_seconds=10.0)
        report.update(baseline_config_sha256=baseline.sha256, controller_config_sha256=controller.sha256,
                      evaluation_config=evaluation.to_dict(), evaluation_config_sha256=evaluation.sha256,
                      protocol="same-original-route-center-state-controller-speed-clock-and-termination-for-both-methods")
        carla = importlib.import_module("carla")
        condition = {"name": "center", "lateral_m": 0.0, "yaw_deg": 0.0}
        for geometry in catalog["geometry_cases"]:
            for method in ("privileged_reference", "B0"):
                case, route = geometry["case"], geometry["route"]
                report["stage"] = f"native-{method}-{case}"
                print("Baseline evaluation:", report["stage"], flush=True)
                relative = f"episodes/{method}/{case}__center"
                result = run_baseline_trial(route, case, condition, directory / relative, collection, carla_module=carla,
                                            baseline_config=baseline, evaluation_config=evaluation, controller_config=controller,
                                            method=method)
                report["cases"].append({"method": method, **result})
                report["episode_count"] = len(report["cases"])
                report["frame_count"] += result.get("step_count", 0)
                manifest["episodes"].append({"method": method, "case": case, "condition": "center", "episode": relative})
                metrics.append(_attempt_metric(directory / relative, method, route, evaluation))
                write_report(directory / "baseline_manifest.json", manifest)
                write_report(directory / "baseline_report.json", report)
                if result["status"] != "PASS":
                    raise DrivingError("baseline-native-trial-operation-failed")
        report["stage"] = "independent-rgb-controller-and-metric-gate"
        report["gate"] = validate_baseline_manifest(directory / "baseline_manifest.json")
        if report["gate"]["status"] != "PASS":
            raise DrivingError("baseline-independent-artifact-gate-failed")
        metrics = report["gate"]["episode_metrics"]
        report["aggregate"] = write_metric_csv(metrics, directory / "metrics.csv", config=evaluation)
        report["metrics_csv_sha256"] = hashlib.sha256((directory / "metrics.csv").read_bytes()).hexdigest()
        write_report(directory / "metrics.json", {"schema_version": 1, "episodes": metrics, "aggregate": report["aggregate"]})
        report["gate"] = validate_baseline_manifest(directory / "baseline_manifest.json")
        if report["gate"]["status"] != "PASS":
            raise DrivingError("baseline-exported-metrics-gate-failed")
        report["connection_after"] = check_connection(ConnectionConfig(timeout=30, ticks=3), carla_module=carla)
        if report["connection_after"]["status"] != "PASS":
            raise DrivingError("baseline-post-driving-RPC-failed")
        report.update(status="PASS", stage="verified")
    except BaseException as error:
        report["error"] = {"code": getattr(error, "code", "baseline-pipeline-operation-failed"),
                           "exception_type": type(error).__name__, "stage": report["stage"]}
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        if owned:
            if metrics and not (directory / "metrics.csv").exists():
                try:
                    report["aggregate"] = write_metric_csv(metrics, directory / "metrics.csv")
                    write_report(directory / "metrics.json", {"schema_version": 1, "episodes": metrics, "aggregate": report["aggregate"]})
                except Exception as error:
                    report["metric_publication_error"] = type(error).__name__
            write_report(directory / "baseline_report.json", report)
            try:
                with zipfile.ZipFile(directory.parent / "baseline_artifacts.zip", "x", compression=zipfile.ZIP_DEFLATED,
                                     compresslevel=1) as archive:
                    for path in sorted(directory.rglob("*")):
                        if path.is_file() and not path.is_symlink():
                            archive.write(path, Path("baseline") / path.relative_to(directory))
            except Exception as error:
                report.update(status="FAIL", packaging_error=type(error).__name__)
                write_report(directory / "baseline_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_pipeline(args.output_dir)
    print(json.dumps({key: report[key] for key in ("status", "stage", "episode_count", "frame_count", "elapsed_seconds", "error")
                      if key in report}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
