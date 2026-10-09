"""Collect a moving route-group pilot and recheck native development baselines."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.baseline_driving import run_baseline_trial
from carla_tools.config import load_collection_config
from carla_tools.pilot import (SCOPE, PilotError, build_pilot_manifest, load_pilot_config,
                              survey_pilot_routes, validate_pilot_manifest)
from scripts.check_carla_connection import ConnectionConfig, check_connection
from scripts.run_baseline_pipeline import run_pipeline as run_baseline_pipeline
from scripts.verify_collection import write_report


def run_pipeline(output_dir, *, config_path=ROOT / "configs/pilot.json"):
    directory, started, owned = Path(output_dir), time.perf_counter(), False
    report = {"schema_version": 1, "status": "FAIL", "scope": SCOPE, "stage": "output-preflight",
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "episodes": [], "native_frame_count": 0,
              "sample_count": 0, "reserved_final_town": "Town02", "reserved_final_state": "UNOPENED"}
    try:
        if directory.exists() or directory.is_symlink():
            raise PilotError("pilot-output-already-exists")
        directory.mkdir(parents=True, exist_ok=False)
        owned = True
        config, collection = load_pilot_config(config_path), load_collection_config(ROOT / "configs/collection_d4.json")
        report["config_sha256"] = config.sha256
        carla = importlib.import_module("carla")
        report["stage"] = "native-topology-and-scene-split-survey"
        client = carla.Client(collection.connection.host, collection.connection.port)
        client.set_timeout(collection.connection.timeout_seconds)
        if client.get_client_version() != "0.9.16" or client.get_server_version() != "0.9.16":
            raise PilotError("pilot-native-version-mismatch")
        catalog = survey_pilot_routes(client.get_world().get_map(), carla, config)
        write_report(directory / "pilot_route_catalog.json", catalog)
        report["route_count"] = len(catalog["routes"])
        native_started, descriptors = time.perf_counter(), []
        for item in catalog["routes"]:
            route, case, split = item["route"], item["case"], item["split"]
            for condition in config.conditions:
                relative = f"episodes/{split}/{route['route_id']}__{condition['name']}"
                report["stage"] = f"native-{split}-{case}-{condition['name']}"
                print("Moving pilot:", report["stage"], "spawn", route["spawn_point_index"], flush=True)
                result = run_baseline_trial(route, case, condition, directory / relative, collection,
                                            carla_module=carla, method="privileged_reference")
                report["episodes"].append({"split": split, "route_id": route["route_id"], **result})
                report["native_frame_count"] += result.get("step_count", 0)
                write_report(directory / "pilot_report.json", report)
                if result["status"] != "PASS":
                    raise PilotError("pilot-native-trial-operation-failed")
                metadata = json.loads((directory / relative / "episode.json").read_text(encoding="utf-8"))
                if metadata["evaluation_metrics"]["driving_success"] is not True:
                    raise PilotError("pilot-reference-driving-failed")
                descriptors.append({"split": split, "case": case, "condition": condition["name"], "episode": relative})
        report["native_collection_elapsed_seconds"] = time.perf_counter() - native_started
        report["stage"] = "immutable-moving-sample-manifest"
        manifest = build_pilot_manifest(directory, catalog["routes"], descriptors, config)
        report["stage"] = "independent-native-and-split-gate"
        report["gate"] = validate_pilot_manifest(manifest)
        if report["gate"]["status"] != "PASS":
            raise PilotError("pilot-source-or-split-gate-failed")
        report.update(sample_count=report["gate"]["sample_count"], split_counts=report["gate"]["split_counts"],
                      manifest_sha256=report["gate"]["manifest_sha256"],
                      moving_frames_per_collection_second=report["gate"]["sample_count"] / report["native_collection_elapsed_seconds"])
        report["stage"] = "development-baseline-and-privileged-reference"
        report["development_baseline"] = run_baseline_pipeline(directory / "development_baseline")
        if report["development_baseline"]["status"] != "PASS":
            raise PilotError("pilot-development-baseline-runtime-or-source-gate-failed")
        reference_metrics = [metric for metric in report["development_baseline"]["gate"]["episode_metrics"]
                             if metric["method"] == "privileged_reference"]
        if len(reference_metrics) != 3 or not all(metric["driving_success"] is True for metric in reference_metrics):
            raise PilotError("pilot-development-reference-driving-failed")
        report["stage"] = "post-collection-connection"
        report["connection_after"] = check_connection(ConnectionConfig(timeout=30, ticks=3), carla_module=carla)
        if report["connection_after"]["status"] != "PASS":
            raise PilotError("pilot-post-collection-RPC-failed")
        files = [path for path in directory.rglob("*") if path.is_file() and not path.is_symlink() and path.suffix != ".zip"]
        report["artifact_uncompressed_bytes"] = sum(path.stat().st_size for path in files)
        report["bytes_per_moving_sample"] = report["artifact_uncompressed_bytes"] / report["sample_count"]
        report.update(status="PASS", stage="verified")
    except BaseException as error:
        report["error"] = {"code": getattr(error, "code", "pilot-pipeline-operation-failed"),
                           "exception_type": type(error).__name__, "stage": report["stage"]}
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        if owned:
            write_report(directory / "pilot_report.json", report)
            try:
                archive_path = directory.parent / "pilot_artifacts.zip"
                with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                    for path in sorted(directory.rglob("*")):
                        if path.is_file() and not path.is_symlink() and path.suffix != ".zip":
                            archive.write(path, Path("pilot") / path.relative_to(directory))
                report["archive_bytes"] = archive_path.stat().st_size
                report["archive_sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
                # Archive publication metadata stays outside the archive to
                # avoid a circular archive hash. Sources inside remain stable.
                write_report(directory.parent / "pilot_archive_report.json", {"status": report["status"],
                             "archive_sha256": report["archive_sha256"], "archive_bytes": report["archive_bytes"],
                             "manifest_sha256": report.get("manifest_sha256")})
            except Exception as error:
                report.update(status="FAIL", packaging_error=type(error).__name__)
                write_report(directory / "pilot_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pilot.json")
    args = parser.parse_args(argv)
    report = run_pipeline(args.output_dir, config_path=args.config)
    print(json.dumps({key: report[key] for key in ("status", "stage", "sample_count", "split_counts", "native_frame_count",
                                                "elapsed_seconds", "archive_bytes", "archive_sha256", "error") if key in report}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
