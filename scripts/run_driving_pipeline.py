"""D8-D10 privileged reference/recovery on an already running owned CARLA server."""
from __future__ import annotations
import argparse
import datetime as dt
import importlib
import json
import sys
import time
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from carla_tools.config import load_collection_config
from carla_tools.controller import load_controller_config
from carla_tools.driving import DrivingError, load_validation_config, run_drive_trial
from carla_tools.driving_validation import validate_drive_episode, validate_driving_manifest
from carla_tools.geometry_survey import load_geometry_catalog
from scripts.carla_rgb_probe import gpu_sample
from scripts.check_carla_connection import ConnectionConfig, check_connection
from scripts.verify_collection import write_report

def run_pipeline(output_dir):
    directory = Path(output_dir)
    report = {"schema_version": 1, "status": "FAIL", "scope": "D8-D10-privileged-controller-development-QA",
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "cases": [], "stage": "output-preflight"}
    started, owned = time.perf_counter(), False
    try:
        if directory.exists() or directory.is_symlink():
            raise DrivingError("driving-pipeline-output-already-exists")
        directory.mkdir(parents=True)
        owned = True
        catalog = load_geometry_catalog(ROOT / "configs/routes_d6_geometry.json")
        collection = load_collection_config(ROOT / "configs/collection_d4.json")
        controller, limits = load_controller_config(), load_validation_config()
        carla = importlib.import_module("carla")
        report["gpu_before"] = gpu_sample()
        manifest = {"schema_version": 1, "episodes": []}
        for geometry in catalog["geometry_cases"]:
            for condition in limits["initial_conditions"]:
                case = geometry["case"]
                report["stage"] = f"native-{case}-{condition['name']}"
                print("D8/D9/D10:", report["stage"], flush=True)
                relative = f"episodes/{case}__{condition['name']}"
                result = run_drive_trial(geometry["route"], case, condition, directory / relative, collection,
                                         carla_module=carla, validation_config=limits, controller_config=controller)
                report["cases"].append(result)
                if result["status"] != "PASS":
                    raise DrivingError("native-driving-trial-failed")
                checked = validate_drive_episode(directory / relative)
                result["validation"] = checked
                if checked["status"] != "PASS":
                    raise DrivingError("native-driving-disk-gate-failed")
                manifest["episodes"].append({"case": case, "condition": condition["name"], "episode": relative})
                write_report(directory / "driving_report.json", report)
        write_report(directory / "driving_manifest.json", manifest)
        report["stage"] = "independent-full-driving-recovery-gate"
        report["gate"] = validate_driving_manifest(directory / "driving_manifest.json")
        if report["gate"]["status"] != "PASS":
            raise DrivingError("driving-manifest-gate-failed")
        report["episode_count"] = 15
        report["frame_count"] = sum(item["step_count"] for item in report["cases"])
        report["connection_after"] = check_connection(ConnectionConfig(timeout=30, ticks=3), carla_module=carla)
        if report["connection_after"]["status"] != "PASS":
            raise DrivingError("post-driving-RPC-failed")
        report["gpu_after"] = gpu_sample()
        report.update(status="PASS", stage="verified")
    except BaseException as error:
        report["error"] = {"code": getattr(error, "code", "driving-pipeline-operation-failed"),
                           "exception_type": type(error).__name__, "stage": report["stage"]}
    report["elapsed_seconds"] = time.perf_counter() - started
    report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    if owned:
        write_report(directory / "driving_report.json", report)
        try:
            # One bounded archive avoids hundreds of per-file HTTP transfers.
            with zipfile.ZipFile(directory.parent / "driving_artifacts.zip", "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                for path in sorted(directory.rglob("*")):
                    if path.is_file() and not path.is_symlink():
                        archive.write(path, Path("driving") / path.relative_to(directory))
        except Exception as error:
            report.update(status="FAIL", error={"code": "driving-artifact-packaging-failed", "exception_type": type(error).__name__})
            write_report(directory / "driving_report.json", report)
    return report

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_pipeline(args.output_dir)
    print(json.dumps({key: report[key] for key in ("status", "stage", "episode_count", "frame_count", "elapsed_seconds", "error") if key in report}, indent=2))
    return 0 if report["status"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
