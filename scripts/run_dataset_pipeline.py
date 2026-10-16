"""Collect a frozen, scene-separated central/recovery RGB waypoint dataset.

Native images remain in the caller's scratch directory. Only the archive and
small evidence views are published to the artifact directory. No CARLA server
or GPU is started by importing this module or by the dry-run command.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import datetime as dt
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.baseline import load_baseline_config
from carla_tools.baseline_driving import PrivilegedEvaluationPolicy
from carla_tools.baseline_validation import _validate_camera_episode
from carla_tools.config import collection_config_from_dict, load_collection_config
from carla_tools.contracts import canonical_sha256, read_strict_json
from carla_tools.driving import load_validation_config, route_initial_location, run_drive_trial
from carla_tools.evaluation import load_evaluation_config
from carla_tools.pilot import survey_pilot_routes
from carla_tools.session import fields_match, public_fields
from scripts.check_carla_connection import ConnectionConfig, check_connection
from scripts.verify_collection import write_report

SCOPE = "scene-separated-central-recovery-rgb-dataset"
DEVELOPMENT_WEATHERS = ("ClearNoon", "CloudyNoon", "WetNoon", "SoftRainNoon")
REQUESTED_YAW_SERIALIZATION_TOLERANCE_DEG = 1e-4


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collection_matrix(routes, config):
    """One central and four recovery trials per physical/weather/start group."""
    result = []
    for item in routes:
        route = item["route"]
        for weather in config.weather_presets:
            for run in config.collection_runs:
                pair = f"{route['route_id']}__{weather}__{run['name']}"
                for condition in config.conditions:
                    name = pair + "__" + condition["name"]
                    result.append({"episode": f"episodes/{item['split']}/{name}",
                        "split": item["split"], "case": item["case"], "route_id": route["route_id"],
                        "condition": dict(condition), "role": "central" if condition["name"] == "center" else "recovery",
                        "pair_group_id": pair, "weather": weather, "run_name": run["name"],
                        "initial_reference_m": run["initial_reference_m"], "seed": run["seed"]})
    if not result or len({item["episode"] for item in result}) != len(result):
        raise ValueError("dataset-matrix-empty-or-duplicate")
    expected = sum(config.train_routes_per_case.values()) + sum(config.dev_routes_per_case.values())
    if len(routes) != expected or len(result) != expected * len(config.weather_presets) * len(config.collection_runs) * len(config.conditions):
        raise ValueError("dataset-matrix-route-count-mismatch")
    return result


def episode_descriptor(trial):
    return {key: (trial[key]["name"] if key == "condition" else trial[key]) for key in
            ("episode", "split", "case", "condition", "role", "pair_group_id", "weather", "run_name", "initial_reference_m")}


class DatasetReferencePolicy(PrivilegedEvaluationPolicy):
    """Preserve the original privileged controller while varying the start."""

    def __init__(self, route, condition, *, initial_reference_m=0.):
        super().__init__(load_baseline_config(), replace(load_evaluation_config(), timeout_seconds=10.), route, condition)
        if type(initial_reference_m) not in (int, float) or not math.isfinite(initial_reference_m) or not 0 <= initial_reference_m <= 4:
            raise ValueError("dataset-initial-reference-invalid")
        self.initial_reference_m = float(initial_reference_m)


def run_dataset_v1_trial(route, case, condition, output_dir, collection_config, *, carla_module,
                         initial_reference_m=0., session_factory=None):
    return run_drive_trial(route, case, condition, output_dir, collection_config, carla_module=carla_module,
        trial_policy=DatasetReferencePolicy(route, condition, initial_reference_m=initial_reference_m),
        allowed_weather_presets=DEVELOPMENT_WEATHERS, session_factory=session_factory)


def validate_dataset_v1_episode(path, *, allowed_conditions, expected_initial_reference_m, expected_weather,
                                 carla_module=None):
    """Replay every frame and bind weather to native CARLA preset definitions.

    Importing the CARLA client to inspect WeatherParameters performs no RPC,
    rendering or GPU operation. Preset names alone cannot certify readbacks.
    """
    gate = _validate_camera_episode(path, allowed_conditions=[dict(value) for value in allowed_conditions],
        expected_initial_reference_m=expected_initial_reference_m, allowed_weather_presets=DEVELOPMENT_WEATHERS,
        initial_heading_reference="declared-route-start")
    if gate["status"] != "PASS":
        return gate
    try:
        if expected_weather not in DEVELOPMENT_WEATHERS:
            raise ValueError("dataset-weather-not-development-preset")
        metadata_path = Path(path) / "episode.json"
        before = metadata_path.read_bytes()
        metadata = read_strict_json(metadata_path)
        initialization = metadata.get("initialization")
        nominal_yaw = route_initial_location(metadata["route_contract"], expected_initial_reference_m)[1]
        if type(initialization) is not dict:
            raise ValueError("dataset-requested-heading-record-missing")
        # These recorded requests are audited independently; they never
        # supply the expected heading for the measured first-pose check.
        # CARLA serializes rotation through float32, so retain a small
        # serialization allowance while leaving the measured .5 deg bound.
        for name, expected_yaw in (("center_requested_transform", nominal_yaw),
                                   ("perturbed_requested_transform", nominal_yaw + metadata["condition"]["yaw_deg"])):
            transform = initialization.get(name)
            rotation = transform.get("rotation") if type(transform) is dict else None
            yaw = rotation.get("yaw") if type(rotation) is dict else None
            if (type(yaw) not in (int, float) or not math.isfinite(yaw)
                    or abs((yaw - expected_yaw + 180.) % 360. - 180.) > REQUESTED_YAW_SERIALIZATION_TOLERANCE_DEG):
                raise ValueError("dataset-requested-heading-source-binding-mismatch")
        carla = carla_module or importlib.import_module("carla")
        expected = public_fields(getattr(carla.WeatherParameters, expected_weather))
        observed = metadata["applied_manifest"]["weather"]
        collection = collection_config_from_dict(metadata["collection_config"], allowed_weather_presets=DEVELOPMENT_WEATHERS)
        if (metadata["collection_config"]["weather"]["preset"] != expected_weather
                or observed["preset"] != expected_weather or not expected
                or metadata["applied_manifest"].get("config_sha256") != collection.sha256
                or not fields_match(expected, observed["values"], exact_keys=True)):
            raise ValueError("dataset-native-weather-readback-mismatch")
        initial = metadata.get("initialization", {}).get("initial_reference_m", 0.)
        if type(initial) not in (int, float) or abs(initial - expected_initial_reference_m) > 1e-8:
            raise ValueError("dataset-requested-start-binding-mismatch")
        if metadata_path.read_bytes() != before or file_sha256(metadata_path) != gate["source"]["metadata_sha256"]:
            raise ValueError("dataset-weather-source-changed")
        gate["checks"].update(native_weather_preset_readback=True, requested_start_binding=True,
                              requested_heading_source_binding=True)
    except Exception as error:
        gate["status"] = "FAIL"
        gate["errors"].append({"code": str(error) if isinstance(error, ValueError) else "dataset-weather-validation-failed",
                               "exception_type": type(error).__name__})
    return gate


def runtime_source_hashes(manifest_path=None):
    if manifest_path is not None:
        value = read_strict_json(manifest_path)
        expected = value["source_sha256"]
    else:
        expected = {path.relative_to(ROOT).as_posix(): file_sha256(path) for pattern in
                    ("carla_tools/*.py", "scripts/*.py", "configs/*.json") for path in ROOT.glob(pattern)}
    actual = {}
    for name, digest in expected.items():
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()) or file_sha256(path) != digest:
            raise ValueError("dataset-runtime-source-hash-mismatch")
        actual[name] = digest
    return actual


def archive_dataset(directory, artifact_dir):
    """Publish all owned native bytes, including failed or interrupted trials."""
    source, destination = Path(directory), Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if source.is_symlink() or not source.is_dir() or destination.is_symlink():
        raise ValueError("dataset-archive-directory-invalid")
    archive_path = destination / "dataset_artifacts.zip"
    if archive_path.exists():
        raise ValueError("dataset-archive-already-exists")
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if any(path.is_symlink() or not path.resolve().is_relative_to(source.resolve()) for path in files):
        raise ValueError("dataset-archive-source-escaped")
    size = sum(path.stat().st_size for path in files)
    if shutil.disk_usage(destination).free < size + 512 * 2**20:
        raise ValueError("dataset-archive-free-space-insufficient")
    with zipfile.ZipFile(archive_path, "x", allowZip64=True) as archive:
        for path in files:
            archive.write(path, Path("dataset") / path.relative_to(source),
                          compress_type=zipfile.ZIP_STORED if path.suffix == ".png" else zipfile.ZIP_DEFLATED,
                          compresslevel=None if path.suffix == ".png" else 1)
    return {"archive_bytes": archive_path.stat().st_size, "archive_sha256": file_sha256(archive_path),
            "native_uncompressed_bytes": size, "archive_file_count": len(files)}


def dry_run(config_path):
    from carla_tools.dataset_v1 import load_dataset_v1_config
    config = load_dataset_v1_config(config_path)
    route_count = sum(config.train_routes_per_case.values()) + sum(config.dev_routes_per_case.values())
    return {"status": "PASS", "scope": SCOPE, "config_sha256": config.sha256, "route_count": route_count,
            "expected_episode_count": route_count * len(config.weather_presets) * len(config.collection_runs) * len(config.conditions),
            "weather_presets": config.to_dict()["weather_presets"], "collection_runs": config.to_dict()["collection_runs"],
            "maximum_samples": config.maximum_samples, "minimum_samples": config.minimum_samples,
            "train_selection_samples": config.train_selection_samples, "carla_imported": False,
            "native_collection_started": False, "reserved_final_state": "UNOPENED"}


def run_pipeline(output_dir, *, artifact_dir=None, config_path=ROOT / "configs/dataset_v1.json",
                 runtime_sources=None, frozen_route_catalog=None, resume=False):
    """Run the predeclared matrix; no adaptive padding or final-town inspection."""
    from carla_tools.dataset_v1 import (build_dataset_v1_manifest, load_dataset_v1_config,
                                      validate_dataset_v1_manifest)
    directory, began = Path(output_dir), time.monotonic()
    published = Path(artifact_dir) if artifact_dir is not None else directory.parent
    report = {"schema_version": 1, "scope": SCOPE, "status": "FAIL", "stage": "output-preflight",
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "episodes": [],
              "episode_count": 0, "native_frame_count": 0, "sample_count": 0,
              "reserved_final_town": "Town02", "reserved_final_state": "UNOPENED", "resumed": resume}
    owned = False
    try:
        if directory.is_symlink() or (directory.exists() and not resume):
            raise ValueError("dataset-output-already-exists")
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=False)
        elif not directory.is_dir():
            raise ValueError("dataset-output-not-directory")
        owned = True
        config = load_dataset_v1_config(config_path)
        hashes = runtime_source_hashes(runtime_sources)
        report.update(config_sha256=config.sha256, source_sha256=hashes)
        source_config = load_collection_config(ROOT / "configs/collection_d4.json")
        carla = importlib.import_module("carla")
        report["stage"] = "native-topology-survey"
        client = carla.Client(source_config.connection.host, source_config.connection.port)
        client.set_timeout(source_config.connection.timeout_seconds)
        if client.get_client_version() != "0.9.16" or client.get_server_version() != "0.9.16":
            raise ValueError("dataset-native-version-mismatch")
        catalog = survey_pilot_routes(client.get_world().get_map(), carla, config)
        if frozen_route_catalog is not None and read_strict_json(frozen_route_catalog)["routes"] != catalog["routes"]:
            raise ValueError("dataset-live-frozen-route-catalog-mismatch")
        plan = collection_matrix(catalog["routes"], config)
        protocol = {"schema_version": 1, "scope": SCOPE, "config": config.to_dict(), "config_sha256": config.sha256,
                    "routes": catalog["routes"], "trials": plan, "source_sha256": hashes,
                    "weather_definitions": {name: public_fields(getattr(carla.WeatherParameters, name))
                                            for name in config.weather_presets},
                    "reserved_final_town": "Town02", "reserved_final_state": "UNOPENED"}
        protocol_path = directory / "dataset_protocol.json"
        if protocol_path.exists():
            if not resume or read_strict_json(protocol_path) != protocol:
                raise ValueError("dataset-resume-protocol-mismatch")
        else:
            write_report(protocol_path, protocol)
        write_report(directory / "dataset_route_catalog.json", catalog)
        report.update(protocol_sha256=file_sha256(protocol_path), route_count=len(catalog["routes"]),
                      planned_episode_count=len(plan), matrix_sha256=canonical_sha256(plan))
        descriptors = []
        for index, trial in enumerate(plan):
            report["stage"] = "native-collection"
            episode = directory / trial["episode"]
            if episode.exists():
                if not resume or episode.is_symlink():
                    raise ValueError("dataset-resume-episode-refused")
                gate = validate_dataset_v1_episode(episode, allowed_conditions=config.conditions,
                    expected_initial_reference_m=trial["initial_reference_m"], expected_weather=trial["weather"], carla_module=carla)
                if gate["status"] != "PASS":
                    raise ValueError("dataset-incomplete-resume-episode-retained")
                result = {"status": "PASS", "step_count": gate["metrics"]["step_count"], "resumed": True}
            else:
                native = source_config.to_dict()
                native["weather"]["preset"], native["seed"] = trial["weather"], trial["seed"]
                collection = collection_config_from_dict(native, allowed_weather_presets=DEVELOPMENT_WEATHERS)
                route = next(item["route"] for item in catalog["routes"] if item["route"]["route_id"] == trial["route_id"])
                result = run_dataset_v1_trial(route, trial["case"], trial["condition"], episode, collection,
                    carla_module=carla, initial_reference_m=trial["initial_reference_m"])
            entry = {**episode_descriptor(trial), "execution": result}
            report["episodes"].append(entry)
            report["episode_count"] += 1
            report["native_frame_count"] += result.get("step_count", 0)
            write_report(directory / "dataset_report.json", report)
            # This small view is available while the large native directory is
            # still in scratch. Completed evidence is checked again below.
            published.mkdir(parents=True, exist_ok=True)
            write_report(published / "dataset_progress.json", {**{key: report[key] for key in
                ("stage", "episode_count", "native_frame_count", "planned_episode_count", "config_sha256")}, "status": "RUNNING"})
            print(json.dumps({"completed_episodes": index + 1, "planned_episodes": len(plan),
                "episode": trial["episode"], "status": result["status"], "frames": result.get("step_count", 0)}, separators=(",", ":")), flush=True)
            if result["status"] != "PASS":
                raise ValueError("dataset-native-trial-operation-failed")
            descriptors.append(episode_descriptor(trial))
        report["native_collection_elapsed_seconds"] = time.monotonic() - began
        report["stage"] = "immutable-manifest-and-qa"
        write_report(published / "dataset_progress.json", {"status": "RUNNING", "stage": report["stage"],
            "episode_count": report["episode_count"], "native_frame_count": report["native_frame_count"], "config_sha256": config.sha256})
        manifest_path = directory / "dataset_v1_manifest.json"
        if manifest_path.exists() and not resume:
            raise ValueError("dataset-manifest-already-exists")
        if not manifest_path.exists():
            manifest_path = build_dataset_v1_manifest(directory, catalog["routes"], descriptors, config)
        def qa_progress(value):
            write_report(published / "dataset_progress.json", {"status": "RUNNING", "stage": "independent-native-qa", **value})
            print(json.dumps({"stage": "independent-native-qa", **value}, separators=(",", ":")), flush=True)
        report["gate"] = validate_dataset_v1_manifest(manifest_path, progress_callback=qa_progress)
        if report["gate"]["status"] != "PASS":
            raise ValueError("dataset-native-source-label-pair-split-gate-failed")
        write_report(directory / "dataset_validation_report.json", report["gate"])
        report.update(sample_count=report["gate"]["sample_count"], split_counts=report["gate"]["split_counts"],
                      manifest_sha256=file_sha256(manifest_path))
        report["stage"] = "split-selections-and-label-previews"
        from carla_tools.dataset_artifacts import export_dataset_artifacts
        report["dataset_artifacts"] = export_dataset_artifacts(manifest_path, report["gate"], directory / "qa")
        if report["dataset_artifacts"]["status"] != "PASS":
            raise ValueError("dataset-label-preview-or-export-gate-failed")
        report["stage"] = "post-collection-connection"
        report["connection_after"] = check_connection(ConnectionConfig(timeout=30, ticks=3), carla_module=carla)
        if report["connection_after"]["status"] != "PASS":
            raise ValueError("dataset-post-collection-rpc-failed")
        if runtime_source_hashes(runtime_sources) != hashes:
            raise ValueError("dataset-source-changed-during-collection")
        report.update(status="PASS", stage="verified", sources_unchanged=True)
    except BaseException as error:
        report["error"] = {"code": getattr(error, "code", str(error) if isinstance(error, ValueError) else "dataset-pipeline-operation-failed"),
                           "exception_type": type(error).__name__, "stage": report["stage"]}
    finally:
        report.update(elapsed_seconds=time.monotonic() - began,
                      finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        if owned:
            write_report(directory / "dataset_report.json", report)
            try:
                for name in ("dataset_protocol.json", "dataset_route_catalog.json", "dataset_report.json"):
                    if (directory / name).is_file() and directory.resolve() != published.resolve():
                        published.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(directory / name, published / name)
                publication = archive_dataset(directory, published)
                write_report(published / "dataset_archive_report.json", {"status": report["status"], **publication,
                    "manifest_sha256": report.get("manifest_sha256"), "config_sha256": report.get("config_sha256")})
                report.update(publication)
            except Exception as error:
                report.update(status="FAIL", packaging_error={"code": str(error) if isinstance(error, ValueError) else "dataset-packaging-failed",
                              "exception_type": type(error).__name__})
            write_report(published / "dataset_report.json", report)
            write_report(published / "dataset_progress.json", {key: report[key] for key in
                ("status", "stage", "episode_count", "native_frame_count", "sample_count", "elapsed_seconds")})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dataset_v1.json")
    parser.add_argument("--runtime-sources", type=Path)
    parser.add_argument("--frozen-route-catalog", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run and args.output_dir is None:
        parser.error("--output-dir is required for native collection")
    report = dry_run(args.config) if args.dry_run else run_pipeline(args.output_dir, artifact_dir=args.artifact_dir,
        config_path=args.config, runtime_sources=args.runtime_sources, frozen_route_catalog=args.frozen_route_catalog, resume=args.resume)
    print(json.dumps({key: report[key] for key in ("status", "stage", "config_sha256", "expected_episode_count", "episode_count",
        "native_frame_count", "sample_count", "split_counts", "elapsed_seconds", "archive_bytes", "error", "packaging_error") if key in report}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
