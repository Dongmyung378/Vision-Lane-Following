"""Run one frozen nominal or shifted pipeline confirmation on development routes."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_pipeline(output, routes_file, protocol_file, models_file, *, device="cpu"):
    from carla_tools.baseline import load_baseline_config
    from carla_tools.config import load_collection_config
    from carla_tools.contracts import strict_json_loads
    from carla_tools.development_final_driving import (final_source_hashes, freeze_final_protocol,
        run_final_trial, validate_final_episode)
    from carla_tools.development_selection import ARMS
    from carla_tools.development_study import ProfiledBaselinePredictor, digest, summarize_stage
    from carla_tools.driving_media import export_driving_media
    from carla_tools.evaluation import load_evaluation_config, write_metric_csv
    from carla_tools.pipeline_latency import aggregate_pipeline_latency
    from carla_tools.runtime_driving import RuntimeCheckpointPredictor
    from carla_tools.guarded_path_correction import wrap_preprocessing
    from scripts.verify_collection import write_report
    output = Path(output)
    if output.exists() or output.is_symlink() or (output.parent / "native_artifacts.zip").exists():
        raise ValueError("development-final-new-output-required")
    routes_bytes, protocol_bytes, models_bytes = (Path(path).read_bytes() for path in (routes_file, protocol_file, models_file))
    routes, protocol, models = (strict_json_loads(value.decode()) for value in (routes_bytes, protocol_bytes, models_bytes))
    if set(models) != set(ARMS[:-1]):
        raise ValueError("development-final-model-file-coverage-invalid")
    hashes = {arm: models[arm]["sha256"] for arm in ARMS[:-1]}; hashes["B0"] = None
    if (protocol != freeze_final_protocol(routes, hashes, protocol["controller_configs"], phase=protocol["phase"],
            representative_seed=protocol["representative_seed"], training_comparison_sha256=protocol["training_comparison_sha256"],
            parent_study_sha256=protocol["parent_study_sha256"], offline_checkpoint_sha256=protocol["offline_checkpoint_sha256"],
            training_audit_sha256=protocol["training_audit_sha256"], preprocessing_configs=protocol["preprocessing_configs"])
            or any(hashlib.sha256(Path(model["path"]).read_bytes()).hexdigest() != model["sha256"] for model in models.values())):
        raise ValueError("development-final-preexecution-protocol-or-model-mismatch")
    baseline = load_baseline_config()
    # The incumbent and control roles have identical tensor state. Share one
    # stateless predictor without sharing the per-episode controller state.
    unique = {}
    predictors = {}
    for arm in ARMS[:-1]:
        expected = models[arm]["sha256"]
        key = (expected, digest(protocol["preprocessing_configs"][arm]))
        if key not in unique:
            base = RuntimeCheckpointPredictor(models[arm]["path"], expected_sha256=expected,
                device=device, correction="path")
            unique[key] = wrap_preprocessing(base, protocol["preprocessing_configs"][arm])
        predictors[arm] = unique[key]
    predictors["B0"] = ProfiledBaselinePredictor(baseline)
    sources = final_source_hashes()
    output.mkdir(parents=True)
    (output / "protocol.json").write_bytes(protocol_bytes)
    (output / "routes.json").write_bytes(routes_bytes)
    report = {"schema_version": 1, "status": "FAIL", "scope": protocol["protocol_version"], "phase": protocol["phase"],
        "protocol_sha256": digest(protocol), "protocol_file_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "checkpoint_sha256": hashes, "preprocessing_configs": protocol["preprocessing_configs"],
        "source_sha256": sources, "models": {arm: predictor.identity for arm, predictor in predictors.items()},
        "planned_trials": protocol["planned_trials"], "episodes": [], "episode_count": 0, "frame_count": 0,
        "expected_episode_count": protocol["expected_episodes"], "closed_loop_executed": False,
        "final_town_used": False, "final_conditions_used": False, "pointer_updated": False}
    started = time.monotonic()
    try:
        collection = load_collection_config(ROOT / "configs/collection_d4.json")
        evaluation = replace(load_evaluation_config(), timeout_seconds=protocol["timeout_seconds"])
        carla = importlib.import_module("carla")
        latency_episodes = []
        for trial in protocol["planned_trials"]:
            route = next(record["route"] for record in routes["routes"] if record["route"]["route_id"] == trial["route_id"])
            directory, predictor = output / trial["episode"], predictors[trial["arm"]]
            execution = run_final_trial(route, trial, directory, collection, predictor=predictor,
                protocol=protocol, carla_module=carla, baseline_config=baseline, evaluation_config=evaluation)
            gate = (validate_final_episode(directory, predictor, trial, protocol=protocol, routes=routes)
                if execution.get("status") == "PASS" else {"status": "FAIL", "errors": ["development-final-operation-failed"]})
            record = dict(trial, execution=execution, gate=gate)
            report["episodes"].append(record)
            report["episode_count"] += 1
            report["frame_count"] += gate.get("frame_count", 0)
            write_report(output / "native_report.json", report)
            print(json.dumps({"event": "development-final-episode", "phase": protocol["phase"],
                "index": report["episode_count"], "expected": protocol["expected_episodes"], "arm": trial["arm"],
                "case": trial["case"], "environment": trial["environment"]["name"],
                "integrity": gate["status"], "metrics": gate.get("metrics"), "errors": gate.get("errors")}), flush=True)
            if gate["status"] != "PASS":
                raise ValueError("development-final-native-integrity-failed:" + str(gate.get("errors")))
            rows = [strict_json_loads(line) for line in (directory / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
            latency_input = {"episode_id": trial["episode"], "method": trial["arm"], "case": trial["case"],
                "driving_success": gate["driving_success"], "expected_frame_count": gate["frame_count"],
                "rows": [{"frame": row["frame"], "latency": row["perception"]["pipeline_timing"]} for row in rows]}
            latency_episodes.append(latency_input)
            latency = aggregate_pipeline_latency([latency_input],
                warmup_frames=protocol["selection_policy"]["warmup_frames_per_episode"], budget_ms=50.)
            record["steady_processing_p95_ms"] = next(group["steady_state"]["total_processing_ms"]["p95_ms"]
                for group in latency["groups"] if group["case"] == "all")
            record["media"] = export_driving_media(directory)
            if record["media"].get("status") != "PASS":
                raise ValueError("development-final-media-failed")
            write_report(output / "native_report.json", report)
        report["summary"] = summarize_stage(report["episodes"], protocol, expected_trials=protocol["planned_trials"])
        write_report(output / "pipeline_latency.json", aggregate_pipeline_latency(latency_episodes,
            warmup_frames=protocol["selection_policy"]["warmup_frames_per_episode"], budget_ms=50.))
        report["aggregates"] = {arm: write_metric_csv([record["gate"]["metrics"] for record in report["episodes"]
            if record["arm"] == arm], output / (arm + "_metrics.csv"), config=evaluation) for arm in ARMS}
        if (sources != final_source_hashes() or any(hashlib.sha256(Path(model["path"]).read_bytes()).hexdigest()
                != model["sha256"] for model in models.values()) or Path(routes_file).read_bytes() != routes_bytes
                or Path(protocol_file).read_bytes() != protocol_bytes or Path(models_file).read_bytes() != models_bytes):
            raise ValueError("development-final-sources-or-weights-changed")
        if report["episode_count"] != protocol["expected_episodes"]:
            raise ValueError("development-final-incomplete-paired-episodes")
        report.update(status="PASS", closed_loop_executed=True)
    except Exception as error:
        report.update(status="FAIL", error={"type": type(error).__name__, "code": str(error)})
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_report(output / "native_report.json", report)
        with zipfile.ZipFile(output.parent / "native_artifacts.zip", "x", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    archive.write(path, Path("native") / path.relative_to(output))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "routes", "protocol", "models"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    result = run_pipeline(args.output_dir, args.routes, args.protocol, args.models, device=args.device)
    print(json.dumps({key: result[key] for key in ("status", "episode_count", "frame_count", "error") if key in result}), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
