"""Execute frozen paired development driving without changing deployment pointers."""
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


def run_pipeline(output, routes_file, protocol_file, models_file, comparison_file, *, device="cpu"):
    from carla_tools.baseline import load_baseline_config
    from carla_tools.config import load_collection_config
    from carla_tools.contracts import strict_json_loads
    from carla_tools.controller import load_controller_config
    from carla_tools.driving_media import export_driving_media
    from carla_tools.evaluation import load_evaluation_config, write_metric_csv
    from carla_tools.pipeline_latency import aggregate_pipeline_latency
    from carla_tools.recovery_comparison import (ARMS, RecoveryRuntimePredictor,
        comparison_source_hashes, digest, freeze_driving_protocol, run_recovery_trial,
        select_recovery_model, validate_common_dev_reports, validate_recovery_episode)
    from carla_tools.recovery_training import (_equal_budget, load_recovery_protocol,
        recovery_source_hashes, representative_seed)
    from scripts.verify_collection import write_report

    output = Path(output)
    if output.exists() or output.is_symlink() or (output.parent / "native_artifacts.zip").exists():
        raise ValueError("recovery-driving-new-output-required")
    routes = strict_json_loads(Path(routes_file).read_text(encoding="utf-8"))
    protocol = strict_json_loads(Path(protocol_file).read_text(encoding="utf-8"))
    models = strict_json_loads(Path(models_file).read_text(encoding="utf-8"))
    comparison_bytes = Path(comparison_file).read_bytes()
    training = strict_json_loads(comparison_bytes.decode("utf-8"))
    hashes = {arm: model["sha256"] for arm, model in models.items()}
    expected = freeze_driving_protocol(routes, hashes, hashlib.sha256(comparison_bytes).hexdigest(),
        representative_seed=training["representative_seed"])
    if expected != protocol or training.get("status") != "PASS":
        raise ValueError("recovery-driving-preexecution-protocol-mismatch")
    train_protocol = load_recovery_protocol()
    if (training.get("equal_budget_verified") is not True
            or training.get("fixture_injection") is not False or training.get("final_town_used") is not False
            or training.get("protocol") != train_protocol.to_dict()
            or training.get("source_sha256") != recovery_source_hashes()
            or training.get("initial_checkpoint_sha256") != hashes["incumbent"]
            or training["representative_seed"] != representative_seed(training["training"], train_protocol)):
        raise ValueError("recovery-driving-training-provenance-invalid")
    _equal_budget(training["training"], train_protocol)
    offline = {}
    for arm in ARMS:
        value = Path(models[arm]["evaluation"]).read_bytes()
        if hashlib.sha256(value).hexdigest() != models[arm]["evaluation_sha256"]:
            raise ValueError("recovery-driving-offline-evaluation-changed")
        offline[arm] = strict_json_loads(value.decode("utf-8"))
        if (any(offline[arm].get(key) != training.get(key) for key in ("manifest_sha256",
                "validation_report_sha256", "protocol_sha256", "common_dev_selection_sha256"))
                or offline[arm].get("source_sha256") != recovery_source_hashes()):
            raise ValueError("recovery-driving-common-DEV-training-binding-mismatch")
        if arm != "incumbent":
            name, seed = arm.rsplit("_seed", 1)
            if training["training"][seed][name]["checkpoint_sha256"] != hashes[arm]:
                raise ValueError("recovery-driving-trained-checkpoint-report-mismatch")
    validate_common_dev_reports(offline, protocol)
    sources = comparison_source_hashes()
    predictors = {arm: RecoveryRuntimePredictor(models[arm]["path"], arm=arm,
        expected_sha256=hashes[arm], device=device, expected_manifest_sha256=training["manifest_sha256"],
        expected_initial_checkpoint_sha256=hashes["incumbent"]) for arm in ARMS}
    output.mkdir(parents=True)
    protocol_bytes = Path(protocol_file).read_bytes()
    (output / "protocol.json").write_bytes(protocol_bytes)
    (output / "training_comparison.json").write_bytes(comparison_bytes)
    report = {"status": "FAIL", "scope": protocol["protocol_version"],
        "protocol_sha256": digest(protocol), "protocol_file_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "checkpoint_sha256": hashes, "source_sha256": sources,
        "models": {arm: predictors[arm].identity for arm in ARMS},
        "planned_trials": protocol["planned_trials"], "episodes": [],
        "episode_count": 0, "frame_count": 0, "expected_episode_count": len(protocol["planned_trials"]),
        "closed_loop_executed": False, "final_town_used": False, "pointer_updated": False}
    began = time.monotonic()
    try:
        collection = load_collection_config(ROOT / "configs/collection_d4.json")
        evaluation = replace(load_evaluation_config(), timeout_seconds=protocol["timeout_seconds"])
        baseline, controller = load_baseline_config(), load_controller_config()
        carla = importlib.import_module("carla")
        latency_episodes = []
        for trial in protocol["planned_trials"]:
            route = next(r["route"] for r in routes["routes"] if r["route"]["route_id"] == trial["route_id"])
            directory, predictor = output / trial["episode"], predictors[trial["arm"]]
            execution = run_recovery_trial(route, trial, directory, collection, predictor=predictor,
                protocol_sha256=report["protocol_sha256"], carla_module=carla,
                baseline_config=baseline, evaluation_config=evaluation, controller_config=controller)
            gate = (validate_recovery_episode(directory, predictor, trial, protocol_sha256=report["protocol_sha256"])
                if execution.get("status") == "PASS" else {"status": "FAIL", "errors": ["recovery-driving-operation-failed"]})
            record = dict(trial, execution=execution, gate=gate)
            report["episodes"].append(record)
            report["episode_count"] += 1
            report["frame_count"] += gate.get("frame_count", 0)
            write_report(output / "native_report.json", report)
            print(json.dumps({"arm": trial["arm"], "case": trial["case"],
                "start": trial["initial_reference_m"], "condition": trial["condition"]["name"],
                "integrity": gate["status"], "metrics": gate.get("metrics")}), flush=True)
            if gate["status"] != "PASS":
                raise ValueError("recovery-driving-native-integrity-failed:" + str(gate.get("errors")))
            metadata = strict_json_loads((directory / "episode.json").read_text(encoding="utf-8"))
            rows = [strict_json_loads(line) for line in (directory / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
            latency_episodes.append({"episode_id": trial["episode"], "method": trial["arm"],
                "case": trial["case"], "driving_success": gate["driving_success"],
                "expected_frame_count": metadata["step_count"],
                "rows": [{"frame": row["frame"], "latency": row["perception"]["pipeline_timing"]} for row in rows]})
            record["media"] = export_driving_media(directory)
            if record["media"].get("status") != "PASS":
                raise ValueError("recovery-driving-media-failed")
            write_report(output / "native_report.json", report)
        latency = aggregate_pipeline_latency(latency_episodes,
            warmup_frames=protocol["driving_policy"]["warmup_frames_per_episode"], budget_ms=50.)
        write_report(output / "pipeline_latency.json", latency)
        report["steady_processing_latency_ms"] = {arm: {
            "p95": max(g["steady_state"]["total_processing_ms"]["p95_ms"] for g in latency["groups"]
                if g["method"] == arm),
            "mean": next(g["steady_state"]["total_processing_ms"]["mean_ms"] for g in latency["groups"]
                if g["method"] == arm and g["case"] == "all"),
            "scope": "maximum-of-all-and-each-case-steady-state-P95"} for arm in ARMS}
        report["aggregates"] = {arm: write_metric_csv([e["gate"]["metrics"] for e in report["episodes"]
            if e["arm"] == arm], output / (arm + "_metrics.csv"), config=evaluation) for arm in ARMS}
        if sources != comparison_source_hashes() or any(hashlib.sha256(Path(models[a]["path"]).read_bytes()).hexdigest()
                != hashes[a] for a in ARMS):
            raise ValueError("recovery-driving-source-or-weights-changed")
        report.update(status="PASS", closed_loop_executed=True)
        # Evaluation reports are transported separately from checkpoint tensors.
        # Their full common-DEV provenance remains in the training comparison.
        for arm in ARMS:
            value = Path(models[arm]["evaluation"]).read_bytes()
            if hashlib.sha256(value).hexdigest() != models[arm]["evaluation_sha256"]:
                raise ValueError("recovery-driving-offline-evaluation-changed")
        report["selection"] = select_recovery_model(offline, report, protocol)
    except Exception as error:
        report.update(status="FAIL", error={"type": type(error).__name__, "code": str(error)})
    finally:
        report["elapsed_seconds"] = time.monotonic() - began
        write_report(output / "native_report.json", report)
        with zipfile.ZipFile(output.parent / "native_artifacts.zip", "x", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    archive.write(path, Path("native") / path.relative_to(output))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "routes", "protocol", "models", "training-comparison"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    result = run_pipeline(args.output_dir, args.routes, args.protocol, args.models,
        args.training_comparison, device=args.device)
    print(json.dumps({k: result[k] for k in ("status", "episode_count", "frame_count", "error") if k in result}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
