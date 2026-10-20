"""Execute a frozen staged controller and camera robustness development study."""
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


def run_pipeline(output, routes_file, protocol_file, checkpoint, *, device="cpu"):
    from carla_tools.baseline import load_baseline_config
    from carla_tools.config import load_collection_config
    from carla_tools.contracts import strict_json_loads
    from carla_tools.development_study import (ProfiledBaselinePredictor, STAGES, confirm_controller,
        digest, freeze_development_protocol, planned_stage_trials, run_study_trial, select_screen_candidate,
        select_tuned_controller, study_source_hashes, summarize_stage, validate_study_episode)
    from carla_tools.driving_media import export_driving_media
    from carla_tools.evaluation import load_evaluation_config, write_metric_csv
    from carla_tools.pipeline_latency import aggregate_pipeline_latency
    from carla_tools.runtime_driving import RuntimeCheckpointPredictor
    from scripts.verify_collection import write_report
    output, checkpoint = Path(output), Path(checkpoint)
    if output.exists() or output.is_symlink() or (output.parent / "native_artifacts.zip").exists():
        raise ValueError("development-new-output-directory-required")
    routes_bytes, protocol_bytes = Path(routes_file).read_bytes(), Path(protocol_file).read_bytes()
    routes, protocol = strict_json_loads(routes_bytes.decode()), strict_json_loads(protocol_bytes.decode())
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if protocol != freeze_development_protocol(routes, checkpoint_hash, protocol["controller_configs"],
            environments=protocol["environments"]):
        raise ValueError("development-preexecution-frozen-protocol-mismatch")
    baseline = load_baseline_config()
    predictors = {"M1": RuntimeCheckpointPredictor(checkpoint, expected_sha256=checkpoint_hash,
        device=device, correction="path"), "B0": ProfiledBaselinePredictor(baseline)}
    sources = study_source_hashes()
    output.mkdir(parents=True)
    (output / "protocol.json").write_bytes(protocol_bytes)
    (output / "routes.json").write_bytes(routes_bytes)
    report = {"schema_version": 1, "status": "FAIL", "scope": protocol["protocol_version"],
        "protocol_sha256": digest(protocol), "protocol_file_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "checkpoint_sha256": checkpoint_hash, "source_sha256": sources,
        "models": {name: value.identity for name, value in predictors.items()},
        "expected_episode_count": protocol["expected_episodes"], "episodes": [], "episode_count": 0,
        "frame_count": 0, "resolved_stage_trials": {}, "stage_summaries": {}, "stage_decisions": {},
        "closed_loop_executed": False, "final_town_used": False, "final_conditions_used": False,
        "pointer_updated": False, "active_stage": None}
    started = time.monotonic()
    try:
        carla = importlib.import_module("carla")
        collection = load_collection_config(ROOT / "configs/collection_d4.json")
        evaluation = replace(load_evaluation_config(), timeout_seconds=protocol["timeout_seconds"])
        shortlist, candidate, latency_episodes = None, None, []
        for stage in STAGES:
            trials = planned_stage_trials(protocol, routes, stage,
                candidate=shortlist if stage == "tune" else candidate)
            report["active_stage"] = stage
            report["resolved_stage_trials"][stage] = trials
            write_report(output / "native_report.json", report)
            print(json.dumps({"event": "study-stage-start", "stage": stage,
                "episodes": len(trials), "candidate": shortlist if stage == "tune" else candidate}), flush=True)
            stage_episodes = []
            for trial in trials:
                route = next(row["route"] for row in routes["routes"] if row["route"]["route_id"] == trial["route_id"])
                directory, predictor = output / trial["episode"], predictors[trial["method"]]
                execution = run_study_trial(route, trial, directory, collection, predictor=predictor,
                    protocol=protocol, carla_module=carla, baseline_config=baseline, evaluation_config=evaluation)
                gate = (validate_study_episode(directory, predictor, trial, protocol=protocol, routes=routes)
                    if execution.get("status") == "PASS" else {"status": "FAIL", "errors": ["development-native-operation-failed"]})
                record = dict(trial, execution=execution, gate=gate)
                report["episodes"].append(record)
                report["episode_count"] += 1
                report["frame_count"] += gate.get("frame_count", 0)
                write_report(output / "native_report.json", report)
                print(json.dumps({"event": "study-episode-complete", "index": report["episode_count"],
                    "expected": protocol["expected_episodes"], "stage": stage, "arm": trial["arm"],
                    "case": trial["case"], "environment": trial["environment"]["name"],
                    "integrity": gate["status"], "metrics": gate.get("metrics"), "errors": gate.get("errors")}), flush=True)
                if gate["status"] != "PASS":
                    raise ValueError("development-native-integrity-failed:" + str(gate.get("errors")))
                metadata = strict_json_loads((directory / "episode.json").read_text(encoding="utf-8"))
                rows = [strict_json_loads(line) for line in (directory / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
                latency_input = {"episode_id": trial["episode"], "method": stage + "__" + trial["arm"],
                    "case": trial["case"], "driving_success": gate["driving_success"],
                    "expected_frame_count": metadata["step_count"],
                    "rows": [{"frame": row["frame"], "latency": row["perception"]["pipeline_timing"]} for row in rows]}
                latency_episodes.append(latency_input)
                latency = aggregate_pipeline_latency([latency_input],
                    warmup_frames=protocol["selection_policy"]["warmup_frames_per_episode"], budget_ms=50.)
                record["steady_processing_p95_ms"] = next(group["steady_state"]["total_processing_ms"]["p95_ms"]
                    for group in latency["groups"] if group["case"] == "all")
                record["media"] = export_driving_media(directory)
                if record["media"].get("status") != "PASS":
                    raise ValueError("development-native-media-failed")
                stage_episodes.append(record)
                write_report(output / "native_report.json", report)
            summary = summarize_stage(stage_episodes, protocol, expected_trials=trials)
            report["stage_summaries"][stage] = summary
            if stage == "screen":
                decision = select_screen_candidate(summary, protocol)
                shortlist = decision["selected_nonincumbent"]
            elif stage == "tune":
                decision = select_tuned_controller(summary, protocol, shortlist)
                candidate = decision["selected_controller"]
            elif stage == "confirmation":
                decision = {"status": "PASS", "candidate": candidate, "summary_sha256": digest(summary),
                    "selection_pending_robustness": True}
            else:
                decision = confirm_controller(report["stage_summaries"]["confirmation"], summary, protocol, candidate)
                report["selection"] = decision
            report["stage_decisions"][stage] = decision
            # Persist the earlier-stage decision before obtaining later-stage data.
            write_report(output / (stage + "_decision.json"), decision)
            write_report(output / "native_report.json", report)
            print(json.dumps({"event": "study-stage-complete", "stage": stage, "decision": decision}), flush=True)
        latency = aggregate_pipeline_latency(latency_episodes,
            warmup_frames=protocol["selection_policy"]["warmup_frames_per_episode"], budget_ms=50.)
        write_report(output / "pipeline_latency.json", latency)
        report["aggregates"] = {stage: write_metric_csv([episode["gate"]["metrics"]
            for episode in report["episodes"] if episode["stage"] == stage], output / (stage + "_metrics.csv"),
            config=evaluation) for stage in STAGES}
        if (report["episode_count"] != protocol["expected_episodes"] or sources != study_source_hashes()
                or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_hash
                or Path(routes_file).read_bytes() != routes_bytes or Path(protocol_file).read_bytes() != protocol_bytes):
            raise ValueError("development-source-checkpoint-protocol-changed")
        report.update(status="PASS", closed_loop_executed=True, active_stage="complete")
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
    for name in ("output-dir", "routes", "protocol", "checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    result = run_pipeline(args.output_dir, args.routes, args.protocol, args.checkpoint, device=args.device)
    print(json.dumps({key: result[key] for key in ("status", "episode_count", "frame_count", "error") if key in result}), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
