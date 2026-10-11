"""Isolate development driving failures with a frozen paired correction protocol."""
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


def failure_trial_matrix(routes, candidate="hybrid_geometry"):
    from carla_tools.refinement_driving import CONDITIONS, INITIAL_REFERENCES_M
    if (candidate != "hybrid_geometry" or len(routes) != 3
            or {r["case"] for r in routes} != {"straight", "left_curve", "right_curve"}
            or len({r["route"]["route_id"] for r in routes}) != 3):
        raise ValueError("failure-paired-routes-or-candidate-invalid")
    result = []
    for record in routes:
        for start in INITIAL_REFERENCES_M:
            for condition in CONDITIONS:
                for variant in ("privileged_reference", candidate, candidate + "_near"):
                    result.append({"variant": variant, "case": record["case"], "route_id": record["route"]["route_id"],
                        "initial_reference_m": start, "condition": dict(condition),
                        "episode": f"episodes/{variant}/{record['case']}__start{int(start):02d}__{condition['name']}"})
    return result


def run_pipeline(output, routes_file, candidate, *, candidate_sha256, device="cpu"):
    from carla_tools.baseline import load_baseline_config
    from carla_tools.config import load_collection_config
    from carla_tools.contracts import route_sha256
    from carla_tools.controller import load_controller_config
    from carla_tools.driving_media import export_driving_media
    from carla_tools.driving_quality import summarize_driving_quality
    from carla_tools.evaluation import load_evaluation_config, write_metric_csv
    from carla_tools.failure_analysis import analyze_failure_episode
    from carla_tools.failure_driving import REFERENCE_IDENTITY, SCOPE, source_hashes, run_failure_trial, validate_failure_episode
    from carla_tools.near_correction import NearCorrectedPredictor
    from carla_tools.refinement_driving import NativeCheckpointPredictor
    from scripts.run_refinement_pipeline import summarize_attempts
    from scripts.verify_collection import write_report

    output, routes_file = Path(output), Path(routes_file)
    if output.exists() or output.is_symlink():
        raise ValueError("failure-output-already-exists")
    data = json.loads(routes_file.read_text())
    routes = data["routes"]
    plan = failure_trial_matrix(routes)
    if (data.get("split") != "dev" or data.get("final_town_used") is not False
            or any(r["split"] != "dev" or r["route"]["town"] != "Town01"
                   or route_sha256(r["route"]) != r["route"]["route_sha256"] for r in routes)):
        raise ValueError("failure-frozen-dev-route-invalid")
    baseline, controller = load_baseline_config(), load_controller_config()
    raw = NativeCheckpointPredictor(candidate, kind="refined", expected_sha256=candidate_sha256, device=device)
    corrected = NearCorrectedPredictor(raw, baseline_config=baseline)
    predictors = {"privileged_reference": None, raw.identity["variant"]: raw, corrected.identity["variant"]: corrected}
    if set(predictors) != {p["variant"] for p in plan}:
        raise ValueError("failure-model-variant-mismatch")
    models = [REFERENCE_IDENTITY, raw.identity, corrected.identity]
    output.mkdir(parents=True)
    evaluation = replace(load_evaluation_config(), timeout_seconds=10.)
    report = {"status": "FAIL", "scope": SCOPE, "final_town_used": False, "episodes": [], "models": models,
        "expected_episode_count": len(plan), "episode_count": 0, "frame_count": 0, "closed_loop_executed": False,
        "protocol": "same-original-DEV-route-start-state-checkpoint-controller-speed-clock-and-termination",
        "planned_trials": plan, "protocol_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "routes_sha256": hashlib.sha256(routes_file.read_bytes()).hexdigest(), "runtime_source_sha256": source_hashes(),
        "controller_config_sha256": controller.sha256, "goal_reference_m": 19.,
        "clean_completion_frozen_before_execution": True, "correction_config": corrected.identity["correction_config"]}
    started = time.monotonic()
    try:
        collection = load_collection_config(ROOT / "configs/collection_d4.json")
        carla = importlib.import_module("carla")
        for trial in plan:
            route = next(r["route"] for r in routes if r["route"]["route_id"] == trial["route_id"])
            predictor = predictors[trial["variant"]]
            directory = output / trial["episode"]
            execution = run_failure_trial(route, trial["case"], trial["condition"], directory, collection,
                predictor=predictor, carla_module=carla, baseline_config=baseline, evaluation_config=evaluation,
                controller_config=controller, initial_reference_m=trial["initial_reference_m"])
            gate = (validate_failure_episode(directory, predictor, expected_initial_reference_m=trial["initial_reference_m"])
                if execution["status"] == "PASS" else {"status": "FAIL", "errors": ["failure-operation-failed"]})
            record = {"variant": trial["variant"], "case": trial["case"], "condition": trial["condition"]["name"],
                "initial_reference_m": trial["initial_reference_m"], "episode": trial["episode"], "execution": execution, "gate": gate}
            report["episodes"].append(record)
            report["episode_count"] += 1
            report["frame_count"] += gate.get("frame_count", 0)
            print(json.dumps({"variant": trial["variant"], "case": trial["case"], "condition": record["condition"],
                "initial_reference_m": trial["initial_reference_m"], "integrity": gate["status"], "metrics": gate.get("metrics")}), flush=True)
            write_report(output / "native_report.json", report)
            if gate["status"] != "PASS":
                raise ValueError("failure-episode-integrity-failed:" + str(gate.get("errors")))
            record["media"] = export_driving_media(directory)
            record["failure_analysis"] = analyze_failure_episode(directory, integrity_verified=True)
            write_report(output / "native_report.json", report)
        report["aggregates"] = {variant: write_metric_csv([e["gate"]["metrics"] for e in report["episodes"] if e["variant"] == variant],
            output / (variant + "_metrics.csv"), config=evaluation) for variant in predictors}
        report.update(status="PASS", closed_loop_executed=True, quality=summarize_driving_quality(report["episodes"]))
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": str(error)}
    finally:
        report["attempt_summary"] = summarize_attempts(plan, report["episodes"])
        report["elapsed_seconds"] = time.monotonic() - started
        write_report(output / "native_report.json", report)
        with zipfile.ZipFile(output.parent / "native_artifacts.zip", "x", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    archive.write(path, Path("native") / path.relative_to(output))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "routes", "candidate"):
        parser.add_argument("--" + name, type=Path, required=True)
    # The reusable transport can retain an archived reference attachment; it is
    # not an evaluated ResNet arm in this diagnostic protocol.
    for name in ("reference", "reference-report", "reference-sha256"):
        parser.add_argument("--" + name)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    report = run_pipeline(args.output_dir, args.routes, args.candidate, candidate_sha256=args.candidate_sha256, device=args.device)
    print(json.dumps({k: report[k] for k in ("status", "episode_count", "frame_count", "error") if k in report}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
