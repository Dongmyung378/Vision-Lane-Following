"""Compare B0, a frozen ResNet and a refined RGB model on frozen DEV routes."""
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


def trial_matrix(routes, variants):
    """Full paired protocol; original routes are never shortened or re-anchored."""
    from carla_tools.refinement_driving import CONDITIONS, INITIAL_REFERENCES_M
    if len(variants) != len(set(variants)) or len(variants) != 3 or variants[:2] != ["B0", "resnet"]:
        raise ValueError("native-paired-methods-invalid")
    trials = []
    for record in routes:
        for start in INITIAL_REFERENCES_M:
            for condition in CONDITIONS:
                for variant in variants:
                    trials.append({"variant": variant, "case": record["case"], "route_id": record["route"]["route_id"],
                        "initial_reference_m": start, "condition": dict(condition),
                        "episode": f"episodes/{variant}/{record['case']}__start{int(start):02d}__{condition['name']}"})
    if len(trials) != 81 or len({t["episode"] for t in trials}) != 81:
        raise ValueError("native-paired-protocol-incomplete-or-duplicate")
    return trials


def summarize_attempts(plan, episodes):
    """Account for incomplete operations separately; driving failures stay counted."""
    result = {}
    planned = {t["episode"]: t for t in plan}
    if len(planned) != len(plan):
        raise ValueError("native-summary-duplicate-planned-episode")
    seen = set()
    for episode in episodes:
        if episode["episode"] not in planned or episode["episode"] in seen:
            raise ValueError("native-summary-unplanned-or-duplicate-attempt")
        if episode["variant"] != planned[episode["episode"]]["variant"]:
            raise ValueError("native-summary-method-mismatch")
        seen.add(episode["episode"])
    for variant in dict.fromkeys(t["variant"] for t in plan):
        expected = sum(t["variant"] == variant for t in plan)
        attempted = [e for e in episodes if e["variant"] == variant]
        verified = [e for e in attempted if e["gate"]["status"] == "PASS"]
        successful = sum(e["gate"]["metrics"]["driving_success"] for e in verified)
        result[variant] = {"planned_episodes": expected, "attempted_episodes": len(attempted),
            "integrity_verified_episodes": len(verified), "operation_or_integrity_failures": len(attempted)-len(verified),
            "unexecuted_episodes": expected-len(attempted), "successful_episodes": successful,
            "driving_failed_episodes": len(verified)-successful,
            "verified_driving_success_rate": successful/len(verified) if verified else None,
            "full_protocol_driving_success_rate": successful/expected if len(verified) == expected else None}
    return result


def run_pipeline(output, routes_file, reference, candidate, *, reference_sha256, candidate_sha256, reference_report, device="cpu"):
    from carla_tools.baseline import load_baseline_config
    from carla_tools.config import load_collection_config
    from carla_tools.controller import load_controller_config
    from carla_tools.evaluation import load_evaluation_config, write_metric_csv
    from carla_tools.refinement_driving import NativeCheckpointPredictor, NativeBaselinePredictor, SCOPE, run_refinement_trial
    from carla_tools.driving_media import export_driving_media
    from carla_tools.refinement_validation import validate_refinement_episode
    from carla_tools.contracts import route_sha256
    from scripts.verify_collection import write_report

    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("refinement-native-output-already-exists")
    route_data = json.loads(Path(routes_file).read_text(encoding="utf-8"))
    routes = route_data["routes"]
    if (route_data.get("split") != "dev" or route_data.get("final_town_used") is not False
            or len(routes) != 3 or {r["case"] for r in routes} != {"straight", "left_curve", "right_curve"}
            or any(r["split"] != "dev" or r["route"]["town"] != "Town01" or route_sha256(r["route"]) != r["route"]["route_sha256"] for r in routes)):
        raise ValueError("refinement-native-frozen-dev-routes-invalid")
    baseline = load_baseline_config()
    predictors = [NativeBaselinePredictor(baseline), NativeCheckpointPredictor(reference, kind="spatial", expected_sha256=reference_sha256,
                    training_report=reference_report, device=device),
                  NativeCheckpointPredictor(candidate, kind="refined", expected_sha256=candidate_sha256, device=device)]
    plan = trial_matrix(routes, [p.identity["variant"] for p in predictors])
    output.mkdir(parents=True)
    report = {"status": "FAIL", "scope": SCOPE, "final_town_used": False, "episodes": [],
              "protocol": "same-original-DEV-route-start-state-controller-speed-clock-and-termination",
              "models": [p.identity for p in predictors], "routes_sha256": hashlib.sha256(Path(routes_file).read_bytes()).hexdigest(),
              "episode_count": 0, "frame_count": 0, "expected_episode_count": len(plan), "planned_trials": plan,
              "closed_loop_executed": False, "initial_reference_m": [0., 2., 4.], "goal_reference_m": 19.}
    report["protocol_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    began = time.monotonic()
    try:
        collection, controller = load_collection_config(ROOT / "configs/collection_d4.json"), load_controller_config()
        evaluation = replace(load_evaluation_config(), timeout_seconds=10.)
        carla = importlib.import_module("carla")
        for trial in plan:
            record = next(r for r in routes if r["route"]["route_id"] == trial["route_id"])
            predictor = next(p for p in predictors if p.identity["variant"] == trial["variant"])
            variant, condition, relative = trial["variant"], trial["condition"], trial["episode"]
            directory = output / relative
            result = run_refinement_trial(record["route"], record["case"], condition, directory, collection,
                predictor=predictor, carla_module=carla, baseline_config=baseline,
                evaluation_config=evaluation, controller_config=controller, initial_reference_m=trial["initial_reference_m"])
            gate = (validate_refinement_episode(directory, predictor, expected_initial_reference_m=trial["initial_reference_m"])
                    if result["status"] == "PASS" else {"status": "FAIL", "errors": ["native-operation-failed"]})
            report["episodes"].append({"variant": variant, "case": record["case"], "condition": condition["name"],
                "initial_reference_m": trial["initial_reference_m"], "episode": relative, "execution": result, "gate": gate})
            report["episode_count"] += 1
            report["frame_count"] += gate.get("frame_count", 0)
            print(json.dumps({"variant": variant, "case": record["case"], "condition": condition["name"],
                              "initial_reference_m": trial["initial_reference_m"],
                              "integrity": gate["status"], "metrics": gate.get("metrics")}), flush=True)
            write_report(output / "native_report.json", report)
            if gate["status"] != "PASS":
                raise ValueError("refinement-native-episode-integrity-failed:" + str(gate.get("errors")))
            report["episodes"][-1]["media"] = export_driving_media(directory)
            write_report(output / "native_report.json", report)
        aggregates = {}
        for predictor in predictors:
            variant = predictor.identity["variant"]
            metrics = [e["gate"]["metrics"] for e in report["episodes"] if e["variant"] == variant]
            aggregates[variant] = write_metric_csv(metrics, output / (variant + "_metrics.csv"), config=evaluation)
        report.update(status="PASS", aggregates=aggregates, closed_loop_executed=True)
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": str(error)}
    finally:
        report["attempt_summary"] = summarize_attempts(plan, report["episodes"])
        report["elapsed_seconds"] = time.monotonic() - began
        write_report(output / "native_report.json", report)
        with zipfile.ZipFile(output.parent / "native_artifacts.zip", "x", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    archive.write(path, Path("native") / path.relative_to(output))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "routes", "reference", "candidate", "reference-report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    report = run_pipeline(args.output_dir, args.routes, args.reference, args.candidate,
        reference_sha256=args.reference_sha256, candidate_sha256=args.candidate_sha256,
        reference_report=args.reference_report, device=args.device)
    print(json.dumps({k: report[k] for k in ("status", "episode_count", "frame_count", "error") if k in report}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
