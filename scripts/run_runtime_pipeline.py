"""Measure and compare development RGB driving with a fixed evaluation protocol."""
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


def runtime_trial_matrix(routes, candidate_variant):
    from carla_tools.refinement_driving import CONDITIONS, INITIAL_REFERENCES_M
    if (len(routes) != 3 or {r["case"] for r in routes} != {"straight", "left_curve", "right_curve"}
            or len({r["route"]["route_id"] for r in routes}) != 3
            or candidate_variant != "hybrid_geometry_path_near"):
        raise ValueError("runtime-paired-protocol-invalid")
    trials = []
    for record in routes:
        for start in INITIAL_REFERENCES_M:
            for condition in CONDITIONS:
                for variant in ("privileged_reference", "hybrid_geometry_near", candidate_variant):
                    trials.append({"variant": variant, "case": record["case"],
                        "route_id": record["route"]["route_id"], "initial_reference_m": start,
                        "condition": dict(condition),
                        "episode": f"episodes/{variant}/{record['case']}__start{int(start):02d}__{condition['name']}"})
    return trials


def collect_latency_episodes(output, episodes):
    from carla_tools.contracts import strict_json_loads
    records = []
    for episode in episodes:
        if episode["variant"] == "privileged_reference":
            continue
        root = Path(output) / episode["episode"]
        metadata = strict_json_loads((root / "episode.json").read_text())
        rows = [strict_json_loads(line) for line in (root / "steps.jsonl").read_text().splitlines()]
        records.append({"episode_id": episode["episode"], "method": episode["variant"],
            "case": episode["case"], "driving_success": episode["gate"]["driving_success"],
            "expected_frame_count": metadata["step_count"],
            "rows": [{"frame": row["frame"], "latency": row["perception"]["pipeline_timing"]} for row in rows]})
    return records


def benchmark_same_images(output, episodes, predictors):
    """Alternate candidate order on identical RGB states after explicit warmup."""
    import numpy as np
    from carla_tools.controller import PurePursuitController, VehicleGeometry, load_controller_config
    from carla_tools.projection import decode_rgb8_png
    from carla_tools.pipeline_latency import aggregate_pipeline_latency
    from carla_tools.runtime_driving import RuntimeTrialPolicy
    from carla_tools.baseline import load_baseline_config
    from carla_tools.evaluation import load_evaluation_config
    from types import SimpleNamespace

    controls, samples = {}, []
    for case in ("straight", "left_curve", "right_curve"):
        record = next(e for e in episodes if e["variant"] == "privileged_reference"
            and e["case"] == case and e["initial_reference_m"] == 0. and e["condition"] == "center")
        root = Path(output) / record["episode"]
        metadata = json.loads((root / "episode.json").read_text())
        rows = [json.loads(line) for line in (root / "steps.jsonl").read_text().splitlines()]
        for index in np.linspace(0, len(rows) - 1, 60).round().astype(int):
            row = rows[index]
            width, height, rgb = decode_rgb8_png((root / row["image"]["path"]).read_bytes(), row["image"]["sha256"], (640, 360))
            pixels = np.frombuffer(rgb, np.uint8).reshape(height, width, 3)
            bgra = np.empty((height, width, 4), np.uint8)
            bgra[:, :, :3], bgra[:, :, 3] = pixels[:, :, ::-1], 255
            samples.append((case, row, SimpleNamespace(width=width, height=height, raw_data=bgra.tobytes()), metadata))
    methods = [name for name in predictors if predictors[name] is not None]
    records = {name: [] for name in methods}
    for name in methods:
        predictor = predictors[name]
        first = samples[0]
        controller = PurePursuitController(load_controller_config(), VehicleGeometry(**first[3]["vehicle_geometry"]))
        policy = RuntimeTrialPolicy(load_baseline_config(), load_evaluation_config(), first[3]["route_contract"],
            first[3]["condition"], predictor=predictor)
        for _ in range(10):
            policy.command(first[2], first[1]["speed_mps"], controller)
        controls[name] = (policy, controller)
    for index, (case, row, image, _) in enumerate(samples):
        for name in methods if index % 2 == 0 else list(reversed(methods)):
            policy, controller = controls[name]
            _, prediction, _ = policy.command(image, row["speed_mps"], controller)
            if prediction["status"] != "PASS":
                raise ValueError("runtime-same-image-benchmark-prediction-failed")
            records[name].append({"frame": index + 1, "latency": prediction["pipeline_timing"],
                "case": case, "image_sha256": row["image"]["sha256"],
                "rgb_sha256": hashlib.sha256(np.frombuffer(image.raw_data, np.uint8)
                    .reshape(360, 640, 4)[:, :, [2, 1, 0]].tobytes()).hexdigest(),
                "waypoints_ego_m": prediction["waypoints_ego_m"]})
    grouped = []
    for name in methods:
        for case in ("straight", "left_curve", "right_curve"):
            rows = [r for r in records[name] if r["case"] == case]
            grouped.append({"episode_id": name + "/same-image/" + case, "method": name,
                "case": case, "driving_success": True, "expected_frame_count": len(rows), "rows": rows})
    result = aggregate_pipeline_latency(grouped, warmup_frames=0, budget_ms=50.)
    result.update(scope="alternating-same-RGB-processing-benchmark-not-new-driving",
        samples_per_method=len(samples), warmup_predictions_per_method=10,
        selection="60 uniformly spaced recorded frames from each privileged start00/center case",
        order="alternating-per-RGB-sample", predictions=records,
        reconstructed_BGRA="same recorded RGB bytes with opaque alpha; decode and reconstruction excluded")
    return result


def run_pipeline(output, routes_file, checkpoint, *, candidate_sha256, device="cpu"):
    from carla_tools.baseline import load_baseline_config
    from carla_tools.config import load_collection_config
    from carla_tools.contracts import route_sha256
    from carla_tools.controller import load_controller_config
    from carla_tools.driving_media import export_driving_media
    from carla_tools.driving_quality import summarize_driving_quality
    from carla_tools.evaluation import load_evaluation_config, write_metric_csv
    from carla_tools.failure_analysis import analyze_failure_episode
    from carla_tools.pipeline_latency import aggregate_pipeline_latency
    from carla_tools.runtime_driving import SCOPE, REFERENCE_IDENTITY, RuntimeCheckpointPredictor, runtime_source_hashes, run_runtime_trial, validate_runtime_episode
    from scripts.run_refinement_pipeline import summarize_attempts
    from scripts.verify_collection import write_report

    output, routes_file = Path(output), Path(routes_file)
    if output.exists() or output.is_symlink():
        raise ValueError("runtime-output-already-exists")
    data = json.loads(routes_file.read_text())
    routes = data["routes"]
    if (data.get("split") != "dev" or data.get("final_town_used") is not False
            or any(r["split"] != "dev" or r["route"]["town"] != "Town01"
                or route_sha256(r["route"]) != r["route"]["route_sha256"] for r in routes)):
        raise ValueError("runtime-frozen-development-route-invalid")
    prior = RuntimeCheckpointPredictor(checkpoint, expected_sha256=candidate_sha256, device=device, correction="near")
    candidate = RuntimeCheckpointPredictor(checkpoint, expected_sha256=candidate_sha256, device=device, correction="path")
    predictors = {"privileged_reference": None, prior.identity["variant"]: prior, candidate.identity["variant"]: candidate}
    plan = runtime_trial_matrix(routes, candidate.identity["variant"])
    output.mkdir(parents=True)
    evaluation = replace(load_evaluation_config(), timeout_seconds=10.)
    baseline, controller = load_baseline_config(), load_controller_config()
    report = {"status": "FAIL", "scope": SCOPE, "final_town_used": False,
        "episodes": [], "models": [REFERENCE_IDENTITY, prior.identity, candidate.identity],
        "expected_episode_count": len(plan), "episode_count": 0, "frame_count": 0,
        "closed_loop_executed": False, "planned_trials": plan,
        "protocol_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "routes_sha256": hashlib.sha256(routes_file.read_bytes()).hexdigest(),
        "runtime_source_sha256": runtime_source_hashes(), "goal_reference_m": 19.,
        "controller_config_sha256": controller.sha256, "clean_completion_frozen_before_execution": True,
        "latency_warmup_frames_per_episode": 10, "processing_budget_ms": 50.}
    began = time.monotonic()
    try:
        collection = load_collection_config(ROOT / "configs/collection_d4.json")
        carla = importlib.import_module("carla")
        for trial in plan:
            route = next(r["route"] for r in routes if r["route"]["route_id"] == trial["route_id"])
            predictor = predictors[trial["variant"]]
            directory = output / trial["episode"]
            execution = run_runtime_trial(route, trial["case"], trial["condition"], directory, collection,
                predictor=predictor, carla_module=carla, baseline_config=baseline, evaluation_config=evaluation,
                controller_config=controller, initial_reference_m=trial["initial_reference_m"])
            gate = (validate_runtime_episode(directory, predictor, expected_initial_reference_m=trial["initial_reference_m"])
                if execution["status"] == "PASS" else {"status": "FAIL", "errors": ["runtime-operation-failed"]})
            record = {"variant": trial["variant"], "case": trial["case"], "condition": trial["condition"]["name"],
                "initial_reference_m": trial["initial_reference_m"], "episode": trial["episode"],
                "execution": execution, "gate": gate}
            report["episodes"].append(record)
            report["episode_count"] += 1
            report["frame_count"] += gate.get("frame_count", 0)
            print(json.dumps({"variant": trial["variant"], "case": trial["case"], "condition": record["condition"],
                "initial_reference_m": trial["initial_reference_m"], "integrity": gate["status"], "metrics": gate.get("metrics")}), flush=True)
            write_report(output / "native_report.json", report)
            if gate["status"] != "PASS":
                raise ValueError("runtime-integrity-failed:" + str(gate.get("errors")))
            record["media"] = export_driving_media(directory)
            record["failure_analysis"] = analyze_failure_episode(directory, integrity_verified=True)
            write_report(output / "native_report.json", report)
        report["aggregates"] = {name: write_metric_csv([e["gate"]["metrics"] for e in report["episodes"] if e["variant"] == name],
            output / (name + "_metrics.csv"), config=evaluation) for name in predictors}
        latency = aggregate_pipeline_latency(collect_latency_episodes(output, report["episodes"]), warmup_frames=10, budget_ms=50.)
        write_report(output / "pipeline_latency.json", latency)
        benchmark = benchmark_same_images(output, report["episodes"], predictors)
        write_report(output / "same_image_benchmark.json", benchmark)
        report.update(status="PASS", closed_loop_executed=True, quality=summarize_driving_quality(report["episodes"]),
            pipeline_latency_sha256=hashlib.sha256((output / "pipeline_latency.json").read_bytes()).hexdigest(),
            same_image_benchmark_sha256=hashlib.sha256((output / "same_image_benchmark.json").read_bytes()).hexdigest())
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
    for name in ("output-dir", "routes", "candidate"):
        parser.add_argument("--" + name, type=Path, required=True)
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
