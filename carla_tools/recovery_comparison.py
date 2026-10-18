"""Freeze paired RGB driving and select by mean error with absolute case bounds.

Development data may select a model. Independent towns and final conditions
are deliberately outside this protocol. Decisions never update model pointers.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from .contracts import route_sha256, strict_json_loads
from .refinement_driving import CONDITIONS, INITIAL_REFERENCES_M

SEEDS = (42, 43)
TRAINING_ARMS = ("central_only", "central_recovery")
ARMS = ("incumbent", *(f"{arm}_seed{seed}" for seed in SEEDS for arm in TRAINING_ARMS))
CASES = ("straight", "left_curve", "right_curve")
VERSION = "paired-recovery-development-comparison-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def comparison_source_hashes():
    from .recovery_training import recovery_source_hashes
    from .runtime_driving import runtime_source_hashes
    root = Path(__file__).resolve().parents[1]
    names = ("carla_tools/recovery_comparison.py", "scripts/run_recovery_driving.py",
        "carla_tools/model_selection.py", "configs/model_selection.json", "configs/collection_d4.json")
    return dict(recovery_source_hashes(), **runtime_source_hashes(), **{
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names})


def paired_trial_matrix(routes):
    """Alternate arm order on the same route, start, and perturbation."""
    if (type(routes) is not list or len(routes) != 3
            or {r["case"] for r in routes} != set(CASES)
            or len({r["route"]["route_id"] for r in routes}) != 3):
        raise ValueError("recovery-driving-three-development-cases-required")
    trials = []
    for record in sorted(routes, key=lambda r: CASES.index(r["case"])):
        for start in INITIAL_REFERENCES_M:
            for condition in CONDITIONS:
                state_index = len(trials) // len(ARMS)
                order = ARMS if state_index % 2 == 0 else tuple(reversed(ARMS))
                for arm in order:
                    trials.append({"arm": arm, "case": record["case"],
                        "route_id": record["route"]["route_id"],
                        "initial_reference_m": start, "condition": dict(condition),
                        "episode": f"episodes/{arm}/{record['case']}__start{int(start):02d}__{condition['name']}"})
    return trials


def freeze_driving_protocol(routes, checkpoints, training_comparison_sha256, *, representative_seed):
    from .baseline import load_baseline_config
    from .config import load_collection_config
    from .controller import load_controller_config
    from .evaluation import load_evaluation_config
    from .model_selection import load_selection_policy
    from .path_correction import RGBPathCorrection
    if (routes.get("split") != "dev" or routes.get("final_town_used") is not False
            or any(r.get("split") != "dev" or r["route"]["town"] != "Town01"
                or route_sha256(r["route"]) != r["route"]["route_sha256"]
                for r in routes["routes"])):
        raise ValueError("recovery-driving-frozen-development-routes-required")
    import re
    if (type(representative_seed) is not int or representative_seed not in SEEDS
            or set(checkpoints) != set(ARMS) or any(not re.fullmatch("[0-9a-f]{64}", str(h))
            for h in (*checkpoints.values(), training_comparison_sha256))):
        raise ValueError("recovery-driving-checkpoint-bindings-invalid")
    correction = RGBPathCorrection()
    collection = load_collection_config(Path(__file__).resolve().parents[1] / "configs/collection_d4.json")
    return {"schema_version": 1, "protocol_version": VERSION,
        "split": "dev", "town": "Town01", "weather": "ClearNoon", "final_town_used": False,
        "scope": "paired-short-development-routes-not-independent-generalization",
        "training_comparison_sha256": training_comparison_sha256,
        "training_seeds": list(SEEDS), "representative_seed": representative_seed,
        "checkpoint_sha256": dict(checkpoints), "routes_sha256": digest(routes),
        "source_sha256": comparison_source_hashes(), "planned_trials": paired_trial_matrix(routes["routes"]),
        "goal_reference_m": 19., "timeout_seconds": 10.,
        "controller_config_sha256": load_controller_config().sha256,
        "collection_base_config_sha256": collection.sha256,
        "configured_collection_seed": collection.seed, "simulator_repeats_per_state": 1,
        "physics_determinism_claimed": False,
        "baseline_config_sha256": load_baseline_config().sha256,
        "evaluation_config_sha256": replace(load_evaluation_config(), timeout_seconds=10.).sha256,
        "correction": "identical-frozen-RGB-path-correction-for-all-arms",
        "correction_config_sha256": correction.config.sha256,
        "offline_policy": load_selection_policy(),
        "driving_policy": {"mean_lateral_max_m": .5, "p95_lateral_max_m": 1.,
            "collision_count_max": 0, "lane_invasion_count_max": 0,
            "clean_completion_not_lower_than_incumbent": True,
            "steady_processing_p95_max_ms": 50., "warmup_frames_per_episode": 10,
            "primary_metric": "equal-episode-mean-absolute-lateral-error-m"},
        "representative_rule": "one-common-seed-pair-selected-by-mean-fixed-epoch-DEV-loss-before-full-DEV; all-seeds-driven",
        "perception_inputs": "RGB-only-learned-prediction-and-fixed-nominal-RGB-correction",
        "learned_raw_predictions_retained": True}


class RecoveryRuntimePredictor:
    """Adapt source-bound recovery weights to the unchanged RGB runtime path."""
    def __new__(cls, checkpoint, *, arm, expected_sha256, device="cpu",
            expected_manifest_sha256=None, expected_initial_checkpoint_sha256=None):
        from .runtime_driving import RuntimeCheckpointPredictor
        if arm == "incumbent":
            return RuntimeCheckpointPredictor(checkpoint, expected_sha256=expected_sha256,
                device=device, correction="path")
        if arm not in ARMS[1:]:
            raise ValueError("recovery-driving-arm-invalid")
        from .recovery_training import load_recovery_checkpoint
        from .path_correction import PathCorrectedPredictor
        model, config, state = load_recovery_checkpoint(checkpoint, device=device,
            expected_sha256=expected_sha256, manifest_sha256=expected_manifest_sha256)
        training_arm, seed = arm.rsplit("_seed", 1)
        if (state["arm"] != training_arm or config.seed != int(seed)
                or state["variant"] != "hybrid_geometry"
                or expected_initial_checkpoint_sha256 is not None
                    and state.get("initial_checkpoint_sha256") != expected_initial_checkpoint_sha256):
            raise ValueError("recovery-driving-checkpoint-arm-mismatch")
        identity = {"variant": state["variant"], "checkpoint_sha256": expected_sha256,
            "checkpoint_role": state["checkpoint_role"], "checkpoint_epoch": state["epoch"],
            "model_version": state["model_contract"]["model_version"],
            "model_contract": state["model_contract"], "model_contract_sha256": state["model_contract_sha256"],
            "source_sha256": state["source_sha256"], "manifest_sha256": state["manifest_sha256"],
            "arm": training_arm, "seed": config.seed, "device": next(model.parameters()).device.type,
            "closed_loop_verified": False}
        identity.update({key: state[key] for key in ("protocol_sha256", "initial_checkpoint_sha256",
            "validation_report_sha256", "training_selection_sha256") if key in state})
        result = RuntimeCheckpointPredictor.__new__(RuntimeCheckpointPredictor)
        result.raw = SimpleNamespace(model=model, config=config, identity=identity)
        wrapped = PathCorrectedPredictor(result.raw)
        result.corrector, result._identity, result.correction = wrapped.corrector, wrapped.identity, "path"
        return result


def run_recovery_trial(route, trial, output, collection_config, *, predictor,
        protocol_sha256, carla_module, baseline_config, evaluation_config, controller_config):
    from .runtime_driving import RuntimeTrialPolicy
    from .driving import load_validation_config, run_drive_trial

    class Policy(RuntimeTrialPolicy):
        def configure_metadata(self, metadata):
            super().configure_metadata(metadata)
            metadata["recovery_comparison"] = {"arm": trial["arm"],
                "protocol_sha256": protocol_sha256, "source_sha256": comparison_source_hashes()}

    policy = Policy(baseline_config, evaluation_config, route, trial["condition"],
        predictor=predictor, initial_reference_m=trial["initial_reference_m"])
    return run_drive_trial(route, trial["case"], trial["condition"], output, collection_config,
        carla_module=carla_module, validation_config=load_validation_config(),
        controller_config=controller_config, trial_policy=policy)


def validate_recovery_episode(directory, predictor, trial, *, protocol_sha256):
    from .runtime_driving import validate_runtime_episode
    from .baseline_validation import _safe_file
    metadata = strict_json_loads(_safe_file(Path(directory), "episode.json").read_text(encoding="utf-8"))
    expected = {"arm": trial["arm"], "protocol_sha256": protocol_sha256,
        "source_sha256": comparison_source_hashes()}
    if metadata.get("recovery_comparison") != expected:
        return {"status": "FAIL", "errors": ["recovery-driving-episode-binding-mismatch"]}
    return validate_runtime_episode(directory, predictor,
        expected_initial_reference_m=trial["initial_reference_m"])


def summarize_paired_driving(report, protocol):
    """Reject missing, duplicated, replaced, or unverified paired states."""
    from .driving_quality import summarize_driving_quality
    if (report.get("status") != "PASS" or report.get("protocol_sha256") != digest(protocol)
            or report.get("closed_loop_executed") is not True
            or report.get("final_town_used") is not False
            or report.get("checkpoint_sha256") != protocol["checkpoint_sha256"]
            or report.get("source_sha256") != protocol["source_sha256"]
            or report.get("planned_trials") != protocol["planned_trials"]):
        raise ValueError("recovery-driving-report-protocol-mismatch")
    plan = {t["episode"]: t for t in protocol["planned_trials"]}
    episodes = report["episodes"]
    if len(episodes) != len(plan) or {e["episode"] for e in episodes} != set(plan):
        raise ValueError("recovery-driving-incomplete-or-duplicate-pairs")
    quality_input, groups = [], {}
    for episode in episodes:
        trial = plan[episode["episode"]]
        if (any(episode.get(k) != trial[k] for k in
                ("arm", "case", "route_id", "initial_reference_m", "condition"))
                or episode["gate"].get("status") != "PASS"
                or episode["execution"].get("status") != "PASS"):
            raise ValueError("recovery-driving-state-or-integrity-mismatch")
        metrics = episode["gate"]["metrics"]
        mean, p95 = metrics["lateral_abs_m"]["mean"], metrics["lateral_abs_m"]["p95"]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (mean, p95)):
            raise ValueError("recovery-driving-invalid-lateral-metrics")
        quality_input.append(dict(episode, variant=episode["arm"]))
        for case in ("all", episode["case"]):
            group = groups.setdefault((episode["arm"], case), {"episodes": 0,
                "episode_mean_lateral_m": [], "episode_p95_lateral_m": []})
            group["episodes"] += 1
            group["episode_mean_lateral_m"].append(mean)
            group["episode_p95_lateral_m"].append(p95)
    quality = summarize_driving_quality(quality_input)
    for quality_group in quality["groups"]:
        key = (quality_group["method"], quality_group["case"])
        group = groups[key]
        group.update(quality_group)
        group["mean_lateral_m"] = math.fsum(group.pop("episode_mean_lateral_m")) / group["episodes"]
        # Bound every paired episode's P95, rather than hiding a bad start
        # behind a pooled percentile from easier/longer episodes.
        group["maximum_episode_p95_lateral_m"] = max(group.pop("episode_p95_lateral_m"))
    return {arm: {case: groups[(arm, case)] for case in ("all", *CASES)} for arm in ARMS}


def validate_common_dev_reports(offline_reports, protocol):
    """Bind identical DEV images and inference representation across all seeds."""
    from .model_selection import _groups
    if set(offline_reports) != set(ARMS):
        raise ValueError("recovery-selection-all-seed-offline-reports-required")
    groups = {arm: _groups(dict(report, metrics=[row for row in report["metrics"]
        if row.get("weather", "all") == "all"])) for arm, report in offline_reports.items()}
    base = offline_reports["incumbent"]
    for arm, report in offline_reports.items():
        if (report["checkpoint_sha256"] != protocol["checkpoint_sha256"][arm]
                or digest(report["evaluation_contract"]) != report["evaluation_contract_sha256"]
                or any(report[k] != base[k] for k in ("manifest_sha256", "sample_count",
                    "evaluation_contract_sha256", "common_dev_selection_sha256"))
                or any(groups[arm][case]["sample_count"] != groups["incumbent"][case]["sample_count"]
                    for case in ("all", *CASES))):
            raise ValueError("recovery-selection-common-DEV-or-checkpoint-mismatch")
    return groups


def select_recovery_model(offline_reports, driving_report, protocol):
    """Allow a case to worsen inside its budget; require overall improvement."""
    groups = validate_common_dev_reports(offline_reports, protocol)
    driving = summarize_paired_driving(driving_report, protocol)
    reasons, eligible = {}, []
    offline_policy, policy = protocol["offline_policy"], protocol["driving_policy"]
    for arm in ARMS[1:]:
        failed = []
        for case in ("all", *CASES):
            group, native = groups[arm][case], driving[arm][case]
            if group["coverage"] != 1.:
                failed.append(case + ":offline-coverage")
            for key, bound in (("near_mean_m", "near_mean_max_m"), ("far_mean_m", "far_mean_max_m")):
                if group[key] > offline_policy[bound]:
                    failed.append(case + ":" + key)
            if native["mean_lateral_m"] > policy["mean_lateral_max_m"]:
                failed.append(case + ":driving-mean-lateral-bound")
            if native["maximum_episode_p95_lateral_m"] > policy["p95_lateral_max_m"]:
                failed.append(case + ":driving-p95-lateral-bound")
            if native["collision_episodes"]:
                failed.append(case + ":collision")
            if native["lane_invasion_episodes"]:
                failed.append(case + ":lane-invasion")
            if native["body_lane_margin_negative_episodes"]:
                failed.append(case + ":body-outside-lane")
            if case == "all" and native["clean_completion_rate"] < driving["incumbent"][case]["clean_completion_rate"]:
                failed.append(case + ":clean-completion-regression")
        if groups[arm]["all"]["mean_euclidean_m"] >= groups["incumbent"]["all"]["mean_euclidean_m"]:
            failed.append("overall-offline-mean-improvement")
        if driving[arm]["all"]["mean_lateral_m"] >= driving["incumbent"]["all"]["mean_lateral_m"]:
            failed.append("overall-driving-mean-improvement")
        if offline_reports[arm]["latency_ms"]["p95"] > offline_policy["forward_p95_max_ms"]:
            failed.append("offline-forward-latency")
        latency = driving_report["steady_processing_latency_ms"][arm]
        if (type(latency.get("p95")) not in (int, float) or not math.isfinite(latency["p95"])
                or latency["p95"] < 0 or latency["p95"] > policy["steady_processing_p95_max_ms"]):
            failed.append("steady-processing-latency")
        reasons[arm] = failed
        if not failed:
            eligible.append(arm)
    representatives = tuple(f"{arm}_seed{protocol['representative_seed']}" for arm in TRAINING_ARMS)
    deployment_eligible = [arm for arm in eligible if arm in representatives]
    selected = min(deployment_eligible, key=lambda arm: (driving[arm]["all"]["mean_lateral_m"],
        groups[arm]["all"]["mean_euclidean_m"], arm)) if deployment_eligible else "incumbent"
    import statistics
    seed_effects = []
    for seed in SEEDS:
        a, b = f"central_only_seed{seed}", f"central_recovery_seed{seed}"
        seed_effects.append({"seed": seed,
            "offline_mean_delta_m_B_minus_A": groups[b]["all"]["mean_euclidean_m"] - groups[a]["all"]["mean_euclidean_m"],
            "driving_mean_delta_m_B_minus_A": driving[b]["all"]["mean_lateral_m"] - driving[a]["all"]["mean_lateral_m"],
            "clean_completion_delta_B_minus_A": driving[b]["all"]["clean_completion_rate"] - driving[a]["all"]["clean_completion_rate"]})
    effect_summary = {key: {"mean": statistics.fmean(row[key] for row in seed_effects),
        "std": statistics.pstdev(row[key] for row in seed_effects)} for key in seed_effects[0] if key != "seed"}
    return {"schema_version": 1, "status": "PASS", "scope": VERSION,
        "selected_arm": selected, "selected_checkpoint_sha256": protocol["checkpoint_sha256"][selected],
        "eligible_improvements": eligible, "deployment_eligible_improvements": deployment_eligible,
        "representative_seed": protocol["representative_seed"], "failed_criteria": reasons,
        "offline_mean_m": {arm: groups[arm]["all"]["mean_euclidean_m"] for arm in ARMS},
        "driving": driving, "recovery_effect": {"per_seed": seed_effects,
            "two_seed_summary": effect_summary, "scope": "paired-training-seed-variation-on-one-simulator-configuration"},
        "protocol_sha256": digest(protocol), "pointer_updated": False, "final_town_used": False,
        "limitations": "two training seeds, one simulator configuration; short correlated development routes; no final-town evidence"}
