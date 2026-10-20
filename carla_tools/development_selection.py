"""Select a development pipeline from paired nominal and shifted drives.

The score balances nominal driving and the five distinct development shifts.
The repeated clear robustness drive is retained for checks but is not counted
as another independent environment. Decisions never publish runtime pointers.
"""
from __future__ import annotations

import copy
import math
import re

from .contracts import canonical_sha256
from .development_study import CASES, DEFAULT_ENVIRONMENTS, summarize_stage
from .model_selection import _groups

ARMS = ("incumbent", "control", "continued", "appearance_robust", "B0")
LEARNED_ARMS = ARMS[:-1]
VERSION = "balanced-development-pipeline-selection-v1"
DEFAULT_FINAL_POLICY = {
    "policy_version": VERSION,
    "nominal_weight": .5,
    "shift_weight": .5,
    "minimum_mean_improvement_m": .002,
    "minimum_relative_improvement": .01,
    "nominal_mean_regression_budget_m": .01,
    "case_mean_max_m": .5,
    "episode_p95_max_m": 1.,
    "processing_p95_max_ms": 50.,
    "offline_near_mean_max_m": .3,
    "offline_far_mean_max_m": 1.,
    "forward_p95_max_ms": 50.,
    "warmup_frames_per_episode": 10,
    "offline_role": "complete-common-DEV-and-absolute-bounds",
    "relative_casewise_dominance_required": False,
    "clear_robustness_repeat_scored_as_new_environment": False,
}


def final_selection_policy():
    return copy.deepcopy(DEFAULT_FINAL_POLICY)


def _require(value, code):
    if not value:
        raise ValueError("development-final-selection-" + code)


def _finite(value):
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
             "invalid-nonnegative-number")
    return float(value)


def _shared_protocol(nominal, robustness):
    keys = ("checkpoint_sha256", "offline_checkpoint_sha256", "training_audit_sha256",
            "controller_configs", "preprocessing_configs", "representative_seed",
            "training_comparison_sha256", "parent_study_sha256", "routes_sha256",
            "source_sha256", "selection_policy", "environments")
    _require(all(key in nominal and nominal[key] == robustness.get(key) for key in keys),
             "phase-context-mismatch")
    _require(nominal["selection_policy"] == DEFAULT_FINAL_POLICY, "changed-frozen-policy")
    _require(nominal["environments"] == list(DEFAULT_ENVIRONMENTS), "changed-environments")
    hashes = nominal["checkpoint_sha256"]
    _require(set(hashes) == set(ARMS) and hashes["B0"] is None
             and hashes["incumbent"] == hashes["control"]
             and all(re.fullmatch("[a-f0-9]{64}", str(hashes[arm])) for arm in LEARNED_ARMS),
             "model-binding-mismatch")
    primary = nominal["offline_checkpoint_sha256"]
    _require(set(primary) == set(LEARNED_ARMS) and primary["incumbent"] == primary["control"]
             and all(re.fullmatch("[a-f0-9]{64}", str(value)) for value in primary.values())
             and re.fullmatch("[a-f0-9]{64}", str(nominal["training_audit_sha256"])),
             "audited-primary-to-compatible-lineage-required")
    _require(set(nominal["controller_configs"]) == set(ARMS), "controller-binding-mismatch")
    _require(all(nominal["controller_configs"][arm] == nominal["controller_configs"]["control"]
                 for arm in ARMS[1:]), "shared-candidate-controller-required")
    preprocessing = nominal["preprocessing_configs"]
    _require(type(preprocessing) is dict and set(preprocessing) == set(ARMS)
             and preprocessing["B0"] is None, "preprocessing-arm-coverage")
    from .path_correction import RGBPathCorrection
    base_correction_sha = RGBPathCorrection().config.sha256
    for arm in LEARNED_ARMS:
        config = preprocessing[arm]
        _require(type(config) is dict and config.get("correction") in ("path", "guarded_path")
                 and config.get("correction_config_sha256") == base_correction_sha,
                 "preprocessing-base-contract")
        expected = {"correction", "correction_config_sha256"}
        if config["correction"] == "guarded_path":
            expected |= {"guard_config", "guard_config_sha256"}
            from .guarded_path_correction import guard_config_from_dict
            guard_config_from_dict(config.get("guard_config"))
            _require(type(config.get("guard_config")) is dict and bool(config["guard_config"])
                     and config.get("guard_config_sha256") == canonical_sha256(config["guard_config"]),
                     "preprocessing-guard-contract")
        _require(set(config) == expected, "preprocessing-fields")
    _require(preprocessing["incumbent"]["correction"] == "path"
             and all(preprocessing[arm] == preprocessing["control"] for arm in ARMS[1:-1]),
             "shared-candidate-preprocessing-required")
    _require(type(nominal["representative_seed"]) is int and nominal["representative_seed"] >= 0,
             "representative-seed-missing")
    for protocol, phase, count in ((nominal, "final_nominal", 135),
                                   (robustness, "final_robustness", 90)):
        _require(protocol.get("phase") == phase and protocol.get("split") == "dev"
                 and protocol.get("town") == "Town01" and protocol.get("final_town_used") is False
                 and protocol.get("final_conditions_used") is False
                 and protocol.get("expected_episodes") == count, "invalid-development-phase")
    return hashes, primary


def _native(report, protocol):
    plan = protocol["planned_trials"]
    _require(report.get("status") == "PASS" and report.get("closed_loop_executed") is True
             and report.get("final_town_used") is False and report.get("final_conditions_used") is False
             and report.get("pointer_updated") is False
             and report.get("protocol_sha256") == canonical_sha256(protocol)
             and report.get("checkpoint_sha256") == protocol["checkpoint_sha256"]
             and report.get("preprocessing_configs") == protocol["preprocessing_configs"]
             and report.get("source_sha256") == protocol["source_sha256"]
             and report.get("planned_trials") == plan
             and report.get("episode_count") == report.get("expected_episode_count")
                 == protocol["expected_episodes"] == len(plan), "incomplete-or-unbound-native-report")
    _require({trial["arm"] for trial in plan} == set(ARMS), "native-arm-coverage")
    environment_names = {environment["name"] for environment in DEFAULT_ENVIRONMENTS}
    states = {}
    for trial in plan:
        _require(trial["case"] in CASES and trial["environment"] in DEFAULT_ENVIRONMENTS
                 and trial["method"] == ("B0" if trial["arm"] == "B0" else "M1"),
                 "native-case-environment-or-method")
        identity = canonical_sha256({key: trial[key] for key in
            ("case", "route_id", "condition", "environment", "initial_reference_m")})
        arm_states = states.setdefault(trial["arm"], set())
        _require(identity not in arm_states, "duplicate-physical-paired-state")
        arm_states.add(identity)
    _require(all(value == states["incumbent"] for value in states.values()), "unpaired-native-states")
    summary = summarize_stage(report["episodes"], protocol, expected_trials=plan)
    by_environment = {}
    for name in environment_names:
        selected = [trial for trial in plan if trial["environment"]["name"] == name]
        if not selected:
            continue
        episodes = [episode for episode in report["episodes"] if episode["environment"]["name"] == name]
        by_environment[name] = summarize_stage(episodes, protocol, expected_trials=selected)
    expected_environments = {"clear_nominal"} if protocol["phase"] == "final_nominal" else environment_names
    _require(set(by_environment) == expected_environments, "native-environment-coverage")
    expected_per_arm = 27 if protocol["phase"] == "final_nominal" else 18
    _require(all(summary[arm]["all"]["episodes"] == expected_per_arm for arm in ARMS),
             "native-episode-counts")
    return summary, by_environment


def _offline(reports, hashes):
    _require(set(reports) == set(LEARNED_ARMS), "offline-arm-coverage")
    groups = {}
    base = reports["incumbent"]
    for arm in LEARNED_ARMS:
        report = reports[arm]
        filtered = dict(report, metrics=[row for row in report["metrics"] if row.get("weather", "all") == "all"])
        groups[arm] = _groups(filtered)
        _require(report["checkpoint_sha256"] == hashes[arm]
                 and report.get("view") == "clean"
                 and canonical_sha256(report["evaluation_contract"]) == report["evaluation_contract_sha256"]
                 and all(report[key] == base[key] for key in
                     ("manifest_sha256", "sample_count", "evaluation_contract_sha256", "common_dev_selection_sha256",
                      "source_sha256", "protocol_sha256", "view", "validation_report_sha256"))
                 and all(groups[arm][case]["sample_count"] == groups["incumbent"][case]["sample_count"]
                         for case in ("all", *CASES)), "offline-common-DEV-or-model-mismatch")
    return groups


def _absolute(summary, policy):
    failed = []
    for case in ("all", *CASES):
        row = summary[case]
        if row["clean_completed"] != row["episodes"]:
            failed.append(case + ":incomplete-clean-driving")
        for key in ("collision_episodes", "lane_invasion_episodes", "body_lane_margin_negative_episodes"):
            if row[key]:
                failed.append(case + ":" + key)
        for key, limit in (("mean_lateral_m", policy["case_mean_max_m"]),
                           ("maximum_episode_p95_lateral_m", policy["episode_p95_max_m"]),
                           ("maximum_episode_processing_p95_ms", policy["processing_p95_max_ms"])):
            if row[key] is None or _finite(row[key]) > limit:
                failed.append(case + ":" + key)
        if row["episodes_with_steady_latency"] != row["episodes"]:
            failed.append(case + ":missing-steady-latency")
    return failed


def select_development_pipeline(offline_reports, nominal_report, robustness_report,
                                nominal_protocol, robustness_protocol):
    """Apply a predeclared development objective with explicit safety bounds."""
    hashes, primary_hashes = _shared_protocol(nominal_protocol, robustness_protocol)
    # Offline evidence keeps its original primary checkpoint identity. Native
    # runs use the separately audited compatible export; no metric report is
    # rewritten to impersonate that byte-distinct file.
    groups = _offline(offline_reports, primary_hashes)
    nominal, _ = _native(nominal_report, nominal_protocol)
    robust, environments = _native(robustness_report, robustness_protocol)
    policy = nominal_protocol["selection_policy"]
    shifted = sorted(set(environments) - {"clear_nominal"})
    scores = {arm: policy["nominal_weight"] * nominal[arm]["all"]["mean_lateral_m"]
        + policy["shift_weight"] * math.fsum(environments[name][arm]["all"]["mean_lateral_m"]
            for name in shifted) / len(shifted) for arm in ARMS}
    failed, eligible = {}, []
    for arm in LEARNED_ARMS[1:]:
        reasons = []
        for stage, summary in (("nominal", nominal), ("robustness", robust)):
            reasons.extend(stage + ":" + reason for reason in _absolute(summary[arm], policy))
            if summary[arm]["all"]["clean_completion_rate"] < summary["incumbent"]["all"]["clean_completion_rate"]:
                reasons.append(stage + ":clean-completion-regression")
        for name, summary in environments.items():
            reasons.extend(name + ":" + reason for reason in _absolute(summary[arm], policy))
        for case, group in groups[arm].items():
            if group["coverage"] != 1.:
                reasons.append(case + ":offline-coverage")
            if group["near_mean_m"] > policy["offline_near_mean_max_m"]:
                reasons.append(case + ":offline-near-bound")
            if group["far_mean_m"] > policy["offline_far_mean_max_m"]:
                reasons.append(case + ":offline-far-bound")
        if _finite(offline_reports[arm]["latency_ms"]["p95"]) > policy["forward_p95_max_ms"]:
            reasons.append("offline-forward-latency")
        required = max(policy["minimum_mean_improvement_m"],
                       policy["minimum_relative_improvement"] * scores["incumbent"])
        if scores["incumbent"] - scores[arm] < required:
            reasons.append("balanced-mean-improvement-below-practical-margin")
        if nominal[arm]["all"]["mean_lateral_m"] > nominal["incumbent"]["all"]["mean_lateral_m"] + policy["nominal_mean_regression_budget_m"]:
            reasons.append("nominal-overall-regression-budget")
        if (hashes[arm] == hashes["incumbent"]
                and nominal_protocol["controller_configs"][arm] == nominal_protocol["controller_configs"]["incumbent"]
                and nominal_protocol["preprocessing_configs"][arm] == nominal_protocol["preprocessing_configs"]["incumbent"]):
            reasons.append("identical-pipeline-is-not-an-improvement")
        failed[arm] = reasons
        if not reasons:
            eligible.append(arm)
    selected = min(eligible, key=lambda arm: (scores[arm], nominal[arm]["all"]["mean_lateral_m"],
        groups[arm]["all"]["mean_euclidean_m"], arm)) if eligible else "incumbent"
    return {"schema_version": 1, "status": "PASS", "selection_version": VERSION,
        "selected_arm": selected, "selected_checkpoint_sha256": hashes[selected],
        "selected_primary_checkpoint_sha256": primary_hashes[selected],
        "training_audit_sha256": nominal_protocol["training_audit_sha256"],
        "selected_controller_config": copy.deepcopy(nominal_protocol["controller_configs"][selected]),
        "selected_preprocessing_config": copy.deepcopy(nominal_protocol["preprocessing_configs"][selected]),
        "preprocessing_configs": copy.deepcopy(nominal_protocol["preprocessing_configs"]),
        "representative_seed": nominal_protocol["representative_seed"],
        "eligible_improvements": eligible, "failed_criteria": failed, "balanced_error_score_m": scores,
        "nominal": nominal, "robustness": robust, "by_environment": environments,
        "offline_mean_m": {arm: groups[arm]["all"]["mean_euclidean_m"] for arm in LEARNED_ARMS},
        "nominal_protocol_sha256": canonical_sha256(nominal_protocol),
        "robustness_protocol_sha256": canonical_sha256(robustness_protocol),
        "selection_policy": copy.deepcopy(policy), "final_town_used": False, "final_conditions_used": False,
        "pointer_updated": False,
        "scope": "short-correlated-Town01-development-routes-with-fixed-nominal-RGB-path-correction",
        "limitations": "development selection; two training seeds, one simulator configuration; no final-town generalization"}
