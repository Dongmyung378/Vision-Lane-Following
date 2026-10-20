"""Synthetic complete-plan selection fixtures; no experiment results or GPU."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import math
import unittest

from carla_tools.contracts import canonical_sha256
from carla_tools.controller import load_controller_config
from carla_tools.controller_study import ControllerStudyConfig
from carla_tools.development_selection import (
    ARMS, LEARNED_ARMS, final_selection_policy, select_development_pipeline,
)
from carla_tools.development_study import CASES, DEFAULT_ENVIRONMENTS
from carla_tools.refinement_driving import CONDITIONS


def checksum(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def offline_report(arm, checkpoint):
    counts = {"straight": 7900, "left_curve": 7800, "right_curve": 7944}
    contract = {"input": "RGB", "waypoints": 5, "near_indices": [0, 1], "far_indices": [2, 3, 4]}
    groups = [{"case": case, "condition": "all", "weather": "all", "sample_count": count,
               "predicted_count": count, "failed_prediction_count": 0, "coverage": 1.,
               "near_mean_m": .12, "far_mean_m": .32, "far_p95_m": .7,
               "mean_euclidean_m": (2 * .12 + 3 * .32) / 5}
              for case, count in (("all", sum(counts.values())), *counts.items())]
    return {"status": "PASS", "variant": "hybrid_geometry", "arm": arm, "split": "dev", "view": "clean",
            "fixture_injection": False, "final_town_used": False, "sample_count": sum(counts.values()),
            "manifest_sha256": checksum("common-DEV-manifest"), "config_sha256": checksum("learning-config"),
            "validation_report_sha256": checksum("independent-common-dataset-validation"),
            "checkpoint_sha256": checkpoint, "evaluation_contract": contract,
            "evaluation_contract_sha256": canonical_sha256(contract),
            "common_dev_selection_sha256": checksum("23644-common-sample-identities"),
            "protocol_sha256": checksum("common-development-training-protocol"),
            "source_sha256": {"synthetic_evaluation_source.py": checksum("common-evaluation-source")},
            "metrics": groups, "latency_ms": {"p95": 10.}}


def refresh_offline(report):
    """Keep exact weighted counts/means valid when testing one case's gate."""
    rows = {row["case"]: row for row in report["metrics"]}
    all_row = rows["all"]
    all_row["sample_count"] = sum(rows[case]["sample_count"] for case in CASES)
    all_row["predicted_count"] = sum(rows[case]["predicted_count"] for case in CASES)
    all_row["failed_prediction_count"] = sum(rows[case]["failed_prediction_count"] for case in CASES)
    for row in rows.values():
        row["coverage"] = row["predicted_count"] / row["sample_count"]
        row["mean_euclidean_m"] = (2 * row["near_mean_m"] + 3 * row["far_mean_m"]) / 5
    for field in ("near_mean_m", "far_mean_m", "mean_euclidean_m"):
        all_row[field] = math.fsum(rows[case][field] * rows[case]["predicted_count"] for case in CASES) / all_row["predicted_count"]
    report["sample_count"] = all_row["sample_count"]


def phase_fixture(shared, phase):
    protocol = copy.deepcopy(shared)
    protocol.update(phase=phase, split="dev", town="Town01", final_town_used=False,
                    final_conditions_used=False, expected_episodes=135 if phase == "final_nominal" else 90)
    starts = (0., 2., 4.) if phase == "final_nominal" else (4.,)
    environments = DEFAULT_ENVIRONMENTS[:1] if phase == "final_nominal" else DEFAULT_ENVIRONMENTS
    conditions = CONDITIONS if phase == "final_nominal" else CONDITIONS[:1]
    plan, episodes = [], []
    for case in CASES:
        for start in starts:
            for environment in environments:
                for condition in conditions:
                    for arm in ARMS:
                        trial = {"stage": phase, "arm": arm, "method": "B0" if arm == "B0" else "M1",
                            "controller": arm, "case": case, "route_id": "fixture_" + case,
                            "initial_reference_m": start, "condition": copy.deepcopy(condition),
                            "environment": copy.deepcopy(environment),
                            "episode": f"episodes/{phase}/{arm}/{case}__{start:g}__{environment['name']}__{condition['name']}"}
                        nominal = phase == "final_nominal" or environment["name"] == "clear_nominal"
                        values = {"incumbent": (.20, .30), "control": (.18, .28),
                                  "continued": (.17, .24), "appearance_robust": (.18, .20), "B0": (.001, .001)}
                        mean = values[arm][0 if nominal else 1]
                        plan.append(trial)
                        episodes.append(dict(copy.deepcopy(trial), execution={"status": "PASS"}, media={"status": "PASS"},
                            steady_processing_p95_ms=35., gate={"status": "PASS", "metrics": {
                                "driving_success": True, "termination_reason": "route_completed", "collision_count": 0,
                                "lane_invasion_count": 0, "minimum_body_lane_margin_m": .3,
                                "lateral_abs_m": {"mean": mean, "p95": .75}}}))
    protocol["planned_trials"] = plan
    report = {"status": "PASS", "declared_synthetic_selection_fixture": True, "closed_loop_executed": True,
        "final_town_used": False, "final_conditions_used": False, "pointer_updated": False,
        "protocol_sha256": canonical_sha256(protocol), "source_sha256": copy.deepcopy(protocol["source_sha256"]),
        "checkpoint_sha256": copy.deepcopy(protocol["checkpoint_sha256"]), "planned_trials": copy.deepcopy(plan),
        "preprocessing_configs": copy.deepcopy(protocol["preprocessing_configs"]),
        "episode_count": len(episodes), "expected_episode_count": len(episodes), "episodes": episodes}
    return report, protocol


def fixture():
    from carla_tools.path_correction import RGBPathCorrection
    base = load_controller_config()
    incumbent = ControllerStudyConfig(base=base).to_dict()
    candidate = ControllerStudyConfig(base=replace(base, lookahead_base_m=2.5, lookahead_min_m=3.,
                                                   lookahead_max_m=6.)).to_dict()
    native_hashes = {arm: checksum("compatible-" + arm) for arm in LEARNED_ARMS}
    native_hashes["control"] = native_hashes["incumbent"]
    native_hashes["B0"] = None
    primary_hashes = {arm: checksum("primary-" + arm) for arm in LEARNED_ARMS}
    primary_hashes["control"] = primary_hashes["incumbent"]
    shared = {"checkpoint_sha256": native_hashes, "offline_checkpoint_sha256": primary_hashes,
              "training_audit_sha256": checksum("independently-verified-primary-compatible-tensors"),
              "controller_configs": {arm: copy.deepcopy(incumbent if arm == "incumbent" else candidate) for arm in ARMS},
              "preprocessing_configs": {arm: None if arm == "B0" else {
                  "correction": "path", "correction_config_sha256": RGBPathCorrection().config.sha256}
                  for arm in ARMS},
              "representative_seed": 43, "training_comparison_sha256": checksum("fixed-comparison"),
              "parent_study_sha256": checksum("previous-stage-audit"), "routes_sha256": checksum("same-DEV-routes"),
              "source_sha256": {"synthetic_native_source.py": checksum("frozen-native-source")},
              "selection_policy": final_selection_policy(), "environments": copy.deepcopy(list(DEFAULT_ENVIRONMENTS))}
    reports = {arm: offline_report(arm, primary_hashes[arm]) for arm in LEARNED_ARMS}
    nominal, nominal_protocol = phase_fixture(shared, "final_nominal")
    robust, robust_protocol = phase_fixture(shared, "final_robustness")
    return reports, nominal, robust, nominal_protocol, robust_protocol


def means(inputs, arm, *, nominal=None, shifted=None, repeat=None, case=None):
    for index, phase in ((1, "final_nominal"), (2, "final_robustness")):
        for row in inputs[index]["episodes"]:
            if row["arm"] != arm or case is not None and row["case"] != case:
                continue
            value = nominal if phase == "final_nominal" else (
                repeat if row["environment"]["name"] == "clear_nominal" else shifted)
            if value is not None:
                row["gate"]["metrics"]["lateral_abs_m"]["mean"] = value


def target_only(inputs, target="appearance_robust"):
    for arm in LEARNED_ARMS[1:]:
        if arm != target:
            means(inputs, arm, nominal=.2, shifted=.3, repeat=.2)


def resign(inputs, phase_index):
    report, protocol = inputs[phase_index], inputs[phase_index + 2]
    report["protocol_sha256"] = canonical_sha256(protocol)


def guarded_preprocessing(inputs):
    from carla_tools.guarded_path_correction import guarded_preprocessing_config
    config = guarded_preprocessing_config()
    for phase_index in (1, 2):
        protocol = inputs[phase_index + 2]
        for arm in LEARNED_ARMS[1:]:
            protocol["preprocessing_configs"][arm] = copy.deepcopy(config)
        inputs[phase_index]["preprocessing_configs"] = copy.deepcopy(protocol["preprocessing_configs"])
        resign(inputs, phase_index)
    return config


class DevelopmentSelectionTests(unittest.TestCase):
    def test_actual_final_protocol_producer_and_selection_share_complete_plan_contract(self):
        from carla_tools.development_final_driving import freeze_final_protocol
        from tests.test_development_study import routes
        inputs = list(fixture())
        guarded = guarded_preprocessing(inputs)
        context = inputs[3]
        for phase_index, phase in ((1, "final_nominal"), (2, "final_robustness")):
            generated = freeze_final_protocol(routes(), context["checkpoint_sha256"], context["controller_configs"],
                phase=phase, representative_seed=context["representative_seed"],
                training_comparison_sha256=context["training_comparison_sha256"],
                parent_study_sha256=context["parent_study_sha256"],
                offline_checkpoint_sha256=context["offline_checkpoint_sha256"],
                training_audit_sha256=context["training_audit_sha256"],
                preprocessing_configs=context["preprocessing_configs"])
            key = lambda row: (row["arm"], row["case"], row["initial_reference_m"],
                               row["environment"]["name"], row["condition"]["name"])
            previous = {key(row): row for row in inputs[phase_index]["episodes"]}
            inputs[phase_index]["episodes"] = [dict(copy.deepcopy(previous[key(trial)]), **copy.deepcopy(trial))
                for trial in generated["planned_trials"]]
            inputs[phase_index]["planned_trials"] = copy.deepcopy(generated["planned_trials"])
            inputs[phase_index]["source_sha256"] = copy.deepcopy(generated["source_sha256"])
            inputs[phase_index]["preprocessing_configs"] = copy.deepcopy(generated["preprocessing_configs"])
            inputs[phase_index + 2] = generated
            resign(inputs, phase_index)
        result = select_development_pipeline(*inputs)
        self.assertEqual(result["selected_arm"], "appearance_robust")
        self.assertEqual(result["selected_preprocessing_config"], guarded)

    def test_complete_plans_distinct_primary_compatible_and_offline_ordering(self):
        inputs = list(fixture())
        self.assertEqual(len(inputs[1]["episodes"]), 135)
        self.assertEqual(len(inputs[2]["episodes"]), 90)
        self.assertEqual(inputs[0]["incumbent"]["sample_count"], 23644)
        self.assertNotEqual(inputs[3]["checkpoint_sha256"]["incumbent"],
                            inputs[3]["offline_checkpoint_sha256"]["incumbent"])
        inputs[0] = dict(reversed(list(inputs[0].items())))
        result = select_development_pipeline(*inputs)
        self.assertEqual(result["selected_arm"], "appearance_robust")
        self.assertEqual(result["selected_checkpoint_sha256"], inputs[3]["checkpoint_sha256"]["appearance_robust"])
        self.assertFalse(result["pointer_updated"])

    def test_balanced_shift_objective_can_accept_small_nominal_regression(self):
        inputs = fixture()
        target_only(inputs)
        means(inputs, "continued", nominal=.16, shifted=.28, repeat=.16)
        means(inputs, "appearance_robust", nominal=.205, shifted=.18, repeat=.205)
        result = select_development_pipeline(*inputs)
        self.assertEqual(result["selected_arm"], "appearance_robust")
        self.assertAlmostEqual(result["balanced_error_score_m"]["appearance_robust"], .1925)
        self.assertGreater(result["nominal"]["appearance_robust"]["all"]["mean_lateral_m"],
                           result["nominal"]["incumbent"]["all"]["mean_lateral_m"])

    def test_clear_robustness_repeat_is_retained_but_not_counted_twice_in_objective(self):
        inputs = fixture()
        first = select_development_pipeline(*inputs)
        means(inputs, "appearance_robust", repeat=.49)
        second = select_development_pipeline(*inputs)
        self.assertEqual(first["balanced_error_score_m"], second["balanced_error_score_m"])
        self.assertEqual(second["selected_arm"], "appearance_robust")
        self.assertAlmostEqual(second["by_environment"]["clear_nominal"]["appearance_robust"]["all"]["mean_lateral_m"], .49)
        means(inputs, "appearance_robust", repeat=.501)
        third = select_development_pipeline(*inputs)
        self.assertNotIn("appearance_robust", third["eligible_improvements"])

    def test_control_only_improvement_uses_same_offline_model_without_forcing_offline_win(self):
        inputs = fixture()
        target_only(inputs, "control")
        result = select_development_pipeline(*inputs)
        self.assertEqual(result["selected_arm"], "control")
        self.assertEqual(result["offline_mean_m"]["control"], result["offline_mean_m"]["incumbent"])

    def test_guarded_control_with_identical_weights_and_controller_can_win(self):
        inputs = fixture()
        target_only(inputs, "control")
        guarded = guarded_preprocessing(inputs)
        for protocol in inputs[3:]:
            original = protocol["controller_configs"]["incumbent"]
            protocol["controller_configs"] = {arm: copy.deepcopy(original) for arm in ARMS}
        resign(inputs, 1); resign(inputs, 2)
        result = select_development_pipeline(*inputs)
        self.assertEqual(result["selected_arm"], "control")
        self.assertEqual(result["selected_checkpoint_sha256"], inputs[3]["checkpoint_sha256"]["incumbent"])
        self.assertEqual(result["selected_controller_config"], inputs[3]["controller_configs"]["incumbent"])
        self.assertEqual(result["selected_preprocessing_config"], guarded)
        self.assertEqual(result["offline_mean_m"]["control"], result["offline_mean_m"]["incumbent"])
        result["selected_preprocessing_config"]["guard_config"]["maximum_first_point_change_m"] = .9
        self.assertEqual(inputs[3]["preprocessing_configs"]["control"], guarded)
        for phase_index in (1, 2):
            protocol = inputs[phase_index + 2]
            path = protocol["preprocessing_configs"]["incumbent"]
            for arm in LEARNED_ARMS[1:]:
                protocol["preprocessing_configs"][arm] = copy.deepcopy(path)
            inputs[phase_index]["preprocessing_configs"] = copy.deepcopy(protocol["preprocessing_configs"])
            resign(inputs, phase_index)
        result = select_development_pipeline(*inputs)
        self.assertEqual(result["selected_arm"], "incumbent")
        self.assertIn("identical-pipeline-is-not-an-improvement", result["failed_criteria"]["control"])

    def test_guarded_preprocessing_cross_phase_and_shared_candidate_context_must_match(self):
        from carla_tools.guarded_path_correction import GuardConfig, guarded_preprocessing_config
        alternative = guarded_preprocessing_config(GuardConfig(maximum_first_point_change_m=.4))
        for changed_phase, changed_arms in ((2, LEARNED_ARMS[1:]), (None, ("continued",)),
                                            (None, ("appearance_robust",)), (None, ("incumbent",))):
            inputs = fixture()
            guarded_preprocessing(inputs)
            for phase_index in ((changed_phase,) if changed_phase else (1, 2)):
                protocol = inputs[phase_index + 2]
                for arm in changed_arms:
                    protocol["preprocessing_configs"][arm] = copy.deepcopy(alternative)
                inputs[phase_index]["preprocessing_configs"] = copy.deepcopy(protocol["preprocessing_configs"])
                resign(inputs, phase_index)
            with self.subTest(phase=changed_phase, arms=changed_arms), self.assertRaises(ValueError):
                select_development_pipeline(*inputs)

    def test_guarded_preprocessing_report_map_cannot_be_missing_or_substituted(self):
        for phase in (1, 2):
            for missing in (False, True):
                inputs = fixture()
                guarded_preprocessing(inputs)
                if missing:
                    inputs[phase].pop("preprocessing_configs")
                else:
                    inputs[phase]["preprocessing_configs"]["control"] = copy.deepcopy(
                        inputs[phase]["preprocessing_configs"]["incumbent"])
                with self.subTest(phase=phase, missing=missing), self.assertRaises(ValueError):
                    select_development_pipeline(*inputs)

    def test_guarded_preprocessing_unknown_fields_versions_bounds_and_hashes_rejected(self):
        mutations = (lambda config: config.update(extra="unknown"),
                     lambda config: config.update(correction="unrecognized"),
                     lambda config: config.update(correction_config_sha256="0" * 64),
                     lambda config: config.update(guard_config_sha256="0" * 64),
                     lambda config: config["guard_config"].update(extra="unknown"),
                     lambda config: config["guard_config"].update(version="unknown"),
                     lambda config: config["guard_config"].update(maximum_first_point_change_m=True),
                     lambda config: config["guard_config"].update(maximum_first_point_change_m=.049),
                     lambda config: config["guard_config"].update(maximum_near_mean_change_m=1.001))
        for index, mutate in enumerate(mutations):
            inputs = fixture()
            guarded_preprocessing(inputs)
            for phase in (1, 2):
                config = copy.deepcopy(inputs[phase + 2]["preprocessing_configs"]["control"])
                mutate(config)
                if index >= 4:
                    config["guard_config_sha256"] = canonical_sha256(config["guard_config"])
                for arm in LEARNED_ARMS[1:]:
                    inputs[phase + 2]["preprocessing_configs"][arm] = copy.deepcopy(config)
                inputs[phase]["preprocessing_configs"] = copy.deepcopy(inputs[phase + 2]["preprocessing_configs"])
                resign(inputs, phase)
            with self.subTest(index=index), self.assertRaises(ValueError):
                select_development_pipeline(*inputs)

    def test_small_case_regression_inside_absolute_bounds_is_accepted(self):
        inputs = fixture()
        target_only(inputs)
        means(inputs, "appearance_robust", nominal=.13, shifted=.15, repeat=.13)
        means(inputs, "appearance_robust", nominal=.25, case="straight")
        result = select_development_pipeline(*inputs)
        self.assertEqual(result["selected_arm"], "appearance_robust")
        self.assertGreater(result["nominal"]["appearance_robust"]["straight"]["mean_lateral_m"],
                           result["nominal"]["incumbent"]["straight"]["mean_lateral_m"])

    def test_any_single_collision_lane_body_P95_latency_or_missing_latency_rejects(self):
        changes = (
            lambda row: row["gate"]["metrics"].update(collision_count=1),
            lambda row: row["gate"]["metrics"].update(lane_invasion_count=1),
            lambda row: row["gate"]["metrics"].update(minimum_body_lane_margin_m=-.001),
            lambda row: row["gate"]["metrics"]["lateral_abs_m"].update(p95=1.001),
            lambda row: row.update(steady_processing_p95_ms=50.01),
            lambda row: row.update(steady_processing_p95_ms=None),
            lambda row: row["gate"]["metrics"].update(driving_success=False, termination_reason="timeout"),
        )
        for phase in (1, 2):
            for change in changes:
                inputs = fixture()
                row = next(row for row in inputs[phase]["episodes"] if row["arm"] == "appearance_robust")
                change(row)
                with self.subTest(phase=phase, change=change):
                    result = select_development_pipeline(*inputs)
                    self.assertNotIn("appearance_robust", result["eligible_improvements"])
                    self.assertTrue(result["failed_criteria"]["appearance_robust"])

    def test_one_environment_case_mean_bound_cannot_be_hidden_in_aggregate(self):
        inputs = fixture()
        means(inputs, "appearance_robust", shifted=.10)
        for row in inputs[2]["episodes"]:
            if row["arm"] == "appearance_robust" and row["environment"]["name"] == "cloudy_nominal" and row["case"] == "left_curve":
                row["gate"]["metrics"]["lateral_abs_m"]["mean"] = .501
        result = select_development_pipeline(*inputs)
        self.assertNotIn("appearance_robust", result["eligible_improvements"])
        self.assertTrue(any("cloudy_nominal:left_curve" in reason for reason in result["failed_criteria"]["appearance_robust"]))

    def test_practical_margin_requires_both_two_millimetres_and_one_percent(self):
        for baseline, losing, winning in ((.10, .0981, .0979), (.25, .2476, .2474)):
            inputs = fixture()
            for arm in LEARNED_ARMS:
                means(inputs, arm, nominal=baseline, shifted=baseline, repeat=baseline)
            means(inputs, "appearance_robust", nominal=losing, shifted=losing, repeat=losing)
            first = select_development_pipeline(*inputs)
            self.assertEqual(first["selected_arm"], "incumbent")
            means(inputs, "appearance_robust", nominal=winning, shifted=winning, repeat=winning)
            second = select_development_pipeline(*inputs)
            self.assertEqual(second["selected_arm"], "appearance_robust")

    def test_nominal_overall_regression_budget_is_one_centimetre(self):
        inputs = fixture()
        target_only(inputs)
        means(inputs, "appearance_robust", nominal=.209, shifted=.10, repeat=.209)
        self.assertEqual(select_development_pipeline(*inputs)["selected_arm"], "appearance_robust")
        means(inputs, "appearance_robust", nominal=.211)
        result = select_development_pipeline(*inputs)
        self.assertEqual(result["selected_arm"], "incumbent")
        self.assertIn("nominal-overall-regression-budget", result["failed_criteria"]["appearance_robust"])

    def test_no_eligible_pipeline_retains_incumbent_and_B0_is_never_candidate(self):
        inputs = fixture()
        for arm in LEARNED_ARMS[1:]:
            means(inputs, arm, nominal=.2, shifted=.3, repeat=.2)
        result = select_development_pipeline(*inputs)
        self.assertEqual(result["selected_arm"], "incumbent")
        self.assertLess(result["balanced_error_score_m"]["B0"], result["balanced_error_score_m"]["incumbent"])
        self.assertNotIn("B0", result["eligible_improvements"])

    def test_identical_control_pipeline_is_rejected_despite_repeated_run_noise(self):
        inputs = fixture()
        target_only(inputs, "control")
        for protocol in inputs[3:]:
            original = protocol["controller_configs"]["incumbent"]
            protocol["controller_configs"] = {arm: copy.deepcopy(original) for arm in ARMS}
        resign(inputs, 1)
        resign(inputs, 2)
        result = select_development_pipeline(*inputs)
        self.assertEqual(result["selected_arm"], "incumbent")
        self.assertIn("identical-pipeline-is-not-an-improvement", result["failed_criteria"]["control"])

    def test_B0_failures_are_retained_as_reference_without_rejecting_safe_candidate(self):
        inputs = fixture()
        for report in inputs[1:3]:
            for row in report["episodes"]:
                if row["arm"] == "B0":
                    row["gate"]["metrics"].update(driving_success=False, termination_reason="timeout", collision_count=1)
                    row["steady_processing_p95_ms"] = None
        result = select_development_pipeline(*inputs)
        self.assertEqual(result["selected_arm"], "appearance_robust")
        self.assertEqual(result["nominal"]["B0"]["all"]["clean_completed"], 0)
        self.assertEqual(result["nominal"]["B0"]["all"]["collision_episodes"], 27)

    def test_native_missing_duplicate_integrity_or_changed_episode_rejected(self):
        changes = (lambda report: report["episodes"].pop(),
                   lambda report: report["episodes"].__setitem__(1, copy.deepcopy(report["episodes"][0])),
                   lambda report: report["episodes"][0]["gate"].update(status="FAIL"),
                   lambda report: report["episodes"][0].update(initial_reference_m=.25),
                   lambda report: report.update(episode_count=1),
                   lambda report: report.update(protocol_sha256="0" * 64),
                   lambda report: report["source_sha256"].update(synthetic_native_source="0" * 64),
                   lambda report: report["checkpoint_sha256"].update(continued="0" * 64))
        for change in changes:
            inputs = fixture()
            change(inputs[1])
            with self.subTest(change=change), self.assertRaises(ValueError):
                select_development_pipeline(*inputs)

    def test_consistently_rebound_single_arm_state_drift_and_duplicate_state_rejected(self):
        for duplicate in (False, True):
            inputs = fixture()
            report, protocol = inputs[1], inputs[3]
            index = next(index for index, row in enumerate(protocol["planned_trials"]) if row["arm"] == "appearance_robust")
            changed = copy.deepcopy(protocol["planned_trials"][index])
            changed["initial_reference_m"] = .25
            if duplicate:
                previous = next(row for row in protocol["planned_trials"][index + 1:] if row["arm"] == "appearance_robust")
                for field in ("case", "route_id", "condition", "environment", "initial_reference_m"):
                    changed[field] = copy.deepcopy(previous[field])
            protocol["planned_trials"][index] = changed
            report["planned_trials"][index] = copy.deepcopy(changed)
            report["episodes"][index].update(copy.deepcopy(changed))
            resign(inputs, 1)
            with self.subTest(duplicate=duplicate), self.assertRaises(ValueError):
                select_development_pipeline(*inputs)

    def test_candidate_and_reference_B0_share_exact_common_controller(self):
        for arm in ("continued", "appearance_robust", "B0"):
            inputs = fixture()
            for protocol in inputs[3:]:
                protocol["controller_configs"][arm]["steer_smoothing_alpha"] = .5
            resign(inputs, 1); resign(inputs, 2)
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, "shared-candidate-controller"):
                select_development_pipeline(*inputs)

    def test_primary_hash_compatible_hash_and_training_audit_phase_mismatches_rejected(self):
        changes = (lambda inputs: inputs[0]["continued"].update(checkpoint_sha256=inputs[3]["checkpoint_sha256"]["continued"]),
                   lambda inputs: inputs[4]["offline_checkpoint_sha256"].update(continued="0" * 64),
                   lambda inputs: inputs[4].update(training_audit_sha256="0" * 64),
                   lambda inputs: inputs[3].update(training_audit_sha256="not-a-sha256"))
        for change in changes:
            inputs = fixture()
            change(inputs)
            with self.subTest(change=change), self.assertRaises(ValueError):
                select_development_pipeline(*inputs)

    def test_offline_case_bounds_coverage_and_forward_latency_are_absolute(self):
        for field, value in (("near_mean_m", .301), ("far_mean_m", 1.001), ("failed_prediction_count", 1)):
            inputs = fixture()
            report = inputs[0]["appearance_robust"]
            group = next(row for row in report["metrics"] if row["case"] == "straight")
            group[field] = value
            if field == "failed_prediction_count":
                group["predicted_count"] -= 1
            refresh_offline(report)
            with self.subTest(field=field):
                result = select_development_pipeline(*inputs)
                self.assertNotIn("appearance_robust", result["eligible_improvements"])
        inputs = fixture()
        inputs[0]["appearance_robust"]["latency_ms"]["p95"] = 50.01
        self.assertNotIn("appearance_robust", select_development_pipeline(*inputs)["eligible_improvements"])

    def test_offline_common_sample_identity_model_and_evaluation_contract_tamper_rejected(self):
        changes = (lambda report: report.update(common_dev_selection_sha256="0" * 64),
                   lambda report: report.update(manifest_sha256="0" * 64),
                   lambda report: report.update(validation_report_sha256="0" * 64),
                   lambda report: report.update(checkpoint_sha256="0" * 64),
                   lambda report: report["evaluation_contract"].update(waypoints=4),
                   lambda report: report.update(evaluation_contract_sha256="0" * 64),
                   lambda report: report.update(final_town_used=True),
                   lambda report: report.update(fixture_injection=True))
        for change in changes:
            inputs = fixture()
            change(inputs[0]["appearance_robust"])
            with self.subTest(change=change), self.assertRaises(ValueError):
                select_development_pipeline(*inputs)
        inputs = fixture()
        report = inputs[0]["appearance_robust"]
        report["evaluation_contract"]["waypoints"] = 4
        report["evaluation_contract_sha256"] = canonical_sha256(report["evaluation_contract"])
        with self.assertRaises(ValueError):
            select_development_pipeline(*inputs)

    def test_offline_training_source_protocol_or_nonclean_view_substitution_rejected(self):
        for key, value in (("source_sha256", {"different.py": "0" * 64}),
                           ("protocol_sha256", "0" * 64), ("view", "dim"),
                           ("validation_report_sha256", "0" * 64)):
            inputs = fixture()
            inputs[0]["appearance_robust"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                select_development_pipeline(*inputs)

    def test_frozen_policy_environment_town_and_cross_phase_context_cannot_change(self):
        changes = (lambda protocol: protocol["selection_policy"].update(nominal_weight=.6),
                   lambda protocol: protocol["environments"].reverse(),
                   lambda protocol: protocol.update(town="Town02"),
                   lambda protocol: protocol.update(final_conditions_used=True),
                   lambda protocol: protocol.update(parent_study_sha256="0" * 64))
        for change in changes:
            inputs = fixture()
            change(inputs[3])
            resign(inputs, 1)
            with self.subTest(change=change), self.assertRaises(ValueError):
                select_development_pipeline(*inputs)

    def test_nonfinite_or_boolean_metrics_rejected_and_policy_copy_is_independent(self):
        for value in (float("nan"), float("inf"), True, -.1):
            inputs = fixture()
            inputs[1]["episodes"][0]["gate"]["metrics"]["lateral_abs_m"]["mean"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_development_pipeline(*inputs)
        changed = final_selection_policy()
        changed["nominal_weight"] = 0.
        self.assertEqual(final_selection_policy()["nominal_weight"], .5)


if __name__ == "__main__":
    unittest.main()
