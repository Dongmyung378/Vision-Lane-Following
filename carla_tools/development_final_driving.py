"""Freeze paired pipeline confirmation without reopening independent final data.

Nominal and shifted development phases are separate bounded GPU jobs. Both
bind the same models, chosen controller, routes and predeclared selection rule.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import re

from .development_study import (CASES, DEFAULT_ENVIRONMENTS, LATENCY_SCOPE,
    _routes, digest, run_study_trial, study_source_hashes, validate_study_episode)
from .development_selection import ARMS, final_selection_policy
from .refinement_driving import CONDITIONS

VERSION = "paired-development-pipeline-confirmation-v1"
PHASES = {"final_nominal": 135, "final_robustness": 90}


def final_source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ("carla_tools/development_final_driving.py", "carla_tools/development_selection.py",
        "scripts/run_development_final.py", "scripts/prepare_kaggle_development_final.py",
        "carla_tools/guarded_path_correction.py")
    return dict(study_source_hashes(), **{name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names})


def freeze_final_protocol(routes, checkpoint_sha256, controller_configs, *, phase,
        representative_seed, training_comparison_sha256, parent_study_sha256,
        offline_checkpoint_sha256, training_audit_sha256, preprocessing_configs=None):
    from .baseline import load_baseline_config
    from .config import load_collection_config
    from .controller import load_controller_config
    from .controller_study import ControllerStudyConfig, study_controller_config_from_dict
    from .evaluation import load_evaluation_config
    from .path_correction import RGBPathCorrection
    from .guarded_path_correction import path_preprocessing_config, validate_preprocessing_config
    _routes(routes)
    if (phase not in PHASES or type(representative_seed) is not int or representative_seed not in (42, 43)
            or type(checkpoint_sha256) is not dict or set(checkpoint_sha256) != set(ARMS)
            or checkpoint_sha256["B0"] is not None or checkpoint_sha256["incumbent"] != checkpoint_sha256["control"]
            or type(offline_checkpoint_sha256) is not dict or set(offline_checkpoint_sha256) != set(ARMS[:-1])
            or offline_checkpoint_sha256["incumbent"] != offline_checkpoint_sha256["control"]
            or any(not re.fullmatch("[0-9a-f]{64}", str(value)) for value in
                (training_comparison_sha256, parent_study_sha256, training_audit_sha256,
                    *offline_checkpoint_sha256.values(), *(checkpoint_sha256[arm] for arm in ARMS[:-1])))):
        raise ValueError("development-final-frozen-model-context-invalid")
    if type(controller_configs) is not dict or set(controller_configs) != set(ARMS):
        raise ValueError("development-final-controller-arm-coverage-invalid")
    configs = {arm: study_controller_config_from_dict(value) for arm, value in controller_configs.items()}
    if (configs["incumbent"] != ControllerStudyConfig(base=load_controller_config())
            or any(configs[arm] != configs["control"] for arm in ARMS[1:])):
        raise ValueError("development-final-candidate-controller-must-be-shared")
    if preprocessing_configs is None:
        preprocessing_configs = {arm: None if arm == "B0" else path_preprocessing_config() for arm in ARMS}
    if (type(preprocessing_configs) is not dict or set(preprocessing_configs) != set(ARMS)
            or preprocessing_configs["B0"] is not None):
        raise ValueError("development-final-preprocessing-arm-coverage-invalid")
    preprocessing_configs = {arm: None if arm == "B0" else validate_preprocessing_config(value)
        for arm, value in preprocessing_configs.items()}
    if (preprocessing_configs["incumbent"] != path_preprocessing_config()
            or any(preprocessing_configs[arm] != preprocessing_configs["control"] for arm in ARMS[1:-1])):
        raise ValueError("development-final-incumbent-or-shared-candidate-preprocessing-invalid")
    collection = load_collection_config(Path(__file__).resolve().parents[1] / "configs/collection_d4.json")
    protocol = {"schema_version": 1, "protocol_version": VERSION, "phase": phase, "split": "dev", "town": "Town01",
        "final_town_used": False, "final_conditions_used": False,
        "scope": "paired-short-correlated-development-pipeline-confirmation-not-independent-final-evaluation",
        "checkpoint_sha256": copy.deepcopy(checkpoint_sha256), "controller_configs": copy.deepcopy(controller_configs),
        "preprocessing_configs": copy.deepcopy(preprocessing_configs),
        "controller_config_sha256": {arm: config.sha256 for arm, config in configs.items()},
        "representative_seed": representative_seed, "training_comparison_sha256": training_comparison_sha256,
        "offline_checkpoint_sha256": copy.deepcopy(offline_checkpoint_sha256), "training_audit_sha256": training_audit_sha256,
        "parent_study_sha256": parent_study_sha256, "routes_sha256": digest(routes), "source_sha256": final_source_hashes(),
        "environments": copy.deepcopy(list(DEFAULT_ENVIRONMENTS)), "initial_conditions": copy.deepcopy(list(CONDITIONS)),
        "expected_episodes": PHASES[phase], "goal_reference_m": 19., "timeout_seconds": 10.,
        "selection_policy": final_selection_policy(), "collection_base_config_sha256": collection.sha256,
        "configured_collection_seed": collection.seed, "base_controller_config_sha256": load_controller_config().sha256,
        "baseline_config_sha256": load_baseline_config().sha256,
        "evaluation_config_sha256": replace(load_evaluation_config(), timeout_seconds=10.).sha256,
        "correction_config_sha256": RGBPathCorrection().config.sha256,
        "correction": "per-arm-frozen-RGB-path-fusion-with-identical-nominal-camera-calibration",
        "perception_inputs": "RGB-only; speed-to-controller-only; no-native-calibration-or-route-or-pose",
        "latency_scope": LATENCY_SCOPE, "simulator_repeats_per_state": 1, "physics_determinism_claimed": False}
    protocol["planned_trials"] = planned_final_trials(protocol, routes)
    return protocol


def planned_final_trials(protocol, routes):
    phase = protocol["phase"]
    if phase not in PHASES:
        raise ValueError("development-final-phase-invalid")
    if phase == "final_nominal":
        environments, starts, conditions = [protocol["environments"][0]], (0., 2., 4.), CONDITIONS
    else:
        environments, starts, conditions = protocol["environments"], (4.,), (CONDITIONS[0],)
    trials, state = [], 0
    for record in sorted(routes["routes"], key=lambda row: CASES.index(row["case"])):
        for environment in environments:
            for start in starts:
                for condition in conditions:
                    for arm in (ARMS if state % 2 == 0 else tuple(reversed(ARMS))):
                        trials.append({"stage": phase, "arm": arm, "method": "B0" if arm == "B0" else "M1",
                            "controller": arm, "case": record["case"], "route_id": record["route"]["route_id"],
                            "initial_reference_m": start, "condition": copy.deepcopy(condition),
                            "environment": copy.deepcopy(environment),
                            "episode": f"episodes/{phase}/{arm}/{record['case']}__start{int(start):02d}__{environment['name']}__{condition['name']}"})
                    state += 1
    if len(trials) != protocol["expected_episodes"]:
        raise ValueError("development-final-paired-state-budget-mismatch")
    return trials


def episode_protocol(protocol, trial):
    """Bind the unchanged native replay contract to this arm's exact weights."""
    if trial not in protocol["planned_trials"]:
        raise ValueError("development-final-trial-not-in-frozen-plan")
    result = copy.deepcopy(protocol)
    result["parent_final_protocol_sha256"] = digest(protocol)
    # B0 has no checkpoint; this field is never consulted by B0 perception.
    result["checkpoint_sha256"] = protocol["checkpoint_sha256"][trial["arm"]]
    return result


def run_final_trial(route, trial, output, collection_config, *, predictor, protocol,
        carla_module, baseline_config, evaluation_config):
    return run_study_trial(route, trial, output, collection_config, predictor=predictor,
        protocol=episode_protocol(protocol, trial), carla_module=carla_module,
        baseline_config=baseline_config, evaluation_config=evaluation_config)


def validate_final_episode(directory, predictor, trial, *, protocol, routes):
    if (protocol.get("routes_sha256") != digest(routes) or protocol.get("source_sha256") != final_source_hashes()
            or protocol.get("planned_trials") != planned_final_trials(protocol, routes)):
        return {"status": "FAIL", "errors": ["development-final-source-routes-or-trials-changed"]}
    return _validate_final_perception(directory, predictor, trial,
        protocol=episode_protocol(protocol, trial), routes=routes)


def _validate_final_perception(directory, predictor, trial, *, protocol, routes):
    """Independently reproduce RGB perception before replaying native controls."""
    from .development_study import SCOPE, LATENCY_SCOPE, _routes, _validate_study_native
    from .contracts import strict_json_loads
    from .driving_validation import _require, _same
    from .baseline_validation import _safe_file
    from .projection import decode_rgb8_png
    from .guarded_path_correction import validate_recorded_guarded_correction
    from .learning_data import validate_waypoint_points
    from .pipeline_latency import validate_pipeline_timing
    from .refinement_validation import _RecordedReplay
    from .runtime_driving import validate_recorded_correction, validate_runtime_timing
    root = Path(directory)
    report = {"status": "FAIL", "scope": SCOPE, "errors": []}
    try:
        _routes(routes)
        _require(protocol["routes_sha256"] == digest(routes), "development-routes-changed")
        metadata_bytes = _safe_file(root, "episode.json").read_bytes()
        trace_bytes = _safe_file(root, "steps.jsonl").read_bytes()
        metadata = strict_json_loads(metadata_bytes.decode())
        observed_identity, identity = copy.deepcopy(metadata["native_model"]), predictor.identity
        observed_device = observed_identity.pop("device", None)
        identity.pop("device", None)
        _require(observed_device in ("cpu", "cuda") and _same(observed_identity, identity, 1e-12),
            "development-perception-identity-mismatch")
        _require(metadata.get("requested_initial_reference_m") == trial["initial_reference_m"]
            and metadata.get("detailed_latency_required") is True and metadata.get("latency_scope") == LATENCY_SCOPE,
            "development-start-or-latency-scope-mismatch")
        _require(metadata.get("evaluation_config_sha256") == protocol["evaluation_config_sha256"]
            and metadata.get("baseline_config_sha256") == protocol["baseline_config_sha256"]
            and metadata.get("controller_config_sha256") == protocol["base_controller_config_sha256"],
            "development-frozen-configuration-mismatch")
        if trial["method"] == "M1":
            _require(identity.get("checkpoint_sha256") == protocol["checkpoint_sha256"]
                and predictor.correction == protocol["preprocessing_configs"][trial["arm"]]["correction"],
                "development-final-M1-checkpoint-or-correction-mismatch")
            from .guarded_path_correction import guarded_preprocessing_config, path_preprocessing_config
            actual_preprocessing = (guarded_preprocessing_config(predictor.config)
                if predictor.correction == "guarded_path" else path_preprocessing_config())
            _require(actual_preprocessing == protocol["preprocessing_configs"][trial["arm"]],
                "development-final-frozen-preprocessing-config-mismatch")
            base = predictor.base if predictor.correction == "guarded_path" else predictor
            _require(base.corrector.config.sha256 == actual_preprocessing["correction_config_sha256"],
                "development-final-nominal-path-config-mismatch")
        else:
            _require(trial["method"] == "B0" and identity.get("variant") == "B0",
                "development-method-invalid")
        rows = [strict_json_loads(line) for line in trace_bytes.decode().splitlines()]
        _require(rows and len(rows) == metadata["step_count"], "development-frame-count-mismatch")
        frames, maximum = [], 0.
        for row in rows:
            image = row["image"]
            _require(image["path"] == f"images/{row['frame']:08d}.png", "development-original-image-path-mismatch")
            width, height, rgb = decode_rgb8_png(_safe_file(root, image["path"]).read_bytes(), image["sha256"], (640, 360))
            reproduced = predictor.predict(rgb, width=width, height=height)
            observed = row["perception"]
            observed_rgb = copy.deepcopy(observed)
            rejection = observed_rgb.pop("controller_rejection", None)
            if rejection is not None:
                fields = {"exception_type", "error_code", "waypoints_ego_m"}
                if trial["method"] == "M1":
                    fields.add("learned_waypoints_ego_m")
                _require(type(rejection) is dict and set(rejection) == fields
                    and rejection["exception_type"] == "ControllerError" and type(rejection["error_code"]) is str
                    and bool(rejection["error_code"]) and observed["status"] == "FAIL"
                    and observed["error"] == "development-controller-rejection" and observed["waypoints_ego_m"] is None,
                    "development-controller-rejection-schema-invalid")
                observed_rgb.update(status="PASS", error=None, waypoints_ego_m=rejection["waypoints_ego_m"])
                if trial["method"] == "M1":
                    _require(observed.get("learned_waypoints_ego_m") is None, "development-rejected-raw-path-not-separated")
                    observed_rgb["learned_waypoints_ego_m"] = rejection["learned_waypoints_ego_m"]
            timing = observed["pipeline_timing"]
            validate_pipeline_timing(timing)
            _require(timing["device"] == observed_device, "development-timing-device-mismatch")
            ignored = ("timing_ms", "pipeline_timing", "waypoints_ego_m", "learned_waypoints_ego_m")
            observed_content = {key: copy.deepcopy(value) for key, value in observed_rgb.items() if key not in ignored}
            reproduced_content = {key: copy.deepcopy(value) for key, value in reproduced.items() if key not in ignored}
            if trial["method"] == "M1":
                validate_runtime_timing(timing, row["latency_ms"], observed["status"])
                if rejection is not None:
                    from .pipeline_latency import STAGES
                    _require(timing["executed_stages"] == list(STAGES) and timing["failed_stages"] == ["controller_ms"]
                        and (timing["device"] != "cuda" or set(timing["synchronized_stages"]) ==
                            {"learned_preprocessing_ms", "forward_decode_ms"}), "development-controller-rejection-timing-invalid")
                if observed_rgb["status"] == "PASS":
                    validate_waypoint_points(observed_rgb["learned_waypoints_ego_m"])
                    validate_waypoint_points(reproduced["learned_waypoints_ego_m"])
                    difference = max(abs(a - b) for p, q in zip(observed_rgb["learned_waypoints_ego_m"],
                        reproduced["learned_waypoints_ego_m"]) for a, b in zip(p, q))
                    _require(difference <= .05, "development-raw-learned-replay-mismatch")
                    if predictor.correction == "guarded_path":
                        validate_recorded_guarded_correction(predictor, rgb, observed_rgb, width=width, height=height)
                        observed_content["diagnostics"].pop("guarded_path_correction")
                        reproduced_content["diagnostics"].pop("guarded_path_correction")
                    else:
                        validate_recorded_correction(predictor, rgb, observed_rgb, width=width, height=height)
                    observed_content["diagnostics"].pop("path_correction")
                    reproduced_content["diagnostics"].pop("path_correction")
            else:
                expected_stages = ["bgra_conversion_ms", "forward_decode_ms"]
                if observed_rgb["status"] == "PASS":
                    expected_stages.append("controller_ms")
                _require(timing["executed_stages"] == expected_stages and timing["device"] == "cpu"
                    and timing["failed_stages"] == (["controller_ms"] if rejection is not None else []),
                    "development-B0-stage-coverage-mismatch")
                legacy, detail = row["latency_ms"], timing["stages_ms"]
                _require(timing["total_processing_ms"] + 1e-5 >= legacy["pipeline_ms"]
                    and detail["bgra_conversion_ms"] <= legacy["preprocess_ms"] + 1e-5
                    and detail["forward_decode_ms"] <= legacy["inference_ms"] + 1e-5
                    and detail["controller_ms"] <= legacy["controller_ms"] + 1e-5,
                    "development-B0-detailed-interval-mismatch")
            _require(_same(observed_content, reproduced_content, 1e-8), "development-RGB-content-replay-mismatch")
            if observed_rgb["status"] == "PASS":
                validate_waypoint_points(observed_rgb["waypoints_ego_m"])
                validate_waypoint_points(reproduced["waypoints_ego_m"])
                difference = max(abs(a - b) for p, q in zip(observed_rgb["waypoints_ego_m"],
                    reproduced["waypoints_ego_m"]) for a, b in zip(p, q))
                _require(difference <= (.05 if trial["method"] == "M1" else 1e-8),
                    "development-RGB-waypoint-replay-mismatch")
                maximum = max(maximum, difference)
            else:
                _require(observed["waypoints_ego_m"] is None and reproduced["waypoints_ego_m"] is None,
                    "development-failed-perception-retained-waypoints")
                if trial["method"] == "M1":
                    _require(observed.get("learned_waypoints_ego_m") is None
                        and reproduced.get("learned_waypoints_ego_m") is None,
                        "development-failed-perception-retained-raw-waypoints")
            frames.append({"rgb_sha256": hashlib.sha256(rgb).hexdigest(), "perception": observed})
        replay = _RecordedReplay(frames)
        native = _validate_study_native(root, protocol=protocol, trial=trial, routes=routes, predictor=replay)
        _require(native["status"] == "PASS", "development-native-replay-failed:" + str(native.get("errors")))
        _require(replay.index == len(rows), "development-native-RGB-replay-incomplete")
        _require(_safe_file(root, "episode.json").read_bytes() == metadata_bytes
            and _safe_file(root, "steps.jsonl").read_bytes() == trace_bytes, "development-sources-changed")
        report.update(status="PASS", frame_count=len(rows), native=native,
            maximum_coordinate_replay_difference_m=maximum, metrics=metadata["evaluation_metrics"],
            driving_success=metadata["evaluation_metrics"]["driving_success"], closed_loop_executed=True)
    except Exception as error:
        report["errors"].append(str(error))
    return report
