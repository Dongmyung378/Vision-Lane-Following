"""Paired development-only controller and camera robustness experiments.

The protocol reserves final towns, difficult rain, sunset and camera pitch.
Selection uses earlier development stages only; every executed failure stays
in the result. Camera readbacks and ground truth never enter RGB perception.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
import time

from .contracts import canonical_sha256, route_sha256, strict_json_loads
from .refinement_driving import CONDITIONS

VERSION = "paired-controller-camera-development-v1"
SCOPE = VERSION
CASES = ("straight", "left_curve", "right_curve")
WEATHERS = ("ClearNoon", "CloudyNoon", "SoftRainNoon")
STAGES = ("screen", "tune", "confirmation", "robustness")
DEFAULT_ENVIRONMENTS = (
    {"name": "clear_nominal", "weather": "ClearNoon", "camera_yaw_deg": 0., "camera_height_delta_m": 0.},
    {"name": "cloudy_nominal", "weather": "CloudyNoon", "camera_yaw_deg": 0., "camera_height_delta_m": 0.},
    {"name": "soft_rain_nominal", "weather": "SoftRainNoon", "camera_yaw_deg": 0., "camera_height_delta_m": 0.},
    {"name": "clear_yaw_plus", "weather": "ClearNoon", "camera_yaw_deg": 1., "camera_height_delta_m": 0.},
    {"name": "clear_yaw_minus", "weather": "ClearNoon", "camera_yaw_deg": -1., "camera_height_delta_m": 0.},
    {"name": "clear_height_plus", "weather": "ClearNoon", "camera_yaw_deg": 0., "camera_height_delta_m": .05},
)
LATENCY_SCOPE = ("sensor-delivered BGRA conversion, RGB perception and shared study controller; "
    "M1 retains synchronized learned preprocessing/forward and fixed nominal RGB path correction; "
    "B0 forward_decode_ms contains CPU OpenCV; excludes simulator wait, GT generation, disk and video")


def digest(value):
    return canonical_sha256(value)


def study_source_hashes():
    from .recovery_comparison import comparison_source_hashes
    root = Path(__file__).resolve().parents[1]
    names = ("carla_tools/development_study.py", "carla_tools/controller_study.py",
        "scripts/run_development_study.py", "scripts/prepare_kaggle_development_study.py")
    return dict(comparison_source_hashes(), **{name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in names})


def _controllers(values):
    from .controller_study import study_controller_config_from_dict
    from .controller import load_controller_config
    if (type(values) is not dict or len(values) != 6 or "incumbent" not in values
            or any(type(name) is not str or not name.replace("_", "").isalnum() for name in values)):
        raise ValueError("development-six-named-controller-configs-required")
    configs = {name: study_controller_config_from_dict(value) for name, value in values.items()}
    baseline = configs["incumbent"]
    if (baseline.method != "pure_pursuit" or baseline.base != load_controller_config()
            or baseline.steer_smoothing_alpha != 1.):
        raise ValueError("development-incumbent-controller-must-match-current-default")
    if len({config.sha256 for config in configs.values()}) != 6:
        raise ValueError("development-controller-configs-must-be-distinct")
    return configs


def _routes(value):
    if (type(value) is not dict or value.get("split") != "dev" or value.get("final_town_used") is not False
            or type(value.get("routes")) is not list or len(value["routes"]) != 3
            or {row.get("case") for row in value["routes"]} != set(CASES)
            or len({row["route"]["route_id"] for row in value["routes"]}) != 3
            or any(row.get("split") != "dev" or row["route"].get("town") != "Town01"
                or route_sha256(row["route"]) != row["route"].get("route_sha256") for row in value["routes"])):
        raise ValueError("development-three-source-bound-Town01-DEV-routes-required")
    return value


def freeze_development_protocol(routes, checkpoint_sha256, controller_configs, *, environments=None):
    """Declare the complete search and stage decision rules before simulation."""
    import re
    from .baseline import load_baseline_config
    from .config import load_collection_config
    from .controller import load_controller_config
    from .evaluation import load_evaluation_config
    from .path_correction import RGBPathCorrection
    _routes(routes)
    configs = _controllers(controller_configs)
    if not re.fullmatch("[0-9a-f]{64}", str(checkpoint_sha256)):
        raise ValueError("development-checkpoint-hash-invalid")
    environments = copy.deepcopy(list(DEFAULT_ENVIRONMENTS if environments is None else environments))
    if environments != list(DEFAULT_ENVIRONMENTS):
        raise ValueError("development-reserved-final-conditions-or-environment-grid-changed")
    collection = load_collection_config(Path(__file__).resolve().parents[1] / "configs/collection_d4.json")
    protocol = {"schema_version": 1, "protocol_version": VERSION, "split": "dev", "town": "Town01",
        "final_town_used": False, "final_conditions_used": False,
        "scope": "short-correlated-Town01-development-only-not-independent-final-evaluation",
        "checkpoint_sha256": checkpoint_sha256, "routes_sha256": digest(routes),
        "source_sha256": study_source_hashes(), "controller_configs": copy.deepcopy(controller_configs),
        "controller_config_sha256": {name: cfg.sha256 for name, cfg in configs.items()},
        "incumbent_controller": "incumbent", "environments": environments,
        "stage_starts_m": {"screen": 0., "tune": 2., "confirmation": 4., "robustness": 4.},
        "initial_conditions": copy.deepcopy(list(CONDITIONS)), "goal_reference_m": 19., "timeout_seconds": 10.,
        "expected_stage_episodes": {"screen": 54, "tune": 18, "confirmation": 27, "robustness": 54},
        "expected_episodes": 153, "collection_base_config_sha256": collection.sha256,
        "configured_collection_seed": collection.seed, "simulator_repeats_per_state": 1,
        "physics_determinism_claimed": False, "base_controller_config_sha256": load_controller_config().sha256,
        "baseline_config_sha256": load_baseline_config().sha256,
        "evaluation_config_sha256": replace(load_evaluation_config(), timeout_seconds=10.).sha256,
        "correction": "identical-frozen-nominal-RGB-path-correction-for-all-M1-arms",
        "correction_config_sha256": RGBPathCorrection().config.sha256,
        "perception_inputs": "RGB-only; speed-to-controller-only; no-native-calibration-or-route-or-pose",
        "latency_scope": LATENCY_SCOPE,
        "selection_policy": {"primary_metric": "equal-episode-mean-absolute-lateral-error-m",
            "case_mean_max_m": .5, "episode_p95_max_m": 1., "processing_p95_max_ms": 50.,
            "minimum_mean_improvement_m": .002, "minimum_relative_mean_improvement": .01,
            "collision_count_max": 0, "lane_invasion_count_max": 0,
            "negative_body_margin_episodes_max": 0, "clean_completion_not_lower_than_incumbent": True,
            "screen_rule": "rank-nonincumbents-by-clean-completion-desc-then-mean-lateral-then-id; keep-incumbent-plus-one",
            "tune_rule": "candidate-absolute-gates-and-mean-improvement-over-incumbent-otherwise-retain-incumbent",
            "confirmation_rule": "same-absolute-gates-and-mean-improvement; no-casewise-relative-dominance",
            "robustness_rule": "retain-all-failures; eligibility-requires-all-absolute-gates-and-no-clean-completion-regression",
            "warmup_frames_per_episode": 10}}
    protocol["potential_stage_trials"] = {
        "screen": planned_stage_trials(protocol, routes, "screen"),
        "tune": {name: planned_stage_trials(protocol, routes, "tune", candidate=name) for name in configs if name != "incumbent"},
        "confirmation": {name: planned_stage_trials(protocol, routes, "confirmation", candidate=name) for name in configs},
        "robustness": {name: planned_stage_trials(protocol, routes, "robustness", candidate=name) for name in configs}}
    return protocol


def planned_stage_trials(protocol, routes, stage, *, candidate=None):
    """Alternate arm order on exactly paired original route/initial states."""
    if stage not in STAGES:
        raise ValueError("development-stage-invalid")
    configs = protocol["controller_configs"]
    if stage == "screen":
        order = ("incumbent", *sorted(name for name in configs if name != "incumbent"))
        arms = [("M1__" + name, "M1", name) for name in order]
    elif stage == "tune":
        if candidate not in configs or candidate == "incumbent":
            raise ValueError("development-tune-needs-one-nonincumbent")
        arms = [("M1_incumbent", "M1", "incumbent"), ("M1_candidate", "M1", candidate)]
    else:
        if candidate not in configs:
            raise ValueError("development-confirmation-controller-invalid")
        arms = [("M1_incumbent", "M1", "incumbent"), ("M1_candidate", "M1", candidate),
            ("B0_candidate", "B0", candidate)]
    envs = protocol["environments"] if stage == "robustness" else [protocol["environments"][0]]
    conditions = [protocol["initial_conditions"][0]] if stage == "robustness" else protocol["initial_conditions"]
    result, state = [], 0
    for row in sorted(routes["routes"], key=lambda r: CASES.index(r["case"])):
        for environment in envs:
            for condition in conditions:
                for arm, method, controller in (arms if state % 2 == 0 else list(reversed(arms))):
                    result.append({"stage": stage, "arm": arm, "method": method, "controller": controller,
                        "case": row["case"], "route_id": row["route"]["route_id"],
                        "initial_reference_m": protocol["stage_starts_m"][stage], "condition": copy.deepcopy(condition),
                        "environment": copy.deepcopy(environment),
                        "episode": f"episodes/{stage}/{arm}/{row['case']}__{environment['name']}__{condition['name']}"})
                state += 1
    if len(result) != protocol["expected_stage_episodes"][stage]:
        raise ValueError("development-stage-episode-budget-mismatch")
    return result


def collection_for_environment(collection_config, environment):
    from .config import collection_config_from_dict
    if environment not in DEFAULT_ENVIRONMENTS:
        raise ValueError("development-environment-not-frozen")
    value = collection_config.to_dict()
    value["weather"]["preset"] = environment["weather"]
    value["camera"]["mount"]["location"]["z"] += environment["camera_height_delta_m"]
    value["camera"]["mount"]["rotation"]["yaw"] += environment["camera_yaw_deg"]
    return collection_config_from_dict(value, allowed_weather_presets=WEATHERS)


class ProfiledBaselinePredictor:
    """Keep the same nominal OpenCV input while recording comparable boundaries."""
    def __init__(self, config):
        from .refinement_driving import NativeBaselinePredictor
        self.raw = NativeBaselinePredictor(config)

    @property
    def identity(self):
        return self.raw.identity

    def timer(self):
        from .pipeline_latency import PipelineFrameTimer
        return PipelineFrameTimer(device="cpu")

    def predict(self, rgb, *, width=640, height=360):
        return self.raw.predict(rgb, width=width, height=height)

    def predict_profiled(self, rgb, timer, *, width=640, height=360):
        return timer.measure("forward_decode_ms", lambda: self.predict(rgb, width=width, height=height))


def run_study_trial(route, trial, output, collection_config, *, predictor, protocol,
        carla_module, baseline_config, evaluation_config):
    from .baseline_driving import BaselineTrialPolicy
    from .controller import load_controller_config
    from .controller_study import build_controller, study_controller_config_from_dict
    from .driving import load_validation_config, run_drive_trial
    study_config = study_controller_config_from_dict(protocol["controller_configs"][trial["controller"]])

    class Policy(BaselineTrialPolicy):
        scope = SCOPE

        def __init__(self):
            super().__init__(baseline_config, evaluation_config, route, trial["condition"], predictor=predictor)
            self.method, self.initial_reference_m = trial["method"], trial["initial_reference_m"]
            self.study_controller = None

        def configure_metadata(self, metadata):
            super().configure_metadata(metadata)
            metadata.update(native_model=predictor.identity, requested_initial_reference_m=self.initial_reference_m,
                development_study={"trial": copy.deepcopy(trial), "protocol_sha256": digest(protocol),
                    "source_sha256": study_source_hashes(), "controller_config": study_config.to_dict(),
                    "controller_config_sha256": study_config.sha256}, latency_scope=LATENCY_SCOPE,
                detailed_latency_required=True)

        def command(self, image, speed_mps, controller):
            from .controller import ControllerError
            from scripts.carla_rgb_probe import rgb_from_bgra
            if self.study_controller is None:
                self.study_controller = build_controller(study_config, controller.geometry)
            # These inputs contain no native pose, route or actual calibration.
            timer = predictor.timer()
            began = time.perf_counter()
            rgb, _ = timer.measure("bgra_conversion_ms", lambda: rgb_from_bgra(
                bytes(image.raw_data), image.width, image.height))
            converted = time.perf_counter()
            prediction = predictor.predict_profiled(rgb, timer, width=image.width, height=image.height)
            json.dumps(prediction, allow_nan=False)
            predicted = time.perf_counter()
            command = None
            if prediction["status"] == "PASS":
                try:
                    command = timer.measure("controller_ms", lambda: self.study_controller.step(
                        prediction["waypoints_ego_m"], speed_mps, dt_seconds=.05))
                except ControllerError as error:
                    # A typed input rejection is an observed driving failure.
                    # Keep the exact original path for independent replay of
                    # the same exception; never substitute privileged points.
                    rejection = {"exception_type": "ControllerError", "error_code": str(error),
                        "waypoints_ego_m": copy.deepcopy(prediction["waypoints_ego_m"])}
                    if "learned_waypoints_ego_m" in prediction:
                        rejection["learned_waypoints_ego_m"] = copy.deepcopy(prediction["learned_waypoints_ego_m"])
                        prediction["learned_waypoints_ego_m"] = None
                    prediction.update(status="FAIL", error="development-controller-rejection",
                        waypoints_ego_m=None, controller_rejection=rejection)
            finished = time.perf_counter()
            prediction["pipeline_timing"] = timer.finish()
            return command, prediction, {"preprocess_ms": (converted - began) * 1000,
                "inference_ms": (predicted - converted) * 1000,
                "controller_ms": (finished - predicted) * 1000, "pipeline_ms": (finished - began) * 1000}

    config = collection_for_environment(collection_config, trial["environment"])
    return run_drive_trial(route, trial["case"], trial["condition"], output, config,
        carla_module=carla_module, validation_config=load_validation_config(), controller_config=load_controller_config(),
        trial_policy=Policy(), allowed_weather_presets=WEATHERS)


def summarize_stage(episodes, protocol, *, expected_trials, _include_environments=True):
    """Aggregate episode means, retaining short failed drives and every event."""
    from .driving_quality import summarize_driving_quality
    if (len(episodes) != len(expected_trials) or len({e["episode"] for e in episodes}) != len(episodes)
            or {e["episode"] for e in episodes} != {t["episode"] for t in expected_trials}):
        raise ValueError("development-stage-incomplete-or-duplicate")
    planned = {t["episode"]: t for t in expected_trials}
    output = {}
    quality = []
    for episode in episodes:
        if (any(episode.get(k) != v for k, v in planned[episode["episode"]].items())
                or episode.get("gate", {}).get("status") != "PASS"
                or episode.get("execution", {}).get("status") != "PASS"
                or episode.get("media", {}).get("status") != "PASS"):
            raise ValueError("development-stage-source-or-integrity-mismatch")
        metrics = episode["gate"]["metrics"]
        mean, p95 = metrics["lateral_abs_m"]["mean"], metrics["lateral_abs_m"]["p95"]
        latency = episode["steady_processing_p95_ms"]
        if (any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in (mean, p95))
                or latency is not None and (type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0)):
            raise ValueError("development-nonfinite-driving-summary")
        quality.append(dict(episode, variant=episode["arm"]))
        for case in ("all", episode["case"]):
            row = output.setdefault(episode["arm"], {}).setdefault(case, {"episode_means": [], "episode_p95": [], "episode_latency": []})
            row["episode_means"].append(mean)
            row["episode_p95"].append(p95)
            row["episode_latency"].append(latency)
    for group in summarize_driving_quality(quality)["groups"]:
        row = output[group["method"]][group["case"]]
        row.update(group)
        row["mean_lateral_m"] = math.fsum(row.pop("episode_means")) / row["episodes"]
        row["maximum_episode_p95_lateral_m"] = max(row.pop("episode_p95"))
        latencies = row.pop("episode_latency")
        row["episodes_with_steady_latency"] = sum(value is not None for value in latencies)
        row["maximum_episode_processing_p95_ms"] = max((value for value in latencies if value is not None), default=None)
    if _include_environments:
        for arm in output:
            output[arm]["environments"] = {}
        for environment in sorted({trial["environment"]["name"] for trial in expected_trials}):
            selected_trials = [trial for trial in expected_trials if trial["environment"]["name"] == environment]
            selected_records = [record for record in episodes if record["environment"]["name"] == environment]
            grouped = summarize_stage(selected_records, protocol, expected_trials=selected_trials, _include_environments=False)
            for arm, value in grouped.items():
                output[arm]["environments"][environment] = value
    return output


def absolute_gate(summary, protocol):
    limits = protocol["selection_policy"]
    errors = []
    groups = [(case, summary[case]) for case in ("all", *CASES)]
    groups += [("environment:" + environment + ":" + case, row)
        for environment, value in summary.get("environments", {}).items()
        for case, row in value.items()]
    for case, row in groups:
        for key in ("collision_episodes", "lane_invasion_episodes", "body_lane_margin_negative_episodes"):
            if row[key]:
                errors.append(case + ":" + key)
        if row["clean_completed"] != row["episodes"]:
            errors.append(case + ":incomplete-clean-driving")
        for field, limit in (("mean_lateral_m", limits["case_mean_max_m"]),
                ("maximum_episode_p95_lateral_m", limits["episode_p95_max_m"]),
                ("maximum_episode_processing_p95_ms", limits["processing_p95_max_ms"])):
            if row[field] is None or row[field] > limit:
                errors.append(case + ":" + field)
        if row["episodes_with_steady_latency"] != row["episodes"]:
            errors.append(case + ":missing-steady-latency")
    return errors


def select_screen_candidate(summary, protocol):
    expected = {"M1__" + name for name in protocol["controller_configs"]}
    if set(summary) != expected:
        raise ValueError("development-screen-config-coverage-mismatch")
    names = [name for name in protocol["controller_configs"] if name != "incumbent"]
    chosen = min(names, key=lambda name: (-summary["M1__" + name]["all"]["clean_completed"],
        summary["M1__" + name]["all"]["mean_lateral_m"], name))
    return {"status": "PASS", "selected_nonincumbent": chosen, "retained": ["incumbent", chosen],
        "rule": protocol["selection_policy"]["screen_rule"], "summary_sha256": digest(summary)}


def select_tuned_controller(summary, protocol, candidate):
    if set(summary) != {"M1_incumbent", "M1_candidate"} or candidate == "incumbent":
        raise ValueError("development-tune-arm-coverage-mismatch")
    failed = absolute_gate(summary["M1_candidate"], protocol)
    old, new = summary["M1_incumbent"]["all"], summary["M1_candidate"]["all"]
    improvement = old["mean_lateral_m"] - new["mean_lateral_m"]
    if (improvement < protocol["selection_policy"]["minimum_mean_improvement_m"]
            or improvement < old["mean_lateral_m"] * protocol["selection_policy"]["minimum_relative_mean_improvement"]):
        failed.append("mean-improvement-below-frozen-practical-margin")
    if new["clean_completed"] < old["clean_completed"]:
        failed.append("clean-completion-regression")
    return {"status": "PASS", "selected_controller": "incumbent" if failed else candidate,
        "candidate": candidate, "failed_criteria": failed, "summary_sha256": digest(summary),
        "rule": protocol["selection_policy"]["tune_rule"]}


def confirm_controller(confirmation, robustness, protocol, candidate):
    if any(set(value) != {"M1_incumbent", "M1_candidate", "B0_candidate"} for value in (confirmation, robustness)):
        raise ValueError("development-confirmation-arm-coverage-mismatch")
    failed = []
    for stage, summary in (("confirmation", confirmation), ("robustness", robustness)):
        failed.extend(stage + ":" + error for error in absolute_gate(summary["M1_candidate"], protocol))
        old, new = summary["M1_incumbent"]["all"], summary["M1_candidate"]["all"]
        if new["clean_completed"] < old["clean_completed"]:
            failed.append(stage + ":clean-completion-regression")
        improvement = old["mean_lateral_m"] - new["mean_lateral_m"]
        if stage == "confirmation" and (improvement < protocol["selection_policy"]["minimum_mean_improvement_m"]
                or improvement < old["mean_lateral_m"] * protocol["selection_policy"]["minimum_relative_mean_improvement"]):
            failed.append(stage + ":mean-improvement-below-frozen-practical-margin")
    if candidate == "incumbent":
        failed.append("search-retained-incumbent")
    return {"status": "PASS", "selected_controller": "incumbent" if failed else candidate,
        "proposed_controller": candidate, "eligible_improvement": not failed, "failed_criteria": failed,
        "confirmation_sha256": digest(confirmation), "robustness_sha256": digest(robustness),
        "scope": protocol["scope"], "final_town_used": False, "pointer_updated": False}


def validate_study_episode(directory, predictor, trial, *, protocol, routes):
    """Independently reproduce RGB perception before replaying native controls."""
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
                and predictor.correction == "path", "development-M1-checkpoint-or-correction-mismatch")
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


# This explicit replay retains the original native invariants. It does not
# modify legacy validators or rewrite captured metadata to satisfy a gate.
from .baseline import OpenCVLanePredictor, baseline_config_from_dict
from .baseline_validation import BASELINE_ROW_KEYS, _native_angles, _safe_file, _stopping
from .capture import _calibration, _mount_matrix
from .color import verify_rgb_color
from .config import collection_config_from_dict
from .contracts import validate_identifier, validate_waypoint_label
from .controller import controller_config_from_dict
from .driving_route import FixedRouteTracker, _transform
from .driving_validation import (CONTROL_KEYS, _body_margin, _cleanup, _control,
    _finite, _geometry, _integer, _require, _same, _sha)
from .evaluation import EvaluationTracker, deduplicate_events, evaluate_episode, evaluation_config_from_dict
from .geometry_survey import classify_geometry
from .projection import decode_rgb8_png


def _validate_study_native(episode_path, *, protocol, trial, routes, predictor):
    """Replay the original native geometry, labels, controls, events and PNGs."""
    allowed_conditions = protocol["initial_conditions"]
    scope, methods = SCOPE, (trial["method"],)
    prediction_inputs = "rgb-and-fixed-nominal-calibration"
    expected_initial_reference_m = trial["initial_reference_m"]
    allowed_weather_presets = WEATHERS
    initial_heading_reference = "tracked-route-tangent"
    root = Path(episode_path)
    report = {"status": "FAIL", "scope": scope, "gate_version": VERSION, "errors": []}
    try:
        metadata_path, trace_path = _safe_file(root, "episode.json"), _safe_file(root, "steps.jsonl")
        metadata_bytes, trace_bytes = metadata_path.read_bytes(), trace_path.read_bytes()
        metadata = strict_json_loads(metadata_bytes.decode("utf-8"))
        required = {"schema_version", "status", "scope", "episode_id", "geometry_case", "condition", "route_contract",
                    "collection_config", "controller_config", "controller_config_sha256", "vehicle_geometry", "geometry_evidence",
                    "camera_calibration", "bounding_box", "applied_manifest", "cleanup", "events", "event_callback_errors",
                    "drive_end_frame", "post_stop_frame", "step_count", "trace_sha256", "validation_config", "stop_control",
                    "stop_observations", "method", "baseline_config", "baseline_config_sha256", "evaluation_config",
                    "evaluation_config_sha256", "evaluation_start_reference_m", "evaluation_route_length_m",
                    "evaluation_goal_reference_m", "evaluation_metrics", "termination_reason", "prediction_inputs",
                    "perception_uses_route", "perception_uses_vehicle_pose", "non_driving_events"}
        _require(isinstance(metadata, dict) and required <= metadata.keys(), "baseline-metadata-fields-missing")
        _require(type(metadata["schema_version"]) is int and metadata["schema_version"] == 1
                 and metadata["status"] == "COMPLETE" and metadata["scope"] == scope,
                 "baseline-episode-not-complete")
        validate_identifier(metadata["episode_id"])
        method = metadata["method"]
        _require(method in methods and metadata["geometry_case"] in CASES, "baseline-method-or-case-invalid")
        is_reference = method == "privileged_reference"
        _require(type(initial_heading_reference) is str and initial_heading_reference in
                 ("tracked-route-tangent", "declared-route-start"), "baseline-initial-heading-reference-invalid")
        _require(initial_heading_reference == "tracked-route-tangent" or is_reference,
                 "baseline-initial-heading-reference-requires-privileged-method")
        _require(metadata["perception_uses_route"] is is_reference and metadata["perception_uses_vehicle_pose"] is is_reference,
                 "baseline-prediction-input-contract-mismatch")
        expected_inputs = "privileged-original-route-and-same-frame-pose" if is_reference else prediction_inputs
        _require(metadata["prediction_inputs"] == expected_inputs, "baseline-prediction-input-contract-mismatch")
        from .driving import load_validation_config, route_initial_location
        policy = load_validation_config()
        _require(_same(metadata["validation_config"], policy), "baseline-native-validation-policy-mismatch")
        approved_conditions = policy["initial_conditions"]
        if allowed_conditions is None:
            permitted_conditions = [next(condition for condition in approved_conditions if condition["name"] == "center")]
        else:
            _require(isinstance(allowed_conditions, (list, tuple)) and 1 <= len(allowed_conditions) <= len(approved_conditions)
                     and all(any(_same(condition, approved) for approved in approved_conditions)
                             for condition in allowed_conditions)
                     and len({canonical_sha256(condition) for condition in allowed_conditions}) == len(allowed_conditions),
                     "baseline-allowed-conditions-invalid")
            permitted_conditions = allowed_conditions
        _require(any(_same(metadata["condition"], condition) for condition in permitted_conditions), "baseline-condition-invalid")
        baseline_config = baseline_config_from_dict(metadata["baseline_config"])
        evaluation_config = evaluation_config_from_dict(metadata["evaluation_config"])
        controller_config = controller_config_from_dict(metadata["controller_config"])
        _require(metadata["baseline_config_sha256"] == baseline_config.sha256
                 and metadata["evaluation_config_sha256"] == evaluation_config.sha256
                 and metadata["controller_config_sha256"] == controller_config.sha256, "baseline-configuration-hash-mismatch")
        cfg = collection_config_from_dict(metadata["collection_config"],
                                          allowed_weather_presets=allowed_weather_presets)
        from .baseline import load_baseline_config
        from .config import load_collection_config
        from .controller import load_controller_config
        from .controller_study import build_controller, study_controller_config_from_dict
        _require(baseline_config == load_baseline_config(), "development-nominal-perception-calibration-changed")
        expected_cfg = collection_for_environment(load_collection_config(
            Path(__file__).resolve().parents[1] / "configs/collection_d4.json"), trial["environment"]).to_dict()
        declared_route = next(r["route"] for r in routes["routes"] if r["route"]["route_id"] == trial["route_id"])
        expected_cfg["vehicle"]["spawn_point_index"] = declared_route["spawn_point_index"]
        _require(cfg.to_dict() == expected_cfg, "development-actual-camera-or-weather-mismatch")
        _require(controller_config == load_controller_config(), "development-runner-base-controller-changed")
        study_config = study_controller_config_from_dict(protocol["controller_configs"][trial["controller"]])
        _require(metadata.get("development_study") == {"trial": trial, "protocol_sha256": digest(protocol),
            "source_sha256": study_source_hashes(), "controller_config": study_config.to_dict(),
            "controller_config_sha256": study_config.sha256}, "development-study-episode-binding-mismatch")
        route = metadata["route_contract"]
        tracker = FixedRouteTracker(route, max_speed_mps=policy["maximum_speed_mps"], minimum_dt_seconds=.05)
        _require(route["town"] == cfg.map and route["spawn_point_index"] == cfg.vehicle.spawn_point_index
                 and route["town"] != "Town02" and classify_geometry(route)["case"] == metadata["geometry_case"],
                 "baseline-original-route-source-mismatch")
        geometry = _geometry(metadata)
        _require(route == declared_route, "development-original-route-contract-mismatch")
        controller = build_controller(study_config, geometry)
        predictor = None if is_reference else (predictor if predictor is not None else OpenCVLanePredictor(baseline_config))
        _require(_same(metadata["camera_calibration"], _calibration(cfg, _mount_matrix(cfg))), "baseline-calibration-mismatch")
        _require(metadata["applied_manifest"].get("applied_verified") is True, "baseline-initial-application-not-verified")
        _cleanup(metadata["cleanup"])
        _require(metadata["event_callback_errors"] == [], "baseline-event-callback-failed")
        events = metadata["events"]
        _require(isinstance(events, list), "baseline-events-invalid")
        deduplicate_events(events)
        _require(metadata["trace_sha256"] == _sha(trace_bytes), "baseline-trace-hash-mismatch")
        lines = trace_bytes.decode("utf-8").splitlines()
        _require(lines and all(line.strip() for line in lines), "baseline-trace-empty-or-blank")
        rows = [strict_json_loads(line) for line in lines]
        _require(len(rows) == _integer(metadata["step_count"], 1), "baseline-step-count-mismatch")
        start = _finite(metadata["evaluation_start_reference_m"])
        length, goal = _finite(metadata["evaluation_route_length_m"]), _finite(metadata["evaluation_goal_reference_m"])
        _require(abs(goal - start - length) <= 1e-6 and goal == policy["goal_reference_m"], "baseline-evaluation-length-mismatch")
        evaluation = EvaluationTracker(evaluation_config, length, start, metadata["condition"])
        previous, snapshots, image_hashes = None, [], {}
        for index, row in enumerate(rows):
            _require(isinstance(row, dict) and set(row) == BASELINE_ROW_KEYS, "baseline-step-fields-invalid")
            _require(type(row["schema_version"]) is int and row["schema_version"] == 1 and _integer(row["index"]) == index,
                     "baseline-step-index-mismatch")
            frame, timestamp = _integer(row["frame"]), _finite(row["timestamp"])
            _require(all(type(row[key]) is int and row[key] == frame for key in
                         ("world_frame", "snapshot_frame", "sensor_frame", "command_source_frame")), "baseline-frame-alignment-mismatch")
            _require(timestamp >= 0 and all(abs(_finite(row[key]) - timestamp) <= 1e-6 for key in
                                           ("snapshot_timestamp", "sensor_timestamp")), "baseline-timestamp-alignment-mismatch")
            _require(abs(_finite(row["delta_seconds"]) - .05) <= 1e-6 and _finite(row["tick_to_rgb_ms"]) >= 0,
                     "baseline-timestep-mismatch")
            if previous is not None:
                _require(frame == previous["frame"] + 1 and abs(timestamp - previous["timestamp"] - .05) <= 1e-6,
                         "baseline-frame-sequence-mismatch")
            terminal = index == len(rows) - 1
            _require(type(row["terminal"]) is bool and row["terminal"] is terminal, "baseline-terminal-row-mismatch")
            _require(row["command_target_frame"] is None if terminal else
                     type(row["command_target_frame"]) is int and row["command_target_frame"] == frame + 1,
                     "baseline-command-target-frame-mismatch")
            vehicle = row["vehicle"]
            _require(isinstance(vehicle, dict) and set(vehicle) == {"transform", "velocity", "applied_control"},
                     "baseline-vehicle-record-invalid")
            pose = _transform(vehicle["transform"])
            _require(isinstance(vehicle["velocity"], dict) and set(vehicle["velocity"]) == {"x", "y", "z"},
                     "baseline-velocity-invalid")
            speed = math.sqrt(sum(_finite(number) ** 2 for number in vehicle["velocity"].values()))
            _require(abs(_finite(row["speed_mps"]) - speed) <= 1e-5, "baseline-speed-readback-mismatch")
            expected_applied = cfg.vehicle.control.to_dict() if previous is None else {key: previous["command"][key] for key in CONTROL_KEYS}
            _require(_same(_control(vehicle["applied_control"]), expected_applied), "baseline-command-application-mismatch")
            tracking = tracker.observe(pose, frame, timestamp)
            _require(_same(row["tracking"], tracking), "baseline-original-tracking-replay-mismatch")
            validate_waypoint_label(row["tracking"]["label"], {"frame": frame, "vehicle": {"transform": pose}},
                                    {"route_id": route["route_id"], "route_contract": route})
            if index == 0:
                condition = metadata["condition"]
                if initial_heading_reference == "declared-route-start":
                    nominal_yaw = route_initial_location(route, expected_initial_reference_m)[1]
                    yaw_error = (pose["rotation"]["yaw"] - nominal_yaw - condition["yaw_deg"] + 180.) % 360. - 180.
                else:
                    yaw_error = (tracking["heading_error_deg"] - condition["yaw_deg"] + 180.) % 360. - 180.
                _require(abs(tracking["reference_arc_m"] - start) <= 1e-6
                         and abs(start - expected_initial_reference_m) <= .5
                         and abs(tracking["lateral_error_m"] - condition["lateral_m"]) <= policy["initial_lateral_tolerance_m"]
                         and abs(yaw_error) <= policy["initial_yaw_tolerance_deg"],
                         "baseline-initial-reference-or-condition-mismatch")
            camera = row["camera"]
            _require(isinstance(camera, dict) and set(camera) == {"transform"}, "baseline-camera-record-invalid")
            cm, vm = _transform(camera["transform"])["matrix"], pose["matrix"]
            relative = [[sum(vm[k][r] * cm[k][c] for k in range(3)) if c < 3 else
                         sum(vm[k][r] * (cm[k][3] - vm[k][3]) for k in range(3)) for c in range(4)] for r in range(3)] + [[0., 0., 0., 1.]]
            _require(_same(relative, metadata["camera_calibration"]["requested_mount_matrix"], .002), "baseline-camera-mount-mismatch")
            margin = _body_margin(pose, tracking, metadata["bounding_box"], route)
            _require(abs(_finite(row["body_lane_margin_m"]) - margin) <= 1e-5, "baseline-body-margin-replay-mismatch")
            _native_angles(row["native_wheel_angles_deg"])
            image = row["image"]
            _require(isinstance(image, dict) and set(image) == {"path", "sha256", "width", "height", "color_provenance"},
                     "baseline-image-record-missing-or-invalid")
            expected_path = f"images/{frame:08d}.png"
            _require(image["path"] == expected_path and type(image["width"]) is int and type(image["height"]) is int
                     and (image["width"], image["height"]) == (cfg.camera.width, cfg.camera.height), "baseline-image-identity-mismatch")
            png = _safe_file(root, expected_path).read_bytes()
            width, height, rgb = decode_rgb8_png(png, image["sha256"], (cfg.camera.width, cfg.camera.height))
            verify_rgb_color(rgb, image["color_provenance"], width, height, require_witness=True)
            image_hashes[expected_path] = _sha(png)
            if is_reference:
                prediction = {"status": "PASS", "waypoints_ego_m": copy.deepcopy(tracking["label"]["waypoints_ego_m"]),
                              "diagnostics": {"method": method, "input": "privileged-original-route"}, "timing_ms": 0.0}
            else:
                prediction = predictor.predict(rgb, width=width, height=height)
            observed = row["perception"]
            _require(isinstance(observed, dict) and _finite(observed.get("timing_ms")) >= 0, "baseline-perception-timing-invalid")
            _require(_same({key: value for key, value in observed.items() if key != "timing_ms"},
                          {key: value for key, value in prediction.items() if key != "timing_ms"}, 1e-6),
                     "baseline-rgb-prediction-replay-mismatch")
            rejection = prediction.get("controller_rejection")
            if rejection is not None:
                from .controller import ControllerError
                try:
                    controller.step(rejection["waypoints_ego_m"], speed, dt_seconds=.05)
                except ControllerError as error:
                    _require(str(error) == rejection["error_code"]
                        and rejection["exception_type"] == "ControllerError", "development-controller-rejection-replay-mismatch")
                else:
                    _require(False, "development-recorded-controller-rejection-did-not-reproduce")
                command = None
            else:
                command = controller.step(prediction["waypoints_ego_m"], speed, dt_seconds=.05) if prediction["status"] == "PASS" else None
            _require(_same(row["command"], command), "baseline-controller-replay-mismatch")
            if command is None:
                _require(terminal and row["technical_failure"] is True and row["termination_reason"] == "technical_failure",
                         "baseline-null-command-not-terminal-failure")
            expected_technical = prediction["status"] != "PASS" or speed > policy["maximum_speed_mps"]
            _require(row["evaluation_callback_errors"] == [], "baseline-per-frame-callback-failure")
            latency = row["latency_ms"]
            _require(isinstance(latency, dict) and set(latency) == {"preprocess_ms", "inference_ms", "controller_ms", "pipeline_ms"},
                     "baseline-latency-fields-invalid")
            _require(all(_finite(number) >= 0 for number in latency.values()), "baseline-latency-number-invalid")
            _require(abs(latency["pipeline_ms"] - math.fsum(latency[key] for key in
                         ("preprocess_ms", "inference_ms", "controller_ms"))) <= 1e-5,
                     "baseline-latency-component-sum-mismatch")
            if is_reference:
                _require(latency["preprocess_ms"] == 0. and latency["inference_ms"] == 0.,
                         "baseline-reference-latency-scope-mismatch")
            _require(type(row["technical_failure"]) is bool and row["technical_failure"] == expected_technical,
                     "baseline-technical-failure-flag-mismatch")
            remaining = route["route_length_m"] - tracking["reference_arc_m"]
            _require(abs(_finite(row["remaining_horizon_m"]) - remaining) <= 1e-6, "baseline-route-horizon-replay-mismatch")
            snapshot = row["evaluation_events"]
            _require(isinstance(snapshot, list) and len(snapshot) >= len(snapshots) and _same(snapshot[:len(snapshots)], snapshots)
                     and _same(events[:len(snapshot)], snapshot) and all(event["frame"] <= frame for event in snapshot),
                     "baseline-event-snapshot-mismatch")
            decision = evaluation.update(row, events=snapshot, technical_failure=expected_technical, remaining_horizon_m=remaining)
            _require(_same(row["evaluation"], decision) and row["termination_reason"] == decision["reason"]
                     and row["terminal"] == decision["terminated"], "baseline-evaluation-decision-replay-mismatch")
            snapshots, previous = snapshot, row
        _require(metadata["drive_end_frame"] == rows[-1]["frame"] and type(metadata["drive_end_frame"]) is int,
                 "baseline-drive-end-frame-mismatch")
        stopping = _stopping(metadata, rows[-1], route)
        expected_non_driving = [event for event in events if not rows[0]["frame"] <= event["frame"] <= rows[-1]["frame"]]
        _require(_same(metadata["non_driving_events"], expected_non_driving), "baseline-non-driving-event-set-mismatch")
        replay_metadata = dict(metadata, termination_reason=rows[-1]["termination_reason"])
        metrics = evaluate_episode(replay_metadata, rows, events, config=evaluation_config)
        _require(metrics["integrity_valid"] is True and metadata["termination_reason"] == metrics["termination_reason"]
                 and _same(metadata["evaluation_metrics"], metrics),
                 "baseline-final-evaluation-metrics-mismatch")
        actual_images = {path.relative_to(root).as_posix() for path in (root / "images").rglob("*") if path.is_file()}
        _require(actual_images == set(image_hashes), "baseline-orphan-or-missing-image")
        _require(metadata_path.read_bytes() == metadata_bytes and trace_path.read_bytes() == trace_bytes
                 and all(_sha(_safe_file(root, relative).read_bytes()) == digest for relative, digest in image_hashes.items()),
                 "baseline-source-changed-during-validation")
        config_except_spawn = cfg.to_dict()
        config_except_spawn["vehicle"].pop("spawn_point_index")
        report.update(status="PASS", source={"method": method, "episode_id": metadata["episode_id"],
                      "geometry_case": metadata["geometry_case"], "condition": metadata["condition"], "route_id": route["route_id"],
                      "route_sha256": route_sha256(route), "town": route["town"], "opendrive_sha256": route["opendrive_sha256"],
                      "metadata_sha256": _sha(metadata_bytes), "trace_sha256": _sha(trace_bytes), "image_sha256": image_hashes,
                      "controller_config_sha256": controller_config.sha256, "baseline_config_sha256": baseline_config.sha256,
                      "evaluation_config_sha256": evaluation_config.sha256, "calibration_sha256": canonical_sha256(metadata["camera_calibration"]),
                      "configuration_except_spawn_sha256": canonical_sha256(config_except_spawn), "vehicle_geometry": geometry.to_dict()},
                      metrics={"step_count": len(rows), "image_count": len(image_hashes), **stopping}, evaluation_metrics=metrics,
                      checks={"every_frame_rgb_color": True, "prediction_replay": True, "shared_controller_replay": True,
                              "original_route_evaluation_replay": True, "command_application": True, "stop_readbacks": True,
                              "owned_actors_cleaned": True, "sources_unchanged": True})
        if initial_heading_reference == "declared-route-start":
            report["checks"]["declared_start_heading_binding"] = True
    except BaseException as error:
        report["errors"].append({"code": getattr(error, "code", str(error) if isinstance(error, ValueError) else "baseline-source-validation-failed"),
                                 "exception_type": type(error).__name__})
    return report
