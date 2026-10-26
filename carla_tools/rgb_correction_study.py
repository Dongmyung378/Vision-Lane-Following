"""Paired RGB-correction development study, separate from frozen v1 evaluation.

Weights, nominal calibration and controller are shared. Only current-frame RGB
correction changes. The exposed Town02 regression is explicitly development
evidence; this module never selects or replaces a deployed pipeline.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from . import final_evaluation_driving as fd
from .contracts import canonical_sha256, strict_json_loads
from .final_evaluation import runtime_route_contract, validate_route

SCOPE = "paired-RGB-correction-development-and-exposed-regression"
ARMS = ("raw", "current", "supported")
WEATHERS = fd.WEATHERS
LATENCY_SCOPE = fd.LATENCY_SCOPE
REFERENCE_LATENCY_SCOPE = fd.REFERENCE_LATENCY_SCOPE
BOOTSTRAP_HEIGHT_TOLERANCE_M = fd.BOOTSTRAP_HEIGHT_TOLERANCE_M
adapter_dependency_hashes = fd.adapter_dependency_hashes
condition_for_trial = fd.condition_for_trial
collection_for_trial = fd.collection_for_trial
_validate_native_application = fd._validate_native_application


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def study_id(protocol):
    return canonical_sha256({k: v for k, v in protocol.items() if k != "study_id"})


def validate_protocol(protocol, routes, selection=None):
    """Validate a predeclared 36-episode development contract, never a final split."""
    _require(protocol.get("scope") == SCOPE and protocol.get("split") == "development"
             and protocol.get("study_id") == study_id(protocol), "study-protocol-identity")
    _require(protocol.get("v1_results_preserved") is True
             and protocol.get("automatic_pipeline_activation") is False, "study-preservation-contract")
    _require(protocol["routes_sha256"] == canonical_sha256(routes), "study-routes-hash")
    _require(set(protocol["arms"]) == set(ARMS), "study-arms")
    route_index = {route["route_id"]: route for route in routes["routes"]}
    _require(len(route_index) == 4, "study-route-count")
    for route in route_index.values():
        validate_route(route)
    trials = protocol["trials"]
    _require(len(trials) == 36 and len({t["trial_id"] for t in trials}) == 36,
             "study-trial-cardinality")
    pairs = {}
    for index, trial in enumerate(trials):
        _require(trial["schedule_index"] == index and trial["method"] == "M1"
                 and trial["arm"] in ARMS and trial["route_id"] in route_index,
                 "study-trial-fields")
        _require(route_index[trial["route_id"]]["town"] == trial["town"]
                 and trial["environment"] == {"weather": "ClearNoon", "camera_pitch_delta_deg": 0.0},
                 "study-trial-physical-scope")
        role = "development" if trial["town"] == "Town01" else "previously_exposed_failure_regression"
        _require(trial["evidence_role"] == role, "study-evidence-role")
        pairs.setdefault(trial["pair_id"], []).append(trial)
    _require(len(pairs) == 12 and sum(t["town"] == "Town01" for t in trials) == 27
             and sum(t["town"] == "Town02" for t in trials) == 9, "study-cohort-size")
    for rows in pairs.values():
        _require(len(rows) == 3 and {r["arm"] for r in rows} == set(ARMS), "study-paired-arms")
        base = rows[0]
        for row in rows[1:]:
            _require(all(row[k] == base[k] for k in ("route_id", "initial_state", "environment",
                "town", "group_id", "evidence_role", "geometry_case")), "study-unpaired-condition")
    if selection is not None:
        _require(selection["pipeline_id"] == protocol["study_id"]
                 and selection["model"] == protocol["model"]
                 and selection["study_arm"] in ARMS, "study-arm-selection")
    return {"status": "PASS", "study_id": protocol["study_id"], "episode_count": len(trials)}


def verify_sources(project_root, protocol):
    root = Path(project_root).resolve()
    for relative, expected in protocol["source_sha256"].items():
        target = (root / relative).resolve()
        _require(target.is_relative_to(root) and target.is_file()
                 and hashlib.sha256(target.read_bytes()).hexdigest() == expected, "study-source-mismatch:" + relative)


def arm_selection(protocol, arm):
    _require(arm in ARMS, "study-arm-unknown")
    return {"pipeline_id": protocol["study_id"], "study_arm": arm,
            "model": copy.deepcopy(protocol["model"]), "arm_contract": copy.deepcopy(protocol["arms"][arm])}


class IdentityRGBCorrection:
    """A timed identity operation: no paint proposal or route input."""
    def apply(self, rgb, points, *, width=640, height=360):
        from .learning_data import validate_waypoint_points
        validate_waypoint_points(points)
        _require((width, height) == (640, 360), "study-raw-dimensions")
        return copy.deepcopy(points), {"version": "rgb-study-no-correction-v1", "accepted": False,
            "selected_source": "current-learned-RGB-path", "reason": "correction-disabled"}


def build_arm_predictor(checkpoint, protocol, arm, *, device="cpu"):
    from .runtime_driving import RuntimeCheckpointPredictor
    from .guarded_path_correction import build_guarded_predictor
    from .supported_path_correction import SupportedPathPredictor, supported_config_from_dict
    _require(arm in ARMS, "study-arm-unknown")
    base = RuntimeCheckpointPredictor(checkpoint,
        expected_sha256=protocol["model"]["checkpoint_sha256"], device=device, correction="path")
    if arm == "raw":
        base.corrector = IdentityRGBCorrection()
        base.correction = "none"
        base._identity.update(variant=protocol["model"]["architecture"] + "_raw", correction="none")
        predictor = base
    elif arm == "current":
        predictor = build_guarded_predictor(base)
    else:
        predictor = SupportedPathPredictor(base, supported_config_from_dict(protocol["arms"][arm]["config"]))
    predictor.study_arm = arm
    validate_arm_predictor(predictor, arm_selection(protocol, arm))
    return predictor


def validate_arm_predictor(predictor, selection):
    arm, model, identity = selection["study_arm"], selection["model"], predictor.identity
    _require(getattr(predictor, "study_arm", None) == arm, "study-predictor-arm")
    base = predictor if arm == "raw" else predictor.base
    _require(base.raw.identity["variant"] == model["architecture"]
        and identity["checkpoint_sha256"] == model["checkpoint_sha256"]
        and identity["model_contract_sha256"] == model["model_contract_sha256"]
        and base.raw.config.sha256 == model["learning_config_sha256"], "study-model-contract")
    _require(predictor.correction == selection["arm_contract"]["correction"], "study-correction-contract")
    if arm == "supported":
        _require(predictor.config.to_dict() == selection["arm_contract"]["config"], "study-supported-config")
    elif arm == "current":
        _require(predictor.config.to_dict() == selection["arm_contract"]["guard_config"], "study-current-config")


def validation_policy(manifest, trial):
    from .driving import load_validation_config
    result = load_validation_config()
    result["goal_reference_m"] = trial["initial_state"]["initial_reference_m"] + manifest["route_length_m"]
    conditions = {r["initial_state"]["state_id"]: condition_for_trial(r) for r in manifest["trials"]
                  if r["route_id"] == trial["route_id"] and r["group_id"] == trial["group_id"]}
    result["initial_conditions"] = list(conditions.values())
    return result


def run_study_trial(route, trial, output, *, manifest, routes, predictor, carla_module, carla_map, session_factory=None):
    """Call inside the post-drive-observation context, before adapter creation."""
    selection = arm_selection(manifest, trial["arm"])
    validate_protocol(manifest, routes, selection)
    validate_arm_predictor(predictor, selection)
    _require(trial in manifest["trials"], "study-trial-not-planned")
    bootstrap = fd.resolve_bootstrap(route, carla_map)
    lowered = runtime_route_contract(route, bootstrap["spawn_point_index"])
    fd.verify_topology_in_world(lowered, carla_map, carla_module)
    timings = {}
    policy = fd._isolated(fd.build_trial_policy, SCOPE=SCOPE)(lowered, trial, manifest, selection, predictor, timings)
    configure = policy.configure_metadata
    def metadata_hook(metadata):
        configure(metadata)
        metadata.update(declared_topology_route=copy.deepcopy(route), native_bootstrap=copy.deepcopy(bootstrap),
            rgb_correction_study={"study_id": manifest["study_id"], "arm": trial["arm"],
                "pair_id": trial["pair_id"], "evidence_role": trial["evidence_role"],
                "independent_final_evaluation": False})
    policy.configure_metadata = metadata_hook
    from .controller import load_controller_config
    # Deliberately instantiate here, after the caller installs the observer.
    operation = fd._native_adapter(lowered, timings)
    return operation(lowered, "heldout_route", condition_for_trial(trial), output,
        collection_for_trial(manifest, trial, bootstrap["spawn_point_index"]), carla_module=carla_module,
        validation_config=validation_policy(manifest, trial), controller_config=load_controller_config(),
        trial_policy=policy, allowed_weather_presets=WEATHERS, session_factory=session_factory)


def _perception_replay(predictor, rgb, observed, *, method, width, height):
    """Fresh CPU/device inference plus exact correction on recorded learned points."""
    from .driving_validation import _same
    from .learning_data import validate_waypoint_points
    from .pipeline_latency import validate_pipeline_timing, STAGES
    from .runtime_driving import validate_runtime_timing
    reproduced, actual = predictor.predict(rgb, width=width, height=height), copy.deepcopy(observed)
    rejection = actual.pop("controller_rejection", None)
    if rejection is not None:
        _require(set(rejection) == {"exception_type", "error_code", "waypoints_ego_m", "learned_waypoints_ego_m"}
                 and rejection["exception_type"] == "ControllerError" and actual["status"] == "FAIL"
                 and actual["error"] == "final-controller-rejection" and actual["waypoints_ego_m"] is None
                 and actual["learned_waypoints_ego_m"] is None, "study-controller-rejection-schema")
        actual.update(status="PASS", error=None, waypoints_ego_m=rejection["waypoints_ego_m"],
                      learned_waypoints_ego_m=rejection["learned_waypoints_ego_m"])
    timing = validate_pipeline_timing(actual["pipeline_timing"])
    validate_runtime_timing(timing, observed["_legacy_latency"], observed["status"])
    if rejection is not None:
        _require(timing["executed_stages"] == list(STAGES) and timing["failed_stages"] == ["controller_ms"]
                 and (timing["device"] != "cuda" or set(timing["synchronized_stages"]) ==
                      {"learned_preprocessing_ms", "forward_decode_ms"}), "study-rejection-timing")
    maximum = 0.
    for key in ("waypoints_ego_m", "learned_waypoints_ego_m"):
        a, b = actual.get(key), reproduced.get(key)
        if actual["status"] == "PASS":
            validate_waypoint_points(a); validate_waypoint_points(b)
            delta = max(abs(x-y) for p,q in zip(a,b) for x,y in zip(p,q))
            _require(delta <= .05, "study-fresh-RGB-coordinate-replay")
            maximum = max(maximum, delta)
        else:
            _require(a is None and b is None, "study-failed-prediction-points")
    ignored = {"timing_ms", "pipeline_timing", "_legacy_latency", "waypoints_ego_m", "learned_waypoints_ego_m"}
    aa, bb = ({k:copy.deepcopy(v) for k,v in value.items() if k not in ignored} for value in (actual,reproduced))
    if actual["status"] == "PASS":
        corrected, details = predictor.corrector.apply(rgb, actual["learned_waypoints_ego_m"], width=width,height=height)
        _require(_same(corrected, actual["waypoints_ego_m"], 1e-8), "study-recorded-raw-exact-correction")
        _require(corrected[2:] == actual["learned_waypoints_ego_m"][2:], "study-far-points-changed")
        if predictor.study_arm == "raw":
            details = {"path_correction":details}
        elif predictor.study_arm == "current":
            details = {"path_correction":details["path_correction"],
                       "guarded_path_correction":details["guarded_path_correction"]}
        for key, detail in details.items():
            _require(_same(actual["diagnostics"][key], detail, 1e-8), "study-recorded-diagnostic:"+key)
        for key in ("path_correction", "guarded_path_correction", "supported_path_correction"):
            aa["diagnostics"].pop(key, None); bb["diagnostics"].pop(key, None)
    _require(_same(aa,bb,1e-8), "study-fresh-RGB-content-replay")
    return maximum


validate_manifest = validate_protocol


def validate_study_episode(directory, predictor, trial, *, manifest, routes, selection):
    """Fresh replay of original RGB, fixed-route GT, commands and evaluation.

    PASS denotes persisted integrity. Driving failures remain valid outcomes.
    This is an offline gate and never executes a simulator.
    """
    from .baseline import baseline_config_from_dict
    from .baseline_validation import BASELINE_ROW_KEYS, _native_angles, _safe_file, _stopping
    from .capture import _calibration, _mount_matrix
    from .color import verify_rgb_color
    from .contracts import validate_waypoint_label
    from .controller import ControllerError, load_controller_config
    from .controller_study import build_controller, study_controller_config_from_dict
    from .driving import route_initial_location
    from .driving_route import FixedRouteTracker, _transform
    from .driving_validation import CONTROL_KEYS, _body_margin, _cleanup, _control, _finite, _geometry, _same
    from .evaluation import EvaluationTracker, deduplicate_events, evaluate_episode, evaluation_config_from_dict
    from .projection import decode_rgb8_png
    root = Path(directory)
    report = {"status": "FAIL", "scope": SCOPE, "errors": []}
    try:
        validate_manifest(manifest, routes, selection)
        _require(selection["study_arm"] == trial["arm"], "study-replay-arm-binding")
        _require(trial in manifest["trials"], "final-trial-not-frozen")
        original = next(route for route in routes["routes"] if route["route_id"] == trial["route_id"])
        mb, tb = _safe_file(root, "episode.json").read_bytes(), _safe_file(root, "steps.jsonl").read_bytes()
        metadata = strict_json_loads(mb.decode("utf-8"))
        rows = [strict_json_loads(line) for line in tb.decode("utf-8").splitlines()]
        bootstrap = metadata["native_bootstrap"]
        route = runtime_route_contract(original, bootstrap["spawn_point_index"])
        cfg = collection_for_trial(manifest, trial, bootstrap["spawn_point_index"])
        config = study_controller_config_from_dict(manifest["controller_config"])
        condition = condition_for_trial(trial)
        expected_binding = {"trial": trial, "manifest_sha256": canonical_sha256(manifest),
            "selected_pipeline_id": selection["pipeline_id"], "controller_config": config.to_dict(),
            "controller_config_sha256": config.sha256, "adapter_dependency_sha256": adapter_dependency_hashes()}
        _require(metadata["status"] == "COMPLETE" and metadata["scope"] == SCOPE and metadata["method"] == trial["method"]
                 and metadata["route_id"] == route["route_id"] and metadata["episode_id"] == root.name
                 and metadata["geometry_case"] == "heldout_route"
                 and metadata["requested_initial_reference_m"] == trial["initial_state"]["initial_reference_m"]
                 and metadata["route_contract"] == route and metadata["declared_topology_route"] == original
                 and metadata["collection_config"] == cfg.to_dict() and metadata["condition"] == condition
                 and metadata["final_evaluation"] == expected_binding, "final-native-plan-binding-mismatch")
        _require(metadata.get("rgb_correction_study") == {"study_id": manifest["study_id"], "arm": trial["arm"], "pair_id": trial["pair_id"], "evidence_role": trial["evidence_role"], "independent_final_evaluation": False}, "study-native-role-binding")
        _require(bootstrap["rule"] == "nearest-live-spawn-XY-to-frozen-start-then-lowest-index"
                 and type(bootstrap["spawn_count"]) is int and 0 <= bootstrap["spawn_point_index"] < bootstrap["spawn_count"]
                 and 0 <= _finite(bootstrap["height_difference_m"]) <= BOOTSTRAP_HEIGHT_TOLERANCE_M
                 and _finite(bootstrap["distance_xy_m"]) >= 0, "final-bootstrap-record-invalid")
        _require(metadata["validation_config"] == validation_policy(manifest, trial)
                 and metadata["controller_config"] == load_controller_config().to_dict()
                 and metadata["controller_config_sha256"] == load_controller_config().sha256
                 and metadata["baseline_config"] == manifest["baseline_config"]
                 and metadata["baseline_config_sha256"] == baseline_config_from_dict(manifest["baseline_config"]).sha256
                 and metadata["evaluation_config"] == manifest["evaluation_config"], "final-configuration-mismatch")
        evaluation_config = evaluation_config_from_dict(manifest["evaluation_config"])
        _require(metadata["evaluation_config_sha256"] == evaluation_config.sha256, "final-evaluation-hash-mismatch")
        _require(_same(metadata["camera_calibration"], _calibration(cfg, _mount_matrix(cfg)))
                 and metadata["applied_manifest"].get("applied_verified") is True,
                 "final-actual-camera-or-readback-mismatch")
        _validate_native_application(metadata, cfg, route, trial)
        _require(abs(metadata["initialization"]["spawn_settled_transform"]["location"]["z"] - original["start"]["world_location"]["z"])
                 <= BOOTSTRAP_HEIGHT_TOLERANCE_M, "final-settled-elevation-mismatch")
        _cleanup(metadata["cleanup"])
        _require(metadata["event_callback_errors"] == [] and rows and len(rows) == metadata["step_count"]
                 and metadata["trace_sha256"] == hashlib.sha256(tb).hexdigest(), "final-source-count-hash-or-events-mismatch")
        events = metadata["events"]; deduplicate_events(events)
        reference = trial["method"] == "privileged"
        _require(metadata["perception_uses_route"] is reference and metadata["perception_uses_vehicle_pose"] is reference
                 and metadata["prediction_inputs"] == ("privileged-original-route-and-same-frame-pose" if reference
                    else "rgb-and-fixed-nominal-calibration")
                 and metadata["latency_scope"] == (REFERENCE_LATENCY_SCOPE if reference else LATENCY_SCOPE)
                 and metadata["detailed_latency_required"] is (not reference), "final-perception-input-or-timing-scope-mismatch")
        if not reference:
            from .rgb_correction_study import validate_arm_predictor as _validate_predictor
            actual_identity, identity = copy.deepcopy(metadata["native_model"]), copy.deepcopy(predictor.identity)
            observed_device = actual_identity.pop("device", None)
            _require(observed_device in ("cpu", "cuda"), "final-model-device-invalid")
            identity.pop("device", None)
            _require(_same(actual_identity, identity, 1e-12), "final-model-identity-mismatch")
            if trial["method"] == "M1":
                _validate_predictor(predictor, selection)
            else:
                _require(identity.get("variant") == "B0", "final-B0-model-mismatch")
        policy = metadata["validation_config"]
        tracker = FixedRouteTracker(route, max_speed_mps=policy["maximum_speed_mps"], minimum_dt_seconds=.05)
        geometry = _geometry(metadata)
        controller = build_controller(config, geometry)
        start = _finite(metadata["evaluation_start_reference_m"])
        goal = trial["initial_state"]["initial_reference_m"] + manifest["route_length_m"]
        _require(abs(start - trial["initial_state"]["initial_reference_m"]) <= .5
                 and metadata["evaluation_goal_reference_m"] == goal
                 and abs(metadata["evaluation_route_length_m"] - (goal - start)) <= 1e-6,
                 "final-initial-reference-or-effective-length-mismatch")
        evaluator = EvaluationTracker(evaluation_config, goal - start, start, condition)
        previous, snapshots, images, maximum = None, [], {}, 0.
        for index, row in enumerate(rows):
            _require(type(row) is dict and set(row) == BASELINE_ROW_KEYS and row["index"] == index
                     and type(row["schema_version"]) is int and row["schema_version"] == 1, "final-row-schema-or-index-mismatch")
            frame, stamp = row["frame"], _finite(row["timestamp"])
            _require(type(frame) is int and frame >= 0 and stamp >= 0
                     and all(type(row[k]) is int and row[k] == frame for k in ("world_frame", "snapshot_frame", "sensor_frame", "command_source_frame"))
                     and abs(row["snapshot_timestamp"] - stamp) <= 1e-6 and abs(row["sensor_timestamp"] - stamp) <= 1e-6
                     and abs(row["delta_seconds"] - .05) <= 1e-6 and _finite(row["tick_to_rgb_ms"]) >= 0,
                     "final-frame-or-timestamp-alignment-mismatch")
            _require(previous is None or (frame == previous["frame"] + 1 and abs(stamp - previous["timestamp"] - .05) <= 1e-6),
                     "final-frame-sequence-mismatch")
            terminal = index == len(rows) - 1
            _require(row["terminal"] is terminal and row["command_target_frame"] == (None if terminal else frame + 1),
                     "final-terminal-or-target-frame-mismatch")
            pose = _transform(row["vehicle"]["transform"])
            speed = math.sqrt(sum(_finite(value) ** 2 for value in row["vehicle"]["velocity"].values()))
            _require(abs(_finite(row["speed_mps"]) - speed) <= 1e-5, "final-speed-readback-mismatch")
            applied = cfg.vehicle.control.to_dict() if previous is None else {k: previous["command"][k] for k in CONTROL_KEYS}
            _require(_same(_control(row["vehicle"]["applied_control"]), applied), "final-command-application-mismatch")
            tracking = tracker.observe(pose, frame, stamp)
            _require(_same(row["tracking"], tracking), "final-original-route-GT-replay-mismatch")
            validate_waypoint_label(tracking["label"], {"frame": frame, "vehicle": {"transform": pose}},
                                    {"route_id": route["route_id"], "route_contract": route})
            if index == 0:
                yaw = route_initial_location(route, trial["initial_state"]["initial_reference_m"])[1]
                _require(abs(tracking["reference_arc_m"] - start) <= 1e-6
                         and abs(tracking["lateral_error_m"] - condition["lateral_m"]) <= policy["initial_lateral_tolerance_m"]
                         and abs((pose["rotation"]["yaw"] - yaw - condition["yaw_deg"] + 180.) % 360. - 180.) <= policy["initial_yaw_tolerance_deg"],
                         "final-physical-initial-condition-mismatch")
            vm, cm = pose["matrix"], _transform(row["camera"]["transform"])["matrix"]
            relative = [[sum(vm[k][r] * cm[k][c] for k in range(3)) if c < 3 else
                         sum(vm[k][r] * (cm[k][3] - vm[k][3]) for k in range(3)) for c in range(4)] for r in range(3)] + [[0., 0., 0., 1.]]
            _require(_same(relative, metadata["camera_calibration"]["requested_mount_matrix"], .002), "final-camera-mount-mismatch")
            _require(abs(_finite(row["body_lane_margin_m"]) - _body_margin(pose, tracking, metadata["bounding_box"], route)) <= 1e-5,
                     "final-body-margin-replay-mismatch")
            _native_angles(row["native_wheel_angles_deg"])
            _require(all(abs(angle) <= geometry.max_steer_angle_deg + .5 for angle in row["native_wheel_angles_deg"].values()),
                     "final-native-wheel-angle-exceeds-measured-limit")
            image = row["image"]; path = f"images/{frame:08d}.png"
            _require(image["path"] == path and (image["width"], image["height"]) == (640, 360), "final-original-image-identity-mismatch")
            png = _safe_file(root, path).read_bytes()
            width, height, rgb = decode_rgb8_png(png, image["sha256"], (640, 360))
            verify_rgb_color(rgb, image["color_provenance"], width, height, require_witness=True)
            images[path] = hashlib.sha256(png).hexdigest()
            observed = row["perception"]
            if reference:
                reference_prediction = {k: copy.deepcopy(v) for k, v in observed.items() if k != "reference_timing"}
                rejection = reference_prediction.pop("controller_rejection", None)
                if rejection is not None:
                    _require(type(rejection) is dict and set(rejection) == {"exception_type", "error_code", "waypoints_ego_m"}
                             and rejection["exception_type"] == "ControllerError" and type(rejection["error_code"]) is str
                             and bool(rejection["error_code"]) and reference_prediction["status"] == "FAIL"
                             and reference_prediction.pop("error", None) == "final-controller-rejection"
                             and reference_prediction["waypoints_ego_m"] is None, "final-reference-rejection-schema-invalid")
                    reference_prediction.update(status="PASS", waypoints_ego_m=rejection["waypoints_ego_m"])
                _require(_same(reference_prediction,
                    {"status": "PASS", "waypoints_ego_m": tracking["label"]["waypoints_ego_m"],
                     "diagnostics": {"method": "privileged", "input": "privileged-original-route"}, "timing_ms": 0.}),
                    "final-privileged-GT-replay-mismatch")
                rt = observed["reference_timing"]
                _require(set(rt) == {"label_generation_ms", "controller_ms", "total_processing_ms"}
                         and all(_finite(value) >= 0 for value in rt.values())
                         and abs(rt["total_processing_ms"] - rt["label_generation_ms"] - rt["controller_ms"]) <= 1e-6,
                         "final-reference-timing-mismatch")
            else:
                _require(observed["pipeline_timing"]["device"] == observed_device, "final-recorded-timing-device-mismatch")
                maximum = max(maximum, _perception_replay(predictor, rgb, dict(observed, _legacy_latency=row["latency_ms"]),
                    method=trial["method"], width=width, height=height))
            rejection = observed.get("controller_rejection")
            if rejection:
                try:
                    controller.step(rejection["waypoints_ego_m"], speed, dt_seconds=.05)
                except ControllerError as error:
                    _require(str(error) == rejection["error_code"], "final-controller-rejection-replay-mismatch")
                else:
                    _require(False, "final-recorded-controller-rejection-not-reproduced")
                command = None
            else:
                command = controller.step(observed["waypoints_ego_m"], speed, dt_seconds=.05) if observed["status"] == "PASS" else None
            _require(_same(row["command"], command), "final-controller-replay-mismatch")
            latency = row["latency_ms"]
            _require(set(latency) == {"preprocess_ms", "inference_ms", "controller_ms", "pipeline_ms"}
                     and all(_finite(value) >= 0 for value in latency.values())
                     and abs(latency["pipeline_ms"] - math.fsum(latency[k] for k in ("preprocess_ms", "inference_ms", "controller_ms"))) <= 1e-5,
                     "final-latency-sum-mismatch")
            if reference:
                _require(_same(latency, {"preprocess_ms": rt["label_generation_ms"], "inference_ms": 0.,
                    "controller_ms": rt["controller_ms"], "pipeline_ms": rt["total_processing_ms"]}), "final-reference-latency-scope-mismatch")
            technical = observed["status"] != "PASS" or speed > policy["maximum_speed_mps"]
            _require(row["technical_failure"] is technical and row["evaluation_callback_errors"] == []
                     and (command is not None or (terminal and technical)), "final-technical-failure-record-mismatch")
            remaining = route["route_length_m"] - tracking["reference_arc_m"]
            _require(abs(row["remaining_horizon_m"] - remaining) <= 1e-6, "final-route-horizon-mismatch")
            snapshot = row["evaluation_events"]
            _require(len(snapshot) >= len(snapshots) and _same(snapshot[:len(snapshots)], snapshots)
                     and _same(events[:len(snapshot)], snapshot) and all(event["frame"] <= frame for event in snapshot),
                     "final-event-snapshot-mismatch")
            decision = evaluator.update(row, events=snapshot, technical_failure=technical, remaining_horizon_m=remaining)
            _require(_same(row["evaluation"], decision) and row["termination_reason"] == decision["reason"]
                     and row["terminal"] is decision["terminated"], "final-termination-replay-mismatch")
            previous, snapshots = row, snapshot
        _require(metadata["drive_end_frame"] == rows[-1]["frame"], "final-drive-end-frame-mismatch")
        stopping = _stopping(metadata, rows[-1], route)
        _require(_same(metadata["non_driving_events"], [event for event in events
            if not rows[0]["frame"] <= event["frame"] <= rows[-1]["frame"]]), "final-non-driving-event-set-mismatch")
        metrics = evaluate_episode(dict(metadata, termination_reason=rows[-1]["termination_reason"]), rows, events, config=evaluation_config)
        _require(metrics["integrity_valid"] is True and _same(metrics, metadata["evaluation_metrics"])
                 and metadata["termination_reason"] == metrics["termination_reason"], "final-metric-replay-mismatch")
        _require({p.relative_to(root).as_posix() for p in (root / "images").rglob("*") if p.is_file()} == set(images),
                 "final-missing-or-orphan-image")
        _require(_safe_file(root, "episode.json").read_bytes() == mb and _safe_file(root, "steps.jsonl").read_bytes() == tb
                 and all(hashlib.sha256(_safe_file(root, path).read_bytes()).hexdigest() == digest for path, digest in images.items()),
                 "final-sources-changed-during-replay")
        report.update(status="PASS", frame_count=len(rows), metrics=metrics, driving_success=metrics["driving_success"],
                      maximum_coordinate_replay_difference_m=maximum, stopping=stopping,
                      metadata_sha256=hashlib.sha256(mb).hexdigest(), trace_sha256=hashlib.sha256(tb).hexdigest(),
                      image_sha256=images, original_RGB_native_GT_control_events_replayed=True)
    except Exception as error:
        report["errors"].append({"type": type(error).__name__, "code": str(error)})
    return report
