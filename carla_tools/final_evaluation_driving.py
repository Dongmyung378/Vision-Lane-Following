"""Execute and replay a pre-frozen, paired held-out driving plan.

Importing this module does not load CARLA or a model. The historical native
loop remains unchanged: isolated FunctionType namespaces substitute only the
final config parser, owned session, topology verifier and timed route tracker.
No original function globals are mutated. Their source bytes belong to the
evaluation freeze, together with this adapter's source.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time
from types import FunctionType

from .contracts import canonical_sha256, strict_json_loads
from .final_evaluation import (FinalEvaluationError, TOWNS, runtime_route_contract,
    validate_manifest, validate_route)

SCOPE = "frozen-heldout-paired-driving-evaluation"
WEATHERS = ("ClearNoon", "HardRainNoon", "ClearSunset")
ANCHOR_POSITION_TOLERANCE_M = .01
ANCHOR_ROTATION_TOLERANCE_DEG = .1
BOOTSTRAP_HEIGHT_TOLERANCE_M = .5
LATENCY_SCOPE = "BGRA conversion, fixed-nominal RGB perception and shared controller; excludes simulator wait, GT and disk"
REFERENCE_LATENCY_SCOPE = "original-route same-frame label construction plus shared controller; reported separately from RGB perception"


def _require(condition, code):
    if not condition:
        raise FinalEvaluationError(code)


def adapter_dependency_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ("carla_tools/config.py", "carla_tools/session.py", "carla_tools/driving.py",
             "carla_tools/driving_route.py", "carla_tools/geometry_survey.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _isolated(function, **dependencies):
    """Retain original bytecode/closure with an explicit, copied namespace."""
    _require(set(dependencies) <= set(function.__globals__), "final-adapter-unknown-dependency")
    result = FunctionType(function.__code__, dict(function.__globals__, **dependencies),
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = copy.copy(function.__kwdefaults__)
    return result


def final_collection_config_from_dict(value, *, allowed_weather_presets=WEATHERS):
    """Extend only town/weather; run every other historical config check."""
    from .config import collection_config_from_dict
    _require(type(value) is dict and value.get("map") in TOWNS,
             "final-unsupported-town")
    _require(type(allowed_weather_presets) in (tuple, list) and bool(allowed_weather_presets)
             and all(type(preset) is str and preset in WEATHERS for preset in allowed_weather_presets)
             and len(allowed_weather_presets) == len(set(allowed_weather_presets)),
             "final-unsupported-weather-allowlist")
    weather = value.get("weather")
    _require(type(weather) is dict and weather.get("preset") in allowed_weather_presets,
             "final-unsupported-weather")
    normalized = copy.deepcopy(value)
    town, preset = normalized["map"], weather["preset"]
    normalized["map"], normalized["weather"]["preset"] = "Town01", "ClearNoon"
    parsed = collection_config_from_dict(normalized)
    return replace(parsed, map=town, weather=replace(parsed.weather, preset=preset))


def condition_for_trial(trial):
    state = trial["initial_state"]
    return {"name": state["state_id"], "lateral_m": state["lateral_m"], "yaw_deg": state["yaw_deg"]}


def collection_for_trial(manifest, trial, spawn_point_index):
    _require(trial in manifest["trials"], "final-trial-not-frozen")
    value = copy.deepcopy(manifest["collection_config"])
    value["map"], value["seed"] = trial["town"], trial["initial_state"]["seed"]
    value["weather"]["preset"] = trial["environment"]["weather"]
    value["camera"]["mount"]["rotation"]["pitch"] += trial["environment"]["camera_pitch_delta_deg"]
    value["vehicle"]["spawn_point_index"] = spawn_point_index
    value["verification"]["warmup_frames"] = manifest["warmup_frames"]
    return final_collection_config_from_dict(value)


def validation_policy(manifest, trial):
    from .driving import load_validation_config
    value = load_validation_config()
    value["goal_reference_m"] = trial["initial_state"]["initial_reference_m"] + manifest["route_length_m"]
    value["initial_conditions"] = [condition_for_trial(row) for row in manifest["trials"]
                                   if row["method"] == "B0" and row["route_id"] == trial["route_id"]
                                   and row["group_id"] == trial["group_id"]]
    return value


def resolve_bootstrap(route, carla_map):
    """Pick nearest actual spawn in XY, index tie-break; never invent one."""
    validate_route(route)
    points = carla_map.get_spawn_points()
    _require(bool(points), "final-live-map-has-no-bootstrap-spawns")
    _require(all(type(getattr(point.location, axis)) in (int, float)
                 and math.isfinite(getattr(point.location, axis)) for point in points for axis in ("x", "y", "z")),
             "final-bootstrap-nonfinite")
    start = route["start"]["world_location"]
    distances = [math.hypot(point.location.x - start["x"], point.location.y - start["y"])
                 for point in points]
    _require(all(math.isfinite(value) for value in distances), "final-bootstrap-nonfinite")
    index = min(range(len(points)), key=lambda i: (distances[i], i))
    height = abs(points[index].location.z - start["z"])
    _require(math.isfinite(height) and height <= BOOTSTRAP_HEIGHT_TOLERANCE_M,
             "final-bootstrap-elevation-mismatch")
    return {"spawn_point_index": index, "spawn_count": len(points),
            "spawn_locations_m": [{axis: float(getattr(point.location, axis)) for axis in ("x", "y", "z")} for point in points],
            "distance_xy_m": distances[index], "height_difference_m": height,
            "rule": "nearest-live-spawn-XY-to-frozen-start-then-lowest-index"}


def verify_topology_in_world(route, carla_map, carla_module):
    """Prove the frozen corridor against native XODR; do not reselect a lane."""
    from .contracts import validate_route_contract
    validate_route_contract(route)
    _require(carla_map.name.replace("\\", "/").rsplit("/", 1)[-1] == route["town"]
             and hashlib.sha256(carla_map.to_opendrive().encode("utf-8")).hexdigest() == route["opendrive_sha256"],
             "final-live-map-identity-mismatch")
    for anchor in route["centerline_anchor_sequence"]:
        waypoint = carla_map.get_waypoint_xodr(anchor["road_id"], anchor["lane_id"], anchor["s"])
        _require(waypoint is not None, "final-live-anchor-unavailable")
        _require(all(getattr(waypoint, key) == anchor[key] for key in ("road_id", "section_id", "lane_id"))
                 and abs(waypoint.s - anchor["s"]) <= .001
                 and waypoint.is_junction is False and waypoint.lane_type == carla_module.LaneType.Driving
                 and len(waypoint.next(.05)) == 1, "final-live-anchor-topology-mismatch")
        _require(math.dist([getattr(waypoint.transform.location, axis) for axis in ("x", "y", "z")],
                          [anchor["world_location"][axis] for axis in ("x", "y", "z")]) <= ANCHOR_POSITION_TOLERANCE_M
                 and abs(waypoint.lane_width - anchor["lane_width_m"]) <= .005,
                 "final-live-anchor-position-or-width-mismatch")
        _require(all(abs((getattr(waypoint.transform.rotation, axis) - anchor["world_rotation"][axis] + 180.)
                         % 360. - 180.) <= ANCHOR_ROTATION_TOLERANCE_DEG for axis in ("pitch", "yaw", "roll")),
                 "final-live-anchor-heading-mismatch")


def _native_adapter(route, tracker_timings):
    from .driving import run_drive_trial
    from .driving_route import FixedRouteTracker
    from .session import CollectionSession
    enter = _isolated(CollectionSession.__enter__, collection_config_from_dict=final_collection_config_from_dict)

    class FinalSession(CollectionSession):
        def __enter__(self):
            result = enter(self)
            from .session import public_fields
            self.manifest["requested_weather"] = {"preset": self.cfg.weather.preset,
                "values": public_fields(getattr(self._carla.WeatherParameters, self.cfg.weather.preset))}
            return result

        def verify_applied(self):
            super().verify_applied()
            _require(abs(self.vehicle.get_transform().location.z - route["start"]["world_location"]["z"])
                     <= BOOTSTRAP_HEIGHT_TOLERANCE_M, "final-settled-elevation-mismatch")

    class TimedTracker(FixedRouteTracker):
        def observe(self, *args, **kwargs):
            started = time.perf_counter()
            result = super().observe(*args, **kwargs)
            tracker_timings["label_generation_ms"] = (time.perf_counter() - started) * 1000.
            return result

    return _isolated(run_drive_trial, collection_config_from_dict=final_collection_config_from_dict,
                     CollectionSession=FinalSession, verify_geometry_route_in_world=verify_topology_in_world,
                     FixedRouteTracker=TimedTracker)


def build_trial_policy(route, trial, manifest, selection, predictor, tracker_timings):
    from .baseline import baseline_config_from_dict
    from .baseline_driving import BaselineTrialPolicy
    from .controller import ControllerError
    from .controller_study import build_controller, study_controller_config_from_dict
    from .evaluation import evaluation_config_from_dict
    config = study_controller_config_from_dict(manifest["controller_config"])
    reference = trial["method"] == "privileged"

    class Policy(BaselineTrialPolicy):
        scope = SCOPE

        def __init__(self):
            super().__init__(baseline_config_from_dict(manifest["baseline_config"]),
                evaluation_config_from_dict(manifest["evaluation_config"]), route, condition_for_trial(trial),
                predictor=predictor, goal_reference_m=trial["initial_state"]["initial_reference_m"] + manifest["route_length_m"])
            self.method = trial["method"]
            self.initial_reference_m = trial["initial_state"]["initial_reference_m"]
            self.selected_controller = None

        def configure_metadata(self, metadata):
            super().configure_metadata(metadata)
            metadata.update(final_evaluation={"trial": copy.deepcopy(trial), "manifest_sha256": canonical_sha256(manifest),
                "selected_pipeline_id": selection["pipeline_id"], "controller_config": config.to_dict(),
                "controller_config_sha256": config.sha256, "adapter_dependency_sha256": adapter_dependency_hashes()},
                requested_initial_reference_m=self.initial_reference_m,
                detailed_latency_required=not reference,
                latency_scope=REFERENCE_LATENCY_SCOPE if reference else LATENCY_SCOPE)
            if reference:
                metadata.update(prediction_inputs="privileged-original-route-and-same-frame-pose",
                                perception_uses_route=True, perception_uses_vehicle_pose=True)
            else:
                metadata["native_model"] = predictor.identity

        def _controller(self, controller):
            if self.selected_controller is None:
                self.selected_controller = build_controller(config, controller.geometry)
            return self.selected_controller

        def command(self, image, speed_mps, controller):
            from scripts.carla_rgb_probe import rgb_from_bgra
            selected = self._controller(controller)
            timer = predictor.timer()
            began = time.perf_counter()
            rgb, _ = timer.measure("bgra_conversion_ms", lambda: rgb_from_bgra(bytes(image.raw_data), image.width, image.height))
            converted = time.perf_counter()
            prediction = predictor.predict_profiled(rgb, timer, width=image.width, height=image.height)
            json.dumps(prediction, allow_nan=False)
            predicted = time.perf_counter()
            command = None
            if prediction["status"] == "PASS":
                try:
                    command = timer.measure("controller_ms", lambda: selected.step(prediction["waypoints_ego_m"], speed_mps, dt_seconds=.05))
                except ControllerError as error:
                    rejection = {"exception_type": "ControllerError", "error_code": str(error),
                                 "waypoints_ego_m": copy.deepcopy(prediction["waypoints_ego_m"])}
                    if "learned_waypoints_ego_m" in prediction:
                        rejection["learned_waypoints_ego_m"] = prediction["learned_waypoints_ego_m"]
                        prediction["learned_waypoints_ego_m"] = None
                    prediction.update(status="FAIL", error="final-controller-rejection", waypoints_ego_m=None,
                                      controller_rejection=rejection)
            ended = time.perf_counter()
            prediction["pipeline_timing"] = timer.finish()
            return command, prediction, {"preprocess_ms": (converted - began) * 1000,
                "inference_ms": (predicted - converted) * 1000, "controller_ms": (ended - predicted) * 1000,
                "pipeline_ms": (ended - began) * 1000}

    if reference:
        def privileged_command(self, label, speed_mps, controller):
            started = time.perf_counter()
            prediction = {"status": "PASS", "waypoints_ego_m": copy.deepcopy(label["waypoints_ego_m"]),
                "diagnostics": {"method": "privileged", "input": "privileged-original-route"}, "timing_ms": 0.}
            try:
                command = self._controller(controller).step(label["waypoints_ego_m"], speed_mps, dt_seconds=.05)
            except ControllerError as error:
                command = None
                prediction.update(status="FAIL", error="final-controller-rejection", waypoints_ego_m=None,
                    controller_rejection={"exception_type": "ControllerError", "error_code": str(error),
                                          "waypoints_ego_m": copy.deepcopy(label["waypoints_ego_m"])})
            elapsed = (time.perf_counter() - started) * 1000
            label_ms = tracker_timings["label_generation_ms"]
            prediction["reference_timing"] = {"label_generation_ms": label_ms, "controller_ms": elapsed,
                                              "total_processing_ms": label_ms + elapsed}
            return command, prediction, {
                "preprocess_ms": label_ms, "inference_ms": 0., "controller_ms": elapsed, "pipeline_ms": label_ms + elapsed}
        Policy.privileged_command = privileged_command
    return Policy()


def run_heldout_trial(route, trial, output, *, manifest, selection, predictor,
                      carla_module, carla_map, session_factory=None):
    """Low-level native hook; callers must verify the whole freeze first."""
    _require(trial in manifest["trials"] and trial["route_id"] == route["route_id"], "final-trial-not-frozen")
    bootstrap = resolve_bootstrap(route, carla_map)
    lowered = runtime_route_contract(route, bootstrap["spawn_point_index"])
    verify_topology_in_world(lowered, carla_map, carla_module)
    tracker_timings = {}
    policy = build_trial_policy(lowered, trial, manifest, selection, predictor, tracker_timings)
    # Bind declared topology separately from its measured native bootstrap.
    configure = policy.configure_metadata
    def metadata_hook(metadata):
        configure(metadata)
        metadata["declared_topology_route"] = copy.deepcopy(route)
        metadata["native_bootstrap"] = copy.deepcopy(bootstrap)
    policy.configure_metadata = metadata_hook
    from .controller import load_controller_config
    operation = _native_adapter(lowered, tracker_timings)
    return operation(lowered, "heldout_route", condition_for_trial(trial), output,
        collection_for_trial(manifest, trial, bootstrap["spawn_point_index"]), carla_module=carla_module,
        validation_config=validation_policy(manifest, trial), controller_config=load_controller_config(),
        trial_policy=policy, allowed_weather_presets=WEATHERS, session_factory=session_factory)


def _validate_native_application(metadata, cfg, route, trial):
    """Validate persisted native requests/readbacks, not just a success flag."""
    from .driving_validation import _finite, _same
    from .driving import route_initial_location
    from .driving_route import _pose_matrix, _transform, perturbation_transform
    from .session import attributes_match, fields_match
    applied = metadata["applied_manifest"]
    _require(applied["carla_version"] == cfg.carla_version and applied["map"] == cfg.map
             and applied["seed"] == cfg.seed and applied["config_sha256"] == cfg.sha256
             and applied["versions"] == {"client": cfg.carla_version, "server": cfg.carla_version}
             and applied["seed_scope"] == "recorded-fixed-configuration-no-randomized-actors"
             and fields_match(cfg.world.to_dict(), applied["world_settings"]), "final-native-configuration-readback-mismatch")
    weather = applied["weather"]
    _require(weather["preset"] == cfg.weather.preset and weather == applied["requested_weather"]
             and type(weather["values"]) is dict and bool(weather["values"])
             and {"cloudiness", "precipitation", "sun_altitude_angle"} <= set(weather["values"])
             and all(type(value) in (int, float) and math.isfinite(value) for value in weather["values"].values()),
             "final-native-weather-readback-mismatch")
    vehicle, camera = applied["vehicle"], applied["camera"]
    bootstrap = metadata["native_bootstrap"]
    _require(vehicle["blueprint"] == cfg.vehicle.blueprint and vehicle["attributes"] == dict(cfg.vehicle.attributes)
             and vehicle["spawn_point_index"] == cfg.vehicle.spawn_point_index
             and vehicle["spawn_point_count"] == bootstrap["spawn_count"]
             and _same(vehicle["control_requested"], cfg.vehicle.control.to_dict())
             and _same(vehicle["control_applied"], cfg.vehicle.control.to_dict()) and vehicle["control_verified"] is True,
             "final-native-vehicle-readback-mismatch")
    expected_attributes = {"image_size_x": str(cfg.camera.width), "image_size_y": str(cfg.camera.height),
        "fov": str(cfg.camera.fov), "sensor_tick": str(cfg.camera.sensor_tick),
        "enable_postprocess_effects": str(cfg.camera.enable_postprocess_effects).lower(),
        "gamma": str(cfg.camera.gamma), "lens_k": str(cfg.camera.lens_k), "lens_kcube": str(cfg.camera.lens_kcube)}
    _require(camera["blueprint"] == cfg.camera.blueprint and attributes_match(expected_attributes, camera["attributes"])
             and camera["parent_actor_id"] == vehicle["actor_id"] and camera["attachment_type"] == "Rigid"
             and camera["mount_verified"] is True and _same(camera["intrinsics"], cfg.camera.intrinsics())
             and _same(camera["mount_requested"], cfg.camera.mount.to_dict())
             and _same(camera["mount_relative_matrix"], camera["requested_mount_matrix"], .002)
             and _same(camera["requested_mount_matrix"], metadata["camera_calibration"]["requested_mount_matrix"])
             and 0 <= _finite(camera["mount_matrix_max_abs_error"]) <= .002
             and camera["mount_matrix_tolerance"] == .002, "final-native-camera-readback-mismatch")
    locations = bootstrap["spawn_locations_m"]
    _require(type(locations) is list and len(locations) == bootstrap["spawn_count"]
             and all(type(point) is dict and set(point) == {"x", "y", "z"}
                     and all(type(value) in (int, float) and math.isfinite(value) for value in point.values()) for point in locations),
             "final-bootstrap-location-inventory-invalid")
    start = route["start"]["world_location"]
    distances = [math.hypot(point["x"] - start["x"], point["y"] - start["y"]) for point in locations]
    chosen = min(range(len(locations)), key=lambda i: (distances[i], i))
    _require(chosen == bootstrap["spawn_point_index"] and abs(distances[chosen] - bootstrap["distance_xy_m"]) <= 1e-6
             and abs(abs(locations[chosen]["z"] - start["z"]) - bootstrap["height_difference_m"]) <= 1e-6
             and _same(vehicle["spawn_transform"]["location"], locations[chosen]), "final-bootstrap-selection-replay-mismatch")
    initialization = metadata["initialization"]
    settled = _transform(initialization["spawn_settled_transform"])
    initial_reference = trial["initial_state"]["initial_reference_m"]
    location, yaw = route_initial_location(route, initial_reference)
    location["z"] = settled["location"]["z"]
    rotation = dict(settled["rotation"], yaw=yaw)
    center = {"location": location, "rotation": rotation, "matrix": _pose_matrix(location, rotation)}
    expected = {"policy": "pinned-route-start-xy-and-tangent-preserve-settled-height",
        "spawn_settled_transform": settled, "center_requested_transform": center,
        "perturbed_requested_transform": perturbation_transform(center, trial["initial_state"]["lateral_m"], trial["initial_state"]["yaw_deg"])}
    if initial_reference != 0.:
        expected.update(policy="pinned-original-route-reference-xy-and-tangent-preserve-settled-height",
                        initial_reference_m=initial_reference)
    _require(_same(initialization, expected), "final-initialization-request-replay-mismatch")


def _perception_replay(predictor, rgb, observed, *, method, width, height):
    """Check fresh RGB output and exact recorded-raw correction separately."""
    from .driving_validation import _same
    from .pipeline_latency import validate_pipeline_timing
    from .runtime_driving import validate_recorded_correction
    from .guarded_path_correction import validate_recorded_guarded_correction
    reproduced = predictor.predict(rgb, width=width, height=height)
    actual = copy.deepcopy(observed)
    rejection = actual.pop("controller_rejection", None)
    if rejection is not None:
        expected = {"exception_type", "error_code", "waypoints_ego_m"}
        if method == "M1":
            expected.add("learned_waypoints_ego_m")
        _require(type(rejection) is dict and set(rejection) == expected and rejection["exception_type"] == "ControllerError"
                 and type(rejection["error_code"]) is str and bool(rejection["error_code"])
                 and actual["status"] == "FAIL" and actual["error"] == "final-controller-rejection"
                 and actual["waypoints_ego_m"] is None, "final-controller-rejection-invalid")
        actual.update(status="PASS", error=None, waypoints_ego_m=rejection["waypoints_ego_m"])
        if method == "M1":
            _require(actual["learned_waypoints_ego_m"] is None, "final-rejected-raw-not-separated")
            actual["learned_waypoints_ego_m"] = rejection["learned_waypoints_ego_m"]
    timing = validate_pipeline_timing(actual["pipeline_timing"])
    if method == "M1":
        from .runtime_driving import validate_runtime_timing
        validate_runtime_timing(timing, observed["_legacy_latency"], observed["status"])
        if rejection is not None:
            from .pipeline_latency import STAGES
            _require(timing["executed_stages"] == list(STAGES) and timing["failed_stages"] == ["controller_ms"]
                     and (timing["device"] != "cuda" or set(timing["synchronized_stages"]) ==
                          {"learned_preprocessing_ms", "forward_decode_ms"}), "final-rejection-stage-coverage")
    else:
        stages = ["bgra_conversion_ms", "forward_decode_ms"] + (["controller_ms"] if actual["status"] == "PASS" else [])
        _require(timing["executed_stages"] == stages and timing["device"] == "cpu"
                 and timing["failed_stages"] == (["controller_ms"] if rejection else []), "final-B0-timing-coverage")
        legacy, detail = observed["_legacy_latency"], timing["stages_ms"]
        _require(timing["total_processing_ms"] + 1e-5 >= legacy["pipeline_ms"]
                 and detail["bgra_conversion_ms"] <= legacy["preprocess_ms"] + 1e-5
                 and detail["forward_decode_ms"] <= legacy["inference_ms"] + 1e-5
                 and detail["controller_ms"] <= legacy["controller_ms"] + 1e-5, "final-B0-interval-mismatch")
    maximum = 0.
    for key in ("waypoints_ego_m", "learned_waypoints_ego_m"):
        if key not in actual and key not in reproduced:
            continue
        a, b = actual.get(key), reproduced.get(key)
        if actual["status"] == "PASS":
            from .learning_data import validate_waypoint_points
            validate_waypoint_points(a); validate_waypoint_points(b)
            delta = max(abs(x - y) for p, q in zip(a, b) for x, y in zip(p, q))
            _require(delta <= (.05 if method == "M1" else 1e-8), "final-RGB-coordinate-replay-mismatch")
            maximum = max(maximum, delta)
        else:
            _require(a is None and b is None, "final-failed-prediction-retains-points")
    ignored = ("timing_ms", "pipeline_timing", "_legacy_latency", "waypoints_ego_m", "learned_waypoints_ego_m")
    aa, bb = ({k: copy.deepcopy(v) for k, v in item.items() if k not in ignored} for item in (actual, reproduced))
    if method == "M1" and actual["status"] == "PASS":
        if predictor.correction == "guarded_path":
            validate_recorded_guarded_correction(predictor, rgb, actual, width=width, height=height)
            aa["diagnostics"].pop("guarded_path_correction"); bb["diagnostics"].pop("guarded_path_correction")
        else:
            validate_recorded_correction(predictor, rgb, actual, width=width, height=height)
        aa["diagnostics"].pop("path_correction"); bb["diagnostics"].pop("path_correction")
    _require(_same(aa, bb, 1e-8), "final-RGB-content-replay-mismatch")
    return maximum


def validate_heldout_episode(directory, predictor, trial, *, manifest, routes, selection):
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
            from .selected_pipeline import _validate_predictor
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
