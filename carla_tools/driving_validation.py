"""Fresh, offline validation of privileged D8-D10 controller QA traces.

This gate certifies reproducible short-route controller/recovery development
checks. It does not certify a camera model, pilot dataset or held-out driving
performance. Source files are read again before PASS is published.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .capture import _calibration, _mount_matrix
from .color import verify_rgb_color
from .config import collection_config_from_dict
from .contracts import canonical_sha256, read_strict_json, route_sha256, strict_json_loads, validate_identifier, validate_waypoint_label
from .controller import PurePursuitController, VehicleGeometry, controller_config_from_dict
from .driving_route import FixedRouteTracker, _transform
from .geometry_survey import CASES, classify_geometry
from .projection import decode_rgb8_png


SCOPE = "D8-D10-privileged-controller-development-QA"
GATE_VERSION = "vlf-privileged-driving-d10-v1"
DEFAULT_VALIDATION_PATH = Path(__file__).resolve().parents[1] / "configs" / "driving_validation_d10.json"
CONTROL_KEYS = {"throttle", "brake", "steer", "hand_brake", "reverse"}
ROW_KEYS = {"schema_version", "index", "frame", "world_frame", "snapshot_frame", "sensor_frame", "timestamp",
            "snapshot_timestamp", "sensor_timestamp", "delta_seconds", "tick_to_rgb_ms", "vehicle", "camera",
            "speed_mps", "tracking", "command", "command_source_frame", "command_target_frame", "terminal",
            "body_lane_margin_m", "native_wheel_angles_deg", "image"}


class DrivingValidationError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _require(condition, code):
    if not condition:
        raise DrivingValidationError(code)


def _finite(value):
    try:
        _require(type(value) in (int, float) and math.isfinite(value), "nonfinite-driving-number")
        return float(value)
    except (OverflowError, TypeError):
        raise DrivingValidationError("nonfinite-driving-number") from None


def _integer(value, minimum=0):
    _require(type(value) is int and value >= minimum, "invalid-driving-integer")
    return value


def _same(actual: Any, expected: Any, tolerance=1e-5) -> bool:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return type(actual) is type(expected) and actual == expected
    if type(expected) in (int, float):
        return type(actual) in (int, float) and abs(_finite(actual) - _finite(expected)) <= tolerance
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(_same(actual[key], child, tolerance) for key, child in expected.items())
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(_same(a, b, tolerance) for a, b in zip(actual, expected))
    return False


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _control(value):
    _require(isinstance(value, dict) and set(value) == CONTROL_KEYS, "invalid-driving-control")
    for key in ("throttle", "brake"):
        _require(0 <= _finite(value[key]) <= 1, "invalid-driving-control-range")
    _require(-1 <= _finite(value["steer"]) <= 1, "invalid-driving-control-range")
    _require(type(value["hand_brake"]) is bool and type(value["reverse"]) is bool, "invalid-driving-control-flags")
    return value


def _geometry(metadata):
    proof = metadata["geometry_evidence"]
    _require(isinstance(proof, dict) and set(proof) == {"measurement_vehicle_transform", "wheels", "original_steering_curve",
             "applied_steering_curve", "position_units", "measurement_after_warmup", "steering_normalization"}, "invalid-native-geometry-proof")
    _require(proof["position_units"] == "world-centimetres" and proof["measurement_after_warmup"] is True
             and proof["steering_normalization"] == "measured-front-steer-limit-times-constant-factor", "invalid-native-geometry-proof")
    _require(_same(proof["applied_steering_curve"], [[0., 1.], [100., 1.]]), "native-steering-curve-not-constant")
    pose = _transform(proof["measurement_vehicle_transform"])
    matrix = pose["matrix"]
    inverse = [[matrix[column][row] for column in range(3)] + [-sum(matrix[column][row] * matrix[column][3] for column in range(3))] for row in range(3)]
    inverse.append([0., 0., 0., 1.])
    _require(isinstance(proof["wheels"], list) and len(proof["wheels"]) == 4, "invalid-native-wheel-count")
    wheels = []
    for item in proof["wheels"]:
        _require(isinstance(item, dict) and set(item) == {"position_cm", "max_steer_angle_deg"}, "invalid-native-wheel-proof")
        _require(isinstance(item["position_cm"], dict) and set(item["position_cm"]) == {"x", "y", "z"}, "invalid-native-wheel-position")
        wheels.append(SimpleNamespace(position=SimpleNamespace(**{key: _finite(value) for key, value in item["position_cm"].items()}),
                                      max_steer_angle=_finite(item["max_steer_angle_deg"])))
    physics = SimpleNamespace(wheels=wheels, steering_curve=[SimpleNamespace(x=x, y=y) for x, y in proof["applied_steering_curve"]])
    observed = VehicleGeometry.from_physics_control(physics, SimpleNamespace(get_inverse_matrix=lambda: inverse))
    _require(_same(metadata["vehicle_geometry"], observed.to_dict()), "native-vehicle-geometry-mismatch")
    # Validate the original curve independently even though the applied curve
    # is deliberately constant for an exact normalized-angle conversion.
    original = dict(observed.to_dict(), steering_curve=proof["original_steering_curve"])
    VehicleGeometry(**original)
    return observed


def _project_route_xy(point, route):
    candidates = []
    anchors = route["centerline_anchor_sequence"]
    for first, second in zip(anchors, anchors[1:]):
        a, b = first["world_location"], second["world_location"]
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        denominator = dx * dx + dy * dy
        _require(denominator > 0, "degenerate-driving-route-projection")
        fraction = min(1., max(0., ((point[0] - a["x"]) * dx + (point[1] - a["y"]) * dy) / denominator))
        x, y = a["x"] + fraction * dx, a["y"] + fraction * dy
        gap = math.hypot(point[0] - x, point[1] - y)
        lateral = ((point[1] - y) * dx - (point[0] - x) * dy) / math.sqrt(denominator)
        width = _finite(first["lane_width_m"]) + fraction * (_finite(second["lane_width_m"]) - _finite(first["lane_width_m"]))
        arc = first["arc_distance_m"] + fraction * (second["arc_distance_m"] - first["arc_distance_m"])
        candidates.append((gap, lateral, width, arc))
    return min(candidates, key=lambda item: item[0])


def _body_margin(pose, tracking, bbox, route=None):
    _require(isinstance(bbox, dict) and set(bbox) == {"location", "extent"}, "invalid-driving-bounding-box")
    for key in ("location", "extent"):
        _require(isinstance(bbox[key], dict) and set(bbox[key]) == {"x", "y", "z"}, "invalid-driving-bounding-box")
        for number in bbox[key].values():
            _finite(number)
    _require(all(bbox["extent"][key] > 0 for key in bbox["extent"]), "invalid-driving-bounding-box-extent")
    if route is not None:
        margins = []
        matrix = pose["matrix"]
        for x_sign in (-1., 1.):
            for y_sign in (-1., 1.):
                for z_sign in (-1., 1.):
                    local = [bbox["location"][axis] + sign * bbox["extent"][axis]
                             for axis, sign in zip(("x", "y", "z"), (x_sign, y_sign, z_sign))]
                    world = [sum(matrix[row][column] * local[column] for column in range(3)) + matrix[row][3] for row in range(3)]
                    _, lateral, width, _ = _project_route_xy(world, route)
                    margins.append(width / 2 - abs(lateral))
        return min(margins)
    yaw = math.radians(tracking["route_tangent_yaw_deg"])
    normal = [-math.sin(yaw), math.cos(yaw), 0.]
    normal_local = [sum(normal[row] * pose["matrix"][row][column] for row in range(3)) for column in range(3)]
    center_lateral = tracking["lateral_error_m"] + sum(normal_local[index] * bbox["location"][axis] for index, axis in enumerate(("x", "y", "z")))
    half_width = sum(abs(normal_local[index]) * bbox["extent"][axis] for index, axis in enumerate(("x", "y", "z")))
    return _finite(tracking["lane_width_m"]) / 2 - abs(center_lateral) - half_width


def _stopping(metadata, final_row, policy, route):
    """Verify braking with the last applied steer, then two parking readbacks."""
    expected = {"throttle": 0., "brake": 1., "steer": final_row["vehicle"]["applied_control"]["steer"],
                "hand_brake": False, "reverse": False}
    _require(_same(_control(metadata["stop_control"]), expected), "driving-stop-control-mismatch")
    parking = dict(expected, hand_brake=True)
    observations = metadata["stop_observations"]
    _require(isinstance(observations, list) and len(observations) >= policy["settling_frames"] + 2, "driving-stop-observations-missing")
    previous_frame, previous_time = final_row["frame"], final_row["timestamp"]
    first_stopped = None
    margins, last_speed, last_braking_speed = [], None, None
    braking_count = len(observations) - 2
    for index, observation in enumerate(observations):
        _require(isinstance(observation, dict) and set(observation) == {"frame", "timestamp", "vehicle", "speed_mps", "native_wheel_angles_deg", "body_lane_margin_m"}, "invalid-driving-stop-observation")
        frame, timestamp = _integer(observation["frame"]), _finite(observation["timestamp"])
        _require(frame == previous_frame + 1 and abs(timestamp - previous_time - .05) <= 1e-6, "driving-stop-frame-sequence-mismatch")
        vehicle = observation["vehicle"]
        _require(isinstance(vehicle, dict) and set(vehicle) == {"transform", "velocity", "applied_control"}, "invalid-driving-stop-vehicle")
        pose = _transform(vehicle["transform"])
        _require(isinstance(vehicle["velocity"], dict) and set(vehicle["velocity"]) == {"x", "y", "z"}, "invalid-driving-stop-velocity")
        speed = math.sqrt(sum(_finite(value) ** 2 for value in vehicle["velocity"].values()))
        _require(abs(_finite(observation["speed_mps"]) - speed) <= 1e-5 and speed <= policy["maximum_speed_mps"], "driving-stop-speed-mismatch")
        _require(_same(_control(vehicle["applied_control"]), expected if index < braking_count else parking), "driving-stop-control-readback-mismatch")
        angles = observation["native_wheel_angles_deg"]
        _require(isinstance(angles, dict) and set(angles) == {"front_left", "front_right"}
                 and all(abs(_finite(value)) <= 90. for value in angles.values()), "invalid-native-stop-steering-angle")
        _, lateral, _, _ = _project_route_xy([pose["location"]["x"], pose["location"]["y"]], route)
        _require(abs(lateral) <= policy["maximum_lateral_error_m"], "driving-stop-lateral-bound")
        margin = _body_margin(pose, {}, metadata["bounding_box"], route)
        _require(abs(_finite(observation["body_lane_margin_m"]) - margin) <= 1e-5
                 and margin >= policy["minimum_body_lane_margin_m"], "driving-stop-body-lane-margin-failed")
        elapsed = timestamp - final_row["timestamp"]
        if index < braking_count:
            _require(elapsed <= policy["maximum_stop_seconds"] + 1e-6, "driving-stop-deadline-exceeded")
            last_braking_speed = speed
            if first_stopped is None and speed <= policy["stop_speed_tolerance_mps"]:
                first_stopped = elapsed
        else:
            _require(speed <= policy["stop_speed_tolerance_mps"], "driving-parking-not-stationary")
        margins.append(margin)
        previous_frame, previous_time, last_speed = frame, timestamp, speed
    _require(first_stopped is not None and first_stopped <= policy["maximum_stop_seconds"] + 1e-6, "driving-stop-not-within-deadline")
    _require(last_braking_speed is not None and last_braking_speed <= policy["stop_speed_tolerance_mps"],
             "driving-parking-started-before-stop")
    _require(metadata["post_stop_frame"] == previous_frame and type(metadata["post_stop_frame"]) is int, "driving-stop-end-frame-mismatch")
    return {"stopping_observation_count": len(observations), "stopping_completed_seconds": first_stopped,
            "last_braking_speed_mps": last_braking_speed, "final_parking_speed_mps": last_speed,
            "minimum_stopping_body_lane_margin_m": min(margins)}


def _cleanup(value):
    _require(isinstance(value, dict) and value.get("status") == "PASS" and value.get("errors") == [], "driving-cleanup-not-verified")
    for key in ("sensor_stopped", "settings_restored", "weather_restored", "owned_actors_absent",
                "owned_actors_absent_in_snapshot", "owned_actors_absent_in_current_list"):
        _require(value.get(key) is True, "driving-cleanup-not-verified")
    _require(value.get("absence_verification") == "fresh-snapshot-and-current-actor-list"
             and _integer(value.get("absence_observed_frame")) > _integer(value.get("cleanup_reference_frame")), "driving-cleanup-stale-proof")
    _require(isinstance(value.get("owned_actor_ids"), list) and len(value["owned_actor_ids"]) >= 4
             and len(set(value["owned_actor_ids"])) == len(value["owned_actor_ids"]), "driving-owned-sensors-missing")
    for actor in value["owned_actor_ids"]:
        _integer(actor, 1)


def validate_drive_episode(episode_path: str | Path) -> dict:
    """Reopen all source bytes and replay geometry/controller state before PASS."""
    root = Path(episode_path)
    report = {"status": "FAIL", "gate_version": GATE_VERSION, "scope": SCOPE, "errors": []}
    try:
        metadata_bytes = (root / "episode.json").read_bytes()
        trace_bytes = (root / "steps.jsonl").read_bytes()
        metadata = strict_json_loads(metadata_bytes.decode("utf-8"))
        required = {"schema_version", "status", "scope", "episode_id", "geometry_case", "condition", "route_contract",
                    "collection_config", "controller_config", "controller_config_sha256", "vehicle_geometry", "geometry_evidence",
                    "camera_calibration", "bounding_box", "applied_manifest", "cleanup", "events", "event_callback_errors",
                    "drive_end_frame", "post_stop_frame", "step_count", "trace_sha256", "validation_config", "stop_control", "stop_observations"}
        _require(isinstance(metadata, dict) and required <= metadata.keys(), "driving-metadata-fields-missing")
        _require(type(metadata["schema_version"]) is int and metadata["schema_version"] == 1 and metadata["status"] == "COMPLETE"
                 and metadata["scope"] == SCOPE, "driving-episode-not-complete")
        validate_identifier(metadata["episode_id"])
        policy = read_strict_json(DEFAULT_VALIDATION_PATH)
        _require(canonical_sha256(metadata["validation_config"]) == canonical_sha256(policy), "driving-validation-policy-changed")
        _require(metadata["geometry_case"] in CASES, "invalid-driving-geometry-case")
        conditions = {item["name"]: item for item in policy["initial_conditions"]}
        condition = metadata["condition"]
        _require(isinstance(condition, dict) and condition.get("name") in conditions
                 and canonical_sha256(condition) == canonical_sha256(conditions[condition["name"]]), "invalid-driving-condition")
        cfg = collection_config_from_dict(metadata["collection_config"])
        controller_cfg = controller_config_from_dict(metadata["controller_config"])
        _require(metadata["controller_config_sha256"] == controller_cfg.sha256, "driving-controller-config-hash-mismatch")
        route = metadata["route_contract"]
        tracker = FixedRouteTracker(route, max_speed_mps=policy["maximum_speed_mps"], minimum_dt_seconds=.05)
        _require(route["town"] == cfg.map and route["spawn_point_index"] == cfg.vehicle.spawn_point_index
                 and route["town"] != "Town02", "driving-route-source-mismatch")
        _require(classify_geometry(route)["case"] == metadata["geometry_case"], "driving-geometry-case-mismatch")
        geometry = _geometry(metadata)
        controller = PurePursuitController(controller_cfg, geometry)
        _require(_same(metadata["camera_calibration"], _calibration(cfg, _mount_matrix(cfg))), "driving-camera-calibration-mismatch")
        _require(metadata["applied_manifest"].get("applied_verified") is True, "driving-initial-application-not-verified")
        _cleanup(metadata["cleanup"])
        _require(isinstance(metadata["events"], list) and metadata["events"] == [], "driving-collision-or-lane-event")
        _require(metadata["event_callback_errors"] == [], "driving-event-callback-failed")
        _require(metadata["trace_sha256"] == _sha(trace_bytes), "driving-trace-hash-mismatch")
        lines = trace_bytes.decode("utf-8").splitlines()
        _require(lines and all(line.strip() for line in lines), "driving-trace-empty-or-blank")
        rows = [strict_json_loads(line) for line in lines]
        _require(len(rows) == _integer(metadata["step_count"], 2), "driving-step-count-mismatch")
        seen_images, image_hashes, lateral_values, speeds, body_margins = set(), {}, [], [], []
        previous = None
        recovery_start = recovery_complete = None
        for index, row in enumerate(rows):
            _require(isinstance(row, dict) and set(row) == ROW_KEYS, "invalid-driving-step-fields")
            _require(type(row["schema_version"]) is int and row["schema_version"] == 1 and _integer(row["index"]) == index, "driving-step-index-mismatch")
            frame = _integer(row["frame"])
            _require(all(type(row[key]) is int and row[key] == frame for key in ("world_frame", "snapshot_frame", "sensor_frame", "command_source_frame")), "driving-frame-alignment-mismatch")
            timestamp = _finite(row["timestamp"])
            _require(timestamp >= 0 and all(abs(_finite(row[key]) - timestamp) <= 1e-6 for key in ("snapshot_timestamp", "sensor_timestamp")), "driving-timestamp-alignment-mismatch")
            _require(abs(_finite(row["delta_seconds"]) - .05) <= 1e-6 and _finite(row["tick_to_rgb_ms"]) >= 0, "driving-timestep-mismatch")
            if previous is not None:
                _require(frame == previous["frame"] + 1 and abs(timestamp - previous["timestamp"] - .05) <= 1e-6, "driving-frame-sequence-mismatch")
            terminal = index == len(rows) - 1
            _require(type(row["terminal"]) is bool and row["terminal"] == terminal, "driving-terminal-row-mismatch")
            target = None if terminal else frame + 1
            _require(row["command_target_frame"] is None if target is None else type(row["command_target_frame"]) is int and row["command_target_frame"] == target, "driving-command-target-frame-mismatch")
            vehicle = row["vehicle"]
            _require(isinstance(vehicle, dict) and set(vehicle) == {"transform", "velocity", "applied_control"}, "invalid-driving-vehicle-record")
            pose = _transform(vehicle["transform"])
            _require(isinstance(vehicle["velocity"], dict) and set(vehicle["velocity"]) == {"x", "y", "z"}, "invalid-driving-velocity")
            speed = math.sqrt(sum(_finite(value) ** 2 for value in vehicle["velocity"].values()))
            _require(abs(_finite(row["speed_mps"]) - speed) <= 1e-5 and speed <= policy["maximum_speed_mps"], "driving-speed-bound-or-readback-mismatch")
            applied = _control(vehicle["applied_control"])
            expected_applied = {"throttle": 0., "brake": 1., "steer": 0., "hand_brake": True, "reverse": False} if previous is None else {key: previous["command"][key] for key in CONTROL_KEYS}
            _require(_same(applied, expected_applied), "driving-command-application-mismatch")
            tracking = tracker.observe(pose, frame, timestamp)
            _require(_same(row["tracking"], tracking), "driving-tracking-replay-mismatch")
            validate_waypoint_label(row["tracking"]["label"], {"frame": frame, "vehicle": {"transform": pose}},
                                    {"route_id": route["route_id"], "route_contract": route})
            command = controller.step(tracking["label"]["waypoints_ego_m"], speed)
            _require(_same(row["command"], command), "driving-controller-replay-mismatch")
            camera = row["camera"]
            _require(isinstance(camera, dict) and set(camera) == {"transform"}, "invalid-driving-camera-record")
            camera_pose = _transform(camera["transform"])
            vm, cm, desired = pose["matrix"], camera_pose["matrix"], metadata["camera_calibration"]["requested_mount_matrix"]
            relative = [[sum(vm[k][r] * cm[k][c] for k in range(3)) if c < 3 else sum(vm[k][r] * (cm[k][3] - vm[k][3]) for k in range(3)) for c in range(4)] for r in range(3)] + [[0., 0., 0., 1.]]
            _require(_same(relative, desired, .002), "driving-camera-mount-mismatch")
            _require(abs(tracking["lateral_error_m"]) <= policy["maximum_lateral_error_m"], "driving-lateral-error-bound")
            margin = _body_margin(pose, tracking, metadata["bounding_box"], route)
            _require(abs(_finite(row["body_lane_margin_m"]) - margin) <= 1e-5 and margin >= policy["minimum_body_lane_margin_m"], "driving-body-lane-margin-failed")
            angles = row["native_wheel_angles_deg"]
            _require(isinstance(angles, dict) and set(angles) == {"front_left", "front_right"}
                     and all(abs(_finite(value)) <= 90. for value in angles.values()), "invalid-native-steering-angle")
            image_required = index % policy["image_stride"] == 0 or terminal
            image = row["image"]
            _require(image is not None if image_required else image is None, "driving-image-sampling-mismatch")
            if image is not None:
                _require(isinstance(image, dict) and set(image) == {"path", "sha256", "width", "height", "color_provenance"}, "invalid-driving-image-record")
                expected_path = f"images/{frame:08d}.png"
                _require(image["path"] == expected_path and image["width"] == cfg.camera.width and image["height"] == cfg.camera.height
                         and type(image["width"]) is int and type(image["height"]) is int, "driving-image-identity-mismatch")
                path = (root / expected_path).resolve()
                _require(path.is_relative_to((root / "images").resolve()), "driving-image-path-escaped")
                png = path.read_bytes()
                width, height, rgb = decode_rgb8_png(png, image["sha256"], (cfg.camera.width, cfg.camera.height))
                verify_rgb_color(rgb, image["color_provenance"], width, height, require_witness=True)
                seen_images.add(expected_path)
                image_hashes[expected_path] = _sha(png)
            within = abs(tracking["lateral_error_m"]) <= policy["recovery_lateral_tolerance_m"] and abs(tracking["heading_error_deg"]) <= policy["recovery_heading_tolerance_deg"]
            if within:
                if recovery_start is None:
                    recovery_start = timestamp
                if recovery_complete is None and timestamp - recovery_start >= policy["recovery_hold_seconds"] - 1e-6:
                    recovery_complete = timestamp - rows[0]["timestamp"]
            else:
                recovery_start = None
            lateral_values.append(abs(tracking["lateral_error_m"]))
            speeds.append(speed)
            body_margins.append(margin)
            previous = row
        initial, final = rows[0]["tracking"], rows[-1]["tracking"]
        _require(initial["reference_arc_m"] <= .5, "driving-start-reference-too-far")
        _require(abs(initial["lateral_error_m"] - condition["lateral_m"]) <= policy["initial_lateral_tolerance_m"]
                 and abs(initial["heading_error_deg"] - condition["yaw_deg"]) <= policy["initial_yaw_tolerance_deg"], "driving-initial-perturbation-not-applied")
        duration = rows[-1]["timestamp"] - rows[0]["timestamp"]
        _require(duration <= policy["maximum_simulation_seconds"] + 1e-6 and final["reference_arc_m"] >= policy["goal_reference_m"], "driving-goal-or-duration-failed")
        if condition["name"] != "center":
            _require(recovery_complete is not None and recovery_complete <= policy["recovery_deadline_seconds"] + 1e-6, "driving-recovery-not-held-within-deadline")
        stopping = _stopping(metadata, rows[-1], policy, route)
        _require(metadata["drive_end_frame"] == rows[-1]["frame"] and type(metadata["drive_end_frame"]) is int
                 and _integer(metadata["post_stop_frame"]) >= metadata["drive_end_frame"] + policy["settling_frames"], "driving-event-settling-window-missing")
        _require(metadata["cleanup"]["cleanup_reference_frame"] >= metadata["post_stop_frame"], "driving-cleanup-before-event-window")
        actual_images = {path.relative_to(root).as_posix() for path in (root / "images").rglob("*") if path.is_file()}
        _require(actual_images == seen_images, "driving-orphan-or-missing-image")
        _require((root / "episode.json").read_bytes() == metadata_bytes and (root / "steps.jsonl").read_bytes() == trace_bytes
                 and all(_sha((root / relative).read_bytes()) == digest for relative, digest in image_hashes.items()), "driving-source-changed-during-validation")
        config_except_spawn = cfg.to_dict()
        config_except_spawn["vehicle"].pop("spawn_point_index")
        report.update(status="PASS", source={"episode_id": metadata["episode_id"], "geometry_case": metadata["geometry_case"], "condition": condition,
                      "route_id": route["route_id"], "route_sha256": route_sha256(route), "town": route["town"],
                      "opendrive_sha256": route["opendrive_sha256"], "metadata_sha256": _sha(metadata_bytes), "trace_sha256": _sha(trace_bytes),
                      "controller_config_sha256": controller_cfg.sha256, "configuration_except_spawn_sha256": canonical_sha256(config_except_spawn),
                      "vehicle_geometry_sha256": canonical_sha256(geometry.to_dict()), "vehicle_geometry": geometry.to_dict(),
                      "calibration_sha256": canonical_sha256(metadata["camera_calibration"])},
                      metrics={"step_count": len(rows), "image_count": len(seen_images), "first_frame": rows[0]["frame"], "last_frame": rows[-1]["frame"],
                               "simulation_seconds": duration, "final_reference_m": final["reference_arc_m"], "progress_m": final["progress_m"],
                               "initial_lateral_error_m": initial["lateral_error_m"], "initial_heading_error_deg": initial["heading_error_deg"],
                               "final_lateral_error_m": final["lateral_error_m"], "final_heading_error_deg": final["heading_error_deg"],
                               "maximum_abs_lateral_error_m": max(lateral_values), "mean_abs_lateral_error_m": sum(lateral_values) / len(lateral_values),
                               "maximum_speed_mps": max(speeds), "minimum_body_lane_margin_m": min(min(body_margins), stopping["minimum_stopping_body_lane_margin_m"]),
                               "minimum_driving_body_lane_margin_m": min(body_margins),
                               "recovery_completed_seconds": recovery_complete, "collision_count": 0, "lane_invasion_count": 0, **stopping},
                      checks={"same_frame_original_route_labels": True, "controller_replay": True, "command_application": True,
                              "sampled_rgb_color_verified": True, "body_lane_margin": True, "owned_actors_cleaned_and_world_restored": True,
                              "events_observed_through_post_stop": True, "sources_unchanged": True}, final_test_used=False, model_training_verified=False)
    except BaseException as error:
        report["status"] = "FAIL"
        report["errors"].append({"code": getattr(error, "code", str(error) if isinstance(error, DrivingValidationError) else "driving-source-validation-failed"),
                                 "exception_type": type(error).__name__})
    return report


def validate_driving_manifest(manifest_path: str | Path) -> dict:
    """Require all 3 geometries x 5 prescribed conditions, with common sources."""
    path = Path(manifest_path)
    report = {"status": "FAIL", "gate_version": GATE_VERSION, "scope": SCOPE, "errors": [], "episodes": []}
    try:
        manifest = read_strict_json(path)
        _require(isinstance(manifest, dict) and set(manifest) == {"schema_version", "episodes"}
                 and type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
                 and isinstance(manifest["episodes"], list), "invalid-driving-manifest")
        policy = read_strict_json(DEFAULT_VALIDATION_PATH)
        required = {(case, item["name"]) for case in CASES for item in policy["initial_conditions"]}
        found, roots, episode_ids = set(), set(), set()
        same_case_routes = {}
        shared_fields = ("town", "opendrive_sha256", "controller_config_sha256", "configuration_except_spawn_sha256", "calibration_sha256")
        shared = None
        shared_geometry = None
        for item in manifest["episodes"]:
            _require(isinstance(item, dict) and set(item) == {"case", "condition", "episode"}
                     and all(isinstance(value, str) and value.strip() for value in item.values()), "invalid-driving-manifest-episode")
            identity = (item["case"], item["condition"])
            _require(identity in required and identity not in found, "duplicate-or-unknown-driving-case")
            root = (path.parent / item["episode"]).resolve()
            _require(root not in roots, "duplicate-driving-episode-path")
            result = validate_drive_episode(root)
            report["episodes"].append({"case": item["case"], "condition": item["condition"], "report": result})
            _require(result["status"] == "PASS", "driving-episode-gate-failed")
            source = result["source"]
            _require((source["geometry_case"], source["condition"]["name"]) == identity
                     and source["episode_id"] not in episode_ids, "driving-episode-identity-mismatch")
            common = tuple(source[key] for key in shared_fields)
            _require(shared is None or shared == common, "driving-common-source-mismatch")
            shared = common
            geometry = source["vehicle_geometry"]
            if shared_geometry is not None:
                _require(set(shared_geometry) == set(geometry), "driving-vehicle-geometry-source-mismatch")
                for key in geometry:
                    tolerance = .001 if key.endswith("_m") else 1e-5
                    _require(_same(geometry[key], shared_geometry[key], tolerance), "driving-vehicle-geometry-source-mismatch")
            shared_geometry = geometry if shared_geometry is None else shared_geometry
            case_route = same_case_routes.setdefault(item["case"], source["route_sha256"])
            _require(case_route == source["route_sha256"], "driving-recovery-target-route-changed")
            found.add(identity)
            roots.add(root)
            episode_ids.add(source["episode_id"])
        _require(found == required, "driving-cases-incomplete")
        report.update(status="PASS", episode_count=len(found), step_count=sum(item["report"]["metrics"]["step_count"] for item in report["episodes"]),
                      image_count=sum(item["report"]["metrics"]["image_count"] for item in report["episodes"]),
                      geometry_cases=list(CASES), condition_names=[item["name"] for item in policy["initial_conditions"]],
                      final_test_used=False, model_training_verified=False)
    except BaseException as error:
        report["status"] = "FAIL"
        report["errors"].append({"code": getattr(error, "code", "driving-manifest-validation-failed"), "exception_type": type(error).__name__})
    return report
