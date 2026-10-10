"""Offline RGB, control, evaluation, and cleanup replay of camera trials.

Driving failures remain evaluated episodes. A PASS certifies persisted source
integrity and reproducibility, not successful driving or held-out performance.
"""
from __future__ import annotations

import copy
import csv
import io
import math
from pathlib import Path, PurePosixPath

from .baseline import OpenCVLanePredictor, baseline_config_from_dict
from .baseline_driving import SCOPE
from .capture import _calibration, _mount_matrix
from .color import verify_rgb_color
from .config import collection_config_from_dict
from .contracts import canonical_sha256, route_sha256, strict_json_loads, validate_identifier, validate_waypoint_label
from .controller import PurePursuitController, controller_config_from_dict
from .driving_route import FixedRouteTracker, _transform
from .driving_validation import (CONTROL_KEYS, ROW_KEYS, _body_margin, _cleanup, _control,
                                 _finite, _geometry, _integer, _require, _same, _sha)
from .evaluation import CSV_FIELDS, EvaluationTracker, aggregate_metrics, deduplicate_events, evaluate_episode, evaluation_config_from_dict
from .geometry_survey import CASES, classify_geometry
from .projection import decode_rgb8_png

GATE_VERSION = "vlf-camera-baseline-replay-v1"
METHODS = ("B0", "privileged_reference")
BASELINE_ROW_KEYS = ROW_KEYS | {"perception", "latency_ms", "evaluation", "evaluation_events",
                               "evaluation_callback_errors", "technical_failure", "remaining_horizon_m", "termination_reason"}


def _safe_file(root, relative):
    value = PurePosixPath(relative)
    _require(isinstance(relative, str) and not value.is_absolute() and ".." not in value.parts,
             "baseline-source-path-escaped")
    path = root.joinpath(*value.parts)
    _require(not root.is_symlink() and all(not parent.is_symlink() for parent in (path, *path.parents)
                                         if parent == root or parent.is_relative_to(root)), "baseline-source-symlink")
    _require(path.resolve().is_relative_to(root.resolve()) and path.is_file(), "baseline-source-file-missing-or-escaped")
    return path


def _native_angles(value):
    _require(isinstance(value, dict) and set(value) == {"front_left", "front_right"}
             and all(abs(_finite(number)) <= 90 for number in value.values()), "baseline-native-steering-angle-invalid")


def _stopping(metadata, last, route):
    policy = metadata["validation_config"]
    brake = {"throttle": 0., "brake": 1., "steer": last["vehicle"]["applied_control"]["steer"],
             "hand_brake": False, "reverse": False}
    _require(_same(_control(metadata["stop_control"]), brake), "baseline-stop-control-mismatch")
    rows = metadata["stop_observations"]
    _require(isinstance(rows, list) and len(rows) >= policy["settling_frames"] + 2,
             "baseline-stop-observations-missing")
    braking = len(rows) - 2
    previous_frame, previous_time = last["frame"], last["timestamp"]
    first_stopped, margins = None, []
    for index, row in enumerate(rows):
        _require(isinstance(row, dict) and set(row) == {"frame", "timestamp", "vehicle", "speed_mps",
                 "native_wheel_angles_deg", "body_lane_margin_m"}, "baseline-stop-observation-invalid")
        frame, timestamp = _integer(row["frame"]), _finite(row["timestamp"])
        _require(frame == previous_frame + 1 and abs(timestamp - previous_time - .05) <= 1e-6,
                 "baseline-stop-frame-sequence-mismatch")
        vehicle = row["vehicle"]
        _require(isinstance(vehicle, dict) and set(vehicle) == {"transform", "velocity", "applied_control"},
                 "baseline-stop-vehicle-invalid")
        pose = _transform(vehicle["transform"])
        _require(isinstance(vehicle["velocity"], dict) and set(vehicle["velocity"]) == {"x", "y", "z"},
                 "baseline-stop-velocity-invalid")
        speed = math.sqrt(sum(_finite(number) ** 2 for number in vehicle["velocity"].values()))
        _require(abs(_finite(row["speed_mps"]) - speed) <= 1e-5, "baseline-stop-speed-mismatch")
        expected = brake if index < braking else dict(brake, hand_brake=True)
        _require(_same(_control(vehicle["applied_control"]), expected), "baseline-stop-control-readback-mismatch")
        _native_angles(row["native_wheel_angles_deg"])
        margin = _body_margin(pose, {}, metadata["bounding_box"], route)
        _require(abs(_finite(row["body_lane_margin_m"]) - margin) <= 1e-5, "baseline-stop-body-margin-mismatch")
        elapsed = timestamp - last["timestamp"]
        if index < braking:
            _require(elapsed <= policy["maximum_stop_seconds"] + 1e-6, "baseline-stop-deadline-exceeded")
            if first_stopped is None and speed <= policy["stop_speed_tolerance_mps"]:
                first_stopped = elapsed
        else:
            _require(speed <= policy["stop_speed_tolerance_mps"], "baseline-parking-not-stationary")
        if index == braking - 1:
            _require(speed <= policy["stop_speed_tolerance_mps"], "baseline-parking-before-stop")
        margins.append(margin)
        previous_frame, previous_time = frame, timestamp
    _require(first_stopped is not None and metadata["post_stop_frame"] == previous_frame,
             "baseline-stop-end-or-speed-mismatch")
    _require(type(metadata["post_stop_frame"]) is int and metadata["cleanup"]["cleanup_reference_frame"] >= previous_frame,
             "baseline-cleanup-before-stop")
    return {"stopping_observation_count": len(rows), "stopping_completed_seconds": first_stopped,
            "minimum_stopping_body_lane_margin_m": min(margins)}


def _validate_camera_episode(episode_path: str | Path, *, allowed_conditions=None,
                             scope=SCOPE, methods=METHODS,
                             prediction_inputs="rgb-and-fixed-nominal-calibration",
                             predictor=None, expected_initial_reference_m=0.0) -> dict:
    """Read unchanged source bytes and reconstruct RGB predictions independently."""
    root = Path(episode_path)
    report = {"status": "FAIL", "scope": scope, "gate_version": GATE_VERSION, "errors": []}
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
        _require(metadata["perception_uses_route"] is is_reference and metadata["perception_uses_vehicle_pose"] is is_reference,
                 "baseline-prediction-input-contract-mismatch")
        expected_inputs = "privileged-original-route-and-same-frame-pose" if is_reference else prediction_inputs
        _require(metadata["prediction_inputs"] == expected_inputs, "baseline-prediction-input-contract-mismatch")
        from .driving import load_validation_config
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
        cfg = collection_config_from_dict(metadata["collection_config"])
        camera = cfg.camera
        _require((camera.width, camera.height) == (baseline_config.image_width, baseline_config.image_height)
                 and camera.fov == baseline_config.fov_deg
                 and camera.mount.location.x == baseline_config.camera_x_m
                 and camera.mount.location.y == baseline_config.camera_y_m
                 and camera.mount.location.z == baseline_config.camera_z_m
                 and camera.mount.rotation.pitch == baseline_config.camera_pitch_deg
                 and camera.mount.rotation.yaw == 0. and camera.mount.rotation.roll == 0.,
                 "baseline-unsupported-actual-camera-calibration")
        route = metadata["route_contract"]
        tracker = FixedRouteTracker(route, max_speed_mps=policy["maximum_speed_mps"], minimum_dt_seconds=.05)
        _require(route["town"] == cfg.map and route["spawn_point_index"] == cfg.vehicle.spawn_point_index
                 and route["town"] != "Town02" and classify_geometry(route)["case"] == metadata["geometry_case"],
                 "baseline-original-route-source-mismatch")
        geometry = _geometry(metadata)
        controller = PurePursuitController(controller_config, geometry)
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
    except BaseException as error:
        report["errors"].append({"code": getattr(error, "code", str(error) if isinstance(error, ValueError) else "baseline-source-validation-failed"),
                                 "exception_type": type(error).__name__})
    return report


def validate_baseline_episode(episode_path: str | Path, *, allowed_conditions=None) -> dict:
    """Preserve the strict original B0/reference episode validation contract."""
    return _validate_camera_episode(episode_path, allowed_conditions=allowed_conditions)


def validate_baseline_manifest(manifest_path: str | Path) -> dict:
    """Require both methods on all three shared original routes and source contracts."""
    path = Path(manifest_path)
    report = {"status": "FAIL", "scope": SCOPE, "gate_version": GATE_VERSION, "errors": [], "episodes": []}
    try:
        _require(not path.is_symlink(), "baseline-manifest-symlink")
        manifest_bytes = path.read_bytes()
        manifest = strict_json_loads(manifest_bytes.decode("utf-8"))
        _require(isinstance(manifest, dict) and set(manifest) == {"schema_version", "scope", "episodes"}
                 and type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
                 and manifest["scope"] == SCOPE and isinstance(manifest["episodes"], list), "baseline-manifest-invalid")
        required = {(method, case, "center") for method in METHODS for case in CASES}
        found, roots, shared, shared_geometry, routes, episode_metrics, source_proofs = set(), set(), None, None, {}, [], []
        common_fields = ("town", "opendrive_sha256", "controller_config_sha256", "baseline_config_sha256",
                         "evaluation_config_sha256", "calibration_sha256", "configuration_except_spawn_sha256")
        for item in manifest["episodes"]:
            _require(isinstance(item, dict) and set(item) == {"method", "case", "condition", "episode"}
                     and all(isinstance(value, str) and value for value in item.values()), "baseline-manifest-episode-invalid")
            identity = (item["method"], item["case"], item["condition"])
            _require(identity in required and identity not in found, "baseline-duplicate-or-unknown-case")
            expected = f"episodes/{item['method']}/{item['case']}__center"
            _require(item["episode"] == expected, "baseline-manifest-episode-path-invalid")
            source_root = path.parent.joinpath(*PurePosixPath(expected).parts)
            _require(all(not node.is_symlink() for node in (source_root, *source_root.parents) if
                         node == path.parent or node.is_relative_to(path.parent)), "baseline-source-symlink")
            root = source_root.resolve()
            _require(root.is_relative_to(path.parent.resolve()) and root not in roots, "baseline-episode-path-escaped-or-duplicate")
            checked = validate_baseline_episode(source_root)
            report["episodes"].append({"method": item["method"], "case": item["case"], "condition": item["condition"], "report": checked})
            _require(checked["status"] == "PASS", "baseline-episode-gate-failed")
            source = checked["source"]
            _require((source["method"], source["geometry_case"], source["condition"]["name"]) == identity,
                     "baseline-episode-identity-mismatch")
            common = tuple(source[field] for field in common_fields)
            _require(shared is None or common == shared, "baseline-common-source-mismatch")
            shared = common
            if shared_geometry is not None:
                _require(set(source["vehicle_geometry"]) == set(shared_geometry)
                         and all(_same(source["vehicle_geometry"][key], shared_geometry[key], .001 if key.endswith("_m") else 1e-5)
                                 for key in shared_geometry), "baseline-vehicle-geometry-source-mismatch")
            else:
                shared_geometry = source["vehicle_geometry"]
            reference = routes.setdefault(item["case"], source["route_sha256"])
            _require(reference == source["route_sha256"], "baseline-comparison-route-changed")
            found.add(identity)
            roots.add(root)
            episode_metrics.append(checked["evaluation_metrics"])
            source_proofs.append((source_root, source))
        _require(found == required, "baseline-cases-incomplete")
        aggregate = aggregate_metrics(episode_metrics)
        optional_hashes = {}
        csv_path = path.parent / "metrics.csv"
        if csv_path.exists() or csv_path.is_symlink():
            csv_bytes = _safe_file(path.parent, "metrics.csv").read_bytes()
            reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8"), newline=""))
            csv_rows = list(reader)
            _require(tuple(reader.fieldnames or ()) == CSV_FIELDS and len(csv_rows) == len(episode_metrics),
                     "baseline-metric-csv-schema-or-count-mismatch")
            for actual, expected in zip(csv_rows, episode_metrics):
                _require(set(actual) == set(CSV_FIELDS), "baseline-metric-csv-schema-or-count-mismatch")
                for field in CSV_FIELDS:
                    value, encoded = expected[field], actual[field]
                    if value is None:
                        matches = encoded == "N/A"
                    elif isinstance(value, bool):
                        matches = encoded == str(value)
                    elif isinstance(value, str):
                        matches = encoded == value
                    else:
                        matches = _same(strict_json_loads(encoded), value)
                    _require(matches, "baseline-metric-csv-replay-mismatch")
            optional_hashes["metrics.csv"] = _sha(csv_bytes)
        json_path = path.parent / "metrics.json"
        if json_path.exists() or json_path.is_symlink():
            json_bytes = _safe_file(path.parent, "metrics.json").read_bytes()
            expected = {"schema_version": 1, "episodes": episode_metrics, "aggregate": aggregate}
            _require(_same(strict_json_loads(json_bytes.decode("utf-8")), expected), "baseline-metric-json-replay-mismatch")
            optional_hashes["metrics.json"] = _sha(json_bytes)
        _require(path.read_bytes() == manifest_bytes and all(
            _sha(_safe_file(root, "episode.json").read_bytes()) == source["metadata_sha256"]
            and _sha(_safe_file(root, "steps.jsonl").read_bytes()) == source["trace_sha256"]
            and all(_sha(_safe_file(root, relative).read_bytes()) == digest
                    for relative, digest in source["image_sha256"].items())
            for root, source in source_proofs) and all(
                _sha(_safe_file(path.parent, relative).read_bytes()) == digest for relative, digest in optional_hashes.items()),
                 "baseline-manifest-source-changed-during-validation")
        report.update(status="PASS", episode_count=len(found),
                      step_count=sum(item["report"]["metrics"]["step_count"] for item in report["episodes"]),
                      image_count=sum(item["report"]["metrics"]["image_count"] for item in report["episodes"]),
                      episode_metrics=episode_metrics, aggregate=aggregate,
                      final_test_used=False, held_out_performance_certified=False)
    except BaseException as error:
        report["errors"].append({"code": getattr(error, "code", str(error) if isinstance(error, ValueError) else "baseline-manifest-validation-failed"),
                                 "exception_type": type(error).__name__})
    return report
