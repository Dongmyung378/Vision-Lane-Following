"""Connect RGB-only lane prediction to the shared native vehicle controller."""
from __future__ import annotations

import copy
import math
import time
import json
from dataclasses import replace

from .driving import DrivingError, load_validation_config, run_drive_trial
from .controller import _rear_points
from scripts.carla_rgb_probe import rgb_from_bgra

SCOPE = "opencv-camera-only-development-evaluation"


class BaselineTrialPolicy:
    """Keep RGB prediction separate from privileged route evaluation."""

    scope = SCOPE
    method = "B0"

    def __init__(self, baseline_config, evaluation_config, route, condition, *, predictor=None,
                 goal_reference_m=19.0):
        from .baseline import OpenCVLanePredictor
        self.baseline_config = baseline_config
        self.evaluation_config = evaluation_config
        self.predictor = predictor or OpenCVLanePredictor(baseline_config)
        self.route = copy.deepcopy(route)
        self.condition = copy.deepcopy(condition)
        self.goal_reference_m = goal_reference_m
        self.tracker = None
        self.start_reference_m = None

    def configure_metadata(self, metadata):
        metadata.update(method=self.method, route_id=self.route["route_id"],
                        prediction_inputs="rgb-and-fixed-nominal-calibration",
                        baseline_config=self.baseline_config.to_dict(), baseline_config_sha256=self.baseline_config.sha256,
                        evaluation_config=self.evaluation_config.to_dict(), evaluation_config_sha256=self.evaluation_config.sha256,
                        evaluation_goal_reference_m=self.goal_reference_m,
                        perception_uses_route=False, perception_uses_vehicle_pose=False,
                        latency_scope="CPU BGRA-to-RGB, OpenCV prediction and shared controller; excludes simulator wait and disk writes")

    def command(self, image, speed_mps, controller):
        # This method receives neither pose, calibration readback nor GT labels.
        started = time.perf_counter()
        rgb, _ = rgb_from_bgra(bytes(image.raw_data), image.width, image.height)
        converted = time.perf_counter()
        try:
            prediction = self.predictor.predict(rgb, width=image.width, height=image.height)
            if not isinstance(prediction, dict) or prediction.get("status") not in ("PASS", "FAIL"):
                raise ValueError("invalid-prediction-record")
            json.dumps(prediction, allow_nan=False)
            if prediction["status"] == "PASS":
                points = prediction.get("waypoints_ego_m")
                _rear_points(points, controller.geometry.rear_axle_x_m, controller.geometry.rear_axle_y_m)
                if any(point[0] <= 0 for point in points):
                    raise ValueError("baseline-actor-waypoint-not-forward")
        except Exception as error:
            # Keep this frame and brake before cleanup. The disk replay gate
            # still rejects an exception result that cannot be reproduced.
            prediction = {"status": "FAIL", "waypoints_ego_m": None, "error": "baseline-predictor-exception",
                          "diagnostics": {"exception_type": type(error).__name__, "failure_origin": "predictor-exception"},
                          "timing_ms": (time.perf_counter() - converted) * 1000}
        predicted = time.perf_counter()
        command = None
        if prediction.get("status") == "PASS":
            command = controller.step(prediction["waypoints_ego_m"], speed_mps, dt_seconds=.05)
        finished = time.perf_counter()
        latency = {"preprocess_ms": (converted - started) * 1000,
                   "inference_ms": (predicted - converted) * 1000,
                   "controller_ms": (finished - predicted) * 1000,
                   "pipeline_ms": (finished - started) * 1000}
        return command, prediction, latency

    def observe(self, row, events, callback_errors):
        from .evaluation import EvaluationTracker
        if self.tracker is None:
            self.start_reference_m = row["tracking"]["reference_arc_m"]
            length = self.goal_reference_m - self.start_reference_m
            if not math.isfinite(length) or length <= 0:
                raise DrivingError("baseline-evaluation-length-invalid")
            self.tracker = EvaluationTracker(self.evaluation_config, route_length_m=length,
                                             start_reference_m=self.start_reference_m, condition=self.condition)
        row["technical_failure"] = bool(callback_errors) or row["perception"]["status"] != "PASS" or row["speed_mps"] > 12.
        row["remaining_horizon_m"] = self.route["route_length_m"] - row["tracking"]["reference_arc_m"]
        return self.tracker.update(
            row, events=events,
            technical_failure=row["technical_failure"], remaining_horizon_m=row["remaining_horizon_m"])

    def finalize(self, metadata, rows):
        from .evaluation import evaluate_episode
        metadata["evaluation_start_reference_m"] = self.start_reference_m
        metadata["evaluation_route_length_m"] = (None if self.start_reference_m is None else
                                                  self.goal_reference_m - self.start_reference_m)
        reason = rows[-1].get("termination_reason") if rows else "technical_failure"
        if metadata.get("status") != "COMPLETE" or metadata.get("event_callback_errors"):
            reason = "technical_failure"
        metadata["termination_reason"] = reason
        metadata["evaluation_metrics"] = evaluate_episode(metadata, rows, metadata.get("events", []),
                                                           config=self.evaluation_config)
        metadata["termination_reason"] = metadata["evaluation_metrics"]["termination_reason"]
        first_frame = rows[0]["frame"] if rows else 0
        last_frame = rows[-1]["frame"] if rows else -1
        metadata["non_driving_events"] = [event for event in metadata.get("events", [])
                                          if not first_frame <= event["frame"] <= last_frame]


class PrivilegedEvaluationPolicy(BaselineTrialPolicy):
    """Reference labels use the identical native controller/evaluation policy."""

    method = "privileged_reference"

    def configure_metadata(self, metadata):
        super().configure_metadata(metadata)
        metadata.update(prediction_inputs="privileged-original-route-and-same-frame-pose",
                        perception_uses_route=True, perception_uses_vehicle_pose=True,
                        latency_scope="shared controller only; privileged route-label generation is excluded")

    def privileged_command(self, label, speed_mps, controller):
        started = time.perf_counter()
        command = controller.step(label["waypoints_ego_m"], speed_mps, dt_seconds=.05)
        elapsed = (time.perf_counter() - started) * 1000
        return command, {"status": "PASS", "waypoints_ego_m": copy.deepcopy(label["waypoints_ego_m"]),
                         "diagnostics": {"method": self.method, "input": "privileged-original-route"}, "timing_ms": 0.0}, {
                             "preprocess_ms": 0.0, "inference_ms": 0.0, "controller_ms": elapsed, "pipeline_ms": elapsed}


def run_baseline_trial(route, case, condition, output_dir, collection_config, *, carla_module,
                       baseline_config=None, evaluation_config=None, controller_config=None, predictor=None, method="B0"):
    from .baseline import load_baseline_config
    from .evaluation import load_evaluation_config
    if method not in ("B0", "privileged_reference"):
        raise DrivingError("baseline-method-invalid")
    policy_type = BaselineTrialPolicy if method == "B0" else PrivilegedEvaluationPolicy
    policy = policy_type(baseline_config or load_baseline_config(),
                         evaluation_config or replace(load_evaluation_config(), timeout_seconds=10.0),
                         route, condition, predictor=predictor)
    return run_drive_trial(route, case, condition, output_dir, collection_config, carla_module=carla_module,
                           validation_config=load_validation_config(), controller_config=controller_config,
                           trial_policy=policy)
