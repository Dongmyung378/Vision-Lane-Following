"""Classify verified development failures without feeding labels to prediction."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import statistics

from .baseline_validation import _safe_file
from .contracts import strict_json_loads
from .controller import controller_config_from_dict, VehicleGeometry, PurePursuitController
from .driving_validation import _same
from .driving_route import FixedRouteTracker


def summarize_errors(values):
    if not values:
        return {"count": 0, "mean": None, "maximum": None}
    return {"count": len(values), "mean": statistics.fmean(values), "maximum": max(values)}


def analyze_failure_episode(directory, *, integrity_verified=False):
    if integrity_verified is not True:
        raise ValueError("failure-analysis-independent-integrity-proof-required")
    root = Path(directory)
    metadata_bytes = _safe_file(root, "episode.json").read_bytes()
    metadata = strict_json_loads(metadata_bytes.decode())
    trace_bytes = _safe_file(root, "steps.jsonl").read_bytes()
    if hashlib.sha256(trace_bytes).hexdigest() != metadata["trace_sha256"]:
        raise ValueError("failure-analysis-trace-changed")
    rows = [strict_json_loads(line) for line in trace_bytes.decode().splitlines()]
    if not rows or len(rows) != metadata["step_count"]:
        raise ValueError("failure-analysis-frame-count-invalid")
    tracker = FixedRouteTracker(metadata["route_contract"], max_speed_mps=12., minimum_dt_seconds=.05)
    geometry = VehicleGeometry(**metadata["vehicle_geometry"])
    cfg = controller_config_from_dict(metadata["controller_config"])
    observed_controller, labelled_controller = PurePursuitController(cfg, geometry), PurePursuitController(cfg, geometry)
    near, far, steering_difference, evidence_before_issue = [], [], [], []
    statuses, first_issue, label_mismatches = {}, None, []
    for index, row in enumerate(rows):
        reconstructed = tracker.observe(row["vehicle"]["transform"], row["frame"], row["timestamp"])
        if not _same(reconstructed, row["tracking"], 1e-8):
            label_mismatches.append(row["frame"])
        driving_events = [e for e in row.get("evaluation_events", []) if rows[0]["frame"] <= e["frame"] <= row["frame"]]
        if first_issue is None and (row["body_lane_margin_m"] < 0 or driving_events):
            first_issue = index
        prediction = row["perception"]
        status = prediction["status"]
        statuses[status] = statuses.get(status, 0) + 1
        if status != "PASS":
            continue
        target = reconstructed["label"]["waypoints_ego_m"]
        points = prediction["waypoints_ego_m"]
        errors = [math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(points, target)]
        if len(errors) != 5 or any(not math.isfinite(e) for e in errors):
            raise ValueError("failure-analysis-waypoint-error-invalid")
        near.extend(errors[:2]); far.extend(errors[2:])
        predicted_command = observed_controller.step(points, row["speed_mps"])
        labelled_command = labelled_controller.step(target, row["speed_mps"])
        if not _same(predicted_command, row["command"], 1e-8):
            raise ValueError("failure-analysis-controller-replay-mismatch")
        difference = abs(predicted_command["steer"] - labelled_command["steer"])
        steering_difference.append(difference)
        evidence_before_issue.append({"index": index, "near_mean_m": statistics.fmean(errors[:2]),
            "first_point_lateral_error_m": points[0][1] - target[0][1], "counterfactual_steering_difference": difference,
            "opposite_steering": predicted_command["steer"] * labelled_command["steer"] < 0
                and abs(predicted_command["steer"]) > .02 and abs(labelled_command["steer"]) > .02})
    if label_mismatches:
        raise ValueError("failure-analysis-label-or-coordinate-mismatch")
    limit = first_issue if first_issue is not None else len(rows)
    before = [e for e in evidence_before_issue if max(0, limit - 20) <= e["index"] < limit]
    window_mean = statistics.fmean(e["near_mean_m"] for e in before) if before else None
    metrics = metadata["evaluation_metrics"]
    lane_issue = metrics["lane_invasion_count"] > 0 or min(r["body_lane_margin_m"] for r in rows) < 0
    completed_with_issue = metrics["termination_reason"] == "route_completed" and lane_issue
    if statuses.get("FAIL"):
        classification = "perception-invalid-path"
    elif lane_issue and (window_mean or 0) > .3:
        classification = "perception-near-path-error-before-lane-issue"
    elif lane_issue:
        classification = "control-or-small-perception-error-needs-paired-reference"
    elif metrics["termination_reason"] != "route_completed":
        classification = "driving-failure-needs-paired-reference"
    else:
        classification = "clean-completion"
    return {"episode_id": metadata["episode_id"], "method": metadata["method"], "case": metadata["geometry_case"],
        "condition": metadata["condition"], "initial_reference_m": metadata.get("requested_initial_reference_m", 0.),
        "classification": classification, "source_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "source_trace_sha256": hashlib.sha256(trace_bytes).hexdigest(), "frame_count": len(rows),
        "labels_reconstructed": True, "controller_replayed": True, "execution_integrity_verified": True,
        "perception_status_counts": statuses, "near_error_m": summarize_errors(near), "far_error_m": summarize_errors(far),
        "same-state_counterfactual_steering_difference": summarize_errors(steering_difference),
        "counterfactual_is_new_driving": False, "first_lane_issue_index": first_issue,
        "last_second_before_issue": {"count": len(before), "near_mean_m": window_mean,
            "opposite_steering_frames": sum(e["opposite_steering"] for e in before)},
        "completed_with_lane_issue": completed_with_issue, "termination_reason": metrics["termination_reason"],
        "environment": {"callback_errors": len(metadata["event_callback_errors"]), "operation_status": metadata["status"]}}
