"""Paired development remediation with an explicitly privileged control arm."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from .baseline_driving import BaselineTrialPolicy, PrivilegedEvaluationPolicy
from .baseline_validation import _safe_file, _validate_camera_episode
from .contracts import strict_json_loads
from .driving import load_validation_config, run_drive_trial
from .driving_validation import _require, _same
from .projection import decode_rgb8_png
from .refinement_driving import CONDITIONS, INITIAL_REFERENCES_M, NATIVE_SOURCES
from .refinement_validation import _RecordedReplay

SCOPE = "rgb-waypoint-failure-remediation-development-driving"
SOURCES = (*NATIVE_SOURCES, "carla_tools/near_correction.py", "carla_tools/failure_driving.py",
           "carla_tools/failure_analysis.py", "carla_tools/driving_quality.py", "scripts/run_failure_pipeline.py")
REFERENCE_IDENTITY = {"variant": "privileged_reference", "device": "cpu", "input": "original-route-and-same-frame-pose"}


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}


class FailureTrialPolicy(BaselineTrialPolicy):
    scope = SCOPE

    def __init__(self, *args, method, initial_reference_m=0., **kwargs):
        if type(initial_reference_m) not in (int, float) or initial_reference_m not in INITIAL_REFERENCES_M:
            raise ValueError("failure-protocol-start-invalid")
        super().__init__(*args, **kwargs)
        self.method, self.initial_reference_m = method, float(initial_reference_m)

    def configure_metadata(self, metadata):
        super().configure_metadata(metadata)
        metadata.update(prediction_inputs="rgb-and-fixed-nominal-calibration" if self.method.endswith("_near") else "rgb-only",
            native_model=self.predictor.identity, requested_initial_reference_m=self.initial_reference_m,
            native_source_sha256=source_hashes(), comparison_scope="same-checkpoint-controller-speed-and-termination")


class FailureReferencePolicy(PrivilegedEvaluationPolicy):
    scope = SCOPE

    def __init__(self, *args, initial_reference_m=0., **kwargs):
        if type(initial_reference_m) not in (int, float) or initial_reference_m not in INITIAL_REFERENCES_M:
            raise ValueError("failure-protocol-start-invalid")
        super().__init__(*args, **kwargs)
        self.initial_reference_m = float(initial_reference_m)

    def configure_metadata(self, metadata):
        super().configure_metadata(metadata)
        metadata.update(native_model=copy.deepcopy(REFERENCE_IDENTITY), requested_initial_reference_m=self.initial_reference_m,
            native_source_sha256=source_hashes(), comparison_scope="privileged-control-diagnostic-not-camera-only-method")


def run_failure_trial(route, case, condition, output, collection_config, *, predictor, carla_module,
                      baseline_config, evaluation_config, controller_config, initial_reference_m=0.):
    if predictor is None:
        policy = FailureReferencePolicy(baseline_config, evaluation_config, route, condition,
            initial_reference_m=initial_reference_m)
    else:
        policy = FailureTrialPolicy(baseline_config, evaluation_config, route, condition,
            predictor=predictor, method=predictor.identity["variant"], initial_reference_m=initial_reference_m)
    return run_drive_trial(route, case, condition, output, collection_config, carla_module=carla_module,
        validation_config=load_validation_config(), controller_config=controller_config, trial_policy=policy)


def validate_failure_episode(directory, predictor, *, expected_initial_reference_m):
    root = Path(directory)
    report = {"status": "FAIL", "scope": SCOPE, "errors": []}
    try:
        _require(type(expected_initial_reference_m) in (int, float) and expected_initial_reference_m in INITIAL_REFERENCES_M,
                 "failure-protocol-start-invalid")
        metadata_bytes, trace_bytes = _safe_file(root, "episode.json").read_bytes(), _safe_file(root, "steps.jsonl").read_bytes()
        metadata = strict_json_loads(metadata_bytes.decode())
        identity = copy.deepcopy(REFERENCE_IDENTITY if predictor is None else predictor.identity)
        _require(metadata.get("requested_initial_reference_m") == expected_initial_reference_m, "failure-start-mismatch")
        observed_identity = copy.deepcopy(metadata.get("native_model", {}))
        _require(observed_identity.pop("device", None) in ("cpu", "cuda"), "failure-runtime-missing")
        actual_identity = copy.deepcopy(identity)
        actual_identity.pop("device", None)
        _require(_same(observed_identity, actual_identity, 1e-12), "failure-model-identity-mismatch")
        _require(metadata.get("native_source_sha256") == source_hashes(), "failure-runtime-source-mismatch")
        _require(metadata.get("scope") == SCOPE and metadata.get("method") == identity["variant"], "failure-method-mismatch")
        lines = trace_bytes.decode().splitlines()
        _require(lines and all(lines), "failure-empty-trace")
        rows = [strict_json_loads(line) for line in lines]
        _require(len(rows) == metadata.get("step_count"), "failure-frame-count-mismatch")
        replay_frames, maximum_difference = [], 0.
        if predictor is not None:
            for row in rows:
                image = row["image"]
                _require(image["path"] == f"images/{row['frame']:08d}.png", "failure-image-identity-mismatch")
                width, height, rgb = decode_rgb8_png(_safe_file(root, image["path"]).read_bytes(), image["sha256"], (640, 360))
                reproduced, observed = predictor.predict(rgb, width=width, height=height), row["perception"]
                _require(_same({k: v for k, v in observed.items() if k not in ("timing_ms", "waypoints_ego_m")},
                               {k: v for k, v in reproduced.items() if k not in ("timing_ms", "waypoints_ego_m")}, 1e-8),
                         "failure-RGB-status-or-diagnostics-mismatch")
                if observed["status"] == "PASS":
                    from .learning_data import validate_waypoint_points
                    validate_waypoint_points(observed["waypoints_ego_m"])
                    validate_waypoint_points(reproduced["waypoints_ego_m"])
                    difference = max(abs(a - b) for first, second in zip(observed["waypoints_ego_m"], reproduced["waypoints_ego_m"]) for a, b in zip(first, second))
                    _require(difference <= .05, "failure-RGB-waypoint-reproduction-mismatch")
                    maximum_difference = max(maximum_difference, difference)
                else:
                    _require(observed["waypoints_ego_m"] is None and reproduced["waypoints_ego_m"] is None, "failure-failed-path-retained")
                replay_frames.append({"rgb_sha256": hashlib.sha256(rgb).hexdigest(), "perception": observed})
        replay = None if predictor is None else _RecordedReplay(replay_frames)
        native = _validate_camera_episode(root, allowed_conditions=CONDITIONS, scope=SCOPE,
            methods=(identity["variant"],), prediction_inputs=("rgb-and-fixed-nominal-calibration" if identity["variant"].endswith("_near") else "rgb-only"),
            predictor=replay, expected_initial_reference_m=expected_initial_reference_m)
        _require(native["status"] == "PASS", "failure-control-or-label-or-metric-replay-failed:" + str(native.get("errors")))
        if replay is not None:
            _require(replay.index == len(rows), "failure-replay-incomplete")
        _require(_safe_file(root, "episode.json").read_bytes() == metadata_bytes and _safe_file(root, "steps.jsonl").read_bytes() == trace_bytes,
                 "failure-source-changed")
        report.update(status="PASS", native=native, frame_count=len(rows), maximum_coordinate_replay_difference_m=maximum_difference,
            metrics=metadata["evaluation_metrics"], closed_loop_executed=True, driving_success=metadata["evaluation_metrics"]["driving_success"])
    except Exception as error:
        report["errors"].append(str(error))
    return report
