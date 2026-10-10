"""Replay original RGB first, then independently reconstruct controls and metrics."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from .baseline_validation import _safe_file, _validate_camera_episode
from .contracts import strict_json_loads
from .driving_validation import _require, _same
from .projection import decode_rgb8_png
from .refinement_driving import SCOPE, CONDITIONS, INITIAL_REFERENCES_M, native_source_hashes


class _RecordedReplay:
    def __init__(self, frames):
        self.frames, self.index = frames, 0
    def predict(self, rgb, *, width, height):
        _require(self.index < len(self.frames), "refinement-replay-extra-frame")
        expected = self.frames[self.index]
        _require((width, height) == (640, 360) and hashlib.sha256(rgb).hexdigest() == expected["rgb_sha256"],
                 "refinement-replay-RGB-sequence-mismatch")
        self.index += 1
        return copy.deepcopy(expected["perception"])


def validate_refinement_episode(directory, predictor, *, expected_initial_reference_m=0.):
    root = Path(directory)
    report = {"status": "FAIL", "scope": SCOPE, "errors": []}
    try:
        _require(type(expected_initial_reference_m) in (int, float) and expected_initial_reference_m in INITIAL_REFERENCES_M,
                 "refinement-native-start-outside-frozen-protocol")
        metadata_bytes = _safe_file(root, "episode.json").read_bytes()
        trace_bytes = _safe_file(root, "steps.jsonl").read_bytes()
        metadata = strict_json_loads(metadata_bytes.decode())
        identity = predictor.identity
        _require(metadata.get("requested_initial_reference_m") == expected_initial_reference_m,
                 "refinement-native-requested-start-mismatch")
        # GPU execution may be independently reproduced on CPU. Every other
        # model identity field remains exact and the numerical tolerance is fixed.
        observed_identity = dict(metadata.get("native_model", {}))
        _require(observed_identity.pop("device", None) in ("cpu", "cuda"), "refinement-native-runtime-missing")
        actual_identity = dict(identity)
        actual_identity.pop("device", None)
        _require(_same(observed_identity, actual_identity, 1e-12), "refinement-native-model-identity-mismatch")
        _require(metadata.get("native_source_sha256") == native_source_hashes(), "refinement-native-source-mismatch")
        _require(metadata.get("scope") == SCOPE and metadata.get("method") == identity["variant"], "refinement-native-method-mismatch")
        lines = trace_bytes.decode().splitlines()
        _require(lines and all(lines), "refinement-native-empty-trace")
        rows = [strict_json_loads(line) for line in lines]
        _require(len(rows) == metadata.get("step_count"), "refinement-native-frame-count-mismatch")
        replay_frames, maximum_difference = [], 0.
        for row in rows:
            image = row["image"]
            _require(image["path"] == f"images/{row['frame']:08d}.png", "refinement-native-image-identity-mismatch")
            width, height, rgb = decode_rgb8_png(_safe_file(root, image["path"]).read_bytes(), image["sha256"], (640, 360))
            reproduced = predictor.predict(rgb, width=width, height=height)
            observed = row["perception"]
            _require(_same({k: v for k, v in observed.items() if k not in ("timing_ms", "waypoints_ego_m")},
                           {k: v for k, v in reproduced.items() if k not in ("timing_ms", "waypoints_ego_m")}, 1e-8),
                     "refinement-native-RGB-status-or-diagnostics-mismatch")
            if observed["status"] == "PASS":
                from .learning_data import validate_waypoint_points
                validate_waypoint_points(observed["waypoints_ego_m"])
                validate_waypoint_points(reproduced["waypoints_ego_m"])
                difference = max(abs(a-b) for first, second in zip(observed["waypoints_ego_m"], reproduced["waypoints_ego_m"]) for a, b in zip(first, second))
                _require(difference <= .05, "refinement-native-RGB-waypoint-reproduction-mismatch")
                maximum_difference = max(maximum_difference, difference)
            else:
                _require(observed["waypoints_ego_m"] is None and reproduced["waypoints_ego_m"] is None, "refinement-native-failed-path-retained")
            replay_frames.append({"rgb_sha256": hashlib.sha256(rgb).hexdigest(), "perception": observed})
        replay = _RecordedReplay(replay_frames)
        native = _validate_camera_episode(root, allowed_conditions=CONDITIONS, scope=SCOPE,
            methods=(identity["variant"],), prediction_inputs=("rgb-and-fixed-nominal-calibration" if identity["variant"] == "B0" else "rgb-only"),
            predictor=replay, expected_initial_reference_m=expected_initial_reference_m)
        _require(native["status"] == "PASS", "refinement-native-control-or-metric-replay-failed:" + str(native.get("errors")))
        _require(replay.index == len(rows), "refinement-native-replay-incomplete")
        _require(_safe_file(root, "episode.json").read_bytes() == metadata_bytes and _safe_file(root, "steps.jsonl").read_bytes() == trace_bytes,
                 "refinement-native-source-changed")
        report.update(status="PASS", native=native, frame_count=len(rows), maximum_coordinate_replay_difference_m=maximum_difference,
                      metrics=metadata["evaluation_metrics"], closed_loop_executed=True, driving_success=metadata["evaluation_metrics"]["driving_success"])
    except Exception as error:
        report["errors"].append(str(error))
    return report
