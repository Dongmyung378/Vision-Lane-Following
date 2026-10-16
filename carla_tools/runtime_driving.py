"""Profile learned RGB driving without changing the frozen controller protocol."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .baseline_driving import BaselineTrialPolicy, PrivilegedEvaluationPolicy
from .baseline_validation import _safe_file, _validate_camera_episode
from .contracts import strict_json_loads
from .driving import load_validation_config, run_drive_trial
from .driving_validation import _require, _same
from .failure_driving import REFERENCE_IDENTITY, SOURCES as FAILURE_SOURCES
from .learning_data import preprocess_rgb, validate_waypoint_points
from .near_correction import NearCorrectedPredictor
from .projection import decode_rgb8_png
from .refinement_driving import CONDITIONS, INITIAL_REFERENCES_M, NativeCheckpointPredictor
from .refinement_validation import _RecordedReplay

SCOPE = "profiled-rgb-waypoint-development-driving"
RUNTIME_SOURCES = tuple(dict.fromkeys((*FAILURE_SOURCES,
    "carla_tools/runtime_driving.py", "carla_tools/pipeline_latency.py",
    "carla_tools/path_correction.py", "carla_tools/driving_demo.py",
    "scripts/run_runtime_pipeline.py")))
TIMING_SCOPE = ("BGRA conversion, learned RGB preprocessing and device transfer, "
    "synchronized model forward and output decoding, RGB path correction, "
    "prediction validation and shared controller; excludes simulator wait, "
    "route-label generation, disk writes and video encoding")


def runtime_source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in RUNTIME_SOURCES}


def validate_path_correction_replay(observed, reproduced):
    """Replay fixed RGB and recorded learned inputs, retaining exact gates."""
    _require(type(observed) is dict and type(reproduced) is dict and set(observed) == set(reproduced),
        "runtime-path-diagnostic-schema-mismatch")
    for key in observed:
        if key in ("paint_pixels", "qualified_components"):
            _require(type(observed[key]) is int and type(reproduced[key]) is int,
                "runtime-path-discrete-diagnostic-type-mismatch")
        _require(_same(observed[key], reproduced[key], 1e-8),
            "runtime-path-diagnostic-replay-mismatch:" + key)


def validate_runtime_timing(timing, legacy, status):
    """Bind detailed intervals to the independently recorded enclosing wall time."""
    from .pipeline_latency import STAGES, validate_pipeline_timing
    validate_pipeline_timing(timing)
    _require(type(legacy) is dict and set(legacy) == {"preprocess_ms", "inference_ms", "controller_ms", "pipeline_ms"},
        "runtime-legacy-timing-schema-mismatch")
    import math
    _require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in legacy.values()),
        "runtime-legacy-timing-invalid")
    detail = timing["stages_ms"]
    epsilon = 1e-5
    _require(timing["total_processing_ms"] + epsilon >= legacy["pipeline_ms"]
        and detail["bgra_conversion_ms"] <= legacy["preprocess_ms"] + epsilon
        and sum(detail[k] for k in ("learned_preprocessing_ms", "forward_decode_ms", "near_correction_ms"))
            <= legacy["inference_ms"] + epsilon
        and detail["controller_ms"] <= legacy["controller_ms"] + epsilon,
        "runtime-detailed-and-legacy-interval-mismatch")
    if status == "PASS":
        _require(timing["executed_stages"] == list(STAGES) and not timing["failed_stages"],
            "runtime-success-stage-coverage-mismatch")
        _require(timing["device"] != "cuda" or set(timing["synchronized_stages"]) ==
            {"learned_preprocessing_ms", "forward_decode_ms"}, "runtime-cuda-stages-not-synchronized")


def validate_recorded_correction(predictor, rgb, observed, *, width=640, height=360):
    """Independently replay deterministic correction on the recorded raw path."""
    validate_waypoint_points(observed["learned_waypoints_ego_m"])
    validate_waypoint_points(observed["waypoints_ego_m"])
    corrected, diagnostic = predictor.corrector.apply(rgb, observed["learned_waypoints_ego_m"],
        width=width, height=height)
    _require(_same(corrected, observed["waypoints_ego_m"], 1e-8), "runtime-recorded-raw-correction-mismatch")
    _require(observed["waypoints_ego_m"][2:] == observed["learned_waypoints_ego_m"][2:],
        "runtime-distant-points-changed")
    correction_key = "near_correction" if predictor.correction == "near" else "path_correction"
    validate_path_correction_replay(observed["diagnostics"][correction_key], diagnostic)


class RuntimeCheckpointPredictor:
    """Use identical learned preprocessing/weights with separate timed stages."""

    def __init__(self, checkpoint, *, expected_sha256, device="cpu", correction="near"):
        if correction not in ("near", "path"):
            raise ValueError("runtime-correction-invalid")
        self.raw = NativeCheckpointPredictor(checkpoint, kind="refined",
            expected_sha256=expected_sha256, device=device)
        if self.raw.identity["variant"] != "hybrid_geometry":
            raise ValueError("runtime-geometric-hybrid-required")
        if correction == "near":
            wrapped = NearCorrectedPredictor(self.raw)
            self.corrector = wrapped.corrector
            self._identity = wrapped.identity
        else:
            from .path_correction import PathCorrectedPredictor
            wrapped = PathCorrectedPredictor(self.raw)
            self.corrector = wrapped.corrector
            self._identity = wrapped.identity
        self.correction = correction

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def timer(self):
        from .pipeline_latency import PipelineFrameTimer
        device = next(self.raw.model.parameters()).device
        sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None
        return PipelineFrameTimer(device=device.type, synchronize=sync)

    def predict_profiled(self, rgb, timer, *, width=640, height=360):
        if (width, height) != (640, 360):
            raise ValueError("runtime-camera-dimensions-mismatch")
        began = time.perf_counter()
        model, config = self.raw.model, self.raw.config
        device = next(model.parameters()).device
        diagnostics = {"model_version": self.raw.identity["model_version"],
            "input": "RGB-only", "variant": self._identity["variant"],
            "checkpoint_sha256": self._identity["checkpoint_sha256"]}
        try:
            def prepare():
                image = (np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3).copy()
                    if isinstance(rgb, (bytes, bytearray)) else np.asarray(rgb))
                return preprocess_rgb(image, config)[None].to(device)
            tensor = timer.measure("learned_preprocessing_ms", prepare, gpu=device.type == "cuda")
            def forward():
                with torch.inference_mode(), torch.autocast(device_type=device.type,
                        enabled=config.mixed_precision and device.type == "cuda"):
                    points = model(tensor).float().cpu().tolist()[0]
                validate_waypoint_points(points)
                return points
            points = timer.measure("forward_decode_ms", forward, gpu=device.type == "cuda")
            corrected, detail = timer.measure("near_correction_ms",
                lambda: self.corrector.apply(rgb, points, width=width, height=height))
            diagnostics["near_correction" if self.correction == "near" else "path_correction"] = detail
            validate_waypoint_points(corrected)
            result = {"status": "PASS", "waypoints_ego_m": corrected,
                "learned_waypoints_ego_m": points, "error": None, "diagnostics": diagnostics}
        except (ValueError, TypeError, RuntimeError) as error:
            diagnostics["exception_type"] = type(error).__name__
            result = {"status": "FAIL", "waypoints_ego_m": None,
                "learned_waypoints_ego_m": None, "error": str(error), "diagnostics": diagnostics}
        result["timing_ms"] = (time.perf_counter() - began) * 1000
        return result

    def predict(self, rgb, *, width=640, height=360):
        timer = self.timer()
        return self.predict_profiled(rgb, timer, width=width, height=height)


class RuntimeTrialPolicy(BaselineTrialPolicy):
    scope = SCOPE

    def __init__(self, *args, initial_reference_m=0., **kwargs):
        if type(initial_reference_m) not in (int, float) or initial_reference_m not in INITIAL_REFERENCES_M:
            raise ValueError("runtime-start-invalid")
        super().__init__(*args, **kwargs)
        self.method = self.predictor.identity["variant"]
        self.initial_reference_m = float(initial_reference_m)

    def configure_metadata(self, metadata):
        super().configure_metadata(metadata)
        metadata.update(native_model=self.predictor.identity,
            requested_initial_reference_m=self.initial_reference_m,
            native_source_sha256=runtime_source_hashes(),
            latency_scope=TIMING_SCOPE, detailed_latency_required=True)

    def command(self, image, speed_mps, controller):
        from .controller import _rear_points
        from scripts.carla_rgb_probe import rgb_from_bgra
        timer = self.predictor.timer()
        began = time.perf_counter()
        rgb, _ = timer.measure("bgra_conversion_ms",
            lambda: rgb_from_bgra(bytes(image.raw_data), image.width, image.height))
        converted = time.perf_counter()
        prediction = self.predictor.predict_profiled(rgb, timer, width=image.width, height=image.height)
        json.dumps(prediction, allow_nan=False)
        if prediction["status"] == "PASS":
            _rear_points(prediction["waypoints_ego_m"], controller.geometry.rear_axle_x_m,
                controller.geometry.rear_axle_y_m)
            if any(p[0] <= 0 for p in prediction["waypoints_ego_m"]):
                raise ValueError("runtime-waypoints-not-forward")
        predicted = time.perf_counter()
        command = (timer.measure("controller_ms", lambda: controller.step(
            prediction["waypoints_ego_m"], speed_mps, dt_seconds=.05))
            if prediction["status"] == "PASS" else None)
        finished = time.perf_counter()
        prediction["pipeline_timing"] = timer.finish()
        legacy = {"preprocess_ms": (converted - began) * 1000,
            "inference_ms": (predicted - converted) * 1000,
            "controller_ms": (finished - predicted) * 1000,
            "pipeline_ms": (finished - began) * 1000}
        return command, prediction, legacy


class RuntimeReferencePolicy(PrivilegedEvaluationPolicy):
    scope = SCOPE

    def __init__(self, *args, initial_reference_m=0., **kwargs):
        if type(initial_reference_m) not in (int, float) or initial_reference_m not in INITIAL_REFERENCES_M:
            raise ValueError("runtime-reference-start-invalid")
        super().__init__(*args, **kwargs)
        self.initial_reference_m = float(initial_reference_m)

    def configure_metadata(self, metadata):
        super().configure_metadata(metadata)
        metadata.update(native_model=copy.deepcopy(REFERENCE_IDENTITY),
            requested_initial_reference_m=self.initial_reference_m,
            native_source_sha256=runtime_source_hashes(), detailed_latency_required=False)


def run_runtime_trial(route, case, condition, output, collection_config, *, predictor,
        carla_module, baseline_config, evaluation_config, controller_config, initial_reference_m=0.):
    policy_type = RuntimeReferencePolicy if predictor is None else RuntimeTrialPolicy
    kwargs = {} if predictor is None else {"predictor": predictor}
    policy = policy_type(baseline_config, evaluation_config, route, condition,
        initial_reference_m=initial_reference_m, **kwargs)
    return run_drive_trial(route, case, condition, output, collection_config, carla_module=carla_module,
        validation_config=load_validation_config(), controller_config=controller_config, trial_policy=policy)


def validate_runtime_episode(directory, predictor, *, expected_initial_reference_m):
    """Verify CPU/CUDA predictions, then replay original control/event records."""
    report = {"status": "FAIL", "scope": SCOPE, "errors": []}
    root = Path(directory)
    try:
        _require(type(expected_initial_reference_m) in (int, float)
            and expected_initial_reference_m in INITIAL_REFERENCES_M, "runtime-start-invalid")
        meta_bytes, trace_bytes = _safe_file(root, "episode.json").read_bytes(), _safe_file(root, "steps.jsonl").read_bytes()
        metadata = strict_json_loads(meta_bytes.decode())
        identity = copy.deepcopy(REFERENCE_IDENTITY if predictor is None else predictor.identity)
        observed_identity = copy.deepcopy(metadata["native_model"])
        observed_device = observed_identity.pop("device", None)
        _require(observed_device in ("cpu", "cuda"), "runtime-device-invalid")
        identity.pop("device", None)
        _require(_same(observed_identity, identity, 1e-12), "runtime-model-identity-mismatch")
        _require(metadata["native_source_sha256"] == runtime_source_hashes(), "runtime-source-mismatch")
        _require(metadata["requested_initial_reference_m"] == expected_initial_reference_m,
            "runtime-requested-start-mismatch")
        _require(metadata["scope"] == SCOPE and metadata["method"] == identity["variant"], "runtime-scope-or-method-mismatch")
        rows = [strict_json_loads(line) for line in trace_bytes.decode().splitlines()]
        _require(rows and len(rows) == metadata["step_count"], "runtime-frame-count-mismatch")
        frames, maximum_difference = [], 0.
        if predictor is not None:
            _require(metadata.get("detailed_latency_required") is True and metadata["latency_scope"] == TIMING_SCOPE,
                "runtime-latency-scope-mismatch")
            for row in rows:
                image = row["image"]
                _require(image["path"] == f"images/{row['frame']:08d}.png", "runtime-image-path-mismatch")
                w, h, rgb = decode_rgb8_png(_safe_file(root, image["path"]).read_bytes(), image["sha256"], (640, 360))
                reproduced, observed = predictor.predict(rgb, width=w, height=h), row["perception"]
                timing = observed.get("pipeline_timing")
                validate_runtime_timing(timing, row["latency_ms"], observed["status"])
                _require(timing["device"] == observed_device, "runtime-timing-device-mismatch")
                observed_content = {k: v for k, v in observed.items() if k not in ("timing_ms", "pipeline_timing", "waypoints_ego_m", "learned_waypoints_ego_m")}
                reproduced_content = {k: v for k, v in reproduced.items() if k not in ("timing_ms", "pipeline_timing", "waypoints_ego_m", "learned_waypoints_ego_m")}
                if observed["status"] == "PASS":
                    observed_content, reproduced_content = copy.deepcopy(observed_content), copy.deepcopy(reproduced_content)
                    validate_waypoint_points(observed["learned_waypoints_ego_m"])
                    raw_difference = max(abs(a - b) for p, q in zip(observed["learned_waypoints_ego_m"],
                        reproduced["learned_waypoints_ego_m"]) for a, b in zip(p, q))
                    _require(raw_difference <= .05, "runtime-raw-checkpoint-replay-mismatch")
                    validate_recorded_correction(predictor, rgb, observed, width=w, height=h)
                    correction_key = "near_correction" if predictor.correction == "near" else "path_correction"
                    observed_content["diagnostics"].pop(correction_key)
                    reproduced_content["diagnostics"].pop(correction_key)
                _require(_same(observed_content, reproduced_content, 1e-8),
                    "runtime-RGB-diagnostics-mismatch")
                if observed["status"] == "PASS":
                    validate_waypoint_points(observed["waypoints_ego_m"])
                    validate_waypoint_points(reproduced["waypoints_ego_m"])
                    difference = max(abs(a - b) for p, q in zip(observed["waypoints_ego_m"], reproduced["waypoints_ego_m"]) for a, b in zip(p, q))
                    _require(difference <= .05, "runtime-RGB-waypoint-mismatch")
                    maximum_difference = max(maximum_difference, difference)
                else:
                    _require(observed["waypoints_ego_m"] is None and reproduced["waypoints_ego_m"] is None,
                        "runtime-failed-path-retained")
                    _require(observed.get("learned_waypoints_ego_m") is None
                        and reproduced.get("learned_waypoints_ego_m") is None, "runtime-failed-raw-path-retained")
                frames.append({"rgb_sha256": hashlib.sha256(rgb).hexdigest(), "perception": observed})
        replay = None if predictor is None else _RecordedReplay(frames)
        gate = _validate_camera_episode(root, allowed_conditions=CONDITIONS, scope=SCOPE,
            methods=(identity["variant"],), prediction_inputs="rgb-and-fixed-nominal-calibration",
            predictor=replay, expected_initial_reference_m=expected_initial_reference_m)
        _require(gate["status"] == "PASS", "runtime-native-replay-failed:" + str(gate.get("errors")))
        _require(replay is None or replay.index == len(rows), "runtime-replay-incomplete")
        _require(_safe_file(root, "episode.json").read_bytes() == meta_bytes
            and _safe_file(root, "steps.jsonl").read_bytes() == trace_bytes, "runtime-source-changed")
        report.update(status="PASS", frame_count=len(rows), native=gate,
            maximum_coordinate_replay_difference_m=maximum_difference,
            metrics=metadata["evaluation_metrics"], closed_loop_executed=True,
            driving_success=metadata["evaluation_metrics"]["driving_success"])
    except Exception as error:
        report["errors"].append(str(error))
    return report
