"""RGB-only checkpoint adapters and native comparison with a shared controller."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .baseline_driving import BaselineTrialPolicy
from .driving import run_drive_trial, load_validation_config
from .inference import _input_file, _sha256_file
from .refined_training import load_refined_checkpoint
from .spatial_inference import SpatialCheckpointPredictor

SCOPE = "rgb-waypoint-refinement-development-driving"
CONDITIONS = ({"name": "center", "lateral_m": 0., "yaw_deg": 0.},
              {"name": "left_offset", "lateral_m": -.3, "yaw_deg": 0.},
              {"name": "right_offset", "lateral_m": .3, "yaw_deg": 0.})
INITIAL_REFERENCES_M = (0., 2., 4.)
NATIVE_SOURCES = ("carla_tools/refinement_driving.py", "carla_tools/refinement_validation.py",
                 "carla_tools/baseline_driving.py", "carla_tools/baseline_validation.py",
                 "carla_tools/driving.py", "carla_tools/driving_validation.py", "carla_tools/controller.py",
                 "carla_tools/evaluation.py", "carla_tools/spatial_inference.py", "carla_tools/inference.py",
                 "carla_tools/driving_media.py", "scripts/run_refinement_pipeline.py")


def native_source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {name: _sha256_file(root / name) for name in NATIVE_SOURCES}


class NativeBaselinePredictor:
    """Expose the frozen OpenCV baseline without supplying measured calibration."""
    def __init__(self, config):
        from .baseline import OpenCVLanePredictor
        self.predictor = OpenCVLanePredictor(config)
        self._identity = {"variant": "B0", "baseline_config_sha256": config.sha256,
                          "baseline_version": config.baseline_version,
                          "source_sha256": _sha256_file(Path(__file__).with_name("baseline.py")),
                          "device": "cpu", "closed_loop_verified": False}

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def predict(self, rgb, *, width=640, height=360):
        return self.predictor.predict(rgb, width=width, height=height)


class NativeCheckpointPredictor:
    def __init__(self, checkpoint, *, kind, expected_sha256, device="cpu", training_report=None):
        path = _input_file(checkpoint)
        if _sha256_file(path) != expected_sha256:
            raise ValueError("native-checkpoint-hash-mismatch")
        self.kind = kind
        if kind == "spatial":
            if training_report is None:
                raise ValueError("native-archived-training-source-required")
            report = json.loads(_input_file(training_report).read_text(encoding="utf-8"))
            if report.get("checkpoint_sha256") != expected_sha256 or report.get("status") != "PASS":
                raise ValueError("native-archived-training-binding-invalid")
            self.spatial = SpatialCheckpointPredictor(path, device=device, expected_checkpoint_sha256=expected_sha256,
                                                     source_sha256=report["source_sha256"])
            self._identity = self.spatial.identity
        elif kind == "refined":
            self.model, self.config, state = load_refined_checkpoint(path, device=device, expected_sha256=expected_sha256)
            self._identity = {"variant": state["variant"], "checkpoint_sha256": expected_sha256,
                "checkpoint_role": state["checkpoint_role"], "checkpoint_epoch": state["epoch"],
                "model_version": state["model_contract"]["model_version"], "model_contract": state["model_contract"],
                "model_contract_sha256": state["model_contract_sha256"], "source_sha256": state["source_sha256"],
                "manifest_sha256": state["manifest_sha256"], "device": str(next(self.model.parameters()).device),
                "closed_loop_verified": False}
        else:
            raise ValueError("native-unknown-checkpoint-kind")
        # Tensor devices name the default GPU 'cuda:0'; bind the runtime type
        # consistently so CPU audit can compare all other identity fields.
        self._identity["device"] = torch.device(self._identity["device"]).type

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def predict(self, rgb, *, width=640, height=360):
        if (width, height) != (640, 360):
            raise ValueError("native-camera-dimensions-mismatch")
        if isinstance(rgb, (bytes, bytearray)):
            image = np.frombuffer(rgb, dtype=np.uint8).reshape(360, 640, 3).copy()
        else:
            image = np.asarray(rgb)
        if self.kind == "spatial":
            return self.spatial.predict_rgb(image)
        device = next(self.model.parameters()).device
        with torch.autocast(device_type=device.type, enabled=self.config.mixed_precision and device.type == "cuda"):
            result = self.model.predict_rgb(image)
        result["diagnostics"].update(model_version=self._identity["model_version"],
            variant=self._identity["variant"], checkpoint_sha256=self._identity["checkpoint_sha256"])
        return result


class RefinementTrialPolicy(BaselineTrialPolicy):
    scope = SCOPE
    def __init__(self, *args, method, initial_reference_m=0., **kwargs):
        if type(initial_reference_m) not in (int, float) or initial_reference_m not in INITIAL_REFERENCES_M:
            raise ValueError("native-initial-reference-not-in-frozen-protocol")
        super().__init__(*args, **kwargs)
        self.method = method
        self.initial_reference_m = float(initial_reference_m)

    @property
    def prediction_inputs(self):
        return "rgb-and-fixed-nominal-calibration" if self.method == "B0" else "rgb-only"

    def configure_metadata(self, metadata):
        super().configure_metadata(metadata)
        metadata.update(prediction_inputs=self.prediction_inputs, native_model=self.predictor.identity,
            requested_initial_reference_m=self.initial_reference_m,
            native_source_sha256=native_source_hashes(),
            latency_scope="BGRA conversion, RGB preprocessing, checkpoint inference and shared controller; excludes simulator wait and disk writes")


def run_refinement_trial(route, case, condition, output, collection_config, *, predictor,
                         carla_module, baseline_config, evaluation_config, controller_config, initial_reference_m=0.):
    policy = RefinementTrialPolicy(baseline_config, evaluation_config, route, condition,
                                   predictor=predictor, method=predictor.identity["variant"], initial_reference_m=initial_reference_m)
    return run_drive_trial(route, case, condition, output, collection_config, carla_module=carla_module,
        validation_config=load_validation_config(), controller_config=controller_config, trial_policy=policy)
