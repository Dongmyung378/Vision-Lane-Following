"""Strict RGB inference for an explicitly selected spatial experiment checkpoint."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch

from .inference import InferenceError, _expected_sha256, _input_file, _sha256_file
from .spatial_training import load_spatial_checkpoint


class SpatialCheckpointPredictor:
    def __init__(self, checkpoint_path, *, expected_checkpoint_sha256=None, device="cpu", source_sha256=None):
        path = _input_file(checkpoint_path)
        digest = _sha256_file(path)
        expected = _expected_sha256(expected_checkpoint_sha256)
        if expected is not None and digest != expected:
            raise InferenceError("spatial-inference-checkpoint-sha256-mismatch")
        try:
            model, config, checkpoint = load_spatial_checkpoint(path, device=device,source_sha256=source_sha256)
        except (ValueError, KeyError, RuntimeError, TypeError, OSError):
            raise InferenceError("spatial-inference-checkpoint-invalid") from None
        epoch, loss = checkpoint.get("epoch"), checkpoint.get("selected_dev_loss")
        if (type(epoch) is not int or epoch < 1 or type(loss) not in (int, float)
                or not math.isfinite(loss) or loss < 0 or _sha256_file(path) != digest):
            raise InferenceError("spatial-inference-selection-or-file-changed")
        self._model, self._config = model, config
        self._identity = {"inference_version": "vlf-spatial-rgb-inference-v1", "mode": "RGB-only",
                          "variant": checkpoint["variant"], "checkpoint_sha256": digest,
                          "checkpoint_role": checkpoint["checkpoint_role"], "checkpoint_epoch": epoch,
                          "model_version": checkpoint["model_contract"]["model_version"],
                          "model_contract": checkpoint["model_contract"],
                          "model_contract_sha256": checkpoint["model_contract_sha256"],
                          "source_sha256": checkpoint["source_sha256"],
                          "inference_source_sha256": {"carla_tools/spatial_inference.py": _sha256_file(Path(__file__)),
                              "carla_tools/inference.py": _sha256_file(Path(__file__).with_name("inference.py"))},
                          "manifest_sha256": checkpoint["manifest_sha256"],
                          "device": str(next(model.parameters()).device),
                          "closed_loop_verified": False}

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def predict_rgb(self, rgb):
        device = next(self._model.parameters()).device
        with torch.autocast(device_type=device.type, enabled=self._config.mixed_precision and device.type == "cuda"):
            result = self._model.predict_rgb(rgb)
        result["diagnostics"].update(model_version=self._identity["model_version"],
                                      variant=self._identity["variant"], checkpoint_sha256=self._identity["checkpoint_sha256"])
        return result
