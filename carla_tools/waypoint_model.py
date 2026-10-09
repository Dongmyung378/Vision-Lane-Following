"""A pretrained ResNet18 RGB encoder and actor-coordinate waypoint head."""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import torch
from torch import nn

from .learning_data import (
    HORIZONS_M, ROAD_ROI_TOP_PX, LearningConfig, LearningDataError, load_learning_config,
    preprocess_rgb, validate_waypoint_points,
)

PRETRAINED_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
PRETRAINED_HASH_PREFIX = "f37072fd"
SPATIAL_ROW_CUTS = (0, 3, 6)
SPATIAL_COL_CUTS = (0, 2, 5, 7, 10)


class WaypointModelError(ValueError):
    """Model inputs or pretrained provenance cannot be verified."""


def model_contract(config=None):
    config = config or load_learning_config()
    return {
        "schema_version": 1,
        "model_version": config.model_version,
        "architecture": "torchvision-resnet18-imagenet1k-v1-rgb-road-roi-spatial-residual-head",
        "encoder": {"feature_shape": [512, 6, 10],
                    "pooling": {"method": "nonoverlap-slice-mean", "output_grid": [2, 4],
                                "row_cuts": list(SPATIAL_ROW_CUTS), "column_cuts": list(SPATIAL_COL_CUTS)},
                    "batch_normalization": "frozen-running-statistics-trainable-affine"},
        "head": {"features": [4096, config.head_hidden_features, 10],
                 "activation": "ReLU", "dropout": config.head_dropout},
        "input": {"source": "RGB-only", "source_width": 640, "source_height": 360,
                  "width": config.image_width, "height": config.image_height,
                  "source_crop_xyxy": [0, ROAD_ROI_TOP_PX, 640, 360],
                  "crop_bounds": "half-open-fixed-native-pixels",
                  "resize": "fixed-road-roi-area", "normalization": "fixed-ImageNet-mean-std",
                  "speed_input": False, "pose_input": False, "map_input": False,
                  "per_frame_calibration_input": False},
        "output": {"shape": [5, 2], "origin": "vehicle-actor-origin", "x": "forward",
                   "y": "right", "units": "meters", "path_arc_distances_m": list(HORIZONS_M),
                   "prior": "straight-path-plus-learned-residual", "residual_scale_m": config.label_scale_m},
    }


class DeterministicSpatialPool(nn.Module):
    """Keep image location in eight disjoint feature-map bins.

    The fixed 180x320 input produces a 6x10 ResNet feature map. Explicit
    nonoverlapping slice means avoid the CUDA adaptive-average-pool backward
    operation, which is unavailable under strict deterministic algorithms.
    Flattening preserves channel, row and column order.
    """

    def forward(self, feature):
        if not isinstance(feature, torch.Tensor) or feature.ndim != 4 or tuple(feature.shape[-2:]) != (6, 10):
            raise WaypointModelError("learning-invalid-spatial-feature-shape")
        bins = [feature[:, :, row_start:row_end, col_start:col_end].mean(dim=(-2, -1))
                for row_start, row_end in zip(SPATIAL_ROW_CUTS[:-1], SPATIAL_ROW_CUTS[1:])
                for col_start, col_end in zip(SPATIAL_COL_CUTS[:-1], SPATIAL_COL_CUTS[1:])]
        return torch.stack(bins, dim=-1).reshape(feature.shape[0], feature.shape[1], 2, 4)


def _pretrained_state(weights_path=None):
    source = Path(weights_path) if weights_path is not None else Path(torch.hub.get_dir()) / "checkpoints" / PRETRAINED_URL.rsplit("/", 1)[1]
    if source.is_symlink():
        raise WaypointModelError("learning-pretrained-symlink")
    if weights_path is not None and not source.is_file():
        raise WaypointModelError("learning-pretrained-file-missing")
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.hub.download_url_to_file(PRETRAINED_URL, str(source), hash_prefix=PRETRAINED_HASH_PREFIX, progress=False)
        except Exception as error:
            raise WaypointModelError("learning-pretrained-download-failed: " + type(error).__name__) from error
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if not digest.startswith(PRETRAINED_HASH_PREFIX):
        raise WaypointModelError("learning-pretrained-sha256-mismatch")
    try:
        state = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as error:
        raise WaypointModelError("learning-pretrained-state-invalid: " + type(error).__name__) from error
    if not isinstance(state, dict) or not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise WaypointModelError("learning-pretrained-state-invalid")
    return state, {"pretrained": True, "weights_enum": "ResNet18_Weights.IMAGENET1K_V1",
                   "url": PRETRAINED_URL, "sha256": digest, "required_hash_prefix": PRETRAINED_HASH_PREFIX,
                   "hash_verified": True}


class WaypointModel(nn.Module):
    def __init__(self, *, pretrained=True, weights_path=None, config=None):
        super().__init__()
        self.config = config or load_learning_config()
        if not isinstance(self.config, LearningConfig) or type(pretrained) is not bool:
            raise WaypointModelError("learning-invalid-model-config")
        if not pretrained and weights_path is not None:
            raise WaypointModelError("learning-weights-path-requires-pretrained")
        # Calling resnet18(weights=None) keeps initialization explicit. When
        # pretrained is requested, a verified complete encoder state is loaded
        # strictly; download or state errors never produce a random fallback.
        from torchvision.models import resnet18
        self.encoder = resnet18(weights=None)
        if pretrained:
            state, self.pretrained_provenance = _pretrained_state(weights_path)
            try:
                self.encoder.load_state_dict(state, strict=True)
            except RuntimeError as error:
                raise WaypointModelError("learning-pretrained-architecture-mismatch") from error
        else:
            self.pretrained_provenance = {"pretrained": False, "hash_verified": False, "purpose": "explicit-random-or-checkpoint-initialization"}
        # The complete original topology is loaded strictly before replacing
        # its parameterless global pool and classification projection.
        self.encoder.avgpool = DeterministicSpatialPool()
        self.encoder.fc = nn.Identity()
        self.head = nn.Sequential(nn.Linear(4096, self.config.head_hidden_features), nn.ReLU(inplace=True),
                                  nn.Dropout(self.config.head_dropout), nn.Linear(self.config.head_hidden_features, 10))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("straight_prior", torch.tensor([[distance, 0.0] for distance in HORIZONS_M], dtype=torch.float32))
        self.train(self.training)

    def train(self, mode=True):
        """Preserve pretrained BatchNorm statistics while learning RGB features."""
        super().train(mode)
        if mode:
            for layer in self.encoder.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
        return self

    def forward(self, image):
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or tuple(image.shape[1:]) != (3, self.config.image_height, self.config.image_width):
            raise WaypointModelError("learning-invalid-input-tensor-shape")
        if not image.is_floating_point() or image.shape[0] < 1 or not bool(torch.isfinite(image).all()):
            raise WaypointModelError("learning-nonfinite-or-invalid-input-tensor")
        residual = self.head(self.encoder(image)).reshape(-1, 5, 2)
        return self.straight_prior[None, :, :] + residual * self.config.label_scale_m

    def predict_rgb(self, rgb):
        """Stateless image-only inference in the shared 5x2 waypoint contract."""
        started = time.perf_counter()
        try:
            tensor = preprocess_rgb(rgb, self.config)[None]
            device = next(self.parameters()).device
            tensor = tensor.to(device)
            self.eval()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            with torch.inference_mode():
                points = self(tensor).float().cpu().tolist()[0]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            try:
                validate_waypoint_points(points)
            except LearningDataError as error:
                raise WaypointModelError("learning-invalid-prediction: " + str(error)) from error
            return {"status": "PASS", "waypoints_ego_m": points, "error": None,
                    "diagnostics": {"model_version": self.config.model_version, "input": "RGB-only"},
                    "timing_ms": (time.perf_counter() - started) * 1000}
        except (ValueError, TypeError, RuntimeError) as error:
            return {"status": "FAIL", "waypoints_ego_m": None, "error": str(error),
                    "diagnostics": {"model_version": self.config.model_version, "exception_type": type(error).__name__},
                    "timing_ms": (time.perf_counter() - started) * 1000}


def create_model(config=None, pretrained=True, weights_path=None):
    return WaypointModel(config=config, pretrained=pretrained, weights_path=weights_path)


def waypoint_smooth_l1_loss(prediction, target, config=None):
    """SmoothL1 on both actor-coordinate axes with a declared fixed scale."""
    config = config or load_learning_config()
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor) or prediction.shape != target.shape or prediction.ndim != 3 or tuple(prediction.shape[1:]) != (5, 2):
        raise WaypointModelError("learning-invalid-loss-tensor-shape")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise WaypointModelError("learning-nonfinite-loss-input")
    return nn.functional.smooth_l1_loss(prediction / config.label_scale_m, target / config.label_scale_m, beta=config.smooth_l1_beta)
