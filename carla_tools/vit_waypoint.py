"""Frozen, verified DINOv3 spatial features for RGB waypoint regression.

This candidate is separate from the ResNet deployment contract. No network
download or random pretrained fallback occurs in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .learning_data import HORIZONS_M, load_learning_config, preprocess_rgb, validate_waypoint_points

MODEL_ID = "vit_base_patch16_dinov3.lvd1689m"
MODEL_VERSION = "vlf-dinov3-vitb16-frozen-spatial-waypoint-v1"
WEIGHTS_REPOSITORY = "https://huggingface.co/timm/" + MODEL_ID + "/resolve/"
PATCH_ROWS, PATCH_COLUMNS, FEATURE_CHANNELS = 12, 20, 768
ROW_CUTS, COLUMN_CUTS = (0, 6, 12), (0, 5, 10, 15, 20)
FEATURE_COUNT = FEATURE_CHANNELS * 2 * 4


class VitWaypointError(ValueError):
    """The ViT candidate cannot establish its input or weight identity."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class VitConfig:
    schema_version: int = 1
    model_version: str = MODEL_VERSION
    backbone_id: str = MODEL_ID
    seed: int = 42
    batch_size: int = 64
    feature_batch_size: int = 16
    epochs: int = 32
    early_stopping_patience: int = 8
    warmup_epochs: int = 1
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    smooth_l1_beta: float = 0.03
    gradient_clip_norm: float = 1.0
    head_hidden_features: int = 256
    head_dropout: float = 0.2
    label_scale_m: float = 15.0
    mixed_precision: bool = True
    overfit_steps: int = 60
    overfit_batch_size: int = 16
    overfit_learning_rate: float = 0.001
    overfit_loss_ratio_target: float = 0.35
    near_mean_target_m: float = 0.30
    far_mean_target_m: float = 1.0

    def __post_init__(self):
        if self.schema_version != 1 or type(self.schema_version) is not int or self.model_version != MODEL_VERSION or self.backbone_id != MODEL_ID:
            raise VitWaypointError("vit-unsupported-model-version")
        for name in ("seed", "batch_size", "feature_batch_size", "epochs", "early_stopping_patience", "warmup_epochs",
                     "head_hidden_features", "overfit_steps", "overfit_batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name in ("seed", "warmup_epochs") else 1):
                raise VitWaypointError("vit-invalid-" + name)
        for name in ("learning_rate", "weight_decay", "smooth_l1_beta", "gradient_clip_norm", "head_dropout", "label_scale_m",
                     "overfit_learning_rate", "overfit_loss_ratio_target", "near_mean_target_m", "far_mean_target_m"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise VitWaypointError("vit-invalid-" + name)
        if self.label_scale_m != 15 or min(self.learning_rate, self.smooth_l1_beta, self.gradient_clip_norm,
                                          self.overfit_learning_rate, self.near_mean_target_m) <= 0:
            raise VitWaypointError("vit-invalid-scale-or-learning-rate")
        if not 0 <= self.head_dropout < 1 or not 0 < self.overfit_loss_ratio_target < 1 or type(self.mixed_precision) is not bool:
            raise VitWaypointError("vit-invalid-ratio-or-precision")
        if self.near_mean_target_m > self.far_mean_target_m:
            raise VitWaypointError("vit-invalid-performance-target")

    def to_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return canonical_sha256(self.to_dict())


def vit_config_from_dict(raw):
    if type(raw) is not dict or set(raw) != {field.name for field in fields(VitConfig)}:
        raise VitWaypointError("vit-invalid-config-fields")
    return VitConfig(**raw)


def model_contract(config=None):
    config = config or VitConfig()
    return {
        "schema_version": 1, "model_version": config.model_version, "backbone_id": config.backbone_id,
        "encoder": {"pretrained": "DINOv3-LVD1689M", "training": "frozen-eval-no-gradient",
                    "attention": "torch-SDPA-math; FP32-intermediates-for-FP16-inputs",
                    "patch_size": [16, 16], "feature_shape": [768, 12, 20], "prefix_tokens": "excluded",
                    "pooling": {"method": "nonoverlap-slice-mean", "output_grid": [2, 4],
                                "row_cuts": list(ROW_CUTS), "column_cuts": list(COLUMN_CUTS)}},
        "head": {"features": [FEATURE_COUNT, config.head_hidden_features, 10], "activation": "ReLU", "dropout": config.head_dropout},
        "input": {"source": "RGB-only", "source_shape": [360, 640, 3], "source_crop_xyxy": [0, 144, 640, 360],
                  "resize": "INTER_AREA-to-320x180", "normalization": "fixed-ImageNet-mean-std",
                  "padding": {"mode": "replicate", "left": 0, "right": 0, "top": 0, "bottom": 12},
                  "encoder_shape": [3, 192, 320], "speed_input": False, "pose_input": False, "map_input": False,
                  "per_frame_calibration_input": False},
        "output": {"shape": [5, 2], "origin": "vehicle-actor-origin", "x": "forward", "y": "right", "units": "meters",
                   "path_arc_distances_m": list(HORIZONS_M), "prior": "straight-path-plus-learned-residual",
                   "residual_scale_m": config.label_scale_m},
        "training_inputs": {"feature_cache": "fixed-unaugmented-RGB-only", "augmentation": "none",
                            "labels": "loss-only; never supplied to encoder or prediction head"},
    }


def _valid_hash(value):
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def validate_pretrained_provenance(value):
    if (type(value) is not dict or value.get("pretrained") is not True or value.get("hash_verified") is not True or
            value.get("backbone_id") != MODEL_ID or not _valid_hash(value.get("sha256")) or
            type(value.get("url")) is not str or not value["url"].startswith(WEIGHTS_REPOSITORY)):
        raise VitWaypointError("vit-pretrained-provenance-invalid")


def _pretrained_state(path, expected_sha256, weights_url):
    if not _valid_hash(expected_sha256):
        raise VitWaypointError("vit-pretrained-full-sha256-required")
    if type(weights_url) is not str or not weights_url.startswith(WEIGHTS_REPOSITORY):
        raise VitWaypointError("vit-pretrained-official-source-required")
    source = Path(path)
    if source.is_symlink() or any(parent.is_symlink() for parent in source.parents):
        raise VitWaypointError("vit-pretrained-symlink")
    if not source.is_file():
        raise VitWaypointError("vit-pretrained-file-missing")
    if sha256_file(source) != expected_sha256:
        raise VitWaypointError("vit-pretrained-sha256-mismatch")
    try:
        if source.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(source), device="cpu")
        else:
            state = torch.load(source, map_location="cpu", weights_only=True)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        raise VitWaypointError("vit-pretrained-state-invalid: " + type(error).__name__) from error
    if not isinstance(state, dict) or not state or not all(type(key) is str and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise VitWaypointError("vit-pretrained-state-invalid")
    if sha256_file(source) != expected_sha256:
        raise VitWaypointError("vit-pretrained-changed-during-load")
    return state, {"pretrained": True, "hash_verified": True, "backbone_id": MODEL_ID,
                   "sha256": expected_sha256, "url": weights_url}


def _timm_backbone(config):
    try:
        import timm
    except ImportError as error:
        raise VitWaypointError("vit-timm-dependency-missing") from error
    # Both topology and input dimensions are explicit. pretrained=False avoids
    # implicit network access; the complete official state is loaded below.
    model = timm.create_model(config.backbone_id, pretrained=False, img_size=(192, 320), num_classes=0,
                             dynamic_img_size=True)
    # timm's manual q @ k attention can overflow half-precision scores. Its
    # SDPA branch is used with PyTorch's math backend explicitly selected in
    # encode(). Math SDPA accumulates FP16 inputs in FP32 and requires neither
    # FlashAttention nor a GPU architecture newer than T4.
    for block in getattr(model, "blocks", ()):
        if hasattr(getattr(block, "attn", None), "fused_attn"):
            block.attn.fused_attn = True
    return model


class PatchSpatialPool(nn.Module):
    """Disjoint patch means retain coarse image location without class tokens."""

    def forward(self, feature):
        if not isinstance(feature, torch.Tensor) or feature.ndim != 4 or tuple(feature.shape[1:]) != (768, 12, 20):
            raise VitWaypointError("vit-invalid-spatial-feature-shape")
        bins = [feature[:, :, top:bottom, left:right].mean(dim=(-2, -1))
                for top, bottom in zip(ROW_CUTS[:-1], ROW_CUTS[1:])
                for left, right in zip(COLUMN_CUTS[:-1], COLUMN_CUTS[1:])]
        return torch.stack(bins, dim=-1).flatten(1)


class VitWaypointModel(nn.Module):
    def __init__(self, *, config=None, pretrained=True, weights_path=None, expected_weights_sha256=None,
                 weights_url=None, backbone_factory=None):
        super().__init__()
        self.config = config or VitConfig()
        if not isinstance(self.config, VitConfig) or type(pretrained) is not bool:
            raise VitWaypointError("vit-invalid-model-config")
        if not pretrained and any(value is not None for value in (weights_path, expected_weights_sha256, weights_url)):
            raise VitWaypointError("vit-checkpoint-initialization-must-not-use-pretrained-file")
        # Verify the file before even creating the encoder. An unavailable or
        # corrupt official artifact cannot silently become a random model.
        if pretrained:
            if weights_path is None:
                raise VitWaypointError("vit-pretrained-file-required")
            state, provenance = _pretrained_state(weights_path, expected_weights_sha256, weights_url)
        self.encoder = (backbone_factory or _timm_backbone)(self.config)
        if pretrained:
            try:
                self.encoder.load_state_dict(state, strict=True)
            except RuntimeError as error:
                raise VitWaypointError("vit-pretrained-architecture-mismatch") from error
            self.pretrained_provenance = provenance
            del state
        else:
            self.pretrained_provenance = {"pretrained": False, "hash_verified": False,
                                         "purpose": "explicit-unit-fixture-or-complete-checkpoint-initialization"}
        self.encoder.requires_grad_(False).eval()
        self.pool = PatchSpatialPool()
        self.head = nn.Sequential(nn.Linear(FEATURE_COUNT, self.config.head_hidden_features), nn.ReLU(),
                                  nn.Dropout(self.config.head_dropout), nn.Linear(self.config.head_hidden_features, 10))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("straight_prior", torch.tensor([[distance, 0.0] for distance in HORIZONS_M], dtype=torch.float32))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def encode(self, image):
        if (not isinstance(image, torch.Tensor) or image.ndim != 4 or tuple(image.shape[1:]) != (3, 180, 320) or
                image.shape[0] < 1 or not image.is_floating_point() or not bool(torch.isfinite(image).all())):
            raise VitWaypointError("vit-invalid-input-tensor")
        image = nn.functional.pad(image, (0, 0, 0, 12), mode="replicate")
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type=image.device.type, dtype=torch.float16,
                                             enabled=self.config.mixed_precision and image.device.type == "cuda"):
            tokens = self.encoder.forward_features(image)
        prefix = getattr(self.encoder, "num_prefix_tokens", None)
        if (type(prefix) is not int or prefix < 0 or not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or
                tuple(tokens.shape[1:]) != (prefix + PATCH_ROWS * PATCH_COLUMNS, FEATURE_CHANNELS)):
            raise VitWaypointError("vit-invalid-patch-token-shape")
        finite = torch.isfinite(tokens)
        if not bool(finite.all()):
            # Only numerical tensor diagnostics are emitted: no source image,
            # labels, poses or authentication details enter this error.
            diagnostics = {"shape": list(tokens.shape), "dtype": str(tokens.dtype),
                           "finite_count": int(finite.sum()), "total_count": tokens.numel(),
                           "finite_max_abs": float(tokens[finite].abs().max()) if bool(finite.any()) else None}
            raise VitWaypointError("vit-nonfinite-patch-tokens: " + json.dumps(diagnostics, allow_nan=False))
        feature = tokens[:, prefix:].transpose(1, 2).reshape(image.shape[0], FEATURE_CHANNELS, PATCH_ROWS, PATCH_COLUMNS)
        return self.pool(feature).float()

    def predict_features(self, features):
        if (not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] != FEATURE_COUNT or
                features.shape[0] < 1 or not features.is_floating_point() or not bool(torch.isfinite(features).all())):
            raise VitWaypointError("vit-invalid-cached-feature")
        # The deployment path uses the same encoder FP16 / head FP32 protocol
        # as feature extraction and training, including inside outer autocast.
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self.straight_prior[None] + self.head(features.float()).reshape(-1, 5, 2) * self.config.label_scale_m

    def forward(self, image):
        return self.predict_features(self.encode(image))

    def predict_rgb(self, rgb):
        started = time.perf_counter()
        try:
            device = next(self.parameters()).device
            image = preprocess_rgb(rgb, load_learning_config())[None].to(device)
            self.eval()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            with torch.inference_mode():
                points = self(image).float().cpu().tolist()[0]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            validate_waypoint_points(points)
            return {"status": "PASS", "waypoints_ego_m": points, "error": None,
                    "diagnostics": {"model_version": self.config.model_version, "input": "RGB-only"},
                    "timing_ms": (time.perf_counter() - started) * 1000}
        except (ValueError, TypeError, RuntimeError) as error:
            return {"status": "FAIL", "waypoints_ego_m": None, "error": str(error),
                    "diagnostics": {"model_version": self.config.model_version, "exception_type": type(error).__name__},
                    "timing_ms": (time.perf_counter() - started) * 1000}


def waypoint_loss(prediction, target, config):
    if (not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor) or
            prediction.shape != target.shape or prediction.ndim != 3 or tuple(prediction.shape[1:]) != (5, 2) or
            not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all())):
        raise VitWaypointError("vit-invalid-loss-input")
    return nn.functional.smooth_l1_loss(prediction / config.label_scale_m, target / config.label_scale_m,
                                      beta=config.smooth_l1_beta)
