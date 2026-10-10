"""Compact attention candidates with mirrored training and path geometry.

The decoder uses fixed path-distance priors, never measured vehicle pose,
map data or a per-image curve label. All five output points remain in meters
relative to the vehicle actor origin.
"""
from __future__ import annotations

import copy
import hashlib
import math

import torch

from .learning_data import LaneWaypointDataset, LearningDataError, augment_rgb, preprocess_rgb
from .spatial_waypoint import SpatialWaypointModel, spatial_model_contract, GRID
from .waypoint_model import WaypointModelError

VARIANTS = ("hybrid_mirror", "hybrid_geometry")
HORIZON_WEIGHTS = (1., 1., 2., 2., 3.)
LENGTH_RADIUS_M = (0.5, 0.15, 0.15, 0.15, 0.15)


def mirror_rgb_and_points(rgb, points):
    """Reflect the fixed camera about cx=320 and negate actor-coordinate y.

    CARLA's 640px camera has cx=width/2. Reflecting about (width-1)/2 would
    introduce a pixel shift. The one missing boundary column is replicated.
    """
    import numpy as np
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (360, 640, 3):
        raise LearningDataError("refined-mirror-native-rgb-required")
    if not isinstance(points, torch.Tensor) or points.shape != (5, 2) or not bool(torch.isfinite(points).all()):
        raise LearningDataError("refined-mirror-target-invalid")
    mirrored = np.concatenate((rgb[:, -1:, :], rgb[:, :0:-1, :]), axis=1)
    target = points.clone()
    target[:, 1].neg_()
    return mirrored, target


class MirroredWaypointDataset(LaneWaypointDataset):
    def __getitem__(self, index):
        sample = self.samples[index]
        rgb = self._read_image(sample)
        target = torch.tensor(sample["waypoints_ego_m"], dtype=torch.float32)
        flipped = False
        if self.augment:
            key = f"{self.seed}:{self.epoch}:{sample['sample_id']}".encode()
            digest = hashlib.sha256(key).digest()
            rgb = augment_rgb(rgb, seed=int.from_bytes(digest[:8], "big"), config=self.config)
            flipped = bool(digest[8] & 1)
            if flipped:
                rgb, target = mirror_rgb_and_points(rgb, target)
        case = sample["case"]
        if flipped:
            case = {"left_curve": "right_curve", "right_curve": "left_curve", "straight": "straight"}[case]
        return {"image": preprocess_rgb(rgb, self.config), "target": target, "case": case,
                **{key: sample[key] for key in ("sample_id", "condition", "route_id")}}


def refined_contract(variant, config=None):
    if variant not in VARIANTS:
        raise WaypointModelError("refined-unknown-variant")
    value = copy.deepcopy(spatial_model_contract("hybrid", config))
    value["model_version"] = "vlf-resnet18-" + variant.replace("_", "-") + "-waypoint-v1"
    value["comparison_variant"] = variant
    value["training_augmentation"] = {"train_only_horizontal_mirror_probability": 0.5,
                                      "reflection_cx_px": 320, "target_transform": "x-unchanged-y-negated"}
    value["loss"] = {"name": "horizon-weighted-SmoothL1", "horizon_weights": list(HORIZON_WEIGHTS)}
    if variant == "hybrid_geometry":
        value["output"].update(prior="cumulative-learned-segments", decoder={
            "angle": "pi*tanh(raw)", "length": "3m+radius*tanh(raw)",
            "length_radius_m": list(LENGTH_RADIUS_M), "axes": "x=length*cos(angle),y=length*sin(angle)"})
    return value


def decode_segments(raw):
    if raw.ndim != 3 or tuple(raw.shape[1:]) != (5, 2) or not raw.is_floating_point() or not bool(torch.isfinite(raw).all()):
        raise WaypointModelError("refined-invalid-segment-output")
    angle = math.pi * raw[..., 0].tanh()
    length = 3.0 + raw.new_tensor(LENGTH_RADIUS_M) * raw[..., 1].tanh()
    return torch.stack((length * angle.cos(), length * angle.sin()), dim=-1).cumsum(dim=1)


class RefinedWaypointModel(SpatialWaypointModel):
    def __init__(self, *, variant="hybrid_geometry", **kwargs):
        if variant not in VARIANTS:
            raise WaypointModelError("refined-unknown-variant")
        super().__init__(variant="hybrid", **kwargs)
        self.variant = variant

    def forward(self, image):
        if self.variant == "hybrid_mirror":
            return super().forward(image)
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or tuple(image.shape[1:]) != (3, 180, 320):
            raise WaypointModelError("refined-invalid-input-shape")
        if image.shape[0] < 1 or not image.is_floating_point() or not bool(torch.isfinite(image).all()):
            raise WaypointModelError("refined-invalid-input-values")
        features = self.encoder(image).reshape(-1, 512, *GRID)
        return decode_segments(self.head(features))


def refined_loss(prediction, target, config):
    if prediction.shape != target.shape or prediction.ndim != 3 or tuple(prediction.shape[1:]) != (5, 2):
        raise WaypointModelError("refined-loss-shape-mismatch")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise WaypointModelError("refined-nonfinite-loss")
    error = torch.nn.functional.smooth_l1_loss(prediction / config.label_scale_m,
        target / config.label_scale_m, beta=config.smooth_l1_beta, reduction="none").mean(-1)
    weights = error.new_tensor(HORIZON_WEIGHTS)
    return (error * weights).sum(-1).div(weights.sum()).mean()
