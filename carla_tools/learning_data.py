"""Leak-free pilot samples and fixed, RGB-only learning preprocessing.

Pose, map, speed and measured calibration are retained in source records for
auditing. They are deliberately absent from the model input tensor.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath

from .contracts import read_strict_json

DEFAULT_LEARNING_PATH = Path(__file__).resolve().parents[1] / "configs" / "learning.json"
HORIZONS_M = (3.0, 6.0, 9.0, 12.0, 15.0)
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)
ROAD_ROI_TOP_PX = 144


class LearningDataError(ValueError):
    """A learning input does not satisfy the frozen source contract."""


@dataclass(frozen=True)
class LearningConfig:
    schema_version: int = 1
    model_version: str = "vlf-resnet18-road-roi-waypoint-v3"
    image_width: int = 320
    image_height: int = 180
    waypoint_distances_m: tuple = HORIZONS_M
    label_scale_m: float = 15.0
    seed: int = 42
    batch_size: int = 64
    num_workers: int = 2
    epochs: int = 32
    learning_rate: float = 0.001
    encoder_learning_rate: float = 0.00001
    weight_decay: float = 0.001
    smooth_l1_beta: float = 0.03
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    head_hidden_features: int = 256
    head_dropout: float = 0.2
    warmup_epochs: int = 1
    early_stopping_patience: int = 8
    augmentation_brightness: float = 0.2
    augmentation_contrast: float = 0.2
    augmentation_saturation: float = 0.2
    overfit_steps: int = 60
    overfit_batch_size: int = 16
    overfit_learning_rate: float = 0.003
    overfit_loss_ratio_target: float = 0.35
    near_mean_target_m: float = 0.30
    far_mean_target_m: float = 1.0

    def __post_init__(self):
        if self.schema_version != 1 or type(self.schema_version) is not int or self.model_version != "vlf-resnet18-road-roi-waypoint-v3":
            raise LearningDataError("learning-invalid-version")
        if (self.image_width, self.image_height) != (320, 180) or tuple(self.waypoint_distances_m) != HORIZONS_M:
            raise LearningDataError("learning-unsupported-input-or-horizons")
        for name in ("image_width", "image_height", "seed", "batch_size", "num_workers", "epochs", "head_hidden_features",
                     "warmup_epochs", "early_stopping_patience", "overfit_steps", "overfit_batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name in ("seed", "num_workers", "warmup_epochs") else 1):
                raise LearningDataError("learning-invalid-" + name)
        if type(self.mixed_precision) is not bool:
            raise LearningDataError("learning-invalid-mixed-precision")
        for name in ("label_scale_m", "learning_rate", "encoder_learning_rate", "weight_decay", "smooth_l1_beta",
                     "gradient_clip_norm", "head_dropout", "augmentation_brightness", "augmentation_contrast",
                     "augmentation_saturation", "overfit_learning_rate", "overfit_loss_ratio_target",
                     "near_mean_target_m", "far_mean_target_m"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise LearningDataError("learning-invalid-" + name)
        if self.label_scale_m != 15.0 or min(self.learning_rate, self.encoder_learning_rate, self.smooth_l1_beta,
                                            self.gradient_clip_norm, self.overfit_learning_rate) <= 0:
            raise LearningDataError("learning-invalid-loss-or-learning-rate")
        if not 0 <= self.head_dropout < 1 or not 0 < self.overfit_loss_ratio_target < 1:
            raise LearningDataError("learning-invalid-ratio")
        if not 0 < self.near_mean_target_m <= self.far_mean_target_m:
            raise LearningDataError("learning-invalid-development-targets")
        if any(getattr(self, name) > 0.5 for name in ("augmentation_brightness", "augmentation_contrast", "augmentation_saturation")):
            raise LearningDataError("learning-invalid-photometric-augmentation")

    def to_dict(self):
        raw = asdict(self)
        raw["waypoint_distances_m"] = list(self.waypoint_distances_m)
        return raw

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def learning_config_from_dict(raw):
    if type(raw) is not dict or set(raw) != {f.name for f in fields(LearningConfig)}:
        raise LearningDataError("learning-invalid-config-fields")
    copied = dict(raw)
    if type(copied["waypoint_distances_m"]) not in (list, tuple):
        raise LearningDataError("learning-invalid-horizons")
    copied["waypoint_distances_m"] = tuple(copied["waypoint_distances_m"])
    return LearningConfig(**copied)


def load_learning_config(path=DEFAULT_LEARNING_PATH):
    return learning_config_from_dict(read_strict_json(Path(path)))


def _dependencies():
    try:
        import cv2
        import numpy as np
        import torch
    except ImportError as error:
        raise LearningDataError("learning-dependency-missing: " + error.name) from None
    return cv2, np, torch


def preprocess_rgb(rgb, config=None):
    """Crop a fixed full-width road ROI and apply ImageNet normalization.

    Source rows [144, 360) and all columns are always selected, regardless of
    split, labels, pose or measured per-frame calibration. Actor-coordinate
    labels do not change when the image representation changes.
    """
    config = config or load_learning_config()
    cv2, np, torch = _dependencies()
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != (360, 640):
        raise LearningDataError("learning-invalid-rgb-shape-or-dtype")
    road = image[ROAD_ROI_TOP_PX:, :, :]
    resized = cv2.resize(road, (config.image_width, config.image_height), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).to(dtype=torch.float32).div_(255.0)
    mean = torch.tensor(RGB_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(RGB_STD, dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


def augment_rgb(rgb, *, seed, config=None):
    """Photometric-only augmentation; pixels and labels keep their geometry."""
    config = config or load_learning_config()
    _, np, _ = _dependencies()
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.shape != (360, 640, 3):
        raise LearningDataError("learning-invalid-rgb-shape-or-dtype")
    rng = np.random.default_rng(seed)
    values = image.astype(np.float32)
    brightness = rng.uniform(1 - config.augmentation_brightness, 1 + config.augmentation_brightness)
    contrast = rng.uniform(1 - config.augmentation_contrast, 1 + config.augmentation_contrast)
    saturation = rng.uniform(1 - config.augmentation_saturation, 1 + config.augmentation_saturation)
    values = values * brightness
    values = (values - 127.5) * contrast + 127.5
    gray = (values * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(axis=2, keepdims=True)
    values = gray + (values - gray) * saturation
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def _sample_path(root, relative):
    if type(relative) is not str or "\\" in relative:
        raise LearningDataError("learning-invalid-image-path")
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in ("..", ".") for part in path.parts) or ":" in relative:
        raise LearningDataError("learning-image-path-escape")
    source = root.joinpath(*path.parts)
    try:
        source.resolve().relative_to(root.resolve())
    except ValueError:
        raise LearningDataError("learning-image-path-escape") from None
    for part in (source, *source.parents):
        if part == root.parent:
            break
        if part.is_symlink():
            raise LearningDataError("learning-image-symlink")
    return source


def validate_waypoint_points(points):
    """Check actor-origin points ordered outward along the future path.

    A bend beyond 90 degrees can decrease forward x while future arc distance
    and actor-origin radius still increase. The downstream controller performs
    its own rear-axle-origin check after applying actual vehicle geometry.
    """
    if type(points) is not list or len(points) != 5 or any(type(row) is not list or len(row) != 2 for row in points):
        raise LearningDataError("learning-invalid-waypoint-shape")
    if any(type(value) not in (int, float) or not math.isfinite(value) for row in points for value in row):
        raise LearningDataError("learning-nonfinite-waypoint")
    radii = [math.hypot(*row) for row in points]
    if any(not 0 < row[0] <= 25 or abs(row[1]) > 20 for row in points) or any(radii[i] >= radii[i+1] for i in range(4)):
        raise LearningDataError("learning-invalid-waypoint-order-or-range")


class LaneWaypointDataset:
    """Torch-compatible map dataset with whole-scene split validation.

    Files are hashed and decoded during initialization, then rechecked whenever
    used. Augmentation seeds depend only on sample identity, epoch and seed,
    so worker scheduling cannot change the training input.
    """

    def __init__(self, manifest_path, split, config=None, *, augment=False, seed=None):
        if split not in ("train", "dev"):
            raise LearningDataError("learning-split-must-be-train-or-dev")
        if augment and split != "train":
            raise LearningDataError("learning-dev-augmentation-forbidden")
        self.config = config or load_learning_config()
        if Path(manifest_path).is_symlink():
            raise LearningDataError("learning-manifest-symlink")
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.manifest_sha256 = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        self.manifest = read_strict_json(self.manifest_path)
        manifest = self.manifest
        if type(manifest) is not dict or manifest.get("schema_version") != 1 or manifest.get("status") != "COMPLETE":
            raise LearningDataError("learning-incomplete-manifest")
        if manifest.get("scope") != "moving-rgb-waypoint-pilot" or manifest.get("town") == manifest.get("reserved_final_town"):
            raise LearningDataError("learning-invalid-pilot-scope-or-reserved-town")
        samples = manifest.get("samples")
        if type(samples) is not list or not samples:
            raise LearningDataError("learning-empty-samples")
        if manifest.get("sample_count") != len(samples):
            raise LearningDataError("learning-sample-count-mismatch")
        memberships = {key: {} for key in ("sample_id", "episode_id", "route_id", "scene_group_id", "image_sha256", "image")}
        counts = {"train": 0, "dev": 0}
        self.samples = []
        self._paths = {}
        for sample in samples:
            if type(sample) is not dict or sample.get("split") not in counts:
                raise LearningDataError("learning-invalid-sample")
            required = {"sample_id", "episode_id", "route_id", "scene_group_id", "image", "image_sha256", "waypoints_ego_m",
                        "image_width", "image_height", "case", "condition", "frame", "timestamp"}
            if not required.issubset(sample):
                raise LearningDataError("learning-missing-sample-fields")
            sample_split = sample["split"]
            for key, seen in memberships.items():
                value = sample[key]
                if type(value) is not str or not value:
                    raise LearningDataError("learning-invalid-" + key)
                if value in seen and (seen[value] != sample_split or key in ("sample_id", "image_sha256", "image")):
                    raise LearningDataError("learning-duplicate-or-split-leakage-" + key)
                seen[value] = sample_split
            if sample["case"] not in ("straight", "left_curve", "right_curve"):
                raise LearningDataError("learning-invalid-case")
            if type(sample["frame"]) is not int or sample["frame"] < 0 or type(sample["timestamp"]) not in (int, float) or not math.isfinite(sample["timestamp"]):
                raise LearningDataError("learning-invalid-frame-or-timestamp")
            if (sample["image_width"], sample["image_height"]) != (640, 360):
                raise LearningDataError("learning-unsupported-source-dimensions")
            if len(sample["image_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in sample["image_sha256"]):
                raise LearningDataError("learning-invalid-image-hash")
            validate_waypoint_points(sample["waypoints_ego_m"])
            path = _sample_path(self.root, sample["image"])
            self._paths[sample["sample_id"]] = path
            self._read_image(sample)
            counts[sample_split] += 1
            if sample_split == split:
                self.samples.append(sample)
        if counts != manifest.get("split_counts") or not all(counts.values()):
            raise LearningDataError("learning-split-count-mismatch-or-empty")
        self.split = split
        self.augment = bool(augment)
        self.seed = self.config.seed if seed is None else seed
        if type(self.seed) is not int or self.seed < 0:
            raise LearningDataError("learning-invalid-seed")
        self.epoch = 0

    def _read_image(self, sample):
        cv2, np, _ = _dependencies()
        source = self._paths[sample["sample_id"]]
        # Resolve and inspect again to catch replacement with a link after init.
        if _sample_path(self.root, sample["image"]) != source:
            raise LearningDataError("learning-source-path-changed")
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != sample["image_sha256"]:
            raise LearningDataError("learning-image-sha256-mismatch")
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise LearningDataError("learning-source-is-not-png")
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.uint8 or image.shape != (360, 640, 3):
            raise LearningDataError("learning-invalid-png-shape-or-dtype")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise LearningDataError("learning-invalid-epoch")
        self.epoch = epoch

    def assert_manifest_unchanged(self):
        """Check the frozen manifest at epoch/checkpoint boundaries."""
        if self.manifest_path.is_symlink() or hashlib.sha256(self.manifest_path.read_bytes()).hexdigest() != self.manifest_sha256:
            raise LearningDataError("learning-manifest-changed")

    def __getitem__(self, index):
        _, _, torch = _dependencies()
        sample = self.samples[index]
        rgb = self._read_image(sample)
        if self.augment:
            key = f"{self.seed}:{self.epoch}:{sample['sample_id']}".encode()
            rgb = augment_rgb(rgb, seed=int.from_bytes(hashlib.sha256(key).digest()[:8], "big"), config=self.config)
        # Keep model input limited to RGB. Source metadata is returned only as
        # identifiers for stratified evaluation; it is not concatenated to RGB.
        return {"image": preprocess_rgb(rgb, self.config), "target": torch.tensor(sample["waypoints_ego_m"], dtype=torch.float32),
                **{key: sample[key] for key in ("sample_id", "case", "condition", "route_id")}}


def seed_worker(worker_id):
    """Seed third-party worker RNGs from the seeded Torch DataLoader generator."""
    import random
    _, np, torch = _dependencies()
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
