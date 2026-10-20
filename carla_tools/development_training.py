"""Controlled appearance robustness fine-tuning using development evidence.

The unchanged incumbent and both training arms have identical RGB geometry.
Only the candidate's additional appearance augmentation changes. All epochs,
optimizer updates, initial tensors and development views are paired by seed.
The output is a development candidate; native validation remains mandatory.
"""
from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .contracts import canonical_sha256, read_strict_json
from .dataset_learning import DatasetV1WaypointDataset
from .learning_data import (LearningDataError, augment_rgb, learning_config_from_dict,
                            preprocess_rgb, validate_waypoint_points)
from .recovery_training import (RecoveryProtocol, _binding, _contract, _groups,
    _latency, _loader, epoch_order, evaluation_contract, fixed_dev_indices,
    learning_rate_factor, model_state_sha256, recovery_protocol_from_dict,
    recovery_source_hashes)
from .refined_training import load_refined_checkpoint, refined_source_hashes
from .refined_waypoint import RefinedWaypointModel, mirror_rgb_and_points, refined_loss
from .training import (_atomic_checkpoint, _device, _epoch, _json, _new_output,
    _optimizer, _validate_pretrained_provenance, seed_everything, sha256_file)

ARMS = ("continued", "appearance_robust")
VIEWS = ("clean", "dim", "bright", "low_contrast", "blur", "noise")
DEFAULT_PROTOCOL = Path(__file__).resolve().parents[1] / "configs/development_training.json"
SOURCES = ("carla_tools/development_training.py", "scripts/train_development_waypoints.py",
           "configs/development_training.json")


@dataclass(frozen=True)
class AppearanceAugmentation:
    brightness: float = .35
    contrast: float = .25
    blur_probability: float = .35
    blur_sigma: float = 1.0
    noise_probability: float = .35
    noise_std: float = 5.0

    def __post_init__(self):
        maxima = {"brightness": .6, "contrast": .6, "blur_probability": 1.,
                  "blur_sigma": 2., "noise_probability": 1., "noise_std": 12.}
        for name, maximum in maxima.items():
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError("development-invalid-appearance-" + name)
        if self.blur_probability > 0 and self.blur_sigma <= 0:
            raise ValueError("development-zero-active-blur-sigma")

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class DevelopmentProtocol:
    schema_version: int = 1
    training: RecoveryProtocol = dataclasses.field(default_factory=lambda: RecoveryProtocol(
        epochs=12, learning_rate=.00005, encoder_learning_rate=.000005))
    appearance: AppearanceAugmentation = dataclasses.field(default_factory=AppearanceAugmentation)
    training_selection: str = "central_only"
    epoch_dev_views: tuple = ("clean", "dim", "bright", "blur", "noise")
    view_selection_policy: str = "equal-weight-fixed-development-views"

    def __post_init__(self):
        if (type(self.schema_version) is not int or self.schema_version != 1
                or not isinstance(self.training, RecoveryProtocol)
                or not isinstance(self.appearance, AppearanceAugmentation)
                or self.training_selection not in ("central_only", "central_recovery")
                or self.view_selection_policy != "equal-weight-fixed-development-views"):
            raise ValueError("development-invalid-protocol")
        if (type(self.epoch_dev_views) is not tuple or not self.epoch_dev_views
                or "clean" not in self.epoch_dev_views or len(set(self.epoch_dev_views)) != len(self.epoch_dev_views)
                or any(view not in VIEWS for view in self.epoch_dev_views)):
            raise ValueError("development-invalid-epoch-dev-views")

    def to_dict(self):
        return {"schema_version": self.schema_version, "training": self.training.to_dict(),
            "appearance": self.appearance.to_dict(), "training_selection": self.training_selection,
            "epoch_dev_views": list(self.epoch_dev_views), "view_selection_policy": self.view_selection_policy}

    @property
    def sha256(self):
        return canonical_sha256(self.to_dict())


def development_protocol_from_dict(value):
    if type(value) is not dict or set(value) != {field.name for field in fields(DevelopmentProtocol)}:
        raise ValueError("development-protocol-fields-invalid")
    copied = dict(value)
    appearance = copied["appearance"]
    if type(appearance) is not dict or set(appearance) != {field.name for field in fields(AppearanceAugmentation)}:
        raise ValueError("development-appearance-fields-invalid")
    if type(copied["epoch_dev_views"]) not in (list, tuple):
        raise ValueError("development-invalid-epoch-dev-views")
    copied.update(training=recovery_protocol_from_dict(copied["training"]),
                  appearance=AppearanceAugmentation(**appearance), epoch_dev_views=tuple(copied["epoch_dev_views"]))
    return DevelopmentProtocol(**copied)


def load_development_protocol(path=DEFAULT_PROTOCOL):
    return development_protocol_from_dict(read_strict_json(Path(path)))


def development_source_hashes():
    root = Path(__file__).resolve().parents[1]
    return dict(recovery_source_hashes(), **{name: sha256_file(root / name) for name in SOURCES})


def _image(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (360, 640, 3):
        raise LearningDataError("development-invalid-rgb-shape-or-dtype")
    return rgb


def augment_appearance(rgb, *, seed, config):
    """Deterministic appearance-only changes, without changing waypoint labels."""
    import cv2
    image = _image(rgb)
    if type(seed) is not int or seed < 0 or not isinstance(config, AppearanceAugmentation):
        raise ValueError("development-invalid-augmentation-input")
    rng = np.random.default_rng(seed)
    value = image.astype(np.float32) * rng.uniform(1 - config.brightness, 1 + config.brightness)
    value = (value - 127.5) * rng.uniform(1 - config.contrast, 1 + config.contrast) + 127.5
    if rng.random() < config.blur_probability:
        value = cv2.GaussianBlur(value, (5, 5), sigmaX=config.blur_sigma, borderType=cv2.BORDER_REFLECT_101)
    if rng.random() < config.noise_probability:
        value += rng.normal(0., config.noise_std, size=value.shape).astype(np.float32)
    return np.clip(np.rint(value), 0, 255).astype(np.uint8)


def development_view(rgb, view, *, sample_id):
    """Fixed RGB stress views are development diagnostics, not unseen weather."""
    import cv2
    image = _image(rgb)
    if view not in VIEWS or type(sample_id) is not str or not sample_id:
        raise ValueError("development-invalid-view")
    if view == "clean":
        return image.copy()
    value = image.astype(np.float32)
    if view == "dim":
        value *= .65
    elif view == "bright":
        value *= 1.35
    elif view == "low_contrast":
        value = (value - 127.5) * .65 + 127.5
    elif view == "blur":
        value = cv2.GaussianBlur(value, (5, 5), sigmaX=1., borderType=cv2.BORDER_REFLECT_101)
    else:
        seed = int.from_bytes(hashlib.sha256(("development-view:" + sample_id).encode()).digest()[:8], "big")
        value += np.random.default_rng(seed).normal(0., 5., size=value.shape).astype(np.float32)
    return np.clip(np.rint(value), 0, 255).astype(np.uint8)


class DevelopmentTrainingDataset:
    def __init__(self, source, arm, appearance):
        if source.split != "train" or source.augment or arm not in ARMS:
            raise ValueError("development-unaugmented-training-source-required")
        self.source, self.arm, self.appearance, self.epoch = source, arm, appearance, 0

    def __len__(self):
        return len(self.source)

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index):
        sample = self.source.samples[index]
        rgb, target = self.source._read_image(sample), torch.tensor(sample["waypoints_ego_m"], dtype=torch.float32)
        key = hashlib.sha256(f"{self.source.seed}:{self.epoch}:{index}".encode()).digest()
        # Existing augmentation, mirror draws and sampler order are identical
        # in both arms. The extra channel has its own RNG and cannot perturb them.
        rgb = augment_rgb(rgb, seed=int.from_bytes(key[:8], "big"), config=self.source.config)
        if self.arm == "appearance_robust":
            rgb = augment_appearance(rgb, seed=int.from_bytes(key[9:17], "big"), config=self.appearance)
        flipped = bool(key[8] & 1)
        if flipped:
            rgb, target = mirror_rgb_and_points(rgb, target)
        case = sample["case"]
        if flipped:
            case = {"left_curve": "right_curve", "right_curve": "left_curve", "straight": "straight"}[case]
        return {"image": preprocess_rgb(rgb, self.source.config), "target": target, "case": case,
                **{name: sample[name] for name in ("sample_id", "condition", "route_id")}}


class DevelopmentViewDataset:
    def __init__(self, source, views):
        if source.split != "dev" or source.augment or not views or any(view not in VIEWS for view in views):
            raise ValueError("development-unaugmented-dev-source-required")
        self.source, self.views = source, tuple(views)

    def __len__(self):
        return len(self.source) * len(self.views)

    def __getitem__(self, index):
        sample = self.source.samples[index // len(self.views)]
        view = self.views[index % len(self.views)]
        rgb = development_view(self.source._read_image(sample), view, sample_id=sample["sample_id"])
        return {"image": preprocess_rgb(rgb, self.source.config),
                "target": torch.tensor(sample["waypoints_ego_m"], dtype=torch.float32), "view": view,
                **{name: sample[name] for name in ("sample_id", "case", "condition", "route_id")}}


def _valid_sha(value):
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def load_development_checkpoint(path, *, device="cpu", expected_sha256=None, manifest_sha256=None):
    path = Path(path)
    if path.is_symlink() or expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("development-checkpoint-file-binding-invalid")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (type(value) is not dict or value.get("schema_version") != 1
            or value.get("checkpoint_role") != "selected-development-model"
            or value.get("variant") != "hybrid_geometry" or value.get("arm") not in ARMS):
        raise ValueError("development-checkpoint-role-invalid")
    protocol = development_protocol_from_dict(value["protocol"])
    config = learning_config_from_dict(value["config"])
    contract, contract_sha = _contract(config)
    if (value.get("protocol_sha256") != protocol.sha256 or value.get("config_sha256") != config.sha256
            or config.to_dict() != protocol.training.learning_config(value["seed"]).to_dict()
            or value.get("model_contract") != contract or value.get("model_contract_sha256") != contract_sha
            or value.get("source_sha256") != development_source_hashes()
            or type(value.get("epoch")) is not int or not 1 <= value["epoch"] <= protocol.training.epochs
            or value.get("model_state_sha256") != model_state_sha256(value["model_state"])):
        raise ValueError("development-checkpoint-contract-or-source-mismatch")
    if manifest_sha256 is not None and value.get("manifest_sha256") != manifest_sha256:
        raise ValueError("development-checkpoint-manifest-mismatch")
    for key in ("manifest_sha256", "validation_report_sha256", "initial_checkpoint_sha256",
                "initial_model_state_sha256", "training_selection_sha256", "epoch_dev_selection_sha256"):
        if not _valid_sha(value.get(key)):
            raise ValueError("development-checkpoint-provenance-invalid")
    _validate_pretrained_provenance(value["pretrained_provenance"])
    model = RefinedWaypointModel(variant="hybrid_geometry", config=config, pretrained=False)
    model.load_state_dict(value["model_state"], strict=True)
    model.pretrained_provenance = value["pretrained_provenance"]
    return model.to(_device(torch, device)).eval(), config, value


def export_refined_checkpoint(source_path, destination_path):
    # Strictly validate primary provenance before producing the existing runtime
    # adapter. Exported tensors must remain byte-for-byte identical.
    model, _, value = load_development_checkpoint(source_path, device="cpu")
    del model
    provenance_keys = ("protocol_sha256", "manifest_sha256", "validation_report_sha256",
                       "initial_checkpoint_sha256", "initial_model_state_sha256", "seed", "arm")
    exported = dict(value, checkpoint_role="selected-refined-model", source_sha256=refined_source_hashes(),
        development_provenance={"checkpoint_sha256": sha256_file(Path(source_path)),
            "training_source_sha256": value["source_sha256"],
            **{name: value[name] for name in provenance_keys}})
    _atomic_checkpoint(Path(destination_path), exported, torch)
    check = torch.load(destination_path, map_location="cpu", weights_only=True)
    if model_state_sha256(check["model_state"]) != value["model_state_sha256"]:
        raise ValueError("development-export-model-state-changed")
    return sha256_file(Path(destination_path))


def _train(data, dev, dev_indices, initial, protocol, seed, arm, output, device):
    output = _new_output(Path(output))
    budget, config = protocol.training, protocol.training.learning_config(seed)
    data.config = dev.config = config
    data.seed = dev.seed = seed
    source, binding = development_source_hashes(), _binding(data)
    contract, contract_sha = _contract(config)
    selected_samples = [dev.samples[index] for index in dev_indices]
    # The view wrapper consumes the same unaugmented source; Subset selects its
    # expanded sample-major indices without copying mutable dataset state.
    from torch.utils.data import Subset
    views = DevelopmentViewDataset(dev, protocol.epoch_dev_views)
    view_indices = [index * len(protocol.epoch_dev_views) + local
                    for index in dev_indices for local in range(len(protocol.epoch_dev_views))]
    report = {"schema_version": 1, "status": "FAIL", "variant": budget.variant, "arm": arm, "seed": seed,
        "protocol": protocol.to_dict(), "protocol_sha256": protocol.sha256,
        "config": config.to_dict(), "config_sha256": config.sha256, "source_sha256": source, **binding,
        "initial_checkpoint_sha256": initial["file_sha256"], "initial_model_state_sha256": initial["state_sha256"],
        "training_selection_sha256": canonical_sha256([sample["sample_id"] for sample in data.samples]),
        "epoch_dev_selection_sha256": canonical_sha256([sample["sample_id"] for sample in selected_samples]),
        "train_sample_count": len(data), "epoch_dev_sample_count": len(dev_indices),
        "epoch_dev_views": list(protocol.epoch_dev_views), "epoch_dev_view_count": len(view_indices),
        "planned_epochs": budget.epochs, "planned_updates": math.ceil(len(data) / budget.batch_size) * budget.epochs,
        "early_stopping": False, "fixture_injection": False, "final_town_used": False, "history": [],
        "augmentation": "matched-base-photometric-mirror-plus-appearance" if arm == ARMS[1] else "matched-base-photometric-mirror",
        "runtime": {"device": str(device), "torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}
    _json(output / "training_report.json", report)
    began = time.monotonic()
    try:
        seed_everything(seed, torch, np)
        model = RefinedWaypointModel(variant=budget.variant, config=config, pretrained=False)
        model.load_state_dict(initial["state"], strict=True)
        if model_state_sha256(model.state_dict()) != initial["state_sha256"]:
            raise ValueError("development-initial-state-mismatch")
        model.pretrained_provenance = initial["pretrained_provenance"]
        model.to(device)
        optimizer = _optimizer(model, config, torch)
        scaler = torch.amp.GradScaler(device.type, enabled=config.mixed_precision and device.type == "cuda")
        wrapped = DevelopmentTrainingDataset(data, arm, protocol.appearance)
        dev_loader = _loader(Subset(views, view_indices), budget, device)
        seed_everything(seed, torch, np)
        best_loss, updates = math.inf, 0
        for epoch in range(budget.epochs):
            data.assert_manifest_unchanged(); dev.assert_manifest_unchanged()
            wrapped.set_epoch(epoch)
            order, factor = epoch_order(len(data), seed, epoch), learning_rate_factor(epoch, budget)
            train_loader = _loader(wrapped, budget, device, sampler=order)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            train_loss, train_count = _epoch(model, train_loader, device, config, torch, refined_loss,
                                            optimizer=optimizer, scaler=scaler)
            dev_loss, dev_count = _epoch(model, dev_loader, device, config, torch, refined_loss)
            if train_count != len(data) or dev_count != len(view_indices):
                raise ValueError("development-sample-budget-mismatch")
            updates += math.ceil(train_count / budget.batch_size)
            selected = dev_loss < best_loss
            item = {"epoch": epoch + 1, "train_loss": train_loss, "dev_loss": dev_loss,
                "train_samples": train_count, "dev_samples": dev_count, "optimizer_updates": updates,
                "sampler_order_sha256": canonical_sha256(order), "learning_rate_factor": factor, "selected": selected}
            report["history"].append(item)
            if selected:
                best_loss = dev_loss
                state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                checkpoint = {"schema_version": 1, "checkpoint_role": "selected-development-model",
                    "variant": budget.variant, "arm": arm, "seed": seed, "epoch": epoch + 1,
                    "selected_dev_loss": best_loss, "config": config.to_dict(), "config_sha256": config.sha256,
                    "model_contract": contract, "model_contract_sha256": contract_sha,
                    "protocol": protocol.to_dict(), "protocol_sha256": protocol.sha256, "source_sha256": source,
                    **binding, "initial_checkpoint_sha256": initial["file_sha256"],
                    "initial_model_state_sha256": initial["state_sha256"],
                    "training_selection_sha256": report["training_selection_sha256"],
                    "epoch_dev_selection_sha256": report["epoch_dev_selection_sha256"],
                    "pretrained_provenance": initial["pretrained_provenance"],
                    "model_state_sha256": model_state_sha256(state), "model_state": state}
                _atomic_checkpoint(output / "checkpoint.pt", checkpoint, torch)
                report["selected_epoch"] = epoch + 1
            report.update(completed_epochs=epoch + 1, optimizer_updates=updates, best_dev_loss=best_loss)
            _json(output / "training_report.json", report)
            print(json.dumps({"phase": "development-training", "arm": arm, "seed": seed, **item}), flush=True)
        data.assert_manifest_unchanged(); dev.assert_manifest_unchanged()
        if source != development_source_hashes() or updates != report["planned_updates"]:
            raise ValueError("development-source-or-budget-changed")
        report.update(status="PASS", checkpoint_sha256=sha256_file(output / "checkpoint.pt"),
                      parameter_count=sum(parameter.numel() for parameter in model.parameters()))
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": str(error)}
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - began
        _json(output / "training_report.json", report)
    del model, optimizer, scaler, train_loader, dev_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def evaluate_development_model(data, checkpoint_path, output_dir, *, protocol, device="cpu", initial=False, view="clean"):
    output, device = _new_output(Path(output_dir)), _device(torch, device)
    model, config, checkpoint = (load_refined_checkpoint if initial else load_development_checkpoint)(checkpoint_path, device=str(device))
    data.config = config
    data.assert_manifest_unchanged()
    wrapped = DevelopmentViewDataset(data, (view,))
    source, records, offset = development_source_hashes(), [], 0
    began = time.monotonic()
    try:
        with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
            for batch in _loader(wrapped, protocol.training, device):
                with torch.inference_mode(), torch.autocast(device_type=device.type,
                        enabled=config.mixed_precision and device.type == "cuda"):
                    predictions = model(batch["image"].to(device)).float().cpu()
                if tuple(predictions.shape) != (len(batch["image"]), 5, 2):
                    raise ValueError("development-evaluation-output-shape")
                for local, points in enumerate(predictions.tolist()):
                    sample, target = data.samples[offset], batch["target"][local].tolist()
                    if batch["sample_id"][local] != sample["sample_id"]:
                        raise ValueError("development-evaluation-order")
                    error = None
                    try:
                        validate_waypoint_points(points)
                    except LearningDataError:
                        points, error = None, "invalid-model-output-order-or-range"
                    row = {name: sample[name] for name in ("sample_id", "case", "condition", "weather", "route_id", "image_sha256")}
                    row.update(view=view, target=target, prediction=points, error=error)
                    if points is not None:
                        row.update(errors_m=[math.dist(p, t) for p, t in zip(points, target)],
                            x_errors_m=[abs(p[0] - t[0]) for p, t in zip(points, target)],
                            y_errors_m=[abs(p[1] - t[1]) for p, t in zip(points, target)])
                    records.append(row); stream.write(json.dumps(row, allow_nan=False) + "\n")
                    offset += 1
        data.assert_manifest_unchanged()
        if offset != len(data) or source != development_source_hashes():
            raise ValueError("development-evaluation-source-or-count-changed")
        metrics = _groups(records)
        report = {"schema_version": 1, "status": "PASS", "variant": "hybrid_geometry",
            "arm": "incumbent" if initial else checkpoint["arm"], "seed": checkpoint.get("seed"), "split": "dev",
            "view": view, "view_scope": "fixed-appearance-diagnostic-not-independent-weather", "sample_count": len(data),
            "metrics": metrics, "common_dev_selection_sha256": canonical_sha256([sample["sample_id"] for sample in data.samples]),
            **_binding(data), "checkpoint_sha256": sha256_file(Path(checkpoint_path)), "config_sha256": config.sha256,
            "protocol_sha256": protocol.sha256, "source_sha256": source, "fixture_injection": False, "final_town_used": False,
            "evaluation_contract": evaluation_contract(config), "evaluation_contract_sha256": canonical_sha256(evaluation_contract(config)),
            "case_sample_counts": {case: sum(sample["case"] == case for sample in data.samples)
                                   for case in sorted({sample["case"] for sample in data.samples})},
            "performance_target_passed": metrics[0]["coverage"] == 1 and metrics[0]["near_mean_m"] <= config.near_mean_target_m
                and metrics[0]["far_mean_m"] <= config.far_mean_target_m,
            "latency_ms": _latency(model, data, protocol.training, config, device),
            "runtime": {"device": str(device), "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
            "predictions_sha256": sha256_file(output / "predictions.jsonl"), "elapsed_seconds": time.monotonic() - began}
        _json(output / "metrics.json", report)
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return report


def representative_seed(training, protocol):
    values = []
    for seed in protocol.training.seeds:
        scores = [training[str(seed)][arm]["best_dev_loss"] for arm in ARMS]
        if any(type(score) not in (int, float) or not math.isfinite(score) for score in scores):
            raise ValueError("development-invalid-representative-loss")
        values.append((sum(scores) / len(scores), seed))
    return min(values)[1]


def verify_equal_budget(training, protocol):
    for seed in protocol.training.seeds:
        first, second = (training[str(seed)][arm] for arm in ARMS)
        for report in (first, second):
            if (report["status"] != "PASS" or report["completed_epochs"] != protocol.training.epochs
                    or report["optimizer_updates"] != report["planned_updates"]
                    or report["train_sample_count"] != protocol.training.train_samples_per_arm
                    or len(report["history"]) != protocol.training.epochs):
                raise ValueError("development-incomplete-training-budget")
        for name in ("initial_checkpoint_sha256", "initial_model_state_sha256", "training_selection_sha256",
                     "epoch_dev_selection_sha256", "epoch_dev_views", "config_sha256", "protocol_sha256",
                     "manifest_sha256", "validation_report_sha256", "optimizer_updates"):
            if first[name] != second[name]:
                raise ValueError("development-paired-arm-mismatch-" + name)
        for a, b in zip(first["history"], second["history"]):
            for name in ("epoch", "train_samples", "dev_samples", "optimizer_updates", "sampler_order_sha256", "learning_rate_factor"):
                if a[name] != b[name]:
                    raise ValueError("development-paired-update-schedule-mismatch")


def run_development_comparison(manifest_path, validation_report_path, initial_checkpoint_path,
                               output_dir, *, protocol=None, device="cpu"):
    protocol = protocol or load_development_protocol()
    output, device = _new_output(Path(output_dir)), _device(torch, device)
    source = development_source_hashes()
    report = {"schema_version": 1, "status": "FAIL", "protocol": protocol.to_dict(), "protocol_sha256": protocol.sha256,
        "source_sha256": source, "fixture_injection": False, "final_town_used": False, "training": {},
        "evaluations": {}, "closed_loop_verified": False, "incumbent_pointer_updated": False}
    _json(output / "comparison.json", report)
    began = time.monotonic()
    try:
        config = protocol.training.learning_config(protocol.training.seeds[0])
        train = DatasetV1WaypointDataset(manifest_path, "train", selection=protocol.training_selection,
            validation_report_path=validation_report_path, config=config, augment=False)
        dev = DatasetV1WaypointDataset(manifest_path, "dev", selection=protocol.training_selection,
            validation_report_path=validation_report_path, config=config, augment=False)
        if len(train) != protocol.training.train_samples_per_arm:
            raise ValueError("development-fixed-training-sample-budget-mismatch")
        dev_indices = fixed_dev_indices(dev.samples, protocol.training.epoch_dev_samples)
        path = Path(initial_checkpoint_path)
        if path.is_symlink():
            raise ValueError("development-initial-checkpoint-symlink")
        initial_sha = sha256_file(path)
        model, _, initial_checkpoint = load_refined_checkpoint(path, device="cpu", expected_sha256=initial_sha)
        if initial_checkpoint["variant"] != "hybrid_geometry":
            raise ValueError("development-initial-architecture-mismatch")
        initial = {"file_sha256": initial_sha, "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                   "pretrained_provenance": initial_checkpoint["pretrained_provenance"]}
        initial["state_sha256"] = model_state_sha256(initial["state"])
        del model
        initial_output = output / "initial"; initial_output.mkdir()
        shutil.copyfile(path, initial_output / "checkpoint.pt")
        if sha256_file(initial_output / "checkpoint.pt") != initial_sha:
            raise ValueError("development-initial-copy-mismatch")
        report.update(**_binding(dev), initial_checkpoint_sha256=initial_sha,
            initial_model_state_sha256=initial["state_sha256"],
            common_dev_sample_count=len(dev), common_dev_selection_sha256=canonical_sha256([sample["sample_id"] for sample in dev.samples]),
            epoch_dev_sample_ids=[dev.samples[index]["sample_id"] for index in dev_indices],
            epoch_dev_selection_sha256=canonical_sha256([dev.samples[index]["sample_id"] for index in dev_indices]),
            selection_scope="fixed-development-subset-and-fixed-appearance-views-before-full-dev-and-native")
        _json(output / "protocol.json", dict(report))
        for seed in protocol.training.seeds:
            report["training"][str(seed)] = {}
            for arm in ARMS:
                destination = output / f"seed{seed}" / arm
                trained = _train(train, dev, dev_indices, initial, protocol, seed, arm, destination, device)
                trained["compatible_checkpoint_sha256"] = export_refined_checkpoint(destination / "checkpoint.pt", destination / "refined_checkpoint.pt")
                _json(destination / "training_report.json", trained)
                report["training"][str(seed)][arm] = trained
                _json(output / "comparison.json", report)
        verify_equal_budget(report["training"], protocol)
        chosen = representative_seed(report["training"], protocol)
        report["representative_seed"] = chosen
        _json(output / "native_pair_selection.json", {"status": "PASS", "representative_seed": chosen,
            "policy": protocol.training.representative_seed_policy, "protocol_sha256": protocol.sha256,
            "epoch_dev_selection_sha256": report["epoch_dev_selection_sha256"], "full_dev_evaluations_completed": False,
            "training_report_sha256": {f"{seed}/{arm}": sha256_file(output / f"seed{seed}" / arm / "training_report.json")
                for seed in protocol.training.seeds for arm in ARMS}})
        report["initial_evaluation"] = evaluate_development_model(dev, initial_output / "checkpoint.pt",
            output / "initial_evaluation", protocol=protocol, device=str(device), initial=True)
        for seed in protocol.training.seeds:
            report["evaluations"][str(seed)] = {}
            for arm in ARMS:
                report["evaluations"][str(seed)][arm] = evaluate_development_model(dev, output / f"seed{seed}" / arm / "checkpoint.pt",
                    output / f"seed{seed}" / arm / "evaluation", protocol=protocol, device=str(device))
                _json(output / "comparison.json", report)
        for arm in ARMS:
            target = output / arm; target.mkdir()
            for name in ("checkpoint.pt", "refined_checkpoint.pt"):
                shutil.copyfile(output / f"seed{chosen}" / arm / name, target / name)
            report.setdefault("representative_checkpoint_sha256", {})[arm] = sha256_file(target / "checkpoint.pt")
            report.setdefault("compatible_checkpoint_sha256", {})[arm] = sha256_file(target / "refined_checkpoint.pt")
        train.assert_manifest_unchanged(); dev.assert_manifest_unchanged()
        if source != development_source_hashes() or sha256_file(path) != initial_sha:
            raise ValueError("development-final-source-or-initial-changed")
        report.update(status="PASS", equal_budget_verified=True,
                      native_pair_selection_sha256=sha256_file(output / "native_pair_selection.json"))
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": str(error)}
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - began
        _json(output / "comparison.json", report)
    return report
