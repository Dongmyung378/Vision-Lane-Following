"""Equal-budget RGB-only central/recovery fine-tuning and common DEV scoring.

Both arms restart from the same complete model state. A fixed stratified DEV
subset selects epochs; full common DEV scores and native driving remain
separate evidence. Reserved towns never enter any loader or selection.
"""
from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import math
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .contracts import canonical_sha256, read_strict_json
from .dataset_learning import DatasetV1WaypointDataset
from .dataset_v1 import _implementation_hashes
from .learning_data import (LearningDataError, augment_rgb, learning_config_from_dict,
                           load_learning_config, preprocess_rgb, validate_waypoint_points)
from .refined_training import load_refined_checkpoint, refined_source_hashes
from .refined_waypoint import (RefinedWaypointModel, mirror_rgb_and_points,
                               refined_contract, refined_loss)
from .training import (_atomic_checkpoint, _device, _epoch, _json, _metric_group,
                       _new_output, _optimizer, _validate_pretrained_provenance,
                       seed_everything, sha256_file)

ARMS = ("central_only", "central_recovery")
DEFAULT_PROTOCOL = Path(__file__).resolve().parents[1] / "configs/recovery_training.json"
SOURCES = ("carla_tools/recovery_training.py", "scripts/train_recovery_waypoints.py",
           "configs/recovery_training.json")


@dataclass(frozen=True)
class RecoveryProtocol:
    schema_version: int = 1
    variant: str = "hybrid_geometry"
    seeds: tuple = (42, 43)
    train_samples_per_arm: int = 5000
    epochs: int = 32
    batch_size: int = 64
    evaluation_batch_size: int = 64
    num_workers: int = 2
    epoch_dev_samples: int = 2048
    learning_rate: float = 0.0002
    encoder_learning_rate: float = 0.00002
    weight_decay: float = 0.001
    warmup_epochs: int = 2
    minimum_lr_factor: float = 0.05
    mixed_precision: bool = True
    latency_warmup: int = 16
    latency_samples: int = 128
    representative_seed_policy: str = "minimum-paired-epoch-dev-loss"

    def __post_init__(self):
        if (type(self.schema_version) is not int or self.schema_version != 1
                or self.variant != "hybrid_geometry"
                or self.representative_seed_policy != "minimum-paired-epoch-dev-loss"):
            raise ValueError("recovery-invalid-protocol-version-or-policy")
        if (type(self.seeds) is not tuple or not self.seeds or len(self.seeds) > 8
                or len(set(self.seeds)) != len(self.seeds)
                or any(type(seed) is not int or seed < 0 for seed in self.seeds)):
            raise ValueError("recovery-invalid-seeds")
        for name in ("train_samples_per_arm", "epochs", "batch_size", "evaluation_batch_size",
                     "num_workers", "epoch_dev_samples", "warmup_epochs", "latency_warmup", "latency_samples"):
            minimum = 0 if name in ("num_workers", "warmup_epochs") else 1
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError("recovery-invalid-" + name)
        if self.epochs > 128 or self.warmup_epochs >= self.epochs and self.epochs != 1:
            raise ValueError("recovery-invalid-epoch-budget")
        for name in ("learning_rate", "encoder_learning_rate", "weight_decay", "minimum_lr_factor"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("recovery-invalid-" + name)
        if not 0 < self.minimum_lr_factor <= 1 or type(self.mixed_precision) is not bool:
            raise ValueError("recovery-invalid-schedule-or-amp")

    def to_dict(self):
        value = dataclasses.asdict(self)
        value["seeds"] = list(self.seeds)
        return value

    @property
    def sha256(self):
        return canonical_sha256(self.to_dict())

    def learning_config(self, seed):
        if seed not in self.seeds:
            raise ValueError("recovery-seed-outside-protocol")
        return dataclasses.replace(load_learning_config(), seed=seed, epochs=self.epochs,
            batch_size=self.batch_size, num_workers=self.num_workers,
            learning_rate=self.learning_rate, encoder_learning_rate=self.encoder_learning_rate,
            weight_decay=self.weight_decay, warmup_epochs=self.warmup_epochs,
            mixed_precision=self.mixed_precision)


def recovery_protocol_from_dict(value):
    if type(value) is not dict or set(value) != {field.name for field in fields(RecoveryProtocol)}:
        raise ValueError("recovery-protocol-fields-invalid")
    value = dict(value)
    if type(value["seeds"]) not in (list, tuple):
        raise ValueError("recovery-invalid-seeds")
    value["seeds"] = tuple(value["seeds"])
    return RecoveryProtocol(**value)


def load_recovery_protocol(path=DEFAULT_PROTOCOL):
    return recovery_protocol_from_dict(read_strict_json(Path(path)))


def recovery_source_hashes():
    root = Path(__file__).resolve().parents[1]
    return dict(refined_source_hashes(), **_implementation_hashes(),
                **{name: sha256_file(root / name) for name in SOURCES})


def model_state_sha256(state):
    """Hash named tensor values independently of Torch archive serialization."""
    digest = hashlib.sha256()
    if not isinstance(state, dict) or not state:
        raise ValueError("recovery-empty-initial-state")
    for name in sorted(state):
        value = state[name]
        if type(name) is not str or not isinstance(value, torch.Tensor):
            raise ValueError("recovery-state-tensor-required")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("recovery-nonfinite-initial-state")
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        payload = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big")); digest.update(header)
        digest.update(len(payload).to_bytes(8, "big")); digest.update(payload)
    return digest.hexdigest()


def fixed_dev_indices(samples, count, seed=0):
    """Round-robin case/weather/condition strata, without looking at predictions."""
    if type(count) is not int or not 0 < count <= len(samples):
        raise ValueError("recovery-insufficient-common-dev")
    groups = defaultdict(list)
    seen = set()
    for index, sample in enumerate(samples):
        identifier = sample["sample_id"]
        if identifier in seen or sample["split"] != "dev":
            raise ValueError("recovery-invalid-common-dev-membership")
        seen.add(identifier)
        key = (sample["case"], sample["weather"], sample["condition"])
        rank = hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()
        groups[key].append((rank, identifier, index))
    for values in groups.values():
        values.sort()
    keys = sorted(groups)
    result, cursor = [], 0
    while len(result) < count:
        for key in keys:
            if cursor < len(groups[key]):
                result.append(groups[key][cursor][2])
                if len(result) == count:
                    break
        cursor += 1
    return result


def epoch_order(count, seed, epoch):
    if any(type(value) is not int for value in (count, seed, epoch)) or count < 1 or min(seed, epoch) < 0:
        raise ValueError("recovery-invalid-order-input")
    return torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch)).tolist()


def learning_rate_factor(epoch, protocol):
    if not 0 <= epoch < protocol.epochs:
        raise ValueError("recovery-epoch-outside-budget")
    if epoch < protocol.warmup_epochs:
        return (epoch + 1) / max(1, protocol.warmup_epochs)
    progress = (epoch - protocol.warmup_epochs) / max(1, protocol.epochs - protocol.warmup_epochs - 1)
    return protocol.minimum_lr_factor + (1 - protocol.minimum_lr_factor) * (1 + math.cos(math.pi * progress)) / 2


class MirroredRecoveryDataset:
    """Use identical augmentation draws for each matched arm index and epoch."""

    def __init__(self, source):
        if source.split != "train" or source.augment:
            raise ValueError("recovery-wrapper-requires-unaugmented-train")
        self.source, self.epoch = source, 0

    def __len__(self):
        return len(self.source)

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index):
        sample = self.source.samples[index]
        rgb = self.source._read_image(sample)
        target = torch.tensor(sample["waypoints_ego_m"], dtype=torch.float32)
        key = hashlib.sha256(f"{self.source.seed}:{self.epoch}:{index}".encode()).digest()
        rgb = augment_rgb(rgb, seed=int.from_bytes(key[:8], "big"), config=self.source.config)
        flipped = bool(key[8] & 1)
        if flipped:
            rgb, target = mirror_rgb_and_points(rgb, target)
        case = sample["case"]
        if flipped:
            case = {"left_curve": "right_curve", "right_curve": "left_curve", "straight": "straight"}[case]
        return {"image": preprocess_rgb(rgb, self.source.config), "target": target, "case": case,
                **{name: sample[name] for name in ("sample_id", "condition", "route_id")}}


def _loader(dataset, protocol, device, *, sampler=None):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=protocol.evaluation_batch_size if sampler is None else protocol.batch_size,
        sampler=sampler, shuffle=False, num_workers=protocol.num_workers if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda", drop_last=False)


def _contract(config):
    value = refined_contract("hybrid_geometry", config)
    return value, canonical_sha256(value)


def _binding(data):
    return {"manifest_sha256": data.manifest_sha256,
            "validation_report_sha256": data.validation_report_sha256,
            "snapshot_sha256": data.manifest["snapshot_sha256"],
            "dataset_config_sha256": data.manifest["config_sha256"]}


def evaluation_contract(config):
    """Describe inference representation without training seed or optimizer."""
    from .learning_data import RGB_MEAN, RGB_STD, ROAD_ROI_TOP_PX
    return {"input": "RGB-only", "source_shape": [360, 640, 3],
        "roi_top_px": ROAD_ROI_TOP_PX, "tensor_shape": [3, config.image_height, config.image_width],
        "resize": "opencv-INTER_AREA", "mean": list(RGB_MEAN), "std": list(RGB_STD),
        "target": "vehicle-actor-origin-meters", "horizons_m": list(config.waypoint_distances_m),
        "model_contract": refined_contract("hybrid_geometry", config), "augmentation": False}


def load_recovery_checkpoint(path, *, device="cpu", expected_sha256=None, manifest_sha256=None):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("recovery-checkpoint-symlink")
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("recovery-checkpoint-sha256-mismatch")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (type(value) is not dict or value.get("schema_version") != 1
            or value.get("checkpoint_role") != "selected-recovery-model"
            or value.get("variant") != "hybrid_geometry" or value.get("arm") not in ARMS):
        raise ValueError("recovery-checkpoint-role-or-variant-invalid")
    protocol = recovery_protocol_from_dict(value["protocol"])
    config = learning_config_from_dict(value["config"])
    contract, contract_hash = _contract(config)
    if (value.get("protocol_sha256") != protocol.sha256 or value.get("config_sha256") != config.sha256
            or config.to_dict() != protocol.learning_config(value["seed"]).to_dict()
            or value.get("model_contract") != contract or value.get("model_contract_sha256") != contract_hash
            or value.get("source_sha256") != recovery_source_hashes()
            or not 1 <= value.get("epoch", 0) <= protocol.epochs
            or value.get("model_state_sha256") != model_state_sha256(value["model_state"])):
        raise ValueError("recovery-checkpoint-contract-or-source-mismatch")
    if manifest_sha256 is not None and value.get("manifest_sha256") != manifest_sha256:
        raise ValueError("recovery-checkpoint-manifest-mismatch")
    for key in ("manifest_sha256", "validation_report_sha256", "initial_checkpoint_sha256",
                "initial_model_state_sha256", "training_selection_sha256", "epoch_dev_selection_sha256"):
        if (type(value.get(key)) is not str or len(value[key]) != 64
                or any(character not in "0123456789abcdef" for character in value[key])):
            raise ValueError("recovery-checkpoint-provenance-invalid")
    _validate_pretrained_provenance(value["pretrained_provenance"])
    model = RefinedWaypointModel(variant="hybrid_geometry", config=config, pretrained=False)
    model.load_state_dict(value["model_state"], strict=True)
    model.pretrained_provenance = value["pretrained_provenance"]
    return model.to(_device(torch, device)).eval(), config, value


def _export_refined_checkpoint(source_path, destination_path):
    """Preserve complete recovery provenance for the legacy runtime adapter."""
    value = torch.load(source_path, map_location="cpu", weights_only=True)
    exported = dict(value, checkpoint_role="selected-refined-model", source_sha256=refined_source_hashes(),
        recovery_provenance={"checkpoint_sha256": sha256_file(source_path),
            "training_source_sha256": value["source_sha256"], "protocol_sha256": value["protocol_sha256"],
            "manifest_sha256": value["manifest_sha256"], "validation_report_sha256": value["validation_report_sha256"],
            "initial_checkpoint_sha256": value["initial_checkpoint_sha256"], "seed": value["seed"], "arm": value["arm"]})
    _atomic_checkpoint(destination_path, exported, torch)
    return sha256_file(destination_path)


def _train_arm(data, dev, dev_indices, initial, protocol, seed, output, device):
    output = _new_output(output)
    config = protocol.learning_config(seed)
    data.config, dev.config = config, config
    data.seed, dev.seed = seed, seed
    source = recovery_source_hashes()
    contract, contract_hash = _contract(config)
    train_ids = [sample["sample_id"] for sample in data.samples]
    dev_ids = [dev.samples[index]["sample_id"] for index in dev_indices]
    binding = _binding(data)
    report = {"schema_version": 1, "status": "FAIL", "arm": data.selection, "seed": seed,
        "variant": protocol.variant, **binding, "source_sha256": source,
        "protocol": protocol.to_dict(), "protocol_sha256": protocol.sha256,
        "config": config.to_dict(), "config_sha256": config.sha256,
        "initial_checkpoint_sha256": initial["file_sha256"],
        "initial_model_state_sha256": initial["state_sha256"],
        "training_selection_sha256": canonical_sha256(train_ids),
        "epoch_dev_selection_sha256": canonical_sha256(dev_ids),
        "train_sample_count": len(data), "epoch_dev_sample_count": len(dev_indices),
        "planned_epochs": protocol.epochs, "planned_updates": math.ceil(len(data) / protocol.batch_size) * protocol.epochs,
        "early_stopping": False, "augmentation": "matched-index-epoch-photometric-and-half-mirror",
        "fixture_injection": False, "final_town_used": False, "history": []}
    report["runtime"] = {"device": str(device), "torch": str(torch.__version__), "cuda": torch.version.cuda,
                         "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    _json(output / "training_report.json", report)
    began = time.monotonic()
    try:
        seed_everything(seed, torch, np)
        model = RefinedWaypointModel(variant=protocol.variant, config=config, pretrained=False)
        model.load_state_dict(initial["state"], strict=True)
        if model_state_sha256(model.state_dict()) != initial["state_sha256"]:
            raise ValueError("recovery-arm-initialization-mismatch")
        model.pretrained_provenance = initial["pretrained_provenance"]
        model.to(device)
        optimizer = _optimizer(model, config, torch)
        scaler = torch.amp.GradScaler(device.type, enabled=config.mixed_precision and device.type == "cuda")
        wrapped = MirroredRecoveryDataset(data)
        from torch.utils.data import Subset
        dev_loader = _loader(Subset(dev, dev_indices), protocol, device)
        best_loss, updates = math.inf, 0
        # Model construction consumes randomness. Reset before actual updates so
        # dropout, sampler order and augmentation schedules match between arms.
        seed_everything(seed, torch, np)
        for epoch in range(protocol.epochs):
            data.assert_manifest_unchanged(); dev.assert_manifest_unchanged()
            wrapped.set_epoch(epoch)
            order = epoch_order(len(data), seed, epoch)
            train_loader = _loader(wrapped, protocol, device, sampler=order)
            factor = learning_rate_factor(epoch, protocol)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            train_loss, train_count = _epoch(model, train_loader, device, config, torch, refined_loss,
                                            optimizer=optimizer, scaler=scaler)
            dev_loss, dev_count = _epoch(model, dev_loader, device, config, torch, refined_loss)
            if train_count != len(data) or dev_count != len(dev_indices):
                raise ValueError("recovery-epoch-sample-budget-mismatch")
            updates += math.ceil(train_count / protocol.batch_size)
            selected = dev_loss < best_loss
            item = {"epoch": epoch + 1, "train_loss": train_loss, "dev_loss": dev_loss,
                    "train_samples": train_count, "dev_samples": dev_count,
                    "optimizer_updates": updates, "sampler_order_sha256": canonical_sha256(order),
                    "learning_rate_factor": factor, "selected": selected}
            report["history"].append(item)
            if selected:
                best_loss = dev_loss
                state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                checkpoint = {"schema_version": 1, "checkpoint_role": "selected-recovery-model",
                    "variant": protocol.variant, "arm": data.selection, "seed": seed,
                    "epoch": epoch + 1, "selected_dev_loss": best_loss,
                    "config": config.to_dict(), "config_sha256": config.sha256,
                    "model_contract": contract, "model_contract_sha256": contract_hash,
                    "protocol": protocol.to_dict(), "protocol_sha256": protocol.sha256,
                    "source_sha256": source, **binding,
                    "initial_checkpoint_sha256": initial["file_sha256"],
                    "initial_model_state_sha256": initial["state_sha256"],
                    "training_selection_sha256": report["training_selection_sha256"],
                    "epoch_dev_selection_sha256": report["epoch_dev_selection_sha256"],
                    "pretrained_provenance": initial["pretrained_provenance"],
                    "model_state_sha256": model_state_sha256(state), "model_state": state}
                _atomic_checkpoint(output / "checkpoint.pt", checkpoint, torch)
                report["selected_epoch"] = epoch + 1
            report.update(completed_epochs=epoch + 1, optimizer_updates=updates, best_dev_loss=best_loss)
            _json(output / "training_report.json", report)
            print(json.dumps({"phase": "training", "arm": data.selection, "seed": seed, **item}), flush=True)
        if source != recovery_source_hashes():
            raise ValueError("recovery-runtime-source-changed")
        data.assert_manifest_unchanged(); dev.assert_manifest_unchanged()
        if updates != report["planned_updates"]:
            raise ValueError("recovery-update-budget-mismatch")
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


def _groups(records):
    result = [{"case": "all", "condition": "all", "weather": "all", **_metric_group(records)}]
    for dimensions in (("case",), ("condition",), ("weather",), ("case", "condition"), ("case", "weather")):
        groups = defaultdict(list)
        for record in records:
            groups[tuple(record[name] for name in dimensions)].append(record)
        for key, members in sorted(groups.items()):
            labels = dict(case="all", condition="all", weather="all")
            labels.update(zip(dimensions, key))
            result.append(dict(labels, **_metric_group(members)))
    return result


def _latency(model, data, protocol, config, device):
    indices = fixed_dev_indices(data.samples, min(protocol.latency_samples, len(data)), seed=2718)
    images = [data[index]["image"][None] for index in indices]
    def forward(image):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        began = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type=device.type,
                enabled=config.mixed_precision and device.type == "cuda"):
            model(image.to(device)).float().cpu()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return (time.perf_counter() - began) * 1000
    for index in range(protocol.latency_warmup):
        forward(images[index % len(images)])
    values = [forward(image) for image in images]
    return {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)),
            "max": max(values), "sample_count": len(values), "warmup_count": protocol.latency_warmup,
            "sample_ids_sha256": canonical_sha256([data.samples[index]["sample_id"] for index in indices]),
            "scope": "batch-one-forward-and-transfer-excludes-decode-preprocess-control"}


def evaluate_recovery_model(data, checkpoint_path, output_dir, *, protocol, device="cpu", initial=False):
    output = _new_output(Path(output_dir))
    device = _device(torch, device)
    loader = load_refined_checkpoint if initial else load_recovery_checkpoint
    model, config, checkpoint = loader(checkpoint_path, device=device)
    data.config = config
    data.assert_manifest_unchanged()
    source = recovery_source_hashes()
    began, records = time.monotonic(), []
    try:
        offset = 0
        with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
            for batch in _loader(data, protocol, device):
                with torch.inference_mode(), torch.autocast(device_type=device.type,
                        enabled=config.mixed_precision and device.type == "cuda"):
                    predictions = model(batch["image"].to(device)).float().cpu()
                if tuple(predictions.shape) != (len(batch["image"]), 5, 2):
                    raise ValueError("recovery-evaluation-output-shape")
                targets = batch["target"].tolist()
                for local, points in enumerate(predictions.tolist()):
                    sample, target = data.samples[offset], targets[local]
                    if batch["sample_id"][local] != sample["sample_id"]:
                        raise ValueError("recovery-evaluation-order-mismatch")
                    error = None
                    try:
                        validate_waypoint_points(points)
                    except LearningDataError:
                        points, error = None, "invalid-model-output-order-or-range"
                    row = {name: sample[name] for name in ("sample_id", "case", "condition", "weather", "route_id", "image_sha256")}
                    row.update(target=target, prediction=points, error=error)
                    if points is not None:
                        row.update(errors_m=[math.dist(p, t) for p, t in zip(points, target)],
                            x_errors_m=[abs(p[0] - t[0]) for p, t in zip(points, target)],
                            y_errors_m=[abs(p[1] - t[1]) for p, t in zip(points, target)])
                    records.append(row); stream.write(json.dumps(row, allow_nan=False) + "\n")
                    offset += 1
        if offset != len(data) or source != recovery_source_hashes():
            raise ValueError("recovery-evaluation-count-or-source-changed")
        data.assert_manifest_unchanged()
        metrics = _groups(records)
        report = {"schema_version": 1, "status": "PASS", "variant": "hybrid_geometry",
            "arm": "initial" if initial else checkpoint["arm"], "seed": checkpoint.get("seed"),
            "split": "dev", "sample_count": len(data), "metrics": metrics,
            "common_dev_selection_sha256": canonical_sha256([sample["sample_id"] for sample in data.samples]),
            **_binding(data), "checkpoint_sha256": sha256_file(Path(checkpoint_path)),
            "config_sha256": config.sha256, "protocol_sha256": protocol.sha256,
            "source_sha256": source, "fixture_injection": False, "final_town_used": False,
            "evaluation_contract": evaluation_contract(config),
            "evaluation_contract_sha256": canonical_sha256(evaluation_contract(config)),
            "case_sample_counts": {case: sum(sample["case"] == case for sample in data.samples)
                                   for case in sorted({sample["case"] for sample in data.samples})},
            "performance_target_passed": metrics[0]["coverage"] == 1
                and metrics[0]["near_mean_m"] <= config.near_mean_target_m
                and metrics[0]["far_mean_m"] <= config.far_mean_target_m,
            "latency_ms": _latency(model, data, protocol, config, device),
            "runtime": {"device": str(device), "torch": str(torch.__version__), "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
            "predictions_sha256": sha256_file(output / "predictions.jsonl"),
            "elapsed_seconds": time.monotonic() - began}
        _json(output / "metrics.json", report)
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return report


def representative_seed(training, protocol):
    scores = []
    for seed in protocol.seeds:
        arms = training[str(seed)]
        losses = [arms[arm]["best_dev_loss"] for arm in ARMS]
        if any(type(loss) not in (int, float) or not math.isfinite(loss) for loss in losses):
            raise ValueError("recovery-invalid-representative-loss")
        scores.append((sum(losses) / len(losses), seed))
    return min(scores)[1]


def _equal_budget(training, protocol):
    for seed in protocol.seeds:
        first, second = (training[str(seed)][arm] for arm in ARMS)
        for report in (first, second):
            if (report["status"] != "PASS" or report["completed_epochs"] != protocol.epochs
                    or report["optimizer_updates"] != report["planned_updates"]
                    or report["train_sample_count"] != protocol.train_samples_per_arm
                    or len(report["history"]) != protocol.epochs):
                raise ValueError("recovery-incomplete-equal-budget-arm")
        for name in ("initial_checkpoint_sha256", "initial_model_state_sha256", "epoch_dev_selection_sha256",
                     "config_sha256", "protocol_sha256", "manifest_sha256", "validation_report_sha256",
                     "optimizer_updates", "planned_epochs"):
            if first[name] != second[name]:
                raise ValueError("recovery-arm-protocol-mismatch-" + name)
        for a, b in zip(first["history"], second["history"]):
            for name in ("epoch", "train_samples", "dev_samples", "optimizer_updates",
                         "sampler_order_sha256", "learning_rate_factor"):
                if a[name] != b[name]:
                    raise ValueError("recovery-arm-update-schedule-mismatch")


def run_recovery_comparison(manifest_path, validation_report_path, initial_checkpoint_path,
                            output_dir, *, protocol=None, device="cpu"):
    protocol = protocol or load_recovery_protocol()
    device = _device(torch, device)
    output = _new_output(Path(output_dir))
    source = recovery_source_hashes()
    report = {"schema_version": 1, "status": "FAIL", "protocol": protocol.to_dict(),
        "protocol_sha256": protocol.sha256, "source_sha256": source,
        "fixture_injection": False, "final_town_used": False,
        "training": {}, "evaluations": {}, "closed_loop_verified": False}
    _json(output / "comparison.json", report)
    began = time.monotonic()
    try:
        first_config = protocol.learning_config(protocol.seeds[0])
        datasets = {arm: DatasetV1WaypointDataset(manifest_path, "train", selection=arm,
                        validation_report_path=validation_report_path, config=first_config, augment=False)
                    for arm in ARMS}
        dev = DatasetV1WaypointDataset(manifest_path, "dev", selection=ARMS[0],
                    validation_report_path=validation_report_path, config=first_config, augment=False)
        if any(len(data) != protocol.train_samples_per_arm for data in datasets.values()):
            raise ValueError("recovery-fixed-train-budget-mismatch")
        if dev.manifest["selections"][ARMS[0]]["dev_sample_ids"] != dev.manifest["selections"][ARMS[1]]["dev_sample_ids"]:
            raise ValueError("recovery-dev-arm-membership-mismatch")
        dev_indices = fixed_dev_indices(dev.samples, protocol.epoch_dev_samples)
        report.update(**_binding(dev), train_samples_per_arm=protocol.train_samples_per_arm,
            common_dev_sample_count=len(dev), common_dev_selection_sha256=canonical_sha256([sample["sample_id"] for sample in dev.samples]),
            epoch_dev_selection_sha256=canonical_sha256([dev.samples[index]["sample_id"] for index in dev_indices]),
            epoch_dev_sample_ids=[dev.samples[index]["sample_id"] for index in dev_indices],
            selection_scope="fixed-stratified-dev-subset-epoch-and-paired-seed-selection",
            native_comparison_scope="all-training-seeds-against-unchanged-incumbent",
            representative_pair_scope="deployment-candidate-pair-selected-before-full-dev-and-native-driving")
        initial_path = Path(initial_checkpoint_path)
        initial_model, _, initial_checkpoint = load_refined_checkpoint(initial_path, device="cpu")
        if initial_checkpoint["variant"] != protocol.variant:
            raise ValueError("recovery-initial-variant-mismatch")
        initial = {"file_sha256": sha256_file(initial_path),
            "state": {key: value.detach().cpu().clone() for key, value in initial_model.state_dict().items()},
            "pretrained_provenance": initial_checkpoint["pretrained_provenance"]}
        initial["state_sha256"] = model_state_sha256(initial["state"])
        del initial_model
        initial_output = output / "initial"; initial_output.mkdir()
        shutil.copyfile(initial_path, initial_output / "checkpoint.pt")
        if sha256_file(initial_output / "checkpoint.pt") != initial["file_sha256"]:
            raise ValueError("recovery-initial-copy-mismatch")
        report.update(initial_checkpoint_sha256=initial["file_sha256"],
                      initial_model_state_sha256=initial["state_sha256"])
        # All seeds and arm budgets are frozen before the first training batch.
        _json(output / "protocol.json", dict(report))
        for seed in protocol.seeds:
            report["training"][str(seed)] = {}
            for arm in ARMS:
                report["training"][str(seed)][arm] = _train_arm(datasets[arm], dev, dev_indices,
                    initial, protocol, seed, output / f"seed{seed}" / arm, device)
                report["training"][str(seed)][arm]["compatible_checkpoint_sha256"] = _export_refined_checkpoint(
                    output / f"seed{seed}" / arm / "checkpoint.pt",
                    output / f"seed{seed}" / arm / "refined_checkpoint.pt")
                _json(output / f"seed{seed}" / arm / "training_report.json", report["training"][str(seed)][arm])
                _json(output / "comparison.json", report)
        _equal_budget(report["training"], protocol)
        # Choose the native pair before evaluating full DEV. Both arms share
        # this seed; no individual full-DEV score can change the choice.
        chosen_seed = representative_seed(report["training"], protocol)
        report["representative_seed"] = chosen_seed
        report["representative_checkpoints"] = {arm: arm + "/checkpoint.pt" for arm in ARMS}
        _json(output / "native_pair_selection.json", {
            "status": "PASS", "representative_seed": chosen_seed,
            "policy": protocol.representative_seed_policy,
            "epoch_dev_selection_sha256": report["epoch_dev_selection_sha256"],
            "protocol_sha256": protocol.sha256,
            "training_report_sha256": {f"{seed}/{arm}": sha256_file(output / f"seed{seed}" / arm / "training_report.json")
                for seed in protocol.seeds for arm in ARMS},
            "full_dev_evaluations_completed": False})
        for arm in ARMS:
            target = output / arm; target.mkdir()
            selected = output / f"seed{chosen_seed}" / arm / "checkpoint.pt"
            shutil.copyfile(selected, target / "checkpoint.pt")
            shutil.copyfile(output / f"seed{chosen_seed}" / arm / "refined_checkpoint.pt", target / "refined_checkpoint.pt")
            report.setdefault("representative_checkpoint_sha256", {})[arm] = sha256_file(target / "checkpoint.pt")
            report.setdefault("compatible_checkpoint_sha256", {})[arm] = sha256_file(target / "refined_checkpoint.pt")
        report["initial_evaluation"] = evaluate_recovery_model(dev, initial_output / "checkpoint.pt",
            output / "initial_evaluation", protocol=protocol, device=str(device), initial=True)
        for seed in protocol.seeds:
            report["evaluations"][str(seed)] = {}
            for arm in ARMS:
                report["evaluations"][str(seed)][arm] = evaluate_recovery_model(dev,
                    output / f"seed{seed}" / arm / "checkpoint.pt", output / f"seed{seed}" / arm / "evaluation",
                    protocol=protocol, device=str(device))
                _json(output / "comparison.json", report)
        for arm in ARMS:
            destination = output / arm / "evaluation"
            destination.mkdir()
            shutil.copyfile(output / f"seed{chosen_seed}" / arm / "evaluation" / "metrics.json",
                            destination / "metrics.json")
        aggregate = {}
        for arm in ARMS:
            aggregate[arm] = {}
            for name in ("mean_euclidean_m", "near_mean_m", "far_mean_m", "far_p95_m", "coverage"):
                values = [report["evaluations"][str(seed)][arm]["metrics"][0][name] for seed in protocol.seeds]
                valid = all(type(value) in (int, float) and math.isfinite(value) for value in values)
                aggregate[arm][name] = {"per_seed": values, "mean": float(np.mean(values)) if valid else None,
                    "std": float(np.std(values)) if valid else None,
                    "minimum": min(values) if valid else None, "maximum": max(values) if valid else None,
                    "seed_count": len(values), "interpretation": "descriptive-seed-variation"}
        if source != recovery_source_hashes():
            raise ValueError("recovery-final-source-changed")
        for data in (*datasets.values(), dev):
            data.assert_manifest_unchanged()
        if sha256_file(initial_path) != initial["file_sha256"]:
            raise ValueError("recovery-initial-checkpoint-changed")
        report.update(status="PASS", equal_budget_verified=True, aggregate_metrics=aggregate,
            native_pair_selection_sha256=sha256_file(output / "native_pair_selection.json"))
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": str(error)}
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - began
        _json(output / "comparison.json", report)
    return report
