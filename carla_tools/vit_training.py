"""Isolated frozen-ViT candidate training and complete development scoring."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .learning_data import LaneWaypointDataset, load_learning_config, validate_waypoint_points
from .training import _metric_group, _rng_state, seed_everything
from .vit_waypoint import (
    FEATURE_COUNT, MODEL_ID, VitConfig, VitWaypointModel, canonical_sha256, model_contract,
    sha256_file, validate_pretrained_provenance, vit_config_from_dict, waypoint_loss,
)

CHECKPOINT_SCHEMA = 1
PRETRAINED_REVISION = "c6a5fb7d12bbd3cf3b0079253141c3332aaed7da"
PRETRAINED_SHA256 = "ed8a1265847086c913535fe7d84c0156a68dc98c0edd52298b8bb4bb7bdcbbb1"
PRETRAINED_URL = "https://huggingface.co/timm/" + MODEL_ID + "/resolve/" + PRETRAINED_REVISION + "/pytorch_model.bin"
SOURCE_FILES = (
    "carla_tools/vit_waypoint.py", "carla_tools/vit_training.py", "scripts/train_vit_waypoints.py",
    "carla_tools/learning_data.py", "carla_tools/training.py", "carla_tools/waypoint_model.py",
    "carla_tools/baseline.py", "carla_tools/contracts.py", "configs/learning.json", "configs/baseline.json",
)


def runtime_source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {name: sha256_file(root / name) for name in SOURCE_FILES}


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _new_output(path):
    path = Path(path)
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("vit-output-symlink")
    if path.exists():
        raise ValueError("vit-output-already-exists")
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def _device(requested):
    # A requested accelerator is never substituted with local CPU training.
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("vit-unsupported-device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("vit-requested-cuda-unavailable")
    return device


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _atomic_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _factory(factory, config, *, pretrained, weights_path=None, expected_weights_sha256=None, weights_url=None):
    return factory(config=config, pretrained=pretrained, weights_path=weights_path,
                   expected_weights_sha256=expected_weights_sha256, weights_url=weights_url)


def _tensor_sha256(value):
    array = value.detach().cpu().contiguous().numpy()
    return canonical_sha256({"dtype": str(array.dtype), "shape": list(array.shape),
                             "bytes_sha256": __import__("hashlib").sha256(array.tobytes()).hexdigest()})


def cache_features(model, dataset, device, config, *, deadline=None):
    """Encode only images; identifiers and labels remain separate cache fields."""
    model.eval()
    images, targets, identities = [], [], []
    loader = DataLoader(dataset, batch_size=config.feature_batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    started = time.monotonic()
    for batch in loader:
        if deadline is not None and time.monotonic() >= deadline:
            raise ValueError("vit-feature-cache-time-budget-exhausted")
        image = batch["image"].to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                             enabled=config.mixed_precision and device.type == "cuda"):
            features = model.encode(image)
        if features.shape != (image.shape[0], FEATURE_COUNT) or not bool(torch.isfinite(features).all()):
            raise ValueError("vit-invalid-feature-cache-output")
        images.append(features.float().cpu())
        targets.append(batch["target"].float().cpu())
        identities.extend(list(batch["sample_id"]))
    features, targets = torch.cat(images), torch.cat(targets)
    expected = [sample["sample_id"] for sample in dataset.samples]
    if identities != expected or len(identities) != len(dataset):
        raise ValueError("vit-feature-cache-sample-order-mismatch")
    dataset.assert_manifest_unchanged()
    return {"features": features, "targets": targets, "sample_ids": identities,
            "report": {"sample_count": len(identities), "features_shape": list(features.shape),
                       "features_sha256": _tensor_sha256(features), "targets_sha256": _tensor_sha256(targets),
                       "sample_order_sha256": canonical_sha256(identities),
                       "image_bindings_sha256": canonical_sha256([
                           {"sample_id": sample["sample_id"], "image_sha256": sample["image_sha256"]}
                           for sample in dataset.samples]),
                       "elapsed_seconds": time.monotonic() - started,
                       "source": "RGB-only", "augmentation": "none", "encoder_frozen": True,
                       "feature_dtype": "float32", "encoder_amp_fp16": config.mixed_precision and device.type == "cuda"}}


def _cached_epoch(model, cache, device, config, *, optimizer=None, generator=None):
    training = optimizer is not None
    model.train(training)
    count = len(cache["sample_ids"])
    order = torch.randperm(count, generator=generator) if training else torch.arange(count)
    total, seen = 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for start in range(0, count, config.batch_size):
            indices = order[start:start + config.batch_size]
            features = cache["features"][indices].to(device)
            target = cache["targets"][indices].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model.predict_features(features)
            loss = waypoint_loss(prediction, target, config)
            if training:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.head.parameters(), config.gradient_clip_norm)
                if not bool(torch.isfinite(norm)):
                    raise ValueError("vit-head-nonfinite-gradient")
                optimizer.step()
            total += float(loss.detach()) * len(indices)
            seen += len(indices)
    if seen != count or not seen:
        raise ValueError("vit-head-epoch-sample-coverage-mismatch")
    return total / seen, seen


def overfit_check(model, cache, device, config):
    count = min(config.overfit_batch_size, len(cache["sample_ids"]))
    indices = sorted({round(index * (len(cache["sample_ids"]) - 1) / max(1, count - 1)) for index in range(count)})
    features = cache["features"][indices].to(device)
    targets = cache["targets"][indices].to(device)
    head_before = copy.deepcopy(model.head.state_dict())
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=config.overfit_learning_rate, weight_decay=0)
    model.eval()  # Dropout disabled in this fixed-batch wiring diagnostic.
    with torch.no_grad():
        initial = float(waypoint_loss(model.predict_features(features), targets, config))
    for _ in range(config.overfit_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = waypoint_loss(model.predict_features(features), targets, config)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.head.parameters(), config.gradient_clip_norm)
        if not bool(torch.isfinite(norm)):
            raise ValueError("vit-overfit-nonfinite-gradient")
        optimizer.step()
    with torch.no_grad():
        final = float(waypoint_loss(model.predict_features(features), targets, config))
    ratio = final / max(initial, 1e-12)
    model.head.load_state_dict(head_before, strict=True)
    restored = all(torch.equal(model.head.state_dict()[name], value) for name, value in head_before.items())
    if not restored:
        raise ValueError("vit-overfit-head-reset-failed")
    return {"status": "PASS" if final < initial and ratio <= config.overfit_loss_ratio_target else "FAIL",
            "initial_loss": initial, "final_loss": final, "loss_ratio": ratio,
            "required_max_loss_ratio": config.overfit_loss_ratio_target, "sample_count": len(indices),
            "steps": config.overfit_steps, "indices": indices, "head_reset_to_fresh_initialization": restored,
            "optimizer_discarded_before_training": True, "encoder_frozen": True}


def _validate_weights_identity(provenance):
    validate_pretrained_provenance(provenance)
    if provenance["sha256"] != PRETRAINED_SHA256 or provenance["url"] != PRETRAINED_URL:
        raise ValueError("vit-pretrained-official-revision-mismatch")


def load_vit_checkpoint(path, *, device="cpu", manifest_sha256=None, model_factory=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if type(checkpoint) is not dict or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA or checkpoint.get("checkpoint_role") != "selected-model":
        raise ValueError("vit-selected-checkpoint-required")
    config = vit_config_from_dict(checkpoint["config"])
    contract = model_contract(config)
    if checkpoint.get("config_sha256") != config.sha256 or checkpoint.get("model_contract") != contract or checkpoint.get("model_contract_sha256") != canonical_sha256(contract):
        raise ValueError("vit-checkpoint-contract-mismatch")
    if checkpoint.get("source_sha256") != runtime_source_hashes():
        raise ValueError("vit-checkpoint-source-mismatch")
    if manifest_sha256 is not None and checkpoint.get("manifest_sha256") != manifest_sha256:
        raise ValueError("vit-checkpoint-manifest-mismatch")
    if checkpoint.get("fixture_injection") is True and model_factory is None:
        raise ValueError("vit-fixture-checkpoint-requires-explicit-factory")
    if model_factory is None:
        _validate_weights_identity(checkpoint.get("pretrained_provenance"))
    history = checkpoint.get("history", [])
    if not history or checkpoint.get("epoch") != min(history, key=lambda row: row["dev_loss"])["epoch"]:
        raise ValueError("vit-selected-epoch-history-mismatch")
    model = _factory(model_factory or VitWaypointModel, config, pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.pretrained_provenance = checkpoint["pretrained_provenance"]
    model.to(_device(device)).eval()
    return model, config, checkpoint


def train_vit_waypoints(manifest_path, output_dir, *, weights_path, expected_weights_sha256=PRETRAINED_SHA256,
                        weights_url=PRETRAINED_URL, config=None, device="cuda", max_seconds=1200,
                        max_epochs=None, model_factory=None, dataset_factory=None, evaluate=True):
    config = config or VitConfig()
    if not isinstance(config, VitConfig) or type(evaluate) is not bool:
        raise ValueError("vit-invalid-training-config")
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("vit-invalid-time-budget")
    if max_epochs is not None and (type(max_epochs) is not int or max_epochs < 1):
        raise ValueError("vit-invalid-maximum-epochs")
    if model_factory is None and (expected_weights_sha256 != PRETRAINED_SHA256 or weights_url != PRETRAINED_URL):
        raise ValueError("vit-pretrained-official-revision-mismatch")
    manifest_path = Path(manifest_path).resolve()
    manifest_hash, source = sha256_file(manifest_path), runtime_source_hashes()
    output = _new_output(output_dir)
    started = time.monotonic()
    deadline = started + max_seconds
    report = {"schema_version": 1, "status": "FAIL", "backbone_id": MODEL_ID,
              "pretrained_source_repository": "timm/" + MODEL_ID, "pretrained_revision": PRETRAINED_REVISION,
              "manifest_sha256": manifest_hash, "source_sha256": source,
              "config": config.to_dict(), "config_sha256": config.sha256,
              "model_contract": model_contract(config), "model_contract_sha256": canonical_sha256(model_contract(config)),
              "training_protocol": "frozen-feature-head-only", "augmentation": "none-fixed-RGB-feature-cache",
              "comparison_limitation": "ResNet candidate used photometric augmentation; this frozen-cache candidate uses none",
              "split": "train/dev", "final_town_used": False, "fixture_injection": model_factory is not None or dataset_factory is not None,
              "time_budget_seconds": max_seconds, "time_budget_scope": "feature-extraction-and-head-training; full-dev-evaluation-follows",
              "history": []}
    _json(output / "training_report.json", report)
    try:
        device = _device(device)
        report["runtime"] = {"torch": str(torch.__version__), "numpy": str(np.__version__), "device": str(device),
                             "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                             "encoder_amp_fp16": config.mixed_precision and device.type == "cuda", "head_precision": "float32"}
        seed_everything(config.seed, torch, np)
        data_factory = dataset_factory or LaneWaypointDataset
        learning_config = load_learning_config()
        train = data_factory(manifest_path, "train", config=learning_config, augment=False, seed=config.seed)
        dev = data_factory(manifest_path, "dev", config=learning_config, augment=False, seed=config.seed)
        if model_factory is None and dataset_factory is None and (len(train), len(dev)) != (1665, 1305):
            raise ValueError("vit-frozen-pilot-count-mismatch")
        report.update(train_sample_count=len(train), dev_sample_count=len(dev), preprocessing_config_sha256=learning_config.sha256)
        model = _factory(model_factory or VitWaypointModel, config, pretrained=True, weights_path=weights_path,
                         expected_weights_sha256=expected_weights_sha256, weights_url=weights_url).to(device)
        if model_factory is None:
            _validate_weights_identity(model.pretrained_provenance)
        report["pretrained_provenance"] = model.pretrained_provenance
        if any(parameter.requires_grad for parameter in model.encoder.parameters()):
            raise ValueError("vit-encoder-not-frozen")
        train_cache = cache_features(model, train, device, config, deadline=deadline)
        dev_cache = cache_features(model, dev, device, config, deadline=deadline)
        report["feature_cache"] = {"train": train_cache["report"], "dev": dev_cache["report"], "tensors_exported": False}
        _json(output / "feature_cache_report.json", report["feature_cache"])
        report["overfit_check"] = overfit_check(model, train_cache, device, config)
        _json(output / "overfit_check.json", report["overfit_check"])
        if report["overfit_check"]["status"] != "PASS":
            raise ValueError("vit-small-batch-overfit-check-failed")
        # Reset global/dropout and sampler state after the diagnostic. The head
        # was restored bit-for-bit; the encoder never received a gradient.
        seed_everything(config.seed, torch, np)
        sampler = torch.Generator().manual_seed(config.seed)
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        best_loss, best_epoch, bad_epochs = math.inf, None, 0
        best_head = None
        for epoch in range(min(config.epochs, max_epochs or config.epochs)):
            if time.monotonic() >= deadline:
                report["time_budget_stopped"] = True
                break
            train.assert_manifest_unchanged()
            dev.assert_manifest_unchanged()
            factor = ((epoch + 1) / max(1, config.warmup_epochs) if epoch < config.warmup_epochs else
                      0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (epoch - config.warmup_epochs) /
                                                        max(1, config.epochs - config.warmup_epochs - 1))))
            optimizer.param_groups[0]["lr"] = config.learning_rate * factor
            began = time.monotonic()
            train_loss, train_count = _cached_epoch(model, train_cache, device, config, optimizer=optimizer, generator=sampler)
            dev_loss, dev_count = _cached_epoch(model, dev_cache, device, config)
            improved = dev_loss < best_loss
            if improved:
                best_loss, best_epoch, bad_epochs = dev_loss, epoch + 1, 0
                best_head = {name: value.detach().cpu().clone() for name, value in model.head.state_dict().items()}
            else:
                bad_epochs += 1
            row = {"epoch": epoch + 1, "train_loss": train_loss, "dev_loss": dev_loss, "train_samples": train_count,
                   "dev_samples": dev_count, "selected": improved, "learning_rate_factor": factor,
                   "elapsed_seconds": time.monotonic() - began}
            report["history"].append(row)
            report.update(completed_epochs=epoch + 1, best_dev_loss=best_loss, selected_epoch=best_epoch)
            _json(output / "training_report.json", report)
            print(json.dumps(row, allow_nan=False), flush=True)
            if bad_epochs >= config.early_stopping_patience:
                report["early_stopped"] = True
                break
        if best_head is None:
            raise ValueError("vit-training-budget-exhausted-before-first-epoch")
        model.head.load_state_dict(best_head, strict=True)
        model.eval()
        checkpoint = {"schema_version": CHECKPOINT_SCHEMA, "checkpoint_role": "selected-model", "epoch": best_epoch,
                      "config": config.to_dict(), "config_sha256": config.sha256, "model_contract": model_contract(config),
                      "model_contract_sha256": canonical_sha256(model_contract(config)), "manifest_sha256": manifest_hash,
                      "source_sha256": source, "pretrained_provenance": model.pretrained_provenance,
                      "pretrained_revision": PRETRAINED_REVISION, "training_protocol": "frozen-feature-head-only",
                      "fixture_injection": report["fixture_injection"], "model_state": model.state_dict(),
                      "history": report["history"], "best_dev_loss": best_loss, "overfit_check": report["overfit_check"],
                      "feature_cache_report": report["feature_cache"]}
        _atomic_checkpoint(output / "checkpoint.pt", checkpoint)
        report["checkpoint_sha256"] = sha256_file(output / "checkpoint.pt")
        with (output / "history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report["history"][0]))
            writer.writeheader()
            writer.writerows(report["history"])
        if sha256_file(manifest_path) != manifest_hash or runtime_source_hashes() != source:
            raise ValueError("vit-training-input-source-changed")
        if evaluate:
            metrics = evaluate_vit_waypoints(manifest_path, output / "checkpoint.pt", output / "evaluation", device=str(device),
                                             model_factory=model_factory, dataset_factory=dataset_factory,
                                             cached_dev=dev_cache)
            report["evaluation"] = {"status": metrics["status"], "sample_count": metrics["sample_count"],
                                    "performance_target_passed": metrics["performance_target_passed"],
                                    "metrics_sha256": sha256_file(output / "evaluation/metrics.json")}
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": str(error)}
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        _json(output / "training_report.json", report)
    return report


def evaluate_vit_waypoints(manifest_path, checkpoint_path, output_dir, *, device="cpu", model_factory=None,
                           dataset_factory=None, cached_dev=None):
    manifest_path, checkpoint_path = Path(manifest_path).resolve(), Path(checkpoint_path).resolve()
    manifest_hash, checkpoint_hash, source = sha256_file(manifest_path), sha256_file(checkpoint_path), runtime_source_hashes()
    model, config, checkpoint = load_vit_checkpoint(checkpoint_path, device=device, manifest_sha256=manifest_hash, model_factory=model_factory)
    dataset = (dataset_factory or LaneWaypointDataset)(manifest_path, "dev", config=load_learning_config(), augment=False, seed=config.seed)
    if model_factory is None and dataset_factory is None and len(dataset) != 1305:
        raise ValueError("vit-frozen-development-count-mismatch")
    output = _new_output(output_dir)
    device = _device(device)
    # Warmup uses the first dev image without consuming its labels or recording
    # a scored result. The complete ordered pass below includes it again.
    image = dataset[0]["image"][None].to(device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                               enabled=config.mixed_precision and device.type == "cuda"):
        model.encode(image)
    _synchronize(device)
    records, cache_max_difference = [], 0.0
    for index in range(len(dataset)):
        item = dataset[index]
        sample = dataset.samples[index]
        if sample["sample_id"] != item["sample_id"] or sha256_file(manifest_path.parent / sample["image"]) != sample["image_sha256"]:
            raise ValueError("vit-evaluation-sample-binding-mismatch")
        _synchronize(device)
        began = time.perf_counter()
        try:
            image = item["image"][None].to(device)
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                       enabled=config.mixed_precision and device.type == "cuda"):
                features = model.encode(image)
            with torch.inference_mode():
                prediction = model.predict_features(features.float()).float().cpu()
            _synchronize(device)
            latency = (time.perf_counter() - began) * 1000
            points = prediction[0].tolist()
            validate_waypoint_points(points)
            error = None
        except (ValueError, RuntimeError, TypeError) as exception:
            _synchronize(device)
            latency = (time.perf_counter() - began) * 1000
            points, error = None, str(exception)
        # GT is accessed for scoring only after the RGB-only prediction returns.
        target = item["target"].tolist()
        value = {"sample_id": sample["sample_id"], "case": sample["case"], "condition": sample["condition"],
                 "route_id": sample["route_id"], "image_sha256": sample["image_sha256"], "target": target,
                 "prediction": points, "error": error, "method": "DINOv3-ViT-B16-frozen", "latency_ms": latency}
        if points is not None:
            value.update(errors_m=[math.hypot(p[0] - t[0], p[1] - t[1]) for p, t in zip(points, target)],
                         x_errors_m=[abs(p[0] - t[0]) for p, t in zip(points, target)],
                         y_errors_m=[abs(p[1] - t[1]) for p, t in zip(points, target)])
            if cached_dev is not None:
                if cached_dev["sample_ids"][index] != sample["sample_id"]:
                    raise ValueError("vit-evaluation-cache-sample-order-mismatch")
                with torch.inference_mode():
                    cached_prediction = model.predict_features(cached_dev["features"][index:index + 1].to(device)).float().cpu()
                difference = float((prediction - cached_prediction).abs().max())
                cache_max_difference = max(cache_max_difference, difference)
                value["cached_prediction_max_abs_difference_m"] = difference
        records.append(value)
    dataset.assert_manifest_unchanged()
    if sha256_file(checkpoint_path) != checkpoint_hash or runtime_source_hashes() != source:
        raise ValueError("vit-evaluation-checkpoint-or-source-changed")
    groups = [dict(method="DINOv3-ViT-B16-frozen", case="all", condition="all", **_metric_group(records))]
    for case in sorted({row["case"] for row in records}):
        groups.append(dict(method="DINOv3-ViT-B16-frozen", case=case, condition="all", **_metric_group([row for row in records if row["case"] == case])))
    for case, condition in sorted({(row["case"], row["condition"]) for row in records}):
        groups.append(dict(method="DINOv3-ViT-B16-frozen", case=case, condition=condition,
                           **_metric_group([row for row in records if (row["case"], row["condition"]) == (case, condition)])))
    overall = groups[0]
    passed = (overall["coverage"] == 1 and overall["near_mean_m"] <= config.near_mean_target_m and overall["far_mean_m"] <= config.far_mean_target_m)
    latencies = np.asarray([row["latency_ms"] for row in records], dtype=np.float64)
    report = {"schema_version": 1, "status": "PASS", "split": "dev", "final_town_used": False,
              "backbone_id": MODEL_ID, "training_protocol": "frozen-feature-head-only", "selected_epoch": checkpoint["epoch"],
              "manifest_sha256": manifest_hash, "checkpoint_sha256": checkpoint_hash, "source_sha256": source,
              "config_sha256": config.sha256, "pretrained_provenance": checkpoint["pretrained_provenance"],
              "pretrained_revision": PRETRAINED_REVISION, "sample_count": len(records), "metrics": groups,
              "performance_target": {"near_mean_max_m": config.near_mean_target_m, "far_mean_max_m": config.far_mean_target_m, "required_coverage": 1.0},
              "performance_target_passed": passed, "predictions": records,
              "latency": {"mean_ms": float(latencies.mean()), "p95_ms": float(np.percentile(latencies, 95)), "max_ms": float(latencies.max()),
                          "scope": "full-encoder-and-head-plus-device-transfer; warmup excluded; decode/resize/controller excluded",
                          "device_synchronized": device.type == "cuda", "device": str(device)},
              "cached_feature_prediction_check": {"checked": cached_dev is not None, "maximum_abs_difference_m": cache_max_difference,
                                                  "note": "FP16 encoder batch shape can introduce rounding differences; full RGB predictions are authoritative"},
              "fixture_injection": checkpoint["fixture_injection"]}
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    _json(output / "metrics.json", report)
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        names = [name for name in groups[0] if name != "per_waypoint_mean_m"]
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows({name: row[name] if row[name] is not None else "N/A" for name in names} for row in groups)
    return report
