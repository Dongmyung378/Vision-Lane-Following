"""Controlled fine-tuning and full DEV comparison of spatial waypoint heads.

All arms share the pretrained CNN, data split, photometric augmentation and
optimizer budget. Execution reports are private artifacts, not public docs.
"""

from __future__ import annotations

import gc
import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .learning_data import LaneWaypointDataset, LearningDataError, learning_config_from_dict, load_learning_config, validate_waypoint_points
from .spatial_waypoint import VARIANTS, create_spatial_model, spatial_model_contract
from .training import (_atomic_checkpoint, _device, _epoch, _json, _metric_group, _new_output,
                       _optimizer, _overfit_check, _validate_pretrained_provenance, runtime_source_hashes,
                       seed_everything, sha256_file)
from .waypoint_model import waypoint_smooth_l1_loss

SOURCE_FILES = ("carla_tools/spatial_waypoint.py", "carla_tools/spatial_training.py",
                "scripts/train_spatial_waypoints.py")
SCHEMA_VERSION = 2


def spatial_source_hashes():
    root = Path(__file__).resolve().parents[1]
    return dict(runtime_source_hashes(), **{name: sha256_file(root / name) for name in SOURCE_FILES})


def _contract(variant, config):
    value = spatial_model_contract(variant, config)
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return value, digest


def load_spatial_checkpoint(path, *, device="cpu", manifest_sha256=None, source_sha256=None):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != SCHEMA_VERSION or checkpoint.get("checkpoint_role") != "selected-spatial-model":
        raise ValueError("spatial-checkpoint-schema-or-role-invalid")
    variant = checkpoint.get("variant")
    config = learning_config_from_dict(checkpoint["config"])
    contract, digest = _contract(variant, config)
    if config.sha256 != checkpoint.get("config_sha256") or checkpoint.get("model_contract") != contract or checkpoint.get("model_contract_sha256") != digest:
        raise ValueError("spatial-checkpoint-config-or-contract-mismatch")
    current_source = spatial_source_hashes()
    expected_source = current_source if source_sha256 is None else source_sha256
    if checkpoint.get("source_sha256") != expected_source:
        raise ValueError("spatial-checkpoint-source-mismatch")
    # An explicitly supplied archived training source can differ in the runner,
    # but the current prediction implementation and all shared core files must
    # still match. There is no automatic acceptance of historical checkpoints.
    for name in (*runtime_source_hashes(), "carla_tools/spatial_waypoint.py"):
        if expected_source.get(name) != current_source[name]:
            raise ValueError("spatial-checkpoint-inference-source-mismatch")
    if manifest_sha256 is not None and checkpoint.get("manifest_sha256") != manifest_sha256:
        raise ValueError("spatial-checkpoint-dataset-mismatch")
    _validate_pretrained_provenance(checkpoint.get("pretrained_provenance"))
    model = create_spatial_model(variant, config=config, pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.pretrained_provenance = checkpoint["pretrained_provenance"]
    model.to(_device(torch, device)).eval()
    return model, config, checkpoint


def train_variant(manifest_path, output_dir, variant, *, device="cpu", config=None, weights_path=None):
    config = config or load_learning_config()
    if variant not in VARIANTS:
        raise ValueError("spatial-unknown-variant")
    device = _device(torch, device)
    source = spatial_source_hashes()
    manifest_path = Path(manifest_path).resolve()
    manifest_hash = sha256_file(manifest_path)
    train = LaneWaypointDataset(manifest_path, "train", config=config, augment=True, seed=config.seed)
    fixed = LaneWaypointDataset(manifest_path, "train", config=config, augment=False, seed=config.seed)
    dev = LaneWaypointDataset(manifest_path, "dev", config=config, augment=False, seed=config.seed)
    output = _new_output(Path(output_dir))
    contract, contract_hash = _contract(variant, config)
    report = {"schema_version": SCHEMA_VERSION, "status": "FAIL", "variant": variant,
              "manifest_sha256": manifest_hash, "source_sha256": source,
              "config": config.to_dict(), "config_sha256": config.sha256,
              "model_contract": contract, "model_contract_sha256": contract_hash,
              "train_sample_count": len(train), "dev_sample_count": len(dev),
              "fixture_injection": False, "final_town_used": False,
              "training_protocol": "full-encoder-finetuning-identical-augmentation-and-budget",
              "runtime": {"device": str(device), "torch": str(torch.__version__),
                          "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                          "amp_enabled": bool(config.mixed_precision and device.type == "cuda")}, "history": []}
    began = time.monotonic()

    def factory(**kwargs):
        return create_spatial_model(variant, **kwargs)

    try:
        seed_everything(config.seed, torch, np)
        diagnostic_config = dataclasses.replace(config, overfit_learning_rate=min(config.learning_rate, config.overfit_learning_rate))
        overfit = _overfit_check(factory, fixed, diagnostic_config, device, torch, waypoint_smooth_l1_loss, weights_path)
        report["overfit_check"] = dict(overfit, model_reset_to_fresh_initialization=True,
                                      head_learning_rate=diagnostic_config.overfit_learning_rate,
                                      diagnostic_config_sha256=diagnostic_config.sha256)
        _json(output / "overfit_check.json", report["overfit_check"])
        if overfit["status"] != "PASS":
            raise ValueError("spatial-small-batch-overfit-failed")
        seed_everything(config.seed, torch, np)
        model = factory(config=config, pretrained=True, weights_path=weights_path).to(device)
        report["pretrained_provenance"] = model.pretrained_provenance
        _validate_pretrained_provenance(model.pretrained_provenance)
        report["parameter_count"] = {"total": sum(p.numel() for p in model.parameters()),
                                     "encoder": sum(p.numel() for p in model.encoder.parameters()),
                                     "head": sum(p.numel() for p in model.head.parameters())}
        optimizer = _optimizer(model, config, torch)
        scaler = torch.amp.GradScaler(device.type, enabled=config.mixed_precision and device.type == "cuda")
        from torch.utils.data import DataLoader
        generator = torch.Generator().manual_seed(config.seed)
        train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, generator=generator,
                                  num_workers=config.num_workers if device.type == "cuda" else 0,
                                  pin_memory=device.type == "cuda")
        dev_loader = DataLoader(dev, batch_size=config.batch_size, shuffle=False, num_workers=0,
                                generator=torch.Generator().manual_seed(config.seed + 1), pin_memory=device.type == "cuda")
        best_loss, bad_epochs = math.inf, 0
        for epoch in range(config.epochs):
            train.assert_manifest_unchanged()
            dev.assert_manifest_unchanged()
            train.set_epoch(epoch)
            progress = (epoch - config.warmup_epochs) / max(1, config.epochs - config.warmup_epochs - 1)
            factor = ((epoch + 1) / max(1, config.warmup_epochs) if epoch < config.warmup_epochs
                      else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            epoch_start = time.monotonic()
            train_loss, train_count = _epoch(model, train_loader, device, config, torch, waypoint_smooth_l1_loss,
                                            optimizer=optimizer, scaler=scaler)
            dev_loss, dev_count = _epoch(model, dev_loader, device, config, torch, waypoint_smooth_l1_loss)
            selected = dev_loss < best_loss
            best_loss, bad_epochs = (dev_loss, 0) if selected else (best_loss, bad_epochs + 1)
            item = {"epoch": epoch + 1, "train_loss": train_loss, "dev_loss": dev_loss,
                    "train_samples": train_count, "dev_samples": dev_count, "selected": selected,
                    "learning_rate_factor": factor, "elapsed_seconds": time.monotonic() - epoch_start}
            report["history"].append(item)
            report.update(completed_epochs=epoch + 1, best_dev_loss=best_loss)
            if selected:
                checkpoint = {"schema_version": SCHEMA_VERSION, "checkpoint_role": "selected-spatial-model",
                              "variant": variant, "epoch": epoch + 1, "selected_dev_loss": best_loss,
                              "config": config.to_dict(), "config_sha256": config.sha256,
                              "model_contract": contract, "model_contract_sha256": contract_hash,
                              "manifest_sha256": manifest_hash, "source_sha256": source,
                              "pretrained_provenance": model.pretrained_provenance,
                              "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
                _atomic_checkpoint(output / "checkpoint.pt", checkpoint, torch)
            _json(output / "training_report.json", report)
            print(json.dumps(dict(variant=variant, **item)), flush=True)
            if bad_epochs >= config.early_stopping_patience:
                report["early_stopped"] = True
                break
        train.assert_manifest_unchanged()
        if spatial_source_hashes() != source:
            raise ValueError("spatial-training-source-changed")
        report.update(status="PASS", checkpoint_sha256=sha256_file(output / "checkpoint.pt"))
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": str(error)}
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - began
        _json(output / "training_report.json", report)
    del model, optimizer, train_loader, dev_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def _forward(model, image, config, device):
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=config.mixed_precision and device.type == "cuda"):
        points = model(image.to(device)).float().cpu().tolist()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return points


def evaluate_variant(manifest_path, checkpoint_path, output_dir, *, device="cpu"):
    manifest_path, checkpoint_path = Path(manifest_path).resolve(), Path(checkpoint_path)
    digest = sha256_file(manifest_path)
    model, config, checkpoint = load_spatial_checkpoint(checkpoint_path, device=device, manifest_sha256=digest)
    device = next(model.parameters()).device
    dataset = LaneWaypointDataset(manifest_path, "dev", config=config, augment=False)
    output = _new_output(Path(output_dir))
    warmup = dataset[0]["image"][None]
    for _ in range(12):
        _forward(model, warmup, config, device)
    records = []
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for index in range(len(dataset)):
            item, sample = dataset[index], dataset.samples[index]
            began = time.perf_counter()
            points = _forward(model, item["image"][None], config, device)[0]
            latency = (time.perf_counter() - began) * 1000
            target = item["target"].tolist()
            error = None
            try:
                validate_waypoint_points(points)
            except LearningDataError:
                points, error = None, "invalid-model-output-order-or-range"
            record = {key: item[key] for key in ("sample_id", "case", "condition", "route_id")}
            record.update(image_sha256=sample["image_sha256"], target=target, prediction=points,
                          error=error, latency_ms=latency, method=checkpoint["variant"])
            if points is not None:
                record.update(errors_m=[math.dist(p, t) for p, t in zip(points, target)],
                              x_errors_m=[abs(p[0] - t[0]) for p, t in zip(points, target)],
                              y_errors_m=[abs(p[1] - t[1]) for p, t in zip(points, target)])
            records.append(record)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    dataset.assert_manifest_unchanged()
    groups = [dict(case="all", condition="all", **_metric_group(records))]
    for case in sorted({r["case"] for r in records}):
        groups.append(dict(case=case, condition="all", **_metric_group([r for r in records if r["case"] == case])))
    for case, condition in sorted({(r["case"], r["condition"]) for r in records}):
        groups.append(dict(case=case, condition=condition, **_metric_group([r for r in records if (r["case"], r["condition"]) == (case, condition)])))
    overall = groups[0]
    report = {"schema_version": SCHEMA_VERSION, "status": "PASS", "variant": checkpoint["variant"],
              "split": "dev", "final_town_used": False, "fixture_injection": False,
              "sample_count": len(records), "manifest_sha256": digest,
              "checkpoint_sha256": sha256_file(checkpoint_path), "source_sha256": spatial_source_hashes(),
              "config_sha256": config.sha256, "pretrained_provenance": model.pretrained_provenance,
              "metrics": groups, "performance_target_passed": overall["coverage"] == 1.0 and
              overall["near_mean_m"] <= config.near_mean_target_m and overall["far_mean_m"] <= config.far_mean_target_m,
              "latency_scope": "batch-1-model-forward-and-device-transfer; excludes decoding and preprocessing",
              "latency_ms": {"mean": float(np.mean([r["latency_ms"] for r in records])),
                             "p95": float(np.percentile([r["latency_ms"] for r in records], 95)),
                             "warmup_frames_excluded": 12}, "runtime": {"device": str(device),
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "amp_enabled": bool(config.mixed_precision and device.type == "cuda")}}
    _json(output / "metrics.json", report)
    return report


def choose_candidate(evaluations):
    """Only promote an arm that improves overall error and both curve tails.

    This is a DEV-only candidate decision. It does not certify driving safety
    or replace the eventual closed-loop selection and reserved final test.
    """
    if set(evaluations) != set(VARIANTS):
        raise ValueError("spatial-comparison-arm-missing")
    signatures = {(r["manifest_sha256"], r["config_sha256"], r["sample_count"]) for r in evaluations.values()}
    if len(signatures) != 1 or any(r["status"] != "PASS" or r["split"] != "dev" or r["final_town_used"] is not False for r in evaluations.values()):
        raise ValueError("spatial-comparison-evaluation-mismatch")
    groups = {name: {g["case"]: g for g in report["metrics"] if g["condition"] == "all"}
              for name, report in evaluations.items()}
    reference = groups["resnet"]
    eligible = []
    for name in ("spatial", "hybrid"):
        candidate = groups[name]
        if (candidate["all"]["coverage"] == 1.0 and evaluations[name]["performance_target_passed"]
                and candidate["all"]["far_mean_m"] < reference["all"]["far_mean_m"]
                and all(candidate[case]["far_mean_m"] <= reference[case]["far_mean_m"] and
                        candidate[case]["far_p95_m"] <= reference[case]["far_p95_m"]
                        for case in ("left_curve", "right_curve"))
                and evaluations[name]["latency_ms"]["p95"] <= max(50.0, evaluations["resnet"]["latency_ms"]["p95"] * 1.5)):
            eligible.append(name)
    selected = min(eligible, key=lambda name: groups[name]["all"]["far_mean_m"]) if eligible else "resnet"
    return {"selected_candidate": selected, "eligible_improvements": eligible,
            "selection_scope": "DEV-offline-candidate-only", "closed_loop_verified": False,
            "final_town_used": False, "promotion_requires": "accuracy-targets, coverage, both-curve-mean-and-p95, latency"}


def validate_reused_baseline(value, config, manifest_sha256):
    if not isinstance(value, dict):
        raise ValueError("spatial-reused-baseline-invalid")
    train, evaluation = value.get("training", {}), value.get("evaluation", {})
    if (train.get("status") != "PASS" or evaluation.get("status") != "PASS"
            or train.get("variant") != "resnet" or evaluation.get("variant") != "resnet"
            or train.get("config_sha256") != config.sha256 or evaluation.get("config_sha256") != config.sha256
            or train.get("manifest_sha256") != manifest_sha256 or evaluation.get("manifest_sha256") != manifest_sha256
            or train.get("checkpoint_sha256") != evaluation.get("checkpoint_sha256") or not train.get("checkpoint_sha256")
            or train.get("source_sha256") != evaluation.get("source_sha256")
            or train.get("final_town_used") is not False or evaluation.get("final_town_used") is not False
            or train.get("fixture_injection") is not False or evaluation.get("fixture_injection") is not False
            or evaluation.get("split") != "dev" or train.get("dev_sample_count") != evaluation.get("sample_count")
            or train.get("runtime", {}).get("device") != "cuda" or "T4" not in train["runtime"].get("gpu", "")
            or train.get("training_protocol") != "full-encoder-finetuning-identical-augmentation-and-budget"):
        raise ValueError("spatial-reused-baseline-protocol-mismatch")
    current = spatial_source_hashes()
    for name in (*runtime_source_hashes(), "carla_tools/spatial_waypoint.py"):
        if train["source_sha256"].get(name) != current[name]:
            raise ValueError("spatial-reused-baseline-core-source-mismatch")
    return train, evaluation


def run_spatial_comparison(manifest_path, output_dir, *, device="cpu", config=None, weights_path=None,
                           reused_baseline_path=None):
    config = config or load_learning_config()
    output = _new_output(Path(output_dir))
    evaluations, training = {}, {}
    began = time.monotonic()
    reuse = None
    if reused_baseline_path is not None:
        reused_baseline_path = Path(reused_baseline_path)
        value = json.loads(reused_baseline_path.read_text(encoding="utf-8"))
        training["resnet"], evaluations["resnet"] = validate_reused_baseline(value,config,sha256_file(Path(manifest_path)))
        reuse = dict(value["provenance"], frozen_input_sha256=sha256_file(reused_baseline_path))
        _json(output / "reused_baseline.json", value)
    for variant in (v for v in VARIANTS if v not in training):
        training[variant] = train_variant(manifest_path, output / variant, variant, device=device,
                                         config=config, weights_path=weights_path)
        evaluations[variant] = evaluate_variant(manifest_path, output / variant / "checkpoint.pt",
                                                output / variant / "evaluation", device=device)
    report = {"schema_version": SCHEMA_VERSION, "status": "PASS", "training": training,
              "evaluations": evaluations, "decision": choose_candidate(evaluations),
              "elapsed_seconds": time.monotonic() - began, "fixture_injection": False, "final_town_used": False,
              "baseline_reuse": reuse}
    _json(output / "comparison.json", report)
    return report
