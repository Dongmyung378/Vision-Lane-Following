"""Train compact attention refinements without altering archived model code."""
from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .learning_data import LaneWaypointDataset, load_learning_config, learning_config_from_dict, validate_waypoint_points, LearningDataError
from .refined_waypoint import (VARIANTS, RefinedWaypointModel, MirroredWaypointDataset,
                               refined_contract, refined_loss)
from .spatial_training import spatial_source_hashes, validate_reused_baseline
from .training import (_atomic_checkpoint, _device, _epoch, _json, _metric_group,
                       _new_output, _optimizer, _overfit_check,
                       _validate_pretrained_provenance, seed_everything, sha256_file)

SOURCES = ("carla_tools/refined_waypoint.py", "carla_tools/refined_training.py", "scripts/train_refined_waypoints.py")


def refined_source_hashes():
    root = Path(__file__).resolve().parents[1]
    return dict(spatial_source_hashes(), **{name: sha256_file(root / name) for name in SOURCES})


def _contract(variant, config):
    value = refined_contract(variant, config)
    return value, hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_refined_checkpoint(path, *, device="cpu", expected_sha256=None, manifest_sha256=None):
    path = Path(path)
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("refined-checkpoint-hash-mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 1 or checkpoint.get("checkpoint_role") != "selected-refined-model":
        raise ValueError("refined-checkpoint-schema-or-role-invalid")
    config = learning_config_from_dict(checkpoint["config"])
    contract, digest = _contract(checkpoint["variant"], config)
    if (checkpoint.get("model_contract") != contract or checkpoint.get("model_contract_sha256") != digest
            or checkpoint.get("config_sha256") != config.sha256 or checkpoint.get("source_sha256") != refined_source_hashes()):
        raise ValueError("refined-checkpoint-contract-or-source-mismatch")
    if manifest_sha256 is not None and checkpoint.get("manifest_sha256") != manifest_sha256:
        raise ValueError("refined-checkpoint-manifest-mismatch")
    _validate_pretrained_provenance(checkpoint["pretrained_provenance"])
    model = RefinedWaypointModel(variant=checkpoint["variant"], config=config, pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.pretrained_provenance = checkpoint["pretrained_provenance"]
    return model.to(_device(torch, device)).eval(), config, checkpoint


def train_refinement(manifest_path, output_dir, variant, *, config=None, device="cpu", weights_path=None):
    config = config or load_learning_config()
    device = _device(torch, device)
    source = refined_source_hashes()
    digest = sha256_file(Path(manifest_path))
    train = MirroredWaypointDataset(manifest_path, "train", config=config, augment=True)
    fixed = LaneWaypointDataset(manifest_path, "train", config=config, augment=False)
    dev = LaneWaypointDataset(manifest_path, "dev", config=config, augment=False)
    output = _new_output(Path(output_dir))
    contract, contract_hash = _contract(variant, config)
    report = {"status": "FAIL", "variant": variant, "manifest_sha256": digest,
              "source_sha256": source, "config": config.to_dict(), "config_sha256": config.sha256,
              "model_contract": contract, "model_contract_sha256": contract_hash,
              "train_sample_count": len(train), "dev_sample_count": len(dev),
              "fixture_injection": False, "final_town_used": False,
              "training_protocol": "full-encoder-finetuning-mirrored-train-weighted-horizons",
              "runtime": {"device": str(device), "torch": str(torch.__version__),
                          "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                          "amp_enabled": bool(config.mixed_precision and device.type == "cuda")}, "history": []}
    began = time.monotonic()
    factory = lambda **kwargs: RefinedWaypointModel(variant=variant, **kwargs)
    try:
        seed_everything(config.seed, torch, np)
        diagnostic = dataclasses.replace(config, overfit_learning_rate=min(config.learning_rate, config.overfit_learning_rate))
        report["overfit_check"] = dict(_overfit_check(factory, fixed, diagnostic, device, torch, refined_loss, weights_path),
                                      model_reset_to_fresh_initialization=True)
        _json(output / "overfit_check.json", report["overfit_check"])
        if report["overfit_check"]["status"] != "PASS":
            raise ValueError("refined-small-batch-overfit-failed")
        seed_everything(config.seed, torch, np)
        model = factory(config=config, pretrained=True, weights_path=weights_path).to(device)
        report["pretrained_provenance"] = model.pretrained_provenance
        _validate_pretrained_provenance(model.pretrained_provenance)
        report["parameter_count"] = sum(p.numel() for p in model.parameters())
        optimizer = _optimizer(model, config, torch)
        scaler = torch.amp.GradScaler(device.type, enabled=config.mixed_precision and device.type == "cuda")
        from torch.utils.data import DataLoader
        loader = DataLoader(train, batch_size=config.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(config.seed), num_workers=config.num_workers if device.type == "cuda" else 0,
            pin_memory=device.type == "cuda")
        dev_loader = DataLoader(dev, batch_size=config.batch_size, shuffle=False,
            generator=torch.Generator().manual_seed(config.seed + 1), num_workers=0, pin_memory=device.type == "cuda")
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
            train_loss, train_count = _epoch(model, loader, device, config, torch, refined_loss, optimizer=optimizer, scaler=scaler)
            dev_loss, dev_count = _epoch(model, dev_loader, device, config, torch, refined_loss)
            selected = dev_loss < best_loss
            best_loss, bad_epochs = (dev_loss, 0) if selected else (best_loss, bad_epochs + 1)
            item = {"epoch": epoch + 1, "train_loss": train_loss, "dev_loss": dev_loss,
                    "train_samples": train_count, "dev_samples": dev_count, "selected": selected}
            report["history"].append(item)
            if selected:
                checkpoint = {"schema_version": 1, "checkpoint_role": "selected-refined-model",
                    "variant": variant, "epoch": epoch + 1, "selected_dev_loss": best_loss,
                    "config": config.to_dict(), "config_sha256": config.sha256,
                    "model_contract": contract, "model_contract_sha256": contract_hash,
                    "manifest_sha256": digest, "source_sha256": source,
                    "pretrained_provenance": model.pretrained_provenance,
                    "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
                _atomic_checkpoint(output / "checkpoint.pt", checkpoint, torch)
            report.update(completed_epochs=epoch + 1, best_dev_loss=best_loss)
            _json(output / "training_report.json", report)
            print(json.dumps(dict(variant=variant, **item)), flush=True)
            if bad_epochs >= config.early_stopping_patience:
                report["early_stopped"] = True
                break
        if source != refined_source_hashes():
            raise ValueError("refined-runtime-source-changed")
        train.assert_manifest_unchanged()
        report.update(status="PASS", checkpoint_sha256=sha256_file(output / "checkpoint.pt"))
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": str(error)}
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - began
        _json(output / "training_report.json", report)
    del model, optimizer, loader, dev_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def evaluate_refinement(manifest_path, checkpoint_path, output_dir, *, device="cpu"):
    digest = sha256_file(Path(manifest_path))
    model, config, checkpoint = load_refined_checkpoint(checkpoint_path, device=device, manifest_sha256=digest)
    device = next(model.parameters()).device
    data = LaneWaypointDataset(manifest_path, "dev", config=config, augment=False)
    output = _new_output(Path(output_dir))
    def forward(image):
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=config.mixed_precision and device.type == "cuda"):
            result = model(image.to(device)).float().cpu().tolist()[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return result
    for _ in range(12):
        forward(data[0]["image"][None])
    records = []
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for index in range(len(data)):
            item, sample = data[index], data.samples[index]
            began = time.perf_counter()
            points = forward(item["image"][None])
            elapsed = (time.perf_counter() - began) * 1000
            target, error = item["target"].tolist(), None
            try:
                validate_waypoint_points(points)
            except LearningDataError:
                points, error = None, "invalid-model-output-order-or-range"
            row = {key: item[key] for key in ("sample_id", "case", "condition", "route_id")}
            row.update(image_sha256=sample["image_sha256"], target=target, prediction=points, error=error,
                       latency_ms=elapsed, method=checkpoint["variant"])
            if points is not None:
                row.update(errors_m=[math.dist(p, t) for p, t in zip(points, target)],
                           x_errors_m=[abs(p[0]-t[0]) for p, t in zip(points, target)],
                           y_errors_m=[abs(p[1]-t[1]) for p, t in zip(points, target)])
            records.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    data.assert_manifest_unchanged()
    groups = [dict(case="all", condition="all", **_metric_group(records))]
    for case in sorted({r["case"] for r in records}):
        groups.append(dict(case=case, condition="all", **_metric_group([r for r in records if r["case"] == case])))
    overall = groups[0]
    report = {"status": "PASS", "variant": checkpoint["variant"], "sample_count": len(data),
        "split": "dev", "final_town_used": False, "fixture_injection": False,
        "manifest_sha256": digest, "checkpoint_sha256": sha256_file(Path(checkpoint_path)),
        "source_sha256": refined_source_hashes(), "config_sha256": config.sha256, "metrics": groups,
        "performance_target_passed": overall["coverage"] == 1 and overall["near_mean_m"] <= config.near_mean_target_m and overall["far_mean_m"] <= config.far_mean_target_m,
        "latency_ms": {"mean": float(np.mean([r["latency_ms"] for r in records])), "p95": float(np.percentile([r["latency_ms"] for r in records], 95))},
        "latency_scope": "batch-one-forward-and-transfer-excludes-decoding-preprocessing-control",
        "runtime": {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}
    _json(output / "metrics.json", report)
    return report


def choose_refinement(reference, candidates):
    expected = (reference["manifest_sha256"], reference["config_sha256"], reference["sample_count"])
    base = {m["case"]: m for m in reference["metrics"] if m["condition"] == "all"}
    eligible, reasons = [], {}
    for name, report in candidates.items():
        if (report["status"] != "PASS" or report.get("fixture_injection") is not False or report.get("final_town_used") is not False
                or report["split"] != "dev" or (report["manifest_sha256"], report["config_sha256"], report["sample_count"]) != expected):
            raise ValueError("refined-comparison-dataset-or-protocol-mismatch")
        groups = {m["case"]: m for m in report["metrics"] if m["condition"] == "all"}
        failed = []
        if not report["performance_target_passed"]:
            failed.append("overall-target-or-coverage")
        if groups["all"]["far_mean_m"] >= base["all"]["far_mean_m"]:
            failed.append("overall-far-mean")
        for case in ("left_curve", "right_curve", "straight"):
            for metric in ("far_mean_m", "far_p95_m"):
                if groups[case]["coverage"] != 1 or groups[case][metric] > base[case][metric]:
                    failed.append(case + ":" + metric)
        if report["latency_ms"]["p95"] > max(50., 1.5 * reference["latency_ms"]["p95"]):
            failed.append("forward-latency")
        reasons[name] = failed
        if not failed:
            eligible.append(name)
    selected = min(eligible, key=lambda n: candidates[n]["metrics"][0]["far_mean_m"]) if eligible else "resnet"
    return {"selected_candidate": selected, "eligible_improvements": eligible, "failed_criteria": reasons,
            "selection_scope": "DEV-offline-candidate-only", "closed_loop_verified": False, "final_town_used": False}


def run_refinements(manifest_path, output_dir, reference_path, *, device="cpu", config=None, weights_path=None):
    config = config or load_learning_config()
    output = _new_output(Path(output_dir))
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    _, baseline = validate_reused_baseline(reference, config, sha256_file(Path(manifest_path)))
    _json(output / "reused_baseline.json", reference)
    training, evaluations = {}, {}
    for variant in VARIANTS:
        training[variant] = train_refinement(manifest_path, output / variant, variant, device=device, config=config, weights_path=weights_path)
        evaluations[variant] = evaluate_refinement(manifest_path, output / variant / "checkpoint.pt", output / variant / "evaluation", device=device)
    report = {"status": "PASS", "training": training, "evaluations": evaluations,
              "decision": choose_refinement(baseline, evaluations), "baseline_reuse": reference["provenance"],
              "fixture_injection": False, "final_town_used": False}
    _json(output / "comparison.json", report)
    return report
