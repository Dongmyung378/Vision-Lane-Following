"""Reproducible RGB-only waypoint training and held-out development evaluation.

Execution records and weights belong in an ignored local output directory. The
development set selects checkpoints; reserved final towns are never opened.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path


CHECKPOINT_SCHEMA = 1
SOURCE_FILES = (
    "carla_tools/training.py", "carla_tools/learning_data.py",
    "carla_tools/waypoint_model.py", "carla_tools/baseline.py",
    "configs/baseline.json", "configs/learning.json",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_source_hashes() -> dict:
    root = Path(__file__).resolve().parents[1]
    return {name: sha256_file(root / name) for name in SOURCE_FILES}


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _dependencies():
    import numpy as np
    import torch
    from carla_tools.learning_data import LaneWaypointDataset, load_learning_config
    from carla_tools.waypoint_model import WaypointModel, waypoint_smooth_l1_loss
    return np, torch, LaneWaypointDataset, load_learning_config, WaypointModel, waypoint_smooth_l1_loss


def seed_everything(seed: int, torch, np) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _rng_state(torch, np) -> dict:
    # NumPy's ndarray is converted to a tensor so weights_only=True can safely
    # deserialize the complete checkpoint without arbitrary object pickles.
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": {
        "name": state[0], "keys": torch.tensor(state[1].astype("int64")),
        "pos": state[2], "has_gauss": state[3], "cached_gaussian": state[4]},
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state, torch, np) -> None:
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value["name"], value["keys"].numpy().astype("uint32"),
                         value["pos"], value["has_gauss"], value["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("cuda-resume-requires-cuda")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def _states_equal(first, second, torch) -> bool:
    """Compare restored state recursively, including actual optimizer tensors."""
    if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
        return (isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor) and
                first.dtype == second.dtype and first.shape == second.shape and
                torch.equal(first.detach().cpu(), second.detach().cpu()))
    if isinstance(first, dict) or isinstance(second, dict):
        return (isinstance(first, dict) and isinstance(second, dict) and first.keys() == second.keys() and
                all(_states_equal(first[key], second[key], torch) for key in first))
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        return (type(first) is type(second) and len(first) == len(second) and
                all(_states_equal(a, b, torch) for a, b in zip(first, second)))
    return type(first) is type(second) and first == second


def _new_output(path: Path) -> Path:
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("output-parent-is-symlink")
    path = path.resolve()
    if path.exists():
        raise ValueError("output-directory-already-exists")
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError("output-parent-is-symlink")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _device(torch, requested):
    requested = requested or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("unsupported-training-device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda-required-but-unavailable")
    return device


def _model(factory, config, pretrained, weights_path):
    return factory(config=config, pretrained=pretrained, weights_path=weights_path)


def _optimizer(model, config, torch):
    encoder, head = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (encoder if name.startswith(("encoder.", "backbone.")) else head).append(parameter)
    groups = []
    if encoder:
        groups.append({"params": encoder, "lr": config.encoder_learning_rate,
                       "initial_lr": config.encoder_learning_rate, "role": "encoder"})
    if head:
        groups.append({"params": head, "lr": config.learning_rate,
                       "initial_lr": config.learning_rate, "role": "head"})
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def _loss(prediction, target, config, torch, loss_fn):
    if prediction.shape != target.shape or prediction.ndim != 3 or tuple(prediction.shape[1:]) != (5, 2):
        raise ValueError("model-output-shape-mismatch")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("nonfinite-waypoint-output-or-target")
    return loss_fn(prediction, target, config=config)


def _epoch(model, loader, device, config, torch, loss_fn, *, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total, samples = 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            image = batch["image"].to(device, non_blocking=device.type == "cuda")
            target = batch["target"].to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,
                                enabled=bool(config.mixed_precision and device.type == "cuda")):
                prediction = model(image)
                loss = _loss(prediction, target, config, torch, loss_fn)
            if not math.isfinite(float(loss.detach())):
                raise ValueError("nonfinite-training-loss")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                if not torch.isfinite(norm):
                    raise ValueError("nonfinite-training-gradient")
                scaler.step(optimizer)
                scaler.update()
            count = image.shape[0]
            samples += count
            total += float(loss.detach()) * count
    if not samples:
        raise ValueError("empty-training-or-development-loader")
    return total / samples, samples


def _overfit_check(factory, dataset, config, device, torch, loss_fn, weights_path):
    """Actual forward/backward on a fixed RGB batch, then discard its weights."""
    from torch.utils.data import DataLoader, Subset
    count = min(config.overfit_batch_size, len(dataset))
    # Use deterministic samples spread over the training set, rather than only
    # temporally adjacent frames at the beginning of its first episode.
    indices = sorted({round(index * (len(dataset) - 1) / max(1, count - 1)) for index in range(count)})
    batch = next(iter(DataLoader(Subset(dataset, indices), batch_size=len(indices), num_workers=0)))
    model = _model(factory, config, True, weights_path).to(device)
    # The diagnostic head learns faster; the pretrained encoder retains the
    # same conservative fine-tuning rate used in ordinary training.
    optimizer = _optimizer(model, config, torch)
    for group in optimizer.param_groups:
        group["lr"] = (min(config.encoder_learning_rate, config.overfit_learning_rate)
                       if group["role"] == "encoder" else config.overfit_learning_rate)
        group["weight_decay"] = 0.0
    image, target = batch["image"].to(device), batch["target"].to(device)
    # Disable dropout and batch-statistic updates for the fixed-batch diagnostic
    # while retaining gradients through the real pretrained encoder and head.
    model.eval()
    initial = None
    for step in range(config.overfit_steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(image)
        loss = _loss(prediction, target, config, torch, loss_fn)
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        if not torch.isfinite(norm):
            raise ValueError("overfit-nonfinite-gradient")
        optimizer.step()
    with torch.no_grad():
        final = float(_loss(model(image), target, config, torch, loss_fn))
    ratio = final / max(initial, 1e-12)
    passed = final < initial and ratio <= config.overfit_loss_ratio_target
    report = {"status": "PASS" if passed else "FAIL", "sample_count": len(indices),
              "indices": indices, "steps": config.overfit_steps, "initial_loss": initial,
              "final_loss": final, "loss_ratio": ratio,
              "required_max_loss_ratio": config.overfit_loss_ratio_target,
              "device": str(device), "pretrained": getattr(model, "pretrained_provenance", {})}
    del optimizer, model, image, target
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def _atomic_checkpoint(path, checkpoint, torch):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def _model_contract(config):
    from carla_tools.waypoint_model import model_contract
    contract = model_contract(config)
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    return contract, digest


def _validate_pretrained_provenance(value):
    from carla_tools.waypoint_model import PRETRAINED_HASH_PREFIX, PRETRAINED_URL
    if (not isinstance(value, dict) or value.get("pretrained") is not True or
            value.get("hash_verified") is not True or value.get("url") != PRETRAINED_URL or
            not isinstance(value.get("sha256"), str) or len(value["sha256"]) != 64 or
            not value["sha256"].startswith(PRETRAINED_HASH_PREFIX) or
            any(character not in "0123456789abcdef" for character in value["sha256"])):
        raise ValueError("checkpoint-pretrained-provenance-invalid")


def load_checkpoint(path: Path, *, device="cpu", config=None, manifest_sha256=None,
                    source_sha256=None):
    _, torch, _, _, factory, _ = _dependencies()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported-waypoint-checkpoint")
    from carla_tools.learning_data import learning_config_from_dict
    embedded_config = learning_config_from_dict(checkpoint["config"])
    if embedded_config.sha256 != checkpoint.get("config_sha256"):
        raise ValueError("checkpoint-config-hash-mismatch")
    contract, contract_hash = _model_contract(embedded_config)
    if checkpoint.get("model_contract") != contract or checkpoint.get("model_contract_sha256") != contract_hash:
        raise ValueError("checkpoint-model-contract-mismatch")
    if checkpoint.get("checkpoint_role") not in ("training-state", "selected-model"):
        raise ValueError("unsupported-checkpoint-role")
    _validate_pretrained_provenance(checkpoint.get("pretrained_provenance"))
    if config is not None and config.sha256 != embedded_config.sha256:
        raise ValueError("resume-configuration-mismatch")
    if manifest_sha256 is not None and checkpoint.get("manifest_sha256") != manifest_sha256:
        raise ValueError("resume-dataset-mismatch")
    if checkpoint.get("source_sha256") != (source_sha256 or runtime_source_hashes()):
        raise ValueError("resume-source-mismatch")
    model = _model(factory, embedded_config, False, None)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(torch.device(device))
    model.pretrained_provenance = checkpoint.get("pretrained_provenance", {})
    return model, embedded_config, checkpoint


def train_waypoints(manifest_path: Path, output_dir: Path, *, config=None, device=None,
                    resume: Path | None = None, weights_path: Path | None = None,
                    model_factory=None, dataset_factory=None, max_epochs=None) -> dict:
    np, torch, dataset_type, load_config, default_factory, loss_fn = _dependencies()
    config = config or load_config()
    factory, dataset_type = model_factory or default_factory, dataset_factory or dataset_type
    source = runtime_source_hashes()
    contract, contract_hash = _model_contract(config)
    manifest_path = manifest_path.resolve()
    manifest_hash = sha256_file(manifest_path)
    train = dataset_type(manifest_path, "train", config=config, augment=True, seed=config.seed)
    train_fixed = dataset_type(manifest_path, "train", config=config, augment=False, seed=config.seed)
    dev = dataset_type(manifest_path, "dev", config=config, augment=False, seed=config.seed)
    target = {"near_mean_max_m": config.near_mean_target_m,
              "far_mean_max_m": config.far_mean_target_m, "required_coverage": 1.0,
              "selection": "held-out-dev-normalized-smooth-l1",
              "split": "dev", "final_town_used": False}
    output = _new_output(output_dir)
    report = {"schema_version": 1, "status": "FAIL", "manifest_sha256": manifest_hash,
              "source_sha256": source, "config": config.to_dict(), "config_sha256": config.sha256,
              "model_contract": contract, "model_contract_sha256": contract_hash,
              "performance_target": target, "train_sample_count": len(train),
              "dev_sample_count": len(dev), "started_unix": time.time(), "history": []}
    _json(output / "training_report.json", report)
    try:
        device = _device(torch, device)
        report["runtime"] = {"torch": str(torch.__version__), "numpy": str(np.__version__),
                             "device": str(device), "cuda": torch.version.cuda,
                             "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                             "amp_enabled": bool(config.mixed_precision and device.type == "cuda")}
        seed_everything(config.seed, torch, np)
        generator = torch.Generator().manual_seed(config.seed)
        model = _model(factory, config, resume is None, weights_path).to(device)
        optimizer = _optimizer(model, config, torch)
        scaler = torch.amp.GradScaler(device.type, enabled=config.mixed_precision and device.type == "cuda")
        start_epoch, best_loss, bad_epochs, history = 0, math.inf, 0, []
        original_best_state = None
        if resume is None:
            report["overfit_check"] = _overfit_check(factory, train_fixed, config, device, torch,
                                                    loss_fn, weights_path)
            _json(output / "overfit_check.json", report["overfit_check"])
            if report["overfit_check"]["status"] != "PASS":
                raise ValueError("small-batch-overfit-check-failed")
            # Reset all randomness and reinitialize the complete model so the
            # diagnostic cannot leak its special optimization into training.
            del model, optimizer
            seed_everything(config.seed, torch, np)
            generator.manual_seed(config.seed)
            model = _model(factory, config, True, weights_path).to(device)
            optimizer = _optimizer(model, config, torch)
        else:
            checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
            if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
                raise ValueError("unsupported-waypoint-checkpoint")
            if checkpoint.get("checkpoint_role") != "training-state":
                raise ValueError("resume-requires-last-training-state-checkpoint")
            for key, expected in (("config_sha256", config.sha256),
                                  ("manifest_sha256", manifest_hash), ("source_sha256", source)):
                if checkpoint.get(key) != expected:
                    raise ValueError("resume-provenance-mismatch-" + key)
            if checkpoint.get("config") != config.to_dict():
                raise ValueError("resume-embedded-configuration-mismatch")
            if checkpoint.get("model_contract") != contract or checkpoint.get("model_contract_sha256") != contract_hash:
                raise ValueError("resume-model-contract-mismatch")
            model.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if not _states_equal(optimizer.state_dict(), checkpoint["optimizer_state"], torch):
                raise ValueError("resume-optimizer-state-not-restored")
            scaler.load_state_dict(checkpoint["scaler_state"])
            if scaler.state_dict() != checkpoint["scaler_state"]:
                raise ValueError("resume-scaler-state-not-restored")
            generator.set_state(checkpoint["sampler_rng"].cpu())
            if not torch.equal(generator.get_state(), checkpoint["sampler_rng"].cpu()):
                raise ValueError("resume-sampler-state-not-restored")
            _restore_rng(checkpoint["rng_state"], torch, np)
            if not _states_equal(_rng_state(torch, np), checkpoint["rng_state"], torch):
                raise ValueError("resume-rng-state-not-restored")
            model.pretrained_provenance = checkpoint["pretrained_provenance"]
            start_epoch, best_loss, bad_epochs = checkpoint["epoch"], checkpoint["best_dev_loss"], checkpoint["bad_epochs"]
            history = list(checkpoint["history"])
            original_best_state = checkpoint["best_model_state"]
            report["overfit_check"] = checkpoint["overfit_check"]
            report["resume"] = {"checkpoint_sha256": sha256_file(resume), "start_epoch": start_epoch,
                                "optimizer_state_restored": True, "scaler_state_restored": True,
                                "rng_state_restored": True, "sampler_state_restored": True,
                                "scaler_state_sha256": hashlib.sha256(json.dumps(
                                    checkpoint["scaler_state"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
        report["pretrained_provenance"] = getattr(model, "pretrained_provenance", {})
        if model_factory is None:
            _validate_pretrained_provenance(report["pretrained_provenance"])
        workers = config.num_workers if device.type == "cuda" else 0
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, generator=generator,
                                  num_workers=workers, pin_memory=device.type == "cuda", drop_last=False)
        # A separate generator prevents dev-worker seeding from advancing the
        # training sampler or global dropout RNG between epochs.
        dev_generator = torch.Generator().manual_seed(config.seed + 1)
        dev_loader = DataLoader(dev, batch_size=config.batch_size, shuffle=False, num_workers=0,
                                generator=dev_generator, pin_memory=device.type == "cuda")
        best_state = original_best_state
        end_epoch = config.epochs if max_epochs is None else min(config.epochs, max_epochs)
        if end_epoch <= start_epoch:
            raise ValueError("resume-has-no-remaining-epochs")
        for epoch in range(start_epoch, end_epoch):
            if hasattr(train, "assert_manifest_unchanged"):
                train.assert_manifest_unchanged()
                dev.assert_manifest_unchanged()
            train.set_epoch(epoch)
            # Reproducible warmup and cosine schedule require no mutable
            # scheduler state beyond the persisted completed epoch.
            if epoch < config.warmup_epochs:
                factor = (epoch + 1) / max(1, config.warmup_epochs)
            else:
                progress = (epoch - config.warmup_epochs) / max(1, config.epochs - config.warmup_epochs - 1)
                factor = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            began = time.monotonic()
            train_loss, train_count = _epoch(model, train_loader, device, config, torch, loss_fn,
                                            optimizer=optimizer, scaler=scaler)
            dev_loss, dev_count = _epoch(model, dev_loader, device, config, torch, loss_fn)
            improved = dev_loss < best_loss
            if improved:
                best_loss, bad_epochs = dev_loss, 0
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            else:
                bad_epochs += 1
            item = {"epoch": epoch + 1, "train_loss": train_loss, "dev_loss": dev_loss,
                    "train_samples": train_count, "dev_samples": dev_count,
                    "elapsed_seconds": time.monotonic() - began, "learning_rate_factor": factor,
                    "selected": improved}
            history.append(item)
            checkpoint = {"schema_version": CHECKPOINT_SCHEMA, "epoch": epoch + 1,
                          "checkpoint_role": "training-state", "model_contract": contract,
                          "model_contract_sha256": contract_hash,
                          "config": config.to_dict(), "config_sha256": config.sha256,
                          "manifest_sha256": manifest_hash, "source_sha256": source,
                          "pretrained_provenance": report["pretrained_provenance"],
                          "model_state": model.state_dict(), "best_model_state": best_state,
                          "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(),
                          "rng_state": _rng_state(torch, np), "sampler_rng": generator.get_state(),
                          "best_dev_loss": best_loss, "bad_epochs": bad_epochs,
                          "history": history, "overfit_check": report["overfit_check"]}
            _atomic_checkpoint(output / "last_checkpoint.pt", checkpoint, torch)
            if improved:
                _atomic_checkpoint(output / "checkpoint.pt", dict(checkpoint, checkpoint_role="selected-model"), torch)
            report["history"] = history
            report["completed_epochs"] = epoch + 1
            report["best_dev_loss"] = best_loss
            _json(output / "training_report.json", report)
            print(json.dumps(item), flush=True)
            if bad_epochs >= config.early_stopping_patience:
                report["early_stopped"] = True
                break
        if not (output / "checkpoint.pt").exists():
            # Resume may never beat its previous best. Preserve that selected
            # checkpoint rather than substituting the last epoch's weights.
            selected = dict(checkpoint)
            selected["model_state"] = best_state
            selected["checkpoint_role"] = "selected-model"
            _atomic_checkpoint(output / "checkpoint.pt", selected, torch)
        with (output / "history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        if sha256_file(manifest_path) != manifest_hash or runtime_source_hashes() != source:
            raise ValueError("training-input-source-changed")
        report["checkpoint_sha256"] = sha256_file(output / "checkpoint.pt")
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": str(error)}
        raise
    finally:
        report["finished_unix"] = time.time()
        report["elapsed_seconds"] = report["finished_unix"] - report["started_unix"]
        _json(output / "training_report.json", report)
    return report


def _metric_group(records: list[dict]) -> dict:
    successful = [item for item in records if item.get("prediction") is not None]
    errors = [item["errors_m"] for item in successful]
    count = len(successful)
    def mean(values):
        return sum(values) / len(values) if values else None
    def p95(values):
        if not values:
            return None
        values = sorted(values)
        index = 0.95 * (len(values) - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        return values[lower] * (upper - index) + values[upper] * (index - lower) if lower != upper else values[lower]
    all_errors = [value for error in errors for value in error]
    near_errors = [value for error in errors for value in error[:2]]
    far_errors = [value for error in errors for value in error[2:]]
    return {"sample_count": len(records), "predicted_count": count,
            "failed_prediction_count": len(records) - count,
            "coverage": count / len(records) if records else 0.0,
            "mean_euclidean_m": mean(all_errors), "p95_euclidean_m": p95(all_errors),
            "max_euclidean_m": max(all_errors) if all_errors else None,
            "near_mean_m": mean(near_errors), "near_p95_m": p95(near_errors),
            "far_mean_m": mean(far_errors), "far_p95_m": p95(far_errors),
            "per_waypoint_mean_m": [mean([error[index] for error in errors]) for index in range(5)],
            "x_mae_m": mean([value for item in successful for value in item["x_errors_m"]]),
            "y_mae_m": mean([value for item in successful for value in item["y_errors_m"]])}


def evaluate_waypoints(manifest_path: Path, checkpoint_path: Path, output_dir: Path, *,
                       device="cpu", model=None, config=None, dataset_factory=None,
                       compare_baseline=True, overlay_count=15) -> dict:
    np, torch, dataset_type, _, _, _ = _dependencies()
    manifest_path = manifest_path.resolve()
    manifest_hash = sha256_file(manifest_path)
    if model is None:
        model, config, checkpoint = load_checkpoint(checkpoint_path, device=device,
                                                   manifest_sha256=manifest_hash)
    elif config is None:
        raise ValueError("injected-evaluation-model-requires-config")
    dataset = (dataset_factory or dataset_type)(manifest_path, "dev", config=config, augment=False)
    output = _new_output(output_dir)
    device = _device(torch, device)
    model.to(device).eval()
    records, baseline_records, prior_records = [], [], []
    from PIL import Image
    if compare_baseline:
        from carla_tools.baseline import OpenCVLanePredictor, load_baseline_config
        baseline_config = load_baseline_config()
        baseline = OpenCVLanePredictor(baseline_config)
    overlay_dir = output / "overlays"
    overlay_dir.mkdir()
    overlay_groups, overlays = set(), []
    for index in range(len(dataset)):
        item = dataset[index]
        sample = dataset.samples[index]
        rgb_path = manifest_path.parent / sample["image"]
        if sha256_file(rgb_path) != sample["image_sha256"]:
            raise ValueError("evaluation-rgb-sha256-mismatch")
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        target = item["target"].numpy().tolist()
        base = {"sample_id": item["sample_id"], "case": item["case"],
                "condition": item["condition"], "route_id": item["route_id"],
                "image_sha256": sample["image_sha256"], "target": target}
        began = time.perf_counter()
        with torch.no_grad():
            prediction = model(item["image"].unsqueeze(0).to(device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency = (time.perf_counter() - began) * 1000
        if prediction.shape != (1, 5, 2) or not torch.isfinite(prediction).all():
            points, error = None, "invalid-model-output"
        else:
            points, error = prediction[0].detach().cpu().numpy().tolist(), None
            from carla_tools.learning_data import LearningDataError, validate_waypoint_points
            try:
                # Arc-length samples on a curve need not have increasing x.
                # The image-only contract uses actor-origin radial ordering;
                # the downstream controller additionally checks its measured
                # rear-axle geometry without passing it into the predictor.
                validate_waypoint_points(points)
            except LearningDataError:
                points, error = None, "invalid-model-output-order-or-range"
        def record(points, error, method, timing):
            value = dict(base, prediction=points, error=error, method=method, latency_ms=timing)
            if points is not None:
                value["errors_m"] = [math.hypot(p[0] - t[0], p[1] - t[1]) for p, t in zip(points, target)]
                value["x_errors_m"] = [abs(p[0] - t[0]) for p, t in zip(points, target)]
                value["y_errors_m"] = [abs(p[1] - t[1]) for p, t in zip(points, target)]
            return value
        records.append(record(points, error, "ResNet18", latency))
        prior_records.append(record([[float(distance), 0.0] for distance in config.waypoint_distances_m],
                                    None, "straight_prior", 0.0))
        if compare_baseline:
            detected = baseline.predict(rgb, width=rgb.shape[1], height=rgb.shape[0])
            baseline_records.append(record(detected.get("waypoints_ego_m") if detected["status"] == "PASS" else None,
                                           detected.get("error"), "B0", detected.get("timing_ms")))
        overlay_group = (item["case"], item["condition"])
        if len(overlays) < overlay_count and overlay_group not in overlay_groups and points is not None:
            from carla_tools.baseline import load_baseline_config, nominal_ground_to_pixels
            import cv2
            image = rgb.copy()
            nominal = load_baseline_config()
            for path_points, color in ((target, (0, 255, 0)), (points, (255, 50, 50))):
                projected, front = nominal_ground_to_pixels(np.asarray(path_points), nominal)
                for pixel, visible in zip(projected, front):
                    if (visible and np.isfinite(pixel).all() and
                            0 <= pixel[0] < image.shape[1] and 0 <= pixel[1] < image.shape[0]):
                        cv2.circle(image, tuple(int(round(v)) for v in pixel), 5, color, -1)
            image_name = f"{index:05d}.png"
            Image.fromarray(image).save(overlay_dir / image_name)
            overlay_groups.add(overlay_group)
            overlays.append({"sample_id": item["sample_id"], "case": item["case"],
                             "condition": item["condition"], "image_sha256": sample["image_sha256"],
                             "overlay": "overlays/" + image_name})
    methods = {"ResNet18": records, "straight_prior": prior_records}
    if compare_baseline:
        methods["B0"] = baseline_records
    grouped = []
    for method, values in methods.items():
        grouped.append(dict(method=method, case="all", condition="all", **_metric_group(values)))
        for case in sorted({item["case"] for item in values}):
            subset = [item for item in values if item["case"] == case]
            grouped.append(dict(method=method, case=case, condition="all", **_metric_group(subset)))
        for case, condition in sorted({(item["case"], item["condition"]) for item in values}):
            subset = [item for item in values if (item["case"], item["condition"]) == (case, condition)]
            grouped.append(dict(method=method, case=case, condition=condition, **_metric_group(subset)))
    overall = grouped[0]
    target_pass = (overall["coverage"] == 1.0 and
                   overall["near_mean_m"] <= config.near_mean_target_m and
                   overall["far_mean_m"] <= config.far_mean_target_m)
    report = {"schema_version": 1, "status": "PASS", "split": "dev", "final_town_used": False,
              "manifest_sha256": manifest_hash, "checkpoint_sha256": sha256_file(checkpoint_path),
              "config_sha256": config.sha256, "source_sha256": runtime_source_hashes(),
              "performance_target": {"near_mean_max_m": config.near_mean_target_m,
                                     "far_mean_max_m": config.far_mean_target_m, "required_coverage": 1.0},
              "performance_target_passed": target_pass, "sample_count": len(records),
              "metrics": grouped, "predictions": records, "baseline_predictions": baseline_records,
              "straight_prior_predictions": prior_records,
              "overlays": overlays,
              "overlay_legend": {"green": "privileged-target-for-QA-only", "red": "RGB-only-model"},
              "latency_scope": "model-forward-and-device-transfer; excludes image decoding and resizing"}
    if compare_baseline:
        matched = [(learned, basic) for learned, basic in zip(records, baseline_records)
                   if basic["prediction"] is not None]
        report["matched_baseline_metrics"] = {
            "sample_count": len(matched),
            "selection": "same-dev-RGB-samples-with-successful-B0-detection",
            "ResNet18": _metric_group([learned for learned, _ in matched]),
            "B0": _metric_group([basic for _, basic in matched])}
    if sha256_file(manifest_path) != manifest_hash:
        raise ValueError("evaluation-manifest-changed")
    _json(output / "metrics.json", report)
    fields = [key for key in grouped[0] if key != "per_waypoint_mean_m"]
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group in grouped:
            writer.writerow({key: group[key] if group[key] is not None else "N/A" for key in fields})
    return report
