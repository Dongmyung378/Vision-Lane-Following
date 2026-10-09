"""Evaluate actual OpenCV-first, learned-fallback inference on development RGB."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.baseline import OpenCVLanePredictor, load_baseline_config, nominal_ground_to_pixels
from carla_tools.learning_data import LaneWaypointDataset, LearningDataError, validate_waypoint_points
from carla_tools.rgb_predictor import BaselineAssistedWaypointPredictor, PREDICTOR_VERSION
from carla_tools.training import (
    _device, _metric_group, _new_output, load_checkpoint, runtime_source_hashes, sha256_file,
)

PREDICTOR_SOURCE_FILES = ("carla_tools/rgb_predictor.py", "scripts/evaluate_rgb_predictor.py")
METHOD = "B0-assisted-ResNet18"
TARGET = {"near_mean_max_m": 0.3, "far_mean_max_m": 1.0, "required_coverage": 1.0}


def predictor_source_hashes():
    return {name: sha256_file(ROOT / name) for name in PREDICTOR_SOURCE_FILES}


def _input_path(path):
    path = Path(path)
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("evaluation-input-symlink")
    return path.resolve()


def _groups(records):
    groups = [dict(method=METHOD, case="all", condition="all", **_metric_group(records))]
    for case in sorted({item["case"] for item in records}):
        groups.append(dict(method=METHOD, case=case, condition="all",
                           **_metric_group([item for item in records if item["case"] == case])))
    for case, condition in sorted({(item["case"], item["condition"]) for item in records}):
        groups.append(dict(method=METHOD, case=case, condition=condition,
                           **_metric_group([item for item in records
                                            if (item["case"], item["condition"]) == (case, condition)])))
    return groups


def evaluate_rgb_predictor(manifest_path, checkpoint_path, output_dir, *, device="cpu"):
    import cv2
    import numpy as np
    import torch

    manifest_path, checkpoint_path = map(_input_path, (manifest_path, checkpoint_path))
    manifest_hash, checkpoint_hash = map(sha256_file, (manifest_path, checkpoint_path))
    core_sources, predictor_sources = runtime_source_hashes(), predictor_source_hashes()
    device = _device(torch, device)
    learned, config, checkpoint = load_checkpoint(checkpoint_path, device=str(device),
                                                  manifest_sha256=manifest_hash,
                                                  source_sha256=core_sources)
    if (config.near_mean_target_m, config.far_mean_target_m) != (0.3, 1.0):
        raise ValueError("assisted-evaluation-requires-frozen-targets")
    learned.eval()
    dataset = LaneWaypointDataset(manifest_path, "dev", config=config, augment=False)
    baseline_config = load_baseline_config()
    predictor = BaselineAssistedWaypointPredictor(learned, baseline=OpenCVLanePredictor(baseline_config))
    output = _new_output(Path(output_dir))
    overlay_dir = output / "overlays"
    overlay_dir.mkdir()
    records, overlays, overlay_groups = [], [], set()
    counts = {"B0": 0, "ResNet18": 0, "failed": 0}
    learned_attempts = 0
    for sample in dataset.samples:
        # The dataset verifies the original PNG hash, shape and dtype. No
        # preprocessed image, target or route metadata reaches the predictor.
        rgb = dataset._read_image(sample)
        started = time.perf_counter()
        result = predictor.predict_rgb(rgb)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency = (time.perf_counter() - started) * 1000
        diagnostics = result.get("diagnostics", {})
        selected = diagnostics.get("selected_predictor")
        learned_attempts += bool(diagnostics.get("fallback_used"))
        points = result.get("waypoints_ego_m") if result.get("status") == "PASS" else None
        error = result.get("error")
        if points is not None:
            try:
                validate_waypoint_points(points)
                if selected not in ("B0", "ResNet18"):
                    raise LearningDataError("invalid-selected-predictor")
            except LearningDataError:
                points, selected, error = None, None, "invalid-assisted-output"
        if points is None:
            selected, error = None, error or "assisted-prediction-failed"
        counts[selected if selected is not None else "failed"] += 1
        # Privileged labels enter only after this frame's RGB prediction ends.
        target = torch.tensor(sample["waypoints_ego_m"], dtype=torch.float32).tolist()
        record = {key: sample[key] for key in ("sample_id", "case", "condition", "route_id", "image_sha256")}
        record.update(target=target, prediction=points, error=error if points is None else None,
                      method=METHOD, selected_predictor=selected, latency_ms=latency,
                      baseline_error=diagnostics.get("baseline_error"))
        if points is not None:
            record["errors_m"] = [math.hypot(p[0]-t[0], p[1]-t[1]) for p, t in zip(points, target)]
            record["x_errors_m"] = [abs(p[0]-t[0]) for p, t in zip(points, target)]
            record["y_errors_m"] = [abs(p[1]-t[1]) for p, t in zip(points, target)]
        records.append(record)
        group = (sample["case"], sample["condition"])
        if points is not None and group not in overlay_groups and len(overlays) < 15:
            image = rgb.copy()
            for path_points, color in ((target, (0, 255, 0)), (points, (255, 50, 50))):
                projected, front = nominal_ground_to_pixels(np.asarray(path_points), baseline_config)
                for pixel, visible in zip(projected, front):
                    if visible and np.isfinite(pixel).all():
                        x, y = (int(round(float(value))) for value in pixel)
                        if 0 <= x < 640 and 0 <= y < 360:
                            cv2.circle(image, (x, y), 3, color, -1)
            name = f"overlays/{len(records)-1:05d}.png"
            if not cv2.imwrite(str(output / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise OSError("overlay-write-failed")
            overlays.append({"sample_id": sample["sample_id"], "case": group[0], "condition": group[1],
                             "image_sha256": sample["image_sha256"], "overlay": name})
            overlay_groups.add(group)
    dataset.assert_manifest_unchanged()
    # Check consumed RGB again, and freeze both the trained core and the new
    # inference policy throughout the complete evaluation.
    for sample in dataset.samples:
        dataset._read_image(sample)
    if (sha256_file(manifest_path) != manifest_hash or sha256_file(checkpoint_path) != checkpoint_hash or
            runtime_source_hashes() != core_sources or predictor_source_hashes() != predictor_sources):
        raise ValueError("assisted-evaluation-input-or-source-changed")
    grouped = _groups(records)
    overall = grouped[0]
    target_passed = (overall["coverage"] == 1.0 and overall["near_mean_m"] is not None and
                     overall["far_mean_m"] is not None and overall["near_mean_m"] <= 0.3 and
                     overall["far_mean_m"] <= 1.0)
    report = {"schema_version": 1, "status": "PASS", "split": "dev", "final_town_used": False,
              "scope": "actual-RGB-inference; offline-development-waypoint-accuracy",
              "predictor_version": PREDICTOR_VERSION, "policy": "valid-B0-else-learned",
              "actual_inference_performed": True, "selection_uses_target_or_case": False,
              "learned_model_version": config.model_version,
              "manifest_sha256": manifest_hash, "checkpoint_sha256": checkpoint_hash,
              "config_sha256": config.sha256, "baseline_config_sha256": baseline_config.sha256,
              "source_sha256": core_sources, "predictor_source_sha256": predictor_sources,
              "inputs_and_sources_unchanged": True, "performance_target": dict(TARGET),
              "performance_target_passed": target_passed, "sample_count": len(records),
              "branch_counts": counts, "baseline_attempt_count": len(records),
              "learned_attempt_count": learned_attempts, "metrics": grouped, "predictions": records,
              "overlays": overlays,
              "overlay_legend": {"green": "privileged-target-for-QA-only", "red": "RGB-only-assisted-prediction"},
              "latency_scope": "total-RGB-predictor-call: input-check, B0, optional CNN preprocessing/transfer/forward; "
                               "excludes file decoding, integrity checks, target scoring and overlays",
              "runtime": {"device": str(device), "torch": torch.__version__, "numpy": np.__version__},
              "limitations": ["Development-set measurements do not establish final-town or closed-loop driving performance."]}
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    fields = [key for key in grouped[0] if key != "per_waypoint_mean_m"]
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group in grouped:
            writer.writerow({key: group[key] if group[key] is not None else "N/A" for key in fields})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    try:
        report = evaluate_rgb_predictor(args.manifest, args.checkpoint, args.output_dir, device=args.device)
        print(json.dumps({key: report[key] for key in ("status", "sample_count", "performance_target_passed", "branch_counts")}
                         | {"metrics": report["metrics"][0]}))
        return 0
    except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(json.dumps({"status": "FAIL", "code": str(error), "exception_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
