"""Predict actor-origin metre waypoints from one original RGB8 PNG."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.inference import (
    InferenceError, OfflineWaypointPredictor, _reject_links, new_output_path, write_prediction_json,
)


def _private_output(path):
    try:
        path = Path(path).absolute()
        _reject_links(path)
        path = path.resolve()
        private_roots = (ROOT / "results/local", ROOT / "docs/local")
        if not any(path.is_relative_to(root.resolve()) for root in private_roots):
            raise InferenceError("inference-output-must-be-private")
        path.parent.mkdir(parents=True, exist_ok=True)
        _reject_links(path)
        return new_output_path(path)
    except InferenceError:
        raise
    except (OSError, ValueError, TypeError):
        raise InferenceError("inference-output-parent-unavailable") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file under results/local or docs/local; parents are created")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=("learned", "assisted"), default="learned")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args(argv)
    try:
        output = _private_output(args.output)
    except InferenceError as error:
        print(json.dumps({"status": "FAIL", "waypoints_ego_m": None, "error": error.code}))
        return 1
    try:
        predictor = OfflineWaypointPredictor(args.checkpoint, device=args.device, mode=args.mode,
                    expected_checkpoint_sha256=args.expected_checkpoint_sha256,
                    expected_manifest_sha256=args.expected_manifest_sha256)
        report = predictor.predict_png(args.image)
    except InferenceError as error:
        report = {"status": "FAIL", "waypoints_ego_m": None, "error": error.code,
                  "diagnostics": {"mode": args.mode, "selected_predictor": None}, "timing_ms": 0.0,
                  "input": None, "input_sha256": None, "model": None}
    report["cli_source_sha256"] = {"scripts/predict_waypoints.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        write_prediction_json(output, report)
    except InferenceError as error:
        print(json.dumps({"status": "FAIL", "waypoints_ego_m": None, "error": error.code}))
        return 1
    print(json.dumps({key: report[key] for key in ("status", "waypoints_ego_m", "error")}, allow_nan=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
