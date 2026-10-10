"""Verify a frozen DINOv3 waypoint candidate on the existing train/dev pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--expected-weights-sha256")
    parser.add_argument("--weights-url")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args(argv)
    try:
        from carla_tools.contracts import read_strict_json
        from carla_tools.vit_waypoint import VitConfig, vit_config_from_dict
        from carla_tools.vit_training import PRETRAINED_SHA256, PRETRAINED_URL, train_vit_waypoints
        config = vit_config_from_dict(read_strict_json(args.config)) if args.config else VitConfig()
        report = train_vit_waypoints(args.manifest, args.output_dir, weights_path=args.weights,
                                     expected_weights_sha256=args.expected_weights_sha256 or PRETRAINED_SHA256,
                                     weights_url=args.weights_url or PRETRAINED_URL,
                                     config=config, device=args.device, max_seconds=args.max_seconds,
                                     max_epochs=args.max_epochs)
        print(json.dumps({name: report[name] for name in ("status", "completed_epochs", "selected_epoch", "best_dev_loss",
                                                         "checkpoint_sha256", "evaluation")}, allow_nan=False))
        return 0
    except (ImportError, OSError, ValueError, RuntimeError, KeyError) as error:
        # This command never makes network requests. Do not include dependency
        # tracebacks or HTTP exception representations in its public output.
        print(json.dumps({"status": "FAIL", "code": str(error), "exception_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
