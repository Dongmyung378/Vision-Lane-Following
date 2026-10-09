"""Train the RGB waypoint model using a frozen train/dev pilot manifest."""

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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--max-epochs", type=int,
                        help="Stop at this completed epoch without changing the frozen total-epoch schedule.")
    args = parser.parse_args(argv)
    try:
        from carla_tools.learning_data import load_learning_config
        from carla_tools.training import train_waypoints
        config = load_learning_config(args.config) if args.config else load_learning_config()
        report = train_waypoints(args.manifest, args.output_dir, config=config, device=args.device,
                                 resume=args.resume, weights_path=args.weights, max_epochs=args.max_epochs)
        print(json.dumps({key: report[key] for key in ("status", "completed_epochs", "best_dev_loss",
                                                       "checkpoint_sha256")}))
        return 0
    except (ImportError, OSError, ValueError, RuntimeError, KeyError) as error:
        print(json.dumps({"status": "FAIL", "code": str(error), "exception_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
