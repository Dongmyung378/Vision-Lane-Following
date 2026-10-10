"""Train three controlled RGB waypoint candidates and evaluate the full DEV set."""

from __future__ import annotations

import argparse
import dataclasses
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--max-epochs", type=int, default=32)
    parser.add_argument("--reused-baseline", type=Path)
    args = parser.parse_args(argv)
    from carla_tools.learning_data import load_learning_config
    from carla_tools.spatial_training import run_spatial_comparison
    if not 1 <= args.max_epochs <= 64:
        parser.error("max-epochs must be between 1 and 64")
    config = dataclasses.replace(load_learning_config(), epochs=args.max_epochs)
    result = run_spatial_comparison(args.manifest, args.output_dir, device=args.device,
                                    config=config, weights_path=args.weights, reused_baseline_path=args.reused_baseline)
    print(json.dumps({"status": result["status"], "decision": result["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
