"""Evaluate a selected waypoint checkpoint on its held-out development RGB."""

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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args(argv)
    try:
        from carla_tools.training import evaluate_waypoints
        report = evaluate_waypoints(args.manifest, args.checkpoint, args.output_dir, device=args.device,
                                    compare_baseline=not args.no_baseline)
        print(json.dumps({key: report[key] for key in ("status", "sample_count", "performance_target_passed")}
                         | {"metrics": report["metrics"][0]}))
        return 0
    except (ImportError, OSError, ValueError, RuntimeError, KeyError) as error:
        print(json.dumps({"status": "FAIL", "code": str(error), "exception_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
