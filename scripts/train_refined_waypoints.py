"""Train two compact hybrid refinements against a frozen ResNet reference."""
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
    parser.add_argument("--reused-baseline", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-epochs", type=int, default=32)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.max_epochs <= 64:
        parser.error("max-epochs must be between 1 and 64")
    from carla_tools.learning_data import load_learning_config
    from carla_tools.refined_training import run_refinements
    report = run_refinements(args.manifest, args.output_dir, args.reused_baseline,
        device=args.device, config=dataclasses.replace(load_learning_config(), epochs=args.max_epochs), weights_path=args.weights)
    print(json.dumps({"status": report["status"], "decision": report["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
