"""Compare matched continuation and appearance augmentation on RGB-only data."""
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
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/development_training.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    from carla_tools.development_training import load_development_protocol, run_development_comparison
    report = run_development_comparison(args.manifest, args.validation_report, args.initial_checkpoint,
        args.output_dir, protocol=load_development_protocol(args.protocol), device=args.device)
    print(json.dumps({"status": report["status"], "representative_seed": report["representative_seed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
