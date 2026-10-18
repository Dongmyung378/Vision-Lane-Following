"""Compare source-bound central and recovery data at an equal update budget."""
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
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/recovery_training.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    from carla_tools.recovery_training import load_recovery_protocol, run_recovery_comparison
    report = run_recovery_comparison(args.manifest, args.validation_report,
        args.initial_checkpoint, args.output_dir,
        protocol=load_recovery_protocol(args.protocol), device=args.device)
    print(json.dumps({"status": report["status"], "representative_seed": report["representative_seed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
