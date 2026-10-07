"""Build a D5 episode/route/scene grouping manifest from verified raw episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.contracts import DEFAULT_CONTRACT_PATH, ContractError, read_strict_json
from carla_tools.splits import build_split_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True, help="JSON object mapping episode IDs to train/dev.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        assignments = read_strict_json(args.assignments)
    except (OSError, ValueError, ContractError):
        print(json.dumps({"status": "FAIL", "code": "invalid-split-assignments"}))
        return 1
    report = build_split_manifest(args.episodes, args.output, assignments, contract_path=args.contract)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
