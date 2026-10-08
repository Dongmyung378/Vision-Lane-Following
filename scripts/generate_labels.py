"""Generate or verify a new D6 waypoint sidecar without changing raw episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.contracts import DEFAULT_CONTRACT_PATH
from carla_tools.labels import generate_labels, validate_labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", "--labels", dest="labels", type=Path, required=True,
                        help="New output file for generation; existing sidecar for --validate-only.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    operation = validate_labels if args.validate_only else generate_labels
    report = operation(args.episode, args.labels, args.contract)
    summary = {key: report[key] for key in ("status", "scope", "source", "metrics", "labels_sha256", "written", "errors") if key in report}
    summary["operation"] = "validate-existing-derived-labels" if args.validate_only else "generate-new-derived-labels"
    summary["raw_episode_modified"] = False
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
