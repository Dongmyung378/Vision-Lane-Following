"""Revalidate immutable moving-pilot images, labels and scene groups on CPU."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.pilot import SCOPE, validate_pilot_manifest
from scripts.verify_collection import write_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    boundary = args.manifest.resolve().parent
    if args.output.resolve().is_relative_to(boundary):
        parser.error("output must be outside the source artifact directory")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump({"status": "FAIL", "scope": SCOPE, "errors": [{"code": "pilot-revalidation-incomplete"}]}, handle)
    report = validate_pilot_manifest(args.manifest)
    write_report(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "sample_count", "split_counts", "route_count", "errors") if key in report}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
