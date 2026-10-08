"""Revalidate D7 alignment from persisted raw episodes and derived labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.contracts import read_strict_json
from carla_tools.week1 import verify_week1_cases
from scripts.verify_collection import write_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Case/episode/label paths relative to this JSON file.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_report(args.output, {"status": "RUNNING", "errors": [], "scope": "D7-offline-revalidation"})
    try:
        manifest = read_strict_json(args.manifest)
        if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "cases"}
                or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
                or not isinstance(manifest["cases"], list)):
            raise ValueError("week1-manifest-schema-invalid")
        for item in manifest["cases"]:
            if (not isinstance(item, dict) or set(item) != {"case", "episode", "labels"}
                    or any(not isinstance(item[key], str) or not item[key].strip() for key in item)):
                raise ValueError("week1-manifest-case-invalid")
        cases = [{"case": item["case"], "episode": args.manifest.parent / item["episode"],
                  "labels": args.manifest.parent / item["labels"]} for item in manifest["cases"]]
        report = verify_week1_cases(cases)
    except BaseException as error:
        report = {"status": "FAIL", "errors": [{"code": "week1-manifest-invalid", "exception_type": type(error).__name__}]}
    write_report(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "gate_version", "episode_count", "frame_count", "geometry_cases", "errors") if key in report}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
