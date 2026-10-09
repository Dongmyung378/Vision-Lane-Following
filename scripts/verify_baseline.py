"""Revalidate camera baseline RGB/control/evaluation artifacts without a GPU."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.baseline_validation import SCOPE, validate_baseline_episode, validate_baseline_manifest
from scripts.verify_collection import write_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--manifest", type=Path)
    sources.add_argument("--episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    selected = args.manifest if args.manifest is not None else args.episode
    target, source = args.output.resolve(), selected.resolve()
    # The whole manifest directory contains source artifacts, including CSVs,
    # previews and failed trials that may not be listed in a malformed manifest.
    boundary = source.parent if args.manifest is not None else source
    if target.is_relative_to(boundary):
        parser.error("output must be outside the source artifact directory")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"status": "FAIL", "scope": SCOPE,
                                     "errors": [{"code": "baseline-revalidation-incomplete"}]}) + "\n")
    except FileExistsError:
        parser.error("output already exists")
    try:
        report = validate_baseline_manifest(args.manifest) if args.manifest is not None else validate_baseline_episode(args.episode)
    except BaseException as error:
        report = {"status": "FAIL", "scope": SCOPE, "errors": [{"code": "baseline-revalidation-interrupted",
                                                                "exception_type": type(error).__name__}]}
    write_report(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "gate_version", "episode_count", "step_count", "image_count", "errors")
                      if key in report}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
