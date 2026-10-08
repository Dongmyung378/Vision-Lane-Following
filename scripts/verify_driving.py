"""Revalidate persisted D8-D10 native controller/recovery QA without a GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.driving_validation import validate_drive_episode, validate_driving_manifest
from carla_tools.contracts import read_strict_json
from scripts.verify_collection import write_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    target = args.output.resolve()
    selected = args.manifest if args.manifest is not None else args.episode
    if args.manifest is not None and target == selected.resolve() or args.episode is not None and target.is_relative_to(selected.resolve()):
        parser.error("output must not overwrite a source artifact")
    if args.manifest is not None:
        try:
            manifest = read_strict_json(args.manifest)
        except Exception:
            manifest = None
        if isinstance(manifest, dict) and isinstance(manifest.get("episodes"), list):
            for item in manifest["episodes"]:
                if isinstance(item, dict) and isinstance(item.get("episode"), str):
                    if target.is_relative_to((args.manifest.parent / item["episode"]).resolve()):
                        parser.error("output must not overwrite a source artifact")
    write_report(args.output, {"status": "RUNNING", "scope": "D8-D10-offline-revalidation", "errors": []})
    try:
        report = validate_driving_manifest(args.manifest) if args.manifest is not None else validate_drive_episode(args.episode)
    except BaseException as error:
        report = {"status": "FAIL", "errors": [{"code": "driving-revalidation-interrupted", "exception_type": type(error).__name__}]}
    write_report(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "gate_version", "episode_count", "step_count", "image_count", "errors") if key in report}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
