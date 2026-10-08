"""Render a fresh same-frame waypoint overlay from independently validated D6 labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.projection import render_episode_overlay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory outside the raw episode.")
    parser.add_argument("--frame", type=int, help="Simulator frame; defaults to the first labelled frame.")
    args = parser.parse_args(argv)
    report = render_episode_overlay(args.episode, args.labels, args.output_dir, args.frame)
    print(json.dumps({key: report[key] for key in ("status", "episode_id", "route_id", "frame", "visible_points", "all_points_in_front", "errors") if key in report}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
