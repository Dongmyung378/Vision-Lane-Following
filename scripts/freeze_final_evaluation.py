"""Create or verify a write-once final evaluation freeze without running it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.evaluation_freeze import build_freeze, verify_freeze, write_freeze


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Bind the plan, model, provenance and complete public sources.")
    create.add_argument("--project-root", type=Path, default=ROOT)
    for name in ("manifest", "routes", "selection", "development-review", "output"):
        create.add_argument("--" + name, type=Path, required=True)
    create.add_argument("--artifact", type=Path, action="append", required=True,
        help="Repeat for every manifest evidence artifact; no image archive is scanned.")
    create.add_argument("--source", type=Path, action="append", default=None,
        help="Optional explicit source list; must cover the automatic public source closure.")
    verify = commands.add_parser("verify", help="Recheck every bound file and the current source inventory.")
    verify.add_argument("--project-root", type=Path, default=ROOT)
    verify.add_argument("--freeze", type=Path, required=True)
    verify.add_argument("--expected-freeze-id", help="Retain independently to check external identity.")
    args = parser.parse_args(argv)
    if args.command == "create":
        value = build_freeze(args.project_root, args.manifest, args.routes, args.selection,
            args.artifact, args.source, development_review_path=args.development_review)
        # Keep the published artifact in the same project as its portable paths.
        from carla_tools.evaluation_freeze import _relative, _root
        root = _root(args.project_root)
        output = root / _relative(root, args.output)
        report = write_freeze(output, value)
    else:
        report = verify_freeze(args.freeze, args.project_root, expected_freeze_id=args.expected_freeze_id)
    print(json.dumps(report, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
