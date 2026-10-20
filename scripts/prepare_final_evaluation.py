"""Prepare a held-out evaluation manifest without starting a simulator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--artifact-bindings", required=True,
                        help="JSON array of project-relative artifact paths, hashes and roles.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        from carla_tools.contracts import read_strict_json
        from carla_tools.final_evaluation import build_manifest, FinalEvaluationError
        from carla_tools.inference import _sha256_file
        from carla_tools.selected_pipeline import safe_relative_file
        from scripts.predict_selected_pipeline import _write_result
        routes, selection, bindings = (read_strict_json(path) for path in
                                      (args.routes, args.selection, args.artifact_bindings))
        manifest = build_manifest(routes, selection, bindings)
        for row in bindings:
            source = safe_relative_file(ROOT, row["path"])
            if _sha256_file(source) != row["sha256"]:
                raise FinalEvaluationError("final-artifact-hash-mismatch")
        _write_result(args.output, manifest)
        print(json.dumps({"status": "PASS", "expected_episodes": manifest["expected_episodes"],
                          "final_results_observed": False, "GPU_used": False}))
        return 0
    except (ValueError, OSError, TypeError, KeyError) as error:
        print(json.dumps({"status": "FAIL", "error_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
