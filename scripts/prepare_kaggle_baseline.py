"""Prepare the camera baseline for the same existing CARLA GPU notebook."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_kaggle_week1 import prepare


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-metadata", type=Path, required=True,
                        help="Metadata obtained by a read-only pull of the existing notebook.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.existing_metadata, args.output_dir, workload="baseline")
        print(json.dumps({key: result[key] for key in (
            "notebook_id", "title", "workload", "code_sha256", "submitted", "creates_new_notebook")}, indent=2))
        return 0
    except (OSError, ValueError, SyntaxError, StopIteration) as error:
        print(json.dumps({"status": "FAIL", "code": "baseline-package-preparation-failed",
                          "exception_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
