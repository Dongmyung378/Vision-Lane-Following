"""Replay closed-loop episodes and exclusively publish metric CSV + aggregate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.contracts import ContractError, read_strict_json, strict_json_loads
from carla_tools.evaluation import (EvaluationError,
                                    evaluate_episode, load_evaluation_config, write_metric_csv)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, help="Optional override; must match each episode's frozen configuration.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_evaluation_config(args.config) if args.config is not None else None
        metrics, sources = [], set()
        for directory in args.episodes:
            directory = directory.resolve()
            native_metadata, generic_metadata = directory / "episode.json", directory / "metadata.json"
            if native_metadata.exists() and generic_metadata.exists():
                raise EvaluationError("ambiguous-episode-metadata")
            metadata_path = native_metadata if native_metadata.exists() else generic_metadata
            rows_path = directory / "steps.jsonl"
            sources.update((native_metadata, generic_metadata, rows_path, directory / "events.json"))
            try:
                metadata = read_strict_json(metadata_path)
            except ContractError:
                metadata = {"evaluation_route_length_m": None, "termination_reason": "technical_failure"}
            if type(metadata) is not dict:
                metadata = {"evaluation_route_length_m": None, "termination_reason": "technical_failure"}
            metadata = dict(metadata)
            metadata.setdefault("episode_id", directory.name)
            # Invalid/truncated row JSON is represented, not silently omitted.
            # Valid rows before the fault still supply distance and error samples.
            rows = []
            try:
                with rows_path.open(encoding="utf-8") as handle:
                    for line in handle:
                        rows.append(strict_json_loads(line))
            except (OSError, UnicodeError, ContractError):
                rows.append(None)
                metadata["termination_reason"] = "technical_failure"
            events = metadata.get("events", [])
            if (directory / "events.json").is_file():
                try:
                    events = read_strict_json(directory / "events.json")
                except ContractError:
                    events = [None]
            metrics.append(evaluate_episode(metadata, rows, events, config=config))
        if args.output.resolve() in sources:
            raise EvaluationError("metric-output-conflicts-with-source")
        aggregate = write_metric_csv(metrics, args.output)
        print(json.dumps({"status": "PASS", "episode_count": len(metrics), "aggregate": aggregate},
                         indent=2, allow_nan=False))
        return 0
    except (EvaluationError, ContractError, OSError) as error:
        print(json.dumps({"status": "FAIL", "code": str(error) if isinstance(error, EvaluationError) else "evaluation-input-read-failed"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
