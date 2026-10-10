"""Independently replay learned CARLA RGB/control artifacts and checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from carla_tools.inference import InferenceError, _reject_links, new_output_path, write_prediction_json
from carla_tools.learned_validation import SCOPE, GATE_VERSION, validate_learned_episode, validate_learned_manifest
from carla_tools.driving import load_validation_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--manifest",type=Path)
    sources.add_argument("--episode",type=Path)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True,help="New JSON under results/local or docs/local")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--initial-reference-m",type=float,choices=(0.,2.,4.),default=0.,
                        help="Requested initial original-route arc for a single episode")
    args = parser.parse_args(argv)
    try:
        target = args.output.absolute()
        _reject_links(target)
        target = target.resolve()
        if not any(target.is_relative_to(root.resolve()) for root in (ROOT/"results/local",ROOT/"docs/local")):
            raise InferenceError("learned-validation-output-must-be-private")
        selected = (args.manifest if args.manifest is not None else args.episode).resolve()
        boundary = selected.parent if args.manifest is not None else selected
        if target.is_relative_to(boundary):
            raise InferenceError("learned-validation-output-inside-source")
        target.parent.mkdir(parents=True,exist_ok=True)
        target = new_output_path(target)
    except (InferenceError,OSError,ValueError) as error:
        print(json.dumps({"status":"FAIL","error":getattr(error,"code","learned-validation-output-unavailable")}))
        return 1
    try:
        if args.manifest is not None:
            report = validate_learned_manifest(args.manifest,args.checkpoint,device=args.device)
        else:
            report = validate_learned_episode(args.episode,args.checkpoint,device=args.device,
                allowed_conditions=load_validation_config()["initial_conditions"],
                expected_initial_reference_m=args.initial_reference_m)
    except BaseException as error:
        report = {"status":"FAIL","scope":SCOPE,"gate_version":GATE_VERSION,
                  "errors":[{"code":"learned-revalidation-interrupted","exception_type":type(error).__name__}]}
    try:
        write_prediction_json(target,report)
    except InferenceError as error:
        print(json.dumps({"status":"FAIL","error":error.code}))
        return 1
    print(json.dumps({key:report[key] for key in ("status","gate_version","episode_count","step_count","image_count","errors")
                      if key in report},indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
