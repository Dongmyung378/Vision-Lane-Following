"""Execute selected groups of an externally identified held-out freeze.

This command performs native driving only with --execute-heldout. Planning and
freeze verification are separate CPU commands; importing this script does not
open final imagery, import CARLA or load a model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_verified_inputs(proof, project_root):
    from carla_tools.contracts import strict_json_loads
    from carla_tools.selected_pipeline import safe_relative_file
    result = {}
    for name, binding in proof["verified_inputs"].items():
        path = safe_relative_file(project_root, binding["path"])
        payload = path.read_bytes()
        if len(payload) != binding["bytes"] or hashlib.sha256(payload).hexdigest() != binding["sha256"]:
            raise ValueError("heldout-verified-input-changed:" + name)
        result[name] = path if name == "checkpoint" else strict_json_loads(payload.decode("utf-8"))
        if path.read_bytes() != payload:
            raise ValueError("heldout-input-changed-during-parse:" + name)
    return result


def prepare_native_world(carla_module, collection, route):
    """Load only the declared map, refusing an occupied or synchronous world."""
    client = carla_module.Client(collection.connection.host, collection.connection.port)
    client.set_timeout(collection.connection.timeout_seconds)
    if client.get_client_version() != "0.9.16" or client.get_server_version() != "0.9.16":
        raise ValueError("heldout-native-version-mismatch")
    world = client.get_world()
    def require_idle(candidate):
        if candidate.get_settings().synchronous_mode or any(
            actor.type_id.startswith(("vehicle.", "walker.", "controller.ai.walker", "sensor."))
            for actor in candidate.get_actors()):
            raise ValueError("heldout-busy-world-refused")
    require_idle(world)
    if world.get_map().name.replace("\\", "/").rsplit("/", 1)[-1] != route["town"]:
        world = client.load_world(route["town"])
        require_idle(world)
    live = world.get_map()
    if (live.name.replace("\\", "/").rsplit("/", 1)[-1] != route["town"] or
        hashlib.sha256(live.to_opendrive().encode("utf-8")).hexdigest() != route["opendrive_sha256"]):
        raise ValueError("heldout-live-map-identity-mismatch")
    return live


def run_pipeline(output, freeze_path, *, project_root=ROOT, expected_freeze_id,
                 groups, device="cpu", execute_heldout=False):
    from carla_tools.evaluation_freeze import verify_freeze
    from carla_tools.final_evaluation import validate_manifest
    if execute_heldout is not True:
        raise ValueError("heldout-explicit-execution-required")
    if type(expected_freeze_id) is not str or re.fullmatch("[0-9a-f]{64}", expected_freeze_id) is None:
        raise ValueError("heldout-external-freeze-identity-required")
    if device not in ("cpu", "cuda") or (type(groups) not in (tuple, list)) or not groups or (
        len(groups) != len(set(groups)) or any(group not in "ABCDE" or len(group) != 1 for group in groups)):
        raise ValueError("heldout-device-or-group-selection-invalid")
    output, project_root = Path(output), Path(project_root)
    if output.exists() or output.is_symlink():
        raise ValueError("heldout-new-output-required")
    proof = verify_freeze(freeze_path, project_root, expected_freeze_id=expected_freeze_id)
    inputs = read_verified_inputs(proof, project_root)
    manifest, routes, selection = (inputs[key] for key in ("manifest", "routes", "selection"))
    validate_manifest(manifest, routes, selection)
    trials = [trial for trial in manifest["trials"] if trial["group_id"] in groups]
    # All contract and source checks precede model/device or CARLA access.
    from carla_tools.baseline import baseline_config_from_dict
    from carla_tools.development_study import ProfiledBaselinePredictor
    from carla_tools.final_evaluation_driving import (collection_for_trial, run_heldout_trial, validate_heldout_episode)
    from carla_tools.guarded_path_correction import wrap_preprocessing
    from carla_tools.runtime_driving import RuntimeCheckpointPredictor
    from carla_tools.selected_pipeline import _validate_predictor
    from scripts.verify_collection import write_report
    base = RuntimeCheckpointPredictor(inputs["checkpoint"], expected_sha256=selection["model"]["checkpoint_sha256"],
                                      device=device, correction="path")
    model = wrap_preprocessing(base, {key: value for key, value in manifest["preprocessing"].items() if key != "input"})
    _validate_predictor(model, selection)
    predictors = {"M1": model, "B0": ProfiledBaselinePredictor(baseline_config_from_dict(manifest["baseline_config"])),
                  "privileged": None}
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "RUNNING", "freeze": proof,
              "selected_pipeline_id": selection["pipeline_id"], "groups": list(groups), "device": device,
              "planned_trials": trials, "expected_episode_count": len(trials), "episodes": [],
              "episode_count": 0, "frame_count": 0, "closed_loop_executed": False,
              "complete_final_matrix": False, "selection_changed": False}
    started = time.monotonic()
    try:
        carla = importlib.import_module("carla")
        for trial in trials:
            if verify_freeze(freeze_path, project_root, expected_freeze_id=expected_freeze_id) != proof:
                raise ValueError("heldout-freeze-changed-before-trial")
            route = next(route for route in routes["routes"] if route["route_id"] == trial["route_id"])
            record = dict(trial, execution={"status": "NOT_RUN"}, gate={"status": "NOT_RUN"})
            report["episodes"].append(record)
            report["active_trial"] = trial["trial_id"]
            write_report(output / "native_report.json", report)
            collection = collection_for_trial(manifest, trial, 0)
            live_map = prepare_native_world(carla, collection, route)
            predictor = predictors[trial["method"]]
            record["execution"] = run_heldout_trial(route, trial, output / trial["episode"],
                manifest=manifest, selection=selection, predictor=predictor, carla_module=carla, carla_map=live_map)
            if record["execution"]["status"] == "PASS":
                record["gate"] = validate_heldout_episode(output / trial["episode"], predictor, trial,
                    manifest=manifest, routes=routes, selection=selection)
            else:
                record["gate"] = {"status": "FAIL", "errors": ["heldout-native-operation-failed"]}
            report["episode_count"] += 1
            report["frame_count"] += record["gate"].get("frame_count", 0)
            report["closed_loop_executed"] |= record["execution"].get("step_count", 0) > 0
            write_report(output / "native_report.json", report)
            print(json.dumps({"event": "heldout-episode", "trial_id": trial["trial_id"],
                "index": report["episode_count"], "expected": len(trials), "integrity": record["gate"]["status"],
                "metrics": record["gate"].get("metrics")}), flush=True)
            if record["gate"]["status"] != "PASS":
                raise ValueError("heldout-native-integrity-failed")
        final_proof = verify_freeze(freeze_path, project_root, expected_freeze_id=expected_freeze_id)
        if final_proof != proof:
            raise ValueError("heldout-freeze-changed-during-execution")
        report.update(status="PASS", complete_final_matrix=len(trials) == manifest["expected_episodes"])
    except Exception as error:
        report.update(status="FAIL", error={"type": type(error).__name__, "code": str(error)})
        if report["episodes"] and report["episodes"][-1]["execution"]["status"] == "NOT_RUN":
            report["episodes"][-1].update(execution={"status": "BLOCKED", "error": report["error"]},
                                         gate={"status": "FAIL", "errors": ["heldout-trial-blocked-before-driving"]})
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_report(output / "native_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-id", required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--groups", nargs="+", choices=tuple("ABCDE"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--execute-heldout", action="store_true")
    args = parser.parse_args(argv)
    report = run_pipeline(args.output_dir, args.freeze, project_root=args.project_root,
        expected_freeze_id=args.expected_freeze_id, groups=args.groups, device=args.device,
        execute_heldout=args.execute_heldout)
    print(json.dumps({key: report[key] for key in ("status", "episode_count", "frame_count", "complete_final_matrix")}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
