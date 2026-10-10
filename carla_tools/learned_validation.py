"""Independent RGB/checkpoint and native control replay of learned trials.

A PASS certifies persisted sources and replay, not successful driving. Failed
driving episodes remain in the metric denominator and retain their RGB files.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import math
from pathlib import Path, PurePosixPath

from .baseline_validation import _safe_file, _validate_camera_episode
from .contracts import canonical_sha256, strict_json_loads
from .driving import load_validation_config, route_initial_location
from .driving_route import _transform, perturbation_transform
from .driving_validation import _finite, _require, _same
from .evaluation import CSV_FIELDS, aggregate_metrics, evaluate_episode, evaluation_config_from_dict
from .geometry_survey import CASES, load_geometry_catalog
from .projection import decode_rgb8_png

SCOPE = "learned-camera-only-development-evaluation"
GATE_VERSION = "vlf-learned-camera-native-replay-v1"
METHOD_MODES = {"M1": "learned", "M1-assisted": "assisted"}
REPLAY_TOLERANCE_M = 1e-4
ROOT = Path(__file__).resolve().parents[1]
MODEL_IDENTITY_FIELDS = ("inference_version", "mode", "checkpoint_role", "checkpoint_sha256", "checkpoint_epoch",
                         "selected_epoch", "manifest_sha256", "config_sha256", "model_version", "config",
                         "model_contract", "model_contract_sha256", "source_sha256", "inference_source_sha256",
                         "pretrained_provenance")


def _sha_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _no_links(path):
    path = Path(path).absolute()
    _require(all(not node.is_symlink() and not getattr(node, "is_junction", lambda: False)()
                 for node in (path, *path.parents)), "learned-source-symlink-or-junction")
    return path


def learned_runtime_source_hashes():
    """Bind the runner bytes separately from the six trained model sources."""
    from .learned_driving import learned_driving_source_hashes
    sources = learned_driving_source_hashes()
    _require(isinstance(sources,dict) and sources and all(isinstance(key,str) and isinstance(value,str)
             for key,value in sources.items()), "learned-driving-source-closure-invalid")
    return sources


def _initialization(metadata, expected_initial_reference_m):
    reference = _finite(expected_initial_reference_m)
    _require(reference in (0., 2., 4.) and _finite(metadata.get("initial_reference_m")) == reference,
             "learned-initial-reference-mismatch")
    initial = metadata.get("initialization")
    _require(isinstance(initial, dict), "learned-initialization-missing")
    expected_keys = {"policy", "spawn_settled_transform", "center_requested_transform", "perturbed_requested_transform"}
    if reference == 0.:
        _require(set(initial) == expected_keys and initial["policy"] ==
                 "pinned-route-start-xy-and-tangent-preserve-settled-height", "learned-initialization-policy-mismatch")
    else:
        _require(set(initial) == expected_keys | {"initial_reference_m"} and initial["policy"] ==
                 "pinned-original-route-reference-xy-and-tangent-preserve-settled-height" and
                 _finite(initial["initial_reference_m"]) == reference, "learned-initialization-policy-mismatch")
    location, tangent = route_initial_location(metadata["route_contract"], reference)
    settled, center, moved = [_transform(initial[key]) for key in
                              ("spawn_settled_transform", "center_requested_transform", "perturbed_requested_transform")]
    _require(all(abs(center["location"][axis]-location[axis]) <= .002 for axis in ("x", "y")) and
             abs(center["location"]["z"]-settled["location"]["z"]) <= .002 and
             abs((center["rotation"]["yaw"]-tangent+180.) % 360.-180.) <= .001 and
             all(abs(center["rotation"][axis]-settled["rotation"][axis]) <= .001 for axis in ("pitch", "roll")),
             "learned-requested-original-route-pose-mismatch")
    requested = perturbation_transform(center, metadata["condition"]["lateral_m"], metadata["condition"]["yaw_deg"])
    _require(_same(moved, requested, 1e-4), "learned-requested-perturbation-mismatch")


def _model_identity(metadata, runtime):
    recorded = metadata.get("learned_model")
    actual = runtime.summary
    _require(isinstance(recorded, dict) and set(recorded) == set(actual), "learned-model-summary-fields-mismatch")
    _require(all(_same(recorded.get(key), actual.get(key), 1e-12) for key in MODEL_IDENTITY_FIELDS),
             "learned-model-checkpoint-or-source-mismatch")
    _require(recorded["checkpoint_role"] == "selected-model" and
             type(recorded["selected_epoch"]) is int and recorded["selected_epoch"] >= 1 and
             METHOD_MODES[metadata["method"]] == recorded["mode"], "learned-method-or-selected-model-mismatch")
    _require(recorded["model_contract_sha256"] == canonical_sha256(recorded["model_contract"]) and
             recorded["model_contract"]["input"]["source"] == "RGB-only" and
             (recorded["model_contract"]["input"]["source_width"], recorded["model_contract"]["input"]["source_height"]) == (640, 360) and
             all(recorded["model_contract"]["input"][key] is False for key in
                 ("speed_input", "pose_input", "map_input", "per_frame_calibration_input")) and
             recorded["model_contract"]["output"]["shape"] == [5, 2] and
             recorded["model_contract"]["output"]["origin"] == "vehicle-actor-origin" and
             recorded["model_contract"]["output"]["units"] == "meters", "learned-RGB-actor-waypoint-contract-mismatch")
    _require(isinstance(recorded.get("runtime"), dict) and recorded["runtime"].get("device") in ("cpu", "cuda"),
             "learned-runtime-record-invalid")
    sources = learned_runtime_source_hashes()
    _require(metadata.get("learned_driving_source_sha256") == sources, "learned-driving-source-mismatch")
    return sources


class _RecordedPredictionReplay:
    """Sequence-bound saved results used only after independent RGB reproduction.

    Native controller state is reconstructed from the original recorded floats.
    A fixed tight CPU/GPU reproduction tolerance does not accumulate through
    stateful steering or change the saved controls used for evaluation.
    """

    def __init__(self, frames):
        self.frames, self.index = frames, 0

    def predict(self, rgb, *, width, height):
        _require(self.index < len(self.frames), "learned-native-replay-extra-frame")
        expected = self.frames[self.index]
        _require((width, height) == (640, 360) and hashlib.sha256(rgb).hexdigest() == expected["rgb_sha256"],
                 "learned-native-replay-RGB-sequence-mismatch")
        self.index += 1
        return copy.deepcopy(expected["perception"])


def _compare_perception(observed, replayed):
    _require(isinstance(observed, dict) and set(observed) == set(replayed) and
             observed.get("status") in ("PASS", "FAIL") and
             type(observed.get("timing_ms")) in (int, float) and math.isfinite(observed["timing_ms"]) and observed["timing_ms"] >= 0,
             "learned-perception-record-invalid")
    # API diagnostics are deterministic and contain no GT/case/pose fields.
    _require(_same(observed.get("diagnostics"), replayed.get("diagnostics"), 1e-8) and
             observed.get("status") == replayed.get("status") and observed.get("error") == replayed.get("error"),
             "learned-RGB-status-or-diagnostics-replay-mismatch")
    if observed["status"] == "FAIL":
        _require(observed.get("waypoints_ego_m") is None and replayed.get("waypoints_ego_m") is None,
                 "learned-failure-retained-waypoint")
        return 0.
    from .learning_data import validate_waypoint_points
    validate_waypoint_points(observed.get("waypoints_ego_m"))
    validate_waypoint_points(replayed.get("waypoints_ego_m"))
    difference = max(abs(a-b) for row_a,row_b in zip(observed["waypoints_ego_m"],replayed["waypoints_ego_m"])
                     for a,b in zip(row_a,row_b))
    _require(difference <= REPLAY_TOLERANCE_M, "learned-RGB-waypoint-replay-mismatch")
    return difference


def validate_learned_episode(episode_path: str | Path, checkpoint_path: str | Path, *,
                             allowed_conditions=None, expected_initial_reference_m=0.0, device="cpu") -> dict:
    """Restore a pinned checkpoint and replay every original frame before GT QA."""
    root = Path(episode_path)
    report = {"status":"FAIL", "scope":SCOPE, "gate_version":GATE_VERSION, "errors":[]}
    try:
        from .inference import OfflineWaypointPredictor
        import numpy as np
        _no_links(root)
        checkpoint = _no_links(checkpoint_path)
        _require(checkpoint.is_file(), "learned-checkpoint-missing")
        metadata_path, trace_path = _safe_file(root,"episode.json"), _safe_file(root,"steps.jsonl")
        metadata_bytes, trace_bytes = metadata_path.read_bytes(), trace_path.read_bytes()
        metadata = strict_json_loads(metadata_bytes.decode("utf-8"))
        _require(isinstance(metadata, dict) and metadata.get("method") in METHOD_MODES and metadata.get("scope") == SCOPE and
                 metadata.get("status") == "COMPLETE" and metadata.get("prediction_inputs") == "rgb-only" and
                 metadata.get("perception_uses_route") is False and metadata.get("perception_uses_vehicle_pose") is False,
                 "learned-episode-method-scope-or-input-mismatch")
        _initialization(metadata,expected_initial_reference_m)
        stored_model = metadata.get("learned_model")
        _require(isinstance(stored_model, dict), "learned-model-summary-missing")
        runtime = OfflineWaypointPredictor(checkpoint, device=device, mode=METHOD_MODES[metadata["method"]],
            expected_checkpoint_sha256=stored_model.get("checkpoint_sha256"),
            expected_manifest_sha256=stored_model.get("manifest_sha256"))
        driving_sources = _model_identity(metadata,runtime)
        checkpoint_sha = _sha_file(checkpoint)
        lines = trace_bytes.decode("utf-8").splitlines()
        _require(lines and all(line.strip() for line in lines), "learned-trace-empty-or-blank")
        rows = [strict_json_loads(line) for line in lines]
        _require(len(rows) == metadata.get("step_count"), "learned-step-count-mismatch")
        reproduced, maximum_difference = [], 0.
        for row in rows:
            _require(isinstance(row,dict) and isinstance(row.get("image"),dict), "learned-frame-image-missing")
            image = row["image"]
            _require(type(row.get("frame")) is int and image.get("path") == f"images/{row['frame']:08d}.png",
                     "learned-frame-image-path-mismatch")
            png = _safe_file(root,image["path"]).read_bytes()
            width,height,rgb_bytes = decode_rgb8_png(png,image["sha256"],(640,360))
            rgb = np.frombuffer(rgb_bytes,dtype=np.uint8).reshape(height,width,3).copy()
            # No labels, transform, route, condition, frame number or speed reach
            # this call. Native evaluation is replayed only after this pass.
            result = runtime.predict_rgb(rgb)
            perception = {key:value for key,value in result.items() if key != "timing_ms"}
            perception["timing_ms"] = result["timing_ms"]
            maximum_difference = max(maximum_difference,_compare_perception(row.get("perception"),perception))
            reproduced.append({"rgb_sha256":hashlib.sha256(rgb_bytes).hexdigest(), "perception":row["perception"]})
        replay = _RecordedPredictionReplay(reproduced)
        native = _validate_camera_episode(root,allowed_conditions=allowed_conditions,scope=SCOPE,
            methods=tuple(METHOD_MODES),prediction_inputs="rgb-only",predictor=replay,
            expected_initial_reference_m=expected_initial_reference_m)
        _require(native["status"] == "PASS", "learned-native-controller-or-evaluation-gate-failed")
        _require(replay.index == len(reproduced), "learned-native-replay-frame-count-mismatch")
        from .training import runtime_source_hashes
        from .inference import _inference_source_hashes
        _require(metadata_path.read_bytes() == metadata_bytes and trace_path.read_bytes() == trace_bytes and
                 _sha_file(checkpoint) == checkpoint_sha and learned_runtime_source_hashes() == driving_sources and
                 runtime_source_hashes() == stored_model["source_sha256"] and
                 _inference_source_hashes(METHOD_MODES[metadata["method"]]) == stored_model["inference_source_sha256"],
                 "learned-source-changed-during-validation")
        report.update(status="PASS",source={**native["source"],"initial_reference_m":expected_initial_reference_m,
            "learned_model":stored_model,"learned_driving_source_sha256":driving_sources,
            "checkpoint_sha256":checkpoint_sha,"trained_core6_sha256":stored_model["source_sha256"]},
            metrics={**native["metrics"],"independently_replayed_RGB_count":len(reproduced),
                     "maximum_RGB_replay_coordinate_difference_m":maximum_difference,"RGB_replay_tolerance_m":REPLAY_TOLERANCE_M},
            evaluation_metrics=native["evaluation_metrics"],checks={**native["checks"],"checkpoint_and_model_identity":True,
                "actual_original_RGB_inference":True,"GT_not_supplied_to_predictor":True,
                "requested_original_route_initial_pose":True,"original_checkpoint_and_driving_sources_unchanged":True},
            replay_runtime=runtime.summary["runtime"],final_test_used=False,held_out_performance_certified=False,
            learned_driving_success=native["evaluation_metrics"]["driving_success"])
    except BaseException as error:
        report["errors"].append({"code":getattr(error,"code",str(error) if isinstance(error,ValueError) else
                                               "learned-episode-validation-failed"),"exception_type":type(error).__name__})
        if "native" in locals() and native.get("status") != "PASS":
            report["native_gate"] = native
    return report


IDENTITY_KEYS = {"method", "case", "condition", "initial_reference_m", "route_id", "episode"}
EPISODE_KEYS = IDENTITY_KEYS | {"status", "step_count", "episode_json_sha256", "steps_jsonl_sha256"}
CONFIG_KEYS = {"collection_config", "controller_config", "controller_config_sha256", "baseline_config",
               "baseline_config_sha256", "evaluation_config", "evaluation_config_sha256", "validation_config",
               "initial_reference_m", "initial_conditions", "geometry_catalog_sha256", "goal_reference_m", "comparison_scope"}
MANIFEST_KEYS = {"schema_version", "scope", "status", "mode", "model", "config", "driving_source_sha256",
                 "planned_trials", "episodes", "remaining_trials"}


def _identity(item):
    _require(isinstance(item,dict) and IDENTITY_KEYS <= set(item) and
             all(isinstance(item[key],str) and item[key] for key in IDENTITY_KEYS-{"initial_reference_m"}) and
             type(item["initial_reference_m"]) in (int,float) and math.isfinite(item["initial_reference_m"]),
             "learned-manifest-trial-identity-invalid")
    relative = PurePosixPath(item["episode"])
    _require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in item["episode"] and
             ":" not in item["episode"] and relative.as_posix() == item["episode"], "learned-manifest-trial-path-invalid")
    return tuple(item[key] for key in ("method","case","condition","initial_reference_m","route_id","episode"))


def _attempt_metric(root,item,evaluation_config):
    """Keep failed or incomplete attempts in the denominator with unknown bounds."""
    try:
        metadata = strict_json_loads(_safe_file(root,"episode.json").read_text(encoding="utf-8"))
    except Exception:
        metadata = {"episode_id":root.name,"evaluation_route_length_m":None}
    if not isinstance(metadata,dict):
        metadata = {"episode_id":root.name,"evaluation_route_length_m":None}
    rows = []
    try:
        lines = _safe_file(root,"steps.jsonl").read_text(encoding="utf-8").splitlines()
        rows = [strict_json_loads(line) for line in lines]
    except Exception:
        rows = [None]
    metadata.update(method=item["method"],route_id=item["route_id"])
    if item.get("status") != "PASS" or metadata.get("status") != "COMPLETE":
        metadata["termination_reason"] = "technical_failure"
    return evaluate_episode(metadata,rows,metadata.get("events",[]),config=evaluation_config)


def _exported_metrics(root,episode_metrics,aggregate):
    hashes = {}
    json_path = root/"metrics.json"
    if json_path.exists() or json_path.is_symlink():
        value = _safe_file(root,"metrics.json").read_bytes()
        expected = {"schema_version":1,"episodes":episode_metrics,"aggregate":aggregate}
        _require(_same(strict_json_loads(value.decode("utf-8")),expected), "learned-exported-metrics-json-mismatch")
        hashes["metrics.json"] = hashlib.sha256(value).hexdigest()
    csv_path = root/"metrics.csv"
    if csv_path.exists() or csv_path.is_symlink():
        value = _safe_file(root,"metrics.csv").read_bytes()
        reader = csv.DictReader(io.StringIO(value.decode("utf-8"),newline=""))
        rows = list(reader)
        _require(tuple(reader.fieldnames or ()) == CSV_FIELDS and len(rows) == len(episode_metrics), "learned-exported-metrics-csv-schema")
        for actual,expected in zip(rows,episode_metrics):
            for field in CSV_FIELDS:
                encoded,original = actual[field],expected[field]
                if original is None:
                    match = encoded == "N/A"
                elif isinstance(original,(bool,str)):
                    match = encoded == str(original)
                else:
                    match = _same(strict_json_loads(encoded),original)
                _require(match,"learned-exported-metrics-csv-mismatch")
        hashes["metrics.csv"] = hashlib.sha256(value).hexdigest()
    return hashes


def validate_learned_manifest(manifest_path: str | Path, checkpoint_path: str | Path, *, device="cpu") -> dict:
    """Require the full declared trial plan and compare its paired center subset.

    Integrity PASS is separate from driving success. Operationally failed
    attempts make the gate FAIL and still receive metric rows. No missing
    attempt can become a successful shortened suite.
    """
    path = Path(manifest_path)
    report = {"status":"FAIL","scope":SCOPE,"gate_version":GATE_VERSION,"errors":[],"episodes":[]}
    episode_metrics = []
    try:
        _no_links(path)
        manifest_bytes = path.read_bytes()
        manifest = strict_json_loads(manifest_bytes.decode("utf-8"))
        _require(isinstance(manifest,dict) and set(manifest) == MANIFEST_KEYS and
                 type(manifest["schema_version"]) is int and manifest["schema_version"] == 1 and
                 manifest["scope"] == SCOPE and manifest["mode"] in ("learned","assisted") and
                 manifest["status"] in ("PLANNED","RUNNING","FAILED","COMPLETE"), "learned-manifest-schema-invalid")
        cfg = manifest["config"]
        _require(isinstance(cfg,dict) and set(cfg) == CONFIG_KEYS,"learned-manifest-config-fields")
        policy = load_validation_config()
        _require(_same(cfg["validation_config"],policy) and cfg["initial_reference_m"] == [0.,2.,4.] and
                 _same(cfg["initial_conditions"],policy["initial_conditions"]) and cfg["goal_reference_m"] == policy["goal_reference_m"],
                 "learned-manifest-native-plan-config-mismatch")
        _require(manifest["driving_source_sha256"] == learned_runtime_source_hashes(),"learned-manifest-driving-source-mismatch")
        catalog_path = ROOT/"configs/routes_d6_geometry.json"
        _require(cfg["geometry_catalog_sha256"] == _sha_file(_no_links(catalog_path)),"learned-manifest-geometry-catalog-mismatch")
        catalog = load_geometry_catalog(catalog_path)
        routes = {item["case"]:item["route"] for item in catalog["geometry_cases"]}
        _require(set(routes) == set(CASES) and all(route["town"] == "Town01" for route in routes.values()),"learned-manifest-original-route-cases")
        conditions = {item["name"]:item for item in policy["initial_conditions"]}
        primary = "M1" if manifest["mode"] == "learned" else "M1-assisted"
        requested = {(primary,case,condition,start,routes[case]["route_id"])
                     for case in CASES for condition in conditions for start in (0.,2.,4.)}
        requested |= {(method,case,"center",start,routes[case]["route_id"])
                      for method in ("B0","privileged_reference") for case in CASES for start in (0.,2.,4.)}
        plan = manifest["planned_trials"]
        _require(isinstance(plan,list) and all(isinstance(item,dict) and set(item) == IDENTITY_KEYS for item in plan),
                 "learned-manifest-plan-fields")
        plan_ids = [_identity(item) for item in plan]
        _require(len(plan_ids) == len(set(plan_ids)) == 63 and {value[:5] for value in plan_ids} == requested and
                 len({value[-1] for value in plan_ids}) == 63,"learned-manifest-required-trial-plan-incomplete")
        episodes,remaining = manifest["episodes"],manifest["remaining_trials"]
        _require(isinstance(episodes,list) and isinstance(remaining,list) and
                 all(isinstance(item,dict) and set(item) == EPISODE_KEYS for item in episodes) and
                 all(isinstance(item,dict) and set(item) == IDENTITY_KEYS for item in remaining), "learned-manifest-attempt-fields")
        attempts = [_identity(item) for item in episodes]
        outstanding = [_identity(item) for item in remaining]
        _require(len(attempts) == len(set(attempts)) and len(outstanding) == len(set(outstanding)) and
                 not set(attempts)&set(outstanding) and set(attempts)|set(outstanding) == set(plan_ids), "learned-manifest-dropped-or-duplicate-attempt")
        report.update(planned_episode_count=63,attempted_episode_count=len(attempts),remaining_episode_count=len(outstanding))
        evaluation_config = evaluation_config_from_dict(cfg["evaluation_config"])
        _require(evaluation_config.sha256 == cfg["evaluation_config_sha256"],"learned-manifest-evaluation-config-hash")
        from .baseline import baseline_config_from_dict
        from .controller import controller_config_from_dict
        from .config import collection_config_from_dict
        _require(baseline_config_from_dict(cfg["baseline_config"]).sha256 == cfg["baseline_config_sha256"] and
                 controller_config_from_dict(cfg["controller_config"]).sha256 == cfg["controller_config_sha256"], "learned-manifest-controller-or-baseline-config-hash")
        collection = collection_config_from_dict(cfg["collection_config"]).to_dict()
        collection["vehicle"].pop("spawn_point_index")
        common_hash = canonical_sha256(collection)
        shared_geometry,proofs,gate_errors = None,[],[]
        for item in episodes:
            identity = _identity(item)
            root = path.parent.joinpath(*PurePosixPath(item["episode"]).parts)
            _no_links(root)
            _require(root.resolve().is_relative_to(path.parent.resolve()),"learned-manifest-episode-escaped")
            _require(item["status"] in ("PASS","FAIL") and type(item["step_count"]) is int and item["step_count"] >= 0,
                     "learned-manifest-attempt-status-invalid")
            if item["status"] == "PASS":
                if item["method"] == primary:
                    checked = validate_learned_episode(root,checkpoint_path,allowed_conditions=list(conditions.values()),
                        expected_initial_reference_m=item["initial_reference_m"],device=device)
                else:
                    metadata = strict_json_loads(_safe_file(root,"episode.json").read_text(encoding="utf-8"))
                    _initialization(metadata,item["initial_reference_m"])
                    from .baseline_driving import SCOPE as BASELINE_SCOPE
                    checked = _validate_camera_episode(root,allowed_conditions=[conditions["center"]],scope=BASELINE_SCOPE,
                        expected_initial_reference_m=item["initial_reference_m"])
            else:
                checked = {"status":"FAIL","errors":[{"code":"learned-runtime-attempt-failed"}]}
            report["episodes"].append({**{key:item[key] for key in IDENTITY_KEYS},"report":checked})
            if checked["status"] != "PASS":
                gate_errors.append({"code":"learned-manifest-episode-gate-failed","identity":list(identity[:5])})
                episode_metrics.append(_attempt_metric(root,item,evaluation_config))
                continue
            source = checked["source"]
            _require((source["method"],source["geometry_case"],source["condition"]["name"],item["initial_reference_m"],source["route_id"]) == identity[:5],
                     "learned-manifest-episode-identity-mismatch")
            _require(source["metadata_sha256"] == item["episode_json_sha256"] and source["trace_sha256"] == item["steps_jsonl_sha256"] and
                     checked["metrics"]["step_count"] == item["step_count"],"learned-manifest-episode-source-hash-or-count-mismatch")
            _require(source["route_sha256"] == routes[item["case"]]["route_sha256"] and source["town"] == "Town01" and
                     source["configuration_except_spawn_sha256"] == common_hash and
                     source["controller_config_sha256"] == cfg["controller_config_sha256"] and
                     source["baseline_config_sha256"] == cfg["baseline_config_sha256"] and
                     source["evaluation_config_sha256"] == cfg["evaluation_config_sha256"], "learned-manifest-shared-route-or-control-config-mismatch")
            if item["method"] == primary:
                _require(source["learned_model"] == manifest["model"] and source["learned_driving_source_sha256"] == manifest["driving_source_sha256"],
                         "learned-manifest-model-identity-mismatch")
            if shared_geometry is not None:
                _require(set(source["vehicle_geometry"]) == set(shared_geometry) and all(
                         _same(source["vehicle_geometry"][key],shared_geometry[key],.001 if key.endswith("_m") else 1e-5)
                         for key in shared_geometry),"learned-manifest-shared-native-geometry-mismatch")
            shared_geometry = source["vehicle_geometry"]
            episode_metrics.append(checked["evaluation_metrics"])
            proofs.append((root,source))
        report.update(episode_count=len(episode_metrics),episode_metrics=episode_metrics,aggregate=aggregate_metrics(episode_metrics))
        _require(manifest["status"] == "COMPLETE" and not outstanding and len(attempts) == 63,"learned-manifest-incomplete-run")
        _require(not gate_errors,"learned-manifest-operational-or-integrity-failures")
        aggregate = report["aggregate"]
        center = [metric for item,metric in zip(episodes,episode_metrics) if item["condition"] == "center"]
        _require(len(center) == 27,"learned-manifest-paired-center-count")
        report["paired_center_comparison"] = {"scope":"same three original routes/center conditions/0,2,4m starts/shared controller/speed/clock",
            "episode_count":27,"aggregate":aggregate_metrics(center),"full_primary_recovery_vs_baseline_center_is_not_paired":True}
        extra = _exported_metrics(path.parent,episode_metrics,aggregate)
        _require(path.read_bytes() == manifest_bytes and learned_runtime_source_hashes() == manifest["driving_source_sha256"] and all(
                 _sha_file(_safe_file(root,"episode.json")) == source["metadata_sha256"] and
                 _sha_file(_safe_file(root,"steps.jsonl")) == source["trace_sha256"] and all(
                 _sha_file(_safe_file(root,name)) == digest for name,digest in source["image_sha256"].items())
                 for root,source in proofs) and all(_sha_file(_safe_file(path.parent,name)) == digest for name,digest in extra.items()),
                 "learned-manifest-source-changed-during-validation")
        report.update(status="PASS",step_count=sum(item["report"]["metrics"]["step_count"] for item in report["episodes"]),
            image_count=sum(item["report"]["metrics"]["image_count"] for item in report["episodes"]),
            successful_driving_episodes=sum(metric["driving_success"] for metric in episode_metrics),
            failed_driving_episodes=sum(not metric["driving_success"] for metric in episode_metrics),
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),final_test_used=False,held_out_performance_certified=False,
            checks={"all_required_attempts_in_denominator":True,"same_frame_original_RGB_and_model_identity":True,
                    "shared_native_controller_and_evaluation":True,"paired_comparison_center_subset_only":True,"sources_unchanged":True})
    except BaseException as error:
        report["errors"].append({"code":getattr(error,"code",str(error) if isinstance(error,ValueError) else
                                               "learned-manifest-validation-failed"),"exception_type":type(error).__name__})
        if "gate_errors" in locals():
            report["errors"].extend(gate_errors)
        if episode_metrics:
            report.update(episode_count=len(episode_metrics),episode_metrics=episode_metrics,aggregate=aggregate_metrics(episode_metrics))
    return report
