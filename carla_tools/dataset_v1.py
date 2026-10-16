"""Source-bound moving RGB datasets and matched-budget learning selections.

The manifest contains labels and audit metadata; the model input remains RGB.
Every weather, start and perturbation of a physical scene shares its split.
"""
from __future__ import annotations

import bisect
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType

from .contracts import (canonical_sha256, read_strict_json, strict_json_loads,
                        validate_identifier, validate_route_contract, validate_waypoint_label)
from .geometry_survey import CASES, classify_geometry
from .pilot import SAMPLE_FIELDS as PILOT_SAMPLE_FIELDS
from .pilot import _read_episode, _safe_file, _scene_distance
from .splits import _scene

SCOPE = "moving-rgb-waypoint-dataset-v1"
GATE_VERSION = "vlf-dataset-v1-source-replay-v2"
VALIDATION_CHECKS = ("all_native_episodes_replayed", "same_frame_gt_labels", "every_source_png",
                     "all_excluded_sources_retained", "route_scene_spatial_split", "reference_arc_pairs",
                     "matched_training_budget", "common_dev", "immutable_source_snapshot", "reserved_final_town_unopened")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/dataset_v1.json"
SAMPLE_FIELDS = PILOT_SAMPLE_FIELDS | {"role", "pair_group_id", "run_name", "lateral_error_m", "heading_error_deg",
                                      "pairing_heading_error_deg"}
DESCRIPTOR_FIELDS = {"episode", "split", "case", "condition", "role", "pair_group_id", "weather",
                     "run_name", "initial_reference_m"}
DERIVED_DESCRIPTOR_FIELDS = {"episode_id", "route_id", "scene_group_id", "episode_json_sha256", "steps_jsonl_sha256"}
SOURCE_MODULES = ("carla_tools/dataset_v1.py", "carla_tools/pilot.py", "carla_tools/contracts.py",
                  "carla_tools/baseline_validation.py", "carla_tools/driving_route.py",
                  "carla_tools/splits.py", "carla_tools/geometry_survey.py", "carla_tools/config.py",
                  "carla_tools/session.py", "carla_tools/capture.py", "carla_tools/driving.py",
                  "carla_tools/driving_validation.py", "carla_tools/controller.py", "carla_tools/evaluation.py",
                  "carla_tools/baseline.py", "carla_tools/baseline_driving.py", "carla_tools/projection.py",
                  "carla_tools/color.py", "carla_tools/episode.py", "carla_tools/routes.py",
                  "carla_tools/dataset_learning.py", "carla_tools/learning_data.py", "scripts/run_dataset_pipeline.py",
                  "scripts/carla_rgb_probe.py", "configs/driving_validation_d10.json", "configs/controller_d8.json",
                  "configs/baseline.json", "configs/evaluation.json", "configs/learning.json",
                  "scripts/verify_collection.py", "scripts/check_carla_connection.py")


class DatasetV1Error(ValueError):
    def __init__(self, code, *, context=None):
        super().__init__(code)
        self.code = code
        self.context = copy.deepcopy(context) if context is not None else None


def _require(condition, code):
    if not condition:
        raise DatasetV1Error(code)


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value), "dataset-nonfinite-number")
    return float(value)


def _freeze(value):
    if type(value) is dict:
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) in (list, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, MappingProxyType):
        return {key: _thaw(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class DatasetV1Config:
    schema_version: int
    dataset_version: str
    minimum_samples: int
    maximum_samples: int
    minimum_train_samples: int
    minimum_dev_samples: int
    minimum_central_samples: int
    minimum_recovery_samples: int
    minimum_speed_mps: float
    minimum_step_distance_m: float
    scene_buffer_m: float
    pair_arc_tolerance_m: float
    minimum_recovery_lateral_difference_m: float
    minimum_recovery_heading_difference_deg: float
    minimum_pair_count: int
    train_selection_samples: int
    mixed_recovery_fraction: float
    selection_seed: int
    train_routes_per_case: dict
    dev_routes_per_case: dict
    conditions: list
    weather_presets: list
    collection_runs: list

    def __post_init__(self):
        raw = {item.name: _thaw(getattr(self, item.name)) for item in fields(self)}
        _validate_config(raw)
        for item in fields(self):
            object.__setattr__(self, item.name, _freeze(raw[item.name]))

    def to_dict(self):
        return {item.name: _thaw(getattr(self, item.name)) for item in fields(self)}

    @property
    def sha256(self):
        return canonical_sha256(self.to_dict())


def _validate_config(value):
    _require(type(value["schema_version"]) is int and value["schema_version"] == 1, "dataset-config-schema-invalid")
    validate_identifier(value["dataset_version"])
    count_fields = ("minimum_samples", "maximum_samples", "minimum_train_samples", "minimum_dev_samples",
                    "minimum_central_samples", "minimum_recovery_samples", "minimum_pair_count", "train_selection_samples")
    _require(all(type(value[key]) is int and 1 <= value[key] <= 1_000_000 for key in count_fields),
             "dataset-config-count-invalid")
    _require(value["minimum_samples"] <= value["maximum_samples"]
             and value["minimum_train_samples"] + value["minimum_dev_samples"] <= value["minimum_samples"]
             and value["minimum_central_samples"] + value["minimum_recovery_samples"] <= value["minimum_samples"]
             and value["train_selection_samples"] <= value["minimum_train_samples"], "dataset-config-budget-invalid")
    _require(type(value["selection_seed"]) is int and 0 <= value["selection_seed"] <= 2 ** 31 - 1,
             "dataset-config-seed-invalid")
    _require(.05 <= _number(value["minimum_speed_mps"]) <= 1.
             and .001 <= _number(value["minimum_step_distance_m"]) <= .05
             and _number(value["scene_buffer_m"]) >= 75.
             and .001 <= _number(value["pair_arc_tolerance_m"]) <= .5,
             "dataset-config-motion-buffer-or-pair-tolerance-invalid")
    _require(.01 <= _number(value["minimum_recovery_lateral_difference_m"]) <= .3
             and .1 <= _number(value["minimum_recovery_heading_difference_deg"]) <= 5.,
             "dataset-config-active-recovery-threshold-invalid")
    fraction = _number(value["mixed_recovery_fraction"])
    _require(0 < fraction < 1 and abs(value["train_selection_samples"] * fraction -
                                     round(value["train_selection_samples"] * fraction)) <= 1e-9,
             "dataset-config-recovery-fraction-invalid")
    for key in ("train_routes_per_case", "dev_routes_per_case"):
        _require(type(value[key]) is dict and set(value[key]) == set(CASES)
                 and all(type(count) is int and 1 <= count <= 100 for count in value[key].values()),
                 "dataset-config-route-coverage-invalid")
    from .driving import load_validation_config
    _require(value["conditions"] == load_validation_config()["initial_conditions"], "dataset-config-conditions-invalid")
    weather = value["weather_presets"]
    _require(type(weather) is list and 1 <= len(weather) <= 8 and len(set(weather)) == len(weather)
             and all(item in ("ClearNoon", "CloudyNoon", "WetNoon", "SoftRainNoon") for item in weather),
             "dataset-config-weather-invalid")
    runs = value["collection_runs"]
    _require(type(runs) is list and 1 <= len(runs) <= 32, "dataset-config-runs-invalid")
    names, starts = set(), set()
    for run in runs:
        _require(type(run) is dict and set(run) == {"name", "initial_reference_m", "seed"}, "dataset-config-run-fields-invalid")
        validate_identifier(run["name"])
        _require(run["name"] not in names and type(run["seed"]) is int and 0 <= run["seed"] <= 2 ** 31 - 1
                 and 0 <= _number(run["initial_reference_m"]) <= 10., "dataset-config-run-invalid")
        identity = run["initial_reference_m"], run["seed"]
        _require(identity not in starts, "dataset-config-duplicate-run")
        names.add(run["name"])
        starts.add(identity)


def dataset_v1_config_from_dict(value):
    _require(type(value) is dict and set(value) == {item.name for item in fields(DatasetV1Config)},
             "dataset-config-fields-invalid")
    return DatasetV1Config(**copy.deepcopy(value))


def load_dataset_v1_config(path=DEFAULT_CONFIG):
    return dataset_v1_config_from_dict(read_strict_json(path))


def _implementation_hashes():
    root = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_MODULES}


def pair_group_id(route_id, weather, run_name):
    return f"{route_id}__{weather}__{run_name}"


def _routes(routes, config):
    _require(type(routes) is list and routes, "dataset-routes-empty")
    lookup, scene_splits, counts, maps = {}, {}, Counter(), set()
    for item in routes:
        _require(type(item) is dict and set(item) == {"split", "case", "route"}, "dataset-route-fields-invalid")
        route, split, case = item["route"], item["split"], item["case"]
        validate_route_contract(route)
        _require(route["town"] == "Town01" and split in ("train", "dev") and case in CASES
                 and classify_geometry(route)["case"] == case, "dataset-route-case-town-or-split-invalid")
        _require(route["route_id"] not in lookup, "dataset-route-duplicate")
        group = route["scene_group_id"]
        _require(group not in scene_splits or scene_splits[group] == split, "dataset-scene-split-leakage")
        scene_splits[group] = split
        lookup[route["route_id"]] = item
        counts[split, case] += 1
        maps.add(route["opendrive_sha256"])
    _require(len(maps) == 1 and all(counts[split, case] == getattr(config, split + "_routes_per_case")[case]
                                  for split in ("train", "dev") for case in CASES), "dataset-route-coverage-invalid")
    return lookup


def _descriptor(root, item, route_lookup, config):
    _require(type(item) is dict and set(item) == DESCRIPTOR_FIELDS, "dataset-episode-descriptor-fields-invalid")
    metadata, rows, metadata_bytes, trace_bytes = _read_episode(root, item)
    route = metadata["route_contract"]
    route_item = route_lookup.get(route["route_id"])
    _require(route_item is not None and route == route_item["route"] and item["split"] == route_item["split"]
             and item["case"] == route_item["case"] and metadata["geometry_case"] == item["case"],
             "dataset-episode-route-mismatch")
    validate_identifier(metadata["episode_id"])
    _require(metadata.get("status") == "COMPLETE" and metadata.get("method") == "privileged_reference",
             "dataset-episode-not-complete-reference")
    conditions = {condition["name"]: _thaw(condition) for condition in config.conditions}
    _require(item["condition"] in conditions and metadata["condition"] == conditions[item["condition"]],
             "dataset-episode-condition-mismatch")
    expected_role = "central" if item["condition"] == "center" else "recovery"
    _require(item["role"] == expected_role, "dataset-episode-role-mismatch")
    runs = {run["name"]: run for run in config.collection_runs}
    run = runs.get(item["run_name"])
    _require(run is not None and type(item["initial_reference_m"]) in (int, float)
             and item["initial_reference_m"] == run["initial_reference_m"], "dataset-episode-run-mismatch")
    initialization = metadata.get("initialization", {})
    actual_start = initialization.get("initial_reference_m", 0.)
    _require(type(actual_start) in (int, float) and actual_start == item["initial_reference_m"],
             "dataset-episode-start-mismatch")
    _require(item["weather"] in config.weather_presets
             and metadata["collection_config"]["weather"]["preset"] == item["weather"]
             and metadata["collection_config"]["seed"] == run["seed"], "dataset-episode-weather-or-seed-mismatch")
    _require(item["pair_group_id"] == pair_group_id(route["route_id"], item["weather"], item["run_name"]),
             "dataset-episode-pair-group-mismatch")
    descriptor = copy.deepcopy(item)
    descriptor.update(episode_id=metadata["episode_id"], route_id=route["route_id"], scene_group_id=route["scene_group_id"],
                      episode_json_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
                      steps_jsonl_sha256=hashlib.sha256(trace_bytes).hexdigest())
    return descriptor, metadata, rows


def _continuous_route_yaw_deg(route, reference_arc_m):
    """Interpolate frozen anchor rotations through the shortest wrapped angle.

    The route contract is validated before this helper is called. Its endpoint
    rotations are included; a polyline projection's chosen chord is irrelevant.
    """
    reference = _number(reference_arc_m)
    anchors = route["centerline_anchor_sequence"]
    _require(type(anchors) is list and len(anchors) >= 2, "dataset-pairing-route-heading-invalid")
    first_arc, last_arc = _number(anchors[0]["arc_distance_m"]), _number(anchors[-1]["arc_distance_m"])
    _require(first_arc <= reference <= last_arc, "dataset-pairing-heading-reference-outside-route")
    index = min(bisect.bisect_right(anchors, reference, key=lambda anchor: anchor["arc_distance_m"]) - 1,
                len(anchors) - 2)
    first, second = anchors[index:index + 2]
    start, end = _number(first["arc_distance_m"]), _number(second["arc_distance_m"])
    _require(start <= reference <= end and start < end, "dataset-pairing-route-heading-invalid")
    yaw = _number(first["world_rotation"]["yaw"])
    change = (_number(second["world_rotation"]["yaw"]) - yaw + 180.) % 360. - 180.
    return (yaw + (reference - start) / (end - start) * change + 180.) % 360. - 180.


def _pairing_heading_error_deg(route, vehicle_transform, reference_arc_m):
    actual_yaw = _number(vehicle_transform["rotation"]["yaw"])
    return (actual_yaw - _continuous_route_yaw_deg(route, reference_arc_m) + 180.) % 360. - 180.


def _quality_samples(metadata, rows, descriptor, config):
    """Record every excluded frame; malformed source bytes never become samples."""
    samples, ledger, previous = [], [], None
    route = metadata["route_contract"]
    calibration = canonical_sha256(metadata["camera_calibration"])
    metrics = metadata["evaluation_metrics"]
    clean = (metrics.get("driving_success") is True and metrics.get("collision_count") == 0
             and metrics.get("lane_invasion_count") == 0 and _number(metrics["minimum_body_lane_margin_m"]) >= 0.)
    for index, row in enumerate(rows):
        frame = row.get("frame") if type(row) is dict else None
        entry = {"episode_id": descriptor["episode_id"], "row_index": index, "frame": frame}
        try:
            current = row["vehicle"]["transform"]["location"]
            x, y = _number(current["x"]), _number(current["y"])
            distance = 0. if previous is None else math.hypot(x - previous[0], y - previous[1])
            previous = x, y
            speed = _number(row["speed_mps"])
            _require(type(frame) is int and frame >= 0, "dataset-frame-invalid")
            _number(row["timestamp"])
            _number(row["tracking"]["reference_arc_m"])
            if not clean:
                reason = "reference_episode_not_clean"
            elif speed < config.minimum_speed_mps or distance < config.minimum_step_distance_m:
                reason = "stationary_or_startup"
            elif _number(row["body_lane_margin_m"]) < 0:
                reason = "body_outside_lane"
            else:
                reason = None
            if reason is not None:
                ledger.append(dict(entry, reason=reason))
                continue
            image, label = row["image"], row["tracking"]["label"]
            validate_waypoint_label(label, row, {"route_id": route["route_id"], "route_contract": route})
            _require(type(image["sha256"]) is str and len(image["sha256"]) == 64
                     and all(char in "0123456789abcdef" for char in image["sha256"])
                     and (image["width"], image["height"]) == (640, 360), "dataset-image-contract-invalid")
            samples.append({"sample_id": f"{metadata['episode_id']}:{frame:010d}", "episode_id": metadata["episode_id"],
                            "route_id": route["route_id"], "scene_group_id": route["scene_group_id"],
                            "split": descriptor["split"], "case": descriptor["case"], "condition": descriptor["condition"],
                            "role": descriptor["role"], "pair_group_id": descriptor["pair_group_id"],
                            "run_name": descriptor["run_name"], "frame": frame, "timestamp": row["timestamp"],
                            "lateral_error_m": _number(row["tracking"]["lateral_error_m"]),
                            "heading_error_deg": _number(row["tracking"]["heading_error_deg"]),
                            "pairing_heading_error_deg": _pairing_heading_error_deg(
                                route, row["vehicle"]["transform"], row["tracking"]["reference_arc_m"]),
                            "image": descriptor["episode"] + "/" + image["path"], "image_sha256": image["sha256"],
                            "image_width": image["width"], "image_height": image["height"],
                            "waypoints_ego_m": copy.deepcopy(label["waypoints_ego_m"]), "speed_mps": speed,
                            "reference_arc_m": row["tracking"]["reference_arc_m"],
                            "vehicle_transform": copy.deepcopy(row["vehicle"]["transform"]), "calibration_sha256": calibration,
                            "target_lane": copy.deepcopy(label["target_lane"]), "weather": descriptor["weather"],
                            "perturbation": {key: metadata["condition"][key] for key in ("lateral_m", "yaw_deg")}})
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            ledger.append(dict(entry, reason="malformed_frame", error_code=getattr(error, "code", "invalid-frame")))
    return samples, ledger


def match_reference_arc_pairs(samples, config):
    """Nearest one-to-one arc matches per recovery condition in the same scene.

    A central frame may anchor different recovery conditions, but each frame
    is used at most once within a condition. Timestamps never define a pair.
    """
    groups = defaultdict(list)
    ids = set()
    for sample in samples:
        _require(sample["sample_id"] not in ids, "dataset-pair-sample-duplicate")
        ids.add(sample["sample_id"])
        _number(sample["reference_arc_m"])
        _require(sample["role"] in ("central", "recovery")
                 and (sample["condition"] == "center") is (sample["role"] == "central"),
                 "dataset-pair-role-condition-invalid")
        groups[sample["pair_group_id"]].append(sample)
    pairs = []
    for group, members in sorted(groups.items()):
        _require(len({(item["route_id"], item["scene_group_id"], item["split"], item["case"], item["weather"], item["run_name"])
                      for item in members}) == 1, "dataset-pair-group-identity-mismatch")
        central = sorted((item for item in members if item["role"] == "central"),
                         key=lambda item: (item["reference_arc_m"], item["sample_id"]))
        _require(len({item["episode_id"] for item in central}) <= 1, "dataset-multiple-central-episodes-in-pair-group")
        arcs = [item["reference_arc_m"] for item in central]
        recoveries = defaultdict(list)
        for item in members:
            _require(item["role"] in ("central", "recovery"), "dataset-pair-role-invalid")
            if item["role"] == "recovery":
                recoveries[item["condition"]].append(item)
        for condition, recovery in sorted(recoveries.items()):
            used = set()
            _require(len({item["episode_id"] for item in recovery}) <= 1, "dataset-multiple-recovery-episodes-in-pair-group")
            # Process original progressing frames. Nearest unused central is
            # found in the local tolerance window, not by aligning row numbers.
            for item in sorted(recovery, key=lambda row: (row["reference_arc_m"], row["sample_id"])):
                arc = item["reference_arc_m"]
                lo = bisect.bisect_left(arcs, arc - config.pair_arc_tolerance_m)
                hi = bisect.bisect_right(arcs, arc + config.pair_arc_tolerance_m)
                candidates = [index for index in range(lo, hi) if index not in used]
                if not candidates:
                    continue
                index = min(candidates, key=lambda candidate: (abs(arcs[candidate] - arc), central[candidate]["sample_id"]))
                used.add(index)
                first = central[index]
                lateral_difference = abs(_number(first["lateral_error_m"]) - _number(item["lateral_error_m"]))
                heading_difference = abs((_number(first["pairing_heading_error_deg"]) -
                                          _number(item["pairing_heading_error_deg"]) + 180.) % 360. - 180.)
                identity = {"pair_group_id": group, "central_sample_id": first["sample_id"],
                            "recovery_sample_id": item["sample_id"], "recovery_condition": condition}
                pairs.append({"pair_id": canonical_sha256(identity), **identity, "split": item["split"], "case": item["case"],
                              "reference_arc_difference_m": abs(first["reference_arc_m"] - arc),
                              "lateral_difference_m": lateral_difference, "heading_difference_deg": heading_difference,
                              "active_recovery": lateral_difference >= config.minimum_recovery_lateral_difference_m
                              or heading_difference >= config.minimum_recovery_heading_difference_deg})
    return pairs


def _rank(value, seed):
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _balanced_pick(items, count, seed):
    """Deterministically round-robin geometry/weather/run strata, without reuse."""
    _require(type(count) is int and 0 <= count <= len(items), "dataset-selection-insufficient-samples")
    groups = defaultdict(list)
    for item in items:
        groups[item["case"], item["weather"], item["run_name"]].append(item)
    for values in groups.values():
        values.sort(key=lambda item: (_rank(item["sample_id"], seed), item["sample_id"]))
    selected, cursor = [], Counter()
    keys = sorted(groups, key=lambda item: (_rank("|".join(item), seed), item))
    while len(selected) < count:
        for key in keys:
            if cursor[key] < len(groups[key]):
                selected.append(groups[key][cursor[key]]["sample_id"])
                cursor[key] += 1
                if len(selected) == count:
                    break
    return selected


def select_training_sets(samples, pairs, config):
    """Freeze equal-size A/B train sets and a common scene-held-out dev set."""
    lookup = {item["sample_id"]: item for item in samples}
    _require(len(lookup) == len(samples), "dataset-selection-sample-duplicate")
    eligible_pairs = defaultdict(list)
    for pair in pairs:
        first, second = lookup[pair["central_sample_id"]], lookup[pair["recovery_sample_id"]]
        lateral_difference = abs(_number(first["lateral_error_m"]) - _number(second["lateral_error_m"]))
        heading_difference = abs((_number(first["pairing_heading_error_deg"]) -
                                  _number(second["pairing_heading_error_deg"]) + 180.) % 360. - 180.)
        active = lateral_difference >= config.minimum_recovery_lateral_difference_m or heading_difference >= config.minimum_recovery_heading_difference_deg
        _require(first["role"] == "central" and second["role"] == "recovery"
                 and first["pair_group_id"] == second["pair_group_id"] and first["split"] == second["split"]
                 and abs(first["reference_arc_m"] - second["reference_arc_m"]) <= config.pair_arc_tolerance_m
                 and type(pair["active_recovery"]) is bool and pair["active_recovery"] is active,
                 "dataset-selection-pair-invalid")
        if first["split"] == "train" and pair["active_recovery"]:
            eligible_pairs[first["sample_id"]].append(pair)
    central = [item for item in samples if item["split"] == "train" and item["role"] == "central"]
    count = config.train_selection_samples
    recovery_count = round(count * config.mixed_recovery_fraction)
    replaced_ids = _balanced_pick([item for item in central if item["sample_id"] in eligible_pairs],
                                  recovery_count, config.selection_seed)
    replaced_set = set(replaced_ids)
    shared_ids = _balanced_pick([item for item in central if item["sample_id"] not in replaced_set],
                                count - recovery_count, config.selection_seed)
    central_ids = shared_ids + replaced_ids
    # Both arms share their central examples. Recovery replaces exactly the
    # configured fraction of A, preserving the sample budget and dev source.
    mixed_central = shared_ids
    recovery_ids, used_recovery, condition_counts = [], set(), Counter()
    for identifier in replaced_ids:
        candidates = [pair for pair in eligible_pairs[identifier] if pair["recovery_sample_id"] not in used_recovery]
        _require(candidates, "dataset-selection-active-recovery-insufficient")
        pair = min(candidates, key=lambda item: (condition_counts[item["case"], item["recovery_condition"]],
                                                _rank(item["pair_id"], config.selection_seed), item["pair_id"]))
        recovery_ids.append(pair["recovery_sample_id"])
        used_recovery.add(pair["recovery_sample_id"])
        condition_counts[pair["case"], pair["recovery_condition"]] += 1
    dev_ids = sorted(item["sample_id"] for item in samples if item["split"] == "dev")
    _require(dev_ids and len(set(mixed_central + recovery_ids)) == count, "dataset-selection-budget-or-dev-invalid")
    return {"selection_version": "vlf-matched-budget-central-recovery-v2", "training_samples_per_arm": count,
            "recovery_fraction": config.mixed_recovery_fraction, "selection_seed": config.selection_seed,
            "central_only": {"train_sample_ids": central_ids, "dev_sample_ids": dev_ids},
            "central_recovery": {"train_sample_ids": mixed_central + recovery_ids, "dev_sample_ids": dev_ids},
            "common_dev_sample_ids": dev_ids,
            "replaced_central_sample_ids": replaced_ids,
            "eligible_active_recovery_anchors": len(eligible_pairs),
            "note": "Independent training trials must keep preprocessing, initialization and optimizer budget equal."}


def _native_episode_gate(path, *, allowed_conditions, expected_initial_reference_m, expected_weather):
    from scripts.run_dataset_pipeline import validate_dataset_v1_episode
    return validate_dataset_v1_episode(path, allowed_conditions=allowed_conditions,
                                       expected_initial_reference_m=expected_initial_reference_m,
                                       expected_weather=expected_weather)


def _scene_separation_lower_bound(first, second, required_m):
    """Certify separated full-source boxes, with exact geometry fallback.

    Both boxes include surveyed segments and every observed actor position,
    including excluded frames. A box gap bounds every segment pair below.
    """
    _require(first["town"] == second["town"], "dataset-scene-town-mismatch")
    def bounds(scene):
        points = [point for segment in scene["segments"] for point in segment]
        _require(points and all(len(point) == 2 and all(math.isfinite(_number(value)) for value in point)
                                for point in points), "dataset-scene-geometry-invalid")
        return tuple((min(point[axis] for point in points), max(point[axis] for point in points)) for axis in (0, 1))
    a, b = bounds(first), bounds(second)
    lower = math.hypot(*(max(0., a[axis][0] - b[axis][1], b[axis][0] - a[axis][1]) for axis in (0, 1)))
    return lower if lower > required_m else _scene_distance(first, second)


def _derive(directory, routes, episodes, config, *, replay_native, progress_callback=None):
    root = Path(directory)
    route_lookup = _routes(routes, config)
    descriptors, samples, ledger, gates, scenes, sources = [], [], [], [], {}, {}
    seen_episodes, combinations, seen_images, seen_samples = set(), set(), {}, set()
    for item in episodes:
        descriptor, metadata, rows = _descriptor(root, item, route_lookup, config)
        episode_id = descriptor["episode_id"]
        identity = (descriptor["route_id"], descriptor["weather"], descriptor["run_name"], descriptor["condition"])
        _require(episode_id not in seen_episodes and identity not in combinations, "dataset-episode-or-condition-duplicate")
        seen_episodes.add(episode_id)
        combinations.add(identity)
        if replay_native:
            gate = _native_episode_gate(root / descriptor["episode"], allowed_conditions=_thaw(config.conditions),
                                        expected_initial_reference_m=descriptor["initial_reference_m"],
                                        expected_weather=descriptor["weather"])
            if gate.get("status") != "PASS":
                raise DatasetV1Error("dataset-native-episode-replay-failed", context={
                    "episode_index": len(descriptors) + 1, "episode_count": len(episodes),
                    "episode_id": episode_id, "episode": descriptor["episode"],
                    "native_errors": copy.deepcopy(gate.get("errors", []))})
            _require(gate["evaluation_metrics"] == metadata["evaluation_metrics"], "dataset-native-metrics-mismatch")
            gates.append({"episode_id": episode_id, "checks": gate["checks"], "evaluation_metrics": gate["evaluation_metrics"]})
        for suffix, key in (("episode.json", "episode_json_sha256"), ("steps.jsonl", "steps_jsonl_sha256")):
            sources[descriptor["episode"] + "/" + suffix] = descriptor[key]
        for row in rows:
            # Include startup, duplicates and rejected frames in the immutable
            # source inventory even when they are absent from learning samples.
            image = row["image"]
            relative = descriptor["episode"] + "/" + image["path"]
            _safe_file(root, relative)
            _require(relative not in sources, "dataset-source-path-duplicate")
            sources[relative] = image["sha256"]
        moving, rejected = _quality_samples(metadata, rows, descriptor, config)
        ledger.extend(rejected)
        for sample in moving:
            _require(sample["sample_id"] not in seen_samples, "dataset-sample-id-duplicate")
            seen_samples.add(sample["sample_id"])
            digest = sample["image_sha256"]
            if digest in seen_images:
                previous = seen_images[digest]
                _require(previous["split"] == sample["split"], "dataset-cross-split-duplicate-rgb")
                ledger.append({"episode_id": episode_id, "row_index": next(index for index, row in enumerate(rows)
                               if row["frame"] == sample["frame"]), "frame": sample["frame"], "reason": "duplicate_rgb",
                               "duplicate_of_sample_id": previous["sample_id"]})
                continue
            seen_images[digest] = sample
            samples.append(sample)
        scene = _scene(dict(metadata, town="Town01"), rows)
        if descriptor["route_id"] not in scenes:
            scenes[descriptor["route_id"]] = {"town": "Town01", "split": descriptor["split"], "segments": set()}
        scenes[descriptor["route_id"]]["segments"].update(scene["segments"])
        descriptors.append(descriptor)
        if progress_callback is not None:
            progress_callback({"episode_index": len(descriptors), "episode_count": len(episodes),
                               "episode_id": episode_id, "sample_count_so_far": len(samples),
                               "native_replay": replay_native})
    expected = {(route, weather, run["name"], condition["name"]) for route in route_lookup
                for weather in config.weather_presets for run in config.collection_runs for condition in config.conditions}
    _require(combinations == expected, "dataset-incomplete-collection-matrix")
    minimum_distance = math.inf
    scene_items = list(scenes.values())
    for scene in scene_items:
        scene["segments"] = list(scene["segments"])
    for index, first in enumerate(scene_items):
        for second in scene_items[index + 1:]:
            if first["split"] != second["split"]:
                distance = _scene_separation_lower_bound(first, second, config.scene_buffer_m)
                _require(distance > config.scene_buffer_m, "dataset-spatial-split-leakage")
                minimum_distance = min(minimum_distance, distance)
    split_counts = Counter(sample["split"] for sample in samples)
    role_counts = Counter(sample["role"] for sample in samples)
    _require(config.minimum_samples <= len(samples) <= config.maximum_samples
             and split_counts["train"] >= config.minimum_train_samples and split_counts["dev"] >= config.minimum_dev_samples
             and role_counts["central"] >= config.minimum_central_samples
             and role_counts["recovery"] >= config.minimum_recovery_samples, "dataset-sample-budget-not-met")
    coverage = {split: {case: Counter(sample["role"] for sample in samples if sample["split"] == split and sample["case"] == case)
                        for case in CASES} for split in ("train", "dev")}
    _require(all(coverage[split][case][role] > 0 for split in coverage for case in CASES for role in ("central", "recovery")),
             "dataset-sample-geometry-role-coverage-missing")
    pairs = match_reference_arc_pairs(samples, config)
    _require(len(pairs) >= config.minimum_pair_count, "dataset-pair-budget-not-met")
    selections = select_training_sets(samples, pairs, config)
    return {"schema_version": 1, "dataset_version": config.dataset_version, "scope": SCOPE, "status": "COMPLETE",
            "town": "Town01", "reserved_final_town": "Town02", "reserved_final_state": "UNOPENED",
            "config": config.to_dict(), "config_sha256": config.sha256, "routes": copy.deepcopy(routes),
            "episodes": descriptors, "samples": samples, "sample_count": len(samples),
            "split_counts": dict(split_counts), "role_counts": dict(role_counts),
            "geometry_role_counts": {split: {case: dict(counts) for case, counts in cases.items()} for split, cases in coverage.items()},
            "exclusion_ledger": ledger, "exclusion_counts": dict(Counter(item["reason"] for item in ledger)),
            "pairs": pairs, "pair_count": len(pairs), "selections": selections,
            "minimum_cross_split_scene_distance_lower_bound_m": minimum_distance,
            "spatial_separation_policy": "full-route-and-all-observed-position-AABB-gap-or-exact-segment-fallback",
            "source_files_sha256": dict(sorted(sources.items())), "implementation_sha256": _implementation_hashes()}, gates


def build_dataset_v1_manifest(directory, routes, episodes, config):
    """Write once from original episodes; final acceptance requires full replay."""
    path = Path(directory) / "dataset_v1_manifest.json"
    _require(not path.exists(), "dataset-manifest-already-exists")
    manifest, _ = _derive(directory, routes, episodes, config, replay_native=False)
    manifest["snapshot_sha256"] = canonical_sha256(manifest)
    with path.open("xb") as stream:
        stream.write(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n")
    return path


def validate_dataset_v1_manifest(path, *, progress_callback=None):
    """Reconstruct the snapshot, including every PNG/control/GT and selection."""
    report = {"status": "FAIL", "scope": SCOPE, "gate_version": GATE_VERSION, "errors": []}
    try:
        path = Path(path)
        root = path.parent
        payload = _safe_file(root, path.name).read_bytes()
        manifest = strict_json_loads(payload.decode("utf-8"))
        _require(type(manifest) is dict and manifest.get("scope") == SCOPE and manifest.get("status") == "COMPLETE",
                 "dataset-manifest-scope-invalid")
        config = dataset_v1_config_from_dict(manifest["config"])
        original = copy.deepcopy(manifest)
        snapshot = original.pop("snapshot_sha256")
        _require(snapshot == canonical_sha256(original), "dataset-snapshot-hash-mismatch")
        _require(manifest["implementation_sha256"] == _implementation_hashes(), "dataset-implementation-changed")
        # Keep source stat fences around native replay. Native gates decode and
        # hash every PNG; this avoids a third read of the whole image archive.
        before = {}
        for relative in manifest["source_files_sha256"]:
            source = _safe_file(root, relative)
            stat = source.stat()
            before[relative] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        episodes = [{key: item[key] for key in DESCRIPTOR_FIELDS} for item in manifest["episodes"]]
        derived, gates = _derive(root, manifest["routes"], episodes, config, replay_native=True, progress_callback=progress_callback)
        derived["snapshot_sha256"] = canonical_sha256(derived)
        _require(manifest == derived, "dataset-source-or-selection-reconstruction-mismatch")
        _require(path.read_bytes() == payload, "dataset-manifest-changed-during-validation")
        for relative, identity in before.items():
            stat = _safe_file(root, relative).stat()
            _require(identity == (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns),
                     "dataset-source-changed-during-validation")
        _require(manifest["implementation_sha256"] == _implementation_hashes(), "dataset-implementation-changed")
        report.update(status="PASS", manifest_sha256=hashlib.sha256(payload).hexdigest(), snapshot_sha256=snapshot,
                      config_sha256=config.sha256, sample_count=manifest["sample_count"],
                      episode_count=len(gates), route_count=len(manifest["routes"]),
                      split_counts=manifest["split_counts"], role_counts=manifest["role_counts"],
                      pair_count=manifest["pair_count"], training_samples_per_arm=config.train_selection_samples,
                      geometry_role_counts=manifest["geometry_role_counts"],
                      minimum_cross_split_scene_distance_lower_bound_m=manifest["minimum_cross_split_scene_distance_lower_bound_m"],
                      eligible_active_recovery_anchors=manifest["selections"]["eligible_active_recovery_anchors"],
                      exclusion_counts=manifest["exclusion_counts"], episode_metrics=gates,
                      checks={name: True for name in VALIDATION_CHECKS})
    except (Exception, KeyboardInterrupt) as error:
        detail = {"code": getattr(error, "code", "dataset-validation-operation-failed"),
                  "exception_type": type(error).__name__}
        if isinstance(error, DatasetV1Error) and error.context is not None:
            detail["context"] = copy.deepcopy(error.context)
        report["errors"].append(detail)
    return report
