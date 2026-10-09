"""Moving RGB/waypoint pilot collection, source integrity and grouped splits.

The native privileged controller supplies labels, never inference inputs. All
conditions for a physical scene stay together; the final town stays unopened.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from .contracts import (ContractError, canonical_sha256, read_strict_json, route_sha256,
                        strict_json_loads, validate_identifier, validate_route_contract,
                        validate_waypoint_label)
from .driving_route import FixedRouteTracker
from .geometry_survey import CASES, classify_geometry, survey_geometry_route
from .routes import RouteError, corridor_distance
from .splits import _scene

SCOPE = "moving-rgb-waypoint-pilot"
GATE_VERSION = "vlf-moving-pilot-v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/pilot.json"
SAMPLE_FIELDS = {"sample_id", "episode_id", "route_id", "scene_group_id", "split", "case", "condition",
                 "frame", "timestamp", "image", "image_sha256", "image_width", "image_height",
                 "waypoints_ego_m", "speed_mps", "reference_arc_m", "vehicle_transform", "calibration_sha256",
                 "target_lane", "weather", "perturbation"}


class PilotError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _require(condition, code):
    if not condition:
        raise PilotError(code)


def _finite(value):
    _require(type(value) in (int, float) and math.isfinite(value), "pilot-nonfinite-number")
    return float(value)


@dataclass(frozen=True)
class PilotConfig:
    schema_version: int
    dataset_version: str
    minimum_samples: int
    maximum_samples: int
    minimum_train_samples: int
    minimum_dev_samples: int
    minimum_speed_mps: float
    minimum_step_distance_m: float
    scene_buffer_m: float
    train_routes_per_case: dict
    dev_routes_per_case: dict
    conditions: list

    def to_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return canonical_sha256(self.to_dict())


def pilot_config_from_dict(value):
    names = {item.name for item in fields(PilotConfig)}
    _require(type(value) is dict and set(value) == names, "pilot-config-fields-invalid")
    _require(type(value["schema_version"]) is int and value["schema_version"] == 1, "pilot-config-schema-invalid")
    validate_identifier(value["dataset_version"])
    for key in ("minimum_samples", "maximum_samples", "minimum_train_samples", "minimum_dev_samples"):
        _require(type(value[key]) is int and value[key] > 0, "pilot-config-sample-count-invalid")
    _require(2000 <= value["minimum_samples"] <= value["maximum_samples"] <= 5000
             and value["minimum_train_samples"] + value["minimum_dev_samples"] <= value["minimum_samples"],
             "pilot-config-sample-budget-invalid")
    _require(.05 <= _finite(value["minimum_speed_mps"]) <= 1.
             and .001 <= _finite(value["minimum_step_distance_m"]) <= .05
             and _finite(value["scene_buffer_m"]) >= 75., "pilot-config-motion-or-buffer-invalid")
    for key in ("train_routes_per_case", "dev_routes_per_case"):
        counts = value[key]
        _require(type(counts) is dict and set(counts) == set(CASES)
                 and all(type(count) is int and 1 <= count <= 3 for count in counts.values()),
                 "pilot-config-route-count-invalid")
    from .driving import load_validation_config
    _require(value["conditions"] == load_validation_config()["initial_conditions"], "pilot-config-condition-invalid")
    return PilotConfig(**copy.deepcopy(value))


def load_pilot_config(path=DEFAULT_CONFIG):
    return pilot_config_from_dict(read_strict_json(path))


def _lower_bound(first, second):
    # Any point on an anchor segment is at most half its chord from an endpoint.
    # Subtract both radii: the result is conservative for the whole corridors.
    return max(0., corridor_distance(first, second) -
               .5 * (first["sample_step_m"] + second["sample_step_m"]))


def select_pilot_routes(candidates, config):
    """Choose fixed corridors before collection; refuse missing held-out curves."""
    _require(type(candidates) is list and candidates, "pilot-candidate-routes-empty")
    grouped = {case: [] for case in CASES}
    seen = set()
    for item in candidates:
        route, case = item["route"], item["case"]
        validate_route_contract(route)
        _require(route["town"] == "Town01" and case in CASES and classify_geometry(route)["case"] == case,
                 "pilot-candidate-route-invalid")
        _require(route["spawn_point_index"] not in seen, "pilot-candidate-route-duplicate")
        seen.add(route["spawn_point_index"])
        grouped[case].append(copy.deepcopy(item))
    for items in grouped.values():
        items.sort(key=lambda item: item["route"]["spawn_point_index"])
    _require(all(grouped.values()), "pilot-geometry-case-unavailable")
    cache = {}

    def distant(first, second):
        ids = tuple(sorted((first["route"]["spawn_point_index"], second["route"]["spawn_point_index"])))
        if ids not in cache:
            cache[ids] = _lower_bound(first["route"], second["route"])
        return cache[ids] > config.scene_buffer_m

    def pick(items, count, opposite):
        selected = []
        for item in items:
            if not all(distant(item, other) for other in opposite):
                continue
            # Extra same-case routes must add physical coverage, not clone it.
            if any(_lower_bound(item["route"], other["route"]) <= 35. for other in selected):
                continue
            selected.append(item)
            if len(selected) == count:
                return selected
        return None

    # Curves are scarce. Resolve their split first, then choose the abundant
    # straight corridors; split choices never depend on labels or model scores.
    from itertools import combinations, product
    for left_dev, right_dev in product(combinations(grouped["left_curve"], config.dev_routes_per_case["left_curve"]),
                                       combinations(grouped["right_curve"], config.dev_routes_per_case["right_curve"])):
        curves_dev = [*left_dev, *right_dev]
        left_train = pick(grouped["left_curve"], config.train_routes_per_case["left_curve"], curves_dev)
        right_train = pick(grouped["right_curve"], config.train_routes_per_case["right_curve"], curves_dev)
        if left_train is None or right_train is None:
            continue
        curves_train = [*left_train, *right_train]
        for straight_dev in combinations(grouped["straight"], config.dev_routes_per_case["straight"]):
            if not all(distant(item, other) for item in straight_dev for other in curves_train):
                continue
            dev = [*straight_dev, *curves_dev]
            straight_train = pick(grouped["straight"], config.train_routes_per_case["straight"], dev)
            if straight_train is None:
                continue
            selected = [{"split": "train", **item} for item in [*straight_train, *curves_train]] + [
                {"split": "dev", **item} for item in dev]
            # Connected nearby routes share one scene identity, even if the
            # vehicle direction and road ID differ. Cross-split edges cannot occur.
            parent = list(range(len(selected)))
            def find(index):
                while parent[index] != index:
                    index = parent[index]
                return index
            for index, first in enumerate(selected):
                for second_index in range(index + 1, len(selected)):
                    if _lower_bound(first["route"], selected[second_index]["route"]) <= config.scene_buffer_m:
                        parent[find(second_index)] = find(index)
            for index, item in enumerate(selected):
                route = item["route"]
                route["scene_group_id"] = f"town01_pilot_scene_{find(index):03d}"
                route["route_sha256"] = route_sha256(route)
            return selected
    raise PilotError("pilot-separated-train-dev-geometry-unavailable")


def survey_pilot_routes(carla_map, carla_module, config):
    """Read Town01 native topology without ticking, loading or inspecting Town02."""
    _require(carla_map.name.replace("\\", "/").rsplit("/", 1)[-1] == "Town01", "pilot-reserved-or-unexpected-town")
    points, source = carla_map.get_spawn_points(), carla_map.to_opendrive()
    view = SimpleNamespace(name=carla_map.name, get_spawn_points=lambda: points,
                           get_waypoint=carla_map.get_waypoint, to_opendrive=lambda: source)
    candidates, rejected = [], {}
    for index in range(len(points)):
        try:
            route = survey_geometry_route(view, carla_module, index, f"town01_pilot_p{index:03d}",
                                          f"town01_pilot_candidate_p{index:03d}")
            geometry = classify_geometry(route)
            candidates.append({"case": geometry["case"], "route": route})
        except RouteError as error:
            rejected[error.code] = rejected.get(error.code, 0) + 1
    selected = select_pilot_routes(candidates, config)
    return {"schema_version": 1, "town": "Town01", "selection": "native-spawn-order-separated-geometry-groups",
            "world_ticks_issued": 0, "world_mutated": False, "spawn_points_scanned": len(points),
            "candidate_counts": {case: sum(item["case"] == case for item in candidates) for case in CASES},
            "rejected_candidate_counts": rejected, "routes": selected,
            "reserved_final_town": "Town02", "reserved_final_state": "UNOPENED"}


def _safe_file(root, relative):
    _require(isinstance(relative, str) and relative and "\\" not in relative and ":" not in relative,
             "pilot-source-path-invalid")
    value = PurePosixPath(relative)
    _require(not value.is_absolute() and ".." not in value.parts and "." not in value.parts,
             "pilot-source-path-escaped")
    path = root.joinpath(*value.parts)
    _require(not root.is_symlink() and all(not parent.is_symlink() for parent in (path, *path.parents)
                 if parent == root or parent.is_relative_to(root)), "pilot-source-symlink")
    _require(path.resolve().is_relative_to(root.resolve()) and path.is_file(), "pilot-source-file-missing")
    return path


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _scene_distance(first, second):
    """Exact XY segment distance in bounded CPU blocks, including intersections."""
    import numpy as np
    if first["town"] != second["town"]:
        return math.inf
    minimum = math.inf
    source = np.asarray(first["segments"], dtype=np.float64)
    target = np.asarray(second["segments"], dtype=np.float64)
    _require(source.ndim == target.ndim == 3 and source.shape[1:] == target.shape[1:] == (2, 2)
             and np.isfinite(source).all() and np.isfinite(target).all(), "pilot-scene-geometry-invalid")

    def points_to_segments(points, starts, ends):
        delta = ends - starts
        offset = points[:, None, :] - starts[None, :, :]
        square = np.sum(delta * delta, axis=1)
        fraction = np.divide(np.sum(offset * delta[None, :, :], axis=2), square[None, :],
                             out=np.zeros(offset.shape[:2]), where=square[None, :] > 0)
        fraction = np.clip(fraction, 0., 1.)
        return np.sum((offset - fraction[:, :, None] * delta[None, :, :]) ** 2, axis=2)

    def orientation(delta, offset):
        return delta[..., 0] * offset[..., 1] - delta[..., 1] * offset[..., 0]

    for begin in range(0, len(source), 256):
        block = source[begin:begin + 256]
        a, b, c, d = block[:, 0], block[:, 1], target[:, 0], target[:, 1]
        ab, cd = b - a, d - c
        ac, ad = c[None] - a[:, None], d[None] - a[:, None]
        ca, cb = a[:, None] - c[None], b[:, None] - c[None]
        crossed = ((orientation(ab[:, None], ac) * orientation(ab[:, None], ad) < 0)
                   & (orientation(cd[None], ca) * orientation(cd[None], cb) < 0))
        if np.any(crossed):
            return 0.
        square = min(float(np.min(points_to_segments(a, c, d))), float(np.min(points_to_segments(b, c, d))),
                     float(np.min(points_to_segments(c, a, b))), float(np.min(points_to_segments(d, a, b))))
        minimum = min(minimum, math.sqrt(square))
    return minimum


def _read_episode(root, descriptor):
    relative = descriptor["episode"]
    metadata_path = _safe_file(root, relative + "/episode.json")
    trace_path = _safe_file(root, relative + "/steps.jsonl")
    metadata_bytes, trace_bytes = metadata_path.read_bytes(), trace_path.read_bytes()
    metadata = strict_json_loads(metadata_bytes.decode("utf-8"))
    lines = trace_bytes.decode("utf-8").splitlines()
    _require(lines and all(line.strip() for line in lines), "pilot-empty-or-blank-trace")
    return metadata, [strict_json_loads(line) for line in lines], metadata_bytes, trace_bytes


def _moving_samples(metadata, rows, descriptor, config):
    """Derive samples from measured consecutive motion, retaining same-frame GT."""
    route, samples = metadata["route_contract"], []
    calibration = canonical_sha256(metadata["camera_calibration"])
    previous = None
    for row in rows:
        current = row["vehicle"]["transform"]["location"]
        distance = 0. if previous is None else math.hypot(current["x"] - previous["x"], current["y"] - previous["y"])
        previous = current
        if row["speed_mps"] < config.minimum_speed_mps or distance < config.minimum_step_distance_m:
            continue
        image, label = row["image"], row["tracking"]["label"]
        validate_waypoint_label(label, row, {"route_id": route["route_id"], "route_contract": route})
        samples.append({"sample_id": f"{metadata['episode_id']}:{row['frame']:010d}",
                        "episode_id": metadata["episode_id"], "route_id": route["route_id"],
                        "scene_group_id": route["scene_group_id"], "split": descriptor["split"],
                        "case": descriptor["case"], "condition": descriptor["condition"],
                        "frame": row["frame"], "timestamp": row["timestamp"],
                        "image": descriptor["episode"] + "/" + image["path"], "image_sha256": image["sha256"],
                        "image_width": image["width"], "image_height": image["height"],
                        "waypoints_ego_m": copy.deepcopy(label["waypoints_ego_m"]), "speed_mps": row["speed_mps"],
                        "reference_arc_m": row["tracking"]["reference_arc_m"],
                        "vehicle_transform": copy.deepcopy(row["vehicle"]["transform"]), "calibration_sha256": calibration,
                        "target_lane": copy.deepcopy(label["target_lane"]),
                        "weather": metadata["collection_config"]["weather"]["preset"],
                        "perturbation": {key: metadata["condition"][key] for key in ("lateral_m", "yaw_deg")}})
    return samples


def build_pilot_manifest(directory, routes, episodes, config):
    """Build a new immutable manifest from native sources; no padding or reuse."""
    root, descriptors, samples, seen_images = Path(directory), [], [], {}
    rejected = {"stationary_or_startup_frames": 0, "duplicate_rgb_frames": 0}
    for item in episodes:
        descriptor = copy.deepcopy(item)
        metadata, rows, metadata_bytes, trace_bytes = _read_episode(root, descriptor)
        _require(metadata.get("status") == "COMPLETE" and metadata.get("method") == "privileged_reference"
                 and metadata.get("evaluation_metrics", {}).get("driving_success") is True,
                 "pilot-reference-driving-failed")
        route = metadata["route_contract"]
        descriptor.update(episode_id=metadata["episode_id"], route_id=route["route_id"], scene_group_id=route["scene_group_id"],
                          episode_json_sha256=_sha(metadata_bytes), steps_jsonl_sha256=_sha(trace_bytes))
        moving = _moving_samples(metadata, rows, descriptor, config)
        rejected["stationary_or_startup_frames"] += len(rows) - len(moving)
        for sample in moving:
            digest = sample["image_sha256"]
            if digest in seen_images:
                _require(seen_images[digest] == sample["split"], "pilot-cross-split-duplicate-rgb")
                rejected["duplicate_rgb_frames"] += 1
                continue
            seen_images[digest] = sample["split"]
            samples.append(sample)
        descriptors.append(descriptor)
    manifest = {"schema_version": 1, "dataset_version": config.dataset_version, "status": "COMPLETE", "scope": SCOPE,
                "town": "Town01", "reserved_final_town": "Town02", "reserved_final_state": "UNOPENED",
                "config": config.to_dict(), "config_sha256": config.sha256,
                "sample_count": len(samples), "split_counts": {split: sum(sample["split"] == split for sample in samples)
                                                              for split in ("train", "dev")},
                "routes": copy.deepcopy(routes), "episodes": descriptors, "samples": samples, "rejected_frames": rejected}
    path = root / "pilot_manifest.json"
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
    return path


def validate_pilot_manifest(path):
    """Independently replay every native frame, GT label, PNG and split boundary."""
    from .baseline_validation import validate_baseline_episode
    manifest_path = Path(path)
    root = manifest_path.parent
    report = {"status": "FAIL", "scope": SCOPE, "gate_version": GATE_VERSION, "errors": []}
    try:
        manifest_bytes = _safe_file(root, manifest_path.name).read_bytes()
        manifest = strict_json_loads(manifest_bytes.decode("utf-8"))
        _require(type(manifest) is dict and type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1
                 and manifest.get("status") == "COMPLETE" and manifest.get("scope") == SCOPE
                 and manifest.get("town") == "Town01" and manifest.get("reserved_final_town") == "Town02"
                 and manifest.get("reserved_final_state") == "UNOPENED", "pilot-manifest-scope-invalid")
        config = pilot_config_from_dict(manifest["config"])
        _require(manifest["dataset_version"] == config.dataset_version and manifest["config_sha256"] == config.sha256,
                 "pilot-config-hash-mismatch")
        route_items, route_lookup, route_counts, map_hashes = manifest["routes"], {}, {}, set()
        scene_splits = {}
        for item in route_items:
            _require(type(item) is dict and set(item) == {"split", "case", "route"}, "pilot-route-fields-invalid")
            route, split, case = item["route"], item["split"], item["case"]
            validate_route_contract(route)
            _require(split in ("train", "dev") and route["town"] == "Town01" and case in CASES
                     and classify_geometry(route)["case"] == case, "pilot-route-case-or-split-invalid")
            _require(route["route_id"] not in route_lookup, "pilot-route-identity-duplicate")
            _require(route["scene_group_id"] not in scene_splits or scene_splits[route["scene_group_id"]] == split,
                     "pilot-scene-group-split-leakage")
            scene_splits[route["scene_group_id"]] = split
            route_lookup[route["route_id"]] = item
            route_counts[split, case] = route_counts.get((split, case), 0) + 1
            map_hashes.add(route["opendrive_sha256"])
        _require(len(map_hashes) == 1 and all(route_counts.get((split, case), 0) ==
                 getattr(config, split + "_routes_per_case")[case] for split in ("train", "dev") for case in CASES),
                 "pilot-route-coverage-mismatch")
        descriptors, episode_ids, combinations_seen = manifest["episodes"], set(), set()
        expected_samples, scenes, gates, sources, seen_images = [], [], [], {}, {}
        rejected = {"stationary_or_startup_frames": 0, "duplicate_rgb_frames": 0}
        for descriptor in descriptors:
            required = {"episode_id", "route_id", "scene_group_id", "split", "case", "condition", "episode",
                        "episode_json_sha256", "steps_jsonl_sha256"}
            _require(type(descriptor) is dict and set(descriptor) == required, "pilot-episode-fields-invalid")
            validate_identifier(descriptor["episode_id"])
            route_item = route_lookup.get(descriptor["route_id"])
            _require(route_item is not None and descriptor["split"] == route_item["split"]
                     and descriptor["case"] == route_item["case"]
                     and descriptor["scene_group_id"] == route_item["route"]["scene_group_id"], "pilot-episode-route-identity-mismatch")
            identity = descriptor["route_id"], descriptor["condition"]
            _require(descriptor["episode_id"] not in episode_ids and identity not in combinations_seen,
                     "pilot-episode-or-condition-duplicate")
            episode_ids.add(descriptor["episode_id"])
            combinations_seen.add(identity)
            metadata, rows, metadata_bytes, trace_bytes = _read_episode(root, descriptor)
            _require(_sha(metadata_bytes) == descriptor["episode_json_sha256"]
                     and _sha(trace_bytes) == descriptor["steps_jsonl_sha256"], "pilot-episode-source-hash-mismatch")
            _require(metadata["episode_id"] == descriptor["episode_id"] and metadata["route_contract"] == route_item["route"]
                     and metadata["geometry_case"] == descriptor["case"]
                     and metadata["condition"]["name"] == descriptor["condition"]
                     and metadata["method"] == "privileged_reference", "pilot-native-source-identity-mismatch")
            gate = validate_baseline_episode(root / descriptor["episode"], allowed_conditions=config.conditions)
            _require(gate["status"] == "PASS", "pilot-native-episode-replay-failed")
            _require(gate["evaluation_metrics"]["driving_success"] is True, "pilot-reference-driving-failed")
            gates.append({"episode_id": descriptor["episode_id"], "metric": gate["evaluation_metrics"], "checks": gate["checks"]})
            sources[descriptor["episode"] + "/episode.json"] = _sha(metadata_bytes)
            sources[descriptor["episode"] + "/steps.jsonl"] = _sha(trace_bytes)
            sources.update({descriptor["episode"] + "/" + relative: digest
                            for relative, digest in gate["source"]["image_sha256"].items()})
            moving = _moving_samples(metadata, rows, descriptor, config)
            rejected["stationary_or_startup_frames"] += len(rows) - len(moving)
            for sample in moving:
                digest = sample["image_sha256"]
                if digest in seen_images:
                    _require(seen_images[digest] == sample["split"], "pilot-cross-split-duplicate-rgb")
                    rejected["duplicate_rgb_frames"] += 1
                    continue
                seen_images[digest] = sample["split"]
                expected_samples.append(sample)
            scene = _scene(dict(metadata, town="Town01"), rows)
            scenes.append({"split": descriptor["split"], **scene})
        expected_combinations = {(identifier, condition["name"]) for identifier in route_lookup for condition in config.conditions}
        _require(combinations_seen == expected_combinations, "pilot-incomplete-route-condition-coverage")
        samples = manifest["samples"]
        _require(type(samples) is list and all(type(sample) is dict and set(sample) == SAMPLE_FIELDS for sample in samples)
                 and samples == expected_samples and manifest["rejected_frames"] == rejected, "pilot-sample-source-or-filter-mismatch")
        _require(type(manifest["sample_count"]) is int and manifest["sample_count"] == len(samples)
                 and config.minimum_samples <= len(samples) <= config.maximum_samples, "pilot-sample-budget-not-met")
        counts = {split: sum(sample["split"] == split for sample in samples) for split in ("train", "dev")}
        _require(manifest["split_counts"] == counts and counts["train"] >= config.minimum_train_samples
                 and counts["dev"] >= config.minimum_dev_samples, "pilot-split-sample-budget-not-met")
        _require(len({sample["sample_id"] for sample in samples}) == len(samples)
                 and len({sample["image_sha256"] for sample in samples}) == len(samples), "pilot-sample-or-rgb-duplicate")
        minimum_distance = math.inf
        # Merge whole corridors and measured actor positions by route first;
        # this catches off-centre recovery entering the spatial exclusion zone.
        grouped_scenes = {}
        for descriptor, scene in zip(descriptors, scenes):
            identifier = descriptor["route_id"]
            if identifier not in grouped_scenes:
                grouped_scenes[identifier] = {"town": "Town01", "split": descriptor["split"], "segments": []}
            grouped_scenes[identifier]["segments"].extend(scene["segments"])
        merged = list(grouped_scenes.values())
        for scene in merged:
            scene["segments"] = list(set(scene["segments"]))
        for index, first in enumerate(merged):
            for second in merged[index + 1:]:
                if first["split"] != second["split"]:
                    distance = _scene_distance(first, second)
                    _require(distance > config.scene_buffer_m, "pilot-spatial-scene-split-leakage")
                    minimum_distance = min(minimum_distance, distance)
        _require(manifest_path.read_bytes() == manifest_bytes and all(_sha(_safe_file(root, relative).read_bytes()) == digest
                    for relative, digest in sources.items()), "pilot-source-changed-during-validation")
        report.update(status="PASS", manifest_sha256=_sha(manifest_bytes), dataset_version=config.dataset_version,
                      sample_count=len(samples), episode_count=len(descriptors), route_count=len(route_items),
                      split_counts=counts, rejected_frames=rejected,
                      geometry_counts={split: {case: sum(sample["split"] == split and sample["case"] == case for sample in samples)
                                               for case in CASES} for split in ("train", "dev")},
                      minimum_cross_split_scene_distance_m=minimum_distance, episode_metrics=gates,
                      checks={"native_controller_replay": True, "same_frame_gt_labels": True, "every_source_png": True,
                              "rgb_hashes_unique": True, "moving_frames_only": True, "route_episode_scene_groups": True,
                              "spatial_scene_buffer": True, "reserved_final_town_unopened": True, "source_bytes_unchanged": True})
    except (Exception, KeyboardInterrupt) as error:
        report["errors"].append({"code": getattr(error, "code", "pilot-validation-operation-failed"),
                                 "exception_type": type(error).__name__})
    return report
