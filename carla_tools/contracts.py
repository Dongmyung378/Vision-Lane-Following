"""D5 data contracts and future-label checks; no CARLA or label generation.

JSON Schema describes the files for tooling. These standard-library validators
also check cross-field geometry and identity, which JSON Schema cannot express.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


DEFAULT_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "configs" / "dataset_contract_d5.json"
HORIZON_ARC_M = [3, 6, 9, 12, 15]
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ContractError(ValueError):
    """A stable validation code without raw file contents or paths."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _object(value: Any, required: set[str], optional: set[str] = frozenset(), *, code: str) -> dict:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - optional:
        raise ContractError(code)
    return value


def _finite(value: Any, *, code: str = "nonfinite-number") -> float:
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except (OverflowError, TypeError):
        valid = False
    if not valid:
        raise ContractError(code)
    return float(value)


def _integer(value: Any, *, minimum: int | None = None, code: str = "invalid-integer") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or minimum is not None and value < minimum:
        raise ContractError(code)
    return value


def validate_identifier(value: Any) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ContractError("invalid-identifier")
    return value


def _xyz(value: Any) -> list[float]:
    _object(value, {"x", "y", "z"}, code="invalid-world-location")
    return [_finite(value[key]) for key in ("x", "y", "z")]


def _duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("duplicate-json-key")
        result[key] = value
    return result


def _bad_constant(value):
    raise ContractError("nonfinite-json")


def strict_json_loads(value: str) -> Any:
    try:
        parsed = json.loads(value, object_pairs_hook=_duplicates, parse_constant=_bad_constant)
        def inspect(item):
            if isinstance(item, float):
                _finite(item, code="nonfinite-json")
            elif isinstance(item, list):
                for child in item:
                    inspect(child)
            elif isinstance(item, dict):
                for child in item.values():
                    inspect(child)
        inspect(parsed)
        return parsed
    except (json.JSONDecodeError, UnicodeError, RecursionError) as error:
        raise ContractError("invalid-json") from error


def read_strict_json(path: str | Path) -> Any:
    try:
        return strict_json_loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as error:
        raise ContractError("json-read-failed") from error


def canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                             allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as error:
        raise ContractError("noncanonical-json") from error
    return hashlib.sha256(encoded).hexdigest()


def load_dataset_contract(path: str | Path = DEFAULT_CONTRACT_PATH) -> dict:
    """Load the exact D5 contract; changes require a new reviewed version."""
    value = read_strict_json(path)
    _object(value, {"schema_version", "contract_version", "input", "waypoint", "route", "split"},
            code="invalid-dataset-contract")
    expected = {
        "schema_version": 1, "contract_version": "vlf-dataset-d5-v1",
        "input": {"model_fields": ["rgb"], "privileged_fields": ["vehicle", "camera", "route_contract", "label"],
                  "camera_calibration_policy": "nominal-fixed"},
        "waypoint": {"coordinate_origin": "vehicle_actor_transform", "axes": "x-forward_y-right", "units": "m",
                     "distance_reference": "target-lane-centerline-arc-length", "horizon_arc_m": HORIZON_ARC_M,
                     "output_shape": [5, 2], "generation_status": "NOT_GENERATED", "geometry_tolerance_m": .001},
        "route": {"horizon_m": 15, "minimum_route_length_m": 35, "maximum_anchor_step_m": 1,
                  "junctions_allowed": False, "branches_allowed": False, "lane_changes_allowed": False},
        "split": {"allowed": ["train", "dev"], "group_keys": ["route_id", "episode_id", "scene_group_id"],
                  "same_town_scene_buffer_m": 75, "temporal_adjacency_seconds": 1,
                  "reserved_final": {"town": "Town02", "collection_status": "UNCOLLECTED",
                                     "verification_status": "UNVERIFIED", "available_after": "D42-freeze"}},
    }
    # Canonical bytes retain the distinction between a bool and an integer.
    if canonical_sha256(value) != canonical_sha256(expected):
        raise ContractError("unsupported-dataset-contract")
    return value


def route_sha256(route: Mapping[str, Any]) -> str:
    return canonical_sha256({key: value for key, value in route.items() if key != "route_sha256"})


def _anchor(anchor: Any) -> None:
    _object(anchor, {"road_id", "section_id", "lane_id", "s", "world_location", "arc_distance_m",
                     "is_junction", "successor_count"}, {"lane_width_m", "world_rotation"}, code="invalid-route-anchor")
    _integer(anchor["road_id"], minimum=0)
    _integer(anchor["section_id"], minimum=0)
    if _integer(anchor["lane_id"]) == 0:
        raise ContractError("invalid-target-lane")
    if _finite(anchor["s"]) < 0 or _finite(anchor["arc_distance_m"]) < 0:
        raise ContractError("invalid-route-distance")
    _xyz(anchor["world_location"])
    if anchor["is_junction"] is not False or type(anchor["successor_count"]) is not int or anchor["successor_count"] != 1:
        raise ContractError("ambiguous-route-horizon")
    if "lane_width_m" in anchor and _finite(anchor["lane_width_m"]) <= 0:
        raise ContractError("invalid-lane-width")
    if "world_rotation" in anchor:
        _object(anchor["world_rotation"], {"pitch", "yaw", "roll"}, code="invalid-route-rotation")
        for angle in anchor["world_rotation"].values():
            _finite(angle)


def validate_route_contract(route: Any) -> None:
    """Validate surveyed topology, not simulator execution or D6 labels."""
    _object(route, {"schema_version", "route_id", "scene_group_id", "town", "carla_version", "map_package",
                    "opendrive_sha256", "spawn_point_index", "start", "horizon_m", "min_required_route_m",
                    "sample_step_m", "route_length_m", "centerline_anchor_sequence", "selection_scope"},
            {"route_sha256"}, code="invalid-route-contract")
    for key in ("route_id", "scene_group_id", "town"):
        validate_identifier(route[key])
    if type(route["schema_version"]) is not int or route["schema_version"] != 1 or route["carla_version"] != "0.9.16":
        raise ContractError("unsupported-route-version")
    package = route["map_package"]
    if (not isinstance(package, str) or package not in (f"Carla/Maps/{route['town']}", f"/Game/Carla/Maps/{route['town']}")):
        raise ContractError("invalid-map-package")
    if not isinstance(route["opendrive_sha256"], str) or not _SHA256.fullmatch(route["opendrive_sha256"]):
        raise ContractError("invalid-map-hash")
    _integer(route["spawn_point_index"], minimum=0)
    if _finite(route["horizon_m"]) != 15 or _finite(route["min_required_route_m"]) < 35:
        raise ContractError("insufficient-route-horizon")
    step, length = _finite(route["sample_step_m"]), _finite(route["route_length_m"])
    if not 0 < step <= 1 or length < route["min_required_route_m"]:
        raise ContractError("insufficient-route-horizon")
    if route["selection_scope"] != "D5-topology-only-not-frame-labels":
        raise ContractError("invalid-route-evidence-scope")
    anchors = route["centerline_anchor_sequence"]
    if not isinstance(anchors, list) or len(anchors) < 2:
        raise ContractError("missing-route-anchors")
    for anchor in anchors:
        _anchor(anchor)
    _anchor(route["start"])
    if route["start"] != anchors[0] or abs(anchors[0]["arc_distance_m"]) > 1e-9:
        raise ContractError("route-start-mismatch")
    if abs(anchors[-1]["arc_distance_m"] - length) > 1e-6:
        raise ContractError("route-length-mismatch")
    for first, second in zip(anchors, anchors[1:]):
        difference = second["arc_distance_m"] - first["arc_distance_m"]
        if not 0 < difference <= step + 1e-6:
            raise ContractError("invalid-route-anchor-spacing")
        xyz_first, xyz_second = _xyz(first["world_location"]), _xyz(second["world_location"])
        chord = math.dist(xyz_first, xyz_second)
        if chord <= 0 or chord > difference + .01:
            raise ContractError("invalid-route-anchor-geometry")
    if "route_sha256" in route and route["route_sha256"] != route_sha256(route):
        raise ContractError("route-hash-mismatch")


def sample_id(episode_id: str, frame: int) -> str:
    return f"{validate_identifier(episode_id)}:{_integer(frame, minimum=0):010d}"


def _matrix(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in value):
        raise ContractError("invalid-transform-matrix")
    matrix = [[_finite(number) for number in row] for row in value]
    if any(abs(matrix[3][i] - target) > 1e-6 for i, target in enumerate((0, 0, 0, 1))):
        raise ContractError("invalid-transform-matrix")
    for i in range(3):
        for j in range(3):
            if abs(sum(matrix[i][k] * matrix[j][k] for k in range(3)) - (1 if i == j else 0)) > .001:
                raise ContractError("nonrigid-transform-matrix")
    determinant = (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
                   - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
                   + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))
    if abs(determinant - 1) > .001:
        raise ContractError("nonrigid-transform-matrix")
    return matrix


def _transform(value: Any) -> list[list[float]]:
    _object(value, {"location", "rotation", "matrix"}, code="invalid-transform")
    location = _xyz(value["location"])
    _object(value["rotation"], {"pitch", "yaw", "roll"}, code="invalid-rotation")
    for angle in value["rotation"].values():
        _finite(angle)
    matrix = _matrix(value["matrix"])
    if any(abs(matrix[i][3] - location[i]) > .001 for i in range(3)):
        raise ContractError("transform-location-mismatch")
    return matrix


def validate_episode_identity(metadata: Any) -> None:
    if not isinstance(metadata, dict):
        raise ContractError("invalid-episode-metadata")
    for key in ("dataset_version", "episode_id", "route_id", "scene_group_id", "town"):
        validate_identifier(metadata.get(key))
    route = metadata.get("route_contract")
    validate_route_contract(route)
    for key in ("route_id", "scene_group_id", "town"):
        if metadata[key] != route[key]:
            raise ContractError("episode-route-identity-mismatch")
    if "collection_config" in metadata:
        config = metadata["collection_config"]
        if (not isinstance(config, dict) or config.get("map") != route["town"]
                or not isinstance(config.get("vehicle"), dict)
                or config["vehicle"].get("spawn_point_index") != route["spawn_point_index"]):
            raise ContractError("episode-route-spawn-mismatch")
    if metadata.get("control_mode") != "stationary_D4_smoke" or metadata.get("labels_state") != "not-generated-D6":
        raise ContractError("unsupported-D5-episode-scope")
    perturbation = metadata.get("perturbation")
    _object(perturbation, {"lateral_m", "yaw_deg", "provenance"}, code="invalid-perturbation")
    _finite(perturbation["lateral_m"])
    _finite(perturbation["yaw_deg"])
    if not isinstance(perturbation["provenance"], str) or not perturbation["provenance"]:
        raise ContractError("invalid-perturbation-provenance")


def _route_point(route: dict, distance: float) -> list[float]:
    anchors = route["centerline_anchor_sequence"]
    if distance < 0 or distance > anchors[-1]["arc_distance_m"]:
        raise ContractError("label-outside-route-horizon")
    for first, second in zip(anchors, anchors[1:]):
        if first["arc_distance_m"] <= distance <= second["arc_distance_m"]:
            fraction = (distance - first["arc_distance_m"]) / (second["arc_distance_m"] - first["arc_distance_m"])
            a, b = _xyz(first["world_location"]), _xyz(second["world_location"])
            return [x + fraction * (y - x) for x, y in zip(a, b)]
    raise ContractError("label-outside-route-horizon")


def _reference_arc_for_pose(route: dict, location: dict) -> float:
    """Project onto the already pinned route, never onto a newly nearest lane."""
    closest_distance, closest_arc = math.inf, 0.
    anchors = route["centerline_anchor_sequence"]
    for first, second in zip(anchors, anchors[1:]):
        a, b = first["world_location"], second["world_location"]
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        denominator = dx * dx + dy * dy
        if not denominator:
            raise ContractError("invalid-route-anchor-geometry")
        fraction = max(0., min(1., ((location["x"] - a["x"]) * dx + (location["y"] - a["y"]) * dy) / denominator))
        distance = math.hypot(location["x"] - a["x"] - fraction * dx, location["y"] - a["y"] - fraction * dy)
        if distance < closest_distance:
            closest_distance = distance
            closest_arc = first["arc_distance_m"] + fraction * (second["arc_distance_m"] - first["arc_distance_m"])
    return closest_arc


def validate_waypoint_label(label: Any, frame: dict, metadata: dict) -> None:
    """Check a future D6 generated label against its fixed route and frame pose.

This does not create labels. NOT_GENERATED is represented by the frame envelope
with label=None, so zeros cannot masquerade as a generated target.
"""
    _object(label, {"schema_version", "ego_frame", "route_id", "target_lane", "coordinate_origin", "axes", "units",
                    "distance_reference", "horizon_arc_m", "reference_arc_m", "waypoints_world_m", "waypoints_ego_m"},
            code="invalid-waypoint-label")
    if type(label["schema_version"]) is not int or label["schema_version"] != 1:
        raise ContractError("unsupported-label-version")
    if (_integer(label["ego_frame"], minimum=0) != frame["frame"] or label["route_id"] != metadata["route_id"]):
        raise ContractError("label-frame-route-mismatch")
    route = metadata["route_contract"]
    validate_route_contract(route)
    lane = label["target_lane"]
    _object(lane, {"road_id", "section_id", "lane_id"}, code="invalid-target-lane")
    for key in lane:
        _integer(lane[key])
        if lane[key] != route["start"][key]:
            raise ContractError("label-target-lane-mismatch")
    expected = {"coordinate_origin": "vehicle_actor_transform", "axes": "x-forward_y-right", "units": "m",
                "distance_reference": "target-lane-centerline-arc-length", "horizon_arc_m": HORIZON_ARC_M}
    for key, value in expected.items():
        if canonical_sha256(label[key]) != canonical_sha256(value):
            raise ContractError("label-coordinate-contract-mismatch")
    reference = _finite(label["reference_arc_m"])
    projected_arc = _reference_arc_for_pose(route, frame["vehicle"]["transform"]["location"])
    if abs(projected_arc - reference) > .001:
        raise ContractError("label-reference-pose-mismatch")
    world, ego = label["waypoints_world_m"], label["waypoints_ego_m"]
    if not isinstance(world, list) or not isinstance(ego, list) or len(world) != 5 or len(ego) != 5:
        raise ContractError("invalid-waypoint-shape")
    matrix = _transform(frame["vehicle"]["transform"])
    for index, distance in enumerate(HORIZON_ARC_M):
        if not isinstance(world[index], list) or len(world[index]) != 3 or not isinstance(ego[index], list) or len(ego[index]) != 2:
            raise ContractError("invalid-waypoint-shape")
        xyz, xy = [_finite(value) for value in world[index]], [_finite(value) for value in ego[index]]
        expected_world = _route_point(route, reference + distance)
        if math.dist(xyz, expected_world) > .001:
            raise ContractError("label-world-route-mismatch")
        local = [sum(matrix[k][axis] * (xyz[k] - matrix[k][3]) for k in range(3)) for axis in range(2)]
        if math.dist(xy, local) > .001:
            raise ContractError("label-ego-transform-mismatch")


def validate_frame_contract(frame: Any, metadata: dict, contract: dict | None = None) -> None:
    if contract is not None and contract["contract_version"] != "vlf-dataset-d5-v1":
        raise ContractError("unsupported-dataset-contract")
    _object(frame, {"schema_version", "index", "frame", "world_frame", "snapshot_frame", "sensor_frame", "timestamp",
                    "snapshot_timestamp", "sensor_timestamp", "delta_seconds", "image", "vehicle", "camera", "label_state", "label"},
            {"tick_to_rgb_ms", "mount_matrix_max_abs_error", "rgb", "color_provenance"}, code="invalid-episode-frame")
    if type(frame["schema_version"]) is not int or frame["schema_version"] != 1:
        raise ContractError("unsupported-frame-version")
    _integer(frame["index"], minimum=0)
    identifier = _integer(frame["frame"], minimum=0)
    for key in ("world_frame", "snapshot_frame", "sensor_frame"):
        if _integer(frame[key], minimum=0) != identifier:
            raise ContractError("frame-identity-mismatch")
    timestamp = _finite(frame["timestamp"])
    if timestamp < 0 or _finite(frame["delta_seconds"]) <= 0:
        raise ContractError("invalid-frame-time")
    if any(abs(_finite(frame[key]) - timestamp) > 1e-5 for key in ("snapshot_timestamp", "sensor_timestamp")):
        raise ContractError("frame-timestamp-mismatch")
    image = _object(frame["image"], {"path", "sha256", "width", "height"}, code="invalid-image-record")
    path = image["path"]
    if (not isinstance(path, str) or "\\" in path or ":" in path or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts or path != f"images/{identifier:08d}.png"):
        raise ContractError("invalid-image-path")
    if not isinstance(image["sha256"], str) or not _SHA256.fullmatch(image["sha256"]):
        raise ContractError("invalid-image-hash")
    for key in ("width", "height"):
        _integer(image[key], minimum=1)
    if "color_provenance" in frame:
        from .color import ColorError, validate_color_provenance
        try:
            validate_color_provenance(frame["color_provenance"], image["width"], image["height"])
        except ColorError as error:
            raise ContractError(error.code) from None
    if "tick_to_rgb_ms" in frame and _finite(frame["tick_to_rgb_ms"]) < 0:
        raise ContractError("invalid-capture-latency")
    if "mount_matrix_max_abs_error" in frame and not 0 <= _finite(frame["mount_matrix_max_abs_error"]) <= .001:
        raise ContractError("invalid-mount-diagnostic")
    if "rgb" in frame:
        stats = _object(frame["rgb"], {"width", "height", "rgb_bytes", "channel_min", "channel_max", "spatially_nonconstant"},
                        code="invalid-rgb-diagnostic")
        for key in ("width", "height", "rgb_bytes"):
            _integer(stats[key], minimum=1)
        if (stats["width"] != image["width"] or stats["height"] != image["height"]
                or stats["rgb_bytes"] != image["width"] * image["height"] * 3 or stats["spatially_nonconstant"] is not True):
            raise ContractError("invalid-rgb-diagnostic")
        for key in ("channel_min", "channel_max"):
            if not isinstance(stats[key], list) or len(stats[key]) != 3:
                raise ContractError("invalid-rgb-diagnostic")
            for number in stats[key]:
                if not 0 <= _integer(number) <= 255:
                    raise ContractError("invalid-rgb-diagnostic")
        if (any(low > high for low, high in zip(stats["channel_min"], stats["channel_max"]))
                or not any(low < high for low, high in zip(stats["channel_min"], stats["channel_max"]))):
            raise ContractError("invalid-rgb-diagnostic")
    vehicle = _object(frame["vehicle"], {"transform", "velocity", "angular_velocity", "acceleration", "control"},
                      {"actor_id", "control_source"}, code="invalid-vehicle-record")
    if "actor_id" in vehicle:
        _integer(vehicle["actor_id"], minimum=0)
    if "control_source" in vehicle and vehicle["control_source"] != "client-cache-at-world-frame":
        raise ContractError("invalid-control-provenance")
    _transform(vehicle["transform"])
    for key in ("velocity", "angular_velocity", "acceleration"):
        _xyz(vehicle[key])
    if not isinstance(vehicle["control"], dict) or not vehicle["control"]:
        raise ContractError("invalid-vehicle-control")
    for value in vehicle["control"].values():
        if not isinstance(value, bool):
            _finite(value, code="invalid-vehicle-control")
    camera = _object(frame["camera"], {"transform"}, {"actor_id"}, code="invalid-camera-record")
    if "actor_id" in camera:
        _integer(camera["actor_id"], minimum=0)
    _transform(camera["transform"])
    if frame["label_state"] == "NOT_GENERATED":
        if frame["label"] is not None:
            raise ContractError("ungenerated-label-must-be-null")
    elif frame["label_state"] == "GENERATED":
        validate_waypoint_label(frame["label"], frame, metadata)
    else:
        raise ContractError("invalid-label-state")
