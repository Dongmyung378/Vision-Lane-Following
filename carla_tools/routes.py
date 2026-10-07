"""Read-only D5 route topology survey; this does not generate frame labels."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


class RouteError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def canonical_hash(value: dict) -> str:
    payload = {key: item for key, item in value.items() if key != "route_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _anchor(waypoint: Any, distance: float, successor_count: int) -> dict:
    transform = waypoint.transform
    return {
        "road_id": int(waypoint.road_id), "section_id": int(waypoint.section_id),
        "lane_id": int(waypoint.lane_id), "s": float(waypoint.s),
        "world_location": {key: float(getattr(transform.location, key)) for key in ("x", "y", "z")},
        "world_rotation": {key: float(getattr(transform.rotation, key)) for key in ("pitch", "yaw", "roll")},
        "lane_width_m": float(waypoint.lane_width), "arc_distance_m": distance,
        "is_junction": bool(waypoint.is_junction), "successor_count": successor_count,
    }


def survey_route(carla_map: Any, carla_module: Any, spawn_point_index: int,
                 route_id: str, scene_group_id: str, length_m: int = 35) -> dict:
    """Pin one lane corridor and reject junctions, branches and lane ends.

    next(1m) follows the centerline, rather than fixing world or ego x positions.
    The selected target lane is retained in this contract for later labeling.
    """
    if type(length_m) is not int or not 35 <= length_m <= 500:
        raise RouteError("invalid-route-length")
    points = carla_map.get_spawn_points()
    if type(spawn_point_index) is not int or not 0 <= spawn_point_index < len(points):
        raise RouteError("spawn-index-unavailable")
    spawn = points[spawn_point_index]
    waypoint = carla_map.get_waypoint(spawn.location, project_to_road=True,
                                     lane_type=carla_module.LaneType.Driving)
    if waypoint is None:
        raise RouteError("spawn-has-no-driving-lane")
    lateral_distance = math.hypot(spawn.location.x - waypoint.transform.location.x,
                                  spawn.location.y - waypoint.transform.location.y)
    angle = (spawn.rotation.yaw - waypoint.transform.rotation.yaw + 180) % 360 - 180
    if lateral_distance > waypoint.lane_width / 2 or abs(angle) > 10:
        raise RouteError("spawn-lane-alignment-mismatch")
    anchors = []
    for distance in range(length_m + 1):
        if waypoint.is_junction or waypoint.lane_type != carla_module.LaneType.Driving:
            raise RouteError("junction-or-nondriving-lane")
        following = waypoint.next(1.0)
        if len(following) != 1:
            raise RouteError("route-branch-or-lane-end")
        anchors.append(_anchor(waypoint, float(distance), len(following)))
        waypoint = following[0]
    town = carla_map.name.replace("\\", "/").rsplit("/", 1)[-1]
    route = {
        "schema_version": 1, "route_id": route_id, "scene_group_id": scene_group_id,
        "town": town, "carla_version": "0.9.16", "spawn_point_index": spawn_point_index,
        "start": anchors[0], "horizon_m": 15.0, "min_required_route_m": float(length_m),
        "route_length_m": float(length_m), "sample_step_m": 1.0,
        "selection_scope": "D5-topology-only-not-frame-labels",
        "map_package": carla_map.name,
        "opendrive_sha256": hashlib.sha256(carla_map.to_opendrive().encode("utf-8")).hexdigest(),
        "centerline_anchor_sequence": anchors,
    }
    route["route_sha256"] = canonical_hash(route)
    return route


def corridor_distance(first: dict, second: dict) -> float:
    """Conservative scene separation, including opposite lane directions."""
    if first["town"] != second["town"]:
        return math.inf
    return min(math.hypot(a["world_location"]["x"] - b["world_location"]["x"],
                          a["world_location"]["y"] - b["world_location"]["y"])
               for a in first["centerline_anchor_sequence"] for b in second["centerline_anchor_sequence"])


def select_route_catalog(carla_map: Any, carla_module: Any, min_separation_m: float = 150) -> dict:
    if not math.isfinite(min_separation_m) or min_separation_m < 75:
        raise RouteError("invalid-route-separation")
    town = carla_map.name.replace("\\", "/").rsplit("/", 1)[-1]
    if town != "Town01":
        raise RouteError("unexpected-map")
    selected, rejected = [], {}
    for index in range(len(carla_map.get_spawn_points())):
        role = "train" if not selected else "dev"
        route_id = f"town01_{role}_p{index:03d}"
        try:
            route = survey_route(carla_map, carla_module, index, route_id, f"town01_scene_p{index:03d}")
        except RouteError as error:
            rejected[error.code] = rejected.get(error.code, 0) + 1
            continue
        if selected and corridor_distance(selected[0], route) < min_separation_m:
            rejected["near-existing-scene"] = rejected.get("near-existing-scene", 0) + 1
            continue
        selected.append(route)
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise RouteError("two-disjoint-routes-unavailable")
    return {
        "schema_version": 1, "status": "SURVEY_VERIFIED", "carla_version": "0.9.16",
        "town": town, "dataset_scope": "stationary-smoke-not-training-pilot",
        "route_selection": "first-valid-index-then-first-separated-valid-index",
        "min_separation_m": min_separation_m,
        "measured_corridor_separation_m": corridor_distance(*selected), "routes": selected,
        "rejected_candidate_counts": rejected,
        "final_test": {"town": "Town02", "status": "RESERVED_UNCOLLECTED_UNVERIFIED"},
    }


def load_route_catalog(path: Path) -> dict:
    from .contracts import ContractError, read_strict_json, validate_route_contract
    try:
        catalog = read_strict_json(path)
        if (type(catalog) is not dict or type(catalog.get("schema_version")) is not int or
                catalog["schema_version"] != 1 or catalog.get("status") != "SURVEY_VERIFIED" or
                catalog.get("carla_version") != "0.9.16" or catalog.get("town") != "Town01" or
                catalog.get("dataset_scope") != "stationary-smoke-not-training-pilot" or
                catalog.get("final_test") != {"town": "Town02", "status": "RESERVED_UNCOLLECTED_UNVERIFIED"}):
            raise RouteError("invalid-route-catalog")
        separation = catalog.get("min_separation_m")
        measured = catalog.get("measured_corridor_separation_m")
        if (type(separation) not in (int, float) or not math.isfinite(separation) or separation < 75 or
                type(measured) not in (int, float) or not math.isfinite(measured)):
            raise RouteError("invalid-route-separation")
        routes = catalog["routes"]
        if type(routes) is not list or len(routes) != 2:
            raise RouteError("invalid-route-catalog")
        for route in routes:
            validate_route_contract(route)
            if route["town"] != catalog["town"] or route["carla_version"] != catalog["carla_version"]:
                raise RouteError("route-catalog-identity-mismatch")
            if canonical_hash(route) != route["route_sha256"]:
                raise RouteError("route-hash-mismatch")
        if len({r["route_id"] for r in routes}) != 2 or len({r["scene_group_id"] for r in routes}) != 2:
            raise RouteError("duplicate-route-catalog-identities")
        if len({r["opendrive_sha256"] for r in routes}) != 1 or len({r["map_package"] for r in routes}) != 1:
            raise RouteError("route-catalog-map-source-mismatch")
        actual_distance = corridor_distance(*routes)
        if actual_distance < catalog["min_separation_m"] or not math.isclose(
                actual_distance, catalog["measured_corridor_separation_m"], rel_tol=0, abs_tol=1e-6):
            raise RouteError("route-separation-mismatch")
        return catalog
    except RouteError:
        raise
    except ContractError as error:
        raise RouteError(error.code) from None
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        raise RouteError("invalid-route-catalog") from None


def verify_route_in_world(route: dict, world_map: Any, carla_module: Any) -> None:
    actual = survey_route(world_map, carla_module, route["spawn_point_index"], route["route_id"],
                           route["scene_group_id"], int(route["route_length_m"]))
    if actual["route_sha256"] != route["route_sha256"]:
        raise RouteError("live-route-contract-mismatch")
