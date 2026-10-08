"""Read-only D6 geometry selection from existing Town01 spawn points.

Surveyed routes retain the D5 topology contract. Geometry evidence is kept in
catalog wrappers and creates neither sensor frames nor waypoint labels.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .contracts import ContractError, read_strict_json, validate_route_contract
from .routes import RouteError, _anchor, canonical_hash


CASES = ("straight", "left_curve", "right_curve")
POLICY = {"horizon_m": 15.0, "straight_max_total_heading_deg": 3.0,
          "curve_min_net_heading_deg": 20.0, "curve_min_chord_heading_deg": 15.0,
          "minimum_direction_consistency": 0.9,
          "reference_sampling_step_m": 0.1, "physical_anchor_step_m": 0.25,
          "arc_distance_basis": "dense-XYZ-lane-centerline-polyline",
          "signed_heading_convention": "CARLA-positive-yaw-is-right"}


def _wrapped_change(first: float, second: float) -> float:
    return (second - first + 180.0) % 360.0 - 180.0


def _interpolate_anchor(first: dict, second: dict, target_arc: float) -> dict:
    fraction = (target_arc - first["arc_distance_m"]) / (second["arc_distance_m"] - first["arc_distance_m"])
    result = dict(first, arc_distance_m=target_arc)
    result["world_location"] = {axis: first["world_location"][axis] + fraction * (
        second["world_location"][axis] - first["world_location"][axis]) for axis in ("x", "y", "z")}
    result["world_rotation"] = {axis: (first["world_rotation"][axis] + fraction * _wrapped_change(
        first["world_rotation"][axis], second["world_rotation"][axis]) + 180) % 360 - 180
        for axis in ("pitch", "yaw", "roll")}
    if all(first[key] == second[key] for key in ("road_id", "section_id", "lane_id")):
        result["s"] = first["s"] + fraction * (second["s"] - first["s"])
    result["lane_width_m"] = first["lane_width_m"] + fraction * (second["lane_width_m"] - first["lane_width_m"])
    return result


def survey_geometry_route(carla_map: Any, carla_module: Any, spawn_point_index: int,
                          route_id: str, scene_group_id: str) -> dict:
    """Resample actual lane-centerline arc rather than OpenDRIVE reference s.

    On native Town01 outer bends, next(1m) produces 1.17-1.24m lane-centerline
    chords. Dense reference sampling followed by XYZ arc accumulation preserves
    the 15m physical label horizon without weakening D5 geometry checks.
    """
    points = carla_map.get_spawn_points()
    if type(spawn_point_index) is not int or not 0 <= spawn_point_index < len(points):
        raise RouteError("spawn-index-unavailable")
    spawn = points[spawn_point_index]
    waypoint = carla_map.get_waypoint(spawn.location, project_to_road=True, lane_type=carla_module.LaneType.Driving)
    if waypoint is None:
        raise RouteError("spawn-has-no-driving-lane")
    location = waypoint.transform.location
    if (math.hypot(spawn.location.x - location.x, spawn.location.y - location.y) > waypoint.lane_width / 2 or
            abs(_wrapped_change(spawn.rotation.yaw, waypoint.transform.rotation.yaw)) > 10):
        raise RouteError("spawn-lane-alignment-mismatch")
    dense, total = [], 0.0
    for _ in range(10000):
        if waypoint.is_junction or waypoint.lane_type != carla_module.LaneType.Driving:
            raise RouteError("junction-or-nondriving-lane")
        following = waypoint.next(POLICY["reference_sampling_step_m"])
        if len(following) != 1:
            raise RouteError("route-branch-or-lane-end")
        anchor = _anchor(waypoint, total, 1)
        dense.append(anchor)
        if total >= 35.0:
            break
        next_point = following[0]
        chord = math.dist([getattr(waypoint.transform.location, axis) for axis in ("x", "y", "z")],
                          [getattr(next_point.transform.location, axis) for axis in ("x", "y", "z")])
        if not math.isfinite(chord) or not 0 < chord <= 0.5:
            raise RouteError("invalid-dense-route-step")
        total += chord
        waypoint = next_point
    else:
        raise RouteError("dense-route-sampling-limit")
    if total < 35.0:
        raise RouteError("insufficient-route-horizon")
    anchors, dense_index = [], 0
    for index in range(141):
        target = index * POLICY["physical_anchor_step_m"]
        while dense_index + 1 < len(dense) and dense[dense_index + 1]["arc_distance_m"] < target:
            dense_index += 1
        if target == 0:
            anchors.append(dict(dense[0]))
        else:
            anchors.append(_interpolate_anchor(dense[dense_index], dense[dense_index + 1], target))
    route = {"schema_version": 1, "route_id": route_id, "scene_group_id": scene_group_id,
             "town": carla_map.name.replace("\\", "/").rsplit("/", 1)[-1], "carla_version": "0.9.16",
             "spawn_point_index": spawn_point_index, "start": anchors[0], "horizon_m": 15.0,
             "min_required_route_m": 35.0, "route_length_m": 35.0,
             "sample_step_m": POLICY["physical_anchor_step_m"], "selection_scope": "D5-topology-only-not-frame-labels",
             "map_package": carla_map.name,
             "opendrive_sha256": hashlib.sha256(carla_map.to_opendrive().encode("utf-8")).hexdigest(),
             "centerline_anchor_sequence": anchors}
    route["route_sha256"] = canonical_hash(route)
    try:
        validate_route_contract(route)
    except ContractError as error:
        raise RouteError(error.code) from None
    return route


def verify_geometry_route_in_world(route: dict, carla_map: Any, carla_module: Any) -> None:
    actual = survey_geometry_route(carla_map, carla_module, route["spawn_point_index"],
                                   route["route_id"], route["scene_group_id"])
    if actual["route_sha256"] != route["route_sha256"]:
        raise RouteError("live-geometry-route-contract-mismatch")


def classify_geometry(route: dict) -> dict:
    """Require substantial, consistent bending in the actual 15m geometry.

    Both waypoint yaw and position-derived chord headings must agree. Tiny
    heading changes and curves after the label horizon cannot prove a bend.
    """
    try:
        validate_route_contract(route)
        anchors = [anchor for anchor in route["centerline_anchor_sequence"] if anchor["arc_distance_m"] <= 15.0 + 1e-8]
        if not anchors or not math.isclose(anchors[-1]["arc_distance_m"], 15.0, abs_tol=1e-8):
            raise RouteError("geometry-horizon-not-sampled")
        yaws = [anchor["world_rotation"]["yaw"] for anchor in anchors]
        if any(type(yaw) not in (int, float) or not math.isfinite(yaw) for yaw in yaws):
            raise RouteError("invalid-geometry-heading")
        yaw_steps = [_wrapped_change(first, second) for first, second in zip(yaws, yaws[1:])]
        chords = [math.degrees(math.atan2(second["world_location"]["y"] - first["world_location"]["y"],
                                         second["world_location"]["x"] - first["world_location"]["x"]))
                  for first, second in zip(anchors, anchors[1:])]
        chord_steps = [_wrapped_change(first, second) for first, second in zip(chords, chords[1:])]
        net_yaw, total_yaw = sum(yaw_steps), sum(abs(value) for value in yaw_steps)
        net_chord, total_chord = sum(chord_steps), sum(abs(value) for value in chord_steps)
        consistency = abs(net_yaw) / total_yaw if total_yaw else 1.0
        if total_yaw <= POLICY["straight_max_total_heading_deg"] and total_chord <= POLICY["straight_max_total_heading_deg"]:
            case = "straight"
        elif (abs(net_yaw) >= POLICY["curve_min_net_heading_deg"] and
              abs(net_chord) >= POLICY["curve_min_chord_heading_deg"] and net_yaw * net_chord > 0 and
              consistency >= POLICY["minimum_direction_consistency"]):
            case = "right_curve" if net_yaw > 0 else "left_curve"
        else:
            raise RouteError("geometry-case-insufficient-or-mixed-bend")
        return {"case": case, "horizon_m": 15.0, "waypoint_net_heading_change_deg": net_yaw,
                "waypoint_total_heading_change_deg": total_yaw, "chord_net_heading_change_deg": net_chord,
                "chord_total_heading_change_deg": total_chord, "direction_consistency": consistency,
                "signed_heading_convention": POLICY["signed_heading_convention"]}
    except RouteError:
        raise
    except ContractError as error:
        raise RouteError(error.code) from None
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        raise RouteError("invalid-geometry-route") from None


def survey_geometry_catalog(carla_map: Any, carla_module: Any) -> dict:
    """Scan every fixed spawn index deterministically without ticking the world."""
    town = carla_map.name.replace("\\", "/").rsplit("/", 1)[-1]
    if town != "Town01":
        raise RouteError("unexpected-map")
    points = carla_map.get_spawn_points()
    source = carla_map.to_opendrive()
    # D5 survey_route can reuse these immutable map reads for all candidates.
    view = SimpleNamespace(name=carla_map.name, get_spawn_points=lambda: points,
                           get_waypoint=carla_map.get_waypoint, to_opendrive=lambda: source)
    selected, candidate_counts, rejected = {}, {case: 0 for case in CASES}, {}
    for index in range(len(points)):
        try:
            route = survey_geometry_route(view, carla_module, index, f"town01_geometry_p{index:03d}",
                                          f"town01_geometry_scene_p{index:03d}")
            geometry = classify_geometry(route)
            case = geometry["case"]
            candidate_counts[case] += 1
            if case not in selected:
                selected[case] = {"case": case, "route": route, "geometry": geometry}
        except RouteError as error:
            rejected[error.code] = rejected.get(error.code, 0) + 1
    cases = [selected[case] for case in CASES if case in selected]
    missing = [case for case in CASES if case not in selected]
    return {"schema_version": 1, "status": "GEOMETRY_SURVEY_VERIFIED" if not missing else "INCOMPLETE",
            "carla_version": "0.9.16", "town": town, "scope": "D6-static-geometry-QA-not-training-split",
            "selection": "first-valid-fixed-spawn-index-per-geometry-case",
            "classification_policy": dict(POLICY), "spawn_points_scanned": len(points),
            "candidate_counts": candidate_counts, "rejected_candidate_counts": rejected,
            "missing_cases": missing, "geometry_cases": cases, "routes": [item["route"] for item in cases],
            "frame_labels_generated": False, "world_ticks_issued": 0, "world_mutated": False}


def load_geometry_catalog(path: str | Path) -> dict:
    """Validate complete source-pinned geometry evidence without CARLA."""
    try:
        catalog = read_strict_json(path)
        if (type(catalog) is not dict or type(catalog.get("schema_version")) is not int or catalog["schema_version"] != 1 or
                catalog.get("status") != "GEOMETRY_SURVEY_VERIFIED" or catalog.get("carla_version") != "0.9.16" or
                catalog.get("town") != "Town01" or catalog.get("scope") != "D6-static-geometry-QA-not-training-split" or
                catalog.get("classification_policy") != POLICY or catalog.get("missing_cases") != [] or
                catalog.get("selection") != "first-valid-fixed-spawn-index-per-geometry-case" or
                catalog.get("frame_labels_generated") is not False or type(catalog.get("world_ticks_issued")) is not int or
                catalog["world_ticks_issued"] != 0 or
                catalog.get("world_mutated") is not False):
            raise RouteError("invalid-geometry-catalog")
        scanned = catalog.get("spawn_points_scanned")
        counts = catalog.get("candidate_counts")
        if (type(scanned) is not int or scanned < 3 or type(counts) is not dict or set(counts) != set(CASES) or
                any(type(value) is not int or not 1 <= value <= scanned for value in counts.values()) or
                sum(counts.values()) > scanned):
            raise RouteError("invalid-geometry-survey-counts")
        cases = catalog["geometry_cases"]
        if type(cases) is not list or [item.get("case") for item in cases] != list(CASES):
            raise RouteError("invalid-geometry-cases")
        routes = []
        for item in cases:
            if type(item) is not dict or set(item) != {"case", "route", "geometry"}:
                raise RouteError("invalid-geometry-cases")
            route = item["route"]
            validate_route_contract(route)
            if route["town"] != "Town01" or route.get("route_sha256") != canonical_hash(route):
                raise RouteError("geometry-route-source-mismatch")
            anchors = route["centerline_anchor_sequence"]
            if (route["spawn_point_index"] >= scanned or route["sample_step_m"] != POLICY["physical_anchor_step_m"] or
                    route["route_length_m"] != 35.0 or len(anchors) != 141 or
                    any(anchor["arc_distance_m"] != index * POLICY["physical_anchor_step_m"] for index, anchor in enumerate(anchors))):
                raise RouteError("geometry-physical-arc-policy-mismatch")
            actual = classify_geometry(route)
            if actual != item["geometry"] or actual["case"] != item["case"]:
                raise RouteError("geometry-evidence-mismatch")
            routes.append(route)
        if catalog.get("routes") != routes or len({route["spawn_point_index"] for route in routes}) != 3:
            raise RouteError("invalid-geometry-route-selection")
        if len({route["route_id"] for route in routes}) != 3 or len({route["scene_group_id"] for route in routes}) != 3:
            raise RouteError("duplicate-geometry-identities")
        if len({route["opendrive_sha256"] for route in routes}) != 1 or len({route["map_package"] for route in routes}) != 1:
            raise RouteError("geometry-map-source-mismatch")
        return catalog
    except RouteError:
        raise
    except ContractError as error:
        raise RouteError(error.code) from None
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        raise RouteError("invalid-geometry-catalog") from None
