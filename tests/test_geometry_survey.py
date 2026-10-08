"""Geometry classification and deterministic read-only topology survey."""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from carla_tools.geometry_survey import (CASES, classify_geometry, load_geometry_catalog, survey_geometry_catalog,
                                        survey_geometry_route, verify_geometry_route_in_world)
from carla_tools.routes import RouteError, survey_route
from scripts import survey_geometry_routes


CARLA = SimpleNamespace(LaneType=SimpleNamespace(Driving="Driving"))


class Waypoint:
    def __init__(self, definition, index=0):
        self.definition, self.index = definition, index
        self.road_id, self.section_id, self.lane_id, self.s = 1, 0, -1, float(index)
        self.lane_width, self.lane_type = 3.5, "Driving"
        self.is_junction = definition.get("fail") == "junction" and index >= 15
        base_index = int(math.floor(index))
        fraction = index - base_index
        first, second = definition["positions"][base_index:base_index + 2]
        x, y = (first[axis] + fraction * (second[axis] - first[axis]) for axis in (0, 1))
        heading = definition["headings"][base_index] + fraction * (
            definition["headings"][base_index + 1] - definition["headings"][base_index])
        yaw = (heading + 180) % 360 - 180
        self.transform = SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=0.0),
                                         rotation=SimpleNamespace(pitch=0.0, yaw=yaw, roll=0.0))

    def next(self, distance):
        following = Waypoint(self.definition, self.index + distance)
        if self.index >= 15 and self.definition.get("fail") == "branch":
            return [following, following]
        if self.index >= 15 and self.definition.get("fail") == "end":
            return []
        return [following]


def definition(yaw_step=0.0, base_yaw=0.0, fail=None, mixed=False, inconsistent=False, origin_y=0.0, physical_scale=1.0):
    headings = [base_yaw + yaw_step * index for index in range(40)]
    if mixed:
        headings = [base_yaw + 2.0 * min(index, 15 - index) if index <= 15 else base_yaw for index in range(40)]
    positions = [(0.0, origin_y)]
    for first, second in zip(headings, headings[1:]):
        heading = base_yaw if inconsistent else (first + second) / 2.0
        x, y = positions[-1]
        positions.append((x + physical_scale * math.cos(math.radians(heading)), y + physical_scale * math.sin(math.radians(heading))))
    return {"headings": headings, "positions": positions, "fail": fail}


class Map:
    name = "Carla/Maps/Town01"

    def __init__(self, definitions=None):
        self.definitions = definitions or [definition(), definition(2.0, origin_y=100.0), definition(-2.0, origin_y=200.0)]
        self.points = [Waypoint(item).transform for item in self.definitions]
        self.read_calls = 0

    def get_spawn_points(self):
        return self.points

    def get_waypoint(self, location, **kwargs):
        for index, point in enumerate(self.points):
            if point.location is location:
                return Waypoint(self.definitions[index])
        raise AssertionError("Unknown spawn location")

    def to_opendrive(self):
        self.read_calls += 1
        return "<OpenDRIVE>synthetic-geometry-fixture</OpenDRIVE>"


class GeometrySurveyTests(unittest.TestCase):
    def route(self, item):
        return survey_route(Map([item]), CARLA, 0, "fixture-route", "fixture-scene")

    def test_curve_directions_follow_carla_positive_yaw_right_and_actual_path(self):
        for step, expected in ((0.0, "straight"), (2.0, "right_curve"), (-2.0, "left_curve")):
            with self.subTest(step=step):
                result = classify_geometry(self.route(definition(step)))
                self.assertEqual(result["case"], expected)
                self.assertAlmostEqual(result["waypoint_net_heading_change_deg"], step * 15)
                self.assertAlmostEqual(result["chord_net_heading_change_deg"], step * 14)

    def test_one_degree_variation_does_not_masquerade_as_a_curve(self):
        geometry = classify_geometry(self.route(definition(1.0 / 15)))
        self.assertEqual(geometry["case"], "straight")
        self.assertAlmostEqual(geometry["waypoint_net_heading_change_deg"], 1.0)

    def test_heading_wrap_has_no_false_reversal(self):
        geometry = classify_geometry(self.route(definition(2.0, base_yaw=179.0)))
        self.assertEqual(geometry["case"], "right_curve")
        self.assertAlmostEqual(geometry["waypoint_net_heading_change_deg"], 30.0)

    def test_mixed_bends_and_heading_only_bend_are_not_accepted(self):
        for item in (definition(mixed=True), definition(2.0, inconsistent=True), definition(0.5)):
            with self.subTest(item=item), self.assertRaisesRegex(RouteError, "geometry-case-insufficient-or-mixed-bend"):
                classify_geometry(self.route(item))

    def test_survey_selects_first_valid_index_for_each_case_and_caches_map_source(self):
        world_map = Map()
        result = survey_geometry_catalog(world_map, CARLA)
        self.assertEqual(result["status"], "GEOMETRY_SURVEY_VERIFIED")
        self.assertEqual([case["case"] for case in result["geometry_cases"]], list(CASES))
        self.assertEqual([route["spawn_point_index"] for route in result["routes"]], [0, 2, 1])
        self.assertEqual(result["candidate_counts"], {"straight": 1, "left_curve": 1, "right_curve": 1})
        self.assertEqual(result["world_ticks_issued"], 0)
        self.assertFalse(result["world_mutated"])
        self.assertFalse(result["frame_labels_generated"])
        self.assertEqual(world_map.read_calls, 1)

    def test_branch_junction_and_lane_end_candidates_are_excluded(self):
        invalid = [definition(2.0, fail=cause, origin_y=index * 100.0) for index, cause in enumerate(("branch", "junction", "end"))]
        world_map = Map(invalid + [definition(), definition(2.0, origin_y=400.0), definition(-2.0, origin_y=500.0)])
        result = survey_geometry_catalog(world_map, CARLA)
        self.assertEqual(result["status"], "GEOMETRY_SURVEY_VERIFIED")
        self.assertEqual([route["spawn_point_index"] for route in result["routes"]], [3, 5, 4])
        self.assertEqual(result["rejected_candidate_counts"], {"route-branch-or-lane-end": 2, "junction-or-nondriving-lane": 1})

    def test_missing_native_geometry_is_reported_incomplete_without_fabricated_route(self):
        result = survey_geometry_catalog(Map([definition(), definition(2.0, origin_y=100.0)]), CARLA)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["missing_cases"], ["left_curve"])
        self.assertEqual(len(result["routes"]), 2)

    def test_complete_catalog_loader_recomputes_geometry_and_source_hash(self):
        result = survey_geometry_catalog(Map(), CARLA)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            self.assertEqual(load_geometry_catalog(path)["status"], "GEOMETRY_SURVEY_VERIFIED")
            altered = copy.deepcopy(result)
            altered["geometry_cases"][1]["geometry"]["waypoint_net_heading_change_deg"] = -1.0
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(RouteError, "geometry-evidence-mismatch"):
                load_geometry_catalog(path)
            altered = copy.deepcopy(result)
            altered["geometry_cases"][0]["route"]["route_sha256"] = "0" * 64
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaises(RouteError):
                load_geometry_catalog(path)

    def test_loader_refuses_partial_catalog_false_header_and_duplicate_json(self):
        complete = survey_geometry_catalog(Map(), CARLA)
        incomplete = survey_geometry_catalog(Map([definition()]), CARLA)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.json"
            for value in (incomplete, dict(complete, schema_version=True), dict(complete, world_mutated=True),
                          dict(complete, world_ticks_issued=False), dict(complete, spawn_points_scanned=0)):
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(RouteError):
                    load_geometry_catalog(path)
            path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaises(RouteError):
                load_geometry_catalog(path)

    def test_outer_lane_reference_step_is_resampled_in_real_physical_arc_metres(self):
        world_map = Map([definition(-2.0, physical_scale=1.24)])
        legacy = survey_route(world_map, CARLA, 0, "legacy", "legacy-scene")
        from carla_tools.contracts import ContractError, validate_route_contract
        with self.assertRaisesRegex(ContractError, "invalid-route-anchor-geometry"):
            validate_route_contract(legacy)
        route = survey_geometry_route(world_map, CARLA, 0, "physical", "physical-scene")
        validate_route_contract(route)
        self.assertEqual(len(route["centerline_anchor_sequence"]), 141)
        self.assertEqual(route["centerline_anchor_sequence"][60]["arc_distance_m"], 15.0)
        self.assertEqual(route["centerline_anchor_sequence"][-1]["arc_distance_m"], 35.0)
        self.assertEqual(route["sample_step_m"], 0.25)
        evidence = classify_geometry(route)
        self.assertEqual(evidence["case"], "left_curve")
        self.assertAlmostEqual(evidence["waypoint_net_heading_change_deg"], -30.0 / 1.24, places=6)
        verify_geometry_route_in_world(route, world_map, CARLA)
        world_map.to_opendrive = lambda: "changed-map"
        with self.assertRaisesRegex(RouteError, "live-geometry-route-contract-mismatch"):
            verify_geometry_route_in_world(route, world_map, CARLA)

    def test_cli_validate_only_has_no_carla_import_or_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.json"
            path.write_text(json.dumps(survey_geometry_catalog(Map(), CARLA)), encoding="utf-8")
            with patch("builtins.print"), patch.object(survey_geometry_routes.importlib, "import_module", side_effect=AssertionError("No runtime import")):
                self.assertEqual(survey_geometry_routes.main(["--validate-only", "--output", str(path)]), 0)


if __name__ == "__main__":
    unittest.main()
