"""Read-only route selection, topology failures and pinned catalog integrity."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from carla_tools import routes


CARLA = SimpleNamespace(LaneType=SimpleNamespace(Driving="Driving"))


def transform(x, y):
    return SimpleNamespace(location=SimpleNamespace(x=float(x), y=float(y), z=.01),
                           rotation=SimpleNamespace(pitch=0., yaw=0., roll=0.))


class Waypoint:
    def __init__(self, x, y, distance=0, fail=None):
        self.road_id, self.section_id, self.lane_id, self.s = 1, 0, -1, float(distance)
        self.transform = transform(x + distance, y)
        self.lane_width, self.lane_type = 3.5, "Driving"
        self.is_junction = fail == "junction" and distance >= 15
        self.x, self.y, self.distance, self.fail = x, y, distance, fail

    def next(self, length):
        following = Waypoint(self.x, self.y, self.distance + length, self.fail)
        if self.distance >= 15 and self.fail == "branch":
            return [following, following]
        if self.distance >= 15 and self.fail == "end":
            return []
        return [following]


class Map:
    name = "/Game/Carla/Maps/Town01"

    def __init__(self, fail=None):
        self.points = [transform(0, 0), transform(0, 50), transform(0, 200)]
        self.fail = fail

    def get_spawn_points(self):
        return self.points

    def get_waypoint(self, location, **kwargs):
        return Waypoint(location.x, location.y, fail=self.fail)

    def to_opendrive(self):
        return "<OpenDRIVE>fixture</OpenDRIVE>"


class RouteTests(unittest.TestCase):
    def test_selects_fixed_first_valid_then_separated_corridor(self):
        catalog = routes.select_route_catalog(Map(), CARLA)
        self.assertEqual([r["spawn_point_index"] for r in catalog["routes"]], [0, 2])
        self.assertEqual(catalog["measured_corridor_separation_m"], 200.)
        self.assertEqual(catalog["rejected_candidate_counts"]["near-existing-scene"], 1)
        self.assertEqual(len(catalog["routes"][0]["centerline_anchor_sequence"]), 36)
        self.assertEqual(catalog["final_test"]["status"], "RESERVED_UNCOLLECTED_UNVERIFIED")

    def test_junction_branch_or_end_is_never_selected(self):
        for fail in ("junction", "branch", "end"):
            with self.subTest(fail=fail), self.assertRaises(routes.RouteError):
                routes.survey_route(Map(fail), CARLA, 0, "route", "scene")

    def test_invalid_index_and_ambiguous_spawn_alignment_are_rejected(self):
        with self.assertRaisesRegex(routes.RouteError, "spawn-index-unavailable"):
            routes.survey_route(Map(), CARLA, 999, "route", "scene")
        world_map = Map()
        world_map.points[0].rotation.yaw = 180
        with self.assertRaisesRegex(routes.RouteError, "spawn-lane-alignment-mismatch"):
            routes.survey_route(world_map, CARLA, 0, "route", "scene")

    def test_live_revalidation_detects_changed_map_source(self):
        world_map = Map()
        route = routes.survey_route(world_map, CARLA, 0, "route", "scene")
        routes.verify_route_in_world(route, world_map, CARLA)
        world_map.to_opendrive = lambda: "different source"
        with self.assertRaisesRegex(routes.RouteError, "live-route-contract-mismatch"):
            routes.verify_route_in_world(route, world_map, CARLA)

    def test_catalog_rejects_hash_tamper_and_duplicate_json(self):
        catalog = routes.select_route_catalog(Map(), CARLA)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "routes.json"
            path.write_text(json.dumps(catalog), encoding="utf-8")
            self.assertEqual(routes.load_route_catalog(path)["status"], "SURVEY_VERIFIED")
            catalog["routes"][0]["start"]["world_location"]["x"] += 1
            path.write_text(json.dumps(catalog), encoding="utf-8")
            with self.assertRaises(routes.RouteError):
                routes.load_route_catalog(path)
            path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaises(routes.RouteError):
                routes.load_route_catalog(path)

    def test_stationary_separation_is_direction_independent(self):
        catalog = routes.select_route_catalog(Map(), CARLA)
        reverse = copy.deepcopy(catalog["routes"][0])
        reverse["centerline_anchor_sequence"].reverse()
        self.assertEqual(routes.corridor_distance(catalog["routes"][0], reverse), 0.)

    def test_catalog_rejects_false_header_claims_and_invalid_separation(self):
        catalog = routes.select_route_catalog(Map(), CARLA)
        changes = [("schema_version", True), ("carla_version", "0.9.15"),
                   ("town", "Town02"), ("min_separation_m", -1),
                   ("min_separation_m", True), ("min_separation_m", 74),
                   ("measured_corridor_separation_m", float("inf")),
                   ("final_test", {"town": "Town02", "status": "VERIFIED"})]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "routes.json"
            for key, value in changes:
                invalid = copy.deepcopy(catalog)
                invalid[key] = value
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.subTest(key=key, value=value), self.assertRaises(routes.RouteError):
                    routes.load_route_catalog(path)


if __name__ == "__main__":
    unittest.main()
