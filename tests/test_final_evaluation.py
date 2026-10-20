"""Held-out pairing, physical independence and immutable plan semantics."""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
import unittest

from carla_tools.contracts import route_sha256, validate_route_contract
from carla_tools.controller import load_controller_config
from carla_tools.controller_study import ControllerStudyConfig
from carla_tools.final_evaluation import (
    FinalEvaluationError, REQUIRED_ARTIFACT_ROLES, ROUTE_SCOPE,
    audit_route_exclusion, build_manifest, corridor_separation, planned_trials,
    portable_path, runtime_route_contract, validate_manifest, validate_route,
    validate_route_catalog)
from carla_tools.guarded_path_correction import guarded_preprocessing_config
from carla_tools.selected_pipeline import (
    INPUT_CONTRACT, VERSION, pipeline_id, selected_pipeline_source_hashes)

try:
    from tests.test_data_contracts import route_fixture
except ModuleNotFoundError:
    from test_data_contracts import route_fixture


def selection_fixture():
    controller = ControllerStudyConfig(base=load_controller_config())
    value = {"schema_version": 1, "pipeline_version": VERSION, "status": "PASS",
             "model": {"kind": "refined", "architecture": "hybrid_geometry",
                       "checkpoint_path": "checkpoint.pt", "checkpoint_sha256": "a" * 64,
                       "model_contract_sha256": "b" * 64, "learning_config_sha256": "c" * 64},
             "preprocessing": {"input": INPUT_CONTRACT, **guarded_preprocessing_config()},
             "controller_config": controller.to_dict(),
             "controller_config_sha256": controller.sha256,
             "source_sha256": selected_pipeline_source_hashes()}
    value["pipeline_id"] = pipeline_id(value)
    return value


def routes_fixture():
    records = []
    for town in ("Town01", "Town02"):
        for index in range(2):
            route = route_fixture(x=300. * (index + 1),
                                  route_id=f"{town}_heldout_{index}",
                                  scene_group=f"{town}_heldout_scene_{index}")
            route.update(town=town, map_package="Carla/Maps/" + town,
                         spawn_point_index=None, selection_scope=ROUTE_SCOPE)
            for anchor in route["centerline_anchor_sequence"]:
                anchor["lane_width_m"] = 3.5
                anchor["world_rotation"] = {"pitch": 0., "yaw": 0., "roll": 0.}
            route["start"] = copy.deepcopy(route["centerline_anchor_sequence"][0])
            route["route_sha256"] = route_sha256(route)
            records.append(route)
    return {"schema_version": 1, "split": "final", "routes": records}


def bindings_fixture():
    return [{"path": "evidence/" + role + ".json", "sha256": "d" * 64, "roles": [role]}
            for role in sorted(REQUIRED_ARTIFACT_ROLES)]


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.routes, self.selection = routes_fixture(), selection_fixture()
        self.manifest = build_manifest(self.routes, self.selection, bindings_fixture())

    def test_ninety_trials_thirty_exactly_paired_states_and_distinct_physical_starts(self):
        trials = self.manifest["trials"]
        self.assertEqual(len(trials), 90)
        self.assertEqual(Counter(t["method"] for t in trials),
                         {"B0": 30, "M1": 30, "privileged": 30})
        pairs = defaultdict(list)
        for trial in trials:
            pairs[(trial["group_id"], trial["route_id"], trial["initial_state"]["state_id"])].append(trial)
        self.assertEqual(len(pairs), 30)
        for pair in pairs.values():
            self.assertEqual({t["method"] for t in pair}, {"B0", "M1", "privileged"})
            self.assertTrue(all(t["initial_state"] == pair[0]["initial_state"]
                                and t["environment"] == pair[0]["environment"] for t in pair))
        states = self.manifest["initial_states"]
        self.assertEqual(len({(s["initial_reference_m"], s["lateral_m"], s["yaw_deg"]) for s in states}), 3)
        self.assertEqual(Counter(trials[i]["method"] for i in range(0, 90, 3)),
                         {"B0": 10, "M1": 10, "privileged": 10})

    def test_final_weather_and_pitch_only_change_physical_environment(self):
        environments = {g["group_id"]: g["environment"] for g in self.manifest["groups"]}
        self.assertEqual(environments["C"]["weather"], "HardRainNoon")
        self.assertEqual(environments["D"]["weather"], "ClearSunset")
        self.assertEqual(environments["E"]["camera_pitch_delta_deg"], 2.)
        self.assertEqual(self.manifest["baseline_config"]["camera_pitch_deg"], -12.)
        self.assertEqual(self.manifest["preprocessing"], self.selection["preprocessing"])
        self.assertFalse(self.manifest["final_results_observed"])

    def test_omitted_duplicate_reordered_or_reassigned_trials_rejected(self):
        for change in ("missing", "duplicate", "reorder", "different_seed"):
            value = copy.deepcopy(self.manifest)
            if change == "missing":
                value["trials"].pop()
            elif change == "duplicate":
                value["trials"][0] = copy.deepcopy(value["trials"][1])
            elif change == "reorder":
                value["trials"].reverse()
            else:
                value["trials"][0]["initial_state"]["seed"] += 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_manifest(value, self.routes, self.selection)

    def test_settings_cannot_be_changed_even_when_trial_list_rebuilt(self):
        for key, value in (("route_length_m", 14.), ("warmup_frames", True),
                           ("expected_episodes", 89), ("final_results_observed", True),
                           ("selection_after_results_allowed", True)):
            changed = copy.deepcopy(self.manifest)
            changed[key] = value
            changed["trials"] = planned_trials(changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_manifest(changed, self.routes, self.selection)
        for field in ("controller_config", "baseline_config", "evaluation_config", "metrics", "retry_policy"):
            changed = copy.deepcopy(self.manifest)
            changed[field]["unexpected"] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_manifest(changed, self.routes, self.selection)

    def test_source_selection_and_guard_tampering_are_rejected(self):
        for change in ("source", "pipeline", "guard"):
            selection = copy.deepcopy(self.selection)
            if change == "source":
                selection["source_sha256"].pop(next(iter(selection["source_sha256"])))
            elif change == "pipeline":
                selection["pipeline_id"] = "0" * 64
            else:
                selection["preprocessing"]["guard_config"]["maximum_first_point_change_m"] = 5.
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_manifest(self.manifest, self.routes, selection)

    def test_boolean_policy_and_state_fields_are_not_interchangeable_with_integers(self):
        for change in ("automatic_retry", "state_coordinate"):
            manifest = copy.deepcopy(self.manifest)
            if change == "automatic_retry":
                manifest["retry_policy"]["automatic_retries"] = 0
            else:
                manifest["initial_states"][0]["initial_reference_m"] = False
                manifest["trials"] = planned_trials(manifest)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_manifest(manifest, self.routes, self.selection)

    def test_static_catalog_never_claims_invented_native_spawn_index(self):
        route = self.routes["routes"][0]
        before = copy.deepcopy(route)
        validate_route(route)
        runtime = runtime_route_contract(route, 7)
        validate_route_contract(runtime)
        self.assertEqual(runtime["spawn_point_index"], 7)
        self.assertEqual(route, before)
        self.assertIsNone(route["spawn_point_index"])
        self.assertNotEqual(runtime["route_sha256"], route["route_sha256"])
        wrong = copy.deepcopy(route)
        wrong["spawn_point_index"] = 0
        wrong["route_sha256"] = route_sha256(wrong)
        with self.assertRaisesRegex(FinalEvaluationError, "static-topology"):
            validate_route(wrong)

    def test_seen_corridor_is_rejected_despite_new_ids_and_scene_names(self):
        seen = route_fixture(x=305., route_id="train-old", scene_group="old-scene")
        with self.assertRaisesRegex(FinalEvaluationError, "near-observed"):
            audit_route_exclusion(self.routes, [seen])
        safe = route_fixture()
        report = audit_route_exclusion(self.routes, [safe])
        self.assertEqual(report["status"], "PASS")
        self.assertGreaterEqual(min(r["separation_m"] for r in report["comparisons"]), 75.)

    def test_continuous_segment_distance_detects_crossing_between_anchors(self):
        a = {"town": "Town01", "centerline_anchor_sequence": [
            {"world_location": {"x": -1., "y": 0.}}, {"world_location": {"x": 1., "y": 0.}}]}
        b = {"town": "Town01", "centerline_anchor_sequence": [
            {"world_location": {"x": 0., "y": -1.}}, {"world_location": {"x": 0., "y": 1.}}]}
        self.assertEqual(corridor_separation(a, b), 0.)
        b["town"] = "Town02"
        self.assertEqual(corridor_separation(a, b), float("inf"))

    def test_route_missing_town_duplicate_scene_bad_geometry_and_short_horizon_rejected(self):
        for change in ("town", "scene", "geometry", "heading"):
            routes = copy.deepcopy(self.routes)
            route = routes["routes"][-1]
            if change == "town":
                routes["routes"].pop()
            elif change == "scene":
                route["scene_group_id"] = routes["routes"][0]["scene_group_id"]
            elif change == "geometry":
                route["centerline_anchor_sequence"][10]["world_location"]["x"] += 100
            else:
                route["centerline_anchor_sequence"][10].pop("world_rotation")
            route["route_sha256"] = route_sha256(route)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_route_catalog(routes)

    def test_provenance_role_omission_duplicates_and_nonportable_paths_rejected(self):
        for transform in (lambda b: b[:-1], lambda b: b + [b[0]],
                          lambda b: b + [dict(b[0], path=b[0]["path"].upper())],
                          lambda b: [dict(b[0], sha256="invalid"), *b[1:]]):
            with self.assertRaises(ValueError):
                build_manifest(self.routes, self.selection, transform(bindings_fixture()))
        for path in ("../data.json", "/data.json", "C:/data.json", "a\\b", "a//b", "a/./b", "a:b"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                portable_path(path)


if __name__ == "__main__":
    unittest.main()
