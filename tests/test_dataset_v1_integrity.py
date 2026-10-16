"""Adversarial checks for scene-matched, equal-budget training selections."""
from __future__ import annotations

import copy
import math
import random
import unittest
from dataclasses import replace

from carla_tools.dataset_v1 import (
    DatasetV1Error,
    _continuous_route_yaw_deg,
    _pairing_heading_error_deg,
    load_dataset_v1_config,
    match_reference_arc_pairs,
    select_training_sets,
)
from carla_tools.driving_route import FixedRouteTracker
from carla_tools.driving import route_initial_location
from test_data_contracts import transform_fixture
from test_dataset_native_contract import curved_vertex_fixture


def sample_fixture(identifier, *, role="central", group="scene__ClearNoon__start00",
                   split="train", case="straight", arc=1.0, condition=None):
    condition = condition or ("center" if role == "central" else "left_offset")
    return {"sample_id": identifier, "episode_id": "episode-" + group + "-" + condition,
            "route_id": "route-" + group.split("__")[0],
            "scene_group_id": "physical-" + group.split("__")[0],
            "split": split, "case": case, "condition": condition, "role": role,
            "pair_group_id": group, "reference_arc_m": arc,
            "lateral_error_m": 0.0 if role == "central" else 0.1,
            "heading_error_deg": 0.0,
            "pairing_heading_error_deg": 0.0,
            "weather": group.split("__")[1], "run_name": group.split("__")[2],
            "image": identifier + ".png", "image_sha256": "0" * 64,
            "frame": 10, "timestamp": 0.5,
            "waypoints_ego_m": [[float(x), 0.0] for x in (3, 6, 9, 12, 15)]}


class DatasetV1PairIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(load_dataset_v1_config(), pair_arc_tolerance_m=0.2,
                              minimum_pair_count=1, train_selection_samples=4,
                              mixed_recovery_fraction=0.5)

    def test_nearby_arcs_cannot_cross_weather_start_or_physical_group(self):
        groups = ("scene__ClearNoon__start00", "scene__CloudyNoon__start00",
                  "scene__ClearNoon__start04", "other__ClearNoon__start00")
        samples = [sample_fixture("center", group=groups[0])]
        for index, group in enumerate(groups):
            samples.append(sample_fixture("recovery-" + str(index), group=group, role="recovery", arc=1.01))
        pairs = match_reference_arc_pairs(samples, self.config)
        self.assertEqual({pair["recovery_sample_id"] for pair in pairs}, {"recovery-0"})
        self.assertEqual(pairs[0]["central_sample_id"], "center")
        self.assertAlmostEqual(pairs[0]["reference_arc_difference_m"], 0.01)

    def test_arc_matching_refuses_a_recovery_outside_the_frozen_tolerance(self):
        samples = [sample_fixture("center"),
                   sample_fixture("near", role="recovery", arc=1.19),
                   sample_fixture("far", role="recovery", arc=1.21)]
        pairs = match_reference_arc_pairs(samples, self.config)
        self.assertEqual({pair["recovery_sample_id"] for pair in pairs}, {"near"})
        self.assertLessEqual(pairs[0]["reference_arc_difference_m"], self.config.pair_arc_tolerance_m)

    def test_pair_identity_and_tie_breaks_do_not_depend_on_input_order(self):
        samples = [sample_fixture("central-b", arc=1.1), sample_fixture("central-a", arc=0.9),
                   sample_fixture("recovery-b", role="recovery", arc=1.0),
                   sample_fixture("recovery-a", role="recovery", arc=1.0)]
        expected = match_reference_arc_pairs(samples, self.config)
        shuffled = copy.deepcopy(samples)
        random.Random(19).shuffle(shuffled)
        self.assertEqual(match_reference_arc_pairs(shuffled, self.config), expected)
        self.assertEqual(len({pair["pair_id"] for pair in expected}), len(expected))
        self.assertEqual(len({pair["recovery_sample_id"] for pair in expected}), len(expected))

    def test_training_and_development_cannot_be_paired_as_one_scene(self):
        samples = [sample_fixture("center"), sample_fixture("other-split", role="recovery", split="dev")]
        with self.assertRaises(DatasetV1Error) as caught:
            match_reference_arc_pairs(samples, self.config)
        self.assertEqual(caught.exception.code, "dataset-pair-group-identity-mismatch")

    def test_recovered_frames_cannot_fill_the_active_recovery_training_budget(self):
        samples = []
        for index in range(8):
            samples.append(sample_fixture("central-" + str(index), arc=float(index + 1)))
            recovered = sample_fixture("recovered-" + str(index), role="recovery", arc=index + 1.01)
            recovered["lateral_error_m"] = 0.0
            samples.append(recovered)
        samples.append(sample_fixture("dev", split="dev", group="devscene__ClearNoon__start00"))
        pairs = match_reference_arc_pairs(samples, self.config)
        self.assertEqual(len(pairs), 8)
        self.assertTrue(all(pair["active_recovery"] is False for pair in pairs))
        with self.assertRaises(DatasetV1Error):
            select_training_sets(samples, pairs, self.config)

    def test_heading_difference_wraps_at_one_complete_turn(self):
        central = sample_fixture("central")
        recovery = sample_fixture("recovery", role="recovery", arc=1.01)
        central["heading_error_deg"] = 179.9
        recovery["heading_error_deg"] = -179.9
        central["pairing_heading_error_deg"] = 179.9
        recovery["pairing_heading_error_deg"] = -179.9
        recovery["lateral_error_m"] = 0.0
        pair = match_reference_arc_pairs([central, recovery], self.config)[0]
        self.assertAlmostEqual(pair["heading_difference_deg"], 0.2)
        self.assertFalse(pair["active_recovery"])

    def test_repeated_sample_ids_cannot_increase_the_pair_count(self):
        central = sample_fixture("central")
        recovery = sample_fixture("recovery", role="recovery", arc=1.01)
        with self.assertRaises(DatasetV1Error) as caught:
            match_reference_arc_pairs([central, recovery, copy.deepcopy(recovery)], self.config)
        self.assertEqual(caught.exception.code, "dataset-pair-sample-duplicate")

    def test_same_actor_yaw_across_a_chord_vertex_is_not_active_recovery(self):
        route, samples = curved_vertex_fixture(), []
        anchors = route["centerline_anchor_sequence"]
        vertex = next(index for index, anchor in enumerate(anchors) if anchor["arc_distance_m"] == 2.0)
        origin = anchors[vertex]["world_location"]
        next_point = anchors[vertex + 1]["world_location"]
        actor_yaw = math.degrees(math.atan2(next_point["y"] - origin["y"], next_point["x"] - origin["x"]))
        for role, sign, segment in (("central", -1, anchors[vertex - 1:vertex + 1]),
                                    ("recovery", 1, anchors[vertex:vertex + 2])):
            first, second = (point["world_location"] for point in segment)
            dx, dy = second["x"] - first["x"], second["y"] - first["y"]
            length = math.hypot(dx, dy)
            pose = transform_fixture(x=origin["x"] + sign * .0005 * dx / length,
                                     y=origin["y"] + sign * .0005 * dy / length, yaw=actor_yaw)
            observed = FixedRouteTracker(route).observe(pose, 10, .5)
            sample = sample_fixture(role, role=role, arc=observed["reference_arc_m"])
            sample.update(vehicle_transform=pose, lateral_error_m=observed["lateral_error_m"],
                          heading_error_deg=observed["heading_error_deg"],
                          pairing_heading_error_deg=_pairing_heading_error_deg(route, pose, observed["reference_arc_m"]))
            samples.append(sample)
        self.assertGreater(abs(samples[0]["heading_error_deg"] - samples[1]["heading_error_deg"]), 1.0)
        pair = match_reference_arc_pairs(samples, self.config)[0]
        self.assertLess(pair["heading_difference_deg"], .01)
        self.assertFalse(pair["active_recovery"])

    def test_natural_progress_on_a_tight_curve_does_not_count_as_recovery(self):
        route, samples = curved_vertex_fixture(radius=6.0), []
        for role, arc in (("central", 2.0), ("recovery", 2.14)):
            location, _ = route_initial_location(route, arc)
            pose = transform_fixture(x=location["x"], y=location["y"], yaw=math.degrees(arc / 6.0))
            sample = sample_fixture(role, role=role, arc=arc)
            sample.update(vehicle_transform=pose, lateral_error_m=0.0,
                          pairing_heading_error_deg=_pairing_heading_error_deg(route, pose, arc))
            samples.append(sample)
        self.assertGreater(abs(samples[0]["vehicle_transform"]["rotation"]["yaw"] -
                               samples[1]["vehicle_transform"]["rotation"]["yaw"]), 1.0)
        pair = match_reference_arc_pairs(samples, self.config)[0]
        self.assertAlmostEqual(pair["heading_difference_deg"], 0.0)
        self.assertFalse(pair["active_recovery"])

    def test_wrapped_route_interpolation_includes_endpoints_and_detects_actual_yaw_error(self):
        route = {"centerline_anchor_sequence": [
            {"arc_distance_m": 0.0, "world_rotation": {"yaw": 179.0}},
            {"arc_distance_m": 1.0, "world_rotation": {"yaw": -179.0}}]}
        self.assertEqual(_continuous_route_yaw_deg(route, 0.0), 179.0)
        self.assertEqual(_continuous_route_yaw_deg(route, 0.5), -180.0)
        self.assertEqual(_continuous_route_yaw_deg(route, 1.0), -179.0)
        self.assertAlmostEqual(_pairing_heading_error_deg(route, transform_fixture(yaw=-175.0), 0.5), 5.0)
        for outside in (-.01, 1.01, float("nan"), True):
            with self.subTest(reference=outside), self.assertRaises(DatasetV1Error):
                _continuous_route_yaw_deg(route, outside)

    def test_genuine_route_relative_yaw_difference_remains_active(self):
        route = curved_vertex_fixture()
        central = sample_fixture("central", arc=2.0)
        recovery = sample_fixture("recovery", role="recovery", arc=2.1)
        for sample, extra_yaw in ((central, 0.0), (recovery, 5.0)):
            arc = sample["reference_arc_m"]
            location, _ = route_initial_location(route, arc)
            pose = transform_fixture(x=location["x"], y=location["y"], yaw=math.degrees(arc / 10.0) + extra_yaw)
            sample.update(vehicle_transform=pose, lateral_error_m=0.0,
                          pairing_heading_error_deg=_pairing_heading_error_deg(route, pose, arc))
        pair = match_reference_arc_pairs([central, recovery], self.config)[0]
        self.assertAlmostEqual(pair["heading_difference_deg"], 5.0)
        self.assertTrue(pair["active_recovery"])

    def test_equal_budget_replaces_central_samples_and_keeps_common_development(self):
        samples = []
        for index in range(8):
            samples.append(sample_fixture("central-" + str(index), arc=float(index + 1)))
            samples.append(sample_fixture("recovery-" + str(index), role="recovery", arc=index + 1.01))
        for index in range(3):
            samples.append(sample_fixture("dev-" + str(index), split="dev",
                                          group="devscene__ClearNoon__start00", arc=float(index + 1)))
        pairs = match_reference_arc_pairs(samples, self.config)
        selected = select_training_sets(samples, pairs, self.config)
        central = selected["central_only"]
        mixed = selected["central_recovery"]
        a, b = central["train_sample_ids"], mixed["train_sample_ids"]
        self.assertEqual(len(a), 4)
        self.assertEqual(len(b), 4)
        self.assertEqual(len(set(a)), 4)
        self.assertEqual(len(set(b)), 4)
        self.assertTrue(all(identifier.startswith("central-") for identifier in a))
        recoveries = {identifier for identifier in b if identifier.startswith("recovery-")}
        self.assertEqual(len(recoveries), 2)
        replaced = {pair["central_sample_id"] for pair in pairs if pair["recovery_sample_id"] in recoveries}
        self.assertEqual(set(a) - set(b), replaced)
        self.assertEqual(len(set(a) & set(b)), 2)
        expected_dev = {"dev-0", "dev-1", "dev-2"}
        self.assertEqual(set(central["dev_sample_ids"]), expected_dev)
        self.assertEqual(central["dev_sample_ids"], mixed["dev_sample_ids"])
        self.assertEqual(central["dev_sample_ids"], selected["common_dev_sample_ids"])
        self.assertFalse(expected_dev & (set(a) | set(b)))
        shuffled = copy.deepcopy(samples)
        random.Random(29).shuffle(shuffled)
        self.assertEqual(select_training_sets(shuffled, match_reference_arc_pairs(shuffled, self.config), self.config), selected)


if __name__ == "__main__":
    unittest.main()
