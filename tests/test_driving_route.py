"""Pinned-route recovery signs, same-frame labels and temporal guards."""

from __future__ import annotations

import copy
import math
import unittest

from carla_tools.contracts import route_sha256, validate_waypoint_label
from carla_tools.driving_route import DrivingRouteError, FixedRouteTracker, perturbation_transform

try:
    from tests.test_data_contracts import route_fixture, transform_fixture
except ModuleNotFoundError:
    from test_data_contracts import route_fixture, transform_fixture


def dense_route(sign=0):
    route = route_fixture()
    anchors = []
    for index in range(141):
        arc = index / 4
        x = arc if sign == 0 else 30. * math.sin(arc / 30.)
        y = 0. if sign == 0 else sign * 30. * (1 - math.cos(arc / 30.))
        anchors.append({"road_id": 1, "section_id": 0, "lane_id": -1, "s": arc, "arc_distance_m": arc,
                        "world_location": {"x": x, "y": y, "z": 0.}, "is_junction": False,
                        "successor_count": 1, "lane_width_m": 3.5})
    route.update(start=copy.deepcopy(anchors[0]), sample_step_m=.25, centerline_anchor_sequence=anchors)
    route["route_sha256"] = route_sha256(route)
    return route


class DrivingRouteTests(unittest.TestCase):
    def test_straight_label_uses_actor_origin_and_same_frame(self):
        route = dense_route()
        pose = transform_fixture(y=.3, z=.02)
        result = FixedRouteTracker(route).observe(pose, 123, 6.15)
        self.assertEqual(result["label"]["ego_frame"], 123)
        self.assertEqual(result["label"]["waypoints_ego_m"], [[float(x), -.3] for x in (3, 6, 9, 12, 15)])
        self.assertAlmostEqual(result["lateral_error_m"], .3)
        self.assertEqual(result["lane_width_m"], 3.5)
        self.assertEqual(result["reference_world_m"], [0., 0., 0.])
        self.assertEqual(result["route_tangent_yaw_deg"], 0.)
        validate_waypoint_label(result["label"], {"frame": 123, "vehicle": {"transform": pose}},
                                {"route_id": route["route_id"], "route_contract": route})

    def test_left_right_curves_have_opposite_ego_y_and_target_lane_stays_original(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                result = FixedRouteTracker(dense_route(sign)).observe(transform_fixture(), 100, 5.)
                self.assertTrue(all(sign * point[1] > 0 for point in result["label"]["waypoints_ego_m"]))
                self.assertEqual(result["label"]["target_lane"], {"road_id": 1, "section_id": 0, "lane_id": -1})
                self.assertAlmostEqual(result["label"]["waypoints_world_m"][-1][0], 30 * math.sin(.5))
                self.assertAlmostEqual(result["label"]["waypoints_world_m"][-1][1], sign * 30 * (1 - math.cos(.5)))

    def test_recovery_heading_and_lateral_signs_carla_positive_is_right(self):
        for lateral in (-.5, -.3, .3, .5):
            for yaw in (-5., -3., 3., 5.):
                with self.subTest(lateral=lateral, yaw=yaw):
                    initial = transform_fixture()
                    pose = perturbation_transform(initial, lateral, yaw)
                    result = FixedRouteTracker(dense_route()).observe(pose, 1, .05)
                    self.assertAlmostEqual(result["lateral_error_m"], lateral)
                    self.assertAlmostEqual(result["heading_error_deg"], yaw)
                    self.assertEqual(initial, transform_fixture())

    def test_perturbation_uses_original_right_axis_before_yaw_and_preserves_z_pitch_roll(self):
        pose = transform_fixture(x=12., y=4., z=.3, yaw=90.)
        changed = perturbation_transform(pose, .5, -5.)
        self.assertAlmostEqual(changed["location"]["x"], 11.5)
        self.assertAlmostEqual(changed["location"]["y"], 4.)
        self.assertEqual(changed["location"]["z"], .3)
        self.assertEqual(changed["rotation"], {"pitch": 0., "yaw": 85., "roll": 0.})
        self.assertAlmostEqual(changed["matrix"][0][0], math.cos(math.radians(85.)))
        self.assertEqual(pose["location"]["x"], 12.)

    def test_perturbation_keeps_nonzero_pitch_and_roll_with_rigid_matrix(self):
        pitch, roll = math.radians(10.), math.radians(5.)
        cp, sp, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
        pose = {"location": {"x": 0., "y": 0., "z": .5},
                "rotation": {"pitch": 10., "yaw": 0., "roll": 5.},
                "matrix": [[cp, sp * sr, -sp * cr, 0.], [0., cr, sr, 0.],
                           [sp, -cp * sr, cp * cr, .5], [0., 0., 0., 1.]]}
        changed = perturbation_transform(pose, .3, 5.)
        self.assertEqual(changed["rotation"], {"pitch": 10., "yaw": 5., "roll": 5.})
        self.assertEqual(changed["location"], {"x": 0., "y": .3, "z": .5})
        for row in range(3):
            self.assertAlmostEqual(sum(changed["matrix"][row][column] ** 2 for column in range(3)), 1.)
        # The 5x2 label is computed from the actual tilted actor transform,
        # with z still participating in its inverse, and validated separately.
        result = FixedRouteTracker(dense_route()).observe(changed, 1, .05)
        self.assertEqual(result["label"]["ego_frame"], 1)

    def test_yaw_wrap_has_no_false_360_degree_error(self):
        route = dense_route()
        for anchor in route["centerline_anchor_sequence"]:
            point = anchor["world_location"]
            point["x"] = -point["x"]
        route["start"] = copy.deepcopy(route["centerline_anchor_sequence"][0])
        route["route_sha256"] = route_sha256(route)
        result = FixedRouteTracker(route).observe(transform_fixture(yaw=-179.), 1, .05)
        self.assertAlmostEqual(result["heading_error_deg"], 1.)
        changed = perturbation_transform(transform_fixture(yaw=179.), 0., 3.)
        self.assertEqual(changed["rotation"]["yaw"], -178.)

    def test_input_route_and_returned_copy_cannot_change_pinned_target_lane(self):
        route = dense_route()
        tracker = FixedRouteTracker(route)
        original_hash = tracker.route_sha256
        route["start"]["lane_id"] = 5
        returned = tracker.route
        returned["start"]["lane_id"] = 10
        result = tracker.observe(transform_fixture(), 1, .05)
        self.assertEqual(result["route_sha256"], original_hash)
        self.assertEqual(result["label"]["target_lane"]["lane_id"], -1)

    def test_native_width_is_interpolated_and_missing_width_not_fabricated(self):
        route = dense_route()
        route["centerline_anchor_sequence"][1]["lane_width_m"] = 3.7
        route["route_sha256"] = route_sha256(route)
        result = FixedRouteTracker(route).observe(transform_fixture(x=.125), 1, .05)
        self.assertAlmostEqual(result["lane_width_m"], 3.6)
        absent = FixedRouteTracker(route_fixture()).observe(transform_fixture(), 1, .05)
        self.assertIsNone(absent["lane_width_m"])

    def test_progress_reports_reverse_motion_without_masking_it_as_maximum(self):
        tracker = FixedRouteTracker(dense_route())
        tracker.observe(transform_fixture(x=.2), 1, .05)
        forward = tracker.observe(transform_fixture(x=.7), 2, .1)
        reverse = tracker.observe(transform_fixture(x=.5), 3, .15)
        self.assertAlmostEqual(forward["progress_m"], .5)
        self.assertAlmostEqual(reverse["progress_m"], .3)
        self.assertAlmostEqual(reverse["maximum_progress_m"], .5)
        self.assertAlmostEqual(reverse["reference_arc_m"], .5)

    def test_stale_frame_or_timestamp_is_refused_without_advancing_state(self):
        tracker = FixedRouteTracker(dense_route())
        tracker.observe(transform_fixture(), 100, 5.)
        for frame, timestamp, code in ((100, 5.05, "stale-driving-frame"), (99, 5.05, "stale-driving-frame"),
                                       (101, 5., "stale-driving-timestamp"), (101, 4.95, "stale-driving-timestamp")):
            with self.subTest(frame=frame, timestamp=timestamp), self.assertRaisesRegex(DrivingRouteError, code):
                tracker.observe(transform_fixture(x=.1), frame, timestamp)
        self.assertAlmostEqual(tracker.observe(transform_fixture(x=.2), 101, 5.05)["progress_m"], .2)

    def test_frame_timestamp_and_transform_nonfinite_bool_and_pose_mismatch_refused(self):
        for frame, timestamp in ((True, .05), (1, True), (-1, .05), (1, -.05), (1, float("nan"))):
            with self.subTest(frame=frame, timestamp=timestamp), self.assertRaises(DrivingRouteError):
                FixedRouteTracker(dense_route()).observe(transform_fixture(), frame, timestamp)
        for mutate in (lambda p: p["location"].update(x=float("inf")),
                       lambda p: p["matrix"][0].__setitem__(0, -1.),
                       lambda p: p["rotation"].update(yaw=90.),
                       lambda p: p.update(extra=1)):
            pose = transform_fixture()
            mutate(pose)
            with self.assertRaises(DrivingRouteError):
                FixedRouteTracker(dense_route()).observe(pose, 1, .05)

    def test_teleport_outside_progress_window_is_refused(self):
        tracker = FixedRouteTracker(dense_route())
        tracker.observe(transform_fixture(), 1, .05)
        with self.assertRaisesRegex(DrivingRouteError, "outside-progress-window"):
            tracker.observe(transform_fixture(x=8.), 2, 1.05)

    def test_sub_window_teleport_still_obeys_physical_speed_bound(self):
        tracker = FixedRouteTracker(dense_route(), max_speed_mps=4.)
        tracker.observe(transform_fixture(), 1, .05)
        with self.assertRaisesRegex(DrivingRouteError, "driving-progress-speed-bound"):
            tracker.observe(transform_fixture(x=1.), 2, .1)
        self.assertAlmostEqual(tracker.observe(transform_fixture(x=.2), 2, .1)["progress_m"], .2)

    def test_small_before_start_offset_is_explicitly_zero_reference_not_negative_label_arc(self):
        result = FixedRouteTracker(dense_route()).observe(transform_fixture(x=-.25, y=.3), 1, .05)
        self.assertEqual(result["reference_arc_m"], 0.)
        self.assertEqual(result["label"]["waypoints_world_m"][0], [3., 0., 0.])
        self.assertEqual(result["label"]["waypoints_ego_m"][0], [3.25, -.3])
        with self.assertRaisesRegex(DrivingRouteError, "before-route-start"):
            FixedRouteTracker(dense_route()).observe(transform_fixture(x=-.6), 1, .05)

    def test_short_horizon_never_clamps_or_repeats_last_target(self):
        exact = FixedRouteTracker(dense_route()).observe(transform_fixture(x=20.), 1, .05)
        self.assertEqual(exact["label"]["waypoints_world_m"][-1], [35., 0., 0.])
        for x in (20.001, 34.9, 35., 35.1):
            with self.subTest(x=x), self.assertRaisesRegex(DrivingRouteError, "insufficient-driving-label-horizon"):
                FixedRouteTracker(dense_route()).observe(transform_fixture(x=x), 1, .05)

    def test_excessive_lateral_departure_does_not_assign_another_lane(self):
        with self.assertRaisesRegex(DrivingRouteError, "outside-fixed-route"):
            FixedRouteTracker(dense_route(), maximum_lateral_error_m=.8).observe(transform_fixture(y=1.), 1, .05)

    def test_invalid_policy_and_unbounded_perturbations_are_refused(self):
        for policy in ({"max_speed_mps": 0}, {"max_speed_mps": True}, {"max_speed_mps": 100},
                       {"maximum_lateral_error_m": 0}, {"minimum_dt_seconds": -.05}):
            with self.subTest(policy=policy), self.assertRaises(DrivingRouteError):
                FixedRouteTracker(dense_route(), **policy)
        for lateral, yaw in ((.51, 0.), (0., 5.1), (True, 0.), (0., float("nan"))):
            with self.subTest(lateral=lateral, yaw=yaw), self.assertRaises(DrivingRouteError):
                perturbation_transform(transform_fixture(), lateral, yaw)

    def test_optional_minimum_time_step_guard(self):
        tracker = FixedRouteTracker(dense_route(), minimum_dt_seconds=.05)
        tracker.observe(transform_fixture(), 1, .05)
        with self.assertRaisesRegex(DrivingRouteError, "step-too-small"):
            tracker.observe(transform_fixture(x=.1), 2, .051)
        tracker.observe(transform_fixture(x=.1), 2, .1)

    def test_bad_source_hash_and_self_crossing_route_are_refused(self):
        route = dense_route()
        route["route_sha256"] = "0" * 64
        with self.assertRaisesRegex(DrivingRouteError, "route-hash-mismatch"):
            FixedRouteTracker(route)
        # Revisit the original centerline on a parallel reverse leg. It is not
        # a permitted pinned unambiguous lane even with a recomputed hash.
        crossing = route_fixture()
        for index, anchor in enumerate(crossing["centerline_anchor_sequence"]):
            if index >= 20:
                anchor["world_location"]["x"] = 38. - index
        crossing["route_sha256"] = route_sha256(crossing)
        with self.assertRaisesRegex(DrivingRouteError, "self-crossing-driving-route"):
            FixedRouteTracker(crossing)


if __name__ == "__main__":
    unittest.main()
