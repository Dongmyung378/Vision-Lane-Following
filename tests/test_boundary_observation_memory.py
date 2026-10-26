"""CPU checks for distance-bounded RGB point memory, without lane GT inputs."""
from dataclasses import replace
import inspect
import math
import unittest

from carla_tools.boundary_observation_memory import BoundaryMemoryConfig, BoundaryObservationMemory


def motion(dx=0., dy=0., yaw=0., **changes):
    c, s = math.cos(yaw), math.sin(yaw)
    result = {"usable": True, "available": True, "R": [[c, -s], [s, c]], "t": [dx, dy],
              "inliers": 40, "inlier_fraction": .8, "SE2_inlier_mean_residual_m": .005,
              "affine_scale": 1.}
    result.update(changes)
    return result


class BoundaryObservationMemoryTests(unittest.TestCase):
    def seed(self, points=None, config=None):
        memory = BoundaryObservationMemory(config)
        first = memory.update(points or [[[4.36, -2.]], [[4.36, 2.]]], None, 0.)
        self.assertEqual(first["past_points_by_side"], [[], []])
        return memory

    def test_slow_motion_preserves_observation_past_old_two_second_TTL(self):
        memory = self.seed()
        for index in range(1, 101):
            result = memory.update([[], []], motion(-.025), index*.05)
        point = result["point_records_by_side"][0][0]
        self.assertAlmostEqual(point["xy"][0], 1.86)
        self.assertAlmostEqual(point["source_age_seconds"], 5.)
        self.assertAlmostEqual(point["transported_distance_m"], 2.5)
        self.assertLess(point["uncertainty_m"], .15)
        self.assertTrue(-1 <= point["xy"][0] <= 2)

    def test_continuous_low_speed_observations_do_not_erase_transport_lineage(self):
        observations = [[[4.36+.25*i, side] for i in range(40)] for side in (-2., 2.)]
        memory = self.seed(observations)
        first_supported = None
        for index in range(1, 241):
            result = memory.update(observations, motion(-.025), index*.05)
            near = [p for p in result["past_points_by_side"][0] if -1 <= p[0] <= 2]
            supported = len(near) >= 6 and max(p[0] for p in near)-min(p[0] for p in near) >= 1.5
            if supported and first_supported is None:
                first_supported = index*.05
            if index >= 160:
                self.assertTrue(supported, (index, near))
        self.assertIsNotNone(first_supported)
        self.assertGreater(first_supported, 2.)
        self.assertLessEqual(first_supported, 8.)

    def test_bucket_keeps_oldest_lineage_and_freshest_actual_observation(self):
        memory = self.seed([[[4.36, -2.]], []])
        result = memory.update([[[4.36, -2.]], []], motion(-.025), .05)
        records = result["point_records_by_side"][0]
        self.assertEqual(len(records), 2)
        self.assertEqual({r["observed_at"] for r in records}, {0., .05})
        self.assertAlmostEqual(min(p["xy"][0] for p in records), 4.335)
        self.assertAlmostEqual(max(p["xy"][0] for p in records), 4.36)

    def test_identical_reobserved_coordinate_deduplicates_to_fresh_actual_observation(self):
        memory = self.seed([[[4.36, -2.]], []])
        result = memory.update([[[4.36, -2.]], []], motion(), .05)
        self.assertEqual(len(result["point_records_by_side"][0]), 1)
        self.assertEqual(result["point_records_by_side"][0][0]["observed_at"], .05)

    def test_moving_then_stopped_eight_seconds_holds_without_fake_variance_growth(self):
        memory = self.seed()
        for index in range(1, 101):
            result = memory.update([[], []], motion(-.025), index*.05)
        before = result["point_records_by_side"][0][0]
        for index in range(1, 161):
            result = memory.update([[], []], motion(), 5.+index*.05)
        after = result["point_records_by_side"][0][0]
        self.assertEqual(after["xy"], before["xy"])
        self.assertEqual(after["uncertainty_m"], before["uncertainty_m"])
        self.assertEqual(after["transported_distance_m"], before["transported_distance_m"])
        self.assertAlmostEqual(after["stationary_hold_seconds"], 8.)
        self.assertGreater(after["source_age_seconds"], 10.)

    def test_unobserved_point_expires_after_ten_second_stationary_cap(self):
        memory = self.seed([[[1., -2.]], []])
        for index in range(1, 201):
            result = memory.update([[], []], motion(), index*.05)
        self.assertEqual(len(result["points_by_side"][0]), 1)
        result = memory.update([[], []], motion(), 10.05)
        self.assertEqual(result["points_by_side"], [[], []])
        self.assertEqual(result["stats"]["expired_by_reason"]["stationary_hold"], 1)

    def test_stationary_cold_start_does_not_invent_near_support(self):
        memory = self.seed()
        for index in range(1, 81):
            result = memory.update([[[4.36, -2.]], [[4.36, 2.]]], motion(), index*.05)
        near = [p for side in result["past_points_by_side"] for p in side if -1 <= p[0] <= 2]
        self.assertEqual(near, [])
        self.assertTrue(all(p[0] == 4.36 for side in result["points_by_side"] for p in side))

    def test_transform_is_old_static_point_to_current_not_ego_motion(self):
        memory = self.seed([[[5., -2.]], [[5., 2.]]])
        result = memory.update([[], []], motion(-.2, .1, .1), .05)
        expected = [5*math.cos(.1)+2*math.sin(.1)-.2,
                    5*math.sin(.1)-2*math.cos(.1)+.1]
        for observed, target in zip(result["past_points_by_side"][0][0], expected):
            self.assertAlmostEqual(observed, target)

    def test_failed_motion_clears_history_and_only_seeds_current_points(self):
        for invalid in (None, {"usable": False}, motion(available=False)):
            with self.subTest(motion=invalid):
                memory = self.seed()
                result = memory.update([[[8., -2.]], []], invalid, .05)
                self.assertEqual(result["past_points_by_side"], [[], []])
                self.assertEqual(result["points_by_side"], [[[8., -2.]], []])
                self.assertFalse(result["stats"]["motion_usable"])

    def test_ambiguous_RGB_quality_cannot_keep_old_points(self):
        for changes in ({"inliers": 7}, {"inlier_fraction": .3},
                        {"SE2_inlier_mean_residual_m": .031}, {"affine_scale": 1.2},
                        {"inlier_fraction": float("nan")}):
            with self.subTest(changes=changes):
                result = self.seed().update([[], []], motion(**changes), .05)
                self.assertEqual(result["points_by_side"], [[], []])
                self.assertEqual(result["stats"]["reset_reason"], "relative-RGB-motion-quality-unqualified")
        missing = motion()
        del missing["inlier_fraction"]
        result = self.seed().update([[], []], missing, .05)
        self.assertEqual(result["points_by_side"], [[], []])

    def test_reflection_nonfinite_and_jump_reset_instead_of_transport(self):
        for invalid in (motion(R=[[1., 0.], [0., -1.]]), motion(t=[float("nan"), 0.]),
                        motion(1.01), motion(yaw=.151), motion(R=[[1., 0.]])):
            with self.subTest(motion=invalid):
                result = self.seed().update([[], []], invalid, .05)
                self.assertEqual(result["points_by_side"], [[], []])
                self.assertFalse(result["stats"]["motion_usable"])

    def test_uncertainty_and_distance_expire_independently(self):
        config = replace(BoundaryMemoryConfig(), max_uncertainty_m=.021)
        result = self.seed(config=config).update([[], []], motion(-.1), .05)
        self.assertEqual(result["stats"]["expired_by_reason"]["uncertainty"], 2)
        config = replace(BoundaryMemoryConfig(), max_transport_distance_m=.05)
        result = self.seed(config=config).update([[], []], motion(-.1), .05)
        self.assertEqual(result["stats"]["expired_by_reason"]["distance"], 2)

    def test_spatial_buckets_and_point_budget_favor_retained_near_points(self):
        config = replace(BoundaryMemoryConfig(), max_points_per_side=4)
        memory = self.seed([[[.5, -2.], [.51, -2.]], [[.5, 2.]]], config)
        self.assertEqual(memory.snapshot()["points_by_side"], [[[.5, -2.]], [[.5, 2.]]])
        far = [[4.+index*.11, -2.] for index in range(100)]
        result = memory.update([far, []], motion(), .05)
        self.assertEqual(len(result["points_by_side"][0]), 4)
        self.assertIn([.5, -2.], result["points_by_side"][0])
        self.assertEqual(result["points_by_side"][1], [[.5, 2.]])
        result = memory.update([far, []], motion(), .1)
        self.assertEqual(len(result["points_by_side"][0]), 4)

    def test_current_observations_do_not_appear_as_transported_past_support(self):
        memory = self.seed()
        result = memory.update([[[1., -2.]], []], motion(), .05)
        self.assertNotIn([1., -2.], result["past_points_by_side"][0])
        self.assertIn([1., -2.], result["points_by_side"][0])
        self.assertTrue(all(r["observed_at"] < .05 for side in result["past_point_records_by_side"] for r in side))

    def test_timestamp_gap_and_reverse_reset_history(self):
        for timestamp in (-.1, 1.):
            result = self.seed().update([[], []], motion(), timestamp)
            self.assertEqual(result["points_by_side"], [[], []])
            self.assertEqual(result["stats"]["reset_reason"], "timestamp-discontinuity")

    def test_snapshot_is_pure_and_reset_has_no_route_pose_or_speed_arguments(self):
        memory = self.seed()
        before = memory.snapshot()
        changed = memory.snapshot()
        changed["point_records_by_side"][0][0]["xy"][0] = 100.
        changed["stats"]["expired_by_reason"]["distance"] = 100
        self.assertEqual(memory.snapshot(), before)
        self.assertEqual(list(inspect.signature(memory.update).parameters),
                         ["observations_by_side", "motion", "timestamp"])
        with self.assertRaises(TypeError):
            memory.update([[], []], motion(), .05, speed_mps=0.)
        memory.reset()
        self.assertEqual(memory.snapshot()["points_by_side"], [[], []])
        self.assertIsNone(memory.snapshot()["timestamp"])

    def test_nonfinite_observation_clears_state_before_raising(self):
        memory = self.seed()
        with self.assertRaises(ValueError):
            memory.update([[[float("inf"), -2.]], []], motion(), .05)
        self.assertEqual(memory.snapshot()["points_by_side"], [[], []])


if __name__ == "__main__":
    unittest.main()
