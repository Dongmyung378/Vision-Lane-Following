import copy
import unittest

from carla_tools.driving_quality import summarize_driving_quality


def episode(identity, *, reason="route_completed", invasions=0, collisions=0, margin=.1):
    return {"episode": identity, "variant": "hybrid_geometry", "case": "left_curve", "gate": {"status": "PASS", "metrics": {
        "driving_success": reason == "route_completed", "termination_reason": reason,
        "collision_count": collisions, "lane_invasion_count": invasions, "minimum_body_lane_margin_m": margin}}}


class DrivingQualityTests(unittest.TestCase):
    def test_terminal_lane_crossing_stays_completed_but_is_not_clean(self):
        records = [episode("clean"), episode("crossed-before-goal", invasions=1, margin=-.6),
                   episode("body-over-boundary", margin=-.001), episode("collision-before-goal", collisions=1),
                   episode("failed", reason="lane_departure", invasions=1, margin=-.1)]
        before = copy.deepcopy(records)
        group = next(g for g in summarize_driving_quality(records)["groups"] if g["case"] == "all")
        self.assertEqual(group["route_completed"], 4)
        self.assertEqual(group["clean_completed"], 1)
        self.assertEqual(group["completed_with_lane_or_collision_issue"], 3)
        self.assertEqual(group["clean_completion_rate"], .2)
        self.assertEqual(group["lane_invasion_episodes"], 2)
        self.assertEqual(records, before)

    def test_nonnegative_boundary_is_inclusive_and_missing_proof_is_rejected(self):
        self.assertEqual(summarize_driving_quality([episode("edge", margin=0.)])["groups"][0]["clean_completed"], 1)
        unverified = episode("no-proof")
        unverified["gate"]["status"] = "FAIL"
        with self.assertRaisesRegex(ValueError, "unverified"):
            summarize_driving_quality([unverified])
        for margin in (None, float("nan")):
            with self.assertRaisesRegex(ValueError, "margin"):
                summarize_driving_quality([episode("invalid", margin=margin)])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_driving_quality([episode("same"), episode("same")])


if __name__ == "__main__":
    unittest.main()
