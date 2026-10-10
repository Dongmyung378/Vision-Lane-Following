"""Summarize recorded route completion and additional lane-retention diagnostics."""
from __future__ import annotations

import math


def summarize_driving_quality(episodes):
    """Preserve the frozen termination metric; report a stricter diagnostic too.

    Clean completion requires a completed route, zero collision/lane-invasion
    events, and a nonnegative minimum vehicle-body lane margin during driving.
    It is an additional diagnostic, not a retrospective protocol alteration.
    """
    groups = {}
    seen = set()
    for episode in episodes:
        identity = episode["episode"]
        if identity in seen:
            raise ValueError("driving-quality-duplicate-episode")
        seen.add(identity)
        gate = episode["gate"]
        if gate.get("status") != "PASS":
            raise ValueError("driving-quality-unverified-episode")
        metrics = gate["metrics"]
        completed = metrics["driving_success"]
        if type(completed) is not bool or completed != (metrics["termination_reason"] == "route_completed"):
            raise ValueError("driving-quality-completion-reason-mismatch")
        collisions, invasions = metrics["collision_count"], metrics["lane_invasion_count"]
        margin = metrics["minimum_body_lane_margin_m"]
        if (type(collisions) is not int or type(invasions) is not int or min(collisions, invasions) < 0
                or type(margin) not in (int, float) or not math.isfinite(margin)):
            raise ValueError("driving-quality-invalid-events-or-margin")
        clean = completed and collisions == 0 and invasions == 0 and margin >= 0
        for case in ("all", episode["case"]):
            key = (episode["variant"], case)
            group = groups.setdefault(key, {"method": key[0], "case": key[1], "episodes": 0,
                "route_completed": 0, "clean_completed": 0, "completed_with_lane_or_collision_issue": 0,
                "collision_episodes": 0, "lane_invasion_episodes": 0, "body_lane_margin_negative_episodes": 0})
            group["episodes"] += 1
            group["route_completed"] += int(completed)
            group["clean_completed"] += int(clean)
            group["completed_with_lane_or_collision_issue"] += int(completed and not clean)
            group["collision_episodes"] += int(collisions > 0)
            group["lane_invasion_episodes"] += int(invasions > 0)
            group["body_lane_margin_negative_episodes"] += int(margin < 0)
    for group in groups.values():
        group["route_completion_rate"] = group["route_completed"] / group["episodes"]
        group["clean_completion_rate"] = group["clean_completed"] / group["episodes"]
    return {"scope": "additional-diagnostic-preserving-frozen-native-metrics",
        "clean_completion_rule": "route_completed AND collision_count=0 AND lane_invasion_count=0 AND minimum_body_lane_margin_m>=0",
        "groups": sorted(groups.values(), key=lambda g: (g["method"], g["case"]))}
