"""Track a pinned lane centerline during privileged D9/D10 driving.

No CARLA import or nearest-lane lookup occurs here. The original route and its
start-lane identity survive lateral/yaw perturbations. Progress is measured in
the surveyed physical centerline arc; labels use the same frame's actor pose.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from .contracts import (HORIZON_ARC_M, ContractError, route_sha256,
                        validate_route_contract, validate_waypoint_label)


class DrivingRouteError(ValueError):
    """Stable failure codes without raw native API or filesystem messages."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _finite(value: Any) -> float:
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise DrivingRouteError("invalid-driving-number")
        return float(value)
    except OverflowError:
        raise DrivingRouteError("invalid-driving-number") from None


def _wrap(angle: float) -> float:
    return (angle + 180.) % 360. - 180.


def _pose_matrix(location: dict, rotation: dict) -> list[list[float]]:
    pitch, yaw, roll = (math.radians(rotation[axis]) for axis in ("pitch", "yaw", "roll"))
    cp, sp, cy, sy, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw), math.cos(roll), math.sin(roll)
    return [[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, location["x"]],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, location["y"]],
            [sp, -cp * sr, cp * cr, location["z"]], [0., 0., 0., 1.]]


def _transform(value: Any) -> dict:
    if not isinstance(value, dict) or set(value) != {"location", "rotation", "matrix"}:
        raise DrivingRouteError("invalid-driving-transform")
    for key, axes in (("location", {"x", "y", "z"}), ("rotation", {"pitch", "yaw", "roll"})):
        if not isinstance(value[key], dict) or set(value[key]) != axes:
            raise DrivingRouteError("invalid-driving-transform")
        for number in value[key].values():
            _finite(number)
    matrix = value["matrix"]
    if not isinstance(matrix, list) or len(matrix) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in matrix):
        raise DrivingRouteError("invalid-driving-matrix")
    expected = _pose_matrix(value["location"], value["rotation"])
    if any(abs(_finite(matrix[row][column]) - expected[row][column]) > 1e-4 for row in range(4) for column in range(4)):
        raise DrivingRouteError("driving-transform-matrix-mismatch")
    return copy.deepcopy(value)


def perturbation_transform(transform_record: dict, lateral_m: float, yaw_deg: float) -> dict:
    """Apply a horizontal actor-right offset and positive-right yaw in CARLA.

    z, pitch and roll are preserved. The returned matrix is rebuilt from the
    returned pose; the caller's record is never changed. This is a requested
    initial pose, not proof that the simulator applied it.
    """
    record = _transform(transform_record)
    lateral, yaw = _finite(lateral_m), _finite(yaw_deg)
    if abs(lateral) > .5 or abs(yaw) > 5.:
        raise DrivingRouteError("unsupported-recovery-perturbation")
    initial_yaw = math.radians(record["rotation"]["yaw"])
    record["location"]["x"] -= lateral * math.sin(initial_yaw)
    record["location"]["y"] += lateral * math.cos(initial_yaw)
    record["rotation"]["yaw"] = _wrap(record["rotation"]["yaw"] + yaw)
    record["matrix"] = _pose_matrix(record["location"], record["rotation"])
    return record


def _point(anchor: dict) -> list[float]:
    return [anchor["world_location"][axis] for axis in ("x", "y", "z")]


def _segment_projection(point: list[float], a: list[float], b: list[float]) -> tuple[float, float]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-16:
        raise DrivingRouteError("degenerate-driving-route-projection")
    fraction = min(1., max(0., ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / denominator))
    return fraction, math.hypot(point[0] - a[0] - fraction * dx, point[1] - a[1] - fraction * dy)


def _orientation(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_touch(a, b, c, d):
    if _orientation(a, b, c) * _orientation(a, b, d) < 0 and _orientation(c, d, a) * _orientation(c, d, b) < 0:
        return True
    return min(_segment_projection(a, c, d)[1], _segment_projection(b, c, d)[1],
               _segment_projection(c, a, b)[1], _segment_projection(d, a, b)[1]) <= 1e-8


class FixedRouteTracker:
    """Fail closed on stale frames, jumps, ambiguous references and short horizons.

    The first reference is found on the entire already pinned route. Later
    references must remain in the previous arc's [-1, +3] m window and obey the
    configured speed bound. The global nearest reference is also checked to
    prevent projecting a teleported pose onto the edge of that window.
    """

    def __init__(self, route_contract: dict, *, max_speed_mps: float = 20.,
                 maximum_lateral_error_m: float = 3., minimum_dt_seconds: float | None = None):
        try:
            validate_route_contract(route_contract)
        except ContractError as error:
            raise DrivingRouteError(error.code) from error
        self._route = copy.deepcopy(route_contract)
        self._route_sha256 = route_sha256(self._route)
        self.maximum_reference_arc_m = self._route["route_length_m"] - max(HORIZON_ARC_M)
        self.max_speed_mps = _finite(max_speed_mps)
        self.maximum_lateral_error_m = _finite(maximum_lateral_error_m)
        self.minimum_dt_seconds = None if minimum_dt_seconds is None else _finite(minimum_dt_seconds)
        if (not 0 < self.max_speed_mps <= 50 or not 0 < self.maximum_lateral_error_m <= 10
                or self.minimum_dt_seconds is not None and not 0 < self.minimum_dt_seconds <= 1.):
            raise DrivingRouteError("invalid-driving-tracker-policy")
        self._anchors = self._route["centerline_anchor_sequence"]
        points = [_point(anchor) for anchor in self._anchors]
        for index, (a, b) in enumerate(zip(points, points[1:])):
            _segment_projection(a, a, b)
            for other in range(index + 2, len(points) - 1):
                if _segments_touch(a, b, points[other], points[other + 1]):
                    raise DrivingRouteError("self-crossing-driving-route")
        self._last_frame = self._last_timestamp = self._last_reference = self._initial_reference = None
        self._maximum_progress = 0.

    @property
    def route_sha256(self) -> str:
        return self._route_sha256

    @property
    def route(self) -> dict:
        """Return a copy so callers cannot change the pinned target lane."""
        return copy.deepcopy(self._route)

    def _reference(self, location: dict) -> tuple[float, float, int, float]:
        point = [location[axis] for axis in ("x", "y", "z")]
        candidates = []
        for index, (first, second) in enumerate(zip(self._anchors, self._anchors[1:])):
            fraction, distance = _segment_projection(point, _point(first), _point(second))
            arc = first["arc_distance_m"] + fraction * (second["arc_distance_m"] - first["arc_distance_m"])
            candidates.append((distance, arc, index, fraction))
        distance, arc, index, fraction = min(candidates, key=lambda candidate: candidate[0])
        alternatives = [candidate[1] for candidate in candidates if abs(candidate[0] - distance) <= 1e-8]
        if max(alternatives) - min(alternatives) > .001:
            raise DrivingRouteError("ambiguous-driving-route-reference")
        if self._last_reference is not None and not self._last_reference - 1. <= arc <= self._last_reference + 3.:
            raise DrivingRouteError("driving-reference-outside-progress-window")
        return arc, distance, index, fraction

    def _interpolate(self, arc: float) -> list[float]:
        for first, second in zip(self._anchors, self._anchors[1:]):
            if first["arc_distance_m"] <= arc <= second["arc_distance_m"]:
                fraction = (arc - first["arc_distance_m"]) / (second["arc_distance_m"] - first["arc_distance_m"])
                return [a + fraction * (b - a) for a, b in zip(_point(first), _point(second))]
        raise DrivingRouteError("insufficient-driving-label-horizon")

    def observe(self, vehicle_transform_record: dict, frame: int, timestamp: float) -> dict:
        pose = _transform(vehicle_transform_record)
        if type(frame) is not int or frame < 0:
            raise DrivingRouteError("invalid-driving-frame")
        now = _finite(timestamp)
        if now < 0:
            raise DrivingRouteError("invalid-driving-timestamp")
        if self._last_frame is not None and frame <= self._last_frame:
            raise DrivingRouteError("stale-driving-frame")
        if self._last_timestamp is not None and now <= self._last_timestamp:
            raise DrivingRouteError("stale-driving-timestamp")
        dt = None if self._last_timestamp is None else now - self._last_timestamp
        if dt is not None and self.minimum_dt_seconds is not None and dt < self.minimum_dt_seconds - 1e-9:
            raise DrivingRouteError("driving-timestamp-step-too-small")
        reference, centerline_distance, segment, fraction = self._reference(pose["location"])
        if centerline_distance > self.maximum_lateral_error_m:
            raise DrivingRouteError("driving-pose-outside-fixed-route")
        if dt is not None and abs(reference - self._last_reference) > self.max_speed_mps * dt + .05:
            raise DrivingRouteError("driving-progress-speed-bound")
        if reference > self.maximum_reference_arc_m:
            raise DrivingRouteError("insufficient-driving-label-horizon")
        first, second = self._anchors[segment:segment + 2]
        a, b = _point(first), _point(second)
        dx, dy = b[0] - a[0], b[1] - a[1]
        horizontal_length = math.hypot(dx, dy)
        longitudinal = ((pose["location"]["x"] - a[0]) * dx + (pose["location"]["y"] - a[1]) * dy) / horizontal_length
        if segment == 0 and fraction == 0. and longitudinal < -.5:
            raise DrivingRouteError("driving-pose-before-route-start")
        center = [x + fraction * (y - x) for x, y in zip(a, b)]
        lateral = ((pose["location"]["y"] - center[1]) * dx - (pose["location"]["x"] - center[0]) * dy) / horizontal_length
        heading = _wrap(pose["rotation"]["yaw"] - math.degrees(math.atan2(dy, dx)))
        world = [self._interpolate(reference + horizon) for horizon in HORIZON_ARC_M]
        matrix = pose["matrix"]
        ego = [[sum(matrix[k][axis] * (point[k] - matrix[k][3]) for k in range(3)) for axis in range(2)] for point in world]
        label = {"schema_version": 1, "ego_frame": frame, "route_id": self._route["route_id"],
                 "target_lane": {key: self._route["start"][key] for key in ("road_id", "section_id", "lane_id")},
                 "coordinate_origin": "vehicle_actor_transform", "axes": "x-forward_y-right", "units": "m",
                 "distance_reference": "target-lane-centerline-arc-length", "horizon_arc_m": list(HORIZON_ARC_M),
                 "reference_arc_m": reference, "waypoints_world_m": world, "waypoints_ego_m": ego}
        try:
            validate_waypoint_label(label, {"frame": frame, "vehicle": {"transform": pose}},
                                    {"route_id": self._route["route_id"], "route_contract": self._route})
        except ContractError as error:
            raise DrivingRouteError(error.code) from error
        initial = reference if self._initial_reference is None else self._initial_reference
        progress = reference - initial
        maximum_progress = max(self._maximum_progress, progress)
        lane_width = None
        if "lane_width_m" in first and "lane_width_m" in second:
            lane_width = first["lane_width_m"] + fraction * (second["lane_width_m"] - first["lane_width_m"])
        result = {"frame": frame, "timestamp": now, "reference_arc_m": reference, "progress_m": progress,
                  "maximum_progress_m": maximum_progress, "lateral_error_m": lateral,
                  "centerline_distance_xy_m": centerline_distance, "heading_error_deg": heading,
                  "route_tangent_yaw_deg": math.degrees(math.atan2(dy, dx)), "reference_world_m": center,
                  "lane_width_m": lane_width, "route_sha256": self.route_sha256, "label": label}
        # Invalid observations leave the state unchanged, allowing a caller to
        # report the failure without advancing progress or hiding stale data.
        self._initial_reference = initial
        self._last_frame, self._last_timestamp, self._last_reference = frame, now, reference
        self._maximum_progress = maximum_progress
        return result
