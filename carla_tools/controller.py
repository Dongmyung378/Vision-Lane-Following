"""D8 Pure Pursuit at the measured rear axle, plus bounded speed PI control.

This module has no CARLA import and no simulator side effects. Waypoints use the
D5 actor-origin contract: five future points in metres, x forward and y right.
Equivalent bicycle steering is converted to CARLA's inner front-wheel command
using measured front track width, the native front-wheel steering gain, and the
applied speed-dependent steering curve.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .contracts import ContractError, read_strict_json


DEFAULT_CONTROLLER_PATH = Path(__file__).resolve().parents[1] / "configs" / "controller_d8.json"
CONTROLLER_VERSION = "vlf-pure-pursuit-d8-v4"


class ControllerError(ValueError):
    """An invalid controller input, geometry measurement, or configuration."""


def _number(value: Any, code: str) -> float:
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except (OverflowError, TypeError):
        valid = False
    if not valid:
        raise ControllerError(code)
    return float(value)


def _bounded(value: Any, minimum: float, maximum: float, code: str) -> float:
    number = _number(value, code)
    if not minimum <= number <= maximum:
        raise ControllerError(code)
    return number


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class ControllerConfig:
    schema_version: int = 4
    controller_version: str = CONTROLLER_VERSION
    target_speed_kmh: float = 25.0
    max_lateral_acceleration_mps2: float = 2.0
    fixed_delta_seconds: float = 0.05
    lookahead_base_m: float = 3.5
    lookahead_gain_seconds: float = 0.35
    lookahead_min_m: float = 4.0
    lookahead_max_m: float = 8.0
    speed_kp: float = 0.45
    speed_ki: float = 0.15
    integral_limit_m: float = 10.0
    throttle_max: float = 0.65
    brake_max: float = 0.5
    speed_deadband_mps: float = 0.05
    steer_rate_per_second: float = 2.0
    max_normalized_steer: float = 1.0

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 4:
            raise ControllerError("invalid-controller-schema-version")
        if self.controller_version != CONTROLLER_VERSION:
            raise ControllerError("invalid-controller-version")
        target_speed = _number(self.target_speed_kmh, "invalid-target-speed")
        if target_speed != 25.0:
            raise ControllerError("d8-target-speed-must-be-25-kmh")
        timestep = _number(self.fixed_delta_seconds, "invalid-controller-timestep")
        if timestep != 0.05:
            raise ControllerError("d8-controller-timestep-must-be-0.05-seconds")
        object.__setattr__(self, "target_speed_kmh", target_speed)
        object.__setattr__(self, "fixed_delta_seconds", timestep)
        bounds = {
            "max_lateral_acceleration_mps2": (0.1, 10.0),
            "lookahead_base_m": (0.5, 15.0),
            "lookahead_gain_seconds": (0.0, 2.0),
            "lookahead_min_m": (0.5, 15.0),
            "lookahead_max_m": (0.5, 15.0),
            "speed_kp": (0.001, 5.0),
            "speed_ki": (0.0, 5.0),
            "integral_limit_m": (0.0, 100.0),
            "throttle_max": (0.01, 1.0),
            "brake_max": (0.01, 1.0),
            "speed_deadband_mps": (0.0, 0.5),
            "steer_rate_per_second": (0.01, 20.0),
            "max_normalized_steer": (0.01, 1.0),
        }
        for name, limits in bounds.items():
            object.__setattr__(self, name, _bounded(getattr(self, name), *limits,
                                                  f"invalid-controller-{name}"))
        if self.lookahead_min_m > self.lookahead_max_m:
            raise ControllerError("invalid-controller-lookahead-order")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def controller_config_from_dict(value: Any) -> ControllerConfig:
    expected = {item.name for item in fields(ControllerConfig)}
    if type(value) is not dict or set(value) != expected:
        raise ControllerError("invalid-controller-config-fields")
    return ControllerConfig(**value)


def load_controller_config(path: str | Path = DEFAULT_CONTROLLER_PATH) -> ControllerConfig:
    try:
        return controller_config_from_dict(read_strict_json(path))
    except ContractError as error:
        raise ControllerError(f"controller-config-{error.code}") from error


def _curve(value: Any) -> tuple[tuple[float, float], ...]:
    if type(value) not in (list, tuple) or len(value) < 2:
        raise ControllerError("invalid-native-steering-curve")
    result = []
    for pair in value:
        if type(pair) not in (list, tuple) or len(pair) != 2:
            raise ControllerError("invalid-native-steering-curve")
        speed = _bounded(pair[0], 0.0, 300.0, "invalid-steering-curve-speed-kmh")
        factor = _bounded(pair[1], 0.001, 1.0, "invalid-steering-curve-factor")
        if result and speed <= result[-1][0]:
            raise ControllerError("non-increasing-steering-curve-speed-kmh")
        result.append((speed, factor))
    if result[0][0] != 0.0:
        raise ControllerError("steering-curve-must-start-at-zero-kmh")
    return tuple(result)


@dataclass(frozen=True)
class VehicleGeometry:
    """Actor-local measured axle coordinates in metres, with native steering.

    Explicit construction supports measurement supplied by a simulator harness;
    it never inserts a guessed wheelbase or steering limit. The front and rear
    axle are the average x coordinates of each measured left/right wheel pair.
    """

    wheelbase_m: float
    rear_axle_x_m: float
    front_axle_x_m: float
    max_steer_angle_deg: float
    steering_curve: tuple[tuple[float, float], ...]
    front_track_m: float
    rear_axle_y_m: float = 0.0
    front_axle_y_m: float = 0.0

    def __post_init__(self) -> None:
        wheelbase = _bounded(self.wheelbase_m, 1.0, 6.0, "invalid-measured-wheelbase")
        rear = _bounded(self.rear_axle_x_m, -5.0, -0.01, "invalid-measured-rear-axle")
        front = _bounded(self.front_axle_x_m, 0.01, 5.0, "invalid-measured-front-axle")
        steer_limit = _bounded(self.max_steer_angle_deg, 1.0, 85.0, "invalid-measured-front-steer-angle")
        front_track = _bounded(self.front_track_m, 0.5, 3.0, "invalid-measured-front-track")
        rear_y = _bounded(self.rear_axle_y_m, -0.2, 0.2, "invalid-measured-rear-axle-y")
        front_y = _bounded(self.front_axle_y_m, -0.2, 0.2, "invalid-measured-front-axle-y")
        if abs(front - rear - wheelbase) > 1e-6:
            raise ControllerError("measured-wheelbase-does-not-match-axles")
        if abs(front_y - rear_y) > 0.02:
            raise ControllerError("measured-axles-not-aligned-with-actor")
        for name, value in (("wheelbase_m", wheelbase), ("rear_axle_x_m", rear),
                            ("front_axle_x_m", front), ("max_steer_angle_deg", steer_limit),
                            ("front_track_m", front_track),
                            ("rear_axle_y_m", rear_y), ("front_axle_y_m", front_y)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "steering_curve", _curve(self.steering_curve))

    def to_dict(self) -> dict:
        result = asdict(self)
        result["steering_curve"] = [list(pair) for pair in self.steering_curve]
        return result

    def steering_factor(self, speed_mps: float) -> float:
        """Piecewise-linear configured curve in kilometres/hour.

        Original Unreal interpolation tangents are absent from the Python API.
        The native D9 harness applies/readbacks a flat curve to make this factor
        exact; arbitrary native curves require a matching applied curve policy.
        """
        speed_kmh = _bounded(speed_mps, 0.0, 100.0, "invalid-forward-speed") * 3.6
        for (x0, y0), (x1, y1) in zip(self.steering_curve, self.steering_curve[1:]):
            if speed_kmh <= x1:
                fraction = (speed_kmh - x0) / (x1 - x0)
                return y0 + fraction * (y1 - y0)
        return self.steering_curve[-1][1]

    @classmethod
    def from_physics_control(cls, physics: Any, vehicle_transform: Any = None) -> "VehicleGeometry":
        """Measure CARLA 0.9.16 world-centimetre wheel positions at a pose.

        The vehicle transform must be read for the same stationary measurement.
        Its inverse transforms world metres to actor metres; assuming the wheel
        positions are actor-local produces incorrect geometry after translation
        or yaw. CARLA's four wheel order is front-left, front-right, rear-left,
        rear-right. PhysX uses the larger positive front wheel steering angle as
        its normalized inner-wheel steering gain, then adjusts the outer wheel.
        """
        if vehicle_transform is None:
            raise ControllerError("wheel-measurement-requires-vehicle-transform")
        try:
            wheels = list(physics.wheels)
            inverse = vehicle_transform.get_inverse_matrix()
            native_curve = [(point.x, point.y) for point in physics.steering_curve]
        except (AttributeError, TypeError) as error:
            raise ControllerError("invalid-native-vehicle-physics") from error
        if (len(wheels) != 4 or type(inverse) not in (list, tuple) or len(inverse) != 4
                or any(type(row) not in (list, tuple) or len(row) != 4 for row in inverse)):
            raise ControllerError("unsupported-native-wheel-layout")
        inverse = [[_number(component, "invalid-vehicle-inverse-transform") for component in row]
                   for row in inverse]
        if any(abs(inverse[3][i] - (1.0 if i == 3 else 0.0)) > 1e-6 for i in range(4)):
            raise ControllerError("invalid-vehicle-inverse-transform")
        for i in range(3):
            for j in range(3):
                dot = sum(inverse[i][k] * inverse[j][k] for k in range(3))
                if abs(dot - (1.0 if i == j else 0.0)) > 1e-5:
                    raise ControllerError("non-rigid-vehicle-inverse-transform")
        determinant = (inverse[0][0] * (inverse[1][1] * inverse[2][2] - inverse[1][2] * inverse[2][1])
                       - inverse[0][1] * (inverse[1][0] * inverse[2][2] - inverse[1][2] * inverse[2][0])
                       + inverse[0][2] * (inverse[1][0] * inverse[2][1] - inverse[1][1] * inverse[2][0]))
        if abs(determinant - 1.0) > 1e-5:
            raise ControllerError("non-rigid-vehicle-inverse-transform")
        local = []
        try:
            for wheel in wheels:
                position = [_number(getattr(wheel.position, axis), "invalid-world-wheel-position") / 100.0
                            for axis in ("x", "y", "z")] + [1.0]
                local.append([sum(inverse[i][j] * position[j] for j in range(4)) for i in range(3)])
            limits = [_bounded(wheel.max_steer_angle, 1.0, 85.0,
                               "invalid-measured-front-steer-angle") for wheel in wheels[:2]]
        except AttributeError as error:
            raise ControllerError("invalid-native-wheel-measurement") from error
        for pair in (local[:2], local[2:]):
            if abs(pair[0][0] - pair[1][0]) > 0.15 or not pair[0][1] < 0.0 < pair[1][1]:
                raise ControllerError("inconsistent-native-wheel-axle")
            if not 0.5 <= pair[1][1] - pair[0][1] <= 3.0:
                raise ControllerError("invalid-native-wheel-track")
        front = sum(point[0] for point in local[:2]) / 2.0
        rear = sum(point[0] for point in local[2:]) / 2.0
        front_y = sum(point[1] for point in local[:2]) / 2.0
        rear_y = sum(point[1] for point in local[2:]) / 2.0
        front_track = local[1][1] - local[0][1]
        return cls(front - rear, rear, front, max(limits), _curve(native_curve),
                   front_track, rear_y, front_y)


def _rear_points(waypoints: Any, rear_axle_x_m: float, rear_axle_y_m: float) -> list[list[float]]:
    if type(waypoints) not in (list, tuple) or len(waypoints) != 5:
        raise ControllerError("controller-requires-five-ego-xy-waypoints")
    points = []
    previous_radius = 0.0
    for pair in waypoints:
        if type(pair) not in (list, tuple) or len(pair) != 2:
            raise ControllerError("invalid-controller-waypoint-shape")
        x = _bounded(pair[0], -100.0, 100.0, "invalid-controller-waypoint-number") - rear_axle_x_m
        y = _bounded(pair[1], -100.0, 100.0, "invalid-controller-waypoint-number") - rear_axle_y_m
        radius = math.hypot(x, y)
        if x <= 0.01:
            raise ControllerError("controller-waypoint-behind-rear-axle")
        if radius <= previous_radius + 1e-6:
            raise ControllerError("controller-waypoints-not-ordered-forward")
        points.append([x, y])
        previous_radius = radius
    return points


def _lookahead_target(points: list[list[float]], radius: float) -> tuple[list[float], str, int]:
    if math.hypot(*points[0]) >= radius:
        return list(points[0]), "first-available-target", 0
    for index, (start, end) in enumerate(zip(points, points[1:])):
        if math.hypot(*end) >= radius:
            dx, dy = end[0] - start[0], end[1] - start[1]
            a = dx * dx + dy * dy
            b = 2.0 * (start[0] * dx + start[1] * dy)
            c = start[0] * start[0] + start[1] * start[1] - radius * radius
            discriminant = max(0.0, b * b - 4.0 * a * c)
            fraction = _clip((-b + math.sqrt(discriminant)) / (2.0 * a), 0.0, 1.0)
            return [start[0] + fraction * dx, start[1] + fraction * dy], "circle-segment-interpolation", index
    return list(points[-1]), "last-available-target", 4


def _preview_curvatures(points: list[list[float]]) -> list[float]:
    """Absolute Menger curvature of the three future waypoint triples.

    This geometry is unchanged by rigid translation, yaw, or mirroring. It
    previews the bend before the steering target enters it; instantaneous
    commanded curvature alone would often brake only after turn entry.
    """
    curvatures = []
    for first, second, third in zip(points, points[1:], points[2:]):
        ab = [second[axis] - first[axis] for axis in range(2)]
        bc = [third[axis] - second[axis] for axis in range(2)]
        ac = [third[axis] - first[axis] for axis in range(2)]
        denominator = math.hypot(*ab) * math.hypot(*bc) * math.hypot(*ac)
        if denominator <= 1e-12:
            raise ControllerError("controller-preview-points-too-close")
        cross = ab[0] * ac[1] - ab[1] * ac[0]
        curvatures.append(2.0 * abs(cross) / denominator)
    return curvatures


def _actor_circle_curvature(target_actor_xy: list[float], rear_offset_xy: list[float]) -> tuple[float, float]:
    """Rear bicycle curvature whose ACTOR trajectory passes through the target.

    In rear coordinates the actor currently sits at -rear_offset. A rear ICR
    (0,R) gives an actor circle radius |(-rear_offset)-(0,R)|. Requiring the
    future actor target to lie on that same circle yields
    kappa = 2*target_actor_y / (|target_actor-rear_offset|**2-|rear_offset|**2).
    This tracks the original actor-origin path without shifting stored targets.
    """
    x, y = (_number(value, "invalid-actor-circle-target") for value in target_actor_xy)
    rx, ry = (_number(value, "invalid-actor-circle-rear-offset") for value in rear_offset_xy)
    denominator = (x-rx)**2 + (y-ry)**2 - rx**2 - ry**2
    if not math.isfinite(denominator) or denominator <= 1e-8:
        raise ControllerError("actor-circle-target-not-forward-of-control-origin")
    return 2.0*y/denominator, denominator


class PurePursuitController:
    """Stateful single-episode controller; call reset before every episode.

    Invalid inputs are rejected before any state changes. The speed PI uses
    conditional integration to avoid adding error that pushes an already
    saturated command further into saturation. Steering has an explicit slew
    bound in normalized CARLA command units per simulated second.
    """

    def __init__(self, config: ControllerConfig, geometry: VehicleGeometry):
        if not isinstance(config, ControllerConfig) or not isinstance(geometry, VehicleGeometry):
            raise ControllerError("controller-requires-validated-config-and-geometry")
        self.config = config
        self.geometry = geometry
        self._config_sha256 = config.sha256
        self.reset()

    def reset(self) -> None:
        self._speed_integral_m = 0.0
        self._previous_steer = 0.0
        self._step_index = 0

    def step(self, waypoints: Any, speed_mps: float, dt_seconds: float = 0.05) -> dict:
        speed = _bounded(speed_mps, 0.0, 100.0, "invalid-forward-speed")
        timestep = _number(dt_seconds, "invalid-controller-timestep")
        if abs(timestep - self.config.fixed_delta_seconds) > 1e-9:
            raise ControllerError("controller-timestep-does-not-match-config")
        points = _rear_points(waypoints, self.geometry.rear_axle_x_m, self.geometry.rear_axle_y_m)
        cfg = self.config
        requested_lookahead = _clip(cfg.lookahead_base_m + cfg.lookahead_gain_seconds * speed,
                                    cfg.lookahead_min_m, cfg.lookahead_max_m)
        target, selection, segment_index = _lookahead_target(points, requested_lookahead)
        distance_squared = target[0] * target[0] + target[1] * target[1]
        target_actor = [target[0] + self.geometry.rear_axle_x_m,
                        target[1] + self.geometry.rear_axle_y_m]
        curvature, actor_circle_denominator = _actor_circle_curvature(
            target_actor, [self.geometry.rear_axle_x_m, self.geometry.rear_axle_y_m])
        preview_curvatures = _preview_curvatures(points)
        preview_curvature = max(preview_curvatures)
        speed_curvature = max(preview_curvature, abs(curvature))
        nominal_target_speed = cfg.target_speed_kmh / 3.6
        # None is explicit for a straight target; never emit JSON Infinity.
        curve_speed_cap = (math.sqrt(cfg.max_lateral_acceleration_mps2 / speed_curvature)
                           if speed_curvature > 1e-9 else None)
        effective_target_speed = (min(nominal_target_speed, curve_speed_cap)
                                  if curve_speed_cap is not None else nominal_target_speed)
        steering_rad = math.atan(self.geometry.wheelbase_m * curvature)
        # PhysX steers the INNER wheel by normalized input * steering gain and
        # derives the outer angle. PP's atan(L*kappa) is the bicycle-centre angle,
        # so sending it directly understeers by ignoring half the front track.
        # atan2 keeps the requested turn sign even for an infeasible tight target;
        # ordinary command clipping then limits it to native actuator range.
        inner_radius_factor = 1.0 - self.geometry.front_track_m * abs(curvature) / 2.0
        native_inner_steering_rad = (math.copysign(
            math.atan2(self.geometry.wheelbase_m * abs(curvature), inner_radius_factor), curvature)
            if curvature != 0.0 else 0.0)
        steering_factor = self.geometry.steering_factor(speed)
        effective_limit = math.radians(self.geometry.max_steer_angle_deg) * steering_factor
        unclipped_steer = native_inner_steering_rad / effective_limit
        clipped_steer = _clip(unclipped_steer, -cfg.max_normalized_steer, cfg.max_normalized_steer)
        maximum_change = cfg.steer_rate_per_second * timestep
        steer = _clip(clipped_steer, self._previous_steer - maximum_change,
                      self._previous_steer + maximum_change)
        error = effective_target_speed - speed
        integration_error = 0.0 if abs(error) <= cfg.speed_deadband_mps else error
        candidate_integral = _clip(self._speed_integral_m + integration_error * timestep,
                                   -cfg.integral_limit_m, cfg.integral_limit_m)
        candidate_effort = cfg.speed_kp * error + cfg.speed_ki * candidate_integral
        antiwindup = ((candidate_effort > cfg.throttle_max and integration_error > 0.0)
                      or (candidate_effort < -cfg.brake_max and integration_error < 0.0))
        integral = self._speed_integral_m if antiwindup else candidate_integral
        effort = cfg.speed_kp * error + cfg.speed_ki * integral
        bounded_effort = _clip(effort, -cfg.brake_max, cfg.throttle_max)
        throttle, brake = max(0.0, bounded_effort), max(0.0, -bounded_effort)
        diagnostics = {
            "controller_version": CONTROLLER_VERSION,
            "configuration_sha256": self._config_sha256,
            "step_index": self._step_index,
            "dt_seconds": timestep,
            "waypoint_origin": "vehicle_actor_transform",
            "control_origin": "measured-rear-axle",
            "rear_axle_x_m": self.geometry.rear_axle_x_m,
            "rear_axle_y_m": self.geometry.rear_axle_y_m,
            "target_rear_axle_xy_m": target,
            "target_actor_xy_m": target_actor,
            "lookahead_requested_m": requested_lookahead,
            "lookahead_effective_m": math.sqrt(distance_squared),
            "lookahead_selection": selection,
            "lookahead_segment_index": segment_index,
            "curvature_per_m": curvature,
            "curvature_geometry": "actor-origin-path-circle-with-measured-rear-offset",
            "actor_circle_denominator_m2": actor_circle_denominator,
            "steering_rad": steering_rad,
            "steering_normalization": "ackermann-inner-front-wheel-angle",
            "front_track_m": self.geometry.front_track_m,
            "native_inner_steering_rad": native_inner_steering_rad,
            "native_inner_steering_deg": math.degrees(native_inner_steering_rad),
            "ackermann_inner_radius_nonpositive": inner_radius_factor <= 0.0,
            "steering_curve_factor": steering_factor,
            "effective_max_steering_rad": effective_limit,
            "steer_unclipped": unclipped_steer,
            "steer_clipped": clipped_steer,
            "steer_rate_limited": abs(steer - clipped_steer) > 1e-12,
            "target_speed_kmh": cfg.target_speed_kmh,
            "nominal_target_speed_mps": nominal_target_speed,
            "preview_curvatures_per_m": preview_curvatures,
            "preview_curvature_per_m": preview_curvature,
            "speed_limit_curvature_per_m": speed_curvature,
            "max_lateral_acceleration_mps2": cfg.max_lateral_acceleration_mps2,
            "curvature_speed_cap_mps": curve_speed_cap,
            "effective_target_speed_mps": effective_target_speed,
            "effective_target_speed_kmh": effective_target_speed * 3.6,
            "curvature_speed_limited": effective_target_speed < nominal_target_speed - 1e-12,
            "speed_mps": speed,
            "speed_error_mps": error,
            "speed_integral_m": integral,
            "speed_pi_effort": effort,
            "speed_pi_antiwindup": antiwindup,
            "speed_pi_saturated": abs(effort - bounded_effort) > 1e-12,
        }
        self._speed_integral_m = integral
        self._previous_steer = steer
        self._step_index += 1
        return {"throttle": throttle, "brake": brake, "steer": steer,
                "hand_brake": False, "reverse": False, "diagnostics": diagnostics}
