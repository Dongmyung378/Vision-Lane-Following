"""Bounded controller alternatives without changing the legacy control contract.

Waypoints describe the vehicle actor's future path in metres (x forward, y
right). Stanley projects the measured front axle onto the *corresponding front
axle path*, rather than treating actor-origin labels as front-axle labels.
There is no CARLA import or simulator side effect in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .contracts import ContractError, read_strict_json
from .controller import (
    ControllerConfig, ControllerError, PurePursuitController, VehicleGeometry,
    _bounded, _clip, _number, _preview_curvatures, _rear_points,
    controller_config_from_dict,
)


STUDY_CONTROLLER_VERSION = "vlf-controller-study-v1"
DEFAULT_STUDY_CONTROLLER_PATH = Path(__file__).resolve().parents[1] / "configs" / "controller_study.json"


@dataclass(frozen=True)
class ControllerStudyConfig:
    """An explicit method wrapper around an unchanged legacy configuration."""

    schema_version: int = 1
    controller_version: str = STUDY_CONTROLLER_VERSION
    method: str = "pure_pursuit"
    base: ControllerConfig = field(default_factory=ControllerConfig)
    stanley_cross_track_gain: float = 1.0
    stanley_softening_mps: float = 1.0
    stanley_heading_gain: float = 1.0
    steer_smoothing_alpha: float = 1.0

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ControllerError("invalid-controller-study-schema-version")
        if self.controller_version != STUDY_CONTROLLER_VERSION:
            raise ControllerError("invalid-controller-study-version")
        if type(self.method) is not str or self.method not in ("pure_pursuit", "stanley"):
            raise ControllerError("invalid-controller-study-method")
        if type(self.base) is not ControllerConfig:
            raise ControllerError("controller-study-requires-legacy-base-config")
        bounds = {
            "stanley_cross_track_gain": (0.01, 10.0),
            "stanley_softening_mps": (0.05, 10.0),
            "stanley_heading_gain": (0.1, 2.0),
            "steer_smoothing_alpha": (0.01, 1.0),
        }
        for name, limits in bounds.items():
            object.__setattr__(self, name, _bounded(
                getattr(self, name), *limits, f"invalid-controller-study-{name}"))

    def __getattr__(self, name: str) -> Any:
        # Existing runners read target_speed_kmh/fixed_delta_seconds directly.
        # Forward only declared legacy fields; do not silently accept new keys.
        if name in ControllerConfig.__dataclass_fields__:
            return getattr(object.__getattribute__(self, "base"), name)
        raise AttributeError(name)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def study_controller_config_from_dict(value: Any) -> ControllerStudyConfig:
    expected = {item.name for item in fields(ControllerStudyConfig)}
    if type(value) is not dict or set(value) != expected:
        raise ControllerError("invalid-controller-study-config-fields")
    arguments = dict(value)
    arguments["base"] = controller_config_from_dict(arguments["base"])
    return ControllerStudyConfig(**arguments)


def load_study_controller_config(path: str | Path = DEFAULT_STUDY_CONTROLLER_PATH) -> ControllerStudyConfig:
    try:
        return study_controller_config_from_dict(read_strict_json(path))
    except ContractError as error:
        raise ControllerError(f"controller-study-config-{error.code}") from error


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _stanley_front_reference(points_actor: list[list[float]], geometry: VehicleGeometry) -> dict:
    """Fit the first local arc and project onto its front-axle counterpart.

    Collinear actor points define a line. Otherwise three points determine an
    osculating actor circle. For a constant rear bicycle curvature k, the actor
    radius is sqrt(rear_x**2 + (rear_y + 1/k)**2). This identity converts the
    actor circle into a front-axle circle with the SAME world centre. The front
    path tangent supplies the usual Stanley heading term, including the curve
    feedforward at zero cross-track error. No measured or privileged route is
    used; all reference geometry comes from the supplied five waypoints.
    """
    first, second, third = points_actor[:3]
    ax, ay = second[0] - first[0], second[1] - first[1]
    bx, by = third[0] - first[0], third[1] - first[1]
    determinant = ax * by - ay * bx
    scale = math.hypot(ax, ay) * math.hypot(bx, by)
    if scale <= 1e-12:
        raise ControllerError("controller-stanley-local-points-too-close")
    front = [geometry.front_axle_x_m, geometry.front_axle_y_m]
    if abs(determinant) <= 1e-8 * scale:
        length = math.hypot(ax, ay)
        ux, uy = ax / length, ay / length
        if ux <= 1e-6:
            raise ControllerError("controller-stanley-local-path-not-forward")
        # Desired front position = actor position + R(path heading)*front offset.
        origin = [first[0] + ux * front[0] - uy * front[1],
                  first[1] + uy * front[0] + ux * front[1]]
        fraction = (front[0] - origin[0]) * ux + (front[1] - origin[1]) * uy
        target = [origin[0] + fraction * ux, origin[1] + fraction * uy]
        heading = math.atan2(uy, ux)
        cross_track = -uy * (target[0] - front[0]) + ux * (target[1] - front[1])
        return {"geometry": "actor-line-to-measured-front-axle-line",
                "target_front_axle_xy_m": target, "heading_error_rad": heading,
                "cross_track_error_m": cross_track, "rear_reference_curvature_per_m": 0.0,
                "actor_reference_curvature_per_m": 0.0,
                "actor_circle_center_xy_m": None, "actor_circle_radius_m": None,
                "front_circle_radius_m": None}
    # Solve using coordinates relative to the first point to avoid cancellation.
    a2, b2 = ax * ax + ay * ay, bx * bx + by * by
    cx = (a2 * by - b2 * ay) / (2.0 * determinant)
    cy = (ax * b2 - bx * a2) / (2.0 * determinant)
    centre = [first[0] + cx, first[1] + cy]
    actor_radius = math.hypot(cx, cy)
    sign = math.copysign(1.0, determinant)
    radius_squared = actor_radius * actor_radius - geometry.rear_axle_x_m ** 2
    if radius_squared <= 1e-10:
        raise ControllerError("controller-stanley-actor-circle-not-physically-reachable")
    rear_radius = -geometry.rear_axle_y_m + sign * math.sqrt(radius_squared)
    if rear_radius * sign <= 1e-6:
        raise ControllerError("controller-stanley-rear-circle-not-forward")
    front_radius = math.hypot(geometry.front_axle_x_m - geometry.rear_axle_x_m,
                             geometry.front_axle_y_m - geometry.rear_axle_y_m - rear_radius)
    dx, dy = front[0] - centre[0], front[1] - centre[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-8:
        raise ControllerError("controller-stanley-front-axle-at-circle-centre")
    target = [centre[0] + front_radius * dx / distance,
              centre[1] + front_radius * dy / distance]
    # In the x-forward/y-right convention positive curvature is clockwise on
    # the physical road, but uses the ordinary signed planar cross product.
    heading = _wrap_angle(math.atan2(dy, dx) + sign * math.pi / 2.0)
    if math.cos(heading) <= 1e-6:
        raise ControllerError("controller-stanley-local-path-not-forward")
    cross_track = (-math.sin(heading) * (target[0] - front[0])
                   + math.cos(heading) * (target[1] - front[1]))
    return {"geometry": "actor-circle-to-measured-front-axle-circle",
            "target_front_axle_xy_m": target, "heading_error_rad": heading,
            "cross_track_error_m": cross_track,
            "rear_reference_curvature_per_m": 1.0 / rear_radius,
            "actor_reference_curvature_per_m": sign / actor_radius,
            "actor_circle_center_xy_m": centre, "actor_circle_radius_m": actor_radius,
            "front_circle_radius_m": front_radius}


class _StudyController:
    def __init__(self, config: ControllerStudyConfig, geometry: VehicleGeometry):
        if type(config) is not ControllerStudyConfig or not isinstance(geometry, VehicleGeometry):
            raise ControllerError("controller-study-requires-validated-config-and-geometry")
        self.config, self.geometry = config, geometry
        self._config_sha256 = config.sha256
        self.reset()

    def reset(self) -> None:
        self._speed_integral_m = 0.0
        self._previous_steer = 0.0
        self._step_index = 0

    def _steering(self, clipped: float, timestep: float) -> tuple[float, dict]:
        alpha = self.config.steer_smoothing_alpha
        # Preserve exact legacy arithmetic when smoothing is disabled.
        filtered = (clipped if alpha == 1.0
                    else self._previous_steer + alpha * (clipped - self._previous_steer))
        maximum_change = self.config.steer_rate_per_second * timestep
        steer = _clip(filtered, self._previous_steer - maximum_change,
                      self._previous_steer + maximum_change)
        return steer, {"steer_smoothing_alpha": alpha,
                       "steer_before_smoothing": clipped, "steer_filtered": filtered,
                       "steer_smoothed": abs(filtered - clipped) > 1e-12,
                       "steer_rate_limited": abs(steer - filtered) > 1e-12}

    def _metadata(self, diagnostics: dict) -> None:
        diagnostics.update(controller_version=STUDY_CONTROLLER_VERSION,
                           configuration_sha256=self._config_sha256,
                           base_controller_configuration_sha256=self.config.base.sha256,
                           controller_method=self.config.method)


class StudyPurePursuitController(_StudyController):
    """Legacy PP geometry and PI, followed by explicit smoothing and slew."""

    def __init__(self, config: ControllerStudyConfig, geometry: VehicleGeometry):
        if type(config) is not ControllerStudyConfig or config.method != "pure_pursuit":
            raise ControllerError("pure-pursuit-study-requires-pure-pursuit-method")
        self._delegate = PurePursuitController(config.base, geometry)
        super().__init__(config, geometry)

    def reset(self) -> None:
        super().reset()
        self._delegate.reset()

    def step(self, waypoints: Any, speed_mps: float, dt_seconds: float = 0.05) -> dict:
        result = self._delegate.step(waypoints, speed_mps, dt_seconds)
        diagnostics = result["diagnostics"]
        steer, filter_diagnostics = self._steering(diagnostics["steer_clipped"], dt_seconds)
        diagnostics.update(filter_diagnostics)
        self._metadata(diagnostics)
        result["steer"] = steer
        self._previous_steer = steer
        self._step_index += 1
        return result


class StanleyController(_StudyController):
    """Front-axle Stanley with low-speed softening and measured path offsets."""

    def __init__(self, config: ControllerStudyConfig, geometry: VehicleGeometry):
        if type(config) is not ControllerStudyConfig or config.method != "stanley":
            raise ControllerError("stanley-requires-stanley-study-method")
        super().__init__(config, geometry)

    def step(self, waypoints: Any, speed_mps: float, dt_seconds: float = 0.05) -> dict:
        speed = _bounded(speed_mps, 0.0, 100.0, "invalid-forward-speed")
        timestep = _number(dt_seconds, "invalid-controller-timestep")
        cfg = self.config
        if abs(timestep - cfg.fixed_delta_seconds) > 1e-9:
            raise ControllerError("controller-timestep-does-not-match-config")
        rear = _rear_points(waypoints, self.geometry.rear_axle_x_m, self.geometry.rear_axle_y_m)
        preview_curvatures = _preview_curvatures(rear)
        actor = [[x + self.geometry.rear_axle_x_m, y + self.geometry.rear_axle_y_m] for x, y in rear]
        reference = _stanley_front_reference(actor, self.geometry)
        heading_term = cfg.stanley_heading_gain * reference["heading_error_rad"]
        cross_track_term = math.atan2(cfg.stanley_cross_track_gain * reference["cross_track_error_m"],
                                     speed + cfg.stanley_softening_mps)
        requested_steering = heading_term + cross_track_term
        # Bicycle steering remains in its single-track domain before the native
        # Ackermann conversion. Native normalized limits are applied afterwards.
        steering_rad = _clip(requested_steering, -math.pi / 2.0 + 1e-6, math.pi / 2.0 - 1e-6)
        curvature = math.tan(steering_rad) / self.geometry.wheelbase_m
        inner_radius_factor = 1.0 - self.geometry.front_track_m * abs(curvature) / 2.0
        native_inner = (math.copysign(math.atan2(self.geometry.wheelbase_m * abs(curvature),
                                               inner_radius_factor), curvature)
                        if curvature else 0.0)
        factor = self.geometry.steering_factor(speed)
        effective_limit = math.radians(self.geometry.max_steer_angle_deg) * factor
        unclipped = native_inner / effective_limit
        clipped = _clip(unclipped, -cfg.max_normalized_steer, cfg.max_normalized_steer)
        steer, filter_diagnostics = self._steering(clipped, timestep)
        speed_curvature = max(max(preview_curvatures), abs(curvature))
        nominal_speed = cfg.target_speed_kmh / 3.6
        curve_cap = (math.sqrt(cfg.max_lateral_acceleration_mps2 / speed_curvature)
                     if speed_curvature > 1e-9 else None)
        target_speed = min(nominal_speed, curve_cap) if curve_cap is not None else nominal_speed
        error = target_speed - speed
        integration_error = 0.0 if abs(error) <= cfg.speed_deadband_mps else error
        candidate_integral = _clip(self._speed_integral_m + integration_error * timestep,
                                   -cfg.integral_limit_m, cfg.integral_limit_m)
        candidate_effort = cfg.speed_kp * error + cfg.speed_ki * candidate_integral
        antiwindup = ((candidate_effort > cfg.throttle_max and integration_error > 0.0)
                      or (candidate_effort < -cfg.brake_max and integration_error < 0.0))
        integral = self._speed_integral_m if antiwindup else candidate_integral
        effort = cfg.speed_kp * error + cfg.speed_ki * integral
        bounded_effort = _clip(effort, -cfg.brake_max, cfg.throttle_max)
        diagnostics = {
            "step_index": self._step_index, "dt_seconds": timestep,
            "waypoint_origin": "vehicle_actor_transform", "control_origin": "measured-front-axle",
            "rear_axle_x_m": self.geometry.rear_axle_x_m, "rear_axle_y_m": self.geometry.rear_axle_y_m,
            "front_axle_x_m": self.geometry.front_axle_x_m, "front_axle_y_m": self.geometry.front_axle_y_m,
            "stanley_reference": reference, "stanley_heading_term_rad": heading_term,
            "stanley_cross_track_term_rad": cross_track_term,
            "stanley_requested_steering_rad": requested_steering,
            "stanley_softening_mps": cfg.stanley_softening_mps,
            "curvature_per_m": curvature, "steering_rad": steering_rad,
            "steering_normalization": "ackermann-inner-front-wheel-angle",
            "front_track_m": self.geometry.front_track_m, "native_inner_steering_rad": native_inner,
            "native_inner_steering_deg": math.degrees(native_inner),
            "ackermann_inner_radius_nonpositive": inner_radius_factor <= 0.0,
            "steering_curve_factor": factor, "effective_max_steering_rad": effective_limit,
            "steer_unclipped": unclipped, "steer_clipped": clipped,
            "target_speed_kmh": cfg.target_speed_kmh, "nominal_target_speed_mps": nominal_speed,
            "preview_curvatures_per_m": preview_curvatures,
            "preview_curvature_per_m": max(preview_curvatures),
            "speed_limit_curvature_per_m": speed_curvature,
            "max_lateral_acceleration_mps2": cfg.max_lateral_acceleration_mps2,
            "curvature_speed_cap_mps": curve_cap, "effective_target_speed_mps": target_speed,
            "effective_target_speed_kmh": target_speed * 3.6,
            "curvature_speed_limited": target_speed < nominal_speed - 1e-12,
            "speed_mps": speed, "speed_error_mps": error, "speed_integral_m": integral,
            "speed_pi_effort": effort, "speed_pi_antiwindup": antiwindup,
            "speed_pi_saturated": abs(effort - bounded_effort) > 1e-12,
            **filter_diagnostics,
        }
        self._metadata(diagnostics)
        self._speed_integral_m, self._previous_steer = integral, steer
        self._step_index += 1
        return {"throttle": max(0.0, bounded_effort), "brake": max(0.0, -bounded_effort),
                "steer": steer, "hand_brake": False, "reverse": False, "diagnostics": diagnostics}


def build_controller(config: ControllerConfig | ControllerStudyConfig,
                     geometry: VehicleGeometry) -> PurePursuitController | _StudyController:
    """Dispatch explicitly; legacy configurations retain their original object."""
    if type(config) is ControllerConfig:
        return PurePursuitController(config, geometry)
    if type(config) is not ControllerStudyConfig:
        raise ControllerError("controller-factory-requires-validated-config")
    return (StudyPurePursuitController(config, geometry) if config.method == "pure_pursuit"
            else StanleyController(config, geometry))
