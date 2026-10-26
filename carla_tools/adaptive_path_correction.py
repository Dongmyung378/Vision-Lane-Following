"""Continuously bound RGB proposals and reduce poorly observed turn weakening.

Only the current RGB image, its learned path and fixed nominal calibration are
inputs.  The original RGB proposal implementation is retained unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
import hashlib
import math
from pathlib import Path

from .contracts import canonical_sha256
from .guarded_path_correction import GuardConfig, guard_path_proposal
from .learning_data import validate_waypoint_points
from .path_correction import PathCorrectionConfig, RGBPathCorrection
from .supported_path_correction import SupportedRGBPathCorrection, _shape_change

VERSION = "vlf-rgb-adaptive-bounded-path-v1"
BUDGET_VERSION = "vlf-continuous-near-disagreement-budget-v1"


@dataclass(frozen=True)
class AdaptivePathConfig:
    schema_version: int = 1
    version: str = VERSION
    maximum_first_point_change_m: float = .5
    maximum_near_mean_change_m: float = .5
    disagreement_fade_start_m: float = .5
    disagreement_fade_end_m: float = .75
    uncertain_span_m: float = 1.
    trusted_span_m: float = 1.8
    fit_residual_full_trust_m: float = .03
    fit_residual_no_trust_m: float = .10
    curvature_risk_start_per_m: float = .01
    curvature_risk_full_per_m: float = .04
    turn_weakening_full_per_m: float = .015
    maximum_segment_heading_change_rad: float = .12
    maximum_curvature_change_per_m: float = .04
    scale_bisection_steps: int = 20

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1 or self.version != VERSION:
            raise ValueError("adaptive-path-version-invalid")
        for name, value in self.to_dict().items():
            if name in ("schema_version", "version", "scale_bisection_steps"):
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("adaptive-path-bound-invalid:" + name)
        if self.maximum_first_point_change_m != .5 or self.maximum_near_mean_change_m != .5:
            raise ValueError("adaptive-path-original-output-budgets-required")
        if not (.5 <= self.disagreement_fade_start_m < self.disagreement_fade_end_m <= 1.
                and .25 <= self.uncertain_span_m < self.trusted_span_m <= 5.
                and .001 <= self.fit_residual_full_trust_m < self.fit_residual_no_trust_m <= .15
                and .001 <= self.curvature_risk_start_per_m < self.curvature_risk_full_per_m <= .15
                and .001 <= self.turn_weakening_full_per_m <= .1
                and .01 <= self.maximum_segment_heading_change_rad <= .3
                and .005 <= self.maximum_curvature_change_per_m <= .15):
            raise ValueError("adaptive-path-bound-order-invalid")
        if type(self.scale_bisection_steps) is not int or not 16 <= self.scale_bisection_steps <= 24:
            raise ValueError("adaptive-path-bisection-steps-invalid")

    def to_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return canonical_sha256(self.to_dict())


def adaptive_config_from_dict(value):
    if type(value) is not dict or set(value) != {field.name for field in fields(AdaptivePathConfig)}:
        raise ValueError("adaptive-path-config-fields-invalid")
    return AdaptivePathConfig(**value)


def adaptive_source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _smoothstep(value):
    value = max(0., min(1., value))
    return value * value * (3. - 2. * value)


def _ramp(value, low, high):
    return _smoothstep((value - low) / (high - low))


def _near_curvature(points):
    a, b, c = points[:3]
    denominator = math.dist(a, b) * math.dist(b, c) * math.dist(c, a)
    if denominator <= 1e-12:
        raise ValueError("adaptive-path-degenerate-near-shape")
    return 2. * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / denominator


def ego_circle_curvature(point):
    """Curvature of the circle tangent to ego x and passing through a point.

    This is a spatial path property. No controller, speed or native pose is used.
    """
    x, y = point
    squared = x * x + y * y
    if squared <= 1e-12:
        raise ValueError("adaptive-path-origin-point-invalid")
    return 2. * y / squared


def proposal_quality(points, proposed, support_span_m, residual_p95_m, config):
    """Combine measured RGB observability and symmetric path-turn consistency."""
    if (type(support_span_m) not in (int, float) or not math.isfinite(support_span_m)
            or support_span_m <= 0):
        raise ValueError("adaptive-path-observed-span-invalid")
    if residual_p95_m is not None and (type(residual_p95_m) not in (int, float)
            or not math.isfinite(residual_p95_m) or residual_p95_m < 0):
        raise ValueError("adaptive-path-residual-invalid")
    curvature = _near_curvature(points)
    direction = math.copysign(1., curvature) if curvature != 0 else 0.
    raw_circles = [ego_circle_curvature(p) for p in points[:2]]
    proposed_circles = [ego_circle_curvature(p) for p in proposed[:2]]
    weakening = max(0., *(direction * (raw - proposal)
        for raw, proposal in zip(raw_circles, proposed_circles)))
    short_span = 1. - _ramp(support_span_m, config.uncertain_span_m, config.trusted_span_m)
    fit_uncertainty = (0. if residual_p95_m is None else
        _ramp(residual_p95_m, config.fit_residual_full_trust_m, config.fit_residual_no_trust_m))
    # Union of evidence limitations; a short span alone never deletes a
    # straight-path correction or a proposal that maintains the raw turn.
    uncertainty = 1. - (1. - short_span) * (1. - fit_uncertainty)
    bend = _ramp(abs(curvature), config.curvature_risk_start_per_m, config.curvature_risk_full_per_m)
    turn_risk = _smoothstep(weakening / config.turn_weakening_full_per_m)
    confidence = 1. - uncertainty * bend * turn_risk
    return {"observed_forward_span_m": float(support_span_m), "residual_p95_m": residual_p95_m,
        "raw_signed_near_curvature_per_m": curvature, "raw_ego_circle_curvatures_per_m": raw_circles,
        "proposed_ego_circle_curvatures_per_m": proposed_circles,
        "maximum_turn_weakening_per_m": weakening, "short_span_uncertainty": short_span,
        "fit_uncertainty": fit_uncertainty, "combined_observation_uncertainty": uncertainty,
        "bend_risk": bend, "turn_weakening_risk": turn_risk, "geometry_confidence": confidence,
        "distance_cutoff_used": False, "controller_or_speed_used": False}


def adaptive_guard_path_proposal(points, proposed, support_span_m, residual_p95_m=None, config=None):
    """Use one continuous blend for both near points, preserving output budgets."""
    config = AdaptivePathConfig() if config is None else config
    if not isinstance(config, AdaptivePathConfig):
        raise ValueError("adaptive-path-config-required")
    validate_waypoint_points(points); validate_waypoint_points(proposed)
    _, legacy = guard_path_proposal(points, proposed, GuardConfig())
    original = copy.deepcopy(points)
    distances = [math.dist(a, b) for a, b in zip(points[:2], proposed[:2])]
    mean = math.fsum(distances) / 2
    cap = min(1., config.maximum_first_point_change_m / distances[0] if distances[0] else 1.,
        config.maximum_near_mean_change_m / mean if mean else 1.)
    # The proposal may exceed a budget, but the selected output never does.
    # No discontinuous accept/reject decision is made at the 0.5m boundary.
    disagreement = max(distances[0], mean)
    outer = 1. - _ramp(disagreement, config.disagreement_fade_start_m, config.disagreement_fade_end_m)
    quality = proposal_quality(points, proposed, support_span_m, residual_p95_m, config)
    requested = cap * outer * quality["geometry_confidence"]
    delta = [[b[j] - a[j] for j in range(2)] for a, b in zip(points[:2], proposed[:2])]
    def scaled(scale):
        result = copy.deepcopy(original)
        for index in range(2):
            result[index] = [original[index][j] + scale * delta[index][j] for j in range(2)]
        return result
    def admissible(scale):
        selected = scaled(scale)
        try:
            validate_waypoint_points(selected)
            shape = _shape_change(original, selected)
        except ValueError:
            return False
        return (shape["maximum_segment_heading_change_rad"] <= config.maximum_segment_heading_change_rad
            and shape["maximum_signed_curvature_change_per_m"] <= config.maximum_curvature_change_per_m)
    scale = requested
    if not admissible(scale):
        low, high = 0., scale
        for _ in range(config.scale_bisection_steps):
            middle = (low + high) / 2
            if admissible(middle):
                low = middle
            else:
                high = middle
        scale = low
    selected = scaled(scale)
    # Tiny contraction accounts for subtraction rounding at an exact budget;
    # it tightens the original budgets rather than adding a tolerance to them.
    _, final_guard = guard_path_proposal(original, selected, GuardConfig())
    if not final_guard["accepted"]:
        scale *= 1. - 1e-12
        selected = scaled(scale)
        _, final_guard = guard_path_proposal(original, selected, GuardConfig())
    if not final_guard["accepted"] or not admissible(scale):
        raise ValueError("adaptive-path-output-budget-or-shape-invalid")
    actual_changes = [math.dist(a, b) for a, b in zip(original[:2], selected[:2])]
    accepted = any(value > 1e-12 for value in actual_changes)
    guard = {"version": BUDGET_VERSION, "config_sha256": config.sha256,
        "input": "current-RGB-learned-and-current-RGB-proposed-waypoints",
        "temporal_state": "none", "accepted": accepted, "rejection_reasons": [] if accepted else ["zero-continuous-blend"],
        "proposal_near_point_change_m": distances, "proposal_near_mean_change_m": mean,
        "near_point_change_m": actual_changes, "near_mean_change_m": math.fsum(actual_changes) / 2,
        "budget_projection_scale": cap, "outer_disagreement_confidence": outer,
        "legacy_guard_would_accept": legacy["accepted"], "original_output_bounds_verified": True,
        "maximum_first_point_change_m": .5, "maximum_near_mean_change_m": .5,
        "selected_source": "continuously-bounded-RGB-proposal" if accepted else "current-learned-RGB-path",
        "distant_points_unchanged": True}
    detail = {"version": VERSION, "config_sha256": config.sha256,
        "input": "current-RGB-proposal-and-current-learned-path-and-observed-RGB-support",
        "temporal_state": "none", "accepted": accepted, "rejection_reasons": guard["rejection_reasons"],
        "quality": quality, "requested_common_scale": requested, "common_scale": scale,
        "shape_scale": scale / requested if requested else 0.,
        "shape_change": _shape_change(original, selected), "distant_points_unchanged": True}
    return selected, guard, detail


class AdaptiveRGBPathCorrection:
    def __init__(self, base=None, config=None):
        self.base = RGBPathCorrection() if base is None else base
        self.config = AdaptivePathConfig() if config is None else config
        if not isinstance(self.config, AdaptivePathConfig) or self.base.config.sha256 != PathCorrectionConfig().sha256:
            raise ValueError("adaptive-path-original-base-config-required")
        self.baseline = self.base.baseline
        self._base_sha256, self._baseline_sha256 = self.base.config.sha256, self.baseline.sha256
        self._support_reader = SupportedRGBPathCorrection(self.base)

    def apply(self, rgb, points, *, width=640, height=360):
        if self.base.config.sha256 != self._base_sha256 or self.baseline.sha256 != self._baseline_sha256:
            raise ValueError("adaptive-path-base-config-changed")
        original = copy.deepcopy(points)
        proposed, legacy_detail = self.base.apply(rgb, original, width=width, height=height)
        if original != points:
            raise ValueError("adaptive-path-base-mutated-input")
        support, evidence = self._support_reader._support(rgb, legacy_detail, width=width, height=height)
        if support is None:
            selected, guard, detail = adaptive_guard_path_proposal(points, points, 1., None, self.config)
            detail.update(accepted=False, rejection_reasons=["observed-support-unavailable"])
            guard.update(accepted=False, rejection_reasons=["observed-support-unavailable"])
        else:
            residual = (legacy_detail["local_correction"].get("residual_p95_m")
                if legacy_detail.get("method") == "local-near-boundary" else None)
            selected, guard, detail = adaptive_guard_path_proposal(points, proposed,
                support[1] - support[0], residual, self.config)
        detail.update(observed_support=evidence, center_support_forward_m=support,
            proposed_waypoints_ego_m=copy.deepcopy(proposed), base_path_config_sha256=self._base_sha256)
        return selected, {"path_correction": legacy_detail, "guarded_path_correction": guard,
            "supported_path_correction": detail}


class AdaptivePathPredictor:
    def __init__(self, base, config=None):
        if (getattr(base, "correction", None) != "path" or not callable(getattr(base, "timer", None))
                or not hasattr(base, "raw") or base.raw.identity.get("variant") != "hybrid_geometry"):
            raise ValueError("adaptive-path-runtime-RGB-base-required")
        self.base = base
        self.config = AdaptivePathConfig() if config is None else config
        self.raw = self.raw_predictor = base.raw
        self.corrector = AdaptiveRGBPathCorrection(base.corrector, self.config)
        self.correction = "supported_path"
        self._identity = copy.deepcopy(base.identity)
        if self._identity.get("correction_config_sha256") != self.corrector.base.config.sha256:
            raise ValueError("adaptive-path-base-identity-config-mismatch")
        self._identity.update(variant=self._identity["variant"] + "_adaptive", correction=self.correction,
            adaptive_version=VERSION, adaptive_config=self.config.to_dict(), adaptive_config_sha256=self.config.sha256,
            adaptive_source_sha256=adaptive_source_sha256(), base_correction_variant=base.identity["variant"])

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def timer(self):
        return self.base.timer()

    def predict_profiled(self, rgb, timer, *, width=640, height=360):
        from .runtime_driving import RuntimeCheckpointPredictor
        prediction = RuntimeCheckpointPredictor.predict_profiled(self, rgb, timer, width=width, height=height)
        detail = prediction.get("diagnostics", {}).get("path_correction")
        if detail is not None:
            if type(detail) is not dict or set(detail) != {"path_correction", "guarded_path_correction", "supported_path_correction"}:
                raise ValueError("adaptive-path-composite-diagnostic-invalid")
            prediction["diagnostics"].update(detail)
        return prediction

    def predict(self, rgb, *, width=640, height=360):
        return self.predict_profiled(rgb, self.timer(), width=width, height=height)
