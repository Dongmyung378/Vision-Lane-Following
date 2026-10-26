"""Limit current RGB path proposals to observed paint support and local shape.

This development candidate retains the original proposal and 0.5m guard.
It only attenuates an accepted near-point proposal. It has no route, pose,
ground truth, speed, camera readback or previous-frame input.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np

from .baseline import _marking_mask, _rgb_array
from .contracts import canonical_sha256
from .guarded_path_correction import GuardConfig, guard_path_proposal
from .learning_data import validate_waypoint_points
from .path_correction import PathCorrectionConfig, RGBPathCorrection

VERSION = "vlf-rgb-observed-support-path-v1"


@dataclass(frozen=True)
class SupportedPathConfig:
    schema_version: int = 1
    version: str = VERSION
    maximum_extrapolation_m: float = 1.0
    maximum_segment_heading_change_rad: float = .12
    maximum_curvature_change_per_m: float = .04
    scale_bisection_steps: int = 16

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1 or self.version != VERSION:
            raise ValueError("supported-path-version-invalid")
        for name, low, high in (("maximum_extrapolation_m", .1, 1.5),
                ("maximum_segment_heading_change_rad", .01, .3),
                ("maximum_curvature_change_per_m", .005, .15)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError("supported-path-bound-invalid:" + name)
        if type(self.scale_bisection_steps) is not int or not 12 <= self.scale_bisection_steps <= 24:
            raise ValueError("supported-path-bisection-steps-invalid")

    def to_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return canonical_sha256(self.to_dict())


def supported_config_from_dict(value):
    if type(value) is not dict or set(value) != {field.name for field in fields(SupportedPathConfig)}:
        raise ValueError("supported-path-config-fields-invalid")
    return SupportedPathConfig(**value)


def supported_source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _shape(points):
    headings = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(points, points[1:])]
    curvatures = []
    for a, b, c in zip(points, points[1:], points[2:]):
        denominator = math.dist(a, b) * math.dist(b, c) * math.dist(c, a)
        if denominator <= 1e-12:
            raise ValueError("supported-path-degenerate-shape")
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        curvatures.append(2. * cross / denominator)
    return headings, curvatures


def _shape_change(original, candidate):
    old_h, old_k = _shape(original)
    new_h, new_k = _shape(candidate)
    return {"maximum_segment_heading_change_rad": max(abs(math.atan2(math.sin(b - a), math.cos(b - a)))
                for a, b in zip(old_h, new_h)),
            "maximum_signed_curvature_change_per_m": max(abs(b - a) for a, b in zip(old_k, new_k)),
            "original_signed_curvatures_per_m": old_k, "selected_signed_curvatures_per_m": new_k}


def support_guard_path_proposal(learned_points, proposed_points, support_interval, config=None):
    """Attenuate a legacy-accepted proposal, leaving every far point exact."""
    config = SupportedPathConfig() if config is None else config
    if not isinstance(config, SupportedPathConfig):
        raise ValueError("supported-path-config-required")
    validate_waypoint_points(learned_points)
    validate_waypoint_points(proposed_points)
    legacy_selected, legacy = guard_path_proposal(learned_points, proposed_points, GuardConfig())
    original = copy.deepcopy(learned_points)
    detail = {"version": VERSION, "config_sha256": config.sha256,
        "input": "current-RGB-proposal-and-current-learned-path-and-observed-RGB-support",
        "temporal_state": "none", "accepted": False, "rejection_reasons": [],
        "center_support_forward_m": None, "outside_support_m": [0., 0.],
        "support_weights": [0., 0.], "shape_scale": 0., "distant_points_unchanged": True}
    if not legacy["accepted"]:
        detail["rejection_reasons"] = ["legacy-disagreement-guard-rejected"]
        return original, legacy, detail
    if support_interval is None:
        detail["rejection_reasons"] = ["observed-support-unavailable"]
        return original, legacy, detail
    if (type(support_interval) not in (list, tuple) or len(support_interval) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in support_interval)
            or not 0 <= support_interval[0] < support_interval[1]):
        raise ValueError("supported-path-support-interval-invalid")
    low, high = map(float, support_interval)
    detail["center_support_forward_m"] = [low, high]
    def outside(x):
        return max(low - x, x - high, 0.)
    distances = [max(outside(raw[0]), outside(proposed[0]))
        for raw, proposed in zip(original[:2], legacy_selected[:2])]
    weights = [max(0., 1. - distance / config.maximum_extrapolation_m) for distance in distances]
    detail.update(outside_support_m=distances, support_weights=weights)
    changes = [[weight * (proposed[axis] - raw[axis]) for axis in range(2)]
               for raw, proposed, weight in zip(original[:2], legacy_selected[:2], weights)]
    if not any(abs(value) > 1e-12 for change in changes for value in change):
        detail["rejection_reasons"] = ["no-supported-proposal-change"]
        detail["shape_change"] = _shape_change(original, original)
        return original, legacy, detail
    def scaled(amount):
        result = copy.deepcopy(original)
        for index in range(2):
            result[index] = [original[index][axis] + amount * changes[index][axis] for axis in range(2)]
        return result
    def admissible(amount):
        candidate = scaled(amount)
        try:
            validate_waypoint_points(candidate)
            change = _shape_change(original, candidate)
        except ValueError:
            return False
        return (change["maximum_segment_heading_change_rad"] <= config.maximum_segment_heading_change_rad
            and change["maximum_signed_curvature_change_per_m"] <= config.maximum_curvature_change_per_m)
    if admissible(1.):
        scale = 1.
    else:
        # Fixed-iteration bounded search avoids discontinuous 1/.5/.25 choices.
        # The final actual path is always checked, not inferred from a scale.
        low_scale, high_scale = 0., 1.
        for _ in range(config.scale_bisection_steps):
            middle = (low_scale + high_scale) / 2
            if admissible(middle):
                low_scale = middle
            else:
                high_scale = middle
        scale = low_scale
    selected = scaled(scale)
    validate_waypoint_points(selected)
    if not admissible(scale):
        raise ValueError("supported-path-final-shape-not-admissible")
    detail.update(shape_scale=scale, accepted=any(math.dist(a, b) > 1e-12
        for a, b in zip(selected[:2], original[:2])), shape_change=_shape_change(original, selected))
    if scale < 1.:
        detail["attenuation_reason"] = "segment-heading-or-signed-curvature-bound"
    if not detail["accepted"]:
        detail["rejection_reasons"] = ["shape-constraint-retained-raw-path"]
    return selected, legacy, detail


class SupportedRGBPathCorrection:
    """Reuse the exact RGB proposal; derive support from its selected paint."""
    def __init__(self, base=None, config=None):
        self.base = RGBPathCorrection() if base is None else base
        self.config = SupportedPathConfig() if config is None else config
        if not isinstance(self.config, SupportedPathConfig) or self.base.config.sha256 != PathCorrectionConfig().sha256:
            raise ValueError("supported-path-original-base-config-required")
        self.baseline = self.base.baseline
        self._base_sha256, self._baseline_sha256 = self.base.config.sha256, self.baseline.sha256

    def _support(self, rgb, detail, *, width, height):
        if detail.get("available") is not True:
            return None, {"source": "none", "reason": "legacy-proposal-unavailable"}
        method = detail.get("method")
        half_width = self.baseline.nominal_lane_width_m / 2
        if method == "local-near-boundary":
            local = detail.get("local_correction", {})
            low, high = local.get("observed_minimum_forward_m"), local.get("observed_maximum_forward_m")
            coefficients = local.get("polynomial_coefficients")
            if (type(low) not in (int, float) or type(high) not in (int, float)
                    or not math.isfinite(low) or not math.isfinite(high) or not 0 <= low < high
                    or type(coefficients) is not list or len(coefficients) not in (2, 3)
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in coefficients)):
                return None, {"source": "local-near-boundary", "reason": "observed-support-metadata-invalid"}
            x = np.linspace(low, high, 32)
            slope = np.polynomial.polynomial.polyval(x, np.polynomial.polynomial.polyder(coefficients))
            center_x = x - half_width * slope / np.sqrt(1 + slope * slope)
            if not np.isfinite(center_x).all() or np.any(np.diff(center_x) <= 0) or center_x[0] < 0:
                return None, {"source": "local-near-boundary", "reason": "observed-center-support-not-forward"}
            interval = [float(center_x[0]), float(center_x[-1])]
            return interval, {"source": "observed-local-polynomial-center-support",
                "observed_boundary_forward_m": [float(low), float(high)], "center_support_forward_m": interval}
        if method != "paint-normal-and-heading-alignment-with-arc-resampling":
            return None, {"source": "none", "reason": "unrecognized-proposal-method"}
        observed = detail.get("observed_paint_center_ego_m")
        if (type(observed) is not list or len(observed) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in observed)):
            return None, {"source": "paint-component", "reason": "selected-component-center-invalid"}
        # A center measurement alone is not treated as an observed span. Rebuild
        # the actual selected RGB component using the unchanged proposal mask.
        image = _rgb_array(rgb, width, height, self.baseline)
        cfg, grid = self.base.config, self.base.grid
        yellow, _ = _marking_mask(image, self.baseline, grid)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(yellow.astype(np.uint8), 8)
        matches = []
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < cfg.minimum_pixels:
                continue
            mask = ((labels == label) & (grid[0] >= cfg.minimum_forward_m)
                & (grid[0] <= cfg.maximum_forward_m) & (np.abs(grid[1]) <= cfg.maximum_absolute_boundary_y_m))
            paint = np.column_stack((grid[0][mask], grid[1][mask]))
            if len(paint) < cfg.minimum_pixels or np.linalg.norm(np.median(paint, axis=0) - observed) > 1e-7:
                continue
            values, vectors = np.linalg.eigh(np.cov(paint.T))
            if (values[-1] < cfg.minimum_major_variance_m2 or values[0] > cfg.maximum_minor_variance_m2
                    or values[-1] < cfg.minimum_variance_ratio * values[0]):
                continue
            tangent = vectors[:, -1]
            if tangent[0] < 0:
                tangent = -tangent
            normal = np.array([-tangent[1], tangent[0]])
            center_paint = paint + half_width * normal
            interval = [float(center_paint[:, 0].min()), float(center_paint[:, 0].max())]
            if 0 <= interval[0] < interval[1]:
                matches.append((interval, len(paint)))
        if len(matches) != 1:
            return None, {"source": "paint-component", "reason": "selected-component-not-uniquely-reidentified"}
        interval, pixels = matches[0]
        return interval, {"source": "reidentified-observed-RGB-component-center-support",
            "center_support_forward_m": interval, "paint_pixels": pixels,
            "nominal_half_lane_width_m": half_width}

    def apply(self, rgb, points, *, width=640, height=360):
        if self.base.config.sha256 != self._base_sha256 or self.baseline.sha256 != self._baseline_sha256:
            raise ValueError("supported-path-base-config-changed")
        original = copy.deepcopy(points)
        proposed, legacy_detail = self.base.apply(rgb, original, width=width, height=height)
        if original != points:
            raise ValueError("supported-path-base-mutated-input")
        support, source = self._support(rgb, legacy_detail, width=width, height=height)
        selected, legacy_guard, supported = support_guard_path_proposal(points, proposed, support, self.config)
        supported.update(observed_support=source, proposed_waypoints_ego_m=copy.deepcopy(proposed),
            base_path_config_sha256=self._base_sha256)
        return selected, {"path_correction": legacy_detail, "guarded_path_correction": legacy_guard,
                          "supported_path_correction": supported}


class SupportedPathPredictor:
    """Run the unchanged timed RGB/model forward once with the new corrector."""
    def __init__(self, base, config=None):
        if (getattr(base, "correction", None) != "path" or not callable(getattr(base, "timer", None))
                or not hasattr(base, "raw") or base.raw.identity.get("variant") != "hybrid_geometry"):
            raise ValueError("supported-path-runtime-RGB-base-required")
        self.base, self.config = base, SupportedPathConfig() if config is None else config
        self.raw = self.raw_predictor = base.raw
        self.corrector = SupportedRGBPathCorrection(base.corrector, self.config)
        self.correction = "supported_path"
        self._identity = copy.deepcopy(base.identity)
        if self._identity.get("correction_config_sha256") != self.corrector.base.config.sha256:
            raise ValueError("supported-path-base-identity-config-mismatch")
        self._identity.update(variant=self._identity["variant"] + "_supported", correction="supported_path",
            supported_version=VERSION, supported_config=self.config.to_dict(),
            supported_config_sha256=self.config.sha256, supported_source_sha256=supported_source_sha256(),
            legacy_guard_config=GuardConfig().to_dict(), legacy_guard_config_sha256=GuardConfig().sha256,
            base_correction_variant=base.identity["variant"])

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
                raise ValueError("supported-path-composite-diagnostic-invalid")
            prediction["diagnostics"].update(detail)
        return prediction

    def predict(self, rgb, *, width=640, height=360):
        return self.predict_profiled(rgb, self.timer(), width=width, height=height)
