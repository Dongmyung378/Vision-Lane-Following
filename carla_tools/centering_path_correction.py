"""Curvature-aware RGB alignment with bounded longitudinal phase confidence.

A local osculating arc extends the learned path through the camera blind zone.
For oblique near paths, RGB alignment changes lateral position while retaining
more of the learned longitudinal phase. The original RGB evidence, continuous
displacement budgets and path-shape limits remain in force.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import hashlib
import math
from pathlib import Path

import numpy as np

from .contracts import canonical_sha256
from .learning_data import validate_waypoint_points
from .path_correction import RGBPathCorrection
from .sharp_turn_path_correction import (SharpTurnConfig, SharpTurnPathPredictor,
    SharpTurnRGBPathCorrection, _isolated)

VERSION = "vlf-curvature-aware-centering-path-v1"


@dataclass(frozen=True)
class CenteringPathConfig:
    version: str = VERSION
    blindzone_extension: str = "first-three-point-osculating-arc"
    longitudinal_confidence: str = "one-minus-learned-turn-severity"
    sharp_turn_config: SharpTurnConfig = field(default_factory=lambda:
        SharpTurnConfig(fade_start_rad=.5, fade_end_rad=.7))

    def __post_init__(self):
        if (self.version != VERSION
                or self.blindzone_extension != "first-three-point-osculating-arc"
                or self.longitudinal_confidence != "one-minus-learned-turn-severity"
                or not isinstance(self.sharp_turn_config, SharpTurnConfig)):
            raise ValueError("centering-config-invalid")

    def to_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {
                "version", "blindzone_extension", "longitudinal_confidence", "sharp_turn_config"}:
            raise ValueError("centering-config-fields")
        return cls(value["version"], value["blindzone_extension"], value["longitudinal_confidence"],
            SharpTurnConfig.from_dict(value["sharp_turn_config"]))

    fromdict = from_dict


def centering_config_from_dict(value):
    return CenteringPathConfig.from_dict(value)


def centering_source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def blindzone_arc_anchors(points):
    """Continue near-path curvature backwards without imposing an ego origin.

    Sinc integration has a continuous straight-line limit and avoids subtracting
    the large circle centres produced by nearly straight paths.
    """
    original = np.asarray(points, dtype=float)
    if original.shape != (5, 2) or not np.isfinite(original).all():
        raise ValueError("centering-invalid-arc-points")
    a, b = original[1] - original[0], original[2] - original[0]
    chord = np.linalg.norm(a)
    denominator = chord * np.linalg.norm(b) * np.linalg.norm(b - a)
    if denominator <= 1e-12:
        raise ValueError("centering-degenerate-near-shape")
    curvature = 2. * (a[0] * b[1] - a[1] * b[0]) / denominator
    turn = 2. * math.asin(float(np.clip(curvature * chord / 2., -1., 1.)))
    if abs(turn) >= math.pi / 2 or float(a @ (b - a)) <= 0:
        raise ValueError("centering-reversing-near-shape")
    arc_step = chord if curvature == 0. else turn / curvature
    heading = math.atan2(a[1], a[0]) - turn / 2.
    tangent = np.array([math.cos(heading), math.sin(heading)])
    normal = np.array([-tangent[1], tangent[0]])
    tail = []
    for arc in (-2. * arc_step, -arc_step):
        angle = curvature * arc
        along = arc * np.sinc(angle / math.pi)
        across = .5 * curvature * arc * arc * np.sinc(angle / (2. * math.pi)) ** 2
        tail.append(original[0] + along * tangent + across * normal)
    anchors = np.vstack((*tail, original))
    if not np.isfinite(anchors).all():
        raise ValueError("centering-nonfinite-arc")
    return anchors, {"menger_curvature_per_m": float(curvature),
        "backward_arc_step_m": float(arc_step), "first_point_tangent_heading_rad": heading,
        "fixed_horizon_norm_assumption_used": False}


def curvature_aware_learned_curve(points, samples):
    """Use the existing ordered Hermite interpolation on an arc-derived tail."""
    anchors, _ = blindzone_arc_anchors(points)
    tangents = np.empty_like(anchors)
    tangents[0], tangents[-1] = anchors[1] - anchors[0], anchors[-1] - anchors[-2]
    tangents[1:-1] = (anchors[2:] - anchors[:-2]) / 2
    t = np.linspace(0., 1., samples, endpoint=False)
    t2, t3 = t * t, t * t * t
    segments = []
    for index in range(len(anchors) - 1):
        segments.append((2*t3 - 3*t2 + 1)[:, None] * anchors[index]
            + (t3 - 2*t2 + t)[:, None] * tangents[index]
            + (-2*t3 + 3*t2)[:, None] * anchors[index + 1]
            + (t3 - t2)[:, None] * tangents[index + 1])
    center = np.vstack((*segments, anchors[-1:]))
    derivative = np.gradient(center, axis=0)
    magnitude = np.linalg.norm(derivative, axis=1)
    if not np.isfinite(center).all() or np.any(magnitude <= 1e-8):
        raise ValueError("centering-degenerate-curve")
    return center, derivative / magnitude[:, None]


def longitudinal_confidence(points, config):
    dx, dy = points[1][0] - points[0][0], points[1][1] - points[0][1]
    if math.hypot(dx, dy) <= 1e-12:
        raise ValueError("centering-degenerate-first-segment")
    heading = abs(math.atan2(dy, dx))
    start, end = config.sharp_turn_config.fade_start_rad, config.sharp_turn_config.fade_end_rad
    phase = max(0., min(1., (heading - start) / (end - start)))
    return 1. - phase * phase * (3. - 2. * phase)


class CenteringRGBProposal(RGBPathCorrection):
    """Change proposal geometry before the original continuous safety fusion."""
    def __init__(self, base=None, config=None):
        self.centering_config = CenteringPathConfig() if config is None else config
        if not isinstance(self.centering_config, CenteringPathConfig):
            raise ValueError("centering-config-required")
        self._centering_sha256 = self.centering_config.sha256
        if base is None:
            super().__init__()
        else:
            super().__init__(baseline_config=base.baseline, config=base.config)
        self._operation = _isolated(RGBPathCorrection.apply, _learned_curve=curvature_aware_learned_curve)

    def apply(self, rgb, points, *, width=640, height=360):
        if self.centering_config.sha256 != self._centering_sha256:
            raise ValueError("centering-proposal-config-changed")
        proposed, detail = self._operation(self, rgb, points, width=width, height=height)
        confidence = longitudinal_confidence(points, self.centering_config)
        projected = detail["method"] == "paint-normal-and-heading-alignment-with-arc-resampling"
        if projected:
            for index in range(2):
                proposed[index][0] = points[index][0] + confidence * (proposed[index][0] - points[index][0])
        try:
            validate_waypoint_points(proposed)
        except ValueError:
            proposed = copy.deepcopy(points)
            detail.update(available=False, method="unchanged-learned-path",
                rejection="centering-longitudinal-projection-invalid-path")
        try:
            geometry = dict(available=True, **blindzone_arc_anchors(points)[1])
        except ValueError as error:
            geometry = {"available": False, "rejection": str(error)}
        detail.update(centering_version=VERSION, centering_config_sha256=self.centering_config.sha256,
            blindzone_geometry=geometry, longitudinal_confidence=confidence,
            longitudinal_projection_before_fusion=projected)
        return proposed, detail


class CenteringRGBPathCorrection:
    def __init__(self, base=None, config=None):
        self.config = CenteringPathConfig() if config is None else config
        if not isinstance(self.config, CenteringPathConfig):
            raise ValueError("centering-config-required")
        self._config_sha256 = self.config.sha256
        self.base = CenteringRGBProposal(base, self.config)
        self._sharp = SharpTurnRGBPathCorrection(self.base, self.config.sharp_turn_config)
        self.baseline = self.base.baseline

    def apply(self, rgb, points, *, width=640, height=360):
        if (self.config.sha256 != self._config_sha256
                or self._sharp.config.sha256 != self.config.sharp_turn_config.sha256):
            raise ValueError("centering-config-changed")
        selected, detail = self._sharp.apply(rgb, points, width=width, height=height)
        for name in ("guarded_path_correction", "supported_path_correction"):
            detail[name].update(centering_version=VERSION, centering_config_sha256=self.config.sha256)
        return selected, detail


class CenteringPathPredictor:
    """Retain the original single model forward and measured pipeline stages."""
    def __init__(self, base, config=None):
        self.config = CenteringPathConfig() if config is None else config
        if not isinstance(self.config, CenteringPathConfig):
            raise ValueError("centering-config-required")
        original = SharpTurnPathPredictor(base, self.config.sharp_turn_config)
        self.base, self.raw, self.raw_predictor = base, original.raw, original.raw_predictor
        self.corrector = CenteringRGBPathCorrection(base.corrector, self.config)
        self.correction = "supported_path"
        self._identity = original.identity
        self._identity.update(variant=base.identity["variant"] + "_curvature_centering",
            centering_version=VERSION, centering_config=self.config.to_dict(),
            centering_config_sha256=self.config.sha256, centering_source_sha256=centering_source_sha256())

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def timer(self):
        return self.base.timer()

    def predict_profiled(self, rgb, timer, *, width=640, height=360):
        if self.config.sha256 != self.corrector.config.sha256:
            raise ValueError("centering-predictor-config-changed")
        # This method calls RuntimeCheckpointPredictor directly, once.
        from .adaptive_path_correction import AdaptivePathPredictor
        return AdaptivePathPredictor.predict_profiled(self, rgb, timer, width=width, height=height)

    def predict(self, rgb, *, width=640, height=360):
        return self.predict_profiled(rgb, self.timer(), width=width, height=height)
