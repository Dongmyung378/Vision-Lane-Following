"""Condition RGB proposal uncertainty on the learned path's turn severity.

The original continuous correction budget and shape limits remain unchanged.
Only the current RGB image, learned path and fixed calibration enter inference.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import hashlib
import math
from pathlib import Path
from types import FunctionType

from . import adaptive_path_correction as original
from .contracts import canonical_sha256

VERSION = "vlf-sharp-turn-confidence-path-v1"


def _isolated(function, **dependencies):
    """Copy a function namespace so the original module is never mutated."""
    if not set(dependencies) <= set(function.__globals__):
        raise ValueError("sharp-turn-unknown-original-global")
    result = FunctionType(function.__code__, dict(function.__globals__, **dependencies),
        function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = copy.copy(function.__kwdefaults__)
    return result


@dataclass(frozen=True)
class SharpTurnConfig:
    version: str = VERSION
    fade_start_rad: float = .3
    fade_end_rad: float = .7
    adaptive_config: original.AdaptivePathConfig = field(default_factory=original.AdaptivePathConfig)

    def __post_init__(self):
        if self.version != VERSION or not isinstance(self.adaptive_config, original.AdaptivePathConfig):
            raise ValueError("sharp-turn-config-version-or-base")
        if any(type(v) not in (int, float) or not math.isfinite(v)
               for v in (self.fade_start_rad, self.fade_end_rad)):
            raise ValueError("sharp-turn-heading-nonfinite")
        if not 0. < self.fade_start_rad < self.fade_end_rad < math.pi / 2:
            raise ValueError("sharp-turn-heading-order")

    def to_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {"version", "fade_start_rad", "fade_end_rad", "adaptive_config"}:
            raise ValueError("sharp-turn-config-fields")
        return cls(value["version"], value["fade_start_rad"], value["fade_end_rad"],
            original.adaptive_config_from_dict(value["adaptive_config"]))

    fromdict = from_dict


SharpTurnPathConfig = SharpTurnConfig


def sharp_turn_config_from_dict(value):
    return SharpTurnConfig.from_dict(value)


def sharp_turn_source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def quality_with_sharp_turn(points, proposed, support_span_m, residual_p95_m, adaptive_config, *, config):
    quality = original.proposal_quality(points, proposed, support_span_m, residual_p95_m, adaptive_config)
    dx, dy = points[1][0] - points[0][0], points[1][1] - points[0][1]
    if math.hypot(dx, dy) <= 1e-12:
        raise ValueError("sharp-turn-degenerate-first-segment")
    heading = math.atan2(dy, dx)
    phase = max(0., min(1., (abs(heading) - config.fade_start_rad) / (config.fade_end_rad - config.fade_start_rad)))
    severity = phase * phase * (3. - 2. * phase)
    old = quality["geometry_confidence"]
    new = old if severity == 1. else 1. if severity == 0. else 1. - severity * (1. - old)
    quality.update(unconditional_geometry_confidence=old, geometry_confidence=new,
        raw_first_segment_heading_rad=heading, sharp_turn_severity=severity,
        sharp_turn_fade_start_rad=config.fade_start_rad, sharp_turn_fade_end_rad=config.fade_end_rad)
    return quality


def _guard(points, proposed, support_span_m, residual_p95_m, config):
    def quality(*args):
        return quality_with_sharp_turn(*args, config=config)
    function = _isolated(original.adaptive_guard_path_proposal, proposal_quality=quality)
    selected, guard, detail = function(points, proposed, support_span_m, residual_p95_m, config.adaptive_config)
    for diagnostics in (guard, detail):
        diagnostics.update(version=VERSION, config_sha256=config.sha256,
            base_adaptive_config_sha256=config.adaptive_config.sha256,
            confidence_scope="learned-first-segment-turn-severity")
    return selected, guard, detail


def sharp_turn_guard_path_proposal(points, proposed, support_span_m, residual_p95_m=None, *, fade_start_rad, fade_end_rad):
    return _guard(points, proposed, support_span_m, residual_p95_m,
        SharpTurnConfig(fade_start_rad=fade_start_rad, fade_end_rad=fade_end_rad))


class SharpTurnRGBPathCorrection:
    def __init__(self, base=None, config=None):
        self.config = SharpTurnConfig() if config is None else config
        if not isinstance(self.config, SharpTurnConfig):
            raise ValueError("sharp-turn-config-required")
        self._config_sha256 = self.config.sha256
        self._adaptive = original.AdaptiveRGBPathCorrection(base, self.config.adaptive_config)
        self.base, self.baseline = self._adaptive.base, self._adaptive.baseline
        fixed_config = self.config
        def guard(points, proposed, support_span_m, residual_p95_m=None, config=None):
            if config.sha256 != fixed_config.adaptive_config.sha256:
                raise ValueError("sharp-turn-base-config-changed")
            return _guard(points, proposed, support_span_m, residual_p95_m, fixed_config)
        self._apply = _isolated(original.AdaptiveRGBPathCorrection.apply, adaptive_guard_path_proposal=guard)

    def apply(self, rgb, points, *, width=640, height=360):
        if (self.config.sha256 != self._config_sha256
                or self.config.adaptive_config.sha256 != self._adaptive.config.sha256):
            raise ValueError("sharp-turn-config-changed")
        return self._apply(self._adaptive, rgb, points, width=width, height=height)


class SharpTurnPathPredictor:
    """Reuse the original timed preprocessing and model forward exactly once."""
    def __init__(self, base, config=None):
        self.config = SharpTurnConfig() if config is None else config
        if not isinstance(self.config, SharpTurnConfig):
            raise ValueError("sharp-turn-config-required")
        candidate = original.AdaptivePathPredictor(base, self.config.adaptive_config)
        self.base, self.raw, self.raw_predictor = base, candidate.raw, candidate.raw_predictor
        self.corrector = SharpTurnRGBPathCorrection(base.corrector, self.config)
        self.correction = "supported_path"
        self._identity = candidate.identity
        self._identity.update(variant=base.identity["variant"] + "_sharp_turn",
            sharp_turn_version=VERSION, sharp_turn_config=self.config.to_dict(),
            sharp_turn_config_sha256=self.config.sha256, sharp_turn_source_sha256=sharp_turn_source_sha256())

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def timer(self):
        return self.base.timer()

    def predict_profiled(self, rgb, timer, *, width=640, height=360):
        if self.config.sha256 != self.corrector.config.sha256:
            raise ValueError("sharp-turn-predictor-config-changed")
        return original.AdaptivePathPredictor.predict_profiled(self, rgb, timer, width=width, height=height)

    def predict(self, rgb, *, width=640, height=360):
        return self.predict_profiled(rgb, self.timer(), width=width, height=height)
