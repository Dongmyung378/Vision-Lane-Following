"""Bounded near-path feedback from transported RGB lane observations.

Visual memory is updated once for every prediction. Recorded-input correction
replay reuses that same observation and recomputes its path-dependent proposal.
The original pure-pursuit controller, output budgets and shape guard are reused.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import hashlib
import math
from pathlib import Path

from .adaptive_path_correction import AdaptivePathPredictor
from .baseline import _rgb_array
from .contracts import canonical_sha256
from .learning_data import validate_waypoint_points
from .lane_pose_observer import LanePoseObserverConfig, RGBTemporalLanePoseObserver
from .sharp_turn_path_correction import SharpTurnConfig, SharpTurnPathPredictor, SharpTurnRGBPathCorrection, _guard


VERSION = "vlf-temporal-lane-feedback-path-v1"
TEMPORAL_INPUT = "current-and-past-RGB-and-fixed-nominal-calibration"


@dataclass(frozen=True)
class LaneFeedbackConfig:
    version: str = VERSION
    observer_config: LanePoseObserverConfig = field(default_factory=LanePoseObserverConfig)
    sharp_turn_config: SharpTurnConfig = field(default_factory=lambda:
        SharpTurnConfig(fade_start_rad=.5, fade_end_rad=.7))

    def __post_init__(self):
        if (self.version != VERSION or type(self.observer_config) is not LanePoseObserverConfig
                or type(self.sharp_turn_config) is not SharpTurnConfig):
            raise ValueError("lane-feedback-config-invalid")

    def to_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {"version", "observer_config", "sharp_turn_config"}:
            raise ValueError("lane-feedback-config-fields")
        return cls(value["version"], LanePoseObserverConfig.from_dict(value["observer_config"]),
            SharpTurnConfig.from_dict(value["sharp_turn_config"]))

    fromdict = from_dict


def lane_feedback_source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _finite(value, code, *, low=None, high=None):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except (OverflowError, TypeError):
        valid = False
    if not valid:
        raise ValueError(code)
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(code)
    return float(value)


class LaneFeedbackRGBPathCorrection:
    """Episode-scoped visual feedback with exact same-observation reentry.

    Call ``begin_frame`` once before each new image observation, including when
    two consecutive frames have identical pixels. ``apply`` without a new frame
    reuses the current RGB observation and computes a new proposal for its
    supplied learned points. This is the strict recorded-input replay operation.
    """

    def __init__(self, base=None, config=None, *, observer=None):
        self.config = LaneFeedbackConfig() if config is None else config
        if type(self.config) is not LaneFeedbackConfig:
            raise ValueError("lane-feedback-config-required")
        self._config_sha256 = self.config.sha256
        self._fallback = SharpTurnRGBPathCorrection(base, self.config.sharp_turn_config)
        self.base, self.baseline = self._fallback.base, self._fallback.baseline
        self.observer = (RGBTemporalLanePoseObserver(self.config.observer_config)
            if observer is None else observer)
        if not all(callable(getattr(self.observer, key, None)) for key in ("reset", "observe", "estimate")):
            raise ValueError("lane-feedback-observer-contract")
        if getattr(self.observer, "config", None) != self.config.observer_config:
            raise ValueError("lane-feedback-observer-config-mismatch")
        self.reset()

    def reset(self):
        self.observer.reset()
        self._pending_frame = False
        self._observed_rgb_sha256 = None
        self._observation_count = 0

    def begin_frame(self):
        if self._pending_frame:
            raise ValueError("lane-feedback-frame-already-pending")
        self._pending_frame = True

    def cancel_pending_frame(self):
        """Clear a frame that failed before the timed correction stage."""
        self._pending_frame = False

    def _observe(self, rgb, *, width, height):
        image = _rgb_array(rgb, width, height, self.baseline)
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        if self._pending_frame:
            self.observer.observe(image, width=width, height=height)
            self._observed_rgb_sha256 = digest
            self._observation_count += 1
            self._pending_frame = False
        elif digest != self._observed_rgb_sha256:
            raise ValueError("lane-feedback-new-image-requires-begin-frame")
        return digest

    def apply(self, rgb, points, *, width=640, height=360):
        if (self.config.sha256 != self._config_sha256
                or self.observer.config != self.config.observer_config):
            raise ValueError("lane-feedback-config-changed")
        validate_waypoint_points(points)
        original = copy.deepcopy(points)
        digest = self._observe(rgb, width=width, height=height)
        baseline, baseline_detail = self._fallback.apply(rgb, points, width=width, height=height)
        observation = self.observer.estimate(copy.deepcopy(points))
        if points != original:
            raise ValueError("lane-feedback-input-mutated")
        if type(observation) is not dict or type(observation.get("available")) is not bool:
            raise ValueError("lane-feedback-estimate-schema")
        confidence = (_finite(observation.get("confidence"), "lane-feedback-confidence", low=0., high=1.)
            if observation["available"] else 0.)
        feedback = {"version": VERSION, "config_sha256": self.config.sha256,
            "input": TEMPORAL_INPUT, "observation_count": self._observation_count,
            "observed_rgb_sha256": digest, "requires_episode_reset": True,
            "confidence": confidence, "temporal_observation": copy.deepcopy(observation),
            "blend_rule": "observer-confidence-between-fallback-and-temporal-proposals",
            "temporal_proposal_changes_only_near_y": True,
            "controller_changed": False, "distant_points_unchanged": True}
        if confidence == 0.:
            details = copy.deepcopy(baseline_detail)
            details["supported_path_correction"]["lane_feedback"] = feedback
            return baseline, details
        proposal = observation.get("proposed_waypoints_ego_m")
        validate_waypoint_points(proposal)
        if (proposal[2:] != points[2:] or any(proposal[i][0] != points[i][0] for i in range(2))):
            raise ValueError("lane-feedback-observer-changed-longitudinal-or-far-points")
        lo = _finite(observation.get("observed_boundary_min_x_m"), "lane-feedback-support-min")
        hi = _finite(observation.get("observed_boundary_max_x_m"), "lane-feedback-support-max")
        span = hi-lo
        if span <= 0.:
            raise ValueError("lane-feedback-observed-span")
        residual = _finite(observation.get("normal_fit_residual_p95_m"), "lane-feedback-residual", low=0.)
        prior = baseline_detail["supported_path_correction"]
        prior_proposal = prior["proposed_waypoints_ego_m"]
        validate_waypoint_points(prior_proposal)
        if prior_proposal[2:] != points[2:]:
            raise ValueError("lane-feedback-fallback-changed-far-points")
        prior_support = prior.get("center_support_forward_m")
        prior_span = prior_support[1]-prior_support[0] if prior_support is not None else 1.
        prior_residual = (baseline_detail["path_correction"]["local_correction"].get("residual_p95_m")
            if baseline_detail["path_correction"].get("method") == "local-near-boundary" else None)
        mixed = copy.deepcopy(points)
        for index in range(2):
            for axis in range(2):
                mixed[index][axis] = ((1.-confidence)*prior_proposal[index][axis] +
                    confidence*proposal[index][axis])
        mixed_span = (1.-confidence)*prior_span + confidence*span
        mixed_residual = (1.-confidence)*(prior_residual or 0.) + confidence*residual
        selected, guard, detail = _guard(points, mixed, mixed_span, mixed_residual, self.config.sharp_turn_config)
        for item in (guard, detail):
            item.update(version=VERSION, config_sha256=self.config.sha256,
                temporal_state="episode-local-transported-RGB-paint", requires_episode_reset=True)
        feedback.update(fallback_proposal_waypoints_ego_m=copy.deepcopy(prior_proposal),
            temporal_proposal_waypoints_ego_m=copy.deepcopy(proposal),
            mixed_observed_support_span_m=mixed_span, mixed_fit_residual_p95_m=mixed_residual)
        detail.update(lane_feedback=feedback, proposed_waypoints_ego_m=copy.deepcopy(mixed),
            center_support_forward_m=[lo, hi], base_path_config_sha256=self.base.config.sha256)
        if points != original or selected[2:] != original[2:]:
            raise ValueError("lane-feedback-input-mutated-or-far-points-changed")
        return selected, {"path_correction": baseline_detail["path_correction"],
            "guarded_path_correction": guard, "supported_path_correction": detail}


class LaneFeedbackPathPredictor:
    """Reuse the timed preprocessing and model forward exactly once per image."""

    def __init__(self, base, config=None):
        self.config = LaneFeedbackConfig() if config is None else config
        if type(self.config) is not LaneFeedbackConfig:
            raise ValueError("lane-feedback-config-required")
        previous = SharpTurnPathPredictor(base, self.config.sharp_turn_config)
        self.base, self.raw, self.raw_predictor = base, previous.raw, previous.raw_predictor
        self.corrector = LaneFeedbackRGBPathCorrection(base.corrector, self.config)
        self.correction = "supported_path"
        self._identity = previous.identity
        from . import lane_pose_observer
        observer_source = hashlib.sha256(Path(lane_pose_observer.__file__).read_bytes()).hexdigest()
        self._identity.update(variant=base.identity["variant"]+"_temporal_lane_feedback",
            lane_feedback_version=VERSION, lane_feedback_config=self.config.to_dict(),
            lane_feedback_config_sha256=self.config.sha256,
            lane_feedback_source_sha256=lane_feedback_source_sha256(),
            lane_pose_observer_source_sha256=observer_source, requires_episode_reset=True,
            temporal_input=TEMPORAL_INPUT)

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def reset(self):
        self.corrector.reset()

    def timer(self):
        return self.base.timer()

    def predict_profiled(self, rgb, timer, *, width=640, height=360):
        if self.config.sha256 != self.corrector.config.sha256:
            raise ValueError("lane-feedback-predictor-config-changed")
        self.corrector.begin_frame()
        try:
            return AdaptivePathPredictor.predict_profiled(self, rgb, timer, width=width, height=height)
        finally:
            self.corrector.cancel_pending_frame()

    def predict(self, rgb, *, width=640, height=360):
        return self.predict_profiled(rgb, self.timer(), width=width, height=height)
