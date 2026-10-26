"""Bound an auxiliary steering request before the actuator's single filter.

Inputs are normalized steering requests before smoothing and slew limiting.
The caller retains its longitudinal controller and decides whether the
auxiliary estimate is usable. The request bound is not a lane-margin bound.
"""
from dataclasses import dataclass, fields
import math
from numbers import Real


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(name + "-must-be-finite")
    return float(value)


@dataclass(frozen=True)
class BoundedSteeringConfig:
    gain: float = .35
    correction_limit: float = .004
    maximum_steer: float = 1.
    smoothing_alpha: float = 1.
    slew_per_second: float = 2.

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name, _number(getattr(self, field.name), field.name))
        if not 0. <= self.gain <= 1.:
            raise ValueError("gain-out-of-range")
        if not 0. <= self.correction_limit <= 1.:
            raise ValueError("correction-limit-out-of-range")
        if not 0. < self.maximum_steer <= 1.:
            raise ValueError("maximum-steer-out-of-range")
        if not 0. < self.smoothing_alpha <= 1.:
            raise ValueError("smoothing-alpha-out-of-range")
        if self.slew_per_second <= 0.:
            raise ValueError("slew-must-be-positive")


@dataclass(frozen=True)
class SteeringCorrection:
    steer: float
    previous_steer: float
    requested_correction: float
    bounded_request_correction: float
    requested_steer: float
    filtered_steer: float
    correction_limited: bool
    rate_limited: bool


class BoundedSteeringFeedback:
    """Keep one history of actual outputs, including after a fallback."""

    def __init__(self, config=None):
        self.config = BoundedSteeringConfig() if config is None else config
        if not isinstance(self.config, BoundedSteeringConfig):
            raise ValueError("bounded-steering-config-required")
        self.reset()

    def reset(self, steer=0.):
        value = _number(steer, "reset-steer")
        if abs(value) > self.config.maximum_steer:
            raise ValueError("reset-steer-out-of-range")
        self._previous = value

    def step(self, nominal_request, proposal_request=None, *, enabled=False, dt_seconds=.05):
        config = self.config
        nominal = _number(nominal_request, "nominal-request")
        dt = _number(dt_seconds, "dt")
        if type(enabled) is not bool:
            raise ValueError("enabled-must-be-boolean")
        if dt <= 0. or abs(nominal) > config.maximum_steer:
            raise ValueError("invalid-timestep-or-nominal-request")
        requested = 0.
        if enabled:
            proposed = _number(proposal_request, "proposal-request")
            if abs(proposed) > config.maximum_steer:
                raise ValueError("proposal-request-out-of-range")
            requested = config.gain * (proposed - nominal)
        bounded = max(-config.correction_limit, min(config.correction_limit, requested))
        combined = max(-config.maximum_steer, min(config.maximum_steer, nominal + bounded))
        previous = self._previous
        filtered = (combined if config.smoothing_alpha == 1.
                    else previous + config.smoothing_alpha * (combined - previous))
        delta = config.slew_per_second * dt
        actual = max(previous - delta, min(previous + delta, filtered))
        result = SteeringCorrection(actual, previous, requested, bounded, combined, filtered,
            abs(bounded - requested) > 1e-12, abs(actual - filtered) > 1e-12)
        self._previous = actual
        return result
