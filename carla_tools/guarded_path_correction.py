"""Accept RGB paint proposals only when consistent with the current RGB model.

The guard receives current learned points and a current RGB correction proposal.
It has no route, ground truth, pose, speed, previous path or simulator input.
Rejecting a proposal returns the current learned path unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
import hashlib
import math
from pathlib import Path

from .contracts import canonical_sha256
from .learning_data import validate_waypoint_points
from .path_correction import PathCorrectionConfig

VERSION = "vlf-current-rgb-path-consistency-v1"


@dataclass(frozen=True)
class GuardConfig:
    schema_version: int = 1
    version: str = VERSION
    maximum_first_point_change_m: float = .5
    maximum_near_mean_change_m: float = .5

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1 or self.version != VERSION:
            raise ValueError("guarded-path-version-invalid")
        for name in ("maximum_first_point_change_m", "maximum_near_mean_change_m"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not .05 <= value <= 1.:
                raise ValueError("guarded-path-disagreement-bound-invalid")

    def to_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return canonical_sha256(self.to_dict())


GuardedPathConfig = GuardConfig


def guard_config_from_dict(value):
    if type(value) is not dict or set(value) != {field.name for field in fields(GuardConfig)}:
        raise ValueError("guarded-path-config-fields-invalid")
    return GuardConfig(**value)


def guard_source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def guard_path_proposal(learned_points, proposed_points, config=None):
    """Gate only near-point disagreement, retaining the five current far points."""
    config = GuardConfig() if config is None else config
    if not isinstance(config, GuardConfig):
        raise ValueError("guarded-path-config-required")
    validate_waypoint_points(learned_points)
    validate_waypoint_points(proposed_points)
    if any(abs(first - second) > 1e-8 for learned, proposed in zip(learned_points[2:], proposed_points[2:])
           for first, second in zip(learned, proposed)):
        raise ValueError("guarded-path-distant-proposal-changed")
    changes = [math.dist(first, second) for first, second in zip(learned_points[:2], proposed_points[:2])]
    near_mean = math.fsum(changes) / 2
    reasons = []
    if changes[0] > config.maximum_first_point_change_m:
        reasons.append("first-point-disagreement")
    if near_mean > config.maximum_near_mean_change_m:
        reasons.append("near-mean-disagreement")
    accepted = not reasons
    decision = {"version": VERSION, "config_sha256": config.sha256,
        "input": "current-RGB-learned-and-current-RGB-proposed-waypoints",
        "temporal_state": "none", "accepted": accepted, "rejection_reasons": reasons,
        "near_point_change_m": changes, "near_mean_change_m": near_mean,
        "selected_source": "RGB-path-proposal" if accepted else "current-learned-RGB-path",
        "distant_points_unchanged": True}
    selected = copy.deepcopy(learned_points)
    if accepted:
        selected[:2] = copy.deepcopy(proposed_points[:2])
    return selected, decision


def path_preprocessing_config():
    return {"correction": "path", "correction_config_sha256": PathCorrectionConfig().sha256}


def guarded_preprocessing_config(config=None):
    config = GuardConfig() if config is None else config
    if not isinstance(config, GuardConfig):
        raise ValueError("guarded-path-config-required")
    return dict(path_preprocessing_config(), correction="guarded_path", guard_config=config.to_dict(),
                guard_config_sha256=config.sha256)


def validate_preprocessing_config(value):
    if type(value) is not dict:
        raise ValueError("guarded-path-preprocessing-object-required")
    if value.get("correction") == "path":
        expected = path_preprocessing_config()
    elif value.get("correction") == "guarded_path":
        config = guard_config_from_dict(value.get("guard_config"))
        expected = guarded_preprocessing_config(config)
    else:
        raise ValueError("guarded-path-preprocessing-mode-invalid")
    if value != expected or value.keys() != expected.keys():
        raise ValueError("guarded-path-preprocessing-contract-mismatch")
    return copy.deepcopy(expected)


class GuardedRGBPathCorrection:
    def __init__(self, base, config=None):
        self.config = GuardConfig() if config is None else config
        if not isinstance(self.config, GuardConfig) or base.config.sha256 != PathCorrectionConfig().sha256:
            raise ValueError("guarded-path-original-base-correction-required")
        self.base, self.baseline = base, base.baseline
        self._base_sha256, self._baseline_sha256 = base.config.sha256, base.baseline.sha256

    def apply(self, rgb, points, *, width=640, height=360):
        if self.base.config.sha256 != self._base_sha256 or self.base.baseline.sha256 != self._baseline_sha256:
            raise ValueError("guarded-path-base-config-changed")
        original = copy.deepcopy(points)
        proposed, legacy_detail = self.base.apply(rgb, original, width=width, height=height)
        if original != points:
            raise ValueError("guarded-path-base-mutated-learned-points")
        selected, decision = guard_path_proposal(points, proposed, self.config)
        guard_detail = dict(decision, base_path_config_sha256=self._base_sha256,
                            proposed_waypoints_ego_m=copy.deepcopy(proposed))
        return selected, {"path_correction": legacy_detail, "guarded_path_correction": guard_detail}


class GuardedPathPredictor:
    """Reuse legacy timed preprocessing/forward exactly once with a guarded RGB step."""
    def __init__(self, base, config=None):
        if (getattr(base, "correction", None) != "path" or not callable(getattr(base, "timer", None))
                or not hasattr(base, "raw") or base.raw.identity.get("variant") != "hybrid_geometry"):
            raise ValueError("guarded-path-runtime-RGB-base-required")
        self.base, self.config = base, GuardConfig() if config is None else config
        self.raw = self.raw_predictor = base.raw
        self.corrector = GuardedRGBPathCorrection(base.corrector, self.config)
        self.correction = "guarded_path"
        self._identity = copy.deepcopy(base.identity)
        if self._identity.get("correction_config_sha256") != self.corrector.base.config.sha256:
            raise ValueError("guarded-path-base-identity-config-mismatch")
        self._identity.update(variant=self._identity["variant"] + "_guarded", correction="guarded_path",
            guard_version=VERSION, guard_config=self.config.to_dict(), guard_config_sha256=self.config.sha256,
            guard_source_sha256=guard_source_sha256(), base_correction_variant=base.identity["variant"])

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def timer(self):
        return self.base.timer()

    def predict_profiled(self, rgb, timer, *, width=640, height=360):
        from .runtime_driving import RuntimeCheckpointPredictor
        # The legacy method reads this object's raw/corrector fields, so its
        # original preprocessing and forward run once. Guard work is included
        # inside the existing near_correction_ms measurement.
        prediction = RuntimeCheckpointPredictor.predict_profiled(self, rgb, timer, width=width, height=height)
        detail = prediction.get("diagnostics", {}).get("path_correction")
        if detail is not None:
            if type(detail) is not dict or set(detail) != {"path_correction", "guarded_path_correction"}:
                raise ValueError("guarded-path-composite-diagnostic-invalid")
            prediction["diagnostics"].update(detail)
        return prediction

    def predict(self, rgb, *, width=640, height=360):
        return self.predict_profiled(rgb, self.timer(), width=width, height=height)


def build_guarded_predictor(base, config=None):
    return GuardedPathPredictor(base, config)


def wrap_preprocessing(base, value):
    preprocessing = validate_preprocessing_config(value)
    if preprocessing["correction"] == "path":
        if getattr(base, "correction", None) != "path" or base.corrector.config.sha256 != preprocessing["correction_config_sha256"]:
            raise ValueError("guarded-path-original-base-correction-required")
        return base
    return GuardedPathPredictor(base, guard_config_from_dict(preprocessing["guard_config"]))


def _same(first, second):
    if type(first) is dict or type(second) is dict:
        return type(first) is dict and type(second) is dict and first.keys() == second.keys() and all(
            _same(first[key], second[key]) for key in first)
    if type(first) is list or type(second) is list:
        return type(first) is list and type(second) is list and len(first) == len(second) and all(
            _same(a, b) for a, b in zip(first, second))
    if type(first) in (int, float) and type(second) in (int, float):
        return math.isfinite(first) and math.isfinite(second) and math.isclose(first, second, rel_tol=0., abs_tol=1e-8)
    return type(first) is type(second) and first == second


def validate_recorded_guarded_correction(predictor, rgb, observed, *, width=640, height=360):
    """Replay the recorded current learned points; no new model inference or GT."""
    if not isinstance(predictor, GuardedPathPredictor) or type(observed) is not dict or observed.get("status") != "PASS":
        raise ValueError("guarded-path-replay-success-and-predictor-required")
    selected, detail = predictor.corrector.apply(rgb, observed["learned_waypoints_ego_m"], width=width, height=height)
    if (not _same(selected, observed["waypoints_ego_m"])
            or not _same(detail["path_correction"], observed["diagnostics"].get("path_correction"))
            or not _same(detail["guarded_path_correction"], observed["diagnostics"].get("guarded_path_correction"))):
        raise ValueError("guarded-path-recorded-correction-or-diagnostic-mismatch")
    return {"status": "PASS", "guard_config_sha256": predictor.config.sha256,
            "base_path_config_sha256": predictor.corrector.base.config.sha256}
