"""Bounded RGB paint alignment for a learned near path.

A fixed nominal camera and left-yellow lane-width prior are explicit assumptions.
There is no route, pose, measured calibration, or temporal state input. The local
boundary correction remains preferred; when its near support is absent, an
elongated visible dash can align the learned curve without extrapolating that
dash's uncertain curvature into the camera blind zone.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .baseline import _marking_mask, _pixel_ground_grid, _rgb_array, load_baseline_config
from .learning_data import LearningDataError, validate_waypoint_points
from .near_correction import NearCorrectionConfig, RGBNearCorrection

VERSION = "vlf-rgb-near-path-alignment-v1"


@dataclass(frozen=True)
class PathCorrectionConfig:
    version: str = VERSION
    minimum_pixels: int = 24
    minimum_forward_m: float = 3.5
    maximum_forward_m: float = 16.
    maximum_absolute_boundary_y_m: float = 12.
    minimum_major_variance_m2: float = .08
    maximum_minor_variance_m2: float = .04
    minimum_variance_ratio: float = 4.
    maximum_boundary_distance_m: float = 1.5
    maximum_heading_change_rad: float = .3
    maximum_lateral_offset_m: float = 1.5
    maximum_near_change_m: float = 2.5
    maximum_reference_distance_m: float = 3.
    competing_center_gap_m: float = 1.
    competing_correction_difference_m: float = .4
    heading_gain: float = .5
    blend: float = .6
    curve_segment_samples: int = 80

    def __post_init__(self):
        if self.version != VERSION:
            raise ValueError("path-correction-version-invalid")
        for key, value in asdict(self).items():
            if key != "version" and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise ValueError("path-correction-config-invalid")
        if (type(self.minimum_pixels) is not int or self.minimum_pixels < 12
                or type(self.curve_segment_samples) is not int or not 20 <= self.curve_segment_samples <= 200
                or self.minimum_forward_m >= self.maximum_forward_m or self.maximum_forward_m > 20
                or self.maximum_heading_change_rad > .5 or self.maximum_near_change_m > 3
                or self.maximum_boundary_distance_m > 2 or self.maximum_lateral_offset_m > 2
                or self.minimum_variance_ratio < 2 or self.blend > 1 or self.heading_gain > 1):
            raise ValueError("path-correction-config-invalid")

    def to_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _learned_curve(points, samples):
    """Ordered Hermite interpolation, with a declared linear blind-zone tail."""
    original = np.asarray(points, dtype=float)
    anchors = np.vstack((3 * original[0] - 2 * original[1], 2 * original[0] - original[1], original))
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
    magnitudes = np.linalg.norm(derivative, axis=1)
    if not np.isfinite(center).all() or np.any(magnitudes <= 1e-8):
        raise ValueError("path-correction-learned-curve-degenerate")
    return center, derivative / magnitudes[:, None]


class RGBPathCorrection:
    """Prefer local near paint; otherwise align shape to an observed dash."""
    def __init__(self, baseline_config=None, config=None):
        self.baseline = baseline_config or load_baseline_config()
        self.config = config or PathCorrectionConfig()
        if not isinstance(self.config, PathCorrectionConfig):
            raise ValueError("path-correction-config-invalid")
        self.local = RGBNearCorrection(self.baseline, NearCorrectionConfig())
        self.grid = _pixel_ground_grid(self.baseline)
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)

    def apply(self, rgb, points, *, width=640, height=360):
        image = _rgb_array(rgb, width, height, self.baseline)
        validate_waypoint_points(points)
        original = np.asarray(points, dtype=float)
        cfg = self.config
        local_points, local_diagnostics = self.local.apply(image, points, width=width, height=height)
        diagnostics = {"version": VERSION, "config_sha256": cfg.sha256,
                       "baseline_config_sha256": self.baseline.sha256,
                       "local_config_sha256": self.local.config.sha256,
                       "input": "RGB-and-learned-points-and-fixed-nominal-calibration",
                       "boundary": "observed-yellow-left-plus-fixed-width-prior",
                       "temporal_state": "none", "available": False,
                       "method": "unchanged-learned-path", "local_correction": local_diagnostics}
        if np.max(np.abs(np.asarray(local_points) - original)) > 1e-8:
            diagnostics.update(available=True, method="local-near-boundary", corrected_horizons_m=[3., 6.])
            return local_points, diagnostics
        try:
            center, tangents = _learned_curve(original, cfg.curve_segment_samples)
        except ValueError as error:
            diagnostics["rejection"] = str(error)
            return original.tolist(), diagnostics
        normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
        boundary = center - self.baseline.nominal_lane_width_m / 2 * normals
        yellow, _ = _marking_mask(image, self.baseline, self.grid)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(yellow.astype(np.uint8), 8)
        candidates = []
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < cfg.minimum_pixels:
                continue
            mask = ((labels == label) & (self.grid[0] >= cfg.minimum_forward_m)
                    & (self.grid[0] <= cfg.maximum_forward_m)
                    & (np.abs(self.grid[1]) <= cfg.maximum_absolute_boundary_y_m))
            paint = np.column_stack((self.grid[0][mask], self.grid[1][mask]))
            if len(paint) < cfg.minimum_pixels:
                continue
            observed = np.median(paint, axis=0)
            values, vectors = np.linalg.eigh(np.cov(paint.T))
            if (values[-1] < cfg.minimum_major_variance_m2 or values[0] > cfg.maximum_minor_variance_m2
                    or values[-1] < cfg.minimum_variance_ratio * values[0]):
                continue
            tangent = vectors[:, -1]
            match = int(np.argmin(np.sum((boundary - observed)**2, axis=1)))
            predicted, predicted_tangent = boundary[match], tangents[match]
            if np.linalg.norm(predicted - observed) > cfg.maximum_boundary_distance_m:
                continue
            if tangent @ predicted_tangent < 0:
                tangent = -tangent
            angle = float(np.arctan2(predicted_tangent[0] * tangent[1] - predicted_tangent[1] * tangent[0],
                                     predicted_tangent @ tangent))
            if abs(angle) > cfg.maximum_heading_change_rad:
                continue
            normal = np.array([-tangent[1], tangent[0]])
            offset = float((observed - predicted) @ normal)
            if abs(offset) > cfg.maximum_lateral_offset_m:
                continue
            # One short dash measures a local tangent, not the complete bend.
            # A conservative angular gain retains the learned curve's shape.
            applied_angle = cfg.heading_gain * angle
            rotation = np.array([[math.cos(applied_angle), -math.sin(applied_angle)],
                                 [math.sin(applied_angle), math.cos(applied_angle)]])
            aligned = (center - predicted) @ rotation.T + predicted + offset * normal
            reference = int(np.argmin(np.sum(aligned**2, axis=1)))
            if np.linalg.norm(aligned[reference]) > cfg.maximum_reference_distance_m:
                continue
            aligned = aligned[reference:]
            intervals = np.linalg.norm(np.diff(aligned, axis=0), axis=1)
            if not len(intervals) or np.any(intervals <= 1e-8):
                continue
            arc = np.concatenate(([0.], np.cumsum(intervals)))
            if arc[-1] < 6.:
                continue
            target = np.column_stack([np.interp([3., 6.], arc, aligned[:, axis]) for axis in range(2)])
            change = target - original[:2]
            if np.max(np.linalg.norm(change, axis=1)) > cfg.maximum_near_change_m:
                continue
            candidates.append((float(np.linalg.norm(observed)), observed, change,
                               {"observed_paint_center_ego_m": observed.tolist(),
                                "matched_predicted_boundary_ego_m": predicted.tolist(),
                                "heading_change_rad": applied_angle, "observed_heading_difference_rad": angle,
                                "normal_offset_m": offset,
                                "paint_pixels": len(paint), "reference_distance_m": float(np.linalg.norm(aligned[0]))}))
        diagnostics["qualified_components"] = len(candidates)
        if not candidates:
            return original.tolist(), diagnostics
        candidates.sort(key=lambda item: item[0])
        _, observed, change, selected = candidates[0]
        if any(np.linalg.norm(observed - alternative[1]) <= cfg.competing_center_gap_m
               and np.max(np.linalg.norm(change - alternative[2], axis=1)) > cfg.competing_correction_difference_m
               for alternative in candidates[1:]):
            diagnostics["rejection"] = "competing-paint-alignment"
            return original.tolist(), diagnostics
        result = original.copy()
        result[:2] += cfg.blend * change
        try:
            validate_waypoint_points(result.tolist())
        except LearningDataError:
            diagnostics["rejection"] = "corrected-path-not-ordered-or-in-range"
            return original.tolist(), diagnostics
        diagnostics.update(available=True, method="paint-normal-and-heading-alignment-with-arc-resampling",
                           corrected_horizons_m=[3., 6.], **selected)
        return result.tolist(), diagnostics


class PathCorrectedPredictor:
    def __init__(self, predictor, *, baseline_config=None, config=None):
        self.predictor = predictor
        self.corrector = RGBPathCorrection(baseline_config, config)
        self._identity = copy.deepcopy(predictor.identity)
        self._identity.update(variant=self._identity["variant"] + "_path_near", correction_version=VERSION,
                              correction_config=self.corrector.config.to_dict(),
                              correction_config_sha256=self.corrector.config.sha256,
                              nominal_calibration_sha256=self.corrector.baseline.sha256,
                              local_correction_config_sha256=self.corrector.local.config.sha256,
                              correction_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def predict(self, rgb, *, width=640, height=360):
        started = time.perf_counter()
        prediction = copy.deepcopy(self.predictor.predict(rgb, width=width, height=height))
        prediction.setdefault("diagnostics", {})["variant"] = self._identity["variant"]
        if prediction["status"] == "PASS":
            prediction["waypoints_ego_m"], correction = self.corrector.apply(
                rgb, prediction["waypoints_ego_m"], width=width, height=height)
            prediction["diagnostics"]["path_correction"] = correction
        prediction["timing_ms"] = (time.perf_counter() - started) * 1000
        return prediction
