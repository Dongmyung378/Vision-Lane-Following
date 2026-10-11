"""Stateless RGB near-path correction using a fixed nominal yellow-left boundary.

This development option is separate from the learned checkpoint. It accepts
neither vehicle pose nor route labels and preserves all distant model points.
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

from .baseline import load_baseline_config, _pixel_ground_grid, _marking_mask, _rgb_array
from .learning_data import validate_waypoint_points

VERSION = "vlf-rgb-near-boundary-correction-v1"


@dataclass(frozen=True)
class NearCorrectionConfig:
    version: str = VERSION
    minimum_pixels: int = 24
    minimum_bins: int = 3
    bin_width_m: float = .25
    minimum_span_m: float = .5
    quadratic_minimum_span_m: float = 1.3
    minimum_forward_m: float = 3.5
    maximum_forward_m: float = 10.
    nearest_component_maximum_forward_m: float = 8.
    maximum_absolute_boundary_y_m: float = 5.
    maximum_residual_p95_m: float = .15
    maximum_slope: float = 1.8
    backward_extension_m: float = 3.
    forward_extension_m: float = .5
    support_fade_m: float = .5
    maximum_absolute_center_y_m: float = 3.
    maximum_lateral_change_m: float = 2.5
    blend: float = .85

    def __post_init__(self):
        if self.version != VERSION:
            raise ValueError("near-correction-version-invalid")
        for key, value in asdict(self).items():
            if key == "version":
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("near-correction-config-invalid")
        if (type(self.minimum_pixels) is not int or type(self.minimum_bins) is not int
                or self.minimum_pixels < 12 or self.minimum_bins < 3 or self.blend > 1
                or not self.minimum_forward_m < self.nearest_component_maximum_forward_m <= self.maximum_forward_m
                or self.quadratic_minimum_span_m < self.minimum_span_m):
            raise ValueError("near-correction-config-invalid")

    def to_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class RGBNearCorrection:
    def __init__(self, baseline_config=None, config=None):
        self.baseline = baseline_config or load_baseline_config()
        self.config = config or NearCorrectionConfig()
        self.grid = _pixel_ground_grid(self.baseline)
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)

    def apply(self, rgb, points, *, width=640, height=360):
        image = _rgb_array(rgb, width, height, self.baseline)
        validate_waypoint_points(points)
        original = np.asarray(points, dtype=float)
        cfg = self.config
        yellow, _ = _marking_mask(image, self.baseline, self.grid)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(yellow.astype(np.uint8), 8)
        candidates = []
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < cfg.minimum_pixels:
                continue
            mask = ((labels == label) & (self.grid[0] >= cfg.minimum_forward_m)
                    & (self.grid[0] <= cfg.maximum_forward_m)
                    & (np.abs(self.grid[1]) < cfg.maximum_absolute_boundary_y_m))
            x, y = self.grid[0][mask], self.grid[1][mask]
            if len(x) < cfg.minimum_pixels or x.min() > cfg.nearest_component_maximum_forward_m:
                continue
            bins = np.floor(x / cfg.bin_width_m).astype(int)
            binned = []
            for number in np.unique(bins):
                keep = bins == number
                if keep.sum() >= 3:
                    binned.append((float(np.median(x[keep])), float(np.median(y[keep]))))
            observed = np.asarray(binned)
            if len(observed) < cfg.minimum_bins or np.ptp(observed[:, 0]) < cfg.minimum_span_m:
                continue
            degree = 2 if np.ptp(observed[:, 0]) >= cfg.quadratic_minimum_span_m and len(observed) >= 6 else 1
            coefficients = np.polynomial.polynomial.polyfit(observed[:, 0], observed[:, 1], degree)
            residual = np.abs(np.polynomial.polynomial.polyval(observed[:, 0], coefficients) - observed[:, 1])
            residual_p95 = float(np.quantile(residual, .95))
            slopes = np.polynomial.polynomial.polyval(observed[:, 0], np.polynomial.polynomial.polyder(coefficients))
            if residual_p95 > cfg.maximum_residual_p95_m or np.max(np.abs(slopes)) > cfg.maximum_slope:
                continue
            candidates.append((float(observed[0, 0]), label, residual_p95, coefficients, observed))
        diagnostics = {"version": VERSION, "config_sha256": cfg.sha256,
            "baseline_config_sha256": self.baseline.sha256, "input": "RGB-and-fixed-nominal-calibration",
            "boundary": "observed-yellow-left-plus-fixed-width-prior", "temporal_state": "none",
            "available": False, "yellow_pixels": int(yellow.sum()), "qualified_components": len(candidates)}
        if not candidates:
            return original.tolist(), diagnostics
        minimum, _, residual_p95, coefficients, observed = min(candidates, key=lambda value: (value[0], value[1]))
        x = np.linspace(max(0., minimum - cfg.backward_extension_m), observed[-1, 0] + cfg.forward_extension_m, 400)
        y = np.polynomial.polynomial.polyval(x, coefficients)
        slope = np.polynomial.polynomial.polyval(x, np.polynomial.polynomial.polyder(coefficients))
        normal = np.sqrt(1 + slope * slope)
        half_width = self.baseline.nominal_lane_width_m / 2
        center = np.column_stack((x - half_width * slope / normal, y + half_width / normal))
        if not np.isfinite(center).all() or np.any(np.diff(center[:, 0]) <= 0):
            diagnostics["rejection"] = "nonmonotonic-local-centerline"
            return original.tolist(), diagnostics
        diagnostics.update(available=True, observed_minimum_forward_m=minimum,
            observed_maximum_forward_m=float(observed[-1, 0]), residual_p95_m=residual_p95,
            polynomial_coefficients=coefficients.tolist(), polynomial_degree=len(coefficients) - 1)
        result = original.copy()
        # Smooth support fading avoids a hard switch under CPU/CUDA rounding.
        for index in range(2):
            px, py = result[index]
            support = float(np.clip(min((px - center[0, 0]) / cfg.support_fade_m,
                                       (center[-1, 0] - px) / cfg.support_fade_m,
                                       (cfg.backward_extension_m - (minimum - px)) / cfg.support_fade_m), 0., 1.))
            center_y = float(np.interp(px, center[:, 0], center[:, 1]))
            confidence = float(np.clip((cfg.maximum_absolute_center_y_m - abs(center_y)) / cfg.support_fade_m, 0., 1.))
            change = float(np.clip(center_y - py, -cfg.maximum_lateral_change_m, cfg.maximum_lateral_change_m))
            result[index, 1] = py + cfg.blend * support * confidence * change
        # Never replace an ordered path with a geometrically invalid correction.
        radii = np.linalg.norm(result, axis=1)
        if not np.isfinite(result).all() or np.any(np.diff(radii) <= 1e-6):
            diagnostics.update(available=False, rejection="corrected-path-not-ordered")
            return original.tolist(), diagnostics
        return result.tolist(), diagnostics


class NearCorrectedPredictor:
    def __init__(self, predictor, *, baseline_config=None, config=None):
        self.predictor = predictor
        self.corrector = RGBNearCorrection(baseline_config, config)
        self._identity = predictor.identity
        self._identity.update(variant=self._identity["variant"] + "_near", correction_version=VERSION,
            correction_config=self.corrector.config.to_dict(), correction_config_sha256=self.corrector.config.sha256,
            nominal_calibration_sha256=self.corrector.baseline.sha256,
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
            prediction["diagnostics"]["near_correction"] = correction
        prediction["timing_ms"] = (time.perf_counter() - started) * 1000
        return prediction
