"""Stateless RGB-only lane geometry with a fixed nominal pinhole camera.

No CARLA, world pose, route, map, or labelled waypoint is accepted by this module.
An unpainted outside boundary may be inferred from a declared lane-width prior.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .contracts import read_strict_json

BASELINE_VERSION = "vlf-opencv-lane-v2"
DEFAULT_BASELINE_PATH = Path(__file__).resolve().parents[1] / "configs" / "baseline.json"
HORIZONS_M = (3.0, 6.0, 9.0, 12.0, 15.0)


class BaselineError(ValueError):
    """Invalid baseline configuration or RGB input."""


@dataclass(frozen=True)
class BaselineConfig:
    schema_version: int = 1
    baseline_version: str = BASELINE_VERSION
    image_width: int = 640
    image_height: int = 360
    fov_deg: float = 90.0
    camera_x_m: float = 1.5
    camera_y_m: float = 0.0
    camera_z_m: float = 2.4
    camera_pitch_deg: float = -12.0
    nominal_lane_width_m: float = 4.0
    minimum_lane_width_m: float = 2.8
    maximum_lane_width_m: float = 4.8
    maximum_forward_m: float = 35.0
    maximum_lateral_m: float = 15.0
    minimum_marking_pixels: int = 24
    minimum_support_bins: int = 7
    support_bin_m: float = 0.5
    maximum_fit_residual_m: float = 0.22
    maximum_heading_deg: float = 70.0
    maximum_curvature_per_m: float = 0.20
    maximum_lateral_offset_m: float = 1.2
    maximum_far_extrapolation_m: float = 2.0
    white_min_value: int = 155
    white_max_saturation: int = 60
    yellow_min_value: int = 60
    yellow_min_saturation: int = 100
    yellow_min_hue: int = 8
    yellow_max_hue: int = 40

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1 or self.baseline_version != BASELINE_VERSION:
            raise BaselineError("baseline-invalid-version")
        for key in ("image_width", "image_height", "minimum_marking_pixels", "minimum_support_bins", "white_min_value",
                    "white_max_saturation", "yellow_min_value", "yellow_min_saturation", "yellow_min_hue", "yellow_max_hue"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise BaselineError("baseline-invalid-" + key)
        for key in (item.name for item in fields(self) if item.name not in ("schema_version", "baseline_version") and
                    type(getattr(self, item.name)) is not int):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise BaselineError("baseline-invalid-" + key)
        # Calibration is deliberately fixed, not a per-frame simulator measurement.
        expected = {"image_width": 640, "image_height": 360, "fov_deg": 90.0, "camera_x_m": 1.5,
                    "camera_y_m": 0.0, "camera_z_m": 2.4, "camera_pitch_deg": -12.0}
        if any(getattr(self, key) != value for key, value in expected.items()):
            raise BaselineError("baseline-unsupported-nominal-calibration")
        if not (2.0 <= self.minimum_lane_width_m < self.nominal_lane_width_m < self.maximum_lane_width_m <= 6.0):
            raise BaselineError("baseline-invalid-lane-width")
        checks = ((self.maximum_forward_m, 20, 45), (self.maximum_lateral_m, 5, 20),
                  (self.minimum_marking_pixels, 10, 500), (self.minimum_support_bins, 5, 30),
                  (self.support_bin_m, 0.25, 1.0), (self.maximum_fit_residual_m, 0.05, 0.5),
                  (self.maximum_heading_deg, 20, 80), (self.maximum_curvature_per_m, 0.04, 0.25),
                  (self.maximum_lateral_offset_m, 0.2, 1.5), (self.maximum_far_extrapolation_m, 0, 3))
        if any(not lower <= value <= upper for value, lower, upper in checks):
            raise BaselineError("baseline-invalid-threshold")
        if not (0 < self.yellow_min_hue < self.yellow_max_hue < 90 and
                max(self.white_min_value, self.white_max_saturation, self.yellow_min_value, self.yellow_min_saturation) <= 255):
            raise BaselineError("baseline-invalid-color-threshold")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def baseline_config_from_dict(raw: dict) -> BaselineConfig:
    names = {item.name for item in fields(BaselineConfig)}
    if type(raw) is not dict or set(raw) != names:
        raise BaselineError("baseline-invalid-config-fields")
    return BaselineConfig(**raw)


def load_baseline_config(path: Path | str = DEFAULT_BASELINE_PATH) -> BaselineConfig:
    return baseline_config_from_dict(read_strict_json(Path(path)))


def _dependencies():
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise BaselineError("baseline-opencv-dependency-missing") from None
    return cv2, np


def _rgb_array(rgb: Any, width: int, height: int, config: BaselineConfig):
    _, np = _dependencies()
    if type(width) is not int or type(height) is not int or (width, height) != (config.image_width, config.image_height):
        raise BaselineError("baseline-invalid-image-dimensions")
    if type(rgb) is bytes:
        if len(rgb) != width * height * 3:
            raise BaselineError("baseline-invalid-rgb-byte-length")
        image = np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3)
    elif isinstance(rgb, np.ndarray):
        if rgb.dtype != np.uint8 or rgb.shape != (height, width, 3):
            raise BaselineError("baseline-invalid-rgb-array")
        image = rgb
    else:
        raise BaselineError("baseline-invalid-rgb-input")
    return image


def nominal_ground_to_pixels(points, config: BaselineConfig):
    """Project actor ground-plane XY through the fixed nominal camera.

    CARLA pitch is positive looking upward. The optical axes are right, down,
    forward. This function accepts planar actor-relative points only.
    """
    _, np = _dependencies()
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise BaselineError("baseline-invalid-ground-points")
    p = math.radians(config.camera_pitch_deg)
    c, s = math.cos(p), math.sin(p)
    dx = points[:, 0] - config.camera_x_m
    dy = points[:, 1] - config.camera_y_m
    dz = -config.camera_z_m
    depth = c * dx + s * dz
    vertical = -s * dx + c * dz
    focal = config.image_width / (2 * math.tan(math.radians(config.fov_deg) / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.column_stack((focal * dy / depth + config.image_width / 2,
                              -focal * vertical / depth + config.image_height / 2))
    return uv, depth > 1e-6


def _pixel_ground_grid(config: BaselineConfig):
    _, np = _dependencies()
    vv, uu = np.indices((config.image_height, config.image_width), dtype=float)
    focal = config.image_width / (2 * math.tan(math.radians(config.fov_deg) / 2))
    right = (uu - config.image_width / 2) / focal
    up = -(vv - config.image_height / 2) / focal
    p = math.radians(config.camera_pitch_deg)
    c, s = math.cos(p), math.sin(p)
    ray_x, ray_z = c - s * up, s + c * up
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = -config.camera_z_m / ray_z
        x = config.camera_x_m + distance * ray_x
        y = config.camera_y_m + distance * right
    valid = ((ray_z < -0.01) & (distance > 0) & (x >= 2.5) &
             (x <= config.maximum_forward_m) & (np.abs(y) <= config.maximum_lateral_m))
    return x, y, valid


def _marking_mask(image, config, grid):
    cv2, np = _dependencies()
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    yellow = ((hsv[:, :, 0] >= config.yellow_min_hue) & (hsv[:, :, 0] <= config.yellow_max_hue) &
              (hsv[:, :, 1] >= config.yellow_min_saturation) & (hsv[:, :, 2] >= config.yellow_min_value))
    white = ((hsv[:, :, 1] <= config.white_max_saturation) & (hsv[:, :, 2] >= config.white_min_value))
    valid = grid[2]
    return yellow & valid, white & valid


def _binned_points(mask, grid, config):
    """Metric longitudinal medians; equal-weight bins avoid near-pixel bias."""
    _, np = _dependencies()
    x, y = grid[0][mask], grid[1][mask]
    if len(x) < config.minimum_marking_pixels:
        return np.empty((0, 2))
    bins = np.floor(x / config.support_bin_m).astype(int)
    points = []
    for index in np.unique(bins):
        select = bins == index
        if select.sum() >= 2:
            points.append((float(np.median(x[select])), float(np.median(y[select]))))
    return np.asarray(points, dtype=float).reshape(-1, 2)


def _robust_polynomial(points, config):
    _, np = _dependencies()
    if len(points) < config.minimum_support_bins:
        raise BaselineError("baseline-insufficient-marking-support")
    x, y = points[:, 0], points[:, 1]
    if np.ptp(x) < 7.0:
        raise BaselineError("baseline-insufficient-forward-support")
    # Fixed-seed RANSAC is stateless and deterministic under disk replay.
    rng = np.random.default_rng(1643)
    degree = 3 if len(points) >= 12 else 2
    best = None
    for _ in range(80):
        indices = rng.choice(len(points), degree + 1, replace=False)
        if np.ptp(x[indices]) < 5:
            continue
        coefficients = np.polynomial.polynomial.polyfit(x[indices], y[indices], degree)
        residual = np.abs(np.polynomial.polynomial.polyval(x, coefficients) - y)
        inliers = residual <= config.maximum_fit_residual_m
        score = (int(inliers.sum()), -float(np.minimum(residual, 2).sum()))
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None or best[1].sum() < config.minimum_support_bins or best[1].sum() < len(points) * 0.55:
        raise BaselineError("baseline-ambiguous-or-inconsistent-marking")
    keep = best[1]
    for _ in range(3):
        coefficients = np.polynomial.polynomial.polyfit(x[keep], y[keep], degree)
        residual = np.abs(np.polynomial.polynomial.polyval(x, coefficients) - y)
        keep = residual <= config.maximum_fit_residual_m
    if keep.sum() < config.minimum_support_bins:
        raise BaselineError("baseline-inconsistent-marking-fit")
    residual = np.abs(np.polynomial.polynomial.polyval(x[keep], coefficients) - y[keep])
    return coefficients, {"support_bins": int(keep.sum()), "candidate_bins": int(len(points)),
                          "support_forward_min_m": float(x[keep].min()), "support_forward_max_m": float(x[keep].max()),
                          "residual_p95_m": float(np.quantile(residual, 0.95)), "polynomial_degree": degree}


def _curve(coefficients, x):
    _, np = _dependencies()
    y = np.polynomial.polynomial.polyval(x, coefficients)
    first = np.polynomial.polynomial.polyval(x, np.polynomial.polynomial.polyder(coefficients))
    second = np.polynomial.polynomial.polyval(x, np.polynomial.polynomial.polyder(coefficients, 2))
    return y, first, second


def _observed_circle_tangent(near, far, far_tangent):
    """Near tangent of the circle through two RGB-observed paint centers.

    A circle's two endpoint tangents are mirror images about its chord. The
    reflection also covers the straight-line limit without an unstable solve.
    The far tangent is oriented toward increasing path distance beforehand.
    """
    _, np = _dependencies()
    chord = np.asarray(far, dtype=float) - np.asarray(near, dtype=float)
    distance = float(np.linalg.norm(chord))
    tangent = np.asarray(far_tangent, dtype=float)
    magnitude = float(np.linalg.norm(tangent))
    if distance <= 1e-6 or magnitude <= 1e-6 or not np.isfinite(chord).all() or not np.isfinite(tangent).all():
        raise BaselineError("baseline-observed-circle-degenerate")
    direction, tangent = chord / distance, tangent / magnitude
    if float(direction @ tangent) < 0:
        tangent = -tangent
    near_tangent = 2 * float(direction @ tangent) * direction - tangent
    curvature = 2 * float(direction[0] * tangent[1] - direction[1] * tangent[0]) / distance
    return near_tangent, tangent, curvature


def _yellow_component_curve(mask, grid, config, image=None):
    """Trace elongated paint components, preserving curved-path ordering.

    A forward graph and local paint tangents permit a bend whose X coordinate
    ceases to increase. Compact yellow roadside objects are not lane paint.
    """
    cv2, np = _dependencies()
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) if image is not None else None
    components = []
    for index in range(1, count):
        area = int(statistics[index, cv2.CC_STAT_AREA])
        if area < 6:
            continue
        select = labels == index
        points = np.column_stack((grid[0][select], grid[1][select]))
        center = np.median(points, axis=0)
        covariance = np.cov(points.T)
        values, vectors = np.linalg.eigh(covariance)
        tangent = vectors[:, -1]
        major, minor = math.sqrt(max(0, values[-1])), math.sqrt(max(0, values[0]))
        clipped_near_anchor = area >= 80 and center[0] <= 8 and -4.5 < center[1] < -0.3 and minor <= 0.12
        uncertain_tangent = major < 0.12 or major < max(0.025, minor) * 2.0
        if minor > 0.25 or (uncertain_tangent and not clipped_near_anchor):
            continue
        if tangent[0] < 0:
            tangent = -tangent
        hue = float(np.median(hsv[select, 0])) if hsv is not None else None
        components.append({"center": center, "tangent": tangent, "pixels": area,
                           "uncertain_tangent": uncertain_tangent, "hue": hue})
    starts = [item for item in components if 2.5 <= item["center"][0] <= 14 and -4.5 < item["center"][1] < -0.3]
    if not starts:
        raise BaselineError("baseline-near-yellow-boundary-not-observed")
    starts.sort(key=lambda item: float(np.linalg.norm(item["center"])))
    first = starts[0]
    if any(item is not first and abs(item["center"][0] - first["center"][0]) < 1.0
           and abs(item["center"][1] - first["center"][1]) > .5
           and abs(float(np.linalg.norm(item["center"])) - float(np.linalg.norm(first["center"]))) < 1.5
           for item in starts):
        raise BaselineError("baseline-competing-near-yellow-boundaries")
    # Colour is compared with the *observed* near paint, not a Town/map colour.
    # HSV hue is stable under multiplicative shadows unlike a value threshold.
    rejected_color = 0
    if first["hue"] is not None:
        selected = [item for item in components if abs(item["hue"] - first["hue"]) <= 7]
        rejected_color = len(components) - len(selected)
        components = selected
    near_heading_model = "observed-near-paint-PCA"
    inferred_near_curvature = None
    if first["uncertain_tangent"]:
        # A dash clipped by the image bottom supplies position but not heading.
        # Associate the far paint geometrically *before* estimating a tangent.
        # Two observed centers plus the far paint's measured tangent suffice;
        # arbitrary nearest background colour cannot become a third anchor.
        later = []
        for item in components:
            if item is first or item["uncertain_tangent"]:
                continue
            delta = item["center"] - first["center"]
            gap = float(np.linalg.norm(delta))
            if not 1 <= gap <= 16 or delta[0] <= .5:
                continue
            near_tangent, far_tangent, curvature = _observed_circle_tangent(
                first["center"], item["center"], item["tangent"])
            if (near_tangent[0] <= 0 or abs(math.degrees(math.atan2(near_tangent[1], near_tangent[0]))) > 35 or
                    abs(curvature) > config.maximum_curvature_per_m):
                continue
            score = gap + 4 * (1 - near_tangent[0]) + 4 * (1 - float(far_tangent @ (delta/gap)))
            later.append((score, item, near_tangent, curvature))
        if not later:
            raise BaselineError("baseline-clipped-near-marking-heading-unresolved")
        later.sort(key=lambda candidate: candidate[0])
        if len(later) > 1 and later[1][0] - later[0][0] < .8:
            raise BaselineError("baseline-clipped-near-marking-association-ambiguous")
        _, _, first["tangent"], inferred_near_curvature = later[0]
        near_heading_model = "two-observed-centers-and-far-observed-tangent-circle"
    if abs(math.degrees(math.atan2(first["tangent"][1], first["tangent"][0]))) > 35:
        raise BaselineError("baseline-near-marking-heading-ambiguous")
    chain = [first]
    unused = [item for item in components if item is not first]
    while unused:
        previous = chain[-1]
        candidates = []
        for item in unused:
            delta = item["center"] - previous["center"]
            gap = float(np.linalg.norm(delta))
            forward = float(delta @ previous["tangent"])
            if not 1 <= gap <= 16 or forward < 0.5 or forward / gap < 0.42:
                continue
            tangent = item["tangent"]
            if float(tangent @ delta) < 0:
                tangent = -tangent
            alignment = float(tangent @ previous["tangent"])
            if alignment < -0.2:
                continue
            score = gap + 8 * (1 - forward / gap)
            candidates.append((score, item, tangent))
        if not candidates:
            break
        candidates.sort(key=lambda item: item[0])
        if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.8:
            raise BaselineError("baseline-yellow-branch-ambiguous")
        _, chosen, tangent = candidates[0]
        chosen["tangent"] = tangent
        chain.append(chosen)
        unused = [item for item in unused if item is not chosen]
        covered = float(np.linalg.norm(chain[0]["center"])) + sum(
            float(np.linalg.norm(b["center"] - a["center"])) for a, b in zip(chain, chain[1:]))
        # Far roadside colour is irrelevant once the requested horizon plus
        # normal-offset allowance is supported by the selected paint chain.
        if covered >= HORIZONS_M[-1] + config.nominal_lane_width_m:
            break
    if len(chain) < 2 or sum(float(np.linalg.norm(b["center"] - a["center"])) for a, b in zip(chain, chain[1:])) < 7:
        raise BaselineError("baseline-insufficient-dash-components")
    near = chain[0]
    # The closest actor reference is behind the first visible dash. Extending
    # its *observed tangent* is an explicit near-field camera blind-zone model.
    near_length = max(0.0, float(near["center"] @ near["tangent"])) + 1.0
    points = [near["center"] - near_length * near["tangent"]]
    tangents = [near["tangent"]]
    points.extend(item["center"] for item in chain)
    tangents.extend(item["tangent"] for item in chain)
    if config.maximum_far_extrapolation_m:
        points.append(chain[-1]["center"] + config.maximum_far_extrapolation_m * chain[-1]["tangent"])
        tangents.append(chain[-1]["tangent"])
    curve = []
    for index in range(len(points) - 1):
        a, b = points[index], points[index + 1]
        chord = float(np.linalg.norm(b - a))
        angle_cos = float(np.clip(tangents[index] @ tangents[index + 1], -1, 1))
        tangent_length = chord * (1 + 0.171572875 * (1 - angle_cos))
        m0, m1 = tangents[index] * tangent_length, tangents[index + 1] * tangent_length
        t = np.linspace(0, 1, max(3, int(chord / 0.025)), endpoint=False)
        h00, h10 = 2*t**3 - 3*t**2 + 1, t**3 - 2*t**2 + t
        h01, h11 = -2*t**3 + 3*t**2, t**3 - t**2
        curve.append(h00[:, None]*a + h10[:, None]*m0 + h01[:, None]*b + h11[:, None]*m1)
    curve.append(np.asarray([points[-1]]))
    boundary = np.concatenate(curve)
    tangent = np.gradient(boundary, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    centerline = boundary + config.nominal_lane_width_m / 2 * normal
    diagnostics = {"observed_dash_components": len(chain), "observed_paint_pixels": sum(item["pixels"] for item in chain),
                   "dash_centers_ego_m": [item["center"].tolist() for item in chain],
                   "dash_tangents_ego": [item["tangent"].tolist() for item in chain],
                   "curve_model": "ordered-paint-tangent-cubic-Hermite",
                   "paint_hue_consistency_tolerance": 7,
                   "observed_near_paint_hue": first["hue"],
                   "rejected_color_components": rejected_color,
                   "near_heading_model": near_heading_model,
                   "inferred_near_boundary_curvature_per_m": inferred_near_curvature,
                   "near_blind_zone_model": "backward-extension-of-observed-nearest-paint-tangent",
                   "allowed_far_extrapolation_m": config.maximum_far_extrapolation_m}
    return centerline, diagnostics


def _sample_centerline(centerline, config, diagnostics):
    _, np = _dependencies()
    reference = int(np.argmin(np.sum(centerline**2, axis=1)))
    nearest = centerline[reference]
    if abs(float(nearest[1])) > config.maximum_lateral_offset_m:
        raise BaselineError("baseline-ego-lane-offset-excessive")
    centerline = centerline[reference:]
    differences = np.diff(centerline, axis=0)
    distances = np.linalg.norm(differences, axis=1)
    if np.any(distances <= 1e-8):
        raise BaselineError("baseline-degenerate-path")
    arc = np.concatenate(([0.0], np.cumsum(distances)))
    if arc[-1] < HORIZONS_M[-1]:
        raise BaselineError("baseline-insufficient-horizon-support")
    directions = np.unwrap(np.arctan2(differences[:, 1], differences[:, 0]))
    region = arc[:-1] <= HORIZONS_M[-1]
    if np.any(differences[region, 0] <= 0) or np.max(np.abs(np.degrees(directions[region]))) > config.maximum_heading_deg:
        raise BaselineError("baseline-path-not-forward-or-heading-excessive")
    curvature = np.diff(directions) / ((distances[:-1] + distances[1:]) / 2)
    curvature = curvature[arc[1:-1] <= HORIZONS_M[-1]]
    # Component tangent joins can have a one-sample derivative change; use
    # a 0.25m moving average rather than pretending paint is noise-free.
    if len(curvature) >= 9:
        curvature = np.convolve(curvature, np.ones(9)/9, mode="valid")
    maximum_curvature = float(np.max(np.abs(curvature))) if len(curvature) else 0.0
    if maximum_curvature > config.maximum_curvature_per_m:
        raise BaselineError("baseline-path-curvature-excessive")
    predicted = np.column_stack((np.interp(HORIZONS_M, arc, centerline[:, 0]), np.interp(HORIZONS_M, arc, centerline[:, 1])))
    if not np.isfinite(predicted).all() or np.any(predicted[:, 0] <= 0):
        raise BaselineError("baseline-invalid-result")
    diagnostics.update(reference_center_xy_m=nearest.tolist(), maximum_predicted_curvature_per_m=maximum_curvature,
                       observed_centerline_arc_support_m=float(arc[-1]), centerline_normal_shift_m=config.nominal_lane_width_m/2)
    return predicted.tolist()


class OpenCVLanePredictor:
    """RGB-only predictor. No image history or hidden simulator access."""

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or load_baseline_config()
        if not isinstance(self.config, BaselineConfig):
            raise BaselineError("baseline-invalid-config-object")
        cv2, _ = _dependencies()
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        self._grid = _pixel_ground_grid(self.config)

    def predict(self, rgb: Any, width: int = 640, height: int = 360) -> dict:
        started = time.perf_counter()
        diagnostics = {"baseline_version": BASELINE_VERSION, "configuration_sha256": self.config.sha256,
                       "input_contract": "RGB8-and-fixed-nominal-calibration-only",
                       "coordinate_origin": "vehicle_actor_transform", "axes": "x-forward_y-right", "units": "m",
                       "horizon_arc_m": list(HORIZONS_M), "camera_calibration_source": "fixed-nominal-config",
                       "ground_plane_assumption": "actor-z-zero-flat-road", "temporal_state": "none"}
        try:
            image = _rgb_array(rgb, width, height, self.config)
            waypoints = self._predict(image, diagnostics)
            status, error = "PASS", None
        except BaselineError as exc:
            waypoints, status, error = None, "FAIL", str(exc)
        return {"status": status, "waypoints_ego_m": waypoints, "error": error, "diagnostics": diagnostics,
                "timing_ms": (time.perf_counter() - started) * 1000}

    def _predict(self, image, diagnostics):
        _, np = _dependencies()
        config = self.config
        yellow, white = _marking_mask(image, config, self._grid)
        diagnostics.update(yellow_pixel_count=int(yellow.sum()), white_pixel_count=int(white.sum()))
        if yellow.sum() >= config.minimum_marking_pixels:
            centerline, fit = _yellow_component_curve(yellow, self._grid, config, image)
            diagnostics.update(boundary_mode="yellow-left-plus-nominal-width", observed_boundary="left-yellow",
                               inferred_boundary="right", lane_width_source="fixed-nominal-prior",
                               lane_width_m=config.nominal_lane_width_m, confidence="single-boundary-prior", **fit)
            return _sample_centerline(centerline, config, diagnostics)
        else:
            left = _binned_points(white & (self._grid[1] < -0.3), self._grid, config)
            right = _binned_points(white & (self._grid[1] > 0.3), self._grid, config)
            left_coeff, left_fit = _robust_polynomial(left, config)
            right_coeff, right_fit = _robust_polynomial(right, config)
            sample = np.arange(3.0, 18.01, 0.5)
            ly, ld, _ = _curve(left_coeff, sample)
            ry, rd, _ = _curve(right_coeff, sample)
            lane_width = (ry - ly) / np.sqrt(1 + ((ld + rd) / 2) ** 2)
            if np.any(lane_width < config.minimum_lane_width_m) or np.any(lane_width > config.maximum_lane_width_m):
                raise BaselineError("baseline-inconsistent-paired-lane-width")
            coefficients = (left_coeff + right_coeff) / 2
            shift = 0.0
            fit = {"support_forward_max_m": min(left_fit["support_forward_max_m"], right_fit["support_forward_max_m"])}
            diagnostics.update(boundary_mode="paired-white", observed_boundary="left-and-right-white", inferred_boundary=None,
                               lane_width_source="observed-pair", lane_width_m=float(np.median(lane_width)), confidence="paired-boundaries",
                               left_fit=left_fit, right_fit=right_fit)
        # Dense sampled metric centerline, then arc-length interpolation from
        # the closest actor-origin reference. Offset along the normal, not Y.
        x = np.arange(-1.0, min(config.maximum_forward_m, fit["support_forward_max_m"] + config.maximum_far_extrapolation_m) + 0.01, 0.025)
        y, derivative, second = _curve(coefficients, x)
        normal_scale = np.sqrt(1 + derivative ** 2)
        center_x = x - shift * derivative / normal_scale
        center_y = y + shift / normal_scale
        reference = int(np.argmin(center_x ** 2 + center_y ** 2))
        if abs(center_y[reference]) > config.maximum_lateral_offset_m:
            raise BaselineError("baseline-ego-lane-offset-excessive")
        center_x, center_y = center_x[reference:], center_y[reference:]
        x_source, derivative, second = x[reference:], derivative[reference:], second[reference:]
        arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(center_x), np.diff(center_y)))))
        if arc[-1] < HORIZONS_M[-1]:
            raise BaselineError("baseline-insufficient-horizon-support")
        region = arc <= HORIZONS_M[-1]
        if np.any(np.diff(center_x[region]) <= 0) or np.max(np.abs(np.degrees(np.arctan(derivative[region])))) > config.maximum_heading_deg:
            raise BaselineError("baseline-path-not-forward-or-heading-excessive")
        curvature = second[region] / (1 + derivative[region] ** 2) ** 1.5
        # The inside/outside parallel centerline has a different curvature.
        denominator = 1 - shift * curvature
        if np.any(denominator <= 0.1) or np.max(np.abs(curvature / denominator)) > config.maximum_curvature_per_m:
            raise BaselineError("baseline-path-curvature-excessive")
        predicted = np.column_stack((np.interp(HORIZONS_M, arc, center_x), np.interp(HORIZONS_M, arc, center_y)))
        if not np.isfinite(predicted).all() or np.any(predicted[:, 0] <= 0):
            raise BaselineError("baseline-invalid-result")
        diagnostics.update(polynomial_coefficients_ascending=[float(value) for value in coefficients],
                           centerline_normal_shift_m=shift, reference_center_xy_m=[float(center_x[0]), float(center_y[0])],
                           observed_forward_max_m=fit["support_forward_max_m"],
                           requested_horizon_forward_max_m=float(np.interp(HORIZONS_M[-1], arc, x_source)),
                           allowed_far_extrapolation_m=config.maximum_far_extrapolation_m,
                           maximum_predicted_curvature_per_m=float(np.max(np.abs(curvature / denominator))))
        return predicted.tolist()


def overlay_baseline(rgb: Any, prediction: dict, config: BaselineConfig | None = None):
    """Return an RGB diagnostic overlay using only nominal camera projection."""
    config = config or load_baseline_config()
    cv2, np = _dependencies()
    image = _rgb_array(rgb, config.image_width, config.image_height, config).copy()
    grid = _pixel_ground_grid(config)
    yellow, white = _marking_mask(image, config, grid)
    image[yellow] = (255, 220, 0)
    image[white] = (180, 220, 255)
    if prediction.get("status") == "PASS":
        uv, front = nominal_ground_to_pixels(prediction["waypoints_ego_m"], config)
        prior = None
        for point, visible in zip(uv, front):
            if visible and np.isfinite(point).all() and 0 <= point[0] < config.image_width and 0 <= point[1] < config.image_height:
                pixel = tuple(np.rint(point).astype(int))
                cv2.circle(image, pixel, 5, (30, 255, 120), -1, cv2.LINE_AA)
                if prior is not None:
                    cv2.line(image, prior, pixel, (30, 255, 120), 2, cv2.LINE_AA)
                prior = pixel
    label = str(prediction.get("status")) + ": " + str(prediction.get("error") or prediction.get("diagnostics", {}).get("boundary_mode", ""))
    cv2.putText(image, label[:75], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 100, 100), 1, cv2.LINE_AA)
    return image
