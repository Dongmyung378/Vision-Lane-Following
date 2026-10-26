"""Observed near lane geometry from camera-only ground motion and paint memory.

Ground optical flow transports previously visible paint into the camera blind
zone. Only measured spatial support may update a near ordinate; an unsupported
reference, failed motion estimate or ambiguous fit leaves the proposal absent.
"""
from dataclasses import asdict, dataclass, fields
import copy
import hashlib, math
import cv2, numpy as np
from .baseline import load_baseline_config, _pixel_ground_grid, _marking_mask, _rgb_array
from .centering_path_correction import curvature_aware_learned_curve
from .contracts import canonical_sha256
from .learning_data import validate_waypoint_points
VERSION = 'vlf-temporal-RGB-observed-near-lane-pose-v1'

@dataclass(frozen=True)
class LanePoseObserverConfig:
    version: str = VERSION
    history_frames: int = 40
    reference_x_m: float = 3.0
    memory_minimum_forward_m: float = -4.0
    motion_inlier_threshold_m: float = 0.06
    minimum_motion_inliers: int = 24
    minimum_motion_inlier_ratio: float = 0.55
    maximum_motion_yaw_rad: float = 0.06
    maximum_motion_translation_m: float = 0.8
    minimum_observed_span_m: float = 0.5
    quadratic_minimum_span_m: float = 1.3
    maximum_fit_residual_m: float = 0.1
    fit_full_trust_m: float = 0.03

    def __post_init__(self):
        if self.version != VERSION:
            raise ValueError('lane-observer-version')
        if type(self.history_frames) is not int or not 10 <= self.history_frames <= 40:
            raise ValueError('lane-observer-history')
        if type(self.minimum_motion_inliers) is not int or not 24 <= self.minimum_motion_inliers <= 100:
            raise ValueError('lane-observer-inliers')
        for key, value in asdict(self).items():
            if key in ('version', 'history_frames', 'minimum_motion_inliers', 'memory_minimum_forward_m'):
                continue
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError('lane-observer-numeric')
        if self.reference_x_m != 3.0 or self.memory_minimum_forward_m != -4.0 or (not 0.01 <= self.motion_inlier_threshold_m <= 0.06) or (not 0.55 <= self.minimum_motion_inlier_ratio <= 1.0) or (not 0.01 <= self.maximum_motion_yaw_rad <= 0.06) or (not 0.1 <= self.maximum_motion_translation_m <= 0.8) or (not 0.5 <= self.minimum_observed_span_m < self.quadratic_minimum_span_m <= 2.0) or (not 0.01 <= self.fit_full_trust_m < self.maximum_fit_residual_m <= 0.1):
            raise ValueError('lane-observer-bounds')

    def to_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
            raise ValueError('lane-observer-config-fields')
        return cls(**value)
    fromdict = from_dict

def _smooth(value):
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)

def rigid(source, target):
    a = source.mean(axis=0)
    b = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - a).T @ (target - b))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    return (r, b - r @ a)

class RGBTemporalLanePoseObserver:
    """Advance RGB memory once; estimate many raw paths from that fixed state.

    The caller must reset between trajectories and call observe once per scored
    frame. estimate is pure and can replay an alternative model output without
    advancing optical flow, paint transport or observation age.
    """

    def __init__(self, config=None):
        self.config = LanePoseObserverConfig() if config is None else config
        if not isinstance(self.config, LanePoseObserverConfig):
            raise ValueError('lane-observer-config-required')
        self._config_sha256 = self.config.sha256
        self.baseline = load_baseline_config()
        self.grid = _pixel_ground_grid(self.baseline)
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        cv2.setUseOptimized(False)
        self.reset()

    def reset(self):
        self.previous_gray = None
        self.memory = []
        self.index = -1
        self.rgb_hash = None
        self.motion = {'status': 'FIRST_FRAME'}
        self.current_paint = np.empty((0, 2))
        self.history_resets = 0

    def ground(self, pixels):
        pixels = np.asarray(pixels, dtype=np.float64)
        focal = self.baseline.image_width / (2.0 * math.tan(math.radians(self.baseline.fov_deg) / 2.0))
        p = math.radians(self.baseline.camera_pitch_deg)
        c, s = (math.cos(p), math.sin(p))
        right = (pixels[:, 0] - self.baseline.image_width / 2.0) / focal
        up = -(pixels[:, 1] - self.baseline.image_height / 2.0) / focal
        rayx = c - s * up
        rayz = s + c * up
        dist = -self.baseline.camera_z_m / rayz
        return np.column_stack((self.baseline.camera_x_m + dist * rayx, self.baseline.camera_y_m + dist * right))

    def _motion(self, old, gray):
        roi = ((self.grid[0] >= 3.5) & (self.grid[0] <= 18.0) & (np.abs(self.grid[1]) < 6.0)).astype(np.uint8) * 255
        features = cv2.goodFeaturesToTrack(old, maxCorners=400, qualityLevel=0.01, minDistance=7, mask=roi, blockSize=7)
        if features is None or len(features) < self.config.minimum_motion_inliers:
            return {'status': 'FAIL', 'reason': 'few-ground-features'}
        moved, status, _ = cv2.calcOpticalFlowPyrLK(old, gray, features, None, winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if moved is None or status is None:
            return {'status': 'FAIL', 'reason': 'optical-flow-unavailable'}
        reverse, back, _ = cv2.calcOpticalFlowPyrLK(gray, old, moved, None, winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if moved is None or status is None or reverse is None or (back is None):
            return {'status': 'FAIL', 'reason': 'optical-flow-unavailable'}
        p = features[:, 0]
        q = moved[:, 0]
        valid = status[:, 0].astype(bool) & back[:, 0].astype(bool) & (np.linalg.norm(reverse[:, 0] - p, axis=1) < 0.5)
        valid &= (q[:, 0] >= 0) & (q[:, 0] < 640) & (q[:, 1] >= 0) & (q[:, 1] < 360)
        a, b = (self.ground(p[valid]), self.ground(q[valid]))
        finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & (b[:, 0] > 2.0) & (b[:, 0] < 20.0) & (np.abs(b[:, 1]) < 8.0)
        a, b = (a[finite], b[finite])
        if len(a) < self.config.minimum_motion_inliers:
            return {'status': 'FAIL', 'reason': 'few-consistent-ground-tracks', 'tracks': len(a)}
        rng = np.random.default_rng(1643)
        best = None
        for _ in range(100):
            pair = rng.choice(len(a), 2, replace=False)
            if np.linalg.norm(a[pair[1]] - a[pair[0]]) < 1.0:
                continue
            r, t = rigid(a[pair], b[pair])
            res = np.linalg.norm(a @ r.T + t - b, axis=1)
            keep = res < self.config.motion_inlier_threshold_m
            score = (int(keep.sum()), -float(np.minimum(res, 0.3).sum()))
            if best is None or score > best[0]:
                best = (score, keep)
        if best is None or best[1].sum() < self.config.minimum_motion_inliers:
            return {'status': 'FAIL', 'reason': 'few-motion-inliers', 'tracks': len(a)}
        keep = best[1]
        for _ in range(3):
            if keep.sum() < self.config.minimum_motion_inliers:
                return {'status': 'FAIL', 'reason': 'few-refined-motion-inliers'}
            r, t = rigid(a[keep], b[keep])
            res = np.linalg.norm(a @ r.T + t - b, axis=1)
            keep = res < self.config.motion_inlier_threshold_m
        if keep.sum() < self.config.minimum_motion_inliers:
            return {'status': 'FAIL', 'reason': 'few-final-motion-inliers'}
        ratio = float(keep.mean())
        span = np.ptp(a[keep], axis=0)
        theta = math.atan2(r[1, 0], r[0, 0])
        p95 = float(np.quantile(res[keep], 0.95))
        result = {'status': 'PASS', 'rotation': r.tolist(), 'translation_m': t.tolist(), 'yaw_rad': theta, 'tracks': len(a), 'inliers': int(keep.sum()), 'inlier_ratio': ratio, 'residual_p95_m': p95, 'source_span_xy_m': span.tolist()}
        if ratio < self.config.minimum_motion_inlier_ratio or span[0] < 3.0 or span[1] < 1.5 or (abs(theta) > self.config.maximum_motion_yaw_rad) or (not -self.config.maximum_motion_translation_m < t[0] < 0.2) or (abs(t[1]) > 0.25):
            result.update(status='FAIL', reason='motion-support-or-plausibility')
        return result

    def observe(self, rgb, *, width=640, height=360):
        if self.config.sha256 != self._config_sha256:
            raise ValueError('lane-observer-config-changed')
        image = _rgb_array(rgb, width, height, self.baseline)
        self.index += 1
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.rgb_hash = hashlib.sha256(image.tobytes()).hexdigest()
        try:
            self.motion = self._motion(self.previous_gray, gray) if self.previous_gray is not None else {'status': 'FIRST_FRAME'}
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            self.motion = {'status': 'FAIL', 'reason': 'ground-motion-numerical-failure'}
        if self.motion['status'] == 'PASS':
            r = np.asarray(self.motion['rotation'])
            t = np.asarray(self.motion['translation_m'])
            transported = []
            translation_sigma = max(0.001, self.motion['residual_p95_m'] / math.sqrt(self.motion['inliers']))
            rotation_sigma = translation_sigma / max(self.motion['source_span_xy_m'])
            for entry in self.memory:
                age = self.index - entry['index']
                if age > self.config.history_frames:
                    continue
                points = entry['points'] @ r.T + t
                keep = (points[:, 0] > self.config.memory_minimum_forward_m) & (points[:, 0] < 18.0) & (np.abs(points[:, 1]) < 10.0)
                variance = entry['variance_m2'] + translation_sigma ** 2 + (np.linalg.norm(points, axis=1) * rotation_sigma) ** 2
                if keep.any():
                    transported.append(dict(index=entry['index'], points=points[keep], variance_m2=variance[keep]))
            self.memory = transported
        elif self.previous_gray is not None:
            self.memory = []
            self.history_resets += 1
        yellow, _ = _marking_mask(image, self.baseline, self.grid)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(yellow.astype(np.uint8), 8)
        points = []
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < 24:
                continue
            mask = (labels == label) & (self.grid[0] >= 3.5) & (self.grid[0] <= 18.0) & (np.abs(self.grid[1]) < 10.0)
            paint = np.column_stack((self.grid[0][mask], self.grid[1][mask]))
            if len(paint) < 24:
                continue
            val, vec = np.linalg.eigh(np.cov(paint.T))
            if val[-1] < 0.08 or val[0] > 0.04 or val[-1] < 4 * val[0]:
                continue
            bins = np.floor(paint @ vec[:, -1] / 0.15).astype(int)
            points.extend((np.median(paint[bins == number], axis=0) for number in np.unique(bins) if (bins == number).sum() >= 3))
        self.current_paint = np.asarray(points).reshape(-1, 2)
        if len(self.current_paint):
            self.memory.append({'index': self.index, 'points': self.current_paint.copy(), 'variance_m2': np.full(len(self.current_paint), 0.02 ** 2)})
        self.previous_gray = gray.copy()
        return copy.deepcopy(self.motion)

    def estimate(self, raw):
        if self.config.sha256 != self._config_sha256:
            raise ValueError('lane-observer-config-changed')
        validate_waypoint_points(raw)
        result = {'available': False, 'observation_index': self.index, 'RGB_sha256': self.rgb_hash, 'history_resets': self.history_resets, 'motion': copy.deepcopy(self.motion), 'memory_frame_count': len(self.memory), 'input': 'current-and-past-RGB-with-fixed-nominal-camera', 'GT_or_pose_or_speed_used': False, 'curvature_source': 'transported-observed-paint-not-learned-prior', 'confidence': 0.0, 'config_sha256': self.config.sha256, 'baseline_config_sha256': self.baseline.sha256, 'temporal_state': 'RGB-only-ground-motion-and-observed-paint', 'blindzone_extrapolation_used': False}
        if self.motion['status'] == 'FAIL':
            return dict(result, rejection='current-ground-motion-failed')
        if not self.memory:
            return dict(result, rejection='no-observed-paint')
        try:
            center, tan = curvature_aware_learned_curve(raw, 30)
        except ValueError:
            return dict(result, rejection='learned-path-association-unavailable')
        half_width = self.baseline.nominal_lane_width_m / 2.0
        left = center - half_width * np.column_stack((-tan[:, 1], tan[:, 0]))
        allpoints = np.vstack([entry['points'] for entry in self.memory])
        age = np.concatenate([np.full(len(entry['points']), self.index - entry['index']) for entry in self.memory])
        variances = np.concatenate([entry['variance_m2'] for entry in self.memory])
        window = (allpoints[:, 0] >= self.config.memory_minimum_forward_m) & (allpoints[:, 0] <= 8.0)
        p = allpoints[window]
        ages = age[window]
        variances = variances[window]
        if len(p) < 12:
            return dict(result, rejection='little-near-memory-support')
        bins = np.floor(p / 0.25).astype(int)
        binned = []
        binage = []
        binvariance = []
        for number in np.unique(bins, axis=0):
            keep = np.all(bins == number, axis=1)
            if keep.sum() >= 2:
                binned.append(np.median(p[keep], axis=0))
                binage.append(float(np.median(ages[keep])))
                binvariance.append(float(np.median(variances[keep])))
        observed = np.asarray(binned)
        if len(observed) < 3 or np.ptp(observed[:, 0]) < self.config.minimum_observed_span_m:
            return dict(result, rejection='little-near-observed-span')
        fixed_degree = 2 if len(observed) >= 6 and np.ptp(observed[:, 0]) >= self.config.quadratic_minimum_span_m else 1
        distances = np.sqrt(np.min(np.sum((observed[:, None, :] - left[None, :, :]) ** 2, axis=2), axis=1))
        association = np.array([_smooth((1.5 - d) / self.config.minimum_observed_span_m) for d in distances])
        active = association > 0.0
        observed = observed[active]
        association = association[active]
        binage = np.asarray(binage)[active]
        binvariance = np.asarray(binvariance)[active]
        if len(observed) < fixed_degree + 1 or association.sum() <= 0.0:
            return dict(result, rejection='little-associated-memory-support')
        x = observed[:, 0] - 3.0
        y = observed[:, 1]
        degree = fixed_degree
        a = np.column_stack([x ** power for power in range(degree + 1)])
        coefficients = np.linalg.lstsq(a * association[:, None] ** 0.5, y * association ** 0.5, rcond=None)[0]
        for _ in range(5):
            residual = a @ coefficients - y
            w = association * np.minimum(1.0, 0.03 / np.maximum(np.abs(residual), 1e-09))
            coefficients = np.linalg.lstsq(a * w[:, None] ** 0.5, y * w ** 0.5, rcond=None)[0]
        residual = np.abs(a @ coefficients - y)
        order = np.argsort(residual)
        cumulative = np.cumsum(association[order]) / association.sum()
        p95 = float(np.interp(0.95, cumulative, residual[order]))
        bx = np.linspace(observed[:, 0].min(), observed[:, 0].max(), 400)
        dx = bx - 3.0
        c2 = coefficients[2] if degree == 2 else 0.0
        by = coefficients[0] + coefficients[1] * dx + c2 * dx * dx
        slope = coefficients[1] + 2 * c2 * dx
        magnitude = np.sqrt(1 + slope * slope)
        cx = bx - half_width * slope / magnitude
        cy = by + half_width / magnitude
        if np.any(np.diff(cx) <= 0):
            return dict(result, rejection='nonmonotonic-observed-center')
        boundary_curvature = 2 * c2 / (1 + slope * slope) ** 1.5
        denominator = 1.0 - half_width * boundary_curvature
        if np.any(denominator <= 0.1) or np.max(np.abs(boundary_curvature / denominator)) > self.baseline.maximum_curvature_per_m:
            return dict(result, rejection='observed-center-curvature-excessive')
        result.update(observed_boundary_min_x_m=float(observed[:, 0].min()), observed_boundary_max_x_m=float(observed[:, 0].max()), observed_center_min_x_m=float(cx.min()), observed_center_max_x_m=float(cx.max()), observed_bins=len(observed), normal_fit_residual_p95_m=p95, oldest_support_age_frames=int(max(binage)), polynomial_coefficients_about_3m=coefficients.tolist(), polynomial_degree=degree)
        if p95 > self.config.maximum_fit_residual_m or cx[0] > 3.0 or cx[-1] < 3.0:
            return dict(result, rejection='3m-reference-not-observed-or-fit-poor')
        y3 = float(np.interp(3.0, cx, cy))
        h3 = float(np.arctan(np.interp(3.0, cx, slope)))
        raw = np.asarray(raw, float)
        proposal = raw.copy()
        observed_mask = []
        for i in range(2):
            supported = cx[0] <= raw[i, 0] <= cx[-1]
            observed_mask.append(bool(supported))
            if supported:
                margin = min(raw[i, 0] - cx[0], cx[-1] - raw[i, 0])
                support = _smooth(margin / self.config.minimum_observed_span_m)
                proposal[i, 1] += support * (float(np.interp(raw[i, 0], cx, cy)) - raw[i, 1])
        information = a.T @ (w[:, None] * a)
        try:
            covariance = max(0.02, p95) ** 2 * np.linalg.inv(information)
        except np.linalg.LinAlgError:
            return dict(result, rejection='observed-pose-unobservable')
        shape_sigma = math.sqrt(max(0.0, covariance[0, 0])) + half_width * math.sqrt(max(0.0, covariance[1, 1]))
        transport_variance = float(np.average(binvariance, weights=association))
        uncertainty = max(p95, 2.0 * math.sqrt(transport_variance), 2.0 * shape_sigma)
        confidence = 1.0 - _smooth((uncertainty - self.config.fit_full_trust_m) / (self.config.maximum_fit_residual_m - self.config.fit_full_trust_m))
        center_at_observations = np.interp(observed[:, 0], bx, cx)
        left_support = float(np.sum(association[center_at_observations <= 3.0]))
        right_support = float(np.sum(association[center_at_observations >= 3.0]))
        side_confidence = _smooth(min(left_support, right_support))
        reference_margin = min(3.0 - cx[0], cx[-1] - 3.0)
        reference_confidence = _smooth(reference_margin / self.config.minimum_observed_span_m)
        confidence *= side_confidence * reference_confidence
        return dict(result, available=True, reference_x_m=3.0, center_y_at_reference_m=y3, tangent_heading_rad=h3, confidence=confidence, proposed_waypoints_ego_m=proposal.tolist(), near_ordinates_observed=observed_mask, transport_and_fit_uncertainty_m=uncertainty, observed_reference_side_weights=[left_support, right_support], observed_reference_support_confidence=side_confidence * reference_confidence, association='continuous-distance-confidence-with-RGB-only-fixed-spatial-bins', uncertainty_scope='feature-fit-and-transport-heuristic-not-calibrated-safety-bound')
