"""Bounded memory of RGB-derived 2D boundary observations.

Coordinates belong to the current actor frame. Relative motion must map old
static points into that frame as ``p_current = R @ p_previous + t``. No route,
absolute pose, speed, camera calibration or ground truth is accepted here.
Geometry fitting remains the caller's responsibility.

The uncertainty value covers motion/memory transport only, not boundary pixel
or lane-association accuracy. It is not a calibrated covariance, confidence
probability or guarantee about an unseen road boundary. Usable RGB motion
cannot by itself detect a changed scene or association with the wrong lane.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real


VERSION = "rgb-boundary-distance-memory-v2"


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _require(value, reason):
    if not value:
        raise ValueError(reason)


@dataclass(frozen=True)
class BoundaryMemoryConfig:
    spatial_bucket_m: float = .10
    max_points_per_side: int = 256
    max_transport_distance_m: float = 15.
    observation_uncertainty_m: float = .02
    max_uncertainty_m: float = .15
    max_stationary_hold_seconds: float = 10.
    max_frame_gap_seconds: float = .25
    min_inlier_fraction: float = .55
    max_motion_residual_m: float = .03
    max_step_translation_m: float = 1.
    max_step_yaw_rad: float = .15

    def __post_init__(self):
        for name, value in asdict(self).items():
            _require(_finite(value) and value > 0, "boundary-memory-invalid-config:" + name)
        _require(type(self.max_points_per_side) is int and self.max_points_per_side <= 4096,
                 "boundary-memory-point-budget-invalid")
        _require(self.observation_uncertainty_m < self.max_uncertainty_m
                 and self.min_inlier_fraction <= 1
                 and self.max_step_yaw_rad < math.pi,
                 "boundary-memory-invalid-config-range")


class BoundaryObservationMemory:
    """Transport two side-labelled point clouds without a per-frame time TTL.

    Distance/rotation-based uncertainty, travelled distance, spatial extent and
    a point budget bound storage. A ten-second continuous near-stationary hold
    cap additionally prevents indefinitely trusting an unobserved near point.
    New observations can refresh their own spatial cells, never unseen cells.
    """

    def __init__(self, config=None):
        self.config = config if config is not None else BoundaryMemoryConfig()
        _require(type(self.config) is BoundaryMemoryConfig, "boundary-memory-config-required")
        self.reset()

    def reset(self):
        self._points = [[], []]
        self._timestamp = None
        self._frames = 0
        self._resets = 0
        self._distance = 0.
        self._past = [[], []]
        self._stats = {"motion_usable": False, "reset_reason": "explicit-reset"}

    def _motion(self, motion):
        if motion is None or type(motion) is not dict:
            return None, "relative-RGB-motion-unavailable"
        if motion.get("usable") is not True or motion.get("available", True) is not True:
            return None, "relative-RGB-motion-unavailable"
        rotation, translation = motion.get("R"), motion.get("t")
        try:
            _require(len(rotation) == 2 and all(len(row) == 2 for row in rotation)
                     and len(translation) == 2, "relative-RGB-SE2-invalid")
            values = [*rotation[0], *rotation[1], *translation]
            _require(all(_finite(x) for x in values), "relative-RGB-SE2-nonfinite")
            a, b = rotation[0]; c, d = rotation[1]
            _require(max(abs(a*a+c*c-1), abs(b*b+d*d-1), abs(a*b+c*d), abs(a*d-b*c-1)) <= 1e-5,
                     "relative-RGB-SE2-invalid")
            distance = math.hypot(*translation)
            yaw = abs(math.atan2(c, a))
            _require(distance <= self.config.max_step_translation_m and yaw <= self.config.max_step_yaw_rad,
                     "relative-RGB-motion-step-outlier")
            inliers, fraction = motion.get("inliers"), motion.get("inlier_fraction")
            residual, scale = motion.get("SE2_inlier_mean_residual_m"), motion.get("affine_scale")
            _require(_finite(inliers) and inliers >= 8
                     and _finite(fraction) and self.config.min_inlier_fraction <= fraction <= 1
                     and _finite(residual) and 0 <= residual <= self.config.max_motion_residual_m
                     and _finite(scale) and .9 <= scale <= 1.1,
                     "relative-RGB-motion-quality-unqualified")
        except (TypeError, IndexError, KeyError):
            return None, "relative-RGB-SE2-invalid"
        except ValueError as error:
            return None, str(error)
        return {"R": rotation, "t": translation, "distance": distance, "yaw": yaw,
                "residual": float(residual), "inlier_fraction": float(fraction),
                "inliers": int(inliers), "affine_scale": float(scale)}, None

    def _observations(self, observations):
        _require(type(observations) in (list, tuple) and len(observations) == 2,
                 "boundary-memory-two-side-clouds-required")
        result = [[], []]
        for side, points in enumerate(observations):
            _require(type(points) in (list, tuple) and len(points) <= 16384,
                     "boundary-memory-observation-budget")
            for point in points:
                _require(type(point) in (list, tuple) and len(point) == 2
                         and all(_finite(value) for value in point), "boundary-memory-point-invalid")
                x, y = map(float, point)
                if -3. <= x <= 35. and abs(y) <= 35.:
                    result[side].append([x, y])
        return result

    def _bounded(self, records):
        cells = {}
        bucket = self.config.spatial_bucket_m
        # Preserve the oldest transported lineage AND freshest observation.
        # At low speed the previous point and current observation can occupy
        # the same bucket for several frames. Keeping only the freshest would
        # erase the point before it could move into the unobserved near area.
        # Exact coordinate duplicates keep the freshest actual observation,
        # which genuinely reobserves that location; distinct coordinates keep
        # both extremes, bounded by two representatives per spatial bucket.
        for row in sorted(records, key=lambda r: (-r["observed_at"], r["uncertainty_m"], *r["xy"])):
            key = tuple(math.floor(value / bucket) for value in row["xy"])
            cells.setdefault(key, []).append(row)
        representatives = []
        for values in cells.values():
            freshest = values[0]
            oldest = min(values, key=lambda r: (r["observed_at"], r["uncertainty_m"], *r["xy"]))
            representatives.append(freshest)
            if oldest["xy"] != freshest["xy"]:
                representatives.append(oldest)
        kept = sorted(representatives, key=lambda r: (abs(r["xy"][0]), abs(r["xy"][1]),
                                                       r["uncertainty_m"], -r["observed_at"], *r["xy"]))
        return kept[:self.config.max_points_per_side]

    def update(self, observations_by_side, motion, timestamp):
        """Advance old points, then add current observations; return a copy.

        Invalid/unqualified motion clears old memory immediately. Valid current
        observations may seed a new map, but ``past_points_by_side`` stays empty
        on that frame. Reversed time and frame gaps likewise start a fresh map.
        """
        if not _finite(timestamp):
            self.reset()
            raise ValueError("boundary-memory-timestamp-invalid")
        try:
            observations = self._observations(observations_by_side)
        except ValueError:
            self.reset()
            raise
        timestamp = float(timestamp)
        estimate, reason = self._motion(motion)
        dt = 0. if self._timestamp is None else timestamp-self._timestamp
        if self._timestamp is None:
            reason = "first-frame-no-transport-history"
        elif not 0 < dt <= self.config.max_frame_gap_seconds:
            estimate, reason = None, "timestamp-discontinuity"
        expired = {"distance": 0, "uncertainty": 0, "stationary_hold": 0, "spatial_extent": 0}
        moved = [[], []]
        if estimate is None or reason is not None:
            self._resets += int(any(self._points))
            self._distance = 0.
        else:
            self._distance += estimate["distance"]
            stationary = estimate["distance"] <= .002 and estimate["yaw"] <= .0005
            rate = .01 + estimate["residual"] / estimate["inlier_fraction"]
            for side, records in enumerate(self._points):
                for record in records:
                    x, y = record["xy"]
                    rotation, translation = estimate["R"], estimate["t"]
                    xy = [rotation[i][0]*x+rotation[i][1]*y+translation[i] for i in (0, 1)]
                    travel = record["transported_distance_m"] + estimate["distance"]
                    uncertainty = record["uncertainty_m"] + rate*(estimate["distance"]+estimate["yaw"]*math.hypot(x, y))
                    hold = record["stationary_hold_seconds"]+dt if stationary else 0.
                    failed = ("distance" if travel > self.config.max_transport_distance_m else
                              "uncertainty" if uncertainty > self.config.max_uncertainty_m else
                              "stationary_hold" if hold > self.config.max_stationary_hold_seconds+1e-9 else
                              "spatial_extent" if not (-3 <= xy[0] <= 35 and abs(xy[1]) <= 35) else None)
                    if failed:
                        expired[failed] += 1
                        continue
                    moved[side].append({"xy": xy, "observed_at": record["observed_at"],
                                        "transported_distance_m": travel, "uncertainty_m": uncertainty,
                                        "stationary_hold_seconds": hold})
        self._past = [self._bounded(records) for records in moved]
        self._points = []
        for side, points in enumerate(observations):
            current = [{"xy": point, "observed_at": timestamp, "transported_distance_m": 0.,
                        "uncertainty_m": self.config.observation_uncertainty_m,
                        "stationary_hold_seconds": 0.} for point in points]
            self._points.append(self._bounded(self._past[side]+current))
        self._timestamp = timestamp
        self._frames += 1
        self._stats = {"motion_usable": estimate is not None and reason is None, "reset_reason": reason,
                       "history_resets": self._resets, "frames_since_reset": self._frames,
                       "cumulative_distance_m": self._distance, "expired_by_reason": expired,
                       "quality": ({key: estimate[key] for key in ("residual", "inlier_fraction", "inliers", "affine_scale")}
                                   if estimate is not None else None)}
        return self.snapshot()

    def snapshot(self):
        """Return deterministic, independent copies of points and diagnostics."""
        def records(sides):
            return [[{**row, "xy": list(row["xy"]),
                       "source_age_seconds": self._timestamp-row["observed_at"]}
                     for row in side] for side in sides]
        current, past = records(self._points), records(self._past)
        flat = [row for side in current for row in side]
        stats = {**self._stats,
                 "expired_by_reason": dict(self._stats.get("expired_by_reason", {})),
                 "quality": dict(self._stats["quality"]) if self._stats.get("quality") is not None else None,
                 "point_count_by_side": list(map(len, current)), "past_point_count_by_side": list(map(len, past)),
                 "max_source_age_seconds": max((r["source_age_seconds"] for r in flat), default=None),
                 "max_uncertainty_m": max((r["uncertainty_m"] for r in flat), default=None)}
        return {"version": VERSION, "timestamp": self._timestamp,
                "points_by_side": [[list(row["xy"]) for row in side] for side in current],
                "point_records_by_side": current,
                "past_points_by_side": [[list(row["xy"]) for row in side] for side in past],
                "past_point_records_by_side": past, "stats": stats,
                "policy": asdict(self.config),
                "uncertainty_scope": "motion-memory-transport-only-excludes-boundary-pixel-and-lane-association-error"}
