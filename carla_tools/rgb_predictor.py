"""Stateless RGB waypoint prediction with an explicit OpenCV-first policy."""

from __future__ import annotations

import time

from .baseline import OpenCVLanePredictor
from .learning_data import LearningDataError, validate_waypoint_points

PREDICTOR_VERSION = "vlf-b0-assisted-waypoint-v1"


def _validated_result(predict):
    try:
        result = predict()
        if not isinstance(result, dict) or result.get("status") != "PASS":
            error = result.get("error") if isinstance(result, dict) else None
            return None, error or "prediction-failed", {}
        points = result.get("waypoints_ego_m")
        validate_waypoint_points(points)
        return [[float(x), float(y)] for x, y in points], None, result.get("diagnostics", {})
    except (LearningDataError, TypeError, ValueError, RuntimeError, OSError) as error:
        return None, "invalid-prediction: " + type(error).__name__, {}


class BaselineAssistedWaypointPredictor:
    """Use a valid B0 path, otherwise infer a new path from the same RGB.

    Selection receives no labels, route, pose, speed or historical path. A
    successful result means the shared waypoint contract is satisfied; it is
    not an accuracy guarantee. Both failures produce an explicit empty path.
    """

    def __init__(self, learned, *, baseline=None):
        if not callable(getattr(learned, "predict_rgb", None)):
            raise TypeError("learned-predictor-requires-predict-rgb")
        self.learned = learned
        self.baseline = baseline if baseline is not None else OpenCVLanePredictor()
        if not callable(getattr(self.baseline, "predict", None)):
            raise TypeError("baseline-predictor-requires-predict")

    def predict(self, rgb, width=640, height=360):
        started = time.perf_counter()
        diagnostics = {"predictor_version": PREDICTOR_VERSION, "input": "RGB-only",
                       "policy": "valid-B0-else-learned", "selected_predictor": None,
                       "fallback_used": False}
        import numpy as np
        try:
            image = np.asarray(rgb)
            valid_image = image.dtype == np.uint8 and image.shape == (360, 640, 3)
        except (TypeError, ValueError):
            valid_image = False
        if (width, height) != (640, 360) or not valid_image:
            return {"status": "FAIL", "waypoints_ego_m": None,
                    "error": "predictor-requires-640x360-rgb", "diagnostics": diagnostics,
                    "timing_ms": (time.perf_counter() - started) * 1000}
        points, error, source = _validated_result(lambda: self.baseline.predict(rgb, width=width, height=height))
        diagnostics["baseline_error"] = error
        if points is not None:
            diagnostics.update(selected_predictor="B0", source_diagnostics=source)
        else:
            diagnostics["fallback_used"] = True
            points, error, source = _validated_result(lambda: self.learned.predict_rgb(rgb))
            if points is not None:
                diagnostics.update(selected_predictor="ResNet18", source_diagnostics=source)
        return {"status": "PASS" if points is not None else "FAIL", "waypoints_ego_m": points,
                "error": None if points is not None else error, "diagnostics": diagnostics,
                "timing_ms": (time.perf_counter() - started) * 1000}

    def predict_rgb(self, rgb):
        return self.predict(rgb)
