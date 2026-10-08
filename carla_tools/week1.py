"""D7 same-frame geometry/color gate, independent of CARLA and GPU."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .color import ColorError, verify_rgb_color
from .geometry_survey import CASES, classify_geometry
from .labels import validate_labels
from .projection import NEAR_PLANE_M, ProjectionError, decode_rgb8_png, project_waypoints
from .routes import RouteError


GATE_VERSION = "vlf-week1-alignment-gate-d7-v1"


class GateError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def audit_labeled_episode(episode_path: str | Path, labels_path: str | Path) -> dict:
    """Require independent raw/label integrity, color witnesses and projections.

    All five points must lie in front of the camera; at least three must be
    visible. A legitimate off-image target retains its UV rather than being
    clamped. Color proof is required; legacy episodes without it cannot pass.
    """
    report = {"status": "FAIL", "scope": "D7-same-frame-RGB-pose-label-color-QA",
              "gate_version": GATE_VERSION, "errors": []}
    try:
        validated = validate_labels(episode_path, labels_path)
        if validated["status"] != "PASS":
            report["errors"] = [{"code": "raw-or-label-integrity-failed", "causes": validated["errors"]}]
            return report
        metadata, frames, sidecar = validated["metadata"], validated["frames"], validated["sidecar"]
        calibration = metadata["camera_calibration"]
        visibility = {str(distance): {"VISIBLE": 0, "OUTSIDE_IMAGE": 0} for distance in (3, 6, 9, 12, 15)}
        min_visible, color_frames, witnesses, max_roundtrip, first_points = 5, 0, 0, 0.0, None
        for frame, row in zip(frames, sidecar["rows"]):
            points = project_waypoints(row["label"]["waypoints_world_m"], frame["camera"]["transform"]["matrix"], calibration)
            if any(point["depth_m"] <= NEAR_PLANE_M or point["coordinate_clamped"] is not False for point in points):
                raise GateError("label-behind-camera-or-clamped")
            visible = sum(point["status"] == "VISIBLE" for point in points)
            if visible < 3:
                raise GateError("insufficient-visible-label-targets")
            min_visible = min(min_visible, visible)
            matrix = frame["camera"]["transform"]["matrix"]
            intrinsics = calibration["intrinsics"]
            for point in points:
                visibility[str(point["horizon_arc_m"])][point["status"]] += 1
                # Independent inverse pinhole + camera-to-world roundtrip.
                u, v = point["uv_pixels"]
                depth = point["depth_m"]
                local = [depth, (u - intrinsics["cx"]) * depth / intrinsics["fx"],
                         -(v - intrinsics["cy"]) * depth / intrinsics["fy"]]
                world = [sum(matrix[r][c] * local[c] for c in range(3)) + matrix[r][3] for r in range(3)]
                residual = math.dist(world, point["world_xyz_m"])
                if not math.isfinite(residual) or residual > .001:
                    raise GateError("camera-projection-roundtrip-failed")
                max_roundtrip = max(max_roundtrip, residual)
            png = (Path(episode_path) / frame["image"]["path"]).read_bytes()
            width, height, rgb = decode_rgb8_png(png, frame["image"]["sha256"],
                                                (frame["image"]["width"], frame["image"]["height"]))
            if "color_provenance" not in frame:
                raise GateError("same-frame-color-provenance-missing")
            color = verify_rgb_color(rgb, frame["color_provenance"], width, height, require_witness=True)
            color_frames += 1
            witnesses += color["pixel_witnesses"]
            if first_points is None:
                first_points = points
        fresh = validate_labels(episode_path, labels_path)
        if (fresh["status"] != "PASS" or fresh.get("source") != validated["source"]
                or fresh.get("labels_sha256") != validated["labels_sha256"]):
            raise GateError("sources-changed-during-week1-audit")
        cfg = json.loads(json.dumps(metadata["collection_config"]))
        cfg["vehicle"].pop("spawn_point_index")
        report.update(status="PASS", source=validated["source"], label_metrics=validated["metrics"],
                      frame_count=len(frames), first_frame=frames[0]["frame"], last_frame=frames[-1]["frame"],
                      labels_sha256=validated["labels_sha256"],
                      calibration_sha256=_canonical(calibration), configuration_except_spawn_sha256=_canonical(cfg),
                      route_geometry=classify_geometry(metadata["route_contract"]),
                      opendrive_sha256=metadata["route_contract"]["opendrive_sha256"],
                      min_visible_targets=min_visible, visibility_counts=visibility,
                      max_camera_world_roundtrip_error_m=max_roundtrip, first_frame_projections=first_points,
                      color_verified_frames=color_frames, color_pixel_witnesses=witnesses,
                      checks={"raw_disk_integrity": True, "same_frame_and_time": True, "complete_labels": True,
                              "fixed_target_route": True, "actor_origin_metres": True, "finite_geometry": True,
                              "sensor_BGRA_to_decoded_RGB": True, "camera_projection": True,
                              "no_coordinate_clamping": True, "capture_cleanup_verified": True})
    except (GateError, ColorError, ProjectionError, RouteError, OSError, ValueError, TypeError, KeyError, IndexError) as error:
        report["errors"].append({"code": getattr(error, "code", "week1-audit-operation-failed"),
                                 "exception_type": type(error).__name__})
    return report


def verify_week1_cases(cases: list[dict]) -> dict:
    """Require real complete straight, left and right episode/label pairs."""
    report = {"status": "FAIL", "gate_version": GATE_VERSION, "scope": "D7-week1-alignment-not-driving-performance",
              "errors": [], "cases": [], "frame_count": 0}
    try:
        if type(cases) is not list or len(cases) != 3 or {item.get("case") for item in cases} != set(CASES):
            raise GateError("week1-three-distinct-geometry-cases-required")
        paths, identities, versions, maps, calibrations, configurations = set(), set(), set(), set(), set(), set()
        for item in cases:
            if type(item) is not dict or set(item) != {"case", "episode", "labels"}:
                raise GateError("week1-case-schema-invalid")
            path = Path(item["episode"]).resolve()
            if path in paths:
                raise GateError("duplicate-week1-episode-path")
            paths.add(path)
            audited = audit_labeled_episode(path, item["labels"])
            report["cases"].append({"case": item["case"], "audit": audited})
            if audited["status"] != "PASS":
                raise GateError("week1-episode-audit-failed")
            if audited["route_geometry"]["case"] != item["case"]:
                raise GateError("week1-geometry-case-mismatch")
            source = audited["source"]
            if source["episode_id"] in identities:
                raise GateError("duplicate-week1-episode-id")
            identities.add(source["episode_id"])
            versions.add(source["dataset_version"])
            maps.add((source["town"], audited["opendrive_sha256"]))
            calibrations.add(audited["calibration_sha256"])
            configurations.add(audited["configuration_except_spawn_sha256"])
            report["frame_count"] += audited["frame_count"]
        if len(versions) != 1 or len(maps) != 1 or len(calibrations) != 1 or len(configurations) != 1:
            raise GateError("week1-cross-case-configuration-mismatch")
        report.update(status="PASS", episode_count=3, dataset_version=versions.pop(),
                      geometry_cases=list(CASES), label_shape=[5, 2], raw_sources_preserved=True,
                      final_test_used=False, controller_or_training_verified=False)
    except (GateError, OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        report["errors"].append({"code": getattr(error, "code", "week1-gate-operation-failed"),
                                 "exception_type": type(error).__name__})
    return report
