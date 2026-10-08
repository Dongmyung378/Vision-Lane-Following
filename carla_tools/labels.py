"""Generate and independently verify immutable D6 waypoint sidecars offline.

The source episode remains unlabelled. Only a newly published derived file has
GENERATED rows; it is usable only after validate_labels() returns PASS.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from .contracts import (DEFAULT_CONTRACT_PATH, HORIZON_ARC_M, ContractError, canonical_sha256, load_dataset_contract,
                        route_sha256, sample_id, strict_json_loads,
                        validate_episode_identity, validate_frame_contract,
                        validate_waypoint_label)
from .episode import validate_episode


GENERATION_VERSION = "vlf-waypoint-generator-d6-v1"
SCOPE = "D6-derived-waypoint-labels-not-moving-pilot"
ERROR_KEYS = ("max_world_route_error_m", "max_ego_error_m", "max_world_roundtrip_error_m", "max_reference_error_m")


class LabelError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _exact(value: Any, keys: set[str], code: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise LabelError(code)


def _number(value: Any) -> float:
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise LabelError("nonfinite-label-number")
        return float(value)
    except OverflowError:
        raise LabelError("nonfinite-label-number") from None


def _xyz(anchor: dict) -> list[float]:
    return [anchor["world_location"][axis] for axis in ("x", "y", "z")]


def _point_segment(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    denominator = dx * dx + dy * dy
    if denominator <= 0:
        raise LabelError("degenerate-route-projection")
    fraction = max(0., min(1., ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / denominator))
    distance = math.hypot(point[0] - a[0] - fraction * dx, point[1] - a[1] - fraction * dy)
    return fraction, distance


def _orientation(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_touch(a, b, c, d):
    if _orientation(a, b, c) * _orientation(a, b, d) < 0 and _orientation(c, d, a) * _orientation(c, d, b) < 0:
        return True
    return min(_point_segment(a, c, d)[1], _point_segment(b, c, d)[1],
               _point_segment(c, a, b)[1], _point_segment(d, a, b)[1]) <= 1e-8


def _simple_route(route: dict) -> None:
    points = [_xyz(anchor)[:2] for anchor in route["centerline_anchor_sequence"]]
    for index, (a, b) in enumerate(zip(points, points[1:])):
        _point_segment(a, a, b)
        for other in range(index + 2, len(points) - 1):
            if _segments_touch(a, b, points[other], points[other + 1]):
                raise LabelError("self-crossing-fixed-route")


def _reference(route: dict, location: dict) -> tuple[float, float]:
    candidates = []
    point = [location["x"], location["y"]]
    anchors = route["centerline_anchor_sequence"]
    for a, b in zip(anchors, anchors[1:]):
        fraction, distance = _point_segment(point, _xyz(a)[:2], _xyz(b)[:2])
        arc = a["arc_distance_m"] + fraction * (b["arc_distance_m"] - a["arc_distance_m"])
        candidates.append((distance, arc))
    # Preserve the first nearest segment, matching the separately implemented
    # D5 validator at a shared anchor. Reject equally plausible distant arcs.
    distance, arc = min(candidates, key=lambda item: item[0])
    alternatives = [value for gap, value in candidates if abs(gap - distance) <= 1e-8]
    if max(alternatives) - min(alternatives) > .001:
        raise LabelError("ambiguous-route-reference")
    return arc, distance


def _interpolate(route: dict, arc: float) -> list[float]:
    anchors = route["centerline_anchor_sequence"]
    if not 0 <= arc <= anchors[-1]["arc_distance_m"]:
        raise LabelError("insufficient-label-horizon")
    for a, b in zip(anchors, anchors[1:]):
        if a["arc_distance_m"] <= arc <= b["arc_distance_m"]:
            fraction = (arc - a["arc_distance_m"]) / (b["arc_distance_m"] - a["arc_distance_m"])
            return [first + fraction * (second - first) for first, second in zip(_xyz(a), _xyz(b))]
    raise LabelError("insufficient-label-horizon")


def _local_point(world: list[float], matrix: list[list[float]]) -> list[float]:
    return [sum(matrix[k][axis] * (world[k] - matrix[k][3]) for k in range(3)) for axis in range(3)]


def _make_label(frame: dict, metadata: dict) -> dict:
    route = metadata["route_contract"]
    reference, _ = _reference(route, frame["vehicle"]["transform"]["location"])
    world = [_interpolate(route, reference + distance) for distance in HORIZON_ARC_M]
    matrix = frame["vehicle"]["transform"]["matrix"]
    label = {
        "schema_version": 1, "ego_frame": frame["frame"], "route_id": metadata["route_id"],
        "target_lane": {key: route["start"][key] for key in ("road_id", "section_id", "lane_id")},
        "coordinate_origin": "vehicle_actor_transform", "axes": "x-forward_y-right", "units": "m",
        "distance_reference": "target-lane-centerline-arc-length", "horizon_arc_m": list(HORIZON_ARC_M),
        "reference_arc_m": reference, "waypoints_world_m": world,
        "waypoints_ego_m": [_local_point(point, matrix)[:2] for point in world],
    }
    validate_waypoint_label(label, frame, metadata)
    return label


def generate_waypoint_label(frame: dict, metadata: dict) -> dict:
    """Pure single-frame helper; no image IO, CARLA imports or source mutation."""
    validate_episode_identity(metadata)
    validate_frame_contract(frame, metadata)
    if frame["label_state"] != "NOT_GENERATED" or frame["label"] is not None:
        raise LabelError("source-is-not-unlabelled")
    _simple_route(metadata["route_contract"])
    return _make_label(frame, metadata)


def _measure(label: dict, frame: dict, metadata: dict) -> dict:
    # The public D5 validator independently checks frame/target identity,
    # reference, route interpolation, rigid transform and output shape.
    validate_waypoint_label(label, frame, metadata)
    route, transform = metadata["route_contract"], frame["vehicle"]["transform"]
    reference, centerline_distance = _reference(route, transform["location"])
    matrix = transform["matrix"]
    world_error = ego_error = roundtrip_error = 0.
    for index, distance in enumerate(HORIZON_ARC_M):
        truth = _interpolate(route, reference + distance)
        world = label["waypoints_world_m"][index]
        local = _local_point(world, matrix)
        ego = label["waypoints_ego_m"][index]
        reconstructed = [sum(matrix[axis][k] * value for k, value in enumerate([ego[0], ego[1], local[2]])) + matrix[axis][3]
                         for axis in range(3)]
        world_error = max(world_error, math.dist(world, truth))
        ego_error = max(ego_error, math.dist(ego, local[:2]))
        roundtrip_error = max(roundtrip_error, math.dist(world, reconstructed))
    result = {"centerline_distance_xy_m": centerline_distance, "max_world_route_error_m": world_error,
              "max_ego_error_m": ego_error, "max_world_roundtrip_error_m": roundtrip_error,
              "max_reference_error_m": abs(reference - label["reference_arc_m"])}
    for key, value in result.items():
        if _number(value) < 0 or key in ERROR_KEYS and value > .001:
            raise LabelError("label-geometry-error")
    return result


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_source(episode_path: str | Path, contract_path: str | Path):
    root = Path(episode_path)
    contract = load_dataset_contract(contract_path)
    metadata_bytes, frames_bytes = (root / "episode.json").read_bytes(), (root / "frames.jsonl").read_bytes()
    validated = validate_episode(root, require_complete=True)
    if validated.get("status") != "PASS" or validated.get("errors") or validated.get("complete") is not True:
        raise LabelError("source-episode-integrity-failed")
    metadata = strict_json_loads(metadata_bytes.decode("utf-8"))
    if metadata != validated["metadata"] or metadata.get("status") != "COMPLETE":
        raise LabelError("source-metadata-changed")
    validate_episode_identity(metadata)
    if metadata["town"] == contract["split"]["reserved_final"]["town"]:
        raise LabelError("reserved-final-source-not-available")
    _simple_route(metadata["route_contract"])
    lines = frames_bytes.decode("utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise LabelError("source-frame-record-invalid")
    frames = []
    for index, line in enumerate(lines):
        frame = strict_json_loads(line)
        validate_frame_contract(frame, metadata, contract)
        if frame["index"] != index or frame["label_state"] != "NOT_GENERATED" or frame["label"] is not None:
            raise LabelError("source-frame-state-invalid")
        if frames and (frame["frame"] != frames[-1]["frame"] + 1 or not math.isclose(
                frame["timestamp"] - frames[-1]["timestamp"], frame["delta_seconds"], rel_tol=0, abs_tol=1e-6)):
            raise LabelError("source-frame-sequence-invalid")
        frames.append(frame)
    if len(frames) != validated["frame_count"] or len(frames) != metadata["frame_count"]:
        raise LabelError("source-frame-count-invalid")
    if metadata_bytes != (root / "episode.json").read_bytes() or frames_bytes != (root / "frames.jsonl").read_bytes():
        raise LabelError("source-changed-during-validation")
    source = {key: metadata[key] for key in ("dataset_version", "episode_id", "route_id", "scene_group_id", "town", "configuration_sha256")}
    source.update(frame_count=len(frames), first_frame=frames[0]["frame"], last_frame=frames[-1]["frame"],
                  route_sha256=route_sha256(metadata["route_contract"]), metadata_sha256=_sha(metadata_bytes),
                  frames_jsonl_sha256=_sha(frames_bytes))
    return metadata, frames, source, contract


def _metrics(rows: list[dict]) -> dict:
    return {"frame_count": len(rows), "first_frame": rows[0]["frame"], "last_frame": rows[-1]["frame"],
            "waypoint_shape": [5, 2], **{key: max(row["diagnostics"][key] for row in rows) for key in ERROR_KEYS}}


def _checked_destination(episode_path, output_path) -> Path:
    path = Path(output_path)
    if path.resolve().is_relative_to(Path(episode_path).resolve()):
        raise LabelError("label-path-inside-source")
    return path


def _base_report() -> dict:
    return {"status": "FAIL", "scope": SCOPE, "errors": []}


def _error(report: dict, error: Exception) -> dict:
    code = error.code if isinstance(error, (LabelError, ContractError)) else "label-operation-failed"
    report["errors"].append({"code": code, "exception_type": type(error).__name__})
    return report


def _same_numbers(recorded: dict, expected: dict) -> bool:
    _exact(recorded, set(expected), "invalid-label-diagnostics")
    return all(_number(recorded[key]) >= 0 and math.isclose(recorded[key], value, rel_tol=1e-7, abs_tol=1e-10)
               for key, value in expected.items())


def _is_pending(path: Path) -> bool:
    return path.name.startswith(".labels-") and path.suffix == ".pending"


def _validate_labels(episode_path: str | Path, labels_path: str | Path,
                     contract_path: str | Path, allow_pending: bool = False) -> dict[str, Any]:
    report = _base_report()
    try:
        path = _checked_destination(episode_path, labels_path)
        if _is_pending(path) and not allow_pending:
            raise LabelError("unpublished-label-artifact")
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise LabelError("invalid-label-path")
        metadata, frames, source, contract = _load_source(episode_path, contract_path)
        sidecar_bytes = path.read_bytes()
        sidecar = strict_json_loads(sidecar_bytes.decode("utf-8"))
        _exact(sidecar, {"schema_version", "status", "scope", "contract_version", "generation_version", "source", "rows", "metrics"},
               "invalid-label-sidecar")
        if (type(sidecar["schema_version"]) is not int or sidecar["schema_version"] != 1
                or sidecar["status"] != "COMPLETE" or sidecar["scope"] != SCOPE
                or sidecar["contract_version"] != contract["contract_version"] or sidecar["generation_version"] != GENERATION_VERSION):
            raise LabelError("unsupported-or-incomplete-label-sidecar")
        if canonical_sha256(sidecar["source"]) != canonical_sha256(source):
            raise LabelError("label-source-identity-or-hash-mismatch")
        rows = sidecar["rows"]
        if not isinstance(rows, list) or len(rows) != len(frames):
            raise LabelError("label-frame-count-mismatch")
        measured = []
        for index, (row, frame) in enumerate(zip(rows, frames)):
            _exact(row, {"index", "frame", "timestamp", "sample_id", "label_state", "image", "label", "diagnostics"}, "invalid-label-row")
            if (type(row["index"]) is not int or row["index"] != index or type(row["frame"]) is not int
                    or row["frame"] != frame["frame"] or _number(row["timestamp"]) != frame["timestamp"]
                    or row["sample_id"] != sample_id(metadata["episode_id"], frame["frame"])
                    or row["label_state"] != "GENERATED"
                    or row["image"] != {key: frame["image"][key] for key in ("path", "sha256")}):
                raise LabelError("label-frame-or-image-identity-mismatch")
            metrics = _measure(row["label"], frame, metadata)
            if not _same_numbers(row["diagnostics"], metrics):
                raise LabelError("label-diagnostics-mismatch")
            measured.append({**row, "diagnostics": metrics})
        expected = _metrics(measured)
        _exact(sidecar["metrics"], set(expected), "invalid-label-metrics")
        for key in ("frame_count", "first_frame", "last_frame"):
            if type(sidecar["metrics"][key]) is not int or sidecar["metrics"][key] != expected[key]:
                raise LabelError("label-metrics-mismatch")
        if sidecar["metrics"]["waypoint_shape"] != [5, 2] or any(type(value) is not int for value in sidecar["metrics"]["waypoint_shape"]):
            raise LabelError("label-metrics-mismatch")
        if not _same_numbers({key: sidecar["metrics"][key] for key in ERROR_KEYS}, {key: expected[key] for key in ERROR_KEYS}):
            raise LabelError("label-metrics-mismatch")
        # The reported digest must describe the bytes actually verified, and
        # the source cannot change while geometric checks are in progress.
        if path.read_bytes() != sidecar_bytes:
            raise LabelError("labels-changed-during-validation")
        _, _, fresh_source, _ = _load_source(episode_path, contract_path)
        if fresh_source != source:
            raise LabelError("source-changed-during-validation")
        report.update(status="PASS", source=source, metrics=expected, labels_sha256=_sha(sidecar_bytes),
                      metadata=metadata, frames=frames, sidecar=sidecar)
    except (LabelError, ContractError, OSError, UnicodeError, ValueError, TypeError, KeyError, IndexError) as error:
        _error(report, error)
    return report


def validate_labels(episode_path: str | Path, labels_path: str | Path,
                    contract_path: str | Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    """Fresh disk/source/hash/geometry validation; FAIL exposes no usable data."""
    return _validate_labels(episode_path, labels_path, contract_path)


def generate_labels(episode_path: str | Path, output_path: str | Path,
                    contract_path: str | Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    """Publish one newly validated sidecar, preserving every source byte.

Existing output is always refused, including a prior successful run. A failed
generation never reuses stale PASS data and removes its own pending file.
"""
    report, temporary, published, payload = _base_report(), None, False, None
    successful = None
    destination = None
    try:
        destination = _checked_destination(episode_path, output_path)
        if _is_pending(destination):
            raise LabelError("invalid-label-output-path")
        if destination.exists() or destination.is_symlink():
            raise LabelError("label-output-already-exists")
        metadata, frames, source, contract = _load_source(episode_path, contract_path)
        rows = []
        for frame in frames:
            label = _make_label(frame, metadata)
            rows.append({"index": frame["index"], "frame": frame["frame"], "timestamp": frame["timestamp"],
                         "sample_id": sample_id(metadata["episode_id"], frame["frame"]), "label_state": "GENERATED",
                         "image": {key: frame["image"][key] for key in ("path", "sha256")}, "label": label,
                         "diagnostics": _measure(label, frame, metadata)})
        sidecar = {"schema_version": 1, "status": "COMPLETE", "scope": SCOPE, "contract_version": contract["contract_version"],
                   "generation_version": GENERATION_VERSION, "source": source, "rows": rows, "metrics": _metrics(rows)}
        payload = (json.dumps(sidecar, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".labels-", suffix=".pending", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        checked = _validate_labels(episode_path, temporary, contract_path, allow_pending=True)
        if checked["status"] != "PASS":
            raise LabelError(checked["errors"][0]["code"])
        os.link(temporary, destination)
        published = True
        # Publication includes removing the internal name. If that fails, the
        # exception path removes our published artifact before returning FAIL.
        temporary.unlink()
        temporary = None
        checked = validate_labels(episode_path, destination, contract_path)
        if checked["status"] != "PASS":
            raise LabelError(checked["errors"][0]["code"])
        successful = {**checked, "written": True}
    except (LabelError, ContractError, OSError, UnicodeError, ValueError, TypeError, KeyError, IndexError) as error:
        if published and destination is not None:
            try:
                if destination.read_bytes() == payload:
                    destination.unlink()  # Only our new, unchanged artifact.
            except OSError:
                report["errors"].append({"code": "failed-label-artifact-cleanup-failed"})
        _error(report, error)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                report["errors"].append({"code": "label-pending-cleanup-failed"})
                successful = None
    return successful if successful is not None else report
