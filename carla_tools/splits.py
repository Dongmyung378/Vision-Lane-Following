"""Build a D5 smoke-data split only after complete disk-integrity validation.

Assignments are at episode level. Frame, route, scene-group, time and physical
overlap checks prevent nearby or recovery variants from crossing train/dev.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (DEFAULT_CONTRACT_PATH, ContractError, load_dataset_contract, read_strict_json,
                        route_sha256, sample_id, strict_json_loads, validate_episode_identity,
                        validate_frame_contract, validate_identifier)


CSV_FIELDS = ["dataset_version", "sample_id", "route_id", "episode_id", "scene_group_id", "split"]


def _point_segment_distance(point, first, second):
    dx, dy = second[0] - first[0], second[1] - first[1]
    denominator = dx * dx + dy * dy
    if not denominator:
        return math.dist(point, first)
    fraction = max(0., min(1., ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy) / denominator))
    return math.hypot(point[0] - first[0] - fraction * dx, point[1] - first[1] - fraction * dy)


def _orientation(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segment_distance(a, b, c, d):
    first, second = _orientation(a, b, c), _orientation(a, b, d)
    third, fourth = _orientation(c, d, a), _orientation(c, d, b)
    if first * second < 0 and third * fourth < 0:
        return 0.
    return min(_point_segment_distance(a, c, d), _point_segment_distance(b, c, d),
               _point_segment_distance(c, a, b), _point_segment_distance(d, a, b))


def scene_distance_m(first: dict, second: dict) -> float:
    """XY segment separation catches opposite directions and crossed paths.

Each internal scene contains its surveyed corridor and observed actor origins.
Different towns have distinct geometry and return infinity.
"""
    if first["town"] != second["town"]:
        return math.inf
    return min(_segment_distance(a, b, c, d) for a, b in first["segments"] for c, d in second["segments"])


def _scene(metadata: dict, frames: list[dict]) -> dict:
    anchors = metadata["route_contract"]["centerline_anchor_sequence"]
    points = [(p["world_location"]["x"], p["world_location"]["y"]) for p in anchors]
    segments = list(zip(points, points[1:]))
    poses = {(row["vehicle"]["transform"]["location"]["x"], row["vehicle"]["transform"]["location"]["y"]) for row in frames}
    segments.extend((point, point) for point in poses)
    return {"town": metadata["town"], "segments": segments,
            "time_start": frames[0]["timestamp"], "time_end": frames[-1]["timestamp"]}


def _load_frames(root: Path, metadata: dict, contract: dict) -> list[dict]:
    try:
        lines = (root / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ContractError("frames-read-failed") from error
    if not lines or any(not line.strip() for line in lines):
        raise ContractError("empty-or-blank-frame-record")
    frames, ids = [], set()
    for index, line in enumerate(lines):
        row = strict_json_loads(line)
        validate_frame_contract(row, metadata, contract)
        if row["label_state"] != "NOT_GENERATED":
            raise ContractError("D5-frame-labels-not-generated")
        if row["index"] != index or row["frame"] in ids:
            raise ContractError("duplicate-or-unordered-frames")
        if frames:
            if row["frame"] != frames[-1]["frame"] + 1:
                raise ContractError("nonconsecutive-frames")
            elapsed = row["timestamp"] - frames[-1]["timestamp"]
            if not math.isclose(elapsed, row["delta_seconds"], rel_tol=0, abs_tol=1e-5):
                raise ContractError("nonconsecutive-frame-time")
        ids.add(row["frame"])
        frames.append(row)
    if len(frames) != metadata["frame_count"] or len(frames) != metadata["expected_frames"]:
        raise ContractError("frame-count-mismatch")
    return frames


def _existing_rows(path: Path) -> list[dict]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != CSV_FIELDS:
                raise ContractError("invalid-existing-manifest")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ContractError("existing-manifest-read-failed") from error
    if not rows or any(set(row) != set(CSV_FIELDS) or any(not isinstance(value, str) or not value for value in row.values()) for row in rows):
        raise ContractError("invalid-existing-manifest")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ContractError("invalid-existing-manifest")
    return rows


def _write_manifest(path: Path, rows: list[dict]) -> tuple[bool, str]:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = text.getvalue().encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        existing = _existing_rows(path)
        if {row["dataset_version"] for row in existing} != {rows[0]["dataset_version"]}:
            raise ContractError("existing-manifest-version-conflict")
        if existing != rows:
            raise ContractError("existing-manifest-content-conflict")
        return False, hashlib.sha256(path.read_bytes()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".split-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-volume hard-link publication is atomic and refuses an existing
        # destination on Windows and POSIX, matching the episode writer.
        os.link(temporary, path)
    except FileExistsError as error:
        raise ContractError("concurrent-manifest-conflict") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True, digest


def build_split_manifest(episode_paths: Sequence[str | Path], output_csv: str | Path,
                         assignments: Mapping[str, str], contract_path: str | Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    """Return PASS/FAIL and write only a fully validated, nonconflicting manifest.

``assignments`` maps every included episode_id to train or dev. This is D5's
stationary smoke manifest, not a training-pilot or final-test dataset claim.
Existing identical content is reusable; version or assignment changes require
a new destination. Episode sources are never edited.
"""
    report: dict[str, Any] = {"status": "FAIL", "scope": "D5-stationary-smoke-split-not-training-pilot",
                              "errors": [], "episode_count": 0, "frame_count": 0, "written": False,
                              "final_test": {"town": "Town02", "collection_status": "UNCOLLECTED", "verification_status": "UNVERIFIED"}}
    try:
        from .episode import validate_episode

        contract = load_dataset_contract(contract_path)
        if isinstance(episode_paths, (str, bytes)) or not isinstance(episode_paths, Sequence) or not episode_paths:
            raise ContractError("no-episode-paths")
        if not isinstance(assignments, Mapping) or not assignments:
            raise ContractError("invalid-split-assignments")
        for episode_id, split in assignments.items():
            validate_identifier(episode_id)
            if split not in contract["split"]["allowed"]:
                raise ContractError("unsupported-D5-split")
        roots, episodes, seen_episodes = set(), [], set()
        dataset_version = None
        for source in episode_paths:
            root = Path(source).resolve()
            if root in roots:
                raise ContractError("duplicate-episode-path")
            roots.add(root)
            validated = validate_episode(root, require_complete=True)
            if validated.get("status") != "PASS" or validated.get("errors"):
                raise ContractError("episode-integrity-validation-failed")
            metadata = validated.get("metadata")
            if not isinstance(metadata, dict) or metadata.get("status") != "COMPLETE":
                raise ContractError("episode-not-complete")
            # A changed metadata file between integrity validation and reading is refused.
            if read_strict_json(root / "episode.json") != metadata:
                raise ContractError("episode-metadata-changed")
            validate_episode_identity(metadata)
            if metadata["town"] == contract["split"]["reserved_final"]["town"]:
                raise ContractError("reserved-final-town-not-available-D5")
            identifier = metadata["episode_id"]
            if identifier in seen_episodes:
                raise ContractError("duplicate-episode-id")
            if identifier not in assignments:
                raise ContractError("missing-episode-assignment")
            seen_episodes.add(identifier)
            if dataset_version is None:
                dataset_version = metadata["dataset_version"]
            elif dataset_version != metadata["dataset_version"]:
                raise ContractError("mixed-dataset-versions")
            frames = _load_frames(root, metadata, contract)
            if validated.get("frame_count") != len(frames):
                raise ContractError("validated-frame-count-mismatch")
            episodes.append({"metadata": metadata, "frames": frames, "split": assignments[identifier], "scene": _scene(metadata, frames)})
        if seen_episodes != set(assignments):
            raise ContractError("unused-episode-assignment")
        route_hashes, route_splits, scene_splits, map_sources = {}, {}, {}, {}
        for episode in episodes:
            meta, split = episode["metadata"], episode["split"]
            digest = route_sha256(meta["route_contract"])
            route_id, scene_id = meta["route_id"], meta["scene_group_id"]
            if route_id in route_hashes and route_hashes[route_id] != digest:
                raise ContractError("conflicting-route-contract")
            route_hashes[route_id] = digest
            route = meta["route_contract"]
            # Both accepted native package prefixes denote the same town; the
            # version and actual OpenDRIVE bytes establish coordinate identity.
            map_source = (route["carla_version"], route["opendrive_sha256"])
            if meta["town"] in map_sources and map_sources[meta["town"]] != map_source:
                raise ContractError("town-map-source-conflict")
            map_sources[meta["town"]] = map_source
            if route_id in route_splits and route_splits[route_id] != split:
                raise ContractError("route-split-leakage")
            if scene_id in scene_splits and scene_splits[scene_id] != split:
                raise ContractError("scene-group-split-leakage")
            route_splits[route_id], scene_splits[scene_id] = split, split
        minimum_distance = math.inf
        for index, first in enumerate(episodes):
            for second in episodes[index + 1:]:
                if first["split"] == second["split"]:
                    continue
                distance = scene_distance_m(first["scene"], second["scene"])
                minimum_distance = min(distance, minimum_distance)
                if distance <= contract["split"]["same_town_scene_buffer_m"]:
                    gap = max(first["scene"]["time_start"], second["scene"]["time_start"]) - min(first["scene"]["time_end"], second["scene"]["time_end"])
                    if gap <= contract["split"]["temporal_adjacency_seconds"]:
                        raise ContractError("spatial-and-temporal-split-leakage")
                    raise ContractError("spatial-scene-split-leakage")
        rows, unique_samples = [], set()
        for episode in episodes:
            meta = episode["metadata"]
            for frame in episode["frames"]:
                identifier = sample_id(meta["episode_id"], frame["frame"])
                if identifier in unique_samples:
                    raise ContractError("duplicate-sample-id")
                unique_samples.add(identifier)
                rows.append({"dataset_version": dataset_version, "sample_id": identifier, "route_id": meta["route_id"],
                             "episode_id": meta["episode_id"], "scene_group_id": meta["scene_group_id"], "split": episode["split"]})
        rows.sort(key=lambda row: (row["episode_id"], row["sample_id"]))
        written, digest = _write_manifest(Path(output_csv), rows)
        report.update(status="PASS", contract_version=contract["contract_version"], dataset_version=dataset_version,
                      episode_count=len(episodes), frame_count=len(rows), written=written, manifest_sha256=digest,
                      split_counts={split: sum(row["split"] == split for row in rows) for split in contract["split"]["allowed"]},
                      minimum_cross_split_scene_distance_m=None if math.isinf(minimum_distance) else minimum_distance,
                      leakage_checks={"complete_disk_integrity": True, "episode_assignment": True, "route_assignment": True,
                                      "scene_group_assignment": True, "spatial_and_temporal_adjacency": True,
                                      "fixed_target_route_identity": True, "no_final_test_samples": True})
    except ContractError as error:
        report["errors"].append({"code": error.code})
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, ImportError) as error:
        report["errors"].append({"code": "split-build-operation-failed", "exception_type": type(error).__name__})
    return report
