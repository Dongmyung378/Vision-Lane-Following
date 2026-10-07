"""Write and independently validate small, unlabelled D4 RGB episodes.

Images are published exclusively before their JSONL records. Episode metadata
starts RUNNING and can become COMPLETE only after capture and cleanup succeed.
There is no CARLA import and no waypoint or lane-label generation here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any

from .config import ConfigError, collection_config_from_dict


SCHEMA_VERSION = 1
TIME_TOLERANCE = 1e-6
MOUNT_TOLERANCE = 0.001
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_END = b"\x00\x00\x00\x00IEND\xaeB`\x82"
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class EpisodeError(RuntimeError):
    """A stable public error code without raw filesystem or API diagnostics."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _copy_json(value: Any) -> Any:
    def inspect(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                inspect(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                inspect(child)
            return
        raise EpisodeError("non-json-or-nonfinite-value")

    try:
        inspect(value)
        return json.loads(json.dumps(value, allow_nan=False))
    except EpisodeError:
        raise
    except (ValueError, TypeError, RecursionError):
        raise EpisodeError("non-json-or-nonfinite-value") from None


def _strict_json(text: str) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise EpisodeError("duplicate-json-key")
            result[key] = value
        return result

    def invalid(_):
        raise EpisodeError("non-json-or-nonfinite-value")

    try:
        return _copy_json(json.loads(text, object_pairs_hook=unique, parse_constant=invalid))
    except EpisodeError:
        raise
    except (ValueError, TypeError, RecursionError):
        raise EpisodeError("invalid-json") from None


def _integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _matrix(value: Any) -> bool:
    return (type(value) is list and len(value) == 4 and
            all(type(row) is list and len(row) == 4 and all(_number(cell) for cell in row) for row in value))


def _vector(value: Any, keys: tuple[str, ...]) -> bool:
    return type(value) is dict and all(key in value and _number(value[key]) for key in keys)


def _transform(value: Any) -> bool:
    return (type(value) is dict and _vector(value.get("location"), ("x", "y", "z")) and
            _vector(value.get("rotation"), ("pitch", "yaw", "roll")) and _matrix(value.get("matrix")))


def _pose_matrix(location: Any, rotation: Any) -> list[list[float]]:
    pitch, yaw, roll = (math.radians(rotation[key]) for key in ("pitch", "yaw", "roll"))
    cp, sp, cy, sy, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw), math.cos(roll), math.sin(roll)
    return [
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, location["x"]],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, location["y"]],
        [sp, -cp * sr, cp * cr, location["z"]], [0.0, 0.0, 0.0, 1.0],
    ]


def _matrices_close(first: list, second: list, tolerance: float) -> bool:
    differences = [abs(first[r][c] - second[r][c]) for r in range(4) for c in range(4)]
    return all(math.isfinite(value) and value <= tolerance for value in differences)


def _is_link(path: Path) -> bool:
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def _atomic_bytes(path: Path, content: bytes, *, exclusive: bool) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pending", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            # A same-volume hard link publishes complete bytes atomically and
            # refuses an existing destination on Windows and POSIX.
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict, *, exclusive: bool = False) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _atomic_bytes(path, text.encode("utf-8"), exclusive=exclusive)


def _png_dimensions(data: bytes, expected_dimensions: tuple[int, int] | None = None) -> tuple[int, int]:
    """Verify the complete non-interlaced RGB8 stream with bounded inflation.

    The dimension bounds match the collection configuration. Checking the
    expected dimensions before inflation prevents a foreign header from
    choosing a larger allocation than the configured camera requires.
    """
    if (len(data) < 45 or not data.startswith(PNG_SIGNATURE) or
            data[8:12] != b"\x00\x00\x00\r" or data[12:16] != b"IHDR"):
        raise EpisodeError("invalid-png-header")
    if zlib.crc32(data[12:29]) & 0xFFFFFFFF != struct.unpack(">I", data[29:33])[0]:
        raise EpisodeError("invalid-png-header")
    width, height, bit_depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", data[16:29])
    if not 1 <= width <= 4096 or not 1 <= height <= 2160 or (bit_depth, color, compression, filtering, interlace) != (8, 2, 0, 0, 0):
        raise EpisodeError("invalid-png-format")
    if expected_dimensions is not None and (width, height) != expected_dimensions:
        raise EpisodeError("png-dimensions-mismatch")
    row_bytes = 1 + width * 3
    expected_bytes = height * row_bytes
    # Our encoder uses one zlib stream. Permit its worst-case overhead plus a
    # modest margin, while bounding both compressed and ancillary file data.
    compressed_limit = expected_bytes + expected_bytes // 100 + 1024
    if len(data) > compressed_limit + 65536:
        raise EpisodeError("invalid-png-data")
    compressed = bytearray()
    offset = 8
    seen_header = seen_palette = seen_data = ended_data = seen_end = False
    view = memoryview(data)
    while offset < len(data):
        if offset + 12 > len(data):
            raise EpisodeError("invalid-png-structure")
        length = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4:offset + 8]
        end = offset + 12 + length
        if (length > 0x7FFFFFFF or end > len(data) or
                any(not (65 <= value <= 90 or 97 <= value <= 122) for value in kind) or
                kind[2] & 0x20):
            raise EpisodeError("invalid-png-structure")
        payload = view[offset + 8:end - 4]
        crc = zlib.crc32(payload, zlib.crc32(kind)) & 0xFFFFFFFF
        if crc != struct.unpack_from(">I", data, end - 4)[0]:
            raise EpisodeError("invalid-png-header" if kind == b"IHDR" else "invalid-png-crc")
        if kind == b"IHDR":
            if seen_header or offset != 8 or length != 13:
                raise EpisodeError("invalid-png-structure")
            seen_header = True
        elif kind == b"PLTE":
            if seen_palette or seen_data or not length or length % 3 or length > 768:
                raise EpisodeError("invalid-png-structure")
            seen_palette = True
        elif kind == b"IDAT":
            if ended_data or len(compressed) + length > compressed_limit:
                raise EpisodeError("invalid-png-structure")
            seen_data = True
            compressed.extend(payload)
        elif kind == b"IEND":
            if length or not seen_data or end != len(data):
                raise EpisodeError("invalid-png-structure")
            seen_end = True
        else:
            # Unknown critical chunks change the interpretation of the image.
            if not kind[0] & 0x20:
                raise EpisodeError("invalid-png-structure")
            if seen_data:
                ended_data = True
        offset = end
    if not seen_header or not seen_data or not seen_end:
        raise EpisodeError("invalid-png-structure")
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, expected_bytes + 1)
    except zlib.error:
        raise EpisodeError("invalid-png-data") from None
    if (len(raw) != expected_bytes or not decoder.eof or decoder.unused_data or
            decoder.unconsumed_tail or any(value > 4 for value in raw[::row_bytes])):
        raise EpisodeError("invalid-png-data")
    return width, height


def _validate_metadata(metadata: Any):
    required = {
        "schema_version", "status", "dataset_version", "episode_id", "route_id", "scene_group_id", "town",
        "expected_frames", "frame_count", "collection_config", "configuration_sha256", "camera_calibration",
        "applied_manifest", "perturbation", "control_mode", "labels_state",
    }
    if type(metadata) is not dict or not required <= metadata.keys() or metadata["schema_version"] != 1 or type(metadata["schema_version"]) is not int:
        raise EpisodeError("metadata-schema-invalid")
    for key in ("dataset_version", "episode_id", "route_id", "scene_group_id"):
        if type(metadata[key]) is not str or not IDENTIFIER.fullmatch(metadata[key]):
            raise EpisodeError("metadata-identity-invalid")
    if metadata["status"] not in ("RUNNING", "COMPLETE", "FAILED"):
        raise EpisodeError("metadata-status-invalid")
    if (not _integer(metadata["expected_frames"]) or metadata["expected_frames"] == 0 or
            not _integer(metadata["frame_count"])):
        raise EpisodeError("metadata-frame-count-invalid")
    try:
        cfg = collection_config_from_dict(metadata["collection_config"])
    except ConfigError:
        raise EpisodeError("collection-config-invalid") from None
    if (metadata["configuration_sha256"] != cfg.sha256 or metadata["town"] != cfg.map or
            metadata["expected_frames"] != cfg.verification.frames):
        raise EpisodeError("configuration-identity-mismatch")
    calibration = metadata["camera_calibration"]
    if type(calibration) is not dict or not _matrix(calibration.get("requested_mount_matrix")):
        raise EpisodeError("camera-calibration-invalid")
    mount = cfg.camera.mount.to_dict()
    if not _matrices_close(calibration["requested_mount_matrix"], _pose_matrix(mount["location"], mount["rotation"]), 1e-6):
        raise EpisodeError("camera-calibration-invalid")
    if calibration.get("coordinate_convention") != "carla-x-forward-y-right-z-up" or calibration.get("rotation_units") != "degrees":
        raise EpisodeError("camera-calibration-invalid")
    intrinsics = calibration.get("intrinsics")
    if type(intrinsics) is not dict or any(
            not _number(intrinsics.get(key)) or not math.isclose(intrinsics[key], value, rel_tol=0, abs_tol=TIME_TOLERANCE)
            for key, value in cfg.camera.intrinsics().items()):
        raise EpisodeError("camera-calibration-invalid")
    if "K" in calibration:
        expected_k = [[intrinsics["fx"], 0.0, intrinsics["cx"]], [0.0, intrinsics["fy"], intrinsics["cy"]], [0.0, 0.0, 1.0]]
        observed_k = calibration["K"]
        if (type(observed_k) is not list or len(observed_k) != 3 or
                any(type(row) is not list or len(row) != 3 for row in observed_k) or
                any(not _number(observed_k[r][c]) or not math.isclose(observed_k[r][c], expected_k[r][c], rel_tol=0, abs_tol=TIME_TOLERANCE) for r in range(3) for c in range(3))):
            raise EpisodeError("camera-calibration-invalid")
    for key, expected in (("width", cfg.camera.width), ("height", cfg.camera.height)):
        if key in calibration and (type(calibration[key]) is not int or calibration[key] != expected):
            raise EpisodeError("camera-calibration-invalid")
    if "distortion" in calibration:
        distortion = calibration["distortion"]
        if type(distortion) is not dict or any(not _number(distortion.get(key)) or distortion[key] != 0 for key in ("lens_k", "lens_kcube")):
            raise EpisodeError("camera-calibration-invalid")
    if type(metadata["applied_manifest"]) is not dict:
        raise EpisodeError("applied-manifest-invalid")
    perturbation = metadata["perturbation"]
    if (type(perturbation) is not dict or any(not _number(perturbation.get(key)) or perturbation[key] != 0 for key in ("lateral_m", "yaw_deg")) or
            type(perturbation.get("provenance")) is not str or not perturbation["provenance"]):
        raise EpisodeError("perturbation-invalid")
    if metadata["control_mode"] != "stationary_D4_smoke" or metadata["labels_state"] != "not-generated-D6":
        raise EpisodeError("episode-scope-invalid")
    return cfg


def _cleanup_complete(cleanup: Any) -> bool:
    return type(cleanup) is dict and cleanup.get("status") == "PASS" and all(
        cleanup.get(key) is True for key in ("sensor_stopped", "settings_restored", "weather_restored", "owned_actors_absent")
    )


def _validate_row(row: Any, metadata: dict, cfg: Any, index: int, previous: dict | None) -> None:
    required = {"schema_version", "index", "frame", "world_frame", "snapshot_frame", "sensor_frame",
                "timestamp", "snapshot_timestamp", "sensor_timestamp", "delta_seconds", "image", "vehicle", "camera"}
    if type(row) is not dict or not required <= row.keys() or type(row["schema_version"]) is not int or row["schema_version"] != 1:
        raise EpisodeError("frame-schema-invalid")
    if not _integer(row["index"]) or row["index"] != index or not _integer(row["frame"]):
        raise EpisodeError("frame-index-invalid")
    if any(not _integer(row[key]) or row[key] != row["frame"] for key in ("world_frame", "snapshot_frame", "sensor_frame")):
        raise EpisodeError("frame-id-mismatch")
    if previous is not None:
        if row["frame"] == previous["frame"]:
            raise EpisodeError("duplicate-frame")
        if row["frame"] != previous["frame"] + 1:
            raise EpisodeError("dropped-or-unordered-frame")
    if (any(not _number(row[key]) or row[key] < 0 for key in ("timestamp", "snapshot_timestamp", "sensor_timestamp")) or
            any(not math.isclose(row[key], row["timestamp"], rel_tol=0, abs_tol=TIME_TOLERANCE) for key in ("snapshot_timestamp", "sensor_timestamp"))):
        raise EpisodeError("timestamp-mismatch")
    if not _number(row["delta_seconds"]) or not math.isclose(row["delta_seconds"], cfg.world.fixed_delta_seconds, rel_tol=0, abs_tol=TIME_TOLERANCE):
        raise EpisodeError("timestep-mismatch")
    if previous is not None and not math.isclose(row["timestamp"] - previous["timestamp"], cfg.world.fixed_delta_seconds, rel_tol=0, abs_tol=TIME_TOLERANCE):
        raise EpisodeError("timestamp-not-consecutive")
    image = row["image"]
    if type(image) is not dict or not {"path", "sha256", "width", "height"} <= image.keys():
        raise EpisodeError("image-schema-invalid")
    if image["path"] != f"images/{row['frame']:08d}.png":
        raise EpisodeError("image-path-invalid")
    if type(image["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", image["sha256"]):
        raise EpisodeError("image-hash-invalid")
    if type(image["width"]) is not int or type(image["height"]) is not int or (image["width"], image["height"]) != (cfg.camera.width, cfg.camera.height):
        raise EpisodeError("image-dimensions-mismatch")
    vehicle, camera = row["vehicle"], row["camera"]
    if (type(vehicle) is not dict or type(camera) is not dict or not _transform(vehicle.get("transform")) or
            not _transform(camera.get("transform")) or
            any(not _vector(vehicle.get(key), ("x", "y", "z")) for key in ("velocity", "angular_velocity", "acceleration"))):
        raise EpisodeError("frame-pose-invalid")
    for transform in (vehicle["transform"], camera["transform"]):
        if not _matrices_close(transform["matrix"], _pose_matrix(transform["location"], transform["rotation"]), 1e-4):
            raise EpisodeError("frame-pose-matrix-mismatch")
    control = vehicle.get("control")
    if type(control) is not dict or any(control.get(key) != value or type(control.get(key)) not in (int, float, bool) for key, value in cfg.vehicle.control.to_dict().items()):
        raise EpisodeError("vehicle-control-mismatch")
    if any(type(control.get(key)) is not bool for key in ("hand_brake", "reverse")) or any(not _number(control.get(key)) for key in ("throttle", "steer", "brake")):
        raise EpisodeError("vehicle-control-mismatch")
    # Relative rigid mount = inverse(vehicle world) @ camera world. Both poses
    # originate from the record's same sensor/world frame, not live actor state.
    vehicle_matrix = vehicle["transform"]["matrix"]
    camera_matrix = camera["transform"]["matrix"]
    inverse = [[vehicle_matrix[col][axis] for col in range(3)] + [0.0] for axis in range(3)] + [[0.0, 0.0, 0.0, 1.0]]
    for axis in range(3):
        inverse[axis][3] = -sum(inverse[axis][col] * vehicle_matrix[col][3] for col in range(3))
    actual = [[sum(inverse[r][k] * camera_matrix[k][c] for k in range(4)) for c in range(4)] for r in range(4)]
    requested = metadata["camera_calibration"]["requested_mount_matrix"]
    residuals = [abs(actual[r][c] - requested[r][c]) for r in range(4) for c in range(4)]
    if any(not math.isfinite(item) for item in residuals) or max(residuals) > MOUNT_TOLERANCE:
        raise EpisodeError("camera-mount-mismatch")


class EpisodeWriter:
    def __init__(self, destination: str | Path, metadata: dict):
        self.path = Path(destination)
        self._metadata = _copy_json(metadata)
        if type(self._metadata) is not dict:
            raise EpisodeError("metadata-schema-invalid")
        self._metadata.update(schema_version=SCHEMA_VERSION, status="RUNNING", frame_count=0)
        self._cfg = _validate_metadata(self._metadata)
        self._previous = None
        self._count = 0
        try:
            if self.path.exists():
                if _is_link(self.path) or not self.path.is_dir() or any(self.path.iterdir()):
                    raise EpisodeError("destination-not-empty")
            else:
                self.path.mkdir(parents=True, exist_ok=False)
            (self.path / "images").mkdir(exist_ok=False)
            with (self.path / "frames.jsonl").open("x", encoding="utf-8"):
                pass
            _atomic_json(self.path / "episode.json", self._metadata, exclusive=True)
        except EpisodeError:
            raise
        except OSError:
            raise EpisodeError("episode-initialization-io-failed") from None

    @property
    def metadata(self) -> dict:
        return _copy_json(self._metadata)

    def update_metadata(self, fields: dict) -> dict:
        if self._metadata["status"] != "RUNNING":
            raise EpisodeError("episode-not-running")
        supplied = _copy_json(fields)
        if type(supplied) is not dict or not supplied.keys() <= {"applied_manifest", "camera_calibration"}:
            raise EpisodeError("metadata-update-forbidden")
        candidate = self.metadata
        candidate.update(supplied)
        _validate_metadata(candidate)
        try:
            _atomic_json(self.path / "episode.json", candidate)
        except OSError:
            self._fail("metadata-write-failed")
            raise EpisodeError("metadata-write-failed") from None
        self._metadata = candidate
        return self.metadata

    def _fail(self, code: str) -> None:
        self._metadata.update(status="FAILED", error={"code": code}, frame_count=self._count)
        try:
            _atomic_json(self.path / "episode.json", self._metadata)
        except OSError:
            pass  # A previously RUNNING file remains unusable as COMPLETE.

    def append(self, record: dict, png_bytes: bytes) -> dict:
        if self._metadata["status"] != "RUNNING":
            raise EpisodeError("episode-not-running")
        try:
            if self._count >= self._metadata["expected_frames"]:
                raise EpisodeError("frame-count-exceeded")
            row = _copy_json(record)
            if type(row) is not dict or type(row.get("image")) is not dict or not _integer(row.get("frame")):
                raise EpisodeError("frame-schema-invalid")
            if type(png_bytes) is not bytes:
                raise EpisodeError("invalid-png-input")
            row.setdefault("index", self._count)
            row["image"].update(path=f"images/{row['frame']:08d}.png", sha256=hashlib.sha256(png_bytes).hexdigest())
            _validate_row(row, self._metadata, self._cfg, self._count, self._previous)
            _png_dimensions(png_bytes, (row["image"]["width"], row["image"]["height"]))
            _atomic_bytes(self.path / row["image"]["path"], png_bytes, exclusive=True)
            with (self.path / "frames.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._count += 1
            self._previous = row
            self._metadata["frame_count"] = self._count
            _atomic_json(self.path / "episode.json", self._metadata)
            return _copy_json(row)
        except EpisodeError as error:
            self._fail(error.code)
            raise
        except OSError:
            self._fail("frame-write-failed")
            raise EpisodeError("frame-write-failed") from None

    def finalize(self, capture_succeeded: bool, cleanup: dict, error: dict | None = None) -> dict:
        cleanup = _copy_json(cleanup)
        supplied_error = _copy_json(error) if error is not None else None
        if type(capture_succeeded) is not bool:
            raise EpisodeError("capture-result-invalid")
        complete = (capture_succeeded and supplied_error is None and self._metadata["status"] != "FAILED" and
                    self._count == self._metadata["expected_frames"] and _cleanup_complete(cleanup) and
                    self._metadata["applied_manifest"].get("applied_verified") is True)
        self._metadata.update(status="COMPLETE" if complete else "FAILED", cleanup=cleanup,
                              capture_succeeded=capture_succeeded, frame_count=self._count)
        if supplied_error is not None:
            self._metadata["error"] = supplied_error
        elif not complete:
            self._metadata.setdefault("error", {"code": "capture-or-cleanup-incomplete"})
        try:
            _atomic_json(self.path / "episode.json", self._metadata)
        except OSError:
            self._fail("metadata-write-failed")
            raise EpisodeError("metadata-write-failed") from None
        return self.metadata


def validate_episode(path: str | Path, require_complete: bool = True) -> dict:
    """Re-read persisted records and images; return stable integrity errors."""
    root = Path(path)
    report: dict[str, Any] = {"status": "FAIL", "errors": [], "frame_count": 0, "metadata": None, "complete": False}

    def error(code: str, frame: int | None = None):
        item = {"code": code}
        if frame is not None:
            item["frame"] = frame
        report["errors"].append(item)

    if _is_link(root):
        error("episode-path-invalid")
        return report
    try:
        metadata = _strict_json((root / "episode.json").read_text(encoding="utf-8"))
        cfg = _validate_metadata(metadata)
        report["metadata"] = metadata
        report["complete"] = metadata["status"] == "COMPLETE"
    except EpisodeError as failure:
        error(failure.code)
        return report
    except (OSError, UnicodeError):
        error("metadata-read-failed")
        return report
    if require_complete and not report["complete"]:
        error("episode-incomplete")
    if report["complete"] and (metadata.get("capture_succeeded") is not True or
            not _cleanup_complete(metadata.get("cleanup")) or metadata["applied_manifest"].get("applied_verified") is not True):
        error("completion-proof-incomplete")
    images_root = root / "images"
    if _is_link(images_root):
        error("image-path-invalid")
        return report
    try:
        lines = (root / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        error("frames-read-failed")
        return report
    previous = None
    seen = set()
    referenced = set()
    for index, line in enumerate(lines):
        frame = None
        try:
            row = _strict_json(line)
            if type(row) is dict and _integer(row.get("frame")):
                frame = row["frame"]
            if frame is not None and frame in seen:
                raise EpisodeError("duplicate-frame")
            _validate_row(row, metadata, cfg, index, previous)
            image_path = root / row["image"]["path"]
            if _is_link(image_path) or image_path.resolve().parent != images_root.resolve():
                raise EpisodeError("image-path-invalid")
            referenced.add(image_path.name)
            png = image_path.read_bytes()
            if hashlib.sha256(png).hexdigest() != row["image"]["sha256"]:
                raise EpisodeError("image-hash-mismatch")
            _png_dimensions(png, (row["image"]["width"], row["image"]["height"]))
            seen.add(frame)
            previous = row
        except EpisodeError as failure:
            error(failure.code, frame)
        except (OSError, UnicodeError):
            error("image-read-failed", frame)
    report["frame_count"] = len(lines)
    if len(lines) != metadata["frame_count"] or (report["complete"] and len(lines) != metadata["expected_frames"]):
        error("frame-count-mismatch")
    try:
        if not images_root.is_dir():
            error("images-directory-missing")
        elif any(item.name not in referenced for item in images_root.iterdir()):
            error("unreferenced-image-or-artifact")
    except OSError:
        error("images-directory-read-failed")
    report["status"] = "PASS" if not report["errors"] else "FAIL"
    return report
