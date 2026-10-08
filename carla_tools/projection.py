"""Same-frame pinhole reprojection and lossless RGB8 PNG overlays.

CARLA camera-local (forward, right, up) becomes optical (right, down,
forward): (x, y, z) -> (y, -z, x). Points outside the image retain their
unclamped coordinates and are never drawn at an invented border pixel.
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

from .episode import EpisodeError, _png_dimensions


HORIZONS = (3, 6, 9, 12, 15)
COLORS_RGB = ((255, 80, 80), (255, 180, 0), (60, 220, 100), (60, 220, 255), (200, 100, 255))
NEAR_PLANE_M = 1e-6
PROJECTION_VERSION = "vlf-pinhole-overlay-d6-v1"


class ProjectionError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _finite(value: Any) -> float:
    try:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ProjectionError("projection-nonfinite-number")
        return float(value)
    except (OverflowError, TypeError):
        raise ProjectionError("projection-nonfinite-number") from None


def _dimensions(width: Any, height: Any) -> tuple[int, int]:
    if type(width) is not int or type(height) is not int or not 1 <= width <= 4096 or not 1 <= height <= 2160:
        raise ProjectionError("projection-invalid-dimensions")
    return width, height


def _rigid_matrix(value: Any) -> list[list[float]]:
    if type(value) is not list or len(value) != 4 or any(type(row) is not list or len(row) != 4 for row in value):
        raise ProjectionError("projection-invalid-camera-matrix")
    matrix = [[_finite(cell) for cell in row] for row in value]
    if any(abs(a - b) > 1e-6 for a, b in zip(matrix[3], (0, 0, 0, 1))):
        raise ProjectionError("projection-nonrigid-camera-matrix")
    for i in range(3):
        for j in range(3):
            dot = sum(matrix[k][i] * matrix[k][j] for k in range(3))
            if not math.isfinite(dot) or abs(dot - (1 if i == j else 0)) > 1e-5:
                raise ProjectionError("projection-nonrigid-camera-matrix")
    determinant = (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
                   - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
                   + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))
    if not math.isfinite(determinant) or abs(determinant - 1) > 1e-5:
        raise ProjectionError("projection-nonrigid-camera-matrix")
    return matrix


def _calibration(value: Any) -> tuple[int, int, dict[str, float]]:
    if type(value) is not dict or type(value.get("intrinsics")) is not dict:
        raise ProjectionError("projection-invalid-calibration")
    width, height = _dimensions(value.get("width"), value.get("height"))
    source = value["intrinsics"]
    if set(source) != {"fx", "fy", "cx", "cy"}:
        raise ProjectionError("projection-invalid-calibration")
    k = {key: _finite(source[key]) for key in source}
    if (k["fx"] <= 0 or k["fy"] <= 0 or not math.isclose(k["fx"], k["fy"], rel_tol=0, abs_tol=1e-6) or
            not math.isclose(k["cx"], width / 2, rel_tol=0, abs_tol=1e-6) or
            not math.isclose(k["cy"], height / 2, rel_tol=0, abs_tol=1e-6)):
        raise ProjectionError("projection-invalid-calibration")
    expected = [[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]]
    matrix = value.get("K")
    if (type(matrix) is not list or len(matrix) != 3 or any(type(row) is not list or len(row) != 3 for row in matrix) or
            any(abs(_finite(matrix[r][c]) - expected[r][c]) > 1e-6 for r in range(3) for c in range(3))):
        raise ProjectionError("projection-invalid-calibration")
    distortion = value.get("distortion")
    if (type(distortion) is not dict or set(distortion) != {"lens_k", "lens_kcube"} or
            any(_finite(item) != 0 for item in distortion.values()) or
            value.get("coordinate_convention") != "carla-x-forward-y-right-z-up" or value.get("rotation_units") != "degrees"):
        raise ProjectionError("projection-unsupported-camera-model")
    return width, height, k


def project_waypoints(world_points: list[list[float]], camera_world_matrix: list[list[float]],
                      calibration: dict) -> list[dict]:
    """Project five 3/6/9/12/15m targets, retaining explicit visibility diagnostics.

    UV uses the nominal CARLA principal point (width/2, height/2). Raster
    indices use floor(UV) within [0,width) x [0,height). No coordinate clamps
    or behind-camera sign flips are used.
    """
    matrix = _rigid_matrix(camera_world_matrix)
    width, height, k = _calibration(calibration)
    if type(world_points) is not list or len(world_points) != 5:
        raise ProjectionError("projection-invalid-waypoint-shape")
    result = []
    for index, point in enumerate(world_points):
        if type(point) is not list or len(point) != 3:
            raise ProjectionError("projection-invalid-waypoint-shape")
        world = [_finite(cell) for cell in point]
        # Inverse rigid transform: R^T (world - camera translation).
        local = [sum(matrix[r][axis] * (world[r] - matrix[r][3]) for r in range(3)) for axis in range(3)]
        local = [_finite(cell) for cell in local]
        optical = [local[1], -local[2], local[0]]
        depth = optical[2]
        uv = pixel = None
        if depth <= 0:
            status = "BEHIND_CAMERA"
        elif depth <= NEAR_PLANE_M:
            status = "NEAR_PLANE"
        else:
            uv = [_finite(k["fx"] * optical[0] / depth + k["cx"]),
                  _finite(k["fy"] * optical[1] / depth + k["cy"])]
            status = "VISIBLE" if 0 <= uv[0] < width and 0 <= uv[1] < height else "OUTSIDE_IMAGE"
            if status == "VISIBLE":
                pixel = [math.floor(cell) for cell in uv]
        result.append({"index": index, "horizon_arc_m": HORIZONS[index], "world_xyz_m": world,
                       "camera_carla_xyz_m": local, "camera_optical_xyz_m": optical, "depth_m": depth,
                       "uv_pixels": uv, "pixel": pixel, "status": status, "coordinate_clamped": False})
    return result


def _paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    differences = (abs(estimate - left), abs(estimate - above), abs(estimate - upper_left))
    if differences[0] <= differences[1] and differences[0] <= differences[2]:
        return left
    return above if differences[1] <= differences[2] else upper_left


def decode_rgb8_png(data: bytes, expected_sha256: str | None = None,
                    expected_dimensions: tuple[int, int] | None = None) -> tuple[int, int, bytes]:
    """Decode non-interlaced RGB8 PNG filters 0..4; verify SHA before decoding.

    Chunk CRCs, ordering, zlib completion and bounded scanline size are checked
    by the episode integrity checker. Output bytes are R,G,B (no alpha/BGRA).
    """
    if type(data) is not bytes:
        raise ProjectionError("projection-invalid-png-input")
    if expected_sha256 is not None:
        if type(expected_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ProjectionError("projection-invalid-image-hash")
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ProjectionError("projection-image-hash-mismatch")
    if expected_dimensions is not None:
        if type(expected_dimensions) is not tuple or len(expected_dimensions) != 2:
            raise ProjectionError("projection-invalid-dimensions")
        _dimensions(*expected_dimensions)
    try:
        width, height = _png_dimensions(data, expected_dimensions)
    except EpisodeError as error:
        raise ProjectionError(error.code) from None
    compressed = bytearray()
    offset = 8
    while offset < len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        if data[offset + 4:offset + 8] == b"IDAT":
            compressed.extend(data[offset + 8:offset + 8 + length])
        offset += length + 12
    stride = width * 3
    raw = zlib.decompressobj().decompress(compressed, height * (stride + 1) + 1)
    rgb = bytearray(width * height * 3)
    for y in range(height):
        row_start = y * (stride + 1)
        filter_kind = raw[row_start]
        scanline = raw[row_start + 1:row_start + 1 + stride]
        start = y * stride
        if filter_kind == 0:
            rgb[start:start + stride] = scanline
            continue
        for x, value in enumerate(scanline):
            left = rgb[start + x - 3] if x >= 3 else 0
            above = rgb[start + x - stride] if y else 0
            upper_left = rgb[start + x - stride - 3] if y and x >= 3 else 0
            predictor = (left if filter_kind == 1 else above if filter_kind == 2 else
                         (left + above) // 2 if filter_kind == 3 else _paeth(left, above, upper_left))
            rgb[start + x] = (value + predictor) & 255
    return width, height, bytes(rgb)


def encode_rgb8_png(rgb: bytes, width: int, height: int) -> bytes:
    _dimensions(width, height)
    if type(rgb) is not bytes or len(rgb) != width * height * 3:
        raise ProjectionError("projection-invalid-rgb-buffer")

    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    stride = width * 3
    raw = b"".join(b"\x00" + rgb[start:start + stride] for start in range(0, len(rgb), stride))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


_GLYPHS = {
    "0": ("111", "101", "101", "101", "111"), "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"), "3": ("111", "001", "111", "001", "111"),
    "5": ("111", "100", "111", "001", "111"), "6": ("111", "100", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"), "m": ("000", "110", "111", "101", "101"),
    "X": ("101", "101", "010", "101", "101"),
}


def draw_overlay_rgb(rgb: bytes, width: int, height: int, projections: list[dict]) -> bytes:
    """Draw visible targets and their adjacent segments on a fresh RGB copy."""
    _dimensions(width, height)
    if type(rgb) is not bytes or len(rgb) != width * height * 3 or type(projections) is not list or len(projections) != 5:
        raise ProjectionError("projection-invalid-overlay-input")
    output = bytearray(rgb)

    def put(x, y, color):
        if 0 <= x < width and 0 <= y < height:
            start = (y * width + x) * 3
            output[start:start + 3] = bytes(color)

    def circle(x, y, radius, color):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    put(x + dx, y + dy, color)

    def text(x, y, value, color):
        for character in value:
            for dy, row in enumerate(_GLYPHS[character]):
                for dx, cell in enumerate(row):
                    if cell == "1":
                        put(x + dx, y + dy, color)
            x += 4

    def line(first, second):
        x, y = first
        end_x, end_y = second
        dx, dy = abs(end_x - x), -abs(end_y - y)
        sx, sy = 1 if x < end_x else -1, 1 if y < end_y else -1
        error = dx + dy
        while True:
            circle(x, y, 1, (255, 255, 255))
            if (x, y) == (end_x, end_y):
                break
            twice = 2 * error
            if twice >= dy:
                error += dy
                x += sx
            if twice <= dx:
                error += dx
                y += sy

    for index, projection in enumerate(projections):
        if (projection.get("index") != index or projection.get("horizon_arc_m") != HORIZONS[index] or
                projection.get("status") not in ("VISIBLE", "OUTSIDE_IMAGE", "BEHIND_CAMERA", "NEAR_PLANE") or
                projection.get("coordinate_clamped") is not False):
            raise ProjectionError("projection-invalid-overlay-input")
        pixel = projection.get("pixel")
        if projection.get("status") == "VISIBLE":
            uv = projection.get("uv_pixels")
            if (type(pixel) is not list or len(pixel) != 2 or any(type(value) is not int for value in pixel) or
                    not 0 <= pixel[0] < width or not 0 <= pixel[1] < height or type(uv) is not list or len(uv) != 2 or
                    not 0 <= _finite(uv[0]) < width or not 0 <= _finite(uv[1]) < height or
                    pixel != [math.floor(value) for value in uv]):
                raise ProjectionError("projection-invalid-overlay-input")
        elif pixel is not None:
            raise ProjectionError("projection-invalid-overlay-input")
    for first, second in zip(projections, projections[1:]):
        if first["status"] == second["status"] == "VISIBLE":
            line(first["pixel"], second["pixel"])
    for index, projection in enumerate(projections):
        color = COLORS_RGB[index]
        if projection["status"] == "VISIBLE":
            x, y = projection["pixel"]
            circle(x, y, 6, (0, 0, 0))
            circle(x, y, 4, color)
            text(x + 8, y - 2, str(HORIZONS[index]), color)
        # X means the target was not drawn at its world projection.
        legend_y = 8 + index * 10
        circle(7, legend_y + 2, 3, color)
        text(14, legend_y, str(HORIZONS[index]) + "m" + ("X" if projection["status"] != "VISIBLE" else ""), color)
    return bytes(output)


def _publish(path: Path, data: bytes, *, replace_owned: bool = False) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".overlay-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if replace_owned:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def render_episode_overlay(episode_path: str | Path, labels_path: str | Path, output_directory: str | Path,
                           frame_id: int | None = None, *, labels_validator=None) -> dict:
    """Render a fresh independently validated derived label, leaving raw files unchanged."""
    destination, source_root = Path(output_directory), Path(episode_path)
    report: dict = {"schema_version": 1, "status": "FAIL", "scope": "D6-same-frame-waypoint-reprojection-QA",
                    "projection_version": PROJECTION_VERSION, "errors": []}
    owned_destination = False
    stage = "output-preflight"
    try:
        if destination.exists() or destination.is_symlink():
            raise ProjectionError("overlay-destination-already-exists")
        if destination.resolve().is_relative_to(source_root.resolve()):
            raise ProjectionError("overlay-output-inside-raw-episode")
        if frame_id is not None and (type(frame_id) is not int or frame_id < 0):
            raise ProjectionError("overlay-invalid-frame-id")
        stage = "labels-validation"
        if labels_validator is None:
            from .labels import validate_labels
            labels_validator = validate_labels
        validated = labels_validator(source_root, Path(labels_path))
        if validated.get("status") != "PASS":
            raise ProjectionError("overlay-label-validation-failed")
        sidecar, metadata, frames = validated["sidecar"], validated["metadata"], validated["frames"]
        if sidecar.get("status") != "COMPLETE" or not frames:
            raise ProjectionError("overlay-label-validation-failed")
        chosen_frame = frames[0]["frame"] if frame_id is None else frame_id
        matching = [(frame, label) for frame, label in zip(frames, sidecar["rows"])
                    if frame["frame"] == chosen_frame]
        if len(matching) != 1:
            raise ProjectionError("overlay-frame-unavailable")
        frame, label_row = matching[0]
        if (label_row["frame"] != chosen_frame or label_row["label_state"] != "GENERATED" or
                label_row["label"]["ego_frame"] != chosen_frame or label_row["image"] !=
                {key: frame["image"][key] for key in ("path", "sha256")} or
                any(frame[key] != chosen_frame for key in ("world_frame", "snapshot_frame", "sensor_frame"))):
            raise ProjectionError("overlay-frame-identity-mismatch")
        stage = "projection"
        calibration = metadata["camera_calibration"]
        points = project_waypoints(label_row["label"]["waypoints_world_m"], frame["camera"]["transform"]["matrix"], calibration)
        source_png = (source_root / frame["image"]["path"]).read_bytes()
        width, height, rgb = decode_rgb8_png(source_png, frame["image"]["sha256"],
                                            (calibration["width"], calibration["height"]))
        overlay = encode_rgb8_png(draw_overlay_rgb(rgb, width, height, points), width, height)
        report.update({key: metadata[key] for key in ("dataset_version", "episode_id", "route_id", "scene_group_id", "town")})
        canonical = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                                            ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
        report.update(frame=chosen_frame, sample_id=label_row["sample_id"], timestamp=frame["timestamp"],
                      source_image={key: frame["image"][key] for key in ("path", "sha256", "width", "height")},
                      source_label=sidecar["source"], calibration=calibration,
                      labels_reference={"generation_version": sidecar["generation_version"], "contract_version": sidecar["contract_version"],
                                        "canonical_sidecar_sha256": canonical(sidecar), "row_index": label_row["index"],
                                        "label_sha256": canonical(label_row["label"])},
                      source_frame={key: frame[key] for key in ("index", "frame", "world_frame", "snapshot_frame", "sensor_frame",
                                                              "timestamp", "snapshot_timestamp", "sensor_timestamp")},
                      camera_transform=frame["camera"]["transform"],
                      projection_convention="CARLA(x-forward,y-right,z-up)->optical(y,-z,x);UV=K*optical/depth",
                      raster_policy="floor-visible-UV;outside-points-not-drawn;no-coordinate-clamping",
                      points=points, visible_points=sum(point["status"] == "VISIBLE" for point in points),
                      all_points_in_front=all(point["depth_m"] > NEAR_PLANE_M for point in points),
                      legend=[{"horizon_arc_m": distance, "color_rgb": list(color)} for distance, color in zip(HORIZONS, COLORS_RGB)],
                      overlay={"path": "overlay.png", "sha256": hashlib.sha256(overlay).hexdigest(), "width": width, "height": height},
                      color_provenance_certified=False)
        stage = "publication"
        destination.mkdir(parents=True, exist_ok=False)
        owned_destination = True
        report["status"] = "RUNNING"
        _publish(destination / "report.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        _publish(destination / "overlay.png", overlay)
        if hashlib.sha256((destination / "overlay.png").read_bytes()).hexdigest() != report["overlay"]["sha256"]:
            raise ProjectionError("overlay-publication-integrity-failed")
        report["status"] = "PASS"
        _publish(destination / "report.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"), replace_owned=True)
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "FAIL"
        report["errors"] = [{"stage": stage, "code": error.code if isinstance(error, ProjectionError) else
                             "overlay-interrupted" if isinstance(error, KeyboardInterrupt) else "overlay-operation-failed"}]
        if owned_destination:
            try:
                _publish(destination / "report.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"), replace_owned=True)
            except OSError:
                pass
    return report
