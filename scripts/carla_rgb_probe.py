"""Check real CARLA RGB rendering; CARLA is imported only during execution.

The simulator is deliberately advanced one frame at a time. Every tick must
deliver its own RGB image before another tick is allowed. A PASS is evidence of
rendering and capture only, not lane following, training, or external access.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import math
import queue
import re
import struct
import subprocess
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CARLA_VERSION = "0.9.16"
REPORT_NAME = "carla_rgb_report.json"
PREVIEW_NAME = "carla_rgb_preview.png"


class ProbeFailure(RuntimeError):
    """A stable, public error code, never a raw server or filesystem message."""


@dataclass(frozen=True)
class ProbeConfig:
    output_dir: Path
    host: str = "127.0.0.1"
    port: int = 2000
    timeout: float = 30.0
    frames: int = 20
    warmup: int = 5
    width: int = 640
    height: int = 360
    fixed_delta_seconds: float = 0.05
    reuse_current_world: bool = False


def rgb_from_bgra(raw: bytes, width: int, height: int) -> tuple[bytes, dict[str, Any]]:
    """Reject empty, truncated, and spatially constant images (ignore alpha)."""
    if width <= 0 or height <= 0:
        raise ProbeFailure("invalid-image-dimensions")
    if len(raw) != width * height * 4:
        raise ProbeFailure("invalid-bgra-length")
    rgb = bytearray(width * height * 3)
    rgb[0::3], rgb[1::3], rgb[2::3] = raw[2::4], raw[1::4], raw[0::4]
    channels = (rgb[0::3], rgb[1::3], rgb[2::3])
    ranges = [(min(channel), max(channel)) for channel in channels]
    if not any(low != high for low, high in ranges):
        raise ProbeFailure("spatially-constant-rgb")
    return bytes(rgb), {
        "width": width,
        "height": height,
        "rgb_bytes": len(rgb),
        "channel_min": [low for low, _ in ranges],
        "channel_max": [high for _, high in ranges],
        "spatially_nonconstant": True,
    }


def png_from_rgb(rgb: bytes, width: int, height: int) -> bytes:
    """Encode a true-color PNG without Pillow or NumPy."""
    if width <= 0 or height <= 0 or len(rgb) != width * height * 3:
        raise ProbeFailure("invalid-png-input")

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    stride = width * 3
    scanlines = b"".join(b"\x00" + rgb[offset:offset + stride] for offset in range(0, len(rgb), stride))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b"")


def matching_image(images: queue.Queue, frame_id: int, timeout: float) -> Any:
    """Wait for this tick's sensor frame, discarding old startup deliveries."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProbeFailure("rgb-frame-timeout")
        try:
            image = images.get(timeout=remaining)
        except queue.Empty:
            raise ProbeFailure("rgb-frame-timeout") from None
        if image.frame == frame_id:
            return image
        if image.frame > frame_id:
            raise ProbeFailure("rgb-frame-ahead-of-world")


def gpu_sample() -> dict[str, Any]:
    """Optional whole-device snapshot, not a per-process or peak measurement."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode:
            return {"status": "unavailable"}
        devices = []
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row) != 4:
                return {"status": "unavailable"}
            name, total, used, utilization = [field.strip() for field in row]
            devices.append({"name": name, "total_mib": int(total), "used_mib": int(used), "utilization_percent": int(utilization)})
        return {"status": "available", "devices": devices} if devices else {"status": "unavailable"}
    except (OSError, subprocess.TimeoutExpired, ValueError, csv.Error, UnicodeError):
        return {"status": "unavailable"}


def run_probe(config: ProbeConfig, carla_module: Any = None) -> dict[str, Any]:
    """Return a sanitized report and leave settings and owned actors cleaned up."""
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 1, "status": "FAIL", "scope": "carla-rgb-rendering-only",
        "required_carla_version": CARLA_VERSION, "map_requested": "Town01",
        "configuration": {"width": config.width, "height": config.height, "warmup_frames": config.warmup,
                          "measured_frames": config.frames, "fixed_delta_seconds": config.fixed_delta_seconds,
                          "reuse_current_world": config.reuse_current_world},
        "frames": [], "preview_png_written": False,
        "gpu_memory_scope": "whole-device-snapshots-not-peak", "cleanup_errors": [],
    }
    stage = "configuration"
    world = original_settings = None
    actors: list[Any] = []
    camera = None
    completed = False
    try:
        if (config.frames < 1 or config.warmup < 0 or config.width < 1 or config.height < 1
                or config.timeout <= 0 or not math.isfinite(config.timeout)
                or config.fixed_delta_seconds <= 0 or not math.isfinite(config.fixed_delta_seconds)
                or not isinstance(config.reuse_current_world, bool)):
            raise ProbeFailure("invalid-configuration")
        stage = "output-directory"
        config.output_dir.mkdir(parents=True, exist_ok=True)
        # A prior run's preview must not look like evidence from a failed rerun.
        (config.output_dir / PREVIEW_NAME).unlink(missing_ok=True)
        stage = "carla-import"
        carla = carla_module if carla_module is not None else importlib.import_module("carla")
        stage = "server-connection"
        client = carla.Client(config.host, config.port)
        client.set_timeout(config.timeout)
        stage = "version-check"
        client_version, server_version = client.get_client_version(), client.get_server_version()
        for label, value in (("client_version", client_version), ("server_version", server_version)):
            report[label] = value if isinstance(value, str) and re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][A-Za-z0-9_.-]+)?", value) else "invalid-response"
        if client_version != CARLA_VERSION or server_version != CARLA_VERSION:
            raise ProbeFailure("carla-version-mismatch")
        stage = "world-reuse" if config.reuse_current_world else "world-load"
        world = client.get_world() if config.reuse_current_world else client.load_world("Town01")
        if world.get_map().name.rsplit("/", 1)[-1] != "Town01":
            raise ProbeFailure("unexpected-map")
        report["map_verified"] = "Town01"
        report["world_initialization"] = "reuse-current-town01" if config.reuse_current_world else "load-town01"
        stage = "world-settings"
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = config.fixed_delta_seconds
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        applied = world.get_settings()
        if not applied.synchronous_mode or applied.no_rendering_mode or not math.isclose(applied.fixed_delta_seconds or 0, config.fixed_delta_seconds):
            raise ProbeFailure("rendering-settings-not-applied")
        report["rendering_settings_verified"] = True
        stage = "actor-spawn"
        blueprints = world.get_blueprint_library()
        vehicle_bp = blueprints.find("vehicle.tesla.model3")
        vehicle = None
        for transform in world.get_map().get_spawn_points():
            vehicle = world.try_spawn_actor(vehicle_bp, transform)
            if vehicle is not None:
                break
        if vehicle is None:
            raise ProbeFailure("vehicle-spawn-failed")
        actors.append(vehicle)
        camera_bp = blueprints.find("sensor.camera.rgb")
        for attribute, value in {"image_size_x": str(config.width), "image_size_y": str(config.height), "fov": "90", "sensor_tick": "0.0"}.items():
            camera_bp.set_attribute(attribute, value)
        camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-12)), attach_to=vehicle)
        actors.append(camera)
        images: queue.Queue = queue.Queue()
        camera.listen(images.put)
        report["gpu_before_capture"] = gpu_sample()
        stage = "rgb-warmup"
        for _ in range(config.warmup):
            frame = world.tick(config.timeout)
            image = matching_image(images, frame, config.timeout)
            if image.width != config.width or image.height != config.height:
                raise ProbeFailure("unexpected-image-size")
            if len(image.raw_data) != config.width * config.height * 4:
                raise ProbeFailure("invalid-bgra-length")
        stage = "rgb-capture"
        capture_started = time.perf_counter()
        preview_rgb = None
        first_sim_timestamp = last_sim_timestamp = None
        for _ in range(config.frames):
            tick_started = time.perf_counter()
            frame = world.tick(config.timeout)
            image = matching_image(images, frame, config.timeout)
            delivered = time.perf_counter()
            if image.width != config.width or image.height != config.height:
                raise ProbeFailure("unexpected-image-size")
            rgb, stats = rgb_from_bgra(bytes(image.raw_data), image.width, image.height)
            timestamp = float(image.timestamp)
            if not math.isfinite(timestamp):
                raise ProbeFailure("invalid-image-timestamp")
            if last_sim_timestamp is not None and not math.isclose(timestamp - last_sim_timestamp, config.fixed_delta_seconds, abs_tol=1e-5):
                raise ProbeFailure("unexpected-simulation-step")
            if first_sim_timestamp is None:
                first_sim_timestamp = timestamp
            last_sim_timestamp = timestamp
            report["frames"].append({"world_frame": frame, "sensor_frame": image.frame,
                                     "simulation_timestamp_seconds": timestamp,
                                     "tick_to_rgb_seconds": delivered - tick_started, "rgb": stats})
            if preview_rgb is None:
                preview_rgb = rgb
        wall_seconds = time.perf_counter() - capture_started
        simulated_seconds = config.frames * config.fixed_delta_seconds
        latencies = [record["tick_to_rgb_seconds"] for record in report["frames"]]
        report["metrics"] = {
            "captured_frames": len(report["frames"]), "capture_wall_seconds": wall_seconds,
            "simulated_advance_seconds": simulated_seconds,
            "capture_frames_per_second": config.frames / wall_seconds,
            "real_time_factor": simulated_seconds / wall_seconds,
            "tick_to_rgb_mean_seconds": sum(latencies) / len(latencies), "tick_to_rgb_max_seconds": max(latencies),
            "timing_scope": "capture-loop-including-rgb-validation-excluding-gpu-snapshots-and-png-writing",
        }
        report["gpu_after_capture"] = gpu_sample()
        stage = "png-write"
        assert preview_rgb is not None
        (config.output_dir / PREVIEW_NAME).write_bytes(png_from_rgb(preview_rgb, config.width, config.height))
        report["preview_png_written"] = True
        completed = True
    except Exception as error:
        report["error"] = {"stage": stage, "code": str(error) if isinstance(error, ProbeFailure) else "operation-failed",
                           "exception_type": type(error).__name__}
    finally:
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                report["cleanup_errors"].append("camera-stop-failed")
        for actor in reversed(actors):
            try:
                if actor.destroy() is False:
                    report["cleanup_errors"].append("actor-destroy-failed")
            except Exception:
                report["cleanup_errors"].append("actor-destroy-failed")
        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except Exception:
                report["cleanup_errors"].append("world-settings-restore-failed")
        report["cleanup_complete"] = not report["cleanup_errors"]
        report["elapsed_seconds"] = time.perf_counter() - started
    if completed and report["cleanup_complete"]:
        report["status"] = "PASS"
    elif completed:
        report["error"] = {"stage": "cleanup", "code": "cleanup-incomplete", "exception_type": None}
    return report


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reuse-current-world", action="store_true",
                        help="Use the already-running Town01; fail if its map differs, without reloading it.")
    args = parser.parse_args(arguments)
    config = ProbeConfig(**vars(args))
    report = run_probe(config)
    try:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / REPORT_NAME).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError:
        report["status"] = "FAIL"
        report["error"] = {"stage": "report-write", "code": "report-write-failed", "exception_type": "OSError"}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
