"""Dependency-free D3 collection configuration and pinhole calibration.

Load and validate JSON before importing CARLA. The seed is recorded as part of
the fixed configuration; D3 creates no randomized actors or physics seed claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


class ConfigError(ValueError):
    """The collection configuration violates its explicit D3 contract."""


def _export(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _export(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {key: _export(item) for key, item in value.items()}
    return value


class _Serializable:
    def to_dict(self) -> dict[str, Any]:
        """Return a fresh JSON-compatible tree, without exposing immutable state."""
        return _export(self)


@dataclass(frozen=True)
class ConnectionConfig(_Serializable):
    host: str = "127.0.0.1"
    port: int = 2000
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class WorldConfig(_Serializable):
    synchronous_mode: bool = True
    fixed_delta_seconds: float = 0.05
    no_rendering_mode: bool = False
    substepping: bool = True
    max_substep_delta_time: float = 0.01
    max_substeps: int = 10


@dataclass(frozen=True)
class WeatherConfig(_Serializable):
    preset: str = "ClearNoon"


@dataclass(frozen=True)
class VehicleControlConfig(_Serializable):
    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 1.0
    hand_brake: bool = True
    reverse: bool = False


def _attributes(value: Any) -> dict[str, str]:
    attributes = _object(value, "vehicle.attributes", {"role_name", "color"})
    role = _string(attributes["role_name"], "vehicle.attributes.role_name")
    if not re.fullmatch(r"vlf_[A-Za-z0-9_-]{1,36}", role):
        raise ConfigError("vehicle.attributes.role_name must be a short vlf_ project role.")
    color = _string(attributes["color"], "vehicle.attributes.color")
    if not re.fullmatch(r"\d{1,3},\d{1,3},\d{1,3}", color) or any(
        int(component) > 255 for component in color.split(",")
    ):
        raise ConfigError("vehicle.attributes.color must contain three comma-separated integers from 0 to 255.")
    return {"role_name": role, "color": ",".join(str(int(component)) for component in color.split(","))}


@dataclass(frozen=True)
class VehicleConfig(_Serializable):
    blueprint: str = "vehicle.tesla.model3"
    spawn_point_index: int = 0
    attributes: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({"role_name": "vlf_ego", "color": "0,0,0"})
    )
    control: VehicleControlConfig = field(default_factory=VehicleControlConfig)

    def __post_init__(self) -> None:
        # Defensive copy also protects against a caller mutating its original
        # dict after constructing an otherwise frozen dataclass.
        object.__setattr__(self, "attributes", MappingProxyType(_attributes(self.attributes)))


@dataclass(frozen=True)
class LocationConfig(_Serializable):
    x: float = 1.5
    y: float = 0.0
    z: float = 2.4


@dataclass(frozen=True)
class RotationConfig(_Serializable):
    pitch: float = -12.0
    yaw: float = 0.0
    roll: float = 0.0


@dataclass(frozen=True)
class CameraMountConfig(_Serializable):
    location: LocationConfig = field(default_factory=LocationConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)


@dataclass(frozen=True)
class CameraConfig(_Serializable):
    blueprint: str = "sensor.camera.rgb"
    width: int = 640
    height: int = 360
    fov: float = 90.0
    sensor_tick: float = 0.0
    enable_postprocess_effects: bool = False
    gamma: float = 2.2
    lens_k: float = 0.0
    lens_kcube: float = 0.0
    mount: CameraMountConfig = field(default_factory=CameraMountConfig)

    def intrinsics(self) -> dict[str, float]:
        """Return fx, fy, cx, cy in pixels for CARLA's horizontal FOV.

        Assume square pixels and a pinhole camera (lens_k/lens_kcube = 0).
        Principal point is (width/2, height/2), not ((width-1)/2, (height-1)/2).
        Image axes are u right, v down. These are optical intrinsics; mount
        location is in vehicle metres and CARLA rotation is in degrees.
        """
        focal_length = self.width / (2.0 * math.tan(math.radians(self.fov) / 2.0))
        return {"fx": focal_length, "fy": focal_length, "cx": self.width / 2.0, "cy": self.height / 2.0}


@dataclass(frozen=True)
class VerificationConfig(_Serializable):
    warmup_frames: int = 10
    frames: int = 20
    queue_capacity: int = 8


@dataclass(frozen=True)
class CollectionConfig(_Serializable):
    schema_version: int = 1
    carla_version: str = "0.9.16"
    map: str = "Town01"
    seed: int = 42
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)

    @property
    def sha256(self) -> str:
        """Hash effective values in sorted, compact UTF-8 JSON (no file paths)."""
        canonical = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _object(value: Any, path: str, expected: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{path} must be an object with string keys.")
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise ConfigError(f"{path} has unsupported keys: {', '.join(sorted(unknown))}.")
    if missing:
        raise ConfigError(f"{path} is missing keys: {', '.join(sorted(missing))}.")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 100:
        raise ConfigError(f"{path} must be a nonempty string of at most 100 characters.")
    return value


def _exact(value: Any, path: str, expected: str) -> str:
    if _string(value, path) != expected:
        raise ConfigError(f"{path} must be {expected} for the D3 contract.")
    return expected


def _integer(value: Any, path: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ConfigError(f"{path} must be an integer from {lower} to {upper}; booleans are not numbers.")
    return value


def _number(value: Any, path: str, lower: float, upper: float, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise ConfigError(f"{path} must be a finite number; booleans are not numbers.")
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise ConfigError(f"{path} must be a finite number.") from None
    if not math.isfinite(number) or not lower <= number <= upper or (positive and number <= 0):
        raise ConfigError(f"{path} must be finite and within {'(' if positive else '['}{lower}, {upper}].")
    return number


def _boolean(value: Any, path: str, expected: bool | None = None) -> bool:
    if type(value) is not bool:
        raise ConfigError(f"{path} must be a JSON boolean.")
    if expected is not None and value is not expected:
        raise ConfigError(f"{path} must be {str(expected).lower()} for the D3 contract.")
    return value


def collection_config_from_dict(value: Mapping[str, Any]) -> CollectionConfig:
    """Validate every supplied field; omitted fields never silently use defaults."""
    root = _object(value, "config", {item.name for item in fields(CollectionConfig)})
    schema = _integer(root["schema_version"], "schema_version", 1, 1)
    version = _exact(root["carla_version"], "carla_version", "0.9.16")
    map_name = _exact(root["map"], "map", "Town01")
    seed = _integer(root["seed"], "seed", 0, 2**32 - 1)

    raw = _object(root["connection"], "connection", {item.name for item in fields(ConnectionConfig)})
    connection = ConnectionConfig(
        host=_exact(raw["host"], "connection.host", "127.0.0.1"),
        port=_integer(raw["port"], "connection.port", 1, 65534),
        timeout_seconds=_number(raw["timeout_seconds"], "connection.timeout_seconds", 0, 120, positive=True),
    )

    raw = _object(root["world"], "world", {item.name for item in fields(WorldConfig)})
    world = WorldConfig(
        synchronous_mode=_boolean(raw["synchronous_mode"], "world.synchronous_mode", True),
        fixed_delta_seconds=_number(raw["fixed_delta_seconds"], "world.fixed_delta_seconds", 0, 0.1, positive=True),
        no_rendering_mode=_boolean(raw["no_rendering_mode"], "world.no_rendering_mode", False),
        substepping=_boolean(raw["substepping"], "world.substepping", True),
        max_substep_delta_time=_number(raw["max_substep_delta_time"], "world.max_substep_delta_time", 0, 0.01, positive=True),
        max_substeps=_integer(raw["max_substeps"], "world.max_substeps", 1, 10),
    )
    if world.fixed_delta_seconds > world.max_substep_delta_time * world.max_substeps + 1e-12:
        raise ConfigError("world.fixed_delta_seconds exceeds max_substep_delta_time * max_substeps.")

    raw = _object(root["weather"], "weather", {"preset"})
    weather = WeatherConfig(_exact(raw["preset"], "weather.preset", "ClearNoon"))

    raw = _object(root["vehicle"], "vehicle", {item.name for item in fields(VehicleConfig)})
    control_raw = _object(raw["control"], "vehicle.control", {item.name for item in fields(VehicleControlConfig)})
    control = VehicleControlConfig(
        throttle=_number(control_raw["throttle"], "vehicle.control.throttle", 0, 0),
        steer=_number(control_raw["steer"], "vehicle.control.steer", 0, 0),
        brake=_number(control_raw["brake"], "vehicle.control.brake", 1, 1),
        hand_brake=_boolean(control_raw["hand_brake"], "vehicle.control.hand_brake", True),
        reverse=_boolean(control_raw["reverse"], "vehicle.control.reverse", False),
    )
    vehicle = VehicleConfig(
        blueprint=_exact(raw["blueprint"], "vehicle.blueprint", "vehicle.tesla.model3"),
        spawn_point_index=_integer(raw["spawn_point_index"], "vehicle.spawn_point_index", 0, 100000),
        attributes=_attributes(raw["attributes"]),
        control=control,
    )

    raw = _object(root["camera"], "camera", {item.name for item in fields(CameraConfig)})
    mount_raw = _object(raw["mount"], "camera.mount", {"location", "rotation"})
    loc_raw = _object(mount_raw["location"], "camera.mount.location", {"x", "y", "z"})
    rot_raw = _object(mount_raw["rotation"], "camera.mount.rotation", {"pitch", "yaw", "roll"})
    mount = CameraMountConfig(
        location=LocationConfig(
            x=_number(loc_raw["x"], "camera.mount.location.x", -10, 10),
            y=_number(loc_raw["y"], "camera.mount.location.y", -10, 10),
            z=_number(loc_raw["z"], "camera.mount.location.z", 0.05, 10),
        ),
        rotation=RotationConfig(
            pitch=_number(rot_raw["pitch"], "camera.mount.rotation.pitch", -90, 90),
            yaw=_number(rot_raw["yaw"], "camera.mount.rotation.yaw", -180, 180),
            roll=_number(rot_raw["roll"], "camera.mount.rotation.roll", -180, 180),
        ),
    )
    camera = CameraConfig(
        blueprint=_exact(raw["blueprint"], "camera.blueprint", "sensor.camera.rgb"),
        width=_integer(raw["width"], "camera.width", 16, 4096),
        height=_integer(raw["height"], "camera.height", 16, 2160),
        fov=_number(raw["fov"], "camera.fov", 1, 179),
        sensor_tick=_number(raw["sensor_tick"], "camera.sensor_tick", 0, 0),
        enable_postprocess_effects=_boolean(raw["enable_postprocess_effects"], "camera.enable_postprocess_effects", False),
        gamma=_number(raw["gamma"], "camera.gamma", 0, 10, positive=True),
        lens_k=_number(raw["lens_k"], "camera.lens_k", 0, 0),
        lens_kcube=_number(raw["lens_kcube"], "camera.lens_kcube", 0, 0),
        mount=mount,
    )

    raw = _object(root["verification"], "verification", {item.name for item in fields(VerificationConfig)})
    verification = VerificationConfig(
        warmup_frames=_integer(raw["warmup_frames"], "verification.warmup_frames", 0, 1000),
        frames=_integer(raw["frames"], "verification.frames", 1, 1000),
        queue_capacity=_integer(raw["queue_capacity"], "verification.queue_capacity", 1, 64),
    )
    return CollectionConfig(schema, version, map_name, seed, connection, world, weather, vehicle, camera, verification)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"Duplicate JSON key: {key}.")
        result[key] = value
    return result


def _invalid_constant(_: str) -> Any:
    raise ConfigError("JSON cannot contain NaN or Infinity.")


def load_collection_config(path: str | Path) -> CollectionConfig:
    """Read strict UTF-8 JSON with duplicate/nonfinite rejection, then validate."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError):
        raise ConfigError("Could not read the collection configuration as UTF-8 JSON.") from None
    try:
        raw = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except ConfigError:
        raise
    except (ValueError, RecursionError):
        raise ConfigError("Collection configuration must be a valid JSON object.") from None
    return collection_config_from_dict(raw)
