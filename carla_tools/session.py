"""Own a fixed D3 vehicle/camera session and verify restoration on exit.

CARLA is imported only when entering a real session. The caller owns camera
callbacks and ticks, and calls verify_applied() after warmup to verify the
last-tick control and transform readbacks.
"""

from __future__ import annotations

import importlib
import math
import time
from typing import Any, Mapping

from .config import collection_config_from_dict


CARLA_VERSION = "0.9.16"


class SessionError(RuntimeError):
    """Public stable code/stage/type without raw RPC or filesystem messages."""

    def __init__(self, code: str, stage: str = "session", error_type: str | None = None):
        super().__init__(code)
        self.code, self.stage, self.error_type = code, stage, error_type

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "stage": self.stage, "exception_type": self.error_type}


def public_fields(value: Any) -> dict[str, Any]:
    """Snapshot every primitive public API field, including version additions."""
    fields = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        field = getattr(value, name)
        if field is None or isinstance(field, (bool, int, float, str)):
            if isinstance(field, float) and not math.isfinite(field):
                raise SessionError("nonfinite-api-state")
            fields[name] = field
    return fields


def values_equal(first: Any, second: Any) -> bool:
    if isinstance(first, bool) or isinstance(second, bool):
        return type(first) is type(second) and first == second
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return math.isclose(first, second, rel_tol=1e-6, abs_tol=1e-6)
    return first == second


def fields_match(expected: Mapping[str, Any], actual: Mapping[str, Any], *, exact_keys: bool = False) -> bool:
    return (not exact_keys or expected.keys() == actual.keys()) and all(
        key in actual and values_equal(value, actual[key]) for key, value in expected.items()
    )


def attributes_match(expected: Mapping[str, str], actual: Mapping[str, str]) -> bool:
    for key, value in expected.items():
        if key not in actual:
            return False
        observed = str(actual[key])
        if observed == value:
            continue
        if value.lower() in ("true", "false") and observed.lower() == value.lower():
            continue
        try:
            if math.isfinite(float(value)) and math.isfinite(float(observed)) and math.isclose(float(value), float(observed), rel_tol=1e-6, abs_tol=1e-6):
                continue
        except ValueError:
            pass
        return False
    return True


def transform_record(transform: Any) -> dict[str, Any]:
    return {"location": {name: float(getattr(transform.location, name)) for name in ("x", "y", "z")},
            "rotation": {name: float(getattr(transform.rotation, name)) for name in ("pitch", "yaw", "roll")}}


class CollectionSession:
    def __init__(self, cfg: Any, carla_module: Any = None):
        self.cfg = cfg
        self._carla = carla_module
        self.client = self.world = self.vehicle = self.camera = None
        self.manifest: dict[str, Any] = {}
        self.cleanup_report: dict[str, Any] = {"status": "NOT_RUN", "errors": []}
        self._original_settings = self._original_weather = self._mount = None
        self._settings_before: dict[str, Any] = {}
        self._weather_before: dict[str, Any] = {}
        self._world_expected: dict[str, Any] = {}
        self._weather_expected: dict[str, Any] = {}
        self._vehicle_attributes: dict[str, str] = {}
        self._camera_attributes: dict[str, str] = {}
        self._control_expected: dict[str, Any] = {}
        self._owned: list[Any] = []
        self._extra_sensors: list[Any] = []
        self._settings_touched = self._weather_touched = self._entered = self._closed = False

    @property
    def camera_mount_transform(self):
        return self._mount

    def __enter__(self):
        if self._entered or self._closed:
            raise SessionError("session-cannot-be-reentered")
        self._entered = True
        stage = "configuration"
        try:
            cfg = collection_config_from_dict(self.cfg.to_dict())
            self.cfg = cfg
            if (cfg.carla_version != CARLA_VERSION or not cfg.world.synchronous_mode
                    or cfg.world.no_rendering_mode or cfg.weather.preset != "ClearNoon"):
                raise SessionError("unsupported-collection-configuration", stage)
            self._control_expected = cfg.vehicle.control.to_dict()
            if (self._control_expected["throttle"] != 0 or self._control_expected["steer"] != 0
                    or self._control_expected["brake"] != 1 or not self._control_expected["hand_brake"]
                    or self._control_expected["reverse"]):
                raise SessionError("stationary-control-required", stage)
            stage = "carla-import"
            carla = self._carla if self._carla is not None else importlib.import_module("carla")
            self._carla = carla
            stage = "version-check"
            client = carla.Client(cfg.connection.host, cfg.connection.port)
            self.client = client
            client.set_timeout(cfg.connection.timeout_seconds)
            client_version, server_version = client.get_client_version(), client.get_server_version()
            if client_version != CARLA_VERSION or server_version != CARLA_VERSION:
                raise SessionError("carla-version-mismatch", stage)
            self.manifest.update(carla_version=CARLA_VERSION, map=cfg.map, seed=cfg.seed,
                                 config_sha256=cfg.sha256, versions={"client": client_version, "server": server_version},
                                 seed_scope="recorded-fixed-configuration-no-randomized-actors",
                                 applied_verified=False)
            stage = "world-preflight"
            self.world = client.get_world()
            world_map = self.world.get_map()
            if world_map.name.replace("\\", "/").rsplit("/", 1)[-1] != cfg.map:
                raise SessionError("current-map-mismatch", stage)
            self._original_settings = self.world.get_settings()
            if self._original_settings.synchronous_mode:
                raise SessionError("existing-synchronous-world-refused", stage)
            if any(actor.type_id.startswith(("vehicle.", "walker.", "controller.ai.walker")) for actor in self.world.get_actors()):
                raise SessionError("existing-dynamic-actors-refused", stage)
            if any(actor.type_id.startswith("sensor.") for actor in self.world.get_actors()):
                raise SessionError("existing-sensors-refused", stage)
            self._original_weather = self.world.get_weather()
            self._settings_before = public_fields(self._original_settings)
            self._weather_before = public_fields(self._original_weather)
            if not self._settings_before or not self._weather_before:
                raise SessionError("original-state-snapshot-empty", stage)
            self.manifest["original_world_settings"] = dict(self._settings_before)
            self.manifest["original_weather"] = dict(self._weather_before)
            stage = "blueprint-preflight"
            library = self.world.get_blueprint_library()
            vehicle_bp = library.find(cfg.vehicle.blueprint)
            camera_bp = library.find(cfg.camera.blueprint)
            if vehicle_bp.id != cfg.vehicle.blueprint or camera_bp.id != cfg.camera.blueprint:
                raise SessionError("blueprint-id-mismatch", stage)
            self._vehicle_attributes = dict(cfg.vehicle.attributes)
            self._camera_attributes = {
                "image_size_x": str(cfg.camera.width), "image_size_y": str(cfg.camera.height),
                "fov": str(cfg.camera.fov), "sensor_tick": str(cfg.camera.sensor_tick),
                "enable_postprocess_effects": str(cfg.camera.enable_postprocess_effects).lower(),
                "gamma": str(cfg.camera.gamma), "lens_k": str(cfg.camera.lens_k),
                "lens_kcube": str(cfg.camera.lens_kcube),
            }
            for blueprint, attributes in ((vehicle_bp, self._vehicle_attributes), (camera_bp, self._camera_attributes)):
                for name, value in attributes.items():
                    if not blueprint.has_attribute(name) or not blueprint.get_attribute(name).is_modifiable:
                        raise SessionError("blueprint-attribute-not-modifiable", stage)
                    blueprint.set_attribute(name, value)
            spawn_points = world_map.get_spawn_points()
            index = cfg.vehicle.spawn_point_index
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(spawn_points):
                raise SessionError("spawn-point-index-unavailable", stage)
            spawn = spawn_points[index]
            stage = "world-settings"
            settings = self.world.get_settings()
            self._world_expected = cfg.world.to_dict()
            for name, value in self._world_expected.items():
                setattr(settings, name, value)
            self._settings_touched = True
            self.world.apply_settings(settings)
            if not fields_match(self._world_expected, public_fields(self.world.get_settings())):
                raise SessionError("world-settings-readback-mismatch", stage)
            stage = "weather"
            preset = carla.WeatherParameters.ClearNoon
            self._weather_expected = public_fields(preset)
            self._weather_touched = True
            self.world.set_weather(preset)
            if not self._wait_weather_readback(self._weather_expected):
                raise SessionError("weather-readback-mismatch", stage)
            stage = "vehicle-spawn"
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn)
            if self.vehicle is None:
                raise SessionError("fixed-spawn-collision", stage)
            self._owned.append(self.vehicle)
            self.vehicle.apply_control(carla.VehicleControl(**self._control_expected))
            stage = "camera-spawn"
            location, rotation = cfg.camera.mount.location, cfg.camera.mount.rotation
            self._mount = carla.Transform(carla.Location(x=location.x, y=location.y, z=location.z),
                                          carla.Rotation(pitch=rotation.pitch, yaw=rotation.yaw, roll=rotation.roll))
            self.camera = self.world.spawn_actor(camera_bp, self._mount, attach_to=self.vehicle,
                                                 attachment_type=carla.AttachmentType.Rigid)
            if self.camera is None:
                raise SessionError("camera-spawn-failed", stage)
            self._owned.append(self.camera)
            stage = "actor-readback"
            self._verify_actor_metadata()
            self.manifest.update(
                world_settings=public_fields(self.world.get_settings()),
                weather={"preset": "ClearNoon", "values": public_fields(self.world.get_weather())},
                vehicle={"actor_id": self.vehicle.id, "blueprint": self.vehicle.type_id,
                         "attributes": {key: self.vehicle.attributes[key] for key in self._vehicle_attributes},
                         "spawn_point_index": index, "spawn_point_count": len(spawn_points),
                         "spawn_transform": transform_record(spawn), "control_requested": dict(self._control_expected),
                         "control_verified": False},
                camera={"actor_id": self.camera.id, "blueprint": self.camera.type_id,
                        "attributes": {key: self.camera.attributes[key] for key in self._camera_attributes},
                        "parent_actor_id": self.camera.parent.id, "attachment_type": "Rigid",
                        "attachment_type_verification": "explicit-spawn-request-and-parent-readback",
                        "mount_requested": transform_record(self._mount), "mount_verified": False,
                        "requested_mount_matrix": self._mount.get_matrix(),
                        "intrinsics": cfg.camera.intrinsics()},
            )
            return self
        except BaseException as error:
            failure = error if isinstance(error, SessionError) else SessionError(
                "setup-operation-failed" if isinstance(error, Exception) else "setup-interrupted", stage, type(error).__name__)
            self.manifest["setup_error"] = failure.to_dict()
            self._cleanup()
            if not isinstance(error, Exception):
                raise
            raise failure from None

    def _wait_weather_readback(self, expected):
        # CARLA 0.9.16 SetWeatherParameters uses AsyncCall. Returning from
        # set_weather is not an application acknowledgement; preserve exact
        # equality and allow bounded delivery before failing the readback.
        if fields_match(expected, public_fields(self.world.get_weather()), exact_keys=True):
            return True
        deadline = time.monotonic() + min(2.0, self.cfg.connection.timeout_seconds)
        while True:
            if fields_match(expected, public_fields(self.world.get_weather()), exact_keys=True):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.02, remaining))

    def register_owned_sensor(self, sensor):
        """Register an attached sensor created by this active session's caller."""
        if not self._entered or self._closed or self.vehicle is None or self.camera is None:
            raise SessionError("session-not-active", "sensor-ownership")
        actor_id = getattr(sensor, "id", None)
        parent = getattr(sensor, "parent", None)
        if (type(actor_id) is not int or actor_id <= 0
                or not getattr(sensor, "type_id", "").startswith("sensor.")
                or parent is None or parent.id != self.vehicle.id
                or any(actor.id == actor_id for actor in self._owned)):
            raise SessionError("attached-sensor-ownership-invalid", "sensor-ownership")
        self._owned.append(sensor)
        self._extra_sensors.append(sensor)
        return sensor

    def _verify_actor_metadata(self):
        if self.vehicle.type_id != self.cfg.vehicle.blueprint or self.camera.type_id != self.cfg.camera.blueprint:
            raise SessionError("actor-blueprint-readback-mismatch", "actor-readback")
        if not attributes_match(self._vehicle_attributes, self.vehicle.attributes) or not attributes_match(self._camera_attributes, self.camera.attributes):
            raise SessionError("actor-attribute-readback-mismatch", "actor-readback")
        if self.camera.parent is None or self.camera.parent.id != self.vehicle.id:
            raise SessionError("camera-parent-readback-mismatch", "actor-readback")

    def verify_applied(self) -> dict[str, Any]:
        """Call after a caller-owned tick; verify cached control and rigid mount."""
        if not self._entered or self._closed or self.vehicle is None or self.camera is None:
            raise SessionError("session-not-active", "applied-readback")
        try:
            self._verify_actor_metadata()
            if not fields_match(self._world_expected, public_fields(self.world.get_settings())):
                raise SessionError("world-settings-readback-mismatch", "applied-readback")
            if not fields_match(self._weather_expected, public_fields(self.world.get_weather()), exact_keys=True):
                raise SessionError("weather-readback-mismatch", "applied-readback")
            control = self.vehicle.get_control()
            observed = {key: getattr(control, key) for key in self._control_expected}
            if not fields_match(self._control_expected, observed, exact_keys=True):
                raise SessionError("vehicle-control-readback-mismatch", "applied-readback")
            inverse = self.vehicle.get_transform().get_inverse_matrix()
            camera = self.camera.get_transform().get_matrix()
            desired = self._mount.get_matrix()
            if any(not math.isfinite(value) for matrix in (inverse, camera, desired)
                   for row in matrix for value in row):
                raise SessionError("camera-mount-readback-nonfinite", "applied-readback")
            relative = [[sum(inverse[row][k] * camera[k][column] for k in range(4)) for column in range(4)] for row in range(4)]
            residuals = [abs(relative[row][column] - desired[row][column]) for row in range(4) for column in range(4)]
            if any(not math.isfinite(value) for value in residuals):
                raise SessionError("camera-mount-readback-nonfinite", "applied-readback")
            max_error = max(residuals)
            if max_error > 0.002:
                raise SessionError("camera-mount-readback-mismatch", "applied-readback")
            self.manifest["vehicle"].update(control_applied=observed, control_verified=True)
            self.manifest["camera"].update(mount_verified=True, mount_relative_matrix=relative,
                                           mount_matrix_max_abs_error=max_error, mount_matrix_tolerance=0.002)
            self.manifest["applied_verified"] = True
            return self.manifest
        except SessionError:
            self.manifest["applied_verified"] = False
            raise
        except Exception as error:
            self.manifest["applied_verified"] = False
            raise SessionError("applied-readback-failed", "applied-readback", type(error).__name__) from None

    def _cleanup(self):
        if self._closed:
            return
        self._closed = True
        report: dict[str, Any] = {"status": "PASS", "errors": [], "owned_actor_ids": [actor.id for actor in self._owned],
                                  "sensor_stopped": None, "settings_restored": None, "weather_restored": None,
                                  "owned_actors_absent": not self._owned, "cleanup_reference_frame": None,
                                  "absence_observed_frame": None, "absence_waited_ticks": 0,
                                  "absence_verification": "fresh-snapshot-and-current-actor-list",
                                  "owned_actors_absent_in_snapshot": not self._owned,
                                  "owned_actors_absent_in_current_list": not self._owned}

        def attempt(label, operation):
            try:
                result = operation()
                if result is False:
                    raise SessionError("operation-returned-false")
                return True
            except Exception as error:
                report["errors"].append({"operation": label, "code": error.code if isinstance(error, SessionError) else "operation-failed",
                                         "exception_type": type(error).__name__})
                return False

        def capture_reference_frame():
            frame = self.world.get_snapshot().frame
            if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
                raise SessionError("cleanup-reference-frame-invalid", "cleanup")
            report["cleanup_reference_frame"] = frame

        reference_available = not self._owned or attempt("cleanup-reference-frame", capture_reference_frame)
        if self.camera is not None:
            report["sensor_stopped"] = attempt("sensor-stop", lambda: self.camera.stop() if self.camera.is_listening else None)
        for sensor in self._extra_sensors:
            stopped = attempt("sensor-stop", lambda sensor=sensor: sensor.stop() if sensor.is_listening else None)
            report["sensor_stopped"] = stopped and report["sensor_stopped"] is True
        for actor in reversed(self._owned):
            attempt("owned-actor-destroy", actor.destroy)
        if self._weather_touched:
            restored = attempt("weather-restore", lambda: self.world.set_weather(self._original_weather))
            if restored:
                restored = attempt("weather-readback", lambda: self._wait_weather_readback(self._weather_before))
            report["weather_restored"] = restored
        if self._settings_touched:
            restored = attempt("world-settings-restore", lambda: self.world.apply_settings(self._original_settings))
            if restored:
                restored = attempt("world-settings-readback", lambda: fields_match(self._settings_before, public_fields(self.world.get_settings()), exact_keys=True))
            report["settings_restored"] = restored
        if self._owned:
            # get_actor(id) retains cached descriptions after destruction in
            # CARLA 0.9.16. Use fresh snapshot membership and get_actors() without
            # an ID filter, which enumerates current streamed actor IDs.
            def observe_absence():
                if not reference_available or report["settings_restored"] is not True:
                    raise SessionError("owned-actor-absence-unverifiable", "cleanup")
                deadline = time.monotonic() + self.cfg.connection.timeout_seconds
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise SessionError("owned-actor-absence-timeout", "cleanup")
                    snapshot = self.world.wait_for_tick(remaining)
                    report["absence_waited_ticks"] += 1
                    if snapshot is None:
                        raise SessionError("post-cleanup-tick-unavailable", "cleanup")
                    frame = snapshot.frame
                    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
                        raise SessionError("post-cleanup-frame-invalid", "cleanup")
                    report["absence_observed_frame"] = frame
                    if frame > report["cleanup_reference_frame"]:
                        snapshot_absent = all(snapshot.find(actor.id) is None for actor in self._owned)
                        active_ids = {actor.id for actor in self.world.get_actors()}
                        list_absent = all(actor.id not in active_ids for actor in self._owned)
                        report["owned_actors_absent_in_snapshot"] = snapshot_absent
                        report["owned_actors_absent_in_current_list"] = list_absent
                        if snapshot_absent and list_absent:
                            return True

            report["owned_actors_absent"] = attempt("owned-actor-absence", observe_absence)
        if report["errors"]:
            report["status"] = "FAIL"
        self.cleanup_report = report

    def __exit__(self, exc_type, exc_value, traceback):
        self._cleanup()
        if self.cleanup_report["status"] != "PASS" and exc_type is None:
            raise SessionError("session-cleanup-failed", "cleanup")
        return False
