"""Owned CARLA D9/D10 privileged driving trials; never a model benchmark."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from .capture import BoundedSensorQueue, _calibration, _tick, _transform, _vector
from .color import color_provenance
from .config import collection_config_from_dict
from .contracts import read_strict_json
from .controller import PurePursuitController, VehicleGeometry, load_controller_config
from .driving_route import FixedRouteTracker, perturbation_transform
from .geometry_survey import verify_geometry_route_in_world
from .projection import encode_rgb8_png, project_waypoints
from .session import CollectionSession, fields_match
from scripts.carla_rgb_probe import rgb_from_bgra
from scripts.verify_collection import write_report

ROOT = Path(__file__).resolve().parents[1]
SCOPE = "D8-D10-privileged-controller-development-QA"

class DrivingError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code

def load_validation_config(path=ROOT / "configs/driving_validation_d10.json"):
    value = read_strict_json(path)
    expected = read_strict_json(ROOT / "configs/driving_validation_d10.json")
    if value != expected or type(value.get("schema_version")) is not int or value.get("schema_version") != 1 or value.get("scope") != SCOPE:
        raise DrivingError("driving-validation-config-invalid")
    for key, number in value.items():
        if key in ("schema_version", "scope", "initial_conditions"):
            continue
        if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
            raise DrivingError("driving-validation-config-invalid")
    return value

def body_lane_margin(pose, tracking, bbox, route=None):
    if route is not None:
        margins = []
        anchors = route["centerline_anchor_sequence"]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    local = [bbox["location"][axis] + sign * bbox["extent"][axis]
                             for axis, sign in zip(("x", "y", "z"), (sx, sy, sz))]
                    point = [sum(pose["matrix"][r][c] * local[c] for c in range(3)) + pose["matrix"][r][3] for r in range(2)]
                    candidates = []
                    for first, second in zip(anchors, anchors[1:]):
                        a, b = first["world_location"], second["world_location"]
                        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
                        length = math.hypot(dx, dy)
                        if length <= 1e-8:
                            raise DrivingError("degenerate-driving-lane-segment")
                        fraction = max(0., min(1., ((point[0] - a["x"]) * dx + (point[1] - a["y"]) * dy) / length**2))
                        px, py = a["x"] + fraction * dx, a["y"] + fraction * dy
                        lateral = ((point[0] - px) * -dy + (point[1] - py) * dx) / length
                        width = first["lane_width_m"] + fraction * (second["lane_width_m"] - first["lane_width_m"])
                        candidates.append((math.hypot(point[0] - px, point[1] - py), width / 2 - abs(lateral)))
                    margins.append(min(candidates, key=lambda value: value[0])[1])
        return min(margins)
    yaw = math.radians(tracking["route_tangent_yaw_deg"])
    normal = [-math.sin(yaw), math.cos(yaw), 0.0]
    projections = [sum(normal[r] * pose["matrix"][r][c] for r in range(3)) for c in range(3)]
    center_lat = tracking["lateral_error_m"] + sum(projections[c] * bbox["location"][axis] for c, axis in enumerate(("x", "y", "z")))
    half = sum(abs(projections[c]) * bbox["extent"][axis] for c, axis in enumerate(("x", "y", "z")))
    return tracking["lane_width_m"] / 2 - abs(center_lat) - half

class EventLog:
    """Drain callbacks admitted in Python; native unsubscribe is not a join."""

    def __init__(self):
        self.events, self.errors = [], []
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self._in_flight = 0
        self._sealed = False
        self._frozen = False

    def receive(self, kind, event):
        with self.condition:
            if self._sealed:
                self.errors.append({"code": "event-callback-after-seal"})
                return
            # Register before reading any native payload property, which may
            # release the GIL or be slow while shutdown proceeds elsewhere.
            self._in_flight += 1
        row, failure = None, None
        try:
            row = {"kind": kind, "frame": int(event.frame), "timestamp": float(event.timestamp)}
            if kind == "collision":
                row.update(other_actor_id=int(event.other_actor.id), other_actor_type=str(event.other_actor.type_id),
                           impulse=_vector(event.normal_impulse))
            else:
                row["markings"] = [{"type": str(mark.type), "color": str(mark.color), "lane_change": str(mark.lane_change),
                                    "width_m": float(mark.width)} for mark in event.crossed_lane_markings]
            json.dumps(row, allow_nan=False)
        except BaseException as error:
            failure = {"code": "event-callback-invalid", "exception_type": type(error).__name__}
        finally:
            with self.condition:
                if self._frozen:
                    self.errors.append({"code": "event-callback-completed-after-seal"})
                elif failure is not None:
                    self.errors.append(failure)
                else:
                    self.events.append(row)
                self._in_flight -= 1
                self.condition.notify_all()

    def seal(self, timeout_seconds=2.0):
        """Close admission, bound the existing Python work, then freeze rows.

        A timeout remains a permanent callback error even if its worker later
        finishes. This cannot prove that native I/O will never invoke a future
        callback that has not reached Python yet.
        """
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("event-callback-drain-timeout-invalid")
        deadline = time.monotonic() + timeout_seconds
        with self.condition:
            if self._frozen:
                return self._in_flight == 0
            self._sealed = True
            while self._in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.errors.append({"code": "event-callback-drain-timeout"})
                    break
                self.condition.wait(remaining)
            self._frozen = True
            return self._in_flight == 0

    def publication(self):
        """Keep the saved snapshot and its PASS/FAIL decision under one lock."""
        return self.lock

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.events)), list(self.errors)

def _controls(control):
    return {key: getattr(control, key) for key in ("throttle", "brake", "steer", "hand_brake", "reverse")}

def _apply_control(carla, session, values):
    """Wait for the owned actor's command handler; advance no hidden frames."""
    responses = session.client.apply_batch_sync(
        [carla.command.ApplyVehicleControl(session.vehicle.id, carla.VehicleControl(**values))], False)
    if len(responses) != 1 or responses[0].has_error() or responses[0].actor_id != session.vehicle.id:
        raise DrivingError("driving-control-batch-response-failed")

def _acknowledge_control(metadata, frame, expected, observed, code):
    if not fields_match(expected, observed, exact_keys=True):
        metadata["control_acknowledgement_failure"] = {
            "frame": frame, "expected": dict(expected), "observed": dict(observed)}
        raise DrivingError(code)

def _mount_error(pose, camera, requested):
    matrix, target = pose["matrix"], camera["matrix"]
    inverse = [[matrix[c][r] for c in range(3)] + [0.] for r in range(3)] + [[0., 0., 0., 1.]]
    for r in range(3):
        inverse[r][3] = -sum(inverse[r][c] * matrix[c][3] for c in range(3))
    error = max(abs(sum(inverse[r][k] * target[k][c] for k in range(4)) - requested[r][c])
                for r in range(4) for c in range(4))
    if not math.isfinite(error) or error > .001:
        raise DrivingError("driving-camera-mount-mismatch")
    return error

def _write_rgb(directory, row, image):
    raw = bytes(image.raw_data)
    rgb, _ = rgb_from_bgra(raw, image.width, image.height)
    png = encode_rgb8_png(rgb, image.width, image.height)
    relative = f"images/{row['frame']:08d}.png"
    with (directory / relative).open("xb") as handle:
        handle.write(png)
        handle.flush()
        os.fsync(handle.fileno())
    row["image"] = {"path": relative, "sha256": hashlib.sha256(png).hexdigest(),
                    "width": image.width, "height": image.height,
                    "color_provenance": color_provenance(raw, rgb, image.width, image.height)}

def run_drive_trial(route, case, condition, output_dir, collection_config, *, carla_module,
                    validation_config=None, controller_config=None):
    """Stationary setup -> native measured geometry -> same-frame drive -> cleanup."""
    limits = validation_config or load_validation_config()
    control_cfg = controller_config or load_controller_config()
    cfg_dict = collection_config.to_dict()
    cfg_dict["vehicle"]["spawn_point_index"] = route["spawn_point_index"]
    cfg = collection_config_from_dict(cfg_dict)
    directory = Path(output_dir)
    report = {"schema_version": 1, "status": "FAIL", "scope": SCOPE, "case": case,
              "condition": condition["name"], "episode_directory": directory.name}
    metadata = {"schema_version": 1, "status": "RUNNING", "scope": SCOPE, "episode_id": directory.name,
                "geometry_case": case, "condition": condition, "route_contract": route,
                "collection_config": cfg.to_dict(), "controller_config": control_cfg.to_dict(),
                "controller_config_sha256": control_cfg.sha256, "validation_config": limits,
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    owned = False
    session, events = None, EventLog()
    rows, stage = [], "output-preflight"
    try:
        if directory.exists() or directory.is_symlink():
            raise DrivingError("driving-output-already-exists")
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "images").mkdir()
        owned = True
        write_report(directory / "episode.json", metadata)
        session = CollectionSession(cfg, carla_module=carla_module)
        stage = "native-session"
        with session:
            verify_geometry_route_in_world(route, session.world.get_map(), carla_module)
            images = BoundedSensorQueue(cfg.verification.queue_capacity)
            session.camera.listen(images.receive)
            previous = None
            for _ in range(cfg.verification.warmup_frames):
                _, _, previous = _tick(session, cfg, images, previous)
            session.verify_applied()
            stage = "fixed-route-perturbation"
            initial_pose = _transform(session.vehicle.get_transform())
            anchors = route["centerline_anchor_sequence"]
            start, next_point = anchors[0]["world_location"], anchors[1]["world_location"]
            tangent_yaw = math.degrees(math.atan2(next_point["y"] - start["y"], next_point["x"] - start["x"]))
            # CARLA spawn transforms can be offset from the lane centre.
            # Anchor every condition to the ORIGINAL route before perturbing.
            centered_native = carla_module.Transform(
                carla_module.Location(x=start["x"], y=start["y"], z=initial_pose["location"]["z"]),
                carla_module.Rotation(pitch=initial_pose["rotation"]["pitch"], yaw=tangent_yaw, roll=initial_pose["rotation"]["roll"]))
            centered = _transform(centered_native)
            moved = perturbation_transform(centered, condition["lateral_m"], condition["yaw_deg"])
            metadata["initialization"] = {"policy": "pinned-route-start-xy-and-tangent-preserve-settled-height",
                "spawn_settled_transform": initial_pose, "center_requested_transform": centered, "perturbed_requested_transform": moved}
            session.vehicle.set_simulate_physics(False)
            session.vehicle.set_transform(carla_module.Transform(carla_module.Location(**moved["location"]),
                                                                carla_module.Rotation(**moved["rotation"])))
            session.vehicle.set_simulate_physics(True)
            session.vehicle.set_target_velocity(carla_module.Vector3D(x=0., y=0., z=0.))
            session.vehicle.set_target_angular_velocity(carla_module.Vector3D(x=0., y=0., z=0.))
            _apply_control(carla_module, session, cfg.vehicle.control.to_dict())
            for _ in range(limits["settling_frames"]):
                _, _, previous = _tick(session, cfg, images, previous)
            session.verify_applied()
            stage = "native-vehicle-geometry"
            physics = session.vehicle.get_physics_control()
            original_curve = [[float(point.x), float(point.y)] for point in physics.steering_curve]
            physics.steering_curve = [carla_module.Vector2D(x=0., y=1.), carla_module.Vector2D(x=100., y=1.)]
            session.vehicle.apply_physics_control(physics)
            _, snapshot, previous = _tick(session, cfg, images, previous)
            session.verify_applied()
            physics = session.vehicle.get_physics_control()
            native_pose = snapshot.find(session.vehicle.id).get_transform()
            applied_curve = [[float(point.x), float(point.y)] for point in physics.steering_curve]
            if applied_curve != [[0., 1.], [100., 1.]]:
                raise DrivingError("native-steering-curve-readback-mismatch")
            measured_pose = _transform(native_pose)
            matrix = measured_pose["matrix"]
            # Reuse the recorded native rigid transform for exact offline replay.
            inverse = [[matrix[c][r] for c in range(3)] +
                       [-sum(matrix[c][r] * matrix[c][3] for c in range(3))] for r in range(3)]
            inverse.append([0., 0., 0., 1.])
            geometry = VehicleGeometry.from_physics_control(physics, SimpleNamespace(get_inverse_matrix=lambda: inverse))
            metadata.update(vehicle_geometry=geometry.to_dict(), geometry_evidence={
                "measurement_vehicle_transform": measured_pose,
                "wheels": [{"position_cm": _vector(wheel.position), "max_steer_angle_deg": float(wheel.max_steer_angle)} for wheel in physics.wheels],
                "original_steering_curve": original_curve, "applied_steering_curve": applied_curve,
                "position_units": "world-centimetres", "measurement_after_warmup": True,
                "steering_normalization": "measured-front-steer-limit-times-constant-factor"},
                camera_calibration=_calibration(cfg, session.camera_mount_transform.get_matrix()),
                applied_manifest=session.manifest)
            bbox = session.vehicle.bounding_box
            if any(abs(getattr(bbox.rotation, axis)) > .001 for axis in ("pitch", "yaw", "roll")):
                raise DrivingError("unsupported-rotated-vehicle-bounding-box")
            metadata["bounding_box"] = {"location": _vector(bbox.location), "extent": _vector(bbox.extent)}
            stage = "event-sensors"
            library = session.world.get_blueprint_library()
            for kind, blueprint in (("collision", "sensor.other.collision"), ("lane", "sensor.other.lane_invasion")):
                sensor = session.world.spawn_actor(library.find(blueprint), carla_module.Transform(), attach_to=session.vehicle)
                session.register_owned_sensor(sensor)
                sensor.listen(lambda event, kind=kind: events.receive(kind, event))
            tracker = FixedRouteTracker(route, max_speed_mps=limits["maximum_speed_mps"], minimum_dt_seconds=.05)
            controller = PurePursuitController(control_cfg, geometry)
            expected_control = cfg.vehicle.control.to_dict()
            start_time = None
            stage = "privileged-driving"
            with (directory / "steps.jsonl").open("x", encoding="utf-8", newline="\n") as log:
                while True:
                    image, snapshot, row = _tick(session, cfg, images, previous)
                    actor = snapshot.find(session.vehicle.id)
                    if actor is None:
                        raise DrivingError("driving-actor-snapshot-missing")
                    pose, camera = _transform(actor.get_transform()), _transform(image.transform)
                    velocity = _vector(actor.get_velocity())
                    speed = math.sqrt(sum(value * value for value in velocity.values()))
                    tracking = tracker.observe(pose, row["frame"], row["timestamp"])
                    applied_control = _controls(session.vehicle.get_control())
                    _acknowledge_control(metadata, row["frame"], expected_control, applied_control,
                                         "driving-control-acknowledgement-mismatch")
                    _mount_error(pose, camera, metadata["camera_calibration"]["requested_mount_matrix"])
                    row.update(index=len(rows), vehicle={"transform": pose, "velocity": velocity, "applied_control": applied_control},
                               camera={"transform": camera}, speed_mps=speed, tracking=tracking,
                               body_lane_margin_m=body_lane_margin(pose, tracking, metadata["bounding_box"], route),
                               native_wheel_angles_deg={"front_left": float(session.vehicle.get_wheel_steer_angle(carla_module.VehicleWheelLocation.FL_Wheel)),
                                                       "front_right": float(session.vehicle.get_wheel_steer_angle(carla_module.VehicleWheelLocation.FR_Wheel))}, image=None)
                    start_time = row["timestamp"] if start_time is None else start_time
                    row["command"] = controller.step(tracking["label"]["waypoints_ego_m"], speed)
                    terminal = tracking["reference_arc_m"] >= limits["goal_reference_m"]
                    row.update(terminal=terminal, command_source_frame=row["frame"], command_target_frame=None if terminal else row["frame"] + 1)
                    if len(rows) % limits["image_stride"] == 0 or terminal:
                        _write_rgb(directory, row, image)
                    log.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                    log.flush()
                    rows.append(row)
                    previous = row
                    logged_events, callback_errors = events.snapshot()
                    if callback_errors or any(event["frame"] >= rows[0]["frame"] for event in logged_events):
                        raise DrivingError("driving-collision-or-lane-event")
                    if speed > limits["maximum_speed_mps"] or abs(tracking["lateral_error_m"]) > limits["maximum_lateral_error_m"]:
                        raise DrivingError("driving-speed-or-lateral-limit")
                    if row["body_lane_margin_m"] < limits["minimum_body_lane_margin_m"]:
                        raise DrivingError("driving-body-outside-safe-lane-margin")
                    if row["timestamp"] - start_time > limits["maximum_simulation_seconds"]:
                        raise DrivingError("driving-simulation-timeout")
                    if terminal:
                        break
                    expected_control = {key: row["command"][key] for key in expected_control}
                    _apply_control(carla_module, session, expected_control)
                os.fsync(log.fileno())
            metadata.update(drive_end_frame=rows[-1]["frame"], step_count=len(rows))
            stage = "controlled-stop"
            stop_control = {"throttle": 0., "steer": rows[-1]["vehicle"]["applied_control"]["steer"],
                            "brake": 1., "hand_brake": False, "reverse": False}
            metadata.update(stop_control=stop_control, stop_observations=[])
            def observe_stop(expected):
                image, snapshot, observation = _tick(session, cfg, images, previous)
                actor = snapshot.find(session.vehicle.id)
                if actor is None:
                    raise DrivingError("driving-stop-actor-missing")
                pose, velocity = _transform(actor.get_transform()), _vector(actor.get_velocity())
                speed = math.sqrt(sum(value * value for value in velocity.values()))
                applied = _controls(session.vehicle.get_control())
                _acknowledge_control(metadata, observation["frame"], expected, applied,
                                     "driving-stop-control-acknowledgement-mismatch")
                _mount_error(pose, _transform(image.transform), metadata["camera_calibration"]["requested_mount_matrix"])
                observation = {"frame": observation["frame"], "timestamp": observation["timestamp"],
                    "vehicle": {"transform": pose, "velocity": velocity, "applied_control": applied}, "speed_mps": speed,
                    "native_wheel_angles_deg": {"front_left": float(session.vehicle.get_wheel_steer_angle(carla_module.VehicleWheelLocation.FL_Wheel)),
                                                "front_right": float(session.vehicle.get_wheel_steer_angle(carla_module.VehicleWheelLocation.FR_Wheel))},
                    "body_lane_margin_m": body_lane_margin(pose, None, metadata["bounding_box"], route)}
                metadata["stop_observations"].append(observation)
                if observation["body_lane_margin_m"] < limits["minimum_body_lane_margin_m"]:
                    raise DrivingError("driving-stop-body-outside-safe-lane-margin")
                return observation
            _apply_control(carla_module, session, stop_control)
            for index in range(math.ceil(limits["maximum_stop_seconds"] / cfg.world.fixed_delta_seconds)):
                previous = observe_stop(stop_control)
                if index + 1 >= limits["settling_frames"] and previous["speed_mps"] <= limits["stop_speed_tolerance_mps"]:
                    break
            else:
                raise DrivingError("driving-stop-speed-timeout")
            parked = dict(stop_control, hand_brake=True)
            _apply_control(carla_module, session, parked)
            for _ in range(2):
                previous = observe_stop(parked)
                if previous["speed_mps"] > limits["stop_speed_tolerance_mps"]:
                    raise DrivingError("driving-parked-speed-mismatch")
            metadata["post_stop_frame"] = previous["frame"]
            stage = "cleanup"
        metadata.update(status="COMPLETE", cleanup=session.cleanup_report,
                        trace_sha256=hashlib.sha256((directory / "steps.jsonl").read_bytes()).hexdigest())
        if session.cleanup_report["status"] != "PASS":
            raise DrivingError("driving-cleanup-incomplete")
        report["status"] = "PASS"
    except BaseException as error:
        report["error"] = {"code": getattr(error, "code", "native-driving-operation-failed"), "stage": stage,
                           "exception_type": type(error).__name__}
        metadata.update(status="FAILED", error=report["error"])
    # The session context has stopped all sensors and completed owned cleanup.
    # Its native unsubscribe does not join already-entered Python callbacks.
    events.seal(2.0)
    with events.publication():
        metadata["events"], metadata["event_callback_errors"] = events.snapshot()
        if report["status"] == "PASS" and (metadata["event_callback_errors"] or
                any(event["frame"] >= rows[0]["frame"] for event in metadata["events"])):
            report.update(status="FAIL", error={"code": "driving-late-collision-or-lane-event", "stage": "cleanup", "exception_type": "DrivingError"})
            metadata.update(status="FAILED", error=report["error"])
        metadata["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        if session is not None:
            metadata["cleanup"] = session.cleanup_report
        if owned:
            write_report(directory / "episode.json", metadata)
    report["step_count"] = len(rows)
    return report
