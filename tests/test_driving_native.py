"""CPU stand-in verifies native driving orchestration, never driving performance."""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from carla_tools.config import CollectionConfig
from carla_tools.driving import EventLog, body_lane_margin, run_drive_trial
from test_collection_session import Actor, Transform, World, carla_for
from test_driving_route import dense_route


def vector(x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


class DrivingActor(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curve = [(0.0, 1.0), (100.0, 0.5)]
        self.velocity = vector()
        self.bounding_box = SimpleNamespace(location=vector(), extent=vector(2.0, 0.8, 0.5),
                                            rotation=SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0))
        self.physics_changes = []
        self.control_requests = []

    def apply_control(self, control):
        self.control_requests.append({"frame": self.world.frame, "control": copy.deepcopy(vars(control)),
                                      "speed_mps": math.sqrt(sum(getattr(self.velocity, axis) ** 2 for axis in ("x", "y", "z")))})
        super().apply_control(control)

    def get_velocity(self):
        return copy.deepcopy(self.velocity)

    def get_physics_control(self):
        matrix = self.transform.get_matrix()
        wheels = []
        for index, local in enumerate(((1.4, -0.8, -0.1), (1.4, 0.8, -0.1),
                                        (-1.4, -0.8, -0.1), (-1.4, 0.8, -0.1))):
            world = [sum(matrix[row][column] * local[column] for column in range(3)) + matrix[row][3]
                     for row in range(3)]
            wheels.append(SimpleNamespace(position=vector(*(value * 100 for value in world)),
                                          max_steer_angle=35.0 if index < 2 else 0.0))
        return SimpleNamespace(wheels=wheels, steering_curve=[SimpleNamespace(x=x, y=y) for x, y in self.curve])

    def apply_physics_control(self, physics):
        self.curve = [(point.x, point.y) for point in physics.steering_curve]

    def get_wheel_steer_angle(self, wheel):
        if wheel not in ("front-left", "front-right"):
            raise AssertionError("wrong native wheel enum")
        return 35.0 * self.cached_control.steer

    def set_simulate_physics(self, enabled):
        self.physics_changes.append(enabled)

    def set_transform(self, transform):
        self.transform = copy.deepcopy(transform)

    def set_target_velocity(self, velocity):
        self.velocity = copy.deepcopy(velocity)

    def set_target_angular_velocity(self, velocity):
        self.angular_velocity = copy.deepcopy(velocity)


class ListeningSensor(Actor):
    def listen(self, callback):
        self.callback = callback
        self.is_listening = True

    def stop(self):
        if self.type_id == "sensor.other.collision" and self.world.late_event:
            if self.world.late_event == "collision":
                event = SimpleNamespace(frame=self.world.frame - 2, timestamp=(self.world.frame - 2) * .05,
                                        other_actor=SimpleNamespace(id=55, type_id="static.wall"),
                                        normal_impulse=vector(1.0, 0.0, 0.0))
            else:
                event = SimpleNamespace(frame="invalid-frame", timestamp=0.0)
            self.callback(event)
        super().stop()


class NativeStandInWorld(World):
    """Caches applied controls at tick, then sends that frame's RGB and snapshot."""

    def __init__(self, *, late_event=None, acknowledge_control=True, spawn_lateral_m=0.0, spawn_yaw_deg=0.0,
                 braking_ticks=0, braking_enabled=True, acknowledge_stop_control=True,
                 batch_response="ok", batch_failure_phase="drive", batch_hidden_tick=False):
        super().__init__()
        self.spawn_points = [Transform(vector(0.0, spawn_lateral_m, 0.1),
                                       SimpleNamespace(pitch=0.0, yaw=spawn_yaw_deg, roll=0.0))]
        self.late_event = late_event
        self.acknowledge_control = acknowledge_control
        self.tick_count = 0
        self.braking_ticks = braking_ticks
        self.braking_enabled = braking_enabled
        self.acknowledge_stop_control = acknowledge_stop_control
        self.braking_tick_count = 0
        self.batch_response = batch_response
        self.batch_failure_phase = batch_failure_phase
        self.batch_hidden_tick = batch_hidden_tick
        self.batch_requests = []
        self.batch_driving_started = False

    def get_blueprint_library(self):
        def find(identifier):
            if identifier.startswith("vehicle."):
                return self.vehicle_bp
            if identifier == "sensor.camera.rgb":
                return self.camera_bp
            return SimpleNamespace(id=identifier)
        return SimpleNamespace(find=find)

    def try_spawn_actor(self, blueprint, transform):
        actor = DrivingActor(self, 100, blueprint.id, blueprint.attributes, copy.deepcopy(transform))
        self.actors[100] = actor
        self.spawned_vehicle = actor
        return actor

    def spawn_actor(self, blueprint, transform, *, attach_to, attachment_type=None):
        actor_id = 101 if blueprint.id == "sensor.camera.rgb" else max(self.actors) + 1
        actor = ListeningSensor(self, actor_id, blueprint.id, getattr(blueprint, "attributes", {}),
                                copy.deepcopy(transform), attach_to)
        self.actors[actor_id] = actor
        return actor

    def get_snapshot(self):
        snapshots = {}
        for actor_id, actor in self.actors.items():
            pose = copy.deepcopy(actor.get_transform()) if actor.transform is not None else Transform()
            velocity = copy.deepcopy(getattr(actor, "velocity", vector()))
            snapshots[actor_id] = SimpleNamespace(id=actor_id, get_transform=lambda pose=pose: pose,
                                                  get_velocity=lambda velocity=velocity: velocity)
        return SimpleNamespace(frame=self.frame,
                               timestamp=SimpleNamespace(elapsed_seconds=self.frame * .05, delta_seconds=.05),
                               find=snapshots.get)

    def tick(self, timeout):
        self.frame += 1
        self.tick_count += 1
        vehicle = self.actors.get(100)
        if vehicle is not None:
            control = vehicle.command
            stopping = control.brake >= .99 and not control.hand_brake
            if (self.acknowledge_control or control.hand_brake) and (not stopping or self.acknowledge_stop_control):
                vehicle.cached_control = copy.deepcopy(control)
            if vehicle.cached_control.hand_brake:
                speed = 0.0
            elif vehicle.cached_control.brake >= .99:
                self.braking_tick_count += 1
                if not self.braking_enabled:
                    speed = 5.0
                else:
                    speed = max(0.0, 5.0 * (1 - self.braking_tick_count / (self.braking_ticks + 1)))
            else:
                speed = 5.0
            yaw = math.radians(vehicle.transform.rotation.yaw)
            vehicle.velocity = vector(speed * math.cos(yaw), speed * math.sin(yaw), 0.0)
            vehicle.transform.location.x += vehicle.velocity.x * .05
            vehicle.transform.location.y += vehicle.velocity.y * .05
        camera = self.actors.get(101)
        if camera is not None and camera.is_listening:
            width, height = int(camera.attributes["image_size_x"]), int(camera.attributes["image_size_y"])
            camera.callback(SimpleNamespace(frame=self.frame, timestamp=self.frame * .05,
                                            width=width, height=height, fov=float(camera.attributes["fov"]),
                                            transform=copy.deepcopy(camera.get_transform()),
                                            raw_data=bytes((20, 50, 200, 255, 70, 80, 140, 255)) * (width * height // 2)))
        return self.frame


def native_module(world):
    module = carla_for(world)
    module.Vector2D = lambda **kwargs: SimpleNamespace(**kwargs)
    module.Vector3D = lambda **kwargs: SimpleNamespace(**kwargs)
    module.VehicleWheelLocation = SimpleNamespace(FL_Wheel="front-left", FR_Wheel="front-right")
    module.command = SimpleNamespace(ApplyVehicleControl=lambda actor_id, control:
                                     SimpleNamespace(actor_id=actor_id, control=control))

    def apply_batch_sync(commands, do_tick):
        # This is the server handler barrier, not the next-tick applied-control
        # cache. Existing acknowledge_control faults still exercise that cache.
        if len(commands) != 1 or commands[0].actor_id != world.spawned_vehicle.id:
            raise AssertionError("Only the single owned vehicle may receive batch control")
        command = commands[0]
        control = command.control
        phase = ("park" if world.batch_driving_started else "static") if control.hand_brake else (
            "stop" if control.brake >= .99 else "drive")
        if phase == "drive":
            world.batch_driving_started = True
        record = {"frame_before": world.frame, "tick_count_before": world.tick_count,
                  "actor_id": command.actor_id, "control": copy.deepcopy(vars(control)),
                  "phase": phase, "do_tick": do_tick}
        world.batch_requests.append(record)
        world.spawned_vehicle.apply_control(control)
        if do_tick or (world.batch_hidden_tick and phase == "drive"):
            world.tick(1.0)
        record["frame_after"] = world.frame
        mode = world.batch_response if phase == world.batch_failure_phase else "ok"
        record["response_mode"] = mode
        response = SimpleNamespace(actor_id=command.actor_id, error="",
                                   has_error=lambda: False)
        if mode == "empty":
            return []
        if mode == "duplicate":
            return [response, copy.copy(response)]
        if mode == "wrong_actor":
            response.actor_id = command.actor_id + 1
        if mode == "error":
            response.error = "private remote response detail must not be serialized"
            response.has_error = lambda: True
        return [response]

    module.Client.return_value.apply_batch_sync = apply_batch_sync
    return module


class DrivingNativeTests(unittest.TestCase):
    def run_case(self, world, destination, condition=None):
        config = CollectionConfig()
        config = replace(config, camera=replace(config.camera, width=16, height=16),
                         verification=replace(config.verification, warmup_frames=2))
        with patch("carla_tools.driving.verify_geometry_route_in_world", return_value=None):
            return run_drive_trial(dense_route(), "straight", condition or {
                "name": "center", "lateral_m": 0.0, "yaw_deg": 0.0}, destination, config,
                carla_module=native_module(world))

    def test_tick_control_acknowledgement_same_frame_labels_and_owned_cleanup(self):
        world = NativeStandInWorld()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trial"
            result = self.run_case(world, directory)
            self.assertEqual(result["status"], "PASS", result)
            metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in (directory / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(metadata["status"], "COMPLETE")
            self.assertEqual(metadata["geometry_evidence"]["position_units"], "world-centimetres")
            self.assertEqual(world.spawned_vehicle.physics_changes, [False, True])
            self.assertEqual(metadata["vehicle_geometry"]["wheelbase_m"], 2.8)
            self.assertEqual(metadata["geometry_evidence"]["applied_steering_curve"], [[0.0, 1.0], [100.0, 1.0]])
            self.assertEqual(set(metadata["cleanup"]["owned_actor_ids"]), {100, 101, 102, 103})
            self.assertTrue(rows[-1]["terminal"])
            self.assertIsNone(rows[-1]["command_target_frame"])
            self.assertGreaterEqual(rows[-1]["tracking"]["reference_arc_m"], 19.0)
            for index, row in enumerate(rows):
                self.assertEqual(row["sensor_frame"], row["frame"])
                self.assertEqual(row["snapshot_frame"], row["frame"])
                self.assertEqual(row["tracking"]["label"]["ego_frame"], row["frame"])
                self.assertEqual(row["sensor_timestamp"], row["timestamp"])
                self.assertEqual(row["command_source_frame"], row["frame"])
                if index:
                    previous = rows[index - 1]
                    self.assertEqual(previous["command_target_frame"], row["frame"])
                    expected = {key: previous["command"][key] for key in row["vehicle"]["applied_control"]}
                    self.assertEqual(row["vehicle"]["applied_control"], expected)
                vehicle = row["vehicle"]["transform"]["matrix"]
                camera = row["camera"]["transform"]["matrix"]
                mount = metadata["camera_calibration"]["requested_mount_matrix"]
                relative = [[sum(vehicle[k][r] * camera[k][c] for k in range(3))
                             for c in range(3)] + [sum(vehicle[k][r] * (camera[k][3] - vehicle[k][3])
                                                      for k in range(3))] for r in range(3)]
                relative.append([0.0, 0.0, 0.0, 1.0])
                self.assertLess(max(abs(relative[r][c] - mount[r][c]) for r in range(4) for c in range(4)), .001)
            self.assertTrue(rows[0]["image"])
            self.assertTrue(rows[-1]["image"])
            self.assertEqual(metadata["events"], [])
            self.assertEqual(metadata["event_callback_errors"], [])
            self.assertEqual(len(world.batch_requests), len(rows) - 1 + 3)
            self.assertEqual([item["phase"] for item in world.batch_requests],
                             ["static"] + ["drive"] * (len(rows) - 1) + ["stop", "park"])
            for request in world.batch_requests:
                self.assertEqual(request["actor_id"], world.spawned_vehicle.id)
                self.assertIs(request["do_tick"], False)
                self.assertEqual(request["frame_before"], request["frame_after"])
            drive_batches = [item for item in world.batch_requests if item["phase"] == "drive"]
            for row, request in zip(rows[:-1], drive_batches):
                self.assertEqual(request["frame_before"], row["command_source_frame"])
                self.assertEqual(request["control"], {key: row["command"][key] for key in request["control"]})
            self.assertEqual(world.batch_requests[-2]["control"], metadata["stop_control"])
            self.assertEqual(world.batch_requests[-1]["control"], dict(metadata["stop_control"], hand_brake=True))
        self.assertEqual(list(world.actors), [1])
        self.assertEqual(vars(world.settings), vars(world.initial_settings))
        self.assertEqual(vars(world.weather), vars(world.initial_weather))

    def test_missing_next_tick_control_acknowledgement_fails_and_restores(self):
        world = NativeStandInWorld(acknowledge_control=False)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trial"
            result = self.run_case(world, directory)
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["error"]["code"], "driving-control-acknowledgement-mismatch")
            metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "FAILED")
            self.assertEqual(metadata["cleanup"]["status"], "PASS")
            failure = metadata["control_acknowledgement_failure"]
            first = json.loads((directory / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(failure["frame"], first["command_target_frame"])
            self.assertEqual(failure["expected"], {key: first["command"][key] for key in failure["expected"]})
            self.assertNotEqual(failure["observed"], failure["expected"])
        self.assertEqual(list(world.actors), [1])

    def test_invalid_batch_response_refuses_next_tick_and_cleans_owned_actors(self):
        for mode in ("empty", "duplicate", "wrong_actor", "error"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                world = NativeStandInWorld(batch_response=mode)
                directory = Path(temporary) / "trial"
                result = self.run_case(world, directory)
                self.assertEqual(result["status"], "FAIL")
                self.assertEqual(result["error"]["code"], "driving-control-batch-response-failed")
                metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["status"], "FAILED")
                self.assertEqual(metadata["cleanup"]["status"], "PASS")
                self.assertEqual(world.batch_requests[-1]["response_mode"], mode)
                self.assertEqual(world.tick_count, world.batch_requests[-1]["tick_count_before"])
                self.assertNotIn("private remote response detail", json.dumps(metadata))
                self.assertEqual(list(world.actors), [1])
                self.assertEqual(vars(world.settings), vars(world.initial_settings))
                self.assertEqual(vars(world.weather), vars(world.initial_weather))

    def test_batch_failure_during_static_drive_stop_or_park_never_reports_complete(self):
        for phase in ("static", "drive", "stop", "park"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                world = NativeStandInWorld(batch_response="error", batch_failure_phase=phase)
                directory = Path(temporary) / "trial"
                result = self.run_case(world, directory)
                self.assertEqual(result["status"], "FAIL")
                self.assertEqual(result["error"]["code"], "driving-control-batch-response-failed")
                metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["status"], "FAILED")
                self.assertEqual(metadata["cleanup"]["status"], "PASS")
                self.assertEqual(world.batch_requests[-1]["phase"], phase)
                self.assertEqual(world.tick_count, world.batch_requests[-1]["tick_count_before"])
                self.assertEqual(list(world.actors), [1])

    def test_hidden_batch_tick_is_rejected_by_consecutive_frame_contract(self):
        world = NativeStandInWorld(batch_hidden_tick=True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trial"
            result = self.run_case(world, directory)
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["error"]["code"], "world-frame-not-consecutive")
            metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "FAILED")
            self.assertEqual(metadata["cleanup"]["status"], "PASS")
            request = world.batch_requests[-1]
            self.assertIs(request["do_tick"], False)
            self.assertEqual(request["frame_after"], request["frame_before"] + 1)
        self.assertEqual(list(world.actors), [1])

    def test_stop_retains_applied_steer_and_only_parks_after_observed_low_speed(self):
        world = NativeStandInWorld(braking_ticks=5)
        condition = {"name": "left_offset", "lateral_m": -.3, "yaw_deg": 0.0}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trial"
            result = self.run_case(world, directory, condition)
            self.assertEqual(result["status"], "PASS", result)
            metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in (directory / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
            stop = metadata["stop_control"]
            last = rows[-1]["vehicle"]["applied_control"]
            self.assertNotEqual(last["steer"], 0.0)
            self.assertEqual(stop, {"throttle": 0.0, "brake": 1.0, "steer": last["steer"],
                                    "hand_brake": False, "reverse": False})
            observations = metadata["stop_observations"]
            self.assertGreaterEqual(len(observations), 10)
            self.assertTrue(any(row["speed_mps"] > .1 for row in observations))
            for previous, current in zip(observations, observations[1:]):
                self.assertEqual(current["frame"], previous["frame"] + 1)
                self.assertAlmostEqual(current["timestamp"] - previous["timestamp"], .05)
            for observation in observations:
                applied = observation["vehicle"]["applied_control"]
                self.assertEqual(applied["steer"], last["steer"])
                self.assertEqual(applied["throttle"], 0.0)
                self.assertEqual(applied["brake"], 1.0)
                if observation["speed_mps"] > .1:
                    self.assertIs(applied["hand_brake"], False)
            parking = [request for request in world.spawned_vehicle.control_requests
                       if request["frame"] >= metadata["drive_end_frame"] and request["control"]["hand_brake"]]
            self.assertTrue(parking)
            for request in parking:
                self.assertLessEqual(request["speed_mps"], .1)
                self.assertEqual(request["control"]["steer"], last["steer"])
            self.assertLessEqual(observations[-1]["speed_mps"], .1)
            self.assertEqual(metadata["cleanup"]["status"], "PASS")
        self.assertEqual(list(world.actors), [1])

    def test_stop_timeout_does_not_park_a_moving_actor_or_report_complete(self):
        world = NativeStandInWorld(braking_enabled=False)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trial"
            result = self.run_case(world, directory)
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["error"]["code"], "driving-stop-speed-timeout")
            metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "FAILED")
            self.assertEqual(metadata["cleanup"]["status"], "PASS")
            self.assertLessEqual(world.braking_tick_count, 40)
            self.assertGreater(world.braking_tick_count, 0)
            driving_end = metadata["drive_end_frame"]
            parking = [request for request in world.spawned_vehicle.control_requests
                       if request["frame"] >= driving_end and request["control"]["hand_brake"]]
            self.assertEqual(parking, [])
        self.assertEqual(list(world.actors), [1])

    def test_missing_stop_control_acknowledgement_fails_and_restores_state(self):
        world = NativeStandInWorld(acknowledge_stop_control=False)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trial"
            result = self.run_case(world, directory)
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["error"]["code"], "driving-stop-control-acknowledgement-mismatch")
            metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "FAILED")
            self.assertEqual(metadata["cleanup"]["status"], "PASS")
        self.assertEqual(list(world.actors), [1])
        self.assertEqual(vars(world.settings), vars(world.initial_settings))
        self.assertEqual(vars(world.weather), vars(world.initial_weather))

    def test_teleported_offset_is_measured_after_settling_and_labels_keep_original_lane(self):
        world = NativeStandInWorld()
        condition = {"name": "left_offset", "lateral_m": -.3, "yaw_deg": 0.0}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trial"
            result = self.run_case(world, directory, condition)
            self.assertEqual(result["status"], "PASS", result)
            metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
            first = json.loads((directory / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(world.spawned_vehicle.physics_changes, [False, True])
            measured = metadata["geometry_evidence"]["measurement_vehicle_transform"]
            self.assertAlmostEqual(measured["location"]["y"], -.3)
            self.assertAlmostEqual(first["tracking"]["lateral_error_m"], -.3)
            self.assertEqual(first["tracking"]["label"]["target_lane"], {"road_id": 1, "section_id": 0, "lane_id": -1})
            self.assertTrue(all(abs(point[1] - .3) < 1e-8 for point in first["tracking"]["label"]["waypoints_ego_m"]))
            self.assertGreater(first["command"]["steer"], 0.0)
            self.assertAlmostEqual(metadata["vehicle_geometry"]["wheelbase_m"], 2.8)
        self.assertEqual(list(world.actors), [1])

    def test_offcenter_and_misaligned_spawn_is_centered_on_the_pinned_route_before_drive(self):
        # The native Town01 spawn is a safe spawning location, not necessarily
        # the lane center. Center condition must use the pinned centerline.
        world = NativeStandInWorld(spawn_lateral_m=-.7554, spawn_yaw_deg=5.0)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trial"
            result = self.run_case(world, directory)
            self.assertEqual(result["status"], "PASS", result)
            metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
            first = json.loads((directory / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(world.spawned_vehicle.physics_changes, [False, True])
            self.assertAlmostEqual(first["tracking"]["reference_arc_m"], 0.0)
            self.assertAlmostEqual(first["tracking"]["lateral_error_m"], 0.0)
            self.assertAlmostEqual(first["tracking"]["heading_error_deg"], 0.0)
            measured = metadata["geometry_evidence"]["measurement_vehicle_transform"]
            self.assertAlmostEqual(measured["location"]["y"], 0.0)
            self.assertAlmostEqual(measured["location"]["z"], .1)
            self.assertAlmostEqual(measured["rotation"]["yaw"], 0.0)
            self.assertEqual(metadata["cleanup"]["status"], "PASS")
        self.assertEqual(list(world.actors), [1])

    def test_recovery_offsets_and_yaw_are_relative_to_pinned_route_not_offcenter_spawn(self):
        conditions = (("left_offset", -.3, 0.0), ("right_offset", .3, 0.0),
                      ("left_yaw", 0.0, -5.0), ("right_yaw", 0.0, 5.0))
        for name, lateral, yaw in conditions:
            with self.subTest(condition=name), tempfile.TemporaryDirectory() as temporary:
                world = NativeStandInWorld(spawn_lateral_m=-.7554, spawn_yaw_deg=5.0)
                directory = Path(temporary) / "trial"
                self.run_case(world, directory, {"name": name, "lateral_m": lateral, "yaw_deg": yaw})
                metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
                first = json.loads((directory / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
                self.assertEqual(world.spawned_vehicle.physics_changes, [False, True])
                self.assertAlmostEqual(first["tracking"]["reference_arc_m"], 0.0)
                self.assertAlmostEqual(first["tracking"]["lateral_error_m"], lateral)
                self.assertAlmostEqual(first["tracking"]["heading_error_deg"], yaw)
                self.assertEqual(first["tracking"]["label"]["target_lane"], {"road_id": 1, "section_id": 0, "lane_id": -1})
                expected_y = -3 * math.sin(math.radians(yaw)) - lateral * math.cos(math.radians(yaw))
                self.assertAlmostEqual(first["tracking"]["label"]["waypoints_ego_m"][0][1], expected_y)
                self.assertAlmostEqual(metadata["geometry_evidence"]["measurement_vehicle_transform"]["location"]["z"], .1)
                self.assertEqual(metadata["cleanup"]["status"], "PASS")
                self.assertEqual(list(world.actors), [1])

    def test_collision_delivered_during_sensor_stop_cannot_be_reported_as_pass(self):
        world = NativeStandInWorld(late_event="collision")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trial"
            result = self.run_case(world, directory)
            self.assertEqual(result["status"], "FAIL", "Late callback must invalidate direct producer PASS")
            metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "FAILED")
            self.assertEqual(metadata["events"][0]["kind"], "collision")
            self.assertGreaterEqual(metadata["events"][0]["frame"], metadata["drive_end_frame"])
            self.assertLessEqual(metadata["events"][0]["frame"], metadata["post_stop_frame"])
            self.assertEqual(metadata["cleanup"]["status"], "PASS")
        self.assertEqual(list(world.actors), [1])

    def test_invalid_callback_during_sensor_stop_cannot_be_reported_as_pass(self):
        world = NativeStandInWorld(late_event="invalid")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trial"
            result = self.run_case(world, directory)
            self.assertEqual(result["status"], "FAIL", "Late callback error must invalidate direct producer PASS")
            metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "FAILED")
            self.assertEqual(metadata["event_callback_errors"][0]["code"], "event-callback-invalid")
        self.assertEqual(list(world.actors), [1])

    def test_existing_output_is_preserved_and_no_world_is_connected(self):
        world = NativeStandInWorld()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trial"
            directory.mkdir()
            sentinel = directory / "episode.json"
            sentinel.write_bytes(b"preserved existing bytes")
            result = self.run_case(world, directory)
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["error"]["code"], "driving-output-already-exists")
            self.assertEqual(sentinel.read_bytes(), b"preserved existing bytes")
        self.assertEqual(world.settings_calls, [])
        self.assertEqual(world.tick_count, 0)

    def test_bounding_box_margin_accounts_for_yaw_and_offset(self):
        pose = Transform(vector(), SimpleNamespace(pitch=0.0, yaw=30.0, roll=0.0))
        pose = {"matrix": pose.get_matrix()}
        tracking = {"route_tangent_yaw_deg": 0.0, "lateral_error_m": .3, "lane_width_m": 3.5}
        bbox = {"location": {"x": .2, "y": 0.0, "z": 0.0},
                "extent": {"x": 2.0, "y": .8, "z": .5}}
        expected = 1.75 - (.3 + .2 * .5) - (2.0 * .5 + .8 * math.cos(math.radians(30)))
        self.assertAlmostEqual(body_lane_margin(pose, tracking, bbox), expected)
        self.assertLess(expected, 0.0)

    def test_invalid_event_payload_is_sanitized(self):
        events = EventLog()
        events.receive("collision", SimpleNamespace(frame="secret-token", timestamp=0.0))
        rows, errors = events.snapshot()
        self.assertEqual(rows, [])
        self.assertEqual(errors, [{"code": "event-callback-invalid", "exception_type": "ValueError"}])
        self.assertNotIn("secret-token", json.dumps(errors))


if __name__ == "__main__":
    unittest.main()
