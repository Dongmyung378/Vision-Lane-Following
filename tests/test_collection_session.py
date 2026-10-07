"""Ownership, applied-state verification, and cleanup contracts without CARLA."""

from __future__ import annotations

import copy
import json
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from carla_tools.config import CollectionConfig
from carla_tools.session import CollectionSession, SessionError


def multiply(first, second):
    return [[sum(first[row][k] * second[k][column] for k in range(4)) for column in range(4)] for row in range(4)]


class Transform:
    def __init__(self, location=None, rotation=None, matrix=None):
        self.location = location or SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.rotation = rotation or SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0)
        self.matrix = matrix

    def get_matrix(self):
        if self.matrix is not None:
            return self.matrix
        pitch, yaw, roll = [math.radians(getattr(self.rotation, name)) for name in ("pitch", "yaw", "roll")]
        cp, sp, cy, sy, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw), math.cos(roll), math.sin(roll)
        return [[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, self.location.x],
                [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, self.location.y],
                [sp, -cp * sr, cp * cr, self.location.z], [0.0, 0.0, 0.0, 1.0]]

    def get_inverse_matrix(self):
        source = self.get_matrix()
        inverse = [[source[column][row] for column in range(3)] + [0.0] for row in range(3)] + [[0.0, 0.0, 0.0, 1.0]]
        for row in range(3):
            inverse[row][3] = -sum(inverse[row][column] * source[column][3] for column in range(3))
        return inverse


class Blueprint:
    def __init__(self, identifier, attributes):
        self.id = identifier
        self.attributes = dict(attributes)
        self.nonmodifiable = set()

    def has_attribute(self, name):
        return name in self.attributes

    def get_attribute(self, name):
        return SimpleNamespace(is_modifiable=name not in self.nonmodifiable)

    def set_attribute(self, name, value):
        self.attributes[name] = value


class Actor:
    def __init__(self, world, actor_id, identifier, attributes=None, transform=None, parent=None):
        self.world, self.id, self.type_id = world, actor_id, identifier
        self.attributes = dict(attributes or {})
        self.transform, self.parent = transform, parent
        self.is_listening = False
        self.destroyed = False
        self.destroy_result = True
        self.stop_error = None
        self.cached_control = SimpleNamespace(throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False)

    def apply_control(self, control):
        self.command = copy.copy(control)

    def get_control(self):
        return self.cached_control

    def get_transform(self):
        if self.parent is not None:
            return Transform(matrix=multiply(self.parent.get_transform().get_matrix(), self.transform.get_matrix()))
        return self.transform

    def stop(self):
        if self.stop_error:
            raise self.stop_error
        self.is_listening = False

    def destroy(self):
        self.destroyed = True
        if self.destroy_result:
            if self.world.disappearance_after_waits is None:
                self.world.actors.pop(self.id, None)
            else:
                self.world.pending_destroy.add(self.id)
        return self.destroy_result


def weather(cloudiness):
    return SimpleNamespace(cloudiness=cloudiness, precipitation=0.0, precipitation_deposits=0.0,
                           wind_intensity=5.0, sun_azimuth_angle=0.0, sun_altitude_angle=75.0,
                           fog_density=2.0, fog_distance=0.0, fog_falloff=0.2, wetness=0.0,
                           scattering_intensity=1.0, mie_scattering_scale=0.03,
                           rayleigh_scattering_scale=0.0331, dust_storm=0.0)


class World:
    def __init__(self):
        self.map_name = "/Game/Carla/Maps/Town01"
        self.settings = SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None,
                                        no_rendering_mode=False, substepping=True,
                                        max_substep_delta_time=0.016, max_substeps=6,
                                        actor_active_distance=777.0, tile_stream_distance=2222.0,
                                        spectator_as_ego=False)
        self.weather = weather(77.0)
        self.weather.future_weather_field = 19.0
        self.initial_settings, self.initial_weather = copy.deepcopy(self.settings), copy.deepcopy(self.weather)
        self.actors = {1: Actor(self, 1, "spectator")}
        self.vehicle_bp = Blueprint("vehicle.tesla.model3", {"role_name": "default", "color": "255,0,0"})
        self.camera_bp = Blueprint("sensor.camera.rgb", {name: "" for name in (
            "image_size_x", "image_size_y", "fov", "sensor_tick", "enable_postprocess_effects", "gamma", "lens_k", "lens_kcube")})
        self.spawn_points = [Transform(SimpleNamespace(x=10.0, y=20.0, z=1.0)), Transform(SimpleNamespace(x=30.0, y=40.0, z=1.0))]
        self.settings_calls, self.weather_calls, self.spawn_calls, self.wait_calls = [], [], [], []
        self.collision = self.camera_error = self.fail_restore = self.ignore_settings = False
        self.wrong_parent = self.wrong_attributes = self.ignore_weather = False
        self.frame = 1000
        self.wait_frames = []
        self.disappearance_after_waits = None
        self.pending_destroy = set()

    def get_map(self):
        return SimpleNamespace(name=self.map_name, get_spawn_points=lambda: self.spawn_points)

    def get_settings(self):
        return copy.deepcopy(self.settings)

    def get_weather(self):
        return copy.deepcopy(self.weather)

    def apply_settings(self, settings):
        self.settings_calls.append(copy.deepcopy(settings))
        if self.fail_restore and not settings.synchronous_mode:
            raise RuntimeError("private-person/secret-token")
        self.settings = copy.deepcopy(settings)
        if self.ignore_settings and settings.synchronous_mode:
            self.settings.fixed_delta_seconds = 0.1
        return 10

    def set_weather(self, value):
        self.weather_calls.append(copy.deepcopy(value))
        if not self.ignore_weather or value.cloudiness == 77.0:
            self.weather = copy.deepcopy(value)

    def get_actors(self):
        return list(self.actors.values())

    def get_actor(self, actor_id):
        return self.actors.get(actor_id)

    def get_snapshot(self):
        actor_ids = set(self.actors)
        return SimpleNamespace(frame=self.frame,
                               find=lambda actor_id: SimpleNamespace(id=actor_id) if actor_id in actor_ids else None)

    def get_blueprint_library(self):
        return SimpleNamespace(find=lambda name: self.vehicle_bp if name.startswith("vehicle.") else self.camera_bp)

    def try_spawn_actor(self, blueprint, transform):
        self.spawn_calls.append(transform)
        if self.collision:
            return None
        actor = Actor(self, 100, blueprint.id, blueprint.attributes, transform)
        self.actors[100] = actor
        return actor

    def spawn_actor(self, blueprint, transform, *, attach_to, attachment_type):
        if self.camera_error:
            raise RuntimeError("private-person/secret-token")
        assert attachment_type == "Rigid"
        actor = Actor(self, 101, blueprint.id, blueprint.attributes, transform, None if self.wrong_parent else attach_to)
        if self.wrong_attributes:
            actor.attributes["lens_k"] = "-1.0"
        self.actors[101] = actor
        return actor

    def user_tick(self):
        self.frame += 1
        for actor in self.actors.values():
            if hasattr(actor, "command"):
                actor.cached_control = copy.copy(actor.command)

    def wait_for_tick(self, timeout):
        self.wait_calls.append(timeout)
        if self.settings.synchronous_mode:
            return None
        self.frame = self.wait_frames.pop(0) if self.wait_frames else self.frame + 1
        if self.disappearance_after_waits is not None and len(self.wait_calls) >= self.disappearance_after_waits:
            for actor_id in self.pending_destroy:
                self.actors.pop(actor_id, None)
            self.pending_destroy.clear()
        return self.get_snapshot()


def carla_for(world, version="0.9.16"):
    client = SimpleNamespace(set_timeout=Mock(), get_client_version=lambda: "0.9.16",
                             get_server_version=lambda: version, get_world=lambda: world,
                             load_world=Mock(side_effect=AssertionError("Map reload is forbidden")))
    return SimpleNamespace(Client=Mock(return_value=client), Transform=Transform,
                           Location=lambda **kw: SimpleNamespace(**kw), Rotation=lambda **kw: SimpleNamespace(**kw),
                           VehicleControl=lambda **kw: SimpleNamespace(**kw), AttachmentType=SimpleNamespace(Rigid="Rigid"),
                           WeatherParameters=SimpleNamespace(ClearNoon=weather(10.0)))


class SessionTests(unittest.TestCase):
    def test_fixed_setup_applied_readback_and_full_state_restoration(self):
        world = World()
        module = carla_for(world)
        session = CollectionSession(CollectionConfig(), module)
        with session:
            self.assertTrue(world.settings.synchronous_mode)
            self.assertFalse(world.settings.no_rendering_mode)
            self.assertEqual(world.settings.fixed_delta_seconds, 0.05)
            self.assertFalse(session.manifest["applied_verified"])
            self.assertEqual(session.camera.parent.id, session.vehicle.id)
            self.assertEqual(session.camera.attributes["lens_k"], "0.0")
            self.assertEqual(session.camera.attributes["enable_postprocess_effects"], "false")
            self.assertEqual(world.spawn_calls, [world.spawn_points[0]])
            self.assertEqual(world.wait_calls, [])
            world.user_tick()
            session.verify_applied()
            self.assertTrue(session.manifest["applied_verified"])
            self.assertTrue(session.manifest["vehicle"]["control_verified"])
            self.assertTrue(session.manifest["camera"]["mount_verified"])
            self.assertEqual(session.manifest["camera"]["requested_mount_matrix"], session.camera_mount_transform.get_matrix())
            session.camera.is_listening = True
        self.assertEqual(session.cleanup_report["status"], "PASS")
        self.assertTrue(session.cleanup_report["owned_actors_absent"])
        self.assertTrue(session.cleanup_report["settings_restored"])
        self.assertTrue(session.cleanup_report["weather_restored"])
        self.assertEqual(vars(world.settings), vars(world.initial_settings))
        self.assertEqual(vars(world.weather), vars(world.initial_weather))
        self.assertEqual(list(world.actors), [1])
        module.Client.return_value.load_world.assert_not_called()

    def test_map_sync_and_busy_world_refuse_before_mutation(self):
        for cause in ("map", "sync", "vehicle", "walker"):
            with self.subTest(cause=cause):
                world = World()
                if cause == "map":
                    world.map_name = "/Game/Carla/Maps/Town02"
                elif cause == "sync":
                    world.settings.synchronous_mode = True
                else:
                    world.actors[2] = Actor(world, 2, "vehicle.other" if cause == "vehicle" else "walker.pedestrian.0001")
                session = CollectionSession(CollectionConfig(), carla_for(world))
                with self.assertRaises(SessionError):
                    session.__enter__()
                self.assertEqual(world.settings_calls, [])
                self.assertEqual(world.weather_calls, [])
                self.assertEqual(world.spawn_calls, [])
                self.assertEqual(world.wait_calls, [])
                self.assertFalse(world.actors[1].destroyed)
                if cause in ("vehicle", "walker"):
                    self.assertFalse(world.actors[2].destroyed)

    def test_version_mismatch_refuses_before_world_mutation(self):
        world = World()
        session = CollectionSession(CollectionConfig(), carla_for(world, "0.9.15"))
        with self.assertRaisesRegex(SessionError, "carla-version-mismatch"):
            session.__enter__()
        self.assertEqual(world.settings_calls, [])
        self.assertEqual(world.spawn_calls, [])

    def test_existing_sensor_is_refused_before_mutation_and_never_destroyed(self):
        world = World()
        sensor = Actor(world, 2, "sensor.camera.rgb")
        world.actors[2] = sensor
        session = CollectionSession(CollectionConfig(), carla_for(world))
        with self.assertRaisesRegex(SessionError, "existing-sensors-refused"):
            session.__enter__()
        self.assertEqual(world.settings_calls, [])
        self.assertEqual(world.weather_calls, [])
        self.assertEqual(world.spawn_calls, [])
        self.assertFalse(sensor.destroyed)

    def test_invalid_direct_config_is_rejected_before_client_creation(self):
        module = carla_for(World())
        session = CollectionSession(replace(CollectionConfig(), map="Town02"), module)
        with self.assertRaises(SessionError) as failure:
            session.__enter__()
        self.assertEqual(failure.exception.stage, "configuration")
        module.Client.assert_not_called()

    def test_fixed_spawn_collision_never_tries_another_point_and_restores_state(self):
        world = World()
        world.collision = True
        session = CollectionSession(CollectionConfig(), carla_for(world))
        with self.assertRaisesRegex(SessionError, "fixed-spawn-collision"):
            session.__enter__()
        self.assertEqual(world.spawn_calls, [world.spawn_points[0]])
        self.assertEqual(session.cleanup_report["status"], "PASS")
        self.assertEqual(vars(world.settings), vars(world.initial_settings))
        self.assertEqual(vars(world.weather), vars(world.initial_weather))

    def test_unavailable_spawn_index_does_not_change_world(self):
        world = World()
        config = replace(CollectionConfig(), vehicle=replace(CollectionConfig().vehicle, spawn_point_index=7))
        session = CollectionSession(config, carla_for(world))
        with self.assertRaisesRegex(SessionError, "spawn-point-index-unavailable"):
            session.__enter__()
        self.assertEqual(world.settings_calls, [])

    def test_nonmodifiable_blueprint_attribute_is_refused_before_mutation(self):
        world = World()
        world.camera_bp.nonmodifiable.add("lens_k")
        session = CollectionSession(CollectionConfig(), carla_for(world))
        with self.assertRaisesRegex(SessionError, "blueprint-attribute-not-modifiable"):
            session.__enter__()
        self.assertEqual(world.settings_calls, [])

    def test_partial_setup_failure_destroys_only_owned_vehicle(self):
        world = World()
        world.camera_error = True
        session = CollectionSession(CollectionConfig(), carla_for(world))
        with self.assertRaises(SessionError) as failure:
            session.__enter__()
        self.assertEqual(failure.exception.code, "setup-operation-failed")
        self.assertEqual(session.cleanup_report["status"], "PASS")
        self.assertEqual(list(world.actors), [1])
        self.assertNotIn("secret-token", json.dumps(session.manifest))

    def test_settings_weather_and_actor_readback_mismatches_fail_and_restore(self):
        for cause, code in (("ignore_settings", "world-settings-readback-mismatch"),
                            ("ignore_weather", "weather-readback-mismatch"),
                            ("wrong_parent", "camera-parent-readback-mismatch"),
                            ("wrong_attributes", "actor-attribute-readback-mismatch")):
            with self.subTest(cause=cause):
                world = World()
                setattr(world, cause, True)
                session = CollectionSession(CollectionConfig(), carla_for(world))
                with self.assertRaisesRegex(SessionError, code):
                    session.__enter__()
                self.assertEqual(session.cleanup_report["status"], "PASS")
                self.assertEqual(vars(world.settings), vars(world.initial_settings))
                self.assertEqual(vars(world.weather), vars(world.initial_weather))
                self.assertEqual(list(world.actors), [1])

    def test_control_readback_requires_a_caller_tick_and_never_claims_early_success(self):
        world = World()
        with CollectionSession(CollectionConfig(), carla_for(world)) as session:
            with self.assertRaisesRegex(SessionError, "vehicle-control-readback-mismatch"):
                session.verify_applied()
            self.assertFalse(session.manifest["applied_verified"])
            world.user_tick()
            session.verify_applied()
            self.assertTrue(session.manifest["applied_verified"])

    def test_changed_mount_is_detected_from_relative_world_transforms(self):
        world = World()
        with CollectionSession(CollectionConfig(), carla_for(world)) as session:
            world.user_tick()
            session.camera.transform = Transform(SimpleNamespace(x=2.0, y=0.0, z=2.4))
            with self.assertRaisesRegex(SessionError, "camera-mount-readback-mismatch"):
                session.verify_applied()
            self.assertFalse(session.manifest["applied_verified"])

    def test_cleanup_failure_raises_after_successful_body(self):
        world = World()
        session = CollectionSession(CollectionConfig(), carla_for(world))
        with self.assertRaisesRegex(SessionError, "session-cleanup-failed"):
            with session:
                world.user_tick()
                session.verify_applied()
                session.camera.destroy_result = False
                world.wait_for_tick = Mock(return_value=None)
        self.assertEqual(session.cleanup_report["status"], "FAIL")
        self.assertFalse(session.cleanup_report["owned_actors_absent"])

    def test_cleanup_waits_for_later_frame_and_delayed_actor_disappearance(self):
        world = World()
        world.disappearance_after_waits = 3
        world.wait_frames = [1000, 1001, 1002]
        session = CollectionSession(CollectionConfig(), carla_for(world))
        with session:
            pass
        self.assertEqual(session.cleanup_report["status"], "PASS")
        self.assertEqual(session.cleanup_report["cleanup_reference_frame"], 1000)
        self.assertEqual(session.cleanup_report["absence_observed_frame"], 1002)
        self.assertEqual(session.cleanup_report["absence_waited_ticks"], 3)
        self.assertEqual(list(world.actors), [1])

    def test_cleanup_rejects_stale_snapshot_even_when_actors_already_absent(self):
        world = World()
        world.wait_frames = [1000, 1001]
        session = CollectionSession(CollectionConfig(), carla_for(world))
        with session:
            pass
        self.assertEqual(session.cleanup_report["status"], "PASS")
        self.assertEqual(session.cleanup_report["absence_waited_ticks"], 2)
        self.assertGreater(session.cleanup_report["absence_observed_frame"], session.cleanup_report["cleanup_reference_frame"])

    def test_cleanup_uses_current_membership_despite_persistent_get_actor_description(self):
        world = World()
        session = CollectionSession(CollectionConfig(), carla_for(world))
        with session:
            cached_descriptions = dict(world.actors)
            world.get_actor = Mock(side_effect=cached_descriptions.get)
        self.assertIsNotNone(world.get_actor(100))
        self.assertIsNotNone(world.get_actor(101))
        self.assertEqual(session.cleanup_report["status"], "PASS")
        self.assertTrue(session.cleanup_report["owned_actors_absent_in_snapshot"])
        self.assertTrue(session.cleanup_report["owned_actors_absent_in_current_list"])
        self.assertEqual(world.get_actor.call_count, 2)

    def test_cleanup_requires_absence_in_both_fresh_snapshot_and_current_list(self):
        for remaining_in in ("snapshot", "list"):
            with self.subTest(remaining_in=remaining_in):
                world = World()
                session = CollectionSession(CollectionConfig(), carla_for(world))
                with patch("carla_tools.session.time.monotonic", side_effect=[0.0, 0.0, 31.0]):
                    with self.assertRaisesRegex(SessionError, "session-cleanup-failed"):
                        with session:
                            if remaining_in == "snapshot":
                                original_wait = world.wait_for_tick

                                def wait_with_actor(timeout):
                                    snapshot = original_wait(timeout)
                                    snapshot.find = lambda actor_id: SimpleNamespace(id=actor_id)
                                    return snapshot

                                world.wait_for_tick = wait_with_actor
                            else:
                                camera = session.camera
                                world.get_actors = Mock(return_value=[camera])
                self.assertEqual(session.cleanup_report["status"], "FAIL")
                self.assertFalse(session.cleanup_report["owned_actors_absent"])
                self.assertEqual(session.cleanup_report["errors"][-1]["code"], "owned-actor-absence-timeout")
                self.assertEqual(session.cleanup_report["owned_actors_absent_in_snapshot"], remaining_in != "snapshot")
                self.assertEqual(session.cleanup_report["owned_actors_absent_in_current_list"], remaining_in != "list")

    def test_cleanup_timeout_never_reports_stale_snapshot_as_absence_proof(self):
        world = World()
        world.wait_frames = [1000, 1000]
        cfg = replace(CollectionConfig(), connection=replace(CollectionConfig().connection, timeout_seconds=0.5))
        session = CollectionSession(cfg, carla_for(world))
        with patch("carla_tools.session.time.monotonic", side_effect=[0.0, 0.0, 0.1, 0.6]):
            with self.assertRaisesRegex(SessionError, "session-cleanup-failed"):
                with session:
                    pass
        self.assertEqual(session.cleanup_report["status"], "FAIL")
        self.assertFalse(session.cleanup_report["owned_actors_absent"])
        self.assertEqual(session.cleanup_report["absence_waited_ticks"], 2)
        self.assertEqual(session.cleanup_report["errors"][-1]["code"], "owned-actor-absence-timeout")
        self.assertEqual(world.wait_calls, [0.5, 0.4])

    def test_cleanup_failure_preserves_existing_error_and_captures_report(self):
        world = World()
        session = CollectionSession(CollectionConfig(), carla_for(world))
        original = ValueError("existing-body-error")
        with self.assertRaises(ValueError) as caught:
            with session:
                session.camera.is_listening = True
                session.camera.stop_error = RuntimeError("private-person/secret-token")
                raise original
        self.assertIs(caught.exception, original)
        self.assertEqual(session.cleanup_report["status"], "FAIL")
        self.assertNotIn("secret-token", json.dumps(session.cleanup_report))
        self.assertEqual(list(world.actors), [1])

    def test_settings_restore_failure_cannot_be_reported_as_pass(self):
        world = World()
        session = CollectionSession(CollectionConfig(), carla_for(world))
        with self.assertRaisesRegex(SessionError, "session-cleanup-failed"):
            with session:
                world.fail_restore = True
        self.assertEqual(session.cleanup_report["status"], "FAIL")
        self.assertFalse(session.cleanup_report["settings_restored"])
        self.assertNotIn("secret-token", json.dumps(session.cleanup_report))

    def test_interrupted_setup_still_restores_state_and_propagates_interrupt(self):
        world = World()
        world.spawn_actor = Mock(side_effect=KeyboardInterrupt())
        session = CollectionSession(CollectionConfig(), carla_for(world))
        with self.assertRaises(KeyboardInterrupt):
            session.__enter__()
        self.assertEqual(session.cleanup_report["status"], "PASS")
        self.assertEqual(list(world.actors), [1])
        self.assertEqual(vars(world.settings), vars(world.initial_settings))

    def test_session_cannot_be_reused_after_cleanup(self):
        session = CollectionSession(CollectionConfig(), carla_for(World()))
        with session:
            pass
        with self.assertRaisesRegex(SessionError, "session-cannot-be-reentered"):
            session.__enter__()

    def test_nonfinite_camera_pose_cannot_be_marked_verified(self):
        for invalid in (float("nan"), float("inf")):
            with self.subTest(value=invalid):
                world = World()
                session = CollectionSession(CollectionConfig(), carla_for(world))
                with session:
                    world.user_tick()
                    matrix = session.camera.get_transform().get_matrix()
                    matrix[2][3] = invalid
                    session.camera.get_transform = Mock(return_value=Transform(matrix=matrix))
                    with self.assertRaisesRegex(SessionError, "camera-mount-readback-nonfinite"):
                        session.verify_applied()
                    self.assertFalse(session.manifest["applied_verified"])
                self.assertEqual(session.cleanup_report["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
