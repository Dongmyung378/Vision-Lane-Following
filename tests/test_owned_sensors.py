"""Additional sensor ownership and cleanup, using CPU-only CARLA fixtures."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from carla_tools.config import CollectionConfig
from carla_tools.session import CollectionSession, SessionError
from test_collection_session import Actor, Transform, World, carla_for


def sensor_for(world, vehicle, actor_id=102, identifier="sensor.other.collision"):
    sensor = Actor(world, actor_id, identifier, transform=Transform(), parent=vehicle)
    sensor.is_listening = True
    world.actors[actor_id] = sensor
    return sensor


class OwnedSensorTests(unittest.TestCase):
    def test_collision_and_lane_sensors_stop_before_any_owned_actor_is_destroyed(self):
        world = World()
        session = CollectionSession(CollectionConfig(), carla_for(world))
        operations = []
        with session:
            world.user_tick()
            session.verify_applied()
            collision = sensor_for(world, session.vehicle)
            invasion = sensor_for(world, session.vehicle, 103, "sensor.other.lane_invasion")
            session.register_owned_sensor(collision)
            session.register_owned_sensor(invasion)
            session.camera.is_listening = True
            for actor in (session.vehicle, session.camera, collision, invasion):
                original_destroy = actor.destroy

                def destroy(actor=actor, original=original_destroy):
                    operations.append(("destroy", actor.id))
                    return original()

                actor.destroy = destroy
                if actor.type_id.startswith("sensor."):
                    original_stop = actor.stop

                    def stop(actor=actor, original=original_stop):
                        operations.append(("stop", actor.id))
                        return original()

                    actor.stop = stop
        report = session.cleanup_report
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(set(report["owned_actor_ids"]), {100, 101, 102, 103})
        self.assertTrue(report["sensor_stopped"])
        self.assertTrue(report["owned_actors_absent_in_snapshot"])
        self.assertTrue(report["owned_actors_absent_in_current_list"])
        stops = [index for index, item in enumerate(operations) if item[0] == "stop"]
        destroys = [index for index, item in enumerate(operations) if item[0] == "destroy"]
        self.assertEqual(len(stops), 3)
        self.assertEqual(len(destroys), 4)
        self.assertLess(max(stops), min(destroys))
        self.assertFalse(collision.is_listening)
        self.assertFalse(invasion.is_listening)
        self.assertEqual(list(world.actors), [1])
        self.assertEqual(vars(world.settings), vars(world.initial_settings))
        self.assertEqual(vars(world.weather), vars(world.initial_weather))

    def test_registration_requires_an_active_session_with_owned_vehicle_and_camera(self):
        world = World()
        session = CollectionSession(CollectionConfig(), carla_for(world))
        candidate = SimpleNamespace(id=102, type_id="sensor.other.collision", parent=None)
        with self.assertRaises(SessionError):
            session.register_owned_sensor(candidate)
        with session:
            candidate.parent = session.vehicle
            camera = session.camera
            session.camera = None
            try:
                with self.assertRaises(SessionError):
                    session.register_owned_sensor(candidate)
            finally:
                session.camera = camera
            vehicle = session.vehicle
            session.vehicle = None
            try:
                with self.assertRaises(SessionError):
                    session.register_owned_sensor(candidate)
            finally:
                session.vehicle = vehicle
        with self.assertRaises(SessionError):
            session.register_owned_sensor(candidate)
        self.assertEqual(session.cleanup_report["owned_actor_ids"], [100, 101])

    def test_registration_rejects_wrong_parent_without_taking_ownership(self):
        for parent_kind in ("missing", "different-vehicle"):
            with self.subTest(parent=parent_kind):
                world = World()
                session = CollectionSession(CollectionConfig(), carla_for(world))
                with session:
                    parent = None if parent_kind == "missing" else SimpleNamespace(id=200)
                    candidate = Actor(world, 102, "sensor.other.collision", transform=Transform(), parent=parent)
                    with self.assertRaises(SessionError):
                        session.register_owned_sensor(candidate)
                self.assertFalse(candidate.destroyed)
                self.assertEqual(session.cleanup_report["owned_actor_ids"], [100, 101])

    def test_registration_rejects_non_sensor_actors_and_invalid_sensor_ids(self):
        invalid = [(102, "vehicle.other"), (102, "spectator"), (102, "sensor"),
                   (0, "sensor.other.collision"), (-1, "sensor.other.collision"),
                   (True, "sensor.other.collision"), (102.5, "sensor.other.collision"),
                   ("102", "sensor.other.collision")]
        for actor_id, identifier in invalid:
            with self.subTest(actor_id=actor_id, identifier=identifier):
                world = World()
                session = CollectionSession(CollectionConfig(), carla_for(world))
                with session:
                    candidate = Actor(world, actor_id, identifier, transform=Transform(), parent=session.vehicle)
                    with self.assertRaises(SessionError):
                        session.register_owned_sensor(candidate)
                self.assertFalse(candidate.destroyed)
                self.assertEqual(session.cleanup_report["owned_actor_ids"], [100, 101])

    def test_duplicate_ids_and_existing_owned_camera_cannot_be_registered_twice(self):
        world = World()
        session = CollectionSession(CollectionConfig(), carla_for(world))
        with session:
            collision = sensor_for(world, session.vehicle)
            session.register_owned_sensor(collision)
            same_id = Actor(world, collision.id, "sensor.other.lane_invasion", transform=Transform(), parent=session.vehicle)
            for candidate in (collision, same_id, session.camera):
                with self.subTest(identifier=candidate.type_id):
                    with self.assertRaises(SessionError):
                        session.register_owned_sensor(candidate)
        self.assertEqual(session.cleanup_report["owned_actor_ids"].count(102), 1)
        self.assertEqual(session.cleanup_report["owned_actor_ids"].count(101), 1)
        self.assertFalse(same_id.destroyed)
        self.assertEqual(list(world.actors), [1])

    def test_non_listening_sensor_is_destroyed_without_calling_stop(self):
        world = World()
        session = CollectionSession(CollectionConfig(), carla_for(world))
        with session:
            collision = sensor_for(world, session.vehicle)
            collision.is_listening = False
            collision.stop = Mock(side_effect=AssertionError("inactive sensor must not be stopped"))
            session.register_owned_sensor(collision)
        collision.stop.assert_not_called()
        self.assertTrue(collision.destroyed)
        self.assertEqual(session.cleanup_report["status"], "PASS")

    def test_extra_sensor_stop_failure_is_recorded_and_other_cleanup_continues(self):
        world = World()
        session = CollectionSession(CollectionConfig(), carla_for(world))
        with self.assertRaisesRegex(SessionError, "session-cleanup-failed"):
            with session:
                collision = sensor_for(world, session.vehicle)
                invasion = sensor_for(world, session.vehicle, 103, "sensor.other.lane_invasion")
                collision.stop_error = RuntimeError("private-person/secret-token")
                session.register_owned_sensor(collision)
                session.register_owned_sensor(invasion)
        report = session.cleanup_report
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["sensor_stopped"])
        self.assertIn("sensor-stop", [error["operation"] for error in report["errors"]])
        self.assertNotIn("secret-token", json.dumps(report))
        self.assertTrue(collision.destroyed)
        self.assertTrue(invasion.destroyed)
        self.assertFalse(invasion.is_listening)
        self.assertTrue(report["settings_restored"])
        self.assertTrue(report["weather_restored"])
        self.assertTrue(report["owned_actors_absent"])
        self.assertEqual(list(world.actors), [1])

    def test_cleanup_failure_does_not_replace_an_existing_body_exception(self):
        world = World()
        session = CollectionSession(CollectionConfig(), carla_for(world))
        body_error = ValueError("driving-episode-failed")
        with self.assertRaises(ValueError) as caught:
            with session:
                collision = sensor_for(world, session.vehicle)
                collision.stop_error = RuntimeError("sensor stop failed")
                session.register_owned_sensor(collision)
                raise body_error
        self.assertIs(caught.exception, body_error)
        self.assertEqual(session.cleanup_report["status"], "FAIL")
        self.assertTrue(collision.destroyed)
        self.assertEqual(list(world.actors), [1])

    def test_cleanup_waits_for_delayed_extra_sensor_disappearance_in_fresh_snapshot(self):
        world = World()
        world.disappearance_after_waits = 3
        world.wait_frames = [1000, 1001, 1002]
        session = CollectionSession(CollectionConfig(), carla_for(world))
        with session:
            collision = sensor_for(world, session.vehicle)
            session.register_owned_sensor(collision)
            cached_descriptions = dict(world.actors)
            world.get_actor = Mock(side_effect=cached_descriptions.get)
        report = session.cleanup_report
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["absence_waited_ticks"], 3)
        self.assertGreater(report["absence_observed_frame"], report["cleanup_reference_frame"])
        self.assertTrue(report["owned_actors_absent_in_snapshot"])
        self.assertTrue(report["owned_actors_absent_in_current_list"])
        world.get_actor.assert_not_called()
        self.assertEqual(list(world.actors), [1])

    def test_remaining_extra_sensor_prevents_false_cleanup_success(self):
        world = World()
        session = CollectionSession(CollectionConfig(), carla_for(world))
        with patch("carla_tools.session.time.monotonic", side_effect=[0.0, 0.0, 31.0]):
            with self.assertRaisesRegex(SessionError, "session-cleanup-failed"):
                with session:
                    collision = sensor_for(world, session.vehicle)
                    collision.destroy_result = False
                    session.register_owned_sensor(collision)
        report = session.cleanup_report
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["owned_actors_absent"])
        self.assertFalse(report["owned_actors_absent_in_snapshot"])
        self.assertFalse(report["owned_actors_absent_in_current_list"])
        self.assertIn(102, report["owned_actor_ids"])
        self.assertIn(102, world.actors)
        self.assertTrue(report["settings_restored"])
        self.assertTrue(report["weather_restored"])


if __name__ == "__main__":
    unittest.main()
