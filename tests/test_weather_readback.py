"""Delayed CARLA weather RPC delivery must not weaken verified readbacks."""

from __future__ import annotations

import copy
import json
import unittest
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import Mock, patch

from carla_tools.config import CollectionConfig
from carla_tools.session import CollectionSession, SessionError, public_fields
from test_collection_session import World, carla_for


class FakeClock:
    """Advance on polling sleeps so deadline tests never wait in real time."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        if not 0 < seconds <= 0.02:
            raise AssertionError("Weather polling must yield for a bounded interval")
        self.sleeps.append(seconds)
        self.now += seconds


@contextmanager
def polling_clock(clock):
    with patch("carla_tools.session.time.monotonic", clock.monotonic), patch(
        "carla_tools.session.time.sleep", clock.sleep
    ):
        yield


class DelayedWeatherWorld(World):
    """Only get_weather RPC processing delivers the queued setter request."""

    def __init__(self, *, apply_stale_reads=2, restore_stale_reads=2):
        super().__init__()
        self.apply_stale_reads = apply_stale_reads
        self.restore_stale_reads = restore_stale_reads
        self.phase = "initial"
        self.pending_weather = None
        self.remaining_stale_reads = 0
        self.events = []
        self.stale_observations = {"apply": 0, "restore": 0}
        self.permanent_apply_mismatch = False
        self.permanent_restore_mismatch = False
        self.extra_applied_field = False
        self.omit_original_field_on_restore = False
        self.tick = Mock(side_effect=AssertionError("Weather RPC delivery must not tick the world"))

    def set_weather(self, value):
        self.weather_calls.append(copy.deepcopy(value))
        self.phase = "restore" if value.cloudiness == self.initial_weather.cloudiness else "apply"
        self.pending_weather = copy.deepcopy(value)
        self.remaining_stale_reads = (
            self.restore_stale_reads if self.phase == "restore" else self.apply_stale_reads
        )
        self.events.append(("set_weather", self.phase))

    def get_weather(self):
        if self.pending_weather is not None:
            if self.remaining_stale_reads:
                self.remaining_stale_reads -= 1
                self.stale_observations[self.phase] += 1
            elif not (
                self.permanent_apply_mismatch if self.phase == "apply" else self.permanent_restore_mismatch
            ):
                self.weather = copy.deepcopy(self.pending_weather)
                self.pending_weather = None
                if self.phase == "apply" and self.extra_applied_field:
                    self.weather.unrequested_weather_field = 1.0
                if self.phase == "restore" and self.omit_original_field_on_restore:
                    del self.weather.future_weather_field
        self.events.append(("get_weather", self.phase, self.weather.cloudiness))
        return copy.deepcopy(self.weather)

    def try_spawn_actor(self, blueprint, transform):
        self.events.append(("spawn_vehicle", self.phase))
        return super().try_spawn_actor(blueprint, transform)


class WeatherReadbackTests(unittest.TestCase):
    def test_immediate_verified_readback_needs_no_clock_sleep_or_world_tick(self):
        world = DelayedWeatherWorld(apply_stale_reads=0)
        module = carla_for(world)
        world.set_weather(module.WeatherParameters.ClearNoon)
        session = CollectionSession(CollectionConfig(), module)
        session.world = world
        with patch("carla_tools.session.time.monotonic") as monotonic, patch(
            "carla_tools.session.time.sleep"
        ) as sleep:
            self.assertTrue(session._wait_weather_readback(public_fields(module.WeatherParameters.ClearNoon)))
        monotonic.assert_not_called()
        sleep.assert_not_called()
        world.tick.assert_not_called()
        self.assertEqual(world.wait_calls, [])

    def test_delayed_setup_and_restore_succeed_without_advancing_capture_frames(self):
        world = DelayedWeatherWorld()
        module = carla_for(world)
        session = CollectionSession(CollectionConfig(), module)
        clock = FakeClock()
        with polling_clock(clock):
            with session:
                self.assertEqual(world.stale_observations["apply"], 2)
                self.assertEqual(public_fields(world.weather), public_fields(module.WeatherParameters.ClearNoon))
                self.assertEqual(world.frame, 1000)
                self.assertEqual(world.wait_calls, [])
                self.assertFalse(session.manifest["applied_verified"])
                spawn_index = world.events.index(("spawn_vehicle", "apply"))
                matching_read = world.events.index(("get_weather", "apply", 10.0))
                self.assertLess(matching_read, spawn_index)
                # Cached controls still require the caller's own physics tick.
                world.user_tick()
                session.verify_applied()
                self.assertTrue(session.manifest["applied_verified"])
        self.assertEqual(world.stale_observations["restore"], 2)
        self.assertEqual([value.cloudiness for value in world.weather_calls], [10.0, 77.0])
        self.assertEqual(public_fields(world.weather), public_fields(world.initial_weather))
        self.assertEqual(world.weather.future_weather_field, 19.0)
        self.assertEqual(public_fields(world.settings), public_fields(world.initial_settings))
        self.assertTrue(session.cleanup_report["weather_restored"])
        self.assertEqual(session.cleanup_report["status"], "PASS")
        self.assertTrue(session.cleanup_report["owned_actors_absent"])
        self.assertEqual(session.cleanup_report["absence_waited_ticks"], 1)
        self.assertEqual(len(world.wait_calls), 1)  # Cleanup absence proof, not weather delivery.
        self.assertGreaterEqual(len(clock.sleeps), 2)
        world.tick.assert_not_called()

    def test_permanent_setup_mismatch_stops_at_grace_or_shorter_connection_timeout(self):
        for connection_timeout in (30.0, 0.055):
            with self.subTest(connection_timeout=connection_timeout):
                cfg = CollectionConfig()
                cfg = replace(cfg, connection=replace(cfg.connection, timeout_seconds=connection_timeout))
                world = DelayedWeatherWorld(apply_stale_reads=0, restore_stale_reads=0)
                world.permanent_apply_mismatch = True
                session = CollectionSession(cfg, carla_for(world))
                clock = FakeClock()
                with polling_clock(clock), self.assertRaises(SessionError) as caught:
                    session.__enter__()
                self.assertEqual(caught.exception.code, "weather-readback-mismatch")
                self.assertEqual(caught.exception.stage, "weather")
                budget = min(2.0, connection_timeout)
                self.assertAlmostEqual(clock.now, budget)
                self.assertAlmostEqual(sum(clock.sleeps), budget)
                self.assertEqual(world.spawn_calls, [])
                self.assertEqual(list(world.actors), [1])
                self.assertFalse(session.manifest["applied_verified"])
                self.assertEqual(session.cleanup_report["status"], "PASS")
                self.assertTrue(session.cleanup_report["weather_restored"])
                self.assertEqual(public_fields(world.settings), public_fields(world.initial_settings))
                self.assertEqual(public_fields(world.weather), public_fields(world.initial_weather))
                self.assertEqual([value.cloudiness for value in world.weather_calls], [10.0, 77.0])
                self.assertEqual(world.wait_calls, [])
                world.tick.assert_not_called()

    def test_matching_weather_values_with_unexpected_key_still_fail_setup(self):
        world = DelayedWeatherWorld(apply_stale_reads=0, restore_stale_reads=0)
        world.extra_applied_field = True
        session = CollectionSession(CollectionConfig(), carla_for(world))
        clock = FakeClock()
        with polling_clock(clock), self.assertRaisesRegex(SessionError, "weather-readback-mismatch"):
            session.__enter__()
        self.assertAlmostEqual(clock.now, 2.0)
        self.assertEqual(world.spawn_calls, [])
        self.assertEqual(session.cleanup_report["status"], "PASS")
        self.assertEqual(public_fields(world.weather), public_fields(world.initial_weather))
        world.tick.assert_not_called()

    def test_restore_must_include_every_original_field_or_cleanup_fails(self):
        world = DelayedWeatherWorld(apply_stale_reads=0, restore_stale_reads=2)
        world.omit_original_field_on_restore = True
        session = CollectionSession(CollectionConfig(), carla_for(world))
        clock = FakeClock()
        with polling_clock(clock), self.assertRaisesRegex(SessionError, "session-cleanup-failed"):
            with session:
                world.user_tick()
                session.verify_applied()
        self.assertEqual(world.weather.cloudiness, world.initial_weather.cloudiness)
        self.assertFalse(hasattr(world.weather, "future_weather_field"))
        self.assertEqual(session.cleanup_report["status"], "FAIL")
        self.assertFalse(session.cleanup_report["weather_restored"])
        self.assertTrue(session.cleanup_report["settings_restored"])
        self.assertTrue(session.cleanup_report["owned_actors_absent"])
        self.assertIn("weather-readback", [error["operation"] for error in session.cleanup_report["errors"]])
        self.assertEqual(public_fields(world.settings), public_fields(world.initial_settings))
        self.assertEqual(list(world.actors), [1])
        self.assertEqual([value.cloudiness for value in world.weather_calls], [10.0, 77.0])
        self.assertAlmostEqual(clock.now, 2.0)
        world.tick.assert_not_called()

    def test_permanent_restore_mismatch_preserves_original_body_error_and_fails_report(self):
        world = DelayedWeatherWorld(apply_stale_reads=0, restore_stale_reads=0)
        world.permanent_restore_mismatch = True
        session = CollectionSession(CollectionConfig(), carla_for(world))
        original = ValueError("body-error")
        with polling_clock(FakeClock()), self.assertRaises(ValueError) as caught:
            with session:
                raise original
        self.assertIs(caught.exception, original)
        self.assertEqual(session.cleanup_report["status"], "FAIL")
        self.assertFalse(session.cleanup_report["weather_restored"])
        self.assertTrue(session.cleanup_report["settings_restored"])
        self.assertTrue(session.cleanup_report["owned_actors_absent"])
        self.assertEqual(session.cleanup_report["errors"], [{
            "operation": "weather-readback", "code": "operation-returned-false", "exception_type": "SessionError"
        }])
        self.assertNotIn("body-error", json.dumps(session.cleanup_report))
        world.tick.assert_not_called()

    def test_later_applied_readback_stays_strict_and_does_not_poll_new_mismatch(self):
        world = DelayedWeatherWorld()
        session = CollectionSession(CollectionConfig(), carla_for(world))
        clock = FakeClock()
        with polling_clock(clock):
            with session:
                world.user_tick()
                session.verify_applied()
                self.assertTrue(session.manifest["applied_verified"])
                sleeps_before_readback = list(clock.sleeps)
                world.weather.dust_storm = 0.5
                with self.assertRaises(SessionError) as caught:
                    session.verify_applied()
                self.assertEqual(caught.exception.code, "weather-readback-mismatch")
                self.assertEqual(caught.exception.stage, "applied-readback")
                self.assertFalse(session.manifest["applied_verified"])
                self.assertEqual(clock.sleeps, sleeps_before_readback)
        self.assertEqual(session.cleanup_report["status"], "PASS")
        self.assertEqual(public_fields(world.weather), public_fields(world.initial_weather))
        world.tick.assert_not_called()


if __name__ == "__main__":
    unittest.main()
