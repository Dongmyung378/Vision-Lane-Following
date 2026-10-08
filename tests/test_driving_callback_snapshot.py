"""A producer PASS and its persisted sensor event evidence must agree."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_driving_native as native_standin
import carla_tools.driving as driving
from carla_tools.driving import EventLog


class DrivingCallbackSnapshotTests(unittest.TestCase):
    def test_entered_slow_callback_cannot_finish_between_final_snapshot_and_write(self):
        world = native_standin.NativeStandInWorld()
        entered, release = threading.Event(), threading.Event()
        workers, logs, drain_cleanup_states = [], [], []
        original_stop = native_standin.ListeningSensor.stop
        original_write = driving.write_report

        class SlowCollision:
            timestamp = 0.0
            other_actor = SimpleNamespace(id=55, type_id="static.wall")
            normal_impulse = native_standin.vector(1.0, 0.0, 0.0)

            @property
            def frame(self):
                entered.set()
                if not release.wait(5.0):
                    raise RuntimeError("test-callback-wait-expired")
                return world.frame - 2

        class ObservedLog(EventLog):
            def __init__(self):
                super().__init__()
                logs.append(self)

            def seal(self, timeout_seconds):
                drain_cleanup_states.append((list(world.actors), world.settings.synchronous_mode))
                release.set()
                return super().seal(timeout_seconds)

        def start_slow_callback(sensor):
            if sensor.type_id == "sensor.other.collision":
                worker = threading.Thread(target=sensor.callback, args=(SlowCollision(),), daemon=True)
                workers.append(worker)
                worker.start()
                if not entered.wait(2.0):
                    raise AssertionError("Callback did not enter payload conversion")
            return original_stop(sensor)

        def publish_after_completion(path, metadata):
            if metadata["status"] != "RUNNING":
                # The unfixed producer reaches here with empty final snapshots
                # while the actual callback is still converting its payload.
                release.set()
                for worker in workers:
                    worker.join(2.0)
                    self.assertFalse(worker.is_alive())
            return original_write(path, metadata)

        try:
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / "trial"
                with patch("carla_tools.driving.EventLog", ObservedLog), \
                        patch.object(native_standin.ListeningSensor, "stop", start_slow_callback), \
                        patch("carla_tools.driving.write_report", publish_after_completion):
                    report = native_standin.DrivingNativeTests().run_case(world, directory)
                metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
                self.assertEqual(len(logs[0].snapshot()[0]), 1)
                self.assertEqual(report["status"], "FAIL", "An entered callback cannot be omitted from producer PASS")
                self.assertEqual(metadata["status"], "FAILED")
                self.assertEqual(metadata["events"][0]["kind"], "collision")
                self.assertEqual(report["error"]["code"], "driving-late-collision-or-lane-event")
                self.assertEqual(drain_cleanup_states, [([1], False)])
        finally:
            release.set()
            for worker in workers:
                worker.join(2.0)

    def test_unfinished_callback_times_out_and_cannot_publish_complete(self):
        world = native_standin.NativeStandInWorld()
        entered, release = threading.Event(), threading.Event()
        workers, logs, waits = [], [], []
        original_stop = native_standin.ListeningSensor.stop

        class BlockedCollision:
            timestamp = 0.0
            other_actor = SimpleNamespace(id=55, type_id="static.wall")
            normal_impulse = native_standin.vector(1.0, 0.0, 0.0)

            @property
            def frame(self):
                entered.set()
                release.wait(8.0)
                return world.frame - 2

        class ObservedLog(EventLog):
            def __init__(self):
                super().__init__()
                logs.append(self)

            def seal(self, timeout_seconds):
                started = time.monotonic()
                result = super().seal(timeout_seconds)
                waits.append((timeout_seconds, time.monotonic() - started, result))
                return result

        def start_blocked_callback(sensor):
            if sensor.type_id == "sensor.other.collision":
                worker = threading.Thread(target=sensor.callback, args=(BlockedCollision(),), daemon=True)
                workers.append(worker)
                worker.start()
                if not entered.wait(2.0):
                    raise AssertionError("Callback did not enter payload conversion")
            return original_stop(sensor)

        try:
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / "trial"
                with patch("carla_tools.driving.EventLog", ObservedLog), \
                        patch.object(native_standin.ListeningSensor, "stop", start_blocked_callback):
                    report = native_standin.DrivingNativeTests().run_case(world, directory)
                metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "FAIL")
                self.assertEqual(metadata["status"], "FAILED")
                self.assertEqual(metadata["cleanup"]["status"], "PASS")
                self.assertEqual(metadata["events"], [])
                self.assertEqual(metadata["event_callback_errors"], [{"code": "event-callback-drain-timeout"}])
                self.assertEqual(waits[0][0], 2.0)
                self.assertFalse(waits[0][2])
                self.assertLess(waits[0][1], 3.0)
                self.assertTrue(workers[0].is_alive())
                release.set()
                workers[0].join(2.0)
                self.assertFalse(workers[0].is_alive())
                rows, errors = logs[0].snapshot()
                self.assertEqual(rows, [], "Frozen event rows cannot change after publication")
                self.assertIn({"code": "event-callback-drain-timeout"}, errors)
                self.assertIn({"code": "event-callback-completed-after-seal"}, errors)
                self.assertTrue(logs[0].seal(0.0))
                self.assertIn({"code": "event-callback-drain-timeout"}, logs[0].snapshot()[1])
                self.assertEqual(json.loads((directory / "episode.json").read_text(encoding="utf-8")), metadata)
        finally:
            release.set()
            for worker in workers:
                worker.join(2.0)
        self.assertEqual(list(world.actors), [1])

    def test_callback_after_seal_before_publication_is_refused_and_persisted_as_failure(self):
        world = native_standin.NativeStandInWorld()

        class UnreadablePayload:
            @property
            def frame(self):
                raise AssertionError("Closed callback admission must not inspect payloads")

        class RejectedBeforePublication(EventLog):
            def publication(self):
                self.receive("collision", UnreadablePayload())
                return super().publication()

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trial"
            with patch("carla_tools.driving.EventLog", RejectedBeforePublication):
                report = native_standin.DrivingNativeTests().run_case(world, directory)
            metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(metadata["status"], "FAILED")
            self.assertEqual(metadata["events"], [])
            self.assertEqual(metadata["event_callback_errors"], [{"code": "event-callback-after-seal"}])
        self.assertEqual(list(world.actors), [1])

    def test_seal_keeps_completed_rows_and_refuses_later_payload_conversion(self):
        log = EventLog()
        event = SimpleNamespace(frame=7, timestamp=.35, other_actor=SimpleNamespace(id=55, type_id="static.wall"),
                                normal_impulse=native_standin.vector(1.0, 0.0, 0.0))
        log.receive("collision", event)
        original_rows, errors = log.snapshot()
        self.assertEqual(errors, [])
        self.assertTrue(log.seal(0.0))
        log.receive("collision", SimpleNamespace(frame="private-payload-value"))
        rows, errors = log.snapshot()
        self.assertEqual(rows, original_rows)
        self.assertEqual(errors, [{"code": "event-callback-after-seal"}])
        self.assertNotIn("private-payload-value", json.dumps(errors))

    def test_in_flight_invalid_payload_finishes_drain_with_sanitized_error(self):
        log = EventLog()
        entered, release = threading.Event(), threading.Event()

        class InvalidPayload:
            @property
            def frame(self):
                entered.set()
                release.wait(2.0)
                return "private-payload-value"

        worker = threading.Thread(target=log.receive, args=("collision", InvalidPayload()), daemon=True)
        worker.start()
        try:
            self.assertTrue(entered.wait(1.0))
            timer = threading.Timer(.03, release.set)
            timer.start()
            self.assertTrue(log.seal(1.0))
            timer.join(1.0)
            worker.join(1.0)
            self.assertFalse(worker.is_alive())
            rows, errors = log.snapshot()
            self.assertEqual(rows, [])
            self.assertEqual(errors, [{"code": "event-callback-invalid", "exception_type": "ValueError"}])
            self.assertNotIn("private-payload-value", json.dumps(errors))
        finally:
            release.set()
            worker.join(1.0)

    def test_final_event_snapshot_used_for_status_is_the_snapshot_saved_on_disk(self):
        world = native_standin.NativeStandInWorld()

        class LateSnapshot(EventLog):
            """One observable error is returned on the first final read."""

            def __init__(self):
                super().__init__()
                self.post_cleanup_reads = 0

            def snapshot(self):
                if world.tick_count and not world.settings.synchronous_mode:
                    self.post_cleanup_reads += 1
                    if self.post_cleanup_reads == 1:
                        return [], [{"code": "event-callback-invalid", "exception_type": "ValueError"}]
                return super().snapshot()

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trial"
            with patch("carla_tools.driving.EventLog", LateSnapshot):
                report = native_standin.DrivingNativeTests().run_case(world, directory)
            metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(metadata["status"], "FAILED")
            self.assertEqual(metadata["event_callback_errors"],
                             [{"code": "event-callback-invalid", "exception_type": "ValueError"}],
                             "The evidence that rejects PASS must be the same snapshot saved on disk")
        self.assertEqual(list(world.actors), [1])


if __name__ == "__main__":
    unittest.main()
