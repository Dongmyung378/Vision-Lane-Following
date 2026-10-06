"""Exercise evidence validation and failure handling without a simulator."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import queue
import struct
import sys
import tempfile
import unittest
import zlib
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("carla_rgb_probe", ROOT / "scripts" / "carla_rgb_probe.py")
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class ImageEvidenceTests(unittest.TestCase):
    def test_bgra_is_converted_to_rgb_without_alpha(self):
        rgb, stats = probe.rgb_from_bgra(bytes([3, 2, 1, 255, 6, 5, 4, 0]), 2, 1)
        self.assertEqual(rgb, bytes([1, 2, 3, 4, 5, 6]))
        self.assertEqual(stats["channel_min"], [1, 2, 3])
        self.assertEqual(stats["channel_max"], [4, 5, 6])

    def test_constant_color_and_alpha_noise_are_not_rgb_evidence(self):
        for raw in (bytes([0, 0, 0, 255] * 2), bytes([255, 0, 0, 255, 255, 0, 0, 0])):
            with self.subTest(raw=raw), self.assertRaisesRegex(probe.ProbeFailure, "spatially-constant-rgb"):
                probe.rgb_from_bgra(raw, 2, 1)

    def test_empty_or_truncated_buffer_is_rejected(self):
        for raw in (b"", bytes(7), bytes(9)):
            with self.subTest(length=len(raw)), self.assertRaisesRegex(probe.ProbeFailure, "invalid-bgra-length"):
                probe.rgb_from_bgra(raw, 2, 1)

    def test_invalid_dimensions_are_rejected(self):
        with self.assertRaisesRegex(probe.ProbeFailure, "invalid-image-dimensions"):
            probe.rgb_from_bgra(b"", 0, 1)

    def test_png_has_valid_crc_dimensions_and_exact_pixels(self):
        rgb = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        encoded = probe.png_from_rgb(rgb, 2, 2)
        self.assertEqual(encoded[:8], b"\x89PNG\r\n\x1a\n")
        offset, chunks = 8, {}
        while offset < len(encoded):
            size = struct.unpack(">I", encoded[offset:offset + 4])[0]
            kind = encoded[offset + 4:offset + 8]
            payload = encoded[offset + 8:offset + 8 + size]
            crc = struct.unpack(">I", encoded[offset + 8 + size:offset + 12 + size])[0]
            self.assertEqual(crc, zlib.crc32(kind + payload))
            chunks[kind] = payload
            offset += 12 + size
        self.assertEqual(offset, len(encoded))
        self.assertEqual(struct.unpack(">IIBBBBB", chunks[b"IHDR"]), (2, 2, 8, 2, 0, 0, 0))
        self.assertEqual(zlib.decompress(chunks[b"IDAT"]), b"\x00" + rgb[:6] + b"\x00" + rgb[6:])
        self.assertEqual(chunks[b"IEND"], b"")

    def test_png_rejects_wrong_rgb_length(self):
        with self.assertRaisesRegex(probe.ProbeFailure, "invalid-png-input"):
            probe.png_from_rgb(bytes(5), 2, 1)


class FrameAlignmentTests(unittest.TestCase):
    def test_stale_deliveries_are_discarded_until_exact_frame(self):
        images = queue.Queue()
        images.put(SimpleNamespace(frame=1))
        expected = SimpleNamespace(frame=2)
        images.put(expected)
        self.assertIs(probe.matching_image(images, 2, 1), expected)

    def test_future_frame_does_not_count_as_success(self):
        images = queue.Queue()
        images.put(SimpleNamespace(frame=3))
        with self.assertRaisesRegex(probe.ProbeFailure, "rgb-frame-ahead-of-world"):
            probe.matching_image(images, 2, 1)

    def test_no_delivery_fails_with_bounded_timeout(self):
        with self.assertRaisesRegex(probe.ProbeFailure, "rgb-frame-timeout"):
            probe.matching_image(queue.Queue(), 1, 0.001)


class FakeActor:
    def __init__(self, destroy_ok=True):
        self.destroyed = False
        self.destroy_ok = destroy_ok

    def destroy(self):
        self.destroyed = True
        return self.destroy_ok


class FakeCamera(FakeActor):
    def listen(self, callback):
        self.callback = callback

    def stop(self):
        self.stopped = True


class FakeWorld:
    def __init__(self, raw=bytes([3, 2, 1, 255, 6, 5, 4, 255]), camera_destroy_ok=True):
        self.raw = raw
        self.settings = SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None, no_rendering_mode=False)
        self.initial_settings = copy.copy(self.settings)
        self.vehicle = FakeActor()
        self.camera = FakeCamera(camera_destroy_ok)
        self.frame = 0
        self.applied = []

    def get_map(self):
        return SimpleNamespace(name="/Game/Carla/Maps/Town01", get_spawn_points=lambda: [object()])

    def get_settings(self):
        return copy.copy(self.settings)

    def apply_settings(self, settings):
        self.settings = copy.copy(settings)
        self.applied.append(copy.copy(settings))

    def get_blueprint_library(self):
        return SimpleNamespace(find=lambda _: SimpleNamespace(set_attribute=lambda *args: None))

    def try_spawn_actor(self, *args):
        return self.vehicle

    def spawn_actor(self, *args, **kwargs):
        return self.camera

    def tick(self, timeout):
        self.frame += 1
        raw = self.raw(self.frame) if callable(self.raw) else self.raw
        self.camera.callback(SimpleNamespace(frame=self.frame, width=2, height=1,
                                             raw_data=raw, timestamp=self.frame * 0.05))
        return self.frame


def fake_carla(world, server_version="0.9.16"):
    client = SimpleNamespace(
        set_timeout=lambda _: None, get_client_version=lambda: "0.9.16",
        get_server_version=lambda: server_version, load_world=Mock(return_value=world),
        get_world=Mock(return_value=world),
    )
    return SimpleNamespace(Client=Mock(return_value=client), Transform=lambda *args: None,
                           Location=lambda **kwargs: None, Rotation=lambda **kwargs: None)


class FailureSemanticsTests(unittest.TestCase):
    def config(self, folder):
        return probe.ProbeConfig(Path(folder), frames=2, warmup=1, width=2, height=1, timeout=1)

    def test_missing_carla_is_fail_and_exception_text_is_not_exported(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(probe.importlib, "import_module", side_effect=ImportError("private/path/access-token")):
            report = probe.run_probe(self.config(folder))
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"]["stage"], "carla-import")
        self.assertNotIn("access-token", json.dumps(report))
        self.assertFalse(report["preview_png_written"])

    def test_version_mismatch_stops_before_world_load_or_rgb(self):
        with tempfile.TemporaryDirectory() as folder:
            world = FakeWorld()
            report = probe.run_probe(self.config(folder), fake_carla(world, "0.9.15"))
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"]["code"], "carla-version-mismatch")
        self.assertEqual(report["frames"], [])
        self.assertEqual(world.applied, [])

    def test_malformed_server_version_is_not_exported(self):
        with tempfile.TemporaryDirectory() as folder:
            report = probe.run_probe(self.config(folder), fake_carla(FakeWorld(), "private/path/access-token"))
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["server_version"], "invalid-response")
        self.assertNotIn("access-token", json.dumps(report))

    def test_invalid_rgb_restores_settings_and_destroys_owned_actors(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(probe, "gpu_sample", return_value={"status": "unavailable"}):
            world = FakeWorld(raw=bytes([0, 0, 0, 255] * 2))
            report = probe.run_probe(self.config(folder), fake_carla(world))
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"]["code"], "spatially-constant-rgb")
        self.assertTrue(world.vehicle.destroyed)
        self.assertTrue(world.camera.destroyed)
        self.assertTrue(world.camera.stopped)
        self.assertEqual(vars(world.settings), vars(world.initial_settings))
        self.assertFalse(report["preview_png_written"])

    def test_rendered_rgb_cannot_pass_if_cleanup_is_incomplete(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(probe, "gpu_sample", return_value={"status": "unavailable"}):
            world = FakeWorld(camera_destroy_ok=False)
            report = probe.run_probe(self.config(folder), fake_carla(world))
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"]["stage"], "cleanup")
        self.assertEqual(report["metrics"]["captured_frames"], 2)
        self.assertTrue(report["preview_png_written"])

    def test_optional_gpu_probe_is_not_a_rendering_requirement(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(probe, "gpu_sample", return_value={"status": "unavailable"}):
            report = probe.run_probe(self.config(folder), fake_carla(FakeWorld()))
            self.assertTrue((Path(folder) / probe.PREVIEW_NAME).read_bytes().startswith(b"\x89PNG"))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["gpu_after_capture"]["status"], "unavailable")
        self.assertEqual(len(report["frames"]), 2)
        self.assertTrue(all(frame["world_frame"] == frame["sensor_frame"] for frame in report["frames"]))

    def test_reuse_current_town01_never_loads_world_and_restores_its_settings(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(probe, "gpu_sample", return_value={"status": "unavailable"}):
            world = FakeWorld()
            carla = fake_carla(world)
            config = probe.ProbeConfig(Path(folder), frames=2, warmup=1, width=2, height=1,
                                       timeout=1, reuse_current_world=True)
            report = probe.run_probe(config, carla)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["world_initialization"], "reuse-current-town01")
        carla.Client.return_value.get_world.assert_called_once_with()
        carla.Client.return_value.load_world.assert_not_called()
        self.assertEqual(vars(world.settings), vars(world.initial_settings))
        self.assertTrue(world.camera.destroyed)
        self.assertTrue(world.vehicle.destroyed)

    def test_reuse_wrong_map_is_refused_without_loading_or_changing_world(self):
        with tempfile.TemporaryDirectory() as folder:
            world = FakeWorld()
            world.get_map = Mock(return_value=SimpleNamespace(name="/Game/Carla/Maps/Town02"))
            carla = fake_carla(world)
            report = probe.run_probe(probe.ProbeConfig(Path(folder), reuse_current_world=True), carla)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["error"], {"stage": "world-reuse", "code": "unexpected-map", "exception_type": "ProbeFailure"})
        carla.Client.return_value.load_world.assert_not_called()
        self.assertEqual(world.applied, [])
        self.assertFalse(report["preview_png_written"])
        self.assertFalse(world.vehicle.destroyed)
        self.assertNotIn("world_initialization", report)

    def test_default_mode_still_loads_town01_for_existing_kaggle_jobs(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(probe, "gpu_sample", return_value={"status": "unavailable"}):
            carla = fake_carla(FakeWorld())
            report = probe.run_probe(self.config(folder), carla)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["world_initialization"], "load-town01")
        carla.Client.return_value.load_world.assert_called_once_with("Town01")
        carla.Client.return_value.get_world.assert_not_called()

    def test_black_startup_frame_is_warmed_up_without_counting_as_evidence(self):
        raw = lambda frame: bytes([0, 0, 0, 255] * 2) if frame == 1 else bytes([3, 2, 1, 255, 6, 5, 4, 255])
        with tempfile.TemporaryDirectory() as folder, patch.object(probe, "gpu_sample", return_value={"status": "unavailable"}):
            report = probe.run_probe(self.config(folder), fake_carla(FakeWorld(raw=raw)))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual([frame["world_frame"] for frame in report["frames"]], [2, 3])

    def test_missing_nvidia_smi_is_optional_and_keeps_no_raw_error(self):
        with patch.object(probe.subprocess, "run", side_effect=FileNotFoundError("private/path/access-token")):
            result = probe.gpu_sample()
        self.assertEqual(result, {"status": "unavailable"})

    def test_main_exports_failure_report_and_nonzero_status(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(probe.importlib, "import_module", side_effect=ImportError("secret")), redirect_stdout(io.StringIO()) as output:
            code = probe.main(["--output-dir", folder])
            saved = json.loads((Path(folder) / probe.REPORT_NAME).read_text(encoding="utf-8"))
        self.assertEqual(code, 1)
        self.assertEqual(saved["status"], "FAIL")
        self.assertNotIn("secret", output.getvalue())

    def test_failed_rerun_removes_stale_preview_evidence(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(probe.importlib, "import_module", side_effect=ImportError()):
            preview = Path(folder) / probe.PREVIEW_NAME
            preview.write_bytes(b"stale")
            report = probe.run_probe(self.config(folder))
            self.assertFalse(preview.exists())
        self.assertEqual(report["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
