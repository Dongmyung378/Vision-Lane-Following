"""Persisted D4 integrity, exclusive writes, and interrupted-capture contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

from carla_tools.config import CollectionConfig, collection_config_from_dict
from carla_tools.episode import EpisodeError, EpisodeWriter, validate_episode
from scripts.carla_rgb_probe import png_from_rgb


def transform(x=0.0, y=0.0, z=0.0):
    return {"location": {"x": x, "y": y, "z": z}, "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "matrix": [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]]}


def fixture():
    raw = CollectionConfig().to_dict()
    raw["camera"].update(width=16, height=16)
    raw["camera"]["mount"]["rotation"]["pitch"] = 0.0
    raw["verification"]["frames"] = 2
    cfg = collection_config_from_dict(raw)
    k = cfg.camera.intrinsics()
    metadata = {
        "schema_version": 1, "dataset_version": "pilot-d4-v1", "episode_id": "episode-001",
        "route_id": "town01-route-001", "scene_group_id": "town01-scene-001", "town": "Town01",
        "expected_frames": 2, "collection_config": cfg.to_dict(), "configuration_sha256": cfg.sha256,
        "camera_calibration": {"intrinsics": k, "requested_mount_matrix": transform(1.5, 0.0, 2.4)["matrix"],
                               "coordinate_convention": "carla-x-forward-y-right-z-up", "rotation_units": "degrees",
                               "K": [[k["fx"], 0.0, k["cx"]], [0.0, k["fy"], k["cy"]], [0.0, 0.0, 1.0]],
                               "width": 16, "height": 16, "distortion": {"lens_k": 0.0, "lens_kcube": 0.0}},
        "applied_manifest": {"applied_verified": True},
        "perturbation": {"lateral_m": 0.0, "yaw_deg": 0.0, "provenance": "stationary-no-perturbation"},
        "control_mode": "stationary_D4_smoke", "labels_state": "not-generated-D6",
        "route_contract": {"source": "map-reference-only"},
    }
    cleanup = {"status": "PASS", "sensor_stopped": True, "settings_restored": True,
               "weather_restored": True, "owned_actors_absent": True}
    png = png_from_rgb(bytes([1, 2, 3] * (16 * 16)), 16, 16)
    return cfg, metadata, cleanup, png


def record(frame=100):
    zero = {"x": 0.0, "y": 0.0, "z": 0.0}
    return {
        "schema_version": 1, "frame": frame, "world_frame": frame, "snapshot_frame": frame, "sensor_frame": frame,
        "timestamp": frame * 0.05, "snapshot_timestamp": frame * 0.05, "sensor_timestamp": frame * 0.05,
        "delta_seconds": 0.05, "image": {"width": 16, "height": 16},
        "vehicle": {"transform": transform(10.0, 20.0, 1.0), "velocity": zero, "angular_velocity": zero,
                    "acceleration": zero, "control": {"throttle": 0.0, "steer": 0.0, "brake": 1.0, "hand_brake": True, "reverse": False}},
        "camera": {"transform": transform(11.5, 20.0, 3.4)},
        "label_state": "not-generated-D6", "label": None,
    }


def png_chunk(kind, payload):
    return (struct.pack(">I", len(payload)) + kind + payload +
            struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "episode"
        self.cfg, self.metadata, self.cleanup, self.png = fixture()

    def complete(self):
        writer = EpisodeWriter(self.path, self.metadata)
        writer.append(record(100), self.png)
        writer.append(record(101), self.png)
        writer.finalize(True, self.cleanup)
        return writer

    def rows(self):
        return [json.loads(line) for line in (self.path / "frames.jsonl").read_text().splitlines()]

    def write_rows(self, rows):
        (self.path / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def codes(self):
        return {item["code"] for item in validate_episode(self.path)["errors"]}

    def test_complete_roundtrip_preserves_physics_records_and_unlabelled_scope(self):
        writer = self.complete()
        report = validate_episode(self.path)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["frame_count"], 2)
        self.assertEqual(report["metadata"]["labels_state"], "not-generated-D6")
        self.assertEqual(report["metadata"]["route_contract"], {"source": "map-reference-only"})
        self.assertEqual(self.rows()[0]["vehicle"]["transform"]["location"], {"x": 10.0, "y": 20.0, "z": 1.0})
        self.assertEqual(self.rows()[0]["image"]["path"], "images/00000100.png")
        self.assertNotIn("waypoints", self.rows()[0])
        exported = writer.metadata
        exported["status"] = "FAILED"
        self.assertEqual(writer.metadata["status"], "COMPLETE")

    def test_nonempty_destination_is_never_overwritten(self):
        self.complete()
        original = (self.path / "episode.json").read_bytes()
        with self.assertRaisesRegex(EpisodeError, "destination-not-empty"):
            EpisodeWriter(self.path, self.metadata)
        self.assertEqual((self.path / "episode.json").read_bytes(), original)

    def test_interrupted_capture_remains_running_and_cannot_be_split_as_complete(self):
        writer = EpisodeWriter(self.path, self.metadata)
        writer.append(record(100), self.png)
        self.assertEqual(writer.metadata["status"], "RUNNING")
        self.assertIn("episode-incomplete", self.codes())
        writer.finalize(False, self.cleanup, {"code": "capture-interrupted"})
        self.assertEqual(writer.metadata["status"], "FAILED")
        self.assertEqual(validate_episode(self.path)["status"], "FAIL")

    def test_count_cleanup_and_applied_readback_are_required_for_completion(self):
        writer = EpisodeWriter(self.path, self.metadata)
        writer.append(record(100), self.png)
        result = writer.finalize(True, self.cleanup)
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("episode-incomplete", self.codes())
        for field in ("sensor_stopped", "settings_restored", "weather_restored", "owned_actors_absent"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                other = EpisodeWriter(Path(folder) / "episode", self.metadata)
                other.append(record(100), self.png)
                other.append(record(101), self.png)
                cleanup = dict(self.cleanup, **{field: False})
                self.assertEqual(other.finalize(True, cleanup)["status"], "FAILED")

    def test_completion_can_be_downgraded_but_failed_episode_cannot_be_upgraded(self):
        writer = self.complete()
        self.assertEqual(writer.finalize(False, self.cleanup, {"code": "integrity-check-failed"})["status"], "FAILED")
        self.assertEqual(writer.finalize(True, self.cleanup)["status"], "FAILED")

    def test_failed_image_write_never_appends_a_row_or_marks_pass(self):
        writer = EpisodeWriter(self.path, self.metadata)
        with patch("carla_tools.episode._atomic_bytes", side_effect=OSError("private-path-token")):
            with self.assertRaisesRegex(EpisodeError, "frame-write-failed"):
                writer.append(record(100), self.png)
        self.assertEqual((self.path / "frames.jsonl").read_bytes(), b"")
        self.assertEqual(writer.metadata["status"], "FAILED")
        self.assertNotIn("private-path-token", json.dumps(writer.metadata))
        self.assertNotEqual(validate_episode(self.path)["status"], "PASS")

    def test_png_published_before_record_but_interrupted_append_stays_incomplete(self):
        writer = EpisodeWriter(self.path, self.metadata)
        original_open = Path.open

        def fail_log(path, *args, **kwargs):
            if path.name == "frames.jsonl" and args and args[0] == "a":
                raise OSError("log-append-failed")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", fail_log):
            with self.assertRaisesRegex(EpisodeError, "frame-write-failed"):
                writer.append(record(100), self.png)
        self.assertTrue((self.path / "images" / "00000100.png").exists())
        self.assertIn("unreferenced-image-or-artifact", self.codes())
        self.assertEqual(writer.metadata["status"], "FAILED")

    def test_writer_rejects_dropped_frame_and_never_overwrites_an_image(self):
        writer = EpisodeWriter(self.path, self.metadata)
        writer.append(record(100), self.png)
        original = (self.path / "images" / "00000100.png").read_bytes()
        with self.assertRaisesRegex(EpisodeError, "dropped-or-unordered-frame"):
            writer.append(record(102), self.png)
        self.assertEqual((self.path / "images" / "00000100.png").read_bytes(), original)
        self.assertEqual(writer.metadata["status"], "FAILED")

    def test_persisted_duplicate_and_missing_frames_fail_validation(self):
        self.complete()
        rows = self.rows()
        duplicate = copy.deepcopy(rows[0])
        duplicate["index"] = 1
        self.write_rows([rows[0], duplicate])
        self.assertIn("duplicate-frame", self.codes())
        self.write_rows([rows[0]])
        self.assertIn("frame-count-mismatch", self.codes())

    def test_timestamp_sensor_frame_schema_and_pose_tampering_fail(self):
        self.complete()
        original = self.rows()
        cases = [
            (lambda row: row.update(sensor_frame=500), "frame-id-mismatch"),
            (lambda row: row.update(sensor_timestamp=55.0), "timestamp-mismatch"),
            (lambda row: row.update(delta_seconds=0.1), "timestep-mismatch"),
            (lambda row: row.update(schema_version=True), "frame-schema-invalid"),
            (lambda row: row["vehicle"]["transform"]["location"].update(x=11.0), "frame-pose-matrix-mismatch"),
            (lambda row: row["vehicle"]["velocity"].update(z=float("nan")), "non-json-or-nonfinite-value"),
        ]
        for mutate, code in cases:
            with self.subTest(code=code):
                rows = copy.deepcopy(original)
                mutate(rows[0])
                self.write_rows(rows)
                self.assertIn(code, self.codes())

    def test_relative_path_traversal_and_dimension_mismatch_fail(self):
        self.complete()
        original = self.rows()
        for path in ("../foreign.png", "/absolute.png", "images\\00000100.png", "images/../00000100.png"):
            with self.subTest(path=path):
                rows = copy.deepcopy(original)
                rows[0]["image"]["path"] = path
                self.write_rows(rows)
                self.assertIn("image-path-invalid", self.codes())
        rows = copy.deepcopy(original)
        rows[0]["image"]["width"] = 20
        self.write_rows(rows)
        self.assertIn("image-dimensions-mismatch", self.codes())

    def test_png_damage_and_missing_images_fail_independent_hash_validation(self):
        self.complete()
        image = self.path / "images" / "00000100.png"
        image.write_bytes(image.read_bytes()[:-1] + b"x")
        self.assertIn("image-hash-mismatch", self.codes())
        image.unlink()
        self.assertIn("image-read-failed", self.codes())

    def test_png_actual_header_dimensions_must_match_config(self):
        writer = EpisodeWriter(self.path, self.metadata)
        wrong = png_from_rgb(bytes([1, 2, 3] * (20 * 16)), 20, 16)
        with self.assertRaisesRegex(EpisodeError, "png-dimensions-mismatch"):
            writer.append(record(100), wrong)
        self.assertFalse((self.path / "images" / "00000100.png").exists())

    def test_crc_correct_corrupt_image_data_cannot_pass_with_a_recomputed_sha(self):
        self.complete()
        image = self.path / "images" / "00000100.png"
        original_rows = self.rows()
        idat_length = struct.unpack_from(">I", self.png, 33)[0]
        idat = self.png[41:41 + idat_length]
        raw = zlib.decompress(idat)
        malformed = {
            "invalid-zlib": b"not-a-zlib-stream",
            "truncated-stream": idat[:-1],
            "short-scanlines": zlib.compress(raw[:-1]),
            "long-scanlines": zlib.compress(raw + b"x"),
            "invalid-filter": zlib.compress(b"\x05" + raw[1:]),
            "extra-stream-bytes": idat + b"junk",
            "second-zlib-stream": idat + zlib.compress(raw),
            "inflation-over-budget": zlib.compress(b"\x00" * (len(raw) * 16)),
        }
        for name, payload in malformed.items():
            with self.subTest(name=name):
                damaged = self.png[:33] + png_chunk(b"IDAT", payload) + self.png[-12:]
                image.write_bytes(damaged)
                rows = copy.deepcopy(original_rows)
                rows[0]["image"]["sha256"] = hashlib.sha256(damaged).hexdigest()
                self.write_rows(rows)
                codes = self.codes()
                self.assertIn("invalid-png-data", codes)
                self.assertNotIn("image-hash-mismatch", codes)

    def test_each_chunk_crc_and_required_order_are_checked_independently_of_sha(self):
        self.complete()
        image = self.path / "images" / "00000100.png"
        original_rows = self.rows()
        idat_length = struct.unpack_from(">I", self.png, 33)[0]
        payload = self.png[41:41 + idat_length]
        damaged_crc = bytearray(self.png)
        damaged_crc[41] ^= 1
        damaged_end_crc = bytearray(self.png)
        damaged_end_crc[-1] ^= 1
        split = len(payload) // 2
        malformed = {
            "IDAT-crc": (bytes(damaged_crc), "invalid-png-crc"),
            "IEND-crc": (bytes(damaged_end_crc), "invalid-png-crc"),
            "missing-IDAT": (self.png[:33] + self.png[-12:], "invalid-png-structure"),
            "missing-IEND": (self.png[:-12], "invalid-png-structure"),
            "duplicate-IHDR": (self.png[:33] + self.png[8:33] + self.png[33:], "invalid-png-structure"),
            "trailing-file-data": (self.png + b"trailing", "invalid-png-structure"),
            "truncated-chunk": (self.png[:33] + struct.pack(">I", 10000) + b"IDAT" + self.png[-12:], "invalid-png-structure"),
            "unknown-critical-chunk": (self.png[:33] + png_chunk(b"ABCD", b"") + self.png[33:], "invalid-png-structure"),
            "noncontiguous-IDAT": (self.png[:33] + png_chunk(b"IDAT", payload[:split]) +
                                    png_chunk(b"tEXt", b"key\x00value") + png_chunk(b"IDAT", payload[split:]) +
                                    self.png[-12:], "invalid-png-structure"),
        }
        for name, (damaged, expected) in malformed.items():
            with self.subTest(name=name):
                image.write_bytes(damaged)
                rows = copy.deepcopy(original_rows)
                rows[0]["image"]["sha256"] = hashlib.sha256(damaged).hexdigest()
                self.write_rows(rows)
                self.assertIn(expected, self.codes())

    def test_contiguous_idat_chunks_and_valid_filter_bytes_are_supported(self):
        row_bytes = 1 + 16 * 3
        raw = bytearray(row_bytes * 16)
        for index in range(16):
            raw[index * row_bytes] = index % 5
        compressed = zlib.compress(raw)
        split = len(compressed) // 2
        png = (self.png[:33] + png_chunk(b"IDAT", compressed[:split]) +
               png_chunk(b"IDAT", compressed[split:]) + self.png[-12:])
        writer = EpisodeWriter(self.path, self.metadata)
        writer.append(record(100), png)
        writer.append(record(101), png)
        writer.finalize(True, self.cleanup)
        self.assertEqual(validate_episode(self.path)["status"], "PASS")

    def test_corrupt_png_fails_before_publication_and_leaves_episode_failed(self):
        writer = EpisodeWriter(self.path, self.metadata)
        corrupt = self.png[:33] + png_chunk(b"IDAT", b"not-zlib") + self.png[-12:]
        with self.assertRaisesRegex(EpisodeError, "invalid-png-data"):
            writer.append(record(100), corrupt)
        self.assertEqual(writer.metadata["status"], "FAILED")
        self.assertEqual((self.path / "frames.jsonl").read_bytes(), b"")
        self.assertEqual(list((self.path / "images").iterdir()), [])

    def test_foreign_huge_header_is_rejected_without_inflating_data(self):
        writer = EpisodeWriter(self.path, self.metadata)
        ihdr = struct.pack(">IIBBBBB", 0x7FFFFFFF, 0x7FFFFFFF, 8, 2, 0, 0, 0)
        huge = self.png[:8] + png_chunk(b"IHDR", ihdr) + self.png[33:]
        with patch("carla_tools.episode.zlib.decompressobj") as decompress:
            with self.assertRaisesRegex(EpisodeError, "invalid-png-format"):
                writer.append(record(100), huge)
        decompress.assert_not_called()

    def test_nonfinite_metadata_and_conflicting_calibration_fail_before_creation(self):
        bad = copy.deepcopy(self.metadata)
        bad["perturbation"]["lateral_m"] = float("inf")
        with self.assertRaisesRegex(EpisodeError, "non-json-or-nonfinite-value"):
            EpisodeWriter(self.path, bad)
        self.assertFalse(self.path.exists())
        for mutate in (
            lambda m: m["camera_calibration"]["K"][0].__setitem__(0, 123),
            lambda m: m["camera_calibration"]["requested_mount_matrix"][2].__setitem__(3, 3.4),
            lambda m: m["camera_calibration"].update(width=17),
            lambda m: m["camera_calibration"]["distortion"].update(lens_k=-1),
        ):
            bad = copy.deepcopy(self.metadata)
            mutate(bad)
            with self.assertRaisesRegex(EpisodeError, "camera-calibration-invalid"):
                EpisodeWriter(self.path, bad)
            self.assertFalse(self.path.exists())

    def test_unverified_manifest_and_forged_complete_cleanup_are_rejected(self):
        writer = EpisodeWriter(self.path, self.metadata)
        writer.update_metadata({"applied_manifest": {"applied_verified": False}})
        writer.append(record(100), self.png)
        writer.append(record(101), self.png)
        self.assertEqual(writer.finalize(True, self.cleanup)["status"], "FAILED")
        stored = json.loads((self.path / "episode.json").read_text())
        stored["status"] = "COMPLETE"
        stored["cleanup"]["weather_restored"] = False
        (self.path / "episode.json").write_text(json.dumps(stored))
        self.assertIn("completion-proof-incomplete", self.codes())

    def test_missing_schema_duplicate_json_keys_and_partial_jsonl_are_rejected(self):
        self.complete()
        metadata_path = self.path / "episode.json"
        original = metadata_path.read_text()
        metadata_path.write_text(original.replace('"status": "COMPLETE"', '"status": "COMPLETE", "status": "COMPLETE"'))
        self.assertIn("duplicate-json-key", self.codes())
        metadata_path.write_text(original)
        (self.path / "frames.jsonl").write_text('{"schema_version":1')
        self.assertIn("invalid-json", self.codes())


if __name__ == "__main__":
    unittest.main()
