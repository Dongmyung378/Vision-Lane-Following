"""Independent camera-axis, PNG-filter, color and overlay publication checks."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import struct
import tempfile
import unittest
import zlib
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from carla_tools import projection
from scripts import render_waypoint_overlay as overlay_cli
from scripts.carla_rgb_probe import rgb_from_bgra


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]


def calibration(width=640, height=360):
    fx = width / 2
    return {"width": width, "height": height, "intrinsics": {"fx": fx, "fy": fx, "cx": width / 2, "cy": height / 2},
            "K": [[fx, 0., width / 2], [0., fx, height / 2], [0., 0., 1.]],
            "distortion": {"lens_k": 0., "lens_kcube": 0.},
            "coordinate_convention": "carla-x-forward-y-right-z-up", "rotation_units": "degrees"}


def chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def png_from_filtered(raw, width=2, height=2, color_type=2):
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class CameraProjectionTests(unittest.TestCase):
    def test_known_optical_axes_project_right_and_up_with_correct_signs(self):
        points = [[10., 0., 0.], [10., 1., 0.], [10., -1., 0.], [10., 0., 1.], [10., 0., -1.]]
        actual = projection.project_waypoints(points, IDENTITY, calibration())
        self.assertEqual([point["uv_pixels"] for point in actual],
                         [[320., 180.], [352., 180.], [288., 180.], [320., 148.], [320., 212.]])
        self.assertEqual(actual[1]["camera_optical_xyz_m"], [1., -0., 10.])
        self.assertTrue(all(point["status"] == "VISIBLE" for point in actual))

    def test_opposite_yaws_use_inverse_camera_rotation_and_translation(self):
        positive_yaw = [[0., -1., 0., 100.], [1., 0., 0., 200.], [0., 0., 1., 3.], [0., 0., 0., 1.]]
        negative_yaw = [[0., 1., 0., 100.], [-1., 0., 0., 200.], [0., 0., 1., 3.], [0., 0., 0., 1.]]
        plus = projection.project_waypoints([[100., 210., 3.], [101., 210., 3.]] + [[100., 210., 3.]] * 3,
                                           positive_yaw, calibration())
        minus = projection.project_waypoints([[100., 190., 3.], [101., 190., 3.]] + [[100., 190., 3.]] * 3,
                                            negative_yaw, calibration())
        self.assertEqual(plus[0]["uv_pixels"], [320., 180.])
        self.assertEqual(minus[0]["uv_pixels"], [320., 180.])
        self.assertEqual(plus[1]["uv_pixels"], [288., 180.])
        self.assertEqual(minus[1]["uv_pixels"], [352., 180.])

    def test_opposite_pitch_signs_shift_a_world_level_target_above_and_below(self):
        projections = []
        for angle in (-12., 12.):
            cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            camera = [[cosine, 0., -sine, 0.], [0., 1., 0., 0.], [sine, 0., cosine, 0.], [0., 0., 0., 1.]]
            projections.append(projection.project_waypoints([[10., 0., 0.]] * 5, camera, calibration())[0])
        self.assertLess(projections[0]["uv_pixels"][1], 180.)
        self.assertGreater(projections[1]["uv_pixels"][1], 180.)
        self.assertAlmostEqual(180. - projections[0]["uv_pixels"][1], projections[1]["uv_pixels"][1] - 180.)

    def test_native_nominal_mount_reports_three_metre_ground_target_outside(self):
        cosine, sine = math.cos(math.radians(-12.)), math.sin(math.radians(-12.))
        camera = [[cosine, 0., -sine, 1.5], [0., 1., 0., 0.], [sine, 0., cosine, 2.4], [0., 0., 0., 1.]]
        points = projection.project_waypoints([[float(distance), 0., 0.] for distance in (3, 6, 9, 12, 15)], camera, calibration())
        self.assertEqual(points[0]["status"], "OUTSIDE_IMAGE")
        self.assertGreater(points[0]["uv_pixels"][1], 360.)
        self.assertIsNone(points[0]["pixel"])
        self.assertEqual(sum(point["status"] == "VISIBLE" for point in points), 4)
        self.assertTrue(all(point["depth_m"] > 0 for point in points))

    def test_behind_near_and_outside_points_are_never_clamped_to_border(self):
        actual = projection.project_waypoints([[-1., 0., 0.], [0., 0., 0.], [5e-7, 0., 0.],
                                               [1., 2., 0.], [10., -10., 0.]], IDENTITY, calibration())
        self.assertEqual([point["status"] for point in actual],
                         ["BEHIND_CAMERA", "BEHIND_CAMERA", "NEAR_PLANE", "OUTSIDE_IMAGE", "VISIBLE"])
        self.assertEqual(actual[3]["uv_pixels"], [960., 180.])
        self.assertIsNone(actual[3]["pixel"])
        self.assertEqual(actual[4]["pixel"], [0, 180])
        self.assertTrue(all(point["coordinate_clamped"] is False for point in actual))

    def test_right_bottom_and_fractional_edge_visibility_use_half_open_bounds(self):
        actual = projection.project_waypoints([[10., 10., 0.], [10., 0., -5.625], [10., 9.9999, 0.],
                                               [10., -10.0001, 0.], [10., 0., 5.625]], IDENTITY, calibration())
        self.assertEqual(actual[0]["uv_pixels"][0], 640.)
        self.assertEqual(actual[1]["uv_pixels"][1], 360.)
        self.assertEqual([point["status"] for point in actual],
                         ["OUTSIDE_IMAGE", "OUTSIDE_IMAGE", "VISIBLE", "OUTSIDE_IMAGE", "VISIBLE"])
        self.assertEqual(actual[2]["pixel"], [639, 180])
        self.assertEqual(actual[4]["pixel"], [320, 0])

    def test_nonrigid_nonfinite_and_nonpinhole_inputs_fail(self):
        points = [[10., 0., 0.]] * 5
        bad_matrices = []
        for cell, value in (((2, 3), float("nan")), ((1, 1), -1.), ((0, 1), .2), ((3, 3), 0.)):
            matrix = copy.deepcopy(IDENTITY)
            matrix[cell[0]][cell[1]] = value
            bad_matrices.append(matrix)
        for matrix in bad_matrices:
            with self.subTest(matrix=matrix), self.assertRaises(projection.ProjectionError):
                projection.project_waypoints(points, matrix, calibration())
        for mutate in (lambda c: c["K"][2].__setitem__(2, float("inf")),
                       lambda c: c["intrinsics"].update(fy=300.),
                       lambda c: c["distortion"].update(lens_k=-1.),
                       lambda c: c["distortion"].update(lens_k=False),
                       lambda c: c.update(width=True)):
            value = calibration()
            mutate(value)
            with self.subTest(calibration=value), self.assertRaises(projection.ProjectionError):
                projection.project_waypoints(points, IDENTITY, value)
        for value in (float("nan"), float("inf"), True):
            bad = copy.deepcopy(points)
            bad[3][2] = value
            with self.subTest(value=value), self.assertRaises(projection.ProjectionError):
                projection.project_waypoints(bad, IDENTITY, calibration())

    def test_finite_input_overflow_is_rejected_instead_of_reported_as_visible(self):
        matrix = copy.deepcopy(IDENTITY)
        matrix[0][3] = 1e308
        with self.assertRaisesRegex(projection.ProjectionError, "projection-nonfinite-number"):
            projection.project_waypoints([[-1e308, 0., 0.]] * 5, matrix, calibration())


class PngProjectionTests(unittest.TestCase):
    def test_all_five_png_filters_reconstruct_hand_computed_rgb_rows(self):
        first, second = [1, 2, 3, 11, 22, 33], [4, 8, 12, 14, 28, 42]
        filtered = {
            0: (first, second),
            1: ([1, 2, 3, 10, 20, 30], [4, 8, 12, 10, 20, 30]),
            2: (first, [3, 6, 9, 3, 6, 9]),
            3: ([1, 2, 3, 11, 21, 32], [4, 7, 11, 7, 13, 20]),
            4: ([1, 2, 3, 10, 20, 30], [3, 6, 9, 3, 6, 9]),
        }
        for kind, rows in filtered.items():
            with self.subTest(filter_kind=kind):
                png = png_from_filtered(bytes([kind, *rows[0], kind, *rows[1]]))
                self.assertEqual(projection.decode_rgb8_png(png), (2, 2, bytes(first + second)))

    def test_bgra_red_blue_and_alpha_witness_roundtrips_as_rgb_not_rgba(self):
        bgra = bytes([1, 17, 231, 255, 241, 18, 3, 0, 31, 32, 33, 88, 4, 5, 6, 255])
        rgb, _ = rgb_from_bgra(bgra, 2, 2)
        expected = bytes([231, 17, 1, 3, 18, 241, 33, 32, 31, 6, 5, 4])
        self.assertEqual(rgb, expected)
        png = projection.encode_rgb8_png(rgb, 2, 2)
        self.assertEqual(projection.decode_rgb8_png(png)[2], expected)
        rgba = png_from_filtered(b"\x00" + bytes([231, 17, 1, 255] * 2) + b"\x00" + bytes([3, 18, 241, 0] * 2), color_type=6)
        with self.assertRaisesRegex(projection.ProjectionError, "invalid-png-format"):
            projection.decode_rgb8_png(rgba)

    def test_source_hash_is_checked_before_any_png_decoder_work(self):
        with patch.object(projection, "_png_dimensions") as decoder:
            with self.assertRaisesRegex(projection.ProjectionError, "projection-image-hash-mismatch"):
                projection.decode_rgb8_png(b"not-png", "0" * 64)
        decoder.assert_not_called()

    def test_recomputed_hash_does_not_hide_corrupt_crc_or_invalid_zlib(self):
        png = projection.encode_rgb8_png(bytes([1, 2, 3] * 4), 2, 2)
        bad_crc = bytearray(png)
        bad_crc[41] ^= 1
        invalid_zlib = png[:33] + chunk(b"IDAT", b"bad-zlib") + png[-12:]
        for value, code in ((bytes(bad_crc), "invalid-png-crc"), (invalid_zlib, "invalid-png-data")):
            with self.subTest(code=code), self.assertRaisesRegex(projection.ProjectionError, code):
                projection.decode_rgb8_png(value, hashlib.sha256(value).hexdigest())

    def test_committed_real_capture_decodes_to_recorded_dimensions_and_channel_extrema(self):
        evidence = ROOT / "docs" / "evidence"
        row = json.loads((evidence / "carla_dataset_d4_d5_2026-10-07_train_smoke_01_frames.json").read_text(encoding="utf-8"))["rows"][0]
        data = (evidence / "carla_dataset_d4_d5_2026-10-07_train_smoke_01.png").read_bytes()
        width, height, rgb = projection.decode_rgb8_png(data, row["image"]["sha256"], (640, 360))
        self.assertEqual((width, height), (row["image"]["width"], row["image"]["height"]))
        self.assertEqual(len(rgb), row["rgb"]["rgb_bytes"])
        self.assertEqual([min(rgb[channel::3]) for channel in range(3)], row["rgb"]["channel_min"])
        self.assertEqual([max(rgb[channel::3]) for channel in range(3)], row["rgb"]["channel_max"])
        self.assertEqual(projection.decode_rgb8_png(projection.encode_rgb8_png(rgb, width, height))[2], rgb)


class OverlayPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.raw = self.directory / "raw"
        (self.raw / "images").mkdir(parents=True)
        self.destination = self.directory / "overlay"
        self.labels_path = self.directory / "labels.json"
        self.labels_path.write_bytes(b"label artifact remains unchanged\n")
        self.rgb = bytes([100, 120, 140] * (160 * 90))
        self.png = projection.encode_rgb8_png(self.rgb, 160, 90)
        (self.raw / "images" / "00000007.png").write_bytes(self.png)
        self.frame = {"index": 0, "frame": 7, "world_frame": 7, "snapshot_frame": 7, "sensor_frame": 7,
                      "timestamp": 1., "snapshot_timestamp": 1., "sensor_timestamp": 1.,
                      "image": {"path": "images/00000007.png", "sha256": hashlib.sha256(self.png).hexdigest(), "width": 160, "height": 90},
                      "camera": {"transform": {"matrix": copy.deepcopy(IDENTITY)}}}
        self.metadata = {"dataset_version": "fixture-v1", "episode_id": "fixture_episode", "route_id": "route", "scene_group_id": "scene",
                         "town": "Town01", "camera_calibration": calibration(160, 90)}
        self.label_row = {"index": 0, "frame": 7, "timestamp": 1., "sample_id": "fixture_episode:0000000007", "label_state": "GENERATED",
                          "image": {key: self.frame["image"][key] for key in ("path", "sha256")},
                          "label": {"ego_frame": 7, "waypoints_world_m": [[10., y, 0.] for y in (-5., -2.5, 0., 2.5, 5.)]}}
        self.validated = {"status": "PASS", "metadata": self.metadata, "frames": [self.frame],
                          "sidecar": {"status": "COMPLETE", "source": {"frames_jsonl_sha256": "a" * 64}, "rows": [self.label_row],
                                      "generation_version": "fixture-generator", "contract_version": "vlf-dataset-d5-v1"}}
        self.validator = Mock(return_value=self.validated)

    def render(self, frame_id=None):
        return projection.render_episode_overlay(self.raw, self.labels_path, self.destination, frame_id, labels_validator=self.validator)

    def test_overlay_is_exclusive_preserves_sources_and_uses_distinct_rgb_marker_colors(self):
        report = self.render()
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["visible_points"], 5)
        self.assertTrue(report["all_points_in_front"])
        self.assertFalse(report["color_provenance_certified"])
        output = (self.destination / "overlay.png").read_bytes()
        _, _, rgb = projection.decode_rgb8_png(output, report["overlay"]["sha256"])
        for point, color in zip(report["points"], projection.COLORS_RGB):
            x, y = point["pixel"]
            self.assertEqual(tuple(rgb[(y * 160 + x) * 3:(y * 160 + x) * 3 + 3]), color)
        self.assertEqual((self.raw / self.frame["image"]["path"]).read_bytes(), self.png)
        self.assertEqual(self.labels_path.read_bytes(), b"label artifact remains unchanged\n")
        self.assertEqual(json.loads((self.destination / "report.json").read_text())["status"], "PASS")
        self.assertEqual(report["source_frame"]["sensor_frame"], 7)
        self.assertEqual(report["labels_reference"]["row_index"], 0)
        second = self.render()
        self.assertEqual(second["errors"][0]["code"], "overlay-destination-already-exists")
        self.assertEqual((self.destination / "overlay.png").read_bytes(), output)

    def test_failed_or_incomplete_labels_never_create_output(self):
        for status, sidecar_status in (("FAIL", "COMPLETE"), ("PASS", "RUNNING")):
            self.validated["status"] = status
            self.validated["sidecar"]["status"] = sidecar_status
            with self.subTest(status=status, sidecar_status=sidecar_status):
                report = self.render()
                self.assertEqual(report["errors"][0]["code"], "overlay-label-validation-failed")
                self.assertFalse(self.destination.exists())

    def test_missing_or_foreign_frame_fails_before_projection(self):
        report = self.render(frame_id=8)
        self.assertEqual(report["errors"][0]["code"], "overlay-frame-unavailable")
        self.label_row["frame"] = 8
        with patch.object(projection, "project_waypoints") as projector:
            report = self.render()
        self.assertEqual(report["errors"][0]["code"], "overlay-frame-identity-mismatch")
        projector.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_changed_source_image_after_label_validation_fails_before_publication(self):
        image = self.raw / self.frame["image"]["path"]
        image.write_bytes(self.png[:-1] + b"x")
        report = self.render()
        self.assertEqual(report["errors"][0]["code"], "projection-image-hash-mismatch")
        self.assertFalse(self.destination.exists())

    def test_raw_episode_output_and_existing_empty_directory_are_refused_before_validator(self):
        self.destination = self.raw / "overlays"
        report = self.render()
        self.assertEqual(report["errors"][0]["code"], "overlay-output-inside-raw-episode")
        self.destination = self.directory / "existing"
        self.destination.mkdir()
        report = self.render()
        self.assertEqual(report["errors"][0]["code"], "overlay-destination-already-exists")
        self.validator.assert_not_called()
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_interrupted_image_publication_cannot_leave_a_pass_report(self):
        original = projection._publish

        def interrupt(path, data, **kwargs):
            if path.name == "overlay.png":
                raise KeyboardInterrupt()
            return original(path, data, **kwargs)

        with patch.object(projection, "_publish", side_effect=interrupt):
            report = self.render()
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["errors"][0]["code"], "overlay-interrupted")
        self.assertEqual(json.loads((self.destination / "report.json").read_text())["status"], "FAIL")
        self.assertFalse((self.destination / "overlay.png").exists())

    def test_cli_forwards_frame_and_returns_nonzero_for_failed_render(self):
        stdout = io.StringIO()
        with patch.object(overlay_cli, "render_episode_overlay", return_value={"status": "FAIL", "errors": [{"code": "fixture"}]}) as render, \
                redirect_stdout(stdout):
            code = overlay_cli.main(["--episode", str(self.raw), "--labels", str(self.labels_path),
                                     "--output-dir", str(self.destination), "--frame", "7"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "FAIL")
        render.assert_called_once_with(self.raw, self.labels_path, self.destination, 7)


if __name__ == "__main__":
    unittest.main()
