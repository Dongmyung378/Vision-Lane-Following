"""D7 persisted geometry/color gate and fail-closed CLI report boundaries."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from carla_tools import week1
from carla_tools.color import color_provenance
from carla_tools.config import CollectionConfig, collection_config_from_dict
from carla_tools.contracts import route_sha256
from carla_tools.episode import EpisodeWriter
from carla_tools.geometry_survey import CASES
from carla_tools.labels import generate_labels, validate_labels
from carla_tools.projection import decode_rgb8_png, encode_rgb8_png, project_waypoints
from scripts import verify_week1 as cli


IDENTITY = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]


def geometry_route(case, spawn):
    sign = 0 if case == "straight" else -1 if case == "left_curve" else 1
    anchors = []
    for index in range(141):
        distance = index * .25
        angle = distance / 20.
        x = distance if not sign else 20. * math.sin(angle)
        y = 0. if not sign else sign * 20. * (1. - math.cos(angle))
        anchors.append({"road_id": 1, "section_id": 0, "lane_id": -1, "s": distance,
                        "world_location": {"x": x, "y": y, "z": 0.},
                        "world_rotation": {"pitch": 0., "yaw": sign * math.degrees(angle), "roll": 0.},
                        "lane_width_m": 3.5, "arc_distance_m": distance,
                        "is_junction": False, "successor_count": 1})
    route = {"schema_version": 1, "route_id": f"route_{case}", "scene_group_id": f"scene_{case}",
             "town": "Town01", "carla_version": "0.9.16", "map_package": "Carla/Maps/Town01",
             "opendrive_sha256": "a" * 64, "spawn_point_index": spawn, "start": anchors[0],
             "horizon_m": 15., "min_required_route_m": 35., "sample_step_m": .25, "route_length_m": 35.,
             "centerline_anchor_sequence": anchors, "selection_scope": "D5-topology-only-not-frame-labels"}
    route["route_sha256"] = route_sha256(route)
    return route


def make_case(directory, case, spawn):
    raw = CollectionConfig().to_dict()
    raw["camera"].update(width=160, height=90)
    raw["verification"].update(warmup_frames=1, frames=2)
    raw["vehicle"]["spawn_point_index"] = spawn
    cfg = collection_config_from_dict(raw)
    k = cfg.camera.intrinsics()
    cosine, sine = math.cos(math.radians(-12.)), math.sin(math.radians(-12.))
    camera_matrix = [[cosine, 0., -sine, 1.5], [0., 1., 0., 0.],
                     [sine, 0., cosine, 2.4], [0., 0., 0., 1.]]
    route = geometry_route(case, spawn)
    metadata = {"schema_version": 1, "dataset_version": "week1-fixture-v1", "episode_id": f"episode_{case}",
                "route_id": route["route_id"], "scene_group_id": route["scene_group_id"], "town": "Town01",
                "expected_frames": 2, "collection_config": cfg.to_dict(), "configuration_sha256": cfg.sha256,
                "camera_calibration": {"intrinsics": k, "requested_mount_matrix": camera_matrix,
                                       "K": [[k["fx"], 0., k["cx"]], [0., k["fy"], k["cy"]], [0., 0., 1.]],
                                       "width": 160, "height": 90, "distortion": {"lens_k": 0., "lens_kcube": 0.},
                                       "coordinate_convention": "carla-x-forward-y-right-z-up", "rotation_units": "degrees"},
                "applied_manifest": {"applied_verified": True},
                "perturbation": {"lateral_m": 0., "yaw_deg": 0., "provenance": "synthetic-stationary-fixture"},
                "control_mode": "stationary_D4_smoke", "labels_state": "not-generated-D6", "route_contract": route}
    count = 160 * 90
    bgra = bytearray([1, 17, 231, 255] * count)
    bgra[(count // 2) * 4:(count // 2) * 4 + 4] = bytes([241, 18, 3, 0])
    bgra[-4:] = bytes([31, 32, 33, 88])
    rgb = bytes(value for index in range(count) for value in (bgra[4 * index + 2], bgra[4 * index + 1], bgra[4 * index]))
    proof = color_provenance(bytes(bgra), rgb, 160, 90)
    png = encode_rgb8_png(rgb, 160, 90)
    episode = directory / f"episode_{case}"
    labels = directory / f"labels_{case}.json"
    writer = EpisodeWriter(episode, metadata)
    for offset in range(2):
        frame = 100 + spawn * 100 + offset
        zero = {"x": 0., "y": 0., "z": 0.}
        writer.append({"schema_version": 1, "frame": frame, "world_frame": frame, "snapshot_frame": frame, "sensor_frame": frame,
                       "timestamp": frame * .05, "snapshot_timestamp": frame * .05, "sensor_timestamp": frame * .05,
                       "delta_seconds": .05, "image": {"width": 160, "height": 90},
                       "vehicle": {"transform": {"location": zero, "rotation": {"pitch": 0., "yaw": 0., "roll": 0.},
                                                 "matrix": IDENTITY}, "velocity": zero, "angular_velocity": zero,
                                   "acceleration": zero, "control": cfg.vehicle.control.to_dict()},
                       "camera": {"transform": {"location": {"x": 1.5, "y": 0., "z": 2.4},
                                                "rotation": {"pitch": -12., "yaw": 0., "roll": 0.}, "matrix": camera_matrix}},
                       "label_state": "NOT_GENERATED", "label": None, "color_provenance": proof}, png)
    writer.finalize(True, {"status": "PASS", "sensor_stopped": True, "settings_restored": True,
                           "weather_restored": True, "owned_actors_absent": True})
    generated = generate_labels(episode, labels)
    if generated["status"] != "PASS":
        raise AssertionError(generated)
    return {"case": case, "episode": episode, "labels": labels}


class Week1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.source_temporary.cleanup)
        cls.source_directory = Path(cls.source_temporary.name)
        cls.cases = [make_case(cls.source_directory, case, index) for index, case in enumerate(CASES)]
        cls.validated = [validate_labels(item["episode"], item["labels"]) for item in cls.cases]
        cls.audits = [week1.audit_labeled_episode(item["episode"], item["labels"]) for item in cls.cases]
        for checked in cls.validated + cls.audits:
            if checked["status"] != "PASS":
                raise AssertionError(checked)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def audit_mock_validation(self, validated):
        item = self.cases[0]
        with patch.object(week1, "validate_labels", return_value=validated):
            return week1.audit_labeled_episode(item["episode"], item["labels"])

    def copied_manifest(self):
        cases = []
        for item in self.cases:
            episode = self.directory / item["episode"].name
            labels = self.directory / item["labels"].name
            shutil.copytree(item["episode"], episode)
            shutil.copyfile(item["labels"], labels)
            cases.append({"case": item["case"], "episode": episode.name, "labels": labels.name})
        path = self.directory / "cases.json"
        path.write_text(json.dumps({"schema_version": 1, "cases": cases}), encoding="utf-8")
        return path

    def test_real_persisted_three_direction_cases_pass_without_gpu_or_carla(self):
        original = {(item["case"], name): (item["episode"] / name).read_bytes()
                    for item in self.cases for name in ("episode.json", "frames.jsonl")}
        report = week1.verify_week1_cases(self.cases)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["episode_count"], 3)
        self.assertEqual(report["frame_count"], 6)
        self.assertEqual(report["geometry_cases"], list(CASES))
        self.assertFalse(report["final_test_used"])
        self.assertFalse(report["controller_or_training_verified"])
        for item in report["cases"]:
            audit = item["audit"]
            self.assertEqual(audit["frame_count"], 2)
            self.assertEqual(audit["color_verified_frames"], 2)
            self.assertEqual(audit["min_visible_targets"], 4)
            self.assertLess(audit["max_camera_world_roundtrip_error_m"], 1e-10)
            self.assertEqual(audit["visibility_counts"]["3"]["OUTSIDE_IMAGE"], 2)
        for item in self.cases:
            for name in ("episode.json", "frames.jsonl"):
                self.assertEqual((item["episode"] / name).read_bytes(), original[(item["case"], name)])

    def test_invalid_duplicate_and_missing_geometry_enums_fail_before_audit(self):
        invalid = [[], self.cases[:2], [self.cases[0]] * 3,
                   [dict(self.cases[0], case="Straight"), *self.cases[1:]],
                   [dict(self.cases[0], case="unknown"), *self.cases[1:]]]
        for cases in invalid:
            with self.subTest(cases=cases), patch.object(week1, "audit_labeled_episode") as audit:
                report = week1.verify_week1_cases(cases)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["errors"][0]["code"], "week1-three-distinct-geometry-cases-required")
            audit.assert_not_called()

    def test_actual_geometry_classification_cannot_be_relabelled_left_as_right(self):
        cases = copy.deepcopy(self.cases)
        cases[1]["case"], cases[2]["case"] = cases[2]["case"], cases[1]["case"]
        report = week1.verify_week1_cases(cases)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["errors"][0]["code"], "week1-geometry-case-mismatch")

    def test_duplicate_resolved_path_and_episode_id_are_rejected(self):
        cases = copy.deepcopy(self.cases)
        cases[1]["episode"] = cases[0]["episode"] / "."
        report = week1.verify_week1_cases(cases)
        self.assertEqual(report["errors"][0]["code"], "duplicate-week1-episode-path")
        audits = copy.deepcopy(self.audits)
        audits[1]["source"]["episode_id"] = audits[0]["source"]["episode_id"]
        with patch.object(week1, "audit_labeled_episode", side_effect=audits):
            report = week1.verify_week1_cases(self.cases)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["errors"][0]["code"], "duplicate-week1-episode-id")

    def test_mixed_dataset_map_calibration_and_configuration_cannot_pass(self):
        mutations = [lambda a: a["source"].update(dataset_version="different-v1"),
                     lambda a: a["source"].update(town="Town02"),
                     lambda a: a.update(opendrive_sha256="b" * 64),
                     lambda a: a.update(calibration_sha256="b" * 64),
                     lambda a: a.update(configuration_except_spawn_sha256="b" * 64)]
        for mutate in mutations:
            audits = copy.deepcopy(self.audits)
            mutate(audits[1])
            with self.subTest(mutate=mutate), patch.object(week1, "audit_labeled_episode", side_effect=audits):
                report = week1.verify_week1_cases(self.cases)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["errors"][0]["code"], "week1-cross-case-configuration-mismatch")

    def test_integrity_failure_stops_before_projection_and_color(self):
        failed = {"status": "FAIL", "errors": [{"code": "label-source-identity-or-hash-mismatch"}]}
        with patch.object(week1, "project_waypoints") as projector, patch.object(week1, "verify_rgb_color") as color:
            report = self.audit_mock_validation(failed)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["errors"][0]["code"], "raw-or-label-integrity-failed")
        projector.assert_not_called()
        color.assert_not_called()
        self.assertNotIn("checks", report)

    def test_legacy_missing_same_frame_color_proof_cannot_claim_color_certification(self):
        validated = copy.deepcopy(self.validated[0])
        validated["frames"][0].pop("color_provenance")
        report = self.audit_mock_validation(validated)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["errors"][0]["code"], "same-frame-color-provenance-missing")
        self.assertNotIn("checks", report)

    def test_red_blue_swap_recomputed_png_and_rgb_hashes_still_fail_source_witness(self):
        item = self.cases[0]
        episode = self.directory / "swapped"
        shutil.copytree(item["episode"], episode)
        validated = copy.deepcopy(self.validated[0])
        frame = validated["frames"][0]
        image = episode / frame["image"]["path"]
        width, height, rgb = decode_rgb8_png(image.read_bytes())
        swapped = bytearray(rgb)
        swapped[0::3], swapped[2::3] = rgb[2::3], rgb[0::3]
        png = encode_rgb8_png(bytes(swapped), width, height)
        image.write_bytes(png)
        frame["image"]["sha256"] = hashlib.sha256(png).hexdigest()
        frame["color_provenance"]["rgb_sha256"] = hashlib.sha256(swapped).hexdigest()
        with patch.object(week1, "validate_labels", return_value=validated):
            report = week1.audit_labeled_episode(episode, item["labels"])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["errors"][0]["code"], "bgra-rgb-channel-order-mismatch")
        self.assertNotIn("checks", report)

    def test_behind_camera_and_fewer_than_three_visible_targets_fail(self):
        for angle, expected in ((180., "label-behind-camera-or-clamped"), (60., "insufficient-visible-label-targets")):
            validated = copy.deepcopy(self.validated[0])
            cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            validated["frames"][0]["camera"]["transform"]["matrix"] = [
                [cosine, -sine, 0., 1.5], [sine, cosine, 0., 0.], [0., 0., 1., 2.4], [0., 0., 0., 1.]]
            with self.subTest(angle=angle):
                report = self.audit_mock_validation(validated)
                self.assertEqual(report["status"], "FAIL")
                self.assertEqual(report["errors"][0]["code"], expected)

    def test_nonfirst_projection_nan_or_wrong_uv_cannot_escape_roundtrip_check(self):
        validated = copy.deepcopy(self.validated[0])
        frame = validated["frames"][0]
        base = project_waypoints(validated["sidecar"]["rows"][0]["label"]["waypoints_world_m"],
                                 frame["camera"]["transform"]["matrix"], validated["metadata"]["camera_calibration"])
        for value in (float("nan"), base[3]["uv_pixels"][1] + 20.):
            points = copy.deepcopy(base)
            points[3]["uv_pixels"][1] = value
            with self.subTest(value=value), patch.object(week1, "project_waypoints", return_value=points), \
                    patch.object(week1, "verify_rgb_color") as color:
                report = self.audit_mock_validation(validated)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["errors"][0]["code"], "camera-projection-roundtrip-failed")
            color.assert_not_called()

    def test_actual_source_or_label_damage_fails_fresh_integrity(self):
        item = self.cases[0]
        episode = self.directory / "damaged"
        shutil.copytree(item["episode"], episode)
        first = self.validated[0]["frames"][0]
        image = episode / first["image"]["path"]
        image.write_bytes(image.read_bytes()[:-1] + b"x")
        report = week1.audit_labeled_episode(episode, item["labels"])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["errors"][0]["code"], "raw-or-label-integrity-failed")
        damaged_labels = self.directory / "damaged-labels.json"
        sidecar = copy.deepcopy(self.validated[0]["sidecar"])
        sidecar["rows"][0]["label"]["waypoints_ego_m"][0][0] += .1
        damaged_labels.write_text(json.dumps(sidecar), encoding="utf-8")
        report = week1.audit_labeled_episode(item["episode"], damaged_labels)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["errors"][0]["code"], "raw-or-label-integrity-failed")

    def test_cli_actual_relative_manifest_passes_and_failure_replaces_old_pass_report(self):
        manifest = self.copied_manifest()
        output = self.directory / "report.json"
        with redirect_stdout(io.StringIO()):
            code = cli.main(["--manifest", str(manifest), "--output", str(output)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.read_text())["status"], "PASS")
        manifest.write_text('{"schema_version":1,"cases":[],"cases":[]}', encoding="utf-8")
        with patch.object(cli, "verify_week1_cases") as verifier, redirect_stdout(io.StringIO()):
            code = cli.main(["--manifest", str(manifest), "--output", str(output)])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output.read_text())["status"], "FAIL")
        verifier.assert_not_called()

    def test_cli_interrupted_rerun_cannot_leave_old_pass_evidence(self):
        manifest = self.copied_manifest()
        output = self.directory / "report.json"
        output.write_text('{"status":"PASS","old":true}', encoding="utf-8")
        with patch.object(cli, "verify_week1_cases", side_effect=KeyboardInterrupt()), redirect_stdout(io.StringIO()):
            code = cli.main(["--manifest", str(manifest), "--output", str(output)])
        self.assertEqual(code, 1)
        self.assertIn(json.loads(output.read_text())["status"], ("FAIL", "RUNNING"))

    def test_cli_rejects_unknown_manifest_fields_before_audit(self):
        manifest = self.copied_manifest()
        original = json.loads(manifest.read_text())
        variants = [dict(original, unexpected=True), dict(original, schema_version=True)]
        extra_item = copy.deepcopy(original)
        extra_item["cases"][0]["unrecognized"] = "not silently discarded"
        variants.append(extra_item)
        for data in variants:
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.subTest(data=data), patch.object(cli, "verify_week1_cases") as verifier, redirect_stdout(io.StringIO()):
                code = cli.main(["--manifest", str(manifest), "--output", str(self.directory / "report.json")])
            self.assertEqual(code, 1)
            verifier.assert_not_called()


if __name__ == "__main__":
    unittest.main()
