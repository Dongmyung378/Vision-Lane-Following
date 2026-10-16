"""Dataset reconstruction around a separately covered native replay boundary.

Small synthetic original PNGs and native-shaped traces exercise dataset
contracts here; they are never represented as simulator or training results.
"""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from carla_tools.contracts import canonical_sha256
from carla_tools.dataset_v1 import (DatasetV1Error, _quality_samples, _scene_separation_lower_bound,
                                    build_dataset_v1_manifest, dataset_v1_config_from_dict,
                                    load_dataset_v1_config, pair_group_id, validate_dataset_v1_manifest)
from carla_tools.driving import route_initial_location
from carla_tools.driving_route import FixedRouteTracker, perturbation_transform
from carla_tools.pilot import load_pilot_config, select_pilot_routes
from carla_tools.projection import encode_rgb8_png
from test_data_contracts import transform_fixture
from test_pilot import candidate_fixture


def small_config():
    value = load_dataset_v1_config().to_dict()
    value.update(minimum_samples=100, maximum_samples=200, minimum_train_samples=60, minimum_dev_samples=40,
                 minimum_central_samples=20, minimum_recovery_samples=80, minimum_pair_count=4,
                 train_selection_samples=4, weather_presets=["ClearNoon"],
                 collection_runs=[{"name": "start00", "initial_reference_m": 0., "seed": 42}])
    return dataset_v1_config_from_dict(value)


def native_boundary_fixture(path, **kwargs):
    """Check bytes at this unit boundary without claiming native controller replay."""
    path = Path(path)
    metadata = json.loads((path / "episode.json").read_text())
    for line in (path / "steps.jsonl").read_text().splitlines():
        row = json.loads(line)
        if hashlib.sha256((path / row["image"]["path"]).read_bytes()).hexdigest() != row["image"]["sha256"]:
            return {"status": "FAIL"}
    return {"status": "PASS", "checks": {"synthetic_native_boundary": True},
            "evaluation_metrics": metadata["evaluation_metrics"]}


def dataset_fixture(directory):
    root, config = Path(directory), small_config()
    routes = select_pilot_routes(candidate_fixture(), load_pilot_config())
    descriptors = []
    for route_item in routes:
        route = route_item["route"]
        for condition in config.to_dict()["conditions"]:
            ordinal = len(descriptors)
            identifier = f"dataset_fixture_{ordinal:03d}"
            relative = "episodes/" + identifier
            folder = root / relative
            (folder / "images").mkdir(parents=True)
            tracker, rows = FixedRouteTracker(route), []
            for index in range(5):
                location, yaw = route_initial_location(route, index * .05)
                pose = transform_fixture(x=location["x"], y=location["y"], yaw=yaw)
                pose = perturbation_transform(pose, condition["lateral_m"], condition["yaw_deg"])
                frame, timestamp = index + 10, .05 * (index + 10)
                tracking = tracker.observe(pose, frame, timestamp)
                image = f"images/{frame:08d}.png"
                payload = encode_rgb8_png(bytes((ordinal + 1, index + 1, 127)) * (640 * 360), 640, 360)
                (folder / image).write_bytes(payload)
                rows.append({"frame": frame, "timestamp": timestamp, "vehicle": {"transform": pose},
                             "speed_mps": 0. if index == 0 else 1., "body_lane_margin_m": .5,
                             "tracking": tracking, "image": {"path": image, "sha256": hashlib.sha256(payload).hexdigest(),
                                                              "width": 640, "height": 360}})
            metrics = {"driving_success": True, "collision_count": 0, "lane_invasion_count": 0,
                       "minimum_body_lane_margin_m": .5}
            metadata = {"status": "COMPLETE", "method": "privileged_reference", "episode_id": identifier,
                        "geometry_case": route_item["case"], "route_contract": route,
                        "condition": condition, "camera_calibration": {"fixed": True}, "initialization": {},
                        "collection_config": {"weather": {"preset": "ClearNoon"}, "seed": 42},
                        "evaluation_metrics": metrics}
            (folder / "episode.json").write_text(json.dumps(metadata), encoding="utf-8")
            (folder / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            descriptors.append({"episode": relative, "split": route_item["split"], "case": route_item["case"],
                                "condition": condition["name"], "role": "central" if condition["name"] == "center" else "recovery",
                                "pair_group_id": pair_group_id(route["route_id"], "ClearNoon", "start00"),
                                "weather": "ClearNoon", "run_name": "start00", "initial_reference_m": 0.})
    return config, routes, descriptors


class DatasetV1ConfigTests(unittest.TestCase):
    def test_config_roundtrip_defensive_freeze_and_source_hash(self):
        config = load_dataset_v1_config()
        raw = config.to_dict()
        self.assertEqual(dataset_v1_config_from_dict(raw).sha256, config.sha256)
        raw["conditions"][0]["yaw_deg"] = 99
        self.assertEqual(config.conditions[0]["yaw_deg"], 0.)
        with self.assertRaises(TypeError):
            config.conditions[0]["yaw_deg"] = 99

    def test_invalid_budget_nonfinite_weather_repeated_run_and_extra_key_rejected(self):
        invalids = (("train_selection_samples", True), ("maximum_samples", 39_999),
                    ("scene_buffer_m", 74.), ("pair_arc_tolerance_m", float("nan")),
                    ("weather_presets", ["ClearSunset"]), ("minimum_recovery_lateral_difference_m", 0.),
                    ("collection_runs", [{"name": "one", "initial_reference_m": 0., "seed": 42},
                                         {"name": "two", "initial_reference_m": 0., "seed": 42}]))
        for key, value in invalids:
            with self.subTest(key=key), self.assertRaises(DatasetV1Error):
                raw = load_dataset_v1_config().to_dict()
                raw[key] = value
                dataset_v1_config_from_dict(raw)
        raw = load_dataset_v1_config().to_dict()
        raw["padding"] = True
        with self.assertRaises(DatasetV1Error):
            dataset_v1_config_from_dict(raw)


class DatasetV1SceneTests(unittest.TestCase):
    def test_full_observed_positions_expand_box_and_ambiguous_boxes_use_exact_segments(self):
        first = {"town": "Town01", "segments": [((0., 0.), (10., 0.)), ((100., 0.), (100., 0.))]}
        second = {"town": "Town01", "segments": [((180., 0.), (190., 0.))]}
        with patch("carla_tools.dataset_v1._scene_distance", side_effect=AssertionError("unneeded quadratic fallback")):
            self.assertEqual(_scene_separation_lower_bound(first, second, 75.), 80.)
        second["segments"] = [((170., 0.), (190., 0.))]
        with patch("carla_tools.dataset_v1._scene_distance", return_value=70.) as exact:
            self.assertEqual(_scene_separation_lower_bound(first, second, 75.), 70.)
            exact.assert_called_once()
        crossing = {"town": "Town01", "segments": [((0., 10.), (10., 0.))]}
        self.assertEqual(_scene_separation_lower_bound({"town": "Town01", "segments": [((0., 0.), (10., 10.))]}, crossing, 75.), 0.)


class DatasetV1ManifestTests(unittest.TestCase):
    def test_complete_snapshot_reconstructs_selections_and_retains_every_excluded_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, routes, descriptors = dataset_fixture(temporary)
            path = build_dataset_v1_manifest(temporary, routes, descriptors, config)
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["sample_count"], 140)
            self.assertEqual(manifest["exclusion_counts"], {"stationary_or_startup": 35})
            self.assertEqual(len(manifest["source_files_sha256"]), 35 * 7)
            with patch("carla_tools.dataset_v1._native_episode_gate", side_effect=native_boundary_fixture):
                report = validate_dataset_v1_manifest(path)
            self.assertEqual(report["status"], "PASS", report)
            self.assertEqual(report["episode_count"], 35)
            self.assertEqual(report["training_samples_per_arm"], 4)
            with self.assertRaisesRegex(DatasetV1Error, "already-exists"):
                build_dataset_v1_manifest(temporary, routes, descriptors, config)

    def test_rehashed_selection_rewrite_cannot_override_source_derived_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, routes, descriptors = dataset_fixture(temporary)
            path = build_dataset_v1_manifest(temporary, routes, descriptors, config)
            manifest = json.loads(path.read_text())
            manifest["selections"]["central_recovery"]["train_sample_ids"].reverse()
            manifest.pop("snapshot_sha256")
            manifest["snapshot_sha256"] = canonical_sha256(manifest)
            path.write_text(json.dumps(manifest))
            with patch("carla_tools.dataset_v1._native_episode_gate", side_effect=native_boundary_fixture):
                report = validate_dataset_v1_manifest(path)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["errors"][0]["code"], "dataset-source-or-selection-reconstruction-mismatch")

    def test_missing_matrix_and_reserved_town_fail_before_snapshot_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, routes, descriptors = dataset_fixture(temporary)
            with self.assertRaisesRegex(DatasetV1Error, "incomplete-collection-matrix"):
                build_dataset_v1_manifest(temporary, routes, descriptors[:-1], config)
            altered = copy.deepcopy(routes)
            altered[0]["route"]["town"] = "Town02"
            with self.assertRaises(ValueError):
                build_dataset_v1_manifest(temporary, altered, descriptors, config)

    def test_changed_excluded_startup_png_is_replayed_and_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, routes, descriptors = dataset_fixture(temporary)
            path = build_dataset_v1_manifest(temporary, routes, descriptors, config)
            startup = Path(temporary) / descriptors[0]["episode"] / "images/00000010.png"
            startup.write_bytes(startup.read_bytes() + b"changed")
            with patch("carla_tools.dataset_v1._native_episode_gate", side_effect=native_boundary_fixture):
                report = validate_dataset_v1_manifest(path)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["errors"][0]["code"], "dataset-native-episode-replay-failed")

    def test_failed_native_replay_retains_episode_and_nested_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, routes, descriptors = dataset_fixture(temporary)
            path = build_dataset_v1_manifest(temporary, routes, descriptors, config)
            failure = {"status": "FAIL", "errors": [{"code": "synthetic-wrong-initial-pose"}]}
            with patch("carla_tools.dataset_v1._native_episode_gate", return_value=failure):
                report = validate_dataset_v1_manifest(path)
            self.assertEqual(report["status"], "FAIL")
            detail = report["errors"][0]
            self.assertEqual(detail["code"], "dataset-native-episode-replay-failed")
            self.assertEqual(detail["context"], {
                "episode_index": 1, "episode_count": len(descriptors),
                "episode_id": "dataset_fixture_000", "episode": descriptors[0]["episode"],
                "native_errors": [{"code": "synthetic-wrong-initial-pose"}]})
            failure["errors"][0]["code"] = "mutated-later"
            self.assertEqual(detail["context"]["native_errors"][0]["code"], "synthetic-wrong-initial-pose")

    def test_duplicate_rgb_exclusion_has_exact_source_and_reason_ledger(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, routes, descriptors = dataset_fixture(temporary)
            folder = Path(temporary) / descriptors[0]["episode"]
            rows = [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
            rows[-1]["image"]["sha256"] = rows[-2]["image"]["sha256"]
            (folder / rows[-1]["image"]["path"]).write_bytes((folder / rows[-2]["image"]["path"]).read_bytes())
            (folder / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            path = build_dataset_v1_manifest(temporary, routes, descriptors, config)
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["exclusion_counts"]["duplicate_rgb"], 1)
            self.assertEqual(manifest["sample_count"], 139)
            duplicate = next(item for item in manifest["exclusion_ledger"] if item["reason"] == "duplicate_rgb")
            self.assertEqual(duplicate["frame"], rows[-1]["frame"])
            self.assertTrue(duplicate["duplicate_of_sample_id"].endswith(":0000000013"))


if __name__ == "__main__":
    unittest.main()
