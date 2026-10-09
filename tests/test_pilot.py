"""Moving sample selection and scene-held-out route allocation checks."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from carla_tools.contracts import ContractError, canonical_sha256, route_sha256
from carla_tools.driving_route import FixedRouteTracker
from carla_tools.geometry_survey import classify_geometry
from carla_tools.pilot import (PilotError, _moving_samples, _safe_file, _scene_distance, build_pilot_manifest,
                              load_pilot_config, pilot_config_from_dict, select_pilot_routes,
                              survey_pilot_routes, validate_pilot_manifest)
from test_data_contracts import transform_fixture

ROOT = Path(__file__).resolve().parents[1]


def candidate_fixture():
    catalog = json.loads((ROOT / "configs/routes_d6_geometry.json").read_text(encoding="utf-8"))
    by_case = {item["case"]: item["route"] for item in catalog["geometry_cases"]}
    result = []
    for index, case in enumerate(("straight", "straight", "straight", "left_curve", "left_curve", "right_curve", "right_curve")):
        route = copy.deepcopy(by_case[case])
        base = route["start"]["world_location"]
        dx, dy = index * 500. - base["x"], -base["y"]
        for anchor in route["centerline_anchor_sequence"]:
            anchor["world_location"]["x"] += dx
            anchor["world_location"]["y"] += dy
        route.update(route_id=f"pilot_fixture_p{index:03d}", scene_group_id=f"pilot_fixture_scene_p{index:03d}",
                     spawn_point_index=index, start=copy.deepcopy(route["centerline_anchor_sequence"][0]))
        route["route_sha256"] = route_sha256(route)
        result.append({"case": case, "route": route})
    return result


def moving_fixture():
    from test_driving_route import dense_route
    route = dense_route()
    tracker, rows = FixedRouteTracker(route), []
    for index, (x, speed) in enumerate(((0., 0.), (0., 0.), (.01, .1), (.06, 1.), (.11, 1.))):
        pose = transform_fixture(x=x)
        tracking = tracker.observe(pose, index + 10, .05 * (index + 10))
        rows.append({"frame": index + 10, "timestamp": .05 * (index + 10), "vehicle": {"transform": pose},
                     "tracking": tracking, "speed_mps": speed,
                     "image": {"path": f"images/{index + 10:08d}.png", "sha256": str(index) * 64,
                               "width": 640, "height": 360}})
    metadata = {"status": "COMPLETE", "method": "privileged_reference", "episode_id": "pilot_fixture_episode",
                "route_contract": route, "camera_calibration": {"fixed": True},
                "collection_config": {"weather": {"preset": "ClearNoon"}},
                "condition": {"name": "center", "lateral_m": 0., "yaw_deg": 0.},
                "evaluation_metrics": {"driving_success": True}}
    descriptor = {"split": "train", "case": "straight", "condition": "center", "episode": "episodes/train/pilot_fixture_episode"}
    return metadata, rows, descriptor


class PilotConfigTests(unittest.TestCase):
    def test_configuration_round_trip_is_frozen_and_canonical(self):
        config = load_pilot_config()
        self.assertEqual(pilot_config_from_dict(config.to_dict()).sha256, config.sha256)
        value = config.to_dict()
        value["conditions"][0]["yaw_deg"] = 99.
        self.assertEqual(config.conditions[0]["yaw_deg"], 0.)

    def test_invalid_budget_buffers_and_conditions_are_rejected(self):
        for key, bad in (("minimum_samples", 1999), ("maximum_samples", 5001), ("scene_buffer_m", 74.),
                         ("minimum_speed_mps", False), ("conditions", [])):
            with self.subTest(key=key):
                value = load_pilot_config().to_dict()
                value[key] = bad
                with self.assertRaises((PilotError, ValueError)):
                    pilot_config_from_dict(value)

    def test_missing_or_additional_configuration_keys_are_rejected(self):
        for remove in (True, False):
            value = load_pilot_config().to_dict()
            if remove:
                value.pop("scene_buffer_m")
            else:
                value["frame_padding"] = True
            with self.assertRaises(PilotError):
                pilot_config_from_dict(value)


class PilotRouteTests(unittest.TestCase):
    def test_all_geometry_cases_and_conditions_are_split_by_native_scene(self):
        config = load_pilot_config()
        selected = select_pilot_routes(candidate_fixture(), config)
        self.assertEqual(len(selected), 7)
        for split in ("train", "dev"):
            for case in ("straight", "left_curve", "right_curve"):
                self.assertEqual(sum(item["split"] == split and item["case"] == case for item in selected),
                                 getattr(config, split + "_routes_per_case")[case])
        self.assertEqual(len({item["route"]["route_id"] for item in selected}), len(selected))
        self.assertTrue(all(item["route"]["route_sha256"] == route_sha256(item["route"]) for item in selected))
        for scene in {item["route"]["scene_group_id"] for item in selected}:
            self.assertEqual(len({item["split"] for item in selected if item["route"]["scene_group_id"] == scene}), 1)

    def test_scene_overlap_is_rejected_even_with_distinct_route_ids(self):
        candidates = candidate_fixture()
        first, second = candidates[-2]["route"], candidates[-1]["route"]
        dx = first["start"]["world_location"]["x"] - second["start"]["world_location"]["x"]
        for anchor in second["centerline_anchor_sequence"]:
            anchor["world_location"]["x"] += dx
        second["start"] = copy.deepcopy(second["centerline_anchor_sequence"][0])
        second["route_sha256"] = route_sha256(second)
        with self.assertRaisesRegex(PilotError, "pilot-separated-train-dev-geometry-unavailable"):
            select_pilot_routes(candidates, load_pilot_config())

    def test_missing_curve_case_refuses_to_substitute_stationary_or_straight_frames(self):
        candidates = [item for item in candidate_fixture() if item["case"] != "right_curve"]
        with self.assertRaises(PilotError):
            select_pilot_routes(candidates, load_pilot_config())

    def test_reserved_town_is_rejected_before_spawn_points_are_read(self):
        class ReservedMap:
            name = "Carla/Maps/Town02"

            def get_spawn_points(self):
                raise AssertionError("reserved map geometry must never be read")

        with self.assertRaisesRegex(PilotError, "pilot-reserved-or-unexpected-town"):
            survey_pilot_routes(ReservedMap(), object(), load_pilot_config())

    def test_vectorized_scene_distance_matches_independent_segment_geometry(self):
        from carla_tools.splits import scene_distance_m
        first = {"town": "Town01", "segments": [((0., 0.), (2., 2.)), ((4., 0.), (4., 0.))]}
        for segments in ([((0., 2.), (2., 0.))], [((10., 0.), (10., 3.))], [((3., 4.), (3., 4.))]):
            second = {"town": "Town01", "segments": segments}
            self.assertAlmostEqual(_scene_distance(first, second), scene_distance_m(first, second), places=12)


class PilotSampleTests(unittest.TestCase):
    def test_stationary_and_nearly_stationary_startup_rows_do_not_count(self):
        metadata, rows, descriptor = moving_fixture()
        samples = _moving_samples(metadata, rows, descriptor, load_pilot_config())
        self.assertEqual([sample["frame"] for sample in samples], [13, 14])
        self.assertTrue(all(sample["speed_mps"] >= .2 for sample in samples))
        self.assertEqual(samples[0]["waypoints_ego_m"], rows[3]["tracking"]["label"]["waypoints_ego_m"])
        self.assertEqual(samples[0]["vehicle_transform"], rows[3]["vehicle"]["transform"])
        self.assertEqual(samples[0]["calibration_sha256"], canonical_sha256(metadata["camera_calibration"]))

    def test_recovery_labels_are_recomputed_from_perturbed_same_frame_pose(self):
        metadata, rows, descriptor = moving_fixture()
        route = metadata["route_contract"]
        for row in rows:
            pose = transform_fixture(x=row["vehicle"]["transform"]["location"]["x"], y=.3)
            row["vehicle"]["transform"] = pose
            row["tracking"] = FixedRouteTracker(route).observe(pose, row["frame"], row["timestamp"])
        metadata["condition"] = {"name": "right_offset", "lateral_m": .3, "yaw_deg": 0.}
        descriptor["condition"] = "right_offset"
        samples = _moving_samples(metadata, rows, descriptor, load_pilot_config())
        self.assertAlmostEqual(samples[0]["waypoints_ego_m"][0][1], -.3)
        self.assertEqual(samples[0]["perturbation"], {"lateral_m": .3, "yaw_deg": 0.})

    def test_source_label_tampering_is_rejected_before_manifest_is_built(self):
        metadata, rows, descriptor = moving_fixture()
        rows[3]["tracking"]["label"]["waypoints_ego_m"][0][1] += 1.
        with self.assertRaises(ContractError):
            _moving_samples(metadata, rows, descriptor, load_pilot_config())

    def test_manifest_is_new_only_and_excludes_duplicates_without_frame_padding(self):
        metadata, rows, descriptor = moving_fixture()
        rows[-1]["image"]["sha256"] = rows[-2]["image"]["sha256"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / descriptor["episode"]
            episode.mkdir(parents=True)
            (episode / "episode.json").write_text(json.dumps(metadata))
            (episode / "steps.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            path = build_pilot_manifest(root, [], [descriptor], load_pilot_config())
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["sample_count"], 1)
            self.assertEqual(manifest["rejected_frames"], {"stationary_or_startup_frames": 3, "duplicate_rgb_frames": 1})
            with self.assertRaises(FileExistsError):
                build_pilot_manifest(root, [], [descriptor], load_pilot_config())

    def test_cross_split_identical_rgb_is_rejected(self):
        metadata, rows, descriptor = moving_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root, descriptors = Path(temporary), []
            for split in ("train", "dev"):
                current = dict(descriptor, split=split, episode=f"episodes/{split}/pilot_fixture_episode")
                episode = root / current["episode"]
                episode.mkdir(parents=True)
                (episode / "episode.json").write_text(json.dumps(metadata))
                (episode / "steps.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
                descriptors.append(current)
            with self.assertRaisesRegex(PilotError, "pilot-cross-split-duplicate-rgb"):
                build_pilot_manifest(root, [], descriptors, load_pilot_config())
            self.assertFalse((root / "pilot_manifest.json").exists())

    def test_source_paths_refuse_escape_windows_drive_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ("../x", "/x", "C:/x", "dir\\x"):
                with self.subTest(relative=relative), self.assertRaises(PilotError):
                    _safe_file(root, relative)

    def test_malformed_manifest_never_reports_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pilot_manifest.json"
            path.write_text('{"schema_version":1,"schema_version":1}')
            report = validate_pilot_manifest(path)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["errors"][0]["code"], "duplicate-json-key")


class PilotManifestReplayTests(unittest.TestCase):
    """Test pilot orchestration around an independently covered native gate.

    Native PNG/control replay is deliberately replaced at this unit boundary;
    these fixtures test sample reconstruction, source hashes and grouping.
    They are never represented as simulator or training evidence.
    """

    def build_fixture(self, root):
        config = load_pilot_config()
        selected = select_pilot_routes(candidate_fixture(), config)
        episodes, frame_counter = [], 100
        for item in selected:
            route = item["route"]
            for condition in config.conditions:
                relative = f"episodes/{item['split']}/{route['route_id']}__{condition['name']}"
                directory = root / relative
                (directory / "images").mkdir(parents=True)
                rows, tracker = [], FixedRouteTracker(route)
                for index in range(73):
                    anchor = route["centerline_anchor_sequence"][index]
                    location, yaw = anchor["world_location"], anchor["world_rotation"]["yaw"]
                    pose = transform_fixture(x=location["x"], y=location["y"], yaw=yaw)
                    frame_counter += 1
                    frame, timestamp = frame_counter, frame_counter * .05
                    label = tracker.observe(pose, frame, timestamp)
                    payload = b"independent-native-gate-fixture:" + str(frame).encode()
                    image = f"images/{frame:08d}.png"
                    (directory / image).write_bytes(payload)
                    rows.append({"frame": frame, "timestamp": timestamp, "vehicle": {"transform": pose},
                                 "tracking": label, "speed_mps": 0. if index == 0 else 5.,
                                 "image": {"path": image, "sha256": hashlib.sha256(payload).hexdigest(),
                                           "width": 640, "height": 360}})
                metadata = {"status": "COMPLETE", "method": "privileged_reference", "episode_id": directory.name,
                            "geometry_case": item["case"], "route_contract": route, "camera_calibration": {"fixed": True},
                            "collection_config": {"weather": {"preset": "ClearNoon"}}, "condition": condition,
                            "evaluation_metrics": {"driving_success": True}}
                (directory / "episode.json").write_text(json.dumps(metadata))
                (directory / "steps.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
                episodes.append({"split": item["split"], "case": item["case"], "condition": condition["name"], "episode": relative})
        return build_pilot_manifest(root, selected, episodes, config)

    @staticmethod
    def native_gate(path, *, allowed_conditions):
        rows = [json.loads(line) for line in (path / "steps.jsonl").read_text().splitlines()]
        return {"status": "PASS", "evaluation_metrics": {"driving_success": True}, "checks": {"native_gate_stub": True},
                "source": {"image_sha256": {row["image"]["path"]: row["image"]["sha256"] for row in rows}}}

    def test_complete_grouped_manifest_replays_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.build_fixture(root)
            original = path.read_bytes()
            with patch("carla_tools.baseline_validation.validate_baseline_episode", side_effect=self.native_gate):
                gate = validate_pilot_manifest(path)
                self.assertEqual(gate["status"], "PASS", gate)
                self.assertEqual(gate["sample_count"], 2520)
                self.assertEqual(gate["split_counts"], {"train": 1440, "dev": 1080})
                self.assertGreater(gate["minimum_cross_split_scene_distance_m"], 75.)
                for change in ("waypoint", "metadata", "png", "missing-condition", "duplicate-sample"):
                    with self.subTest(change=change):
                        value = json.loads(original)
                        mutated_path = None
                        original_source = None
                        if change == "waypoint":
                            value["samples"][0]["waypoints_ego_m"][0][1] += .3
                        elif change == "metadata":
                            value["samples"][0]["split"] = "dev"
                        elif change == "missing-condition":
                            value["episodes"].pop()
                        elif change == "duplicate-sample":
                            value["samples"].append(copy.deepcopy(value["samples"][0]))
                        else:
                            mutated_path = root / value["samples"][0]["image"]
                            original_source = mutated_path.read_bytes()
                            mutated_path.write_bytes(b"tampered-image")
                        path.write_text(json.dumps(value))
                        self.assertEqual(validate_pilot_manifest(path)["status"], "FAIL")
                        if mutated_path is not None:
                            mutated_path.write_bytes(original_source)
                        path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
