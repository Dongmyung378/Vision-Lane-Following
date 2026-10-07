"""Real persisted fixtures exercise D5 split grouping and failure refusal."""

from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from carla_tools.config import CollectionConfig, collection_config_from_dict
from carla_tools.contracts import route_sha256
from carla_tools.episode import EpisodeWriter, validate_episode
from carla_tools.splits import CSV_FIELDS, build_split_manifest, scene_distance_m
from scripts.carla_rgb_probe import png_from_rgb

try:
    from tests.test_data_contracts import frame_fixture, route_fixture, transform_fixture
except ModuleNotFoundError:
    from test_data_contracts import frame_fixture, route_fixture, transform_fixture


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "split_manifest.csv"
        value = CollectionConfig().to_dict()
        value["camera"].update(width=16, height=16)
        value["camera"]["mount"] = {"location": {"x": 0., "y": 0., "z": 1.}, "rotation": {"pitch": 0., "yaw": 0., "roll": 0.}}
        value["verification"].update(frames=2)
        self.cfg = collection_config_from_dict(value)
        self.png = png_from_rgb(bytes([1, 2, 3, 4, 5, 6] * 128), 16, 16)

    def episode(self, identifier="ep-a", route=None, y=0., start_frame=100, version="smoke-v1", complete=True):
        route = route or route_fixture(y=y)
        path = self.root / identifier
        mount = transform_fixture(z=1.)["matrix"]
        metadata = {"schema_version": 1, "dataset_version": version, "episode_id": identifier,
                    "route_id": route["route_id"], "scene_group_id": route["scene_group_id"], "town": route["town"],
                    "expected_frames": 2, "collection_config": self.cfg.to_dict(), "configuration_sha256": self.cfg.sha256,
                    "camera_calibration": {"intrinsics": self.cfg.camera.intrinsics(), "requested_mount_matrix": mount,
                                           "coordinate_convention": "carla-x-forward-y-right-z-up", "rotation_units": "degrees"},
                    "applied_manifest": {"applied_verified": True}, "route_contract": route,
                    "perturbation": {"lateral_m": 0., "yaw_deg": 0., "provenance": "synthetic-stationary-test"},
                    "control_mode": "stationary_D4_smoke", "labels_state": "not-generated-D6"}
        writer = EpisodeWriter(path, metadata)
        for index in range(2):
            row = frame_fixture()
            frame = start_frame + index
            row.update(index=index, frame=frame, world_frame=frame, snapshot_frame=frame, sensor_frame=frame,
                       timestamp=frame * .05, snapshot_timestamp=frame * .05, sensor_timestamp=frame * .05)
            row["vehicle"]["transform"] = transform_fixture(y=y)
            row["vehicle"]["control"] = self.cfg.vehicle.control.to_dict()
            row["camera"]["transform"] = transform_fixture(y=y, z=1.)
            writer.append(row, self.png)
        if complete:
            writer.finalize(True, {"status": "PASS", "sensor_stopped": True, "settings_restored": True,
                                   "weather_restored": True, "owned_actors_absent": True})
        return path

    def build(self, paths, assignments):
        return build_split_manifest(paths, self.output, assignments)

    def assert_failed_without_output(self, report, code=None):
        self.assertEqual(report["status"], "FAIL", report)
        self.assertFalse(self.output.exists())
        if code:
            self.assertEqual(report["errors"][0]["code"], code, report)

    def edit_metadata(self, path, mutate):
        metadata = json.loads((path / "episode.json").read_text(encoding="utf-8"))
        mutate(metadata)
        (path / "episode.json").write_text(json.dumps(metadata), encoding="utf-8")

    def edit_frames(self, path, mutate):
        rows = [json.loads(line) for line in (path / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
        mutate(rows)
        (path / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_complete_three_episode_smoke_manifest_is_reproducible_and_reuses_identical_content(self):
        train = self.episode()
        repeat = self.episode("ep-b", start_frame=200)
        dev = self.episode("ep-c", route_fixture(y=150., route_id="route-dev", scene_group="scene-dev"), y=150., start_frame=300)
        self.assertEqual(validate_episode(train)["status"], "PASS")
        report = self.build([train, repeat, dev], {"ep-a": "train", "ep-b": "train", "ep-c": "dev"})
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["split_counts"], {"train": 4, "dev": 2})
        self.assertEqual(report["minimum_cross_split_scene_distance_m"], 150.)
        self.assertEqual(report["final_test"]["collection_status"], "UNCOLLECTED")
        with self.output.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, CSV_FIELDS)
            rows = list(reader)
        self.assertEqual(len(rows), len({row["sample_id"] for row in rows}))
        original = self.output.read_bytes()
        again = self.build([dev, repeat, train], {"ep-c": "dev", "ep-b": "train", "ep-a": "train"})
        self.assertEqual(again["status"], "PASS", again)
        self.assertFalse(again["written"])
        self.assertEqual(self.output.read_bytes(), original)

    def test_same_route_and_scene_cannot_cross_split(self):
        a, b = self.episode(), self.episode("ep-b", start_frame=200)
        self.assert_failed_without_output(self.build([a, b], {"ep-a": "train", "ep-b": "dev"}), "route-split-leakage")
        other_route = route_fixture(y=150., route_id="route-b", scene_group="scene-a")
        c = self.episode("ep-c", other_route, y=150., start_frame=300)
        self.assert_failed_without_output(self.build([a, c], {"ep-a": "train", "ep-c": "dev"}), "scene-group-split-leakage")

    def test_nearby_scene_with_different_ids_is_still_leakage_and_boundary_is_conservative(self):
        a = self.episode()
        for distance in (74., 75.):
            identifier = f"ep-d{int(distance)}"
            b = self.episode(identifier, route_fixture(y=distance, route_id=f"route-{int(distance)}", scene_group=f"scene-{int(distance)}"),
                             y=distance, start_frame=300)
            self.assert_failed_without_output(self.build([a, b], {"ep-a": "train", identifier: "dev"}), "spatial-scene-split-leakage")

    def test_time_adjacent_nearby_fragments_fail_even_with_new_route_and_scene_ids(self):
        a = self.episode()
        b = self.episode("ep-b", route_fixture(y=50., route_id="route-b", scene_group="scene-b"), y=50., start_frame=102)
        self.assert_failed_without_output(self.build([a, b], {"ep-a": "train", "ep-b": "dev"}), "spatial-and-temporal-split-leakage")

    def test_opposite_direction_physical_scene_is_shared_and_crossed_segments_have_zero_distance(self):
        a = self.episode()
        route = route_fixture(route_id="route-reverse", scene_group="scene-reverse")
        for index, anchor in enumerate(route["centerline_anchor_sequence"]):
            anchor["world_location"]["x"] = float(35 - index)
            anchor["lane_id"] = 1
        route["start"] = copy.deepcopy(route["centerline_anchor_sequence"][0])
        route["route_sha256"] = route_sha256(route)
        b = self.episode("ep-b", route, start_frame=300)
        self.assert_failed_without_output(self.build([a, b], {"ep-a": "train", "ep-b": "dev"}), "spatial-scene-split-leakage")
        first = {"town": "Town01", "segments": [((0., 0.), (2., 2.))]}
        second = {"town": "Town01", "segments": [((0., 2.), (2., 0.))]}
        self.assertEqual(scene_distance_m(first, second), 0.)

    def test_missing_extra_frame_level_and_final_assignments_are_rejected(self):
        a = self.episode()
        b = self.episode("ep-b", start_frame=200)
        self.assert_failed_without_output(self.build([a, b], {"ep-a": "train"}), "missing-episode-assignment")
        self.assert_failed_without_output(self.build([a], {"ep-a": "train", "unused": "dev"}), "unused-episode-assignment")
        self.assert_failed_without_output(self.build([a], {"ep-a:0000000100": "train"}), "invalid-identifier")
        self.assert_failed_without_output(self.build([a], {"ep-a": "final_test"}), "unsupported-D5-split")

    def test_duplicate_path_episode_ids_and_mixed_versions_fail(self):
        a = self.episode()
        self.assert_failed_without_output(self.build([a, a], {"ep-a": "train"}), "duplicate-episode-path")
        b = self.episode("ep-b", start_frame=200)
        self.edit_metadata(b, lambda m: m.update(episode_id="ep-a"))
        self.assert_failed_without_output(self.build([a, b], {"ep-a": "train"}), "duplicate-episode-id")
        self.edit_metadata(b, lambda m: m.update(episode_id="ep-b", dataset_version="smoke-v2"))
        self.assert_failed_without_output(self.build([a, b], {"ep-a": "train", "ep-b": "train"}), "mixed-dataset-versions")

    def test_running_corrupt_and_missing_images_never_enter_manifest(self):
        running = self.episode(complete=False)
        self.assert_failed_without_output(self.build([running], {"ep-a": "train"}), "episode-integrity-validation-failed")
        for identifier, action in (("missing", lambda path: next((path / "images").iterdir()).unlink()),
                                   ("corrupt", lambda path: next((path / "images").iterdir()).write_bytes(b"broken"))):
            path = self.episode(identifier, start_frame=300)
            action(path)
            self.assert_failed_without_output(self.build([path], {identifier: "train"}), "episode-integrity-validation-failed")

    def test_frame_tampering_duplicate_nonfinite_and_path_escape_are_refused(self):
        mutations = [lambda rows: rows.append(copy.deepcopy(rows[0])),
                     lambda rows: rows[0]["vehicle"]["velocity"].update(x=float("nan")),
                     lambda rows: rows[0]["image"].update(path="../escape.png")]
        for index, mutation in enumerate(mutations):
            identifier = f"bad-{index}"
            path = self.episode(identifier, start_frame=300)
            self.edit_frames(path, mutation)
            self.assert_failed_without_output(self.build([path], {identifier: "train"}), "episode-integrity-validation-failed")

    def test_label_placeholders_missing_states_and_unknown_frame_fields_are_refused_after_integrity(self):
        # D4 disk validator is intentionally label-agnostic; D5 adds these checks.
        mutations = [lambda rows: rows[0].update(label=[[0., 0.]] * 5),
                     lambda rows: rows[0].pop("label_state"), lambda rows: rows[0].update(extra=1)]
        for index, mutation in enumerate(mutations):
            identifier = f"labels-{index}"
            path = self.episode(identifier, start_frame=300)
            self.edit_frames(path, mutation)
            self.assertEqual(validate_episode(path)["status"], "PASS")
            self.assert_failed_without_output(self.build([path], {identifier: "train"}))

    def test_conflicting_route_contracts_and_reserved_town_are_refused(self):
        a = self.episode()
        changed = route_fixture()
        changed["opendrive_sha256"] = "b" * 64
        changed["route_sha256"] = route_sha256(changed)
        b = self.episode("ep-b", changed, start_frame=200)
        self.assert_failed_without_output(self.build([a, b], {"ep-a": "train", "ep-b": "train"}), "conflicting-route-contract")
        # D4 configuration would refuse Town02 first. A forged PASS cannot bypass the D5 reservation either.
        meta = validate_episode(a)["metadata"]
        meta["town"] = meta["route_contract"]["town"] = "Town02"
        meta["collection_config"]["map"] = "Town02"
        meta["route_contract"]["map_package"] = "Carla/Maps/Town02"
        meta["route_contract"]["route_sha256"] = route_sha256(meta["route_contract"])
        (a / "episode.json").write_text(json.dumps(meta), encoding="utf-8")
        with patch("carla_tools.episode.validate_episode", return_value={"status": "PASS", "errors": [], "frame_count": 2, "metadata": meta}):
            self.assert_failed_without_output(self.build([a], {"ep-a": "train"}), "reserved-final-town-not-available-D5")

    def test_existing_manifest_version_or_content_conflict_never_overwrites(self):
        a = self.episode()
        self.assertEqual(self.build([a], {"ep-a": "train"})["status"], "PASS")
        original = self.output.read_bytes()
        report = self.build([a], {"ep-a": "dev"})
        self.assertEqual(report["errors"][0]["code"], "existing-manifest-content-conflict")
        self.assertEqual(self.output.read_bytes(), original)
        self.edit_metadata(a, lambda m: m.update(dataset_version="smoke-v2"))
        report = self.build([a], {"ep-a": "train"})
        self.assertEqual(report["errors"][0]["code"], "existing-manifest-version-conflict")
        self.assertEqual(self.output.read_bytes(), original)

    def test_one_town_cannot_mix_different_map_sources_under_one_snapshot(self):
        a = self.episode()
        changed = route_fixture(y=150., route_id="route-b", scene_group="scene-b")
        changed["opendrive_sha256"] = "b" * 64
        changed["route_sha256"] = route_sha256(changed)
        b = self.episode("ep-b", changed, y=150., start_frame=300)
        self.assert_failed_without_output(self.build([a, b], {"ep-a": "train", "ep-b": "dev"}), "town-map-source-conflict")

    def test_legitimate_package_aliases_do_not_change_shared_opendrive_coordinates(self):
        a = self.episode()
        route = route_fixture(y=150., route_id="route-b", scene_group="scene-b")
        route["map_package"] = "/Game/Carla/Maps/Town01"
        route["route_sha256"] = route_sha256(route)
        b = self.episode("ep-b", route, y=150., start_frame=300)
        self.assertEqual(self.build([a, b], {"ep-a": "train", "ep-b": "dev"})["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
