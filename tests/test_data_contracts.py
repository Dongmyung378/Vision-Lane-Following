"""D5 geometry, route identity and honest ungenerated-label contracts."""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from carla_tools.contracts import (ContractError, load_dataset_contract, route_sha256, sample_id,
                                   strict_json_loads, validate_episode_identity, validate_frame_contract,
                                   validate_route_contract, validate_waypoint_label)


def route_fixture(x=0., y=0., route_id="route-a", scene_group="scene-a"):
    anchors = [{"road_id": 1, "section_id": 0, "lane_id": -1, "s": float(i), "arc_distance_m": float(i),
                "world_location": {"x": x + i, "y": y, "z": 0.}, "is_junction": False, "successor_count": 1}
               for i in range(36)]
    route = {"schema_version": 1, "route_id": route_id, "scene_group_id": scene_group, "town": "Town01",
             "carla_version": "0.9.16", "map_package": "Carla/Maps/Town01", "opendrive_sha256": "a" * 64,
             "spawn_point_index": 0, "start": copy.deepcopy(anchors[0]), "horizon_m": 15.,
             "min_required_route_m": 35., "sample_step_m": 1., "route_length_m": 35.,
             "centerline_anchor_sequence": anchors, "selection_scope": "D5-topology-only-not-frame-labels"}
    route["route_sha256"] = route_sha256(route)
    return route


def transform_fixture(x=0., y=0., z=0., yaw=0.):
    sine, cosine = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
    return {"location": {"x": x, "y": y, "z": z}, "rotation": {"pitch": 0., "yaw": yaw, "roll": 0.},
            "matrix": [[cosine, -sine, 0., x], [sine, cosine, 0., y], [0., 0., 1., z], [0., 0., 0., 1.]]}


def frame_fixture():
    return {"schema_version": 1, "index": 0, "frame": 100, "world_frame": 100, "snapshot_frame": 100,
            "sensor_frame": 100, "timestamp": 5., "snapshot_timestamp": 5., "sensor_timestamp": 5., "delta_seconds": .05,
            "image": {"path": "images/00000100.png", "sha256": "a" * 64, "width": 16, "height": 16},
            "vehicle": {"transform": transform_fixture(), "velocity": {"x": 0., "y": 0., "z": 0.},
                        "angular_velocity": {"x": 0., "y": 0., "z": 0.}, "acceleration": {"x": 0., "y": 0., "z": 0.},
                        "control": {"throttle": 0., "steer": 0., "brake": 1., "hand_brake": True, "reverse": False}},
            "camera": {"transform": transform_fixture(z=1.)}, "label_state": "NOT_GENERATED", "label": None}


def metadata_fixture():
    return {"dataset_version": "smoke-v1", "episode_id": "episode-a", "route_id": "route-a",
            "scene_group_id": "scene-a", "town": "Town01", "route_contract": route_fixture(),
            "control_mode": "stationary_D4_smoke", "labels_state": "not-generated-D6",
            "perturbation": {"lateral_m": 0., "yaw_deg": 0., "provenance": "stationary"}}


def label_fixture(frame=None):
    frame = frame or frame_fixture()
    matrix = frame["vehicle"]["transform"]["matrix"]
    world = [[float(distance), 0., 0.] for distance in (3, 6, 9, 12, 15)]
    ego = [[sum(matrix[k][axis] * (point[k] - matrix[k][3]) for k in range(3)) for axis in range(2)] for point in world]
    return {"schema_version": 1, "ego_frame": 100, "route_id": "route-a", "target_lane": {"road_id": 1, "section_id": 0, "lane_id": -1},
            "coordinate_origin": "vehicle_actor_transform", "axes": "x-forward_y-right", "units": "m",
            "distance_reference": "target-lane-centerline-arc-length", "horizon_arc_m": [3, 6, 9, 12, 15],
            "reference_arc_m": 0., "waypoints_world_m": world, "waypoints_ego_m": ego}


class DataContractTests(unittest.TestCase):
    def test_checked_in_contract_and_schemas_have_complete_roles(self):
        contract = load_dataset_contract()
        self.assertEqual(contract["input"]["model_fields"], ["rgb"])
        self.assertEqual(contract["waypoint"]["horizon_arc_m"], [3, 6, 9, 12, 15])
        self.assertEqual(contract["split"]["reserved_final"]["collection_status"], "UNCOLLECTED")
        base = Path(__file__).resolve().parents[1] / "schemas"
        schemas = {name: json.loads((base / name).read_text(encoding="utf-8")) for name in
                   ("waypoint_label.schema.json", "episode_frame.schema.json", "split_manifest.schema.json")}
        self.assertEqual(schemas["waypoint_label.schema.json"]["properties"]["waypoints_ego_m"]["minItems"], 5)
        self.assertEqual(schemas["episode_frame.schema.json"]["properties"]["label_state"]["enum"], ["NOT_GENERATED", "GENERATED"])
        self.assertEqual(schemas["split_manifest.schema.json"]["properties"]["split"]["enum"], ["train", "dev"])

    def test_loader_refuses_duplicates_nonfinite_unknown_fields_and_contract_changes(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e400}'):
            with self.subTest(raw=raw), self.assertRaises(ContractError):
                strict_json_loads(raw)
        for mutate in (lambda c: c.update(extra=1), lambda c: c["input"].update(model_fields=["rgb", "pose"]),
                       lambda c: c.update(schema_version=True), lambda c: c["split"].update(same_town_scene_buffer_m=0)):
            with tempfile.TemporaryDirectory() as directory:
                contract = load_dataset_contract()
                mutate(contract)
                path = Path(directory) / "contract.json"
                path.write_text(json.dumps(contract), encoding="utf-8")
                with self.assertRaises(ContractError):
                    load_dataset_contract(path)

    def test_native_windows_and_unreal_package_paths_are_preserved(self):
        for package in ("Carla/Maps/Town01", "/Game/Carla/Maps/Town01"):
            route = route_fixture()
            route["map_package"] = package
            route["route_sha256"] = route_sha256(route)
            validate_route_contract(route)
        for package in ("Carla/Maps/../Town01", "/Game/Other/Town01", "Carla/Maps/Town02"):
            route = route_fixture()
            route["map_package"] = package
            with self.assertRaisesRegex(ContractError, "invalid-map-package"):
                validate_route_contract(route)

    def test_route_topology_guard_rejects_junction_branch_and_lane_end_anywhere(self):
        for key, value in (("is_junction", True), ("successor_count", 2), ("successor_count", 0)):
            route = route_fixture()
            route["centerline_anchor_sequence"][15][key] = value
            with self.assertRaisesRegex(ContractError, "ambiguous-route-horizon"):
                validate_route_contract(route)

    def test_missing_short_duplicate_nonfinite_and_tampered_route_are_rejected(self):
        mutations = [lambda r: r.pop("opendrive_sha256"), lambda r: r.update(route_length_m=14),
                     lambda r: r["centerline_anchor_sequence"][4].update(arc_distance_m=3),
                     lambda r: r["start"]["world_location"].update(x=float("nan")),
                     lambda r: r.update(route_sha256="0" * 64), lambda r: r["start"].update(lane_id=0)]
        for mutation in mutations:
            route = route_fixture()
            mutation(route)
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                validate_route_contract(route)

    def test_ungenerated_sample_requires_null_without_zero_placeholder(self):
        frame, meta = frame_fixture(), metadata_fixture()
        validate_episode_identity(meta)
        validate_frame_contract(frame, meta)
        for label in ([[0., 0.]] * 5, {}, label_fixture()):
            frame["label"] = label
            with self.assertRaisesRegex(ContractError, "ungenerated-label-must-be-null"):
                validate_frame_contract(frame, meta)

    def test_recovery_pose_changes_local_label_but_keeps_world_route_and_lane(self):
        frame, meta = frame_fixture(), metadata_fixture()
        frame["vehicle"]["transform"] = transform_fixture(y=.5, yaw=10.)
        label = label_fixture(frame)
        validate_waypoint_label(label, frame, meta)
        self.assertLess(label["waypoints_ego_m"][0][1], 0.)  # target lies left of an actor displaced to the right
        label["waypoints_ego_m"] = label_fixture()["waypoints_ego_m"]
        with self.assertRaisesRegex(ContractError, "label-ego-transform-mismatch"):
            validate_waypoint_label(label, frame, meta)

    def test_generated_label_requires_five_finite_same_frame_same_target_route_points(self):
        mutations = [lambda label: label.update(ego_frame=99), lambda label: label.update(route_id="other"),
                     lambda label: label["target_lane"].update(lane_id=-2), lambda label: label.update(units="pixels"),
                     lambda label: label["waypoints_ego_m"].pop(), lambda label: label["waypoints_ego_m"][0].append(0),
                     lambda label: label["waypoints_ego_m"][0].__setitem__(0, float("inf")),
                     lambda label: label["waypoints_world_m"][0].__setitem__(1, .5),
                     lambda label: label.update(reference_arc_m=30.)]
        for mutation in mutations:
            label = label_fixture()
            mutation(label)
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                validate_waypoint_label(label, frame_fixture(), metadata_fixture())
        frame = frame_fixture()
        frame.update(label_state="GENERATED", label=label_fixture())
        validate_frame_contract(frame, metadata_fixture())

    def test_frame_paths_time_pose_and_missing_fields_fail_closed(self):
        for path in ("../images/00000100.png", "C:/images/00000100.png", "/images/00000100.png", "images\\00000100.png", "images/00000101.png"):
            frame = frame_fixture()
            frame["image"]["path"] = path
            with self.assertRaisesRegex(ContractError, "invalid-image-path"):
                validate_frame_contract(frame, metadata_fixture())
        mutations = [lambda row: row.pop("label_state"), lambda row: row.update(snapshot_frame=101),
                     lambda row: row.update(sensor_timestamp=5.1), lambda row: row.update(index=True),
                     lambda row: row["vehicle"]["transform"]["matrix"][0].__setitem__(0, 2.),
                     lambda row: row["vehicle"]["velocity"].update(x=float("nan")),
                     lambda row: row.update(label_state="TBD")]
        for mutation in mutations:
            row = frame_fixture()
            mutation(row)
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                validate_frame_contract(row, metadata_fixture())

    def test_scene_route_identity_and_sample_identity_are_explicit(self):
        meta = metadata_fixture()
        meta["scene_group_id"] = "different-scene"
        with self.assertRaisesRegex(ContractError, "episode-route-identity-mismatch"):
            validate_episode_identity(meta)
        self.assertEqual(sample_id("episode-a", 100), "episode-a:0000000100")
        with self.assertRaises(ContractError):
            sample_id("../escape", 100)

    def test_valid_geometry_at_wrong_reference_is_not_a_future_target_for_this_pose(self):
        label = label_fixture()
        label["reference_arc_m"] = 1.
        label["waypoints_world_m"] = [[float(distance + 1), 0., 0.] for distance in (3, 6, 9, 12, 15)]
        label["waypoints_ego_m"] = [point[:2] for point in label["waypoints_world_m"]]
        with self.assertRaisesRegex(ContractError, "label-reference-pose-mismatch"):
            validate_waypoint_label(label, frame_fixture(), metadata_fixture())

    def test_route_spawn_must_match_capture_configuration(self):
        meta = metadata_fixture()
        meta["collection_config"] = {"map": "Town01", "vehicle": {"spawn_point_index": 4}}
        with self.assertRaisesRegex(ContractError, "episode-route-spawn-mismatch"):
            validate_episode_identity(meta)

    def test_native_capture_diagnostics_are_explicit_and_checked(self):
        row = frame_fixture()
        row.update(tick_to_rgb_ms=5.5, mount_matrix_max_abs_error=.00001,
                   rgb={"width": 16, "height": 16, "rgb_bytes": 768, "channel_min": [0, 1, 2],
                        "channel_max": [10, 11, 12], "spatially_nonconstant": True})
        row["vehicle"].update(actor_id=174, control_source="client-cache-at-world-frame")
        row["camera"]["actor_id"] = 175
        validate_frame_contract(row, metadata_fixture())
        mutations = [lambda value: value["rgb"].update(rgb_bytes=1), lambda value: value["rgb"].update(channel_max=[0, 1, 2]),
                     lambda value: value["vehicle"].update(control_source="current-live-pose"),
                     lambda value: value["camera"].update(actor_id=True),
                     lambda value: value.update(mount_matrix_max_abs_error=float("nan")),
                     lambda value: value.update(unapproved_diagnostic=1)]
        for mutation in mutations:
            invalid = copy.deepcopy(row)
            mutation(invalid)
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                validate_frame_contract(invalid, metadata_fixture())


if __name__ == "__main__":
    unittest.main()
