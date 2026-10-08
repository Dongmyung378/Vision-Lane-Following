"""D6 generation and independent verification of real persisted fixtures."""

from __future__ import annotations

import copy
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from carla_tools.config import CollectionConfig, collection_config_from_dict
from carla_tools.contracts import ContractError, route_sha256
from carla_tools.episode import EpisodeWriter, validate_episode
from carla_tools.labels import LabelError, generate_labels, generate_waypoint_label, validate_labels
from scripts.carla_rgb_probe import png_from_rgb
from scripts.generate_labels import main

try:
    from tests.test_data_contracts import frame_fixture, metadata_fixture, route_fixture, transform_fixture
except ModuleNotFoundError:
    from test_data_contracts import frame_fixture, metadata_fixture, route_fixture, transform_fixture


def curve_route(sign):
    route = route_fixture()
    radius = 30.
    anchors = []
    for index in range(141):
        arc = index / 4
        anchors.append({"road_id": 1, "section_id": 0, "lane_id": -1, "s": arc, "arc_distance_m": arc,
                        "world_location": {"x": radius * math.sin(arc / radius),
                                           "y": sign * radius * (1 - math.cos(arc / radius)), "z": 0.},
                        "is_junction": False, "successor_count": 1})
    route.update(start=copy.deepcopy(anchors[0]), sample_step_m=.25, centerline_anchor_sequence=anchors)
    route["route_sha256"] = route_sha256(route)
    return route


class WaypointLabelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        value = CollectionConfig().to_dict()
        value["camera"].update(width=16, height=16)
        value["camera"]["mount"] = {"location": {"x": 0., "y": 0., "z": 1.}, "rotation": {"pitch": 0., "yaw": 0., "roll": 0.}}
        value["verification"]["frames"] = 3
        self.cfg = collection_config_from_dict(value)
        self.png = png_from_rgb(bytes([1, 2, 3, 4, 5, 6] * 128), 16, 16)
        self.output = self.root / "derived" / "labels.json"

    def episode(self, route=None, x=0., y=0., yaw=0., complete=True):
        route = route or route_fixture()
        path = self.root / "raw"
        meta = metadata_fixture()
        meta.update(route_contract=route, expected_frames=3, collection_config=self.cfg.to_dict(),
                    configuration_sha256=self.cfg.sha256, camera_calibration={"intrinsics": self.cfg.camera.intrinsics(),
                       "requested_mount_matrix": transform_fixture(z=1.)["matrix"],
                       "coordinate_convention": "carla-x-forward-y-right-z-up", "rotation_units": "degrees"},
                    applied_manifest={"applied_verified": True})
        writer = EpisodeWriter(path, meta)
        for index in range(3):
            frame, row = 100 + index, frame_fixture()
            row.update(index=index, frame=frame, world_frame=frame, snapshot_frame=frame, sensor_frame=frame,
                       timestamp=frame * .05, snapshot_timestamp=frame * .05, sensor_timestamp=frame * .05)
            row["vehicle"]["transform"] = transform_fixture(x=x, y=y, yaw=yaw)
            row["vehicle"]["control"] = self.cfg.vehicle.control.to_dict()
            row["camera"]["transform"] = transform_fixture(x=x, y=y, z=1., yaw=yaw)
            writer.append(row, self.png)
        if complete:
            writer.finalize(True, {"status": "PASS", "sensor_stopped": True, "settings_restored": True,
                                   "weather_restored": True, "owned_actors_absent": True})
        return path

    def source_bytes(self, path):
        return {str(item.relative_to(path)): item.read_bytes() for item in path.rglob("*") if item.is_file()}

    def generated(self, **kwargs):
        source = self.episode(**kwargs)
        report = generate_labels(source, self.output)
        self.assertEqual(report["status"], "PASS", report.get("errors"))
        return source, report

    def write_sidecar(self, sidecar):
        self.output.write_text(json.dumps(sidecar, allow_nan=True), encoding="utf-8")

    def assert_invalid(self, source):
        report = validate_labels(source, self.output)
        self.assertEqual(report["status"], "FAIL", report)
        for key in ("metadata", "frames", "sidecar"):
            self.assertNotIn(key, report)
        return report

    def test_straight_complete_sidecar_preserves_source_bytes_and_state(self):
        source = self.episode()
        before = self.source_bytes(source)
        report = generate_labels(source, self.output)
        self.assertEqual(report["status"], "PASS", report)
        verified = validate_labels(source, self.output)
        self.assertEqual(verified["status"], "PASS", verified)
        self.assertEqual(verified["metrics"]["waypoint_shape"], [5, 2])
        self.assertEqual(verified["metrics"]["first_frame"], 100)
        self.assertEqual(verified["metrics"]["last_frame"], 102)
        self.assertEqual(verified["sidecar"]["rows"][0]["label"]["waypoints_ego_m"], [[float(d), 0.] for d in (3, 6, 9, 12, 15)])
        self.assertEqual(self.source_bytes(source), before)
        self.assertEqual(verified["metadata"]["labels_state"], "not-generated-D6")
        self.assertTrue(all(frame["label"] is None for frame in verified["frames"]))
        self.assertEqual(validate_episode(source)["status"], "PASS")

    def test_left_and_right_curves_follow_arc_distance_instead_of_fixed_ego_x(self):
        for sign in (-1, 1):
            meta, frame = metadata_fixture(), frame_fixture()
            meta["route_contract"] = curve_route(sign)
            label = generate_waypoint_label(frame, meta)
            for index, distance in enumerate((3, 6, 9, 12, 15)):
                x, y = label["waypoints_ego_m"][index]
                self.assertAlmostEqual(x, 30 * math.sin(distance / 30), places=10)
                self.assertAlmostEqual(y, sign * 30 * (1 - math.cos(distance / 30)), places=10)
                self.assertLess(x, distance)
                self.assertEqual(math.copysign(1., y), sign)

    def test_lateral_and_yaw_recovery_transform_retains_original_world_lane(self):
        frame, meta = frame_fixture(), metadata_fixture()
        normal = generate_waypoint_label(frame, meta)
        frame["vehicle"]["transform"] = transform_fixture(y=.6, yaw=6.)
        altered = generate_waypoint_label(frame, meta)
        self.assertEqual(altered["waypoints_world_m"], normal["waypoints_world_m"])
        self.assertEqual(altered["target_lane"], normal["target_lane"])
        self.assertLess(altered["waypoints_ego_m"][0][1], 0.)
        self.assertNotEqual(altered["waypoints_ego_m"], normal["waypoints_ego_m"])
        source, report = self.generated(y=.6, yaw=6.)
        self.assertLess(report["metrics"]["max_world_roundtrip_error_m"], 1e-10)

    def test_near_end_and_junction_branch_horizons_never_publish_output(self):
        source = self.episode(x=21.)
        report = generate_labels(source, self.output)
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(self.output.exists())
        for field, value in (("is_junction", True), ("successor_count", 0), ("successor_count", 2)):
            frame, meta = frame_fixture(), metadata_fixture()
            meta["route_contract"]["centerline_anchor_sequence"][10][field] = value
            with self.subTest(field=field, value=value), self.assertRaises((LabelError, ContractError)):
                generate_waypoint_label(frame, meta)

    def test_nonfinite_reflected_and_wrong_frame_geometry_is_rejected(self):
        changes = [lambda f: f["vehicle"]["transform"]["matrix"][0].__setitem__(0, -1.),
                   lambda f: f["vehicle"]["transform"]["location"].update(x=float("nan")),
                   lambda f: f.update(sensor_frame=101)]
        for change in changes:
            frame = frame_fixture()
            change(frame)
            with self.subTest(change=change), self.assertRaises((LabelError, ContractError)):
                generate_waypoint_label(frame, metadata_fixture())

    def test_self_crossing_and_equidistant_remote_route_projection_are_refused(self):
        meta = metadata_fixture()
        route = meta["route_contract"]
        # A small loop has valid spacing and topology flags but ambiguous XY geometry.
        radius = 35 / (2 * math.pi)
        for anchor in route["centerline_anchor_sequence"]:
            angle = anchor["arc_distance_m"] / radius
            anchor["world_location"].update(x=radius * math.sin(angle), y=radius * (1 - math.cos(angle)))
        route["start"] = copy.deepcopy(route["centerline_anchor_sequence"][0])
        route["route_sha256"] = route_sha256(route)
        with self.assertRaisesRegex(LabelError, "self-crossing-fixed-route"):
            generate_waypoint_label(frame_fixture(), meta)
        # An open hairpin with no crossing still has two equal nearest projections.
        route = route_fixture()
        points = [[float(i), 0.] for i in range(16)] + [[15., float(i)] for i in range(1, 6)] + [[float(15-i), 5.] for i in range(1, 16)]
        for anchor, point in zip(route["centerline_anchor_sequence"], points):
            anchor["world_location"].update(x=point[0], y=point[1])
        route["start"] = copy.deepcopy(route["centerline_anchor_sequence"][0])
        route["route_sha256"] = route_sha256(route)
        meta["route_contract"] = route
        frame = frame_fixture()
        frame["vehicle"]["transform"] = transform_fixture(x=5., y=2.5)
        with self.assertRaisesRegex(LabelError, "ambiguous-route-reference"):
            generate_waypoint_label(frame, meta)

    def test_duplicate_missing_reordered_and_wrong_sample_or_image_rows_fail_independent_read(self):
        source, report = self.generated()
        mutations = [lambda s: s["rows"].pop(), lambda s: s["rows"].append(copy.deepcopy(s["rows"][0])),
                     lambda s: s["rows"].reverse(), lambda s: s["rows"][1].update(frame=100),
                     lambda s: s["rows"][0].update(sample_id="other:0000000100"),
                     lambda s: s["rows"][0]["image"].update(path="../wrong.png")]
        for change in mutations:
            sidecar = copy.deepcopy(report["sidecar"])
            change(sidecar)
            self.write_sidecar(sidecar)
            self.assert_invalid(source)

    def test_wrong_axis_shape_lane_reference_and_diagnostic_values_cannot_validate(self):
        source, report = self.generated()
        mutations = [lambda s: s["rows"][0]["label"].update(axes="x-forward_y-left"),
                     lambda s: s["rows"][0]["label"]["waypoints_ego_m"].pop(),
                     lambda s: s["rows"][0]["label"]["target_lane"].update(lane_id=-2),
                     lambda s: s["rows"][0]["label"].update(reference_arc_m=1.),
                     lambda s: s["rows"][0]["label"]["waypoints_world_m"][0].__setitem__(0, 4.),
                     lambda s: s["rows"][0]["label"]["waypoints_ego_m"][0].__setitem__(0, float("nan")),
                     lambda s: s["rows"][0]["diagnostics"].update(max_ego_error_m=.5),
                     lambda s: s["metrics"].update(frame_count=True), lambda s: s.update(status="FAILED")]
        for change in mutations:
            sidecar = copy.deepcopy(report["sidecar"])
            change(sidecar)
            self.write_sidecar(sidecar)
            self.assert_invalid(source)

    def test_wrong_source_hashes_or_changed_source_are_detected(self):
        source, report = self.generated()
        for key in ("episode_id", "route_sha256", "configuration_sha256", "metadata_sha256", "frames_jsonl_sha256"):
            sidecar = copy.deepcopy(report["sidecar"])
            sidecar["source"][key] = "wrong"
            self.write_sidecar(sidecar)
            self.assert_invalid(source)
        self.write_sidecar(report["sidecar"])
        # Semantically unchanged whitespace must still invalidate the source-byte snapshot.
        with (source / "episode.json").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        self.assertEqual(validate_episode(source)["status"], "PASS")
        self.assert_invalid(source)

    def test_running_or_corrupt_source_and_existing_or_inside_source_output_are_refused(self):
        source = self.episode(complete=False)
        self.assertEqual(generate_labels(source, self.output)["status"], "FAIL")
        self.assertFalse(self.output.exists())
        self.assertEqual(generate_labels(source, source / "labels.json")["status"], "FAIL")
        self.assertFalse((source / "labels.json").exists())

    def test_corrupt_source_png_cannot_publish_labels(self):
        source = self.episode()
        image = source / "images" / "00000100.png"
        payload = image.read_bytes()
        image.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
        self.assertEqual(generate_labels(source, self.output)["status"], "FAIL")
        self.assertFalse(self.output.exists())

    def test_successful_output_is_never_overwritten_and_cli_only_prints_summary(self):
        source, report = self.generated()
        before = self.output.read_bytes()
        duplicate = generate_labels(source, self.output)
        self.assertEqual(duplicate["status"], "FAIL")
        self.assertEqual(self.output.read_bytes(), before)
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["--episode", str(source), "--labels", str(self.output), "--validate-only"])
        self.assertEqual(result, 0)
        summary = json.loads(output.getvalue())
        self.assertFalse(summary["raw_episode_modified"])
        self.assertNotIn("frames", summary)
        self.assertNotIn("metadata", summary)

    def test_failed_atomic_publication_never_leaves_pass_or_pending_artifacts(self):
        source = self.episode()
        with patch("carla_tools.labels.os.link", side_effect=OSError("private-path")):
            report = generate_labels(source, self.output)
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob("*.pending")), [])
        self.assertNotIn("private-path", json.dumps(report))

    def test_failed_pending_cleanup_cannot_leave_a_published_pass(self):
        source = self.episode()
        original_unlink = Path.unlink
        denied = []

        def fail_once(path, *args, **kwargs):
            if path.name.startswith(".labels-") and not denied:
                denied.append(path)
                raise OSError("pending-cleanup-denied")
            return original_unlink(path, *args, **kwargs)

        with patch("carla_tools.labels.Path.unlink", new=fail_once):
            report = generate_labels(source, self.output)
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob("*.pending")), [])
        self.assertEqual(validate_episode(source)["status"], "PASS")

    def test_leftover_internal_pending_artifact_is_never_publicly_pass(self):
        source, report = self.generated()
        pending = self.output.with_name(".labels-interrupted.pending")
        pending.write_bytes(self.output.read_bytes())
        checked = validate_labels(source, pending)
        self.assertEqual(checked["status"], "FAIL")
        self.assertEqual(checked["errors"][0]["code"], "unpublished-label-artifact")
        self.assertEqual(generate_labels(source, pending)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
