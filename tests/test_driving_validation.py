"""Persisted QA trace replay and meaningful corrupted/unsafe source checks."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from carla_tools.capture import _calibration, _mount_matrix
from carla_tools.color import color_provenance
from carla_tools.config import CollectionConfig
from carla_tools.contracts import read_strict_json, route_sha256
from carla_tools.controller import PurePursuitController, VehicleGeometry, load_controller_config
from carla_tools.driving_route import FixedRouteTracker
from carla_tools.driving import body_lane_margin
from carla_tools.driving_validation import DEFAULT_VALIDATION_PATH, SCOPE, _body_margin, validate_drive_episode, validate_driving_manifest
from carla_tools.projection import encode_rgb8_png
from scripts.verify_driving import main

try:
    from tests.test_data_contracts import transform_fixture
    from tests.test_driving_route import dense_route
except ModuleNotFoundError:
    from test_data_contracts import transform_fixture
    from test_driving_route import dense_route


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def write_episode(root, *, case="straight", name="center", episode_id="drive-test", slow_recovery=False, interrupted_recovery=False):
    root.mkdir()
    (root / "images").mkdir()
    policy = read_strict_json(DEFAULT_VALIDATION_PATH)
    condition = next(item for item in policy["initial_conditions"] if item["name"] == name)
    cfg, controller_cfg = CollectionConfig(), load_controller_config()
    route = dense_route({"straight": 0, "left_curve": -1, "right_curve": 1}[case])
    route_sign = {"straight": 0, "left_curve": -1, "right_curve": 1}[case]
    for anchor in route["centerline_anchor_sequence"]:
        anchor["world_rotation"] = {"pitch": 0., "yaw": route_sign * math.degrees(anchor["arc_distance_m"] / 30.), "roll": 0.}
    route["start"] = copy.deepcopy(route["centerline_anchor_sequence"][0])
    route["route_sha256"] = route_sha256(route)
    geometry = VehicleGeometry(2.8, -1.4, 1.4, 30., ((0., 1.), (100., 1.)), front_track_m=1.4)
    tracker, controller = FixedRouteTracker(route), PurePursuitController(controller_cfg, geometry)
    bbox = {"location": {"x": 0., "y": 0., "z": 0.}, "extent": {"x": 1.9, "y": .8, "z": .7}}
    mount = _mount_matrix(cfg)
    rgb = bytes([3, 2, 1]) * (cfg.camera.width * cfg.camera.height)
    bgra = bytes([1, 2, 3, 255]) * (cfg.camera.width * cfg.camera.height)
    png = encode_rgb8_png(rgb, cfg.camera.width, cfg.camera.height)
    proof = color_provenance(bgra, rgb, cfg.camera.width, cfg.camera.height)
    rows, previous = [], None
    sign = {"straight": 0, "left_curve": -1, "right_curve": 1}[case]
    for index in range(49):
        # Avoid the mathematically tied projection of an exactly inner-normal
        # point at a polygon corner; the tracker deliberately rejects ties.
        arc = index * .397
        route_yaw = 0. if sign == 0 else sign * math.degrees(arc / 30.)
        decay = 1. if slow_recovery else max(0., 1. - index / 10.)
        if interrupted_recovery:
            decay = 1. if index % 15 < 5 else 1. / 3.
        lateral, yaw_offset = condition["lateral_m"] * decay, condition["yaw_deg"] * decay
        x = arc if sign == 0 else 30. * math.sin(arc / 30.)
        y = 0. if sign == 0 else sign * 30. * (1. - math.cos(arc / 30.))
        x -= lateral * math.sin(math.radians(route_yaw))
        y += lateral * math.cos(math.radians(route_yaw))
        pose = transform_fixture(x=x, y=y, yaw=route_yaw + yaw_offset)
        frame, timestamp = 100 + index, 5. + index * .05
        tracking = tracker.observe(pose, frame, timestamp)
        speed = 0. if index == 0 else 8.
        command = controller.step(tracking["label"]["waypoints_ego_m"], speed)
        matrix = [[sum(pose["matrix"][r][k] * mount[k][c] for k in range(4)) for c in range(4)] for r in range(4)]
        camera = {"location": {axis: matrix[row][3] for row, axis in enumerate(("x", "y", "z"))},
                  "rotation": {"pitch": -12., "yaw": pose["rotation"]["yaw"], "roll": 0.}, "matrix": matrix}
        margin = body_lane_margin(pose, tracking, bbox, route)
        terminal = index == 48
        image = None
        if index % policy["image_stride"] == 0 or terminal:
            relative = f"images/{frame:08d}.png"
            (root / relative).write_bytes(png)
            image = {"path": relative, "sha256": _sha(png), "width": cfg.camera.width, "height": cfg.camera.height,
                     "color_provenance": proof}
        applied = {"throttle": 0., "brake": 1., "steer": 0., "hand_brake": True, "reverse": False} if previous is None else {key: previous["command"][key] for key in ("throttle", "brake", "steer", "hand_brake", "reverse")}
        row = {"schema_version": 1, "index": index, "frame": frame, "world_frame": frame, "snapshot_frame": frame,
               "sensor_frame": frame, "timestamp": timestamp, "snapshot_timestamp": timestamp, "sensor_timestamp": timestamp,
               "delta_seconds": .05, "tick_to_rgb_ms": 2., "vehicle": {"transform": pose,
                 "velocity": {"x": speed, "y": 0., "z": 0.}, "applied_control": applied},
               "camera": {"transform": camera}, "speed_mps": speed, "tracking": tracking, "command": command,
               "command_source_frame": frame, "command_target_frame": None if terminal else frame + 1,
               "terminal": terminal, "body_lane_margin_m": margin,
               "native_wheel_angles_deg": {"front_left": 0., "front_right": 0.}, "image": image}
        rows.append(row)
        previous = row
    trace = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()
    (root / "steps.jsonl").write_bytes(trace)
    wheels = [{"position_cm": {"x": x, "y": y, "z": -20.}, "max_steer_angle_deg": angle}
              for x, y, angle in ((140., -70., 30.), (140., 70., 30.), (-140., -70., 0.), (-140., 70., 0.))]
    stop_control = {"throttle": 0., "brake": 1., "steer": rows[-1]["vehicle"]["applied_control"]["steer"], "hand_brake": False, "reverse": False}
    stop_observations = []
    final_pose = copy.deepcopy(rows[-1]["vehicle"]["transform"])
    for index in range(1, 13):
        speed = max(0., 8. * (1. - index / 3.))
        final_pose = transform_fixture(x=final_pose["location"]["x"] + speed * .05 * math.cos(math.radians(final_pose["rotation"]["yaw"])),
                                       y=final_pose["location"]["y"] + speed * .05 * math.sin(math.radians(final_pose["rotation"]["yaw"])),
                                       yaw=final_pose["rotation"]["yaw"])
        stop_observations.append({"frame": 148 + index, "timestamp": rows[-1]["timestamp"] + index * .05,
                                  "vehicle": {"transform": copy.deepcopy(final_pose), "velocity": {"x": speed, "y": 0., "z": 0.},
                                              "applied_control": dict(stop_control, hand_brake=index > 10)},
                                  "speed_mps": speed, "native_wheel_angles_deg": {"front_left": 0., "front_right": 0.},
                                  "body_lane_margin_m": body_lane_margin(final_pose, {}, bbox, route)})
    metadata = {"schema_version": 1, "status": "COMPLETE", "scope": SCOPE, "episode_id": episode_id,
                "geometry_case": case, "condition": condition, "route_contract": route,
                "collection_config": cfg.to_dict(), "controller_config": controller_cfg.to_dict(),
                "controller_config_sha256": controller_cfg.sha256, "vehicle_geometry": geometry.to_dict(),
                "geometry_evidence": {"measurement_vehicle_transform": transform_fixture(), "wheels": wheels,
                    "original_steering_curve": [[0., 1.], [20., .9], [60., .8], [120., .7]],
                    "applied_steering_curve": [[0., 1.], [100., 1.]], "position_units": "world-centimetres",
                    "measurement_after_warmup": True, "steering_normalization": "measured-front-steer-limit-times-constant-factor"},
                "camera_calibration": _calibration(cfg, mount), "bounding_box": bbox, "applied_manifest": {"applied_verified": True},
                "cleanup": {"status": "PASS", "errors": [], "sensor_stopped": True, "settings_restored": True,
                    "weather_restored": True, "owned_actors_absent": True, "owned_actors_absent_in_snapshot": True,
                    "owned_actors_absent_in_current_list": True, "absence_verification": "fresh-snapshot-and-current-actor-list",
                    "cleanup_reference_frame": 160, "absence_observed_frame": 161, "owned_actor_ids": [1, 2, 3, 4]},
                "events": [], "event_callback_errors": [], "drive_end_frame": 148, "post_stop_frame": 160,
                "step_count": len(rows), "trace_sha256": _sha(trace), "validation_config": policy,
                "stop_control": stop_control, "stop_observations": stop_observations}
    (root / "episode.json").write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    return metadata, rows


class DrivingValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.episode = self.root / "episode"
        self.metadata, self.rows = write_episode(self.episode)

    def metadata_write(self, metadata=None):
        (self.episode / "episode.json").write_text(json.dumps(self.metadata if metadata is None else metadata, sort_keys=True), encoding="utf-8")

    def rows_write(self):
        trace = ("\n".join(json.dumps(row, sort_keys=True) for row in self.rows) + "\n").encode()
        (self.episode / "steps.jsonl").write_bytes(trace)
        self.metadata["trace_sha256"] = _sha(trace)
        self.metadata_write()

    def fail_code(self, expected):
        report = validate_drive_episode(self.episode)
        self.assertEqual(report["status"], "FAIL", report)
        self.assertEqual(report["errors"][0]["code"], expected, report)

    def test_fresh_complete_trace_replays_controller_and_original_route_and_preserves_sources(self):
        before = [(path.relative_to(self.episode).as_posix(), path.read_bytes()) for path in self.episode.rglob("*") if path.is_file()]
        report = validate_drive_episode(self.episode)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["metrics"]["step_count"], 49)
        self.assertEqual(report["metrics"]["image_count"], 4)
        self.assertGreaterEqual(report["metrics"]["final_reference_m"], 19.)
        self.assertFalse(report["final_test_used"])
        self.assertEqual(before, [(path.relative_to(self.episode).as_posix(), path.read_bytes()) for path in self.episode.rglob("*") if path.is_file()])

    def test_failed_and_running_episodes_cannot_pass(self):
        for status in ("RUNNING", "FAILED"):
            self.metadata["status"] = status
            self.metadata_write()
            self.fail_code("driving-episode-not-complete")

    def test_bad_trace_hash_refused_before_replay(self):
        self.metadata["trace_sha256"] = "0" * 64
        self.metadata_write()
        self.fail_code("driving-trace-hash-mismatch")

    def test_same_frame_pose_label_units_sign_or_route_hash_corruption_refused(self):
        original = copy.deepcopy(self.rows)
        changes = [lambda row: row["tracking"]["label"].update(units="cm"),
                   lambda row: row["tracking"]["label"].update(ego_frame=999),
                   lambda row: row["tracking"]["label"]["waypoints_ego_m"][0].__setitem__(1, 10.),
                   lambda row: row["tracking"].update(route_sha256="0" * 64)]
        for mutate in changes:
            self.rows = copy.deepcopy(original)
            mutate(self.rows[1])
            self.rows_write()
            self.fail_code("driving-tracking-replay-mismatch")

    def test_rehashed_wrong_control_and_wrong_control_application_refused(self):
        original = copy.deepcopy(self.rows)
        self.rows[2]["command"]["steer"] = .9
        self.rows_write()
        self.fail_code("driving-controller-replay-mismatch")
        self.rows = copy.deepcopy(original)
        self.rows[2]["vehicle"]["applied_control"]["throttle"] = .1
        self.rows_write()
        self.fail_code("driving-command-application-mismatch")

    def test_missing_duplicate_frame_time_or_wrong_command_target_refused(self):
        original = copy.deepcopy(self.rows)
        for key, value, code in (("frame", 100, "driving-frame-alignment-mismatch"),
                                 ("sensor_timestamp", 50., "driving-timestamp-alignment-mismatch"),
                                 ("command_target_frame", 500, "driving-command-target-frame-mismatch")):
            self.rows = copy.deepcopy(original)
            self.rows[1][key] = value
            self.rows_write()
            self.fail_code(code)

    def test_image_hash_crc_missing_or_orphan_refused(self):
        path = self.episode / self.rows[0]["image"]["path"]
        original = path.read_bytes()
        path.write_bytes(original[:-5] + b"xxxxx")
        self.fail_code("projection-image-hash-mismatch")
        path.write_bytes(original)
        (self.episode / "images/orphan.png").write_bytes(original)
        self.fail_code("driving-orphan-or-missing-image")

    def test_rgb_proof_red_blue_swap_cannot_be_certified(self):
        self.rows[0]["image"]["color_provenance"]["sample_pixels"][0]["bgra"][0] = 33
        self.rows_write()
        report = validate_drive_episode(self.episode)
        self.assertEqual(report["status"], "FAIL", report)

    def test_world_cleanup_late_event_callback_error_and_short_settling_fail(self):
        original = copy.deepcopy(self.metadata)
        mutations = [(lambda meta: meta["cleanup"].update(weather_restored=False), "driving-cleanup-not-verified"),
                     (lambda meta: meta.update(events=[{"kind": "collision", "frame": 153, "timestamp": 7.65}]), "driving-collision-or-lane-event"),
                     (lambda meta: meta.update(event_callback_errors=["late-callback-failed"]), "driving-event-callback-failed"),
                     (lambda meta: meta.update(post_stop_frame=149), "driving-stop-end-frame-mismatch")]
        for mutate, code in mutations:
            self.metadata = copy.deepcopy(original)
            mutate(self.metadata)
            self.metadata_write()
            self.fail_code(code)

    def test_recovery_hold_is_measured_from_actual_pose_and_deadline_not_declared_pass(self):
        episode = self.root / "recovery"
        write_episode(episode, name="right_offset")
        report = validate_drive_episode(episode)
        self.assertEqual(report["status"], "PASS", report)
        self.assertGreaterEqual(report["metrics"]["recovery_completed_seconds"], 1.)
        stuck = self.root / "stuck"
        write_episode(stuck, name="right_offset", slow_recovery=True)
        report = validate_drive_episode(stuck)
        self.assertEqual(report["status"], "FAIL", report)
        self.assertEqual(report["errors"][0]["code"], "driving-recovery-not-held-within-deadline")

    def test_recovery_hold_resets_when_errors_leave_tolerance(self):
        episode = self.root / "interrupted-hold"
        write_episode(episode, name="left_offset", interrupted_recovery=True)
        report = validate_drive_episode(episode)
        self.assertEqual(report["status"], "FAIL", report)
        self.assertEqual(report["errors"][0]["code"], "driving-recovery-not-held-within-deadline")

    def test_rehashed_corrupt_png_is_still_checked_for_crc_and_complete_payload(self):
        image = self.rows[0]["image"]
        path = self.episode / image["path"]
        damaged = bytearray(path.read_bytes())
        damaged[-1] ^= 1
        path.write_bytes(damaged)
        image["sha256"] = _sha(damaged)
        self.rows_write()
        self.assertEqual(validate_drive_episode(self.episode)["status"], "FAIL")

    def test_camera_mount_shift_and_duplicate_aligned_frame_are_rejected(self):
        original = copy.deepcopy(self.rows)
        pose = self.rows[1]["camera"]["transform"]
        pose["location"]["x"] += .1
        pose["matrix"][0][3] += .1
        self.rows_write()
        self.fail_code("driving-camera-mount-mismatch")
        self.rows = copy.deepcopy(original)
        for key in ("frame", "world_frame", "snapshot_frame", "sensor_frame", "command_source_frame"):
            self.rows[1][key] = 100
        self.rows_write()
        self.fail_code("driving-frame-sequence-mismatch")

    def test_native_geometry_world_cm_measurement_and_original_curve_are_required(self):
        self.metadata["geometry_evidence"]["wheels"][0]["position_cm"]["x"] += 20
        self.metadata_write()
        self.assertEqual(validate_drive_episode(self.episode)["status"], "FAIL")

    def test_body_margin_cannot_be_relabelled_safe_or_camera_mount_shifted(self):
        self.rows[1]["body_lane_margin_m"] += 1.
        self.rows_write()
        self.fail_code("driving-body-lane-margin-failed")

    def test_stop_keeps_last_applied_steer_and_brakes_before_two_parking_ticks(self):
        report = validate_drive_episode(self.episode)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["metrics"]["stopping_observation_count"], 12)
        self.assertAlmostEqual(report["metrics"]["stopping_completed_seconds"], .15)
        self.assertEqual(report["metrics"]["final_parking_speed_mps"], 0.)
        self.assertGreaterEqual(report["metrics"]["minimum_stopping_body_lane_margin_m"], .15)
        self.metadata["stop_control"]["steer"] += .1
        self.metadata_write()
        self.fail_code("driving-stop-control-mismatch")

    def test_early_hand_brake_or_missing_final_parking_readback_fails(self):
        original = copy.deepcopy(self.metadata)
        for index, hand_brake in ((0, True), (-1, False)):
            self.metadata = copy.deepcopy(original)
            self.metadata["stop_observations"][index]["vehicle"]["applied_control"]["hand_brake"] = hand_brake
            self.metadata_write()
            self.fail_code("driving-stop-control-readback-mismatch")

    def test_stop_frame_time_speed_and_recorded_corner_margin_must_match(self):
        original = copy.deepcopy(self.metadata)
        mutations = [(lambda obs: obs.update(frame=1), "driving-stop-frame-sequence-mismatch"),
                     (lambda obs: obs.update(timestamp=0.), "driving-stop-frame-sequence-mismatch"),
                     (lambda obs: obs.update(speed_mps=99.), "driving-stop-speed-mismatch"),
                     (lambda obs: obs.update(body_lane_margin_m=5.), "driving-stop-body-lane-margin-failed")]
        for mutate, code in mutations:
            self.metadata = copy.deepcopy(original)
            mutate(self.metadata["stop_observations"][0])
            self.metadata_write()
            self.fail_code(code)

    def test_stop_no_low_speed_and_parking_while_moving_are_rejected(self):
        original = copy.deepcopy(self.metadata)
        for observation in self.metadata["stop_observations"][:-2]:
            observation["vehicle"]["velocity"]["x"] = 1.
            observation["speed_mps"] = 1.
        self.metadata_write()
        self.fail_code("driving-stop-not-within-deadline")
        self.metadata = copy.deepcopy(original)
        self.metadata["stop_observations"][-1]["vehicle"]["velocity"]["x"] = 1.
        self.metadata["stop_observations"][-1]["speed_mps"] = 1.
        self.metadata_write()
        self.fail_code("driving-parking-not-stationary")

    def test_early_low_speed_does_not_allow_parking_after_a_moving_last_braking_tick(self):
        # Earlier braking observations and both parking observations are low
        # speed, but the actual readback just before parking is still moving.
        observation = self.metadata["stop_observations"][-3]
        observation["vehicle"]["velocity"]["x"] = .5
        observation["speed_mps"] = .5
        self.metadata_write()
        self.fail_code("driving-parking-started-before-stop")

    def test_stop_longer_than_two_seconds_is_rejected_even_if_event_free(self):
        old = self.metadata["stop_observations"]
        observations = copy.deepcopy(old[:-2]) + [copy.deepcopy(old[-3]) for _ in range(31)] + copy.deepcopy(old[-2:])
        for index, observation in enumerate(observations, start=1):
            observation["frame"] = self.rows[-1]["frame"] + index
            observation["timestamp"] = self.rows[-1]["timestamp"] + .05 * index
        self.metadata["stop_observations"] = observations
        self.metadata["post_stop_frame"] = observations[-1]["frame"]
        self.metadata["cleanup"]["cleanup_reference_frame"] = observations[-1]["frame"]
        self.metadata["cleanup"]["absence_observed_frame"] = observations[-1]["frame"] + 1
        self.metadata_write()
        self.fail_code("driving-stop-deadline-exceeded")

    def test_corner_margin_uses_curved_original_route_instead_of_one_tangent_plane(self):
        route = dense_route(1)
        arc = 19.
        pose = transform_fixture(x=30. * math.sin(arc / 30.), y=30. * (1. - math.cos(arc / 30.)), yaw=math.degrees(arc / 30.))
        tracking = FixedRouteTracker(route).observe(pose, 1, .05)
        bbox = {"location": {"x": .03, "y": 0., "z": .735}, "extent": {"x": 2.4, "y": 1.1, "z": .74}}
        actual = _body_margin(pose, tracking, bbox, route)
        proxy = _body_margin(pose, tracking, bbox)
        self.assertGreater(proxy - actual, .05)
        self.assertGreater(actual, .15)
        self.assertAlmostEqual(actual, body_lane_margin(pose, tracking, bbox, route), places=10)

    def test_incomplete_duplicate_or_wrong_manifest_cases_fail(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "episodes": [{"case": "straight", "condition": "center", "episode": "episode"}]}), encoding="utf-8")
        self.assertEqual(validate_driving_manifest(manifest)["errors"][0]["code"], "driving-cases-incomplete")
        manifest.write_text(json.dumps({"schema_version": 1, "episodes": [{"case": "straight", "condition": "center", "episode": "episode"}] * 2}), encoding="utf-8")
        self.assertEqual(validate_driving_manifest(manifest)["errors"][0]["code"], "duplicate-or-unknown-driving-case")

    def test_all_fifteen_geometry_and_recovery_cases_validate(self):
        episodes = []
        for case in ("straight", "left_curve", "right_curve"):
            for item in read_strict_json(DEFAULT_VALIDATION_PATH)["initial_conditions"]:
                relative = f"{case}_{item['name']}"
                write_episode(self.root / relative, case=case, name=item["name"], episode_id=relative)
                episodes.append({"case": case, "condition": item["name"], "episode": relative})
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "episodes": episodes}), encoding="utf-8")
        report = validate_driving_manifest(manifest)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["episode_count"], 15)
        self.assertEqual(report["step_count"], 15 * 49)

    def test_cli_writes_running_then_final_and_blocks_source_overwrite(self):
        output = self.root / "report.json"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--episode", str(self.episode), "--output", str(output)]), 0)
        self.assertEqual(read_strict_json(output)["status"], "PASS")
        with self.assertRaises(SystemExit):
            main(["--episode", str(self.episode), "--output", str(self.episode / "episode.json")])
        with patch("scripts.verify_driving.validate_drive_episode", side_effect=KeyboardInterrupt), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--episode", str(self.episode), "--output", str(output)]), 1)
        self.assertEqual(read_strict_json(output)["status"], "FAIL")

    def test_manifest_cli_protects_episode_artifacts_from_report_overwrite(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "episodes": [{"case": "straight", "condition": "center", "episode": "episode"}]}), encoding="utf-8")
        source = self.episode / "steps.jsonl"
        before = source.read_bytes()
        with self.assertRaises(SystemExit):
            main(["--manifest", str(manifest), "--output", str(source)])
        self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
