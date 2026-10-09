"""Camera replay corruption checks and complete source-preserving CPU fixtures."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from carla_tools.baseline import BaselineConfig, OpenCVLanePredictor
from carla_tools.capture import _calibration, _mount_matrix
from carla_tools.config import collection_config_from_dict
from carla_tools.baseline_driving import SCOPE
from carla_tools.baseline_validation import validate_baseline_episode, validate_baseline_manifest
from carla_tools.color import color_provenance
from carla_tools.controller import PurePursuitController, VehicleGeometry, load_controller_config
from carla_tools.evaluation import EvaluationConfig, EvaluationTracker, evaluate_episode, write_metric_csv
from carla_tools.driving import body_lane_margin
from carla_tools.projection import encode_rgb8_png
from scripts.verify_baseline import main

try:
    from tests.test_driving_validation import write_episode
    from tests.test_baseline import scene, np
except ModuleNotFoundError:
    from test_driving_validation import write_episode
    from test_baseline import scene, np


def sha(value):
    return hashlib.sha256(value).hexdigest()


def save_sources(root, metadata, rows):
    trace = ("\n".join(json.dumps(row, sort_keys=True, allow_nan=False) for row in rows) + "\n").encode()
    (root / "steps.jsonl").write_bytes(trace)
    metadata["trace_sha256"] = sha(trace)
    (root / "episode.json").write_text(json.dumps(metadata, sort_keys=True, allow_nan=False), encoding="utf-8")


def write_baseline(root, method="B0", case="straight", *, fail_prediction=False):
    metadata, rows = write_episode(root, case=case, episode_id=case + "__center")
    for path in (root / "images").iterdir():
        path.unlink()
    config = BaselineConfig()
    evaluation_config = replace(EvaluationConfig(), route_completion_tolerance_m=.01, timeout_seconds=10.)
    metadata.update(scope=SCOPE, method=method, route_id=metadata["route_contract"]["route_id"],
                    baseline_config=config.to_dict(), baseline_config_sha256=config.sha256,
                    evaluation_config=evaluation_config.to_dict(), evaluation_config_sha256=evaluation_config.sha256,
                    evaluation_start_reference_m=rows[0]["tracking"]["reference_arc_m"],
                    evaluation_goal_reference_m=19., prediction_inputs=("rgb-and-fixed-nominal-calibration" if method == "B0" else
                                                                      "privileged-original-route-and-same-frame-pose"),
                    perception_uses_route=method != "B0", perception_uses_vehicle_pose=method != "B0")
    metadata["evaluation_route_length_m"] = 19. - metadata["evaluation_start_reference_m"]
    image, _ = scene(curvature={"straight": 0, "left_curve": -1/30, "right_curve": 1/30}[case])
    if fail_prediction:
        image[:] = 50
        image[0, 0] = (3, 2, 1)  # Outside road ROI; retain a red/blue color witness.
        rows = rows[:1]
        metadata["step_count"] = 1
        metadata["drive_end_frame"] = rows[0]["frame"]
        # Reuse native stop schema while keeping the stationary first pose.
        first = rows[0]
        for index, stop in enumerate(metadata["stop_observations"]):
            stop["frame"] = first["frame"] + index + 1
            stop["timestamp"] = first["timestamp"] + (index + 1)*.05
            stop["vehicle"]["transform"] = copy.deepcopy(first["vehicle"]["transform"])
            stop["vehicle"]["velocity"] = {"x": 0., "y": 0., "z": 0.}
            stop["speed_mps"] = 0.
            stop["body_lane_margin_m"] = first["body_lane_margin_m"]
        metadata["post_stop_frame"] = first["frame"] + 12
        metadata["cleanup"]["cleanup_reference_frame"] = first["frame"] + 12
        metadata["cleanup"]["absence_observed_frame"] = first["frame"] + 13
    rgb = image.tobytes()
    bgra = np.empty((360, 640, 4), dtype=np.uint8)
    bgra[:, :, :3] = image[:, :, ::-1]
    bgra[:, :, 3] = 255
    provenance = color_provenance(bgra.tobytes(), rgb, 640, 360)
    png = encode_rgb8_png(rgb, 640, 360)
    predictor = OpenCVLanePredictor(config)
    prediction = predictor.predict(rgb)
    controller = PurePursuitController(load_controller_config(), VehicleGeometry(**metadata["vehicle_geometry"]))
    evaluation = EvaluationTracker(evaluation_config, metadata["evaluation_route_length_m"],
                                   metadata["evaluation_start_reference_m"], metadata["condition"])
    previous = None
    for index, row in enumerate(rows):
        if method == "B0":
            row["perception"] = copy.deepcopy(prediction)
        else:
            row["perception"] = {"status": "PASS", "waypoints_ego_m": copy.deepcopy(row["tracking"]["label"]["waypoints_ego_m"]),
                                 "diagnostics": {"method": method, "input": "privileged-original-route"}, "timing_ms": 0.}
        row["command"] = (controller.step(row["perception"]["waypoints_ego_m"], row["speed_mps"], dt_seconds=.05)
                          if row["perception"]["status"] == "PASS" else None)
        if previous is not None:
            row["vehicle"]["applied_control"] = {key: previous["command"][key] for key in ("throttle", "brake", "steer", "hand_brake", "reverse")}
        row["latency_ms"] = ({"preprocess_ms": 1., "inference_ms": 2., "controller_ms": 1., "pipeline_ms": 4.}
                             if method == "B0" else {"preprocess_ms": 0., "inference_ms": 0., "controller_ms": 1., "pipeline_ms": 1.})
        row["technical_failure"] = row["perception"]["status"] != "PASS"
        row["remaining_horizon_m"] = metadata["route_contract"]["route_length_m"] - row["tracking"]["reference_arc_m"]
        row["evaluation_events"] = []
        row["evaluation_callback_errors"] = []
        row["evaluation"] = evaluation.update(row, technical_failure=row["technical_failure"], remaining_horizon_m=row["remaining_horizon_m"])
        row["termination_reason"] = row["evaluation"]["reason"]
        row["terminal"] = row["evaluation"]["terminated"]
        row["command_target_frame"] = None if row["terminal"] else row["frame"] + 1
        relative = f"images/{row['frame']:08d}.png"
        (root / relative).write_bytes(png)
        row["image"] = {"path": relative, "sha256": sha(png), "width": 640, "height": 360, "color_provenance": provenance}
        previous = row
    metadata["stop_control"]["steer"] = rows[-1]["vehicle"]["applied_control"]["steer"]
    for row in metadata["stop_observations"]:
        row["vehicle"]["applied_control"]["steer"] = metadata["stop_control"]["steer"]
    metadata["termination_reason"] = rows[-1]["termination_reason"]
    metadata["non_driving_events"] = []
    metadata["evaluation_metrics"] = evaluate_episode(metadata, rows, metadata["events"], config=evaluation_config)
    save_sources(root, metadata, rows)
    return metadata, rows


@unittest.skipIf(np is None, "Install requirements-baseline.txt for RGB replay")
class BaselineValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.episode = self.root / "episode"
        self.metadata, self.rows = write_baseline(self.episode)

    def changed(self):
        save_sources(self.episode, self.metadata, self.rows)

    def assert_failed(self, code):
        report = validate_baseline_episode(self.episode)
        self.assertEqual(report["status"], "FAIL", report)
        self.assertEqual(report["errors"][0]["code"], code, report)

    def test_complete_baseline_replays_all_pngs_without_modifying_sources(self):
        before = {path.relative_to(self.episode).as_posix(): path.read_bytes() for path in self.episode.rglob("*") if path.is_file()}
        checked = validate_baseline_episode(self.episode)
        self.assertEqual(checked["status"], "PASS", checked)
        self.assertEqual(checked["metrics"]["image_count"], 49)
        self.assertTrue(checked["evaluation_metrics"]["driving_success"])
        self.assertEqual(before, {path.relative_to(self.episode).as_posix(): path.read_bytes() for path in self.episode.rglob("*") if path.is_file()})

    def test_changed_prediction_and_controller_are_rejected_even_with_updated_trace_hash(self):
        self.rows[2]["perception"]["waypoints_ego_m"][0][1] += .1
        self.changed()
        self.assert_failed("baseline-rgb-prediction-replay-mismatch")
        self.rows[2]["perception"]["waypoints_ego_m"][0][1] -= .1
        self.rows[2]["command"]["steer"] += .1
        self.changed()
        self.assert_failed("baseline-controller-replay-mismatch")

    def test_changed_gt_label_never_becomes_a_baseline_fallback(self):
        original_prediction = copy.deepcopy(self.rows[1]["perception"])
        self.rows[1]["tracking"]["label"]["waypoints_ego_m"][0][1] += 1.
        self.changed()
        self.assert_failed("baseline-original-tracking-replay-mismatch")
        self.assertEqual(self.rows[1]["perception"], original_prediction)

    def test_missing_every_frame_image_corrupted_crc_and_wrong_colour_witness_refused(self):
        original = copy.deepcopy(self.rows[1]["image"])
        self.rows[1]["image"] = None
        self.changed()
        self.assert_failed("baseline-image-record-missing-or-invalid")
        self.rows[1]["image"] = original
        png = self.episode / original["path"]
        data = bytearray(png.read_bytes())
        data[-5] ^= 1
        png.write_bytes(data)
        self.rows[1]["image"]["sha256"] = sha(data)
        self.changed()
        self.assertEqual(validate_baseline_episode(self.episode)["status"], "FAIL")

    def test_wrong_input_contract_unknown_configuration_and_final_metrics_refused(self):
        self.metadata["perception_uses_route"] = True
        self.changed()
        self.assert_failed("baseline-prediction-input-contract-mismatch")
        self.metadata["perception_uses_route"] = False
        self.metadata["evaluation_metrics"]["completion_pct"] = 0.
        self.changed()
        self.assert_failed("baseline-final-evaluation-metrics-mismatch")

    def test_self_consistent_actual_camera_shift_is_rejected_by_fixed_protocol(self):
        self.metadata["collection_config"]["camera"]["mount"]["location"]["z"] += .1
        shifted = collection_config_from_dict(self.metadata["collection_config"])
        self.metadata["camera_calibration"] = _calibration(shifted, _mount_matrix(shifted))
        self.changed()
        self.assert_failed("baseline-unsupported-actual-camera-calibration")

    def test_partial_or_inconsistent_latency_rejected_even_with_recomputed_metrics(self):
        original = copy.deepcopy(self.rows[1]["latency_ms"])
        config = replace(EvaluationConfig(), route_completion_tolerance_m=.01, timeout_seconds=10.)
        for changed, code in (({key: value for key, value in original.items() if key != "controller_ms"}, "baseline-latency-fields-invalid"),
                              ({**original, "pipeline_ms": 4.1}, "baseline-latency-component-sum-mismatch"),
                              ({**original, "controller_ms": -1., "pipeline_ms": 2.}, "baseline-latency-number-invalid")):
            with self.subTest(code=code):
                self.rows[1]["latency_ms"] = changed
                self.metadata["evaluation_metrics"] = evaluate_episode(self.metadata, self.rows, [], config=config)
                self.changed()
                self.assert_failed(code)
        self.rows[1]["latency_ms"] = original

    def test_valid_perception_failure_is_kept_as_failed_metric_row_without_gt_fallback(self):
        root = self.root / "failed"
        metadata, rows = write_baseline(root, fail_prediction=True)
        self.assertIsNone(rows[0]["command"])
        self.assertIsNone(rows[0]["perception"]["waypoints_ego_m"])
        checked = validate_baseline_episode(root)
        self.assertEqual(checked["status"], "PASS", checked)
        self.assertFalse(checked["evaluation_metrics"]["driving_success"])
        self.assertEqual(checked["evaluation_metrics"]["termination_reason"], "technical_failure")
        self.assertTrue(checked["evaluation_metrics"]["integrity_valid"])

    def test_stop_with_departure_margin_is_valid_but_false_margin_and_controls_are_rejected(self):
        failed = self.root / "failed-stop"
        metadata, rows = write_baseline(failed, fail_prediction=True)
        for observation in metadata["stop_observations"]:
            pose = observation["vehicle"]["transform"]
            pose["location"]["y"] = 3.
            pose["matrix"][1][3] = 3.
            observation["body_lane_margin_m"] = body_lane_margin(pose, {}, metadata["bounding_box"], metadata["route_contract"])
        save_sources(failed, metadata, rows)
        checked = validate_baseline_episode(failed)
        self.assertEqual(checked["status"], "PASS", checked)
        self.assertLess(checked["metrics"]["minimum_stopping_body_lane_margin_m"], 0.)
        self.metadata["stop_observations"][0]["body_lane_margin_m"] = -99.
        self.changed()
        self.assert_failed("baseline-stop-body-margin-mismatch")

    def test_source_changed_during_rgb_replay_cannot_publish_pass(self):
        original = OpenCVLanePredictor.predict
        changed = False

        def mutate(predictor, *args, **kwargs):
            nonlocal changed
            result = original(predictor, *args, **kwargs)
            if not changed:
                path = self.episode / "episode.json"
                path.write_bytes(path.read_bytes() + b"\n")
                changed = True
            return result

        with patch.object(OpenCVLanePredictor, "predict", mutate):
            self.assert_failed("baseline-source-changed-during-validation")

    def test_cli_preserves_existing_output_and_all_source_artifacts(self):
        before = (self.episode / "episode.json").read_bytes()
        with self.assertRaises(SystemExit):
            main(["--episode", str(self.episode), "--output", str(self.episode / "episode.json")])
        self.assertEqual((self.episode / "episode.json").read_bytes(), before)
        output = self.root / "existing.json"
        output.write_bytes(b"preserve")
        with self.assertRaises(SystemExit):
            main(["--episode", str(self.episode), "--output", str(output)])
        self.assertEqual(output.read_bytes(), b"preserve")
        with redirect_stdout(io.StringIO()):
            status = main(["--episode", str(self.episode), "--output", str(self.root / "fresh.json")])
        self.assertEqual(status, 0)


@unittest.skipIf(np is None, "Install requirements-baseline.txt for RGB replay")
class BaselineManifestTests(unittest.TestCase):
    def test_both_methods_all_cases_keep_failed_baseline_metric_rows_and_shared_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episodes = []
            for method in ("B0", "privileged_reference"):
                for case in ("straight", "left_curve", "right_curve"):
                    relative = f"episodes/{method}/{case}__center"
                    directory = root / relative
                    directory.parent.mkdir(parents=True, exist_ok=True)
                    write_baseline(directory, method=method, case=case, fail_prediction=method == "B0")
                    episodes.append({"method": method, "case": case, "condition": "center", "episode": relative})
            manifest = root / "baseline_manifest.json"
            data = {"schema_version": 1, "scope": SCOPE, "episodes": episodes}
            manifest.write_text(json.dumps(data), encoding="utf-8")
            report = validate_baseline_manifest(manifest)
            self.assertEqual(report["status"], "PASS", report)
            self.assertEqual(report["episode_count"], 6)
            self.assertEqual(len(report["episode_metrics"]), 6)
            self.assertEqual(sum(not row["driving_success"] for row in report["episode_metrics"]), 3)
            self.assertEqual(report["aggregate"]["episode_count"], 6)
            csv_path, json_path = root / "metrics.csv", root / "metrics.json"
            aggregate = write_metric_csv(report["episode_metrics"], csv_path)
            saved = {"schema_version": 1, "episodes": report["episode_metrics"], "aggregate": aggregate}
            json_path.write_text(json.dumps(saved), encoding="utf-8")
            self.assertEqual(validate_baseline_manifest(manifest)["status"], "PASS")
            csv_bytes = csv_path.read_bytes()
            csv_path.write_bytes(csv_bytes.replace(b"technical_failure", b"route_completed", 1))
            changed_csv = validate_baseline_manifest(manifest)
            self.assertEqual(changed_csv["status"], "FAIL")
            self.assertEqual(changed_csv["errors"][0]["code"], "baseline-metric-csv-replay-mismatch")
            csv_path.write_bytes(csv_bytes)
            saved["aggregate"]["episode_count"] = 5
            json_path.write_text(json.dumps(saved), encoding="utf-8")
            changed_json = validate_baseline_manifest(manifest)
            self.assertEqual(changed_json["status"], "FAIL")
            self.assertEqual(changed_json["errors"][0]["code"], "baseline-metric-json-replay-mismatch")
            data["episodes"][0]["episode"] = "../outside"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            escaped = validate_baseline_manifest(manifest)
            self.assertEqual(escaped["status"], "FAIL")
            self.assertEqual(escaped["errors"][0]["code"], "baseline-manifest-episode-path-invalid")


if __name__ == "__main__":
    unittest.main()
