"""Real CPU checkpoint and selected-controller serving contract regressions."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from carla_tools.controller import VehicleGeometry, load_controller_config
from carla_tools.controller_study import ControllerStudyConfig, build_controller
from carla_tools.learning_data import load_learning_config
from carla_tools.projection import encode_rgb8_png
from carla_tools.refined_training import _contract, refined_source_hashes
from carla_tools.refined_waypoint import RefinedWaypointModel
from carla_tools.selected_pipeline import (SelectedPipelineError, build_pipeline_selection,
    load_selected_pipeline, measured_geometry_from_dict, pipeline_id, selected_pipeline_source_hashes)
from carla_tools.waypoint_model import PRETRAINED_URL
from scripts.predict_selected_pipeline import _write_result, main


def measured_geometry():
    return VehicleGeometry(2.8, -1.4, 1.4, 35., ((0., 1.), (100., 1.)), 1.6)


def geometry_evidence():
    pose = {"location": {"x": 0., "y": 0., "z": 0.}, "rotation": {"pitch": 0., "yaw": 0., "roll": 0.},
        "matrix": [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]}
    wheels = [{"position_cm": {"x": x, "y": y, "z": -10.}, "max_steer_angle_deg": 35. if index < 2 else 0.}
        for index, (x, y) in enumerate(((140., -80.), (140., 80.), (-140., -80.), (-140., 80.)))]
    return {"vehicle_geometry": measured_geometry().to_dict(), "geometry_evidence": {
        "measurement_vehicle_transform": pose, "wheels": wheels, "original_steering_curve": [[0., 1.], [100., .5]],
        "applied_steering_curve": [[0., 1.], [100., 1.]], "position_units": "world-centimetres",
        "measurement_after_warmup": True, "steering_normalization": "measured-front-steer-limit-times-constant-factor"}}


class SelectedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.fixture_root = Path(cls.temporary.name)
        cls.fixture = cls.fixture_root / "checkpoint.pt"
        config = load_learning_config()
        model = RefinedWaypointModel(variant="hybrid_geometry", config=config, pretrained=False)
        with torch.no_grad():
            model.head.output[-1].weight.zero_()
            model.head.output[-1].bias.zero_()
        contract, contract_sha = _contract("hybrid_geometry", config)
        torch.save({"schema_version": 1, "checkpoint_role": "selected-refined-model", "variant": "hybrid_geometry",
            "epoch": 1, "config": config.to_dict(), "config_sha256": config.sha256,
            "model_contract": contract, "model_contract_sha256": contract_sha, "source_sha256": refined_source_hashes(),
            "manifest_sha256": "a" * 64, "model_state": model.state_dict(),
            "pretrained_provenance": {"pretrained": True, "hash_verified": True, "url": PRETRAINED_URL,
                "sha256": "f37072fd" + "0" * 56}}, cls.fixture)
        del model
        cls.selection = build_pipeline_selection(cls.fixture, ControllerStudyConfig(base=load_controller_config()).to_dict(),
            base_directory=cls.fixture_root, provenance={"scope": "synthetic-CPU-serving-fixture-not-driving-evidence"})

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.case = tempfile.TemporaryDirectory()
        self.addCleanup(self.case.cleanup)
        self.root = Path(self.case.name)
        self.checkpoint = self.root / "checkpoint.pt"
        shutil.copyfile(self.fixture, self.checkpoint)
        self.pointer = self.root / "pipeline.json"
        self.value = copy.deepcopy(self.selection)
        self.save()
        self.rgb = np.full((360, 640, 3), 50, dtype=np.uint8)
        self.rgb[0, 0] = [3, 2, 1]
        self.image = self.root / "image.png"
        self.image.write_bytes(encode_rgb8_png(self.rgb.tobytes(), 640, 360))
        self.geometry = self.root / "geometry.json"
        self.geometry.write_text(json.dumps(geometry_evidence()))

    def save(self):
        self.pointer.write_text(json.dumps(self.value), encoding="utf-8")

    def load(self):
        return load_selected_pipeline(self.pointer, measured_geometry())

    def cli(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            code = main(["--selection", str(self.pointer), "--geometry", str(self.geometry), *extra])
        return code, captured.getvalue()

    def test_real_checkpoint_load_predict_and_exact_selected_PP_control(self):
        pipeline = self.load()
        result = pipeline.step(self.rgb, 6.5)
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["prediction"]["learned_waypoints_ego_m"], [[float(x), 0.] for x in (3, 6, 9, 12, 15)])
        independent = build_controller(ControllerStudyConfig(base=load_controller_config()), measured_geometry())
        expected = independent.step(result["prediction"]["waypoints_ego_m"], 6.5, dt_seconds=.05)
        self.assertEqual(result["command"], expected)
        self.assertEqual(pipeline.identity["model"]["device"], "cpu")
        self.assertEqual(pipeline.verify_bindings()["status"], "PASS")

    def test_guarded_pointer_loads_real_model_runs_bound_fusion_and_selected_control(self):
        from carla_tools.guarded_path_correction import guarded_preprocessing_config
        self.value = build_pipeline_selection(self.checkpoint, self.value["controller_config"],
            base_directory=self.root, preprocessing_config=guarded_preprocessing_config())
        self.save()
        pipeline = self.load()
        self.assertEqual(pipeline.predictor.correction, "guarded_path")
        result = pipeline.step(self.rgb, 6.5)
        self.assertEqual(result["status"], "PASS", result)
        self.assertIn("guarded_path_correction", result["prediction"]["diagnostics"])
        expected = build_controller(ControllerStudyConfig(base=load_controller_config()), measured_geometry()).step(
            result["prediction"]["waypoints_ego_m"], 6.5, dt_seconds=.05)
        self.assertEqual(result["command"], expected)
        self.assertEqual(pipeline.identity["preprocessing"], self.value["preprocessing"])
        self.assertEqual(pipeline.verify_bindings()["status"], "PASS")
        code, _ = self.cli("--image", str(self.image), "--speed-mps", "6.5", "--output", str(self.root / "guarded-cli.json"))
        self.assertEqual(code, 0)
        observed = json.loads((self.root / "guarded-cli.json").read_text())
        self.assertEqual(observed["pipeline"]["preprocessing"], self.value["preprocessing"])
        self.value["preprocessing"]["guard_config_sha256"] = "0" * 64
        self.value["pipeline_id"] = pipeline_id(self.value); self.save()
        with self.assertRaises(ValueError):
            self.load()

    def test_Stanley_controller_bound_in_pointer_is_actually_used(self):
        config = ControllerStudyConfig(base=load_controller_config(), method="stanley", stanley_cross_track_gain=.6)
        self.value["controller_config"] = config.to_dict()
        self.value["controller_config_sha256"] = config.sha256
        self.value["pipeline_id"] = pipeline_id(self.value); self.save()
        pipeline = self.load(); result = pipeline.step(self.rgb, 6.5)
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["command"]["diagnostics"]["controller_method"], "stanley")
        expected = build_controller(config, measured_geometry()).step(result["prediction"]["waypoints_ego_m"], 6.5)
        self.assertEqual(result["command"], expected)

    def test_controller_history_preserved_and_explicit_reset_starts_fresh(self):
        pipeline = self.load()
        first, second = pipeline.step(self.rgb, 6.5), pipeline.step(self.rgb, 6.5)
        self.assertEqual([first["command"]["diagnostics"]["step_index"], second["command"]["diagnostics"]["step_index"]], [0, 1])
        self.assertGreater(second["command"]["throttle"], first["command"]["throttle"])
        pipeline.reset(); reset = pipeline.step(self.rgb, 6.5)
        self.assertEqual(reset["command"], first["command"])
        self.assertEqual(reset["sequence"]["frame_index"], 0)
        self.assertEqual(reset["sequence"]["reset_count"], 1)

    def test_actual_typed_controller_rejection_keeps_paths_and_requires_reset(self):
        pipeline = self.load()
        original = pipeline.predictor.raw.model.forward
        pipeline.predictor.raw.model.forward = lambda image: image.new_tensor([
            [[3., 0.], [1., 3.], [.1, 4.], [.1, 6.], [.1, 9.]]])
        result = pipeline.step(self.rgb, 0.)
        self.assertEqual(result["status"], "FAIL")
        self.assertIsNone(result["command"])
        self.assertEqual(result["failure"]["code"], "controller-waypoints-not-ordered-forward")
        self.assertEqual(result["timing"]["failed_stages"], ["controller_ms"])
        self.assertEqual(len(result["prediction"]["controller_rejection"]["waypoints_ego_m"]), 5)
        with self.assertRaisesRegex(SelectedPipelineError, "reset-required"):
            pipeline.step(self.rgb, 0.)
        pipeline.predictor.raw.model.forward = original
        pipeline.reset()
        self.assertEqual(pipeline.step(self.rgb, 0.)["status"], "PASS")

    def test_invalid_speed_RGB_or_timestep_rejected_before_inference(self):
        pipeline = self.load()
        with patch.object(pipeline.predictor, "predict_profiled") as predict:
            for rgb, speed, dt in ((self.rgb, True, .05), (self.rgb, float("nan"), .05),
                    (self.rgb.astype(np.float32), 1., .05), (self.rgb[:, :320], 1., .05), (self.rgb, 1., .1)):
                with self.assertRaises(SelectedPipelineError):
                    pipeline.step(rgb, speed, dt_seconds=dt)
            predict.assert_not_called()

    def test_model_source_controller_correction_and_pipeline_tampering_rejected(self):
        changes = [lambda value: value["model"].update(checkpoint_sha256="0" * 64),
            lambda value: value["controller_config"].update(steer_smoothing_alpha=.5),
            lambda value: value["preprocessing"].update(correction="near"),
            lambda value: value["source_sha256"].update({"carla_tools/controller_study.py": "0" * 64}),
            lambda value: value.update(pipeline_id="0" * 64),
            lambda value: value["model"].update(architecture="B0")]
        for change in changes:
            with self.subTest(change=change):
                self.value = copy.deepcopy(self.selection); change(self.value); self.save()
                with self.assertRaises(ValueError):
                    self.load()

    def test_valid_fingerprint_cannot_hide_wrong_learning_or_correction_contract(self):
        for key, target in (("learning_config_sha256", "model"), ("model_contract_sha256", "model"),
                ("correction_config_sha256", "preprocessing")):
            self.value = copy.deepcopy(self.selection)
            self.value[target][key] = "0" * 64
            self.value["pipeline_id"] = pipeline_id(self.value); self.save()
            with self.assertRaises(ValueError):
                self.load()

    def test_checkpoint_actual_bytes_checked_before_deserialization(self):
        self.checkpoint.write_bytes(b"different-bytes")
        with patch("carla_tools.runtime_driving.RuntimeCheckpointPredictor") as load:
            with self.assertRaisesRegex(SelectedPipelineError, "checkpoint-hash"):
                self.load()
            load.assert_not_called()

    def test_relative_path_escape_and_links_rejected(self):
        for relative in ("../checkpoint.pt", "C:/checkpoint.pt", "nested\\checkpoint.pt", "/checkpoint.pt", "a//checkpoint.pt"):
            self.value = copy.deepcopy(self.selection); self.value["model"]["checkpoint_path"] = relative
            self.value["pipeline_id"] = pipeline_id(self.value); self.save()
            with self.assertRaises(ValueError):
                self.load()
        self.value = copy.deepcopy(self.selection); self.save()
        original = Path.is_symlink
        with patch.object(Path, "is_symlink", lambda path: path == self.checkpoint or original(path)):
            with self.assertRaises(ValueError):
                self.load()

    def test_expected_identity_and_full_geometry_are_required(self):
        with self.assertRaisesRegex(SelectedPipelineError, "unexpected-identity"):
            load_selected_pipeline(self.pointer, measured_geometry(), expected_pipeline_id="0" * 64)
        incomplete = measured_geometry().to_dict(); incomplete.pop("rear_axle_y_m")
        with self.assertRaisesRegex(SelectedPipelineError, "complete-measured-geometry"):
            measured_geometry_from_dict(incomplete)
        wrong = measured_geometry().to_dict(); wrong["wheelbase_m"] = 3.
        with self.assertRaises(ValueError):
            measured_geometry_from_dict(wrong)

    def test_pointer_and_checkpoint_changes_during_stream_rejected(self):
        pipeline = self.load()
        self.value["provenance"]["extra"] = "changed-after-load"; self.save()
        with self.assertRaisesRegex(SelectedPipelineError, "binding-changed"):
            pipeline.verify_bindings()

    def test_public_selection_snapshot_cannot_mutate_loaded_runtime_binding(self):
        pipeline = self.load()
        exposed = pipeline.selection
        exposed["pipeline_id"] = "0" * 64
        exposed["controller_config_sha256"] = "0" * 64
        self.assertEqual(pipeline.step(self.rgb, 6.5)["status"], "PASS")
        self.assertEqual(pipeline.identity["pipeline_id"], self.selection["pipeline_id"])

    def test_source_or_geometry_changes_are_rejected_without_touching_sources(self):
        pipeline = self.load()
        with patch("carla_tools.selected_pipeline.selected_pipeline_source_hashes", return_value={}):
            with self.assertRaisesRegex(SelectedPipelineError, "binding-changed"):
                pipeline.verify_bindings()
        other = measured_geometry().to_dict(); other["max_steer_angle_deg"] = 40.
        pipeline.controller.geometry = VehicleGeometry(**other)
        with self.assertRaisesRegex(SelectedPipelineError, "controller-or-geometry-changed"):
            pipeline.step(self.rgb, 6.5)
        self.value = copy.deepcopy(self.selection); self.save(); pipeline = self.load()
        with self.checkpoint.open("ab") as stream:
            stream.write(b"changed-after-load")
        with self.assertRaisesRegex(SelectedPipelineError, "binding-changed"):
            pipeline.verify_bindings()

    def test_builder_checks_actual_checkpoint_and_never_publishes_or_uses_GPU(self):
        value = build_pipeline_selection(self.checkpoint, ControllerStudyConfig().to_dict(), base_directory=self.root)
        self.assertEqual(value["pipeline_id"], self.selection["pipeline_id"])
        self.assertEqual(value["source_sha256"], selected_pipeline_source_hashes())
        with self.assertRaisesRegex(SelectedPipelineError, "checkpoint-hash"):
            build_pipeline_selection(self.checkpoint, ControllerStudyConfig().to_dict(), base_directory=self.root,
                expected_checkpoint_sha256="0" * 64)

    def test_single_image_CLI_produces_waypoints_and_selected_control(self):
        output = self.root / "single.json"
        code, detail = self.cli("--image", str(self.image), "--speed-mps", "6.5", "--output", str(output))
        self.assertEqual(code, 0, detail)
        result = json.loads(output.read_text())
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["processed_frame_count"], 1)
        self.assertIsNotNone(result["frames"][0]["command"])
        self.assertTrue(result["geometry"]["native_geometry_reconstructed"])
        self.assertFalse(result["closed_loop_executed"])

    def test_sequence_CLI_preserves_control_history_and_records_explicit_reset(self):
        frames = self.root / "frames.jsonl"
        rows = [{"frame": frame, "timestamp": timestamp, "image": "image.png", "speed_mps": 6.5,
            "reset_before": reset} for frame, timestamp, reset in ((100, 5., False), (101, 5.05, False), (200, 10., True))]
        frames.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        output = self.root / "sequence.json"
        code, detail = self.cli("--frames", str(frames), "--output", str(output))
        self.assertEqual(code, 0, detail)
        result = json.loads(output.read_text())
        self.assertEqual([row["command"]["diagnostics"]["step_index"] for row in result["frames"]], [0, 1, 0])
        self.assertEqual([row["sequence"]["reset_count"] for row in result["frames"]], [0, 0, 1])
        self.assertTrue(result["all_input_frames_processed"])

    def test_CLI_rejects_time_gaps_guessed_geometry_missing_image_and_overwrite(self):
        frames = self.root / "frames.jsonl"
        rows = [{"frame": frame, "timestamp": timestamp, "image": "image.png", "speed_mps": 1.}
            for frame, timestamp in ((100, 5.), (102, 5.1))]
        frames.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        output = self.root / "bad.json"
        self.assertEqual(self.cli("--frames", str(frames), "--output", str(output))[0], 2)
        self.assertFalse(output.exists())
        self.geometry.write_text(json.dumps({"vehicle_geometry": measured_geometry().to_dict()}))
        self.assertEqual(self.cli("--image", str(self.image), "--speed-mps", "1", "--output", str(output))[0], 2)
        self.assertFalse(output.exists())
        self.geometry.write_text(json.dumps(geometry_evidence()))
        self.assertEqual(self.cli("--image", str(self.root / "missing.png"), "--speed-mps", "1", "--output", str(output))[0], 2)
        output.write_bytes(b"preserve-existing-result")
        self.assertEqual(self.cli("--image", str(self.image), "--speed-mps", "1", "--output", str(output))[0], 2)
        self.assertEqual(output.read_bytes(), b"preserve-existing-result")

    def test_result_publication_preserves_a_destination_created_at_last_moment(self):
        output = self.root / "race.json"
        def race(source, destination):
            Path(destination).write_bytes(b"racing-result")
            raise FileExistsError("simulated-race")
        operation = "os.rename" if os.name == "nt" else "os.link"
        with patch(operation, side_effect=race):
            with self.assertRaisesRegex(SelectedPipelineError, "output-already-exists"):
                _write_result(output, {"status": "PASS"})
        self.assertEqual(output.read_bytes(), b"racing-result")
        self.assertFalse(list(self.root.glob(".selected-pipeline-*.pending")))

    def test_sequence_stops_on_first_typed_failure_and_publishes_failure(self):
        pipeline = self.load()
        pipeline.predictor.raw.model.forward = lambda image: image.new_tensor([
            [[3., 0.], [1., 3.], [.1, 4.], [.1, 6.], [.1, 9.]]])
        frames = self.root / "failure_frames.jsonl"
        frames.write_text("\n".join(json.dumps({"frame": 100 + index, "timestamp": 5. + .05 * index,
            "image": "image.png", "speed_mps": 0.}) for index in range(2)) + "\n")
        output = self.root / "failure.json"
        with patch("carla_tools.selected_pipeline.load_selected_pipeline", return_value=pipeline):
            code, detail = self.cli("--frames", str(frames), "--output", str(output))
        self.assertEqual(code, 1, detail)
        result = json.loads(output.read_text())
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["processed_frame_count"], 1)
        self.assertEqual(result["input_frame_count"], 2)
        self.assertFalse(result["all_input_frames_processed"])
        self.assertIsNone(result["frames"][0]["command"])
        self.assertIn("controller_rejection", result["frames"][0]["prediction"])

    def test_programming_exception_is_not_converted_into_a_driving_rejection(self):
        pipeline = self.load()
        with patch.object(pipeline.controller, "step", side_effect=RuntimeError("fixture-programming-exception")):
            with self.assertRaisesRegex(RuntimeError, "programming-exception"):
                pipeline.step(self.rgb, 0.)


if __name__ == "__main__":
    unittest.main()
