"""Independent synthetic checkpoints and PNGs verify serving boundaries.

Weights are explicitly synthetic. Provenance fields below exercise the loader
schema and make no claim about actual pretrained training or performance.
"""

import copy
import hashlib
import importlib.util
import io
import json
import os
import shutil
import struct
import tempfile
import unittest
import zlib
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import carla_tools.inference as serving
from carla_tools.contracts import canonical_sha256
from carla_tools.learning_data import load_learning_config
from carla_tools.training import runtime_source_hashes
from carla_tools.waypoint_model import PRETRAINED_URL, create_model, model_contract

POINTS = [[3.0, 0.1], [6.0, 0.3], [9.0, 0.7], [11.8, 1.2], [14.3, 2.0]]
MANIFEST_SHA = "1" * 64


class OfflineInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.shared = tempfile.TemporaryDirectory()
        cls.fixture_path = Path(cls.shared.name) / "fixture.pt"
        config = load_learning_config()
        model = create_model(config=config, pretrained=False)
        with torch.no_grad():
            model.head[-1].bias.copy_(((torch.tensor(POINTS) - model.straight_prior) / 15).reshape(-1))
        contract = model_contract(config)
        cls.fixture = {
            "schema_version": 1, "checkpoint_role": "selected-model", "epoch": 14,
            "config": config.to_dict(), "config_sha256": config.sha256,
            "model_contract": contract, "model_contract_sha256": canonical_sha256(contract),
            "manifest_sha256": MANIFEST_SHA, "source_sha256": runtime_source_hashes(),
            "model_state": model.state_dict(), "best_dev_loss": 0.02,
            "history": [{"epoch": 13, "dev_loss": 0.03, "selected": True},
                        {"epoch": 14, "dev_loss": 0.02, "selected": True}],
            "pretrained_provenance": {"pretrained": True, "hash_verified": True,
                "weights_enum": "ResNet18_Weights.IMAGENET1K_V1", "url": PRETRAINED_URL,
                "sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"},
        }
        torch.save(cls.fixture, cls.fixture_path)

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "selected.pt"
        shutil.copyfile(self.fixture_path, self.checkpoint)
        self.rgb = np.full((360, 640, 3), (21, 42, 84), dtype=np.uint8)
        self.rgb[220, 125] = (200, 30, 10)
        self.image = self.root / "rgb.png"
        Image.fromarray(self.rgb).save(self.image)

    def predictor(self, **kwargs):
        return serving.OfflineWaypointPredictor(self.checkpoint, **kwargs)

    def mutate_checkpoint(self, mutate):
        value = dict(self.fixture)
        for key in ("config", "model_contract", "source_sha256", "pretrained_provenance", "history"):
            value[key] = copy.deepcopy(value[key])
        value["model_state"] = dict(value["model_state"])
        mutate(value)
        torch.save(value, self.checkpoint)

    def cli(self):
        path = Path(__file__).resolve().parents[1] / "scripts/predict_waypoints.py"
        spec = importlib.util.spec_from_file_location("predict_waypoints_fixture", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_actual_selected_loader_and_output_metres_without_extra_scale(self):
        predictor = self.predictor(expected_checkpoint_sha256=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest().upper(),
                                   expected_manifest_sha256=MANIFEST_SHA)
        result = predictor.predict_rgb(self.rgb)
        self.assertEqual(result["status"], "PASS")
        np.testing.assert_allclose(result["waypoints_ego_m"], POINTS, atol=2e-6, rtol=0)
        self.assertEqual(result["diagnostics"]["selected_predictor"], "ResNet18")
        summary = predictor.summary
        self.assertEqual(summary["checkpoint_role"], "selected-model")
        self.assertEqual(summary["checkpoint_epoch"], 14)
        self.assertEqual(summary["selected_epoch"], 14)
        self.assertEqual(summary["manifest_sha256"], MANIFEST_SHA)
        self.assertEqual(summary["runtime"]["device"], "cpu")
        self.assertEqual(summary["model_contract"]["output"]["origin"], "vehicle-actor-origin")
        self.assertEqual(summary["model_contract"]["output"]["units"], "meters")
        self.assertEqual(summary["source_sha256"], runtime_source_hashes())
        self.assertEqual(set(summary["inference_source_sha256"]), {"carla_tools/inference.py"})

    def test_png_decoding_keeps_original_rgb_values_and_records_consumed_hash(self):
        rgb, info = serving.read_rgb_png(self.image)
        np.testing.assert_array_equal(rgb, self.rgb)
        result = self.predictor().predict_png(self.image)
        self.assertEqual(result["status"], "PASS")
        expected = hashlib.sha256(self.image.read_bytes()).hexdigest()
        self.assertEqual(info["sha256"], expected)
        self.assertEqual(result["input_sha256"], expected)
        self.assertEqual(result["input"]["sha256"], expected)
        self.assertEqual(result["input"]["mode"], "RGB")
        self.assertGreaterEqual(result["total_timing_ms"], result["timing_ms"])

    def test_each_batch_item_is_independent_with_no_stale_path_or_alias(self):
        predictor = self.predictor()
        results = predictor.predict_rgb_batch([self.rgb, None, self.rgb])
        self.assertEqual([item["status"] for item in results], ["PASS", "FAIL", "PASS"])
        self.assertIsNone(results[1]["waypoints_ego_m"])
        results[0]["waypoints_ego_m"][0][0] = -100
        np.testing.assert_allclose(results[2]["waypoints_ego_m"], POINTS, atol=2e-6, rtol=0)
        summary = predictor.summary
        summary["model_contract"]["output"]["units"] = "wrong"
        self.assertEqual(predictor.summary["model_contract"]["output"]["units"], "meters")

    def test_only_original_rgb_reaches_eval_and_inference_mode(self):
        predictor = self.predictor()
        predictor.model.train()
        def observe(rgb):
            self.assertIs(rgb, self.rgb)
            self.assertFalse(predictor.model.training)
            self.assertTrue(torch.is_inference_mode_enabled())
            return {"status": "PASS", "waypoints_ego_m": POINTS, "diagnostics": {}}
        with patch.object(predictor.model, "predict_rgb", side_effect=observe) as call:
            self.assertEqual(predictor.predict_rgb(self.rgb)["status"], "PASS")
            self.assertEqual(call.call_args.kwargs, {})
            self.assertEqual(len(call.call_args.args), 1)

    def test_invalid_rgb_never_reaches_model(self):
        predictor = self.predictor()
        with patch.object(predictor.model, "predict_rgb") as call:
            for value in (None, self.rgb.astype(np.float32), self.rgb[:180], self.rgb[:, :, 0], np.zeros((360, 640, 4), dtype=np.uint8)):
                with self.subTest(shape=getattr(value, "shape", None)):
                    result = predictor.predict_rgb(value)
                    self.assertEqual(result["status"], "FAIL")
                    self.assertIsNone(result["waypoints_ego_m"])
            call.assert_not_called()

    def test_failed_and_malformed_pass_results_never_return_a_path(self):
        predictor = self.predictor()
        for value in ({"status": "FAIL", "error": "private-token"}, None,
                      {"status": "PASS", "waypoints_ego_m": [[3.0, 0.0]]*5},
                      {"status": "PASS", "waypoints_ego_m": [[3.0, float("nan")]]*5}):
            with self.subTest(result=value), patch.object(predictor.model, "predict_rgb", return_value=value):
                result = predictor.predict_rgb(self.rgb)
                self.assertEqual(result["status"], "FAIL")
                self.assertIsNone(result["waypoints_ego_m"])
                self.assertNotIn("private-token", result["error"])

    def test_assisted_mode_is_explicit_and_binds_its_selection_source(self):
        predictor = self.predictor(mode="assisted")
        self.assertEqual(set(predictor.summary["inference_source_sha256"]),
                         {"carla_tools/inference.py", "carla_tools/rgb_predictor.py"})
        with patch.object(predictor._predictor.baseline, "predict", return_value={"status": "PASS", "waypoints_ego_m": POINTS}), \
                patch.object(predictor.model, "predict_rgb") as learned:
            result = predictor.predict_rgb(self.rgb)
            self.assertEqual(result["diagnostics"]["mode"], "assisted")
            self.assertEqual(result["diagnostics"]["selected_predictor"], "B0")
            learned.assert_not_called()
        with patch.object(predictor._predictor.baseline, "predict", return_value={"status": "FAIL"}):
            result = predictor.predict_rgb(self.rgb)
            self.assertEqual(result["diagnostics"]["selected_predictor"], "ResNet18")
            np.testing.assert_allclose(result["waypoints_ego_m"], POINTS, atol=2e-6, rtol=0)

    def test_hash_syntax_and_checkpoint_identity_reject_before_deserialization(self):
        for kwargs in ({"expected_checkpoint_sha256": "0"*64}, {"expected_checkpoint_sha256": "bad"},
                       {"expected_manifest_sha256": "g"*64}, {"expected_manifest_sha256": 123}):
            with self.subTest(kwargs=kwargs), patch.object(serving, "load_checkpoint") as loader:
                with self.assertRaises(serving.InferenceError):
                    self.predictor(**kwargs)
                loader.assert_not_called()

    def test_dataset_identity_mismatch_is_rejected_by_strict_loader(self):
        with self.assertRaisesRegex(serving.InferenceError, "resume-dataset-mismatch"):
            self.predictor(expected_manifest_sha256="2"*64)

    def test_source_config_contract_and_pretrained_tampering_are_rejected(self):
        mutations = (
            (lambda value: value["source_sha256"].update({"carla_tools/learning_data.py": "0"*64}), "resume-source-mismatch"),
            (lambda value: value["config"].update(head_dropout=0.25), "checkpoint-config-hash-mismatch"),
            (lambda value: value["model_contract"]["output"].update(origin="rear-axle"), "checkpoint-model-contract-mismatch"),
            (lambda value: value["pretrained_provenance"].update(hash_verified=False), "checkpoint-pretrained-provenance-invalid"),
        )
        for mutate, code in mutations:
            with self.subTest(code=code):
                self.mutate_checkpoint(mutate)
                with self.assertRaisesRegex(serving.InferenceError, code):
                    self.predictor()

    def test_training_state_is_not_a_selected_serving_checkpoint(self):
        self.mutate_checkpoint(lambda value: value.update(checkpoint_role="training-state"))
        with self.assertRaisesRegex(serving.InferenceError, "requires-selected-checkpoint"):
            self.predictor()

    def test_resumed_save_epoch_is_distinct_from_history_selected_epoch(self):
        def resume(value):
            value["epoch"] = 17
            value["history"].append({"epoch": 17, "dev_loss": 0.04, "selected": False})
        self.mutate_checkpoint(resume)
        summary = self.predictor().summary
        self.assertEqual(summary["checkpoint_epoch"], 17)
        self.assertEqual(summary["selected_epoch"], 14)

    def test_incomplete_or_inconsistent_history_does_not_invent_selected_epoch(self):
        for mutate in (lambda value: value.update(history=[]), lambda value: value.update(best_dev_loss=0.01),
                       lambda value: value["history"][-1].update(selected=False)):
            self.mutate_checkpoint(mutate)
            self.assertIsNone(self.predictor().summary["selected_epoch"])

    def test_checkpoint_and_core_source_changes_during_load_are_rejected(self):
        original = serving.load_checkpoint
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            with self.checkpoint.open("ab") as stream:
                stream.write(b"changed-after-load")
            return result
        with patch.object(serving, "load_checkpoint", side_effect=changed):
            with self.assertRaisesRegex(serving.InferenceError, "checkpoint-changed-during-load"):
                self.predictor()
        shutil.copyfile(self.fixture_path, self.checkpoint)
        actual = runtime_source_hashes()
        altered = {**actual, "carla_tools/training.py": "0"*64}
        with patch.object(serving, "runtime_source_hashes", side_effect=[actual, altered]):
            with self.assertRaisesRegex(serving.InferenceError, "source-changed-during-load"):
                self.predictor()

    def test_corrupt_checkpoint_invalid_device_and_mode_have_stable_errors(self):
        for kwargs, code in (({"device": "mps"}, "invalid-device"), ({"mode": "oracle"}, "invalid-mode")):
            with self.assertRaisesRegex(serving.InferenceError, code):
                self.predictor(**kwargs)
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(serving.InferenceError, "cuda-unavailable"):
                self.predictor(device="cuda")
        self.checkpoint.write_bytes(b"not-a-checkpoint")
        with self.assertRaisesRegex(serving.InferenceError, "checkpoint-load-failed"):
            self.predictor()

    def test_wrong_size_alpha_grayscale_palette_and_16bit_png_are_rejected(self):
        predictor = self.predictor()
        sources = []
        for mode, size in (("RGB", (320, 180)), ("RGBA", (640, 360)), ("L", (640, 360)), ("P", (640, 360))):
            path = self.root / f"{mode}-{size[0]}.png"
            Image.new(mode, size).save(path)
            sources.append(path)
        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind+data) & 0xFFFFFFFF)
        header = struct.pack(">IIBBBBB", 640, 360, 16, 2, 0, 0, 0)
        row = b"\x00" + struct.pack(">HHH", 1000, 2000, 3000)*640
        rgb16 = self.root / "rgb16.png"
        rgb16.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                          + chunk(b"IDAT", zlib.compress(row*360)) + chunk(b"IEND", b""))
        sources.append(rgb16)
        with patch.object(predictor.model, "predict_rgb") as call:
            for source in sources:
                with self.subTest(source=source.name):
                    result = predictor.predict_png(source)
                    self.assertEqual(result["status"], "FAIL")
                    self.assertIsNone(result["waypoints_ego_m"])
            call.assert_not_called()

    def test_crc_truncation_trailing_bytes_animation_and_non_png_are_rejected(self):
        encoded = self.image.read_bytes()
        bad_crc = bytearray(encoded)
        bad_crc[29] ^= 1
        for index, value in enumerate((bytes(bad_crc), encoded[:-8], encoded+b"trailing", b"JPEG")):
            source = self.root / f"damaged-{index}.png"
            source.write_bytes(value)
            with self.subTest(index=index), self.assertRaises(serving.InferenceError):
                serving.read_rgb_png(source)
        animated = self.root / "animated.png"
        Image.new("RGB", (640, 360), (20, 40, 60)).save(animated, save_all=True,
            append_images=[Image.new("RGB", (640, 360), (30, 50, 70))], duration=100)
        with self.assertRaisesRegex(serving.InferenceError, "single-image-png"):
            serving.read_rgb_png(animated)

    def test_symlink_and_parent_junction_guards_run_before_reading(self):
        with patch.object(Path, "is_symlink", new=lambda path: path == self.image):
            with self.assertRaisesRegex(serving.InferenceError, "symlink"):
                serving.read_rgb_png(self.image)
        with patch.object(Path, "is_junction", new=lambda path: path == self.root, create=True):
            with self.assertRaisesRegex(serving.InferenceError, "symlink"):
                serving.read_rgb_png(self.image)
        with patch.object(Path, "is_symlink", new=lambda path: path == self.checkpoint), \
                patch.object(serving, "load_checkpoint") as loader:
            with self.assertRaisesRegex(serving.InferenceError, "symlink"):
                self.predictor()
            loader.assert_not_called()

    def test_png_replaced_after_inference_cannot_claim_original_image_identity(self):
        predictor = self.predictor()
        def mutate(rgb):
            self.image.write_bytes(b"changed-after-inference")
            return {"status": "PASS", "waypoints_ego_m": POINTS, "diagnostics": {}}
        with patch.object(predictor.model, "predict_rgb", side_effect=mutate):
            result = predictor.predict_png(self.image)
        self.assertEqual(result["error"], "inference-image-changed-during-prediction")
        self.assertIsNone(result["waypoints_ego_m"])

    def test_atomic_json_roundtrip_and_existing_destination_are_preserved(self):
        destination = self.root / "prediction.json"
        report = {"status": "PASS", "waypoints_ego_m": POINTS, "label": "곡선"}
        serving.write_prediction_json(destination, report)
        self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), report)
        original = destination.read_bytes()
        with self.assertRaisesRegex(serving.InferenceError, "already-exists"):
            serving.write_prediction_json(destination, {"status": "FAIL"})
        self.assertEqual(destination.read_bytes(), original)
        self.assertFalse(list(self.root.glob("*.pending")))

    def test_atomic_json_rejects_race_without_overwriting_competing_file(self):
        destination = self.root / "race.json"
        original_link = os.link
        def compete(source, target):
            Path(target).write_bytes(b"competing-artifact")
            return original_link(source, target)
        with patch.object(serving.os, "link", side_effect=compete):
            with self.assertRaisesRegex(serving.InferenceError, "already-exists"):
                serving.write_prediction_json(destination, {"status": "PASS"})
        self.assertEqual(destination.read_bytes(), b"competing-artifact")
        self.assertFalse(list(self.root.glob("*.pending")))

    def test_failed_publication_and_nonfinite_json_leave_no_output(self):
        destination = self.root / "failed.json"
        with patch.object(serving.os, "link", side_effect=OSError("private-token")):
            with self.assertRaisesRegex(serving.InferenceError, "output-write-failed"):
                serving.write_prediction_json(destination, {"status": "PASS"})
        self.assertFalse(destination.exists())
        self.assertFalse(list(self.root.glob("*.pending")))
        with self.assertRaisesRegex(serving.InferenceError, "not-json"):
            serving.write_prediction_json(destination, {"bad": float("nan")})
        self.assertFalse(destination.exists())

    def test_cli_creates_private_nested_output_and_protects_existing_result(self):
        cli = self.cli()
        destination = self.root / "results/local/fresh/nested/prediction.json"
        args = ["--checkpoint", str(self.checkpoint), "--image", str(self.image), "--output", str(destination)]
        with patch.object(cli, "ROOT", self.root), redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(args), 0)
        report = json.loads(destination.read_text())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["input_sha256"], hashlib.sha256(self.image.read_bytes()).hexdigest())
        self.assertEqual(report["model"]["runtime"]["device"], "cpu")
        self.assertEqual(report["model"]["selected_epoch"], 14)
        self.assertIn("scripts/predict_waypoints.py", report["cli_source_sha256"])
        original = destination.read_bytes()
        with patch.object(cli, "ROOT", self.root), patch.object(cli, "OfflineWaypointPredictor") as model, redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(args), 1)
            model.assert_not_called()
        self.assertEqual(destination.read_bytes(), original)

    def test_cli_outside_private_or_linked_output_never_loads_model(self):
        cli = self.cli()
        outside = self.root / "public/fresh/prediction.json"
        args = ["--checkpoint", str(self.checkpoint), "--image", str(self.image), "--output", str(outside)]
        with patch.object(cli, "ROOT", self.root), patch.object(cli, "OfflineWaypointPredictor") as model, redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(args), 1)
            model.assert_not_called()
        self.assertFalse(outside.parent.exists())
        destination = self.root / "results/local/link/prediction.json"
        args[-1] = str(destination)
        with patch.object(cli, "ROOT", self.root), \
                patch.object(Path, "is_junction", new=lambda path: path == destination.parent, create=True), \
                patch.object(cli, "OfflineWaypointPredictor") as model, redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(args), 1)
            model.assert_not_called()
        self.assertFalse(destination.exists())

    def test_cli_invalid_image_saves_fresh_failure_with_empty_path(self):
        cli = self.cli()
        Image.new("RGBA", (640, 360)).save(self.image)
        destination = self.root / "docs/local/failure/prediction.json"
        with patch.object(cli, "ROOT", self.root), redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["--checkpoint", str(self.checkpoint), "--image", str(self.image),
                                      "--output", str(destination)]), 1)
        report = json.loads(destination.read_text())
        self.assertEqual(report["status"], "FAIL")
        self.assertIsNone(report["waypoints_ego_m"])
        self.assertIsNone(report["input_sha256"])


if __name__ == "__main__":
    unittest.main()
