"""Small CPU models exercise training, resume and failure accounting."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from carla_tools.learning_data import LearningConfig
from carla_tools.training import _metric_group, evaluate_waypoints, train_waypoints

torch.set_num_threads(1)


class TinyModel(nn.Module):
    def __init__(self, *, config, pretrained=True, weights_path=None):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(3, 12), nn.Tanh(), nn.Dropout(config.head_dropout), nn.Linear(12, 10))
        self.pretrained_provenance = {"source": "explicit-unit-test-injection", "pretrained": pretrained}

    def forward(self, image):
        prior = image.new_tensor([[distance, 0.0] for distance in (3, 6, 9, 12, 15)])
        return prior + self.head(image.mean(dim=(-1, -2))).reshape(-1, 5, 2)


class TinyDataset:
    def __init__(self, path, split, *, config, augment=False, seed=None):
        self.epoch, self.augment = 0, augment
        self.samples = [{"image": f"{split}-{index}.png", "image_sha256": hashlib.sha256(
            (path.parent / f"{split}-{index}.png").read_bytes()).hexdigest()} for index in range(8)]

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        channels = torch.tensor([index / 8, (index + 1) / 8, 0.25])
        image = channels[:, None, None].expand(3, 4, 4).clone()
        if self.augment:
            image *= 1 + self.epoch * 0.001
        target = torch.tensor([[distance + float(channels[0]) * 0.1, float(channels[1])]
                               for distance in (3, 6, 9, 12, 15)])
        return {"image": image, "target": target,
                "sample_id": str(index), "case": ("straight", "left_curve", "right_curve")[index % 3],
                "condition": "center", "route_id": "unit"}


def curve_points():
    return [[8 * math.sin(distance / 8), 8 * (1 - math.cos(distance / 8))]
            for distance in (3, 6, 9, 12, 15)]


class CurvedDataset(TinyDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["target"] = torch.tensor(curve_points(), dtype=torch.float32)
        item["case"] = "right_curve"
        return item


class FixedPointModel(nn.Module):
    def __init__(self, points):
        super().__init__()
        self.register_buffer("points", torch.tensor(points, dtype=torch.float32))

    def forward(self, image):
        return self.points[None].expand(image.shape[0], 5, 2)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "pilot_manifest.json"
        self.manifest.write_text('{"unit_fixture":true}', encoding="utf-8")
        from PIL import Image
        for split in ("train", "dev"):
            for index in range(8):
                Image.new("RGB", (640, 360), (index * 10, 10, 10)).save(self.root / f"{split}-{index}.png")
        self.config = dataclasses.replace(LearningConfig(), epochs=3, batch_size=4, num_workers=0,
                                          learning_rate=0.01, encoder_learning_rate=0.01,
                                          overfit_steps=45, overfit_batch_size=8,
                                          overfit_learning_rate=0.02, overfit_loss_ratio_target=0.9,
                                          early_stopping_patience=10)

    def train(self, output, **kwargs):
        return train_waypoints(self.manifest, self.root / output, config=self.config, device="cpu",
                               model_factory=TinyModel, dataset_factory=TinyDataset, **kwargs)

    def test_actual_forward_backward_and_complete_artifacts(self):
        report = self.train("train")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["overfit_check"]["status"], "PASS")
        self.assertEqual(report["completed_epochs"], 3)
        self.assertFalse(report["runtime"]["amp_enabled"])
        checkpoint = torch.load(self.root / "train/last_checkpoint.pt", weights_only=True)
        self.assertEqual(checkpoint["epoch"], 3)
        for name in ("optimizer_state", "scaler_state", "rng_state", "sampler_rng", "source_sha256"):
            self.assertIn(name, checkpoint)

    def test_epoch_boundary_resume_matches_uninterrupted_parameters(self):
        self.train("full")
        self.train("part", max_epochs=1)
        self.train("resumed", resume=self.root / "part/last_checkpoint.pt")
        full = torch.load(self.root / "full/last_checkpoint.pt", weights_only=True)
        resumed = torch.load(self.root / "resumed/last_checkpoint.pt", weights_only=True)
        for name, parameter in full["model_state"].items():
            self.assertTrue(torch.equal(parameter, resumed["model_state"][name]), name)
        self.assertEqual(full["rng_state"]["python"], resumed["rng_state"]["python"])
        self.assertTrue(torch.equal(full["sampler_rng"], resumed["sampler_rng"]))

    def test_resume_refuses_changed_manifest(self):
        self.train("part", max_epochs=1)
        self.manifest.write_text('{"unit_fixture":false}')
        with self.assertRaisesRegex(ValueError, "resume-provenance-mismatch-manifest"):
            self.train("changed", resume=self.root / "part/last_checkpoint.pt")
        report = json.loads((self.root / "changed/training_report.json").read_text())
        self.assertEqual(report["status"], "FAIL")

    def test_resume_refuses_changed_source_hash(self):
        self.train("part", max_epochs=1)
        path = self.root / "part/last_checkpoint.pt"
        checkpoint = torch.load(path, weights_only=True)
        checkpoint["source_sha256"]["carla_tools/training.py"] = "0" * 64
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "resume-provenance-mismatch-source"):
            self.train("changed", resume=path)

    def test_selected_inference_checkpoint_is_not_resumable(self):
        self.train("part", max_epochs=1)
        with self.assertRaisesRegex(ValueError, "resume-requires-last-training-state"):
            self.train("resumed", resume=self.root / "part/checkpoint.pt")

    def test_resume_refuses_tampered_actor_coordinate_contract(self):
        self.train("part", max_epochs=1)
        path = self.root / "part/last_checkpoint.pt"
        checkpoint = torch.load(path, weights_only=True)
        checkpoint["model_contract"]["output"]["origin"] = "camera-origin"
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "resume-model-contract-mismatch"):
            self.train("changed", resume=path)

    def test_production_loader_refuses_false_pretrained_provenance(self):
        from carla_tools.training import load_checkpoint
        self.train("part", max_epochs=1)
        with self.assertRaisesRegex(ValueError, "checkpoint-pretrained-provenance-invalid"):
            load_checkpoint(self.root / "part/checkpoint.pt")

    def test_existing_output_preserved(self):
        path = self.root / "existing"
        path.mkdir()
        sentinel = path / "untouched"
        sentinel.write_text("original")
        with self.assertRaisesRegex(ValueError, "output-directory-already-exists"):
            self.train("existing")
        self.assertEqual(sentinel.read_text(), "original")

    def test_metric_failure_accounting_and_near_far_indices(self):
        report = _metric_group([{"prediction": [[0, 0]] * 5, "errors_m": [1, 3, 5, 7, 9],
                                 "x_errors_m": [1] * 5, "y_errors_m": [2] * 5}, {"prediction": None}])
        self.assertEqual(report["near_mean_m"], 2)
        self.assertEqual(report["far_mean_m"], 7)
        self.assertEqual(report["coverage"], 0.5)
        self.assertEqual(report["failed_prediction_count"], 1)

    def test_evaluation_counts_all_dev_and_emits_separate_groups(self):
        checkpoint = self.root / "unit_checkpoint.pt"
        checkpoint.write_bytes(b"injected-unit-model")
        model = TinyModel(config=self.config)
        report = evaluate_waypoints(self.manifest, checkpoint, self.root / "evaluation", device="cpu",
                                    model=model, config=self.config, dataset_factory=TinyDataset,
                                    compare_baseline=False, overlay_count=0)
        self.assertEqual(report["sample_count"], 8)
        self.assertEqual(report["metrics"][0]["coverage"], 1.0)
        self.assertEqual({row["case"] for row in report["metrics"]},
                         {"all", "straight", "left_curve", "right_curve"})
        self.assertFalse(report["final_town_used"])

    def test_valid_curve_with_decreasing_far_x_has_complete_coverage(self):
        points = curve_points()
        self.assertLess(points[4][0], points[3][0])
        checkpoint = self.root / "unit_curve_checkpoint.pt"
        checkpoint.write_bytes(b"injected-unit-curved-model")
        report = evaluate_waypoints(self.manifest, checkpoint, self.root / "curve_evaluation", device="cpu",
                                    model=FixedPointModel(points), config=self.config,
                                    dataset_factory=CurvedDataset, compare_baseline=False, overlay_count=0)
        metric = report["metrics"][0]
        self.assertEqual(metric["sample_count"], 8)
        self.assertEqual(metric["coverage"], 1.0)
        self.assertEqual(metric["failed_prediction_count"], 0)
        self.assertEqual(metric["near_mean_m"], 0.0)
        self.assertEqual(metric["far_mean_m"], 0.0)
        self.assertTrue(report["performance_target_passed"])

    def test_decreasing_radial_distance_counts_as_failed_prediction(self):
        checkpoint = self.root / "unit_invalid_checkpoint.pt"
        checkpoint.write_bytes(b"injected-unit-invalid-model")
        report = evaluate_waypoints(self.manifest, checkpoint, self.root / "invalid_evaluation", device="cpu",
                                    model=FixedPointModel([[3, 0], [6, 0], [5, 0], [12, 0], [15, 0]]),
                                    config=self.config, dataset_factory=TinyDataset,
                                    compare_baseline=False, overlay_count=0)
        metric = report["metrics"][0]
        self.assertEqual(metric["sample_count"], 8)
        self.assertEqual(metric["coverage"], 0.0)
        self.assertEqual(metric["failed_prediction_count"], 8)
        self.assertIsNone(metric["near_mean_m"])
        self.assertFalse(report["performance_target_passed"])
        self.assertTrue(all(item["error"] == "invalid-model-output-order-or-range" for item in report["predictions"]))

    def test_nominal_projection_overlay_contains_model_and_target_markers(self):
        checkpoint = self.root / "unit_overlay_checkpoint.pt"
        checkpoint.write_bytes(b"injected-unit-overlay-model")
        model = FixedPointModel([[distance, 0.0] for distance in (3, 6, 9, 12, 15)])
        report = evaluate_waypoints(self.manifest, checkpoint, self.root / "overlay_evaluation", device="cpu",
                                    model=model, config=self.config, dataset_factory=CurvedDataset,
                                    compare_baseline=False, overlay_count=2)
        self.assertEqual(len(report["overlays"]), 1)
        from PIL import Image
        import numpy as np
        with Image.open(self.root / "overlay_evaluation" / report["overlays"][0]["overlay"]) as image:
            self.assertEqual(image.size, (640, 360))
            self.assertEqual(image.mode, "RGB")
            pixels = np.asarray(image)
        self.assertTrue(np.any(np.all(pixels == [255, 50, 50], axis=-1)))
        self.assertTrue(np.any(np.all(pixels == [0, 255, 0], axis=-1)))


if __name__ == "__main__":
    unittest.main()
