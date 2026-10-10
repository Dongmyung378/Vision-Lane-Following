"""Small explicit CPU fixtures exercise real frozen-feature head training."""

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from carla_tools.vit_training import (
    PRETRAINED_SHA256, PRETRAINED_URL, cache_features, evaluate_vit_waypoints,
    load_vit_checkpoint, runtime_source_hashes, train_vit_waypoints,
)
from carla_tools.vit_waypoint import FEATURE_COUNT, VitConfig, sha256_file


class TinyFrozenModel(nn.Module):
    def __init__(self, *, config, pretrained=True, weights_path=None, expected_weights_sha256=None, weights_url=None):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(3))
        self.encoder.requires_grad_(False).eval()
        self.head = nn.Sequential(nn.Linear(FEATURE_COUNT, config.head_hidden_features), nn.ReLU(),
                                  nn.Dropout(config.head_dropout), nn.Linear(config.head_hidden_features, 10))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("straight_prior", torch.tensor([[distance, 0.0] for distance in (3, 6, 9, 12, 15)]))
        self.pretrained_provenance = {"purpose": "explicit-unit-fixture-not-pretrained-DINOv3", "pretrained": pretrained}

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def encode(self, image):
        with torch.no_grad():
            features = self.encoder(image.mean((-2, -1)))
        return features.repeat(1, FEATURE_COUNT // 3)

    def predict_features(self, features):
        return self.straight_prior[None] + self.head(features).reshape(-1, 5, 2) * self.config.label_scale_m


class TinyDataset:
    def __init__(self, manifest_path, split, *, config, augment=False, seed=None):
        if augment:
            raise AssertionError("Frozen cache must not request augmented inputs")
        self.path = manifest_path
        self.manifest_hash = sha256_file(manifest_path)
        self.samples = [{"sample_id": f"{split}-{index}", "case": ("straight", "left_curve", "right_curve")[index % 3],
                         "condition": "center", "route_id": "fixture-route-" + split,
                         "image": f"{split}-{index}.png", "image_sha256": sha256_file(manifest_path.parent / f"{split}-{index}.png")}
                        for index in range(8 if split == "train" else 4)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image = torch.tensor([0.1 + index * 0.05, 0.25, 0.5])[:, None, None].expand(3, 180, 320).clone()
        target = torch.tensor([[distance - 0.01 * distance, 0.03 * distance] for distance in (3, 6, 9, 12, 15)])
        return {"image": image, "target": target, **{key: self.samples[index][key] for key in ("sample_id", "case", "condition", "route_id")}}

    def assert_manifest_unchanged(self):
        if sha256_file(self.path) != self.manifest_hash:
            raise ValueError("fixture-manifest-changed")


class VitTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text('{"fixture":true}', encoding="utf-8")
        from PIL import Image
        for split, count in (("train", 8), ("dev", 4)):
            for index in range(count):
                Image.new("RGB", (640, 360), (index * 20, 10, 50)).save(self.root / f"{split}-{index}.png")
        self.config = dataclasses.replace(VitConfig(), epochs=3, batch_size=4, feature_batch_size=4, head_hidden_features=8,
                                          head_dropout=0, overfit_steps=45, overfit_batch_size=8, overfit_learning_rate=0.005,
                                          overfit_loss_ratio_target=0.9, early_stopping_patience=8)

    def train(self, name):
        return train_vit_waypoints(self.manifest, self.root / name, weights_path=self.root / "fixture-weights.pth",
                                    config=self.config, device="cpu", max_seconds=60,
                                    model_factory=TinyFrozenModel, dataset_factory=TinyDataset)

    def test_actual_head_training_fresh_reset_selected_checkpoint_and_full_evaluation(self):
        report = self.train("train")
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["fixture_injection"])
        self.assertEqual(report["completed_epochs"], 3)
        self.assertEqual(report["selected_epoch"], min(report["history"], key=lambda row: row["dev_loss"])["epoch"])
        self.assertTrue(report["overfit_check"]["head_reset_to_fresh_initialization"])
        self.assertFalse(report["runtime"]["encoder_amp_fp16"])
        self.assertEqual(report["runtime"]["head_precision"], "float32")
        self.assertEqual(report["feature_cache"]["train"]["sample_count"], 8)
        self.assertEqual(report["feature_cache"]["dev"]["sample_count"], 4)
        model, _, checkpoint = load_vit_checkpoint(self.root / "train/checkpoint.pt", model_factory=TinyFrozenModel)
        torch.testing.assert_close(model.encoder.weight, torch.eye(3), rtol=0, atol=0)
        self.assertEqual(checkpoint["source_sha256"], runtime_source_hashes())
        metrics = json.loads((self.root / "train/evaluation/metrics.json").read_text())
        self.assertEqual(metrics["sample_count"], 4)
        self.assertEqual(metrics["metrics"][0]["coverage"], 1)
        self.assertEqual(len(metrics["predictions"]), 4)
        self.assertEqual(metrics["cached_feature_prediction_check"]["maximum_abs_difference_m"], 0)
        self.assertEqual(len((self.root / "train/evaluation/predictions.jsonl").read_text().splitlines()), 4)
        self.assertFalse(metrics["final_town_used"])
        self.assertTrue(all("image_sha256" in row and len(row["errors_m"]) == 5 for row in metrics["predictions"]))

    def test_seeded_run_is_reproducible_after_diagnostic_reset(self):
        self.train("first")
        self.train("second")
        first = torch.load(self.root / "first/checkpoint.pt", weights_only=True)
        second = torch.load(self.root / "second/checkpoint.pt", weights_only=True)
        self.assertEqual(first["epoch"], second["epoch"])
        for name, value in first["model_state"].items():
            torch.testing.assert_close(value, second["model_state"][name], rtol=0, atol=0)

    def test_fixture_checkpoint_cannot_masquerade_as_production_and_provenance_tampering_fails(self):
        self.train("train")
        path = self.root / "train/checkpoint.pt"
        with self.assertRaisesRegex(ValueError, "fixture-checkpoint"):
            load_vit_checkpoint(path)
        checkpoint = torch.load(path, weights_only=True)
        checkpoint["source_sha256"]["carla_tools/vit_waypoint.py"] = "0" * 64
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "source-mismatch"):
            load_vit_checkpoint(path, model_factory=TinyFrozenModel)

    def test_invalid_predictions_remain_in_every_evaluation_denominator(self):
        self.train("train")
        path = self.root / "train/checkpoint.pt"
        checkpoint = torch.load(path, weights_only=True)
        checkpoint["model_state"]["head.3.bias"].fill_(float("nan"))
        torch.save(checkpoint, path)
        report = evaluate_vit_waypoints(self.manifest, path, self.root / "failed-eval", model_factory=TinyFrozenModel, dataset_factory=TinyDataset)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["sample_count"], 4)
        self.assertEqual(report["metrics"][0]["coverage"], 0)
        self.assertEqual(report["metrics"][0]["failed_prediction_count"], 4)
        self.assertFalse(report["performance_target_passed"])
        self.assertTrue(all(row["prediction"] is None and row["error"] for row in report["predictions"]))

    def test_no_overwrite_or_silent_cuda_fallback(self):
        self.train("train")
        digest = sha256_file(self.root / "train/checkpoint.pt")
        with self.assertRaisesRegex(ValueError, "already-exists"):
            self.train("train")
        self.assertEqual(sha256_file(self.root / "train/checkpoint.pt"), digest)
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "cuda-unavailable"):
                train_vit_waypoints(self.manifest, self.root / "cuda", weights_path=self.root / "fixture.pth", device="cuda",
                                    model_factory=TinyFrozenModel, dataset_factory=TinyDataset)
        self.assertEqual(json.loads((self.root / "cuda/training_report.json").read_text())["status"], "FAIL")

    def test_cache_identity_and_epoch_budget_are_checked(self):
        dataset = TinyDataset(self.manifest, "train", config=None)
        dataset.samples[0]["sample_id"] = "binding-0"
        cache = cache_features(TinyFrozenModel(config=self.config), dataset, torch.device("cpu"), self.config)
        self.assertEqual(cache["sample_ids"][0], "binding-0")
        with self.assertRaisesRegex(ValueError, "official-revision"):
            train_vit_waypoints(self.manifest, self.root / "bad-pin", weights_path=self.root / "anything", expected_weights_sha256="0" * 64)
        self.assertFalse((self.root / "bad-pin").exists())
        for budget in (0, -1, float("nan")):
            with self.assertRaisesRegex(ValueError, "time-budget"):
                train_vit_waypoints(self.manifest, self.root / "bad-budget", weights_path=self.root / "anything", max_seconds=budget)


if __name__ == "__main__":
    unittest.main()
