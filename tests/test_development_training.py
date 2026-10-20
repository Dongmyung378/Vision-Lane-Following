"""CPU checks for paired continuation and geometry-preserving robustness."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from carla_tools.contracts import canonical_sha256
from carla_tools.development_training import (
    ARMS, VIEWS, AppearanceAugmentation, DevelopmentProtocol, DevelopmentTrainingDataset,
    DevelopmentViewDataset, augment_appearance, development_protocol_from_dict,
    development_view, load_development_checkpoint, load_development_protocol,
    representative_seed, run_development_comparison, verify_equal_budget,
)
from carla_tools.learning_data import load_learning_config
from carla_tools.recovery_training import RecoveryProtocol, model_state_sha256
from carla_tools.waypoint_model import PRETRAINED_URL
from test_dataset_learning import bound_fixture
from test_recovery_training import StubTrainingSource, TinyWaypointModel


class DevelopmentProtocolTests(unittest.TestCase):
    def test_strict_protocol_views_and_numeric_ranges(self):
        protocol = load_development_protocol()
        self.assertEqual(protocol, development_protocol_from_dict(protocol.to_dict()))
        self.assertEqual(protocol.training.seeds, (42, 43))
        self.assertEqual(protocol.training.epochs, 12)
        self.assertEqual(protocol.training_selection, "central_only")
        bad = ({"epoch_dev_views": ["clean", "unknown"]}, {"epoch_dev_views": ["noise"]},
               {"epoch_dev_views": ["clean", "clean"]}, {"training_selection": "final"},
               {"schema_version": True}, {"view_selection_policy": "best-full-dev"})
        for change in bad:
            with self.subTest(change=change), self.assertRaises(ValueError):
                development_protocol_from_dict(dict(protocol.to_dict(), **change))
        for change in ({"brightness": float("nan")}, {"blur_probability": 1.1},
                       {"noise_std": -1}, {"contrast": True},
                       {"blur_probability": .5, "blur_sigma": 0}):
            value = protocol.to_dict()
            value["appearance"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                development_protocol_from_dict(value)
        value = protocol.to_dict(); value["appearance"]["extra"] = 1
        with self.assertRaisesRegex(ValueError, "appearance-fields-invalid"):
            development_protocol_from_dict(value)
        value = protocol.to_dict(); value["extra"] = 1
        with self.assertRaisesRegex(ValueError, "protocol-fields-invalid"):
            development_protocol_from_dict(value)

    def test_appearance_rng_is_repeatable_zero_config_is_identity_and_input_unchanged(self):
        image = StubTrainingSource("train")._read_image(None)
        before = image.copy()
        config = AppearanceAugmentation(blur_probability=1., noise_probability=1.)
        first = augment_appearance(image, seed=37, config=config)
        self.assertEqual(first.shape, (360, 640, 3))
        self.assertEqual(first.dtype, np.uint8)
        np.testing.assert_array_equal(first, augment_appearance(image, seed=37, config=config))
        self.assertFalse(np.array_equal(first, augment_appearance(image, seed=38, config=config)))
        np.testing.assert_array_equal(image, before)
        zero = AppearanceAugmentation(brightness=0., contrast=0., blur_probability=0.,
                                      blur_sigma=0., noise_probability=0., noise_std=0.)
        np.testing.assert_array_equal(augment_appearance(image, seed=37, config=zero), image)
        with self.assertRaisesRegex(ValueError, "rgb-shape-or-dtype"):
            augment_appearance(image.astype(np.float32), seed=37, config=config)

    def test_fixed_views_are_stable_and_do_not_modify_source(self):
        image = StubTrainingSource("dev")._read_image(None)
        before = image.copy()
        for view in VIEWS:
            with self.subTest(view=view):
                first = development_view(image, view, sample_id="dev-one")
                np.testing.assert_array_equal(first, development_view(image, view, sample_id="dev-one"))
                self.assertEqual(first.shape, image.shape)
                self.assertEqual(first.dtype, np.uint8)
        self.assertFalse(np.array_equal(development_view(image, "noise", sample_id="one"),
                                       development_view(image, "noise", sample_id="two")))
        np.testing.assert_array_equal(image, before)
        with self.assertRaisesRegex(ValueError, "invalid-view"):
            development_view(image, "final-weather", sample_id="dev")

    def test_candidate_only_changes_appearance_and_keeps_matching_mirror_and_labels(self):
        original = StubTrainingSource("source")
        a = DevelopmentTrainingDataset(original, ARMS[0], AppearanceAugmentation())
        b = DevelopmentTrainingDataset(StubTrainingSource("source"), ARMS[1], AppearanceAugmentation())
        differences = []
        for epoch in range(6):
            a.set_epoch(epoch); b.set_epoch(epoch)
            first, second = a[0], b[0]
            torch.testing.assert_close(first["target"], second["target"], rtol=0, atol=0)
            self.assertEqual(first["case"], second["case"])
            self.assertEqual(first["sample_id"], second["sample_id"])
            differences.append(not torch.equal(first["image"], second["image"]))
        self.assertTrue(all(differences))
        dev = StubTrainingSource("dev"); dev.split = "dev"
        with self.assertRaisesRegex(ValueError, "training-source-required"):
            DevelopmentTrainingDataset(dev, ARMS[0], AppearanceAugmentation())
        with self.assertRaisesRegex(ValueError, "dev-source-required"):
            DevelopmentViewDataset(original, ("clean",))
        expanded = DevelopmentViewDataset(dev, ("clean", "noise"))
        self.assertEqual(len(expanded), 2)
        torch.testing.assert_close(expanded[0]["target"], expanded[1]["target"], rtol=0, atol=0)

    def test_common_representative_seed_uses_paired_subset_loss_and_numeric_ties(self):
        protocol = DevelopmentProtocol(training=RecoveryProtocol(seeds=(43, 42)))
        training = {"42": {"continued": {"best_dev_loss": 1.}, "appearance_robust": {"best_dev_loss": 3.}},
                    "43": {"continued": {"best_dev_loss": 2.}, "appearance_robust": {"best_dev_loss": 2.}}}
        self.assertEqual(representative_seed(training, protocol), 42)
        training["43"]["appearance_robust"]["best_dev_loss"] = 1.
        self.assertEqual(representative_seed(training, protocol), 43)
        training["43"]["appearance_robust"]["best_dev_loss"] = float("nan")
        with self.assertRaisesRegex(ValueError, "representative-loss"):
            representative_seed(training, protocol)


class DevelopmentTrainingCPUIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_actual_two_seed_backward_paired_budget_export_and_fault_rejections(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, validation = bound_fixture(root / "data")
            initial_path = root / "synthetic_initial.pt"
            initial_path.write_bytes(b"synthetic-incumbent-complete-tensors")
            torch.manual_seed(19)
            initial_model = TinyWaypointModel()
            initial_state = model_state_sha256(initial_model.state_dict())
            provenance = {"pretrained": True, "hash_verified": True, "url": PRETRAINED_URL,
                          "sha256": "f37072fd" + "0" * 56}
            initial_checkpoint = {"variant": "hybrid_geometry", "pretrained_provenance": provenance}
            protocol = DevelopmentProtocol(training=RecoveryProtocol(seeds=(7, 8), train_samples_per_arm=4,
                epochs=2, batch_size=2, evaluation_batch_size=8, num_workers=0, epoch_dev_samples=6,
                warmup_epochs=1, latency_warmup=1, latency_samples=2, mixed_precision=False),
                epoch_dev_views=("clean", "noise"))
            with patch("carla_tools.development_training.RefinedWaypointModel", TinyWaypointModel), \
                 patch("carla_tools.development_training.load_refined_checkpoint",
                       return_value=(initial_model, load_learning_config(), initial_checkpoint)):
                report = run_development_comparison(manifest, validation, initial_path, root / "output",
                                                     protocol=protocol, device="cpu")
                self.assertEqual(report["status"], "PASS")
                self.assertTrue(report["equal_budget_verified"])
                self.assertFalse(report["closed_loop_verified"])
                self.assertFalse(report["incumbent_pointer_updated"])
                self.assertEqual(report["initial_model_state_sha256"], initial_state)
                self.assertEqual(report["initial_checkpoint_sha256"], hashlib.sha256(initial_path.read_bytes()).hexdigest())
                full_count = json.loads(manifest.read_text())["split_counts"]["dev"]
                for seed in protocol.training.seeds:
                    for arm in ARMS:
                        training = report["training"][str(seed)][arm]
                        self.assertEqual((training["completed_epochs"], training["optimizer_updates"]), (2, 4))
                        self.assertEqual(training["epoch_dev_view_count"], 12)
                        self.assertFalse(training["early_stopping"])
                        self.assertEqual(training["initial_model_state_sha256"], initial_state)
                        evaluation = report["evaluations"][str(seed)][arm]
                        self.assertEqual(evaluation["sample_count"], full_count)
                        self.assertEqual(evaluation["metrics"][0]["coverage"], 1.)
                        self.assertEqual(evaluation["common_dev_selection_sha256"], report["common_dev_selection_sha256"])
                        primary = root / "output" / f"seed{seed}" / arm / "checkpoint.pt"
                        compatible = primary.with_name("refined_checkpoint.pt")
                        first, second = (torch.load(path, weights_only=True) for path in (primary, compatible))
                        self.assertEqual(model_state_sha256(first["model_state"]), model_state_sha256(second["model_state"]))
                        self.assertEqual(second["development_provenance"]["checkpoint_sha256"], training["checkpoint_sha256"])
                        self.assertEqual(second["checkpoint_role"], "selected-refined-model")
                pair = json.loads((root / "output/native_pair_selection.json").read_text())
                self.assertFalse(pair["full_dev_evaluations_completed"])
                self.assertEqual(pair["representative_seed"], representative_seed(report["training"], protocol))
                path = root / "output/appearance_robust/checkpoint.pt"
                model, _, checkpoint = load_development_checkpoint(path, expected_sha256=report["representative_checkpoint_sha256"]["appearance_robust"],
                    manifest_sha256=report["manifest_sha256"])
                self.assertIsInstance(model, TinyWaypointModel)
                self.assertEqual(checkpoint["seed"], pair["representative_seed"])
                with self.assertRaisesRegex(ValueError, "file-binding-invalid"):
                    load_development_checkpoint(path, expected_sha256="0" * 64)
                with self.assertRaisesRegex(ValueError, "manifest-mismatch"):
                    load_development_checkpoint(path, manifest_sha256="0" * 64)
                for name in ("source", "state", "role", "protocol", "provenance"):
                    saved = torch.load(path, weights_only=True)
                    if name == "source":
                        saved["source_sha256"]["carla_tools/development_training.py"] = "0" * 64
                    elif name == "state":
                        next(iter(saved["model_state"].values())).add_(1.)
                    elif name == "role":
                        saved["checkpoint_role"] = "selected-refined-model"
                    elif name == "protocol":
                        saved["protocol"]["appearance"]["brightness"] = .1
                    else:
                        saved["initial_model_state_sha256"] = "unbound"
                    torch.save(saved, root / "changed.pt")
                    with self.subTest(tampering=name), self.assertRaises(ValueError):
                        load_development_checkpoint(root / "changed.pt")
                changed = copy.deepcopy(report["training"])
                changed["7"][ARMS[0]]["optimizer_updates"] -= 1
                with self.assertRaisesRegex(ValueError, "incomplete-training-budget"):
                    verify_equal_budget(changed, protocol)
                changed = copy.deepcopy(report["training"])
                changed["7"][ARMS[0]]["training_selection_sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "paired-arm-mismatch"):
                    verify_equal_budget(changed, protocol)
                changed = copy.deepcopy(report["training"])
                changed["7"][ARMS[0]]["history"][0]["sampler_order_sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "update-schedule-mismatch"):
                    verify_equal_budget(changed, protocol)


if __name__ == "__main__":
    unittest.main()
