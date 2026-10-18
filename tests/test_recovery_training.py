"""Synthetic CPU regression of shared warm starts and equal update budgets."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from carla_tools.contracts import canonical_sha256
from carla_tools.learning_data import load_learning_config
from carla_tools.recovery_training import (
    ARMS, MirroredRecoveryDataset, RecoveryProtocol, _equal_budget, epoch_order,
    evaluation_contract, fixed_dev_indices, learning_rate_factor, load_recovery_checkpoint,
    load_recovery_protocol, model_state_sha256, recovery_protocol_from_dict,
    representative_seed, run_recovery_comparison,
)
from carla_tools.waypoint_model import PRETRAINED_URL
from test_dataset_learning import bound_fixture


class TinyWaypointModel(torch.nn.Module):
    """Test-only differentiable RGB network; never a project model artifact."""
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                           torch.nn.Linear(3, 8), torch.nn.Tanh())
        self.head = torch.nn.Linear(8, 5)
        self.register_buffer("distances", torch.tensor([3., 6., 9., 12., 15.]))

    def forward(self, image):
        y = self.head(self.encoder(image)).tanh()
        x = self.distances[None].expand(image.shape[0], -1)
        return torch.stack((x, y), dim=-1)


class StubTrainingSource:
    split, augment, seed, config = "train", False, 42, load_learning_config()

    def __init__(self, identifier):
        self.samples = [{"sample_id": identifier, "case": "left_curve", "condition": "center",
                         "route_id": "synthetic_route", "waypoints_ego_m": [[3. * i, -.2 * i] for i in range(1, 6)]}]

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _read_image(self, sample):
        image = np.empty((360, 640, 3), np.uint8)
        image[:] = (35, 79, 122)
        image[:, 100:200] = (180, 111, 43)
        return image


class RecoveryProtocolTests(unittest.TestCase):
    def test_strict_protocol_and_inference_contract_exclude_training_seed(self):
        protocol = load_recovery_protocol()
        self.assertEqual(recovery_protocol_from_dict(protocol.to_dict()), protocol)
        self.assertEqual((protocol.epochs, protocol.train_samples_per_arm), (32, 5000))
        self.assertEqual(protocol.seeds, (42, 43))
        self.assertEqual(evaluation_contract(protocol.learning_config(42)),
                         evaluation_contract(protocol.learning_config(43)))
        self.assertNotEqual(protocol.learning_config(42).sha256, protocol.learning_config(43).sha256)
        for change in ({"seeds": [42, 42]}, {"seeds": [True]}, {"epochs": 0},
                       {"learning_rate": float("nan")}, {"representative_seed_policy": "full-dev-best"}):
            value = dict(protocol.to_dict(), **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                recovery_protocol_from_dict(value)
        value = protocol.to_dict(); value["extra"] = 1
        with self.assertRaisesRegex(ValueError, "fields-invalid"):
            recovery_protocol_from_dict(value)

    def test_state_hash_handles_scalar_buffers_and_detects_named_value_changes(self):
        state = OrderedDict(a=torch.ones(3), counter=torch.tensor(5, dtype=torch.int64))
        self.assertEqual(model_state_sha256(state), model_state_sha256(dict(reversed(list(state.items())))))
        changed = copy.deepcopy(state); changed["counter"].add_(1)
        self.assertNotEqual(model_state_sha256(state), model_state_sha256(changed))
        changed = copy.deepcopy(state); changed["a"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite-initial-state"):
            model_state_sha256(changed)

    def test_common_dev_rank_ignores_labels_and_balances_strata(self):
        samples = [{"sample_id": f"dev_{case}_{i}", "split": "dev", "case": case,
                    "weather": "ClearNoon", "condition": "center", "prediction": 1.}
                   for case in ("straight", "left_curve", "right_curve") for i in range(7)]
        indices = fixed_dev_indices(samples, 9)
        self.assertEqual([sum(samples[index]["case"] == case for index in indices)
                          for case in ("straight", "left_curve", "right_curve")], [3, 3, 3])
        changed = copy.deepcopy(samples)
        for sample in changed:
            sample["prediction"] = 999.
        self.assertEqual(indices, fixed_dev_indices(changed, 9))
        changed[0]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "membership"):
            fixed_dev_indices(changed, 9)
        with self.assertRaisesRegex(ValueError, "insufficient-common-dev"):
            fixed_dev_indices(samples, 22)

    def test_identical_slot_augmentation_and_epoch_order_across_arms(self):
        a, b = (MirroredRecoveryDataset(StubTrainingSource(name)) for name in ("center_id", "recovery_id"))
        for epoch in (0, 1, 5):
            a.set_epoch(epoch); b.set_epoch(epoch)
            torch.testing.assert_close(a[0]["image"], b[0]["image"], rtol=0, atol=0)
            torch.testing.assert_close(a[0]["target"], b[0]["target"], rtol=0, atol=0)
            self.assertEqual(a[0]["case"], b[0]["case"])
        self.assertEqual(epoch_order(21, 42, 0), epoch_order(21, 42, 0))
        self.assertNotEqual(epoch_order(21, 42, 0), epoch_order(21, 42, 1))
        protocol = RecoveryProtocol()
        self.assertEqual(learning_rate_factor(0, protocol), .5)
        self.assertEqual(learning_rate_factor(1, protocol), 1.)
        self.assertAlmostEqual(learning_rate_factor(31, protocol), .05)

    def test_representative_seed_uses_paired_subset_losses_and_numeric_tie(self):
        protocol = RecoveryProtocol(seeds=(43, 42))
        training = {"42": {"central_only": {"best_dev_loss": 1.}, "central_recovery": {"best_dev_loss": 3.}},
                    "43": {"central_only": {"best_dev_loss": 2.}, "central_recovery": {"best_dev_loss": 2.}}}
        self.assertEqual(representative_seed(training, protocol), 42)
        training["43"]["central_recovery"]["best_dev_loss"] = 1.
        self.assertEqual(representative_seed(training, protocol), 43)


class RecoveryTrainingCPUIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_actual_two_seed_backward_equal_budget_full_dev_and_source_bound_load(self):
        # A tiny test network and independently covered synthetic native boundary
        # keep the regression on CPU; no simulator or GPU result is asserted.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, validation = bound_fixture(root / "data")
            initial_path = root / "synthetic_initial.pt"
            initial_path.write_bytes(b"synthetic-initial-checkpoint-bytes")
            torch.manual_seed(4)
            initial_model = TinyWaypointModel()
            initial_state_sha = model_state_sha256(initial_model.state_dict())
            provenance = {"pretrained": True, "hash_verified": True, "url": PRETRAINED_URL,
                          "sha256": "f37072fd" + "0" * 56}
            initial_checkpoint = {"variant": "hybrid_geometry", "pretrained_provenance": provenance}
            protocol = RecoveryProtocol(seeds=(7, 8), train_samples_per_arm=4, epochs=2,
                batch_size=2, evaluation_batch_size=8, num_workers=0, epoch_dev_samples=6,
                warmup_epochs=1, latency_warmup=1, latency_samples=2, mixed_precision=False)
            with patch("carla_tools.recovery_training.RefinedWaypointModel", TinyWaypointModel), \
                 patch("carla_tools.recovery_training.load_refined_checkpoint",
                       return_value=(initial_model, load_learning_config(), initial_checkpoint)):
                report = run_recovery_comparison(manifest, validation, initial_path, root / "output",
                                                 protocol=protocol, device="cpu")
                self.assertEqual(report["status"], "PASS")
                self.assertTrue(report["equal_budget_verified"])
                self.assertFalse(report["closed_loop_verified"])
                self.assertEqual(report["initial_model_state_sha256"], initial_state_sha)
                self.assertEqual(report["initial_checkpoint_sha256"], hashlib.sha256(initial_path.read_bytes()).hexdigest())
                full_count = json.loads(manifest.read_text())["split_counts"]["dev"]
                for seed in protocol.seeds:
                    for arm in ARMS:
                        training = report["training"][str(seed)][arm]
                        self.assertEqual((training["completed_epochs"], training["optimizer_updates"]), (2, 4))
                        self.assertFalse(training["early_stopping"])
                        self.assertEqual(training["initial_model_state_sha256"], initial_state_sha)
                        evaluation = report["evaluations"][str(seed)][arm]
                        self.assertEqual(evaluation["sample_count"], full_count)
                        self.assertEqual(evaluation["metrics"][0]["coverage"], 1.)
                        self.assertEqual(evaluation["latency_ms"]["sample_count"], 2)
                        self.assertEqual(sum(evaluation["case_sample_counts"].values()), full_count)
                        self.assertEqual(evaluation["common_dev_selection_sha256"], report["common_dev_selection_sha256"])
                pair = json.loads((root / "output/native_pair_selection.json").read_text())
                self.assertFalse(pair["full_dev_evaluations_completed"])
                self.assertEqual(pair["representative_seed"], representative_seed(report["training"], protocol))
                model, config, checkpoint = load_recovery_checkpoint(root / "output/central_recovery/checkpoint.pt",
                    device="cpu", expected_sha256=report["representative_checkpoint_sha256"]["central_recovery"],
                    manifest_sha256=report["manifest_sha256"])
                self.assertIsInstance(model, TinyWaypointModel)
                self.assertEqual(checkpoint["seed"], pair["representative_seed"])
                self.assertTrue((root / "output/central_only/evaluation/metrics.json").is_file())
                with self.assertRaisesRegex(ValueError, "sha256-mismatch"):
                    load_recovery_checkpoint(root / "output/central_recovery/checkpoint.pt", expected_sha256="0" * 64)
                changed = copy.deepcopy(report["training"])
                changed["7"]["central_only"]["optimizer_updates"] -= 1
                with self.assertRaisesRegex(ValueError, "incomplete-equal-budget-arm"):
                    _equal_budget(changed, protocol)
                saved = torch.load(root / "output/central_recovery/checkpoint.pt", weights_only=True)
                saved["source_sha256"]["carla_tools/recovery_training.py"] = "0" * 64
                torch.save(saved, root / "changed.pt")
                with self.assertRaisesRegex(ValueError, "contract-or-source-mismatch"):
                    load_recovery_checkpoint(root / "changed.pt")


if __name__ == "__main__":
    unittest.main()
