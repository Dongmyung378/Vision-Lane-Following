"""Geometry, reflection, source binding and archived selection reproducibility."""
import copy
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from carla_tools.learning_data import load_learning_config
from carla_tools.refined_waypoint import decode_segments, mirror_rgb_and_points, refined_loss, RefinedWaypointModel
from carla_tools.refined_training import choose_refinement, _contract, refined_source_hashes, load_refined_checkpoint
from carla_tools.waypoint_model import PRETRAINED_URL


class RefinementTests(unittest.TestCase):
    def test_reflection_matches_camera_and_negates_only_y_without_mutation(self):
        rgb = np.zeros((360, 640, 3), np.uint8)
        rgb[:, 93] = (20, 80, 210)
        points = torch.tensor([[3., -1.], [6., -2.], [9., -3.], [12., -4.], [15., -5.]])
        mirrored, targets = mirror_rgb_and_points(rgb, points)
        self.assertTrue(np.all(mirrored[:, 640 - 93] == (20, 80, 210)))
        torch.testing.assert_close(targets[:, 0], points[:, 0])
        torch.testing.assert_close(targets[:, 1], -points[:, 1])
        self.assertEqual(points[-1, 1].item(), -5.)

    def test_segment_decoder_zero_is_straight_and_curved_segments_accumulate(self):
        raw = torch.zeros(1, 5, 2)
        torch.testing.assert_close(decode_segments(raw)[0], torch.tensor([[3., 0.], [6., 0.], [9., 0.], [12., 0.], [15., 0.]]))
        raw[0, :, 0] = math.atanh(0.25)
        predicted = decode_segments(raw)
        expected = torch.tensor([[3. / math.sqrt(2), 3. / math.sqrt(2)]])
        torch.testing.assert_close(predicted[:, 0], expected)
        torch.testing.assert_close(predicted[:, -1], expected * 5)

    def test_horizon_weights_give_farthest_point_three_times_near_gradient(self):
        target = torch.zeros(1, 5, 2)
        predicted = torch.full_like(target, .1, requires_grad=True)
        loss = refined_loss(predicted, target, load_learning_config())
        loss.backward()
        self.assertAlmostEqual((predicted.grad[0, 4, 0] / predicted.grad[0, 0, 0]).item(), 3., places=5)

    def test_geometric_output_has_gradients_and_frozen_batchnorm(self):
        model = RefinedWaypointModel(pretrained=False)
        output = model(torch.randn(1, 3, 180, 320))
        output[..., 1].sum().backward()
        self.assertTrue(bool(torch.isfinite(model.head.output[-1].weight.grad).all()))
        self.assertGreater(model.head.output[-1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(not layer.training for layer in model.encoder.modules() if isinstance(layer, torch.nn.BatchNorm2d)))

    def test_checkpoint_strictly_rejects_source_and_manifest_changes(self):
        config = load_learning_config()
        model = RefinedWaypointModel(pretrained=False)
        contract, digest = _contract("hybrid_geometry", config)
        checkpoint = dict(schema_version=1, checkpoint_role="selected-refined-model", variant="hybrid_geometry",
            config=config.to_dict(), config_sha256=config.sha256, model_contract=contract, model_contract_sha256=digest,
            source_sha256=refined_source_hashes(), manifest_sha256="a" * 64, model_state=model.state_dict(),
            pretrained_provenance={"pretrained": True, "hash_verified": True, "url": PRETRAINED_URL, "sha256": "f37072fd" + "0" * 56})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(checkpoint, path)
            loaded, _, _ = load_refined_checkpoint(path, manifest_sha256="a" * 64)
            torch.testing.assert_close(loaded.head.output[-1].weight, model.head.output[-1].weight)
            with self.assertRaisesRegex(ValueError, "manifest-mismatch"):
                load_refined_checkpoint(path, manifest_sha256="b" * 64)
            checkpoint["source_sha256"] = {}
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "source-mismatch"):
                load_refined_checkpoint(path)

    def test_archived_strict_selection_decision_is_reproducible(self):
        reference = dict(status="PASS", split="dev", fixture_injection=False, final_town_used=False,
            manifest_sha256="a", config_sha256="b", sample_count=100, performance_target_passed=True,
            latency_ms={"p95": 6.}, metrics=[dict(case=case, condition="all", coverage=1., far_mean_m=.8, far_p95_m=1.5)
                for case in ("all", "left_curve", "right_curve", "straight")])
        candidate = copy.deepcopy(reference)
        candidate["metrics"][0]["far_mean_m"] = .5
        self.assertEqual(choose_refinement(reference, {"hybrid_geometry": candidate})["selected_candidate"], "hybrid_geometry")
        candidate["metrics"][-1]["far_mean_m"] = .9
        self.assertEqual(choose_refinement(reference, {"hybrid_geometry": candidate})["selected_candidate"], "resnet")
        candidate["sample_count"] = 99
        with self.assertRaisesRegex(ValueError, "dataset-or-protocol"):
            choose_refinement(reference, {"hybrid_geometry": candidate})


if __name__ == "__main__":
    unittest.main()
