"""Actual CPU ResNet topology, geometry units and loss gradient checks."""

import tempfile
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from carla_tools.learning_data import load_learning_config
from carla_tools.waypoint_model import (
    DeterministicSpatialPool, WaypointModel, WaypointModelError, create_model,
    model_contract, waypoint_smooth_l1_loss,
)


class WaypointModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_resnet_rgb_head_shape_prior_and_contract(self):
        torch.manual_seed(9)
        model = create_model(pretrained=False)
        self.assertEqual(model.encoder.conv1.in_channels, 3)
        self.assertEqual(len(model.encoder.layer1), 2)
        self.assertEqual(len(model.encoder.layer4), 2)
        self.assertIsInstance(model.encoder.avgpool, DeterministicSpatialPool)
        self.assertEqual(model.head[0].in_features, 4096)
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 3, 180, 320))
        expected = torch.tensor([[s, 0.0] for s in (3, 6, 9, 12, 15)])
        self.assertEqual(tuple(output.shape), (2, 5, 2))
        torch.testing.assert_close(output[0], expected)
        torch.testing.assert_close(output[1], expected)
        contract = model_contract()
        self.assertEqual(contract["output"]["origin"], "vehicle-actor-origin")
        self.assertEqual(contract["output"]["y"], "right")
        self.assertFalse(contract["input"]["speed_input"])
        self.assertEqual(contract["model_version"], "vlf-resnet18-road-roi-waypoint-v3")
        self.assertEqual(contract["encoder"]["pooling"]["output_grid"], [2, 4])
        self.assertEqual(contract["head"]["features"], [4096, 256, 10])
        self.assertFalse(model.pretrained_provenance["pretrained"])

    def test_spatial_pool_disjoint_means_and_backward_cover_all_pixels(self):
        pool = DeterministicSpatialPool()
        feature = torch.arange(60, dtype=torch.float64).reshape(1, 1, 6, 10).requires_grad_()
        output = pool(feature)
        expected = torch.tensor([[[[10.5, 13, 15.5, 18], [40.5, 43, 45.5, 48]]]], dtype=torch.float64)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            output.sum().backward()
        finally:
            torch.use_deterministic_algorithms(previous)
        expected_gradient = torch.tensor([1/6, 1/6, 1/9, 1/9, 1/9, 1/6, 1/6, 1/9, 1/9, 1/9], dtype=torch.float64)
        torch.testing.assert_close(feature.grad[0, 0], expected_gradient.repeat(6, 1), rtol=0, atol=0)
        with self.assertRaisesRegex(WaypointModelError, "feature-shape"):
            pool(torch.zeros(1, 512, 5, 10))

    def test_spatial_pool_retains_left_right_signal_lost_by_global_mean(self):
        feature = torch.zeros(2, 1, 6, 10)
        feature[0, :, :, :2] = 1
        feature[1, :, :, -2:] = 1
        # Equal global means cannot identify the side containing the signal.
        torch.testing.assert_close(feature.mean(dim=(-2, -1))[0], feature.mean(dim=(-2, -1))[1])
        spatial = DeterministicSpatialPool()(feature)
        side_score = spatial[:, :, :, 0].mean(dim=(-2, -1)) - spatial[:, :, :, -1].mean(dim=(-2, -1))
        self.assertGreater(float(side_score[0]), 0)
        self.assertLess(float(side_score[1]), 0)

    def test_encoder_batchnorm_statistics_are_frozen_and_head_dropout_trains(self):
        model = WaypointModel(pretrained=False)
        batchnorms = [layer for layer in model.encoder.modules() if isinstance(layer, torch.nn.BatchNorm2d)]
        saved = [(layer.running_mean.clone(), layer.running_var.clone(), layer.num_batches_tracked.clone())
                 for layer in batchnorms]
        self.assertTrue(batchnorms)
        self.assertIs(model.train(), model)
        self.assertTrue(model.training)
        self.assertTrue(model.head[2].training)
        self.assertTrue(all(not layer.training and layer.weight.requires_grad and layer.bias.requires_grad
                            for layer in batchnorms))
        with torch.no_grad():
            model(torch.ones(2, 3, 180, 320))
            model(torch.ones(2, 3, 180, 320) * 7)
        for layer, (running_mean, running_var, batches) in zip(batchnorms, saved):
            torch.testing.assert_close(layer.running_mean, running_mean, rtol=0, atol=0)
            torch.testing.assert_close(layer.running_var, running_var, rtol=0, atol=0)
            self.assertEqual(layer.num_batches_tracked, batches)
        model.eval()
        self.assertFalse(model.head[2].training)
        model.train()
        self.assertTrue(model.head[2].training)
        self.assertTrue(all(not layer.training for layer in batchnorms))

    def test_gradients_reach_encoder_and_both_xy_head_axes(self):
        torch.manual_seed(20)
        model = WaypointModel(pretrained=False)
        image = torch.randn(2, 3, 180, 320)
        target = torch.tensor([[[s - 0.1*s, 0.02*s*s] for s in (3, 6, 9, 12, 15)]]).repeat(2, 1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        initial = waypoint_smooth_l1_loss(model(image), target)
        initial.backward()
        self.assertGreater(float(model.head[-1].bias.grad.reshape(5, 2)[:, 0].abs().sum()), 0)
        self.assertGreater(float(model.head[-1].bias.grad.reshape(5, 2)[:, 1].abs().sum()), 0)
        optimizer.step()
        optimizer.zero_grad()
        loss = waypoint_smooth_l1_loss(model(image), target)
        loss.backward()
        self.assertGreater(float(model.encoder.conv1.weight.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(loss))
        for axis in (0, 1):
            optimizer.zero_grad()
            prediction = model(image)
            axis_loss = torch.nn.functional.smooth_l1_loss(prediction[..., axis] / 15, target[..., axis] / 15, beta=0.03)
            axis_loss.backward()
            self.assertGreater(float(model.encoder.conv1.weight.grad.abs().sum()), 0)
            self.assertGreater(float(model.encoder.bn1.weight.grad.abs().sum()), 0)

    def test_smooth_l1_units_match_independent_piecewise_formula(self):
        target = torch.zeros(1, 5, 2)
        prediction = torch.tensor([[[0.1, 1.0]] * 5], requires_grad=True)
        config = load_learning_config()
        error = prediction.detach().numpy() / 15.0
        expected = np.where(np.abs(error) < 0.03, 0.5 * error**2 / 0.03, np.abs(error) - 0.015).mean()
        loss = waypoint_smooth_l1_loss(prediction, target, config)
        self.assertAlmostEqual(float(loss.detach()), float(expected), places=7)
        loss.backward()
        self.assertTrue(torch.all(prediction.grad > 0))
        self.assertEqual(float(waypoint_smooth_l1_loss(target, target)), 0.0)

    def test_pretrained_missing_corrupt_and_download_failure_never_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pth"
            with self.assertRaisesRegex(WaypointModelError, "missing"):
                WaypointModel(pretrained=True, weights_path=missing)
            missing.write_bytes(b"random checkpoint")
            with self.assertRaisesRegex(WaypointModelError, "sha256"):
                WaypointModel(pretrained=True, weights_path=missing)
            with patch("torch.hub.get_dir", return_value=directory), patch("torch.hub.download_url_to_file", side_effect=OSError("offline")):
                with self.assertRaisesRegex(WaypointModelError, "download-failed"):
                    WaypointModel(pretrained=True)

    def test_shape_and_nonfinite_data_rejected(self):
        model = WaypointModel(pretrained=False)
        for bad in (torch.zeros(1, 3, 224, 224), torch.zeros(3, 180, 320), torch.zeros(0, 3, 180, 320),
                    torch.full((1, 3, 180, 320), float("nan")), torch.zeros(1, 3, 180, 320, dtype=torch.uint8)):
            with self.assertRaises(WaypointModelError):
                model(bad)
        for pred, target in ((torch.zeros(5, 2), torch.zeros(5, 2)), (torch.zeros(1, 5, 2), torch.zeros(2, 5, 2)),
                             (torch.full((1, 5, 2), float("inf")), torch.zeros(1, 5, 2))):
            with self.assertRaises(WaypointModelError):
                waypoint_smooth_l1_loss(pred, target)

    def test_rgb_inference_is_stateless_and_failed_predictions_explicit(self):
        model = WaypointModel(pretrained=False)
        rgb = np.zeros((360, 640, 3), dtype=np.uint8)
        result = model.predict_rgb(rgb)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["waypoints_ego_m"], [[float(s), 0.0] for s in (3, 6, 9, 12, 15)])
        invalid = model.predict_rgb(rgb.astype(np.float32))
        self.assertEqual(invalid["status"], "FAIL")
        self.assertIsNone(invalid["waypoints_ego_m"])
        with torch.no_grad():
            model.head[-1].bias[0] = -1.0
        invalid = model.predict_rgb(rgb)
        self.assertEqual(invalid["status"], "FAIL")
        self.assertIsNone(invalid["waypoints_ego_m"])

    def test_rgb_prediction_accepts_arc_ordered_past_ninety_degree_curve(self):
        model = WaypointModel(pretrained=False)
        points = torch.tensor([[8 * math.sin(s / 8), 8 * (1 - math.cos(s / 8))] for s in (3, 6, 9, 12, 15)])
        with torch.no_grad():
            model.head[-1].bias.copy_(((points - model.straight_prior) / 15).reshape(-1))
        result = model.predict_rgb(np.zeros((360, 640, 3), dtype=np.uint8))
        self.assertEqual(result["status"], "PASS")
        self.assertLess(result["waypoints_ego_m"][4][0], result["waypoints_ego_m"][3][0])
        torch.testing.assert_close(torch.tensor(result["waypoints_ego_m"]), points)


if __name__ == "__main__":
    unittest.main()
