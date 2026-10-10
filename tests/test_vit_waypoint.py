"""CPU fixtures verify the ViT spatial, provenance and meter contracts."""

import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from carla_tools.vit_waypoint import (
    FEATURE_COUNT, WEIGHTS_REPOSITORY, PatchSpatialPool, VitConfig, VitWaypointError,
    VitWaypointModel, canonical_sha256, model_contract, sha256_file, vit_config_from_dict, waypoint_loss,
)


class FixtureBackbone(nn.Module):
    """Explicit small test encoder; no claim of pretrained DINO performance."""

    def __init__(self, config):
        super().__init__()
        self.num_prefix_tokens = 5
        self.patch = nn.Conv2d(3, 768, 16, stride=16)
        self.register_buffer("prefix", torch.full((1, 5, 768), 10000.0))
        self.last_input = None

    def forward_features(self, image):
        self.last_input = image.detach().clone()
        tokens = self.patch(image).flatten(2).transpose(1, 2)
        return torch.cat((self.prefix.expand(image.shape[0], -1, -1), tokens), dim=1)


class VitWaypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        return VitWaypointModel(pretrained=False, backbone_factory=FixtureBackbone)

    def test_original_roi_padding_spatial_tokens_and_actor_meter_prior(self):
        model = self.model().eval()
        image = torch.arange(180, dtype=torch.float32)[None, None, :, None].expand(2, 3, 180, 320)
        with torch.no_grad():
            output = model(image)
        self.assertEqual(model.encoder.last_input.shape, (2, 3, 192, 320))
        torch.testing.assert_close(model.encoder.last_input[:, :, :180], image, rtol=0, atol=0)
        torch.testing.assert_close(model.encoder.last_input[:, :, 180:], image[:, :, -1:].expand(-1, -1, 12, -1), rtol=0, atol=0)
        expected = torch.tensor([[distance, 0] for distance in (3, 6, 9, 12, 15)], dtype=torch.float32)
        torch.testing.assert_close(output, expected[None].expand(2, -1, -1), rtol=0, atol=0)
        self.assertEqual(model.head[0].in_features, FEATURE_COUNT)
        contract = model_contract()
        self.assertEqual(contract["input"]["source_crop_xyxy"], [0, 144, 640, 360])
        self.assertEqual(contract["output"]["origin"], "vehicle-actor-origin")
        self.assertEqual(contract["output"]["units"], "meters")
        self.assertFalse(contract["input"]["speed_input"])

    def test_pool_has_disjoint_bins_and_retains_left_right_location(self):
        feature = torch.arange(240, dtype=torch.float64).reshape(1, 1, 12, 20).expand(1, 768, -1, -1).clone().requires_grad_()
        output = PatchSpatialPool()(feature)
        expected = torch.tensor([52, 57, 62, 67, 172, 177, 182, 187], dtype=torch.float64).repeat(768)[None]
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.sum().backward()
        torch.testing.assert_close(feature.grad, torch.full_like(feature, 1 / 30), rtol=0, atol=0)
        mirrored = feature.detach().flip(-1)
        torch.testing.assert_close(mirrored.mean((-2, -1)), feature.detach().mean((-2, -1)))
        self.assertFalse(torch.equal(PatchSpatialPool()(mirrored), output))
        with self.assertRaisesRegex(VitWaypointError, "feature-shape"):
            PatchSpatialPool()(torch.zeros(1, 768, 11, 20))

    def test_prefix_tokens_are_excluded_from_patch_features(self):
        model = self.model()
        with torch.no_grad():
            model.encoder.patch.weight.zero_()
            model.encoder.patch.bias.fill_(2.5)
        features = model.encode(torch.zeros(1, 3, 180, 320))
        torch.testing.assert_close(features, torch.full((1, FEATURE_COUNT), 2.5), rtol=0, atol=0)
        self.assertFalse(features.requires_grad)

    def test_encoder_is_frozen_eval_while_head_trains_xy_in_meters(self):
        model = self.model()
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertTrue(model.head[2].training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.parameters()))
        before = {key: value.clone() for key, value in model.encoder.state_dict().items()}
        image = torch.ones(2, 3, 180, 320)
        target = model.straight_prior[None].repeat(2, 1, 1) + torch.tensor([0.2, 0.5])
        loss = waypoint_loss(model(image), target, model.config)
        loss.backward()
        self.assertIsNone(model.encoder.patch.weight.grad)
        gradient = model.head[-1].bias.grad.reshape(5, 2)
        self.assertTrue(torch.all(gradient.abs().sum(0) > 0))
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.01)
        optimizer.step()
        self.assertTrue(all(torch.equal(before[key], value) for key, value in model.encoder.state_dict().items()))
        with torch.no_grad():
            model.head[-1].weight.zero_()
            model.head[-1].bias.fill_(0.02)
        model.eval()
        torch.testing.assert_close(model(image)[0], model.straight_prior + 0.3)

    def test_pretrained_verification_precedes_topology_and_has_no_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "weights.pth"
            with patch("carla_tools.vit_waypoint._timm_backbone") as factory:
                with self.assertRaisesRegex(VitWaypointError, "file-required"):
                    VitWaypointModel()
                with self.assertRaisesRegex(VitWaypointError, "file-missing"):
                    VitWaypointModel(weights_path=weights, expected_weights_sha256="a" * 64, weights_url=WEIGHTS_REPOSITORY + "pin/model.bin")
                weights.write_bytes(b"not official weights")
                with self.assertRaisesRegex(VitWaypointError, "sha256-mismatch"):
                    VitWaypointModel(weights_path=weights, expected_weights_sha256="a" * 64, weights_url=WEIGHTS_REPOSITORY + "pin/model.bin")
                factory.assert_not_called()

    def test_full_pretrained_state_is_loaded_strictly_with_complete_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "weights.pth"
            state = FixtureBackbone(VitConfig()).state_dict()
            torch.save(state, weights)
            digest = sha256_file(weights)
            model = VitWaypointModel(weights_path=weights, expected_weights_sha256=digest,
                                      weights_url=WEIGHTS_REPOSITORY + "fixture/model.bin", backbone_factory=FixtureBackbone)
            self.assertEqual(model.pretrained_provenance["sha256"], digest)
            self.assertTrue(all(torch.equal(state[key], value) for key, value in model.encoder.state_dict().items()))
            state.pop("patch.bias")
            torch.save(state, weights)
            with self.assertRaisesRegex(VitWaypointError, "architecture-mismatch"):
                VitWaypointModel(weights_path=weights, expected_weights_sha256=sha256_file(weights),
                                  weights_url=WEIGHTS_REPOSITORY + "fixture/model.bin", backbone_factory=FixtureBackbone)

    def test_invalid_tensor_token_and_config_inputs_fail(self):
        model = self.model()
        for bad in (torch.zeros(0, 3, 180, 320), torch.zeros(1, 3, 192, 320), torch.zeros(1, 3, 180, 320, dtype=torch.uint8),
                    torch.full((1, 3, 180, 320), float("nan"))):
            with self.assertRaises(VitWaypointError):
                model(bad)
        with patch.object(model.encoder, "forward_features", return_value=torch.zeros(1, 245, 767)):
            with self.assertRaisesRegex(VitWaypointError, "token-shape"):
                model.encode(torch.zeros(1, 3, 180, 320))
        with self.assertRaises(VitWaypointError):
            model.predict_features(torch.zeros(1, FEATURE_COUNT - 1))
        with self.assertRaises(VitWaypointError):
            dataclasses.replace(VitConfig(), backbone_id="some-other-backbone")
        with self.assertRaises(VitWaypointError):
            vit_config_from_dict({"epochs": 1})
        config = VitConfig()
        self.assertEqual(vit_config_from_dict(config.to_dict()), config)
        self.assertEqual(canonical_sha256(config.to_dict()), config.sha256)

    def test_math_sdpa_keeps_half_precision_large_attention_scores_finite(self):
        # DINO prefix tokens can have large activations. The previous manual
        # half score matmul overflows even though all inputs are finite.
        query = torch.full((1, 1, 2, 64), 1000, dtype=torch.float16)
        key = query.clone()
        value = torch.arange(128, dtype=torch.float16).reshape(1, 1, 2, 64)
        manual_scores = (query * (64 ** -0.5)) @ key.transpose(-2, -1)
        self.assertFalse(bool(torch.isfinite(manual_scores).all()))
        self.assertFalse(bool(torch.isfinite(manual_scores.softmax(-1)).all()))
        with sdpa_kernel(SDPBackend.MATH):
            attended = nn.functional.scaled_dot_product_attention(query, key, value)
        self.assertTrue(bool(torch.isfinite(attended).all()))
        torch.testing.assert_close(attended[0, 0, 0], (value.float().mean(dim=2))[0, 0].half(), rtol=0, atol=0)
        model = self.model()
        with patch("carla_tools.vit_waypoint.sdpa_kernel", wraps=sdpa_kernel) as selected_backend:
            model.encode(torch.zeros(1, 3, 180, 320))
        selected_backend.assert_called_once_with(SDPBackend.MATH)
        self.assertIn("SDPA-math", model_contract()["encoder"]["attention"])

    def test_nonfinite_tokens_have_distinct_numerical_diagnostics(self):
        model = self.model()
        tokens = torch.full((1, 245, 768), float("nan"))
        with patch.object(model.encoder, "forward_features", return_value=tokens):
            with self.assertRaisesRegex(VitWaypointError, "vit-nonfinite-patch-tokens") as raised:
                model.encode(torch.zeros(1, 3, 180, 320))
        self.assertIn('"finite_count": 0', str(raised.exception))
        self.assertIn('"shape": [1, 245, 768]', str(raised.exception))
        self.assertNotIn("token-shape", str(raised.exception))

    def test_predict_rgb_is_stateless_and_invalid_paths_have_no_stale_points(self):
        model = self.model()
        rgb = np.zeros((360, 640, 3), dtype=np.uint8)
        passed = model.predict_rgb(rgb)
        self.assertEqual(passed["status"], "PASS")
        failed = model.predict_rgb(rgb.astype(np.float32))
        self.assertEqual(failed["status"], "FAIL")
        self.assertIsNone(failed["waypoints_ego_m"])
        self.assertEqual(model.predict_rgb(rgb)["waypoints_ego_m"], passed["waypoints_ego_m"])


if __name__ == "__main__":
    unittest.main()
