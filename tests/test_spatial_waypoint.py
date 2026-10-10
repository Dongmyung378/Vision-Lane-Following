"""CPU checks for location information, real gradients and model contracts."""

import unittest

import torch
from torch import nn

from carla_tools.spatial_waypoint import (SpatialHead, create_spatial_model,
                                         fixed_2d_position, spatial_model_contract)
from carla_tools.waypoint_model import WaypointModelError

torch.set_num_threads(1)


class SpatialWaypointTests(unittest.TestCase):
    def test_initial_prediction_is_the_shared_prior_and_encoder_bn_stays_frozen(self):
        for variant in ("spatial", "hybrid"):
            model = create_spatial_model(variant, pretrained=False)
            prediction = model(torch.zeros(2, 3, 180, 320))
            expected = torch.tensor([[3., 0.], [6., 0.], [9., 0.], [12., 0.], [15., 0.]])
            torch.testing.assert_close(prediction, expected[None].expand(2, 5, 2))
            self.assertTrue(all(not m.training for m in model.encoder.modules() if isinstance(m, nn.BatchNorm2d)))
            self.assertLess(sum(p.numel() for p in model.parameters()), 13_000_000)

    def test_real_hybrid_gradients_reach_cnn_and_attention_after_output_unlocks(self):
        model = create_spatial_model("hybrid", pretrained=False)
        nn.init.normal_(model.head.output[-1].weight, std=0.01)
        prediction = model(torch.randn(2, 3, 180, 320))
        prediction.square().mean().backward()
        for parameter in (model.encoder.conv1.weight, model.head.blocks[0].query.weight,
                          model.head.cross_attention.key.weight, model.head.queries):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_fixed_positions_are_distinct_for_every_location_and_reproducible(self):
        first = fixed_2d_position()
        torch.testing.assert_close(first, fixed_2d_position(), rtol=0, atol=0)
        self.assertEqual(tuple(first.shape), (1, 60, 128))
        self.assertEqual(len(torch.unique(first[0], dim=0)), 60)

    def test_attention_uses_location_information_and_eval_is_deterministic(self):
        torch.manual_seed(4)
        head = SpatialHead("hybrid", 0.2).eval()
        nn.init.normal_(head.output[-1].weight, std=0.01)
        features = torch.randn(1, 512, 6, 10)
        with torch.inference_mode():
            original, repeat = head(features), head(features)
            shuffled = head(features.flip(-1))
        torch.testing.assert_close(original, repeat, rtol=0, atol=0)
        self.assertGreater(float((original - shuffled).abs().max()), 1e-6)

    def test_contracts_disclose_different_heads_and_share_rgb_output_geometry(self):
        resnet = spatial_model_contract("resnet")
        for variant in ("spatial", "hybrid"):
            contract = spatial_model_contract(variant)
            self.assertEqual(contract["input"], resnet["input"])
            self.assertEqual(contract["output"], resnet["output"])
            self.assertEqual(contract["head"]["tokens"], 60)
            self.assertNotEqual(contract["architecture"], resnet["architecture"])

    def test_invalid_inputs_and_variant_fail_before_encoder_execution(self):
        model = create_spatial_model("hybrid", pretrained=False)
        for image in (torch.zeros(1, 3, 360, 640), torch.full((1, 3, 180, 320), float("nan")),
                      torch.zeros(0, 3, 180, 320), torch.zeros(1, 3, 180, 320, dtype=torch.int32)):
            with self.assertRaises(WaypointModelError):
                model(image)
        with self.assertRaises(WaypointModelError):
            create_spatial_model("unknown", pretrained=False)


if __name__ == "__main__":
    unittest.main()
