"""The public RGB adapter identifies its actual checkpoint and fails closed."""

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from carla_tools.inference import InferenceError
from carla_tools.learning_data import load_learning_config
from carla_tools.spatial_inference import SpatialCheckpointPredictor
from carla_tools.spatial_training import _contract, spatial_source_hashes
from carla_tools.spatial_waypoint import create_spatial_model
from carla_tools.waypoint_model import PRETRAINED_URL

torch.set_num_threads(1)


class SpatialInferenceTests(unittest.TestCase):
    def test_checkpoint_identity_rgb_contract_and_failed_input(self):
        config = load_learning_config()
        model = create_spatial_model("hybrid", pretrained=False, config=config)
        contract, digest = _contract("hybrid", config)
        checkpoint = {"schema_version": 2, "checkpoint_role": "selected-spatial-model", "variant": "hybrid",
                      "epoch": 1, "selected_dev_loss": 0.1, "config": config.to_dict(), "config_sha256": config.sha256,
                      "model_contract": contract, "model_contract_sha256": digest, "source_sha256": spatial_source_hashes(),
                      "manifest_sha256": "a"*64, "pretrained_provenance": {"pretrained": True, "hash_verified": True,
                          "url": PRETRAINED_URL, "sha256": "f37072fd"+"0"*56}, "model_state": model.state_dict()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"fixture.pt"
            torch.save(checkpoint,path)
            actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            predictor = SpatialCheckpointPredictor(path, expected_checkpoint_sha256=actual_sha)
            result = predictor.predict_rgb(np.zeros((360,640,3),dtype=np.uint8))
            self.assertEqual(result["status"],"PASS")
            self.assertEqual(result["waypoints_ego_m"],[[3.,0.],[6.,0.],[9.,0.],[12.,0.],[15.,0.]])
            self.assertEqual(result["diagnostics"]["variant"],"hybrid")
            self.assertEqual(result["diagnostics"]["model_version"],"vlf-resnet18-hybrid-waypoint-v1")
            self.assertFalse(predictor.identity["closed_loop_verified"])
            changed_identity = predictor.identity
            changed_identity["variant"] = "tampered"
            self.assertEqual(predictor.identity["variant"], "hybrid")
            self.assertEqual(predictor.predict_rgb(np.zeros((180,320,3),dtype=np.uint8))["status"],"FAIL")
            with self.assertRaisesRegex(InferenceError,"sha256-mismatch"):
                SpatialCheckpointPredictor(path,expected_checkpoint_sha256="c"*64)


if __name__ == "__main__":
    unittest.main()
