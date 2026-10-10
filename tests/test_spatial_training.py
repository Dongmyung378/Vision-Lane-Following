"""Strict checkpoint contracts and conservative DEV-only candidate promotion."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from carla_tools.learning_data import load_learning_config
from carla_tools.spatial_training import (_contract, choose_candidate,
                                          load_spatial_checkpoint, spatial_source_hashes,validate_reused_baseline)
from carla_tools.spatial_waypoint import create_spatial_model
from carla_tools.waypoint_model import PRETRAINED_URL


def evaluation(variant, far=0.8, curve=1.0, tail=2.0, latency=15.0):
    return {"status": "PASS", "variant": variant, "split": "dev", "final_town_used": False,
            "manifest_sha256": "a" * 64, "config_sha256": "b" * 64, "sample_count": 100,
            "performance_target_passed": far <= 1.0, "latency_ms": {"p95": latency},
            "metrics": [{"case": case, "condition": "all", "coverage": 1.0,
                         "far_mean_m": far if case == "all" else curve, "far_p95_m": tail}
                        for case in ("all", "left_curve", "right_curve")]}


class SpatialTrainingTests(unittest.TestCase):
    def setUp(self):
        self.reports = {name: evaluation(name, far=1.1 if name == "resnet" else 0.8)
                        for name in ("resnet", "spatial", "hybrid")}

    def test_worse_curve_tail_or_latency_cannot_be_promoted(self):
        self.reports["spatial"]["performance_target_passed"] = False
        for field in ("tail", "latency"):
            reports = copy.deepcopy(self.reports)
            reports["hybrid"] = evaluation("hybrid", tail=2.1 if field == "tail" else 2.,
                                            latency=51.0 if field == "latency" else 15.)
            self.assertEqual(choose_candidate(reports)["selected_candidate"], "resnet")

    def test_best_eligible_arm_is_a_candidate_without_closed_loop_claim(self):
        self.reports["hybrid"] = evaluation("hybrid", far=0.7)
        decision = choose_candidate(self.reports)
        self.assertEqual(decision["selected_candidate"], "hybrid")
        self.assertFalse(decision["closed_loop_verified"])
        self.assertFalse(decision["final_town_used"])

    def test_partial_or_different_development_set_is_rejected(self):
        self.reports["hybrid"]["sample_count"] = 99
        with self.assertRaisesRegex(ValueError, "evaluation-mismatch"):
            choose_candidate(self.reports)

    def test_reused_reference_keeps_original_provenance_and_refuses_different_core(self):
        config=load_learning_config()
        source=spatial_source_hashes()
        common={"status":"PASS","variant":"resnet","config_sha256":config.sha256,
                "manifest_sha256":"a"*64,"checkpoint_sha256":"c"*64,"source_sha256":source,
                "final_town_used":False,"fixture_injection":False}
        train=dict(common,dev_sample_count=100,runtime={"device":"cuda","gpu":"Tesla T4"},
                   training_protocol="full-encoder-finetuning-identical-augmentation-and-budget")
        evaluated=dict(common,split="dev",sample_count=100)
        value={"training":train,"evaluation":evaluated,"provenance":{"original_version":1}}
        self.assertEqual(validate_reused_baseline(value,config,"a"*64),(train,evaluated))
        changed=copy.deepcopy(value)
        changed["training"]["source_sha256"]["carla_tools/waypoint_model.py"]="d"*64
        changed["evaluation"]["source_sha256"]=changed["training"]["source_sha256"]
        with self.assertRaisesRegex(ValueError,"core-source-mismatch"):
            validate_reused_baseline(changed,config,"a"*64)
        with self.assertRaisesRegex(ValueError,"protocol-mismatch"):
            validate_reused_baseline(value,config,"b"*64)

    def test_checkpoint_load_is_bit_exact_and_rejects_contract_source_and_manifest_changes(self):
        config = load_learning_config()
        model = create_spatial_model("hybrid", config=config, pretrained=False)
        contract, digest = _contract("hybrid", config)
        checkpoint = {"schema_version": 2, "checkpoint_role": "selected-spatial-model",
                      "variant": "hybrid", "config": config.to_dict(), "config_sha256": config.sha256,
                      "model_contract": contract, "model_contract_sha256": digest,
                      "source_sha256": spatial_source_hashes(), "manifest_sha256": "a" * 64,
                      "pretrained_provenance": {"pretrained": True, "hash_verified": True,
                                                "url": PRETRAINED_URL, "sha256": "f37072fd" + "0" * 56},
                      "model_state": model.state_dict()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(checkpoint, path)
            restored, _, _ = load_spatial_checkpoint(path, manifest_sha256="a" * 64)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "dataset-mismatch"):
                load_spatial_checkpoint(path, manifest_sha256="c" * 64)
            for field, value in (("variant", "spatial"), ("source_sha256", {}), ("config_sha256", "bad")):
                changed = dict(checkpoint, **{field: value})
                torch.save(changed, path)
                with self.assertRaises(ValueError):
                    load_spatial_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
