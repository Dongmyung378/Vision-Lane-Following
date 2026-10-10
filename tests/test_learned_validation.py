"""Synthetic checkpoint replay fixtures test native learned artifact integrity.

The checkpoint loader is explicitly mocked; these tests make no learned-model
performance claim. Native geometry/controller/PNG/evaluation replay is real.
"""
from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from carla_tools.baseline import OpenCVLanePredictor, load_baseline_config
from carla_tools.contracts import canonical_sha256
from carla_tools.driving_route import _pose_matrix
from carla_tools.inference import _inference_source_hashes
from carla_tools.learning_data import load_learning_config
from carla_tools.training import runtime_source_hashes
from carla_tools.waypoint_model import model_contract
import carla_tools.learned_validation as gate

try:
    from tests.test_baseline_validation import write_baseline, save_sources
except ModuleNotFoundError:
    from test_baseline_validation import write_baseline, save_sources


class FixtureRuntime:
    def __init__(self,summary):
        self._summary = copy.deepcopy(summary)
        self.calls = []
        self.model = OpenCVLanePredictor(load_baseline_config())

    @property
    def summary(self):
        return copy.deepcopy(self._summary)

    def predict_rgb(self,rgb):
        if not isinstance(rgb,np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (360,640,3):
            raise AssertionError("Only the decoded native RGB array may enter inference")
        self.calls.append(rgb.copy())
        return self.model.predict(rgb.tobytes(),width=640,height=360)


class LearnedValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.episode = self.root/"episode"
        self.metadata,self.rows = write_baseline(self.episode)
        self.checkpoint = self.root/"synthetic-selected.pt"
        self.checkpoint.write_bytes(b"explicit synthetic checkpoint fixture, no trained weights")
        config = load_learning_config()
        contract = model_contract(config)
        self.summary = {"inference_version":"vlf-offline-rgb-inference-v1","mode":"learned",
            "checkpoint_role":"selected-model","checkpoint_sha256":hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            "checkpoint_epoch":14,"selected_epoch":14,"manifest_sha256":"1"*64,"config_sha256":config.sha256,
            "model_version":config.model_version,"config":config.to_dict(),"model_contract":contract,
            "model_contract_sha256":canonical_sha256(contract),"source_sha256":runtime_source_hashes(),
            "inference_source_sha256":_inference_source_hashes("learned"),
            "pretrained_provenance":{"fixture":"synthetic, no pretrained-performance claim"},
            "runtime":{"device":"cpu","torch":"fixture","numpy":str(np.__version__),"cuda":None}}
        pose = copy.deepcopy(self.rows[0]["vehicle"]["transform"])
        self.metadata.update(scope=gate.SCOPE,method="M1",prediction_inputs="rgb-only",initial_reference_m=0.,
            learned_model=copy.deepcopy(self.summary),learned_driving_source_sha256={"synthetic-source":"2"*64},
            perception_uses_route=False,perception_uses_vehicle_pose=False,
            initialization={"policy":"pinned-route-start-xy-and-tangent-preserve-settled-height",
                "spawn_settled_transform":pose,"center_requested_transform":copy.deepcopy(pose),
                "perturbed_requested_transform":copy.deepcopy(pose)})
        self.runtime = FixtureRuntime(self.summary)
        for row in self.rows:
            row["evaluation"] = copy.deepcopy(row["evaluation"])
        # Changing a metric's method changes its grouping, not any driving data.
        self.metadata["evaluation_metrics"]["method"] = "M1"
        self.changed()

    def changed(self):
        save_sources(self.episode,self.metadata,self.rows)

    def checked(self,**kwargs):
        with patch("carla_tools.inference.OfflineWaypointPredictor",return_value=self.runtime), \
                patch.object(gate,"learned_runtime_source_hashes",return_value={"synthetic-source":"2"*64}):
            return gate.validate_learned_episode(self.episode,self.checkpoint,**kwargs)

    def assert_failed(self,code):
        report = self.checked()
        self.assertEqual(report["status"],"FAIL",report)
        self.assertEqual(report["errors"][0]["code"],code,report)

    def test_full_original_RGB_and_shared_native_controller_roundtrip_preserves_sources(self):
        before = {path.relative_to(self.episode).as_posix():path.read_bytes() for path in self.episode.rglob("*") if path.is_file()}
        report = self.checked()
        self.assertEqual(report["status"],"PASS",report)
        self.assertEqual(report["metrics"]["independently_replayed_RGB_count"],len(self.rows))
        self.assertEqual(report["metrics"]["maximum_RGB_replay_coordinate_difference_m"],0.)
        self.assertEqual(len(self.runtime.calls),len(self.rows))
        self.assertTrue(report["checks"]["GT_not_supplied_to_predictor"])
        self.assertEqual(before,{path.relative_to(self.episode).as_posix():path.read_bytes() for path in self.episode.rglob("*") if path.is_file()})

    def test_rehashed_prediction_corruption_cannot_pass(self):
        self.rows[2]["perception"]["waypoints_ego_m"][0][1] += .01
        self.changed()
        self.assert_failed("learned-RGB-waypoint-replay-mismatch")

    def test_rehashed_controller_corruption_is_rejected_after_independent_RGB_replay(self):
        self.rows[2]["command"]["steer"] += .1
        self.changed()
        report = self.checked()
        self.assertEqual(report["status"],"FAIL",report)
        self.assertEqual(report["native_gate"]["errors"][0]["code"],"baseline-controller-replay-mismatch")
        self.assertEqual(len(self.runtime.calls),len(self.rows))

    def test_gt_corruption_cannot_change_model_inputs_or_become_fallback(self):
        self.rows[1]["tracking"]["label"]["waypoints_ego_m"][0][1] += 1.
        self.changed()
        report = self.checked()
        self.assertEqual(report["status"],"FAIL",report)
        self.assertEqual(report["native_gate"]["errors"][0]["code"],"baseline-original-tracking-replay-mismatch")
        self.assertEqual(len(self.runtime.calls),len(self.rows))

    def test_method_scope_privileged_flag_and_model_sha_mismatch_are_rejected(self):
        original = copy.deepcopy(self.metadata)
        cases = [("method","B0","learned-episode-method-scope-or-input-mismatch"),
                 ("perception_uses_route",True,"learned-episode-method-scope-or-input-mismatch")]
        for key,value,code in cases:
            self.metadata = copy.deepcopy(original);self.metadata[key] = value;self.changed();self.assert_failed(code)
        self.metadata = copy.deepcopy(original)
        self.metadata["learned_model"]["checkpoint_sha256"] = "0"*64
        self.changed();self.assert_failed("learned-model-checkpoint-or-source-mismatch")

    def test_recorded_training_state_is_not_an_accepted_selected_model(self):
        self.metadata["learned_model"]["checkpoint_role"] = "training-state"
        self.changed();self.assert_failed("learned-model-checkpoint-or-source-mismatch")

    def test_declared_start_is_bound_to_requested_original_route_XY(self):
        self.metadata["initial_reference_m"] = 2.
        self.changed()
        report = self.checked(expected_initial_reference_m=2.)
        self.assertEqual(report["status"],"FAIL")
        self.assertEqual(report["errors"][0]["code"],"learned-initialization-policy-mismatch")

    def test_same_frame_sensor_identity_corruption_is_rejected(self):
        self.rows[1]["sensor_frame"] += 1
        self.changed()
        report = self.checked()
        self.assertEqual(report["status"],"FAIL")
        self.assertEqual(report["native_gate"]["errors"][0]["code"],"baseline-frame-alignment-mismatch")

    def test_tight_replay_limit_and_failed_predictions_keep_no_path(self):
        original = copy.deepcopy(self.rows[0]["perception"])
        close = copy.deepcopy(original);close["waypoints_ego_m"][0][1] += gate.REPLAY_TOLERANCE_M*.5
        self.assertLessEqual(gate._compare_perception(close,original),gate.REPLAY_TOLERANCE_M)
        close["waypoints_ego_m"][0][1] += gate.REPLAY_TOLERANCE_M*2
        with self.assertRaises(ValueError): gate._compare_perception(close,original)
        failed = {"status":"FAIL","waypoints_ego_m":None,"error":"fixture","diagnostics":{},"timing_ms":0.}
        self.assertEqual(gate._compare_perception(failed,failed),0.)
        retained = dict(failed,waypoints_ego_m=original["waypoints_ego_m"])
        with self.assertRaises(ValueError): gate._compare_perception(retained,failed)

    def test_replay_sequence_refuses_missing_extra_or_wrong_RGB(self):
        record = {"rgb_sha256":hashlib.sha256(b"fixture").hexdigest(),"perception":{"status":"FAIL"}}
        replay = gate._RecordedPredictionReplay([record])
        with self.assertRaises(ValueError): replay.predict(b"wrong",width=640,height=360)
        self.assertEqual(replay.index,0)
        replay.predict(b"fixture",width=640,height=360)
        with self.assertRaises(ValueError): replay.predict(b"fixture",width=640,height=360)

    def test_complete_driving_failure_still_has_valid_episode_denominator(self):
        metadata,rows = write_baseline(self.root/"failed",fail_prediction=True)
        metadata.update(scope=gate.SCOPE,method="M1",prediction_inputs="rgb-only",initial_reference_m=0.,
            learned_model=self.summary,learned_driving_source_sha256={"synthetic-source":"2"*64},
            initialization=copy.deepcopy(self.metadata["initialization"]))
        metadata["evaluation_metrics"]["method"] = "M1"
        save_sources(self.root/"failed",metadata,rows)
        with patch("carla_tools.inference.OfflineWaypointPredictor",return_value=self.runtime), \
                patch.object(gate,"learned_runtime_source_hashes",return_value={"synthetic-source":"2"*64}):
            report = gate.validate_learned_episode(self.root/"failed",self.checkpoint)
        self.assertEqual(report["status"],"PASS",report)
        self.assertFalse(report["evaluation_metrics"]["driving_success"])
        self.assertEqual(report["evaluation_metrics"]["termination_reason"],"technical_failure")
        self.assertEqual(report["evaluation_metrics"]["sample_count"],1)


if __name__ == "__main__":
    unittest.main()
