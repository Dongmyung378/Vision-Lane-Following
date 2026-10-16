"""Verify measured RGB composition and unchanged development replay contracts."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from carla_tools.baseline import load_baseline_config
from carla_tools.controller import PurePursuitController, VehicleGeometry, load_controller_config
from carla_tools.evaluation import load_evaluation_config
from carla_tools.learning_data import load_learning_config, preprocess_rgb
from carla_tools.pipeline_latency import STAGES, validate_pipeline_timing
from carla_tools.runtime_driving import (RuntimeCheckpointPredictor, RuntimeTrialPolicy,
    SCOPE, REFERENCE_IDENTITY, runtime_source_hashes, validate_runtime_episode)
from scripts.prepare_kaggle_runtime_driving import prepare_runtime_driving, EXTRA_SOURCES
from scripts.prepare_kaggle_week1 import NOTEBOOK_ID, NOTEBOOK_TITLE
from scripts.run_runtime_pipeline import runtime_trial_matrix

try:
    from tests.test_baseline_validation import write_baseline, save_sources
    from tests.test_near_correction import painted_scene
except ModuleNotFoundError:
    from test_baseline_validation import write_baseline, save_sources
    from test_near_correction import painted_scene


class FixtureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.input = None
    def forward(self, image):
        self.input = image.clone()
        return image.new_tensor([[[float(x), .5] for x in (3, 6, 9, 12, 15)]])


def fixture_predictor(correction="near"):
    identity = {"variant": "hybrid_geometry", "checkpoint_sha256": "fixture-only",
        "model_version": "fixture-only", "device": "cpu"}
    native = SimpleNamespace(model=FixtureModel().eval(), config=load_learning_config(), identity=identity)
    with patch("carla_tools.runtime_driving.NativeCheckpointPredictor", return_value=native):
        return RuntimeCheckpointPredictor("fixture", expected_sha256="fixture-only", correction=correction)


class RuntimeDrivingTests(unittest.TestCase):
    def test_profiled_input_is_exact_shared_learned_preprocessing(self):
        image = painted_scene()
        for correction in ("near", "path"):
            predictor = fixture_predictor(correction)
            result = predictor.predict(image)
            self.assertEqual(result["status"], "PASS", result)
            self.assertTrue(torch.equal(predictor.raw.model.input, preprocess_rgb(image, predictor.raw.config)[None]))
            self.assertEqual(result["waypoints_ego_m"][2:], [[float(x), .5] for x in (9, 12, 15)])
            with self.assertRaises(ValueError): predictor.predict(image, width=320)

    def test_full_command_measures_all_stages_without_pose_or_route_input(self):
        predictor = fixture_predictor()
        image = painted_scene()
        bgra = np.empty((360, 640, 4), np.uint8)
        bgra[:, :, :3], bgra[:, :, 3] = image[:, :, ::-1], 255
        geometry = VehicleGeometry(2.8, -1.4, 1.4, 70., ((0., 1.), (100., 1.)), 1.5)
        controller = PurePursuitController(load_controller_config(), geometry)
        policy = RuntimeTrialPolicy(load_baseline_config(), load_evaluation_config(), {}, {}, predictor=predictor)
        command, prediction, legacy = policy.command(SimpleNamespace(raw_data=bgra.tobytes(), width=640, height=360), 1., controller)
        self.assertIsNotNone(command)
        timing = validate_pipeline_timing(prediction["pipeline_timing"])
        self.assertEqual(timing["executed_stages"], list(STAGES))
        self.assertEqual(timing["device"], "cpu")
        self.assertEqual(set(legacy), {"preprocess_ms", "inference_ms", "controller_ms", "pipeline_ms"})
        self.assertAlmostEqual(legacy["pipeline_ms"], sum(legacy[k] for k in ("preprocess_ms", "inference_ms", "controller_ms")))

    def test_reference_replays_original_labels_and_rejects_source_or_start_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "reference"
            metadata, rows = write_baseline(root, method="privileged_reference")
            metadata.update(scope=SCOPE, native_model=REFERENCE_IDENTITY, native_source_sha256=runtime_source_hashes(),
                requested_initial_reference_m=0., detailed_latency_required=False)
            save_sources(root, metadata, rows)
            self.assertEqual(validate_runtime_episode(root, None, expected_initial_reference_m=0.)["status"], "PASS")
            self.assertEqual(validate_runtime_episode(root, None, expected_initial_reference_m=2.)["status"], "FAIL")
            metadata["native_source_sha256"]["carla_tools/runtime_driving.py"] = "0" * 64
            save_sources(root, metadata, rows)
            self.assertEqual(validate_runtime_episode(root, None, expected_initial_reference_m=0.)["status"], "FAIL")

    def test_plan_keeps_prior_correction_and_candidate_on_identical_states(self):
        routes = [{"case": case, "route": {"route_id": case}} for case in ("straight", "left_curve", "right_curve")]
        trials = runtime_trial_matrix(routes, "hybrid_geometry_path_near")
        self.assertEqual(len(trials), 81)
        self.assertEqual(len({t["episode"] for t in trials}), 81)
        for index in range(0, 81, 3):
            states = [{k: v for k, v in t.items() if k not in ("variant", "episode")} for t in trials[index:index+3]]
            self.assertEqual(states[0], states[1])
            self.assertEqual(states[1], states[2])
        with self.assertRaises(ValueError): runtime_trial_matrix(routes, "hybrid_geometry_near")

    def test_bundle_keeps_notebook_identity_and_embeds_complete_runtime_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps(dict(id=NOTEBOOK_ID, title=NOTEBOOK_TITLE, is_private=True)))
            routes = root / "routes.json"
            routes.write_text(json.dumps({"split": "dev", "final_town_used": False, "routes": [{}, {}, {}]}))
            bundle = prepare_runtime_driving(metadata, root / "bundle", dev_routes=routes,
                candidate_sha256="a" * 64, candidate_bytes=100, reference_archive_sha256="b" * 64,
                reference_checkpoint_sha256="c" * 64, reference_version=1,
                candidate_archive_sha256="d" * 64, candidate_version=2)
            submission = json.loads((root / "bundle/kernel-metadata.json").read_text())
            self.assertEqual((submission["id"], submission["title"]), (NOTEBOOK_ID, NOTEBOOK_TITLE))
            self.assertFalse(bundle["creates_new_notebook"])
            self.assertFalse(bundle["creates_dataset"])
            code = (root / "bundle/native.py").read_text()
            self.assertEqual(hashlib.sha256(code.encode()).hexdigest(), bundle["code_sha256"])
            nodes = []
            for node in ast.parse(code).body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"_project_payload", "PROJECT_SOURCES"} for t in node.targets):
                    nodes.append(node)
                elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and
                        isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name) and
                        node.value.func.value.id == "PROJECT_SOURCES" and node.value.func.attr == "update"):
                    nodes.append(node)
            isolated = "import base64,lzma,json,hashlib\n" + "\n".join(ast.unparse(n) for n in nodes)
            namespace = {}
            exec(compile(isolated, "fixture-review", "exec"), namespace)
            self.assertTrue(set(EXTRA_SOURCES) <= namespace["PROJECT_SOURCES"].keys())
            for name, digest in bundle["source_sha256"].items():
                self.assertEqual(hashlib.sha256(namespace["PROJECT_SOURCES"][name].encode()).hexdigest(), digest)
            self.assertIn("scripts/run_runtime_pipeline.py", code)
            self.assertEqual(bundle["evaluated_methods"], ["privileged_reference", "hybrid_geometry_near", "hybrid_geometry_path_near"])


if __name__ == "__main__":
    unittest.main()
