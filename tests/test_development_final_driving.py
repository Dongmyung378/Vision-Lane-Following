"""Paired development phase and exact representative-model transport checks."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

from carla_tools.baseline import load_baseline_config
from carla_tools.controller import VehicleGeometry, load_controller_config
from carla_tools.controller_study import ControllerStudyConfig, build_controller
from carla_tools.development_final_driving import (PHASES, episode_protocol, final_source_hashes,
    freeze_final_protocol, planned_final_trials, run_final_trial, validate_final_episode)
from carla_tools.development_selection import ARMS
from carla_tools.development_study import ProfiledBaselinePredictor, digest
from scripts.prepare_kaggle_development_final import prepare_development_final, unpack_final_checkpoints
from scripts.prepare_kaggle_week1 import NOTEBOOK_ID, NOTEBOOK_TITLE
from tests.test_development_study import routes


def models():
    return {"incumbent": "a" * 64, "control": "a" * 64, "continued": "b" * 64,
        "appearance_robust": "c" * 64, "B0": None}


def offline_models():
    return {"incumbent": "a" * 64, "control": "a" * 64, "continued": "d" * 64, "appearance_robust": "e" * 64}


def controllers():
    default = load_controller_config()
    candidate = ControllerStudyConfig(base=replace(default, lookahead_base_m=2.5, lookahead_min_m=3., lookahead_max_m=6.))
    return {arm: (ControllerStudyConfig(base=default) if arm == "incumbent" else candidate).to_dict() for arm in ARMS}


def frozen(phase="final_nominal", *, source_routes=None):
    return freeze_final_protocol(routes() if source_routes is None else source_routes, models(), controllers(), phase=phase,
        representative_seed=42, training_comparison_sha256="f" * 64, parent_study_sha256="1" * 64,
        offline_checkpoint_sha256=offline_models(), training_audit_sha256="2" * 64)


class FinalProtocolTests(unittest.TestCase):
    def test_guarded_fusion_is_bound_per_arm_without_changing_paired_states(self):
        from carla_tools.guarded_path_correction import guarded_preprocessing_config, path_preprocessing_config
        baseline = frozen()
        mapping = {arm: None if arm == "B0" else path_preprocessing_config() if arm == "incumbent"
            else guarded_preprocessing_config() for arm in ARMS}
        keyword = dict(phase="final_nominal", representative_seed=42, training_comparison_sha256="f" * 64,
            parent_study_sha256="1" * 64, offline_checkpoint_sha256=offline_models(), training_audit_sha256="2" * 64)
        guarded = freeze_final_protocol(routes(), models(), controllers(), preprocessing_configs=mapping, **keyword)
        self.assertEqual(guarded["planned_trials"], baseline["planned_trials"])
        self.assertEqual(guarded["preprocessing_configs"], mapping)
        for arm in ("incumbent", "continued", "B0"):
            wrong = copy.deepcopy(mapping)
            wrong[arm] = guarded_preprocessing_config() if arm in ("incumbent", "B0") else path_preprocessing_config()
            with self.assertRaises(ValueError):
                freeze_final_protocol(routes(), models(), controllers(), preprocessing_configs=wrong, **keyword)

    def test_two_phases_have_exact_identical_arm_pairs_and_shared_context(self):
        nominal, robust = frozen(), frozen("final_robustness")
        for protocol in (nominal, robust):
            plan = protocol["planned_trials"]
            self.assertEqual(len(plan), PHASES[protocol["phase"]])
            self.assertEqual(len({trial["episode"] for trial in plan}), len(plan))
            for index in range(0, len(plan), 5):
                chunk = plan[index:index + 5]
                state = lambda trial: {key: trial[key] for key in
                    ("case", "route_id", "initial_reference_m", "condition", "environment")}
                self.assertEqual({trial["arm"] for trial in chunk}, set(ARMS))
                self.assertTrue(all(state(trial) == state(chunk[0]) for trial in chunk))
            self.assertEqual([trial["arm"] for trial in plan[:5]], list(reversed([trial["arm"] for trial in plan[5:10]])))
            self.assertFalse(protocol["final_town_used"])
            self.assertFalse(protocol["final_conditions_used"])
        different = {key for key in nominal if nominal[key] != robust[key]}
        self.assertEqual(different, {"phase", "expected_episodes", "planned_trials"})

    def test_primary_offline_and_compatible_native_hashes_remain_separate(self):
        protocol = frozen()
        self.assertEqual(protocol["checkpoint_sha256"]["continued"], "b" * 64)
        self.assertEqual(protocol["offline_checkpoint_sha256"]["continued"], "d" * 64)
        self.assertEqual(protocol["training_audit_sha256"], "2" * 64)
        trial = next(trial for trial in protocol["planned_trials"] if trial["arm"] == "continued")
        per_episode = episode_protocol(protocol, trial)
        self.assertEqual(per_episode["checkpoint_sha256"], "b" * 64)
        self.assertEqual(per_episode["parent_final_protocol_sha256"], digest(protocol))
        self.assertEqual(protocol["checkpoint_sha256"], models())

    def test_unpaired_controller_model_or_undeclared_episode_rejected(self):
        keyword = dict(phase="final_nominal", representative_seed=42, training_comparison_sha256="f" * 64,
            parent_study_sha256="1" * 64, offline_checkpoint_sha256=offline_models(), training_audit_sha256="2" * 64)
        wrong_models = models(); wrong_models["control"] = "3" * 64
        with self.assertRaises(ValueError):
            freeze_final_protocol(routes(), wrong_models, controllers(), **keyword)
        wrong_controllers = controllers(); wrong_controllers["B0"] = wrong_controllers["incumbent"]
        with self.assertRaises(ValueError):
            freeze_final_protocol(routes(), models(), wrong_controllers, **keyword)
        protocol = frozen(); trial = copy.deepcopy(protocol["planned_trials"][0]); trial["initial_reference_m"] = 99.
        with self.assertRaises(ValueError):
            episode_protocol(protocol, trial)

    def test_reserved_environment_is_not_available_through_phase_switch(self):
        protocol = frozen("final_robustness")
        self.assertEqual({trial["environment"]["weather"] for trial in protocol["planned_trials"]},
            {"ClearNoon", "CloudyNoon", "SoftRainNoon"})
        self.assertEqual({trial["initial_reference_m"] for trial in protocol["planned_trials"]}, {4.})
        self.assertTrue(all(trial["condition"]["name"] == "center" for trial in protocol["planned_trials"]))


def write_archive(root, *, changed_representative=False):
    payload = {"initial": b"fixture-incumbent", "continued": b"fixture-continued", "appearance_robust": b"fixture-appearance"}
    native = {arm: hashlib.sha256(payload["initial" if arm in ("incumbent", "control") else arm]).hexdigest() for arm in ARMS[:-1]}
    native["B0"] = None
    primary = {"incumbent": native["incumbent"], "control": native["control"], "continued": "d" * 64, "appearance_robust": "e" * 64}
    members = {arm: "experiment/initial/checkpoint.pt" if arm in ("incumbent", "control")
        else f"experiment/{arm}/refined_checkpoint.pt" for arm in ARMS[:-1]}
    comparison = {"status": "PASS", "representative_seed": 43 if changed_representative else 42,
        "equal_budget_verified": True, "fixture_injection": False, "final_town_used": False,
        "initial_checkpoint_sha256": native["incumbent"],
        "compatible_checkpoint_sha256": {arm: native[arm] for arm in ("continued", "appearance_robust")},
        "representative_checkpoint_sha256": {arm: primary[arm] for arm in ("continued", "appearance_robust")}}
    comparison_bytes = json.dumps(comparison).encode()
    archive = root / "training.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for name, value in payload.items():
            stream.writestr("experiment/initial/checkpoint.pt" if name == "initial"
                else f"experiment/{name}/refined_checkpoint.pt", value)
        stream.writestr("experiment/comparison.json", comparison_bytes)
        stream.writestr("../../unused.txt", b"not-extracted")
    return archive, dict(expected_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        expected_archive_bytes=archive.stat().st_size, checkpoint_sha256=native, offline_checkpoint_sha256=primary,
        checkpoint_members=members, comparison_sha256=hashlib.sha256(comparison_bytes).hexdigest(), representative_seed=42)


class FinalTransportTests(unittest.TestCase):
    def test_exact_three_models_and_original_comparison_extracted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); archive, keyword = write_archive(root)
            result = unpack_final_checkpoints(archive, root / "models", **keyword)
            observed = json.loads(result.read_text())
            self.assertEqual(set(observed), set(ARMS[:-1]))
            self.assertEqual(observed["incumbent"]["path"], observed["control"]["path"])
            self.assertFalse((root / "unused.txt").exists())
            self.assertEqual(hashlib.sha256((root / "models/comparison.json").read_bytes()).hexdigest(), keyword["comparison_sha256"])
            for value in observed.values():
                self.assertEqual(hashlib.sha256(Path(value["path"]).read_bytes()).hexdigest(), value["sha256"])

    def test_changed_representative_archive_or_native_model_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); archive, keyword = write_archive(root, changed_representative=True)
            with self.assertRaisesRegex(ValueError, "lineage"):
                unpack_final_checkpoints(archive, root / "models", **keyword)
            self.assertFalse((root / "models").exists())
            keyword["expected_archive_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                unpack_final_checkpoints(archive, root / "other", **keyword)

    def test_bundle_pins_original_notebook_phase_models_and_all_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); protocol = frozen("final_robustness")
            metadata, route, freeze = root / "metadata.json", root / "routes.json", root / "protocol.json"
            metadata.write_text(json.dumps({"id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE, "is_private": True}))
            route.write_text(json.dumps(routes())); freeze.write_text(json.dumps(protocol))
            members = {arm: "experiment/initial/checkpoint.pt" if arm in ("incumbent", "control")
                else f"experiment/{arm}/refined_checkpoint.pt" for arm in ARMS[:-1]}
            result = prepare_development_final(metadata, root / "bundle", dev_routes=route, protocol_file=freeze,
                training_archive_sha256="a" * 64, training_archive_bytes=100, training_version=21, checkpoint_members=members)
            self.assertFalse(result["creates_new_notebook"])
            self.assertEqual(result["expected_episodes"], 90)
            self.assertEqual(result["checkpoint_sha256"], models())
            self.assertEqual(result["offline_checkpoint_sha256"], offline_models())
            code = (root / "bundle/native.py").read_text(encoding="utf-8")
            compile(code, "native.py", "exec")
            self.assertIn("unpack_final_checkpoints", code)
            self.assertIn("run_logged_study", code)
            self.assertTrue(set(final_source_hashes()) <= set(result["source_sha256"]))


class FinalNativeReplayTests(unittest.TestCase):
    def test_guarded_original_RGB_capture_replay_media_and_typed_rejection(self):
        """CPU simulator stand-in: source/control integrity, no performance evidence."""
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from carla_tools.config import load_collection_config
        from carla_tools.contracts import route_sha256
        from carla_tools.driving_media import export_driving_media
        from carla_tools.evaluation import load_evaluation_config
        from carla_tools.guarded_path_correction import guarded_preprocessing_config, path_preprocessing_config, wrap_preprocessing
        from tests.test_runtime_driving import fixture_predictor
        from tests.test_baseline import scene, np
        with patch.object(sys, "path", [str(Path(__file__).parent), *sys.path]):
            from tests.test_driving_native import NativeStandInWorld, native_module
            from tests.test_driving_route import dense_route
        previous_threads = torch.get_num_threads(); torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous_threads)
        image, _ = scene()
        bgra = np.empty((360, 640, 4), dtype=np.uint8)
        bgra[:, :, :3], bgra[:, :, 3] = image[:, :, ::-1], 255
        raw_data = bgra.tobytes()
        class PaintedWorld(NativeStandInWorld):
            def spawn_actor(self, blueprint, transform, *, attach_to, attachment_type=None):
                import math
                actor = super().spawn_actor(blueprint, transform, attach_to=attach_to, attachment_type=attachment_type)
                if blueprint.id == "sensor.camera.rgb":
                    original_transform = actor.get_transform
                    def readback():
                        result = original_transform(); matrix = result.get_matrix()
                        result.location = SimpleNamespace(**{axis: matrix[index][3] for index, axis in enumerate(("x", "y", "z"))})
                        result.rotation = SimpleNamespace(pitch=math.degrees(math.asin(matrix[2][0])),
                            yaw=math.degrees(math.atan2(matrix[1][0], matrix[0][0])),
                            roll=math.degrees(math.atan2(-matrix[2][1], matrix[2][2])))
                        return result
                    actor.get_transform = readback
                return actor

            def tick(self, timeout):
                camera = self.actors.get(101)
                if camera is None or not camera.is_listening:
                    return super().tick(timeout)
                original = camera.callback
                camera.callback = lambda image: original(SimpleNamespace(**dict(vars(image), raw_data=raw_data)))
                try:
                    return super().tick(timeout)
                finally:
                    camera.callback = original
        source_routes = routes(); route = dense_route()
        for anchor in route["centerline_anchor_sequence"]:
            anchor["world_rotation"] = {"pitch": 0., "yaw": 0., "roll": 0.}
        route["start"] = copy.deepcopy(route["centerline_anchor_sequence"][0])
        route["route_sha256"] = route_sha256(route); source_routes["routes"][0]["route"] = route
        mapping = {arm: None if arm == "B0" else path_preprocessing_config() if arm == "incumbent"
            else guarded_preprocessing_config() for arm in ARMS}
        protocol = freeze_final_protocol(source_routes, models(), controllers(), phase="final_nominal",
            representative_seed=42, training_comparison_sha256="f" * 64, parent_study_sha256="1" * 64,
            offline_checkpoint_sha256=offline_models(), training_audit_sha256="2" * 64, preprocessing_configs=mapping)
        trial = next(row for row in protocol["planned_trials"] if row["arm"] == "control" and row["case"] == "straight"
            and row["initial_reference_m"] == 0. and row["condition"]["name"] == "center")
        baseline = load_baseline_config()
        for rejected in (False, True):
            with self.subTest(controller_rejected=rejected), tempfile.TemporaryDirectory() as directory:
                if rejected:
                    bgra[:, :, :3] = 50; bgra[0, 0, :3] = [1, 2, 3]; raw_data = bgra.tobytes()
                base = fixture_predictor("path")
                base.raw.identity["checkpoint_sha256"] = "a" * 64; base._identity["checkpoint_sha256"] = "a" * 64
                if rejected:
                    base.raw.model.forward = lambda image: image.new_tensor([[[3., 0.], [1., 3.], [.1, 4.], [.1, 6.], [.1, 9.]]])
                predictor = wrap_preprocessing(base, mapping["control"])
                destination = Path(directory) / "episode"
                with patch("carla_tools.driving.verify_geometry_route_in_world", return_value=None):
                    execution = run_final_trial(route, trial, destination, load_collection_config(Path("configs/collection_d4.json")),
                        predictor=predictor, protocol=protocol, carla_module=native_module(PaintedWorld()),
                        baseline_config=baseline, evaluation_config=replace(load_evaluation_config(), timeout_seconds=10.))
                self.assertEqual(execution["status"], "PASS", execution)
                checked = validate_final_episode(destination, predictor, trial, protocol=protocol, routes=source_routes)
                self.assertEqual(checked["status"], "PASS", checked)
                self.assertEqual(export_driving_media(destination)["status"], "PASS")
                metadata = json.loads((destination / "episode.json").read_text())
                rows = [json.loads(line) for line in (destination / "steps.jsonl").read_text().splitlines()]
                self.assertEqual(metadata["native_model"]["correction"], "guarded_path")
                self.assertIn("guarded_path_correction", rows[0]["perception"]["diagnostics"])
                if rejected:
                    self.assertEqual(checked["frame_count"], 1)
                    self.assertFalse(checked["driving_success"])
                    self.assertEqual(rows[0]["perception"]["controller_rejection"]["error_code"], "controller-waypoints-not-ordered-forward")
                    from tests.test_baseline_validation import save_sources
                    rows[0]["perception"]["controller_rejection"]["error_code"] = "forged-error"
                    save_sources(destination, metadata, rows)
                    self.assertEqual(validate_final_episode(destination, predictor, trial, protocol=protocol, routes=source_routes)["status"], "FAIL")
                else:
                    wrong = copy.deepcopy(protocol)
                    wrong["preprocessing_configs"]["control"]["guard_config_sha256"] = "0" * 64
                    self.assertEqual(validate_final_episode(destination, predictor, trial, protocol=wrong, routes=source_routes)["status"], "FAIL")

    def test_final_B0_RGB_native_replay_is_bound_to_top_protocol_and_control(self):
        from tests.test_baseline_validation import save_sources
        from tests.test_development_study import DevelopmentNativeReplayTests
        DevelopmentNativeReplayTests.setUpClass()
        self.addCleanup(DevelopmentNativeReplayTests.tearDownClass)
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory) / "episode"
            shutil.copytree(DevelopmentNativeReplayTests.original, episode)
            source_routes = DevelopmentNativeReplayTests.routes
            protocol = frozen(source_routes=source_routes)
            metadata = json.loads((episode / "episode.json").read_text())
            rows = [json.loads(line) for line in (episode / "steps.jsonl").read_text().splitlines()]
            # Explicit CPU fixture retains the original fixture evaluation tolerance.
            protocol["evaluation_config_sha256"] = metadata["evaluation_config_sha256"]
            trial = next(trial for trial in protocol["planned_trials"] if trial["arm"] == "B0"
                and trial["case"] == "straight" and trial["initial_reference_m"] == 0. and trial["condition"]["name"] == "center")
            derived = episode_protocol(protocol, trial)
            config = ControllerStudyConfig(base=replace(load_controller_config(), lookahead_base_m=2.5, lookahead_min_m=3., lookahead_max_m=6.))
            metadata["development_study"] = {"trial": trial, "protocol_sha256": digest(derived),
                "source_sha256": DevelopmentNativeReplayTests.frozen["source_sha256"],
                "controller_config": config.to_dict(), "controller_config_sha256": config.sha256}
            controller = build_controller(config, VehicleGeometry(**metadata["vehicle_geometry"]))
            previous = None
            for row in rows:
                row["command"] = controller.step(row["perception"]["waypoints_ego_m"], row["speed_mps"], dt_seconds=.05)
                if previous is not None:
                    row["vehicle"]["applied_control"] = {key: previous["command"][key]
                        for key in ("throttle", "steer", "brake", "hand_brake", "reverse")}
                previous = row
            metadata["stop_control"]["steer"] = rows[-1]["vehicle"]["applied_control"]["steer"]
            for observation in metadata["stop_observations"]:
                observation["vehicle"]["applied_control"]["steer"] = metadata["stop_control"]["steer"]
            from carla_tools.evaluation import evaluate_episode, evaluation_config_from_dict
            metadata["evaluation_metrics"] = evaluate_episode(metadata, rows, metadata["events"],
                config=evaluation_config_from_dict(metadata["evaluation_config"]))
            save_sources(episode, metadata, rows)
            predictor = ProfiledBaselinePredictor(load_baseline_config())
            result = validate_final_episode(episode, predictor, trial, protocol=protocol, routes=source_routes)
            self.assertEqual(result["status"], "PASS", result)
            self.assertEqual(result["frame_count"], 49)
            wrong = copy.deepcopy(protocol); wrong["source_sha256"]["carla_tools/development_final_driving.py"] = "0" * 64
            self.assertEqual(validate_final_episode(episode, predictor, trial, protocol=wrong, routes=source_routes)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
