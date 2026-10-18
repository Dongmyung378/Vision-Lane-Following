"""Paired-state, source, safety-bound, and seed-selection regression checks."""
import ast
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from carla_tools.contracts import route_sha256
from carla_tools.learning_data import load_learning_config, preprocess_rgb
from carla_tools.recovery_comparison import (ARMS, CASES, RecoveryRuntimePredictor,
    digest, freeze_driving_protocol, paired_trial_matrix, select_recovery_model,
    summarize_paired_driving)
from scripts.prepare_kaggle_recovery_driving import prepare_recovery_driving, unpack_recovery_archive
from scripts.prepare_kaggle_week1 import NOTEBOOK_ID, NOTEBOOK_TITLE


def routes():
    result = {"split": "dev", "final_town_used": False, "routes": []}
    for case in CASES:
        route = {"route_id": "synthetic_" + case, "town": "Town01"}
        route["route_sha256"] = route_sha256(route)
        result["routes"].append({"split": "dev", "case": case, "route": route})
    return result


def protocol():
    with patch("carla_tools.recovery_comparison.comparison_source_hashes", return_value={"synthetic.py": "a" * 64}):
        return freeze_driving_protocol(routes(), {arm: hashlib.sha256(arm.encode()).hexdigest() for arm in ARMS},
            "b" * 64, representative_seed=42)


def offline(arm, far=(.5, .5, .5), near=(.1, .1, .1)):
    groups = []
    for case, n, f in zip(CASES, near, far):
        groups.append({"case": case, "condition": "all", "weather": "all", "sample_count": 10,
            "predicted_count": 10, "failed_prediction_count": 0, "coverage": 1.,
            "near_mean_m": n, "far_mean_m": f, "far_p95_m": f,
            "mean_euclidean_m": (2 * n + 3 * f) / 5})
    groups.insert(0, {"case": "all", "condition": "all", "weather": "all", "sample_count": 30,
        "predicted_count": 30, "failed_prediction_count": 0, "coverage": 1.,
        "near_mean_m": sum(near) / 3, "far_mean_m": sum(far) / 3, "far_p95_m": max(far),
        "mean_euclidean_m": (2 * sum(near) + 3 * sum(far)) / 15})
    contract = {"input": "RGB-only", "synthetic": True}
    return {"status": "PASS", "split": "dev", "fixture_injection": False, "final_town_used": False,
        "manifest_sha256": "d" * 64, "config_sha256": hashlib.sha256(arm.encode()).hexdigest(),
        "checkpoint_sha256": protocol()["checkpoint_sha256"][arm], "sample_count": 30,
        "evaluation_contract": contract, "evaluation_contract_sha256": digest(contract),
        "common_dev_selection_sha256": "e" * 64, "metrics": groups, "latency_ms": {"p95": 10.}}


def evidence():
    frozen = protocol()
    records = []
    for trial in frozen["planned_trials"]:
        mean = .3 if trial["arm"] == "incumbent" else .2
        metrics = {"driving_success": True, "termination_reason": "route_completed",
            "collision_count": 0, "lane_invasion_count": 0, "minimum_body_lane_margin_m": .2,
            "lateral_abs_m": {"mean": mean, "p95": .7}}
        records.append(dict(trial, execution={"status": "PASS"}, gate={"status": "PASS", "metrics": metrics}))
    native = {"status": "PASS", "protocol_sha256": digest(frozen), "closed_loop_executed": True,
        "final_town_used": False, "checkpoint_sha256": frozen["checkpoint_sha256"],
        "source_sha256": frozen["source_sha256"], "planned_trials": frozen["planned_trials"], "episodes": records,
        "steady_processing_latency_ms": {arm: {"p95": 20.} for arm in ARMS}}
    reports = {arm: offline(arm, (.6, .6, .1) if arm == "incumbent" else (.2, .2, .2)) for arm in ARMS}
    return frozen, native, reports


class RecoveryComparisonTests(unittest.TestCase):
    def test_all_training_seeds_share_exact_27_states_and_order_alternates(self):
        trials = paired_trial_matrix(routes()["routes"])
        self.assertEqual(len(trials), 135)
        self.assertEqual(len({t["episode"] for t in trials}), 135)
        keys = ("case", "route_id", "initial_reference_m", "condition")
        for i in range(0, len(trials), len(ARMS)):
            chunk = trials[i:i + len(ARMS)]
            self.assertEqual({t["arm"] for t in chunk}, set(ARMS))
            self.assertTrue(all({k: t[k] for k in keys} == {k: chunk[0][k] for k in keys} for t in chunk))
        self.assertEqual([t["arm"] for t in trials[:5]], list(ARMS))
        self.assertEqual([t["arm"] for t in trials[5:10]], list(reversed(ARMS)))

    def test_protocol_keeps_final_town_closed_and_requires_every_checkpoint(self):
        frozen = protocol()
        self.assertEqual(frozen["weather"], "ClearNoon")
        self.assertEqual(frozen["training_seeds"], [42, 43])
        self.assertEqual(frozen["driving_policy"]["warmup_frames_per_episode"], 10)
        wrong = routes(); wrong["routes"][0]["route"]["town"] = "Town02"
        with self.assertRaises(ValueError):
            freeze_driving_protocol(wrong, frozen["checkpoint_sha256"], "a" * 64, representative_seed=42)
        with self.assertRaises(ValueError):
            freeze_driving_protocol(routes(), {"incumbent": "a" * 64}, "a" * 64, representative_seed=42)

    def test_average_improvement_allows_a_case_to_worsen_inside_absolute_bounds(self):
        frozen, native, reports = evidence()
        # Straight worsens relative to incumbent in offline and closed loop.
        # Mean still improves, and each case remains under its absolute bound.
        for row in native["episodes"]:
            if row["arm"].startswith("central_recovery") and row["case"] == "straight":
                row["gate"]["metrics"]["lateral_abs_m"]["mean"] = .35
        decision = select_recovery_model(reports, native, frozen)
        self.assertIn("central_recovery_seed42", decision["eligible_improvements"])
        self.assertFalse(decision["pointer_updated"])
        self.assertEqual(len(decision["recovery_effect"]["per_seed"]), 2)

    def test_frozen_representative_prevents_cherry_picking_best_native_seed(self):
        frozen, native, reports = evidence()
        for row in native["episodes"]:
            if row["arm"].endswith("seed43"):
                row["gate"]["metrics"]["lateral_abs_m"]["mean"] = .01
        decision = select_recovery_model(reports, native, frozen)
        self.assertTrue(decision["selected_arm"].endswith("seed42"))
        self.assertTrue(all(arm.endswith("seed42") for arm in decision["deployment_eligible_improvements"]))

    def test_one_hard_start_p95_events_body_margin_and_latency_are_independent_gates(self):
        changes = (
            lambda e: e["gate"]["metrics"]["lateral_abs_m"].update(p95=1.001),
            lambda e: e["gate"]["metrics"].update(collision_count=1),
            lambda e: e["gate"]["metrics"].update(lane_invasion_count=1),
            lambda e: e["gate"]["metrics"].update(minimum_body_lane_margin_m=-.001),
            lambda e: e["gate"]["metrics"].update(driving_success=False, termination_reason="timeout"))
        arm = "central_recovery_seed42"
        for mutate in changes:
            frozen, native, reports = evidence()
            mutate(next(e for e in native["episodes"] if e["arm"] == arm))
            self.assertNotIn(arm, select_recovery_model(reports, native, frozen)["eligible_improvements"])
        frozen, native, reports = evidence()
        native["steady_processing_latency_ms"][arm]["p95"] = 50.001
        self.assertIn("steady-processing-latency", select_recovery_model(reports, native, frozen)["failed_criteria"][arm])

    def test_unpaired_duplicate_replaced_and_failed_states_cannot_form_a_decision(self):
        mutations = (lambda r: r["episodes"].pop(),
            lambda r: r["episodes"].__setitem__(1, copy.deepcopy(r["episodes"][0])),
            lambda r: r["episodes"][0].update(initial_reference_m=99.),
            lambda r: r["episodes"][0]["gate"].update(status="FAIL"),
            lambda r: r.update(source_sha256={}))
        for mutate in mutations:
            frozen, native, _ = evidence(); mutate(native)
            with self.assertRaises(ValueError):
                summarize_paired_driving(native, frozen)

    def test_common_dev_identity_is_stricter_than_equal_sample_counts_and_ignores_training_seed_config(self):
        frozen, native, reports = evidence()
        # Different training config hashes/seeds do not alter RGB preprocessing.
        select_recovery_model(reports, native, frozen)
        reports[ARMS[1]]["common_dev_selection_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            select_recovery_model(reports, native, frozen)

    def test_ineffective_retraining_preserves_incumbent_and_records_negative_result(self):
        frozen, native, reports = evidence()
        reports = {arm: offline(arm, (.6, .6, .6) if arm != "incumbent" else (.2, .2, .2)) for arm in ARMS}
        decision = select_recovery_model(reports, native, frozen)
        self.assertEqual(decision["selected_arm"], "incumbent")
        self.assertFalse(decision["pointer_updated"])

    def test_weather_strata_do_not_duplicate_all_condition_geometry_groups(self):
        frozen, native, reports = evidence()
        for report in reports.values():
            report["metrics"] += [dict(row, weather="ClearNoon") for row in report["metrics"]]
        self.assertEqual(select_recovery_model(reports, native, frozen)["status"], "PASS")

    def test_recovery_adapter_preserves_exact_rgb_preprocessing_and_raw_predictions(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.observed = None
            def forward(self, image):
                self.observed = image.clone()
                return image.new_tensor([[[float(x), .1] for x in (3, 6, 9, 12, 15)]])
        model, config = Model().eval(), replace(load_learning_config(), seed=43)
        state = {"variant": "hybrid_geometry", "arm": "central_only", "checkpoint_role": "selected-recovery-model",
            "epoch": 2, "model_contract": {"model_version": "synthetic"}, "model_contract_sha256": "a" * 64,
            "source_sha256": {}, "manifest_sha256": "b" * 64, "initial_checkpoint_sha256": "d" * 64}
        image = np.zeros((360, 640, 3), np.uint8)
        with patch("carla_tools.recovery_training.load_recovery_checkpoint", return_value=(model, config, state)):
            predictor = RecoveryRuntimePredictor("synthetic.pt", arm="central_only_seed43", expected_sha256="c" * 64)
            result = predictor.predict(image)
            self.assertEqual(result["status"], "PASS", result)
            self.assertTrue(torch.equal(model.observed, preprocess_rgb(image, config)[None]))
            self.assertEqual(len(result["learned_waypoints_ego_m"]), 5)
            self.assertEqual(result["waypoints_ego_m"][2:], result["learned_waypoints_ego_m"][2:])
            self.assertEqual(predictor.identity["seed"], 43)
            with self.assertRaises(ValueError):
                RecoveryRuntimePredictor("synthetic.pt", arm="central_recovery_seed43", expected_sha256="c" * 64)
            with self.assertRaises(ValueError):
                RecoveryRuntimePredictor("synthetic.pt", arm="central_only_seed42", expected_sha256="c" * 64)
            with patch("carla_tools.recovery_training.load_recovery_checkpoint", return_value=(model, config, state)) as loader:
                RecoveryRuntimePredictor("synthetic.pt", arm="central_only_seed43", expected_sha256="c" * 64,
                    expected_manifest_sha256="b" * 64, expected_initial_checkpoint_sha256="d" * 64)
                self.assertEqual(loader.call_args.kwargs["manifest_sha256"], "b" * 64)
                with self.assertRaises(ValueError):
                    RecoveryRuntimePredictor("synthetic.pt", arm="central_only_seed43", expected_sha256="c" * 64,
                        expected_manifest_sha256="b" * 64, expected_initial_checkpoint_sha256="e" * 64)

    def test_recovery_adapter_reaches_full_native_gt_control_and_event_replay(self):
        from tests.test_baseline_validation import write_baseline, save_sources
        from tests.test_runtime_driving import FixtureModel
        from carla_tools.baseline import load_baseline_config
        from carla_tools.controller import PurePursuitController, VehicleGeometry, load_controller_config
        from carla_tools.evaluation import evaluation_config_from_dict, evaluate_episode
        from carla_tools.projection import decode_rgb8_png
        from carla_tools.runtime_driving import RuntimeTrialPolicy, SCOPE
        from carla_tools.recovery_comparison import comparison_source_hashes, validate_recovery_episode
        config = replace(load_learning_config(), seed=43)
        state = {"variant": "hybrid_geometry", "arm": "central_only", "checkpoint_role": "selected-recovery-model",
            "epoch": 2, "model_contract": {"model_version": "synthetic"}, "model_contract_sha256": "a" * 64,
            "source_sha256": {}, "manifest_sha256": "b" * 64}
        with patch("carla_tools.recovery_training.load_recovery_checkpoint", return_value=(FixtureModel().eval(), config, state)):
            predictor = RecoveryRuntimePredictor("synthetic.pt", arm="central_only_seed43", expected_sha256="c" * 64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "episode"
            metadata, rows = write_baseline(root)
            evaluation = evaluation_config_from_dict(metadata["evaluation_config"])
            controller = PurePursuitController(load_controller_config(), VehicleGeometry(**metadata["vehicle_geometry"]))
            policy = RuntimeTrialPolicy(load_baseline_config(), evaluation, metadata["route_contract"],
                metadata["condition"], predictor=predictor, initial_reference_m=0.)
            policy.configure_metadata(metadata)
            trial = {"arm": "central_only_seed43", "initial_reference_m": 0.}
            metadata.update(scope=SCOPE, recovery_comparison={"arm": trial["arm"],
                "protocol_sha256": "d" * 64, "source_sha256": comparison_source_hashes()})
            previous = None
            for row in rows:
                _, _, rgb = decode_rgb8_png((root / row["image"]["path"]).read_bytes(), row["image"]["sha256"], (640, 360))
                image = np.frombuffer(rgb, np.uint8).reshape(360, 640, 3)
                bgra = np.empty((360, 640, 4), np.uint8)
                bgra[:, :, :3], bgra[:, :, 3] = image[:, :, ::-1], 255
                row["command"], row["perception"], row["latency_ms"] = policy.command(
                    SimpleNamespace(raw_data=bgra.tobytes(), width=640, height=360), row["speed_mps"], controller)
                if previous is not None:
                    row["vehicle"]["applied_control"] = {k: previous["command"][k]
                        for k in ("throttle", "brake", "steer", "hand_brake", "reverse")}
                previous = row
            metadata["stop_control"]["steer"] = rows[-1]["vehicle"]["applied_control"]["steer"]
            for stop in metadata["stop_observations"]:
                stop["vehicle"]["applied_control"]["steer"] = metadata["stop_control"]["steer"]
            metadata["evaluation_metrics"] = evaluate_episode(metadata, rows, metadata["events"], config=evaluation)
            save_sources(root, metadata, rows)
            gate = validate_recovery_episode(root, predictor, trial, protocol_sha256="d" * 64)
            self.assertEqual(gate["status"], "PASS", gate)
            rows[1]["tracking"]["label"]["waypoints_ego_m"][0][1] += .1
            save_sources(root, metadata, rows)
            failed = validate_recovery_episode(root, predictor, trial, protocol_sha256="d" * 64)
            self.assertEqual(failed["status"], "FAIL", failed)
            self.assertIn("baseline-original-tracking-replay-mismatch", str(failed["errors"]))

    def test_bootstrap_keeps_existing_identity_and_binds_135_trials_sources_and_transport(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE, "is_private": True}))
            route_file = root / "routes.json"; route_file.write_text(json.dumps(routes()))
            from carla_tools.recovery_comparison import freeze_driving_protocol
            checkpoints = protocol()["checkpoint_sha256"]
            frozen = freeze_driving_protocol(routes(), checkpoints, "a" * 64, representative_seed=42)
            protocol_file = root / "protocol.json"; protocol_file.write_text(json.dumps(frozen))
            bundle = prepare_recovery_driving(metadata, root / "bundle", dev_routes=route_file,
                protocol_file=protocol_file, training_archive_sha256="b" * 64, training_archive_bytes=400,
                training_version=1,
                checkpoint_sha256=checkpoints, evaluation_sha256={arm: "c" * 64 for arm in ARMS},
                comparison_sha256="a" * 64, representative_seed=42)
            submission = json.loads((root / "bundle/kernel-metadata.json").read_text())
            self.assertEqual((submission["id"], submission["title"]), (NOTEBOOK_ID, NOTEBOOK_TITLE))
            self.assertFalse(bundle["creates_new_notebook"])
            self.assertEqual(bundle["expected_episodes"], 135)
            code = (root / "bundle/native.py").read_text()
            self.assertEqual(hashlib.sha256(code.encode()).hexdigest(), bundle["code_sha256"])
            self.assertIn('gate.get("episode_count") != 135', code)
            self.assertIn("scripts/run_recovery_driving.py", code)
            self.assertIn("unpack_recovery_archive", code)
            self.assertEqual(bundle["training_archive_bytes"], 400)
            self.assertEqual(bundle["transport"], "official-versioned-response-with-bounded-resumable-HTTP-ranges")
            helper_node = next(node for node in ast.parse(code).body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "TRAINING_DOWNLOAD_HELPER" for t in node.targets))
            helper = ast.literal_eval(helper_node.value)
            from scripts.kaggle_recovery_training_bootstrap import cloud_download_helper_source
            expected_helper = cloud_download_helper_source(1, Path("__VLF_RECOVERY_DEPS__"),
                Path("__VLF_RECOVERY_DOWNLOAD__"), notebook_id="dongmyungpark/vlf-carla-vulkan-preflight-20261005",
                filename="training_artifacts.zip", expected_sha256="b" * 64, expected_bytes=400, maximum_seconds=1800)
            self.assertEqual(helper, expected_helper)
            self.assertEqual(hashlib.sha256(helper.encode()).hexdigest(), bundle["training_download_helper_sha256"])
            self.assertIn("resumable_http_download", helper)
            self.assertNotIn("NotebookOutputHttpResolver()._resolve", helper)
            # Executing only generated source assignments never starts bootstrap.
            assignments = []
            for node in ast.parse(code).body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PROJECT_SOURCES" for t in node.targets):
                    assignments.append(node)
            namespace = {}
            exec("import base64,lzma,json\n" + "\n".join(ast.unparse(n) for n in assignments), namespace)
            for name, sha in bundle["source_sha256"].items():
                self.assertEqual(hashlib.sha256(namespace["PROJECT_SOURCES"][name].encode()).hexdigest(), sha)
            # Import only the embedded closure in an isolated process. Workspace
            # imports cannot hide a missing evaluator module or policy config.
            embedded = root / "embedded"
            for name, source in namespace["PROJECT_SOURCES"].items():
                path = embedded / name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.encode())
            isolated = ("import json,sys;from pathlib import Path;sys.path.insert(0," + repr(str(embedded)) + ");"
                "from carla_tools.recovery_comparison import freeze_driving_protocol;"
                "routes=json.loads(Path(" + repr(str(route_file)) + ").read_text());"
                "frozen=json.loads(Path(" + repr(str(protocol_file)) + ").read_text());"
                "value=freeze_driving_protocol(routes,frozen['checkpoint_sha256'],'" + "a" * 64 + "',representative_seed=42);"
                "assert value==frozen")
            checked = subprocess.run([sys.executable, "-I", "-c", isolated], cwd=root,
                capture_output=True, text=True, timeout=60)
            self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_archive_extraction_uses_exact_members_and_rejects_tampered_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); archive = root / "models.zip"
            checkpoints, evaluations = {}, {}
            comparison = json.dumps({"status": "PASS", "final_town_used": False}).encode()
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("experiment/comparison.json", comparison)
                for arm in ARMS:
                    if arm == "incumbent":
                        directory, evaluation = "initial", "initial_evaluation"
                    else:
                        name, seed = arm.rsplit("_seed", 1)
                        directory = f"seed{seed}/{name}"; evaluation = directory + "/evaluation"
                    weights, report = arm.encode(), b"{}"
                    checkpoints[arm], evaluations[arm] = hashlib.sha256(weights).hexdigest(), hashlib.sha256(report).hexdigest()
                    stream.writestr("experiment/" + directory + "/checkpoint.pt", weights)
                    stream.writestr("experiment/" + evaluation + "/metrics.json", report)
                stream.writestr("../../unselected-secret", b"unselected")
            kwargs = {"expected_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "checkpoint_sha256": checkpoints, "evaluation_sha256": evaluations,
                "comparison_sha256": hashlib.sha256(comparison).hexdigest()}
            models = json.loads(unpack_recovery_archive(archive, root / "new", **kwargs).read_text())
            self.assertEqual(set(models), set(ARMS))
            self.assertFalse((root / "unselected-secret").exists())
            with self.assertRaises(ValueError):
                unpack_recovery_archive(archive, root / "other", **dict(kwargs, expected_archive_sha256="f" * 64))


if __name__ == "__main__":
    unittest.main()
