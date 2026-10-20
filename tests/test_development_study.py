"""Frozen-stage, native RGB replay, safety and transport regression checks."""
from __future__ import annotations

import contextlib
import copy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from carla_tools.baseline import load_baseline_config
from carla_tools.config import load_collection_config
from carla_tools.contracts import route_sha256
from carla_tools.controller import VehicleGeometry, load_controller_config
from carla_tools.controller_study import ControllerStudyConfig, build_controller
from carla_tools.development_study import (CASES, DEFAULT_ENVIRONMENTS, LATENCY_SCOPE,
    ProfiledBaselinePredictor, SCOPE, absolute_gate, collection_for_environment, confirm_controller,
    digest, freeze_development_protocol, planned_stage_trials, select_screen_candidate,
    select_tuned_controller, study_source_hashes, summarize_stage, validate_study_episode)
from carla_tools.pipeline_latency import STAGES, TIMING_SCOPE
from scripts.prepare_kaggle_development_study import (prepare_development_study,
    run_logged_study, unpack_study_checkpoint)
from scripts.prepare_kaggle_week1 import NOTEBOOK_ID, NOTEBOOK_TITLE


def configs():
    base = load_controller_config()
    return {name: config.to_dict() for name, config in {
        "incumbent": ControllerStudyConfig(base=base),
        "pp_short": ControllerStudyConfig(base=replace(base, lookahead_base_m=2.5, lookahead_min_m=3., lookahead_max_m=6.)),
        "pp_long": ControllerStudyConfig(base=replace(base, lookahead_base_m=4.5, lookahead_min_m=5., lookahead_max_m=9.)),
        "stanley_low": ControllerStudyConfig(base=base, method="stanley", stanley_cross_track_gain=.6),
        "stanley_medium": ControllerStudyConfig(base=base, method="stanley", stanley_cross_track_gain=1.2),
        "pp_smooth": ControllerStudyConfig(base=base, steer_smoothing_alpha=.55)}.items()}


def routes():
    result = {"split": "dev", "final_town_used": False, "routes": []}
    for case in CASES:
        route = {"route_id": "fixture_" + case, "town": "Town01"}
        route["route_sha256"] = route_sha256(route)
        result["routes"].append({"case": case, "split": "dev", "route": route})
    return result


def protocol():
    return freeze_development_protocol(routes(), "a" * 64, configs())


def episodes(frozen, stage, candidate="pp_short"):
    trials = planned_stage_trials(frozen, routes(), stage, candidate=candidate)
    records = []
    for trial in trials:
        mean = .2 if trial["controller"] == "incumbent" else .15
        records.append(dict(trial, execution={"status": "PASS"}, media={"status": "PASS"},
            steady_processing_p95_ms=20., gate={"status": "PASS", "metrics": {
                "driving_success": True, "termination_reason": "route_completed", "collision_count": 0,
                "lane_invasion_count": 0, "minimum_body_lane_margin_m": .2,
                "lateral_abs_m": {"mean": mean, "p95": .6}}}))
    return trials, records


class DevelopmentProtocolTests(unittest.TestCase):
    def test_frozen_stage_budget_and_exact_paired_states(self):
        frozen = protocol()
        self.assertEqual(sum(frozen["expected_stage_episodes"].values()), 153)
        for stage, count in frozen["expected_stage_episodes"].items():
            trials = planned_stage_trials(frozen, routes(), stage, candidate="pp_short")
            self.assertEqual(len(trials), count)
            self.assertEqual(len({trial["episode"] for trial in trials}), count)
            arms = len({trial["arm"] for trial in trials})
            for index in range(0, count, arms):
                chunk = trials[index:index + arms]
                key = lambda trial: {name: trial[name] for name in ("case", "route_id", "condition", "environment", "initial_reference_m")}
                self.assertTrue(all(key(trial) == key(chunk[0]) for trial in chunk))
            self.assertEqual([trial["arm"] for trial in trials[:arms]],
                list(reversed([trial["arm"] for trial in trials[arms:arms * 2]])))
        self.assertEqual(frozen["stage_starts_m"], {"screen": 0., "tune": 2., "confirmation": 4., "robustness": 4.})

    def test_reserved_town_rain_sunset_and_pitch_never_enter_protocol(self):
        for key, value in (("weather", "HardRainNoon"), ("weather", "ClearSunset"), ("camera_pitch_delta_deg", 1.)):
            environment = copy.deepcopy(list(DEFAULT_ENVIRONMENTS))
            environment[-1][key] = value
            with self.assertRaises(ValueError):
                freeze_development_protocol(routes(), "a" * 64, configs(), environments=environment)
        value = routes()
        value["routes"][0]["route"]["town"] = "Town02"
        value["routes"][0]["route"]["route_sha256"] = route_sha256(value["routes"][0]["route"])
        with self.assertRaises(ValueError):
            freeze_development_protocol(value, "a" * 64, configs())

    def test_actual_camera_shifts_leave_nominal_perception_configuration_unchanged(self):
        base = load_collection_config(Path("configs/collection_d4.json"))
        nominal = load_baseline_config().sha256
        for environment in DEFAULT_ENVIRONMENTS:
            actual = collection_for_environment(base, environment)
            self.assertEqual(actual.camera.mount.rotation.pitch, base.camera.mount.rotation.pitch)
            self.assertEqual(actual.camera.mount.rotation.yaw, base.camera.mount.rotation.yaw + environment["camera_yaw_deg"])
            self.assertEqual(load_baseline_config().sha256, nominal)

    def test_duplicate_controller_or_changed_incumbent_rejected(self):
        value = configs(); value["pp_short"] = value["incumbent"]
        with self.assertRaises(ValueError):
            freeze_development_protocol(routes(), "a" * 64, value)
        value = configs(); value["incumbent"]["steer_smoothing_alpha"] = .5
        with self.assertRaises(ValueError):
            freeze_development_protocol(routes(), "a" * 64, value)

    def test_protocol_pair_order_does_not_depend_on_JSON_object_key_order(self):
        first = protocol()
        reordered = json.loads(json.dumps(first, sort_keys=True))
        second = freeze_development_protocol(routes(), "a" * 64, reordered["controller_configs"])
        self.assertEqual(first, second)

    def test_search_and_tune_keep_average_improvement_with_absolute_case_bounds(self):
        frozen = protocol()
        trials, records = episodes(frozen, "screen")
        screen = summarize_stage(records, frozen, expected_trials=trials)
        chosen = select_screen_candidate(screen, frozen)["selected_nonincumbent"]
        self.assertNotEqual(chosen, "incumbent")
        trials, records = episodes(frozen, "tune")
        for record in records:
            if record["arm"] == "M1_candidate" and record["case"] == "straight":
                record["gate"]["metrics"]["lateral_abs_m"]["mean"] = .22
        tuned = summarize_stage(records, frozen, expected_trials=trials)
        self.assertEqual(select_tuned_controller(tuned, frozen, "pp_short")["selected_controller"], "pp_short")

    def test_one_bad_episode_or_event_rejects_low_average_candidate(self):
        frozen = protocol()
        for field, value in (("collision_count", 1), ("lane_invasion_count", 1), ("minimum_body_lane_margin_m", -.01)):
            trials, records = episodes(frozen, "tune")
            candidate = next(record for record in records if record["arm"] == "M1_candidate")
            candidate["gate"]["metrics"][field] = value
            summary = summarize_stage(records, frozen, expected_trials=trials)
            self.assertEqual(select_tuned_controller(summary, frozen, "pp_short")["selected_controller"], "incumbent")
        trials, records = episodes(frozen, "tune")
        next(record for record in records if record["arm"] == "M1_candidate")["gate"]["metrics"]["lateral_abs_m"]["p95"] = 1.01
        self.assertEqual(select_tuned_controller(summarize_stage(records, frozen, expected_trials=trials), frozen,
            "pp_short")["selected_controller"], "incumbent")

    def test_tiny_single_run_improvement_does_not_select_candidate(self):
        frozen = protocol()
        trials, records = episodes(frozen, "tune")
        for record in records:
            if record["arm"] == "M1_candidate":
                record["gate"]["metrics"]["lateral_abs_m"]["mean"] = .199
        decision = select_tuned_controller(summarize_stage(records, frozen, expected_trials=trials), frozen, "pp_short")
        self.assertEqual(decision["selected_controller"], "incumbent")
        self.assertIn("mean-improvement-below-frozen-practical-margin", decision["failed_criteria"])

    def test_missing_short_episode_latency_retained_and_cannot_pass_selection(self):
        frozen = protocol()
        trials, records = episodes(frozen, "confirmation")
        for record in records:
            if record["arm"] == "B0_candidate":
                record["steady_processing_p95_ms"] = None
        summary = summarize_stage(records, frozen, expected_trials=trials)
        self.assertIsNone(summary["B0_candidate"]["all"]["maximum_episode_processing_p95_ms"])
        self.assertIn("all:missing-steady-latency", absolute_gate(summary["B0_candidate"], frozen))

    def test_failure_robustness_is_reported_and_rejects_promotion(self):
        frozen = protocol()
        trial, records = episodes(frozen, "confirmation")
        nominal = summarize_stage(records, frozen, expected_trials=trial)
        trial, records = episodes(frozen, "robustness")
        bad = next(record for record in records if record["arm"] == "M1_candidate")
        bad["gate"]["metrics"].update(driving_success=False, termination_reason="timeout")
        robust = summarize_stage(records, frozen, expected_trials=trial)
        decision = confirm_controller(nominal, robust, frozen, "pp_short")
        self.assertFalse(decision["eligible_improvement"])
        self.assertEqual(decision["selected_controller"], "incumbent")

    def test_one_weak_environment_cannot_hide_inside_weather_average(self):
        frozen = protocol()
        trials, records = episodes(frozen, "robustness")
        for record in records:
            if record["arm"] == "M1_candidate" and record["environment"]["name"] == "soft_rain_nominal":
                record["gate"]["metrics"]["lateral_abs_m"].update(mean=.7, p95=.9)
        summary = summarize_stage(records, frozen, expected_trials=trials)
        self.assertLess(summary["M1_candidate"]["all"]["mean_lateral_m"], .5)
        self.assertIn("environment:soft_rain_nominal:all:mean_lateral_m", absolute_gate(summary["M1_candidate"], frozen))

    def test_missing_duplicate_replaced_or_unverified_episode_rejected(self):
        frozen = protocol()
        trial, records = episodes(frozen, "tune")
        candidates = [records[:-1], records[:-1] + [records[0]], copy.deepcopy(records), copy.deepcopy(records)]
        candidates[2][0]["environment"]["camera_yaw_deg"] = 99.
        candidates[3][0]["gate"]["status"] = "FAIL"
        for value in candidates:
            with self.assertRaises(ValueError):
                summarize_stage(value, frozen, expected_trials=trial)


class DevelopmentTransportTests(unittest.TestCase):
    def test_exact_member_only_extracted_and_bad_archive_binding_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); archive = root / "archive.zip"
            payload = b"synthetic-checkpoint-fixture"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("experiment/model/checkpoint.pt", payload)
                stream.writestr("../../unused.txt", b"never-extracted")
            arguments = {"expected_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "expected_archive_bytes": archive.stat().st_size, "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
                "member": "experiment/model/checkpoint.pt"}
            result = unpack_study_checkpoint(archive, root / "safe/checkpoint.pt", **arguments)
            self.assertEqual(result.read_bytes(), payload)
            self.assertFalse((root / "unused.txt").exists())
            arguments["checkpoint_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                unpack_study_checkpoint(archive, root / "other.pt", **arguments)

    def test_owned_child_progress_is_teed_and_timeout_stops_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()) as displayed:
                run_logged_study([sys.executable, "-u", "-c", "print('fixture-progress',flush=True)"], root / "good.log", 10, None)
            self.assertIn("fixture-progress", displayed.getvalue())
            self.assertIn("fixture-progress", (root / "good.log").read_text())
            with self.assertRaises(TimeoutError):
                run_logged_study([sys.executable, "-c", "import time;time.sleep(10)"], root / "timeout.log", .2, None)

    def test_preparation_reuses_fixed_notebook_and_keeps_source_and_live_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.json"; frozen = root / "protocol.json"; route = root / "routes.json"
            metadata.write_text(json.dumps({"id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE, "is_private": True}))
            frozen.write_text(json.dumps(protocol())); route.write_text(json.dumps(routes()))
            value = prepare_development_study(metadata, root / "bundle", dev_routes=route, protocol_file=frozen,
                training_archive_sha256="b" * 64, training_archive_bytes=100, training_version=20,
                checkpoint_sha256="a" * 64, checkpoint_member="experiment/model/refined_checkpoint.pt")
            self.assertFalse(value["creates_new_notebook"])
            self.assertTrue(value["live_owned_child_progress"])
            self.assertEqual(value["expected_episodes"], 153)
            code = (root / "bundle/native.py").read_text(encoding="utf-8")
            compile(code, "native.py", "exec")
            self.assertIn('MAX_SECONDS = 7200', code)
            self.assertIn('run_logged_study(', code)
            remote = json.loads((root / "bundle/kernel-metadata.json").read_text())
            self.assertEqual((remote["id"], remote["title"]), (NOTEBOOK_ID, NOTEBOOK_TITLE))


class DevelopmentNativeReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_baseline_validation import save_sources, write_baseline
        cls.temporary = tempfile.TemporaryDirectory()
        cls.original = Path(cls.temporary.name) / "source"
        metadata, rows = write_baseline(cls.original)
        frozen = protocol()
        sources = routes()
        sources["routes"][0]["route"] = metadata["route_contract"]
        frozen["routes_sha256"] = digest(sources)
        # A declared CPU fixture uses its existing strict termination tolerance;
        # production freeze() binds the unchanged default evaluation config.
        frozen["evaluation_config_sha256"] = metadata["evaluation_config_sha256"]
        trial = {"stage": "screen", "arm": "B0_fixture", "method": "B0", "controller": "incumbent",
            "case": "straight", "route_id": metadata["route_contract"]["route_id"], "initial_reference_m": 0.,
            "condition": copy.deepcopy(frozen["initial_conditions"][0]), "environment": copy.deepcopy(frozen["environments"][0]),
            "episode": "episodes/screen/B0_fixture/straight__clear_nominal__center"}
        predictor = ProfiledBaselinePredictor(load_baseline_config())
        metadata["collection_config"]["verification"] = load_collection_config(Path("configs/collection_d4.json")).to_dict()["verification"]
        study_config = ControllerStudyConfig()
        metadata.update(scope=SCOPE, method="B0", native_model=predictor.identity, requested_initial_reference_m=0.,
            detailed_latency_required=True, latency_scope=LATENCY_SCOPE,
            development_study={"trial": trial, "protocol_sha256": digest(frozen), "source_sha256": study_source_hashes(),
                "controller_config": study_config.to_dict(), "controller_config_sha256": study_config.sha256})
        controller = build_controller(study_config, VehicleGeometry(**metadata["vehicle_geometry"]))
        for row in rows:
            row["command"] = controller.step(row["perception"]["waypoints_ego_m"], row["speed_mps"], dt_seconds=.05)
            row["perception"]["pipeline_timing"] = {"schema_version": 1, "device": "cpu", "synchronization": "none-cpu",
                "executed_stages": ["bgra_conversion_ms", "forward_decode_ms", "controller_ms"], "synchronized_stages": [],
                "failed_stages": [], "stages_ms": {key: {"bgra_conversion_ms": 1., "forward_decode_ms": 2., "controller_ms": 1.}.get(key, 0.) for key in STAGES},
                "pipeline_overhead_ms": 0., "total_processing_ms": 4., "timing_scope": TIMING_SCOPE}
        save_sources(cls.original, metadata, rows)
        cls.frozen, cls.routes, cls.trial = frozen, sources, trial

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.temporary_case = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_case.cleanup)
        self.episode = Path(self.temporary_case.name) / "episode"
        shutil.copytree(self.original, self.episode)

    def check(self):
        return validate_study_episode(self.episode, ProfiledBaselinePredictor(load_baseline_config()), self.trial,
            protocol=self.frozen, routes=self.routes)

    def mutate(self, change):
        from tests.test_baseline_validation import save_sources
        metadata = json.loads((self.episode / "episode.json").read_text())
        rows = [json.loads(line) for line in (self.episode / "steps.jsonl").read_text().splitlines()]
        change(metadata, rows)
        save_sources(self.episode, metadata, rows)

    def test_complete_original_RGB_native_GT_control_event_replay(self):
        before = {path.relative_to(self.episode).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.episode.rglob("*") if path.is_file()}
        checked = self.check()
        self.assertEqual(checked["status"], "PASS", checked)
        self.assertEqual(checked["frame_count"], 49)
        self.assertTrue(checked["native"]["checks"]["shared_controller_replay"])
        self.assertEqual(before, {path.relative_to(self.episode).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.episode.rglob("*") if path.is_file()})

    def test_forged_command_or_GT_rejected_even_with_updated_trace_hash(self):
        self.mutate(lambda metadata, rows: rows[2]["command"].update(steer=.9))
        self.assertEqual(self.check()["status"], "FAIL")
        shutil.rmtree(self.episode); shutil.copytree(self.original, self.episode)
        self.mutate(lambda metadata, rows: rows[2]["tracking"]["label"]["waypoints_ego_m"][0].__setitem__(1, 2.))
        self.assertEqual(self.check()["status"], "FAIL")

    def test_changed_nominal_or_actual_camera_and_controller_binding_rejected(self):
        self.mutate(lambda metadata, rows: metadata["collection_config"]["camera"]["mount"]["location"].update(z=2.45))
        self.assertEqual(self.check()["status"], "FAIL")
        shutil.rmtree(self.episode); shutil.copytree(self.original, self.episode)
        self.mutate(lambda metadata, rows: metadata["development_study"].update(controller_config_sha256="0" * 64))
        self.assertEqual(self.check()["status"], "FAIL")

    def test_changed_evaluation_configuration_rejected_even_with_self_consistent_hash(self):
        def change(metadata, rows):
            metadata["evaluation_config"]["timeout_seconds"] = 999.
            metadata["evaluation_config_sha256"] = digest(metadata["evaluation_config"])
        self.mutate(change)
        self.assertEqual(self.check()["status"], "FAIL")

    def test_missing_original_image_or_forged_timing_rejected(self):
        (self.episode / "images/00000100.png").unlink()
        self.assertEqual(self.check()["status"], "FAIL")
        shutil.rmtree(self.episode); shutil.copytree(self.original, self.episode)
        self.mutate(lambda metadata, rows: rows[0]["perception"]["pipeline_timing"].update(total_processing_ms=0.))
        self.assertEqual(self.check()["status"], "FAIL")


class DevelopmentNativeOrchestrationTests(unittest.TestCase):
    def test_owned_native_stand_in_runs_PP_Stanley_and_actual_camera_readbacks(self):
        """A CPU stand-in checks orchestration only, never driving performance."""
        from types import SimpleNamespace
        from carla_tools.development_study import run_study_trial
        from carla_tools.driving_media import export_driving_media
        from carla_tools.evaluation import load_evaluation_config
        from tests.test_baseline import scene, np
        from tests.test_runtime_driving import fixture_predictor
        import torch
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with patch.object(sys, "path", [str(Path(__file__).parent), *sys.path]):
            from tests.test_driving_native import NativeStandInWorld, native_module
            from tests.test_driving_route import dense_route
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
                        result = original_transform()
                        matrix = result.get_matrix()
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

        collection = load_collection_config(Path("configs/collection_d4.json"))
        baseline = load_baseline_config()
        evaluation = replace(load_evaluation_config(), timeout_seconds=10.)
        for method, controller, env_index, rejected in (("M1", "incumbent", 0, False), ("M1", "stanley_low", 0, False),
                ("B0", "incumbent", 3, False), ("B0", "incumbent", 5, False), ("M1", "incumbent", 0, True)):
            with self.subTest(method=method, controller=controller, environment=env_index, controller_rejected=rejected), tempfile.TemporaryDirectory() as temporary:
                if rejected:
                    bgra[:, :, :3] = 50
                    bgra[0, 0, :3] = [1, 2, 3]
                    raw_data = bgra.tobytes()
                frozen = protocol(); source_routes = routes()
                route = dense_route()
                for anchor in route["centerline_anchor_sequence"]:
                    anchor["world_rotation"] = {"pitch": 0., "yaw": 0., "roll": 0.}
                route["start"] = copy.deepcopy(route["centerline_anchor_sequence"][0])
                route["route_sha256"] = route_sha256(route)
                source_routes["routes"][0]["route"] = route
                frozen["routes_sha256"] = digest(source_routes)
                trial = {"stage": "screen", "arm": "fixture", "method": method, "controller": controller,
                    "case": "straight", "route_id": route["route_id"], "initial_reference_m": 0.,
                    "condition": copy.deepcopy(frozen["initial_conditions"][0]),
                    "environment": copy.deepcopy(frozen["environments"][env_index]),
                    "episode": "episodes/screen/fixture/straight__fixture__center"}
                if method == "B0":
                    predictor = ProfiledBaselinePredictor(baseline)
                else:
                    predictor = fixture_predictor("path")
                    predictor.raw.identity["checkpoint_sha256"] = "a" * 64
                    predictor._identity["checkpoint_sha256"] = "a" * 64
                    if rejected:
                        # Valid actor-origin radial order, but invalid order
                        # at the measured rear axle: exercise a real typed
                        # controller input rejection, with no mocked control.
                        predictor.raw.model.forward = lambda image: image.new_tensor([
                            [[3., 0.], [1., 3.], [.1, 4.], [.1, 6.], [.1, 9.]]])
                destination = Path(temporary) / "episode"
                world = PaintedWorld()
                with patch("carla_tools.driving.verify_geometry_route_in_world", return_value=None):
                    execution = run_study_trial(route, trial, destination, collection, predictor=predictor,
                        protocol=frozen, carla_module=native_module(world), baseline_config=baseline,
                        evaluation_config=evaluation)
                self.assertEqual(execution["status"], "PASS", execution)
                checked = validate_study_episode(destination, predictor, trial, protocol=frozen, routes=source_routes)
                self.assertEqual(checked["status"], "PASS", checked)
                self.assertGreater(checked["frame_count"], 0 if rejected else 10)
                self.assertTrue(checked["native"]["checks"]["owned_actors_cleaned"])
                media = export_driving_media(destination)
                self.assertEqual(media["status"], "PASS", media)
                self.assertEqual(media["frame_count"], checked["frame_count"])
                if rejected:
                    from tests.test_baseline_validation import save_sources
                    metadata = json.loads((destination / "episode.json").read_text())
                    rows = [json.loads(line) for line in (destination / "steps.jsonl").read_text().splitlines()]
                    self.assertFalse(checked["driving_success"])
                    self.assertEqual(checked["metrics"]["termination_reason"], "technical_failure")
                    self.assertEqual(len(rows), 1)
                    rejection = rows[0]["perception"]["controller_rejection"]
                    self.assertEqual(rejection["error_code"], "controller-waypoints-not-ordered-forward")
                    self.assertEqual(rows[0]["perception"]["pipeline_timing"]["failed_stages"], ["controller_ms"])
                    self.assertIsNone(rows[0]["command"])
                    self.assertEqual(len(rejection["waypoints_ego_m"]), 5)
                    planned, records = episodes(frozen, "tune")
                    failed_record = next(record for record in records if record["arm"] == "M1_candidate")
                    failed_record.update(gate=checked, media=media, steady_processing_p95_ms=None)
                    summary = summarize_stage(records, frozen, expected_trials=planned)
                    self.assertEqual(summary["M1_candidate"]["all"]["episodes"], 9)
                    self.assertEqual(summary["M1_candidate"]["all"]["clean_completed"], 8)
                    self.assertEqual(select_tuned_controller(summary, frozen, "pp_short")["selected_controller"], "incumbent")
                    rejection["error_code"] = "forged-controller-error"
                    save_sources(destination, metadata, rows)
                    changed = validate_study_episode(destination, predictor, trial, protocol=frozen, routes=source_routes)
                    self.assertEqual(changed["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
