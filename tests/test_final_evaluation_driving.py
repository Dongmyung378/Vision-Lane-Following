"""Artificial CPU-only held-out orchestration fixtures, never final outcomes."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from carla_tools.contracts import route_sha256
from carla_tools.final_evaluation import ROUTE_SCOPE, build_manifest
from carla_tools.final_evaluation_driving import (
    BOOTSTRAP_HEIGHT_TOLERANCE_M, FinalEvaluationError, WEATHERS,
    _native_adapter, adapter_dependency_hashes, build_trial_policy,
    collection_for_trial, final_collection_config_from_dict, resolve_bootstrap,
    run_heldout_trial, validate_heldout_episode, verify_topology_in_world)
from tests.test_final_evaluation import bindings_fixture, selection_fixture


def inputs_fixture():
    from tests.test_driving_route import dense_route
    records = []
    for town in ("Town01", "Town02"):
        for index in range(2):
            route = dense_route()
            route.update(route_id=f"{town}_fixture_{index}", scene_group_id=f"{town}_scene_{index}",
                town=town, map_package="Carla/Maps/" + town, spawn_point_index=None,
                selection_scope=ROUTE_SCOPE, opendrive_sha256=hashlib.sha256(b"CPU-only-map-fixture").hexdigest())
            for anchor in route["centerline_anchor_sequence"]:
                anchor["world_location"]["x"] += index * 300.
                anchor["world_rotation"] = {"pitch": 0., "yaw": 0., "roll": 0.}
            route["start"] = copy.deepcopy(route["centerline_anchor_sequence"][0])
            route["route_sha256"] = route_sha256(route)
            records.append(route)
    routes = {"schema_version": 1, "split": "final", "routes": records}
    selection = selection_fixture()
    return routes, selection, build_manifest(routes, selection, bindings_fixture())


def fake_map(route, *, bad=None, spawn_points=None):
    def waypoint(road_id, lane_id, distance):
        record = min(route["centerline_anchor_sequence"], key=lambda row: abs(row["s"] - distance))
        data = copy.deepcopy(record)
        if bad == "position":
            data["world_location"]["x"] += .02
        if bad == "heading":
            data["world_rotation"]["yaw"] += .2
        if bad == "width":
            data["lane_width_m"] += .01
        result = SimpleNamespace(road_id=data["road_id"], section_id=data["section_id"], lane_id=data["lane_id"],
            s=data["s"], lane_type="Driving", is_junction=bad == "junction", lane_width=data["lane_width_m"],
            transform=SimpleNamespace(location=SimpleNamespace(**data["world_location"]),
                                      rotation=SimpleNamespace(**data["world_rotation"])))
        result.next = lambda delta: [result, result] if bad == "branch" else [result]
        return result
    start = route["start"]["world_location"]
    if spawn_points is None:
        spawn_points = [SimpleNamespace(location=SimpleNamespace(**dict(start, z=start["z"] + .1)))]
    return SimpleNamespace(name="/Game/Carla/Maps/" + route["town"], get_spawn_points=lambda: spawn_points,
        get_waypoint_xodr=waypoint, to_opendrive=lambda: "CPU-only-map-fixture" if bad != "map" else "changed")


class FinalConfigAndTopologyTests(unittest.TestCase):
    def setUp(self):
        self.routes, self.selection, self.manifest = inputs_fixture()

    def test_only_final_towns_and_weather_extend_original_parser_without_mutation(self):
        from carla_tools.config import collection_config_from_dict
        value = copy.deepcopy(self.manifest["collection_config"])
        original = copy.deepcopy(value)
        for town in ("Town01", "Town02"):
            for weather in WEATHERS:
                value["map"], value["weather"]["preset"] = town, weather
                parsed = final_collection_config_from_dict(value)
                self.assertEqual(parsed.map, town)
                self.assertEqual(parsed.weather.preset, weather)
                normalized = parsed.to_dict()
                normalized["map"], normalized["weather"]["preset"] = "Town01", "ClearNoon"
                self.assertEqual(collection_config_from_dict(normalized).to_dict(), original)
        for key, invalid in (("map", "Town03"), ("weather", {"preset": "WetNoon"})):
            candidate = copy.deepcopy(original); candidate[key] = invalid
            with self.assertRaises(FinalEvaluationError):
                final_collection_config_from_dict(candidate)
        self.assertEqual(self.manifest["collection_config"], original)

    def test_all_other_config_checks_are_preserved_in_extended_town(self):
        original = copy.deepcopy(self.manifest["collection_config"])
        mutations = [lambda c: c["world"].update(fixed_delta_seconds=0.),
            lambda c: c["connection"].update(host="remote.invalid"),
            lambda c: c["camera"].update(width=0),
            lambda c: c["camera"]["mount"]["rotation"].update(pitch=float("nan")),
            lambda c: c["weather"].update(unknown=1), lambda c: c.update(extra=True)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                value = copy.deepcopy(original); value["map"] = "Town02"; mutation(value)
                with self.assertRaises(ValueError):
                    final_collection_config_from_dict(value)

    def test_environment_changes_actual_camera_only_and_freezes_seed_start_and_goal(self):
        for group in "ABCDE":
            trial = next(t for t in self.manifest["trials"] if t["group_id"] == group)
            cfg = collection_for_trial(self.manifest, trial, 17)
            self.assertEqual(cfg.camera.mount.rotation.pitch, -10. if group == "E" else -12.)
            self.assertEqual(cfg.weather.preset, trial["environment"]["weather"])
            self.assertEqual(cfg.map, trial["town"])
            self.assertEqual(cfg.seed, trial["initial_state"]["seed"])
            self.assertEqual(cfg.verification.warmup_frames, 10)
        self.assertEqual(self.manifest["baseline_config"]["camera_pitch_deg"], -12.)
        self.assertEqual(self.manifest["preprocessing"], self.selection["preprocessing"])
        changed = copy.deepcopy(trial); changed["environment"]["camera_pitch_delta_deg"] = 3.
        with self.assertRaises(FinalEvaluationError):
            collection_for_trial(self.manifest, changed, 17)

    def test_bootstrap_is_live_nearest_xy_with_index_tie_and_elevation_gate(self):
        route = self.routes["routes"][0]
        points = [SimpleNamespace(location=SimpleNamespace(x=x, y=0., z=.1)) for x in (20., -1., 1.)]
        result = resolve_bootstrap(route, fake_map(route, spawn_points=points))
        self.assertEqual(result["spawn_point_index"], 1)
        self.assertEqual(result["spawn_count"], 3)
        self.assertEqual(route["spawn_point_index"], None)
        points[1].location.z = BOOTSTRAP_HEIGHT_TOLERANCE_M + .01
        with self.assertRaisesRegex(FinalEvaluationError, "elevation"):
            resolve_bootstrap(route, fake_map(route, spawn_points=points))
        with self.assertRaisesRegex(FinalEvaluationError, "no-bootstrap"):
            resolve_bootstrap(route, fake_map(route, spawn_points=[]))

    def test_native_anchor_verifier_rejects_map_pose_heading_width_branch_and_junction(self):
        from carla_tools.final_evaluation import runtime_route_contract
        route = runtime_route_contract(self.routes["routes"][0], 0)
        carla = SimpleNamespace(LaneType=SimpleNamespace(Driving="Driving"))
        verify_topology_in_world(route, fake_map(route), carla)
        for bad in ("map", "position", "heading", "width", "branch", "junction"):
            with self.subTest(bad=bad), self.assertRaises(FinalEvaluationError):
                verify_topology_in_world(route, fake_map(route, bad=bad), carla)

    def test_isolated_adapter_leaves_original_namespaces_and_tracker_values_unchanged(self):
        from carla_tools.driving import run_drive_trial
        from carla_tools.driving_route import FixedRouteTracker
        from carla_tools.session import CollectionSession
        from carla_tools.final_evaluation import runtime_route_contract
        from tests.test_data_contracts import transform_fixture
        original_drive, original_enter = dict(run_drive_trial.__globals__), dict(CollectionSession.__enter__.__globals__)
        route = runtime_route_contract(self.routes["routes"][0], 0)
        times = {}
        adapted = _native_adapter(route, times)
        self.assertIs(adapted.__code__, run_drive_trial.__code__)
        self.assertIsNot(adapted.__globals__, run_drive_trial.__globals__)
        self.assertEqual(original_drive, run_drive_trial.__globals__)
        self.assertEqual(original_enter, CollectionSession.__enter__.__globals__)
        adapted_tracker = adapted.__globals__["FixedRouteTracker"](route)
        expected = FixedRouteTracker(route).observe(transform_fixture(), 100, 5.)
        self.assertEqual(adapted_tracker.observe(transform_fixture(), 100, 5.), expected)
        self.assertGreaterEqual(times["label_generation_ms"], 0.)
        self.assertEqual(set(adapter_dependency_hashes()), {"carla_tools/config.py", "carla_tools/session.py",
            "carla_tools/driving.py", "carla_tools/driving_route.py", "carla_tools/geometry_survey.py"})


class FrozenEntrypointTests(unittest.TestCase):
    def test_rgb_command_does_not_accept_gt_pose_or_actual_calibration(self):
        from carla_tools.final_evaluation import runtime_route_contract
        routes, selection, manifest = inputs_fixture()
        trial = manifest["trials"][0]
        policy = build_trial_policy(runtime_route_contract(routes["routes"][0], 0), trial, manifest,
                                    selection, SimpleNamespace(), {})
        for key in ("route", "pose", "calibration", "label"):
            with self.assertRaises(TypeError):
                policy.command(None, 0., None, **{key: {}})

    def test_cli_preserves_all_paired_failures_and_records_blocked_trial(self):
        from contextlib import ExitStack, redirect_stdout
        import io
        from scripts.run_final_evaluation import run_pipeline
        routes, selection, manifest = inputs_fixture()
        inputs = {"routes": routes, "selection": selection, "manifest": manifest, "checkpoint": Path("fixture.pt")}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch("carla_tools.evaluation_freeze.verify_freeze", return_value={"status": "PASS"}))
            stack.enter_context(patch("scripts.run_final_evaluation.read_verified_inputs", return_value=inputs))
            stack.enter_context(patch("carla_tools.runtime_driving.RuntimeCheckpointPredictor"))
            stack.enter_context(patch("carla_tools.guarded_path_correction.wrap_preprocessing"))
            stack.enter_context(patch("carla_tools.selected_pipeline._validate_predictor"))
            stack.enter_context(patch("scripts.run_final_evaluation.prepare_native_world", return_value=SimpleNamespace()))
            stack.enter_context(patch("carla_tools.final_evaluation_driving.run_heldout_trial", return_value={"status": "PASS", "step_count": 1}))
            stack.enter_context(patch("carla_tools.final_evaluation_driving.validate_heldout_episode",
                return_value={"status": "PASS", "frame_count": 1, "driving_success": False, "metrics": {"termination_reason": "technical_failure"}}))
            # Patch only the actual CARLA import, after the other patch targets resolve.
            original_import = __import__("importlib").import_module
            stack.enter_context(patch("scripts.run_final_evaluation.importlib.import_module",
                side_effect=lambda name: SimpleNamespace() if name == "carla" else original_import(name)))
            with redirect_stdout(io.StringIO()):
                report = run_pipeline(Path(directory) / "complete", "fixture", expected_freeze_id="a" * 64,
                                      groups=["A"], execute_heldout=True)
            self.assertEqual(report["status"], "PASS", report)
            self.assertEqual(report["episode_count"], 18)
            self.assertFalse(report["complete_final_matrix"])
            self.assertEqual({row["method"] for row in report["episodes"]}, {"B0", "M1", "privileged"})
            self.assertTrue(all(row["gate"]["driving_success"] is False for row in report["episodes"]))
            with patch("scripts.run_final_evaluation.prepare_native_world", side_effect=ValueError("map-unavailable")):
                blocked = run_pipeline(Path(directory) / "blocked", "fixture", expected_freeze_id="a" * 64,
                                       groups=["B"], execute_heldout=True)
            self.assertEqual(blocked["status"], "FAIL")
            self.assertEqual(blocked["episodes"][0]["execution"]["status"], "BLOCKED")
            self.assertEqual(blocked["episode_count"], 0)

    def test_freeze_failure_precedes_model_import_native_access_and_output_creation(self):
        from scripts.run_final_evaluation import run_pipeline
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "native"
            with patch("carla_tools.evaluation_freeze.verify_freeze", side_effect=ValueError("changed-source")), \
                 patch("carla_tools.runtime_driving.RuntimeCheckpointPredictor") as predictor, \
                 patch("importlib.import_module") as importer:
                with self.assertRaisesRegex(ValueError, "changed-source"):
                    run_pipeline(output, "absent.json", expected_freeze_id="a" * 64, groups=["A"], execute_heldout=True)
                importer.assert_not_called(); predictor.assert_not_called()
                self.assertFalse(output.exists())

    def test_verified_input_hash_change_and_implicit_execution_are_rejected(self):
        from scripts.run_final_evaluation import read_verified_inputs, run_pipeline
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "plan.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "verified-input-changed"):
                read_verified_inputs({"verified_inputs": {"manifest": {"path": "plan.json", "bytes": 2, "sha256": "a" * 64}}}, root)
            with self.assertRaisesRegex(ValueError, "explicit-execution"):
                run_pipeline(root / "native", "absent", expected_freeze_id="a" * 64, groups=["A"])
            with self.assertRaisesRegex(ValueError, "external-freeze-identity"):
                run_pipeline(root / "native", "absent", expected_freeze_id=None, groups=["A"], execute_heldout=True)

    def test_native_map_transition_refuses_busy_world_before_loading_another_map(self):
        from scripts.run_final_evaluation import prepare_native_world
        from carla_tools.config import CollectionConfig
        from unittest.mock import Mock
        world = Mock()
        world.get_settings.return_value.synchronous_mode = False
        world.get_actors.return_value = [SimpleNamespace(type_id="vehicle.foreign")]
        client = Mock()
        client.get_client_version.return_value = client.get_server_version.return_value = "0.9.16"
        client.get_world.return_value = world
        carla = SimpleNamespace(Client=Mock(return_value=client))
        with self.assertRaisesRegex(ValueError, "busy-world"):
            prepare_native_world(carla, CollectionConfig(), {"town": "Town02"})
        client.load_world.assert_not_called()


class FinalNativeCPUStandInTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.previous_threads = torch.get_num_threads(); torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.previous_threads)

    def native_fixture(self, route, *, blank=False, acknowledge_control=True):
        import numpy as np
        from tests.test_baseline import scene
        with patch.object(sys, "path", [str(Path(__file__).parent), *sys.path]):
            from tests.test_driving_native import NativeStandInWorld, native_module
        rgb = np.full((360, 640, 3), 50, np.uint8) if blank else scene()[0]
        # A visible colour witness remains even for the deliberate failed image.
        rgb[0, 0] = [200, 50, 20]
        bgra = np.empty((360, 640, 4), np.uint8)
        bgra[:, :, :3], bgra[:, :, 3] = rgb[:, :, ::-1], 255
        raw = bgra.tobytes()
        live = fake_map(route)
        class PaintedWorld(NativeStandInWorld):
            def get_map(self):
                live.get_spawn_points = lambda: self.spawn_points
                return live
            def spawn_actor(self, blueprint, transform, *, attach_to, attachment_type=None):
                actor = super().spawn_actor(blueprint, transform, attach_to=attach_to, attachment_type=attachment_type)
                if blueprint.id == "sensor.camera.rgb":
                    native_transform = actor.get_transform
                    def readback():
                        import math
                        result = native_transform(); matrix = result.get_matrix()
                        result.location = SimpleNamespace(**{axis: matrix[i][3] for i, axis in enumerate(("x", "y", "z"))})
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
                callback = camera.callback
                camera.callback = lambda image: callback(SimpleNamespace(**dict(vars(image), raw_data=raw)))
                try:
                    return super().tick(timeout)
                finally:
                    camera.callback = callback
        world = PaintedWorld(acknowledge_control=acknowledge_control)
        world.map_name = "/Game/Carla/Maps/" + route["town"]
        module = native_module(world)
        module.LaneType = SimpleNamespace(Driving="Driving")
        module.WeatherParameters.HardRainNoon = copy.deepcopy(module.WeatherParameters.ClearNoon)
        module.WeatherParameters.HardRainNoon.precipitation = 100.
        module.WeatherParameters.ClearSunset = copy.deepcopy(module.WeatherParameters.ClearNoon)
        module.WeatherParameters.ClearSunset.sun_altitude_angle = 15.
        return world, module

    def model_fixture(self, selection):
        from carla_tools.guarded_path_correction import wrap_preprocessing
        from carla_tools.selected_pipeline import pipeline_id
        from tests.test_runtime_driving import fixture_predictor
        predictor = fixture_predictor("path")
        predictor.raw.identity.update(checkpoint_sha256="a" * 64, model_contract_sha256="b" * 64)
        predictor._identity.update(checkpoint_sha256="a" * 64, model_contract_sha256="b" * 64)
        selection["model"]["learning_config_sha256"] = predictor.raw.config.sha256
        selection["pipeline_id"] = pipeline_id(selection)
        return wrap_preprocessing(predictor, {k: v for k, v in selection["preprocessing"].items() if k != "input"})

    def execute(self, group, method, *, blank=False, rejected=False, state_id="center_s0"):
        from carla_tools.baseline import load_baseline_config
        from carla_tools.development_study import ProfiledBaselinePredictor
        routes, selection, manifest = inputs_fixture()
        model = self.model_fixture(selection)
        if rejected:
            model.base.raw.model.forward = lambda image: image.new_tensor([[[3., 0.], [1., 3.], [.1, 4.], [.1, 6.], [.1, 9.]]])
        manifest = build_manifest(routes, selection, bindings_fixture())
        trial = next(t for t in manifest["trials"] if t["group_id"] == group and t["method"] == method
                     and t["initial_state"]["state_id"] == state_id)
        route = next(r for r in routes["routes"] if r["route_id"] == trial["route_id"])
        predictor = model if method == "M1" else ProfiledBaselinePredictor(load_baseline_config()) if method == "B0" else None
        world, carla = self.native_fixture(route, blank=blank)
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        output = Path(temporary.name) / "episode"
        operation = run_heldout_trial(route, trial, output, manifest=manifest, selection=selection,
            predictor=predictor, carla_module=carla, carla_map=world.get_map())
        self.assertEqual(operation["status"], "PASS", operation)
        gate = validate_heldout_episode(output, predictor, trial, manifest=manifest, routes=routes, selection=selection)
        self.assertEqual(gate["status"], "PASS", gate)
        return output, predictor, trial, manifest, routes, selection, gate

    def test_town02_three_methods_capture_native_geometry_shared_control_and_replay(self):
        for method in ("B0", "M1", "privileged"):
            with self.subTest(method=method):
                output, predictor, trial, manifest, routes, selection, gate = self.execute("B", method)
                metadata = json.loads((output / "episode.json").read_text())
                rows = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
                self.assertTrue(gate["original_RGB_native_GT_control_events_replayed"])
                self.assertEqual(metadata["collection_config"]["map"], "Town02")
                self.assertEqual(metadata["final_evaluation"]["controller_config"], manifest["controller_config"])
                self.assertEqual(metadata["evaluation_goal_reference_m"], 15.)
                self.assertEqual(metadata["declared_topology_route"]["spawn_point_index"], None)
                self.assertEqual(metadata["route_contract"]["spawn_point_index"], 0)
                self.assertEqual(metadata["perception_uses_route"], method == "privileged")
                if method == "privileged":
                    self.assertIn("reference_timing", rows[0]["perception"])
                    self.assertNotIn("pipeline_timing", rows[0]["perception"])
                else:
                    self.assertIn("pipeline_timing", rows[0]["perception"])
                # Rehashing a forged command cannot pass independent replay.
                from tests.test_baseline_validation import save_sources
                rows[0]["command"]["steer"] += .1
                save_sources(output, metadata, rows)
                self.assertEqual(validate_heldout_episode(output, predictor, trial,
                    manifest=manifest, routes=routes, selection=selection)["status"], "FAIL")

    def test_shifted_pitch_is_measured_while_nominal_perception_remains_fixed(self):
        output, predictor, trial, manifest, routes, selection, gate = self.execute("E", "B0")
        metadata = json.loads((output / "episode.json").read_text())
        self.assertEqual(metadata["collection_config"]["camera"]["mount"]["rotation"]["pitch"], -10.)
        self.assertEqual(metadata["baseline_config"]["camera_pitch_deg"], -12.)
        metadata["collection_config"]["camera"]["mount"]["rotation"]["pitch"] = -12.
        (output / "episode.json").write_text(json.dumps(metadata))
        self.assertEqual(validate_heldout_episode(output, predictor, trial,
            manifest=manifest, routes=routes, selection=selection)["status"], "FAIL")

    def test_prediction_failure_is_retained_as_valid_failed_driving_episode(self):
        output, predictor, trial, manifest, routes, selection, gate = self.execute("C", "B0", blank=True)
        self.assertFalse(gate["driving_success"])
        self.assertEqual(gate["frame_count"], 1)
        self.assertEqual(gate["metrics"]["termination_reason"], "technical_failure")

    def test_typed_controller_rejection_is_reproduced_without_route_fallback(self):
        output, predictor, trial, manifest, routes, selection, gate = self.execute("A", "M1", blank=True, rejected=True)
        self.assertFalse(gate["driving_success"])
        self.assertEqual(gate["frame_count"], 1)
        rows = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
        rejection = rows[0]["perception"]["controller_rejection"]
        self.assertEqual(rejection["error_code"], "controller-waypoints-not-ordered-forward")
        self.assertIsNone(rows[0]["command"])
        self.assertIsNone(rows[0]["perception"]["learned_waypoints_ego_m"])
        from tests.test_baseline_validation import save_sources
        metadata = json.loads((output / "episode.json").read_text())
        rejection["error_code"] = "forged-error"
        save_sources(output, metadata, rows)
        self.assertEqual(validate_heldout_episode(output, predictor, trial,
            manifest=manifest, routes=routes, selection=selection)["status"], "FAIL")

    def test_requested_weather_initialization_and_native_readbacks_cannot_be_self_declared(self):
        output, predictor, trial, manifest, routes, selection, gate = self.execute("D", "privileged")
        metadata_file = output / "episode.json"
        original = json.loads(metadata_file.read_text())
        mutations = [lambda m: m["applied_manifest"]["weather"]["values"].update(precipitation=99.),
            lambda m: m["applied_manifest"]["world_settings"].update(fixed_delta_seconds=.1),
            lambda m: m["applied_manifest"]["vehicle"]["control_applied"].update(throttle=.1),
            lambda m: m["applied_manifest"]["camera"]["mount_requested"]["rotation"].update(pitch=-9.),
            lambda m: m["initialization"]["perturbed_requested_transform"]["location"].update(x=5.),
            lambda m: m["native_bootstrap"]["spawn_locations_m"][0].update(x=5.)]
        for mutate in mutations:
            changed = copy.deepcopy(original); mutate(changed)
            metadata_file.write_text(json.dumps(changed))
            rejected = validate_heldout_episode(output, predictor, trial,
                manifest=manifest, routes=routes, selection=selection)
            self.assertEqual(rejected["status"], "FAIL", rejected)
        metadata_file.write_text(json.dumps(original))

    def test_nonzero_physical_starts_have_equal_effective_distance_and_recorded_recovery(self):
        for state_id, expected in (("right_s2", 17.), ("left_s4", 19.)):
            with self.subTest(state=state_id):
                output, predictor, trial, manifest, routes, selection, gate = self.execute("A", "privileged", state_id=state_id)
                metadata = json.loads((output / "episode.json").read_text())
                self.assertEqual(metadata["evaluation_goal_reference_m"], expected)
                self.assertAlmostEqual(metadata["evaluation_route_length_m"], 15., delta=.5)
                self.assertEqual(metadata["initialization"]["initial_reference_m"], expected - 15.)
                self.assertEqual(metadata["collection_config"]["seed"], trial["initial_state"]["seed"])

    def test_privileged_controller_rejection_is_an_evaluated_failure_with_exact_replay(self):
        from carla_tools.controller import ControllerError
        def rejected_controller(config, geometry):
            class Reject:
                def step(self, *args, **kwargs):
                    raise ControllerError("fixture-reference-rejection")
            return Reject()
        with patch("carla_tools.controller_study.build_controller", side_effect=rejected_controller):
            output, predictor, trial, manifest, routes, selection, gate = self.execute("A", "privileged")
        self.assertEqual(gate["frame_count"], 1)
        self.assertFalse(gate["driving_success"])
        self.assertEqual(gate["metrics"]["termination_reason"], "technical_failure")
        rows = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
        self.assertEqual(rows[0]["perception"]["controller_rejection"]["error_code"], "fixture-reference-rejection")
        self.assertIsNone(rows[0]["command"])


if __name__ == "__main__":
    unittest.main()
