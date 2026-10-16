"""Explicit development weather must preserve the original collection contract."""
from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from carla_tools.config import CollectionConfig, ConfigError, WeatherConfig, collection_config_from_dict
from carla_tools.contracts import route_sha256
from carla_tools.driving import run_drive_trial
from carla_tools.session import CollectionSession, SessionError
from test_collection_session import Transform, World, carla_for, weather
from test_driving_native import ListeningSensor, NativeStandInWorld, native_module
from test_driving_route import dense_route
import test_baseline_native as baseline_fixture
from carla_tools.baseline_validation import _validate_camera_episode, validate_baseline_episode
from carla_tools.driving import load_validation_config
from carla_tools.driving_route import perturbation_transform
from scripts.run_dataset_pipeline import run_dataset_v1_trial, validate_dataset_v1_episode


DEVELOPMENT_WEATHER = ("ClearNoon", "CloudyNoon", "WetNoon", "SoftRainNoon")


def native_weather(module):
    for index, name in enumerate(DEVELOPMENT_WEATHER):
        value = weather(10.0 + index * 20.0)
        value.precipitation = 15.0 if name == "SoftRainNoon" else 0.0
        value.wetness = 50.0 if name in ("WetNoon", "SoftRainNoon") else 0.0
        setattr(module.WeatherParameters, name, value)
    return module


class DatasetNativeWeatherContractTests(unittest.TestCase):
    def test_original_parser_stays_clear_noon_only_without_explicit_weather_scope(self):
        raw = CollectionConfig().to_dict()
        self.assertEqual(collection_config_from_dict(raw).weather.preset, "ClearNoon")
        for name in DEVELOPMENT_WEATHER[1:]:
            raw["weather"]["preset"] = name
            with self.subTest(weather=name), self.assertRaises(ConfigError):
                collection_config_from_dict(raw)
            self.assertEqual(collection_config_from_dict(raw, allowed_weather_presets=DEVELOPMENT_WEATHER).weather.preset,
                             name)

    def test_an_allowlist_cannot_admit_unsupported_or_reserved_weather(self):
        raw = CollectionConfig().to_dict()
        for presets in ((), [], "ClearNoon", ("ClearNoon", "ClearNoon"), (False,),
                        ("HardRainNoon",), ("ClearSunset",), ("unknown",)):
            with self.subTest(presets=presets), self.assertRaises(ConfigError):
                collection_config_from_dict(raw, allowed_weather_presets=presets)

    def test_default_session_refuses_extra_weather_before_world_mutation(self):
        world = World()
        module = native_weather(carla_for(world))
        config = replace(CollectionConfig(), weather=WeatherConfig("CloudyNoon"))
        with self.assertRaises(SessionError):
            with CollectionSession(config, module):
                self.fail("unapproved weather reached native session")
        self.assertFalse(world.settings_calls)
        self.assertFalse(world.weather_calls)
        self.assertEqual(vars(world.weather), vars(world.initial_weather))

    def test_explicit_weather_is_applied_recorded_and_restored_for_every_preset(self):
        for name in DEVELOPMENT_WEATHER:
            with self.subTest(weather=name):
                world = World()
                module = native_weather(carla_for(world))
                config = replace(CollectionConfig(), weather=WeatherConfig(name))
                session = CollectionSession(config, module, allowed_weather_presets=DEVELOPMENT_WEATHER)
                with session:
                    self.assertEqual(vars(world.weather), vars(getattr(module.WeatherParameters, name)))
                    self.assertEqual(session.manifest["weather"]["preset"], name)
                    self.assertEqual(session.manifest["weather"]["values"]["cloudiness"],
                                     getattr(module.WeatherParameters, name).cloudiness)
                self.assertEqual(session.cleanup_report["status"], "PASS")
                self.assertTrue(session.cleanup_report["weather_restored"])
                self.assertEqual(vars(world.weather), vars(world.initial_weather))

    def test_unacknowledged_weather_cannot_be_certified_by_its_requested_name(self):
        world = World()
        world.ignore_weather = True
        module = native_weather(carla_for(world))
        config = replace(CollectionConfig(), weather=WeatherConfig("WetNoon"))
        session = CollectionSession(config, module, allowed_weather_presets=DEVELOPMENT_WEATHER)
        with self.assertRaises(SessionError) as caught:
            with session:
                self.fail("unapplied weather was accepted")
        self.assertEqual(caught.exception.code, "weather-readback-mismatch")
        self.assertEqual(vars(world.weather), vars(world.initial_weather))

    def test_shared_trial_propagates_explicit_weather_to_the_selected_session_factory(self):
        world = NativeStandInWorld()
        module = native_weather(native_module(world))
        config = replace(CollectionConfig(), weather=WeatherConfig("CloudyNoon"))
        config = replace(config, verification=replace(config.verification, warmup_frames=2),
                         camera=replace(config.camera, width=16, height=16))
        route = dense_route()
        for anchor in route["centerline_anchor_sequence"]:
            anchor["world_rotation"] = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        route["start"] = copy.deepcopy(route["centerline_anchor_sequence"][0])
        route["route_sha256"] = route_sha256(route)
        calls = []

        def factory(selected, **kwargs):
            calls.append((selected, kwargs.copy()))
            return CollectionSession(selected, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "explicit-weather"
            with patch("carla_tools.driving.verify_geometry_route_in_world", return_value=None):
                report = run_drive_trial(route, "straight", {"name": "center", "lateral_m": 0.0, "yaw_deg": 0.0},
                                         output, config, carla_module=module,
                                         allowed_weather_presets=DEVELOPMENT_WEATHER, session_factory=factory)
            self.assertEqual(report["status"], "PASS", report)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1]["allowed_weather_presets"], DEVELOPMENT_WEATHER)
            metadata = json.loads((output / "episode.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["collection_config"]["weather"]["preset"], "CloudyNoon")
            self.assertEqual(metadata["applied_manifest"]["weather"]["preset"], "CloudyNoon")
            self.assertEqual(metadata["cleanup"]["status"], "PASS")
        self.assertEqual(vars(world.weather), vars(world.initial_weather))


class DatasetNativeSourceBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.weather_name = "CloudyNoon"
        cls.start_m = 2.0
        cls.world = NativeStandInWorld()
        cls.module = None

        def expanded_trial(route, case, condition, output, collection, *, carla_module, **_kwargs):
            cls.module = native_weather(carla_module)
            selected = replace(collection, weather=WeatherConfig(cls.weather_name))
            return run_dataset_v1_trial(route, case, condition, output, selected, carla_module=cls.module,
                                        initial_reference_m=cls.start_m)

        with patch("test_baseline_native.run_baseline_trial", side_effect=expanded_trial):
            report = baseline_fixture.BaselineNativeTests().run_case(
                cls.world, cls.root / "source", method="privileged_reference", full_resolution=True)
        if report["status"] != "PASS":
            raise AssertionError(report)
        cls.metadata_path = cls.root / "source" / "episode.json"
        cls.metadata_bytes = cls.metadata_path.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def tearDown(self):
        self.metadata_path.write_bytes(self.metadata_bytes)

    def gate(self, **changes):
        parameters = {"allowed_conditions": load_validation_config()["initial_conditions"],
                      "expected_initial_reference_m": self.start_m, "expected_weather": self.weather_name,
                      "carla_module": self.module}
        parameters.update(changes)
        return validate_dataset_v1_episode(self.metadata_path.parent, **parameters)

    def rewrite_metadata(self, transform):
        metadata = json.loads(self.metadata_bytes)
        transform(metadata)
        self.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    def test_actual_weather_and_nonzero_start_pass_every_rgb_control_and_label_replay(self):
        report = self.gate()
        self.assertEqual(report["status"], "PASS", report)
        self.assertTrue(report["checks"]["native_weather_preset_readback"])
        self.assertTrue(report["checks"]["requested_start_binding"])
        self.assertTrue(report["checks"]["every_frame_rgb_color"])
        self.assertTrue(report["checks"]["shared_controller_replay"])
        self.assertGreater(report["metrics"]["image_count"], 10)
        self.assertEqual(vars(self.world.weather), vars(self.world.initial_weather))

    def test_original_baseline_gate_keeps_its_narrow_weather_and_start_scope(self):
        report = validate_baseline_episode(self.metadata_path.parent)
        self.assertEqual(report["status"], "FAIL")

    def test_expected_start_is_bound_exactly_beyond_the_native_pose_tolerance(self):
        report = self.gate(expected_initial_reference_m=self.start_m + 0.2)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["errors"][-1]["code"], "dataset-requested-start-binding-mismatch")

    def test_relabelling_the_applied_weather_name_cannot_change_the_source_weather(self):
        self.rewrite_metadata(lambda metadata: metadata["applied_manifest"]["weather"].update(preset="ClearNoon"))
        report = self.gate()
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["errors"][-1]["code"], "dataset-native-weather-readback-mismatch")

    def test_correct_weather_name_with_forged_native_values_is_rejected(self):
        self.rewrite_metadata(lambda metadata: metadata["applied_manifest"]["weather"]["values"].update(cloudiness=99.0))
        report = self.gate()
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["errors"][-1]["code"], "dataset-native-weather-readback-mismatch")

    def test_initialization_metadata_cannot_hide_a_different_requested_reference(self):
        self.rewrite_metadata(lambda metadata: metadata["initialization"].update(initial_reference_m=self.start_m + 0.1))
        report = self.gate()
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["errors"][-1]["code"], "dataset-requested-start-binding-mismatch")

    def test_native_manifest_must_acknowledge_the_actual_collection_configuration(self):
        self.rewrite_metadata(lambda metadata: metadata["applied_manifest"].update(config_sha256="0" * 64))
        report = self.gate()
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["errors"][-1]["code"], "dataset-native-weather-readback-mismatch")


def curved_vertex_fixture(radius=10.0):
    """An analytic toy curve; no coordinates originate from a native capture."""
    route = dense_route()
    for anchor in route["centerline_anchor_sequence"]:
        arc = anchor["arc_distance_m"]
        anchor["world_location"].update(x=radius * math.sin(arc / radius),
                                         y=radius * (1 - math.cos(arc / radius)))
        anchor["world_rotation"] = {"pitch": 0.0, "yaw": math.degrees(arc / radius), "roll": 0.0}
    route["start"] = copy.deepcopy(route["centerline_anchor_sequence"][0])
    route["route_sha256"] = route_sha256(route)
    return route


class VertexSettlingWorld(NativeStandInWorld):
    """Apply a tiny parked displacement before a vertex, without steering physics."""

    def __init__(self, route, start_m, yaw_error_deg):
        super().__init__()
        self.settled = False
        self.yaw_error_deg = yaw_error_deg
        anchors = route["centerline_anchor_sequence"]
        vertex = next(index for index, anchor in enumerate(anchors) if anchor["arc_distance_m"] == start_m)
        first, second = anchors[vertex - 1:vertex + 1]
        dx = second["world_location"]["x"] - first["world_location"]["x"]
        dy = second["world_location"]["y"] - first["world_location"]["y"]
        length = math.hypot(dx, dy)
        self.previous_chord_direction = dx / length, dy / length

    def tick(self, timeout):
        actor = self.actors.get(100)
        if actor is not None and len(actor.physics_changes) >= 2 and not self.settled:
            actor.transform.location.x -= 0.0005 * self.previous_chord_direction[0]
            actor.transform.location.y -= 0.0005 * self.previous_chord_direction[1]
            actor.transform.rotation.yaw += self.yaw_error_deg
            self.settled = True
        return super().tick(timeout)


class DatasetCurvedVertexHeadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root, cls.route, cls.start_m = Path(cls.temporary.name), curved_vertex_fixture(), 2.0
        cls.records = {}
        original_transform = ListeningSensor.get_transform

        def sensor_transform(sensor):
            matrix = original_transform(sensor).get_matrix()
            return Transform(location=SimpleNamespace(x=matrix[0][3], y=matrix[1][3], z=matrix[2][3]),
                             rotation=SimpleNamespace(pitch=math.degrees(math.asin(matrix[2][0])),
                                                      yaw=math.degrees(math.atan2(matrix[1][0], matrix[0][0])),
                                                      roll=math.degrees(math.atan2(-matrix[2][1], matrix[2][2]))), matrix=matrix)

        config = CollectionConfig()
        config = replace(config, verification=replace(config.verification, warmup_frames=2))
        for name, yaw_error in (("correct", 0.0), ("inside_bound", 0.49), ("outside_bound", 0.51)):
            world = VertexSettlingWorld(cls.route, cls.start_m, yaw_error)
            module = native_weather(native_module(world))
            output = cls.root / name
            with patch("carla_tools.driving.verify_geometry_route_in_world", return_value=None), \
                    patch.object(ListeningSensor, "get_transform", sensor_transform):
                report = run_dataset_v1_trial(cls.route, "right_curve",
                    {"name": "center", "lateral_m": 0.0, "yaw_deg": 0.0}, output, config,
                    carla_module=module, initial_reference_m=cls.start_m)
            if report["status"] != "PASS":
                raise AssertionError(report)
            cls.records[name] = output, module

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def gate(self, name="correct"):
        output, module = self.records[name]
        return validate_dataset_v1_episode(output, allowed_conditions=load_validation_config()["initial_conditions"],
            expected_initial_reference_m=self.start_m, expected_weather="ClearNoon", carla_module=module)

    def test_declared_vertex_tangent_matches_actual_yaw_despite_previous_chord_projection(self):
        output, _ = self.records["correct"]
        first = json.loads((output / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
        anchors = self.route["centerline_anchor_sequence"]
        vertex = next(index for index, anchor in enumerate(anchors) if anchor["arc_distance_m"] == self.start_m)
        before, after = anchors[vertex:vertex + 2]
        declared_yaw = math.degrees(math.atan2(after["world_location"]["y"] - before["world_location"]["y"],
                                              after["world_location"]["x"] - before["world_location"]["x"]))
        self.assertLess(first["tracking"]["reference_arc_m"], self.start_m)
        self.assertGreater(abs(first["tracking"]["heading_error_deg"]), 0.5)
        self.assertAlmostEqual(first["vehicle"]["transform"]["rotation"]["yaw"], declared_yaw)
        gate = self.gate()
        self.assertEqual(gate["status"], "PASS", gate)
        self.assertTrue(gate["checks"]["declared_start_heading_binding"])
        self.assertTrue(gate["checks"]["original_route_evaluation_replay"])
        self.assertTrue(gate["checks"]["requested_heading_source_binding"])

    def test_legacy_default_keeps_the_original_tracked_tangent_contract(self):
        output, _ = self.records["correct"]
        parameters = {"allowed_conditions": load_validation_config()["initial_conditions"],
                      "expected_initial_reference_m": self.start_m}
        default = _validate_camera_episode(output, **parameters)
        explicit = _validate_camera_episode(output, initial_heading_reference="tracked-route-tangent", **parameters)
        self.assertEqual(default["status"], "FAIL")
        self.assertEqual(default["errors"], explicit["errors"])
        self.assertEqual(default["errors"][0]["code"], "baseline-initial-reference-or-condition-mismatch")

    def test_half_degree_limit_still_accepts_inside_and_rejects_outside_actual_yaw(self):
        self.assertEqual(load_validation_config()["initial_yaw_tolerance_deg"], 0.5)
        inside, outside = self.gate("inside_bound"), self.gate("outside_bound")
        self.assertEqual(inside["status"], "PASS", inside)
        self.assertEqual(outside["status"], "FAIL")
        self.assertEqual(outside["errors"][0]["code"], "baseline-initial-reference-or-condition-mismatch")

    def test_forged_requested_yaw_cannot_hide_a_wrong_actual_actor_yaw(self):
        output, _ = self.records["outside_bound"]
        path = output / "episode.json"
        saved = path.read_bytes()
        try:
            metadata = json.loads(saved)
            actual = json.loads((output / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])["vehicle"]["transform"]
            metadata["initialization"]["center_requested_transform"] = copy.deepcopy(actual)
            metadata["initialization"]["perturbed_requested_transform"] = copy.deepcopy(actual)
            path.write_text(json.dumps(metadata), encoding="utf-8")
            gate = self.gate("outside_bound")
            self.assertEqual(gate["status"], "FAIL")
            self.assertEqual(gate["errors"][0]["code"], "baseline-initial-reference-or-condition-mismatch")
        finally:
            path.write_bytes(saved)

    def test_coherent_requested_yaw_forgery_is_rejected_even_when_actual_pose_is_correct(self):
        output, _ = self.records["correct"]
        path, trace_path = output / "episode.json", output / "steps.jsonl"
        saved, original_trace = path.read_bytes(), trace_path.read_bytes()
        try:
            for name in ("center_requested_transform", "perturbed_requested_transform"):
                with self.subTest(record=name):
                    metadata = json.loads(saved)
                    metadata["initialization"][name] = perturbation_transform(
                        metadata["initialization"][name], 0.0, 0.2)
                    path.write_text(json.dumps(metadata), encoding="utf-8")
                    gate = self.gate()
                    self.assertEqual(gate["status"], "FAIL")
                    self.assertEqual(gate["errors"][-1]["code"], "dataset-requested-heading-source-binding-mismatch")
                    self.assertEqual(trace_path.read_bytes(), original_trace)
        finally:
            path.write_bytes(saved)


if __name__ == "__main__":
    unittest.main()
