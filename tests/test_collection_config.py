"""D3 schema, immutable effective configuration, and calibration regression checks."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from carla_tools.config import (
    CollectionConfig,
    ConfigError,
    VehicleConfig,
    collection_config_from_dict,
    load_collection_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "collection_d3.json"
DEFAULT_SHA256 = "4063be7db617865045a92ba3b73399ea3b747a4d59d713bd9950979724800e78"


class CollectionConfigTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def change(self, path, value):
        updated = copy.deepcopy(self.raw)
        keys = path.split(".")
        node = updated
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
        return updated

    def load_text(self, text):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "configuration.json"
            path.write_text(text, encoding="utf-8")
            return load_collection_config(path)

    def test_default_effective_configuration_and_golden_hash(self):
        config = load_collection_config(CONFIG_PATH)
        self.assertEqual(config, CollectionConfig())
        self.assertEqual(config.sha256, DEFAULT_SHA256)
        self.assertEqual(config.vehicle.control.to_dict(), {"throttle": 0.0, "steer": 0.0, "brake": 1.0, "hand_brake": True, "reverse": False})
        self.assertEqual(config.camera.mount.rotation.pitch, -12.0)
        self.assertEqual(config.camera.mount.location.z, 2.4)
        self.assertEqual(config.weather.preset, "ClearNoon")

    def test_intrinsics_follow_horizontal_fov_and_image_center_convention(self):
        intrinsics = load_collection_config(CONFIG_PATH).camera.intrinsics()
        self.assertEqual(set(intrinsics), {"fx", "fy", "cx", "cy"})
        self.assertAlmostEqual(intrinsics["fx"], 320.0)
        self.assertAlmostEqual(intrinsics["fy"], 320.0)
        self.assertEqual(intrinsics["cx"], 320.0)
        self.assertEqual(intrinsics["cy"], 180.0)
        raw = copy.deepcopy(self.raw)
        raw["camera"].update(width=1280, height=720, fov=60)
        large = collection_config_from_dict(raw).camera.intrinsics()
        self.assertAlmostEqual(large["fx"], 640 * math.sqrt(3))
        self.assertEqual(large["fx"], large["fy"])
        self.assertEqual((large["cx"], large["cy"]), (640.0, 360.0))

    def test_nested_state_and_attributes_are_immutable_and_exports_are_independent(self):
        config = load_collection_config(CONFIG_PATH)
        with self.assertRaises(FrozenInstanceError):
            config.camera.mount.location.x = 4.0
        with self.assertRaises(FrozenInstanceError):
            config.vehicle.control.throttle = 1.0
        with self.assertRaises(TypeError):
            config.vehicle.attributes["color"] = "255,0,0"
        exported = config.to_dict()
        exported["vehicle"]["control"]["throttle"] = 1.0
        exported["vehicle"]["attributes"]["color"] = "255,0,0"
        exported["camera"]["mount"]["location"]["x"] = 99
        self.assertEqual(config.vehicle.control.throttle, 0.0)
        self.assertEqual(config.vehicle.attributes["color"], "0,0,0")
        self.assertEqual(config.camera.mount.location.x, 1.5)
        self.assertEqual(config.sha256, DEFAULT_SHA256)

    def test_constructor_copies_attribute_mapping(self):
        attributes = {"role_name": "vlf_ego", "color": "0,0,0"}
        vehicle = VehicleConfig(attributes=attributes)
        attributes["color"] = "255,0,0"
        self.assertEqual(vehicle.attributes["color"], "0,0,0")

    def test_canonical_hash_ignores_key_order_and_normalizes_effective_numbers(self):
        raw = {key: self.raw[key] for key in reversed(self.raw)}
        raw["camera"]["fov"] = 90
        config = collection_config_from_dict(raw)
        self.assertEqual(config.sha256, DEFAULT_SHA256)
        exported = config.to_dict()
        self.assertEqual(collection_config_from_dict(exported).sha256, config.sha256)
        self.assertEqual(json.loads(json.dumps(exported)), exported)
        changed = collection_config_from_dict(self.change("seed", 43))
        self.assertNotEqual(config.sha256, changed.sha256)

    def test_effective_color_is_normalized_before_hashing_and_export(self):
        config = collection_config_from_dict(self.change("vehicle.attributes.color", "0,00,000"))
        self.assertEqual(config.vehicle.attributes["color"], "0,0,0")
        self.assertEqual(config.sha256, DEFAULT_SHA256)

    def test_missing_fields_and_unknown_fields_are_rejected_at_every_level(self):
        for path in ("seed", "camera.gamma", "vehicle.attributes.color", "vehicle.control.reverse"):
            with self.subTest(missing=path):
                raw = copy.deepcopy(self.raw)
                keys = path.split(".")
                node = raw
                for key in keys[:-1]:
                    node = node[key]
                del node[keys[-1]]
                with self.assertRaisesRegex(ConfigError, "missing keys"):
                    collection_config_from_dict(raw)
        for path in ("password", "camera.unknown", "vehicle.attributes.api_token", "vehicle.control.gear"):
            with self.subTest(unknown=path):
                with self.assertRaisesRegex(ConfigError, "unsupported keys"):
                    collection_config_from_dict(self.change(path, "do-not-store-credentials"))

    def test_duplicate_json_keys_are_rejected_even_if_values_match(self):
        text = json.dumps(self.raw, separators=(",", ":"))
        for existing, duplicate in (
            ('"seed":42', '"seed":42,"seed":42'),
            ('"width":640', '"width":640,"width":320'),
            ('"color":"0,0,0"', '"color":"0,0,0","color":"1,2,3"'),
        ):
            with self.subTest(key=existing):
                self.assertIn(existing, text)
                with self.assertRaisesRegex(ConfigError, "Duplicate JSON key"):
                    self.load_text(text.replace(existing, duplicate, 1))

    def test_numeric_booleans_strings_and_non_boolean_modes_are_rejected(self):
        cases = {
            "seed": True,
            "connection.port": False,
            "camera.width": True,
            "world.max_substeps": True,
            "world.fixed_delta_seconds": True,
            "camera.gamma": "2.2",
            "verification.frames": 20.0,
            "world.synchronous_mode": 1,
            "vehicle.control.hand_brake": "true",
        }
        for path, value in cases.items():
            with self.subTest(path=path):
                with self.assertRaises(ConfigError):
                    collection_config_from_dict(self.change(path, value))

    def test_nonfinite_values_are_rejected_in_json_and_dicts(self):
        for value in (float("nan"), float("inf"), float("-inf"), 10**400):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ConfigError):
                    collection_config_from_dict(self.change("camera.gamma", value))
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant):
                text = json.dumps(self.raw, separators=(",", ":")).replace('"gamma":2.2', f'"gamma":{constant}')
                with self.assertRaisesRegex(ConfigError, "NaN or Infinity"):
                    self.load_text(text)

    def test_version_blueprint_and_rgb_stationary_mode_contracts_are_enforced(self):
        invalid = {
            "schema_version": 2,
            "carla_version": "0.9.15",
            "map": "Town10HD_Opt",
            "connection.host": "example.com",
            "weather.preset": "getattr_or_eval_expression",
            "vehicle.blueprint": "vehicle.audi.a2",
            "camera.blueprint": "sensor.camera.depth",
            "world.synchronous_mode": False,
            "world.no_rendering_mode": True,
            "world.substepping": False,
            "camera.sensor_tick": 0.05,
            "camera.enable_postprocess_effects": True,
            "camera.lens_k": -1.0,
            "vehicle.control.throttle": 0.1,
            "vehicle.control.brake": 0.0,
            "vehicle.control.hand_brake": False,
            "vehicle.control.reverse": True,
        }
        for path, value in invalid.items():
            with self.subTest(path=path):
                with self.assertRaises(ConfigError):
                    collection_config_from_dict(self.change(path, value))

    def test_substep_budget_equality_is_valid_and_exceeding_it_is_rejected(self):
        valid = collection_config_from_dict(self.change("world.max_substeps", 5))
        self.assertEqual(valid.world.fixed_delta_seconds, 0.05)
        with self.assertRaisesRegex(ConfigError, "exceeds max_substep_delta_time"):
            collection_config_from_dict(self.change("world.max_substeps", 4))

    def test_pose_calibration_capacity_and_attribute_bounds_are_checked(self):
        invalid = {
            "camera.mount.location.x": 11,
            "camera.mount.location.y": -11,
            "camera.mount.location.z": 0,
            "camera.mount.rotation.pitch": 91,
            "camera.mount.rotation.yaw": 181,
            "camera.mount.rotation.roll": -181,
            "camera.fov": 180,
            "camera.width": 0,
            "camera.gamma": 0,
            "connection.port": 65535,
            "world.fixed_delta_seconds": 0,
            "world.max_substep_delta_time": 0.02,
            "verification.frames": 0,
            "verification.queue_capacity": 0,
            "vehicle.attributes.color": "256,0,0",
            "vehicle.attributes.role_name": "secret credential value",
        }
        for path, value in invalid.items():
            with self.subTest(path=path):
                with self.assertRaises(ConfigError):
                    collection_config_from_dict(self.change(path, value))

    def test_malformed_json_and_unreadable_files_raise_the_stable_error(self):
        for text in ("{", "[]", "null", '"string"'):
            with self.subTest(text=text):
                with self.assertRaises(ConfigError):
                    self.load_text(text)
        with tempfile.TemporaryDirectory(prefix="private-marker-") as folder:
            with self.assertRaises(ConfigError) as error:
                load_collection_config(Path(folder) / "does-not-exist.json")
            self.assertNotIn("private-marker", str(error.exception))

    def test_configuration_load_never_imports_carla(self):
        code = (
            "import importlib.abc, sys\n"
            "class RejectCarla(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname == 'carla' or fullname.startswith('carla.'):\n"
            "            raise AssertionError('CARLA must not be imported during configuration validation')\n"
            "sys.meta_path.insert(0, RejectCarla())\n"
            "from carla_tools.config import load_collection_config\n"
            "load_collection_config(sys.argv[1])\n"
            "assert 'carla' not in sys.modules\n"
        )
        result = subprocess.run([sys.executable, "-c", code, str(CONFIG_PATH)], cwd=ROOT, capture_output=True, text=True, timeout=10, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
