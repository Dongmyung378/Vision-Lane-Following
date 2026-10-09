"""Independent synthetic image/label fixtures exercise learning boundaries."""

import copy
import hashlib
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

from carla_tools.learning_data import (
    LaneWaypointDataset, LearningConfig, LearningDataError, RGB_MEAN, RGB_STD,
    augment_rgb, learning_config_from_dict, load_learning_config, preprocess_rgb,
    validate_waypoint_points,
)


def make_manifest(root):
    samples = []
    for index, split in enumerate(("train", "train", "dev")):
        # A red channel gradient and independent green marker distinguish RGB
        # order, sample identity and geometry without any implementation label.
        rgb = np.zeros((360, 640, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.arange(640, dtype=np.uint16)[None, :] * 255 // 639
        rgb[150:190, 100 + index * 50:130 + index * 50, 1] = 150
        rgb[:, :, 2] = 20 + index
        image = f"image-{index}.png"
        cv2.imwrite(str(root / image), rgb[:, :, ::-1])
        samples.append({"sample_id": str(index), "episode_id": "ep-" + split, "route_id": "route-" + split,
                        "scene_group_id": "scene-" + split, "split": split, "case": "straight", "condition": "center",
                        "frame": 10 + index, "timestamp": 0.5 + index * 0.05, "image": image,
                        "image_sha256": hashlib.sha256((root / image).read_bytes()).hexdigest(),
                        "image_width": 640, "image_height": 360,
                        "waypoints_ego_m": [[float(s), -0.2 * index] for s in (3, 6, 9, 12, 15)],
                        "speed_mps": 3.0})
    manifest = {"schema_version": 1, "dataset_version": "moving-pilot-v1", "status": "COMPLETE",
                "scope": "moving-rgb-waypoint-pilot", "town": "Town01", "reserved_final_town": "Town02",
                "sample_count": 3, "split_counts": {"train": 2, "dev": 1}, "samples": samples}
    path = root / "pilot_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


class LearningConfigTests(unittest.TestCase):
    def test_defaults_strict_serialization_and_fixed_input(self):
        config = load_learning_config()
        self.assertEqual(config, learning_config_from_dict(config.to_dict()))
        self.assertEqual(config.model_version, LearningConfig().model_version)
        # Training hyperparameters can change independently of Python defaults.
        # An explicit file value must survive loading rather than be replaced
        # by a dataclass default.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learning.json"
            configured = config.to_dict()
            configured["encoder_learning_rate"] = 0.0002
            path.write_text(json.dumps(configured), encoding="utf-8")
            self.assertEqual(load_learning_config(path).encoder_learning_rate, 0.0002)
        self.assertEqual(len(config.sha256), 64)
        for raw in ({**config.to_dict(), "unknown": True}, {k: v for k, v in config.to_dict().items() if k != "seed"}):
            with self.assertRaises(LearningDataError):
                learning_config_from_dict(raw)
        for change in ({"image_width": 224}, {"label_scale_m": 1}, {"num_workers": -1}, {"learning_rate": float("nan")},
                       {"mixed_precision": 1}, {"augmentation_brightness": 1},
                       {"model_version": "vlf-resnet18-waypoint-v1"}):
            with self.assertRaises(LearningDataError):
                replace(config, **change)

    def test_full_image_rgb_normalization_matches_independent_formula(self):
        rgb = np.empty((360, 640, 3), dtype=np.uint8)
        rgb[:] = (255, 0, 128)
        tensor = preprocess_rgb(rgb)
        self.assertEqual(tuple(tensor.shape), (3, 180, 320))
        self.assertEqual(tensor.dtype, torch.float32)
        expected = torch.tensor([(v / 255 - mean) / std for v, mean, std in zip((255, 0, 128), RGB_MEAN, RGB_STD)])
        torch.testing.assert_close(tensor[:, 0, 0], expected)
        torch.testing.assert_close(tensor[:, -1, -1], expected)
        for invalid in (rgb.astype(np.float32), rgb[:, :, 0], rgb[:180]):
            with self.assertRaises(LearningDataError):
                preprocess_rgb(invalid)

    def test_augmentation_changes_color_and_preserves_marker_locations(self):
        rgb = np.full((360, 640, 3), 20, dtype=np.uint8)
        rgb[80:110, 70:100] = (120, 160, 200)
        a = augment_rgb(rgb, seed=4)
        np.testing.assert_array_equal(a, augment_rgb(rgb, seed=4))
        self.assertFalse(np.array_equal(a, augment_rgb(rgb, seed=5)))
        self.assertEqual(a.shape, rgb.shape)
        self.assertGreater(float(a[90, 90].mean()), float(a[79, 90].mean()))
        self.assertGreater(float(a[90, 90].mean()), float(a[90, 69].mean()))

    def test_past_ninety_degree_arc_accepts_nonmonotonic_forward_x(self):
        points = [[8 * math.sin(s / 8), 8 * (1 - math.cos(s / 8))] for s in (3, 6, 9, 12, 15)]
        self.assertLess(points[4][0], points[3][0])
        validate_waypoint_points(points)
        for invalid in ([[3.0, 0.0], [2.0, 1.0], *points[2:]],
                        [[-1.0, 0.0], *points[1:]], [[float("nan"), 0.0], *points[1:]]):
            with self.assertRaises(LearningDataError):
                validate_waypoint_points(invalid)


class LearningDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path, self.manifest = make_manifest(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, manifest):
        self.path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_source_rgb_and_actor_labels_return_without_pose_speed_inputs(self):
        dataset = LaneWaypointDataset(self.path, "train")
        self.assertEqual(len(dataset), 2)
        item = dataset[1]
        self.assertEqual(set(item), {"image", "target", "sample_id", "case", "condition", "route_id"})
        self.assertEqual(tuple(item["image"].shape), (3, 180, 320))
        torch.testing.assert_close(item["target"], torch.tensor([[s, -0.2] for s in (3, 6, 9, 12, 15)]))
        self.assertGreater(float(item["image"][0, 0, -1]), float(item["image"][2, 0, -1]))

    def test_augmentation_epoch_sample_and_schedule_determinism(self):
        dataset = LaneWaypointDataset(self.path, "train", augment=True)
        initial = dataset[0]["image"].clone()
        _ = dataset[1]
        torch.testing.assert_close(initial, dataset[0]["image"])
        dataset.set_epoch(1)
        self.assertFalse(torch.equal(initial, dataset[0]["image"]))
        another = LaneWaypointDataset(self.path, "train", augment=True)
        another.set_epoch(1)
        torch.testing.assert_close(dataset[0]["image"], another[0]["image"])
        with self.assertRaises(LearningDataError):
            LaneWaypointDataset(self.path, "dev", augment=True)

    def test_split_leakage_detected_for_episode_route_scene_and_rgb(self):
        for key in ("episode_id", "route_id", "scene_group_id", "image_sha256", "image", "sample_id"):
            raw = copy.deepcopy(self.manifest)
            raw["samples"][2][key] = raw["samples"][0][key]
            self.write_manifest(raw)
            with self.subTest(key=key), self.assertRaises(LearningDataError):
                LaneWaypointDataset(self.path, "train")

    def test_invalid_targets_and_counts_are_rejected(self):
        for mutate in (lambda x: x["samples"][0].update(waypoints_ego_m=[[3, 0]]),
                       lambda x: x["samples"][0]["waypoints_ego_m"][2].__setitem__(0, 2),
                       lambda x: x["samples"][0]["waypoints_ego_m"][0].__setitem__(1, float("nan")),
                       lambda x: x.update(sample_count=99), lambda x: x.update(split_counts={"train": 3, "dev": 0}),
                       lambda x: x.update(reserved_final_town="Town01"), lambda x: x.update(status="FAILED")):
            raw = copy.deepcopy(self.manifest)
            mutate(raw)
            self.write_manifest(raw)
            with self.assertRaises(ValueError):
                LaneWaypointDataset(self.path, "train")

    def test_dataset_accepts_arc_ordered_past_ninety_degree_curve(self):
        self.manifest["samples"][0]["waypoints_ego_m"] = [[8 * math.sin(s / 8), 8 * (1 - math.cos(s / 8))] for s in (3, 6, 9, 12, 15)]
        self.write_manifest(self.manifest)
        dataset = LaneWaypointDataset(self.path, "train")
        self.assertLess(float(dataset[0]["target"][4, 0]), float(dataset[0]["target"][3, 0]))

    def test_source_change_after_initialization_is_rejected(self):
        dataset = LaneWaypointDataset(self.path, "train")
        image = self.root / "image-0.png"
        image.write_bytes(image.read_bytes() + b"tampering")
        with self.assertRaisesRegex(LearningDataError, "sha256"):
            dataset[0]

    def test_manifest_boundary_recheck_detects_label_replacement(self):
        dataset = LaneWaypointDataset(self.path, "train")
        dataset.assert_manifest_unchanged()
        self.manifest["samples"][0]["waypoints_ego_m"][0][1] = 0.8
        self.write_manifest(self.manifest)
        with self.assertRaisesRegex(LearningDataError, "manifest-changed"):
            dataset.assert_manifest_unchanged()

    def test_path_escape_and_corrupt_image_are_rejected(self):
        for path in ("../outside.png", "C:/outside.png", "image\\sample.png", "/outside.png"):
            raw = copy.deepcopy(self.manifest)
            raw["samples"][0]["image"] = path
            self.write_manifest(raw)
            with self.assertRaises(LearningDataError):
                LaneWaypointDataset(self.path, "train")
        self.write_manifest(self.manifest)
        image = self.root / "image-0.png"
        image.write_bytes(b"not a png")
        self.manifest["samples"][0]["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
        self.write_manifest(self.manifest)
        with self.assertRaisesRegex(LearningDataError, "not-png"):
            LaneWaypointDataset(self.path, "train")

    def test_gray_png_with_matching_hash_is_rejected(self):
        image = self.root / "image-0.png"
        cv2.imwrite(str(image), np.zeros((360, 640), dtype=np.uint8))
        self.manifest["samples"][0]["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
        self.write_manifest(self.manifest)
        with self.assertRaisesRegex(LearningDataError, "shape"):
            LaneWaypointDataset(self.path, "train")


if __name__ == "__main__":
    unittest.main()
