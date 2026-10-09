"""Independent native-image fixtures for the fixed RGB road ROI contract."""

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from carla_tools.learning_data import (
    LaneWaypointDataset, LearningDataError, RGB_MEAN, RGB_STD, augment_rgb,
    learning_config_from_dict, load_learning_config, preprocess_rgb,
)
from carla_tools.waypoint_model import model_contract


class RoadROITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cv2.setNumThreads(1)

    def test_exact_native_row_crop_precedes_area_resize(self):
        rgb = np.empty((360, 640, 3), dtype=np.uint8)
        rgb[:144] = (250, 10, 15)
        rgb[144:] = (10, 120, 200)
        rgb[144] = (0, 0, 240)
        rgb[359] = (200, 40, 0)
        tensor = preprocess_rgb(rgb)
        restored = (tensor * torch.tensor(RGB_STD)[:, None, None]
                    + torch.tensor(RGB_MEAN)[:, None, None]) * 255
        # A 216 -> 180 resize integrates 1.2 source rows per output row.
        # Boundary bins use row144/359 and a fifth of the adjacent row.
        first = np.array([(0 + 0.2*10)/1.2, (0 + 0.2*120)/1.2, (240 + 0.2*200)/1.2])
        last = np.array([(0.2*10 + 200)/1.2, (0.2*120 + 40)/1.2, (0.2*200 + 0)/1.2])
        np.testing.assert_allclose(restored[:, 0, 100].numpy(), first, atol=0.65, rtol=0)
        np.testing.assert_allclose(restored[:, -1, 100].numpy(), last, atol=0.65, rtol=0)
        np.testing.assert_allclose(restored[:, 80, 100].numpy(), (10, 120, 200), atol=0.001, rtol=0)
        self.assertEqual(tuple(tensor.shape), (3, 180, 320))

    def test_full_width_preserves_both_outer_image_edges(self):
        rgb = np.zeros((360, 640, 3), dtype=np.uint8)
        rgb[:, :2] = (255, 0, 0)
        rgb[:, -2:] = (0, 255, 0)
        output = preprocess_rgb(rgb)
        left = torch.tensor([(value/255 - mean)/std for value, mean, std in zip((255, 0, 0), RGB_MEAN, RGB_STD)])
        right = torch.tensor([(value/255 - mean)/std for value, mean, std in zip((0, 255, 0), RGB_MEAN, RGB_STD)])
        torch.testing.assert_close(output[:, 80, 0], left)
        torch.testing.assert_close(output[:, 80, -1], right)

    def test_photometric_augmentation_keeps_original_marker_geometry(self):
        rgb = np.full((360, 640, 3), 20, dtype=np.uint8)
        rgb[210:230, 170:190] = (120, 160, 200)
        augmented = augment_rgb(rgb, seed=4)
        self.assertEqual(augmented.shape, rgb.shape)
        mask = augmented.mean(axis=2) > 80
        expected = np.zeros((360, 640), dtype=bool)
        expected[210:230, 170:190] = True
        np.testing.assert_array_equal(mask, expected)
        np.testing.assert_array_equal(augment_rgb(rgb, seed=4), augmented)

    def test_dataset_actor_labels_are_unchanged_by_crop_and_augmentation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = []
            labels = [[8*math.sin(distance/8), -8*(1-math.cos(distance/8))] for distance in (3, 6, 9, 12, 15)]
            for index, split in enumerate(("train", "dev")):
                rgb = np.full((360, 640, 3), (30+index*10, 80, 120), dtype=np.uint8)
                source = root / f"image-{index}.png"
                self.assertTrue(cv2.imwrite(str(source), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))
                samples.append({"sample_id": f"s-{index}", "episode_id": f"e-{index}",
                    "route_id": f"r-{index}", "scene_group_id": f"g-{index}", "split": split,
                    "image": source.name, "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "image_width": 640, "image_height": 360, "waypoints_ego_m": labels,
                    "case": "left_curve", "condition": "center", "frame": index+1, "timestamp": (index+1)*0.05,
                    "speed_mps": 9.0, "vehicle_transform": {"privileged": True}})
            manifest = {"schema_version": 1, "status": "COMPLETE", "scope": "moving-rgb-waypoint-pilot",
                        "town": "Town01", "reserved_final_town": "Town02", "sample_count": 2,
                        "split_counts": {"train": 1, "dev": 1}, "samples": samples}
            path = root / "pilot_manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            dataset = LaneWaypointDataset(path, "train", augment=True)
            target = torch.tensor(labels, dtype=torch.float32)
            for epoch in (0, 1):
                dataset.set_epoch(epoch)
                item = dataset[0]
                torch.testing.assert_close(item["target"], target, rtol=0, atol=0)
                self.assertEqual(set(item), {"image", "target", "sample_id", "case", "condition", "route_id"})
                self.assertEqual(tuple(item["image"].shape), (3, 180, 320))
            self.assertEqual(json.loads(path.read_text())["samples"][0]["waypoints_ego_m"], labels)

    def test_v3_contract_rejects_old_versions_and_keeps_actor_units(self):
        config = load_learning_config()
        self.assertEqual(learning_config_from_dict(config.to_dict()), config)
        for version in ("vlf-resnet18-waypoint-v1", "vlf-resnet18-spatial-waypoint-v2"):
            with self.assertRaisesRegex(LearningDataError, "invalid-version"):
                learning_config_from_dict({**config.to_dict(), "model_version": version})
        contract = model_contract(config)
        self.assertEqual(contract["model_version"], "vlf-resnet18-road-roi-waypoint-v3")
        self.assertEqual(contract["input"]["source_crop_xyxy"], [0, 144, 640, 360])
        self.assertEqual(contract["output"]["origin"], "vehicle-actor-origin")
        self.assertEqual(contract["output"]["units"], "meters")
        self.assertEqual(contract["encoder"]["pooling"]["output_grid"], [2, 4])
        self.assertFalse(contract["input"]["pose_input"])
        self.assertFalse(contract["input"]["map_input"])


if __name__ == "__main__":
    unittest.main()
