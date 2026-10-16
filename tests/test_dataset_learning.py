"""Actual CPU RGB/tensor consumption of a source-bound dataset selection."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from carla_tools.dataset_learning import DatasetV1WaypointDataset
from carla_tools.dataset_v1 import build_dataset_v1_manifest, validate_dataset_v1_manifest
from carla_tools.learning_data import LaneWaypointDataset, LearningDataError
from test_dataset_v1 import dataset_fixture, native_boundary_fixture


def bound_fixture(directory):
    config, routes, descriptors = dataset_fixture(directory)
    path = build_dataset_v1_manifest(directory, routes, descriptors, config)
    with patch("carla_tools.dataset_v1._native_episode_gate", side_effect=native_boundary_fixture):
        report = validate_dataset_v1_manifest(path)
    if report["status"] != "PASS":
        raise AssertionError(report)
    report_path = Path(directory) / "dataset_validation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return path, report_path


class DatasetV1LearningTests(unittest.TestCase):
    def test_both_equal_budget_arms_consume_actual_rgb_and_pose_labels_on_cpu(self):
        import torch
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary:
            path, report = bound_fixture(temporary)
            a = DatasetV1WaypointDataset(path, "train", selection="central_only", validation_report_path=report)
            b = DatasetV1WaypointDataset(path, "train", selection="central_recovery", validation_report_path=report)
            dev_a = DatasetV1WaypointDataset(path, "dev", selection="central_only", validation_report_path=report)
            dev_b = DatasetV1WaypointDataset(path, "dev", selection="central_recovery", validation_report_path=report)
            self.assertEqual((len(a), len(b)), (4, 4))
            self.assertEqual([item["sample_id"] for item in dev_a.samples], [item["sample_id"] for item in dev_b.samples])
            for dataset in (a, b):
                item = dataset[0]
                self.assertEqual(set(item), {"image", "target", "sample_id", "case", "condition", "route_id"})
                self.assertEqual(tuple(item["image"].shape), (3, 180, 320))
                self.assertEqual(tuple(item["target"].shape), (5, 2))
                self.assertEqual(item["image"].device.type, "cpu")
                self.assertTrue(torch.isfinite(item["image"]).all())
                self.assertTrue(torch.equal(item["target"], torch.tensor(dataset.samples[0]["waypoints_ego_m"], dtype=torch.float32)))
            with self.assertRaisesRegex(LearningDataError, "invalid-pilot-scope"):
                LaneWaypointDataset(path, "train")

    def test_changed_selected_image_or_bound_report_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, report = bound_fixture(temporary)
            data = DatasetV1WaypointDataset(path, "train", selection="central_recovery", validation_report_path=report)
            image = Path(temporary) / data.samples[0]["image"]
            image.write_bytes(image.read_bytes() + b"changed")
            with self.assertRaisesRegex(LearningDataError, "image-sha256-mismatch"):
                data[0]
            report.write_text(report.read_text() + "\n")
            with self.assertRaisesRegex(LearningDataError, "validation-report-changed"):
                data.assert_manifest_unchanged()

    def test_missing_replay_check_unbound_report_and_dev_augmentation_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, report = bound_fixture(temporary)
            value = json.loads(report.read_text())
            value["checks"].pop("same_frame_gt_labels")
            report.write_text(json.dumps(value))
            with self.assertRaisesRegex(LearningDataError, "unbound-native-validation-report"):
                DatasetV1WaypointDataset(path, "train", selection="central_only", validation_report_path=report)
            with self.assertRaisesRegex(LearningDataError, "dev-augmentation-forbidden"):
                DatasetV1WaypointDataset(path, "dev", selection="central_only", validation_report_path=report, augment=True)


if __name__ == "__main__":
    unittest.main()
