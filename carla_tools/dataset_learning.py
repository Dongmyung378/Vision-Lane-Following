"""RGB-only loader for an independently validated Dataset v1 selection.

Native source replay is performed once by the dataset verification command.
This consumer binds that report to the immutable manifest, reconstructs the
fixed selection, and hashes and decodes each original image when it is used.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .contracts import canonical_sha256, read_strict_json
from .dataset_v1 import (GATE_VERSION, SAMPLE_FIELDS, SCOPE, VALIDATION_CHECKS, _implementation_hashes,
                         dataset_v1_config_from_dict, match_reference_arc_pairs, select_training_sets)
from .learning_data import (LaneWaypointDataset, LearningDataError, _sample_path,
                            load_learning_config, validate_waypoint_points)


class DatasetV1WaypointDataset(LaneWaypointDataset):
    """Consume one fixed matched-budget arm and its common development split.

    ``validation_report_path`` must be the PASS report from full native replay.
    No route, pose, measured calibration or speed enters the RGB tensor.
    """

    def __init__(self, manifest_path, split, *, selection, validation_report_path,
                 config=None, augment=False, seed=None):
        if split not in ("train", "dev") or selection not in ("central_only", "central_recovery"):
            raise LearningDataError("dataset-learning-invalid-split-or-selection")
        if type(augment) is not bool or augment and split != "train":
            raise LearningDataError("dataset-learning-dev-augmentation-forbidden")
        original = Path(manifest_path)
        if original.is_symlink() or Path(validation_report_path).is_symlink():
            raise LearningDataError("dataset-learning-source-symlink")
        self.manifest_path = original.resolve()
        self.root = self.manifest_path.parent
        self.manifest_sha256 = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        self.manifest = read_strict_json(self.manifest_path)
        manifest = self.manifest
        if (type(manifest) is not dict or manifest.get("scope") != SCOPE or manifest.get("status") != "COMPLETE"
                or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
                or manifest.get("town") != "Town01" or manifest.get("reserved_final_town") != "Town02"
                or manifest.get("reserved_final_state") != "UNOPENED"):
            raise LearningDataError("dataset-learning-invalid-manifest-scope")
        snapshot = dict(manifest)
        digest = snapshot.pop("snapshot_sha256", None)
        if digest != canonical_sha256(snapshot) or manifest.get("implementation_sha256") != _implementation_hashes():
            raise LearningDataError("dataset-learning-snapshot-or-implementation-changed")
        dataset_config = dataset_v1_config_from_dict(manifest["config"])
        if manifest.get("config_sha256") != dataset_config.sha256:
            raise LearningDataError("dataset-learning-config-hash-mismatch")
        self.validation_report_path = Path(validation_report_path).resolve()
        self.validation_report_sha256 = hashlib.sha256(self.validation_report_path.read_bytes()).hexdigest()
        report = read_strict_json(self.validation_report_path)
        if (type(report) is not dict or report.get("status") != "PASS" or report.get("scope") != SCOPE
                or report.get("gate_version") != GATE_VERSION or report.get("errors") != []
                or report.get("manifest_sha256") != self.manifest_sha256 or report.get("snapshot_sha256") != digest
                or report.get("config_sha256") != dataset_config.sha256
                or report.get("sample_count") != manifest.get("sample_count")
                or report.get("episode_count") != len(manifest["episodes"])
                or type(report.get("checks")) is not dict or set(report["checks"]) != set(VALIDATION_CHECKS)
                or any(value is not True for value in report["checks"].values())):
            raise LearningDataError("dataset-learning-unbound-native-validation-report")
        samples = manifest.get("samples")
        if type(samples) is not list or len(samples) != manifest.get("sample_count") or not samples:
            raise LearningDataError("dataset-learning-samples-invalid")
        memberships = {key: {} for key in ("sample_id", "image", "image_sha256", "route_id", "scene_group_id", "episode_id")}
        counts = {"train": 0, "dev": 0}
        lookup, self._paths = {}, {}
        for sample in samples:
            if type(sample) is not dict or set(sample) != SAMPLE_FIELDS or sample["split"] not in counts:
                raise LearningDataError("dataset-learning-sample-contract-invalid")
            split_name = sample["split"]
            for key, seen in memberships.items():
                value = sample[key]
                if type(value) is not str or not value or value in seen and (
                        seen[value] != split_name or key in ("sample_id", "image", "image_sha256")):
                    raise LearningDataError("dataset-learning-duplicate-or-split-leakage-" + key)
                seen[value] = split_name
            validate_waypoint_points(sample["waypoints_ego_m"])
            if (sample["image_width"], sample["image_height"]) != (640, 360):
                raise LearningDataError("dataset-learning-image-shape-invalid")
            lookup[sample["sample_id"]] = sample
            counts[split_name] += 1
        if counts != manifest["split_counts"]:
            raise LearningDataError("dataset-learning-split-count-mismatch")
        pairs = match_reference_arc_pairs(samples, dataset_config)
        selections = select_training_sets(samples, pairs, dataset_config)
        if pairs != manifest["pairs"] or selections != manifest["selections"]:
            raise LearningDataError("dataset-learning-selection-reconstruction-mismatch")
        selected_ids = selections[selection][split + "_sample_ids"]
        self.samples = [lookup[identifier] for identifier in selected_ids]
        if any(sample["split"] != split for sample in self.samples):
            raise LearningDataError("dataset-learning-selection-split-mismatch")
        for sample in self.samples:
            self._paths[sample["sample_id"]] = _sample_path(self.root, sample["image"])
        self.config = config or load_learning_config()
        self.split, self.selection, self.augment = split, selection, augment
        self.seed = self.config.seed if seed is None else seed
        if type(self.seed) is not int or self.seed < 0:
            raise LearningDataError("dataset-learning-seed-invalid")
        self.epoch = 0

    def assert_manifest_unchanged(self):
        super().assert_manifest_unchanged()
        if (self.validation_report_path.is_symlink()
                or hashlib.sha256(self.validation_report_path.read_bytes()).hexdigest() != self.validation_report_sha256):
            raise LearningDataError("dataset-learning-validation-report-changed")
