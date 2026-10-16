"""Source images and held-out selections constrain exported inspection views."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from carla_tools.capture import _calibration, _mount_matrix
from carla_tools.config import CollectionConfig
from carla_tools.dataset_artifacts import export_dataset_artifacts, validate_dataset_artifacts
from carla_tools.dataset_v1 import build_dataset_v1_manifest, validate_dataset_v1_manifest
from carla_tools.projection import decode_rgb8_png
from test_collection_session import multiply
from test_dataset_v1 import dataset_fixture, native_boundary_fixture


def inspection_fixture(root):
    """Add independently defined fixed camera/colour evidence to unit traces."""
    config, routes, descriptors = dataset_fixture(root)
    collection = CollectionConfig()
    mount = _mount_matrix(collection)
    calibration = _calibration(collection, mount)
    for ordinal, descriptor in enumerate(descriptors):
        folder = root / descriptor["episode"]
        metadata = json.loads((folder / "episode.json").read_text(encoding="utf-8"))
        metadata["camera_calibration"] = copy.deepcopy(calibration)
        rows = [json.loads(line) for line in (folder / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
        for index, row in enumerate(rows):
            rgb = bytes((ordinal + 1, index + 1, 127)) * (640 * 360)
            bgra_pixel = [127, index + 1, ordinal + 1, 255]
            bgra = bytes(bgra_pixel) * (640 * 360)
            row["image"]["color_provenance"] = {"source_format": "BGRA8", "output_format": "RGB8",
                "source_sha256": hashlib.sha256(bgra).hexdigest(), "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
                "sample_pixels": [{"index": pixel, "bgra": bgra_pixel} for pixel in (0, 115200, 230399)]}
            row["camera"] = {"transform": {"matrix": multiply(row["vehicle"]["transform"]["matrix"], mount)}}
        (folder / "episode.json").write_text(json.dumps(metadata), encoding="utf-8")
        (folder / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    path = build_dataset_v1_manifest(root, routes, descriptors, config)
    with patch("carla_tools.dataset_v1._native_episode_gate", side_effect=native_boundary_fixture):
        gate = validate_dataset_v1_manifest(path)
    if gate["status"] != "PASS":
        raise AssertionError(gate)
    return path, gate


class DatasetArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.path, cls.gate = inspection_fixture(cls.root)
        cls.manifest = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.output = cls.root / "derived"
        cls.report = export_dataset_artifacts(cls.path, cls.gate, cls.output)
        if cls.report["status"] != "PASS":
            raise AssertionError(cls.report)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_split_and_ordered_training_tables_keep_development_out_of_training(self):
        def read_csv(name):
            return list(csv.DictReader(io.StringIO((self.output / name).read_text(encoding="utf-8"))))
        all_rows = read_csv("split_manifest.csv")
        self.assertEqual(len(all_rows), 140)
        self.assertEqual({row["sample_id"] for row in all_rows},
                         {sample["sample_id"] for sample in self.manifest["samples"]})
        selections = self.manifest["selections"]
        a, b, dev = read_csv("central_only_train.csv"), read_csv("central_recovery_train.csv"), read_csv("common_dev.csv")
        self.assertEqual([row["sample_id"] for row in a], selections["central_only"]["train_sample_ids"])
        self.assertEqual([row["sample_id"] for row in b], selections["central_recovery"]["train_sample_ids"])
        self.assertEqual([int(row["selection_index"]) for row in a], [0, 1, 2, 3])
        self.assertTrue(all(row["split"] == "train" for row in a + b))
        self.assertTrue(all(row["split"] == "dev" for row in dev))
        self.assertFalse({row["sample_id"] for row in dev} & {row["sample_id"] for row in a + b})
        references = json.loads((self.output / "training_selections.json").read_text(encoding="utf-8"))
        self.assertEqual(references["source_manifest_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(references["common_dev_sample_ids"], [row["sample_id"] for row in dev])

    def test_visuals_cover_fixed_strata_and_preserve_source_pixels_outside_markers(self):
        visuals = json.loads((self.output / "visualization_manifest.json").read_text(encoding="utf-8"))
        records = visuals["records"]
        expected = {(split, case, role) for split in ("train", "dev")
                    for case in ("straight", "left_curve", "right_curve") for role in ("central", "recovery")}
        self.assertEqual({(row["split"], row["case"], row["role"]) for row in records}, expected)
        self.assertEqual(len(records), 24)
        self.assertEqual({row["position"] for row in records}, {"early", "middle"})
        self.assertEqual(len(visuals["contact_sheets"]), 2)
        for record in records:
            width, height, overlay = decode_rgb8_png((self.output / record["overlay"]).read_bytes())
            sw, sh, source = decode_rgb8_png((self.root / record["source_image"]).read_bytes(),
                                            record["source_image_sha256"], (640, 360))
            self.assertEqual((width, height), (sw, sh))
            self.assertEqual(overlay[-3:], source[-3:])
            self.assertFalse(record["prediction_or_inference_used"])
            self.assertEqual(len(record["projections"]), 5)
            self.assertTrue(all(row["coordinate_clamped"] is False for row in record["projections"]))
            # Recompute one projection directly from R^T(world - camera),
            # independently of the production project_waypoints helper.
            projected = record["projections"][2]
            matrix, world = record["camera_world_matrix"], projected["world_xyz_m"]
            local = [sum(matrix[r][axis] * (world[r] - matrix[r][3]) for r in range(3)) for axis in range(3)]
            if local[0] > 1e-6:
                self.assertAlmostEqual(projected["uv_pixels"][0], 320.0 * local[1] / local[0] + 320.0)
                self.assertAlmostEqual(projected["uv_pixels"][1], -320.0 * local[2] / local[0] + 180.0)

    def test_output_is_deterministic_and_never_repeats_full_native_episode_replay(self):
        with patch("carla_tools.dataset_v1._native_episode_gate", side_effect=AssertionError("full replay repeated")):
            other = export_dataset_artifacts(self.path, self.gate, self.root / "same-views")
        self.assertEqual(other["status"], "PASS", other)
        self.assertEqual(other["files_sha256"], self.report["files_sha256"])
        self.assertEqual(other["artifact_manifest_sha256"], self.report["artifact_manifest_sha256"])
        self.assertFalse(other["full_native_replay_repeated"])

    def test_saved_artifacts_reconstruct_and_unbound_additional_files_are_rejected(self):
        report = validate_dataset_artifacts(self.path, self.gate, self.output)
        self.assertEqual(report["status"], "PASS", report)
        extra = self.output / "unbound.csv"
        try:
            extra.write_text("not part of the bound dataset\n", encoding="utf-8")
            report = validate_dataset_artifacts(self.path, self.gate, self.output)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["errors"][0]["code"], "dataset-artifacts-unbound-extra-file")
        finally:
            extra.unlink()

    def test_missing_or_unbound_pass_gate_cannot_publish_outputs(self):
        for index, mutate in enumerate((lambda gate: gate.update(status="FAIL"),
                                       lambda gate: gate.update(manifest_sha256="0" * 64),
                                       lambda gate: gate["checks"].pop("every_source_png"))):
            gate = copy.deepcopy(self.gate)
            mutate(gate)
            output = self.root / ("refused-" + str(index))
            report = export_dataset_artifacts(self.path, gate, output)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["errors"][0]["code"], "dataset-artifacts-unbound-native-gate")
            self.assertFalse(output.exists())

    def test_source_image_changes_are_rejected_even_when_original_native_gate_was_pass(self):
        visuals = json.loads((self.output / "visualization_manifest.json").read_text(encoding="utf-8"))
        image = self.root / visuals["records"][0]["source_image"]
        original = image.read_bytes()
        try:
            image.write_bytes(original + b"changed")
            report = export_dataset_artifacts(self.path, self.gate, self.root / "changed-image")
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["errors"][0]["code"], "projection-image-hash-mismatch")
        finally:
            image.write_bytes(original)

    def test_rehashed_csv_tampering_cannot_redefine_selection_order(self):
        csv_path, inventory_path = self.output / "central_only_train.csv", self.output / "artifact_manifest.json"
        before_csv, before_inventory = csv_path.read_bytes(), inventory_path.read_bytes()
        try:
            lines = before_csv.decode("utf-8").splitlines(keepends=True)
            lines[1], lines[2] = lines[2], lines[1]
            csv_path.write_bytes("".join(lines).encode("utf-8"))
            inventory = json.loads(before_inventory)
            inventory["files_sha256"]["central_only_train.csv"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
            report = validate_dataset_artifacts(self.path, self.gate, self.output)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["errors"][0]["code"], "dataset-artifacts-reconstruction-mismatch")
        finally:
            csv_path.write_bytes(before_csv)
            inventory_path.write_bytes(before_inventory)

    def test_existing_output_is_refused_without_overwriting_derived_files(self):
        before = (self.output / "artifact_manifest.json").read_bytes()
        report = export_dataset_artifacts(self.path, self.gate, self.output)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["errors"][0]["code"], "dataset-artifacts-output-already-exists")
        self.assertEqual((self.output / "artifact_manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
