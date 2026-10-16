"""Collection matrix, failure preservation and CPU-only preflight checks."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from carla_tools.dataset_v1 import load_dataset_v1_config
from carla_tools.pilot import load_pilot_config, select_pilot_routes
from scripts import run_dataset_pipeline as pipeline
from test_pilot import candidate_fixture


def catalog_fixture():
    return {"routes": select_pilot_routes(candidate_fixture(), load_pilot_config())}


class DatasetPipelineTests(unittest.TestCase):
    def test_frozen_matrix_covers_all_groups_with_one_central_four_recoveries(self):
        config = load_dataset_v1_config()
        plan = pipeline.collection_matrix(catalog_fixture()["routes"], config)
        self.assertEqual(len(plan), 700)
        self.assertEqual(len({item["episode"] for item in plan}), 700)
        self.assertEqual(sum(item["role"] == "central" for item in plan), 140)
        for group in {item["pair_group_id"] for item in plan}:
            trials = [item for item in plan if item["pair_group_id"] == group]
            self.assertEqual(len(trials), 5)
            self.assertEqual(sum(item["role"] == "central" for item in trials), 1)
            self.assertEqual(len({(item["route_id"], item["weather"], item["initial_reference_m"], item["seed"])
                                  for item in trials}), 1)
        # Mutable CARLA boundary values cannot mutate the deep-frozen config.
        plan[0]["condition"]["lateral_m"] = 10.
        self.assertEqual(config.conditions[0]["lateral_m"], 0.)

    def test_dry_run_serializes_without_importing_carla_or_starting_collection(self):
        with patch.object(pipeline.importlib, "import_module", side_effect=AssertionError("native import forbidden")):
            report = pipeline.dry_run(pipeline.ROOT / "configs/dataset_v1.json")
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["expected_episode_count"], 700)
        self.assertFalse(report["native_collection_started"])
        self.assertEqual(report["reserved_final_state"], "UNOPENED")

    def test_archive_preserves_partial_images_and_reports_without_source_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native = root / "scratch" / "dataset"
            native.mkdir(parents=True)
            (native / "partial.png").write_bytes(b"original-rgb-png")
            (native / "dataset_report.json").write_text('{"status":"FAIL"}\n')
            report = pipeline.archive_dataset(native, root / "artifacts")
            self.assertGreater(report["archive_bytes"], 0)
            with zipfile.ZipFile(root / "artifacts/dataset_artifacts.zip") as archive:
                self.assertEqual(archive.read("dataset/partial.png"), b"original-rgb-png")
                self.assertEqual(archive.getinfo("dataset/partial.png").compress_type, zipfile.ZIP_STORED)
                self.assertIn("FAIL", archive.read("dataset/dataset_report.json").decode())
            self.assertEqual((native / "partial.png").read_bytes(), b"original-rgb-png")
            with self.assertRaisesRegex(ValueError, "already-exists"):
                pipeline.archive_dataset(native, root / "artifacts")

    def test_first_native_failure_is_accounted_and_all_partial_bytes_are_published(self):
        client = SimpleNamespace(get_client_version=lambda: "0.9.16", get_server_version=lambda: "0.9.16",
            set_timeout=lambda _value: None, get_world=lambda: SimpleNamespace(get_map=lambda: object()))
        module = SimpleNamespace(Client=lambda *_args: client,
            WeatherParameters=SimpleNamespace(**{name: SimpleNamespace(cloudiness=float(index))
                for index, name in enumerate(pipeline.DEVELOPMENT_WEATHERS)}))
        def fail_trial(_route, _case, _condition, path, *_args, **_kwargs):
            Path(path).mkdir(parents=True)
            (Path(path) / "partial.png").write_bytes(b"native-failure-kept")
            return {"status": "FAIL", "step_count": 3}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(pipeline.importlib, "import_module", return_value=module), \
                 patch.object(pipeline, "survey_pilot_routes", return_value=catalog_fixture()), \
                 patch.object(pipeline, "runtime_source_hashes", return_value={"source.py": "a" * 64}), \
                 patch.object(pipeline, "run_dataset_v1_trial", side_effect=fail_trial):
                report = pipeline.run_pipeline(root / "scratch/dataset", artifact_dir=root / "working")
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["episode_count"], 1)
            self.assertEqual(report["native_frame_count"], 3)
            self.assertEqual(report["error"]["code"], "dataset-native-trial-operation-failed")
            with zipfile.ZipFile(root / "working/dataset_artifacts.zip") as archive:
                partial = [name for name in archive.namelist() if name.endswith("partial.png")]
                self.assertEqual(len(partial), 1)
                self.assertEqual(archive.read(partial[0]), b"native-failure-kept")
            self.assertEqual(json.loads((root / "working/dataset_progress.json").read_text())["episode_count"], 1)

    def test_existing_native_output_is_never_overwritten_without_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "keep.txt"
            marker.write_text("keep")
            report = pipeline.run_pipeline(root)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["error"]["code"], "dataset-output-already-exists")
            self.assertEqual(marker.read_text(), "keep")
            self.assertFalse((root / "dataset_report.json").exists())


if __name__ == "__main__":
    unittest.main()
