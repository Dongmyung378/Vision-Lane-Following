"""Offline proof of existing notebook reuse and source-bound T4 reporting."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
import zipfile

from carla_tools.contracts import canonical_sha256
from carla_tools.development_training import ARMS, load_development_protocol
from scripts import kaggle_development_training_bootstrap as bootstrap
from scripts import prepare_kaggle_development_training as packager
import test_kaggle_recovery_training as recovery_fixtures


class DevelopmentKaggleTests(unittest.TestCase):
    def setUp(self):
        fixture = recovery_fixtures.RecoveryKaggleTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture, self.root = fixture, fixture.root
        self.member = "experiment/seed43/central_only/refined_checkpoint.pt"
        self.archive = self.root / "selected_training_artifacts.zip"
        with zipfile.ZipFile(self.archive, "x") as archive:
            archive.write(fixture.initial, self.member)

    def prepare(self, **changes):
        fixture = self.fixture
        arguments = dict(manifest_path=fixture.manifest_path, validation_report_path=fixture.report_path,
            source_audit_path=fixture.audit_path, validation_amendment_path=fixture.amendment_path,
            archive_sha256="a" * 64, source_notebook_version=3, mount_source_version=7,
            mount_source_code_sha256="c" * 64, initial_archive_path=self.archive,
            initial_checkpoint_path=fixture.initial, initial_notebook_version=20, initial_checkpoint_member=self.member)
        arguments.update(changes)
        return packager.prepare(fixture.metadata, self.root / "development_bundle", **arguments)

    def test_bundle_reuses_exact_notebooks_and_checkpoint_without_authentication_material(self):
        report = self.prepare()
        root = self.root / "development_bundle"
        metadata = json.loads((root / "kernel-metadata.json").read_text())
        self.assertEqual(metadata["id"], packager.NOTEBOOK_ID)
        self.assertEqual(metadata["title"], packager.NOTEBOOK_TITLE)
        self.assertEqual(metadata["kernel_sources"], [packager.MOUNT_NOTEBOOK_ID + "/7"])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertTrue(metadata["is_private"])
        self.assertFalse(report["creates_new_notebook"])
        self.assertFalse(report["creates_dataset"])
        self.assertFalse(report["submitted"])
        code = (root / "development_training.py").read_bytes()
        self.assertNotIn(b"SECRET-SENTINEL", code)
        self.assertNotIn(b"api_key", json.dumps(metadata).encode())
        self.assertEqual(hashlib.sha256(code).hexdigest(), report["code_sha256"])
        self.assertLess(len(code), 1_000_000)
        namespace = {}
        exec(compile(code.decode().rsplit("raise SystemExit(main_development())", 1)[0], "bundle", "exec"), namespace)
        frozen = namespace["FROZEN_INPUT"]
        self.assertEqual(frozen["initial_checkpoint_member"], self.member)
        self.assertEqual(frozen["initial_notebook_version"], 20)
        self.assertEqual(frozen["initial_checkpoint_sha256"], hashlib.sha256(self.fixture.initial.read_bytes()).hexdigest())
        self.assertEqual(frozen["mount_source_code_sha256"], "c" * 64)
        self.assertEqual(canonical_sha256(frozen["training_protocol"]), frozen["training_protocol_sha256"])
        for relative, text in namespace["PROJECT_SOURCES"].items():
            self.assertEqual(text.encode(), (packager.ROOT / relative).read_bytes())
        compile(namespace["DOWNLOAD_HELPER_SOURCE"], "download", "exec")

    def test_private_protocol_is_embedded_without_rewriting_public_config(self):
        before = (packager.ROOT / "configs/development_training.json").read_bytes()
        protocol = load_development_protocol().to_dict()
        protocol["epoch_dev_views"] = ["clean", "blur"]
        private = self.root / "private_protocol.json"
        private.write_text(json.dumps(protocol))
        report = self.prepare(protocol_path=private)
        self.assertEqual(report["frozen_input"]["training_protocol"], protocol)
        self.assertEqual((packager.ROOT / "configs/development_training.json").read_bytes(), before)

    def test_new_public_notebook_invalid_mount_and_escaped_initial_member_are_rejected(self):
        for changes in ({"mount_source_version": 0}, {"mount_source_code_sha256": "bad"},
                        {"initial_checkpoint_member": "experiment/../refined_checkpoint.pt"},
                        {"initial_checkpoint_member": "experiment/seed43/checkpoint.pt"}):
            with self.subTest(change=changes), self.assertRaises(ValueError):
                self.prepare(**changes)
        self.fixture.metadata.write_text(json.dumps({"id": "owner/new", "title": packager.NOTEBOOK_TITLE, "is_private": True}))
        with self.assertRaisesRegex(ValueError, "existing-private"):
            self.prepare()
        self.fixture.metadata.write_text(json.dumps({"id": packager.NOTEBOOK_ID, "title": packager.NOTEBOOK_TITLE, "is_private": False}))
        with self.assertRaisesRegex(ValueError, "existing-private"):
            self.prepare()

    def test_initial_tensor_archive_cannot_be_substituted(self):
        self.fixture.initial.write_bytes(b"changed-reviewed-checkpoint")
        with self.assertRaisesRegex(ValueError, "initial-member-invalid|archive-checkpoint-binding-invalid"):
            self.prepare()

    def test_epoch_event_allows_only_bounded_fields_and_finite_learning_metrics(self):
        value = {"phase": "development-training", "arm": "appearance_robust", "seed": 42, "epoch": 1,
                 "train_loss": .2, "dev_loss": .3, "optimizer_updates": 79, "secret": "SENTINEL"}
        clean = bootstrap.visible_epoch_event(value)
        self.assertNotIn("secret", clean)
        self.assertEqual(clean["epoch"], 1)
        for change in ({"dev_loss": float("nan")}, {"epoch": True}, {"arm": "unknown"}, {"phase": "other"}):
            self.assertIsNone(bootstrap.visible_epoch_event(dict(value, **change)))

    def _report(self):
        protocol = load_development_protocol().to_dict()
        budget = protocol["training"]
        common = {"status": "PASS", "fixture_injection": False, "final_town_used": False,
                  "manifest_sha256": "a" * 64, "validation_report_sha256": "b" * 64,
                  "protocol_sha256": canonical_sha256(protocol), "source_sha256": {"source": "c" * 64},
                  "runtime": {"device": "cuda", "gpu": "Tesla T4"}}
        frozen = {"training_protocol_sha256": common["protocol_sha256"], "manifest_sha256": "a" * 64,
                  "transport_proof_sha256": "b" * 64, "initial_checkpoint_sha256": "d" * 64, "split_counts": {"dev": 40}}
        report = dict(common, protocol=protocol, closed_loop_verified=False, incumbent_pointer_updated=False,
            equal_budget_verified=True, initial_checkpoint_sha256="d" * 64, common_dev_sample_count=40,
            common_dev_selection_sha256="e" * 64, training={}, evaluations={})
        for seed in budget["seeds"]:
            report["training"][str(seed)], report["evaluations"][str(seed)] = {}, {}
            for arm in ARMS:
                updates = 79 * budget["epochs"]
                history = [{"epoch": epoch + 1, "train_samples": 5000, "dev_samples": 10240,
                    "optimizer_updates": 79 * (epoch + 1), "sampler_order_sha256": str(epoch), "learning_rate_factor": 1.}
                    for epoch in range(budget["epochs"])]
                report["training"][str(seed)][arm] = dict(common, arm=arm, seed=seed,
                    train_sample_count=5000, completed_epochs=budget["epochs"], optimizer_updates=updates,
                    planned_updates=updates, initial_checkpoint_sha256="d" * 64, initial_model_state_sha256="f" * 64,
                    epoch_dev_views=protocol["epoch_dev_views"], checkpoint_sha256="1" * 64,
                    training_selection_sha256="2" * 64, epoch_dev_selection_sha256="3" * 64,
                    config_sha256="4" * 64, history=history)
                report["evaluations"][str(seed)][arm] = dict(common, arm=arm, seed=seed,
                    split="dev", view="clean", sample_count=40, checkpoint_sha256="1" * 64,
                    common_dev_selection_sha256="e" * 64)
        report["initial_evaluation"] = dict(common, arm="incumbent", checkpoint_sha256="d" * 64,
            split="dev", view="clean", sample_count=40, common_dev_selection_sha256="e" * 64)
        return report, frozen, protocol

    def test_T4_gate_refuses_CPU_missing_arm_short_budget_and_different_evaluation(self):
        report, frozen, protocol = self._report()
        passed = bootstrap.validate_development_report(report, frozen, protocol)
        self.assertTrue(passed["actual_T4_training"])
        self.assertEqual(passed["seed_count"], 2)
        for field in ("cpu", "missing-arm", "short-budget", "different-dev", "wrong-view", "wrong-init"):
            changed = copy.deepcopy(report)
            if field == "cpu":
                changed["training"]["42"]["continued"]["runtime"] = {"device": "cpu", "gpu": None}
            elif field == "missing-arm":
                del changed["training"]["43"]["appearance_robust"]
            elif field == "short-budget":
                changed["training"]["42"]["continued"]["optimizer_updates"] -= 1
            elif field == "different-dev":
                changed["evaluations"]["43"]["appearance_robust"]["common_dev_selection_sha256"] = "9" * 64
            elif field == "wrong-view":
                changed["evaluations"]["43"]["appearance_robust"]["view"] = "noise"
            else:
                changed["initial_evaluation"]["checkpoint_sha256"] = "8" * 64
            with self.subTest(tamper=field), self.assertRaises((ValueError, bootstrap.DevelopmentBootstrapError)):
                bootstrap.validate_development_report(changed, frozen, protocol)


if __name__ == "__main__":
    unittest.main()
