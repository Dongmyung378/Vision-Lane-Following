"""Immutable source transport, private notebook reuse, and archive boundaries."""
import hashlib
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
try:
    import requests
except ModuleNotFoundError:
    # The CPU learning environment does not require the optional cloud SDK.
    # All network operations in this suite are injected streaming fixtures.
    requests = ModuleType("requests")
    class ConnectionFailure(Exception):
        pass
    class ReadTimeout(Exception):
        pass
    class BrokenChunk(Exception):
        pass
    class Session:
        def send(self, *_args, **_kwargs):
            raise AssertionError("network-access-forbidden-in-fixture")
    requests.exceptions = SimpleNamespace(ConnectionError=ConnectionFailure, Timeout=ReadTimeout,
                                           ChunkedEncodingError=BrokenChunk)
    requests.sessions = SimpleNamespace(Session=Session)
    requests.get = lambda *_args, **_kwargs: None

from carla_tools.contracts import canonical_sha256
from carla_tools.dataset_v1 import GATE_VERSION, SCOPE, VALIDATION_CHECKS, _implementation_hashes
from scripts import kaggle_recovery_training_bootstrap as bootstrap
from scripts import prepare_kaggle_recovery_training as packager


class StreamingResponse:
    def __init__(self, status, headers=None, blocks=(), error=None, url="https://object.invalid/pinned?secret=TRANSIENT"):
        self.status_code, self.headers, self.blocks, self.error, self.url = status, headers or {}, blocks, error, url
        self.closed = False

    @property
    def content(self):
        raise AssertionError("whole-response-buffering-forbidden")

    def iter_content(self, *, chunk_size):
        if chunk_size != 1024 * 1024:
            raise AssertionError("unexpected-streaming-chunk-size")
        yield from self.blocks
        if self.error:
            raise self.error

    def close(self):
        self.closed = True


class RecoveryKaggleTests(unittest.TestCase):
    def setUp(self):
        request_patch = patch.dict("sys.modules", {"requests": requests})
        request_patch.start()
        self.addCleanup(request_patch.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.metadata = self.root / "metadata.json"
        self.metadata.write_text(json.dumps({"id": packager.NOTEBOOK_ID, "title": packager.NOTEBOOK_TITLE,
            "is_private": True, "api_key": "SECRET-SENTINEL", "kernel_sources": ["owner/old-source"]}))
        self.config = json.loads((packager.ROOT / "configs/dataset_v1.json").read_text())
        self.manifest = {"scope": SCOPE, "status": "COMPLETE", "town": "Town01",
            "reserved_final_town": "Town02", "reserved_final_state": "UNOPENED",
            "config": self.config, "config_sha256": canonical_sha256(self.config),
            "implementation_sha256": _implementation_hashes(),
            "episodes": [{"episode_id": "fixture_a"}, {"episode_id": "fixture_b"}],
            "sample_count": 10, "split_counts": {"train": 6, "dev": 4}}
        self.manifest["snapshot_sha256"] = canonical_sha256(self.manifest)
        self.manifest_path = self.root / "manifest.json"
        self.write(self.manifest_path, self.manifest)
        self.report = {"status": "PASS", "scope": SCOPE, "gate_version": GATE_VERSION, "errors": [],
            "manifest_sha256": bootstrap.sha256_file(self.manifest_path),
            "snapshot_sha256": self.manifest["snapshot_sha256"], "config_sha256": canonical_sha256(self.config),
            "sample_count": 10, "episode_count": 2, "split_counts": {"train": 6, "dev": 4},
            "checks": dict.fromkeys(VALIDATION_CHECKS, True),
            "episode_metrics": [{"episode_id": name, "checks": {"all_frames_replayed": True},
                                 "evaluation_metrics": {"frame_count": 5}} for name in ("fixture_a", "fixture_b")]}
        self.report_path = self.root / "report.json"
        self.write(self.report_path, self.report)
        self.amendment = {"status": "FROZEN_BEFORE_CPU_DERIVATION", "original_archive_sha256": "a" * 64,
            "same_700_source_episodes": True, "same_sample_budgets": True, "new_GPU_job_started": False}
        self.amendment_path = self.root / "amendment.json"
        self.write(self.amendment_path, self.amendment)
        self.audit = {"status": "PASS", "archive_sha256": "a" * 64,
            "archive_bytes": 16,
            "manifest_sha256": self.report["manifest_sha256"], "snapshot_sha256": self.report["snapshot_sha256"],
            "episode_count": 2, "validation_amendment_sha256": bootstrap.sha256_file(self.amendment_path),
            "parallel_native_replay": {"actual_native_replays": 2}}
        self.audit_path = self.root / "audit.json"
        self.write(self.audit_path, self.audit)
        self.initial = self.root / "checkpoint.pt"
        self.initial.write_bytes(b"synthetic-reviewed-checkpoint")
        self.initial_archive = self.root / "training_artifacts.zip"
        with zipfile.ZipFile(self.initial_archive, "x") as archive:
            archive.write(self.initial, packager.CHECKPOINT_MEMBER)

    def write(self, path, data):
        path.write_text(json.dumps(data, sort_keys=True) + "\n")

    def summary(self):
        return packager.trusted_cpu_summary(self.manifest_path, self.report_path, self.audit_path,
                                            self.amendment_path, archive_sha256="a" * 64)[0]

    def prepare(self, **changes):
        arguments = dict(manifest_path=self.manifest_path, validation_report_path=self.report_path,
            source_audit_path=self.audit_path, validation_amendment_path=self.amendment_path,
            archive_sha256="a" * 64, source_notebook_version=3, initial_archive_path=self.initial_archive,
            initial_checkpoint_path=self.initial, initial_notebook_version=2)
        arguments.update(changes)
        return packager.prepare(self.metadata, self.root / "bundle", **arguments)

    def test_compact_summary_retains_binding_and_distinguishes_prior_CPU_replay(self):
        summary = self.summary()
        self.assertNotIn("episode_metrics", summary)
        self.assertEqual(summary["original_full_report_sha256"], bootstrap.sha256_file(self.report_path))
        self.assertEqual(summary["original_episode_metrics_sha256"], canonical_sha256(self.report["episode_metrics"]))
        self.assertEqual(summary["original_episode_metrics_count"], 2)
        self.assertEqual(summary["transport_kind"], "validated-report-transport-summary")
        self.assertFalse(summary["new_native_replay_performed"])
        self.assertEqual(summary["checks"], self.report["checks"])

    def test_missing_native_episode_and_failed_check_cannot_create_summary(self):
        self.report["episode_metrics"].pop()
        self.write(self.report_path, self.report)
        with self.assertRaisesRegex(ValueError, "complete-native-episode"):
            self.summary()
        self.report["episode_metrics"].append({"episode_id": "fixture_b", "checks": {"all_frames_replayed": False},
                                               "evaluation_metrics": {}})
        self.write(self.report_path, self.report)
        with self.assertRaisesRegex(ValueError, "complete-native-episode"):
            self.summary()

    def test_unbound_CPU_gate_and_changed_manifest_are_refused(self):
        self.report["manifest_sha256"] = "b" * 64
        self.write(self.report_path, self.report)
        with self.assertRaisesRegex(ValueError, "complete-CPU-report-binding"):
            self.summary()
        self.manifest["sample_count"] = 99
        self.write(self.manifest_path, self.manifest)
        with self.assertRaisesRegex(ValueError, "snapshot-or-source"):
            self.summary()

    def test_source_audit_and_original_capture_hash_are_required(self):
        self.audit["archive_sha256"] = "b" * 64
        self.write(self.audit_path, self.audit)
        with self.assertRaisesRegex(ValueError, "independent-source-audit"):
            self.summary()

    def test_exact_existing_private_notebook_and_no_credentials_in_generated_code(self):
        report = self.prepare()
        output = self.root / "bundle"
        metadata = json.loads((output / "kernel-metadata.json").read_text())
        code = (output / "recovery_training.py").read_bytes()
        self.assertEqual(metadata["id"], packager.NOTEBOOK_ID)
        self.assertEqual(metadata["title"], packager.NOTEBOOK_TITLE)
        self.assertEqual(metadata["kernel_sources"], [])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertNotIn("api_key", metadata)
        self.assertNotIn(b"SECRET-SENTINEL", code)
        self.assertFalse(report["creates_new_notebook"])
        self.assertFalse(report["creates_dataset"])
        self.assertFalse(report["submitted"])
        self.assertFalse(report["local_gpu_used"])
        self.assertLess(len(code), 1_000_000)
        self.assertNotIn(b"\r\n", code)
        namespace = {}
        exec(compile(code.decode().rsplit("raise SystemExit(main_recovery())", 1)[0], "bundle", "exec"), namespace)
        self.assertEqual(namespace["FROZEN_INPUT"]["manifest_sha256"], self.report["manifest_sha256"])
        self.assertEqual(namespace["FROZEN_INPUT"]["initial_checkpoint_sha256"], bootstrap.sha256_file(self.initial))
        self.assertEqual(hashlib.sha256(namespace["TRUSTED_CPU_PROOF"].encode()).hexdigest(),
                         report["frozen_input"]["transport_proof_sha256"])
        for name, text in namespace["PROJECT_SOURCES"].items():
            self.assertEqual(text.encode(), (packager.ROOT / name).read_bytes())

    def test_public_or_new_notebook_and_overwrite_are_refused(self):
        self.metadata.write_text(json.dumps({"id": "owner/new-notebook", "title": packager.NOTEBOOK_TITLE, "is_private": True}))
        with self.assertRaisesRegex(ValueError, "existing-private"):
            self.prepare()
        self.metadata.write_text(json.dumps({"id": packager.NOTEBOOK_ID, "title": packager.NOTEBOOK_TITLE, "is_private": False}))
        with self.assertRaisesRegex(ValueError, "existing-private"):
            self.prepare()

    def test_source_versions_and_time_limits_are_bounded(self):
        for changes in ({"source_notebook_version": True}, {"initial_notebook_version": 0},
                        {"maximum_seconds": 50000}, {"training_seconds": 21400}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.prepare(**changes)

    def test_initial_checkpoint_must_match_prior_archive_bytes(self):
        self.initial.write_bytes(b"other-checkpoint")
        with self.assertRaisesRegex(ValueError, "initial-member-invalid"):
            self.prepare()

    def test_archive_paths_duplicates_and_expansion_budget_are_rejected(self):
        for index, name in enumerate(("../escape", "dataset/../../escape", "dataset\\escape", "dataset/C:escape")):
            archive = self.root / ("unsafe" + str(index) + ".zip")
            with zipfile.ZipFile(archive, "x") as packed:
                member = zipfile.ZipInfo("placeholder")
                member.filename = name
                packed.writestr(member, b"unsafe")
            with self.subTest(name=name), self.assertRaisesRegex(bootstrap.RecoveryBootstrapError, "member-invalid"):
                bootstrap.safe_extract_dataset(archive, self.root / ("extract" + str(index)))
        archive = self.root / "oversized.zip"
        with zipfile.ZipFile(archive, "x") as packed:
            packed.writestr("dataset/data.bin", b"123456789")
        with self.assertRaisesRegex(bootstrap.RecoveryBootstrapError, "budget-exceeded"):
            bootstrap.safe_extract_dataset(archive, self.root / "extract_budget", maximum_bytes=8)

    def test_only_hash_verified_existing_cloud_archive_is_accepted(self):
        archive = self.root / "dataset_artifacts.zip"
        archive.write_bytes(b"verified-cloud-bytes")
        expected = bootstrap.sha256_file(archive)
        mounted = self.root / "mounted"
        mounted.mkdir()
        path, transport = bootstrap.resolve_frozen_archive(mounted, expected, download=lambda: archive)
        self.assertEqual(path, archive)
        self.assertEqual(transport, "platform-authenticated-existing-notebook-version")
        with self.assertRaisesRegex(bootstrap.RecoveryBootstrapError, "hash-mismatch"):
            bootstrap.resolve_frozen_archive(mounted, "b" * 64, download=lambda: archive)

    def test_previous_training_checkpoint_is_single_member_extracted_and_hash_checked(self):
        frozen = {"initial_archive_sha256": bootstrap.sha256_file(self.initial_archive),
                  "initial_checkpoint_sha256": bootstrap.sha256_file(self.initial),
                  "initial_checkpoint_member": packager.CHECKPOINT_MEMBER}
        destination = self.root / "isolated/checkpoint.pt"
        result = bootstrap.extract_frozen_initial_checkpoint(self.initial_archive, destination, frozen)
        self.assertEqual(result.read_bytes(), self.initial.read_bytes())
        frozen["initial_checkpoint_sha256"] = "b" * 64
        with self.assertRaisesRegex(bootstrap.RecoveryBootstrapError, "checkpoint-sha256-mismatch"):
            bootstrap.extract_frozen_initial_checkpoint(self.initial_archive, self.root / "bad/checkpoint.pt", frozen)

    def test_cloud_helper_uses_explicit_existing_version_and_suppresses_auth_and_URL_output(self):
        helper = bootstrap.cloud_download_helper_source(7, self.root / "deps", self.root / "cloud",
            expected_sha256="a" * 64, expected_bytes=16)
        compile(helper, "cloud", "exec")
        self.assertIn(packager.DATA_NOTEBOOK_ID + "/versions/7", helper)
        self.assertIn("redirect_stdout", helper)
        self.assertIn("redirect_stderr", helper)
        self.assertIn("client.kernels.kernels_api_client.download_kernel_output(request)", helper)
        self.assertIn("assert request.version_number==7", helper)
        self.assertNotIn("notebook_output_download(", helper)
        self.assertNotIn("api_key", helper)
        initial = bootstrap.cloud_download_helper_source(2, self.root / "deps", self.root / "initial",
            notebook_id=packager.NOTEBOOK_ID, filename="training_artifacts.zip", expected_sha256="a" * 64, expected_bytes=16)
        self.assertIn(packager.NOTEBOOK_ID + "/versions/2", initial)
        self.assertIn("training_artifacts.zip", initial)
        with self.assertRaises(bootstrap.RecoveryBootstrapError):
            bootstrap.cloud_download_helper_source(7, self.root, self.root, notebook_id="owner/new-notebook",
                expected_sha256="a" * 64, expected_bytes=16)

    def test_cloud_helper_checks_returned_version_and_never_forwards_library_chatter(self):
        payload = b"reviewed-output"
        calls = []
        def request(handle, filename):
            calls.append((handle, filename))
            return SimpleNamespace(version_number=7)
        def resolve(_request):
            print("SECRET-SENTINEL-library-chatter")
            return StreamingResponse(200, {"Content-Length": str(len(payload))})
        client = SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=SimpleNamespace(download_kernel_output=resolve)))
        hub, resolver, handles, clients = (ModuleType(name) for name in
            ("kagglehub", "kagglehub.http_resolver", "kagglehub.handle", "kagglehub.clients"))
        hub.__version__ = "synthetic-test"
        resolver._build_notebook_download_request = request
        handles.parse_notebook_handle = lambda value: value
        clients.build_kaggle_client = lambda: contextlib.nullcontext(client)
        modules = {"kagglehub": hub, "kagglehub.http_resolver": resolver,
                   "kagglehub.handle": handles, "kagglehub.clients": clients}
        helper = bootstrap.cloud_download_helper_source(7, self.root / "deps", self.root / "cloud",
            expected_sha256=hashlib.sha256(payload).hexdigest(), expected_bytes=len(payload))
        output = io.StringIO()
        ranged = StreamingResponse(206, {"Content-Range": f"bytes 0-{len(payload)-1}/{len(payload)}",
                                        "Content-Length": str(len(payload))}, [payload])
        with patch.dict("sys.modules", modules), patch("requests.get", return_value=ranged), patch("logging.disable"), contextlib.redirect_stdout(output):
            exec(compile(helper, "cloud", "exec"), {})
        value = json.loads(output.getvalue())
        self.assertEqual(value["status"], "PASS")
        self.assertEqual(value["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(calls[0][0], packager.DATA_NOTEBOOK_ID + "/versions/7")
        self.assertNotIn("SECRET-SENTINEL", output.getvalue())
        self.assertNotIn("TRANSIENT", output.getvalue())
        resolver._build_notebook_download_request = lambda *_args: SimpleNamespace(version_number=8)
        failed = io.StringIO()
        with patch.dict("sys.modules", modules), patch("sys.path", []), patch("logging.disable"), contextlib.redirect_stdout(failed):
            with self.assertRaises(SystemExit):
                exec(compile(helper, "cloud", "exec"), {})
        self.assertEqual(json.loads(failed.getvalue())["status"], "FAIL")

    def ranged_download(self, *, request_get=None, resolve=None, destination="download.zip", **kwargs):
        payload = b"0123456789"
        initial = lambda: StreamingResponse(200, {"Content-Length": "10", "ETag": '"fixed"'})
        def get(_url, *, headers, stream, timeout):
            self.assertTrue(stream)
            self.assertEqual(headers["Accept-Encoding"], "identity")
            self.assertEqual(headers["If-Range"], '"fixed"')
            start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
            return StreamingResponse(206, {"Content-Range": f"bytes {start}-{end}/10",
                "Content-Length": str(end-start+1), "ETag": '"fixed"'}, [payload[start:end+1]])
        return bootstrap.resumable_http_download(self.root / destination, hashlib.sha256(payload).hexdigest(),
            len(payload), resolve or initial, request_get=request_get or get, range_bytes=4, sleep=lambda _delay: None, **kwargs)

    def test_range_streaming_completes_without_buffering_entire_response(self):
        result = self.ranged_download()
        self.assertEqual((self.root / "download.zip").read_bytes(), b"0123456789")
        self.assertEqual(result["range_requests"], 3)
        self.assertEqual(result["retries"], 0)
        self.assertNotIn("TRANSIENT", json.dumps(result))
        self.assertFalse((self.root / "download.zip.part").exists())

    def test_mid_body_disconnect_resumes_exact_written_prefix_without_duplication(self):
        ranges, responses = [], []
        def get(_url, *, headers, **_kwargs):
            requested = headers["Range"]
            ranges.append(requested)
            start, end = map(int, requested.removeprefix("bytes=").split("-"))
            body = b"0123456789"[start:end+1]
            response = StreamingResponse(206, {"Content-Range": f"bytes {start}-{end}/10", "Content-Length": str(len(body)), "ETag": '"fixed"'},
                [body[:2]] if len(ranges)==1 else [body], requests.exceptions.ChunkedEncodingError() if len(ranges)==1 else None)
            responses.append(response)
            return response
        result = self.ranged_download(request_get=get)
        self.assertEqual(ranges, ["bytes=0-3", "bytes=2-5", "bytes=6-9"])
        self.assertEqual(result["retries"], 1)
        self.assertEqual(result["url_renewals"], 2)
        self.assertEqual((self.root / "download.zip").read_bytes(), b"0123456789")
        self.assertTrue(all(response.closed for response in responses))

    def test_expired_signed_url_is_refreshed_only_in_memory(self):
        calls, urls = [], []
        def resolve():
            calls.append(1)
            return StreamingResponse(200, {"Content-Length": "10"}, url=f"https://object.invalid/data?token={len(calls)}")
        def get(url, *, headers, **_kwargs):
            urls.append(url)
            if len(urls)==1:
                return StreamingResponse(403)
            start,end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
            return StreamingResponse(206, {"Content-Range": f"bytes {start}-{end}/10"}, [b"0123456789"[start:end+1]])
        result = self.ranged_download(request_get=get, resolve=resolve)
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(urls[0], urls[1])
        self.assertEqual(result["retries"], 1)
        self.assertNotIn("token=", json.dumps(result))

    def test_unhonored_ranges_invalid_bounds_and_encoded_bodies_never_append(self):
        variants = [(200, {}, "range-not-honored"),
            (206, {"Content-Range": "bytes 1-4/10"}, "content-range-invalid"),
            (206, {"Content-Range": "bytes 0-2/10"}, "content-range-invalid"),
            (206, {"Content-Range": "bytes 0-3/11"}, "content-range-invalid"),
            (206, {"Content-Range": "bytes 0-3/10", "Content-Length": "3"}, "range-length-invalid"),
            (206, {"Content-Range": "bytes 0-3/10", "Content-Encoding": "gzip"}, "content-encoding-invalid")]
        for index,(status,headers,code) in enumerate(variants):
            name=f"invalid{index}.zip"
            with self.subTest(code=code), self.assertRaisesRegex(ValueError,code):
                self.ranged_download(request_get=lambda *_a,**_k: StreamingResponse(status,headers,[b"0123"]), destination=name)
            self.assertFalse((self.root / name).exists())
            self.assertEqual((self.root / (name+".part")).stat().st_size,0)

    def test_changed_object_identity_cannot_extend_a_partial_prefix(self):
        renewals=[]
        def resolve():
            renewals.append(1)
            return StreamingResponse(200,{"Content-Length":"10","ETag":'"first"' if len(renewals)==1 else '"changed"'})
        def get(*_args,**_kwargs):
            return StreamingResponse(206,{"Content-Range":"bytes 0-3/10"},[b"01"],requests.exceptions.ChunkedEncodingError())
        with self.assertRaisesRegex(ValueError,"object-identity-changed"):
            self.ranged_download(request_get=get,resolve=resolve)
        self.assertEqual((self.root/"download.zip.part").read_bytes(),b"01")

    def test_retry_time_and_final_hash_bounds_preserve_failed_partial_only(self):
        def disconnect(*_args,**_kwargs):
            raise requests.exceptions.ConnectionError()
        with self.assertRaisesRegex(ValueError,"retry-budget-exceeded"):
            self.ranged_download(request_get=disconnect,maximum_retries=2,destination="retry.zip")
        clock=iter((0.,2.)).__next__
        with self.assertRaisesRegex(ValueError,"time-budget-exceeded"):
            self.ranged_download(monotonic=clock,maximum_seconds=1,destination="time.zip")
        with self.assertRaisesRegex(ValueError,"final-sha256-mismatch"):
            bootstrap.resumable_http_download(self.root/"bad_sha.zip","a"*64,4,
                lambda:StreamingResponse(200,{"Content-Length":"4"}),
                request_get=lambda *_a,**_k:StreamingResponse(206,{"Content-Range":"bytes 0-3/4"},[b"abcd"]),range_bytes=4)
        self.assertEqual((self.root/"bad_sha.zip.part").read_bytes(),b"abcd")
        self.assertFalse((self.root/"bad_sha.zip").exists())

    def test_live_progress_forwards_only_finite_epoch_fields(self):
        row = {"phase": "training", "arm": "central_recovery", "seed": 3, "epoch": 2,
               "train_loss": 0.04, "dev_loss": 0.08, "optimizer_updates": 158,
               "untrusted_extra": "SECRET-SENTINEL"}
        visible = bootstrap.visible_epoch_event(row)
        self.assertEqual(visible["epoch"], 2)
        self.assertNotIn("untrusted_extra", visible)
        self.assertNotIn("SECRET-SENTINEL", json.dumps(visible))
        for changes in ({"phase": "authentication"}, {"dev_loss": float("nan")},
                        {"arm": "SECRET-SENTINEL"}, {"epoch": True}):
            self.assertIsNone(bootstrap.visible_epoch_event(dict(row, **changes)))

    def test_incomplete_or_CPU_comparison_cannot_claim_remote_training_success(self):
        protocol = json.loads((packager.ROOT / "configs/recovery_training.json").read_text())
        frozen = {"manifest_sha256": "a" * 64, "transport_proof_sha256": "b" * 64,
                  "initial_checkpoint_sha256": "c" * 64, "training_protocol_sha256": canonical_sha256(protocol),
                  "split_counts": {"dev": 20}}
        with self.assertRaisesRegex(bootstrap.RecoveryBootstrapError, "source-or-budget-proof"):
            bootstrap.validate_recovery_report({"status": "PASS"}, frozen, protocol)
        comparison = {"status": "PASS", "fixture_injection": False, "final_town_used": False,
            "closed_loop_verified": False, "equal_budget_verified": True, "protocol": protocol,
            "protocol_sha256": frozen["training_protocol_sha256"], "manifest_sha256": frozen["manifest_sha256"],
            "validation_report_sha256": frozen["transport_proof_sha256"],
            "initial_checkpoint_sha256": frozen["initial_checkpoint_sha256"],
            "common_dev_sample_count": 20, "common_dev_selection_sha256": "d" * 64,
            "source_sha256": {"synthetic-source": "e" * 64}, "training": {}, "evaluations": {}}
        for seed in protocol["seeds"]:
            comparison["training"][str(seed)] = {}
            comparison["evaluations"][str(seed)] = {}
            for arm in ("central_only", "central_recovery"):
                common = {key: comparison[key] for key in ("status", "fixture_injection", "final_town_used",
                    "protocol_sha256", "manifest_sha256", "validation_report_sha256", "source_sha256")}
                common.update(arm=arm, seed=seed, runtime={"device": "cuda", "gpu": "Tesla T4"})
                checkpoint = hashlib.sha256((str(seed) + arm).encode()).hexdigest()
                comparison["training"][str(seed)][arm] = dict(common,
                    train_sample_count=protocol["train_samples_per_arm"], completed_epochs=protocol["epochs"],
                    optimizer_updates=((protocol["train_samples_per_arm"] + protocol["batch_size"] - 1)
                                       // protocol["batch_size"]) * protocol["epochs"],
                    initial_checkpoint_sha256=frozen["initial_checkpoint_sha256"], checkpoint_sha256=checkpoint)
                comparison["evaluations"][str(seed)][arm] = dict(common, split="dev", sample_count=20,
                    checkpoint_sha256=checkpoint, common_dev_selection_sha256="d" * 64)
        comparison["initial_evaluation"] = {"status": "PASS", "arm": "initial", "sample_count": 20,
            "checkpoint_sha256": frozen["initial_checkpoint_sha256"], "common_dev_selection_sha256": "d" * 64}
        self.assertTrue(bootstrap.validate_recovery_report(comparison, frozen, protocol)["actual_T4_training"])
        first = str(protocol["seeds"][0])
        comparison["training"][first]["central_only"]["runtime"] = {"device": "cpu", "gpu": None}
        with self.assertRaisesRegex(bootstrap.RecoveryBootstrapError, "actual-T4-arm"):
            bootstrap.validate_recovery_report(comparison, frozen, protocol)


if __name__ == "__main__":
    unittest.main()
