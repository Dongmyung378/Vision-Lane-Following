"""Run a source-bound paired experiment in an existing private T4 notebook.

The generated notebook uses platform authentication to retrieve an immutable
output from an existing notebook. A compact transport summary reuses a prior
CPU source replay; it never claims a new native replay on the training GPU.
Importing this module starts no process, network request, or GPU workload.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

PROJECT_SOURCES = {}
FROZEN_INPUT = {}
TRUSTED_CPU_PROOF = ""
MAX_SECONDS = 21600
TRAINING_SECONDS = 18000
MAX_EXPANDED_BYTES = 24 * 1024**3
DATA_NOTEBOOK_ID = "dongmyungpark/vlf-carla-rgb-smoke-20261005"
TRAINING_NOTEBOOK_ID = "dongmyungpark/vlf-carla-vulkan-preflight-20261005"
DOWNLOAD_HELPER_SOURCE = None


class RecoveryBootstrapError(RuntimeError):
    pass


def resumable_http_download(destination, expected_sha256, expected_bytes, resolve_response, *,
                            request_get=None, maximum_seconds=1800, maximum_retries=16,
                            range_bytes=32 * 1024**2, monotonic=None, sleep=None):
    """Download pinned bytes with bounded range requests and verified resumes.

    The resolver uses the official authenticated, versioned API. Signed URLs
    remain in memory. Failed partial bytes are retained, never accepted as a
    completed artifact. Existing partial files from another call are refused.
    """
    import hashlib
    import math
    from pathlib import Path
    import re
    import shutil
    import time
    from urllib.parse import urlsplit
    import requests

    clock, pause = monotonic or time.monotonic, sleep or time.sleep
    get = request_get or requests.get
    if (type(expected_bytes) is not int or not 0 < expected_bytes <= 32 * 1024**3
            or type(expected_sha256) is not str or not re.fullmatch("[0-9a-f]{64}", expected_sha256)
            or type(maximum_seconds) not in (int, float) or not 1 <= maximum_seconds <= 3600
            or type(maximum_retries) is not int or not 0 <= maximum_retries <= 32
            or type(range_bytes) is not int or not 1 <= range_bytes <= 64 * 1024**2):
        raise ValueError("recovery-download-budget-or-pin-invalid")
    destination = Path(destination)
    partial = destination.with_name(destination.name + ".part")
    if (destination.exists() or destination.is_symlink() or partial.exists() or partial.is_symlink()
            or any(path.is_symlink() for path in destination.parents)):
        raise ValueError("recovery-download-new-owned-path-required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < expected_bytes + 512 * 1024**2:
        raise ValueError("recovery-download-space-insufficient")
    began, written, failures, requests_made, renewals, resumed = clock(), 0, 0, 0, 0, 0
    digest, url, identity = hashlib.sha256(), None, {}
    request_budget = math.ceil(expected_bytes / range_bytes) + maximum_retries + 1

    def remaining():
        value = maximum_seconds - (clock() - began)
        if value <= 0:
            raise ValueError("recovery-download-time-budget-exceeded")
        return value

    def bind_headers(headers):
        if headers.get("Content-Encoding", "identity").strip().lower() not in ("", "identity"):
            raise ValueError("recovery-download-content-encoding-invalid")
        for name in ("ETag", "Last-Modified", "x-goog-generation"):
            value = headers.get(name)
            if value:
                if name in identity and identity[name] != value:
                    raise ValueError("recovery-download-object-identity-changed")
                identity[name] = value

    retryable = (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                 requests.exceptions.ChunkedEncodingError)
    with partial.open("xb") as output:
        while written < expected_bytes:
            remaining()
            response = None
            try:
                if url is None:
                    initial = None
                    try:
                        initial = resolve_response()
                        renewals += 1
                        if initial.status_code != 200:
                            if initial.status_code in (401, 403, 408, 429) or initial.status_code >= 500:
                                raise requests.exceptions.ConnectionError("renewable-api-response")
                            raise ValueError("recovery-download-resolver-status-invalid")
                        bind_headers(initial.headers)
                        length = initial.headers.get("Content-Length")
                        if length is not None and (not str(length).isdigit() or int(length) != expected_bytes):
                            raise ValueError("recovery-download-resolver-size-mismatch")
                        url = initial.url
                        if urlsplit(url).scheme != "https" or not urlsplit(url).netloc:
                            raise ValueError("recovery-download-resolver-url-invalid")
                    finally:
                        if initial is not None:
                            initial.close()
                if requests_made >= request_budget:
                    raise ValueError("recovery-download-request-budget-exceeded")
                end = min(written + range_bytes, expected_bytes) - 1
                headers = {"Accept-Encoding": "identity", "Range": f"bytes={written}-{end}"}
                etag = identity.get("ETag")
                if etag and not etag.startswith("W/"):
                    headers["If-Range"] = etag
                elif identity.get("Last-Modified"):
                    headers["If-Unmodified-Since"] = identity["Last-Modified"]
                timeout = remaining()
                response = get(url, stream=True, headers=headers,
                               timeout=(min(15, timeout), min(45, timeout)))
                requests_made += 1
                if response.status_code in (401, 403, 408, 429) or response.status_code >= 500:
                    raise requests.exceptions.ConnectionError("renewable-object-response")
                if response.status_code != 206:
                    raise ValueError("recovery-download-range-not-honored")
                bind_headers(response.headers)
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if (not match or tuple(map(int, match.groups())) != (written, end, expected_bytes)):
                    raise ValueError("recovery-download-content-range-invalid")
                length = response.headers.get("Content-Length")
                if length is not None and (not str(length).isdigit() or int(length) != end - written + 1):
                    raise ValueError("recovery-download-range-length-invalid")
                if written:
                    resumed += 1
                expected_end = end + 1
                for block in response.iter_content(chunk_size=1024 * 1024):
                    remaining()
                    if not block:
                        continue
                    if len(block) > 1024 * 1024 or written + len(block) > expected_end:
                        raise ValueError("recovery-download-range-body-overflow")
                    output.write(block)
                    digest.update(block)
                    written += len(block)
                output.flush()
                if written != expected_end:
                    raise requests.exceptions.ConnectionError("truncated-range-body")
                if partial.stat().st_size != written:
                    raise ValueError("recovery-download-written-size-mismatch")
            except retryable:
                output.flush()
                if partial.stat().st_size != written:
                    raise ValueError("recovery-download-partial-size-mismatch")
                failures += 1
                if failures > maximum_retries:
                    raise ValueError("recovery-download-retry-budget-exceeded") from None
                url = None
                pause(min(.25 * 2 ** min(failures - 1, 4), remaining()))
            finally:
                if response is not None:
                    response.close()
    remaining()
    if written != expected_bytes or digest.hexdigest() != expected_sha256:
        raise ValueError("recovery-download-final-sha256-mismatch")
    final_digest = hashlib.sha256()
    with partial.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            remaining()
            final_digest.update(block)
    if final_digest.hexdigest() != expected_sha256 or partial.stat().st_size != expected_bytes:
        raise ValueError("recovery-download-stored-bytes-sha256-mismatch")
    partial.rename(destination)
    return {"path": str(destination), "sha256": final_digest.hexdigest(), "bytes": expected_bytes,
            "range_requests": requests_made, "retries": failures, "url_renewals": renewals,
            "requests_after_initial_prefix": resumed, "chunk_bytes": 1024 * 1024,
            "range_bytes": range_bytes, "elapsed_seconds": clock() - began}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def visible_epoch_event(value):
    """Return a small progress record; never forward arbitrary child text."""
    names = ("phase", "arm", "seed", "epoch", "train_loss", "dev_loss", "optimizer_updates")
    if (type(value) is not dict or value.get("phase") != "training"
            or value.get("arm") not in ("central_only", "central_recovery")
            or any(type(value.get(name)) is not int or value[name] < 0 for name in ("seed", "epoch", "optimizer_updates"))
            or any(type(value.get(name)) not in (int, float) or not math.isfinite(value[name])
                   for name in ("train_loss", "dev_loss"))):
        return None
    return {name: value[name] for name in names}


def safe_extract_dataset(archive_path, target, *, maximum_bytes=MAX_EXPANDED_BYTES):
    """Extract a new, bounded tree; retain the original archive unchanged."""
    archive_path, target = Path(archive_path), Path(target)
    if (archive_path.is_symlink() or not archive_path.is_file() or target.exists()
            or target.is_symlink() or any(parent.is_symlink() for parent in target.parents)):
        raise RecoveryBootstrapError("recovery-extraction-path-invalid")
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        if (not entries or len(entries) > 150000 or sum(item.file_size for item in entries) > maximum_bytes
                or any(item.file_size > 512 * 1024**2 for item in entries)):
            raise RecoveryBootstrapError("recovery-archive-budget-exceeded")
        seen = set()
        for item in entries:
            # ZipInfo normalizes backslashes on Windows. Inspect the original
            # member name so unsafe bytes are rejected identically on Linux.
            name = item.orig_filename
            relative = PurePosixPath(name)
            if (relative.is_absolute() or "\\" in name or ":" in name
                    or any(part in ("", ".", "..") for part in name.split("/"))
                    or relative.parts[0] != "dataset" or len(relative.parts) < 2
                    or item.is_dir() or item.flag_bits & 1
                    or (item.external_attr >> 16) & 0o170000 == 0o120000
                    or name != item.filename or name in seen):
                raise RecoveryBootstrapError("recovery-archive-member-invalid")
            seen.add(name)
        expanded = sum(item.file_size for item in entries)
        if shutil.disk_usage(target.parent).free < expanded + 1024**3:
            raise RecoveryBootstrapError("recovery-extraction-space-insufficient")
        target.mkdir(parents=True, exist_ok=False)
        for item in entries:
            destination = target.joinpath(*PurePosixPath(item.filename).parts)
            if not destination.resolve().is_relative_to(target.resolve()):
                raise RecoveryBootstrapError("recovery-archive-path-escape")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    return target / "dataset"


def resolve_frozen_archive(input_root, expected_sha256, *, download=None):
    """Bind mounted or platform-downloaded bytes to the frozen capture hash."""
    candidates = sorted(Path(input_root).rglob("dataset_artifacts.zip"))
    if len(candidates) > 16:
        raise RecoveryBootstrapError("recovery-mounted-archive-count-exceeded")
    matches = [path for path in candidates if path.is_file() and not path.is_symlink()
               and sha256_file(path) == expected_sha256]
    if len(matches) > 1:
        raise RecoveryBootstrapError("recovery-frozen-archive-not-unique")
    if matches:
        return matches[0], "mounted-existing-notebook-output"
    if download is None:
        raise RecoveryBootstrapError("recovery-frozen-archive-unavailable")
    path = Path(download())
    if (path.name != "dataset_artifacts.zip" or not path.is_file() or path.is_symlink()
            or sha256_file(path) != expected_sha256):
        raise RecoveryBootstrapError("recovery-cloud-archive-hash-mismatch")
    return path, "platform-authenticated-existing-notebook-version"


def cloud_download_helper_source(version, dependencies, destination, *, notebook_id=DATA_NOTEBOOK_ID,
                                 filename="dataset_artifacts.zip", expected_sha256,
                                 expected_bytes, maximum_seconds=1800):
    """Use only platform authentication; suppress URLs and library chatter."""
    if type(version) is not int or version <= 0:
        raise RecoveryBootstrapError("recovery-source-version-invalid")
    if (notebook_id not in (DATA_NOTEBOOK_ID, TRAINING_NOTEBOOK_ID)
            or filename not in ("dataset_artifacts.zip", "training_artifacts.zip")):
        raise RecoveryBootstrapError("recovery-existing-source-identity-invalid")
    import inspect
    downloader = DOWNLOAD_HELPER_SOURCE or inspect.getsource(resumable_http_download)
    return ("import contextlib,hashlib,io,json,logging,sys\nfrom pathlib import Path\n"
            "sys.path.insert(0," + repr(str(dependencies)) + ")\nlogging.disable(logging.CRITICAL)\n"
            + downloader + "\n"
            "try:\n with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):\n"
            "  import kagglehub,requests\n  from unittest.mock import patch\n"
            "  from kagglehub.clients import build_kaggle_client\n"
            "  from kagglehub.http_resolver import _build_notebook_download_request\n"
            "  from kagglehub.handle import parse_notebook_handle\n"
            "  handle=parse_notebook_handle("
            + repr(notebook_id + "/versions/" + str(version))
            + ")\n  request=_build_notebook_download_request(handle," + repr(filename) + ")\n"
            + "  assert request.version_number==" + repr(version) + "\n"
            "  original_send=requests.sessions.Session.send\n"
            "  def bounded_send(session,request,**kwargs):\n"
            "   request.headers['Accept-Encoding']='identity'\n   kwargs['timeout']=(15,45)\n   kwargs['stream']=True\n"
            "   return original_send(session,request,**kwargs)\n"
            "  def resolve_response():\n"
            "   with patch.object(requests.sessions.Session,'send',bounded_send):\n"
            "    with build_kaggle_client() as client:\n"
            "     return client.kernels.kernels_api_client.download_kernel_output(request)\n"
            "  result=resumable_http_download(Path(" + repr(str(destination)) + ")/" + repr(filename)
            + "," + repr(expected_sha256) + "," + repr(expected_bytes) + ",resolve_response,maximum_seconds="
            + repr(maximum_seconds) + ")\n"
            " result.update(status='PASS',platform_authentication=True,kagglehub=kagglehub.__version__)\n"
            " print(json.dumps(result))\n"
            "except BaseException as error:\n"
            " value={'status':'FAIL','exception_type':type(error).__name__}\n"
            " if isinstance(error,ValueError) and str(error).startswith('recovery-download-'):value['code']=str(error)\n"
            " print(json.dumps(value))\n raise SystemExit(1)\n")


def extract_frozen_initial_checkpoint(archive, destination, frozen):
    """Extract only the reviewed checkpoint from a prior training version."""
    archive, destination = Path(archive), Path(destination)
    if archive.is_symlink() or sha256_file(archive) != frozen["initial_archive_sha256"]:
        raise RecoveryBootstrapError("recovery-initial-archive-sha256-mismatch")
    if destination.exists() or destination.is_symlink() or any(p.is_symlink() for p in destination.parents):
        raise RecoveryBootstrapError("recovery-initial-checkpoint-destination-invalid")
    with zipfile.ZipFile(archive) as packed:
        names = packed.namelist()
        selected = frozen["initial_checkpoint_member"]
        if names.count(selected) != 1:
            raise RecoveryBootstrapError("recovery-initial-checkpoint-member-not-unique")
        member = packed.getinfo(selected)
        if (member.is_dir() or member.flag_bits & 1 or member.file_size > 512 * 1024**2
                or (member.external_attr >> 16) & 0o170000 == 0o120000):
            raise RecoveryBootstrapError("recovery-initial-checkpoint-member-invalid")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with packed.open(member) as source, destination.open("xb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
    if sha256_file(destination) != frozen["initial_checkpoint_sha256"]:
        raise RecoveryBootstrapError("recovery-initial-checkpoint-sha256-mismatch")
    return destination


def rebuild_frozen_dataset(archive, extraction, frozen, proof_text):
    """Rebuild exact derived bytes from unchanged raw sources and reviewed code."""
    from carla_tools.contracts import canonical_sha256
    from carla_tools.dataset_v1 import build_dataset_v1_manifest, dataset_v1_config_from_dict
    from scripts.run_dataset_pipeline import episode_descriptor

    if sha256_file(archive) != frozen["archive_sha256"]:
        raise RecoveryBootstrapError("recovery-original-archive-sha256-mismatch")
    if hashlib.sha256(proof_text.encode()).hexdigest() != frozen["transport_proof_sha256"]:
        raise RecoveryBootstrapError("recovery-transport-proof-sha256-mismatch")
    proof = json.loads(proof_text)
    dataset = safe_extract_dataset(archive, extraction)
    original = dataset / "dataset_v1_manifest.json"
    original.rename(dataset / "dataset_v1_manifest.original-capture.json")
    protocol = json.loads((dataset / "dataset_protocol.json").read_text(encoding="utf-8"))
    config = dataset_v1_config_from_dict(protocol["config"])
    if config.sha256 != frozen["config_sha256"]:
        raise RecoveryBootstrapError("recovery-config-mismatch")
    manifest = build_dataset_v1_manifest(dataset, protocol["routes"],
        [episode_descriptor(trial) for trial in protocol["trials"]], config)
    if sha256_file(manifest) != frozen["manifest_sha256"]:
        raise RecoveryBootstrapError("recovery-rebuilt-manifest-sha256-mismatch")
    if (proof.get("transport_kind") != "validated-report-transport-summary"
            or proof.get("new_native_replay_performed") is not False
            or proof.get("manifest_sha256") != frozen["manifest_sha256"]
            or proof.get("snapshot_sha256") != frozen["snapshot_sha256"]
            or proof.get("original_full_report_sha256") != frozen["original_full_report_sha256"]
            or proof.get("config_sha256") != config.sha256
            or proof.get("original_episode_metrics_sha256") != frozen["episode_metrics_sha256"]
            or proof.get("original_episode_metrics_count") != frozen["episode_count"]):
        raise RecoveryBootstrapError("recovery-transport-proof-source-binding-invalid")
    proof_path = dataset / "dataset_validation_transport_summary.json"
    with proof_path.open("xb") as stream:
        stream.write(proof_text.encode())
    return manifest, proof_path


def export_runtime_sources(project, output, sources):
    hashes = {}
    with zipfile.ZipFile(Path(output) / "training_runtime_sources.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name, expected in sorted(sources.items()):
            path = Path(project) / name
            if (path.is_symlink() or not path.resolve().is_relative_to(Path(project).resolve())
                    or sha256_file(path) != hashlib.sha256(expected.encode()).hexdigest()):
                raise RecoveryBootstrapError("recovery-runtime-source-changed")
            hashes[name] = sha256_file(path)
            archive.write(path, name)
    report = {"schema_version": 1, "source_sha256": hashes}
    (Path(output) / "training_runtime_source_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def preserve_training_artifacts(output):
    output = Path(output)
    inventory = {}
    with zipfile.ZipFile(output / "training_artifacts.zip", "x", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        experiment = output / "experiment"
        if experiment.exists():
            for path in sorted(experiment.rglob("*")):
                if path.is_symlink() or not path.resolve().is_relative_to(experiment.resolve()):
                    raise RecoveryBootstrapError("recovery-artifact-path-escaped")
                if path.is_file() and path.suffix in (".json", ".jsonl", ".pt", ".csv", ".png"):
                    name = path.relative_to(output).as_posix()
                    inventory[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                    archive.write(path, name)
    report = {"sha256": sha256_file(output / "training_artifacts.zip"), "files": inventory}
    (output / "training_artifact_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def validate_recovery_report(report, frozen, protocol):
    """Require actual T4 execution for every predeclared arm and seed."""
    seeds = {str(seed) for seed in protocol["seeds"]}
    arms = {"central_only", "central_recovery"}
    if (report.get("status") != "PASS" or report.get("fixture_injection") is not False
            or report.get("final_town_used") is not False or report.get("closed_loop_verified") is not False
            or report.get("equal_budget_verified") is not True or report.get("protocol") != protocol
            or report.get("protocol_sha256") != frozen["training_protocol_sha256"]
            or report.get("manifest_sha256") != frozen["manifest_sha256"]
            or report.get("validation_report_sha256") != frozen["transport_proof_sha256"]
            or report.get("initial_checkpoint_sha256") != frozen["initial_checkpoint_sha256"]
            or report.get("common_dev_sample_count") != frozen["split_counts"]["dev"]
            or set(report.get("training", {})) != seeds or set(report.get("evaluations", {})) != seeds):
        raise RecoveryBootstrapError("recovery-comparison-source-or-budget-proof-invalid")
    source = report.get("source_sha256")
    expected_updates = math.ceil(protocol["train_samples_per_arm"] / protocol["batch_size"]) * protocol["epochs"]
    for seed in seeds:
        if set(report["training"][seed]) != arms or set(report["evaluations"][seed]) != arms:
            raise RecoveryBootstrapError("recovery-comparison-missing-arm")
        for arm in arms:
            train = report["training"][seed][arm]
            evaluation = report["evaluations"][seed][arm]
            for row in (train, evaluation):
                runtime = row.get("runtime", {})
                if (row.get("status") != "PASS" or row.get("arm") != arm or row.get("seed") != int(seed)
                        or row.get("fixture_injection") is not False or row.get("final_town_used") is not False
                        or row.get("manifest_sha256") != frozen["manifest_sha256"]
                        or row.get("validation_report_sha256") != frozen["transport_proof_sha256"]
                        or row.get("protocol_sha256") != frozen["training_protocol_sha256"]
                        or not source or row.get("source_sha256") != source
                        or runtime.get("device") != "cuda" or "T4" not in str(runtime.get("gpu", ""))):
                    raise RecoveryBootstrapError("recovery-actual-T4-arm-source-proof-invalid")
            if (train.get("train_sample_count") != protocol["train_samples_per_arm"]
                    or train.get("completed_epochs") != protocol["epochs"]
                    or train.get("optimizer_updates") != expected_updates
                    or train.get("initial_checkpoint_sha256") != frozen["initial_checkpoint_sha256"]
                    or evaluation.get("split") != "dev" or evaluation.get("sample_count") != frozen["split_counts"]["dev"]
                    or evaluation.get("checkpoint_sha256") != train.get("checkpoint_sha256")
                    or not train.get("checkpoint_sha256")
                    or evaluation.get("common_dev_selection_sha256") != report.get("common_dev_selection_sha256")):
                raise RecoveryBootstrapError("recovery-full-development-or-update-proof-invalid")
    initial = report.get("initial_evaluation", {})
    if (initial.get("status") != "PASS" or initial.get("arm") != "initial"
            or initial.get("checkpoint_sha256") != frozen["initial_checkpoint_sha256"]
            or initial.get("sample_count") != frozen["split_counts"]["dev"]
            or initial.get("common_dev_selection_sha256") != report.get("common_dev_selection_sha256")):
        raise RecoveryBootstrapError("recovery-unchanged-incumbent-evaluation-proof-invalid")
    return {"actual_T4_training": True, "seed_count": len(seeds), "arm_count_per_seed": len(arms),
            "common_dev_samples": frozen["split_counts"]["dev"], "new_native_replay_performed": False}


def main_recovery():
    """Install pinned CUDA dependencies, reconstruct inputs, then train once."""
    output = Path("/kaggle/working")
    scratch = Path(tempfile.mkdtemp(prefix="vlf_recovery_", dir="/tmp"))
    started, current = time.monotonic(), None
    report = {"schema_version": 1, "status": "FAIL", "workload": "matched-budget-central-recovery-training",
              "frozen_input": FROZEN_INPUT, "local_gpu_used": False, "carla_server_started": False,
              "new_native_replay_performed": False, "training_process_stopped": False, "stages": []}

    def expire(_signum, _frame):
        raise RecoveryBootstrapError("recovery-total-time-limit")

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, expire)
    signal.alarm(MAX_SECONDS)

    def stage(name):
        report["stage"] = name
        report["stages"].append({"name": name, "elapsed_seconds": time.monotonic() - started})
        print("Paired recovery training:", name, flush=True)

    def command(args, label, timeout=600, env=None):
        nonlocal current
        remaining = min(timeout, MAX_SECONDS - (time.monotonic() - started))
        if remaining <= 0:
            raise RecoveryBootstrapError("recovery-total-time-limit")
        with (output / (label + ".log")).open("xb") as log:
            if label == "recovery_training":
                current = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           env=env, start_new_session=True)
                child, reader_errors = current, []

                def copy_training_progress():
                    discarding_long_line = False
                    try:
                        with child.stdout:
                            while True:
                                piece = child.stdout.readline(16385)
                                if not piece:
                                    break
                                log.write(piece)
                                log.flush()
                                complete = piece.endswith(b"\n")
                                if discarding_long_line or len(piece) > 16384:
                                    discarding_long_line = not complete
                                    continue
                                if not complete:
                                    discarding_long_line = True
                                    continue
                                try:
                                    event = visible_epoch_event(json.loads(piece.decode("utf-8")))
                                except (UnicodeError, ValueError):
                                    continue
                                if event is not None:
                                    print(json.dumps(event, allow_nan=False), flush=True)
                    except Exception as error:
                        reader_errors.append(type(error).__name__)

                reader = threading.Thread(target=copy_training_progress, daemon=True)
                reader.start()
                code = current.wait(timeout=remaining)
                reader.join(timeout=10)
                if reader.is_alive() or reader_errors:
                    raise RecoveryBootstrapError("recovery-training-log-reader-failed")
            else:
                current = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                code = current.wait(timeout=remaining)
            current = None
        if code:
            raise RecoveryBootstrapError(label + "-failed")

    try:
        stage("materialize-reviewed-sources")
        project = scratch / "project"
        project.mkdir()
        for relative, content in PROJECT_SOURCES.items():
            name = PurePosixPath(relative)
            if (name.is_absolute() or ".." in name.parts or "\\" in relative or ":" in relative
                    or name.parts[0] not in ("scripts", "carla_tools", "configs")):
                raise RecoveryBootstrapError("recovery-project-source-path-invalid")
            target = project / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.encode())
        report["runtime_sources"] = export_runtime_sources(project, output, PROJECT_SOURCES)
        env = os.environ.copy()
        env.update(UV_CACHE_DIR=str(scratch / "uv_cache"), UV_PYTHON_INSTALL_DIR=str(scratch / "uv_python"),
                   UV_NO_PROGRESS="1", UV_LINK_MODE="copy", TORCH_HOME=str(scratch / "torch_cache"),
                   CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4",
                   PYTHONPATH=str(project))
        stage("pinned-cuda-runtime")
        uv = shutil.which("uv")
        if not uv:
            tools = scratch / "uv_tools"
            command([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--target", str(tools),
                     "uv==0.8.22"], "recovery_uv", 180, env)
            uv = str(tools / "bin" / "uv")
        command([uv, "python", "install", "3.12"], "recovery_python", 180, env)
        venv = scratch / "venv"
        command([uv, "venv", "--python", "3.12", "--managed-python", str(venv)], "recovery_venv", 60, env)
        python = str(venv / "bin" / "python")
        command([uv, "pip", "install", "--python", python, "--only-binary", ":all:", "--index-url",
                 "https://download.pytorch.org/whl/cu126", "torch==2.7.1", "torchvision==0.22.1"], "recovery_torch", 900, env)
        command([uv, "pip", "install", "--python", python, "--only-binary", ":all:", "numpy==2.2.6",
                 "opencv-python-headless==4.12.0.88", "Pillow==11.3.0", "carla==0.9.16"], "recovery_dependencies", 180, env)
        stage("frozen-existing-capture")

        dependencies = scratch / "kagglehub_tools"

        def download_existing():
            command([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--only-binary", ":all:",
                     "--target", str(dependencies), "kagglehub==1.0.2"], "recovery_transport_dependencies", 180)
            helper = cloud_download_helper_source(FROZEN_INPUT["source_notebook_version"], dependencies, scratch / "cloud",
                expected_sha256=FROZEN_INPUT["archive_sha256"], expected_bytes=FROZEN_INPUT["archive_bytes"])
            command([sys.executable, "-c", helper], "recovery_capture_transport", 2400)
            value = json.loads((output / "recovery_capture_transport.log").read_text().strip())
            if value.get("status") != "PASS" or value.get("sha256") != FROZEN_INPUT["archive_sha256"]:
                raise RecoveryBootstrapError("recovery-existing-capture-transport-invalid")
            return value["path"]

        archive, report["capture_transport"] = resolve_frozen_archive(Path("/kaggle/input"),
            FROZEN_INPUT["archive_sha256"], download=download_existing)
        stage("reconstruct-exact-dataset-and-trusted-CPU-proof")
        # Run in the pinned interpreter so OpenCV/NumPy reconstruction matches
        # the reviewed dataset implementation, regardless of host packages.
        reconstruction = ("import json,sys;from pathlib import Path;sys.path.insert(0," + repr(str(project)) + ");"
            "from scripts.kaggle_recovery_training_bootstrap import rebuild_frozen_dataset;"
            "frozen=json.loads(Path(" + repr(str(project / "frozen_input.json")) + ").read_text());"
            "proof=Path(" + repr(str(project / "trusted_cpu_proof.json")) + ").read_text();"
            "m,p=rebuild_frozen_dataset(Path(" + repr(str(archive)) + "),Path(" + repr(str(scratch / "capture"))
            + "),frozen,proof);print(json.dumps({'manifest':str(m),'proof':str(p)}))")
        (project / "frozen_input.json").write_text(json.dumps(FROZEN_INPUT))
        (project / "trusted_cpu_proof.json").write_bytes(TRUSTED_CPU_PROOF.encode())
        command([python, "-c", reconstruction], "recovery_dataset_reconstruction", 1800, env)
        reconstructed = json.loads((output / "recovery_dataset_reconstruction.log").read_text().strip().splitlines()[-1])
        stage("exact-previous-training-checkpoint")
        if not dependencies.exists():
            command([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--only-binary", ":all:",
                     "--target", str(dependencies), "kagglehub==1.0.2"], "recovery_initial_transport_dependencies", 180)
        helper = cloud_download_helper_source(FROZEN_INPUT["initial_notebook_version"], dependencies,
            scratch / "initial_cloud", notebook_id=TRAINING_NOTEBOOK_ID, filename="training_artifacts.zip",
            expected_sha256=FROZEN_INPUT["initial_archive_sha256"], expected_bytes=FROZEN_INPUT["initial_archive_bytes"],
            maximum_seconds=600)
        command([sys.executable, "-c", helper], "recovery_initial_transport", 600)
        initial_transport = json.loads((output / "recovery_initial_transport.log").read_text().strip())
        if initial_transport.get("status") != "PASS" or initial_transport.get("sha256") != FROZEN_INPUT["initial_archive_sha256"]:
            raise RecoveryBootstrapError("recovery-initial-transport-invalid")
        initial = extract_frozen_initial_checkpoint(initial_transport["path"], scratch / "initial/checkpoint.pt", FROZEN_INPUT)
        report["initial_checkpoint_sha256"] = sha256_file(initial)
        smoke = ("import json,torch,torchvision;assert torch.__version__.split('+')[0]=='2.7.1';"
                 "assert torchvision.__version__.split('+')[0]=='0.22.1';assert torch.cuda.is_available();"
                 "assert 'T4' in torch.cuda.get_device_name(0);"
                 "print(json.dumps({'device':'cuda','gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__)}))")
        command([python, "-c", smoke], "recovery_cuda_smoke", 60, env)
        report["cuda_smoke"] = json.loads((output / "recovery_cuda_smoke.log").read_text().strip())
        stage("actual-T4-matched-budget-training-and-common-development")
        command([python, str(project / "scripts/train_recovery_waypoints.py"), "--manifest", reconstructed["manifest"],
                 "--validation-report", reconstructed["proof"], "--output-dir", str(output / "experiment"),
                 "--device", "cuda", "--initial-checkpoint", str(initial),
                 "--protocol", str(project / "configs/recovery_training.json")], "recovery_training", TRAINING_SECONDS, env)
        comparison = json.loads((output / "experiment/comparison.json").read_text())
        protocol = json.loads((project / "configs/recovery_training.json").read_text())
        report["experiment"] = validate_recovery_report(comparison, FROZEN_INPUT, protocol)
        report["comparison_sha256"] = sha256_file(output / "experiment/comparison.json")
        report["status"] = "PASS"
        stage("complete")
    except Exception as error:
        report["error"] = {"exception_type": type(error).__name__,
                           "code": str(error) if isinstance(error, RecoveryBootstrapError) else "recovery-bootstrap-operation-failed"}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        if current is not None and current.poll() is None:
            os.killpg(current.pid, signal.SIGTERM)
            try:
                current.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(current.pid, signal.SIGKILL)
                current.wait(timeout=10)
        report["training_process_stopped"] = current is None or current.poll() is not None
        report["elapsed_seconds"] = time.monotonic() - started
        try:
            report["artifacts"] = preserve_training_artifacts(output)
        except Exception as error:
            report.update(status="FAIL", archive_error=type(error).__name__)
        (output / "kaggle_recovery_bootstrap.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 1
