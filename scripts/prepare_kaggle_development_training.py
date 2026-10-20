"""Prepare source-bound appearance learning for an existing Kaggle T4 notebook.

This command creates only a local bundle. Fresh read-only identity, access and
version checks of both existing notebooks are required immediately before any
remote submission. Generated bundles and input evidence belong in local storage.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.contracts import canonical_sha256, read_strict_json
from carla_tools.development_training import load_development_protocol
from scripts.kaggle_development_training_bootstrap import MAX_SECONDS, TRAINING_SECONDS
from scripts.kaggle_recovery_training_bootstrap import sha256_file
from scripts.prepare_kaggle_recovery_training import (NOTEBOOK_ID, NOTEBOOK_TITLE,
    DATA_NOTEBOOK_ID, _digest, _require, _reviewed_sources, trusted_cpu_summary)

MOUNT_NOTEBOOK_ID = "dongmyungpark/vlf-carla-download-probe-20261005"


def prepare(existing_metadata, output_dir, *, manifest_path, validation_report_path, source_audit_path,
            validation_amendment_path, archive_sha256, source_notebook_version, mount_source_version,
            mount_source_code_sha256, initial_archive_path, initial_checkpoint_path, initial_notebook_version,
            initial_checkpoint_member, protocol_path=None, maximum_seconds=MAX_SECONDS, training_seconds=TRAINING_SECONDS):
    metadata = read_strict_json(Path(existing_metadata))
    _require(type(metadata) is dict and metadata.get("id") == NOTEBOOK_ID
        and metadata.get("title") == NOTEBOOK_TITLE and metadata.get("is_private") is True,
        "existing-private-notebook-required")
    for value in (archive_sha256, mount_source_code_sha256):
        _digest(value)
    for version in (source_notebook_version, mount_source_version, initial_notebook_version):
        _require(type(version) is int and version > 0, "development-positive-source-version-required")
    _require(type(maximum_seconds) is int and 1800 <= maximum_seconds <= 43200
        and type(training_seconds) is int and 300 <= training_seconds <= maximum_seconds - 900,
        "development-time-budget-invalid")
    output = Path(output_dir)
    _require(not output.exists() and not output.is_symlink() and not any(path.is_symlink() for path in output.parents),
             "new-local-output-directory-required")
    member_path = PurePosixPath(initial_checkpoint_member)
    _require(type(initial_checkpoint_member) is str and not member_path.is_absolute()
        and not any(part in ("", ".", "..") for part in initial_checkpoint_member.split("/"))
        and "\\" not in initial_checkpoint_member and ":" not in initial_checkpoint_member
        and member_path.parts[0] == "experiment" and member_path.name == "refined_checkpoint.pt",
        "development-reviewed-initial-member-required")
    summary, manifest = trusted_cpu_summary(manifest_path, validation_report_path, source_audit_path,
        validation_amendment_path, archive_sha256=archive_sha256)
    initial_archive, initial_checkpoint = Path(initial_archive_path), Path(initial_checkpoint_path)
    _require(all(path.is_file() and not path.is_symlink() for path in (initial_archive, initial_checkpoint)),
             "development-frozen-initial-artifacts-required")
    initial_sha = sha256_file(initial_checkpoint)
    with zipfile.ZipFile(initial_archive) as archive:
        _require(archive.namelist().count(initial_checkpoint_member) == 1, "development-initial-member-not-unique")
        member = archive.getinfo(initial_checkpoint_member)
        _require(member.file_size == initial_checkpoint.stat().st_size and member.file_size <= 512 * 1024**2
            and not member.is_dir() and not member.flag_bits & 1 and (member.external_attr >> 16) & 0o170000 != 0o120000,
            "development-initial-member-invalid")
        digest = hashlib.sha256()
        with archive.open(member) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        _require(digest.hexdigest() == initial_sha, "development-initial-archive-checkpoint-binding-invalid")
    sources = _reviewed_sources()
    _require(all(name in sources for name in ("carla_tools/development_training.py", "scripts/train_development_waypoints.py",
        "configs/development_training.json", "scripts/kaggle_development_training_bootstrap.py")),
        "development-training-runtime-source-required")
    protocol = load_development_protocol(protocol_path) if protocol_path is not None else load_development_protocol()
    proof = json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    mount_reference = MOUNT_NOTEBOOK_ID + "/" + str(mount_source_version)
    frozen = {"archive_sha256": archive_sha256, "archive_bytes": read_strict_json(source_audit_path)["archive_bytes"],
        "source_notebook_id": DATA_NOTEBOOK_ID, "source_notebook_version": source_notebook_version,
        "mount_source_notebook_id": MOUNT_NOTEBOOK_ID, "mount_source_version": mount_source_version,
        "mount_source_code_sha256": mount_source_code_sha256, "mount_source_reference": mount_reference,
        "manifest_sha256": sha256_file(manifest_path), "snapshot_sha256": manifest["snapshot_sha256"],
        "config_sha256": manifest["config_sha256"], "episode_count": len(manifest["episodes"]),
        "sample_count": manifest["sample_count"], "split_counts": manifest["split_counts"],
        "transport_proof_sha256": hashlib.sha256(proof.encode()).hexdigest(),
        "original_full_report_sha256": summary["original_full_report_sha256"],
        "episode_metrics_sha256": summary["original_episode_metrics_sha256"],
        "initial_notebook_id": NOTEBOOK_ID, "initial_notebook_version": initial_notebook_version,
        "initial_archive_sha256": sha256_file(initial_archive), "initial_archive_bytes": initial_archive.stat().st_size,
        "initial_checkpoint_member": initial_checkpoint_member, "initial_checkpoint_sha256": initial_sha,
        "training_protocol": protocol.to_dict(), "training_protocol_sha256": protocol.sha256}
    payload = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    packed = base64.b64encode(zlib.compress(payload, 9)).decode()
    source = (ROOT / "scripts/kaggle_development_training_bootstrap.py").read_text(encoding="utf-8")
    download_source = (ROOT / "scripts/kaggle_recovery_training_bootstrap.py").read_text(encoding="utf-8")
    downloader = next(node for node in ast.parse(download_source).body
                      if isinstance(node, ast.FunctionDef) and node.name == "resumable_http_download")
    helper = ast.get_source_segment(download_source, downloader)
    code = (source + "\nimport base64,zlib\n_payload=zlib.decompress(base64.b64decode(" + repr(packed) + ",validate=True))\n"
        "if hashlib.sha256(_payload).hexdigest()!=" + repr(hashlib.sha256(payload).hexdigest())
        + ":raise SystemExit('development-project-payload-mismatch')\nPROJECT_SOURCES=json.loads(_payload.decode())\n"
        + "FROZEN_INPUT=" + repr(frozen) + "\nTRUSTED_CPU_PROOF=" + repr(proof)
        + "\nDOWNLOAD_HELPER_SOURCE=" + repr(helper) + "\nMAX_SECONDS=" + repr(maximum_seconds)
        + "\nTRAINING_SECONDS=" + repr(training_seconds) + "\nraise SystemExit(main_development())\n")
    compile(code, "development_training.py", "exec")
    _require(len(code.encode()) < 1_000_000, "development-source-size-exceeded")
    submission = {"id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE, "code_file": "development_training.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "kernel_sources": [mount_reference]}
    for key in ("dataset_sources", "competition_sources", "model_sources"):
        values = metadata.get(key, [])
        _require(type(values) is list and all(type(value) is str and value and not any(c.isspace() for c in value) for value in values),
                 "development-existing-source-list-invalid")
        submission[key] = list(values)
    bundle = {"schema_version": 1, "notebook_id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE,
        "workload": "matched-appearance-development-training", "creates_new_notebook": False,
        "creates_dataset": False, "submitted": False, "local_gpu_used": False,
        "maximum_seconds": maximum_seconds, "training_seconds": training_seconds, "gpu_type": "T4",
        "code_sha256": hashlib.sha256(code.encode()).hexdigest(), "code_bytes": len(code.encode()),
        "source_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()},
        "project_payload_sha256": hashlib.sha256(payload).hexdigest(), "frozen_input": frozen,
        "cpu_proof_transport_kind": summary["transport_kind"], "new_native_replay_performed": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "development_training.py").write_bytes(code.encode())
    for name, value in (("kernel-metadata.json", submission), ("bundle_manifest.json", bundle),
                        ("frozen_input.json", frozen), ("trusted_cpu_proof.json", summary)):
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(proof if name == "trusted_cpu_proof.json" else json.dumps(value, indent=2, allow_nan=False) + "\n")
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("existing-metadata", "output-dir", "manifest", "validation-report", "source-audit",
                 "validation-amendment", "initial-archive", "initial-checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("archive-sha256", "mount-source-code-sha256", "initial-checkpoint-member"):
        parser.add_argument("--" + name, required=True)
    for name in ("source-notebook-version", "mount-source-version", "initial-notebook-version"):
        parser.add_argument("--" + name, type=int, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--maximum-seconds", type=int, default=MAX_SECONDS)
    parser.add_argument("--training-seconds", type=int, default=TRAINING_SECONDS)
    args = parser.parse_args(argv)
    result = prepare(args.existing_metadata, args.output_dir, manifest_path=args.manifest,
        validation_report_path=args.validation_report, source_audit_path=args.source_audit,
        validation_amendment_path=args.validation_amendment, archive_sha256=args.archive_sha256,
        source_notebook_version=args.source_notebook_version, mount_source_version=args.mount_source_version,
        mount_source_code_sha256=args.mount_source_code_sha256, initial_archive_path=args.initial_archive,
        initial_checkpoint_path=args.initial_checkpoint, initial_notebook_version=args.initial_notebook_version,
        initial_checkpoint_member=args.initial_checkpoint_member, protocol_path=args.protocol,
        maximum_seconds=args.maximum_seconds, training_seconds=args.training_seconds)
    print(json.dumps({name: result[name] for name in ("notebook_id", "code_sha256", "code_bytes", "submitted")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
