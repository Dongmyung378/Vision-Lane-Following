"""Prepare paired training for the existing private Kaggle T4 notebook.

This command prepares local files only. Before submission, the caller must
perform a fresh read-only identity and access check of the existing notebook.
The input evidence and generated bundle can contain private execution records
and should be kept outside public source control.
"""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.contracts import canonical_sha256, read_strict_json
from carla_tools.dataset_v1 import (GATE_VERSION, SCOPE, VALIDATION_CHECKS,
                                    _implementation_hashes, dataset_v1_config_from_dict)
from scripts.kaggle_recovery_training_bootstrap import MAX_SECONDS, TRAINING_SECONDS, sha256_file

NOTEBOOK_ID = "dongmyungpark/vlf-carla-vulkan-preflight-20261005"
NOTEBOOK_TITLE = "VLF CARLA Vulkan preflight 20261005"
DATA_NOTEBOOK_ID = "dongmyungpark/vlf-carla-rgb-smoke-20261005"
CHECKPOINT_MEMBER = "experiment/hybrid_geometry/checkpoint.pt"


def _require(value, code):
    if not value:
        raise ValueError(code)


def _digest(value):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "recovery-frozen-sha256-required")
    return value


def trusted_cpu_summary(manifest_path, report_path, audit_path, amendment_path, *, archive_sha256):
    """Summarize prior complete CPU evidence without inventing a new replay."""
    paths = [Path(path) for path in (manifest_path, report_path, audit_path, amendment_path)]
    _require(all(path.is_file() and not path.is_symlink() for path in paths), "recovery-evidence-file-required")
    manifest, report, audit, amendment = [read_strict_json(path) for path in paths]
    original = copy.deepcopy(manifest)
    snapshot = original.pop("snapshot_sha256", None)
    config = dataset_v1_config_from_dict(manifest["config"])
    manifest_sha = sha256_file(paths[0])
    _require(manifest.get("scope") == SCOPE and manifest.get("status") == "COMPLETE"
             and manifest.get("town") == "Town01" and manifest.get("reserved_final_town") == "Town02"
             and manifest.get("reserved_final_state") == "UNOPENED"
             and snapshot == canonical_sha256(original)
             and manifest.get("implementation_sha256") == _implementation_hashes()
             and manifest.get("config_sha256") == config.sha256,
             "recovery-manifest-snapshot-or-source-invalid")
    episodes = manifest["episodes"]
    metrics = report.get("episode_metrics")
    _require(type(episodes) is list and episodes and type(metrics) is list and len(metrics) == len(episodes)
             and all(type(row) is dict and type(row.get("checks")) is dict and row["checks"]
                     and all(value is True for value in row["checks"].values())
                     and type(row.get("evaluation_metrics")) is dict for row in metrics)
             and [row["episode_id"] for row in metrics] == [row["episode_id"] for row in episodes],
             "recovery-complete-native-episode-proof-required")
    _require(report.get("status") == "PASS" and report.get("scope") == SCOPE
             and report.get("gate_version") == GATE_VERSION and report.get("errors") == []
             and report.get("manifest_sha256") == manifest_sha
             and report.get("snapshot_sha256") == snapshot and report.get("config_sha256") == config.sha256
             and report.get("sample_count") == manifest.get("sample_count")
             and report.get("episode_count") == len(episodes)
             and report.get("split_counts") == manifest.get("split_counts")
             and type(report.get("checks")) is dict and set(report["checks"]) == set(VALIDATION_CHECKS)
             and all(value is True for value in report["checks"].values()),
             "recovery-complete-CPU-report-binding-required")
    _require(audit.get("status") == "PASS" and audit.get("archive_sha256") == archive_sha256
             and type(audit.get("archive_bytes")) is int and 0 < audit["archive_bytes"] <= 32 * 1024**3
             and audit.get("manifest_sha256") == manifest_sha and audit.get("snapshot_sha256") == snapshot
             and audit.get("episode_count") == len(episodes)
             and audit.get("validation_amendment_sha256") == sha256_file(paths[3])
             and audit.get("parallel_native_replay", {}).get("actual_native_replays") == len(episodes)
             and amendment.get("status") == "FROZEN_BEFORE_CPU_DERIVATION"
             and amendment.get("original_archive_sha256") == archive_sha256
             and amendment.get("same_700_source_episodes") is True
             and amendment.get("same_sample_budgets") is True
             and amendment.get("new_GPU_job_started") is False,
             "recovery-independent-source-audit-required")
    summary = {key: value for key, value in report.items() if key != "episode_metrics"}
    summary.update(transport_kind="validated-report-transport-summary", new_native_replay_performed=False,
                   original_full_report_sha256=sha256_file(paths[1]),
                   original_episode_metrics_sha256=canonical_sha256(metrics),
                   original_episode_metrics_count=len(metrics),
                   original_independent_audit_sha256=sha256_file(paths[2]),
                   original_validation_amendment_sha256=sha256_file(paths[3]),
                   original_archive_sha256=archive_sha256)
    return summary, manifest


def _reviewed_sources():
    sources = {}
    for pattern in ("carla_tools/*.py", "scripts/*.py", "configs/*.json"):
        for path in sorted(ROOT.glob(pattern)):
            _require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT.resolve()),
                     "recovery-runtime-source-path-invalid")
            data = path.read_bytes()
            text = data.decode("utf-8")
            _require(b"\r\n" not in data, "recovery-source-LF-required")
            relative = path.relative_to(ROOT).as_posix()
            if path.suffix == ".py":
                compile(text, relative, "exec")
            else:
                json.loads(text)
            sources[relative] = text
    _require("scripts/train_recovery_waypoints.py" in sources
             and "carla_tools/recovery_training.py" in sources
             and "configs/recovery_training.json" in sources,
             "recovery-training-runtime-source-required")
    return sources


def prepare(existing_metadata, output_dir, *, manifest_path, validation_report_path, source_audit_path,
            validation_amendment_path, archive_sha256, source_notebook_version,
            initial_archive_path, initial_checkpoint_path, initial_notebook_version,
            maximum_seconds=MAX_SECONDS, training_seconds=TRAINING_SECONDS):
    metadata = read_strict_json(existing_metadata)
    _require(type(metadata) is dict and metadata.get("id") == NOTEBOOK_ID
             and metadata.get("title") == NOTEBOOK_TITLE and metadata.get("is_private") is True,
             "existing-private-notebook-required")
    _digest(archive_sha256)
    for version in (source_notebook_version, initial_notebook_version):
        _require(type(version) is int and version > 0, "recovery-positive-source-version-required")
    _require(type(maximum_seconds) is int and 1800 <= maximum_seconds <= 43200
             and type(training_seconds) is int and 300 <= training_seconds <= maximum_seconds - 900,
             "recovery-time-budget-invalid")
    output = Path(output_dir)
    _require(not output.exists() and not output.is_symlink() and not any(p.is_symlink() for p in output.parents),
             "new-local-output-directory-required")
    summary, manifest = trusted_cpu_summary(manifest_path, validation_report_path, source_audit_path,
                                            validation_amendment_path, archive_sha256=archive_sha256)
    initial_archive, initial_checkpoint = Path(initial_archive_path), Path(initial_checkpoint_path)
    _require(all(path.is_file() and not path.is_symlink() for path in (initial_archive, initial_checkpoint)),
             "recovery-frozen-initial-artifacts-required")
    initial_sha = sha256_file(initial_checkpoint)
    with zipfile.ZipFile(initial_archive) as archive:
        _require(archive.namelist().count(CHECKPOINT_MEMBER) == 1, "recovery-initial-member-not-unique")
        member = archive.getinfo(CHECKPOINT_MEMBER)
        _require(member.file_size == initial_checkpoint.stat().st_size and member.file_size <= 512 * 1024**2
                 and not member.is_dir() and not member.flag_bits & 1
                 and (member.external_attr >> 16) & 0o170000 != 0o120000,
                 "recovery-initial-member-invalid")
        digest = hashlib.sha256()
        with archive.open(member) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        _require(digest.hexdigest() == initial_sha, "recovery-initial-archive-checkpoint-binding-invalid")
    sources = _reviewed_sources()
    protocol = json.loads(sources["configs/recovery_training.json"])
    proof = json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    frozen = {"archive_sha256": archive_sha256, "source_notebook_id": DATA_NOTEBOOK_ID,
              "archive_bytes": read_strict_json(source_audit_path)["archive_bytes"],
              "source_notebook_version": source_notebook_version,
              "manifest_sha256": sha256_file(manifest_path), "snapshot_sha256": manifest["snapshot_sha256"],
              "config_sha256": manifest["config_sha256"], "episode_count": len(manifest["episodes"]),
              "sample_count": manifest["sample_count"], "split_counts": manifest["split_counts"],
              "transport_proof_sha256": hashlib.sha256(proof.encode()).hexdigest(),
              "original_full_report_sha256": summary["original_full_report_sha256"],
              "episode_metrics_sha256": summary["original_episode_metrics_sha256"],
              "initial_notebook_id": NOTEBOOK_ID, "initial_notebook_version": initial_notebook_version,
              "initial_archive_sha256": sha256_file(initial_archive), "initial_checkpoint_member": CHECKPOINT_MEMBER,
              "initial_archive_bytes": initial_archive.stat().st_size,
              "initial_checkpoint_sha256": initial_sha,
              "training_protocol_sha256": canonical_sha256(protocol)}
    payload = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    packed = base64.b64encode(zlib.compress(payload, 9)).decode()
    source = (ROOT / "scripts/kaggle_recovery_training_bootstrap.py").read_bytes().decode()
    downloader_node = next(node for node in ast.parse(source).body
                           if isinstance(node, ast.FunctionDef) and node.name == "resumable_http_download")
    downloader_source = ast.get_source_segment(source, downloader_node)
    code = (source + "\nimport base64,zlib\n_payload=zlib.decompress(base64.b64decode(" + repr(packed) + ",validate=True))\n"
            "if hashlib.sha256(_payload).hexdigest()!=" + repr(hashlib.sha256(payload).hexdigest())
            + ": raise SystemExit('recovery-project-payload-mismatch')\nPROJECT_SOURCES=json.loads(_payload.decode())\n"
            + "FROZEN_INPUT=" + repr(frozen) + "\nTRUSTED_CPU_PROOF=" + repr(proof)
            + "\nDOWNLOAD_HELPER_SOURCE=" + repr(downloader_source)
            + "\nMAX_SECONDS=" + repr(maximum_seconds) + "\nTRAINING_SECONDS=" + repr(training_seconds)
            + "\nraise SystemExit(main_recovery())\n")
    compile(code, "recovery_training.py", "exec")
    _require(len(code.encode()) < 1_000_000, "recovery-source-size-exceeded")
    submission = {"id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE, "code_file": "recovery_training.py",
                  "language": "python", "kernel_type": "script", "is_private": True,
                  "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                  "machine_shape": "NvidiaTeslaT4", "kernel_sources": []}
    for key in ("dataset_sources", "competition_sources", "model_sources"):
        values = metadata.get(key, [])
        _require(type(values) is list and all(type(value) is str and value and not any(c.isspace() for c in value)
                                            for value in values), "recovery-existing-source-list-invalid")
        submission[key] = list(values)
    bundle = {"schema_version": 1, "notebook_id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE,
              "workload": "matched-budget-central-recovery-training", "creates_new_notebook": False,
              "creates_dataset": False, "submitted": False, "local_gpu_used": False,
              "maximum_seconds": maximum_seconds, "training_seconds": training_seconds, "gpu_type": "T4",
              "code_sha256": hashlib.sha256(code.encode()).hexdigest(), "code_bytes": len(code.encode()),
              "source_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()},
              "project_payload_sha256": hashlib.sha256(payload).hexdigest(), "frozen_input": frozen,
              "cpu_proof_transport_kind": summary["transport_kind"], "new_native_replay_performed": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "recovery_training.py").write_bytes(code.encode())
    for name, value in (("kernel-metadata.json", submission), ("bundle_manifest.json", bundle),
                        ("frozen_input.json", frozen), ("trusted_cpu_proof.json", summary)):
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            if name == "trusted_cpu_proof.json":
                stream.write(proof)
            else:
                stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("existing-metadata", "output-dir", "manifest", "validation-report", "source-audit",
                 "validation-amendment", "initial-archive", "initial-checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--source-notebook-version", type=int, required=True)
    parser.add_argument("--initial-notebook-version", type=int, required=True)
    parser.add_argument("--maximum-seconds", type=int, default=MAX_SECONDS)
    parser.add_argument("--training-seconds", type=int, default=TRAINING_SECONDS)
    args = parser.parse_args(argv)
    result = prepare(args.existing_metadata, args.output_dir, manifest_path=args.manifest,
        validation_report_path=args.validation_report, source_audit_path=args.source_audit,
        validation_amendment_path=args.validation_amendment, archive_sha256=args.archive_sha256,
        source_notebook_version=args.source_notebook_version, initial_archive_path=args.initial_archive,
        initial_checkpoint_path=args.initial_checkpoint, initial_notebook_version=args.initial_notebook_version,
        maximum_seconds=args.maximum_seconds, training_seconds=args.training_seconds)
    print(json.dumps({key: result[key] for key in ("notebook_id", "code_sha256", "code_bytes", "submitted")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
