"""Prepare a frozen DINOv3 waypoint experiment for the existing T4 notebook.

Preparation writes a local bundle. It never submits a job, uploads data, or
creates a Kaggle notebook or dataset.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.kaggle_vit_bootstrap import (
    MAX_SECONDS, PRETRAINED_BYTES, PRETRAINED_SHA256, PRETRAINED_URL,
)

NOTEBOOK_ID = "dongmyungpark/vlf-carla-vulkan-preflight-20261005"
NOTEBOOK_TITLE = "VLF CARLA Vulkan preflight 20261005"
PILOT_NOTEBOOK_ID = "dongmyungpark/vlf-carla-rgb-smoke-20261005"
SOURCES = (
    "carla_tools/__init__.py", "carla_tools/contracts.py",
    "carla_tools/learning_data.py", "carla_tools/training.py",
    "carla_tools/waypoint_model.py", "carla_tools/baseline.py",
    "carla_tools/vit_waypoint.py", "carla_tools/vit_training.py",
    "scripts/train_vit_waypoints.py", "configs/learning.json",
    "configs/baseline.json", "configs/dataset_contract_d5.json",
    "requirements-learning.txt", "requirements-vit.txt",
)


def prepare(existing_metadata: Path, output_dir: Path, *, pilot_archive_sha256: str,
            pilot_manifest_sha256: str, pilot_download_version: int,
            maximum_seconds: int = MAX_SECONDS, training_seconds: int = 1200,
            max_epochs: int = 32) -> dict:
    metadata = json.loads(existing_metadata.read_text(encoding="utf-8"))
    if (not isinstance(metadata, dict) or metadata.get("id") != NOTEBOOK_ID
            or metadata.get("title") != NOTEBOOK_TITLE or metadata.get("is_private") is not True):
        raise ValueError("existing-private-training-notebook-id-or-title-mismatch")
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
           for value in (pilot_archive_sha256, pilot_manifest_sha256)):
        raise ValueError("frozen-pilot-hashes-required")
    if type(pilot_download_version) is not int or pilot_download_version <= 0:
        raise ValueError("pilot-download-version-must-be-positive-integer")
    if type(maximum_seconds) is not int or not 900 <= maximum_seconds <= MAX_SECONDS:
        raise ValueError("vit-total-time-limit-invalid")
    if type(training_seconds) is not int or not 60 <= training_seconds <= maximum_seconds - 300:
        raise ValueError("vit-training-time-limit-invalid")
    if type(max_epochs) is not int or not 1 <= max_epochs <= 64:
        raise ValueError("vit-epoch-limit-invalid")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("output-directory-already-exists")
    if any(parent.is_symlink() for parent in output_dir.parents):
        raise ValueError("output-parent-is-symlink")
    sources, hashes = {}, {}
    for relative in SOURCES:
        path = ROOT / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("public-runtime-source-path-invalid")
        data = path.read_bytes()
        source = data.decode("utf-8")
        if relative.endswith(".py"):
            compile(source, relative, "exec")
        elif relative.endswith(".json"):
            json.loads(source)
        sources[relative] = source
        hashes[relative] = hashlib.sha256(data).hexdigest()
    base = (ROOT / "scripts/kaggle_training_bootstrap.py").read_bytes()
    bootstrap = (ROOT / "scripts/kaggle_vit_bootstrap.py").read_bytes()
    payload = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload_sha = hashlib.sha256(payload).hexdigest()
    packed = base64.b64encode(zlib.compress(payload, 9)).decode("ascii")
    code = (base.decode("utf-8") + "\n" + bootstrap.decode("utf-8")
            + "\nPROJECT_SOURCES_PACKED = " + repr(packed)
            + "\nPROJECT_PAYLOAD_SHA256 = " + repr(payload_sha)
            + "\nimport base64,zlib\n"
            + "_project_payload = zlib.decompress(base64.b64decode(PROJECT_SOURCES_PACKED, validate=True))\n"
            + "if hashlib.sha256(_project_payload).hexdigest() != PROJECT_PAYLOAD_SHA256:\n"
            + "    raise SystemExit('embedded-project-transport-hash-mismatch')\n"
            + "PROJECT_SOURCES = json.loads(_project_payload.decode('utf-8'))\n"
            + "PILOT_ARCHIVE_SHA256 = " + repr(pilot_archive_sha256)
            + "\nPILOT_MANIFEST_SHA256 = " + repr(pilot_manifest_sha256)
            + "\nPILOT_DOWNLOAD_VERSION = " + repr(pilot_download_version)
            + "\nMAX_SECONDS = " + repr(maximum_seconds)
            + "\nTRAINING_SECONDS = " + repr(training_seconds)
            + "\nMAX_EPOCHS = " + repr(max_epochs)
            + "\nraise SystemExit(main_vit())\n")
    # Kaggle's SDK reads script text with universal newline conversion. Keep
    # wrapper bytes identical to that submitted text; the compressed payload
    # still preserves every original project source byte, including CRLF.
    code = code.replace("\r\n", "\n")
    compile(code, "vit_training.py", "exec")
    if len(code.encode("utf-8")) >= 1_000_000:
        raise ValueError("vit-script-source-size-exceeded")
    submission = {
        "id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE, "code_file": "vit_training.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        # Mount the already existing private collection output. KaggleHub's
        # platform resolver can require this attachment for private access.
        # The mount is accepted only when both frozen byte hashes match.
        "machine_shape": "NvidiaTeslaT4", "kernel_sources": [PILOT_NOTEBOOK_ID],
    }
    for key in ("dataset_sources", "competition_sources", "model_sources"):
        values = metadata.get(key, [])
        if (not isinstance(values, list) or any(not isinstance(value, str) or not value.strip()
                or any(character.isspace() for character in value) for value in values)):
            raise ValueError("existing-notebook-source-list-invalid")
        submission[key] = list(values)
    bundle = {
        "schema_version": 1, "notebook_id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE,
        "workload": "frozen-dinov3-rgb-waypoint-comparison", "submitted": False,
        "creates_new_notebook": False, "creates_dataset": False,
        "local_gpu_used": False, "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "source_sha256": hashes, "project_payload_sha256": payload_sha,
        "base_bootstrap_sha256": hashlib.sha256(base).hexdigest(),
        "vit_bootstrap_sha256": hashlib.sha256(bootstrap).hexdigest(),
        "pilot_archive_sha256": pilot_archive_sha256,
        "pilot_manifest_sha256": pilot_manifest_sha256,
        "pilot_transport": "verified-existing-notebook-mount-or-explicit-version-download",
        "kernel_sources": submission["kernel_sources"],
        "pilot_mount_version_policy": "latest-existing-output-accepted-only-with-frozen-archive-and-manifest-hashes",
        "pilot_notebook_id": PILOT_NOTEBOOK_ID, "pilot_download_version": pilot_download_version,
        "pretrained": {"url": PRETRAINED_URL, "sha256": PRETRAINED_SHA256, "bytes": PRETRAINED_BYTES},
        "maximum_seconds": maximum_seconds, "training_seconds": training_seconds,
        "max_epochs": max_epochs, "precision": "FP16 frozen encoder; FP32 waypoint head",
        "cuda_required": True, "gpu_type": "T4",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "vit_training.py").write_bytes(code.encode("utf-8"))
    for name, value in (("kernel-metadata.json", submission), ("bundle_manifest.json", bundle)):
        with (output_dir / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, indent=2) + "\n")
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-archive-sha256", required=True)
    parser.add_argument("--pilot-manifest-sha256", required=True)
    parser.add_argument("--pilot-download-version", type=int, required=True)
    parser.add_argument("--maximum-seconds", type=int, default=MAX_SECONDS)
    parser.add_argument("--training-seconds", type=int, default=1200)
    parser.add_argument("--max-epochs", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        report = prepare(args.existing_metadata, args.output_dir,
                         pilot_archive_sha256=args.pilot_archive_sha256,
                         pilot_manifest_sha256=args.pilot_manifest_sha256,
                         pilot_download_version=args.pilot_download_version,
                         maximum_seconds=args.maximum_seconds,
                         training_seconds=args.training_seconds, max_epochs=args.max_epochs)
        print(json.dumps({key: report[key] for key in ("notebook_id", "workload", "code_sha256",
                                                      "submitted", "creates_new_notebook", "creates_dataset")}))
        return 0
    except (OSError, ValueError, SyntaxError, KeyError, TypeError) as error:
        print(json.dumps({"status": "FAIL", "exception_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
