"""Prepare a controlled spatial comparison for the existing private T4 notebook.

Preparation is local only; submission requires a fresh read-only identity check.
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

from scripts.kaggle_spatial_bootstrap import MAX_SECONDS, TRAINING_SECONDS

NOTEBOOK_ID = "dongmyungpark/vlf-carla-vulkan-preflight-20261005"
NOTEBOOK_TITLE = "VLF CARLA Vulkan preflight 20261005"
PILOT_NOTEBOOK_ID = "dongmyungpark/vlf-carla-rgb-smoke-20261005"
SOURCES = ("carla_tools/__init__.py", "carla_tools/contracts.py", "carla_tools/learning_data.py",
           "carla_tools/training.py", "carla_tools/waypoint_model.py", "carla_tools/baseline.py",
           "carla_tools/spatial_waypoint.py", "carla_tools/spatial_training.py", "scripts/train_spatial_waypoints.py",
           "configs/learning.json", "configs/baseline.json", "configs/dataset_contract_d5.json", "requirements-learning.txt")


def prepare(existing_metadata, output_dir, *, pilot_archive_sha256, pilot_manifest_sha256,
            pilot_download_version, maximum_seconds=MAX_SECONDS, training_seconds=TRAINING_SECONDS, max_epochs=32,
            reused_baseline_path=None):
    metadata = json.loads(Path(existing_metadata).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("id") != NOTEBOOK_ID or metadata.get("title") != NOTEBOOK_TITLE or metadata.get("is_private") is not True:
        raise ValueError("existing-private-notebook-required")
    if any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in (pilot_archive_sha256, pilot_manifest_sha256)):
        raise ValueError("frozen-pilot-hashes-required")
    if type(pilot_download_version) is not int or pilot_download_version <= 0:
        raise ValueError("positive-pilot-version-required")
    if type(maximum_seconds) is not int or not 900 <= maximum_seconds <= MAX_SECONDS:
        raise ValueError("total-time-limit-invalid")
    if type(training_seconds) is not int or not 60 <= training_seconds <= maximum_seconds - 300:
        raise ValueError("training-time-limit-invalid")
    if type(max_epochs) is not int or not 1 <= max_epochs <= 64:
        raise ValueError("epoch-limit-invalid")
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink() or any(p.is_symlink() for p in output_dir.parents):
        raise ValueError("new-local-output-directory-required")
    sources, hashes = {}, {}
    for relative in SOURCES:
        path = ROOT / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("runtime-source-path-invalid")
        raw = path.read_bytes()
        source = raw.decode("utf-8")
        if relative.endswith(".py"):
            compile(source, relative, "exec")
        elif relative.endswith(".json"):
            json.loads(source)
        sources[relative], hashes[relative] = source, hashlib.sha256(raw).hexdigest()
    base = (ROOT / "scripts/kaggle_training_bootstrap.py").read_bytes()
    bootstrap = (ROOT / "scripts/kaggle_spatial_bootstrap.py").read_bytes()
    payload = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()
    packed = base64.b64encode(zlib.compress(payload, 9)).decode("ascii")
    frozen_baseline = None
    if reused_baseline_path is not None:
        path = Path(reused_baseline_path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("frozen-baseline-file-required")
        frozen_baseline = path.read_bytes().decode("utf-8")
        if len(frozen_baseline.encode()) > 500_000:
            raise ValueError("frozen-baseline-size-exceeded")
        json.loads(frozen_baseline)
    code = (base.decode() + "\n" + bootstrap.decode()
            + "\nimport base64,zlib\n_payload=zlib.decompress(base64.b64decode(" + repr(packed) + ",validate=True))\n"
            + "if hashlib.sha256(_payload).hexdigest()!=" + repr(digest) + ": raise SystemExit('project-payload-mismatch')\n"
            + "PROJECT_SOURCES=json.loads(_payload.decode())\n"
            + "PILOT_ARCHIVE_SHA256=" + repr(pilot_archive_sha256) + "\n"
            + "PILOT_MANIFEST_SHA256=" + repr(pilot_manifest_sha256) + "\n"
            + "PILOT_DOWNLOAD_VERSION=" + repr(pilot_download_version) + "\n"
            + "MAX_SECONDS=" + repr(maximum_seconds) + "\nTRAINING_SECONDS=" + repr(training_seconds)
            + "\nMAX_EPOCHS=" + repr(max_epochs) + "\nFROZEN_BASELINE=" + repr(frozen_baseline)
            + "\nraise SystemExit(main_spatial())\n").replace("\r\n", "\n")
    compile(code, "spatial_training.py", "exec")
    if len(code.encode()) >= 1_000_000:
        raise ValueError("source-size-exceeded")
    submission = {"id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE, "code_file": "spatial_training.py",
                  "language": "python", "kernel_type": "script", "is_private": True,
                  "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                  "machine_shape": "NvidiaTeslaT4", "kernel_sources": [PILOT_NOTEBOOK_ID]}
    for key in ("dataset_sources", "competition_sources", "model_sources"):
        values = metadata.get(key, [])
        if not isinstance(values, list) or any(not isinstance(v, str) or not v or any(c.isspace() for c in v) for v in values):
            raise ValueError("existing-source-list-invalid")
        submission[key] = list(values)
    bundle = {"schema_version": 1, "notebook_id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE,
              "workload": "controlled-spatial-hybrid-comparison", "creates_new_notebook": False,
              "creates_dataset": False, "submitted": False, "local_gpu_used": False,
              "code_sha256": hashlib.sha256(code.encode()).hexdigest(), "source_sha256": hashes,
              "project_payload_sha256": digest, "base_bootstrap_sha256": hashlib.sha256(base).hexdigest(),
              "spatial_bootstrap_sha256": hashlib.sha256(bootstrap).hexdigest(),
              "pilot_archive_sha256": pilot_archive_sha256, "pilot_manifest_sha256": pilot_manifest_sha256,
              "pilot_download_version": pilot_download_version, "maximum_seconds": maximum_seconds,
              "training_seconds": training_seconds, "max_epochs": max_epochs, "gpu_type": "T4"}
    if frozen_baseline is not None:
        bundle["reused_baseline_sha256"] = hashlib.sha256(frozen_baseline.encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "spatial_training.py").write_bytes(code.encode())
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
    parser.add_argument("--max-epochs", type=int, default=32)
    parser.add_argument("--reused-baseline",type=Path)
    args = parser.parse_args(argv)
    report = prepare(args.existing_metadata, args.output_dir, pilot_archive_sha256=args.pilot_archive_sha256,
                     pilot_manifest_sha256=args.pilot_manifest_sha256, pilot_download_version=args.pilot_download_version,
                     max_epochs=args.max_epochs,reused_baseline_path=args.reused_baseline)
    print(json.dumps({k: report[k] for k in ("notebook_id", "code_sha256", "creates_new_notebook", "submitted")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
