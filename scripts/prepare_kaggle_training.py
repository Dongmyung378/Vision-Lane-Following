"""Package RGB waypoint training for the existing GPU preflight notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ID = "dongmyungpark/vlf-carla-vulkan-preflight-20261005"
NOTEBOOK_TITLE = "VLF CARLA Vulkan preflight 20261005"
PILOT_NOTEBOOK_ID = "dongmyungpark/vlf-carla-rgb-smoke-20261005"
PILOT_DATASET_ID = "dongmyungpark/vlf-moving-rgb-waypoint-pilot"
SOURCES = (
    "carla_tools/__init__.py", "carla_tools/contracts.py", "carla_tools/learning_data.py",
    "carla_tools/waypoint_model.py", "carla_tools/training.py", "carla_tools/baseline.py",
    "configs/learning.json", "configs/baseline.json", "configs/dataset_contract_d5.json",
    "scripts/train_waypoints.py", "scripts/evaluate_waypoints.py", "requirements-learning.txt",
)


def prepare(existing_metadata: Path, output_dir: Path, *, pilot_archive_sha256: str,
            pilot_manifest_sha256: str, pilot_dataset: str | None = None,
            pilot_kernel_version: int | None = None, pilot_download_version: int | None = None) -> dict:
    metadata = json.loads(existing_metadata.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("existing-training-notebook-metadata-not-object")
    if metadata.get("id") != NOTEBOOK_ID or metadata.get("title") != NOTEBOOK_TITLE:
        raise ValueError("existing-training-notebook-id-or-title-mismatch")
    if any(not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value)
           for value in (pilot_archive_sha256, pilot_manifest_sha256)):
        raise ValueError("frozen-pilot-hashes-required")
    if pilot_dataset is not None and pilot_dataset != PILOT_DATASET_ID:
        raise ValueError("pilot-dataset-not-approved-stable-id")
    if pilot_kernel_version is not None and (type(pilot_kernel_version) is not int or pilot_kernel_version <= 0):
        raise ValueError("pilot-kernel-version-must-be-positive-integer")
    if pilot_dataset is not None and pilot_kernel_version is not None:
        raise ValueError("pilot-dataset-and-kernel-version-are-mutually-exclusive")
    if pilot_download_version is not None and (type(pilot_download_version) is not int or pilot_download_version <= 0):
        raise ValueError("pilot-download-version-must-be-positive-integer")
    if pilot_download_version is not None and pilot_dataset is not None:
        raise ValueError("pilot-dataset-and-cloud-version-are-mutually-exclusive")
    if (pilot_download_version is not None and pilot_kernel_version is not None and
            pilot_download_version != pilot_kernel_version):
        raise ValueError("pilot-cloud-and-kernel-version-mismatch")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("output-directory-already-exists")
    if any(parent.is_symlink() for parent in output_dir.parents):
        raise ValueError("output-parent-is-symlink")
    sources, hashes = {}, {}
    for name in SOURCES:
        source = (ROOT / name).read_text(encoding="utf-8")
        if name.endswith(".py"):
            compile(source, name, "exec")
        elif name.endswith(".json"):
            json.loads(source)
        sources[name] = source
        hashes[name] = hashlib.sha256(source.encode()).hexdigest()
    bootstrap = (ROOT / "scripts/kaggle_training_bootstrap.py").read_text(encoding="utf-8")
    code = (bootstrap + "\nPROJECT_SOURCES = " + repr(sources) +
            "\nPILOT_ARCHIVE_SHA256 = " + repr(pilot_archive_sha256) +
            "\nPILOT_MANIFEST_SHA256 = " + repr(pilot_manifest_sha256) +
            "\nPILOT_DOWNLOAD_VERSION = " + repr(pilot_download_version) +
            "\nraise SystemExit(main())\n")
    compile(code, "training.py", "exec")
    submission = {"id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE, "code_file": "training.py",
                  "language": "python", "kernel_type": "script", "is_private": True,
                  "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                  "machine_shape": "NvidiaTeslaT4"}
    for key in ("dataset_sources", "competition_sources", "model_sources"):
        values = metadata.get(key, [])
        if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip()
                or any(character.isspace() for character in value) for value in values):
            raise ValueError("existing-notebook-source-list-invalid")
        submission[key] = list(values)
    # The immutable archive/manifest hashes choose the exact pilot from this
    # existing notebook attachment or an explicitly prepared private dataset.
    # Packaging only describes attachments; it never creates either resource.
    pilot_kernel_source = PILOT_NOTEBOOK_ID + ("/" + str(pilot_kernel_version)
                                              if pilot_kernel_version is not None else "")
    submission["kernel_sources"] = [] if pilot_dataset is not None else [pilot_kernel_source]
    if pilot_dataset is not None and pilot_dataset not in submission["dataset_sources"]:
        submission["dataset_sources"].append(pilot_dataset)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "training.py").write_text(code, encoding="utf-8", newline="\n")
    (output_dir / "kernel-metadata.json").write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, "notebook_id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE,
                "workload": "pretrained-rgb-waypoint-training", "submitted": False,
                "creates_new_notebook": False, "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "source_sha256": hashes, "pilot_archive_sha256": pilot_archive_sha256,
                "pilot_manifest_sha256": pilot_manifest_sha256, "kernel_sources": submission["kernel_sources"],
                "dataset_sources": submission["dataset_sources"],
                "pilot_kernel_version": pilot_kernel_version,
                "pilot_download_version": pilot_download_version,
                "pilot_transport": "private-dataset" if pilot_dataset is not None else "existing-kernel-output"}
    (output_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-archive-sha256", required=True)
    parser.add_argument("--pilot-manifest-sha256", required=True)
    parser.add_argument("--pilot-dataset", choices=(PILOT_DATASET_ID,),
                        help="Attach the independently verified stable private pilot dataset instead of kernel output.")
    parser.add_argument("--pilot-kernel-version", type=int,
                        help="Request a positive numbered version of the same existing collection notebook output.")
    parser.add_argument("--pilot-download-version", type=int,
                        help="If pinned ZIP is not mounted, use KaggleHub platform authentication for this exact existing notebook version.")
    args = parser.parse_args(argv)
    try:
        report = prepare(args.existing_metadata, args.output_dir,
                         pilot_archive_sha256=args.pilot_archive_sha256,
                         pilot_manifest_sha256=args.pilot_manifest_sha256, pilot_dataset=args.pilot_dataset,
                         pilot_kernel_version=args.pilot_kernel_version,
                         pilot_download_version=args.pilot_download_version)
        print(json.dumps({key: report[key] for key in ("notebook_id", "title", "code_sha256",
                                                      "submitted", "creates_new_notebook")}))
        return 0
    except (OSError, ValueError, SyntaxError, KeyError, TypeError) as error:
        print(json.dumps({"status": "FAIL", "code": str(error), "exception_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
