"""Bundle selected-checkpoint driving for the existing private CARLA notebook.

This command only prepares local source and metadata. It neither uploads the
checkpoint nor creates datasets, notebooks, or remote execution sessions.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import re
import zlib
from pathlib import Path

from scripts.kaggle_learned_bootstrap import render_bootstrap
from scripts.prepare_kaggle_week1 import (
    BASELINE_SOURCES, DRIVING_SOURCES, NOTEBOOK_ID, NOTEBOOK_TITLE, SOURCES as COMMON_SOURCES,
)

ROOT = Path(__file__).resolve().parents[1]
LEARNED_SOURCES = (
    "carla_tools/learning_data.py", "carla_tools/waypoint_model.py", "carla_tools/training.py",
    "carla_tools/inference.py", "carla_tools/rgb_predictor.py", "carla_tools/learned_driving.py",
    "carla_tools/learned_validation.py", "scripts/run_learned_pipeline.py", "scripts/verify_learned.py",
    "configs/learning.json", "requirements-learning.txt",
)
SOURCES = COMMON_SOURCES + DRIVING_SOURCES + BASELINE_SOURCES + LEARNED_SOURCES


def prepare(existing_metadata: Path, output_dir: Path, *, checkpoint_dataset: str,
            checkpoint_sha256: str, checkpoint_bytes: int, mode: str = "learned",
            maximum_seconds: int = 3600) -> dict:
    metadata = json.loads(existing_metadata.read_text(encoding="utf-8"))
    if (not isinstance(metadata, dict) or metadata.get("id") != NOTEBOOK_ID
            or metadata.get("title") != NOTEBOOK_TITLE or metadata.get("is_private") is not True):
        raise ValueError("existing-private-notebook-id-or-title-mismatch")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("output-directory-already-exists")
    owner = NOTEBOOK_ID.split("/", 1)[0]
    if (not isinstance(checkpoint_dataset, str)
            or not re.fullmatch(re.escape(owner) + r"/[a-z0-9][a-z0-9-]{5,49}", checkpoint_dataset)):
        raise ValueError("owned-checkpoint-dataset-id-invalid")
    if not isinstance(checkpoint_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256):
        raise ValueError("checkpoint-sha256-invalid")
    if type(checkpoint_bytes) is not int or not 0 < checkpoint_bytes <= 2 * 1024 ** 3:
        raise ValueError("checkpoint-byte-size-invalid")
    if mode not in ("learned", "assisted"):
        raise ValueError("learned-predictor-mode-invalid")
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
        elif not relative.startswith("requirements-") or not relative.endswith(".txt"):
            raise ValueError("unsupported-public-source-extension")
        sources[relative] = source
        hashes[relative] = hashlib.sha256(data).hexdigest()
    base_bytes = (ROOT / "scripts/kaggle_carla_bootstrap.py").read_bytes()
    base_source = base_bytes.decode("utf-8")
    bootstrap = render_bootstrap(base_source, maximum_seconds=maximum_seconds)
    vulkan = (ROOT / "scripts/kaggle_vulkan_probe.py").read_text(encoding="utf-8")
    node = next(node for node in ast.parse(vulkan).body
                if isinstance(node, ast.FunctionDef) and node.name == "vulkan_devices")
    vulkan_child = ("import ctypes,ctypes.util,struct,json\n" + ast.get_source_segment(vulkan, node)
                    + "\nprint(json.dumps(vulkan_devices()))\n")
    payload = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload_sha = hashlib.sha256(payload).hexdigest()
    code = (bootstrap + "\nPROBE_SOURCE = " + repr(sources["scripts/carla_rgb_probe.py"])
            + "\nVULKAN_CHILD_SOURCE = " + repr(vulkan_child)
            + "\nPROJECT_SOURCES_PACKED = " + repr(base64.b64encode(zlib.compress(payload, 9)).decode("ascii"))
            + "\nPROJECT_PAYLOAD_SHA256 = " + repr(payload_sha)
            + "\nimport base64,zlib\n_project_payload = zlib.decompress(base64.b64decode(PROJECT_SOURCES_PACKED, validate=True))"
            + "\nif hashlib.sha256(_project_payload).hexdigest() != PROJECT_PAYLOAD_SHA256:"
            + "\n    raise SystemExit('embedded-project-transport-hash-mismatch')"
            + "\nPROJECT_SOURCES = json.loads(_project_payload.decode('utf-8'))"
            + "\nCHECKPOINT_SHA256 = " + repr(checkpoint_sha256)
            + "\nCHECKPOINT_BYTES = " + repr(checkpoint_bytes)
            + "\nLEARNED_MODE = " + repr(mode) + "\nraise SystemExit(main())\n")
    compile(code, "learned.py", "exec")
    if len(code.encode("utf-8")) >= 1_000_000:
        raise ValueError("kaggle-script-source-size-exceeded")
    submission = {"id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE, "code_file": "learned.py",
                  "language": "python", "kernel_type": "script", "is_private": True,
                  "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                  "machine_shape": "NvidiaTeslaT4"}
    for key in ("dataset_sources", "kernel_sources", "competition_sources", "model_sources"):
        values = metadata.get(key, [])
        if (not isinstance(values, list) or any(not isinstance(value, str) or not value.strip()
                or any(character.isspace() for character in value) for value in values)):
            raise ValueError("existing-notebook-source-list-invalid")
        submission[key] = list(values)
    if checkpoint_dataset not in submission["dataset_sources"]:
        submission["dataset_sources"].append(checkpoint_dataset)
    bundle = {
        "schema_version": 1, "notebook_id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE,
        "workload": "selected-rgb-waypoint-native-driving", "mode": mode,
        "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "source_sha256": hashes, "project_payload_sha256": payload_sha,
        "base_bootstrap_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "bootstrap_generator_sha256": hashlib.sha256((ROOT / "scripts/kaggle_learned_bootstrap.py").read_bytes()).hexdigest(),
        "checkpoint_input": {"dataset_id": checkpoint_dataset, "filename": "checkpoint.pt",
                             "sha256": checkpoint_sha256, "bytes": checkpoint_bytes},
        "maximum_seconds": maximum_seconds, "submitted": False, "checkpoint_uploaded": False,
        "creates_new_notebook": False, "creates_dataset": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "learned.py").write_bytes(code.encode("utf-8"))
    for name, value in (("kernel-metadata.json", submission), ("bundle_manifest.json", bundle)):
        with (output_dir / name).open("x", encoding="utf-8", newline="\n") as target:
            target.write(json.dumps(value, indent=2) + "\n")
    return bundle


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dataset", required=True, help="Existing authorized private checkpoint dataset ID.")
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-bytes", type=int, required=True)
    parser.add_argument("--mode", choices=("learned", "assisted"), default="learned")
    parser.add_argument("--maximum-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    try:
        report = prepare(args.existing_metadata, args.output_dir, checkpoint_dataset=args.checkpoint_dataset,
                         checkpoint_sha256=args.checkpoint_sha256, checkpoint_bytes=args.checkpoint_bytes,
                         mode=args.mode, maximum_seconds=args.maximum_seconds)
        print(json.dumps({key: report[key] for key in ("notebook_id", "workload", "mode", "code_sha256",
                                                      "submitted", "checkpoint_uploaded", "creates_new_notebook")}))
        return 0
    except (OSError, ValueError, SyntaxError, KeyError, TypeError) as error:
        print(json.dumps({"status": "FAIL", "exception_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
