"""Bundle public D6/D7 sources for an update to the existing CARLA notebook."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ID = "dongmyungpark/vlf-carla-rgb-smoke-20261005"
NOTEBOOK_TITLE = "VLF CARLA RGB smoke 20261005"
SOURCES = (
    "carla_tools/__init__.py", "carla_tools/config.py", "carla_tools/session.py",
    "carla_tools/capture.py", "carla_tools/episode.py", "carla_tools/contracts.py",
    "carla_tools/color.py", "carla_tools/routes.py", "carla_tools/geometry_survey.py",
    "carla_tools/labels.py", "carla_tools/projection.py", "carla_tools/week1.py",
    "scripts/carla_rgb_probe.py", "scripts/verify_collection.py", "scripts/check_carla_connection.py",
    "scripts/run_geometry_collection.py", "scripts/run_week1_pipeline.py",
    "configs/collection_d4.json", "configs/dataset_contract_d5.json",
)
DRIVING_SOURCES = (
    "carla_tools/controller.py", "carla_tools/driving_route.py", "carla_tools/driving.py",
    "carla_tools/driving_validation.py", "scripts/run_driving_pipeline.py", "scripts/verify_driving.py",
    "configs/controller_d8.json", "configs/driving_validation_d10.json", "configs/routes_d6_geometry.json",
)
BASELINE_SOURCES = (
    "carla_tools/baseline.py", "carla_tools/baseline_driving.py", "carla_tools/baseline_validation.py",
    "carla_tools/evaluation.py", "scripts/run_baseline_pipeline.py", "scripts/verify_baseline.py",
    "configs/baseline.json", "configs/evaluation.json", "requirements-baseline.txt",
)


def prepare(existing_metadata: Path, output_dir: Path, *, workload="week1") -> dict:
    if workload not in ("week1", "driving", "baseline"):
        raise ValueError("unsupported-existing-notebook-workload")
    metadata = json.loads(existing_metadata.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("existing-notebook-metadata-not-object")
    if metadata.get("id") != NOTEBOOK_ID or metadata.get("title") != NOTEBOOK_TITLE:
        raise ValueError("existing-notebook-id-or-title-mismatch")
    if output_dir.exists():
        raise ValueError("output-directory-already-exists")
    sources = {}
    manifest = {}
    selected = SOURCES
    if workload in ("driving", "baseline"):
        selected += DRIVING_SOURCES
    if workload == "baseline":
        selected += BASELINE_SOURCES
    for relative in selected:
        source = (ROOT / relative).read_text(encoding="utf-8")
        if relative.endswith(".py"):
            compile(source, relative, "exec")
        elif relative.endswith(".json"):
            json.loads(source)
        elif relative == "requirements-baseline.txt":
            requirements = {line.strip() for line in source.splitlines() if line.strip() and not line.lstrip().startswith("#")}
            if requirements != {"opencv-python-headless==4.12.0.88", "numpy==2.2.6"}:
                raise ValueError("baseline-runtime-dependency-pins-mismatch")
        else:
            raise ValueError("unsupported-public-source-extension")
        sources[relative] = source
        manifest[relative] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    bootstrap = (ROOT / "scripts/kaggle_carla_bootstrap.py").read_text(encoding="utf-8")
    vulkan = (ROOT / "scripts/kaggle_vulkan_probe.py").read_text(encoding="utf-8")
    function = next(node for node in ast.parse(vulkan).body if isinstance(node, ast.FunctionDef) and node.name == "vulkan_devices")
    child = "import ctypes,ctypes.util,struct,json\n" + ast.get_source_segment(vulkan, function) + "\nprint(json.dumps(vulkan_devices()))\n"
    code = (bootstrap + "\nPROBE_SOURCE = " + repr(sources["scripts/carla_rgb_probe.py"]) +
            "\nVULKAN_CHILD_SOURCE = " + repr(child) + "\nPROJECT_SOURCES = " + repr(sources) +
            "\nWORKLOAD = " + repr(workload) + "\nraise SystemExit(main())\n")
    compile(code, "week1.py", "exec")
    # Copy only submission fields. Credentials and machine-local configuration
    # cannot enter the package through an arbitrary metadata key.
    submission = {"id": metadata["id"], "title": metadata["title"], "code_file": "week1.py",
                  "language": "python", "kernel_type": "script", "is_private": True,
                  "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                  "machine_shape": "NvidiaTeslaT4"}
    for key in ("dataset_sources", "kernel_sources", "competition_sources", "model_sources"):
        value = metadata.get(key, [])
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip()
                or any(character.isspace() for character in item) for item in value):
            raise ValueError("existing-notebook-source-list-invalid")
        submission[key] = value
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "week1.py").write_text(code, encoding="utf-8", newline="\n")
    (output_dir / "kernel-metadata.json").write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8", newline="\n")
    workload_label = {"week1": "D6-D7-week1", "driving": "D8-D10-privileged-driving", "baseline": "camera-only-opencv-baseline"}[workload]
    bundle = {"notebook_id": submission["id"], "title": submission["title"], "workload": workload_label,
              "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(), "source_sha256": manifest,
              "submitted": False, "creates_new_notebook": False}
    (output_dir / "bundle_manifest.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8", newline="\n")
    return bundle


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-metadata", type=Path, required=True, help="Metadata obtained by a read-only pull of the existing notebook.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = prepare(args.existing_metadata, args.output_dir)
        print(json.dumps({key: report[key] for key in ("notebook_id", "workload", "code_sha256", "submitted", "creates_new_notebook")}, indent=2))
        return 0
    except (OSError, ValueError, StopIteration) as error:
        print(json.dumps({"status": "FAIL", "code": "week1-package-preparation-failed", "exception_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
