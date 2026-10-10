"""Prepare refinements for the existing private T4 notebook, without uploading."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

from scripts.prepare_kaggle_spatial import SOURCES, prepare
from scripts.kaggle_training_bootstrap import TrainingBootstrapError, PILOT_MANIFEST_SHA256

ROOT = Path(__file__).resolve().parents[1]
EXTRA_SOURCES = ("carla_tools/refined_waypoint.py", "carla_tools/refined_training.py", "scripts/train_refined_waypoints.py")
SELECTION_SOURCES = ("carla_tools/model_selection.py", "configs/model_selection.json", "scripts/select_waypoint_model.py")


def validate_refinement_report(report, expected_train, expected_dev):
    if (report.get("status") != "PASS" or report.get("fixture_injection") is not False
            or report.get("final_town_used") is not False or report.get("decision", {}).get("closed_loop_verified") is not False
            or set(report.get("training", {})) != {"hybrid_mirror", "hybrid_geometry"}
            or set(report.get("evaluations", {})) != {"hybrid_mirror", "hybrid_geometry"}):
        raise TrainingBootstrapError("refinement-two-arm-proof-missing")
    sources = set()
    for variant in ("hybrid_mirror", "hybrid_geometry"):
        train, dev = report["training"][variant], report["evaluations"][variant]
        if (train.get("status") != "PASS" or dev.get("status") != "PASS"
                or train.get("variant") != variant or dev.get("variant") != variant
                or train.get("train_sample_count") != expected_train or train.get("dev_sample_count") != expected_dev
                or dev.get("sample_count") != expected_dev or dev.get("split") != "dev"
                or train.get("runtime", {}).get("device") != "cuda" or "T4" not in str(train["runtime"].get("gpu", ""))
                or train["runtime"].get("amp_enabled") is not True
                or train.get("overfit_check", {}).get("status") != "PASS"
                or train["overfit_check"].get("model_reset_to_fresh_initialization") is not True
                or train.get("manifest_sha256") != PILOT_MANIFEST_SHA256 or dev.get("manifest_sha256") != PILOT_MANIFEST_SHA256
                or train.get("checkpoint_sha256") != dev.get("checkpoint_sha256") or not train.get("checkpoint_sha256")
                or train.get("source_sha256") != dev.get("source_sha256")
                or train.get("config_sha256") != dev.get("config_sha256")
                or train.get("pretrained_provenance", {}).get("hash_verified") is not True
                or train.get("fixture_injection") is not False or dev.get("fixture_injection") is not False
                or train.get("final_town_used") is not False or dev.get("final_town_used") is not False):
            raise TrainingBootstrapError("refinement-training-or-full-dev-proof-invalid")
        sources.add(json.dumps(train["source_sha256"], sort_keys=True))
    if len(sources) != 1:
        raise TrainingBootstrapError("refinement-source-closure-mismatch")
    return {"arms_verified": 2, "dev_samples_per_arm": expected_dev,
            "decision": report.get("current_selection", report["decision"]), "actual_t4_training": True}


def prepare_refinement(existing_metadata, output_dir, *, reference_path,
                       pilot_archive_sha256, pilot_manifest_sha256, pilot_download_version):
    # Reuse the audited existing-notebook identity and time-bound validation.
    output = Path(output_dir)
    bundle = prepare(existing_metadata, output, pilot_archive_sha256=pilot_archive_sha256,
        pilot_manifest_sha256=pilot_manifest_sha256, pilot_download_version=pilot_download_version,
        reused_baseline_path=reference_path)
    sources = {name: (ROOT / name).read_text(encoding="utf-8") for name in (*SOURCES, *EXTRA_SOURCES, *SELECTION_SOURCES)}
    for name, value in sources.items():
        if name.endswith(".py"):
            compile(value, name, "exec")
    # Preserve actual byte hashes by reading bytes, including any CRLF sources.
    sources = {name: (ROOT / name).read_bytes().decode("utf-8") for name in sources}
    payload = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    bootstrap = (ROOT / "scripts/kaggle_spatial_bootstrap.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    old = 'str(project / "scripts/train_spatial_waypoints.py")'
    if bootstrap.count(old) != 1:
        raise ValueError("refinement-bootstrap-training-marker-mismatch")
    bootstrap = bootstrap.replace(old, 'str(project / "scripts/train_refined_waypoints.py")')
    bootstrap = bootstrap.replace('validate_spatial_report(comparison,', 'validate_refinement_report(comparison,')
    bootstrap = bootstrap.replace('actual-t4-three-arm-training-and-dev', 'actual-t4-refinement-training-and-dev')
    bootstrap = bootstrap.replace('"workload": "controlled-spatial-hybrid-comparison"', '"workload": "mirrored-and-geometric-hybrid-refinement"')
    marker = '        comparison = json.loads((output / "experiment/comparison.json").read_text())\n'
    if bootstrap.count(marker) != 1:
        raise ValueError('refinement-bootstrap-selection-marker-mismatch')
    bootstrap = bootstrap.replace(marker, marker + '''        command([python, str(project / "scripts/select_waypoint_model.py"),
                 "--reference", str(baseline_path), "--comparison", str(output / "experiment/comparison.json"),
                 "--output", str(output / "experiment/selection.json")], "spatial_selection", 60, env)
        comparison["current_selection"] = json.loads((output / "experiment/selection.json").read_text())["decision"]
''',1)
    source = Path(__file__).read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "validate_refinement_report")
    bootstrap += "\n" + ast.get_source_segment(source, node) + "\n"
    frozen = Path(reference_path).read_text(encoding="utf-8")
    code = ((ROOT / "scripts/kaggle_training_bootstrap.py").read_bytes().decode() + "\n" + bootstrap
        + "\nimport base64,zlib\n_payload=zlib.decompress(base64.b64decode(" + repr(base64.b64encode(zlib.compress(payload, 9)).decode()) + ",validate=True))\n"
        + "if hashlib.sha256(_payload).hexdigest()!=" + repr(hashlib.sha256(payload).hexdigest()) + ": raise SystemExit('project-payload-mismatch')\n"
        + "PROJECT_SOURCES=json.loads(_payload.decode())\nPILOT_ARCHIVE_SHA256=" + repr(pilot_archive_sha256)
        + "\nPILOT_MANIFEST_SHA256=" + repr(pilot_manifest_sha256) + "\nPILOT_DOWNLOAD_VERSION=" + repr(pilot_download_version)
        + "\nFROZEN_BASELINE=" + repr(frozen) + "\nraise SystemExit(main_spatial())\n").replace("\r\n", "\n")
    compile(code, "spatial_training.py", "exec")
    if len(code.encode()) >= 1_000_000:
        raise ValueError("refinement-source-size-exceeded")
    bundle.update(workload="mirrored-and-geometric-hybrid-refinement",
        source_sha256={name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()},
        code_sha256=hashlib.sha256(code.encode()).hexdigest(), project_payload_sha256=hashlib.sha256(payload).hexdigest(),
        refinement_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / "spatial_training.py").write_bytes(code.encode())
    (output / "bundle_manifest.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    return bundle
