"""Package a paired remediation experiment for the existing CARLA notebook."""
from __future__ import annotations

import base64
import hashlib
import json
import lzma
from pathlib import Path

from scripts.prepare_kaggle_refinement_driving import prepare_native, ROOT

EXTRA_SOURCES = ("carla_tools/near_correction.py", "carla_tools/failure_driving.py", "carla_tools/failure_analysis.py",
                 "carla_tools/driving_quality.py", "scripts/run_failure_pipeline.py")


def prepare_failure_driving(metadata_path, output_dir, **kwargs):
    if kwargs.get("candidate_variant", "hybrid_geometry") != "hybrid_geometry":
        raise ValueError("failure-geometric-candidate-required")
    bundle = prepare_native(metadata_path, output_dir, **kwargs)
    output = Path(output_dir)
    code_path = output / "native.py"
    code = code_path.read_text(encoding="utf-8")
    replacements = {
        'str(project / "scripts/run_refinement_pipeline.py")': 'str(project / "scripts/run_failure_pipeline.py")',
        '["B0", "resnet", CANDIDATE_VARIANT]': '["privileged_reference", CANDIDATE_VARIANT, CANDIDATE_VARIANT + "_near"]',
        '[REFERENCE_CHECKPOINT_SHA256, CHECKPOINT_SHA256]': '[CHECKPOINT_SHA256, CHECKPOINT_SHA256]',
    }
    for old, new in replacements.items():
        if code.count(old) != 1:
            raise ValueError("failure-bootstrap-marker-mismatch")
        code = code.replace(old, new, 1)
    sources = {name: (ROOT / name).read_bytes().decode("utf-8") for name in EXTRA_SOURCES}
    for name, source in sources.items():
        compile(source, name, "exec")
    encoded = base64.b64encode(lzma.compress(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode(), preset=9)).decode()
    addition = "\nPROJECT_SOURCES.update(json.loads(lzma.decompress(base64.b64decode(" + repr(encoded) + ",validate=True))))\n"
    marker = "raise SystemExit(main())\n"
    if code.count(marker) != 1:
        raise ValueError("failure-bootstrap-main-marker-mismatch")
    code = code.replace(marker, addition + marker)
    compile(code, "native.py", "exec")
    code_path.write_bytes(code.encode())
    bundle.update(scope="paired-development-failure-remediation", code_sha256=hashlib.sha256(code.encode()).hexdigest(),
        evaluated_methods=["privileged_reference", "hybrid_geometry", "hybrid_geometry_near"], expected_episodes=81)
    bundle["source_sha256"].update({name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()})
    (output / "bundle_manifest.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    return bundle
