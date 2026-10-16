"""Package profiled development driving for the existing private CARLA notebook."""
from __future__ import annotations

import base64
import hashlib
import json
import lzma
from pathlib import Path

from scripts.prepare_kaggle_failure_driving import prepare_failure_driving
from scripts.prepare_kaggle_refinement_driving import ROOT

EXTRA_SOURCES = ("carla_tools/runtime_driving.py", "carla_tools/pipeline_latency.py",
    "carla_tools/path_correction.py", "carla_tools/driving_demo.py", "scripts/run_runtime_pipeline.py")


def prepare_runtime_driving(metadata_path, output_dir, **kwargs):
    """Preserve notebook identity; submission performs a fresh access check."""
    bundle = prepare_failure_driving(metadata_path, output_dir, **kwargs)
    output = Path(output_dir)
    code_path = output / "native.py"
    code = code_path.read_text(encoding="utf-8")
    replacements = {
        'str(project / "scripts/run_failure_pipeline.py")': 'str(project / "scripts/run_runtime_pipeline.py")',
        '["privileged_reference", CANDIDATE_VARIANT, CANDIDATE_VARIANT + "_near"]':
            '["privileged_reference", CANDIDATE_VARIANT + "_near", CANDIDATE_VARIANT + "_path_near"]',
    }
    for old, new in replacements.items():
        if code.count(old) != 1:
            raise ValueError("runtime-bootstrap-marker-mismatch")
        code = code.replace(old, new, 1)
    sources = {name: (ROOT / name).read_bytes().decode("utf-8") for name in EXTRA_SOURCES}
    for name, source in sources.items():
        compile(source, name, "exec")
    payload = base64.b64encode(lzma.compress(json.dumps(sources, sort_keys=True,
        separators=(",", ":")).encode(), preset=9)).decode()
    marker = "raise SystemExit(main())\n"
    if code.count(marker) != 1:
        raise ValueError("runtime-bootstrap-main-marker-mismatch")
    code = code.replace(marker, "\nPROJECT_SOURCES.update(json.loads(lzma.decompress(base64.b64decode(" +
        repr(payload) + ",validate=True))))\n" + marker)
    compile(code, "native.py", "exec")
    code_path.write_bytes(code.encode())
    bundle.update(scope="profiled-paired-development-driving", code_sha256=hashlib.sha256(code.encode()).hexdigest(),
        evaluated_methods=["privileged_reference", "hybrid_geometry_near", "hybrid_geometry_path_near"],
        expected_episodes=81, latency_warmup_frames_per_episode=10, processing_budget_ms=50.)
    bundle["source_sha256"].update({name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()})
    (output / "bundle_manifest.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    return bundle
