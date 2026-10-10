"""Render a bounded learned-driving variant of the existing CARLA bootstrap.

The original bootstrap stays unchanged. Every adaptation requires an exact
source marker; an unexpected upstream change fails before a bundle is written.
Importing this module does not install dependencies or start a simulator.
"""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path


class LearnedBootstrapError(ValueError):
    pass


def _file_sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def resolve_selected_checkpoint(input_root: Path, expected_sha256: str, expected_bytes: int) -> Path:
    """Find the exact attached checkpoint before downloading or starting CARLA."""
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise LearnedBootstrapError("checkpoint-sha256-invalid")
    if type(expected_bytes) is not int or not 0 < expected_bytes <= 2 * 1024 ** 3:
        raise LearnedBootstrapError("checkpoint-byte-size-invalid")
    root = input_root.resolve()
    matches = []
    candidates = list(input_root.rglob("checkpoint.pt"))
    if len(candidates) > 32:
        raise LearnedBootstrapError("checkpoint-input-count-exceeded")
    for path in candidates:
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root)
                or any(parent.is_symlink() for parent in path.parents) or path.stat().st_size != expected_bytes):
            continue
        if _file_sha256(path) == expected_sha256:
            matches.append(path)
    if len(matches) != 1:
        raise LearnedBootstrapError("exact-selected-checkpoint-not-uniquely-mounted")
    return matches[0]


def validate_native_report(report: dict, manifest: dict, checkpoint_sha256: str, mode: str) -> None:
    """Certify execution integrity while retaining unsuccessful driving metrics."""
    if (not isinstance(report, dict) or report.get("status") != "PASS"
            or report.get("gate", {}).get("status") != "PASS"
            or type(report.get("episode_count")) is not int or report["episode_count"] <= 0
            or type(report.get("frame_count")) is not int or report["frame_count"] <= 0):
        raise LearnedBootstrapError("learned-native-artifact-gate-not-pass")
    model = manifest.get("model", {}) if isinstance(manifest, dict) else {}
    if (not isinstance(manifest, dict) or manifest.get("status") != "COMPLETE"
            or manifest.get("mode") != mode or model.get("checkpoint_role") != "selected-model"
            or model.get("checkpoint_sha256") != checkpoint_sha256
            or model.get("runtime", {}).get("device") != "cuda"):
        raise LearnedBootstrapError("learned-native-selected-cuda-model-mismatch")


def _assert_runtime_sources(project: Path, sources: dict) -> None:
    for relative, source in sources.items():
        path = project / relative
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(project.resolve())
                or path.read_bytes() != source.encode("utf-8")):
            raise LearnedBootstrapError("learned-runtime-source-changed")


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise LearnedBootstrapError("native-bootstrap-source-marker-mismatch")
    return source.replace(old, new, 1)


def render_bootstrap(base_source: str, *, maximum_seconds: int = 3600) -> str:
    """Keep audited native setup/cleanup and replace only the learned workload."""
    if type(maximum_seconds) is not int or not 1800 <= maximum_seconds <= 7200:
        raise LearnedBootstrapError("learned-bootstrap-time-budget-invalid")
    # Normalizing the generated executable does not modify the original file.
    source = base_source.replace("\r\n", "\n")
    source = _replace_once(source, 'WORKLOAD = "rgb"', 'WORKLOAD = "learned"')
    source = _replace_once(source, 'MAX_SECONDS = 1200', f'MAX_SECONDS = {maximum_seconds}')
    source = _replace_once(source,
        'WORKLOAD not in ("rgb", "week1", "driving", "baseline", "pilot")',
        'WORKLOAD not in ("rgb", "week1", "driving", "baseline", "pilot", "learned")')
    source = source.replace('WORKLOAD in ("baseline", "pilot")', 'WORKLOAD in ("baseline", "pilot", "learned")')
    source = source.replace('WORKLOAD in ("week1", "driving", "baseline", "pilot")',
                            'WORKLOAD in ("week1", "driving", "baseline", "pilot", "learned")')
    source = source.replace('artifact not in ("baseline", "pilot")', 'artifact not in ("baseline", "pilot", "learned")')
    source = _replace_once(source, '        report["workload"] = WORKLOAD\n',
        '        report["workload"] = WORKLOAD\n'
        '        selected_checkpoint = resolve_selected_checkpoint(Path("/kaggle/input"), CHECKPOINT_SHA256, CHECKPOINT_BYTES)\n'
        '        report["checkpoint_sha256"] = CHECKPOINT_SHA256\n'
        '        report["checkpoint_bytes"] = CHECKPOINT_BYTES\n'
        '        report["learned_mode"] = LEARNED_MODE\n')
    source = _replace_once(source, '        stage("server-download")\n',
        '        stage("learned-cuda-runtime-install")\n'
        '        command([uv, "pip", "install", "--python", python, "--only-binary", ":all:",\n'
        '                 "--index-url", "https://download.pytorch.org/whl/cu126",\n'
        '                 "torch==2.7.1", "torchvision==0.22.1"], "carla_learned_torch", 600, env)\n'
        '        command([uv, "pip", "install", "--python", python, "--only-binary", ":all:",\n'
        '                 "Pillow==11.3.0"], "carla_learned_pillow", 180, env)\n'
        '        cuda_smoke = ("import torch,torchvision,json; "\n'
        '                      "assert torch.__version__==\'2.7.1+cu126\'; "\n'
        '                      "assert torchvision.__version__==\'0.22.1+cu126\'; "\n'
        '                      "assert torch.cuda.is_available(); "\n'
        '                      "x=torch.ones(16,device=\'cuda\'); assert float((x*x).sum())==16.; "\n'
        '                      "torch.cuda.synchronize(); print(json.dumps({\'device\':\'cuda\',"\n'
        '                      "\'torch\':torch.__version__,\'torchvision\':torchvision.__version__,"\n'
        '                      "\'cuda\':torch.version.cuda,\'gpu_name\':torch.cuda.get_device_name(0)}))")\n'
        '        cuda_result = command([python, "-c", cuda_smoke], "carla_learned_cuda_smoke", 60, env, True)\n'
        '        report["learned_runtime"] = json.loads(cuda_result.stdout.strip().splitlines()[-1])\n'
        '        stage("server-download")\n')
    old_start = '        if WORKLOAD in ("week1", "driving", "baseline", "pilot", "learned"):\n            driving = WORKLOAD == "driving"'
    start = source.find(old_start)
    end = source.find('        else:\n            stage("rgb-probe")', start)
    if start < 0 or end < 0 or source.count(old_start) != 1:
        raise LearnedBootstrapError("native-bootstrap-pipeline-block-mismatch")
    source = source[:start] + '''        if WORKLOAD == "learned":
            stage("selected-rgb-model-native-driving-and-replay")
            project = scratch / "project"
            project.mkdir()
            if not PROJECT_SOURCES:
                raise BootstrapFailure("embedded-project-required")
            for relative, source in PROJECT_SOURCES.items():
                path = project / relative
                if not path.resolve().is_relative_to(project.resolve()):
                    raise BootstrapFailure("embedded-project-path-invalid")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.encode("utf-8"))
            report["learned_runtime_sources"] = export_baseline_runtime_sources(
                project, output, PROJECT_SOURCES, artifact="learned")
            command([python, str(project / "scripts/run_learned_pipeline.py"),
                     "--checkpoint", str(selected_checkpoint), "--device", "cuda",
                     "--mode", LEARNED_MODE, "--output-dir", str(output / "learned")],
                    "carla_learned", MAX_SECONDS - 600, env)
            gate = json.loads((output / "learned/learned_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "learned/learned_manifest.json").read_text(encoding="utf-8"))
            validate_native_report(gate, manifest, CHECKPOINT_SHA256, LEARNED_MODE)
            _assert_runtime_sources(project, PROJECT_SOURCES)
            if _file_sha256(selected_checkpoint) != CHECKPOINT_SHA256:
                raise LearnedBootstrapError("selected-checkpoint-changed-during-native-run")
            report["learned_verified"] = True
            report["runtime_sources_unchanged"] = True
            report["learned"] = {key: gate[key] for key in ("status", "episode_count", "frame_count", "gate")}
''' + source[end:]
    source = _replace_once(source, '        if WORKLOAD == "baseline":\n',
        '        if WORKLOAD == "learned":\n'
        '            try:\n'
        '                preserve_baseline_failure_artifacts(output, artifact="learned")\n'
        '            except Exception as error:\n'
        '                report["artifact_cleanup_error"] = type(error).__name__\n'
        '                report["status"] = "FAIL"\n'
        '        if WORKLOAD == "baseline":\n')
    source = source.replace('isinstance(error, BootstrapFailure)',
                            'isinstance(error, (BootstrapFailure, LearnedBootstrapError))')
    source = _replace_once(source, 'output / "kaggle_carla_bootstrap.json"',
                           'output / "kaggle_learned_bootstrap.json"')
    # Helpers remain normal Python source and are present before main is called.
    module_source = Path(__file__).read_text(encoding="utf-8")
    nodes = ast.parse(module_source).body
    names = {"LearnedBootstrapError", "_file_sha256", "resolve_selected_checkpoint", "validate_native_report", "_assert_runtime_sources"}
    helper_source = "\nimport re\n" + "\n\n".join(ast.get_source_segment(module_source, node)
        for node in nodes if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names)
    rendered = source + helper_source + "\n"
    compile(rendered, "learned.py", "exec")
    return rendered
