"""Package paired recovery driving for the existing private CARLA notebook.

Preparation is local and never submits a notebook or starts a GPU. Submission
must first repeat a read-only existence and access check for the fixed ID.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath
import re

from scripts.kaggle_learned_bootstrap import render_bootstrap
from scripts.prepare_kaggle_refinement_driving import ROOT, SOURCES, TRAINING_NOTEBOOK
from scripts.prepare_kaggle_week1 import NOTEBOOK_ID, NOTEBOOK_TITLE


def unpack_recovery_archive(archive, destination, *, expected_archive_sha256,
        checkpoint_sha256, evaluation_sha256, comparison_sha256, member_prefix="experiment"):
    """Extract only five pinned checkpoints, evaluations, and their comparison."""
    import zipfile
    archive, destination = Path(archive), Path(destination)
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_archive_sha256:
        raise ValueError("recovery-native-training-archive-hash-mismatch")
    if (destination.exists() or destination.is_symlink()
            or not member_prefix or "\\" in member_prefix
            or PurePosixPath(member_prefix).is_absolute()
            or any(p in (".", "..") for p in member_prefix.split("/"))):
        raise ValueError("recovery-native-new-safe-extraction-required")
    arms = ("incumbent", *(f"{arm}_seed{seed}" for seed in (42, 43)
        for arm in ("central_only", "central_recovery")))
    if set(checkpoint_sha256) != set(arms) or set(evaluation_sha256) != set(arms):
        raise ValueError("recovery-native-all-seed-model-bindings-required")
    def directories(arm):
        if arm == "incumbent":
            return "initial", "initial_evaluation"
        training_arm, seed = arm.rsplit("_seed", 1)
        directory = f"seed{seed}/{training_arm}"
        return directory, directory + "/evaluation"
    selected = {"comparison.json": (comparison_sha256, 8 * 1024**2)}
    for arm in arms:
        directory, evaluation = directories(arm)
        selected[directory + "/checkpoint.pt"] = (checkpoint_sha256[arm], 128 * 1024**2)
        selected[evaluation + "/metrics.json"] = (evaluation_sha256[arm], 8 * 1024**2)
    payload = {}
    with zipfile.ZipFile(archive) as stream:
        names = stream.namelist()
        if len(names) != len(set(names)):
            raise ValueError("recovery-native-archive-duplicate-members")
        for relative, (expected, maximum) in selected.items():
            member = stream.getinfo(member_prefix.rstrip("/") + "/" + relative)
            if (not 0 < member.file_size <= maximum or member.flag_bits & 1
                    or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("recovery-native-archive-member-invalid")
            value = stream.read(member)
            if hashlib.sha256(value).hexdigest() != expected:
                raise ValueError("recovery-native-selected-artifact-hash-mismatch")
            payload[relative] = value
    comparison = json.loads(payload["comparison.json"])
    if comparison.get("status") != "PASS" or comparison.get("final_town_used") is not False:
        raise ValueError("recovery-native-training-comparison-invalid")
    destination.mkdir(parents=True, exist_ok=False)
    for relative, value in payload.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    models = {}
    for arm in arms:
        directory, evaluation = directories(arm)
        models[arm] = {"path": str(destination / directory / "checkpoint.pt"),
            "sha256": checkpoint_sha256[arm],
            "evaluation": str(destination / evaluation / "metrics.json"),
            "evaluation_sha256": evaluation_sha256[arm]}
    models_file = destination / "models.json"
    models_file.write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
    return models_file


def prepare_recovery_driving(existing_metadata, output_dir, *, dev_routes, protocol_file,
        training_archive_sha256, training_archive_bytes, training_version, checkpoint_sha256,
        evaluation_sha256, comparison_sha256, representative_seed,
        archive_name="training_artifacts.zip",
        member_prefix="experiment"):
    from carla_tools.recovery_comparison import ARMS, comparison_source_hashes, freeze_driving_protocol
    from carla_tools.runtime_driving import RUNTIME_SOURCES
    from scripts.kaggle_recovery_training_bootstrap import cloud_download_helper_source
    metadata = json.loads(Path(existing_metadata).read_text(encoding="utf-8"))
    output = Path(output_dir)
    if (metadata.get("id") != NOTEBOOK_ID or metadata.get("title") != NOTEBOOK_TITLE
            or metadata.get("is_private") is not True):
        raise ValueError("recovery-native-existing-private-notebook-required")
    if output.exists() or output.is_symlink():
        raise ValueError("recovery-native-new-local-bundle-required")
    if (type(training_version) is not int or training_version <= 0
            or type(training_archive_bytes) is not int or not 0 < training_archive_bytes <= 32 * 1024**3
            or type(archive_name) is not str or not re.fullmatch("[A-Za-z0-9_-]+\\.zip", archive_name)
            or not member_prefix or "\\" in member_prefix
            or PurePosixPath(member_prefix).is_absolute()
            or any(p in (".", "..") for p in member_prefix.split("/"))
            or set(checkpoint_sha256) != set(ARMS) or set(evaluation_sha256) != set(ARMS)
            or any(not re.fullmatch("[0-9a-f]{64}", str(h)) for h in
                (training_archive_sha256, comparison_sha256, *checkpoint_sha256.values(), *evaluation_sha256.values()))):
        raise ValueError("recovery-native-pinned-training-transport-required")
    routes_bytes, protocol_bytes = Path(dev_routes).read_bytes(), Path(protocol_file).read_bytes()
    routes, protocol = json.loads(routes_bytes), json.loads(protocol_bytes)
    if protocol != freeze_driving_protocol(routes, checkpoint_sha256, comparison_sha256,
            representative_seed=representative_seed):
        raise ValueError("recovery-native-frozen-protocol-mismatch")
    names = tuple(dict.fromkeys((*SOURCES, *RUNTIME_SOURCES, *comparison_source_hashes())))
    sources = {name: (ROOT / name).read_bytes().decode("utf-8") for name in names}
    for name, value in sources.items():
        if name.endswith(".py"):
            compile(value, name, "exec")
    bootstrap = render_bootstrap((ROOT / "scripts/kaggle_carla_bootstrap.py").read_text(encoding="utf-8"), maximum_seconds=7200)
    old = '        selected_checkpoint = resolve_selected_checkpoint(Path("/kaggle/input"), CHECKPOINT_SHA256, CHECKPOINT_BYTES)\n'
    if bootstrap.count(old) != 1:
        raise ValueError("recovery-native-checkpoint-marker-mismatch")
    bootstrap = bootstrap.replace(old, '        selected_checkpoint = None\n', 1)
    bootstrap = bootstrap.replace('        report["checkpoint_sha256"] = CHECKPOINT_SHA256\n',
        '        report["checkpoint_sha256"] = CHECKPOINTS\n', 1)
    bootstrap = bootstrap.replace('        report["checkpoint_bytes"] = CHECKPOINT_BYTES\n', '', 1)
    marker = '        stage("server-download")\n'
    if bootstrap.count(marker) != 1:
        raise ValueError("recovery-native-transport-marker-mismatch")
    setup = '''        stage("pinned-recovery-training-artifacts")
        dependencies = scratch / "kagglehub_tools"
        command([uv, "pip", "install", "--python", python, "--only-binary", ":all:",
                 "--target", str(dependencies), "kagglehub==1.0.2"], "carla_recovery_transport_dependencies", 180, env)
        transport_helper = TRAINING_DOWNLOAD_HELPER.replace(repr("__VLF_RECOVERY_DEPS__"), repr(str(dependencies))).replace(
            repr("__VLF_RECOVERY_DOWNLOAD__"), repr(str(scratch / "training_download")))
        downloaded = command([python, "-c", transport_helper], "carla_recovery_transport", 1900, env, True)
        value = json.loads(downloaded.stdout.strip().splitlines()[-1])
        if (value.get("status") != "PASS" or value.get("sha256") != TRAINING_ARCHIVE_SHA256
                or value.get("bytes") != TRAINING_ARCHIVE_BYTES):
            raise BootstrapFailure("recovery-native-pinned-training-transport-invalid")
        models_file = unpack_recovery_archive(value["path"], scratch / "trained",
            expected_archive_sha256=TRAINING_ARCHIVE_SHA256, checkpoint_sha256=CHECKPOINTS,
            evaluation_sha256=EVALUATIONS, comparison_sha256=COMPARISON_SHA256, member_prefix=MEMBER_PREFIX)
        stage("server-download")
'''
    bootstrap = bootstrap.replace(marker, setup, 1)
    start = bootstrap.find('        if WORKLOAD == "learned":\n            stage("selected-rgb-model-native-driving-and-replay")')
    end = bootstrap.find('        else:\n            stage("rgb-probe")', start)
    if start < 0 or end < 0:
        raise ValueError("recovery-native-workload-marker-mismatch")
    pipeline = '''        if WORKLOAD == "learned":
            stage("paired-recovery-native-driving-and-replay")
            project = scratch / "project"
            project.mkdir()
            for relative, source in PROJECT_SOURCES.items():
                path = project / relative
                if not path.resolve().is_relative_to(project.resolve()):
                    raise BootstrapFailure("recovery-embedded-project-path-invalid")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.encode("utf-8"))
            report["learned_runtime_sources"] = export_baseline_runtime_sources(project, output, PROJECT_SOURCES, artifact="learned")
            routes_file, protocol_file = project / "frozen_dev_routes.json", project / "frozen_driving_protocol.json"
            routes_file.write_bytes(DEV_ROUTES_BYTES)
            protocol_file.write_bytes(PROTOCOL_BYTES)
            command([python, str(project / "scripts/run_recovery_driving.py"), "--routes", str(routes_file),
                     "--protocol", str(protocol_file), "--models", str(models_file),
                     "--training-comparison", str(scratch / "trained/comparison.json"),
                     "--device", "cuda", "--output-dir", str(output / "native")],
                    "carla_recovery", MAX_SECONDS - 900, env)
            gate = json.loads((output / "native/native_report.json").read_text())
            if (gate.get("status") != "PASS" or gate.get("episode_count") != 135
                    or gate.get("closed_loop_executed") is not True or gate.get("frame_count", 0) <= 0
                    or gate.get("final_town_used") is not False or gate.get("checkpoint_sha256") != CHECKPOINTS
                    or gate.get("protocol_sha256") != PROTOCOL_SHA256
                    or set(gate.get("models", {})) != set(CHECKPOINTS)
                    or any(m.get("device") != "cuda" for m in gate["models"].values())
                    or not all(e.get("gate", {}).get("status") == "PASS" and e.get("media", {}).get("status") == "PASS" for e in gate["episodes"])
                    or gate.get("selection", {}).get("status") != "PASS" or gate.get("pointer_updated") is not False):
                raise BootstrapFailure("recovery-native-integrity-proof-invalid")
            _assert_runtime_sources(project, PROJECT_SOURCES)
            for item in json.loads(models_file.read_text()).values():
                if _file_sha256(Path(item["path"])) != item["sha256"]:
                    raise BootstrapFailure("recovery-native-weights-changed")
            report["learned_verified"] = True
            report["runtime_sources_unchanged"] = True
            report["learned"] = {key: gate[key] for key in ("status", "episode_count", "frame_count", "selection")}
'''
    bootstrap = bootstrap[:start] + pipeline + bootstrap[end:]
    bootstrap = bootstrap.replace('preserve_baseline_failure_artifacts(output, artifact="learned")',
        'preserve_baseline_failure_artifacts(output, artifact="native")')
    bootstrap = bootstrap.replace('artifact not in ("baseline", "pilot", "learned")',
        'artifact not in ("baseline", "pilot", "learned", "native")')
    own = Path(__file__).read_text(encoding="utf-8")
    node = next(n for n in ast.parse(own).body if isinstance(n, ast.FunctionDef) and n.name == "unpack_recovery_archive")
    bootstrap += "\nfrom pathlib import PurePosixPath\n" + ast.get_source_segment(own, node) + "\n"
    vulkan = (ROOT / "scripts/kaggle_vulkan_probe.py").read_text(encoding="utf-8")
    function = next(n for n in ast.parse(vulkan).body if isinstance(n, ast.FunctionDef) and n.name == "vulkan_devices")
    child = "import ctypes,ctypes.util,struct,json\n" + ast.get_source_segment(vulkan, function) + "\nprint(json.dumps(vulkan_devices()))\n"
    code = bootstrap + "\nPROBE_SOURCE=" + repr(sources["scripts/carla_rgb_probe.py"]) + "\nVULKAN_CHILD_SOURCE=" + repr(child)
    for name, payload in (("PROJECT_SOURCES", json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()),
            ("DEV_ROUTES_BYTES", routes_bytes), ("PROTOCOL_BYTES", protocol_bytes)):
        code += "\nimport base64,lzma\n" + name + "=lzma.decompress(base64.b64decode(" + repr(base64.b64encode(lzma.compress(payload, preset=9)).decode()) + ",validate=True))\n"
        if name == "PROJECT_SOURCES":
            code += "PROJECT_SOURCES=json.loads(PROJECT_SOURCES)\n"
    from carla_tools.recovery_comparison import digest
    download_helper = cloud_download_helper_source(training_version, Path("__VLF_RECOVERY_DEPS__"),
        Path("__VLF_RECOVERY_DOWNLOAD__"), notebook_id=TRAINING_NOTEBOOK, filename=archive_name,
        expected_sha256=training_archive_sha256, expected_bytes=training_archive_bytes, maximum_seconds=1800)
    compile(download_helper, "pinned-recovery-training-transport", "exec")
    values = {"CHECKPOINT_SHA256": checkpoint_sha256[f"central_recovery_seed{representative_seed}"], "CHECKPOINT_BYTES": 1,
        "LEARNED_MODE": "paired-recovery", "TRAINING_HANDLE": TRAINING_NOTEBOOK + "/versions/" + str(training_version),
        "TRAINING_VERSION": training_version, "TRAINING_ARCHIVE_NAME": archive_name,
        "TRAINING_ARCHIVE_SHA256": training_archive_sha256, "TRAINING_ARCHIVE_BYTES": training_archive_bytes,
        "TRAINING_DOWNLOAD_HELPER": download_helper, "CHECKPOINTS": checkpoint_sha256,
        "EVALUATIONS": evaluation_sha256, "COMPARISON_SHA256": comparison_sha256,
        "MEMBER_PREFIX": member_prefix, "PROTOCOL_SHA256": digest(protocol)}
    code += "\n" + "\n".join(name + "=" + repr(value) for name, value in values.items()) + "\nraise SystemExit(main())\n"
    compile(code, "native.py", "exec")
    if len(code.encode()) >= 1_000_000:
        raise ValueError("recovery-native-source-size-exceeded")
    submission = {"id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE, "code_file": "native.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4"}
    for name in ("dataset_sources", "kernel_sources", "competition_sources", "model_sources"):
        submission[name] = list(metadata.get(name, []))
    submission["kernel_sources"] = [n for n in submission["kernel_sources"] if n != TRAINING_NOTEBOOK]
    bundle = {"status": "PREPARED", "scope": "paired-recovery-development-driving",
        "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "source_sha256": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()},
        "protocol_sha256": digest(protocol), "protocol_file_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "training_version": training_version, "training_archive_sha256": training_archive_sha256,
        "training_archive_bytes": training_archive_bytes,
        "transport": "official-versioned-response-with-bounded-resumable-HTTP-ranges",
        "training_download_helper_sha256": hashlib.sha256(download_helper.encode()).hexdigest(),
        "training_comparison_sha256": comparison_sha256, "checkpoint_sha256": checkpoint_sha256,
        "expected_episodes": 135, "maximum_seconds": 7200,
        "creates_new_notebook": False, "creates_dataset": False, "local_gpu_used": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "native.py").write_bytes(code.encode())
    for name, value in (("bundle_manifest.json", bundle), ("kernel-metadata.json", submission)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return bundle
