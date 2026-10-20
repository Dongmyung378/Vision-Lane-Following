"""Package a source-bound development confirmation in the existing CARLA notebook."""
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


def unpack_final_checkpoints(archive, destination, *, expected_archive_sha256,
        expected_archive_bytes, checkpoint_sha256, offline_checkpoint_sha256, checkpoint_members,
        comparison_sha256, representative_seed, member_prefix="experiment"):
    """Extract the exact audited representative models; retain original evidence."""
    import hashlib
    import json
    from pathlib import Path, PurePosixPath
    import re
    import zipfile
    arms = ("incumbent", "control", "continued", "appearance_robust")
    archive, destination = Path(archive), Path(destination)
    def safe(relative):
        return (type(relative) is str and bool(relative) and "\\" not in relative
            and not PurePosixPath(relative).is_absolute()
            and not any(part in ("", ".", "..") for part in relative.split("/")))
    if (destination.exists() or destination.is_symlink() or archive.is_symlink()
            or type(expected_archive_bytes) is not int or expected_archive_bytes <= 0
            or archive.stat().st_size != expected_archive_bytes or not safe(member_prefix)
            or set(checkpoint_members) != set(arms) or set(offline_checkpoint_sha256) != set(arms)
            or set(checkpoint_sha256) != {*arms, "B0"} or checkpoint_sha256["B0"] is not None
            or checkpoint_sha256["incumbent"] != checkpoint_sha256["control"]
            or checkpoint_members["incumbent"] != checkpoint_members["control"]
            or type(representative_seed) is not int or representative_seed not in (42, 43)
            or any(not safe(member) or PurePosixPath(member).suffix != ".pt" for member in checkpoint_members.values())
            or any(not re.fullmatch("[0-9a-f]{64}", str(value)) for value in
                (expected_archive_sha256, comparison_sha256, *(checkpoint_sha256[arm] for arm in arms),
                    *offline_checkpoint_sha256.values()))):
        raise ValueError("development-final-extraction-bindings-invalid")
    checksum = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    if checksum.hexdigest() != expected_archive_sha256:
        raise ValueError("development-final-training-archive-hash-mismatch")
    selected = {member: (checkpoint_sha256[arm], 128 * 1024**2) for arm, member in checkpoint_members.items()}
    comparison_member = member_prefix.rstrip("/") + "/comparison.json"
    selected[comparison_member] = (comparison_sha256, 16 * 1024**2)
    payload = {}
    with zipfile.ZipFile(archive) as stream:
        names = stream.namelist()
        if len(names) != len(set(names)):
            raise ValueError("development-final-training-archive-duplicate-members")
        for member, (expected, maximum) in selected.items():
            info = stream.getinfo(member)
            if (not 0 < info.file_size <= maximum or info.flag_bits & 1
                    or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("development-final-training-member-invalid")
            value = stream.read(info)
            if hashlib.sha256(value).hexdigest() != expected:
                raise ValueError("development-final-training-member-hash-mismatch")
            payload[member] = value
    comparison = json.loads(payload[comparison_member])
    if (comparison.get("status") != "PASS" or comparison.get("representative_seed") != representative_seed
            or comparison.get("equal_budget_verified") is not True or comparison.get("fixture_injection") is not False
            or comparison.get("final_town_used") is not False
            or comparison.get("initial_checkpoint_sha256") != checkpoint_sha256["incumbent"]
            or offline_checkpoint_sha256["incumbent"] != checkpoint_sha256["incumbent"]
            or offline_checkpoint_sha256["control"] != checkpoint_sha256["control"]
            or any(comparison.get("compatible_checkpoint_sha256", {}).get(arm) != checkpoint_sha256[arm]
                or comparison.get("representative_checkpoint_sha256", {}).get(arm) != offline_checkpoint_sha256[arm]
                for arm in ("continued", "appearance_robust"))):
        raise ValueError("development-final-representative-model-lineage-mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    paths = {}
    for index, member in enumerate(dict.fromkeys(checkpoint_members.values())):
        path = destination / f"model_{index}.pt"
        path.write_bytes(payload[member])
        paths[member] = path
    models = {arm: {"path": str(paths[checkpoint_members[arm]]), "sha256": checkpoint_sha256[arm]} for arm in arms}
    models_file = destination / "models.json"
    models_file.write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
    (destination / "comparison.json").write_bytes(payload[comparison_member])
    return models_file


def prepare_development_final(existing_metadata, output_dir, *, dev_routes, protocol_file,
        training_archive_sha256, training_archive_bytes, training_version, checkpoint_members,
        archive_name="training_artifacts.zip", member_prefix="experiment"):
    from carla_tools.development_final_driving import final_source_hashes, freeze_final_protocol
    from carla_tools.development_study import digest
    from carla_tools.runtime_driving import RUNTIME_SOURCES
    from scripts.kaggle_recovery_training_bootstrap import cloud_download_helper_source
    metadata = json.loads(Path(existing_metadata).read_text(encoding="utf-8"))
    output = Path(output_dir)
    if (metadata.get("id") != NOTEBOOK_ID or metadata.get("title") != NOTEBOOK_TITLE
            or metadata.get("is_private") is not True):
        raise ValueError("development-final-existing-private-notebook-required")
    if output.exists() or output.is_symlink():
        raise ValueError("development-final-new-local-bundle-required")
    if (type(training_version) is not int or training_version <= 0
            or type(training_archive_bytes) is not int or not 0 < training_archive_bytes <= 32 * 1024**3
            or not re.fullmatch("[0-9a-f]{64}", str(training_archive_sha256))
            or type(archive_name) is not str or not re.fullmatch("[A-Za-z0-9_-]+\\.zip", archive_name)
            or type(member_prefix) is not str or not member_prefix or "\\" in member_prefix
            or PurePosixPath(member_prefix).is_absolute() or any(part in ("", ".", "..") for part in member_prefix.split("/"))):
        raise ValueError("development-final-versioned-transport-required")
    routes_bytes, protocol_bytes = Path(dev_routes).read_bytes(), Path(protocol_file).read_bytes()
    routes, protocol = json.loads(routes_bytes), json.loads(protocol_bytes)
    if protocol != freeze_final_protocol(routes, protocol["checkpoint_sha256"], protocol["controller_configs"],
            phase=protocol["phase"], representative_seed=protocol["representative_seed"],
            training_comparison_sha256=protocol["training_comparison_sha256"], parent_study_sha256=protocol["parent_study_sha256"],
            offline_checkpoint_sha256=protocol["offline_checkpoint_sha256"], training_audit_sha256=protocol["training_audit_sha256"],
            preprocessing_configs=protocol["preprocessing_configs"]):
        raise ValueError("development-final-protocol-changed")
    if (type(checkpoint_members) is not dict or set(checkpoint_members) != set(protocol["checkpoint_sha256"]) - {"B0"}
            or checkpoint_members["incumbent"] != checkpoint_members["control"]
            or any(type(member) is not str or not member or "\\" in member or PurePosixPath(member).is_absolute()
                or any(part in ("", ".", "..") for part in member.split("/")) or PurePosixPath(member).suffix != ".pt"
                for member in checkpoint_members.values())):
        raise ValueError("development-final-selected-members-invalid")
    names = tuple(dict.fromkeys((*SOURCES, *RUNTIME_SOURCES, *final_source_hashes())))
    sources = {name: (ROOT / name).read_bytes().decode("utf-8") for name in names}
    for name, value in sources.items():
        if name.endswith(".py"):
            compile(value, name, "exec")
    bootstrap = render_bootstrap((ROOT / "scripts/kaggle_carla_bootstrap.py").read_text(encoding="utf-8"), maximum_seconds=7200)
    old = '        selected_checkpoint = resolve_selected_checkpoint(Path("/kaggle/input"), CHECKPOINT_SHA256, CHECKPOINT_BYTES)\n'
    if bootstrap.count(old) != 1:
        raise ValueError("development-final-checkpoint-marker-mismatch")
    bootstrap = bootstrap.replace(old, '        selected_checkpoint = None\n', 1)
    bootstrap = bootstrap.replace('        report["checkpoint_sha256"] = CHECKPOINT_SHA256\n',
        '        report["checkpoint_sha256"] = CHECKPOINTS\n', 1)
    bootstrap = bootstrap.replace('        report["checkpoint_bytes"] = CHECKPOINT_BYTES\n', '', 1)
    marker = '        stage("server-download")\n'
    if bootstrap.count(marker) != 1:
        raise ValueError("development-final-transport-marker-mismatch")
    setup = '''        stage("pinned-development-representative-models")
        dependencies = scratch / "kagglehub_tools"
        command([uv, "pip", "install", "--python", python, "--only-binary", ":all:",
                 "--target", str(dependencies), "kagglehub==1.0.2"], "carla_final_transport_dependencies", 180, env)
        transport_helper = TRAINING_DOWNLOAD_HELPER.replace(repr("__VLF_RECOVERY_DEPS__"), repr(str(dependencies))).replace(
            repr("__VLF_RECOVERY_DOWNLOAD__"), repr(str(scratch / "training_download")))
        downloaded = command([python, "-c", transport_helper], "carla_final_transport", 1900, env, True)
        value = json.loads(downloaded.stdout.strip().splitlines()[-1])
        if (value.get("status") != "PASS" or value.get("sha256") != TRAINING_ARCHIVE_SHA256
                or value.get("bytes") != TRAINING_ARCHIVE_BYTES):
            raise BootstrapFailure("development-final-pinned-model-transport-invalid")
        models_file = unpack_final_checkpoints(value["path"], scratch / "trained",
            expected_archive_sha256=TRAINING_ARCHIVE_SHA256, expected_archive_bytes=TRAINING_ARCHIVE_BYTES,
            checkpoint_sha256=CHECKPOINTS, offline_checkpoint_sha256=OFFLINE_CHECKPOINTS,
            checkpoint_members=CHECKPOINT_MEMBERS, comparison_sha256=COMPARISON_SHA256,
            representative_seed=REPRESENTATIVE_SEED, member_prefix=MEMBER_PREFIX)
        stage("server-download")
'''
    bootstrap = bootstrap.replace(marker, setup, 1)
    start = bootstrap.find('        if WORKLOAD == "learned":\n            stage("selected-rgb-model-native-driving-and-replay")')
    end = bootstrap.find('        else:\n            stage("rgb-probe")', start)
    if start < 0 or end < 0:
        raise ValueError("development-final-workload-marker-mismatch")
    pipeline = '''        if WORKLOAD == "learned":
            stage("paired-development-pipeline-confirmation")
            project = scratch / "project"
            project.mkdir()
            for relative, source in PROJECT_SOURCES.items():
                path = project / relative
                if not path.resolve().is_relative_to(project.resolve()):
                    raise BootstrapFailure("development-final-embedded-project-path-invalid")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.encode("utf-8"))
            report["learned_runtime_sources"] = export_baseline_runtime_sources(project, output, PROJECT_SOURCES, artifact="learned")
            routes_file, protocol_file = project / "frozen_dev_routes.json", project / "frozen_final_protocol.json"
            routes_file.write_bytes(DEV_ROUTES_BYTES)
            protocol_file.write_bytes(PROTOCOL_BYTES)
            run_logged_study([python, "-u", str(project / "scripts/run_development_final.py"),
                     "--routes", str(routes_file), "--protocol", str(protocol_file), "--models", str(models_file),
                     "--device", "cuda", "--output-dir", str(output / "native")],
                    output / "carla_development_final.log", MAX_SECONDS - 900, env)
            gate = json.loads((output / "native/native_report.json").read_text())
            if (gate.get("status") != "PASS" or gate.get("episode_count") != EXPECTED_EPISODES
                    or gate.get("closed_loop_executed") is not True or gate.get("frame_count", 0) <= 0
                    or gate.get("final_town_used") is not False or gate.get("final_conditions_used") is not False
                    or gate.get("checkpoint_sha256") != CHECKPOINTS or gate.get("protocol_sha256") != PROTOCOL_SHA256
                    or set(gate.get("models", {})) != set(CHECKPOINTS)
                    or any(gate["models"][arm].get("device") != "cuda" for arm in CHECKPOINTS if arm != "B0")
                    or gate["models"]["B0"].get("device") != "cpu"
                    or not all(e.get("gate", {}).get("status") == "PASS" and e.get("media", {}).get("status") == "PASS" for e in gate["episodes"])
                    or gate.get("pointer_updated") is not False):
                raise BootstrapFailure("development-final-native-integrity-proof-invalid")
            _assert_runtime_sources(project, PROJECT_SOURCES)
            for value in json.loads(models_file.read_text()).values():
                if _file_sha256(Path(value["path"])) != value["sha256"]:
                    raise BootstrapFailure("development-final-model-weights-changed")
            report["learned_verified"] = True
            report["runtime_sources_unchanged"] = True
            report["learned"] = {key: gate[key] for key in ("status", "phase", "episode_count", "frame_count")}
'''
    bootstrap = bootstrap[:start] + pipeline + bootstrap[end:]
    bootstrap = bootstrap.replace('preserve_baseline_failure_artifacts(output, artifact="learned")',
        'preserve_baseline_failure_artifacts(output, artifact="native")')
    bootstrap = bootstrap.replace('artifact not in ("baseline", "pilot", "learned")',
        'artifact not in ("baseline", "pilot", "learned", "native")')
    own = Path(__file__).read_text(encoding="utf-8")
    function = next(node for node in ast.parse(own).body if isinstance(node, ast.FunctionDef) and node.name == "unpack_final_checkpoints")
    bootstrap += "\n" + ast.get_source_segment(own, function) + "\n"
    first = Path(__file__).with_name("prepare_kaggle_development_study.py").read_text(encoding="utf-8")
    function = next(node for node in ast.parse(first).body if isinstance(node, ast.FunctionDef) and node.name == "run_logged_study")
    bootstrap += "\n" + ast.get_source_segment(first, function) + "\n"
    vulkan = (ROOT / "scripts/kaggle_vulkan_probe.py").read_text(encoding="utf-8")
    function = next(node for node in ast.parse(vulkan).body if isinstance(node, ast.FunctionDef) and node.name == "vulkan_devices")
    child = "import ctypes,ctypes.util,struct,json\n" + ast.get_source_segment(vulkan, function) + "\nprint(json.dumps(vulkan_devices()))\n"
    code = bootstrap + "\nPROBE_SOURCE=" + repr(sources["scripts/carla_rgb_probe.py"]) + "\nVULKAN_CHILD_SOURCE=" + repr(child)
    for name, payload in (("PROJECT_SOURCES", json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()),
            ("DEV_ROUTES_BYTES", routes_bytes), ("PROTOCOL_BYTES", protocol_bytes)):
        code += "\nimport base64,lzma\n" + name + "=lzma.decompress(base64.b64decode(" + repr(base64.b64encode(lzma.compress(payload, preset=9)).decode()) + ",validate=True))\n"
        if name == "PROJECT_SOURCES":
            code += "PROJECT_SOURCES=json.loads(PROJECT_SOURCES)\n"
    download_helper = cloud_download_helper_source(training_version, Path("__VLF_RECOVERY_DEPS__"),
        Path("__VLF_RECOVERY_DOWNLOAD__"), notebook_id=TRAINING_NOTEBOOK, filename=archive_name,
        expected_sha256=training_archive_sha256, expected_bytes=training_archive_bytes, maximum_seconds=1800)
    compile(download_helper, "pinned-development-final-transport", "exec")
    values = {"CHECKPOINT_SHA256": protocol["checkpoint_sha256"]["incumbent"], "CHECKPOINT_BYTES": 1,
        "LEARNED_MODE": protocol["phase"], "TRAINING_HANDLE": TRAINING_NOTEBOOK + "/versions/" + str(training_version),
        "TRAINING_VERSION": training_version, "TRAINING_ARCHIVE_NAME": archive_name,
        "TRAINING_ARCHIVE_SHA256": training_archive_sha256, "TRAINING_ARCHIVE_BYTES": training_archive_bytes,
        "TRAINING_DOWNLOAD_HELPER": download_helper, "CHECKPOINTS": protocol["checkpoint_sha256"],
        "OFFLINE_CHECKPOINTS": protocol["offline_checkpoint_sha256"], "CHECKPOINT_MEMBERS": checkpoint_members,
        "COMPARISON_SHA256": protocol["training_comparison_sha256"], "REPRESENTATIVE_SEED": protocol["representative_seed"],
        "MEMBER_PREFIX": member_prefix, "PROTOCOL_SHA256": digest(protocol), "EXPECTED_EPISODES": protocol["expected_episodes"]}
    code += "\n" + "\n".join(name + "=" + repr(value) for name, value in values.items()) + "\nraise SystemExit(main())\n"
    compile(code, "native.py", "exec")
    if len(code.encode()) >= 1_000_000:
        raise ValueError("development-final-notebook-source-size-exceeded")
    submission = {"id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE, "code_file": "native.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4"}
    for name in ("dataset_sources", "kernel_sources", "competition_sources", "model_sources"):
        submission[name] = list(metadata.get(name, []))
    submission["kernel_sources"] = [name for name in submission["kernel_sources"] if name != TRAINING_NOTEBOOK]
    bundle = {"status": "PREPARED", "scope": protocol["protocol_version"], "phase": protocol["phase"],
        "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "source_sha256": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()},
        "protocol_sha256": digest(protocol), "protocol_file_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "training_version": training_version, "training_archive_sha256": training_archive_sha256,
        "training_archive_bytes": training_archive_bytes, "checkpoint_sha256": protocol["checkpoint_sha256"],
        "offline_checkpoint_sha256": protocol["offline_checkpoint_sha256"], "checkpoint_members": checkpoint_members,
        "preprocessing_configs": protocol["preprocessing_configs"],
        "training_audit_sha256": protocol["training_audit_sha256"], "training_comparison_sha256": protocol["training_comparison_sha256"],
        "transport": "official-versioned-response-with-bounded-resumable-HTTP-ranges",
        "training_download_helper_sha256": hashlib.sha256(download_helper.encode()).hexdigest(),
        "expected_episodes": protocol["expected_episodes"], "maximum_seconds": 7200, "live_owned_child_progress": True,
        "creates_new_notebook": False, "creates_dataset": False, "local_gpu_used": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "native.py").write_bytes(code.encode())
    for name, value in (("bundle_manifest.json", bundle), ("kernel-metadata.json", submission)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return bundle
