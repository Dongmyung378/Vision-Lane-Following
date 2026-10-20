"""Package a frozen development study for an existing private CARLA notebook.

This module performs local preparation only. The caller must freshly verify the
existing notebook identity and access before submission; no fallback creation
or local GPU execution is provided.
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


def unpack_study_checkpoint(archive, destination, *, expected_archive_sha256,
        expected_archive_bytes, checkpoint_sha256, member):
    """Extract one exact, hash-bound checkpoint from a completed training run."""
    import hashlib
    from pathlib import Path, PurePosixPath
    import re
    import zipfile
    archive, destination = Path(archive), Path(destination)
    if (type(expected_archive_bytes) is not int or expected_archive_bytes <= 0
            or archive.is_symlink() or archive.stat().st_size != expected_archive_bytes
            or any(not re.fullmatch("[0-9a-f]{64}", str(value)) for value in (expected_archive_sha256, checkpoint_sha256))):
        raise ValueError("development-checkpoint-archive-binding-invalid")
    checksum = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    if checksum.hexdigest() != expected_archive_sha256:
        raise ValueError("development-checkpoint-archive-hash-mismatch")
    relative = PurePosixPath(member)
    if (type(member) is not str or not member or "\\" in member or relative.is_absolute()
            or any(part in ("", ".", "..") for part in member.split("/")) or relative.suffix != ".pt"
            or destination.exists() or destination.is_symlink()):
        raise ValueError("development-checkpoint-new-safe-output-required")
    with zipfile.ZipFile(archive) as stream:
        names = stream.namelist()
        if len(names) != len(set(names)):
            raise ValueError("development-checkpoint-duplicate-archive-members")
        info = stream.getinfo(member)
        if (not 0 < info.file_size <= 128 * 1024**2 or info.flag_bits & 1
                or (info.external_attr >> 16) & 0o170000 == 0o120000):
            raise ValueError("development-checkpoint-archive-member-invalid")
        value = stream.read(info)
    if hashlib.sha256(value).hexdigest() != checkpoint_sha256:
        raise ValueError("development-checkpoint-member-hash-mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(value)
    return destination


def run_logged_study(args, log_path, timeout, env):
    """Tee only the owned study child's stdout while enforcing its deadline."""
    import queue
    import subprocess
    import threading
    import time
    from pathlib import Path
    if type(timeout) not in (int, float) or timeout <= 0:
        raise ValueError("development-child-deadline-invalid")
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env, bufsize=1)
    lines = queue.Queue()
    def receive():
        try:
            for line in process.stdout:
                lines.put(line)
        finally:
            process.stdout.close()
    thread = threading.Thread(target=receive, name="owned-development-output", daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    try:
        with Path(log_path).open("x", encoding="utf-8") as log:
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError("development-owned-child-timeout")
                try:
                    line = lines.get(timeout=.25)
                except queue.Empty:
                    if process.poll() is not None and not thread.is_alive() and lines.empty():
                        break
                    continue
                log.write(line)
                log.flush()
                print(line.rstrip("\r\n"), flush=True)
        returncode = process.wait(timeout=10)
        if returncode != 0:
            raise RuntimeError("development-owned-child-exit:" + str(returncode))
        return returncode
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        thread.join(timeout=2)


def prepare_development_study(existing_metadata, output_dir, *, dev_routes, protocol_file,
        training_archive_sha256, training_archive_bytes, training_version, checkpoint_sha256,
        checkpoint_member, archive_name="training_artifacts.zip"):
    from carla_tools.development_study import digest, freeze_development_protocol, study_source_hashes
    from carla_tools.runtime_driving import RUNTIME_SOURCES
    from scripts.kaggle_recovery_training_bootstrap import cloud_download_helper_source
    metadata = json.loads(Path(existing_metadata).read_text(encoding="utf-8"))
    output = Path(output_dir)
    if (metadata.get("id") != NOTEBOOK_ID or metadata.get("title") != NOTEBOOK_TITLE
            or metadata.get("is_private") is not True):
        raise ValueError("development-existing-private-CARLA-notebook-required")
    if output.exists() or output.is_symlink():
        raise ValueError("development-new-local-bundle-required")
    if (type(training_version) is not int or training_version <= 0
            or type(training_archive_bytes) is not int or not 0 < training_archive_bytes <= 32 * 1024**3
            or type(archive_name) is not str or not re.fullmatch("[A-Za-z0-9_-]+\\.zip", archive_name)
            or type(checkpoint_member) is not str or "\\" in checkpoint_member
            or PurePosixPath(checkpoint_member).is_absolute()
            or any(part in ("", ".", "..") for part in checkpoint_member.split("/"))
            or PurePosixPath(checkpoint_member).suffix != ".pt"
            or any(not re.fullmatch("[0-9a-f]{64}", str(value))
                for value in (training_archive_sha256, checkpoint_sha256))):
        raise ValueError("development-pinned-training-transport-required")
    routes_bytes, protocol_bytes = Path(dev_routes).read_bytes(), Path(protocol_file).read_bytes()
    routes, protocol = json.loads(routes_bytes), json.loads(protocol_bytes)
    if protocol != freeze_development_protocol(routes, checkpoint_sha256, protocol["controller_configs"],
            environments=protocol["environments"]):
        raise ValueError("development-frozen-protocol-mismatch")
    names = tuple(dict.fromkeys((*SOURCES, *RUNTIME_SOURCES, *study_source_hashes())))
    sources = {name: (ROOT / name).read_bytes().decode("utf-8") for name in names}
    for name, value in sources.items():
        if name.endswith(".py"):
            compile(value, name, "exec")
    bootstrap = render_bootstrap((ROOT / "scripts/kaggle_carla_bootstrap.py").read_text(encoding="utf-8"), maximum_seconds=7200)
    old = '        selected_checkpoint = resolve_selected_checkpoint(Path("/kaggle/input"), CHECKPOINT_SHA256, CHECKPOINT_BYTES)\n'
    if bootstrap.count(old) != 1:
        raise ValueError("development-checkpoint-bootstrap-marker-mismatch")
    bootstrap = bootstrap.replace(old, '        selected_checkpoint = None\n', 1)
    bootstrap = bootstrap.replace('        report["checkpoint_bytes"] = CHECKPOINT_BYTES\n', '', 1)
    marker = '        stage("server-download")\n'
    if bootstrap.count(marker) != 1:
        raise ValueError("development-transport-bootstrap-marker-mismatch")
    setup = '''        stage("pinned-development-selected-checkpoint")
        dependencies = scratch / "kagglehub_tools"
        command([uv, "pip", "install", "--python", python, "--only-binary", ":all:",
                 "--target", str(dependencies), "kagglehub==1.0.2"], "carla_development_transport_dependencies", 180, env)
        transport_helper = TRAINING_DOWNLOAD_HELPER.replace(repr("__VLF_RECOVERY_DEPS__"), repr(str(dependencies))).replace(
            repr("__VLF_RECOVERY_DOWNLOAD__"), repr(str(scratch / "training_download")))
        downloaded = command([python, "-c", transport_helper], "carla_development_transport", 1900, env, True)
        value = json.loads(downloaded.stdout.strip().splitlines()[-1])
        if (value.get("status") != "PASS" or value.get("sha256") != TRAINING_ARCHIVE_SHA256
                or value.get("bytes") != TRAINING_ARCHIVE_BYTES):
            raise BootstrapFailure("development-pinned-training-transport-invalid")
        selected_checkpoint = unpack_study_checkpoint(value["path"], scratch / "selected/checkpoint.pt",
            expected_archive_sha256=TRAINING_ARCHIVE_SHA256, expected_archive_bytes=TRAINING_ARCHIVE_BYTES,
            checkpoint_sha256=CHECKPOINT_SHA256, member=CHECKPOINT_MEMBER)
        stage("server-download")
'''
    bootstrap = bootstrap.replace(marker, setup, 1)
    start = bootstrap.find('        if WORKLOAD == "learned":\n            stage("selected-rgb-model-native-driving-and-replay")')
    end = bootstrap.find('        else:\n            stage("rgb-probe")', start)
    if start < 0 or end < 0:
        raise ValueError("development-workload-bootstrap-marker-mismatch")
    pipeline = '''        if WORKLOAD == "learned":
            stage("paired-controller-camera-development-study")
            project = scratch / "project"
            project.mkdir()
            for relative, source in PROJECT_SOURCES.items():
                path = project / relative
                if not path.resolve().is_relative_to(project.resolve()):
                    raise BootstrapFailure("development-embedded-project-path-invalid")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.encode("utf-8"))
            report["learned_runtime_sources"] = export_baseline_runtime_sources(project, output, PROJECT_SOURCES, artifact="learned")
            routes_file, protocol_file = project / "frozen_dev_routes.json", project / "frozen_development_protocol.json"
            routes_file.write_bytes(DEV_ROUTES_BYTES)
            protocol_file.write_bytes(PROTOCOL_BYTES)
            run_logged_study([python, "-u", str(project / "scripts/run_development_study.py"),
                     "--routes", str(routes_file), "--protocol", str(protocol_file),
                     "--checkpoint", str(selected_checkpoint), "--device", "cuda", "--output-dir", str(output / "native")],
                    output / "carla_development.log", MAX_SECONDS - 900, env)
            gate = json.loads((output / "native/native_report.json").read_text())
            if (gate.get("status") != "PASS" or gate.get("episode_count") != EXPECTED_EPISODES
                    or gate.get("closed_loop_executed") is not True or gate.get("frame_count", 0) <= 0
                    or gate.get("final_town_used") is not False or gate.get("final_conditions_used") is not False
                    or gate.get("checkpoint_sha256") != CHECKPOINT_SHA256 or gate.get("protocol_sha256") != PROTOCOL_SHA256
                    or set(gate.get("models", {})) != {"M1", "B0"}
                    or gate["models"]["M1"].get("device") != "cuda" or gate["models"]["B0"].get("device") != "cpu"
                    or not all(e.get("gate", {}).get("status") == "PASS" and e.get("media", {}).get("status") == "PASS" for e in gate["episodes"])
                    or gate.get("selection", {}).get("status") != "PASS" or gate.get("pointer_updated") is not False):
                raise BootstrapFailure("development-native-integrity-proof-invalid")
            _assert_runtime_sources(project, PROJECT_SOURCES)
            if _file_sha256(selected_checkpoint) != CHECKPOINT_SHA256:
                raise BootstrapFailure("development-selected-weights-changed")
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
    for name in ("unpack_study_checkpoint", "run_logged_study"):
        function = next(node for node in ast.parse(own).body if isinstance(node, ast.FunctionDef) and node.name == name)
        bootstrap += "\n" + ast.get_source_segment(own, function) + "\n"
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
    compile(download_helper, "pinned-development-training-transport", "exec")
    values = {"CHECKPOINT_SHA256": checkpoint_sha256, "CHECKPOINT_BYTES": 1, "LEARNED_MODE": "controller-camera-development",
        "TRAINING_HANDLE": TRAINING_NOTEBOOK + "/versions/" + str(training_version), "TRAINING_VERSION": training_version,
        "TRAINING_ARCHIVE_NAME": archive_name, "TRAINING_ARCHIVE_SHA256": training_archive_sha256,
        "TRAINING_ARCHIVE_BYTES": training_archive_bytes, "TRAINING_DOWNLOAD_HELPER": download_helper,
        "CHECKPOINT_MEMBER": checkpoint_member, "PROTOCOL_SHA256": digest(protocol), "EXPECTED_EPISODES": protocol["expected_episodes"]}
    code += "\n" + "\n".join(name + "=" + repr(value) for name, value in values.items()) + "\nraise SystemExit(main())\n"
    compile(code, "native.py", "exec")
    if len(code.encode()) >= 1_000_000:
        raise ValueError("development-notebook-source-size-exceeded")
    submission = {"id": NOTEBOOK_ID, "title": NOTEBOOK_TITLE, "code_file": "native.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4"}
    for name in ("dataset_sources", "kernel_sources", "competition_sources", "model_sources"):
        submission[name] = list(metadata.get(name, []))
    submission["kernel_sources"] = [name for name in submission["kernel_sources"] if name != TRAINING_NOTEBOOK]
    bundle = {"status": "PREPARED", "scope": protocol["protocol_version"],
        "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "source_sha256": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()},
        "protocol_sha256": digest(protocol), "protocol_file_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "training_version": training_version, "training_archive_sha256": training_archive_sha256,
        "training_archive_bytes": training_archive_bytes, "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_member": checkpoint_member, "transport": "official-versioned-response-with-bounded-resumable-HTTP-ranges",
        "training_download_helper_sha256": hashlib.sha256(download_helper.encode()).hexdigest(),
        "expected_episodes": protocol["expected_episodes"], "maximum_seconds": 7200,
        "live_owned_child_progress": True, "creates_new_notebook": False, "creates_dataset": False, "local_gpu_used": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "native.py").write_bytes(code.encode())
    for name, value in (("bundle_manifest.json", bundle), ("kernel-metadata.json", submission)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return bundle
