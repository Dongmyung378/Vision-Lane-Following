"""Run matched appearance fine-tuning in the existing private Kaggle notebook.

Only standard-library imports run before reviewed sources are materialized.
Prior data replay evidence is reused explicitly; no CARLA process is started.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

PROJECT_SOURCES = {}
FROZEN_INPUT = {}
TRUSTED_CPU_PROOF = ""
DOWNLOAD_HELPER_SOURCE = ""
MAX_SECONDS = 14400
TRAINING_SECONDS = 10800


class DevelopmentBootstrapError(RuntimeError):
    pass


def visible_epoch_event(value):
    names = ("phase", "arm", "seed", "epoch", "train_loss", "dev_loss", "optimizer_updates")
    if (type(value) is not dict or value.get("phase") != "development-training"
            or value.get("arm") not in ("continued", "appearance_robust")
            or any(type(value.get(name)) is not int or value[name] < 0 for name in ("seed", "epoch", "optimizer_updates"))
            or any(type(value.get(name)) not in (int, float) or not math.isfinite(value[name])
                   for name in ("train_loss", "dev_loss"))):
        return None
    return {name: value[name] for name in names}


def validate_development_report(report, frozen, protocol):
    """Require both real T4 arms, all seeds, and full identical clean DEV."""
    from carla_tools.contracts import canonical_sha256
    from carla_tools.development_training import development_protocol_from_dict, verify_equal_budget
    parsed = development_protocol_from_dict(protocol)
    budget = protocol["training"]
    seeds, arms = {str(seed) for seed in budget["seeds"]}, {"continued", "appearance_robust"}
    if (report.get("status") != "PASS" or report.get("fixture_injection") is not False
            or report.get("final_town_used") is not False or report.get("closed_loop_verified") is not False
            or report.get("incumbent_pointer_updated") is not False or report.get("equal_budget_verified") is not True
            or report.get("protocol") != protocol or report.get("protocol_sha256") != canonical_sha256(protocol)
            or report.get("protocol_sha256") != frozen["training_protocol_sha256"]
            or report.get("manifest_sha256") != frozen["manifest_sha256"]
            or report.get("validation_report_sha256") != frozen["transport_proof_sha256"]
            or report.get("initial_checkpoint_sha256") != frozen["initial_checkpoint_sha256"]
            or report.get("common_dev_sample_count") != frozen["split_counts"]["dev"]
            or set(report.get("training", {})) != seeds or set(report.get("evaluations", {})) != seeds):
        raise DevelopmentBootstrapError("development-comparison-source-or-budget-proof-invalid")
    source = report.get("source_sha256")
    expected_updates = math.ceil(budget["train_samples_per_arm"] / budget["batch_size"]) * budget["epochs"]
    for seed in seeds:
        if set(report["training"][seed]) != arms or set(report["evaluations"][seed]) != arms:
            raise DevelopmentBootstrapError("development-comparison-missing-arm")
        for arm in arms:
            train, evaluation = report["training"][seed][arm], report["evaluations"][seed][arm]
            for row in (train, evaluation):
                runtime = row.get("runtime", {})
                if (row.get("status") != "PASS" or row.get("arm") != arm or row.get("seed") != int(seed)
                        or row.get("fixture_injection") is not False or row.get("final_town_used") is not False
                        or row.get("manifest_sha256") != frozen["manifest_sha256"]
                        or row.get("validation_report_sha256") != frozen["transport_proof_sha256"]
                        or row.get("protocol_sha256") != frozen["training_protocol_sha256"]
                        or not source or row.get("source_sha256") != source
                        or runtime.get("device") != "cuda" or "T4" not in str(runtime.get("gpu", ""))):
                    raise DevelopmentBootstrapError("development-actual-T4-arm-source-proof-invalid")
            if (train.get("train_sample_count") != budget["train_samples_per_arm"]
                    or train.get("completed_epochs") != budget["epochs"]
                    or train.get("optimizer_updates") != expected_updates
                    or train.get("initial_checkpoint_sha256") != frozen["initial_checkpoint_sha256"]
                    or train.get("epoch_dev_views") != protocol["epoch_dev_views"]
                    or evaluation.get("split") != "dev" or evaluation.get("view") != "clean"
                    or evaluation.get("sample_count") != frozen["split_counts"]["dev"]
                    or evaluation.get("checkpoint_sha256") != train.get("checkpoint_sha256")
                    or not train.get("checkpoint_sha256")
                    or evaluation.get("common_dev_selection_sha256") != report.get("common_dev_selection_sha256")):
                raise DevelopmentBootstrapError("development-full-development-or-update-proof-invalid")
    verify_equal_budget(report["training"], parsed)
    initial = report.get("initial_evaluation", {})
    runtime = initial.get("runtime", {})
    if (initial.get("status") != "PASS" or initial.get("arm") != "incumbent"
            or initial.get("checkpoint_sha256") != frozen["initial_checkpoint_sha256"]
            or initial.get("sample_count") != frozen["split_counts"]["dev"]
            or initial.get("view") != "clean" or initial.get("split") != "dev"
            or initial.get("common_dev_selection_sha256") != report.get("common_dev_selection_sha256")
            or initial.get("source_sha256") != source or initial.get("final_town_used") is not False
            or initial.get("fixture_injection") is not False or runtime.get("device") != "cuda"
            or "T4" not in str(runtime.get("gpu", ""))):
        raise DevelopmentBootstrapError("development-unchanged-incumbent-evaluation-proof-invalid")
    return {"actual_T4_training": True, "seed_count": len(seeds), "arm_count_per_seed": len(arms),
            "common_dev_samples": frozen["split_counts"]["dev"], "new_native_replay_performed": False}


def main_development():
    output = Path("/kaggle/working")
    scratch = Path(tempfile.mkdtemp(prefix="vlf_development_", dir="/tmp"))
    started, current = time.monotonic(), None
    report = {"schema_version": 1, "status": "FAIL", "workload": "matched-appearance-development-training",
        "frozen_input": FROZEN_INPUT, "local_gpu_used": False, "carla_server_started": False,
        "new_native_replay_performed": False, "training_process_stopped": False, "stages": []}

    def expire(_signum, _frame):
        raise DevelopmentBootstrapError("development-total-time-limit")

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, expire)
    signal.alarm(MAX_SECONDS)

    def stage(name):
        report["stage"] = name
        event = {"name": name, "elapsed_seconds": time.monotonic() - started}
        report["stages"].append(event)
        print(json.dumps({"phase": "development-bootstrap", **event}), flush=True)

    def command(args, label, timeout=600, env=None):
        nonlocal current
        remaining = min(timeout, MAX_SECONDS - (time.monotonic() - started))
        if remaining <= 0:
            raise DevelopmentBootstrapError("development-total-time-limit")
        with (output / (label + ".log")).open("xb") as log:
            if label == "development_training":
                current = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           env=env, start_new_session=True)
                child, reader_errors = current, []

                def copy_progress():
                    discarding = False
                    try:
                        with child.stdout:
                            while True:
                                piece = child.stdout.readline(16385)
                                if not piece:
                                    break
                                log.write(piece); log.flush()
                                complete = piece.endswith(b"\n")
                                if discarding or len(piece) > 16384:
                                    discarding = not complete
                                    continue
                                if not complete:
                                    discarding = True
                                    continue
                                try:
                                    event = visible_epoch_event(json.loads(piece.decode("utf-8")))
                                except (UnicodeError, ValueError):
                                    continue
                                if event is not None:
                                    print(json.dumps(event, allow_nan=False), flush=True)
                    except Exception as error:
                        reader_errors.append(type(error).__name__)

                reader = threading.Thread(target=copy_progress, daemon=True)
                reader.start()
                code = current.wait(timeout=remaining)
                reader.join(timeout=10)
                if reader.is_alive() or reader_errors:
                    raise DevelopmentBootstrapError("development-progress-reader-failed")
            else:
                current = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                code = current.wait(timeout=remaining)
            current = None
        if code:
            raise DevelopmentBootstrapError(label + "-failed")

    base = None
    try:
        stage("materialize-reviewed-sources")
        project = scratch / "project"; project.mkdir()
        for relative, content in PROJECT_SOURCES.items():
            name = PurePosixPath(relative)
            if (name.is_absolute() or ".." in name.parts or "\\" in relative or ":" in relative
                    or name.parts[0] not in ("scripts", "carla_tools", "configs")):
                raise DevelopmentBootstrapError("development-source-path-invalid")
            target = project / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.encode())
        sys.path.insert(0, str(project))
        from scripts import kaggle_recovery_training_bootstrap as base
        base.DOWNLOAD_HELPER_SOURCE = DOWNLOAD_HELPER_SOURCE
        report["runtime_sources"] = base.export_runtime_sources(project, output, PROJECT_SOURCES)
        env = os.environ.copy()
        env.update(UV_CACHE_DIR=str(scratch / "uv_cache"), UV_PYTHON_INSTALL_DIR=str(scratch / "uv_python"),
            UV_NO_PROGRESS="1", UV_LINK_MODE="copy", TORCH_HOME=str(scratch / "torch_cache"),
            CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", PYTHONPATH=str(project))
        stage("pinned-cuda-runtime")
        uv = shutil.which("uv")
        if not uv:
            tools = scratch / "uv_tools"
            command([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--target", str(tools), "uv==0.8.22"], "development_uv", 180, env)
            uv = str(tools / "bin" / "uv")
        command([uv, "python", "install", "3.12"], "development_python", 180, env)
        venv = scratch / "venv"
        command([uv, "venv", "--python", "3.12", "--managed-python", str(venv)], "development_venv", 60, env)
        python = str(venv / "bin" / "python")
        command([uv, "pip", "install", "--python", python, "--only-binary", ":all:", "--index-url",
            "https://download.pytorch.org/whl/cu126", "torch==2.7.1", "torchvision==0.22.1"], "development_torch", 900, env)
        command([uv, "pip", "install", "--python", python, "--only-binary", ":all:", "numpy==2.2.6",
            "opencv-python-headless==4.12.0.88", "Pillow==11.3.0", "carla==0.9.16"], "development_dependencies", 180, env)
        stage("exact-mounted-capture")
        # The prepared existing CPU notebook output is compulsory: no giant
        # private dataset upload, alternate notebook, or unpinned fallback.
        archive, report["capture_transport"] = base.resolve_frozen_archive(Path("/kaggle/input"), FROZEN_INPUT["archive_sha256"])
        stage("reconstruct-exact-dataset-and-trusted-CPU-proof")
        (project / "frozen_input.json").write_text(json.dumps(FROZEN_INPUT))
        (project / "trusted_cpu_proof.json").write_bytes(TRUSTED_CPU_PROOF.encode())
        reconstruction = ("import json,sys;from pathlib import Path;sys.path.insert(0," + repr(str(project)) + ");"
            "from scripts.kaggle_recovery_training_bootstrap import rebuild_frozen_dataset;"
            "f=json.loads(Path(" + repr(str(project / "frozen_input.json")) + ").read_text());"
            "p=Path(" + repr(str(project / "trusted_cpu_proof.json")) + ").read_text();"
            "m,v=rebuild_frozen_dataset(Path(" + repr(str(archive)) + "),Path(" + repr(str(scratch / "capture"))
            + "),f,p);print(json.dumps({'manifest':str(m),'proof':str(v)}))")
        command([python, "-c", reconstruction], "development_dataset_reconstruction", 1800, env)
        reconstructed = json.loads((output / "development_dataset_reconstruction.log").read_text().strip().splitlines()[-1])
        stage("exact-previous-selected-checkpoint")
        dependencies = scratch / "kagglehub_tools"
        command([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--only-binary", ":all:",
            "--target", str(dependencies), "kagglehub==1.0.2"], "development_transport_dependencies", 180)
        helper = base.cloud_download_helper_source(FROZEN_INPUT["initial_notebook_version"], dependencies,
            scratch / "initial_cloud", notebook_id=base.TRAINING_NOTEBOOK_ID, filename="training_artifacts.zip",
            expected_sha256=FROZEN_INPUT["initial_archive_sha256"], expected_bytes=FROZEN_INPUT["initial_archive_bytes"], maximum_seconds=900)
        command([sys.executable, "-c", helper], "development_initial_transport", 900)
        transported = json.loads((output / "development_initial_transport.log").read_text().strip())
        if transported.get("status") != "PASS" or transported.get("sha256") != FROZEN_INPUT["initial_archive_sha256"]:
            raise DevelopmentBootstrapError("development-initial-transport-invalid")
        initial = base.extract_frozen_initial_checkpoint(transported["path"], scratch / "initial/checkpoint.pt", FROZEN_INPUT)
        report["initial_checkpoint_sha256"] = base.sha256_file(initial)
        smoke = ("import json,torch,torchvision;assert torch.__version__.split('+')[0]=='2.7.1';"
            "assert torchvision.__version__.split('+')[0]=='0.22.1';assert torch.cuda.is_available();"
            "assert 'T4' in torch.cuda.get_device_name(0);print(json.dumps({'device':'cuda','gpu':torch.cuda.get_device_name(0)}))")
        command([python, "-c", smoke], "development_cuda_smoke", 60, env)
        report["cuda_smoke"] = json.loads((output / "development_cuda_smoke.log").read_text().strip())
        stage("actual-T4-paired-appearance-training-and-common-development")
        protocol_path = project / "development_runtime_protocol.json"
        protocol_path.write_text(json.dumps(FROZEN_INPUT["training_protocol"], allow_nan=False))
        command([python, str(project / "scripts/train_development_waypoints.py"), "--manifest", reconstructed["manifest"],
            "--validation-report", reconstructed["proof"], "--initial-checkpoint", str(initial),
            "--output-dir", str(output / "experiment"), "--protocol", str(protocol_path), "--device", "cuda"],
            "development_training", TRAINING_SECONDS, env)
        validation = ("import json,sys;from pathlib import Path;sys.path.insert(0," + repr(str(project)) + ");"
            "from scripts.kaggle_development_training_bootstrap import validate_development_report;"
            "r=json.loads(Path(" + repr(str(output / "experiment/comparison.json")) + ").read_text());"
            "f=json.loads(Path(" + repr(str(project / "frozen_input.json")) + ").read_text());"
            "print(json.dumps(validate_development_report(r,f,f['training_protocol'])))")
        command([python, "-c", validation], "development_report_validation", 180, env)
        report["experiment"] = json.loads((output / "development_report_validation.log").read_text().strip())
        report["comparison_sha256"] = base.sha256_file(output / "experiment/comparison.json")
        report["status"] = "PASS"
        stage("complete")
    except Exception as error:
        report["error"] = {"exception_type": type(error).__name__,
            "code": str(error) if isinstance(error, (DevelopmentBootstrapError, base.RecoveryBootstrapError if base else DevelopmentBootstrapError))
                    else "development-bootstrap-operation-failed"}
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        if current is not None and current.poll() is None:
            os.killpg(current.pid, signal.SIGTERM)
            try:
                current.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(current.pid, signal.SIGKILL); current.wait(timeout=10)
        report["training_process_stopped"] = current is None or current.poll() is not None
        report["elapsed_seconds"] = time.monotonic() - started
        try:
            if base is None:
                raise DevelopmentBootstrapError("development-runtime-not-materialized")
            report["artifacts"] = base.preserve_training_artifacts(output)
        except Exception as error:
            report.update(status="FAIL", archive_error=type(error).__name__)
        (output / "kaggle_development_bootstrap.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 1
