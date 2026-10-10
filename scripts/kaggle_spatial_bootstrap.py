"""Bounded, T4-only comparison in an existing private Kaggle notebook."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

try:
    TrainingBootstrapError
except NameError:
    from scripts.kaggle_training_bootstrap import (
        PROJECT_SOURCES, PILOT_ARCHIVE_SHA256, PILOT_MANIFEST_SHA256,
        PILOT_DOWNLOAD_VERSION, PILOT_NOTEBOOK_ID, TrainingBootstrapError,
        resolve_frozen_pilot, safe_extract_pilot, cloud_download_helper_source,
        export_runtime_sources,
    )

MAX_SECONDS = 3600
TRAINING_SECONDS = 2700
MAX_EPOCHS = 32
FROZEN_BASELINE = None


def validate_spatial_report(report, expected_train, expected_dev):
    if (report.get("status") != "PASS" or report.get("fixture_injection") is not False
            or report.get("final_town_used") is not False
            or set(report.get("training", {})) != {"resnet", "spatial", "hybrid"}
            or set(report.get("evaluations", {})) != {"resnet", "spatial", "hybrid"}
            or report.get("decision", {}).get("closed_loop_verified") is not False):
        raise TrainingBootstrapError("spatial-three-arm-proof-missing")
    config_hashes, weight_hashes, source_hashes = set(), set(), set()
    baseline_source = report["training"]["resnet"].get("source_sha256",{})
    for variant in ("resnet", "spatial", "hybrid"):
        train, evaluation = report["training"][variant], report["evaluations"][variant]
        runtime = train.get("runtime", {})
        if (train.get("status") != "PASS" or train.get("variant") != variant
                or train.get("train_sample_count") != expected_train or train.get("dev_sample_count") != expected_dev
                or train.get("fixture_injection") is not False or train.get("final_town_used") is not False
                or runtime.get("device") != "cuda" or "T4" not in str(runtime.get("gpu", ""))
                or runtime.get("amp_enabled") is not True
                or train.get("training_protocol") != "full-encoder-finetuning-identical-augmentation-and-budget"
                or train.get("overfit_check", {}).get("status") != "PASS"
                or train["overfit_check"].get("model_reset_to_fresh_initialization") is not True):
            raise TrainingBootstrapError("spatial-real-finetuning-proof-invalid")
        provenance = train.get("pretrained_provenance", {})
        if (provenance.get("pretrained") is not True or provenance.get("hash_verified") is not True
                or not provenance.get("sha256", "").startswith("f37072fd")):
            raise TrainingBootstrapError("spatial-pretrained-proof-invalid")
        if (evaluation.get("status") != "PASS" or evaluation.get("variant") != variant
                or evaluation.get("sample_count") != expected_dev or evaluation.get("split") != "dev"
                or evaluation.get("fixture_injection") is not False or evaluation.get("final_town_used") is not False
                or evaluation.get("manifest_sha256") != PILOT_MANIFEST_SHA256
                or train.get("manifest_sha256") != PILOT_MANIFEST_SHA256
                or evaluation.get("checkpoint_sha256") != train.get("checkpoint_sha256")
                or not train.get("checkpoint_sha256")
                or evaluation.get("source_sha256") != train.get("source_sha256")
                or evaluation.get("config_sha256") != train.get("config_sha256")
                or evaluation.get("pretrained_provenance") != provenance):
            raise TrainingBootstrapError("spatial-complete-dev-model-binding-invalid")
        config_hashes.add(train["config_sha256"])
        weight_hashes.add(provenance["sha256"])
        # A completed reference may come from an earlier execution. Its actual
        # training source remains recorded; core prediction sources must agree.
        for name in baseline_source:
            if name not in ("carla_tools/spatial_training.py","scripts/train_spatial_waypoints.py") and train["source_sha256"].get(name) != baseline_source[name]:
                raise TrainingBootstrapError("spatial-reference-prediction-source-mismatch")
        if variant != "resnet" or not report.get("baseline_reuse"):
            source_hashes.add(json.dumps(train["source_sha256"], sort_keys=True))
    if len(config_hashes) != 1 or len(weight_hashes) != 1 or len(source_hashes) != 1:
        raise TrainingBootstrapError("spatial-controlled-training-protocol-mismatch")
    return {"arms_verified": 3, "dev_samples_per_arm": expected_dev,
            "decision": report["decision"], "actual_t4_training": True}


def preserve_spatial_artifacts(output):
    output = Path(output)
    manifest = {}
    experiment = output / "experiment"
    with zipfile.ZipFile(output / "training_artifacts.zip", "x", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        if experiment.exists():
            for path in sorted(experiment.rglob("*")):
                if not path.is_file() or path.is_symlink() or path.suffix not in (".json", ".jsonl", ".pt", ".csv", ".png"):
                    continue
                if not path.resolve().is_relative_to(experiment.resolve()):
                    raise TrainingBootstrapError("spatial-artifact-path-escape")
                relative = path.relative_to(output).as_posix()
                data = path.read_bytes()
                manifest[relative] = hashlib.sha256(data).hexdigest()
                archive.writestr(relative, data)
    return {"sha256": hashlib.sha256((output / "training_artifacts.zip").read_bytes()).hexdigest(),
            "source_sha256": manifest}


def main_spatial():
    output = Path("/kaggle/working")
    scratch = Path(tempfile.mkdtemp(prefix="vlf_spatial_", dir="/tmp"))
    started, current = time.monotonic(), None
    report = {"schema_version": 1, "status": "FAIL", "workload": "controlled-spatial-hybrid-comparison",
              "local_gpu_used": False, "carla_server_started": False,
              "pilot_archive_sha256": PILOT_ARCHIVE_SHA256, "pilot_manifest_sha256": PILOT_MANIFEST_SHA256,
              "max_seconds": MAX_SECONDS, "training_seconds": TRAINING_SECONDS, "stages": []}

    def expire(_signum, _frame):
        raise TrainingBootstrapError("spatial-total-time-limit")

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, expire)
    signal.alarm(MAX_SECONDS)

    def stage(name):
        report["stage"] = name
        report["stages"].append({"name": name, "elapsed_seconds": time.monotonic() - started})
        print("Spatial waypoint comparison:", name, flush=True)

    def command(args, label, timeout=600, env=None):
        nonlocal current
        remaining = min(timeout, MAX_SECONDS - (time.monotonic() - started))
        if remaining <= 0:
            raise TrainingBootstrapError("spatial-total-time-limit")
        with (output / (label + ".log")).open("x", encoding="utf-8") as log:
            current = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            code = current.wait(timeout=remaining)
            current = None
        if code:
            raise TrainingBootstrapError(label + "-failed")

    def download_existing(version):
        dependencies = scratch / "kagglehub_tools"
        command([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir",
                 "--only-binary", ":all:", "--target", str(dependencies), "kagglehub==1.0.2"], "spatial_transport_dependencies", 180)
        helper = cloud_download_helper_source(version, dependencies, scratch / "pilot_cloud")
        command([sys.executable, "-c", helper], "spatial_pilot_transport", 240)
        value = json.loads((output / "spatial_pilot_transport.log").read_text().strip())
        if value.get("status") != "PASS" or value.get("sha256") != PILOT_ARCHIVE_SHA256:
            raise TrainingBootstrapError("spatial-existing-pilot-transport-invalid")
        return value["path"]

    try:
        stage("frozen-existing-pilot")
        archive, transport = resolve_frozen_pilot(Path("/kaggle/input"), PILOT_ARCHIVE_SHA256,
                                                 download_version=PILOT_DOWNLOAD_VERSION, cloud_download=download_existing)
        manifest = safe_extract_pilot(archive, scratch / "pilot")
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != PILOT_MANIFEST_SHA256:
            raise TrainingBootstrapError("spatial-pilot-manifest-mismatch")
        pilot = json.loads(manifest.read_text())
        report["pilot_transport"] = transport
        stage("materialize-reviewed-sources")
        project = scratch / "project"
        project.mkdir()
        for relative, source in PROJECT_SOURCES.items():
            target = project / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.encode())
        report["runtime_sources"] = export_runtime_sources(project, output, PROJECT_SOURCES)
        stage("pinned-cuda-runtime")
        env = os.environ.copy()
        env.update(UV_CACHE_DIR=str(scratch / "uv_cache"), UV_PYTHON_INSTALL_DIR=str(scratch / "uv_python"),
                   UV_NO_PROGRESS="1", UV_LINK_MODE="copy", TORCH_HOME=str(scratch / "torch_cache"),
                   CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4")
        uv = shutil.which("uv")
        if not uv:
            tools = scratch / "uv_tools"
            command([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--target", str(tools), "uv==0.8.22"], "spatial_uv", 180, env)
            uv = str(tools / "bin" / "uv")
        command([uv, "python", "install", "3.12"], "spatial_python", 180, env)
        venv = scratch / "venv"
        command([uv, "venv", "--python", "3.12", "--managed-python", str(venv)], "spatial_venv", 60, env)
        python = str(venv / "bin" / "python")
        command([uv, "pip", "install", "--python", python, "--only-binary", ":all:",
                 "--index-url", "https://download.pytorch.org/whl/cu126", "torch==2.7.1", "torchvision==0.22.1"], "spatial_torch", 900, env)
        command([uv, "pip", "install", "--python", python, "--only-binary", ":all:", "numpy==2.2.6",
                 "opencv-python-headless==4.12.0.88", "Pillow==11.3.0"], "spatial_dependencies", 180, env)
        stage("actual-t4-three-arm-training-and-dev")
        smoke = "import torch; assert torch.cuda.is_available(); assert 'T4' in torch.cuda.get_device_name(0); print(torch.cuda.get_device_name(0))"
        command([python, "-c", smoke], "spatial_cuda_smoke", 60, env)
        args = [python, str(project / "scripts/train_spatial_waypoints.py"), "--manifest", str(manifest),
                "--output-dir", str(output / "experiment"), "--device", "cuda", "--max-epochs", str(MAX_EPOCHS)]
        if FROZEN_BASELINE is not None:
            baseline_path = project / "frozen_baseline.json"
            baseline_path.write_text(FROZEN_BASELINE,encoding="utf-8")
            args.extend(("--reused-baseline",str(baseline_path)))
            report["frozen_baseline_sha256"] = hashlib.sha256(FROZEN_BASELINE.encode()).hexdigest()
        command(args,"spatial_training",TRAINING_SECONDS,env)
        comparison = json.loads((output / "experiment/comparison.json").read_text())
        report["experiment"] = validate_spatial_report(comparison, pilot["split_counts"]["train"], pilot["split_counts"]["dev"])
        report["status"] = "PASS"
        stage("complete")
    except Exception as error:
        report["error"] = {"type": type(error).__name__,
                           "code": str(error) if isinstance(error, TrainingBootstrapError) else "spatial-bootstrap-operation-failed"}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        if current is not None and current.poll() is None:
            os.killpg(current.pid, signal.SIGTERM)
            try:
                current.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(current.pid, signal.SIGKILL)
                current.wait(timeout=10)
        report["training_process_stopped"] = current is None or current.poll() is not None
        report["elapsed_seconds"] = time.monotonic() - started
        try:
            report["artifacts"] = preserve_spatial_artifacts(output)
        except Exception as error:
            report.update(status="FAIL", archive_error=type(error).__name__)
        (output / "kaggle_spatial_bootstrap.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 1
