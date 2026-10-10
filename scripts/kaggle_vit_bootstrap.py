"""Bounded T4-only DINOv3 training without a local GPU or CARLA server.

The packager embeds the original training bootstrap first, retaining its safe
archive and authenticated existing-output helpers. This module also imports
those helpers normally when used in local CPU-only package tests.
"""

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

MAX_SECONDS = 2400
TRAINING_SECONDS = 1200
MAX_EPOCHS = 32
PRETRAINED_REVISION = "c6a5fb7d12bbd3cf3b0079253141c3332aaed7da"
PRETRAINED_URL = ("https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m/resolve/"
                  + PRETRAINED_REVISION + "/pytorch_model.bin")
PRETRAINED_SHA256 = "ed8a1265847086c913535fe7d84c0156a68dc98c0edd52298b8bb4bb7bdcbbb1"
PRETRAINED_BYTES = 342624338


def vit_cloud_download_helper_source(version: int, dependency_path: Path, output_path: Path) -> str:
    """Resolve an existing version, retaining only allowlisted failure details."""
    if type(version) is not int or version <= 0:
        raise TrainingBootstrapError("pilot-download-version-must-be-positive-integer")
    handle = PILOT_NOTEBOOK_ID + "/versions/" + str(version)
    return ("import contextlib,hashlib,io,json,logging,sys\nfrom pathlib import Path\n"
            + "sys.path.insert(0," + repr(str(dependency_path)) + ")\n"
            + "logging.disable(logging.CRITICAL)\n"
            + "try:\n"
            + " with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):\n"
            + "  import kagglehub\n"
            + "  result=kagglehub.notebook_output_download(" + repr(handle)
            + ",path='pilot_artifacts.zip',output_dir=" + repr(str(output_path)) + ")\n"
            + " path=Path(result)\n"
            + " print(json.dumps({'status':'PASS','path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'kagglehub':kagglehub.__version__}))\n"
            + "except BaseException as error:\n"
            + " report={'status':'FAIL','exception_type':type(error).__name__}\n"
            + " status=getattr(getattr(error,'response',None),'status_code',None)\n"
            + " if type(status) is int and 100<=status<=599: report['http_status']=status\n"
            + " code=getattr(error,'error_code',None)\n"
            + " if type(code) is int and 0<=code<=1000000: report['backend_error_code']=code\n"
            + " message=str(error).lower()\n"
            + " if 'result.mountslug' in message and 'missing' in message: report['reason']='missing-mount-slug'\n"
            + " elif 'permission' in message or 'access denied' in message: report['reason']='access-denied'\n"
            + " elif 'not found' in message: report['reason']='resource-not-found'\n"
            + " elif 'authenticate' in message or 'authentication' in message: report['reason']='authentication-required'\n"
            + " else: report['reason']='existing-output-resolution-failed'\n"
            + " print(json.dumps(report))\n"
            + " raise SystemExit(1)\n")


def pretrained_download_helper_source(destination: Path) -> str:
    """Download exact public pretrained bytes and suppress sensitive errors."""
    return ("import hashlib,json,os,time,urllib.request\nfrom pathlib import Path\n"
            + "target=Path(" + repr(str(destination)) + ")\n"
            + "temporary=target.with_name(target.name+'.partial')\n"
            + "try:\n"
            + " if target.exists() or temporary.exists(): raise ValueError('output-exists')\n"
            + " digest=hashlib.sha256(); count=0; began=time.monotonic()\n"
            + " request=urllib.request.Request(" + repr(PRETRAINED_URL) + ",headers={'User-Agent':'Vision-Lane-Following/1'})\n"
            + " with urllib.request.urlopen(request,timeout=30) as response,temporary.open('xb') as stream:\n"
            + "  if response.status != 200: raise ValueError('http-status')\n"
            + "  while True:\n"
            + "   block=response.read(1024*1024)\n"
            + "   if not block: break\n"
            + "   count+=len(block)\n"
            + "   if count>" + repr(PRETRAINED_BYTES) + " or time.monotonic()-began>240: raise ValueError('download-bound')\n"
            + "   digest.update(block); stream.write(block)\n"
            + "  stream.flush(); os.fsync(stream.fileno())\n"
            + " if count!=" + repr(PRETRAINED_BYTES) + " or digest.hexdigest()!=" + repr(PRETRAINED_SHA256) + ": raise ValueError('hash-or-size')\n"
            + " os.link(temporary,target); temporary.unlink()\n"
            + " print(json.dumps({'status':'PASS','sha256':digest.hexdigest(),'bytes':count,'revision':" + repr(PRETRAINED_REVISION) + "}))\n"
            + "except BaseException as error:\n"
            + " if temporary.exists(): temporary.unlink()\n"
            + " print(json.dumps({'status':'FAIL','exception_type':type(error).__name__}))\n"
            + " raise SystemExit(1)\n")


def validate_cuda_smoke(value: dict) -> dict:
    if (not isinstance(value, dict) or value.get("cuda_available") is not True
            or not isinstance(value.get("gpu"), str) or "T4" not in value["gpu"]
            or value.get("precision") != "float16" or value.get("backward_verified") is not True
            or value.get("torch", "").split("+")[0] != "2.7.1"
            or value.get("torchvision", "").split("+")[0] != "0.22.1"
            or value.get("timm") != "1.0.20"):
        raise TrainingBootstrapError("vit-t4-fp16-runtime-proof-failed")
    return dict(value)


def validate_experiment_reports(train: dict, evaluation: dict, pilot: dict) -> dict:
    expected = pilot.get("split_counts", {}).get("dev")
    if (train.get("status") != "PASS" or type(train.get("completed_epochs")) is not int
            or train["completed_epochs"] < 1 or evaluation.get("status") != "PASS"
            or evaluation.get("split") != "dev" or evaluation.get("final_town_used") is not False
            or type(expected) is not int or expected <= 0 or evaluation.get("sample_count") != expected
            or evaluation.get("manifest_sha256") != PILOT_MANIFEST_SHA256):
        raise TrainingBootstrapError("vit-training-or-complete-dev-evaluation-proof-failed")
    runtime, provenance = train.get("runtime", {}), train.get("pretrained_provenance", {})
    if (runtime.get("device") != "cuda" or "T4" not in str(runtime.get("gpu", ""))
            or runtime.get("encoder_amp_fp16") is not True or runtime.get("head_precision") != "float32"
            or train.get("training_protocol") != "frozen-feature-head-only"
            or train.get("fixture_injection") is not False or evaluation.get("fixture_injection") is not False
            or train.get("final_town_used") is not False
            or provenance.get("pretrained") is not True or provenance.get("hash_verified") is not True
            or provenance.get("sha256") != PRETRAINED_SHA256 or provenance.get("url") != PRETRAINED_URL
            or train.get("pretrained_revision") != PRETRAINED_REVISION
            or evaluation.get("pretrained_revision") != PRETRAINED_REVISION
            or evaluation.get("pretrained_provenance") != provenance
            or train.get("manifest_sha256") != PILOT_MANIFEST_SHA256
            or train.get("source_sha256") != evaluation.get("source_sha256")
            or not train.get("checkpoint_sha256")
            or train.get("checkpoint_sha256") != evaluation.get("checkpoint_sha256")):
        raise TrainingBootstrapError("vit-actual-frozen-cuda-model-provenance-proof-failed")
    cache = train.get("feature_cache", {})
    for split in ("train", "dev"):
        value = cache.get(split, {})
        if (value.get("source") != "RGB-only" or value.get("encoder_frozen") is not True
                or value.get("encoder_amp_fp16") is not True):
            raise TrainingBootstrapError("vit-frozen-rgb-feature-cache-proof-failed")
    overfit = train.get("overfit_check", {})
    if (overfit.get("status") != "PASS" or overfit.get("encoder_frozen") is not True
            or overfit.get("head_reset_to_fresh_initialization") is not True
            or evaluation.get("cached_feature_prediction_check", {}).get("checked") is not True):
        raise TrainingBootstrapError("vit-overfit-reset-or-full-rgb-evaluation-proof-failed")
    # Numerical success and accuracy are distinct. A valid experiment may fail
    # its unchanged development accuracy targets, and that result is retained.
    if type(evaluation.get("performance_target_passed")) is not bool:
        raise TrainingBootstrapError("vit-development-performance-result-missing")
    return {"completed_epochs": train["completed_epochs"], "sample_count": expected,
            "performance_target_passed": evaluation["performance_target_passed"],
            "metrics": evaluation.get("metrics", [])}


def preserve_vit_artifacts(output: Path) -> dict:
    """Export reports, predictions and checkpoints while keeping caches private."""
    artifact = output / "training_artifacts.zip"
    if artifact.exists():
        raise TrainingBootstrapError("vit-artifact-already-exists")
    manifest = {}
    directory = output / "experiment"
    if directory.is_symlink():
        raise TrainingBootstrapError("vit-artifact-directory-is-symlink")
    with zipfile.ZipFile(artifact, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                relative = path.relative_to(directory)
                if (not path.is_file() or path.is_symlink()
                        or any(part in ("cache", "features", "feature_cache") for part in relative.parts)
                        or path.suffix not in (".json", ".jsonl", ".csv", ".pt", ".png")):
                    continue
                if not path.resolve().is_relative_to(directory.resolve()):
                    raise TrainingBootstrapError("vit-artifact-path-escape")
                data = path.read_bytes()
                name = str(path.relative_to(output).as_posix())
                manifest[name] = hashlib.sha256(data).hexdigest()
                archive.writestr(name, data)
    return {"filename": artifact.name, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "source_sha256": manifest, "feature_caches_exported": False}


def main_vit():
    output = Path("/kaggle/working")
    started = time.monotonic()
    scratch = Path(tempfile.mkdtemp(prefix="vlf_vit_", dir="/tmp"))
    report = {"schema_version": 1, "status": "FAIL", "stage": "initializing",
              "workload": "frozen-dinov3-rgb-waypoint-comparison",
              "max_seconds": MAX_SECONDS, "training_seconds": TRAINING_SECONDS,
              "pilot_archive_sha256": PILOT_ARCHIVE_SHA256,
              "pilot_manifest_sha256": PILOT_MANIFEST_SHA256,
              "pilot_mount_version_policy": "latest-existing-output-accepted-only-with-frozen-archive-and-manifest-hashes",
              "local_gpu_used": False, "carla_server_started": False,
              "training_process_stopped": False, "stages": []}
    current = None
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise TrainingBootstrapError("vit-total-time-limit")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(MAX_SECONDS)

    def stage(name):
        report["stage"] = name
        report["stages"].append({"name": name, "elapsed_seconds": time.monotonic() - started})
        print("ViT waypoint experiment:", name, flush=True)

    def command(args, label, timeout=600, env=None):
        nonlocal current
        remaining = min(timeout, MAX_SECONDS - (time.monotonic() - started))
        if remaining <= 0:
            raise TrainingBootstrapError("vit-total-time-limit")
        with (output / (label + ".log")).open("x", encoding="utf-8") as log:
            current = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            code = current.wait(timeout=remaining)
            current = None
        if code:
            raise TrainingBootstrapError(label + "-failed")

    def download_existing_output(version):
        dependencies = scratch / "kagglehub_tools"
        command([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                 "--no-cache-dir", "--only-binary", ":all:", "--target", str(dependencies),
                 "kagglehub==1.0.2"], "vit_pilot_transport_dependencies", 180)
        helper = vit_cloud_download_helper_source(version, dependencies, scratch / "pilot_cloud_download")
        command([sys.executable, "-c", helper], "vit_pilot_cloud_transport", 240)
        value = json.loads((output / "vit_pilot_cloud_transport.log").read_text().strip())
        if value.get("status") != "PASS" or value.get("sha256") != PILOT_ARCHIVE_SHA256:
            raise TrainingBootstrapError("existing-cloud-pilot-transport-not-verified")
        report["cloud_pilot_transport"] = {"notebook_id": PILOT_NOTEBOOK_ID,
                                          "requested_version": version,
                                          "archive_sha256": value["sha256"],
                                          "kagglehub": value["kagglehub"],
                                          "platform_authentication": True}
        return value["path"]

    try:
        if len(PILOT_ARCHIVE_SHA256) != 64 or len(PILOT_MANIFEST_SHA256) != 64:
            raise TrainingBootstrapError("pilot-expected-hashes-missing")
        stage("frozen-pilot")
        archive, transport = resolve_frozen_pilot(Path("/kaggle/input"), PILOT_ARCHIVE_SHA256,
                                                 download_version=PILOT_DOWNLOAD_VERSION,
                                                 cloud_download=download_existing_output)
        report["pilot_transport"] = transport
        manifest = safe_extract_pilot(archive, scratch / "pilot_source")
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != PILOT_MANIFEST_SHA256:
            raise TrainingBootstrapError("frozen-pilot-manifest-sha256-mismatch")
        pilot = json.loads(manifest.read_text(encoding="utf-8"))
        if (pilot.get("scope") != "moving-rgb-waypoint-pilot" or pilot.get("status") != "COMPLETE"
                or pilot.get("split_counts", {}).get("train", 0) <= 0
                or pilot.get("split_counts", {}).get("dev", 0) <= 0):
            raise TrainingBootstrapError("pilot-training-or-development-split-missing")
        stage("materialize-public-sources")
        project = scratch / "project"
        project.mkdir()
        for relative, content in PROJECT_SOURCES.items():
            target = project / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.encode("utf-8"))
        report["runtime_sources"] = export_runtime_sources(project, output, PROJECT_SOURCES)
        stage("install-pinned-cuda-runtime")
        env = os.environ.copy()
        env.update(UV_CACHE_DIR=str(scratch / "uv_cache"), UV_PYTHON_INSTALL_DIR=str(scratch / "uv_python"),
                   UV_NO_PROGRESS="1", UV_LINK_MODE="copy", TORCH_HOME=str(scratch / "torch_cache"),
                   CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4",
                   HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
        uv = shutil.which("uv")
        if not uv:
            tools = scratch / "uv_tools"
            command([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                     "--no-cache-dir", "--target", str(tools), "uv==0.8.22"], "vit_uv", 180, env)
            uv = str(tools / "bin" / "uv")
        command([uv, "python", "install", "3.12"], "vit_python", 180, env)
        venv = scratch / "venv"
        command([uv, "venv", "--python", "3.12", "--managed-python", str(venv)], "vit_venv", 60, env)
        python = str(venv / "bin" / "python")
        command([uv, "pip", "install", "--python", python, "--only-binary", ":all:",
                 "--index-url", "https://download.pytorch.org/whl/cu126",
                 "torch==2.7.1", "torchvision==0.22.1"], "vit_torch", 900, env)
        command([uv, "pip", "install", "--python", python, "--only-binary", ":all:",
                 "numpy==2.2.6", "opencv-python-headless==4.12.0.88", "Pillow==11.3.0", "timm==1.0.20"],
                "vit_dependencies", 180, env)
        smoke = ("import json,torch,torchvision,timm,numpy,cv2,PIL;"
                 "assert torch.cuda.is_available();assert 'T4' in torch.cuda.get_device_name(0);"
                 "x=torch.randn(32,32,device='cuda',requires_grad=True);"
                 "w=torch.randn(32,32,device='cuda',requires_grad=True);"
                 "ctx=torch.autocast(device_type='cuda',dtype=torch.float16);ctx.__enter__();"
                 "y=x@w;assert y.dtype==torch.float16;loss=y.float().square().mean();ctx.__exit__(None,None,None);"
                 "loss.backward();torch.cuda.synchronize();"
                 "print(json.dumps({'torch':torch.__version__,'torchvision':torchvision.__version__,'timm':timm.__version__,"
                 "'gpu':torch.cuda.get_device_name(0),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,"
                 "'cuda_available':True,'cuda':torch.version.cuda,'numpy':numpy.__version__,'opencv':cv2.__version__,"
                 "'pillow':PIL.__version__,'precision':'float16','backward_verified':True}))")
        command([python, "-c", smoke], "vit_cuda_smoke", 60, env)
        report["cuda_smoke"] = validate_cuda_smoke(json.loads((output / "vit_cuda_smoke.log").read_text().strip().splitlines()[-1]))
        stage("pinned-public-pretrained-weights")
        weights = scratch / "dinov3_pretrained.bin"
        command([python, "-c", pretrained_download_helper_source(weights)], "vit_pretrained_download", 300, env)
        downloaded = json.loads((output / "vit_pretrained_download.log").read_text().strip())
        if (downloaded.get("status") != "PASS" or downloaded.get("sha256") != PRETRAINED_SHA256
                or downloaded.get("bytes") != PRETRAINED_BYTES):
            raise TrainingBootstrapError("vit-pretrained-download-proof-failed")
        report["pretrained"] = downloaded
        stage("frozen-feature-training-and-full-dev-evaluation")
        experiment = output / "experiment"
        command([python, str(project / "scripts/train_vit_waypoints.py"), "--manifest", str(manifest),
                 "--output-dir", str(experiment), "--device", "cuda", "--weights", str(weights),
                 "--expected-weights-sha256", PRETRAINED_SHA256, "--weights-url", PRETRAINED_URL,
                 "--max-seconds", str(TRAINING_SECONDS), "--max-epochs", str(MAX_EPOCHS)],
                "vit_training_and_evaluation", MAX_SECONDS, env)
        train = json.loads((experiment / "training_report.json").read_text())
        evaluation = json.loads((experiment / "evaluation/metrics.json").read_text())
        report["experiment"] = validate_experiment_reports(train, evaluation, pilot)
        report["status"] = "PASS"
        stage("complete")
    except Exception as error:
        report["error"] = {"type": type(error).__name__,
                           "code": str(error) if isinstance(error, TrainingBootstrapError) else "vit-bootstrap-operation-failed"}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
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
            report["artifacts"] = preserve_vit_artifacts(output)
        except Exception as error:
            report["status"] = "FAIL"
            report["archive_error"] = type(error).__name__
        (output / "kaggle_vit_bootstrap.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1
