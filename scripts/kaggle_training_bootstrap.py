"""Bounded, provenance-checked training in the existing Kaggle T4 notebook.

The packager supplies source bytes and immutable pilot hashes. Importing this
module neither starts a remote job nor installs dependencies.
"""

from __future__ import annotations

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
from pathlib import Path, PurePosixPath

PROJECT_SOURCES = {}
PILOT_ARCHIVE_SHA256 = ""
PILOT_MANIFEST_SHA256 = ""
PILOT_DOWNLOAD_VERSION = None
PILOT_NOTEBOOK_ID = "dongmyungpark/vlf-carla-rgb-smoke-20261005"
MAX_SECONDS = 5400


class TrainingBootstrapError(RuntimeError):
    pass


def resolve_local_pilot_archive(path: Path, expected_sha256: str, output_root: Path) -> Path:
    """Accept only verified bytes produced in this same owned working directory."""
    path = Path(path)
    if (path.name != "pilot_artifacts.zip" or path.is_symlink() or not path.is_file() or
            path.parent.resolve() != output_root.resolve()):
        raise TrainingBootstrapError("same-session-pilot-archive-location-invalid")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise TrainingBootstrapError("same-session-pilot-archive-sha256-mismatch")
    return path


def resolve_frozen_pilot(input_root: Path, expected_sha256: str, *, download_version=None,
                         cloud_download=None) -> tuple[Path, str]:
    """Locate exact existing cloud bytes; no dataset creation or local upload."""
    candidates = list(input_root.rglob("pilot_artifacts.zip"))
    matches = [path for path in candidates if not path.is_symlink() and path.is_file()
               and hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256]
    if len(matches) > 1:
        raise TrainingBootstrapError("frozen-pilot-archive-not-unique-or-not-mounted")
    if matches:
        return matches[0], "mounted-output"
    if download_version is None or cloud_download is None:
        raise TrainingBootstrapError("frozen-pilot-archive-not-unique-or-not-mounted")
    if type(download_version) is not int or download_version <= 0:
        raise TrainingBootstrapError("pilot-download-version-must-be-positive-integer")
    path = Path(cloud_download(download_version))
    if (path.name != "pilot_artifacts.zip" or path.is_symlink() or not path.is_file() or
            hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256):
        raise TrainingBootstrapError("existing-cloud-pilot-archive-sha256-mismatch")
    return path, "kagglehub-existing-notebook-output"


def cloud_download_helper_source(version: int, dependency_path: Path, output_path: Path) -> str:
    """Use Kaggle's platform authentication without exposing credentials/log URLs."""
    if type(version) is not int or version <= 0:
        raise TrainingBootstrapError("pilot-download-version-must-be-positive-integer")
    handle = PILOT_NOTEBOOK_ID + "/versions/" + str(version)
    return ("import contextlib,hashlib,io,json,logging,sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0," + repr(str(dependency_path)) + ")\n"
            "logging.disable(logging.CRITICAL)\n"
            "try:\n"
            " with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):\n"
            "  import kagglehub\n"
            "  result=kagglehub.notebook_output_download(" + repr(handle) + ",path='pilot_artifacts.zip',output_dir=" + repr(str(output_path)) + ")\n"
            " path=Path(result)\n"
            " print(json.dumps({'status':'PASS','path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'kagglehub':kagglehub.__version__}))\n"
            "except BaseException as error:\n"
            " print(json.dumps({'status':'FAIL','exception_type':type(error).__name__}))\n"
            " raise SystemExit(1)\n")


def safe_extract_pilot(archive_path: Path, target: Path) -> Path:
    """Extract only a bounded private pilot archive, rejecting paths and links."""
    target.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 12 * 1024**3:
            raise TrainingBootstrapError("pilot-archive-expanded-size-exceeded")
        seen = set()
        for item in archive.infolist():
            relative = PurePosixPath(item.filename)
            canonical = tuple(relative.parts)
            if (relative.is_absolute() or "\\" in item.filename or ":" in item.filename or
                    any(part in ("..", ".") for part in relative.parts) or
                    not canonical or (item.external_attr >> 16) & 0o170000 == 0o120000 or canonical in seen):
                raise TrainingBootstrapError("pilot-archive-unsafe-member")
            seen.add(canonical)
            destination = target.joinpath(*relative.parts)
            if not destination.resolve().is_relative_to(target.resolve()):
                raise TrainingBootstrapError("pilot-archive-path-escape")
        archive.extractall(target)
    matches = list(target.rglob("pilot_manifest.json"))
    if len(matches) != 1:
        raise TrainingBootstrapError("pilot-manifest-not-unique")
    return matches[0]


def export_runtime_sources(project: Path, output: Path, sources: dict) -> dict:
    manifest = {}
    with zipfile.ZipFile(output / "training_runtime_sources.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for relative, text in sorted(sources.items()):
            path = project / relative
            data = path.read_bytes()
            if (path.is_symlink() or not path.resolve().is_relative_to(project.resolve()) or
                    hashlib.sha256(data).hexdigest() != hashlib.sha256(text.encode()).hexdigest()):
                raise TrainingBootstrapError("training-runtime-source-mismatch")
            manifest[relative] = hashlib.sha256(data).hexdigest()
            archive.writestr(relative, data)
    report = {"schema_version": 1, "source_sha256": manifest}
    (output / "training_runtime_source_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def preserve_training_artifacts(output: Path):
    artifact = output / "training_artifacts.zip"
    if artifact.exists():
        raise TrainingBootstrapError("training-artifact-already-exists")
    with zipfile.ZipFile(artifact, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for directory in (output / "training", output / "evaluation", output / "training_initial"):
            if directory.exists():
                for path in sorted(directory.rglob("*")):
                    if (path.is_file() and not path.is_symlink() and
                            not (directory.name == "training_initial" and path.suffix == ".pt")):
                        archive.write(path, path.relative_to(output))


def validate_resume_proof(initial: dict, train: dict, checkpoint_sha256: str) -> dict:
    resume = train.get("resume", {})
    if (initial.get("status") != "PASS" or initial.get("completed_epochs") != 2 or
            resume.get("start_epoch") != 2 or resume.get("checkpoint_sha256") != checkpoint_sha256 or
            not all(resume.get(key) is True for key in ("optimizer_state_restored", "scaler_state_restored",
                                                      "rng_state_restored", "sampler_state_restored")) or
            train.get("runtime", {}).get("device") != "cuda" or
            train.get("runtime", {}).get("amp_enabled") is not True or
            train.get("history", [])[:2] != initial.get("history") or train.get("completed_epochs", 0) <= 2):
        raise TrainingBootstrapError("actual-cuda-resume-proof-failed")
    return dict(resume, device="cuda", amp_enabled=True,
                completed_epochs=train["completed_epochs"], history_prefix_preserved=True)


def main(local_archive: Path | None = None):
    output = Path("/kaggle/working")
    started = time.monotonic()
    scratch = Path(tempfile.mkdtemp(prefix="vlf_learning_", dir="/tmp"))
    report = {"schema_version": 1, "status": "FAIL", "stage": "initializing",
              "pilot_archive_sha256": PILOT_ARCHIVE_SHA256,
              "pilot_manifest_sha256": PILOT_MANIFEST_SHA256, "max_seconds": MAX_SECONDS,
              "gpu_job_owned": True, "training_process_stopped": False, "stages": []}
    current = None
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise TrainingBootstrapError("training-total-time-limit")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(MAX_SECONDS)

    def stage(name):
        report["stage"] = name
        report["stages"].append({"name": name, "elapsed_seconds": time.monotonic() - started})
        print("Waypoint learning:", name, flush=True)

    def command(args, label, timeout=600, env=None):
        nonlocal current
        remaining = min(timeout, MAX_SECONDS - (time.monotonic() - started))
        if remaining <= 0:
            raise TrainingBootstrapError("training-total-time-limit")
        with (output / (label + ".log")).open("w", encoding="utf-8") as log:
            current = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            code = current.wait(timeout=remaining)
            current = None
        if code:
            raise TrainingBootstrapError(label + "-failed")

    def download_existing_output(version):
        stage("explicit-existing-notebook-version")
        dependencies = scratch / "kagglehub_tools"
        # Isolate this resolver dependency from the host Kaggle environment;
        # platform authentication is supplied internally by KaggleHub.
        command([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                 "--no-cache-dir", "--only-binary", ":all:", "--target", str(dependencies),
                 "kagglehub==1.0.2"], "learning_pilot_transport_dependencies", 180)
        helper = cloud_download_helper_source(version, dependencies, scratch / "pilot_cloud_download")
        command([sys.executable, "-c", helper], "learning_pilot_cloud_transport", 120)
        result = json.loads((output / "learning_pilot_cloud_transport.log").read_text().strip())
        if result.get("status") != "PASS" or result.get("sha256") != PILOT_ARCHIVE_SHA256:
            raise TrainingBootstrapError("existing-cloud-pilot-transport-not-verified")
        report["cloud_pilot_transport"] = {"notebook_id": PILOT_NOTEBOOK_ID,
                                           "requested_version": version, "kagglehub": result["kagglehub"],
                                           "archive_sha256": result["sha256"], "platform_authentication": True}
        return result["path"]

    try:
        if len(PILOT_ARCHIVE_SHA256) != 64 or len(PILOT_MANIFEST_SHA256) != 64:
            raise TrainingBootstrapError("pilot-expected-hashes-missing")
        stage("find-frozen-pilot")
        if local_archive is not None:
            archive = resolve_local_pilot_archive(local_archive, PILOT_ARCHIVE_SHA256, output)
            transport = "same-session-verified-archive"
        else:
            archive, transport = resolve_frozen_pilot(Path("/kaggle/input"), PILOT_ARCHIVE_SHA256,
                                                     download_version=PILOT_DOWNLOAD_VERSION,
                                                     cloud_download=download_existing_output)
        report["pilot_transport"] = transport
        manifest = safe_extract_pilot(archive, scratch / "pilot_source")
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != PILOT_MANIFEST_SHA256:
            raise TrainingBootstrapError("frozen-pilot-manifest-sha256-mismatch")
        pilot = json.loads(manifest.read_text(encoding="utf-8"))
        if (pilot.get("scope") != "moving-rgb-waypoint-pilot" or pilot.get("status") != "COMPLETE" or
                pilot.get("sample_count", 0) < 2000):
            raise TrainingBootstrapError("pilot-incomplete-or-too-small")
        report["pilot_sample_count"] = pilot["sample_count"]
        stage("materialize-public-sources")
        project = scratch / "project"
        project.mkdir()
        for relative, content in PROJECT_SOURCES.items():
            target = project / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="\n")
        report["runtime_sources"] = export_runtime_sources(project, output, PROJECT_SOURCES)
        stage("install-pinned-cuda-runtime")
        env = os.environ.copy()
        env.update(UV_CACHE_DIR=str(scratch / "uv_cache"), UV_PYTHON_INSTALL_DIR=str(scratch / "uv_python"),
                   UV_NO_PROGRESS="1", UV_LINK_MODE="copy", TORCH_HOME=str(scratch / "torch_cache"),
                   CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4")
        uv = shutil.which("uv")
        if not uv:
            tools = scratch / "uv_tools"
            command([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                     "--no-cache-dir", "--target", str(tools), "uv==0.8.22"], "learning_uv", 180, env)
            uv = str(tools / "bin" / "uv")
        command([uv, "python", "install", "3.12"], "learning_python", 180, env)
        venv = scratch / "venv"
        command([uv, "venv", "--python", "3.12", "--managed-python", str(venv)], "learning_venv", 60, env)
        python = str(venv / "bin" / "python")
        command([uv, "pip", "install", "--python", python, "--only-binary", ":all:",
                 "--index-url", "https://download.pytorch.org/whl/cu126",
                 "torch==2.7.1", "torchvision==0.22.1"], "learning_torch", 900, env)
        command([uv, "pip", "install", "--python", python, "--only-binary", ":all:",
                 "numpy==2.2.6", "opencv-python-headless==4.12.0.88", "Pillow==11.3.0"],
                "learning_dependencies", 180, env)
        smoke = ("import json,torch,torchvision,numpy,cv2,PIL;"
                 "assert torch.__version__.split('+')[0]=='2.7.1';"
                 "assert torchvision.__version__.split('+')[0]=='0.22.1';"
                 "assert torch.cuda.is_available();"
                 "assert 'T4' in torch.cuda.get_device_name(0);"
                 "x=torch.randn(8,8,device='cuda',requires_grad=True);(x*x).sum().backward();"
                 "torch.cuda.synchronize();"
                 "print(json.dumps({'torch':torch.__version__,'torchvision':torchvision.__version__,"
                 "'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'numpy':numpy.__version__,"
                 "'opencv':cv2.__version__,'pillow':PIL.__version__,'backward_verified':True}))")
        command([python, "-c", smoke], "learning_cuda_smoke", 60, env)
        report["cuda_smoke"] = json.loads((output / "learning_cuda_smoke.log").read_text().strip().splitlines()[-1])
        stage("pretrained-rgb-training")
        command([python, str(project / "scripts/train_waypoints.py"), "--manifest", str(manifest),
                 "--output-dir", str(output / "training_initial"), "--device", "cuda", "--max-epochs", "2"],
                "waypoint_training_initial", MAX_SECONDS, env)
        initial = json.loads((output / "training_initial/training_report.json").read_text())
        if initial.get("status") != "PASS" or initial.get("completed_epochs") != 2:
            raise TrainingBootstrapError("initial-gpu-training-not-two-complete-epochs")
        resume_path = output / "training_initial/last_checkpoint.pt"
        resume_hash = hashlib.sha256(resume_path.read_bytes()).hexdigest()
        stage("cuda-checkpoint-resume")
        command([python, str(project / "scripts/train_waypoints.py"), "--manifest", str(manifest),
                 "--output-dir", str(output / "training"), "--device", "cuda", "--resume", str(resume_path)],
                "waypoint_training", MAX_SECONDS, env)
        train = json.loads((output / "training/training_report.json").read_text())
        if train.get("status") != "PASS" or train.get("overfit_check", {}).get("status") != "PASS":
            raise TrainingBootstrapError("training-report-not-passed")
        report["resume_proof"] = validate_resume_proof(initial, train, resume_hash)
        stage("held-out-development-evaluation")
        command([python, str(project / "scripts/evaluate_waypoints.py"), "--manifest", str(manifest),
                 "--checkpoint", str(output / "training/checkpoint.pt"),
                 "--output-dir", str(output / "evaluation"), "--device", "cuda"],
                "waypoint_evaluation", 900, env)
        evaluation = json.loads((output / "evaluation/metrics.json").read_text())
        if evaluation.get("status") != "PASS" or evaluation.get("sample_count") != pilot.get("split_counts", {}).get("dev"):
            raise TrainingBootstrapError("evaluation-report-not-passed-or-count-mismatch")
        report["training"] = {key: train[key] for key in ("completed_epochs", "best_dev_loss", "checkpoint_sha256")}
        report["evaluation"] = {key: evaluation[key] for key in ("sample_count", "performance_target_passed", "metrics")}
        report["status"] = "PASS"
        stage("complete")
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": str(error)}
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
            preserve_training_artifacts(output)
        except Exception as error:
            report["status"] = "FAIL"
            report["archive_error"] = type(error).__name__
        (output / "kaggle_training_bootstrap.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1
