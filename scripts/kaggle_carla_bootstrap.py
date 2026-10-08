"""Bounded CARLA RGB experiment for an ephemeral Kaggle GPU container.

The packager supplies both source constants, then calls main(). This module
does not start jobs, install packages, or launch CARLA merely by being imported.
Only existing NVIDIA driver files are used; no host driver is installed.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import json
import os
import platform
import pwd
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

PROBE_SOURCE = ""
VULKAN_CHILD_SOURCE = ""
PROJECT_SOURCES = {}
WORKLOAD = "rgb"
CARLA_URL = "https://downloads.carlasim.com/Linux/CARLA_0.9.16.tar.gz"
CARLA_MIRROR = "https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.16.tar.gz"
CARLA_PACKAGE_BYTES = 8346095504
MAX_SECONDS = 1200


class BootstrapFailure(RuntimeError):
    pass


def main() -> int:
    output = Path("/kaggle/working")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + MAX_SECONDS
    scratch = Path(tempfile.mkdtemp(prefix="vlf_carla_", dir="/tmp"))
    scratch.chmod(0o755)
    report = {"schema_version": 1, "status": "FAIL", "stage": "initializing",
              "carla_rgb_verified": False, "python": platform.python_version(),
              "stages": [], "max_seconds": MAX_SECONDS}
    process = None
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise BootstrapFailure("total-time-limit")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(MAX_SECONDS)

    def remaining(limit: float) -> float:
        value = min(limit, deadline - time.monotonic())
        if value <= 0:
            raise BootstrapFailure("total-time-limit")
        return value

    def stage(name: str):
        report["stage"] = name
        report["stages"].append({"name": name, "elapsed_seconds": round(time.monotonic() - started, 2)})
        print("CARLA verification:", name, flush=True)

    def command(args, label, timeout=180, env=None, capture=False):
        log_path = output / (label + ".log")
        if capture:
            result = subprocess.run(args, capture_output=True, text=True,
                                    timeout=remaining(timeout), env=env)
            log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
        else:
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=remaining(timeout), env=env)
        if result.returncode:
            raise BootstrapFailure(label + "-failed")
        return result

    def install_userspace(packages):
        if os.geteuid() != 0 or not shutil.which("apt-get"):
            raise BootstrapFailure("userspace-libraries-unavailable")
        env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
        command(["apt-get", "update", "-qq"], "carla_apt_update", 150, env)
        command(["apt-get", "install", "-y", "--no-install-recommends", *packages],
                "carla_apt_install", 180, env)

    try:
        if sys.platform != "linux" or not PROBE_SOURCE or not VULKAN_CHILD_SOURCE or WORKLOAD not in ("rgb", "week1", "driving"):
            raise BootstrapFailure("linux-and-embedded-probes-required")
        report["workload"] = WORKLOAD
        stage("vulkan-preflight")
        if not ctypes.util.find_library("vulkan"):
            install_userspace(["libvulkan1"])
        vulkan_script = scratch / "vulkan_check.py"
        vulkan_script.write_text(VULKAN_CHILD_SOURCE, encoding="utf-8")
        report["vulkan_attempts"] = []
        report["vulkan"] = {"status": "not_verified", "devices": []}
        env = dict(os.environ)
        for flavor, library in (("glx", "libGLX_nvidia.so.0"), ("egl", "libEGL_nvidia.so.0")):
            candidates = sorted({str(Path(path).resolve()) for directory in (
                "/usr/lib/x86_64-linux-gnu", "/usr/local/nvidia/lib64", "/usr/lib64")
                for path in glob.glob(directory + "/" + library) if Path(path).is_file()})
            if not candidates:
                report["vulkan_attempts"].append({"library": library, "status": "not_mounted"})
            for index, driver in enumerate(candidates):
                attempt = {"library": library, "path": driver}
                report["vulkan_attempts"].append(attempt)
                try:
                    ctypes.CDLL(driver)  # Detect dependencies without replacing this driver.
                    manifest = scratch / f"nvidia_{flavor}_{index}_icd.json"
                    manifest.write_text(json.dumps({"file_format_version": "1.0.0", "ICD": {
                        "library_path": driver, "api_version": "1.3.0"}}), encoding="utf-8")
                    candidate_env = dict(os.environ, VK_ICD_FILENAMES=str(manifest),
                                         VK_DRIVER_FILES=str(manifest), VK_LOADER_DEBUG="error,warn,driver")
                    result = command([sys.executable, str(vulkan_script)],
                                     f"carla_vulkan_{flavor}_{index}", 30, candidate_env, True)
                    attempt["vulkan"] = json.loads(result.stdout.strip().splitlines()[-1])
                    attempt["status"] = attempt["vulkan"].get("status", "invalid_result")
                    report["vulkan"] = attempt["vulkan"]
                    if attempt["status"] == "nvidia_hardware":
                        env = candidate_env
                        report["selected_vulkan_library"] = driver
                        break
                except (BootstrapFailure, OSError, ValueError, IndexError) as error:
                    if str(error) == "total-time-limit":
                        raise
                    attempt["status"] = "driver_probe_failed"
                    attempt["error"] = str(error) if isinstance(error, BootstrapFailure) else type(error).__name__
            if report["vulkan"].get("status") == "nvidia_hardware":
                break
        if report["vulkan"].get("status") != "nvidia_hardware":
            raise BootstrapFailure("nvidia-vulkan-hardware-not-verified")

        stage("python312-client")
        uv = shutil.which("uv")
        env.update(UV_CACHE_DIR=str(scratch / "uv_cache"),
                   UV_PYTHON_INSTALL_DIR=str(scratch / "uv_python"), UV_NO_PROGRESS="1",
                   UV_LINK_MODE="copy")
        if not uv:
            tools = scratch / "uv_tools"
            command([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                     "--no-cache-dir", "--target", str(tools), "uv"], "carla_uv_install", 180, env)
            uv = str(tools / "bin" / "uv")
            if not Path(uv).is_file():
                raise BootstrapFailure("uv-executable-missing")
        command([uv, "python", "install", "3.12"], "carla_python_install", 180, env)
        venv = scratch / "py312"
        command([uv, "venv", "--python", "3.12", "--managed-python", str(venv)],
                "carla_venv", 60, env)
        python = str(venv / "bin" / "python")
        command([uv, "pip", "install", "--python", python, "--only-binary", ":all:",
                 "carla==0.9.16"], "carla_client_install", 180, env)

        stage("server-download")
        if shutil.disk_usage(scratch).free < 35 * 2**30:
            raise BootstrapFailure("scratch-space-below-35gib")
        archive = scratch / "CARLA_0.9.16.tar.gz"
        report["download_attempts"] = []
        response = None
        for url in (CARLA_URL, CARLA_MIRROR):
            attempt = {"url": url}
            report["download_attempts"].append(attempt)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                candidate = urllib.request.urlopen(request, timeout=remaining(60))
            except urllib.error.HTTPError as error:
                attempt.update(status="http_error", http_status=error.code)
                error.close()
                continue
            except urllib.error.URLError as error:
                attempt.update(status="network_error", error_type=type(error.reason).__name__)
                continue
            attempt["http_status"] = candidate.status
            try:
                length = int(candidate.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            attempt["content_length"] = length
            if candidate.status != 200 or length != CARLA_PACKAGE_BYTES:
                attempt["status"] = "unexpected_status_or_length"
                candidate.close()
                continue
            attempt["status"] = "selected"
            report["download_url"] = url
            response = candidate
            break
        if response is None:
            raise BootstrapFailure("official-package-sources-unavailable")
        report["download_bytes"] = 0
        with response:
            received, last_update = 0, time.monotonic()
            with archive.open("wb") as target:
                while True:
                    remaining(60)
                    data = response.read(8 * 2**20)
                    if not data:
                        break
                    target.write(data)
                    received += len(data)
                    report["download_bytes"] = received
                    if received > length:
                        raise BootstrapFailure("package-length-exceeded")
                    if time.monotonic() - last_update > 20:
                        print(f"CARLA download: {received / 2**30:.2f}/{length / 2**30:.2f} GiB", flush=True)
                        last_update = time.monotonic()
            if received != length:
                raise BootstrapFailure("incomplete-package-download")
        report["download_bytes"] = received

        stage("server-extraction")
        server = scratch / "server"
        server.mkdir()
        with tarfile.open(archive, "r:gz") as package:
            for member in package:
                remaining(60)
                package.extract(member, server, filter="data")
        archive.unlink()
        launcher = server / "CarlaUE4.sh"
        binary = server / "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
        if not launcher.is_file() or not binary.is_file():
            raise BootstrapFailure("official-package-layout-mismatch")
        if WORKLOAD in ("week1", "driving"):
            # Pin the unpacked package only; reuse the current Town01 in the
            # session instead of reloading a different map during capture.
            ini = server / "CarlaUE4/Config/DefaultEngine.ini"
            text = ini.read_text(encoding="utf-8")
            pattern = re.compile(r"^([ \t]*GameDefaultMap[ \t]*=[ \t]*)([^\r\n]*)(\r?)$", re.MULTILINE)
            matches = list(pattern.finditer(text))
            official = "/Game/Carla/Maps/Town10HD_Opt.Town10HD_Opt"
            desired = "/Game/Carla/Maps/Town01.Town01"
            if len(matches) != 1 or matches[0][2] not in (official, desired):
                raise BootstrapFailure("package-startup-map-ambiguous")
            ini.write_text(pattern.sub(lambda m: m[1] + desired + m[3], text), encoding="utf-8")
            report["startup_map"] = "Town01"
        result = subprocess.run(["ldd", str(binary)], capture_output=True, text=True,
                                timeout=remaining(30))
        (output / "carla_ldd.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if "not found" in result.stdout or not shutil.which("xdg-user-dir"):
            install_userspace(["libsdl2-2.0", "libvulkan1", "libomp5", "xdg-user-dirs",
                               "libx11-6", "libxrandr2", "libxinerama1", "libxcursor1",
                               "libxi6", "libxkbcommon0", "libpulse0"])
        if not shutil.which("xdg-user-dir"):
            raise BootstrapFailure("xdg-user-dir-unavailable")

        stage("server-launch")
        runtime = scratch / "runtime"
        runtime.mkdir(mode=0o700)
        if os.geteuid() == 0:
            if not shutil.which("runuser"):
                raise BootstrapFailure("nonroot-launch-unavailable")
            username = "vlfcarla"
            try:
                account = pwd.getpwnam(username)
            except KeyError:
                command(["useradd", "--system", "--no-create-home", "--home-dir", str(server),
                         "--shell", "/bin/bash", username], "carla_user", 30)
                account = pwd.getpwnam(username)
            if account.pw_uid == 0:
                raise BootstrapFailure("server-user-must-be-nonroot")
            command(["chown", "-R", f"{account.pw_uid}:{account.pw_gid}", str(server), str(runtime)],
                    "carla_permissions", 120)
            prefix = ["runuser", "-u", username, "--"]
        else:
            prefix = []
        env.update(HOME=str(server), XDG_RUNTIME_DIR=str(runtime))
        args = prefix + ["/bin/bash", str(launcher), "-RenderOffScreen", "-nosound",
                         "-quality-level=Low", "-carla-rpc-port=2000", "-graphicsadapter=0"]
        with (output / "carla_server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(args, cwd=server, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        # Actual CARLA RPC calls must succeed; an open TCP socket is insufficient.
        ready_until = time.monotonic() + remaining(180)
        rpc_code = ("import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(5); "
                    "assert c.get_server_version()=='0.9.16'; c.get_world().get_map(); print('RPC_READY')")
        while True:
            if process.poll() is not None:
                raise BootstrapFailure("server-exited-before-rpc-ready")
            rpc = subprocess.run([python, "-c", rpc_code], capture_output=True, text=True,
                                 timeout=remaining(15), env=env)
            (output / "carla_rpc.log").write_text(rpc.stdout + rpc.stderr, encoding="utf-8")
            if rpc.returncode == 0 and "RPC_READY" in rpc.stdout:
                break
            if time.monotonic() >= ready_until:
                raise BootstrapFailure("server-rpc-ready-timeout")
            time.sleep(1)

        if WORKLOAD in ("week1", "driving"):
            driving = WORKLOAD == "driving"
            stage("D8-D10-privileged-driving-and-recovery-gate" if driving else "D6-D7-geometry-labels-and-week1-gate")
            project = scratch / "project"
            project.mkdir()
            if not PROJECT_SOURCES:
                raise BootstrapFailure("embedded-project-required")
            for relative, source in PROJECT_SOURCES.items():
                path = project / relative
                if path.resolve().is_relative_to(project.resolve()) is not True:
                    raise BootstrapFailure("embedded-project-path-invalid")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")
            artifact = "driving" if driving else "week1"
            script = "run_driving_pipeline.py" if driving else "run_week1_pipeline.py"
            command([python, str(project / "scripts" / script), "--output-dir", str(output / artifact)],
                    "carla_driving" if driving else "carla_week1", 600 if driving else 480, env)
            gate = json.loads((output / artifact / (artifact + "_report.json")).read_text(encoding="utf-8"))
            if gate.get("status") != "PASS" or gate.get("gate", {}).get("status") != "PASS":
                raise BootstrapFailure(artifact + "-evidence-not-pass")
            report[artifact + "_verified"] = True
            report[artifact] = {key: gate[key] for key in ("status", "episode_count", "frame_count", "gate")}
        else:
            stage("rgb-probe")
            probe_file = scratch / "carla_rgb_probe.py"
            probe_file.write_text(PROBE_SOURCE, encoding="utf-8")
            probe_output = output / "carla_probe"
            command([python, str(probe_file), "--output-dir", str(probe_output),
                     "--host", "127.0.0.1", "--port", "2000", "--timeout", "15",
                     "--frames", "20", "--warmup", "5"], "carla_rgb_probe", 240, env)
            rgb = json.loads((probe_output / "carla_rgb_report.json").read_text(encoding="utf-8"))
            if rgb.get("status") != "PASS" or not (probe_output / "carla_rgb_preview.png").is_file():
                raise BootstrapFailure("rgb-evidence-not-pass")
        report["carla_rgb_verified"] = True
        report["status"] = "PASS"
        stage("verified")
    except urllib.error.HTTPError as error:
        report["error"] = "package-http-error"
        report["http_status"] = error.code
    except Exception as error:
        report["error"] = str(error) if isinstance(error, BootstrapFailure) else type(error).__name__
    finally:
        # Only the process group created by this invocation is terminated.
        signal.alarm(0)
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired) as error:
                    report["cleanup_error"] = type(error).__name__
                    report["status"] = "FAIL"
            except ProcessLookupError:
                pass
            except OSError as error:
                report["cleanup_error"] = type(error).__name__
                report["status"] = "FAIL"
        signal.signal(signal.SIGALRM, previous_handler)
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        report["server_stopped"] = process is None or process.poll() is not None
        (output / "kaggle_carla_bootstrap.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 1
