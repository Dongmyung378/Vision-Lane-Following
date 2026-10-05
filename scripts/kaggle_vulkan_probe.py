"""Run inside a Kaggle GPU session; enumerate real Vulkan devices before CARLA."""

import ctypes
import ctypes.util
import glob
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path


def vulkan_devices():
    loader = ctypes.util.find_library("vulkan")
    if not loader:
        return {"status": "missing_loader", "devices": []}
    vk = ctypes.CDLL(loader)

    class ApplicationInfo(ctypes.Structure):
        _fields_ = [("sType", ctypes.c_uint32), ("pNext", ctypes.c_void_p),
                    ("name", ctypes.c_char_p), ("version", ctypes.c_uint32),
                    ("engine", ctypes.c_char_p), ("engine_version", ctypes.c_uint32),
                    ("api_version", ctypes.c_uint32)]

    class InstanceInfo(ctypes.Structure):
        _fields_ = [("sType", ctypes.c_uint32), ("pNext", ctypes.c_void_p),
                    ("flags", ctypes.c_uint32), ("application", ctypes.POINTER(ApplicationInfo)),
                    ("layer_count", ctypes.c_uint32), ("layers", ctypes.c_void_p),
                    ("extension_count", ctypes.c_uint32), ("extensions", ctypes.c_void_p)]

    app = ApplicationInfo(0, None, b"CARLA preflight", 1, b"probe", 1, 1 << 22)
    info = InstanceInfo(1, None, 0, ctypes.pointer(app), 0, None, 0, None)
    instance = ctypes.c_void_p()
    vk.vkCreateInstance.argtypes = [ctypes.POINTER(InstanceInfo), ctypes.c_void_p,
                                    ctypes.POINTER(ctypes.c_void_p)]
    vk.vkCreateInstance.restype = ctypes.c_int32
    vk.vkDestroyInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    vk.vkEnumeratePhysicalDevices.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
                                             ctypes.POINTER(ctypes.c_void_p)]
    vk.vkEnumeratePhysicalDevices.restype = ctypes.c_int32
    vk.vkGetPhysicalDeviceProperties.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    code = vk.vkCreateInstance(ctypes.byref(info), None, ctypes.byref(instance))
    if code != 0:
        return {"status": "instance_failed", "vk_result": code, "devices": []}
    try:
        count = ctypes.c_uint32()
        code = vk.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None)
        if code != 0 or count.value == 0:
            return {"status": "enumeration_failed", "vk_result": code, "devices": []}
        handles = (ctypes.c_void_p * count.value)()
        code = vk.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), handles)
        if code != 0:
            return {"status": "enumeration_failed", "vk_result": code, "devices": []}
        devices = []
        for handle in handles[:count.value]:
            properties = ctypes.create_string_buffer(4096)
            vk.vkGetPhysicalDeviceProperties(handle, properties)
            api, driver, vendor, device, kind = struct.unpack_from("=5I", properties.raw)
            name = properties.raw[20:276].split(b"\0", 1)[0].decode("utf-8", "replace")
            devices.append({"name": name, "vendor_id": vendor, "device_type": kind,
                            "api_version": api})
        hardware = any(d["vendor_id"] == 0x10DE and d["device_type"] in (1, 2)
                       for d in devices)
        return {"status": "nvidia_hardware" if hardware else "no_nvidia_hardware",
                "devices": devices}
    finally:
        vk.vkDestroyInstance(instance, None)


def main():
    output = Path("/kaggle/working/kaggle_preflight.json")
    report = {"python": platform.python_version(), "machine": platform.machine(),
              "libc": list(platform.libc_ver()), "uid": os.getuid(),
              "gpu": [], "vulkan": {"status": "not_run", "devices": []},
              "carla_rgb_verified": False}
    for label, directory in (("working", "/kaggle/working"), ("scratch", "/tmp")):
        report[label + "_free_gib"] = round(shutil.disk_usage(directory).free / 2**30, 2)
    report["graphics_driver_files"] = sorted({Path(p).name for pattern in (
        "/usr/lib/x86_64-linux-gnu/libGLX_nvidia*", "/usr/lib/x86_64-linux-gnu/libnvidia-glcore*",
        "/usr/lib/x86_64-linux-gnu/libnvidia-glvkspirv*", "/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp*",
        "/usr/lib/x86_64-linux-gnu/libvulkan_nvidia*",
        "/usr/lib/x86_64-linux-gnu/libEGL_nvidia*", "/usr/lib64/libEGL_nvidia*",
        "/usr/local/nvidia/lib64/libEGL_nvidia*",
        "/usr/lib64/libGLX_nvidia*", "/usr/local/nvidia/lib64/libGLX_nvidia*")
        for p in glob.glob(pattern)})
    report["icd_files"] = sorted(Path(p).name for p in glob.glob("/usr/share/vulkan/icd.d/*.json"))
    try:
        process = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=15)
        if process.returncode == 0:
            for line in process.stdout.splitlines():
                name, memory, driver = [part.strip() for part in line.split(",")]
                report["gpu"].append({"name": name, "memory_mib": int(memory), "driver": driver})
        else:
            report["gpu_error"] = "nvidia_smi_failed"
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        report["gpu_error"] = type(error).__name__
    try:
        report["vulkan"] = vulkan_devices()
    except Exception as error:
        report["vulkan"] = {"status": "probe_error", "error_type": type(error).__name__, "devices": []}
    # CUDA containers sometimes mount graphics libraries without an ICD manifest.
    # Try only the already-mounted matching host driver, never a replacement driver.
    report["vulkan_before_icd"] = report["vulkan"]
    report["vulkan_attempts"] = []
    if report["vulkan"]["status"] != "nvidia_hardware":
        report["nvidia_graphics_library"] = "not_loadable"
        child_source = globals().get("VULKAN_CHILD_SOURCE")
        for flavor, library in (("glx", "libGLX_nvidia.so.0"), ("egl", "libEGL_nvidia.so.0")):
            candidates = sorted({str(Path(path).resolve()) for directory in (
                "/usr/lib/x86_64-linux-gnu", "/usr/local/nvidia/lib64", "/usr/lib64")
                for path in glob.glob(directory + "/" + library) if Path(path).is_file()})
            if not candidates:
                report["vulkan_attempts"].append({"library": library, "status": "not_mounted"})
            for index, driver in enumerate(candidates):
                attempt = {"library": library, "path": driver,
                           "fresh_process_probe": bool(child_source)}
                report["vulkan_attempts"].append(attempt)
                try:
                    ctypes.CDLL(driver)
                    report["nvidia_graphics_library"] = "loadable"
                    manifest = Path(f"/tmp/vlf_nvidia_{flavor}_{index}_icd.json")
                    manifest.write_text(json.dumps({"file_format_version": "1.0.0", "ICD": {
                        "library_path": driver, "api_version": "1.3.0"}}), encoding="utf-8")
                    overrides = {"VK_ICD_FILENAMES": str(manifest), "VK_DRIVER_FILES": str(manifest)}
                    if child_source:
                        env = dict(os.environ, **overrides, VK_LOADER_DEBUG="error,warn,driver")
                        child = subprocess.run([sys.executable, "-c", child_source], env=env,
                                               capture_output=True, text=True, timeout=30)
                        Path(f"/kaggle/working/vulkan_loader_{flavor}_{index}.log").write_text(child.stderr, encoding="utf-8")
                        Path("/kaggle/working/vulkan_loader.log").write_text(child.stderr, encoding="utf-8")
                        result = json.loads(child.stdout) if child.returncode == 0 else {
                            "status": "child_failed", "devices": []}
                    else:
                        previous = {key: os.environ.get(key) for key in overrides}
                        os.environ.update(overrides)
                        try:
                            result = vulkan_devices()
                        finally:
                            for key, value in previous.items():
                                if value is None:
                                    os.environ.pop(key, None)
                                else:
                                    os.environ[key] = value
                    if not isinstance(result, dict) or not isinstance(result.get("devices"), list):
                        raise ValueError("invalid-vulkan-result")
                    attempt["vulkan"] = result
                    attempt["status"] = result.get("status", "invalid_result")
                    report["vulkan"] = result
                    report["fresh_process_probe"] = bool(child_source)
                    report["icd_remediation"] = "attempted_existing_driver_manifest"
                    if attempt["status"] == "nvidia_hardware":
                        report["selected_vulkan_library"] = driver
                        break
                except Exception as error:
                    attempt["status"] = "driver_probe_failed"
                    attempt["error_type"] = type(error).__name__
            if report["vulkan"]["status"] == "nvidia_hardware":
                break
    # Prove a small training step independently of Vulkan graphics support.
    try:
        import torch
        report["cuda_training"] = {"status": "FAIL", "torch": torch.__version__, "devices": []}
        if not torch.cuda.is_available():
            report["cuda_training"]["error"] = "cuda_not_available"
        else:
            for index in range(torch.cuda.device_count()):
                device = torch.device("cuda", index)
                torch.manual_seed(123)
                model = torch.nn.Linear(16, 4).to(device)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                inputs = torch.randn(256, 16, device=device)
                targets = torch.randn(256, 4, device=device)
                before = model.weight.detach().clone()
                for _ in range(3):
                    optimizer.zero_grad()
                    loss = torch.nn.functional.mse_loss(model(inputs), targets)
                    loss.backward()
                    optimizer.step()
                torch.cuda.synchronize(device)
                valid = bool(torch.isfinite(loss).item() and not torch.equal(before, model.weight))
                report["cuda_training"]["devices"].append({"index": index, "name": torch.cuda.get_device_name(index),
                    "three_optimizer_steps_verified": valid, "final_loss": float(loss.detach().item())})
                del model, optimizer, inputs, targets, before
            if all(d["three_optimizer_steps_verified"] for d in report["cuda_training"]["devices"]):
                report["cuda_training"]["status"] = "PASS"
    except Exception as error:
        report["cuda_training"] = {"status": "FAIL", "error_type": type(error).__name__}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    # A negative finding is saved successfully, not mistaken for RGB verification.
    return 0


if __name__ == "__main__":
    sys.exit(main())
