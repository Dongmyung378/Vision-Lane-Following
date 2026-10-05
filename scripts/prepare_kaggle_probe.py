"""Package only public probe code into a private Kaggle job; no credentials."""

import argparse
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--kind", choices=("vulkan", "carla"), default="vulkan")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    vulkan = (ROOT / "scripts/kaggle_vulkan_probe.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(vulkan).body if isinstance(n, ast.FunctionDef) and n.name == "vulkan_devices")
    child = "import ctypes,ctypes.util,struct,json\n" + ast.get_source_segment(vulkan, node) + "\nprint(json.dumps(vulkan_devices()))\n"
    if args.kind == "vulkan":
        code = "VULKAN_CHILD_SOURCE = " + repr(child) + "\n" + vulkan
        slug = "vlf-carla-vulkan-preflight-20261005"
        title = "VLF CARLA Vulkan preflight 20261005"
    else:
        code = (ROOT / "scripts/kaggle_carla_bootstrap.py").read_text(encoding="utf-8")
        rgb = (ROOT / "scripts/carla_rgb_probe.py").read_text(encoding="utf-8")
        code += "\nPROBE_SOURCE = " + repr(rgb) + "\nVULKAN_CHILD_SOURCE = " + repr(child) + "\nraise SystemExit(main())\n"
        slug = "vlf-carla-rgb-smoke-20261005"
        title = "VLF CARLA RGB smoke 20261005"
    compile(code, "probe.py", "exec")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "probe.py").write_text(code, encoding="utf-8")
    metadata = {"id": args.owner + "/" + slug, "title": title, "code_file": "probe.py",
                "language": "python", "kernel_type": "script", "is_private": "true",
                "enable_gpu": "true", "enable_internet": "true", "enable_tpu": "false",
                "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [],
                "kernel_sources": [], "model_sources": []}
    (args.output_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Prepared private " + args.kind + " probe; not submitted.")


if __name__ == "__main__":
    main()
