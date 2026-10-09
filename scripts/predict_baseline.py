"""Predict five ego-frame lane waypoints from an RGB image, without CARLA."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.baseline import DEFAULT_BASELINE_PATH, OpenCVLanePredictor, load_baseline_config, overlay_baseline
from scripts.verify_collection import write_report


def _private_output(parser, path: Path) -> Path:
    resolved = path.resolve()
    if not any(resolved.is_relative_to((ROOT / directory).resolve()) for directory in ("results/local", "docs/local")):
        parser.error("prediction artifacts must stay in results/local/ or docs/local/")
    return resolved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="640x360 RGB image; no simulator metadata")
    parser.add_argument("--config", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "results/local/baseline/prediction.json")
    parser.add_argument("--overlay", type=Path, default=ROOT / "results/local/baseline/overlay.png")
    args = parser.parse_args(argv)
    output = _private_output(parser, args.output)
    overlay_path = _private_output(parser, args.overlay)
    if output == overlay_path or args.image.resolve() in (output, overlay_path) or args.config.resolve() in (output, overlay_path):
        parser.error("prediction outputs must not overwrite each other or source files")
    if overlay_path.suffix.lower() != ".png":
        parser.error("overlay must use a .png extension")
    if output.exists() or overlay_path.exists():
        parser.error("prediction artifacts already exist; select fresh --output and --overlay paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as reserved:
            reserved.write(json.dumps({"status": "RUNNING", "scope": "camera-only-opencv-prediction"}))
    except FileExistsError:
        parser.error("prediction report already exists; select a fresh --output path")
    write_report(output, {"status": "RUNNING", "scope": "camera-only-opencv-prediction"})
    try:
        import cv2
        config = load_baseline_config(args.config)
        encoded = args.image.read_bytes()
        import numpy as np
        bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("baseline-input-must-be-three-channel-RGB")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        prediction = OpenCVLanePredictor(config).predict(rgb, rgb.shape[1], rgb.shape[0])
        report = {**prediction, "scope": "camera-only-opencv-prediction", "input_sha256": hashlib.sha256(encoded).hexdigest(),
                  "configuration": config.to_dict()}
        if rgb.shape == (config.image_height, config.image_width, 3) and rgb.dtype == np.uint8:
            overlay = overlay_baseline(rgb, prediction, config)
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            ok, png = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if not ok:
                raise ValueError("baseline-overlay-encoding-failed")
            # Existing write_report provides atomic JSON publication. PNG is
            # written before final report; consumers trust finalized report.
            with overlay_path.open("xb") as overlay_file:
                overlay_file.write(png.tobytes())
            report["overlay_sha256"] = hashlib.sha256(png.tobytes()).hexdigest()
    except Exception as error:
        report = {"status": "FAIL", "scope": "camera-only-opencv-prediction", "error": str(error),
                  "exception_type": type(error).__name__, "waypoints_ego_m": None}
    write_report(output, report)
    print(json.dumps({key: report[key] for key in ("status", "error", "waypoints_ego_m", "timing_ms") if key in report}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
