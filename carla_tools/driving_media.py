"""Export synchronized CPU-encoded driving video and a compact control log."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from .baseline_validation import _safe_file
from .contracts import strict_json_loads
from .projection import decode_rgb8_png


def export_driving_media(directory):
    """Keep every recorded frame, including the terminal failure frame.

    Red/green waypoints use fixed nominal ground-plane projection for viewing.
    The video is derived evidence; original PNGs and JSONL remain authoritative.
    """
    import cv2
    import numpy as np
    from .baseline import load_baseline_config, nominal_ground_to_pixels

    root = Path(directory)
    metadata = strict_json_loads(_safe_file(root, "episode.json").read_text(encoding="utf-8"))
    trace = _safe_file(root, "steps.jsonl").read_bytes()
    rows = [strict_json_loads(line) for line in trace.decode().splitlines()]
    if not rows:
        raise ValueError("driving-media-empty-trace")
    for index, row in enumerate(rows):
        if (row["image"]["path"] != f"images/{row['frame']:08d}.png" or
            (index and (row["frame"] != rows[index-1]["frame"] + 1 or
                        abs(row["timestamp"] - rows[index-1]["timestamp"] - .05) > 1e-6))):
            raise ValueError("driving-media-frame-sequence-mismatch")
    video, log = root / "driving.mp4", root / "controls.csv"
    if any(path.exists() or path.is_symlink() for path in (video, log, root / "media.json")):
        raise ValueError("driving-media-output-already-exists")
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 20., (640, 420))
    if not writer.isOpened():
        raise ValueError("driving-media-software-encoder-unavailable")
    config = load_baseline_config()
    fields = ("frame", "timestamp", "speed_mps", "reference_arc_m", "lateral_error_m", "heading_error_deg",
              "throttle", "steer", "brake", "prediction_status", "waypoints_ego_m", "termination_reason", "pipeline_ms")
    try:
        with log.open("x", encoding="utf-8", newline="") as stream:
            output = csv.DictWriter(stream, fieldnames=fields)
            output.writeheader()
            for row in rows:
                image = row["image"]
                width, height, rgb = decode_rgb8_png(_safe_file(root, image["path"]).read_bytes(), image["sha256"], (640, 360))
                canvas = np.zeros((420, 640, 3), dtype=np.uint8)
                canvas[:360] = np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3)[:, :, ::-1]
                prediction = row["perception"]
                for points, color in ((row["tracking"]["label"]["waypoints_ego_m"], (0, 255, 0)),
                                      (prediction.get("waypoints_ego_m"), (0, 0, 255))):
                    if points is None:
                        continue
                    pixels, visible = nominal_ground_to_pixels(np.asarray(points), config)
                    for pixel, front in zip(pixels, visible):
                        x, y = pixel
                        if front and math.isfinite(x) and math.isfinite(y) and 0 <= x < 640 and 0 <= y < 360:
                            cv2.circle(canvas, (int(x), int(y)), 4, color, -1)
                command = row.get("command") or {}
                status = row.get("termination_reason") or "driving"
                cv2.putText(canvas, f"{metadata['method']} | frame {row['frame']} | {row['speed_mps']:.2f} m/s | {status}",
                            (8, 379), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
                cv2.putText(canvas, f"steer {command.get('steer', 0):+.3f} | lateral {row['tracking']['lateral_error_m']:+.3f} m | red prediction / green GT (nominal)",
                            (8, 402), cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 255, 255), 1)
                writer.write(canvas)
                output.writerow(dict(frame=row["frame"], timestamp=row["timestamp"], speed_mps=row["speed_mps"],
                    reference_arc_m=row["tracking"]["reference_arc_m"], lateral_error_m=row["tracking"]["lateral_error_m"],
                    heading_error_deg=row["tracking"]["heading_error_deg"], throttle=command.get("throttle"),
                    steer=command.get("steer"), brake=command.get("brake"), prediction_status=prediction["status"],
                    waypoints_ego_m=json.dumps(prediction.get("waypoints_ego_m"), separators=(",", ":")),
                    termination_reason=row.get("termination_reason"), pipeline_ms=row["latency_ms"]["pipeline_ms"]))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(video))
    decoded = 0
    try:
        if not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS) - 20.) > .01:
            raise ValueError("driving-media-video-clock-mismatch")
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame.shape != (420, 640, 3):
                raise ValueError("driving-media-video-dimensions-mismatch")
            decoded += 1
    finally:
        capture.release()
    if decoded != len(rows):
        raise ValueError("driving-media-video-frame-count-mismatch")
    result = {"status": "PASS", "frame_count": decoded, "fps": 20., "encoding": "CPU-mp4v",
              "trace_sha256": hashlib.sha256(trace).hexdigest(), "first_frame": rows[0]["frame"], "last_frame": rows[-1]["frame"],
              "first_timestamp": rows[0]["timestamp"], "last_timestamp": rows[-1]["timestamp"],
              "driving_success": metadata["evaluation_metrics"]["driving_success"],
              "termination_reason": metadata["evaluation_metrics"]["termination_reason"],
              "projection": "fixed-nominal-ground-plane; visualization-only",
              "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (video, log)}}
    (root / "media.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
