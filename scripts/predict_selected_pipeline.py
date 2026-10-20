"""Infer RGB waypoints and use the controller bound to a selected pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_SEQUENCE_BYTES = 16 * 1024 * 1024
MAX_FRAMES = 20000


def _write_result(path, report):
    """Publish complete JSON without replacing an existing result."""
    import os
    import tempfile
    from carla_tools.inference import new_output_path
    from carla_tools.selected_pipeline import SelectedPipelineError
    target = new_output_path(path)
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    pending = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".selected-pipeline-", suffix=".pending", delete=False) as stream:
            pending = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "nt":
            # Windows rename is atomic and rejects an existing destination.
            os.rename(pending, target)
            pending = None
        else:
            # POSIX rename can replace a target; an exclusive link cannot.
            os.link(pending, target)
    except FileExistsError as error:
        raise SelectedPipelineError("selected-pipeline-output-already-exists") from error
    except OSError as error:
        raise SelectedPipelineError("selected-pipeline-output-write-failed") from error
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
    return target


def _read_sequence(path):
    from carla_tools.contracts import strict_json_loads
    from carla_tools.inference import _input_file
    from carla_tools.selected_pipeline import SelectedPipelineError, safe_relative_file
    source = _input_file(path)
    if not 0 < source.stat().st_size <= MAX_SEQUENCE_BYTES:
        raise SelectedPipelineError("selected-pipeline-sequence-size-invalid")
    payload = source.read_bytes()
    if len(payload) > MAX_SEQUENCE_BYTES:
        raise SelectedPipelineError("selected-pipeline-sequence-size-invalid")
    lines = payload.decode("utf-8").splitlines()
    if not 0 < len(lines) <= MAX_FRAMES or any(not line.strip() for line in lines):
        raise SelectedPipelineError("selected-pipeline-sequence-row-count-invalid")
    rows, previous = [], None
    for line in lines:
        row = strict_json_loads(line)
        required = {"frame", "timestamp", "image", "speed_mps"}
        if (type(row) is not dict or not required <= set(row) <= required | {"reset_before"}
                or type(row["frame"]) is not int or row["frame"] < 0
                or type(row["timestamp"]) not in (int, float) or not math.isfinite(row["timestamp"]) or row["timestamp"] < 0
                or type(row["speed_mps"]) not in (int, float) or not math.isfinite(row["speed_mps"])
                or not 0 <= row["speed_mps"] <= 100 or type(row.get("reset_before", False)) is not bool):
            raise SelectedPipelineError("selected-pipeline-sequence-row-invalid")
        if (previous is not None and not row.get("reset_before", False)
                and (row["frame"] != previous["frame"] + 1
                    or abs(row["timestamp"] - previous["timestamp"] - .05) > 1e-6)):
            raise SelectedPipelineError("selected-pipeline-sequence-frame-or-time-gap")
        image = safe_relative_file(source.parent, row["image"])
        rows.append((row, image))
        previous = row
    return source, payload, rows


def run(args):
    from carla_tools.contracts import strict_json_loads
    from carla_tools.driving_validation import _geometry
    from carla_tools.inference import _input_file, new_output_path, read_rgb_png
    from carla_tools.selected_pipeline import (GEOMETRY_CONTRACT, SelectedPipelineError,
        load_selected_pipeline)
    output = new_output_path(args.output)
    geometry_source = _input_file(args.geometry)
    if not 0 < geometry_source.stat().st_size <= 2 * 1024 * 1024:
        raise SelectedPipelineError("selected-pipeline-geometry-evidence-size-invalid")
    geometry_bytes = geometry_source.read_bytes()
    geometry_record = strict_json_loads(geometry_bytes.decode("utf-8"))
    if not isinstance(geometry_record, dict) or not {"vehicle_geometry", "geometry_evidence"} <= geometry_record.keys():
        raise SelectedPipelineError("selected-pipeline-native-geometry-evidence-required")
    # Reconstruct from the native wheel measurements and their same-pose
    # transform; a plausible or guessed wheelbase is insufficient here.
    measured = _geometry(geometry_record)
    sequence_source, sequence_bytes = None, None
    if args.image is not None:
        if args.speed_mps is None:
            raise SelectedPipelineError("selected-pipeline-single-image-speed-required")
        inputs = [({"frame": 0, "timestamp": 0., "speed_mps": args.speed_mps, "reset_before": False}, args.image)]
    else:
        if args.speed_mps is not None:
            raise SelectedPipelineError("selected-pipeline-sequence-has-per-frame-speed")
        sequence_source, sequence_bytes, inputs = _read_sequence(args.frames)
    pipeline = load_selected_pipeline(args.selection, measured, device=args.device,
        expected_pipeline_id=args.expected_pipeline_id)
    report = {"schema_version": 1, "status": "FAIL", "scope": "offline-selected-RGB-prediction-and-control",
        "simulator_started": False, "closed_loop_executed": False, "device": args.device,
        "pipeline": pipeline.identity, "geometry": {"contract": GEOMETRY_CONTRACT,
            "source_sha256": hashlib.sha256(geometry_bytes).hexdigest(), "native_geometry_reconstructed": True},
        "input_mode": "single-image-fresh-controller" if args.image is not None else "ordered-frame-sequence",
        "input_frame_count": len(inputs), "processed_frame_count": 0, "all_input_frames_processed": False,
        "controller_history_policy": "preserve-within-episode; reset-only-when-explicit; stop-on-first-failure",
        "frames": []}
    for row, image in inputs:
        if row.get("reset_before", False):
            pipeline.reset()
        rgb, image_identity = read_rgb_png(image)
        result = pipeline.step(rgb, row["speed_mps"])
        report["frames"].append(dict(result, frame=row["frame"], timestamp=row["timestamp"],
            speed_mps=row["speed_mps"], reset_before=row.get("reset_before", False), image=image_identity))
        report["processed_frame_count"] += 1
        if result["status"] != "PASS":
            break
    report["bindings"] = pipeline.verify_bindings()
    if geometry_source.read_bytes() != geometry_bytes or sequence_source is not None and sequence_source.read_bytes() != sequence_bytes:
        raise SelectedPipelineError("selected-pipeline-input-manifest-changed")
    report["all_input_frames_processed"] = report["processed_frame_count"] == report["input_frame_count"]
    report["status"] = "PASS" if report["all_input_frames_processed"] and all(row["status"] == "PASS" for row in report["frames"]) else "FAIL"
    _write_result(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "geometry", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--frames", type=Path)
    parser.add_argument("--speed-mps", type=float)
    parser.add_argument("--expected-pipeline-id")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except Exception as error:
        from carla_tools.controller import ControllerError
        from carla_tools.inference import InferenceError
        from carla_tools.selected_pipeline import SelectedPipelineError
        code = str(error) if isinstance(error, (SelectedPipelineError, ControllerError, InferenceError)) else "selected-pipeline-input-or-load-failed"
        print(json.dumps({"status": "FAIL", "error": code, "exception_type": type(error).__name__}))
        return 2
    print(json.dumps({"status": result["status"], "pipeline_id": result["pipeline"]["pipeline_id"],
        "processed_frame_count": result["processed_frame_count"], "closed_loop_executed": False}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
