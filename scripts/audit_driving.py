"""CPU-only source-preserving positive/negative QA on native driving output."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.contracts import read_strict_json, strict_json_loads
from carla_tools.driving_validation import validate_drive_episode, validate_driving_manifest
from scripts.verify_collection import write_report


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def source_hashes(root):
    return {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in sorted(root.rglob("*")) if path.is_file()}


def _mutate(name, episode):
    metadata_path, trace_path = episode / "episode.json", episode / "steps.jsonl"
    metadata = read_strict_json(metadata_path)
    rows = [strict_json_loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    if name == "waypoint_units":
        rows[0]["tracking"]["label"]["units"] = "cm"
    elif name == "original_target_lane":
        rows[0]["tracking"]["label"]["target_lane"]["lane_id"] += 1
    elif name == "native_control_application":
        rows[1]["vehicle"]["applied_control"]["steer"] += .123
    elif name == "command_target_frame":
        rows[0]["command_target_frame"] = rows[0]["frame"] + 2
    elif name == "controller_decision":
        rows[0]["command"]["steer"] += .123
    elif name == "measured_vehicle_geometry":
        metadata["vehicle_geometry"]["wheelbase_m"] += .1
    elif name == "png_crc_rehashed":
        image = next(row["image"] for row in rows if row["image"] is not None)
        path = episode / image["path"]
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
        image["sha256"] = digest(payload)
    elif name == "missing_rgb":
        image = next(row["image"] for row in rows if row["image"] is not None)
        path = (episode / image["path"]).resolve()
        if not path.is_relative_to(episode.resolve()):
            raise ValueError("temporary-image-path-escaped")
        path.unlink()
    elif name == "event_callback_error":
        metadata["event_callback_errors"] = [{"code": "injected-callback-error", "exception_type": "ValueError"}]
    elif name == "late_collision_event":
        metadata["events"] = [{"kind": "collision", "frame": metadata["drive_end_frame"] + 1,
                               "timestamp": rows[-1]["timestamp"] + .05, "other_actor_id": 999,
                               "other_actor_type": "injected.qa", "impulse": {"x": 1., "y": 0., "z": 0.}}]
    elif name == "incomplete_cleanup":
        metadata["cleanup"]["weather_restored"] = False
    elif name == "metadata_trace_hash":
        metadata["trace_sha256"] = "0" * 64
    elif name == "duplicate_frame":
        for key in ("frame", "world_frame", "snapshot_frame", "sensor_frame", "command_source_frame"):
            rows[1][key] = rows[0]["frame"]
    elif name == "missing_frame":
        del rows[1]
        for index, row in enumerate(rows):
            row["index"] = index
        metadata["step_count"] = len(rows)
    else:
        raise ValueError("unsupported-negative-mutation")
    trace = ("\n".join(json.dumps(row, sort_keys=True, allow_nan=False) for row in rows) + "\n").encode("utf-8")
    trace_path.write_bytes(trace)
    if name != "metadata_trace_hash":
        metadata["trace_sha256"] = digest(trace)
    metadata_path.write_text(json.dumps(metadata, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_audit(native):
    directory = Path(native).resolve()
    if (directory / "driving").is_dir():
        directory = directory / "driving"
    manifest_path = directory / "driving_manifest.json"
    report = {"schema_version": 1, "status": "FAIL", "scope": "D8-D10-native-artifact-positive-negative-CPU-QA",
              "errors": [], "negative_checks": []}
    before = None
    try:
        before = source_hashes(directory)
        positive = validate_driving_manifest(manifest_path)
        report["positive_gate"] = {key: positive[key] for key in ("status", "gate_version", "episode_count", "step_count", "image_count", "errors") if key in positive}
        if positive["status"] != "PASS":
            failed = [item for item in positive.get("episodes", []) if item["report"]["status"] != "PASS"]
            report["positive_gate"]["failed_episode_errors"] = [item["report"]["errors"] for item in failed]
            raise ValueError("native-positive-gate-not-pass")
        manifest = read_strict_json(manifest_path)
        original_episode = (directory / manifest["episodes"][0]["episode"]).resolve()
        temporary_parent = (ROOT / "results" / "local" / "native-driving-qa-temporary").resolve()
        if not temporary_parent.is_relative_to((ROOT / "results" / "local").resolve()):
            raise ValueError("qa-temporary-parent-escaped")
        temporary_parent.mkdir(parents=True, exist_ok=True)
        mutations = ("waypoint_units", "original_target_lane", "native_control_application", "command_target_frame",
                     "controller_decision", "measured_vehicle_geometry", "png_crc_rehashed", "missing_rgb",
                     "event_callback_error", "late_collision_event", "incomplete_cleanup", "metadata_trace_hash",
                     "duplicate_frame", "missing_frame")
        with tempfile.TemporaryDirectory(prefix="native-driving-", dir=temporary_parent) as generated:
            scratch = Path(generated).resolve()
            if not scratch.is_relative_to(temporary_parent):
                raise ValueError("qa-temporary-directory-escaped")
            for index, name in enumerate(mutations):
                target = scratch / f"episode-{index:02d}"
                shutil.copytree(original_episode, target)
                _mutate(name, target)
                checked = validate_drive_episode(target)
                report["negative_checks"].append({"name": name, "expected_status": "FAIL", "observed_status": checked["status"],
                      "rejected": checked["status"] == "FAIL" and bool(checked["errors"]), "errors": checked["errors"]})
            baseline = copy.deepcopy(manifest)
            for item in baseline["episodes"]:
                item["episode"] = str((directory / item["episode"]).resolve())
            for name in ("duplicate_manifest_case", "missing_manifest_case"):
                changed = copy.deepcopy(baseline)
                if name == "duplicate_manifest_case":
                    changed["episodes"].append(copy.deepcopy(changed["episodes"][0]))
                else:
                    changed["episodes"].pop()
                path = scratch / f"{name}.json"
                path.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
                checked = validate_driving_manifest(path)
                report["negative_checks"].append({"name": name, "expected_status": "FAIL", "observed_status": checked["status"],
                     "rejected": checked["status"] == "FAIL" and bool(checked["errors"]), "errors": checked["errors"]})
        preserved = source_hashes(directory) == before
        report.update(negative_check_count=len(report["negative_checks"]), source_file_count=len(before), sources_preserved=preserved)
        if not preserved:
            raise ValueError("native-sources-changed-during-qa")
        if not all(item["rejected"] for item in report["negative_checks"]):
            raise ValueError("negative-mutation-was-not-rejected")
        report["status"] = "PASS"
    except BaseException as error:
        report["errors"].append({"code": str(error) if type(error) is ValueError else "native-audit-operation-failed", "exception_type": type(error).__name__})
        if before is not None:
            report["sources_preserved"] = source_hashes(directory) == before
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True, help="Downloaded output directory, or its driving subdirectory.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    native = args.native.resolve()
    source_root = native / "driving" if (native / "driving").is_dir() else native
    if args.output.resolve().is_relative_to(source_root):
        parser.error("output must be outside native source artifacts")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; use a new report path")
    initial = {"status": "RUNNING", "scope": "D8-D10-native-artifact-positive-negative-CPU-QA", "errors": []}
    initial_bytes = (json.dumps(initial, allow_nan=False) + "\n").encode("utf-8")
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(initial_bytes.decode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        parser.error("output already exists; use a new report path")
    except OSError as error:
        print(json.dumps({"status": "FAIL", "errors": [{"code": "driving-native-audit-output-unavailable", "exception_type": type(error).__name__}]}))
        return 1
    try:
        report = run_audit(args.native)
    except BaseException as error:
        report = {"status": "FAIL", "errors": [{"code": "driving-native-audit-interrupted", "exception_type": type(error).__name__}]}
    try:
        if args.output.is_symlink() or args.output.read_bytes() != initial_bytes:
            raise ValueError("output-changed-during-audit")
        write_report(args.output, report)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "FAIL", "errors": [{"code": "driving-native-audit-report-publication-failed", "exception_type": type(error).__name__}]}))
        return 1
    print(json.dumps({key: report[key] for key in ("status", "positive_gate", "negative_check_count", "sources_preserved", "errors") if key in report}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
