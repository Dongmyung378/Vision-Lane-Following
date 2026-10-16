"""Assemble complete, deterministically selected driving evidence for review."""
from __future__ import annotations

import copy
import csv
import hashlib
import html
import json
import math
from pathlib import Path, PurePosixPath
import shutil

from .contracts import strict_json_loads

CASES = ("straight", "left_curve", "right_curve")
MINIMUM_SUSTAINED_FRAMES = 40
MINIMUM_SUSTAINED_PROGRESS_M = 10.
CONTROL_FIELDS = ("frame", "timestamp", "speed_mps", "reference_arc_m", "lateral_error_m", "heading_error_deg",
                  "throttle", "steer", "brake", "prediction_status", "waypoints_ego_m", "termination_reason", "pipeline_ms")


def _require(condition, message):
    if not condition:
        raise ValueError("driving-demo-" + message)


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _finite(value):
    _require(type(value) in (int, float) and math.isfinite(value), "nonfinite-number")
    return float(value)


def _file(root, relative):
    _require(isinstance(relative, str) and relative and "\\" not in relative and ":" not in relative,
             "unsafe-relative-path")
    pieces = relative.split("/")
    path = PurePosixPath(relative)
    _require(not path.is_absolute() and all(part not in ("", ".", "..") for part in pieces), "unsafe-relative-path")
    result = root.joinpath(*pieces)
    current = root
    for part in pieces:
        current = current / part
        _require(not current.is_symlink(), "symlink-source")
    _require(result.resolve().is_relative_to(root) and result.is_file(), "missing-or-escaped-source")
    return result


def _read_json(path):
    return strict_json_loads(path.read_text(encoding="utf-8"))


def _verify_controls(path, rows):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(tuple(reader.fieldnames or ()) == CONTROL_FIELDS, "controls-fields-mismatch")
        observed = list(reader)
    _require(len(observed) == len(rows), "controls-frame-count-mismatch")
    for actual, row in zip(observed, rows):
        command = row.get("command") or {}
        expected = dict(frame=row["frame"], timestamp=row["timestamp"], speed_mps=row["speed_mps"],
                        reference_arc_m=row["tracking"]["reference_arc_m"], lateral_error_m=row["tracking"]["lateral_error_m"],
                        heading_error_deg=row["tracking"]["heading_error_deg"], throttle=command.get("throttle"),
                        steer=command.get("steer"), brake=command.get("brake"), prediction_status=row["perception"]["status"],
                        waypoints_ego_m=row["perception"].get("waypoints_ego_m"), termination_reason=row.get("termination_reason"),
                        pipeline_ms=row["latency_ms"]["pipeline_ms"])
        for key, value in expected.items():
            if key == "waypoints_ego_m":
                _require(strict_json_loads(actual[key]) == value, "controls-waypoints-mismatch")
            else:
                _require(actual[key] == ("" if value is None else str(value)), "controls-row-mismatch")


def _verify_video(path, frame_count):
    import cv2

    capture = cv2.VideoCapture(str(path))
    decoded = 0
    try:
        _require(capture.isOpened(), "video-unreadable")
        _require(abs(capture.get(cv2.CAP_PROP_FPS) - 20.) <= .01, "video-clock-mismatch")
        while True:
            success, frame = capture.read()
            if not success:
                break
            _require(frame.shape == (420, 640, 3), "video-dimensions-mismatch")
            decoded += 1
    finally:
        capture.release()
    _require(decoded == frame_count, "video-frame-count-mismatch")


def _verified_episode(root, record):
    gate = record.get("gate", {})
    _require(gate.get("status") == "PASS" and gate.get("closed_loop_executed") is True
             and gate.get("native", {}).get("status") == "PASS", "independent-gate-required")
    source = gate["native"].get("source", {})
    base = record["episode"]
    paths = {name: _file(root, base + "/" + name)
             for name in ("episode.json", "steps.jsonl", "media.json", "driving.mp4", "controls.csv")}
    expected = {"episode.json": source.get("metadata_sha256"), "steps.jsonl": source.get("trace_sha256")}
    for name, digest in expected.items():
        _require(_digest(digest) and _sha(paths[name]) == digest, "gate-source-hash-mismatch")
    metadata, media = _read_json(paths["episode.json"]), _read_json(paths["media.json"])
    rows = [strict_json_loads(line) for line in paths["steps.jsonl"].read_text(encoding="utf-8").splitlines()]
    _require(rows and len(rows) == gate.get("frame_count") == metadata.get("step_count"), "trace-frame-count-mismatch")
    _require(metadata.get("status") == "COMPLETE" and metadata.get("method") == record["variant"]
             and metadata.get("geometry_case") == record["case"]
             and metadata.get("requested_initial_reference_m") == 0.
             and metadata.get("condition", {}).get("name") == "center", "episode-identity-mismatch")
    _require(metadata.get("trace_sha256") == expected["steps.jsonl"], "metadata-trace-hash-mismatch")
    _require(metadata.get("evaluation_metrics") == gate.get("metrics"), "gate-metrics-mismatch")
    metrics = gate["metrics"]
    completed = metrics.get("driving_success")
    _require(type(completed) is bool and completed == (metrics.get("termination_reason") == "route_completed"),
             "completion-reason-mismatch")
    collisions, invasions = metrics.get("collision_count"), metrics.get("lane_invasion_count")
    _require(type(collisions) is int and type(invasions) is int and min(collisions, invasions) >= 0, "events-invalid")
    margin = _finite(metrics.get("minimum_body_lane_margin_m"))
    _require(abs(margin - min(_finite(row["body_lane_margin_m"]) for row in rows)) <= 1e-5, "body-margin-mismatch")
    image_hashes = source.get("image_sha256", {})
    seen_images = set()
    for index, row in enumerate(rows):
        frame, timestamp = row.get("frame"), _finite(row.get("timestamp"))
        _require(type(frame) is int and frame >= 0, "frame-invalid")
        if index:
            _require(frame == rows[index - 1]["frame"] + 1
                     and abs(timestamp - rows[index - 1]["timestamp"] - .05) <= 1e-6, "trace-clock-mismatch")
        image = row["image"]
        relative = "images/" + f"{frame:08d}.png"
        digest = image_hashes.get(relative)
        _require(image.get("path") == relative and _digest(digest) and image.get("sha256") == digest,
                 "image-identity-mismatch")
        _require(_sha(_file(root, base + "/" + relative)) == digest, "image-source-hash-mismatch")
        seen_images.add(relative)
        _require(row.get("terminal") is (index == len(rows) - 1), "terminal-frame-mismatch")
    _require(seen_images == set(image_hashes), "gate-image-set-mismatch")
    _require(rows[-1].get("termination_reason") == metrics["termination_reason"], "terminal-reason-mismatch")
    _require(media.get("status") == "PASS" and media.get("encoding") == "CPU-mp4v"
             and media.get("frame_count") == len(rows) and media.get("fps") == 20.
             and media.get("trace_sha256") == expected["steps.jsonl"]
             and media.get("first_frame") == rows[0]["frame"] and media.get("last_frame") == rows[-1]["frame"]
             and media.get("first_timestamp") == rows[0]["timestamp"] and media.get("last_timestamp") == rows[-1]["timestamp"]
             and media.get("driving_success") == completed and media.get("termination_reason") == metrics["termination_reason"],
             "media-trace-binding-mismatch")
    if "media" in record:
        _require(record["media"] == media, "gate-media-binding-mismatch")
    media_files = media.get("files", {})
    _require(set(media_files) == {"driving.mp4", "controls.csv"}, "media-file-set-mismatch")
    for name, digest in media_files.items():
        _require(_digest(digest) and _sha(paths[name]) == digest, "media-source-hash-mismatch")
        expected[name] = digest
    expected["media.json"] = _sha(paths["media.json"])
    _verify_controls(paths["controls.csv"], rows)
    _verify_video(paths["driving.mp4"], len(rows))
    progress = _finite(rows[-1]["tracking"]["reference_arc_m"]) - _finite(rows[0]["tracking"]["reference_arc_m"])
    _require(progress >= 0., "route-progress-invalid")
    clean = completed and collisions == 0 and invasions == 0 and margin >= 0.
    model = metadata.get("native_model", {})
    learned_camera_only = (_digest(model.get("checkpoint_sha256")) and model.get("variant") == record["variant"]
        and metadata.get("perception_uses_route") is False and metadata.get("perception_uses_vehicle_pose") is False
        and metadata.get("prediction_inputs") in ("rgb-only", "rgb-and-fixed-nominal-calibration"))
    result = {"method": record["variant"], "case": record["case"], "source_episode": base,
        "condition": "center", "initial_reference_m": 0., "frame_count": len(rows),
        "first_frame": rows[0]["frame"], "last_frame": rows[-1]["frame"],
        "simulation_duration_seconds": rows[-1]["timestamp"] - rows[0]["timestamp"], "reference_progress_m": progress,
        "goal_completed": completed, "clean_completed": clean, "learned_camera_only": learned_camera_only,
        "sustained_clean_route": clean and len(rows) >= MINIMUM_SUSTAINED_FRAMES and progress >= MINIMUM_SUSTAINED_PROGRESS_M,
        "termination_reason": metrics["termination_reason"], "collision_count": collisions, "lane_invasion_count": invasions,
        "minimum_body_lane_margin_m": margin, "outcome": ("clean_completed" if clean else
            "goal_completed_with_lane_or_collision_issue" if completed else "driving_failure"),
        "source_file_sha256": expected, "files": {}}
    return result, paths


def _page(manifest):
    escape = html.escape
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>RGB waypoint driving review</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;background:#f5f7fa;color:#18212c}section{background:white;padding:20px;margin:20px 0;border:1px solid #dce2e9;border-radius:8px}video{width:100%;max-width:640px}code,pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#1459aa}li{margin:5px 0}</style>',
             '<h1>RGB waypoint driving review</h1>',
             '<p>Fixed selection: every method, straight / left curve / right curve, start 0 m, centered. '
             'Videos contain complete episodes, including terminal failure frames. '
             'Goal completion and clean lane retention are reported separately.</p>',
             '<p>Red points: predicted path. Green points: privileged ground truth for visualization only, '
             'using nominal ground projection. Video playback at 20 fps follows the simulation clock; '
             'it does not establish real-time wall-clock processing.</p>',
             '<p>Learned driving gate: <strong>' + ('PASS' if manifest['m1_gate_pass'] else 'FAIL') + '</strong>. '
             'A passing gate requires at least one learned camera-only clean route with 40 frames and 10 m progress. '
             'It does not establish unseen-condition reliability.</p>']
    for item in manifest["episodes"]:
        title = escape(item["method"] + " / " + item["case"])
        links = ' · '.join('<a href="' + escape(item["files"][name]["path"], quote=True) + '">' + escape(label) + '</a>'
            for name, label in (("controls.csv", "Complete controls"), ("steps.jsonl", "Complete trace"),
                                ("episode.json", "Episode metadata"), ("media.json", "Media binding")))
        parts.extend(['<section><h2>' + title + '</h2>', '<p><strong>' + escape(item["outcome"]) + '</strong> — '
            + escape(item["termination_reason"]) + '</p>', '<video controls preload="metadata" src="'
            + escape(item["files"]["driving.mp4"]["path"], quote=True) + '"></video>',
            '<p>' + str(item["frame_count"]) + ' frames; progress ' + f'{item["reference_progress_m"]:.2f}'
            + ' m; collisions ' + str(item["collision_count"]) + '; lane invasions ' + str(item["lane_invasion_count"])
            + '; minimum body lane margin ' + f'{item["minimum_body_lane_margin_m"]:.3f}' + ' m.</p>',
            '<p>' + links + '</p></section>'])
    if manifest["latency_report"] is not None:
        parts.append('<details><summary>Latency measurement</summary><pre>'
                     + escape(json.dumps(manifest["latency_report"], indent=2, ensure_ascii=False, allow_nan=False))
                     + '</pre></details>')
    parts.append('<p><a href="manifest.json">Evidence manifest and hashes</a></p></html>')
    return "\n".join(parts) + "\n"


def build_driving_demo(native_dir, episodes, output_dir, latency_report=None):
    """Copy full fixed-state episodes whose independent replay gates are bound.

    Input episode records follow the native/independent-audit record contract.
    PASS describes evidence integrity; ``m1_gate_pass`` separately describes
    the modest learned-driving gate. Outputs are local review artifacts.
    """
    native, output = Path(native_dir), Path(output_dir)
    _require(native.is_dir() and not native.is_symlink(), "native-directory-invalid")
    native = native.resolve()
    _require(not output.exists() and not output.is_symlink(), "output-already-exists")
    for parent in output.absolute().parents:
        _require(not parent.is_symlink(), "symlink-output-parent")
    output = output.absolute()
    _require(not output.resolve().is_relative_to(native), "output-inside-native-source")
    _require(isinstance(episodes, (list, tuple)) and episodes, "episodes-required")
    records, seen, methods = {}, set(), set()
    for record in episodes:
        _require(isinstance(record, dict) and isinstance(record.get("variant"), str) and record["variant"]
                 and record.get("case") in CASES and isinstance(record.get("episode"), str), "record-identity-invalid")
        _require(record["episode"] not in seen, "duplicate-episode")
        seen.add(record["episode"])
        _require(record.get("gate", {}).get("status") == "PASS", "unverified-input-episode")
        methods.add(record["variant"])
        condition = record.get("condition")
        if isinstance(condition, dict):
            condition = condition.get("name")
        start = record.get("initial_reference_m")
        _require(type(start) in (int, float) and math.isfinite(start), "record-start-invalid")
        if start == 0. and condition == "center":
            key = record["variant"], record["case"]
            _require(key not in records, "ambiguous-fixed-state-selection")
            records[key] = record
    expected_keys = {(method, case) for method in methods for case in CASES}
    _require(set(records) == expected_keys, "fixed-state-selection-incomplete")
    # Complete verification precedes output creation, preventing partial demos
    # from silently looking successful after an integrity exception.
    verified = [(method, case, *_verified_episode(native, records[method, case]))
                for method in sorted(methods) for case in CASES]
    latency = copy.deepcopy(latency_report)
    json.dumps(latency, allow_nan=False)
    manifest = {"status": "PASS", "scope": "deterministic-complete-native-driving-evidence",
        "selection": "every-method/straight-left_curve-right_curve/start00/center",
        "selection_uses_outcomes": False, "failed_episodes_truncated": False, "input_episode_count": len(episodes),
        "selected_episode_count": len(verified), "video_clock_fps": 20., "real_time_fps_inferred_from_video": False,
        "clean_completion_rule": "route_completed AND collision_count=0 AND lane_invasion_count=0 AND minimum_body_lane_margin_m>=0",
        "sustained_rule": {"minimum_frames": MINIMUM_SUSTAINED_FRAMES, "minimum_reference_progress_m": MINIMUM_SUSTAINED_PROGRESS_M},
        "latency_report": latency, "episodes": []}
    output.mkdir(parents=True)
    for method_index, method in enumerate(sorted(methods)):
        for selected_method, case, item, paths in verified:
            if selected_method != method:
                continue
            directory = output / f"method{method_index:02d}" / case
            directory.mkdir(parents=True)
            for name, source in paths.items():
                _require(_sha(source) == item["source_file_sha256"][name], "source-changed-before-copy")
                target = directory / name
                shutil.copyfile(source, target)
                digest = _sha(target)
                _require(digest == item["source_file_sha256"][name] and _sha(source) == digest, "copy-identity-mismatch")
                item["files"][name] = {"path": target.relative_to(output).as_posix(), "sha256": digest}
            manifest["episodes"].append(item)
    manifest["m1_gate_pass"] = any(item["learned_camera_only"] and item["sustained_clean_route"] for item in manifest["episodes"])
    manifest["qualifying_routes"] = [{"method": item["method"], "case": item["case"], "source_episode": item["source_episode"]}
        for item in manifest["episodes"] if item["learned_camera_only"] and item["sustained_clean_route"]]
    page = output / "index.html"
    page.write_text(_page(manifest), encoding="utf-8")
    manifest["html_sha256"] = _sha(page)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return manifest
