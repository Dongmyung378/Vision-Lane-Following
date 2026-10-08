"""CPU-only audit of recorded CARLA lane-sensor corners against pinned Town01 XODR.

Only the CARLA Python API's in-process Map is constructed. No Client, RPC,
simulator process, rendering, GPU, or network operation is used.

Pinned official CARLA 0.9.16 definitions:
https://github.com/carla-simulator/carla/blob/0.9.16/LibCarla/source/carla/client/LaneInvasionSensor.cpp#L113-L123
https://github.com/carla-simulator/carla/blob/0.9.16/LibCarla/source/carla/road/Map.cpp#L153-L213

The native sensor adds the unrotated bbox center offset to vehicle location,
then rotates four XY extent corners using yaw alone. Map lookup chooses a
nearest segment in 3D, but tests containment using 2D distance. This audit
preserves native corner height and rejects height-sensitive query results.
It checks recorded poses; native event detection and the controller gate
remain separate evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[1]
XODR = ROOT / "carla/CARLA_0.9.16/CarlaUE4/Content/Carla/Maps/OpenDrive/Town01.xodr"
PINNED_SHA256 = "97a7f6ac67812567e5c8ee0599cd823b23f80f30f3f97c502212e38b72e2b709"
CASES = ("straight", "left_curve", "right_curve")
CONDITIONS = ("center", "left_offset", "right_offset", "left_yaw", "right_yaw")
SENSOR_SOURCE = "https://github.com/carla-simulator/carla/blob/0.9.16/LibCarla/source/carla/client/LaneInvasionSensor.cpp#L113-L123"
MAP_SOURCE = "https://github.com/carla-simulator/carla/blob/0.9.16/LibCarla/source/carla/road/Map.cpp#L153-L213"


class AuditError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def require(condition, code):
    if not condition:
        raise AuditError(code)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def linked_path(path):
    """Reject symlinks and Windows junctions, including ancestor components."""
    absolute = Path(path).absolute()
    for component in (absolute, *absolute.parents):
        if component.is_symlink() or getattr(component, "is_junction", lambda: False)():
            return True
    return False


def public_error(error, fallback):
    return {"code": error.code if isinstance(error, AuditError) else fallback,
            "exception_type": type(error).__name__}


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate-json-key")
            result[key] = value
        return result

    def constant(_):
        raise AuditError("nonfinite-json-number")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "invalid-recorded-coordinate")
    return float(value)


def integer(value):
    require(type(value) is int, "invalid-recorded-integer")
    return value


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def sensor_corners(pose, bbox):
    """Pinned sensor MakeBounds order and float32 yaw-only arithmetic.

    The bbox center offset is added in world axes WITHOUT rotation, exactly
    as CARLA's sensor does. Pitch, roll and vertical extents are not used.
    """
    location = {axis: f32(f32(number(pose["location"][axis])) + f32(number(bbox["location"][axis])))
                for axis in ("x", "y", "z")}
    extent_x, extent_y = (f32(number(bbox["extent"][axis])) for axis in ("x", "y"))
    require(extent_x > 0 and extent_y > 0, "invalid-recorded-bounding-box")
    angle = f32(f32(number(pose["rotation"]["yaw"])) * f32(f32(math.pi) / 180.0))
    cosine, sine = f32(math.cos(angle)), f32(math.sin(angle))
    result = []
    for index, (name, sx, sy) in enumerate((("front_right", 1, 1), ("rear_right", -1, 1),
                                           ("front_left", 1, -1), ("rear_left", -1, -1))):
        x, y = f32(sx * extent_x), f32(sy * extent_y)
        rotated_x = f32(f32(cosine * x) - f32(sine * y))
        rotated_y = f32(f32(sine * x) + f32(cosine * y))
        result.append({"index": index, "name": name,
                       "world_m": {"x": f32(location["x"] + rotated_x),
                                   "y": f32(location["y"] + rotated_y), "z": location["z"]}})
    return result


def lane_tuple(waypoint):
    if waypoint is None:
        return None
    return (int(waypoint.road_id), int(waypoint.section_id), int(waypoint.lane_id))


def audit(native, *, xodr_path=XODR, carla_module=None):
    report = {"schema_version": 1, "status": "FAIL", "scope": "recorded-main-and-stop-sensor-corner-map-audit",
              "execution": "CPU-only CARLA Map; no Client, RPC, server or GPU",
              "corner_rule_source": SENSOR_SOURCE, "map_query_source": MAP_SOURCE,
              "height_semantics": "nearest segment uses 3D; lane containment uses Distance2D < half lane width",
              "claim_limit": "Recorded poses only; this does not replace native events or the driving acceptance gate.",
              "expected_episode_count": 15, "episode_count": 0, "main_pose_count": 0, "stop_pose_count": 0,
              "corner_count": 0, "accepted_corner_count": 0, "bad_corner_count": 0,
              "height_sensitive_corner_count": 0, "first_bad_corner": None,
              "episodes": [], "sources": [], "errors": [], "sources_unchanged": False}
    sources = []
    try:
        native = Path(native)
        xodr_path = Path(xodr_path)
        require(not linked_path(native) and not linked_path(xodr_path), "audit-source-path-link-refused")
        require(native.is_dir(), "native-driving-directory-required")
        require(xodr_path.is_file(), "pinned-local-xodr-required")
        xml_bytes = xodr_path.read_bytes()
        xml_sha = digest(xml_bytes)
        report["opendrive_sha256"] = xml_sha
        require(xml_sha == PINNED_SHA256, "local-xodr-pinned-hash-mismatch")
        sources.append((xodr_path, xml_sha, {"source": "pinned-Town01.xodr", "sha256": xml_sha}))
        episodes_dir = native / "episodes"
        require(episodes_dir.is_dir() and not linked_path(episodes_dir), "native-episodes-directory-required")
        episode_dirs = sorted(episodes_dir.iterdir(), key=lambda item: item.name)
        require(episode_dirs, "native-episodes-empty")
        records = []
        found = set()
        for directory in episode_dirs:
            require(directory.is_dir() and not linked_path(directory), "invalid-episode-directory")
            require(re.fullmatch(r"[a-z][a-z0-9_]{1,63}", directory.name) is not None, "invalid-episode-identifier")
            metadata_path, trace_path = directory / "episode.json", directory / "steps.jsonl"
            require(all(path.is_file() and not linked_path(path) for path in (metadata_path, trace_path)), "episode-source-file-required")
            metadata_bytes, trace_bytes = metadata_path.read_bytes(), trace_path.read_bytes()
            for path, data in ((metadata_path, metadata_bytes), (trace_path, trace_bytes)):
                source = {"source": f"episodes/{directory.name}/{path.name}", "sha256": digest(data)}
                sources.append((path, source["sha256"], source))
            metadata = strict_json(metadata_bytes)
            require(isinstance(metadata, dict) and metadata.get("episode_id") == directory.name, "episode-identifier-mismatch")
            route = metadata["route_contract"]
            require(route["opendrive_sha256"] == xml_sha, "episode-xodr-hash-mismatch")
            require(route["town"] == "Town01" and route["carla_version"] == "0.9.16", "episode-native-map-version-mismatch")
            allowed = {(integer(anchor["road_id"]), integer(anchor["section_id"]), integer(anchor["lane_id"]))
                       for anchor in route["centerline_anchor_sequence"]}
            require(allowed and all(lane[2] != 0 for lane in allowed), "fixed-original-route-lanes-invalid")
            condition = metadata["condition"]["name"]
            case = metadata["geometry_case"]
            require(case in CASES and condition in CONDITIONS and directory.name == f"{case}__{condition}", "unexpected-driving-case-condition")
            require((case, condition) not in found, "duplicate-driving-case-condition")
            found.add((case, condition))
            main = [strict_json(line) for line in trace_bytes.decode("utf-8").splitlines() if line.strip()]
            stop = metadata.get("stop_observations", [])
            require(main and isinstance(stop, list), "recorded-poses-missing")
            if not stop:
                report["errors"].append({"code": "recorded-stop-poses-missing", "episode_id": directory.name})
            if metadata.get("status") != "COMPLETE":
                report["errors"].append({"code": "source-episode-not-complete", "episode_id": directory.name})
            require(metadata.get("trace_sha256") == digest(trace_bytes), "episode-trace-hash-mismatch")
            records.append((directory.name, metadata["bounding_box"], allowed, main, stop))
        report["episode_count"] = len(records)
        if found != {(case, condition) for case in CASES for condition in CONDITIONS}:
            report["errors"].append({"code": "expected-fifteen-case-condition-coverage-missing"})

        # Import only after source files and every metadata XODR hash agree.
        if carla_module is None:
            require(importlib.metadata.version("carla") == "0.9.16", "carla-api-version-mismatch")
            carla_module = importlib.import_module("carla")
        native_map = carla_module.Map("Town01", xml_bytes.decode("utf-8"))
        driving = carla_module.LaneType.Driving
        for episode_id, bbox, allowed, main, stop in records:
            episode = {"episode_id": episode_id, "allowed_original_route_lanes": [list(lane) for lane in sorted(allowed)],
                       "main_pose_count": len(main), "stop_pose_count": len(stop), "bad_corner_count": 0,
                       "height_sensitive_corner_count": 0, "first_bad_corner": None}
            report["episodes"].append(episode)
            report["main_pose_count"] += len(main)
            report["stop_pose_count"] += len(stop)
            previous_frame = None
            for phase, rows in (("main", main), ("stop", stop)):
                for row in rows:
                    frame = integer(row["frame"])
                    require(frame >= 0 and (previous_frame is None or frame == previous_frame + 1), "recorded-frame-continuity-mismatch")
                    previous_frame = frame
                    number(row["timestamp"])
                    for corner in sensor_corners(row["vehicle"]["transform"], bbox):
                        point = corner["world_m"]
                        waypoint = native_map.get_waypoint(carla_module.Location(**point), project_to_road=False, lane_type=driving)
                        actual_lane = lane_tuple(waypoint)
                        ground = native_map.get_waypoint(carla_module.Location(x=point["x"], y=point["y"], z=0.0),
                                                        project_to_road=False, lane_type=driving)
                        ground_lane = lane_tuple(ground)
                        height_sensitive = actual_lane != ground_lane
                        code = None
                        if waypoint is None:
                            code = "sensor-corner-outside-driving-lane"
                        elif waypoint.lane_type != driving:
                            code = "sensor-corner-not-driving-lane"
                        elif actual_lane not in allowed:
                            code = "sensor-corner-outside-original-route-lanes"
                        elif height_sensitive:
                            code = "sensor-corner-height-projection-sensitive"
                        report["corner_count"] += 1
                        if height_sensitive:
                            report["height_sensitive_corner_count"] += 1
                            episode["height_sensitive_corner_count"] += 1
                        if code is None:
                            report["accepted_corner_count"] += 1
                        else:
                            report["bad_corner_count"] += 1
                            episode["bad_corner_count"] += 1
                            bad = {"code": code, "episode_id": episode_id, "phase": phase, "frame": frame,
                                   "corner_index": corner["index"], "corner": corner["name"], "world_m": point,
                                   "actual_lane": list(actual_lane) if actual_lane is not None else None,
                                   "ground_z_zero_lane": list(ground_lane) if ground_lane is not None else None,
                                   "height_sensitive": height_sensitive}
                            if episode["first_bad_corner"] is None:
                                episode["first_bad_corner"] = bad
                            if report["first_bad_corner"] is None:
                                report["first_bad_corner"] = bad
        if report["bad_corner_count"]:
            report["errors"].append({"code": "recorded-sensor-corner-map-check-failed"})
    except BaseException as error:
        report["errors"].append(public_error(error, "map-audit-operation-failed"))
    finally:
        report["sources"] = [source for _, _, source in sources]
        unchanged = bool(sources)
        for path, original_sha, source in sources:
            try:
                source["unchanged"] = path.is_file() and not linked_path(path) and digest(path.read_bytes()) == original_sha
            except BaseException:
                source["unchanged"] = False
            unchanged = unchanged and source["unchanged"]
        report["sources_unchanged"] = unchanged
        if not unchanged:
            report["errors"].append({"code": "audit-source-snapshot-not-unchanged"})
        if not report["errors"]:
            report["status"] = "PASS"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path, help="Expanded native artifact driving directory")
    parser.add_argument("--output", required=True, type=Path, help="New JSON report path; existing output is refused")
    parser.add_argument("--xodr", type=Path, default=XODR, help="Pinned CARLA 0.9.16 Town01 XODR; defaults to the local CARLA installation")
    args = parser.parse_args(argv)
    handle = None
    try:
        require(not linked_path(args.output), "audit-output-path-link-refused")
        require(not args.output.exists(), "audit-output-already-exists")
        require(not linked_path(args.native) and not linked_path(args.xodr), "audit-source-path-link-refused")
        output = args.output.resolve()
        native = args.native.resolve()
        require(not output.is_relative_to(native), "audit-output-inside-native-sources-refused")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        require(not linked_path(args.output), "audit-output-path-link-refused")
        # Reserve a fresh output exclusively BEFORE the audit. An interrupted
        # run leaves FAIL, while an existing file is never reused as evidence.
        handle = args.output.open("x", encoding="utf-8", newline="\n")
        json.dump({"schema_version": 1, "status": "FAIL", "errors": [{"code": "map-audit-not-finished"}]}, handle)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        report = audit(args.native, xodr_path=args.xodr)
        serialized = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        handle.seek(0)
        handle.write(serialized)
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        print(json.dumps({"status": report["status"], "episode_count": report["episode_count"],
                          "corner_count": report["corner_count"], "bad_corner_count": report["bad_corner_count"],
                          "height_sensitive_corner_count": report["height_sensitive_corner_count"],
                          "sources_unchanged": report["sources_unchanged"]}))
        return 0 if report["status"] == "PASS" else 1
    except BaseException as error:
        failure = {"schema_version": 1, "status": "FAIL", "error": public_error(error, "map-audit-cli-operation-failed")}
        if handle is not None:
            try:
                handle.seek(0)
                handle.write(json.dumps(failure, ensure_ascii=False, allow_nan=False) + "\n")
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                pass
        print(json.dumps(failure))
        return 1
    finally:
        if handle is not None:
            try:
                handle.close()
            except BaseException:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
