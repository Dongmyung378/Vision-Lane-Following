"""Source-bound split tables and same-frame GT label inspection artifacts.

This derives small CPU artifacts from a freshly validated dataset. It never
starts a simulator, runs inference or repeats the full native episode audit.
"""
from __future__ import annotations

import csv
import hashlib
import html
import io
import json
from collections import defaultdict
from pathlib import Path

from .color import verify_rgb_color
from .contracts import canonical_sha256, strict_json_loads, validate_waypoint_label
from .dataset_v1 import GATE_VERSION, SCOPE, dataset_v1_config_from_dict, match_reference_arc_pairs, select_training_sets
from .geometry_survey import CASES
from .pilot import _safe_file
from .projection import decode_rgb8_png, draw_overlay_rgb, encode_rgb8_png, project_waypoints

ARTIFACT_SCOPE = "dataset-v1-split-and-same-frame-label-artifacts"
VERSION = "vlf-dataset-artifacts-v1"
SPLIT_FIELDS = ("dataset_version", "sample_id", "route_id", "episode_id", "scene_group_id", "split")
SELECTION_FIELDS = ("selection_index", *SPLIT_FIELDS, "image", "image_sha256")
GATE_CHECKS = ("all_native_episodes_replayed", "same_frame_gt_labels", "every_source_png",
               "all_excluded_sources_retained", "route_scene_spatial_split", "reference_arc_pairs",
               "matched_training_budget", "common_dev", "immutable_source_snapshot", "reserved_final_town_unopened")


class DatasetArtifactError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _require(value, code):
    if not value:
        raise DatasetArtifactError(code)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def _load_bound(manifest_path, gate):
    path = Path(manifest_path)
    payload = _safe_file(path.parent, path.name).read_bytes()
    manifest = strict_json_loads(payload.decode("utf-8"))
    _require(type(manifest) is dict and manifest.get("scope") == SCOPE and manifest.get("status") == "COMPLETE",
             "dataset-artifacts-manifest-invalid")
    snapshot = dict(manifest)
    digest = snapshot.pop("snapshot_sha256", None)
    _require(digest == canonical_sha256(snapshot), "dataset-artifacts-snapshot-mismatch")
    _require(type(gate) is dict and gate.get("status") == "PASS" and gate.get("scope") == SCOPE
             and gate.get("gate_version") == GATE_VERSION and gate.get("errors") == []
             and gate.get("manifest_sha256") == _sha(payload) and gate.get("snapshot_sha256") == digest
             and gate.get("sample_count") == manifest.get("sample_count")
             and gate.get("config_sha256") == manifest.get("config_sha256")
             and all(gate.get("checks", {}).get(name) is True for name in GATE_CHECKS),
             "dataset-artifacts-unbound-native-gate")
    config = dataset_v1_config_from_dict(manifest["config"])
    _require(config.sha256 == manifest["config_sha256"], "dataset-artifacts-config-mismatch")
    samples = manifest["samples"]
    lookup = {sample["sample_id"]: sample for sample in samples}
    _require(len(lookup) == len(samples) == manifest["sample_count"], "dataset-artifacts-sample-identity-invalid")
    pairs = match_reference_arc_pairs(samples, config)
    _require(pairs == manifest["pairs"] and select_training_sets(samples, pairs, config) == manifest["selections"],
             "dataset-artifacts-selection-mismatch")
    return path, payload, manifest, config, lookup


def _csv_bytes(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _table_payloads(manifest, lookup):
    def row(sample):
        return {"dataset_version": manifest["dataset_version"], **{key: sample[key] for key in SPLIT_FIELDS[1:]}}

    payloads = {"split_manifest.csv": _csv_bytes(SPLIT_FIELDS, (row(sample) for sample in manifest["samples"]))}
    selections = manifest["selections"]
    groups = {"central_only_train": selections["central_only"]["train_sample_ids"],
              "central_recovery_train": selections["central_recovery"]["train_sample_ids"],
              "common_dev": selections["common_dev_sample_ids"]}
    for name, identifiers in groups.items():
        expected_split = "dev" if name == "common_dev" else "train"
        _require(len(set(identifiers)) == len(identifiers)
                 and all(lookup[key]["split"] == expected_split for key in identifiers),
                 "dataset-artifacts-selection-split-invalid")
        payloads[name + ".csv"] = _csv_bytes(SELECTION_FIELDS,
            ({"selection_index": index, **row(lookup[key]), "image": lookup[key]["image"],
              "image_sha256": lookup[key]["image_sha256"]} for index, key in enumerate(identifiers)))
    payloads["training_selections.json"] = _json({"schema_version": 1,
        "source_manifest_sha256": None, "image_paths_relative_to": "source-manifest-parent",
        "dataset_version": manifest["dataset_version"], "snapshot_sha256": manifest["snapshot_sha256"],
        "training_samples_per_arm": selections["training_samples_per_arm"],
        "central_only_train_sample_ids": groups["central_only_train"],
        "central_recovery_train_sample_ids": groups["central_recovery_train"],
        "common_dev_sample_ids": groups["common_dev"]})
    return payloads


def _visual_selection(manifest, config):
    groups = defaultdict(list)
    for sample in manifest["samples"]:
        groups[sample["split"], sample["case"], sample["weather"], sample["role"]].append(sample)
    result = []
    for split in ("train", "dev"):
        for case in CASES:
            for weather in config.weather_presets:
                for role in ("central", "recovery"):
                    members = sorted(groups[split, case, weather, role],
                                     key=lambda sample: (sample["reference_arc_m"], sample["sample_id"]))
                    _require(members, "dataset-artifacts-visual-stratum-missing")
                    indices = [("early", 0)]
                    if len(members) > 1:
                        indices.append(("middle", len(members) // 2))
                    for position, index in indices:
                        result.append((position, members[index]))
    return result


def _source_record(root, manifest, sample, cache):
    descriptors = {item["episode_id"]: item for item in manifest["episodes"]}
    descriptor = descriptors[sample["episode_id"]]
    if sample["episode_id"] not in cache:
        meta_name, trace_name = descriptor["episode"] + "/episode.json", descriptor["episode"] + "/steps.jsonl"
        metadata_bytes, trace_bytes = _safe_file(root, meta_name).read_bytes(), _safe_file(root, trace_name).read_bytes()
        _require(_sha(metadata_bytes) == descriptor["episode_json_sha256"] == manifest["source_files_sha256"][meta_name]
                 and _sha(trace_bytes) == descriptor["steps_jsonl_sha256"] == manifest["source_files_sha256"][trace_name],
                 "dataset-artifacts-source-record-changed")
        metadata = strict_json_loads(metadata_bytes.decode("utf-8"))
        rows = [strict_json_loads(line) for line in trace_bytes.decode("utf-8").splitlines()]
        _require(len({row["frame"] for row in rows}) == len(rows), "dataset-artifacts-source-frame-duplicate")
        cache[sample["episode_id"]] = metadata, {row["frame"]: row for row in rows}
    metadata, rows = cache[sample["episode_id"]]
    row = rows[sample["frame"]]
    label = row["tracking"]["label"]
    validate_waypoint_label(label, row, {"route_id": sample["route_id"], "route_contract": metadata["route_contract"]})
    _require(sample["waypoints_ego_m"] == label["waypoints_ego_m"]
             and sample["vehicle_transform"] == row["vehicle"]["transform"]
             and sample["calibration_sha256"] == canonical_sha256(metadata["camera_calibration"])
             and sample["timestamp"] == row["timestamp"]
             and metadata["route_contract"]["route_id"] == sample["route_id"]
             and sample["image"] == descriptor["episode"] + "/" + row["image"]["path"]
             and sample["image_sha256"] == row["image"]["sha256"] == manifest["source_files_sha256"][sample["image"]],
             "dataset-artifacts-same-frame-binding-mismatch")
    return descriptor, metadata, row


def _render(root, manifest, sample, position, cache):
    descriptor, metadata, row = _source_record(root, manifest, sample, cache)
    payload = _safe_file(root, sample["image"]).read_bytes()
    width, height, rgb = decode_rgb8_png(payload, sample["image_sha256"], (640, 360))
    verify_rgb_color(rgb, row["image"]["color_provenance"], width, height, require_witness=True)
    projection = project_waypoints(row["tracking"]["label"]["waypoints_world_m"],
                                   row["camera"]["transform"]["matrix"], metadata["camera_calibration"])
    overlay = draw_overlay_rgb(rgb, width, height, projection)
    key = _sha(sample["sample_id"].encode("utf-8"))[:24] + "__" + position
    name = "overlays/" + key + ".png"
    provenance = {"sample_id": sample["sample_id"], "episode_id": sample["episode_id"], "frame": sample["frame"],
        "timestamp": sample["timestamp"], "split": sample["split"], "case": sample["case"], "weather": sample["weather"],
        "role": sample["role"], "condition": sample["condition"], "position": position,
        "route_id": sample["route_id"], "route_sha256": metadata["route_contract"]["route_sha256"],
        "scene_group_id": sample["scene_group_id"], "source_image": sample["image"],
        "source_image_sha256": sample["image_sha256"], "episode_json_sha256": descriptor["episode_json_sha256"],
        "steps_jsonl_sha256": descriptor["steps_jsonl_sha256"], "calibration_sha256": sample["calibration_sha256"],
        "actor_world_matrix": row["vehicle"]["transform"]["matrix"],
        "camera_world_matrix": row["camera"]["transform"]["matrix"],
        "waypoints_ego_m": sample["waypoints_ego_m"], "projections": projection, "overlay": name,
        "visible_points": sum(point["status"] == "VISIBLE" for point in projection),
        "all_points_in_front": all(point["depth_m"] > 0 for point in projection), "prediction_or_inference_used": False}
    return name, encode_rgb8_png(overlay, width, height), provenance, overlay


def _contact_sheet(cells):
    import cv2
    import numpy as np
    columns, tile_width, tile_height = 4, 320, 210
    rows = (len(cells) + columns - 1) // columns
    output = np.full((rows * tile_height, columns * tile_width, 3), 20, dtype=np.uint8)
    for index, (record, pixels) in enumerate(cells):
        left, top = (index % columns) * tile_width, (index // columns) * tile_height
        thumbnail = np.frombuffer(pixels, dtype=np.uint8).reshape(360, 640, 3)[::2, ::2]
        output[top:top + 180, left:left + tile_width] = thumbnail
        text = f"{record['case']} {record['role']} {record['position']}"
        cv2.putText(output, text, (left + 3, top + 192), cv2.FONT_HERSHEY_SIMPLEX, .34, (240, 240, 240), 1, cv2.LINE_8)
        cv2.putText(output, f"frame {record['frame']} | {record['condition']}", (left + 3, top + 205),
                    cv2.FONT_HERSHEY_SIMPLEX, .34, (240, 240, 240), 1, cv2.LINE_8)
    return encode_rgb8_png(output.tobytes(), output.shape[1], output.shape[0])


def _artifact_payloads(path, payload, manifest, config, lookup):
    files = _table_payloads(manifest, lookup)
    selection_json = strict_json_loads(files["training_selections.json"].decode("utf-8"))
    selection_json["source_manifest_sha256"] = _sha(payload)
    files["training_selections.json"] = _json(selection_json)
    records, cache, sheets = [], {}, defaultdict(list)
    for position, sample in _visual_selection(manifest, config):
        name, encoded, record, pixels = _render(path.parent, manifest, sample, position, cache)
        files[name] = encoded
        files[name[:-4] + ".json"] = _json(record)
        records.append(record)
        sheets[sample["split"], sample["weather"]].append((record, pixels))
    sheet_records = []
    for (split, weather), cells in sorted(sheets.items()):
        name = f"contact_sheets/{split}__{weather}.png"
        files[name] = _contact_sheet(cells)
        sheet_records.append({"image": name, "split": split, "weather": weather,
                              "columns": 4, "sample_ids": [record["sample_id"] for record, _ in cells]})
    files["visualization_manifest.json"] = _json({"schema_version": 1, "source_manifest_sha256": _sha(payload),
        "snapshot_sha256": manifest["snapshot_sha256"], "selection_policy": "lowest-and-middle-reference-arc-per-stratum",
        "overlay_count": len(records), "records": records, "contact_sheets": sheet_records})
    body = ["<!doctype html><meta charset='utf-8'><title>Dataset label inspection</title>",
            "<style>body{font:16px sans-serif;margin:2rem}img{max-width:100%}figure{margin:1rem 0}code{overflow-wrap:anywhere}</style>",
            "<h1>Dataset label inspection</h1><p>Original RGB with same-frame ground-truth projections. No model inference. "
            "Inspect the fixed strata and their original provenance; this view does not certify driving performance.</p>"]
    for sheet in sheet_records:
        body.append(f"<h2>{html.escape(sheet['split'])} / {html.escape(sheet['weather'])}</h2>"
                    f"<a href='{sheet['image']}'><img src='{sheet['image']}'></a>")
    for record in records:
        body.append(f"<figure><figcaption>{html.escape(record['sample_id'])} | {html.escape(record['split'])} | "
                    f"{html.escape(record['case'])} | {html.escape(record['weather'])} | {html.escape(record['role'])} | "
                    f"{record['position']} | {record['visible_points']}/5 visible</figcaption>"
                    f"<a href='{record['overlay'][:-4]}.json'>provenance</a><br><img src='{record['overlay']}'></figure>")
    files["index.html"] = "\n".join(body).encode("utf-8") + b"\n"
    return files, records, sheet_records


def _inspected_sources(manifest, records):
    episodes = {item["episode_id"]: item for item in manifest["episodes"]}
    names = set()
    for record in records:
        descriptor = episodes[record["episode_id"]]
        names.update((record["source_image"], descriptor["episode"] + "/episode.json",
                      descriptor["episode"] + "/steps.jsonl"))
    return {name: manifest["source_files_sha256"][name] for name in sorted(names)}


def _assert_inspected_sources(root, sources):
    _require(all(_sha(_safe_file(root, name).read_bytes()) == digest for name, digest in sources.items()),
             "dataset-artifacts-inspected-source-changed")


def _inventory(payload, manifest, config, gate, files, records, sheets):
    return {"schema_version": 1, "scope": ARTIFACT_SCOPE, "version": VERSION,
        "source_manifest_sha256": _sha(payload), "snapshot_sha256": manifest["snapshot_sha256"],
        "validated_gate_sha256": canonical_sha256(gate), "sample_count": manifest["sample_count"],
        "training_samples_per_arm": config.train_selection_samples, "overlay_count": len(records),
        "contact_sheet_count": len(sheets), "inspected_source_files_sha256": _inspected_sources(manifest, records),
        "files_sha256": {name: _sha(content) for name, content in sorted(files.items())}}


def export_dataset_artifacts(manifest_path, validated_gate, output_dir):
    """Export deterministic private artifacts after a full source gate passes.

    Source PNGs are read only for the selected inspection frames. Errors leave
    a FAIL report in an owned new output directory; existing outputs are never
    overwritten. Returned hashes bind the generated views to the source.
    """
    output, owned = Path(output_dir), False
    report = {"schema_version": 1, "status": "FAIL", "scope": ARTIFACT_SCOPE, "version": VERSION, "errors": []}
    try:
        _require(not output.exists() and not output.is_symlink(), "dataset-artifacts-output-already-exists")
        path, payload, manifest, config, lookup = _load_bound(manifest_path, validated_gate)
        files, records, sheets = _artifact_payloads(path, payload, manifest, config, lookup)
        inventory = _inventory(payload, manifest, config, validated_gate, files, records, sheets)
        _assert_inspected_sources(path.parent, inventory["inspected_source_files_sha256"])
        _require(path.read_bytes() == payload, "dataset-artifacts-manifest-changed")
        output.mkdir(parents=True, exist_ok=False)
        owned = True
        for name, content in sorted(files.items()):
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(content)
        with (output / "artifact_manifest.json").open("xb") as stream:
            stream.write(_json(inventory))
        _assert_inspected_sources(path.parent, inventory["inspected_source_files_sha256"])
        _require(path.read_bytes() == payload, "dataset-artifacts-manifest-changed")
        report.update(status="PASS", manifest_sha256=_sha(payload), snapshot_sha256=manifest["snapshot_sha256"],
            validated_gate_sha256=canonical_sha256(validated_gate), artifact_manifest_sha256=_sha(_json(inventory)),
            sample_count=manifest["sample_count"], training_samples_per_arm=config.train_selection_samples,
            overlay_count=len(records), visual_stratum_count=len({(item["split"], item["case"], item["weather"], item["role"]) for item in records}),
            contact_sheet_count=len(sheets), files_sha256=inventory["files_sha256"],
            full_native_replay_repeated=False, prediction_or_inference_used=False)
    except (Exception, KeyboardInterrupt) as error:
        report["errors"].append({"code": getattr(error, "code", "dataset-artifacts-operation-failed"),
                                 "exception_type": type(error).__name__})
    if owned:
        with (output / "export_report.json").open("xb") as stream:
            stream.write(_json(report))
    return report


def validate_dataset_artifacts(manifest_path, validated_gate, output_dir):
    """Reconstruct the small exported views, without repeating native driving."""
    report = {"status": "FAIL", "scope": ARTIFACT_SCOPE, "version": VERSION, "errors": []}
    try:
        output = Path(output_dir)
        path, payload, manifest, config, lookup = _load_bound(manifest_path, validated_gate)
        inventory = strict_json_loads(_safe_file(output, "artifact_manifest.json").read_bytes().decode("utf-8"))
        _require(inventory["source_manifest_sha256"] == _sha(payload)
                 and inventory["validated_gate_sha256"] == canonical_sha256(validated_gate),
                 "dataset-artifacts-inventory-unbound")
        expected, records, sheets = _artifact_payloads(path, payload, manifest, config, lookup)
        expected_inventory = _inventory(payload, manifest, config, validated_gate, expected, records, sheets)
        _require(inventory == expected_inventory,
                 "dataset-artifacts-reconstruction-mismatch")
        for name, content in expected.items():
            _require(_safe_file(output, name).read_bytes() == content, "dataset-artifacts-file-changed")
        _assert_inspected_sources(path.parent, inventory["inspected_source_files_sha256"])
        actual_files = {item.relative_to(output).as_posix() for item in output.rglob("*") if item.is_file()}
        _require(actual_files == set(expected) | {"artifact_manifest.json", "export_report.json"}
                 and not any(item.is_symlink() for item in output.rglob("*")), "dataset-artifacts-unbound-extra-file")
        exported = strict_json_loads(_safe_file(output, "export_report.json").read_bytes().decode("utf-8"))
        _require(exported["status"] == "PASS" and exported["manifest_sha256"] == _sha(payload)
                 and exported["artifact_manifest_sha256"] == _sha(_json(inventory)),
                 "dataset-artifacts-export-report-mismatch")
        report.update(status="PASS", manifest_sha256=_sha(payload), overlay_count=len(records),
                      contact_sheet_count=len(sheets), full_native_replay_repeated=False)
    except (Exception, KeyboardInterrupt) as error:
        report["errors"].append({"code": getattr(error, "code", "dataset-artifacts-validation-failed"),
                                 "exception_type": type(error).__name__})
    return report
