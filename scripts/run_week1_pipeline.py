"""D6/D7 native geometry capture, immutable labels, overlays and offline gate.

The caller owns the already-running CARLA server. This same workload is bundled
into the reused Kaggle notebook; it never starts a local simulator by itself.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.color import ColorError, verify_rgb_color
from carla_tools.episode import validate_episode
from carla_tools.geometry_survey import load_geometry_catalog, survey_geometry_catalog
from carla_tools.labels import generate_labels, validate_labels
from carla_tools.projection import decode_rgb8_png, render_episode_overlay
from carla_tools.week1 import verify_week1_cases
from scripts.carla_rgb_probe import gpu_sample
from scripts.check_carla_connection import ConnectionConfig, check_connection
from scripts.run_geometry_collection import collect_geometry_cases
from scripts.verify_collection import write_report


class PipelineError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _require(report: dict, code: str):
    if report.get("status") != "PASS":
        raise PipelineError(code)


def negative_checks(episode: Path, labels: Path, temporary_parent: Path) -> dict:
    """Check actual source/label failures on copies; preserve all source bytes."""
    _require(validate_labels(episode, labels), "negative-QA-source-invalid")
    checks = []
    source = json.loads(labels.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="vlf_negative_", dir=temporary_parent) as directory:
        temporary = Path(directory).resolve()
        if not temporary.is_relative_to(temporary_parent.resolve()):
            raise PipelineError("QA-temporary-path-invalid")
        for name, mutate in (
            ("wrong_ego_frame", lambda value: value["rows"][0]["label"].update(ego_frame=value["rows"][0]["frame"] + 1)),
            ("wrong_units", lambda value: value["rows"][0]["label"].update(units="cm")),
            ("wrong_ego_target", lambda value: value["rows"][0]["label"]["waypoints_ego_m"][0].__setitem__(1, 100.0)),
            ("duplicate_label_frame", lambda value: value["rows"].__setitem__(1, copy.deepcopy(value["rows"][0]))),
            ("missing_label_frame", lambda value: value["rows"].pop()),
            ("wrong_source_hash", lambda value: value["source"].update(frames_jsonl_sha256="0" * 64)),
        ):
            value = copy.deepcopy(source)
            mutate(value)
            path = temporary / f"{name}.json"
            write_report(path, value)
            checked = validate_labels(episode, path)
            if checked["status"] != "FAIL":
                raise PipelineError("negative-label-check-incorrectly-passed")
            checks.append({"case": name, "expected_status": "FAIL", "observed_status": checked["status"], "errors": checked["errors"]})
        cloned = temporary / "raw-copy"
        shutil.copytree(episode, cloned)
        rows = [json.loads(line) for line in (cloned / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
        image = (cloned / rows[0]["image"]["path"]).resolve()
        if not image.is_relative_to((cloned / "images").resolve()):
            raise PipelineError("negative-QA-image-outside-owned-clone")
        original = image.read_bytes()
        damaged = bytearray(original)
        damaged[40] ^= 1
        image.write_bytes(damaged)
        checked = validate_episode(cloned)
        if checked["status"] != "FAIL":
            raise PipelineError("negative-png-check-incorrectly-passed")
        checks.append({"case": "corrupt_png", "expected_status": "FAIL", "observed_status": checked["status"], "errors": checked["errors"]})
        image.write_bytes(original)
        (cloned / "frames.jsonl").write_text("\n".join(json.dumps(row, allow_nan=False) for row in rows[1:]) + "\n", encoding="utf-8")
        checked = validate_episode(cloned)
        if checked["status"] != "FAIL":
            raise PipelineError("negative-frame-drop-check-incorrectly-passed")
        checks.append({"case": "dropped_raw_frame", "expected_status": "FAIL", "observed_status": checked["status"], "errors": checked["errors"]})
        width, height, rgb = decode_rgb8_png(original, rows[0]["image"]["sha256"])
        swapped = bytearray(rgb)
        swapped[0::3], swapped[2::3] = rgb[2::3], rgb[0::3]
        evidence = copy.deepcopy(rows[0]["color_provenance"])
        evidence["rgb_sha256"] = hashlib.sha256(swapped).hexdigest()
        try:
            verify_rgb_color(bytes(swapped), evidence, width, height, require_witness=True)
        except ColorError as error:
            checks.append({"case": "red_blue_swap_with_recomputed_rgb_hash", "expected_status": "FAIL", "observed_status": "FAIL", "errors": [{"code": error.code}]})
        else:
            raise PipelineError("negative-color-check-incorrectly-passed")
    return {"status": "PASS", "count": len(checks), "checks": checks, "source_files_modified": False}


def run_pipeline(output_dir: Path) -> dict:
    report = {"schema_version": 1, "status": "FAIL", "scope": "D6-D7-stationary-geometry-not-driving-pilot",
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "stage": "output-preflight"}
    started = time.perf_counter()
    owned = False

    def stage(name):
        report["stage"] = name
        print("D6/D7:", name, flush=True)

    try:
        if output_dir.exists() or output_dir.is_symlink():
            raise PipelineError("pipeline-output-already-exists")
        output_dir.mkdir(parents=True, exist_ok=False)
        owned = True
        stage("read-only-physical-arc-geometry-survey")
        carla = importlib.import_module("carla")
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(30)
        if client.get_client_version() != "0.9.16" or client.get_server_version() != "0.9.16":
            raise PipelineError("carla-version-mismatch")
        catalog = survey_geometry_catalog(client.get_world().get_map(), carla)
        catalog_path = output_dir / "geometry_routes.json"
        write_report(catalog_path, catalog)
        load_geometry_catalog(catalog_path)
        report["geometry_survey"] = {key: catalog[key] for key in ("status", "candidate_counts", "spawn_points_scanned", "classification_policy")}
        report["gpu_before_capture"] = gpu_sample()
        stage("same-frame-RGB-pose-and-color-proof-capture")
        captured = collect_geometry_cases(catalog_path, output_dir / "raw", ROOT / "configs/collection_d4.json", carla_module=carla)
        report["collection"] = captured
        _require(captured, "geometry-capture-failed")
        report["gpu_after_capture"] = gpu_sample()
        stage("immutable-derived-waypoint-labels-and-overlays")
        cases, summaries = [], []
        report["derived_artifacts"] = []
        for item in captured["cases"]:
            case = item["case"]
            episode = output_dir / "raw" / item["episode_directory"]
            labels = output_dir / "labels" / f"{case}.json"
            generated = generate_labels(episode, labels)
            artifact = {"case": case, "labels": {key: value for key, value in generated.items()
                         if key not in {"metadata", "frames", "sidecar"}}}
            report["derived_artifacts"].append(artifact)
            _require(generated, "label-generation-failed")
            overlay = render_episode_overlay(episode, labels, output_dir / "overlays" / case)
            artifact["overlay"] = overlay
            _require(overlay, "overlay-rendering-failed")
            cases.append({"case": case, "episode": str(episode), "labels": str(labels)})
            summaries.append({"case": case, "source": generated["source"], "label_metrics": generated["metrics"],
                              "labels_sha256": generated["labels_sha256"], "visible_points": overlay["visible_points"],
                              "all_points_in_front": overlay["all_points_in_front"]})
        stage("independent-week1-alignment-gate")
        report["gate"] = verify_week1_cases(cases)
        _require(report["gate"], "week1-gate-failed")
        report["episode_count"] = report["gate"]["episode_count"]
        report["frame_count"] = report["gate"]["frame_count"]
        report["episodes"] = summaries
        write_report(output_dir / "week1_manifest.json", {"schema_version": 1, "cases": [
            {"case": item["case"], "episode": str(Path(item["episode"]).relative_to(output_dir)).replace("\\", "/"),
             "labels": str(Path(item["labels"]).relative_to(output_dir)).replace("\\", "/")} for item in cases]})
        stage("native-negative-integrity-and-color-checks")
        report["negative_checks"] = negative_checks(Path(cases[1]["episode"]), Path(cases[1]["labels"]), output_dir)
        stage("post-capture-asynchronous-RPC")
        connection = check_connection(ConnectionConfig(timeout=30, ticks=3), carla_module=carla)
        report["connection_after"] = connection
        _require(connection, "post-capture-RPC-failed")
        report["status"] = "PASS"
        stage("verified")
    except BaseException as error:
        report["error"] = {"code": getattr(error, "code", "week1-pipeline-operation-failed"),
                           "exception_type": type(error).__name__, "stage": report["stage"]}
    report["elapsed_seconds"] = time.perf_counter() - started
    report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    if owned:
        write_report(output_dir / "week1_report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_pipeline(args.output_dir)
    print(json.dumps({key: report[key] for key in ("status", "stage", "episode_count", "frame_count", "elapsed_seconds", "error") if key in report}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
