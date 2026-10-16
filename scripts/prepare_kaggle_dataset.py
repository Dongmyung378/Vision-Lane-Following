"""Package dataset collection for an update to the existing private notebook."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.contracts import canonical_sha256, read_strict_json, validate_route_contract
from carla_tools.geometry_survey import CASES, classify_geometry
from carla_tools.pilot import _lower_bound
from scripts.prepare_kaggle_week1 import NOTEBOOK_ID, NOTEBOOK_TITLE, prepare

EXTRA_SOURCES = ("carla_tools/dataset_v1.py", "carla_tools/dataset_learning.py", "carla_tools/learning_data.py",
                 "carla_tools/dataset_artifacts.py", "configs/learning.json", "scripts/run_dataset_pipeline.py", "configs/dataset_v1.json")


def validate_frozen_catalog(catalog, config):
    routes = catalog.get("routes", [])
    identities, counts = set(), {}
    maps, scene_splits = set(), {}
    for item in routes:
        if type(item) is not dict or set(item) != {"split", "case", "route"}:
            raise ValueError("dataset-frozen-route-fields-invalid")
        route, split, case = item["route"], item["split"], item["case"]
        validate_route_contract(route)
        if (split not in ("train", "dev") or case not in CASES or route["town"] != "Town01"
                or classify_geometry(route)["case"] != case or route["route_id"] in identities):
            raise ValueError("dataset-frozen-route-identity-invalid")
        identities.add(route["route_id"])
        scene = route["scene_group_id"]
        if scene in scene_splits and scene_splits[scene] != split:
            raise ValueError("dataset-frozen-scene-group-split-leakage")
        scene_splits[scene] = split
        counts[split, case] = counts.get((split, case), 0) + 1
        maps.add(route["opendrive_sha256"])
    if (len(maps) != 1 or any(counts.get((split, case), 0) != getattr(config, split + "_routes_per_case")[case]
                             for split in ("train", "dev") for case in CASES)):
        raise ValueError("dataset-frozen-route-coverage-invalid")
    for first in routes:
        for second in routes:
            if first["split"] != second["split"] and _lower_bound(first["route"], second["route"]) <= config.scene_buffer_m:
                raise ValueError("dataset-frozen-cross-split-corridor-overlap")
    return routes


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("dataset-bootstrap-marker-mismatch")
    return source.replace(old, new, 1)


def _failure_retention_source():
    """Generate parent-side recovery; a killed child cannot finish its ZIP."""
    return '''def preserve_baseline_failure_artifacts(output: Path, *, artifact="baseline", source_directory=None, owned_root=None):
    import hashlib
    import json
    import os
    import shutil
    import zipfile
    from pathlib import Path

    if artifact != "dataset" or source_directory is None or owned_root is None:
        raise BootstrapFailure("dataset-retention-owned-root-required")
    output, owner, directory = Path(output), Path(owned_root), Path(source_directory)
    if (output.is_symlink() or owner.is_symlink() or not output.is_dir() or not owner.is_dir()
            or directory.is_symlink() or directory.name != "dataset"
            or directory.resolve().parent != owner.resolve()
            or output.resolve().is_relative_to(directory.resolve())):
        raise BootstrapFailure("dataset-retention-directory-invalid")
    if not directory.exists():
        return {"status": "NOT_PRESENT", "source_file_count": 0}
    if not directory.is_dir():
        raise BootstrapFailure("dataset-retention-source-not-directory")
    destination = output / "dataset_artifacts.zip"
    if destination.is_symlink() or destination.exists() and not destination.is_file():
        raise BootstrapFailure("dataset-retention-destination-invalid")

    def identity(path):
        value = path.stat()
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    inventory = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise BootstrapFailure("dataset-retention-source-escaped")
        if path.is_file():
            inventory[(Path("dataset") / path.relative_to(directory)).as_posix()] = (path, identity(path))

    def digest(path):
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
        return value.hexdigest()

    def verify(path):
        try:
            before = identity(path)
            with zipfile.ZipFile(path) as archive:
                entries = archive.infolist()
                if (len(entries) != len(inventory) or len({item.filename for item in entries}) != len(entries)
                        or {item.filename for item in entries} != set(inventory)):
                    return False, "archive-inventory-mismatch"
                for entry in entries:
                    source, source_identity = inventory[entry.filename]
                    if entry.is_dir() or entry.file_size != source_identity[2] or identity(source) != source_identity:
                        return False, "archive-source-identity-mismatch"
                    # Reading to EOF verifies ZIP CRC. Compare the actual
                    # bytes as well: a valid prefix ZIP or stale same-size ZIP
                    # is insufficient proof of the complete native snapshot.
                    with source.open("rb") as native, archive.open(entry) as packed:
                        while True:
                            original = native.read(1024 * 1024)
                            observed = packed.read(1024 * 1024)
                            if original != observed:
                                return False, "archive-source-bytes-mismatch"
                            if not original:
                                break
                    if identity(source) != source_identity:
                        return False, "archive-source-changed"
            if identity(path) != before:
                return False, "archive-changed-during-verification"
            return True, None
        except (OSError, ValueError, EOFError, RuntimeError, zipfile.BadZipFile):
            return False, "archive-unreadable-or-truncated"

    def unique_path(parent, stem, suffix):
        for index in range(10000):
            candidate = parent / (stem + "_" + str(index).zfill(4) + suffix)
            if not candidate.exists() and not candidate.is_symlink():
                if candidate.resolve().parent != parent.resolve():
                    raise BootstrapFailure("dataset-retention-path-escaped")
                return candidate
        raise BootstrapFailure("dataset-retention-attempt-budget-exhausted")

    def transfer(source, target):
        if target.exists() or target.is_symlink() or source.is_symlink():
            raise BootstrapFailure("dataset-retention-transfer-collision")
        try:
            source.rename(target)
            return
        except OSError:
            if target.exists() or target.is_symlink():
                raise BootstrapFailure("dataset-retention-transfer-destination-changed")
        # /tmp and /kaggle/working can use different mounts. Keep the source
        # until a complete, byte-verified destination has been synchronized.
        before = identity(source)
        checksum = hashlib.sha256()
        with source.open("rb") as reader, target.open("xb") as writer:
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(block)
                checksum.update(block)
            writer.flush()
            os.fsync(writer.fileno())
        if identity(source) != before or digest(target) != checksum.hexdigest():
            raise BootstrapFailure("dataset-retention-transfer-verification-failed")
        source.unlink()

    evidence_path = output / "dataset_archive_retention.json"
    if evidence_path.is_symlink() or evidence_path.exists() and not evidence_path.is_file():
        raise BootstrapFailure("dataset-retention-evidence-path-invalid")
    events = json.loads(evidence_path.read_text(encoding="utf-8"))["events"] if evidence_path.exists() else []
    if not isinstance(events, list):
        raise BootstrapFailure("dataset-retention-evidence-invalid")
    event = {"status": "RUNNING", "artifact": artifact, "source_file_count": len(inventory),
             "source_bytes": sum(value[1][2] for value in inventory.values()),
             "archive_existed": destination.exists(), "dataset_quality_status": "NOT_EVALUATED"}

    def record():
        temporary = unique_path(output, "dataset_archive_retention", ".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump({"schema_version": 1, "events": events + [event]}, stream, indent=2)
            stream.write("\\n")
            stream.flush()
            os.fsync(stream.fileno())
        if evidence_path.is_symlink():
            raise BootstrapFailure("dataset-retention-evidence-path-changed")
        os.replace(temporary, evidence_path)

    if destination.exists():
        complete, reason = verify(destination)
        if complete:
            event.update(status="PRESERVED_COMPLETE", archive_bytes=destination.stat().st_size,
                         archive_sha256=digest(destination))
            record()
            return event
        event.update(incomplete_reason=reason, incomplete_archive_bytes=destination.stat().st_size,
                     incomplete_archive_sha256=digest(destination))
        # Preserve the broken artifact in owned scratch. Keeping two large
        # ZIPs in persistent output could exceed Kaggle's output allowance.
        backup = unique_path(owner, "dataset_incomplete_artifacts", ".zip")
        event["retained_incomplete_archive_in_owned_scratch"] = backup.name
        record()
        try:
            transfer(destination, backup)
        except BaseException as error:
            event.update(status="RECOVERY_FAILED", error=type(error).__name__)
            record()
            raise
    else:
        record()
    temporary = unique_path(owner, "dataset_recovered_artifacts", ".zip")
    try:
        if shutil.disk_usage(owner).free < event["source_bytes"] + 512 * 2**20:
            raise BootstrapFailure("dataset-retention-free-space-insufficient")
        with zipfile.ZipFile(temporary, "x", allowZip64=True) as archive:
            for name, (path, before) in inventory.items():
                if identity(path) != before:
                    raise BootstrapFailure("dataset-retention-source-changed")
                archive.write(path, name, compress_type=zipfile.ZIP_STORED if path.suffix == ".png" else zipfile.ZIP_DEFLATED,
                              compresslevel=None if path.suffix == ".png" else 1)
        complete, reason = verify(temporary)
        if not complete:
            raise BootstrapFailure("dataset-retention-recovery-verification-failed:" + reason)
        transfer(temporary, destination)
        event.update(status="RECOVERED_COMPLETE", archive_bytes=destination.stat().st_size,
                     archive_sha256=digest(destination))
        record()
        return event
    except BaseException as error:
        event.update(status="RECOVERY_FAILED", error=type(error).__name__,
                     retained_recovery_archive_in_owned_scratch=temporary.name if temporary.exists() else None)
        record()
        raise
'''


def prepare_dataset(existing_metadata, output_dir, *, config_path=ROOT / "configs/dataset_v1.json",
                    route_catalog_path, maximum_seconds=18000):
    """Read-only preparation; remote submission must freshly verify access."""
    from carla_tools.dataset_v1 import load_dataset_v1_config
    from scripts.run_dataset_pipeline import collection_matrix, dry_run
    metadata_path, output = Path(existing_metadata), Path(output_dir)
    metadata = read_strict_json(metadata_path)
    if (metadata.get("id") != NOTEBOOK_ID or metadata.get("title") != NOTEBOOK_TITLE
            or metadata.get("is_private") is not True):
        raise ValueError("dataset-existing-private-notebook-identity-required")
    if type(maximum_seconds) is not int or not 14400 <= maximum_seconds <= 18000:
        raise ValueError("dataset-bounded-runtime-invalid")
    config = load_dataset_v1_config(config_path)
    catalog = read_strict_json(route_catalog_path)
    routes = validate_frozen_catalog(catalog, config)
    plan = collection_matrix(routes, config)
    config_source = Path(config_path).read_text(encoding="utf-8")
    frozen_catalog_bytes = (json.dumps({"routes": routes}, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    # Base preparation supplies the established official-CARLA transport,
    # bounded server process group and pinned native dependencies. All changes
    # below affect this newly generated script, never the legacy bootstrap.
    bundle = prepare(metadata_path, output, workload="pilot")
    old_path = output / "week1.py"
    code = old_path.read_text(encoding="utf-8")
    code = _replace_once(code, 'maximum_seconds = 2400 if WORKLOAD == "pilot" else MAX_SECONDS',
                         f'maximum_seconds = {maximum_seconds} if WORKLOAD == "dataset" else MAX_SECONDS')
    # Quoted literals occur only in the bootstrap, not in its compressed
    # PROJECT_SOURCES transport. Keep the local variable name for its branch.
    code = code.replace('"pilot"', '"dataset"')
    code = _replace_once(code, "WORKLOAD = 'pilot'", "WORKLOAD = 'dataset'")
    code = _replace_once(code, '"run_pilot_pipeline.py" if pilot', '"run_dataset_pipeline.py" if pilot')
    code = _replace_once(code, 'report["pilot_runtime_sources"]', 'report["dataset_runtime_sources"]')
    code = _replace_once(code, '"carla_pilot" if pilot', '"carla_dataset" if pilot')
    code = _replace_once(code, '1800 if pilot else 600', f'{maximum_seconds - 1800} if pilot else 600')
    code = _replace_once(code,
        'command([python, str(project / "scripts" / script), "--output-dir", str(output / artifact)],',
        'command([python, str(project / "scripts" / script), "--output-dir", str(scratch / "dataset"),\n'
        '                     "--artifact-dir", str(output), "--frozen-route-catalog", str(frozen_routes),\n'
        '                     "--runtime-sources", str(output / "dataset_runtime_source_manifest.json")],')
    code = _replace_once(code,
        'gate = json.loads((output / artifact / (artifact + "_report.json")).read_text(encoding="utf-8"))',
        'gate = json.loads((output / (artifact + "_report.json")).read_text(encoding="utf-8"))')
    code = _replace_once(code,
        'project = scratch / "project"\n            project.mkdir()',
        'project = scratch / "project"\n            project.mkdir()\n'
        '            frozen_routes = scratch / "dataset_routes.json"\n'
        '            frozen_routes.write_bytes(zlib.decompress(base64.b64decode(FROZEN_DATASET_ROUTES, validate=True)))')
    helper = next(node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)
                  and node.name == "preserve_baseline_failure_artifacts")
    lines = code.splitlines(keepends=True)
    code = "".join(lines[:helper.lineno - 1]) + _failure_retention_source() + "\n" + "".join(lines[helper.end_lineno:])
    code = _replace_once(code,
        'preserve_baseline_failure_artifacts(output, artifact="dataset")',
        'report["dataset_artifact_retention"] = preserve_baseline_failure_artifacts(output, artifact="dataset", '
        'source_directory=scratch / "dataset", owned_root=scratch)')
    from carla_tools.dataset_v1 import SOURCE_MODULES
    required_sources = tuple(dict.fromkeys((*EXTRA_SOURCES, *SOURCE_MODULES)))
    sources = {name: (ROOT / name).read_text(encoding="utf-8") for name in required_sources}
    sources["configs/dataset_v1.json"] = config_source
    for name, source in sources.items():
        if name.endswith(".py"):
            compile(source, name, "exec")
    payload = base64.b64encode(zlib.compress(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode(), 9)).decode()
    marker = "raise SystemExit(main())\n"
    addition = ("\nPROJECT_SOURCES.update(json.loads(zlib.decompress(base64.b64decode(" + repr(payload) + ", validate=True))))\n"
                "FROZEN_DATASET_ROUTES = " + repr(base64.b64encode(zlib.compress(frozen_catalog_bytes, 9)).decode()) + "\n")
    code = _replace_once(code, marker, addition + marker)
    compile(code, "native.py", "exec")
    if len(code.encode()) >= 1_000_000:
        raise ValueError("dataset-kaggle-source-size-exceeded")
    (output / "native.py").write_text(code, encoding="utf-8", newline="\n")
    old_path.unlink()
    submission_path = output / "kernel-metadata.json"
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    submission["code_file"] = "native.py"
    submission_path.write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")
    source_hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
    bundle["source_sha256"].update(source_hashes)
    # Custom config is a reviewed input, not an unnoticed edit to the default.
    # Its actual bytes are retained next to the transport for local inspection.
    (output / "dataset_config.json").write_text(config_source, encoding="utf-8")
    (output / "dataset_route_catalog.json").write_bytes(frozen_catalog_bytes)
    checks = {"status": "PASS", "dry_run": dry_run(config_path), "frozen_route_count": len(routes),
              "matrix_episode_count": len(plan), "matrix_sha256": canonical_sha256(plan),
              "source_compiles": True, "source_contract": "embedded-utf8-sha256",
              "native_collection_started": False, "local_gpu_used": False,
              "reserved_final_state": "UNOPENED", "existing_private_identity_verified": True,
              "remote_access_requires_fresh_read_only_check": True}
    bundle.update(scope="frozen-central-recovery-dataset-collection", workload="dataset-v1-collection",
        maximum_seconds=maximum_seconds, code_sha256=hashlib.sha256(code.encode()).hexdigest(),
        config_sha256=config.sha256, config_file_sha256=hashlib.sha256(config_source.encode()).hexdigest(),
        frozen_route_catalog_sha256=hashlib.sha256(frozen_catalog_bytes).hexdigest(),
        expected_episodes=len(plan), matrix_sha256=canonical_sha256(plan), pre_submit_checks=checks,
        raw_output_location="scratch", artifact="dataset_artifacts.zip", remote_access_verified=False)
    checks.update(code_sha256=bundle["code_sha256"], config_sha256=config.sha256,
                  source_sha256=bundle["source_sha256"], config_file_sha256=bundle["config_file_sha256"],
                  frozen_route_catalog_sha256=bundle["frozen_route_catalog_sha256"])
    (output / "pre_submit_checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    (output / "bundle_manifest.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--route-catalog", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dataset_v1.json")
    parser.add_argument("--maximum-seconds", type=int, default=18000)
    args = parser.parse_args(argv)
    try:
        bundle = prepare_dataset(args.existing_metadata, args.output_dir, config_path=args.config,
            route_catalog_path=args.route_catalog, maximum_seconds=args.maximum_seconds)
        print(json.dumps({key: bundle[key] for key in ("notebook_id", "title", "workload", "expected_episodes",
            "maximum_seconds", "code_sha256", "submitted", "creates_new_notebook")}, indent=2))
        return 0
    except (OSError, ValueError, SyntaxError, StopIteration) as error:
        print(json.dumps({"status": "FAIL", "code": "dataset-package-preparation-failed",
                          "exception_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
