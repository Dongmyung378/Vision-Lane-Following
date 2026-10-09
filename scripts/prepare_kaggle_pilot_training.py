"""Package collection and training consecutively in the existing CARLA notebook."""

from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_kaggle_week1 import prepare as prepare_pilot
from scripts.prepare_kaggle_training import SOURCES as TRAINING_SOURCES


def run_training_after_pilot(carla_return_code, output_root, bootstrap_source, training_sources):
    """Verify the finished producer before starting a separate training runtime."""
    import hashlib
    import json
    import re
    import zipfile
    from pathlib import Path

    output_root = Path(output_root)
    report = {"schema_version": 1, "status": "FAIL", "stage": "verify-pilot-handoff",
              "native_max_seconds": 2400, "training_max_seconds": 5400,
              "pilot_transport": "same-session-verified-archive"}

    def digest(path):
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
        return value.hexdigest()

    try:
        if type(carla_return_code) is not int or carla_return_code != 0:
            raise RuntimeError("native-producer-failed")
        native = json.loads((output_root / "kaggle_carla_bootstrap.json").read_text(encoding="utf-8"))
        gate = native.get("pilot", {}).get("gate", {})
        if (native.get("status") != "PASS" or native.get("pilot_verified") is not True or
                native.get("server_stopped") is not True or gate.get("status") != "PASS"):
            raise RuntimeError("native-producer-not-verified-or-server-still-running")
        artifact = json.loads((output_root / "pilot_archive_report.json").read_text(encoding="utf-8"))
        if artifact.get("status") != "PASS":
            raise RuntimeError("native-pilot-archive-report-failed")
        archive = output_root / "pilot_artifacts.zip"
        manifest = output_root / "pilot" / "pilot_manifest.json"
        for path in (archive, manifest):
            if (path.is_symlink() or not path.is_file() or
                    not path.resolve().is_relative_to(output_root.resolve())):
                raise RuntimeError("native-pilot-handoff-path-invalid")
        archive_sha256, manifest_sha256 = digest(archive), digest(manifest)
        if (not re.fullmatch("[0-9a-f]{64}", str(artifact.get("archive_sha256", ""))) or
                artifact["archive_sha256"] != archive_sha256 or
                artifact.get("archive_bytes") != archive.stat().st_size or
                artifact.get("manifest_sha256") != manifest_sha256 or
                gate.get("manifest_sha256") != manifest_sha256):
            raise RuntimeError("native-pilot-handoff-sha256-mismatch")
        with zipfile.ZipFile(archive) as packed:
            matches = [entry for entry in packed.infolist() if entry.filename == "pilot/pilot_manifest.json"]
            if len(matches) != 1 or hashlib.sha256(packed.read(matches[0])).hexdigest() != manifest_sha256:
                raise RuntimeError("native-pilot-archived-manifest-mismatch")
        report.update(pilot_archive_sha256=archive_sha256, pilot_manifest_sha256=manifest_sha256,
                      native_server_stopped=True, native_pilot_verified=True,
                      training_source_sha256={name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                                              for name, text in sorted(training_sources.items())})
        report["stage"] = "training"
        namespace = {"__name__": "vlf_same_session_training",
                     "__file__": str(output_root / "same_session_training_bootstrap.py")}
        exec(compile(bootstrap_source, namespace["__file__"], "exec"), namespace)
        namespace.update(PROJECT_SOURCES=training_sources, PILOT_ARCHIVE_SHA256=archive_sha256,
                         PILOT_MANIFEST_SHA256=manifest_sha256, PILOT_DOWNLOAD_VERSION=None,
                         MAX_SECONDS=5400)
        code = namespace["main"](local_archive=archive)
        if type(code) is not int or code != 0:
            raise RuntimeError("same-session-training-failed")
        learning = json.loads((output_root / "kaggle_training_bootstrap.json").read_text(encoding="utf-8"))
        if (learning.get("status") != "PASS" or learning.get("training_process_stopped") is not True or
                learning.get("pilot_archive_sha256") != archive_sha256 or
                learning.get("pilot_manifest_sha256") != manifest_sha256):
            raise RuntimeError("same-session-training-report-not-verified")
        report.update(status="PASS", stage="complete", training_process_stopped=True)
        return 0
    except Exception as error:
        # Report only owned guard codes; dependency/backend exceptions may carry
        # signed download URLs or other platform authentication details.
        report["exception_type"] = type(error).__name__
        allowed_codes = {"native-producer-failed", "native-producer-not-verified-or-server-still-running",
                         "native-pilot-archive-report-failed", "native-pilot-handoff-path-invalid",
                         "native-pilot-handoff-sha256-mismatch", "native-pilot-archived-manifest-mismatch",
                         "same-session-training-failed", "same-session-training-report-not-verified"}
        report["code"] = str(error) if str(error) in allowed_codes else "combined-pipeline-failed"
        return 1
    finally:
        (output_root / "combined_pipeline_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: report[key] for key in ("status", "stage")}), flush=True)


def prepare(existing_metadata: Path, output_dir: Path) -> dict:
    """Keep the native package intact and append independently packed training."""
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("output-directory-already-exists")
    if any(parent.is_symlink() for parent in output_dir.parents):
        raise ValueError("output-parent-is-symlink")
    sources, hashes = {}, {}
    for name in TRAINING_SOURCES:
        source = (ROOT / name).read_text(encoding="utf-8")
        if name.endswith(".py"):
            compile(source, name, "exec")
        elif name.endswith(".json"):
            json.loads(source)
        sources[name] = source
        hashes[name] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    bootstrap = (ROOT / "scripts/kaggle_training_bootstrap.py").read_text(encoding="utf-8")
    compile(bootstrap, "same_session_training_bootstrap.py", "exec")
    payload = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode("utf-8")
    training_tail = (
        "\n_carla_return_code = main()\n"
        "TRAINING_SOURCES_PACKED = " + repr(base64.b64encode(zlib.compress(payload, 9)).decode("ascii")) +
        "\nTRAINING_PAYLOAD_SHA256 = " + repr(hashlib.sha256(payload).hexdigest()) +
        "\n_training_payload = zlib.decompress(base64.b64decode(TRAINING_SOURCES_PACKED, validate=True))"
        "\nif hashlib.sha256(_training_payload).hexdigest() != TRAINING_PAYLOAD_SHA256:"
        "\n    raise SystemExit('embedded-training-transport-hash-mismatch')"
        "\nTRAINING_SOURCES = json.loads(_training_payload.decode('utf-8'))"
        "\nTRAINING_BOOTSTRAP_SOURCE = " + repr(bootstrap) + "\n\n" +
        inspect.getsource(run_training_after_pilot) +
        "\nraise SystemExit(run_training_after_pilot(_carla_return_code, Path('/kaggle/working'),"
        " TRAINING_BOOTSTRAP_SOURCE, TRAINING_SOURCES))\n"
    )
    bundle = prepare_pilot(existing_metadata, output_dir, workload="pilot")
    code_file = output_dir / "week1.py"
    native_code = code_file.read_text(encoding="utf-8")
    marker = "\nraise SystemExit(main())\n"
    if not native_code.endswith(marker) or native_code.count(marker) != 1:
        raise ValueError("native-bootstrap-terminal-marker-mismatch")
    code = native_code[:-len(marker)] + training_tail
    compile(code, "week1.py", "exec")
    if len(code.encode("utf-8")) >= 1_000_000:
        raise ValueError("kaggle-script-source-size-exceeded")
    bundle.update(workload="moving-rgb-waypoint-pilot-and-training",
                  code_sha256=hashlib.sha256(code.encode("utf-8")).hexdigest(),
                  training_source_sha256=hashes, pilot_transport="same-session-verified-archive",
                  pilot_hash_origin="verified-same-session-pilot-report",
                  native_max_seconds=2400, training_max_seconds=5400)
    code_file.write_text(code, encoding="utf-8", newline="\n")
    (output_dir / "bundle_manifest.json").write_text(json.dumps(bundle, indent=2) + "\n",
                                                     encoding="utf-8", newline="\n")
    return bundle


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = prepare(args.existing_metadata, args.output_dir)
        print(json.dumps({key: report[key] for key in ("notebook_id", "workload", "code_sha256",
                                                      "submitted", "creates_new_notebook")}))
        return 0
    except (OSError, ValueError, SyntaxError, KeyError, TypeError) as error:
        print(json.dumps({"status": "FAIL", "exception_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
