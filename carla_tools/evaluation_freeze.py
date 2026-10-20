"""Write-once bindings for a final evaluation planned before observation.

This module hashes local inputs and validates contracts on CPU. It neither
starts a simulator nor opens images, runs inference, or submits remote work.
A freeze ID detects changes relative to a separately retained expected ID;
an editable self-hash alone does not establish external authenticity.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile

from .contracts import canonical_sha256, strict_json_loads

VERSION = "vlf-final-evaluation-freeze-v1"
REQUIRED_ARTIFACT_ROLES = frozenset({"dataset", "split", "selection", "topology", "budget"})
SOURCE_DIRECTORIES = ("carla_tools", "scripts", "tests", "configs", "config", "schemas", ".github")
MAX_JSON_BYTES = 32 * 1024 * 1024
_PRIVATE_DOCS = {"ROADMAP.md", "progress.md", "KAGGLE_CARLA_VERIFICATION.md"}
_INPUT_NAMES = {"manifest", "routes", "selection", "development_review", "checkpoint"}
_KEYS = {"schema_version", "freeze_version", "status", "freeze_id", "created_utc", "inputs",
    "selected_pipeline_id", "required_source_paths", "source_paths", "artifact_paths", "files",
    "source_tree_sha256", "artifact_tree_sha256", "informational"}


class EvaluationFreezeError(ValueError):
    """A stable freeze contract failure without sensitive file contents."""


def _require(value, code):
    if not value:
        raise EvaluationFreezeError(code)


def _sha(value):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        "evaluation-freeze-invalid-sha256")
    return value


def _portable(value):
    _require(type(value) is str and value and "\\" not in value and ":" not in value
        and not PurePosixPath(value).is_absolute()
        and all(part not in ("", ".", "..") for part in value.split("/"))
        and all(not part.endswith((".", " ")) for part in value.split("/"))
        and not any(ord(c) < 32 or c in '<>"|?*' for c in value),
        "evaluation-freeze-unsafe-relative-path")
    # Windows device filenames are unsafe even on a non-Windows verifier.
    _require(all(not re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])", part.split(".")[0])
        for part in value.split("/")), "evaluation-freeze-unsafe-relative-path")
    return value


def _no_links(path):
    for part in (path, *path.parents):
        _require(not part.is_symlink() and not getattr(part, "is_junction", lambda: False)(),
            "evaluation-freeze-link-forbidden")


def _root(value):
    try:
        path = Path(value).absolute()
        _no_links(path)
        resolved = path.resolve(strict=True)
        _no_links(resolved)
        _require(resolved.is_dir(), "evaluation-freeze-root-not-directory")
        return resolved
    except (OSError, TypeError, RuntimeError):
        raise EvaluationFreezeError("evaluation-freeze-root-not-directory") from None


def _relative(root, value):
    """Accept a portable project-relative name or an in-project absolute Path."""
    try:
        path = Path(value)
        if path.is_absolute():
            _require(".." not in path.parts, "evaluation-freeze-unsafe-relative-path")
            _no_links(path)
            relative = path.relative_to(root).as_posix()
        else:
            relative = value if type(value) is str else path.as_posix()
        return _portable(relative)
    except EvaluationFreezeError:
        raise
    except (ValueError, TypeError):
        raise EvaluationFreezeError("evaluation-freeze-unsafe-relative-path") from None


def _file(root, relative):
    name = _portable(relative)
    path = root.joinpath(*PurePosixPath(name).parts)
    _no_links(path)
    try:
        resolved = path.resolve(strict=True)
        _require(resolved.is_relative_to(root) and resolved.is_file(), "evaluation-freeze-input-not-file")
        _no_links(resolved)
        return resolved
    except (OSError, RuntimeError):
        raise EvaluationFreezeError("evaluation-freeze-input-not-file") from None


def _signature(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _read(root, relative, *, json_input=False):
    """Hash a stable regular file; JSON parsing consumes these exact bytes."""
    path = _file(root, relative)
    try:
        before = path.stat()
        _require(stat.S_ISREG(before.st_mode), "evaluation-freeze-input-not-file")
        if json_input:
            _require(0 < before.st_size <= MAX_JSON_BYTES, "evaluation-freeze-json-size-invalid")
        digest, chunks, size = hashlib.sha256(), [], 0
        with path.open("rb") as stream:
            _require(_signature(os.fstat(stream.fileno())) == _signature(before),
                "evaluation-freeze-input-changed-during-read")
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
                if json_input:
                    chunks.append(block)
            _require(_signature(os.fstat(stream.fileno())) == _signature(before),
                "evaluation-freeze-input-changed-during-read")
        _require(_file(root, relative) == path and _signature(path.stat()) == _signature(before)
            and size == before.st_size, "evaluation-freeze-input-changed-during-read")
    except OSError:
        raise EvaluationFreezeError("evaluation-freeze-input-read-failed") from None
    binding = {"path": relative, "bytes": size, "sha256": digest.hexdigest()}
    if not json_input:
        return binding
    try:
        return binding, strict_json_loads(b"".join(chunks).decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError):
        raise EvaluationFreezeError("evaluation-freeze-invalid-json") from None


def _unique_paths(values):
    _require(type(values) is list and all(type(value) is str for value in values),
        "evaluation-freeze-path-list-invalid")
    result = [_portable(value) for value in values]
    _require(len({p.casefold() for p in result}) == len(result), "evaluation-freeze-duplicate-path")
    return sorted(result)


def public_source_paths(project_root):
    """Enumerate public code/config/docs, including untracked future additions.

    Private execution records and generated Python caches are excluded. Files
    added to a source directory after freezing cause verification to fail.
    """
    root, names = _root(project_root), []
    directories = [root / name for name in SOURCE_DIRECTORIES] + [root / "docs"]
    for directory in directories:
        if not directory.exists():
            continue
        _no_links(directory)
        _require(directory.is_dir(), "evaluation-freeze-source-directory-invalid")
        for parent, subdirs, files in os.walk(directory, followlinks=False):
            parent = Path(parent)
            _no_links(parent)
            for name in subdirs:
                _no_links(parent / name)
            subdirs[:] = [name for name in subdirs if name not in
                {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
                and not (directory.name == "docs" and parent == directory and name in {"local", "evidence"})]
            for name in files:
                path = parent / name
                if path.suffix in {".pyc", ".pyo"}:
                    continue
                if directory.name == "docs" and parent == directory and (name in _PRIVATE_DOCS
                        or re.search(r"_D[0-9].*\.md$", name) or re.fullmatch(r"controller_d[0-9].*\.md", name)):
                    continue
                _file(root, path.relative_to(root).as_posix())
                names.append(path.relative_to(root).as_posix())
    for name in ("README.md", ".gitignore", ".gitattributes", ".python-version"):
        if (root / name).exists():
            names.append(name)
    names.extend(path.relative_to(root).as_posix() for path in root.glob("requirements*.txt"))
    return _unique_paths(names)


def _manifest_artifacts(manifest):
    bindings = manifest.get("artifact_bindings") if type(manifest) is dict else None
    _require(type(bindings) is list and bindings, "evaluation-freeze-artifact-bindings-required")
    result, roles = {}, set()
    for binding in bindings:
        _require(type(binding) is dict and set(binding) == {"path", "sha256", "roles"},
            "evaluation-freeze-artifact-binding-invalid")
        path = _portable(binding["path"])
        _require(path.casefold() not in {name.casefold() for name in result}, "evaluation-freeze-duplicate-path")
        _sha(binding["sha256"])
        items = binding["roles"]
        _require(type(items) is list and items and all(type(item) is str and item in REQUIRED_ARTIFACT_ROLES
            for item in items) and len(items) == len(set(items)), "evaluation-freeze-artifact-roles-invalid")
        result[path] = binding
        roles.update(items)
    _require(roles == REQUIRED_ARTIFACT_ROLES, "evaluation-freeze-required-artifact-role-missing")
    return result


def _contracts(manifest, routes, selection, review):
    from .final_evaluation import validate_manifest
    from .selected_pipeline import _validate_selection
    validate_manifest(manifest, routes, selection)
    _validate_selection(selection)
    _require(type(review) is dict and review.get("status") == "PASS", "evaluation-freeze-development-review-not-PASS")
    return _manifest_artifacts(manifest)


def _execution_sources(root, selection, sources, recorded):
    """Bind the runtime and validators actually used by this interpreter."""
    expected = dict(selection["source_sha256"])
    implementation_root = Path(__file__).resolve().parents[1]
    for name in ("carla_tools/evaluation_freeze.py", "carla_tools/final_evaluation.py", "carla_tools/contracts.py"):
        expected[name] = _read(implementation_root, name)["sha256"]
    _require(set(expected) <= set(sources), "evaluation-freeze-execution-source-missing")
    _require(all(recorded[path]["sha256"] == checksum for path, checksum in expected.items()),
        "evaluation-freeze-execution-source-mismatch")


def _roles(inputs, sources, artifacts, manifest_artifacts):
    roles = {}
    for name, path in inputs.items():
        roles.setdefault(path, set()).add("input:" + name)
    for path in sources:
        roles.setdefault(path, set()).add("source")
    for path in artifacts:
        roles.setdefault(path, set()).add("artifact")
    for path, binding in manifest_artifacts.items():
        roles.setdefault(path, set()).update("manifest:" + role for role in binding["roles"])
    return {path: sorted(value) for path, value in roles.items()}


def freeze_id(value):
    """Fingerprint every field except this ID itself, including metadata."""
    return canonical_sha256({key: item for key, item in value.items() if key != "freeze_id"})


def _tree(files, paths):
    by_path = {item["path"]: item for item in files}
    return canonical_sha256([{key: by_path[path][key] for key in ("path", "bytes", "sha256")} for path in paths])


def _git_commit(root):
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
        value = result.stdout.strip()
        return value if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40,64}", value) else None
    except OSError:
        return None


def build_freeze(project_root, manifest_path, routes_path, selection_path,
        artifact_paths, source_paths=None, *, development_review_path):
    """Validate and bind the complete current plan without observing test data.

    Supplied source paths may add explicit dependencies but may not omit any
    automatically discovered public source. Artifact paths must cover every
    declared dataset/split/selection/topology/budget evidence binding.
    """
    root = _root(project_root)
    inputs = {name: _relative(root, value) for name, value in {
        "manifest": manifest_path, "routes": routes_path, "selection": selection_path,
        "development_review": development_review_path}.items()}
    original, values = {}, {}
    for name, path in inputs.items():
        original[path], values[name] = _read(root, path, json_input=True)
    manifest_artifacts = _contracts(values["manifest"], values["routes"], values["selection"], values["development_review"])
    checkpoint = values["selection"]["model"]["checkpoint_path"]
    _portable(checkpoint)
    inputs["checkpoint"] = _relative(root, (root / inputs["selection"]).parent.joinpath(*PurePosixPath(checkpoint).parts))
    mandatory = public_source_paths(root)
    _require(source_paths is None or type(source_paths) is list, "evaluation-freeze-path-list-invalid")
    sources = mandatory if source_paths is None else _unique_paths([_relative(root, value) for value in source_paths])
    _require(set(mandatory) <= set(sources), "evaluation-freeze-required-source-missing")
    _require(type(artifact_paths) is list, "evaluation-freeze-path-list-invalid")
    artifacts = _unique_paths([_relative(root, value) for value in artifact_paths])
    _require(set(manifest_artifacts) <= set(artifacts), "evaluation-freeze-required-artifact-missing")
    roles = _roles(inputs, sources, artifacts, manifest_artifacts)
    files = [{**_read(root, path), "roles": roles[path]} for path in sorted(roles)]
    by_path = {item["path"]: item for item in files}
    _execution_sources(root, values["selection"], sources, by_path)
    _require(all({key: by_path[path][key] for key in binding} == binding for path, binding in original.items()),
        "evaluation-freeze-input-changed-during-build")
    _require(by_path[inputs["checkpoint"]]["sha256"] == values["selection"]["model"]["checkpoint_sha256"],
        "evaluation-freeze-selected-checkpoint-mismatch")
    _require(all(by_path[path]["sha256"] == binding["sha256"] for path, binding in manifest_artifacts.items()),
        "evaluation-freeze-declared-artifact-hash-mismatch")
    value = {"schema_version": 1, "freeze_version": VERSION, "status": "FROZEN",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "inputs": inputs,
        "selected_pipeline_id": values["selection"]["pipeline_id"], "required_source_paths": mandatory,
        "source_paths": sources, "artifact_paths": artifacts, "files": files,
        "source_tree_sha256": _tree(files, sources), "artifact_tree_sha256": _tree(files, artifacts),
        "informational": {"git_commit": _git_commit(root), "external_authenticity_requires_expected_freeze_id": True}}
    value["freeze_id"] = freeze_id(value)
    # A second complete read closes changes during contract validation/hashing.
    _verify(value, root)
    return value


def _verify(value, root):
    _require(type(value) is dict and set(value) == _KEYS, "evaluation-freeze-schema-invalid")
    _require(type(value["schema_version"]) is int and value["schema_version"] == 1
        and value["freeze_version"] == VERSION and value["status"] == "FROZEN", "evaluation-freeze-version-invalid")
    _require(_sha(value["freeze_id"]) == freeze_id(value), "evaluation-freeze-identity-mismatch")
    _require(type(value["created_utc"]) is str and bool(value["created_utc"]), "evaluation-freeze-created-time-invalid")
    _require(type(value["informational"]) is dict and set(value["informational"]) ==
        {"git_commit", "external_authenticity_requires_expected_freeze_id"}
        and value["informational"]["external_authenticity_requires_expected_freeze_id"] is True
        and (value["informational"]["git_commit"] is None or type(value["informational"]["git_commit"]) is str
            and re.fullmatch(r"[0-9a-f]{40,64}", value["informational"]["git_commit"])),
        "evaluation-freeze-informational-fields-invalid")
    inputs = value["inputs"]
    _require(type(inputs) is dict and set(inputs) == _INPUT_NAMES, "evaluation-freeze-core-input-missing")
    for path in inputs.values():
        _portable(path)
    mandatory, sources, artifacts = [value[key] for key in ("required_source_paths", "source_paths", "artifact_paths")]
    for items in (mandatory, sources, artifacts):
        _require(items == _unique_paths(items), "evaluation-freeze-path-list-not-canonical")
    _require(public_source_paths(root) == mandatory and set(mandatory) <= set(sources),
        "evaluation-freeze-public-source-inventory-mismatch")
    files = value["files"]
    _require(type(files) is list and files, "evaluation-freeze-files-required")
    recorded = {}
    for binding in files:
        _require(type(binding) is dict and set(binding) == {"path", "bytes", "sha256", "roles"},
            "evaluation-freeze-file-binding-invalid")
        path = _portable(binding["path"])
        _require(path.casefold() not in {p.casefold() for p in recorded}, "evaluation-freeze-duplicate-path")
        _sha(binding["sha256"])
        _require(type(binding["bytes"]) is int and binding["bytes"] >= 0,
            "evaluation-freeze-file-size-invalid")
        recorded[path] = binding
    _require([item["path"] for item in files] == sorted(recorded), "evaluation-freeze-files-not-canonical")
    original, parsed = {}, {}
    for name in _INPUT_NAMES - {"checkpoint"}:
        binding, parsed[name] = _read(root, inputs[name], json_input=True)
        original[inputs[name]] = binding
    manifest_artifacts = _contracts(parsed["manifest"], parsed["routes"], parsed["selection"], parsed["development_review"])
    selected = parsed["selection"]
    _require(value["selected_pipeline_id"] == _sha(selected["pipeline_id"]), "evaluation-freeze-selected-pipeline-mismatch")
    relative_checkpoint = _portable(selected["model"]["checkpoint_path"])
    actual_checkpoint = _relative(root, (root / inputs["selection"]).parent.joinpath(*PurePosixPath(relative_checkpoint).parts))
    _require(inputs["checkpoint"] == actual_checkpoint, "evaluation-freeze-selected-checkpoint-path-mismatch")
    _require(set(manifest_artifacts) <= set(artifacts), "evaluation-freeze-required-artifact-missing")
    roles = _roles(inputs, sources, artifacts, manifest_artifacts)
    _require(set(recorded) == set(roles), "evaluation-freeze-file-closure-mismatch")
    _require(all(recorded[path]["roles"] == role for path, role in roles.items()),
        "evaluation-freeze-file-roles-mismatch")
    _execution_sources(root, selected, sources, recorded)
    # Read every file, not just user-supplied or manifest-listed subsets.
    for path, binding in recorded.items():
        observed = _read(root, path)
        _require(all(binding[key] == observed[key] for key in observed), "evaluation-freeze-bound-file-mismatch")
        if path in original:
            _require(observed == original[path], "evaluation-freeze-input-changed-during-verification")
    _require(recorded[actual_checkpoint]["sha256"] == selected["model"]["checkpoint_sha256"],
        "evaluation-freeze-selected-checkpoint-mismatch")
    _require(all(recorded[path]["sha256"] == binding["sha256"] for path, binding in manifest_artifacts.items()),
        "evaluation-freeze-declared-artifact-hash-mismatch")
    _require(value["source_tree_sha256"] == _tree(files, sources)
        and value["artifact_tree_sha256"] == _tree(files, artifacts), "evaluation-freeze-tree-hash-mismatch")
    for path, binding in recorded.items():
        observed = _read(root, path)
        _require(all(binding[key] == observed[key] for key in observed),
            "evaluation-freeze-binding-changed-during-verification")
    _require(public_source_paths(root) == mandatory, "evaluation-freeze-source-inventory-changed-during-verification")
    return {"status": "PASS", "freeze_id": value["freeze_id"], "selected_pipeline_id": value["selected_pipeline_id"],
        "file_count": len(files), "source_count": len(sources), "artifact_count": len(artifacts),
        "verified_inputs": {name: {key: recorded[path][key] for key in ("path", "bytes", "sha256")}
            for name, path in inputs.items()},
        "GPU_used": False, "heldout_observation_executed": False}


def verify_freeze(freeze_path, project_root, *, expected_freeze_id=None):
    """Verify all current bindings against a retained freeze, without inference."""
    root = _root(project_root)
    relative = _relative(root, freeze_path)
    binding, value = _read(root, relative, json_input=True)
    if expected_freeze_id is not None:
        _require(type(value) is dict and value.get("freeze_id") == _sha(expected_freeze_id),
            "evaluation-freeze-unexpected-identity")
    report = _verify(value, root)
    _require(_read(root, relative) == binding, "evaluation-freeze-record-changed-during-verification")
    report["freeze_sha256"] = binding["sha256"]
    report["expected_identity_checked"] = expected_freeze_id is not None
    return report


def write_freeze(output, value):
    """Atomically publish one complete freeze; never replace an existing path."""
    _require(type(value) is dict and set(value) == _KEYS and value.get("status") == "FROZEN"
        and value.get("freeze_id") == freeze_id(value),
        "evaluation-freeze-identity-mismatch")
    try:
        target = Path(output).absolute()
        _no_links(target)
        _no_links(target.parent.resolve(strict=True))
        _require(target.parent.is_dir(), "evaluation-freeze-output-parent-invalid")
        _require(not target.exists(), "evaluation-freeze-output-already-exists")
        payload = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    except (OSError, TypeError):
        raise EvaluationFreezeError("evaluation-freeze-output-invalid") from None
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".evaluation-freeze-", suffix=".pending", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _no_links(target.parent)
        if os.name == "nt":
            # Windows rename is atomic and fails when the destination exists;
            # unlike replace, it cannot overwrite an artifact created in a race.
            os.rename(temporary, target)
        else:
            os.link(temporary, target)
    except FileExistsError:
        raise EvaluationFreezeError("evaluation-freeze-output-already-exists") from None
    except OSError:
        raise EvaluationFreezeError("evaluation-freeze-publication-failed") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"status": "FROZEN", "freeze_id": value["freeze_id"], "freeze_sha256": hashlib.sha256(payload).hexdigest()}
