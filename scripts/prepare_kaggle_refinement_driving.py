"""Reuse the existing CARLA notebook and existing training outputs for comparison."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import lzma
from pathlib import Path
import re
import zlib

from scripts.kaggle_learned_bootstrap import render_bootstrap
from scripts.prepare_kaggle_week1 import SOURCES as COMMON, DRIVING_SOURCES, BASELINE_SOURCES, NOTEBOOK_ID, NOTEBOOK_TITLE
from scripts.prepare_kaggle_spatial import SOURCES as SPATIAL_SOURCES
from scripts.prepare_kaggle_refinement import EXTRA_SOURCES

ROOT = Path(__file__).resolve().parents[1]
TRAINING_NOTEBOOK = 'dongmyungpark/vlf-carla-vulkan-preflight-20261005'
_UNUSED = frozenset((
    'configs/routes_d6_geometry.json', 'carla_tools/week1.py',
    'scripts/run_geometry_collection.py', 'scripts/run_week1_pipeline.py',
    'scripts/run_driving_pipeline.py', 'scripts/verify_driving.py',
    'scripts/run_baseline_pipeline.py', 'scripts/verify_baseline.py',
))
# The comparison uses the separately frozen DEV routes. Keep the model source
# closure intact while excluding unrelated pipeline entry points and catalog.
SOURCES = tuple(name for name in dict.fromkeys((*COMMON, *DRIVING_SOURCES, *BASELINE_SOURCES, *SPATIAL_SOURCES, *EXTRA_SOURCES,
    'carla_tools/inference.py', 'carla_tools/rgb_predictor.py', 'carla_tools/spatial_inference.py',
    'carla_tools/refinement_driving.py', 'carla_tools/refinement_validation.py', 'carla_tools/driving_media.py', 'scripts/run_refinement_pipeline.py'))
    if name not in _UNUSED)


def unpack_reference(archive, destination, expected_archive_sha256, expected_checkpoint_sha256, variant='resnet'):
    """Extract only the verified reference pair; no generic archive extraction."""
    import zipfile
    archive, destination = Path(archive), Path(destination)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected_archive_sha256:
        raise ValueError('native-archived-reference-archive-hash-mismatch')
    with zipfile.ZipFile(archive) as stream:
        names = stream.namelist()
        if len(names) != len(set(names)):
            raise ValueError('native-archived-reference-duplicate-members')
        if variant not in ('resnet', 'hybrid_mirror', 'hybrid_geometry'):
            raise ValueError('native-archived-model-variant-invalid')
        selected = {name: f'experiment/{variant}/{name}' for name in ('checkpoint.pt', 'training_report.json')}
        payload = {}
        for name, member in selected.items():
            item = stream.getinfo(member)
            if item.file_size > 128 * 1024**2 or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('native-archived-reference-member-invalid')
            payload[name] = stream.read(member)
    if hashlib.sha256(payload['checkpoint.pt']).hexdigest() != expected_checkpoint_sha256:
        raise ValueError('native-archived-reference-checkpoint-mismatch')
    report = json.loads(payload['training_report.json'])
    if report.get('status') != 'PASS' or report.get('variant') != variant or report.get('checkpoint_sha256') != expected_checkpoint_sha256:
        raise ValueError('native-archived-reference-report-mismatch')
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in payload.items():
        (destination / name).write_bytes(data)
    return destination / 'checkpoint.pt', destination / 'training_report.json'


def render_native_bootstrap(base_source, *, archived_candidate=False):
    source = render_bootstrap(base_source, maximum_seconds=3600)
    marker = '        stage("server-download")\n'
    if source.count(marker) != 1:
        raise ValueError('native-bootstrap-reference-marker-mismatch')
    setup = '''        stage("archived-resnet-reference")
        command([uv, "pip", "install", "--python", python, "--only-binary", ":all:",
                 "kagglehub==1.0.2"], "carla_native_transport_dependencies", 180, env)
        reference_helper = ("import contextlib,hashlib,io,json,logging\\nfrom pathlib import Path\\n"
            "logging.disable(logging.CRITICAL)\\ntry:\\n"
            " with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):\\n"
            "  from kagglehub.http_resolver import NotebookOutputHttpResolver\\n"
            "  from kagglehub.handle import parse_notebook_handle\\n"
            "  value,version=NotebookOutputHttpResolver()._resolve(parse_notebook_handle(" + repr(REFERENCE_HANDLE) + "),path='training_artifacts.zip',output_dir=" + repr(str(scratch / "reference_download")) + ")\\n"
            "  assert version==" + repr(REFERENCE_VERSION) + "\\n"
            " print(json.dumps({'status':'PASS','path':str(value)}))\\n"
            "except BaseException as error:\\n print(json.dumps({'status':'FAIL','exception_type':type(error).__name__}));raise SystemExit(1)\\n")
        downloaded = command([python, "-c", reference_helper], "carla_native_reference_transport", 240, env, True)
        reference_value = json.loads(downloaded.stdout.strip().splitlines()[-1])
        reference_checkpoint, reference_report = unpack_reference(reference_value["path"], scratch / "reference",
                                                                  REFERENCE_ARCHIVE_SHA256, REFERENCE_CHECKPOINT_SHA256)
        stage("server-download")
'''
    source = source.replace(marker, setup, 1)
    if archived_candidate:
        mounted = '        selected_checkpoint = resolve_selected_checkpoint(Path("/kaggle/input"), CHECKPOINT_SHA256, CHECKPOINT_BYTES)\n'
        if source.count(mounted) != 1:
            raise ValueError('native-bootstrap-mounted-model-marker-mismatch')
        source = source.replace(mounted, '        selected_checkpoint = None\n', 1)
        candidate_setup = '''        stage("archived-refined-candidate")
        candidate_helper = reference_helper.replace(repr(REFERENCE_HANDLE), repr(CANDIDATE_HANDLE)).replace(
            "assert version==" + repr(REFERENCE_VERSION), "assert version==" + repr(CANDIDATE_VERSION)).replace(
            repr(str(scratch / "reference_download")), repr(str(scratch / "candidate_download")))
        downloaded_candidate = command([python, "-c", candidate_helper], "carla_native_candidate_transport", 240, env, True)
        candidate_value = json.loads(downloaded_candidate.stdout.strip().splitlines()[-1])
        selected_checkpoint, candidate_report = unpack_reference(candidate_value["path"], scratch / "candidate",
            CANDIDATE_ARCHIVE_SHA256, CHECKPOINT_SHA256, CANDIDATE_VARIANT)
        if selected_checkpoint.stat().st_size != CHECKPOINT_BYTES:
            raise BootstrapFailure("native-archived-candidate-size-mismatch")
        stage("server-download")
'''
        source = source.replace(marker, candidate_setup, 1)
    start_marker = '        if WORKLOAD == "learned":\n            stage("selected-rgb-model-native-driving-and-replay")'
    start = source.find(start_marker)
    end = source.find('        else:\n            stage("rgb-probe")', start)
    if start < 0 or end < 0:
        raise ValueError('native-bootstrap-pipeline-marker-mismatch')
    pipeline = '''        if WORKLOAD == "learned":
            stage("refinement-native-driving-and-replay")
            project = scratch / "project"
            project.mkdir()
            for relative, source in PROJECT_SOURCES.items():
                path = project / relative
                if not path.resolve().is_relative_to(project.resolve()):
                    raise BootstrapFailure("embedded-project-path-invalid")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.encode("utf-8"))
            report["learned_runtime_sources"] = export_baseline_runtime_sources(project, output, PROJECT_SOURCES, artifact="learned")
            routes_file = project / "frozen_dev_routes.json"
            routes_file.write_bytes(DEV_ROUTES_BYTES)
            command([python, str(project / "scripts/run_refinement_pipeline.py"), "--routes", str(routes_file),
                     "--reference", str(reference_checkpoint), "--reference-report", str(reference_report),
                     "--reference-sha256", REFERENCE_CHECKPOINT_SHA256, "--candidate", str(selected_checkpoint),
                     "--candidate-sha256", CHECKPOINT_SHA256, "--device", "cuda", "--output-dir", str(output / "native")],
                    "carla_learned", MAX_SECONDS - 600, env)
            gate = json.loads((output / "native/native_report.json").read_text())
            if (gate.get("status") != "PASS" or gate.get("episode_count") != 81 or gate.get("frame_count", 0) <= 0
                    or gate.get("final_town_used") is not False or not all(e.get("gate", {}).get("status") == "PASS" for e in gate["episodes"])
                    or [m.get("variant") for m in gate["models"]] != ["B0", "resnet", CANDIDATE_VARIANT]
                    or [m.get("checkpoint_sha256") for m in gate["models"]][1:] != [REFERENCE_CHECKPOINT_SHA256, CHECKPOINT_SHA256]
                    or gate["models"][0].get("device") != "cpu" or any(m.get("device") != "cuda" for m in gate["models"][1:])
                    or len(gate["planned_trials"]) != 81 or not all(e.get("media", {}).get("status") == "PASS" for e in gate["episodes"])
                    or any(value.get("unexecuted_episodes") or value.get("operation_or_integrity_failures")
                           for value in gate["attempt_summary"].values())):
                raise BootstrapFailure("native-refinement-integrity-proof-invalid")
            _assert_runtime_sources(project, PROJECT_SOURCES)
            if _file_sha256(selected_checkpoint) != CHECKPOINT_SHA256:
                raise BootstrapFailure("native-selected-checkpoint-changed")
            report["learned_verified"] = True
            report["runtime_sources_unchanged"] = True
            report["learned"] = {key: gate[key] for key in ("status", "episode_count", "frame_count", "aggregates")}
'''
    source = source[:start] + pipeline + source[end:]
    # Preserve native output after a timeout before the child can publish its ZIP.
    source = source.replace('preserve_baseline_failure_artifacts(output, artifact="learned")',
                            'preserve_baseline_failure_artifacts(output, artifact="native")')
    source = source.replace('artifact not in ("baseline", "pilot", "learned")',
                            'artifact not in ("baseline", "pilot", "learned", "native")')
    module = Path(__file__).read_text(encoding='utf-8')
    node = next(n for n in ast.parse(module).body if isinstance(n, ast.FunctionDef) and n.name == 'unpack_reference')
    source += '\n' + ast.get_source_segment(module, node) + '\n'
    compile(source, 'native.py', 'exec')
    return source


def prepare_native(existing_metadata, output_dir, *, candidate_sha256, candidate_bytes,
                   dev_routes, reference_archive_sha256, reference_checkpoint_sha256, reference_version=16,
                   candidate_archive_sha256=None, candidate_version=None, candidate_variant='hybrid_geometry'):
    metadata = json.loads(Path(existing_metadata).read_text())
    if metadata.get('id') != NOTEBOOK_ID or metadata.get('title') != NOTEBOOK_TITLE or metadata.get('is_private') is not True:
        raise ValueError('native-existing-private-notebook-required')
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise ValueError('native-new-local-bundle-required')
    for value in (candidate_sha256, reference_archive_sha256, reference_checkpoint_sha256):
        if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
            raise ValueError('native-bound-checkpoint-and-archive-hashes-required')
    if type(candidate_bytes) is not int or not 0 < candidate_bytes <= 128 * 1024**2 or type(reference_version) is not int or reference_version < 1:
        raise ValueError('native-checkpoint-size-or-version-invalid')
    archived_candidate = candidate_archive_sha256 is not None
    if archived_candidate and (not re.fullmatch('[0-9a-f]{64}', candidate_archive_sha256)
            or type(candidate_version) is not int or candidate_version < 1 or candidate_variant not in ('hybrid_mirror','hybrid_geometry')):
        raise ValueError('native-archived-candidate-binding-invalid')
    if not archived_candidate and candidate_version is not None:
        raise ValueError('native-archived-candidate-hash-required')
    routes = Path(dev_routes).read_bytes()
    frozen = json.loads(routes)
    if frozen.get('split') != 'dev' or frozen.get('final_town_used') is not False or len(frozen.get('routes', [])) != 3:
        raise ValueError('native-frozen-dev-routes-required')
    sources = {name: (ROOT / name).read_bytes().decode() for name in SOURCES}
    for name, value in sources.items():
        if name.endswith('.py'):
            compile(value, name, 'exec')
    payload = json.dumps(sources, sort_keys=True, separators=(',', ':')).encode()
    base = (ROOT / 'scripts/kaggle_carla_bootstrap.py').read_text(encoding='utf-8')
    bootstrap = render_native_bootstrap(base, archived_candidate=archived_candidate)
    vulkan = (ROOT / 'scripts/kaggle_vulkan_probe.py').read_text(encoding='utf-8')
    function = next(n for n in ast.parse(vulkan).body if isinstance(n, ast.FunctionDef) and n.name == 'vulkan_devices')
    child = 'import ctypes,ctypes.util,struct,json\n' + ast.get_source_segment(vulkan, function) + '\nprint(json.dumps(vulkan_devices()))\n'
    code = bootstrap + '\nPROBE_SOURCE=' + repr(sources['scripts/carla_rgb_probe.py']) + '\nVULKAN_CHILD_SOURCE=' + repr(child)
    for name, raw in (('_project_payload', payload), ('DEV_ROUTES_BYTES', routes)):
        code += '\nimport base64,lzma\n' + name + '=lzma.decompress(base64.b64decode(' + repr(base64.b64encode(lzma.compress(raw, preset=9)).decode()) + ',validate=True))\n'
        code += 'if hashlib.sha256(' + name + ').hexdigest()!=' + repr(hashlib.sha256(raw).hexdigest()) + ': raise SystemExit("native-transport-hash-mismatch")\n'
    code += 'PROJECT_SOURCES=json.loads(_project_payload.decode())\n'
    values = dict(CHECKPOINT_SHA256=candidate_sha256, CHECKPOINT_BYTES=candidate_bytes, LEARNED_MODE='refinement',
        CANDIDATE_VARIANT=candidate_variant,
        REFERENCE_HANDLE=TRAINING_NOTEBOOK + '/versions/' + str(reference_version), REFERENCE_VERSION=reference_version,
        REFERENCE_ARCHIVE_SHA256=reference_archive_sha256, REFERENCE_CHECKPOINT_SHA256=reference_checkpoint_sha256)
    if archived_candidate:
        values.update(CANDIDATE_ARCHIVE_SHA256=candidate_archive_sha256, CANDIDATE_VERSION=candidate_version,
                      CANDIDATE_VARIANT=candidate_variant, CANDIDATE_HANDLE=TRAINING_NOTEBOOK + '/versions/' + str(candidate_version))
    code += '\n'.join(name + '=' + repr(value) for name, value in values.items()) + '\nraise SystemExit(main())\n'
    code = code.replace('\r\n', '\n')
    compile(code, 'native.py', 'exec')
    if len(code.encode()) >= 1_000_000:
        raise ValueError('native-source-size-exceeded')
    submission = dict(id=NOTEBOOK_ID, title=NOTEBOOK_TITLE, code_file='native.py', language='python', kernel_type='script',
                      is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=True, machine_shape='NvidiaTeslaT4')
    for name in ('dataset_sources', 'kernel_sources', 'competition_sources', 'model_sources'):
        submission[name] = list(metadata.get(name, []))
    if archived_candidate:
        # Retrieve explicit saved versions instead of introducing a circular
        # latest-output dependency between the two existing notebooks.
        submission['kernel_sources'] = [name for name in submission['kernel_sources'] if name != TRAINING_NOTEBOOK]
    elif TRAINING_NOTEBOOK not in submission['kernel_sources']:
        submission['kernel_sources'].append(TRAINING_NOTEBOOK)
    bundle = dict(status='PREPARED', code_sha256=hashlib.sha256(code.encode()).hexdigest(),
        source_sha256={name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()},
        candidate_sha256=candidate_sha256, candidate_bytes=candidate_bytes, routes_sha256=hashlib.sha256(routes).hexdigest(),
        reference_archive_sha256=reference_archive_sha256, reference_checkpoint_sha256=reference_checkpoint_sha256,
        reference_version=reference_version, maximum_seconds=3600, creates_new_notebook=False, creates_dataset=False, local_gpu_used=False)
    bundle['candidate_transport'] = 'hash-pinned-saved-version-download' if archived_candidate else 'attached-existing-output'
    if archived_candidate:
        bundle.update(candidate_archive_sha256=candidate_archive_sha256, candidate_version=candidate_version)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'native.py').write_bytes(code.encode())
    for name, value in (('bundle_manifest.json', bundle), ('kernel-metadata.json', submission)):
        (output / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    return bundle
