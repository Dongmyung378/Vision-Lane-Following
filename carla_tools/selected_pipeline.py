"""Load one selected RGB model, correction and controller as a bound pipeline.

Model inference receives RGB only. The controller additionally receives speed
and explicitly supplied, measured vehicle geometry. This module never starts a
simulator, chooses a GPU automatically or changes a selection pointer.
"""
from __future__ import annotations

import copy
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

import numpy as np

from .contracts import canonical_sha256, strict_json_loads
from .controller import ControllerError, VehicleGeometry
from .controller_study import build_controller, study_controller_config_from_dict
from .inference import _input_file, _reject_links, _sha256_file

VERSION = "vlf-selected-rgb-control-v1"
INPUT_CONTRACT = "RGB8-640x360"
GEOMETRY_CONTRACT = "measured-actor-frame-metres-and-native-inner-front-wheel-steering"
TIMING_SCOPE = ("learned RGB preprocessing/device transfer, synchronized model forward/decode, "
    "bound RGB path correction or guarded fusion and selected controller; excludes BGRA conversion, sensor/simulator wait and disk")
MAX_SELECTION_BYTES = 1024 * 1024


class SelectedPipelineError(ValueError):
    """A stable source, model, input or selection contract failure."""


def _require(value, code):
    if not value:
        raise SelectedPipelineError(code)


def _sha(value):
    _require(type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None,
        "selected-pipeline-invalid-sha256")
    return value


def selected_pipeline_source_hashes():
    """Use an explicit dependency closure, independent of future added files."""
    from .runtime_driving import runtime_source_hashes
    root = Path(__file__).resolve().parents[1]
    names = ("carla_tools/controller_study.py", "carla_tools/selected_pipeline.py",
        "scripts/predict_selected_pipeline.py", "carla_tools/guarded_path_correction.py")
    return dict(runtime_source_hashes(), **{name: _sha256_file(root / name) for name in names})


def _directory(value):
    try:
        path = Path(value).absolute()
        _reject_links(path)
        path = path.resolve(strict=True)
        _reject_links(path)
        _require(path.is_dir(), "selected-pipeline-base-not-directory")
        return path
    except (OSError, TypeError) as error:
        raise SelectedPipelineError("selected-pipeline-base-not-directory") from error


def safe_relative_file(base_directory, relative):
    """Resolve a portable file binding without links or parent traversal."""
    _require(type(relative) is str and bool(relative) and "\\" not in relative and ":" not in relative
        and not PurePosixPath(relative).is_absolute()
        and all(part not in ("", ".", "..") for part in relative.split("/")),
        "selected-pipeline-unsafe-relative-path")
    base = _directory(base_directory)
    path = _input_file(base.joinpath(*PurePosixPath(relative).parts))
    _require(path.is_relative_to(base), "selected-pipeline-checkpoint-outside-bundle")
    return path


def measured_geometry_from_dict(value):
    """Require every native measurement field; never insert guessed defaults."""
    _require(type(value) is dict and set(value) == {field.name for field in fields(VehicleGeometry)},
        "selected-pipeline-complete-measured-geometry-required")
    return VehicleGeometry(**value)


def pipeline_id(selection):
    """Fingerprint execution bindings; provenance remains descriptive metadata."""
    return canonical_sha256({key: value for key, value in selection.items() if key not in ("pipeline_id", "provenance")})


def _validate_selection(value):
    required = {"schema_version", "pipeline_version", "pipeline_id", "status", "model", "preprocessing",
        "controller_config", "controller_config_sha256", "source_sha256"}
    _require(type(value) is dict and required <= set(value) <= required | {"provenance"},
        "selected-pipeline-schema-invalid")
    _require(type(value["schema_version"]) is int and value["schema_version"] == 1
        and value["pipeline_version"] == VERSION and value["status"] == "PASS",
        "selected-pipeline-version-or-status-invalid")
    _require(type(value["model"]) is dict and set(value["model"]) == {"kind", "architecture", "checkpoint_path",
        "checkpoint_sha256", "model_contract_sha256", "learning_config_sha256"}, "selected-pipeline-model-fields-invalid")
    model = value["model"]
    _require(model["kind"] == "refined" and model["architecture"] == "hybrid_geometry",
        "selected-pipeline-requires-selected-geometric-hybrid")
    for key in ("checkpoint_sha256", "model_contract_sha256", "learning_config_sha256"):
        _sha(model[key])
    _require(type(value["preprocessing"]) is dict and value["preprocessing"].get("input") == INPUT_CONTRACT,
        "selected-pipeline-preprocessing-contract-invalid")
    from .guarded_path_correction import validate_preprocessing_config
    validate_preprocessing_config({key: item for key, item in value["preprocessing"].items() if key != "input"})
    config = study_controller_config_from_dict(value["controller_config"])
    _require(value["controller_config_sha256"] == config.sha256, "selected-pipeline-controller-hash-mismatch")
    _require(type(value["source_sha256"]) is dict and value["source_sha256"] == selected_pipeline_source_hashes(),
        "selected-pipeline-runtime-source-mismatch")
    if "provenance" in value:
        _require(type(value["provenance"]) is dict, "selected-pipeline-provenance-must-be-object")
    _require(_sha(value["pipeline_id"]) == pipeline_id(value), "selected-pipeline-identity-mismatch")
    return config


def _validate_predictor(predictor, value):
    model = value["model"]
    identity = predictor.identity
    guarded = value["preprocessing"]["correction"] == "guarded_path"
    base = predictor.base if guarded else predictor
    _require(base.raw.identity["variant"] == model["architecture"]
        and base.identity["variant"] == model["architecture"] + "_path_near"
        and identity["checkpoint_sha256"] == model["checkpoint_sha256"]
        and identity["model_contract_sha256"] == model["model_contract_sha256"]
        and base.raw.config.sha256 == model["learning_config_sha256"],
        "selected-pipeline-loaded-model-contract-mismatch")
    _require(predictor.correction == value["preprocessing"]["correction"] and base.corrector.config.sha256 ==
        value["preprocessing"]["correction_config_sha256"], "selected-pipeline-loaded-correction-mismatch")
    if guarded:
        _require(predictor.config.to_dict() == value["preprocessing"]["guard_config"]
            and predictor.config.sha256 == value["preprocessing"]["guard_config_sha256"],
            "selected-pipeline-loaded-guard-mismatch")


def build_pipeline_selection(checkpoint, controller_config, *, base_directory,
        expected_checkpoint_sha256=None, provenance=None, preprocessing_config=None):
    """Build a reviewable pointer from an actual checkpoint using CPU only.

    The caller separately verifies experiment evidence before publishing it.
    A PASS here means a complete, source-bound runtime configuration.
    """
    from .runtime_driving import RuntimeCheckpointPredictor
    from .guarded_path_correction import path_preprocessing_config, validate_preprocessing_config, wrap_preprocessing
    base = _directory(base_directory)
    path = _input_file(checkpoint)
    _require(path.is_relative_to(base), "selected-pipeline-checkpoint-outside-bundle")
    safe_relative_file(base, path.relative_to(base).as_posix())
    config = study_controller_config_from_dict(controller_config)
    observed = _sha256_file(path)
    if expected_checkpoint_sha256 is not None:
        _require(observed == _sha(expected_checkpoint_sha256), "selected-pipeline-checkpoint-hash-mismatch")
    sources = selected_pipeline_source_hashes()
    preprocessing = validate_preprocessing_config(path_preprocessing_config() if preprocessing_config is None else preprocessing_config)
    base_predictor = RuntimeCheckpointPredictor(path, expected_sha256=observed, device="cpu", correction="path")
    predictor = wrap_preprocessing(base_predictor, preprocessing)
    value = {"schema_version": 1, "pipeline_version": VERSION, "status": "PASS",
        "model": {"kind": "refined", "architecture": "hybrid_geometry",
            "checkpoint_path": path.relative_to(base).as_posix(), "checkpoint_sha256": observed,
            "model_contract_sha256": predictor.identity["model_contract_sha256"],
            "learning_config_sha256": base_predictor.raw.config.sha256},
        "preprocessing": {"input": INPUT_CONTRACT, **preprocessing},
        "controller_config": config.to_dict(), "controller_config_sha256": config.sha256,
        "source_sha256": sources}
    if provenance is not None:
        _require(type(provenance) is dict, "selected-pipeline-provenance-must-be-object")
        value["provenance"] = copy.deepcopy(provenance)
    value["pipeline_id"] = pipeline_id(value)
    _validate_selection(value)
    _validate_predictor(predictor, value)
    _require(_sha256_file(path) == observed and selected_pipeline_source_hashes() == sources,
        "selected-pipeline-source-or-checkpoint-changed-during-build")
    # Reject non-JSON/nonfinite optional provenance as well as runtime fields.
    json.dumps(value, allow_nan=False)
    return value


def load_selected_pipeline(selection_file, geometry, *, device="cpu", expected_pipeline_id=None):
    from .runtime_driving import RuntimeCheckpointPredictor
    from .guarded_path_correction import wrap_preprocessing
    _require(device in ("cpu", "cuda") and type(device) is str, "selected-pipeline-device-must-be-explicit")
    source = _input_file(selection_file)
    _require(0 < source.stat().st_size <= MAX_SELECTION_BYTES, "selected-pipeline-pointer-size-invalid")
    encoded = source.read_bytes()
    value = strict_json_loads(encoded.decode("utf-8"))
    config = _validate_selection(value)
    if expected_pipeline_id is not None:
        _require(value["pipeline_id"] == _sha(expected_pipeline_id), "selected-pipeline-unexpected-identity")
    measured = measured_geometry_from_dict(geometry.to_dict() if isinstance(geometry, VehicleGeometry) else geometry)
    checkpoint = safe_relative_file(source.parent, value["model"]["checkpoint_path"])
    expected = value["model"]["checkpoint_sha256"]
    _require(_sha256_file(checkpoint) == expected, "selected-pipeline-checkpoint-hash-mismatch")
    base_predictor = RuntimeCheckpointPredictor(checkpoint, expected_sha256=expected, device=device, correction="path")
    predictor = wrap_preprocessing(base_predictor, {key: item for key, item in value["preprocessing"].items() if key != "input"})
    _validate_predictor(predictor, value)
    controller = build_controller(config, measured)
    _require(_sha256_file(checkpoint) == expected and source.read_bytes() == encoded
        and value["source_sha256"] == selected_pipeline_source_hashes(),
        "selected-pipeline-source-or-checkpoint-changed-during-load")
    return SelectedPipeline(value, predictor, controller, selection_file=source,
        selection_bytes=encoded, checkpoint=checkpoint)


class SelectedPipeline:
    """Stateful selected control with stateless current-frame RGB perception."""
    def __init__(self, selection, predictor, controller, *, selection_file, selection_bytes, checkpoint):
        self._selection, self.predictor, self.controller = copy.deepcopy(selection), predictor, controller
        self._selection_file, self._selection_bytes, self._checkpoint = selection_file, selection_bytes, checkpoint
        self._geometry_sha256 = canonical_sha256(controller.geometry.to_dict())
        self._frame_index, self._reset_count, self._failed = 0, 0, False

    @property
    def selection(self):
        return copy.deepcopy(self._selection)

    @property
    def identity(self):
        return {"pipeline_version": VERSION, "pipeline_id": self._selection["pipeline_id"],
            "model": self.predictor.identity, "controller_config": self.controller.config.to_dict(),
            "controller_config_sha256": self.controller.config.sha256,
            "geometry_contract": GEOMETRY_CONTRACT, "geometry_sha256": self._geometry_sha256,
            "geometry": self.controller.geometry.to_dict(), "preprocessing": copy.deepcopy(self._selection["preprocessing"])}

    def reset(self):
        """Start a fresh controller episode; never reuse a path after failure."""
        self.controller.reset()
        self._frame_index, self._failed = 0, False
        self._reset_count += 1

    def step(self, rgb, speed_mps, *, dt_seconds=None):
        _require(not self._failed, "selected-pipeline-reset-required-after-failure")
        _require(type(speed_mps) in (int, float) and math.isfinite(speed_mps) and 0 <= speed_mps <= 100,
            "selected-pipeline-speed-invalid")
        dt_seconds = self.controller.config.fixed_delta_seconds if dt_seconds is None else dt_seconds
        _require(type(dt_seconds) in (int, float) and math.isfinite(dt_seconds)
            and abs(dt_seconds - self.controller.config.fixed_delta_seconds) <= 1e-9,
            "selected-pipeline-timestep-mismatch")
        if isinstance(rgb, (bytes, bytearray)):
            _require(len(rgb) == 640 * 360 * 3, "selected-pipeline-requires-RGB8-640x360")
        else:
            _require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.shape == (360, 640, 3),
                "selected-pipeline-requires-RGB8-640x360")
        _require(self.controller.config.sha256 == self._selection["controller_config_sha256"]
            and canonical_sha256(self.controller.geometry.to_dict()) == self._geometry_sha256,
            "selected-pipeline-controller-or-geometry-changed")
        timer = self.predictor.timer()
        prediction = self.predictor.predict_profiled(rgb, timer, width=640, height=360)
        command, failure = None, None
        if prediction["status"] == "PASS":
            try:
                command = timer.measure("controller_ms", lambda: self.controller.step(
                    prediction["waypoints_ego_m"], speed_mps, dt_seconds=dt_seconds))
            except ControllerError as error:
                rejection = {"exception_type": "ControllerError", "error_code": str(error),
                    "waypoints_ego_m": copy.deepcopy(prediction["waypoints_ego_m"]),
                    "learned_waypoints_ego_m": copy.deepcopy(prediction["learned_waypoints_ego_m"])}
                prediction.update(status="FAIL", error="development-controller-rejection", waypoints_ego_m=None,
                    learned_waypoints_ego_m=None, controller_rejection=rejection)
                failure = {"kind": "controller-rejection", "code": str(error)}
        else:
            failure = {"kind": "perception", "code": prediction.get("error")}
        self._failed = prediction["status"] != "PASS"
        result = {"status": "FAIL" if self._failed else "PASS", "pipeline_id": self._selection["pipeline_id"],
            "prediction": prediction, "command": command, "failure": failure, "timing": timer.finish(),
            "timing_scope": TIMING_SCOPE, "sequence": {"frame_index": self._frame_index,
                "reset_count": self._reset_count, "controller_history_preserved": True,
                "reset_required_before_next_frame": self._failed}}
        self._frame_index += 1
        return result

    def verify_bindings(self):
        """Recheck files after a stream without hashing weights on every frame."""
        _require(self._selection_file.read_bytes() == self._selection_bytes
            and _sha256_file(self._checkpoint) == self._selection["model"]["checkpoint_sha256"]
            and selected_pipeline_source_hashes() == self._selection["source_sha256"]
            and self.controller.config.sha256 == self._selection["controller_config_sha256"]
            and canonical_sha256(self.controller.geometry.to_dict()) == self._geometry_sha256,
            "selected-pipeline-binding-changed-during-stream")
        return {"status": "PASS", "pipeline_id": self._selection["pipeline_id"], "checkpoint_and_sources_unchanged": True}
