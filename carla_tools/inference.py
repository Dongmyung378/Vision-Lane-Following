"""Strict local RGB inference from a selected waypoint checkpoint."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import struct
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .learning_data import validate_waypoint_points
from .rgb_predictor import BaselineAssistedWaypointPredictor
from .training import load_checkpoint, runtime_source_hashes

INFERENCE_VERSION = "vlf-offline-rgb-inference-v1"
MAX_PNG_BYTES = 8 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_LOAD_ERRORS = {
    "unsupported-waypoint-checkpoint", "checkpoint-config-hash-mismatch",
    "checkpoint-model-contract-mismatch", "unsupported-checkpoint-role",
    "checkpoint-pretrained-provenance-invalid", "resume-configuration-mismatch",
    "resume-dataset-mismatch", "resume-source-mismatch",
}


class InferenceError(ValueError):
    """A stable error code without file contents or environment secrets."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _expected_sha256(value):
    if value is None:
        return None
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise InferenceError("inference-invalid-sha256")
    return value.lower()


def _reject_links(path):
    for part in (path, *path.parents):
        if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
            raise InferenceError("inference-input-or-output-symlink")


def _input_file(path):
    try:
        path = Path(path).absolute()
        _reject_links(path)
        path = path.resolve(strict=True)
        _reject_links(path)
        if not path.is_file():
            raise InferenceError("inference-input-not-file")
        return path
    except InferenceError:
        raise
    except (OSError, ValueError, TypeError):
        raise InferenceError("inference-input-not-file") from None


def _sha256_file(path):
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        raise InferenceError("inference-input-read-failed") from None
    return digest.hexdigest()


def new_output_path(path):
    """Check a new JSON target; its parent must already exist without links."""
    try:
        path = Path(path).absolute()
        _reject_links(path)
        parent = path.parent.resolve(strict=True)
        _reject_links(parent)
        if not parent.is_dir():
            raise InferenceError("inference-output-parent-not-directory")
        target = parent / path.name
        if target.exists():
            raise InferenceError("inference-output-already-exists")
        return target
    except InferenceError:
        raise
    except (OSError, ValueError, TypeError):
        raise InferenceError("inference-output-parent-unavailable") from None


def write_prediction_json(path, report):
    """Publish complete JSON with an exclusive same-volume hard link.

    An output created after the initial check is still preserved. There is no
    replace operation or fallback that overwrites an existing artifact.
    """
    target = new_output_path(path)
    try:
        encoded = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (ValueError, TypeError):
        raise InferenceError("inference-output-not-json") from None
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".waypoints-", suffix=".pending", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    except FileExistsError:
        raise InferenceError("inference-output-already-exists") from None
    except OSError:
        raise InferenceError("inference-output-write-failed") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                raise InferenceError("inference-output-cleanup-failed") from None
    return target


def _validate_png_bytes(encoded):
    if not encoded.startswith(_PNG_SIGNATURE):
        raise InferenceError("inference-requires-rgb8-png")
    offset, chunks, ended = 8, [], False
    while offset < len(encoded):
        if offset + 12 > len(encoded):
            raise InferenceError("inference-invalid-png")
        length = struct.unpack(">I", encoded[offset:offset+4])[0]
        end = offset + 12 + length
        if end > len(encoded):
            raise InferenceError("inference-invalid-png")
        kind = encoded[offset+4:offset+8]
        data = encoded[offset+8:end-4]
        crc = struct.unpack(">I", encoded[end-4:end])[0]
        if zlib.crc32(kind + data) & 0xFFFFFFFF != crc:
            raise InferenceError("inference-invalid-png")
        if not chunks:
            if kind != b"IHDR" or length != 13:
                raise InferenceError("inference-invalid-png")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", data)
            if (width, height) != (640, 360):
                raise InferenceError("inference-invalid-png-dimensions")
            if (depth, color, compression, filtering) != (8, 2, 0, 0) or interlace not in (0, 1):
                raise InferenceError("inference-requires-rgb8-png")
        elif kind == b"IHDR":
            raise InferenceError("inference-invalid-png")
        if kind in (b"acTL", b"fcTL", b"fdAT"):
            raise InferenceError("inference-requires-single-image-png")
        chunks.append(kind)
        offset = end
        if kind == b"IEND":
            if length != 0 or offset != len(encoded):
                raise InferenceError("inference-invalid-png")
            ended = True
            break
    if not ended or b"IDAT" not in chunks:
        raise InferenceError("inference-invalid-png")


def read_rgb_png(path):
    """Decode original native RGB8 bytes without channel or size conversion."""
    source = _input_file(path)
    try:
        with source.open("rb") as stream:
            encoded = stream.read(MAX_PNG_BYTES + 1)
    except OSError:
        raise InferenceError("inference-input-read-failed") from None
    if len(encoded) > MAX_PNG_BYTES:
        raise InferenceError("inference-png-too-large")
    _validate_png_bytes(encoded)
    try:
        with Image.open(io.BytesIO(encoded)) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (640, 360):
                raise InferenceError("inference-requires-rgb8-png")
            image.load()
            rgb = np.asarray(image, dtype=np.uint8).copy()
    except InferenceError:
        raise
    except (OSError, ValueError, SyntaxError):
        raise InferenceError("inference-invalid-png") from None
    return rgb, {"filename": source.name, "sha256": hashlib.sha256(encoded).hexdigest(),
                 "format": "PNG", "mode": "RGB", "bit_depth": 8, "width": 640, "height": 360}


def _selected_epoch(checkpoint):
    history, best = checkpoint.get("history"), checkpoint.get("best_dev_loss")
    if (not isinstance(history, list) or not history or type(best) not in (int, float)
            or not math.isfinite(best) or best < 0):
        return None
    for row in history:
        if (not isinstance(row, dict) or type(row.get("epoch")) is not int or row["epoch"] < 1
                or type(row.get("dev_loss")) not in (int, float) or not math.isfinite(row["dev_loss"])
                or row["dev_loss"] < 0):
            return None
    if len({row["epoch"] for row in history}) != len(history):
        return None
    selected = min(history, key=lambda row: (row["dev_loss"], row["epoch"]))
    if selected.get("selected") is not True or not math.isclose(selected["dev_loss"], best, rel_tol=1e-12, abs_tol=1e-12):
        return None
    return selected["epoch"]


def _inference_source_hashes(mode):
    sources = {"carla_tools/inference.py": _sha256_file(Path(__file__))}
    if mode == "assisted":
        sources["carla_tools/rgb_predictor.py"] = _sha256_file(Path(__file__).with_name("rgb_predictor.py"))
    return sources


class OfflineWaypointPredictor:
    """Load one selected checkpoint and predict independent RGB frames.

    The sole per-frame inference argument is the original RGB array. Output
    already uses actor-origin metres; decoding applies no additional scale.
    """

    def __init__(self, checkpoint_path, *, device="cpu", mode="learned",
                 expected_checkpoint_sha256=None, expected_manifest_sha256=None):
        expected_checkpoint = _expected_sha256(expected_checkpoint_sha256)
        expected_manifest = _expected_sha256(expected_manifest_sha256)
        if mode not in ("learned", "assisted"):
            raise InferenceError("inference-invalid-mode")
        if device not in ("cpu", "cuda"):
            raise InferenceError("inference-invalid-device")
        if device == "cuda" and not torch.cuda.is_available():
            raise InferenceError("inference-cuda-unavailable")
        checkpoint_path = _input_file(checkpoint_path)
        checkpoint_hash = _sha256_file(checkpoint_path)
        if expected_checkpoint is not None and checkpoint_hash != expected_checkpoint:
            raise InferenceError("inference-checkpoint-sha256-mismatch")
        sources = runtime_source_hashes()
        inference_sources = _inference_source_hashes(mode)
        try:
            model, config, checkpoint = load_checkpoint(checkpoint_path, device=device,
                                                       manifest_sha256=expected_manifest, source_sha256=sources)
        except Exception as error:
            code = str(error)
            raise InferenceError(code if code in _LOAD_ERRORS else "inference-checkpoint-load-failed") from None
        if _sha256_file(_input_file(checkpoint_path)) != checkpoint_hash:
            raise InferenceError("inference-checkpoint-changed-during-load")
        if runtime_source_hashes() != sources:
            raise InferenceError("inference-source-changed-during-load")
        if _inference_source_hashes(mode) != inference_sources:
            raise InferenceError("inference-serving-source-changed-during-load")
        if checkpoint.get("checkpoint_role") != "selected-model":
            raise InferenceError("inference-requires-selected-checkpoint")
        manifest_hash = _expected_sha256(checkpoint.get("manifest_sha256"))
        if manifest_hash is None:
            raise InferenceError("inference-checkpoint-missing-manifest-sha256")
        self.model, self.config = model.eval(), config
        self.device, self.mode = torch.device(device), mode
        self._predictor = model if mode == "learned" else BaselineAssistedWaypointPredictor(model)
        self._summary = {"inference_version": INFERENCE_VERSION, "mode": mode,
                         "checkpoint_role": checkpoint["checkpoint_role"], "checkpoint_sha256": checkpoint_hash,
                         "checkpoint_epoch": checkpoint.get("epoch") if type(checkpoint.get("epoch")) is int else None,
                         "selected_epoch": _selected_epoch(checkpoint),
                         "manifest_sha256": manifest_hash, "config_sha256": config.sha256,
                         "model_version": config.model_version, "config": config.to_dict(),
                         "model_contract": checkpoint["model_contract"],
                         "model_contract_sha256": checkpoint["model_contract_sha256"],
                         "source_sha256": sources,
                         "inference_source_sha256": inference_sources,
                         "pretrained_provenance": checkpoint["pretrained_provenance"],
                         "runtime": {"device": device, "torch": str(torch.__version__),
                                     "numpy": str(np.__version__), "cuda": torch.version.cuda if device == "cuda" else None}}

    @property
    def summary(self):
        return copy.deepcopy(self._summary)

    def predict_rgb(self, rgb):
        started = time.perf_counter()
        diagnostics = {"inference_version": INFERENCE_VERSION, "mode": self.mode,
                       "model_version": self.config.model_version, "input": "RGB-only",
                       "selected_predictor": None}
        try:
            if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (360, 640, 3):
                raise InferenceError("inference-requires-640x360-rgb8")
            self.model.eval()
            with torch.inference_mode():
                result = self._predictor.predict_rgb(rgb)
            if not isinstance(result, dict) or result.get("status") != "PASS":
                raise InferenceError("inference-prediction-failed")
            validate_waypoint_points(result.get("waypoints_ego_m"))
            points = [[float(x), float(y)] for x, y in result["waypoints_ego_m"]]
            source = result.get("diagnostics", {})
            diagnostics["source_diagnostics"] = source if isinstance(source, dict) else {}
            diagnostics["selected_predictor"] = "ResNet18" if self.mode == "learned" else source.get("selected_predictor")
            if diagnostics["selected_predictor"] not in ("B0", "ResNet18"):
                raise InferenceError("inference-invalid-selected-predictor")
            status, error = "PASS", None
        except Exception as failure:
            points, status = None, "FAIL"
            error = failure.code if isinstance(failure, InferenceError) else "inference-invalid-prediction"
            diagnostics["selected_predictor"] = None
        return {"status": status, "waypoints_ego_m": points, "error": error, "diagnostics": diagnostics,
                "timing_ms": (time.perf_counter() - started) * 1000}

    def predict_rgb_batch(self, images):
        """Return one stateless result per supplied native RGB frame."""
        return [self.predict_rgb(rgb) for rgb in images]

    def predict_png(self, path):
        started = time.perf_counter()
        source = None
        try:
            rgb, source = read_rgb_png(path)
            result = self.predict_rgb(rgb)
            if _sha256_file(_input_file(path)) != source["sha256"]:
                raise InferenceError("inference-image-changed-during-prediction")
        except InferenceError as error:
            result = {"status": "FAIL", "waypoints_ego_m": None, "error": error.code,
                      "diagnostics": {"inference_version": INFERENCE_VERSION, "mode": self.mode,
                                      "selected_predictor": None}, "timing_ms": 0.0}
        return {**result, "input": source, "input_sha256": source["sha256"] if source is not None else None,
                "model": self.summary,
                "total_timing_ms": (time.perf_counter() - started) * 1000}
