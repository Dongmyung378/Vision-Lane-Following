"""Same-frame BGRA-to-RGB evidence for D7; no image or CARLA dependencies."""

from __future__ import annotations

import hashlib
import re
from typing import Any


class ColorError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def color_provenance(raw_bgra: bytes, rgb: bytes, width: int, height: int) -> dict:
    """Retain buffer hashes plus a source pixel distinguishing red from blue."""
    count = width * height
    if not count or len(raw_bgra) != count * 4 or len(rgb) != count * 3:
        raise ColorError("color-buffer-length-mismatch")
    witness = max(range(count), key=lambda index: abs(raw_bgra[4 * index] - raw_bgra[4 * index + 2]))
    indices = sorted({0, count // 2, count - 1, witness})
    evidence = {
        "source_format": "BGRA8", "output_format": "RGB8",
        "source_sha256": hashlib.sha256(raw_bgra).hexdigest(),
        "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
        "sample_pixels": [{"index": index, "bgra": list(raw_bgra[4 * index:4 * index + 4])} for index in indices],
    }
    verify_rgb_color(rgb, evidence, width, height)
    return evidence


def validate_color_provenance(value: Any, width: int, height: int) -> None:
    required = {"source_format", "output_format", "source_sha256", "rgb_sha256", "sample_pixels"}
    if type(value) is not dict or value.keys() != required:
        raise ColorError("color-provenance-schema-invalid")
    if value["source_format"] != "BGRA8" or value["output_format"] != "RGB8":
        raise ColorError("color-format-invalid")
    if any(type(value[key]) is not str or not re.fullmatch(r"[0-9a-f]{64}", value[key]) for key in ("source_sha256", "rgb_sha256")):
        raise ColorError("color-hash-invalid")
    samples = value["sample_pixels"]
    if type(samples) is not list or not 1 <= len(samples) <= 4:
        raise ColorError("color-samples-invalid")
    indices = set()
    for sample in samples:
        if type(sample) is not dict or sample.keys() != {"index", "bgra"}:
            raise ColorError("color-samples-invalid")
        index, bgra = sample["index"], sample["bgra"]
        if type(index) is not int or not 0 <= index < width * height or index in indices:
            raise ColorError("color-samples-invalid")
        if type(bgra) is not list or len(bgra) != 4 or any(type(v) is not int or not 0 <= v <= 255 for v in bgra):
            raise ColorError("color-samples-invalid")
        indices.add(index)


def verify_rgb_color(rgb: bytes, evidence: dict, width: int, height: int, *, require_witness: bool = False) -> dict:
    """Compare decoded persisted RGB with its recorded sensor-buffer evidence.

    The source hash identifies the capture buffer; the raw buffer is not stored.
    Pixel witnesses check channel order independently of the converter's hash.
    """
    validate_color_provenance(evidence, width, height)
    if len(rgb) != width * height * 3 or hashlib.sha256(rgb).hexdigest() != evidence["rgb_sha256"]:
        raise ColorError("decoded-rgb-hash-mismatch")
    witness = False
    for sample in evidence["sample_pixels"]:
        index, bgra = sample["index"], sample["bgra"]
        if list(rgb[3 * index:3 * index + 3]) != [bgra[2], bgra[1], bgra[0]]:
            raise ColorError("bgra-rgb-channel-order-mismatch")
        witness |= bgra[0] != bgra[2]
    if require_witness and not witness:
        raise ColorError("no-red-blue-distinguishing-pixel")
    return {"status": "PASS", "decoded_rgb_sha256": evidence["rgb_sha256"],
            "source_pixel_format": "BGRA8", "stored_pixel_format": "RGB8",
            "pixel_witnesses": len(evidence["sample_pixels"]), "red_blue_witness": witness}
