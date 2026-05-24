from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from kd_sensing.preprocessing.multimodal_nf_constants import (
    DEFAULT_DENSE_CODEBOOK_SHAPE,
    DEFAULT_FLATTEN_ORDER,
    DEFAULT_SMALL_CODEBOOK_SHAPE,
)
from kd_sensing.utils.paths import resolve_path

def parse_codebook_metadata(
    codebook_path: str | Path | None = None,
    *,
    codebook_shape: list[int] | tuple[int, int, int] | None = None,
    profile: str | None = None,
    flatten_order: str = DEFAULT_FLATTEN_ORDER,
) -> dict[str, Any]:
    path = resolve_path(codebook_path) if codebook_path is not None else None
    raw: Any = {}
    if path is not None and path.exists() and codebook_shape is None and profile is None:
        suffix = path.suffix.lower()
        if suffix == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
        elif suffix in {".npz", ".npy"}:
            raw = _load_numpy_codebook(path)
        elif suffix in {".pkl", ".pickle"}:
            with path.open("rb") as handle:
                raw = pickle.load(handle)
        else:
            raise ValueError(f"Unsupported Multimodal-NF codebook file extension '{path.suffix}' for {path}.")
    shape = _codebook_shape_from_inputs(raw, codebook_shape=codebook_shape, profile=profile)
    num_classes = int(math.prod(shape))
    metadata = {
        "path": str(path) if path is not None else None,
        "fingerprint": fingerprint_path(path) if path is not None and path.exists() else None,
        "shape": list(shape),
        "flatten_order": str(flatten_order),
        "num_beam_classes": num_classes,
        "profile": profile or _profile_for_shape(shape),
    }
    return metadata


def flatten_beam_triplet(
    triplet: list[int] | tuple[int, int, int] | np.ndarray,
    codebook_shape: list[int] | tuple[int, int, int],
    *,
    flatten_order: str = DEFAULT_FLATTEN_ORDER,
) -> int:
    if str(flatten_order) != DEFAULT_FLATTEN_ORDER:
        raise ValueError(f"Unsupported Multimodal-NF flatten_order '{flatten_order}'.")
    values = np.asarray(triplet, dtype=np.int64).reshape(-1)
    if values.shape[0] != 3:
        raise ValueError(f"Beam triplet must contain 3 indices, got {values.tolist()}.")
    shape = tuple(int(value) for value in codebook_shape)
    if np.any(values < 0) or any(int(values[idx]) >= shape[idx] for idx in range(3)):
        raise ValueError(f"Beam triplet {values.tolist()} is outside codebook shape {list(shape)}.")
    az, el, rg = (int(value) for value in values)
    return int((az * shape[1] + el) * shape[2] + rg)


def unflatten_beam_class(
    class_id: int,
    codebook_shape: list[int] | tuple[int, int, int],
    *,
    flatten_order: str = DEFAULT_FLATTEN_ORDER,
) -> tuple[int, int, int]:
    if str(flatten_order) != DEFAULT_FLATTEN_ORDER:
        raise ValueError(f"Unsupported Multimodal-NF flatten_order '{flatten_order}'.")
    shape = tuple(int(value) for value in codebook_shape)
    total = int(math.prod(shape))
    value = int(class_id)
    if value < 0 or value >= total:
        raise ValueError(f"Beam class {value} is outside codebook size {total}.")
    az, rem = divmod(value, shape[1] * shape[2])
    el, rg = divmod(rem, shape[2])
    return int(az), int(el), int(rg)


def fingerprint_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def _load_numpy_codebook(path: Path) -> dict[str, Any]:
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        return {key: loaded[key] for key in loaded.files}
    return {"array": loaded}


def _codebook_shape_from_inputs(
    raw: Any,
    *,
    codebook_shape: list[int] | tuple[int, int, int] | None,
    profile: str | None,
) -> tuple[int, int, int]:
    if codebook_shape is not None:
        return _normalize_codebook_shape(codebook_shape)
    if profile:
        normalized = str(profile).strip().lower()
        if normalized == "dense":
            return DEFAULT_DENSE_CODEBOOK_SHAPE
        if normalized == "small":
            return DEFAULT_SMALL_CODEBOOK_SHAPE
    if isinstance(raw, dict):
        for key in ("shape", "codebook_shape", "beam_codebook_shape"):
            if key in raw:
                return _normalize_codebook_shape(raw[key])
        if all(key in raw for key in ("num_azimuth", "num_elevation", "num_range")):
            return _normalize_codebook_shape((raw["num_azimuth"], raw["num_elevation"], raw["num_range"]))
        for key in ("codebook", "array", "beam_codebook"):
            if key in raw:
                array = np.asarray(raw[key])
                if array.ndim >= 3:
                    return _normalize_codebook_shape(array.shape[:3])
    if isinstance(raw, (list, tuple, np.ndarray)):
        array = np.asarray(raw)
        if array.ndim >= 3:
            return _normalize_codebook_shape(array.shape[:3])
        if array.size == 3:
            return _normalize_codebook_shape(array.reshape(-1).tolist())
    raise ValueError(
        "Could not parse Multimodal-NF codebook shape. Configure codebook_shape, "
        "codebook_profile ('dense' or 'small'), or provide metadata with shape/codebook_shape."
    )


def _normalize_codebook_shape(value: Any) -> tuple[int, int, int]:
    values = [int(item) for item in np.asarray(value).reshape(-1).tolist()]
    if len(values) != 3 or any(item <= 0 for item in values):
        raise ValueError(f"Multimodal-NF codebook shape must contain three positive integers, got {value}.")
    return int(values[0]), int(values[1]), int(values[2])


def _profile_for_shape(shape: tuple[int, int, int]) -> str | None:
    if tuple(shape) == DEFAULT_DENSE_CODEBOOK_SHAPE:
        return "dense"
    if tuple(shape) == DEFAULT_SMALL_CODEBOOK_SHAPE:
        return "small"
    return None


def _fingerprint(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()

__all__ = [
    "fingerprint_path",
    "flatten_beam_triplet",
    "parse_codebook_metadata",
    "unflatten_beam_class",
]
