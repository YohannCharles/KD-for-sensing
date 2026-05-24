from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

@dataclass(frozen=True)
class MobilityTrace:
    times: np.ndarray
    locations: np.ndarray
    info: Any

def _get_sample(dataset: Any, names: Sequence[str], kwarg_sets: Sequence[dict[str, Any]], default: Any = ...):
    get_sample = getattr(dataset, "get_sample")
    last_exc: Exception | None = None
    for name in names:
        for kwargs in kwarg_sets:
            try:
                return get_sample(name, **kwargs)
            except Exception as exc:
                last_exc = exc
    if default is not ...:
        return default
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No get_sample candidates were attempted.")


def _field(obj: Any, *names: str) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _extract_paths(sample: Any) -> list[str]:
    if sample is None:
        return []
    if isinstance(sample, (str, Path)):
        return [str(sample)]
    if isinstance(sample, Mapping):
        for key in ("path", "paths", "filepath", "file_path", "image_path", "lidar_path", "filename"):
            if key in sample:
                return _extract_paths(sample[key])
    if isinstance(sample, Sequence) and not isinstance(sample, (bytes, bytearray)):
        paths: list[str] = []
        for item in sample:
            paths.extend(_extract_paths(item))
        return paths
    for key in ("path", "paths", "filepath", "file_path", "image_path", "lidar_path", "filename"):
        if hasattr(sample, key):
            return _extract_paths(getattr(sample, key))
    return []


def _path_count(paths: Any) -> int:
    if paths is None:
        return 0
    if isinstance(paths, Mapping):
        if "paths" in paths:
            return _path_count(paths["paths"])
        for value in paths.values():
            try:
                return len(value)
            except TypeError:
                continue
        return len(paths)
    for attr in ("num_paths", "n_paths"):
        if hasattr(paths, attr):
            value = getattr(paths, attr)
            return int(value() if callable(value) else value)
    for attr in ("paths", "ToA", "DoD_theta", "DoA_theta", "phase"):
        if hasattr(paths, attr):
            try:
                return len(getattr(paths, attr))
            except TypeError:
                continue
    try:
        return len(paths)
    except TypeError:
        return 0

def _device_id_candidates(device_id: Any) -> list[Any]:
    candidates = [device_id]
    if isinstance(device_id, (int, np.integer)) and int(device_id) > 0:
        candidates.append(int(device_id) - 1)
    return candidates


def _metadata_value(info: Any, local_idx: int, names: Sequence[str], *, default: Any) -> Any:
    for name in names:
        value = _field(info, name)
        if value is None:
            continue
        if isinstance(value, (str, bytes)):
            return value.decode() if isinstance(value, bytes) else value
        try:
            array = np.asarray(value)
        except Exception:
            return value
        if array.ndim == 0:
            return array.item()
        if len(array) > local_idx:
            return array[local_idx].item() if np.asarray(array[local_idx]).ndim == 0 else array[local_idx]
        return value
    return default


def _dataset_value(dataset: Any, names: Sequence[str], default: Any) -> Any:
    for name in names:
        if hasattr(dataset, name):
            value = getattr(dataset, name)
            return value() if callable(value) else value
    return default


def _make_group_key(scene_id: Any, sequence_id: Any, segment_id: Any, object_id: Any) -> str:
    return "|".join(
        [
            f"scene={_id_part(scene_id)}",
            f"sequence={_id_part(sequence_id)}",
            f"segment={_id_part(segment_id)}",
            f"object={_id_part(object_id)}",
        ]
    )


def _make_sample_id(
    scenario: str,
    *,
    scene_id: Any,
    sequence_id: Any,
    segment_id: Any,
    object_id: Any,
    ue_id: Any,
    bs_id: Any,
    t_anchor: Any,
) -> str:
    return "_".join(
        [
            _id_part(scenario),
            f"scene{_id_part(scene_id)}",
            f"seq{_id_part(sequence_id) or 'single'}",
            f"seg{_id_part(segment_id) or 'full'}",
            f"obj{_id_part(object_id)}",
            f"ue{_id_part(ue_id)}",
            f"bs{_id_part(bs_id)}",
            f"t{_time_to_id(t_anchor)}",
        ]
    )


def _id_part(value: Any) -> str:
    scalar = _json_scalar(value)
    text = str(scalar)
    return (
        text.replace(" ", "")
        .replace("/", "-")
        .replace("\\", "-")
        .replace("|", "-")
        .replace(":", "-")
        .replace(".", "p")
        .replace("-", "m")
    )

def _json_dumps(payload: Any) -> str:
    return json.dumps(_jsonable(payload), separators=(",", ":"))


def _json_scalar(value: Any) -> int | float | str:
    array = np.asarray(value)
    if array.ndim == 0:
        item = array.item()
        if isinstance(item, (np.integer, int)):
            return int(item)
        if isinstance(item, (np.floating, float)):
            return float(item)
        return str(item)
    return str(value)


def _time_to_id(value: Any) -> str:
    scalar = _json_scalar(value)
    if isinstance(scalar, float) and scalar.is_integer():
        scalar = int(scalar)
    return str(scalar).replace(".", "p").replace("-", "m")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value

__all__ = [
    "MobilityTrace",
    "_dataset_value",
    "_device_id_candidates",
    "_extract_paths",
    "_field",
    "_get_sample",
    "_json_dumps",
    "_json_scalar",
    "_jsonable",
    "_make_group_key",
    "_make_sample_id",
    "_metadata_value",
    "_path_count",
]
