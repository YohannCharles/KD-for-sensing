from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from kd_sensing.preprocessing.multimodal_nf_codebook import fingerprint_path
from kd_sensing.preprocessing.multimodal_nf_constants import (
    MULTIMODAL_NF_HDF5_KEYS,
    REQUIRED_MULTIMODAL_NF_FIELDS,
)
from kd_sensing.preprocessing.multimodal_nf_paths import MultimodalNFPaths
from kd_sensing.utils.paths import resolve_path

def _hdf5_file_summary(path: Path) -> dict[str, Any]:
    h5py = _require_h5py("Multimodal-NF audit")
    with h5py.File(path, "r") as handle:
        resolved = _resolve_hdf5_fields(handle)
        datasets = {
            dataset_path: {
                "shape": [int(value) for value in item.shape],
                "dtype": str(item.dtype),
            }
            for dataset_path, item in _iter_hdf5_datasets(handle)
        }
        sample_count = 0
        if "csi" in resolved:
            sample_count = int(handle[resolved["csi"]].shape[0])
        cities = []
        if "city" in resolved:
            cities = sorted({str(item) for item in _decode_hdf5_values(np.asarray(handle[resolved["city"]][:]))})
        elif sample_count:
            cities = [_city_from_path(path)]
        return {
            "path": str(path),
            "fingerprint": fingerprint_path(path),
            "datasets": datasets,
            "resolved_fields": resolved,
            "cities": cities,
            "sample_count": sample_count,
            "missing_fields": [field for field in REQUIRED_MULTIMODAL_NF_FIELDS if field not in resolved],
        }


def _resolve_hdf5_fields(handle) -> dict[str, str]:
    paths = _dataset_paths(handle)
    by_leaf = {Path(path).name.lower(): path for path in paths}
    resolved = {}
    for field, aliases in MULTIMODAL_NF_HDF5_KEYS.items():
        for alias in aliases:
            alias_key = alias.lower()
            if alias in paths:
                resolved[field] = alias
                break
            if alias_key in by_leaf:
                resolved[field] = by_leaf[alias_key]
                break
    return resolved


def _candidate_hdf5_files(
    paths: MultimodalNFPaths,
    *,
    channel_path: str | Path | None,
    image_path: str | Path | None,
    lidar_path: str | Path | None,
) -> list[Path]:
    explicit = [
        resolve_path(item)
        for item in (channel_path, image_path, lidar_path)
        if item is not None
    ]
    if explicit:
        return list(dict.fromkeys(explicit))
    candidates = []
    for root in (paths.raw_root, paths.data_root):
        if root.exists():
            candidates.extend(sorted(root.rglob("*.h5")))
            candidates.extend(sorted(root.rglob("*.hdf5")))
    return list(dict.fromkeys(candidates))


def _candidate_codebook_files(paths: MultimodalNFPaths, *, codebook_path: str | Path | None) -> list[Path]:
    if codebook_path is not None:
        return [resolve_path(codebook_path)]
    candidates = []
    for root in (paths.codebook_root, paths.data_root):
        if root.exists():
            for suffix in ("*.pkl", "*.pickle", "*.json", "*.npz", "*.npy"):
                candidates.extend(sorted(root.rglob(suffix)))
    return list(dict.fromkeys(candidates))


def _resolve_channel_hdf5(paths: MultimodalNFPaths, channel_path: str | Path | None) -> Path:
    return _resolve_channel_hdf5_files(paths, channel_path)[0]


def _resolve_channel_hdf5_files(paths: MultimodalNFPaths, channel_path: str | Path | None) -> list[Path]:
    if channel_path is not None:
        path = resolve_path(channel_path)
        if not path.exists():
            raise FileNotFoundError(f"Multimodal-NF channel HDF5 not found: {path}")
        return [path]
    matches = []
    for candidate in _candidate_hdf5_files(paths, channel_path=None, image_path=None, lidar_path=None):
        stem = candidate.stem.lower()
        if "_img" in stem or "_lidar" in stem:
            continue
        try:
            summary = _hdf5_file_summary(candidate)
        except OSError:
            continue
        if not summary["missing_fields"]:
            matches.append(candidate)
    if matches:
        return sorted(matches)
    raise FileNotFoundError(
        "Could not find a Multimodal-NF channel HDF5 file with required fields "
        f"{list(REQUIRED_MULTIMODAL_NF_FIELDS)} under {paths.raw_root} or {paths.data_root}."
    )


def _resolve_optional_hdf5_path(paths: MultimodalNFPaths, path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Configured Multimodal-NF HDF5 file not found: {resolved}")
    return resolved


def _resolve_optional_hdf5_map(
    paths: MultimodalNFPaths,
    path: str | Path | None,
    *,
    suffix: str,
) -> dict[str, Path]:
    if path is not None:
        resolved = _resolve_optional_hdf5_path(paths, path)
        return {_city_from_path(resolved): resolved} if resolved is not None else {}
    matches: dict[str, Path] = {}
    for candidate in _candidate_hdf5_files(paths, channel_path=None, image_path=None, lidar_path=None):
        stem = candidate.stem.lower()
        if suffix.lower() not in stem:
            continue
        matches.setdefault(_city_from_path(candidate), candidate)
    return dict(sorted(matches.items()))


def _row_tokens(
    handle,
    dataset_path: str | None,
    count: int,
    *,
    fallback_values: list[str] | None = None,
    fallback: str | None = None,
    fallback_sequence: bool = False,
    fallback_frames: bool = False,
    frames_per_traj: int = 20,
) -> list[str]:
    if dataset_path is not None:
        values = _decode_hdf5_values(np.asarray(handle[dataset_path][:]))
        if len(values) >= count:
            return [str(item) for item in values[:count]]
    if fallback_values is not None and len(fallback_values) >= count:
        return [str(item) for item in fallback_values[:count]]
    if fallback is not None:
        return [str(fallback)] * count
    if fallback_sequence:
        return [str(idx // int(frames_per_traj)) for idx in range(count)]
    if fallback_frames:
        return [str(idx % int(frames_per_traj)) for idx in range(count)]
    return [str(idx) for idx in range(count)]


def _metadata_row_tokens(handle, dataset_path: str | None, count: int) -> list[str] | None:
    if dataset_path is None or dataset_path not in handle:
        return None
    values = np.asarray(handle[dataset_path][:count])
    if values.shape[0] < count:
        return None
    rows = values.reshape(count, -1)
    return ["_".join(_decode_token_value(value) for value in row.tolist()) for row in rows]


def _frame_tokens_from_runs(trajectory_tokens: list[str]) -> list[str]:
    frames = []
    previous = None
    frame_idx = 0
    for token in trajectory_tokens:
        if token != previous:
            frame_idx = 0
            previous = token
        frames.append(str(frame_idx))
        frame_idx += 1
    return frames


def _decode_hdf5_values(values: np.ndarray) -> list[Any]:
    decoded = []
    for item in values.reshape(-1).tolist():
        decoded.append(_decode_token_value(item))
    return decoded


def _decode_token_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _dataset_paths(handle) -> list[str]:
    return [path for path, _ in _iter_hdf5_datasets(handle)]


def _iter_hdf5_datasets(handle) -> list[tuple[str, Any]]:
    h5py = _require_h5py("Multimodal-NF HDF5 traversal")
    datasets = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            datasets.append((name, item))

    handle.visititems(visitor)
    return datasets


def _city_from_path(path: Path) -> str:
    match = re.search(r"City[_-]?([A-Za-z0-9]+)", path.stem, flags=re.IGNORECASE)
    if match:
        return f"City_{match.group(1)}"
    return path.stem

def _require_h5py(context: str):
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{context} requires the 'h5py' dependency. Install project dependencies in kd_mm_beam."
        ) from exc
    return h5py

__all__ = [
    "_candidate_codebook_files",
    "_candidate_hdf5_files",
    "_city_from_path",
    "_dataset_paths",
    "_frame_tokens_from_runs",
    "_hdf5_file_summary",
    "_metadata_row_tokens",
    "_require_h5py",
    "_resolve_channel_hdf5",
    "_resolve_channel_hdf5_files",
    "_resolve_hdf5_fields",
    "_resolve_optional_hdf5_map",
    "_row_tokens",
]
