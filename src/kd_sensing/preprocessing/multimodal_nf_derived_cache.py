from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from kd_sensing.preprocessing.multimodal_nf_codebook import fingerprint_path
from kd_sensing.preprocessing.multimodal_nf_constants import MULTIMODAL_NF_HDF5_KEYS
from kd_sensing.preprocessing.multimodal_nf_hdf5 import _dataset_paths, _require_h5py
from kd_sensing.preprocessing.multimodal_nf_paths import resolve_multimodal_nf_paths
from kd_sensing.utils.paths import resolve_path


CACHE_POLICIES = ("off", "auto", "read_only", "rebuild")
CACHE_VERSION = "multimodal_nf_derived_v2"
CACHE_SCHEMA_VERSION = 2
VALIDATION_MODES = ("lightweight", "strong")
DERIVED_MODALITIES = ("image", "lidar")
LIGHTWEIGHT_REQUIRED_METADATA = (
    "version",
    "cache_schema_version",
    "modality",
    "profile",
    "split",
    "seq_len",
    "num_pred",
    "source_path",
    "source_key",
    "source_size_bytes",
    "source_mtime_ns",
    "source_fingerprint",
    "storage_kind",
    "layout",
    "sample_count",
    "bytes",
    "shape",
    "dtype",
    "recommended_access_pattern",
)


def normalize_derived_cache_policy(raw_policy: Any, *, key: str = "derived_cache.policy") -> str:
    policy = str(raw_policy or "off").strip().lower()
    if policy not in CACHE_POLICIES:
        raise ValueError(f"{key} must be one of {', '.join(CACHE_POLICIES)}; got '{raw_policy}'.")
    return policy


def normalize_cache_validation_mode(raw_mode: Any, *, key: str = "derived_cache.validation_mode") -> str:
    mode = str(raw_mode or "lightweight").strip().lower()
    aliases = {
        "light": "lightweight",
        "default": "lightweight",
        "sha256": "strong",
        "fingerprint": "strong",
        "strong_fingerprint": "strong",
    }
    mode = aliases.get(mode, mode)
    if mode not in VALIDATION_MODES:
        raise ValueError(f"{key} must be one of {', '.join(VALIDATION_MODES)}; got '{raw_mode}'.")
    return mode


def derived_cache_path(
    *,
    cache_dir: str | Path,
    source_path: str | Path,
    modality: str,
    profile: str | None,
    split: str,
    seq_len: int,
    num_pred: int,
) -> Path:
    modality = _normalize_modality(modality)
    source = Path(source_path)
    stem = source.stem.replace(os.sep, "_")
    profile_name = str(profile or "default").replace("/", "_")
    return (
        Path(cache_dir)
        / "derived"
        / modality
        / profile_name
        / str(split)
        / f"{stem}_seq{int(seq_len)}_pred{int(num_pred)}.npy"
    )


def sidecar_path(cache_path: str | Path) -> Path:
    return Path(cache_path).with_suffix(Path(cache_path).suffix + ".json")


def build_expected_metadata(
    *,
    source_path: str | Path,
    modality: str,
    profile: str | None,
    split: str,
    seq_len: int,
    num_pred: int,
) -> dict[str, Any]:
    source_identity = lightweight_source_identity(source_path)
    return {
        "version": CACHE_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "modality": _normalize_modality(modality),
        "profile": profile,
        "split": str(split),
        "seq_len": int(seq_len),
        "num_pred": int(num_pred),
        **source_identity,
    }


def lightweight_source_identity(source_path: str | Path) -> dict[str, Any]:
    source = resolve_path(source_path)
    stat = source.stat()
    return {
        "source_path": str(source),
        "source_key": _source_key(source),
        "source_size_bytes": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
    }


def cache_status(
    *,
    cache_path: str | Path,
    expected: dict[str, Any],
    validation_mode: str = "lightweight",
) -> dict[str, Any]:
    validation_mode = normalize_cache_validation_mode(validation_mode, key="validation_mode")
    validation_start = time.perf_counter()
    data_path = Path(cache_path)
    meta_path = sidecar_path(data_path)
    if not data_path.exists():
        return _status_result(
            exists=False,
            valid=False,
            reason="missing_data",
            path=str(data_path),
            validation_mode=validation_mode,
            validation_start=validation_start,
        )
    if not meta_path.exists():
        return _status_result(
            exists=True,
            valid=False,
            reason="missing_metadata",
            path=str(data_path),
            validation_mode=validation_mode,
            validation_start=validation_start,
        )
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _status_result(
            exists=True,
            valid=False,
            reason=f"invalid_metadata_json: {exc}",
            path=str(data_path),
            validation_mode=validation_mode,
            validation_start=validation_start,
        )
    mismatches = _metadata_mismatches(metadata, expected)
    missing = _missing_lightweight_metadata(metadata)
    if missing:
        mismatches["lightweight_metadata"] = {"missing": missing}
    cache_bytes = _safe_file_size(data_path)
    if cache_bytes is not None and "bytes" in metadata and int(metadata["bytes"]) != int(cache_bytes):
        mismatches["bytes"] = {"expected": int(cache_bytes), "actual": int(metadata["bytes"])}
    source_fingerprint_scanned = False
    if validation_mode == "strong":
        source_fingerprint_scanned = True
        actual_fingerprint = fingerprint_path(expected.get("source_path"))
        expected_fingerprint = metadata.get("source_fingerprint")
        if expected_fingerprint is None:
            mismatches["source_fingerprint"] = {"expected": actual_fingerprint, "actual": None}
        elif actual_fingerprint != expected_fingerprint:
            mismatches["source_fingerprint"] = {"expected": expected_fingerprint, "actual": actual_fingerprint}
    validation = _validation_record(
        mode=validation_mode,
        start=validation_start,
        source_fingerprint_scanned=source_fingerprint_scanned,
        mismatches=mismatches,
    )
    return {
        "exists": True,
        "valid": not mismatches,
        "reason": "ok" if not mismatches else "metadata_mismatch",
        "path": str(data_path),
        "metadata_path": str(meta_path),
        "mismatches": mismatches,
        "metadata": metadata,
        "validation": validation,
    }


def ensure_derived_cache(
    *,
    source_path: str | Path,
    cache_path: str | Path,
    modality: str,
    profile: str | None,
    split: str,
    seq_len: int,
    num_pred: int,
    rebuild: bool = False,
    validation_mode: str = "strong",
) -> dict[str, Any]:
    validation_mode = normalize_cache_validation_mode(validation_mode, key="validation_mode")
    expected = build_expected_metadata(
        source_path=source_path,
        modality=modality,
        profile=profile,
        split=split,
        seq_len=seq_len,
        num_pred=num_pred,
    )
    status = cache_status(cache_path=cache_path, expected=expected, validation_mode=validation_mode)
    if status["valid"] and not rebuild:
        return {"generated": False, "cache_path": str(cache_path), "metadata": status.get("metadata"), "status": status}

    fingerprint_start = time.perf_counter()
    source_fingerprint = fingerprint_path(expected["source_path"])
    validation = {
        "mode": "strong",
        "duration_seconds": float(time.perf_counter() - fingerprint_start),
        "source_fingerprint_scanned": True,
        "result": "ok" if source_fingerprint else "unavailable",
        "mismatches": {},
    }
    data = _read_source_array(source_path, modality)
    metadata = {
        **expected,
        "source_fingerprint": source_fingerprint,
        "storage_kind": "npy_mmap",
        "layout": "source_contiguous_rows",
        "shape": [int(value) for value in data.shape],
        "dtype": str(data.dtype),
        "sample_count": int(data.shape[0]) if data.ndim > 0 else 0,
        "recommended_access_pattern": "sequential_or_locality_ordered_windows",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation": validation,
    }
    _atomic_save_npy(Path(cache_path), data)
    metadata["bytes"] = int(Path(cache_path).stat().st_size)
    _atomic_write_json(sidecar_path(cache_path), metadata)
    return {
        "generated": True,
        "cache_path": str(cache_path),
        "metadata": metadata,
        "previous_status": status,
        "status": cache_status(cache_path=cache_path, expected=expected, validation_mode=validation_mode),
    }


def prewarm_multimodal_nf_derived_cache(
    *,
    data_root: str | Path | None = None,
    raw_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    channel_path: str | Path | None = None,
    image_path: str | Path | None = None,
    lidar_path: str | Path | None = None,
    modalities: list[str] | tuple[str, ...] | None = None,
    split: str = "train",
    seq_len: int = 8,
    num_pred: int = 3,
    image_profile: str | None = "rgb_imagenet",
    lidar_profile: str | None = "point_cloud_xyz_10000",
    rebuild: bool = False,
    validation_mode: str = "strong",
    **_: Any,
) -> dict[str, Any]:
    paths = resolve_multimodal_nf_paths(data_root=data_root, raw_root=raw_root, cache_dir=cache_dir)
    selected = tuple(_normalize_modality(item) for item in (modalities or DERIVED_MODALITIES))
    source_paths = {
        "image": _resolve_source_paths(paths.raw_root, channel_path=channel_path, modality_path=image_path, suffix="_img"),
        "lidar": _resolve_source_paths(paths.raw_root, channel_path=channel_path, modality_path=lidar_path, suffix="_lidar"),
    }
    profiles = {"image": image_profile, "lidar": lidar_profile}
    results: dict[str, Any] = {}
    for modality in selected:
        if modality not in DERIVED_MODALITIES:
            raise ValueError(f"Multimodal-NF derived cache only supports image/lidar, got '{modality}'.")
        modality_results = []
        for source in source_paths[modality]:
            cache_path = derived_cache_path(
                cache_dir=paths.cache_dir,
                source_path=source,
                modality=modality,
                profile=profiles[modality],
                split=split,
                seq_len=seq_len,
                num_pred=num_pred,
            )
            modality_results.append(
                ensure_derived_cache(
                    source_path=source,
                    cache_path=cache_path,
                    modality=modality,
                    profile=profiles[modality],
                    split=split,
                    seq_len=seq_len,
                    num_pred=num_pred,
                    rebuild=rebuild,
                    validation_mode=validation_mode,
                )
            )
        results[modality] = {
            "sources": modality_results,
            "generated": sum(1 for item in modality_results if item.get("generated")),
            "total": len(modality_results),
        }
    return {"cache_dir": str(paths.cache_dir), "split": str(split), "modalities": results}


def _read_source_array(source_path: str | Path, modality: str) -> np.ndarray:
    modality = _normalize_modality(modality)
    h5py = _require_h5py("Multimodal-NF derived cache")
    with h5py.File(resolve_path(source_path), "r") as handle:
        key = _dataset_key_for_modality(handle, modality)
        return np.asarray(handle[key][:])


def _dataset_key_for_modality(handle, modality: str) -> str:
    available = list(_dataset_paths(handle))
    by_leaf = {Path(dataset_path).name.lower(): dataset_path for dataset_path in available}
    for alias in MULTIMODAL_NF_HDF5_KEYS[modality]:
        if alias in available:
            return alias
        resolved = by_leaf.get(alias.lower())
        if resolved is not None:
            return resolved
    raise KeyError(
        f"Could not resolve Multimodal-NF {modality} dataset in HDF5 file. "
        f"Expected aliases {MULTIMODAL_NF_HDF5_KEYS[modality]}; available datasets: {available}."
    )


def _metadata_mismatches(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mismatches = {}
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            mismatches[key] = {"expected": expected_value, "actual": actual.get(key)}
    return mismatches


def _missing_lightweight_metadata(metadata: dict[str, Any]) -> list[str]:
    return [key for key in LIGHTWEIGHT_REQUIRED_METADATA if key not in metadata or metadata.get(key) is None]


def _validation_record(
    *,
    mode: str,
    start: float,
    source_fingerprint_scanned: bool,
    mismatches: dict[str, Any],
) -> dict[str, Any]:
    return {
        "mode": mode,
        "duration_seconds": float(time.perf_counter() - start),
        "source_fingerprint_scanned": bool(source_fingerprint_scanned),
        "result": "ok" if not mismatches else "mismatch",
        "mismatches": mismatches,
    }


def _status_result(
    *,
    exists: bool,
    valid: bool,
    reason: str,
    path: str,
    validation_mode: str,
    validation_start: float,
) -> dict[str, Any]:
    return {
        "exists": exists,
        "valid": valid,
        "reason": reason,
        "path": path,
        "validation": _validation_record(
            mode=validation_mode,
            start=validation_start,
            source_fingerprint_scanned=False,
            mismatches={"status": {"reason": reason}},
        ),
    }


def _safe_file_size(path: Path) -> int | None:
    try:
        return int(path.stat().st_size)
    except OSError:
        return None


def _source_key(source: Path) -> str:
    return source.stem.replace(os.sep, "_")


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    tmp.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _resolve_source_paths(
    raw_root: Path,
    *,
    channel_path: str | Path | None,
    modality_path: str | Path | None,
    suffix: str,
) -> tuple[Path, ...]:
    if modality_path is not None:
        path = resolve_path(modality_path)
        if not path.exists():
            raise FileNotFoundError(f"Configured Multimodal-NF derived cache source not found: {path}")
        return (path,)
    if channel_path is not None:
        path = resolve_path(channel_path)
        sibling = path.with_name(path.stem + suffix + path.suffix)
        if sibling.exists():
            return (sibling,)
        if path.exists():
            return (path,)
    matches = sorted(raw_root.rglob(f"*{suffix}.h5")) + sorted(raw_root.rglob(f"*{suffix}.hdf5"))
    if matches:
        return tuple(matches)
    channel_matches = sorted(raw_root.rglob("*.h5")) + sorted(raw_root.rglob("*.hdf5"))
    if channel_matches:
        return (channel_matches[0],)
    raise FileNotFoundError(f"Could not find Multimodal-NF source HDF5 for suffix {suffix} under {raw_root}.")


def _normalize_modality(modality: Any) -> str:
    value = str(modality).strip().lower()
    if value == "lidar":
        return "lidar"
    if value == "image":
        return "image"
    return value


__all__ = [
    "CACHE_POLICIES",
    "CACHE_SCHEMA_VERSION",
    "DERIVED_MODALITIES",
    "VALIDATION_MODES",
    "build_expected_metadata",
    "cache_status",
    "derived_cache_path",
    "ensure_derived_cache",
    "lightweight_source_identity",
    "normalize_cache_validation_mode",
    "normalize_derived_cache_policy",
    "prewarm_multimodal_nf_derived_cache",
    "sidecar_path",
]
