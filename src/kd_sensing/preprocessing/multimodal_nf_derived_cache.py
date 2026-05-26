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
LEGACY_CACHE_VERSION = "multimodal_nf_derived_v1"
VALIDATION_MODES = ("lightweight", "strong")
DERIVED_MODALITIES = ("image", "lidar")
STATUS_VALID = "valid"
STATUS_MIGRATION_PENDING = "migration_pending"
STATUS_INVALID = "invalid"
STATUS_MISSING = "missing"
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
MIGRATION_BACKFILL_METADATA = {
    "version",
    "cache_schema_version",
    "source_key",
    "source_size_bytes",
    "source_mtime_ns",
    "storage_kind",
    "layout",
    "bytes",
    "recommended_access_pattern",
}
MIGRATION_REQUIRED_METADATA = (
    "modality",
    "profile",
    "split",
    "seq_len",
    "num_pred",
    "source_path",
    "source_fingerprint",
    "sample_count",
    "shape",
    "dtype",
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
            status=STATUS_MISSING,
        )
    if not meta_path.exists():
        return _status_result(
            exists=True,
            valid=False,
            reason="missing_metadata",
            path=str(data_path),
            validation_mode=validation_mode,
            validation_start=validation_start,
            status=STATUS_MISSING,
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
            status=STATUS_INVALID,
        )
    try:
        cache_header = _read_npy_header(data_path)
    except Exception as exc:
        return _status_result(
            exists=True,
            valid=False,
            reason=f"invalid_npy_header: {exc}",
            path=str(data_path),
            validation_mode=validation_mode,
            validation_start=validation_start,
            status=STATUS_INVALID,
            metadata_path=str(meta_path),
            metadata=metadata,
        )
    mismatches = _metadata_mismatches(metadata, expected)
    missing = _missing_lightweight_metadata(metadata)
    if missing:
        mismatches["lightweight_metadata"] = {"missing": missing}
    cache_bytes = _safe_file_size(data_path)
    metadata_bytes = _safe_int(metadata.get("bytes")) if "bytes" in metadata else None
    if cache_bytes is not None and "bytes" in metadata and metadata_bytes != int(cache_bytes):
        mismatches["bytes"] = {"expected": int(cache_bytes), "actual": metadata.get("bytes")}
    header_mismatches = _cache_header_mismatches(metadata, cache_header)
    mismatches.update(header_mismatches)
    source_fingerprint_scanned = False
    actual_fingerprint = None
    if validation_mode == "strong":
        source_fingerprint_scanned = True
        actual_fingerprint = fingerprint_path(expected.get("source_path"))
        expected_fingerprint = metadata.get("source_fingerprint")
        if expected_fingerprint is None:
            mismatches["source_fingerprint"] = {"expected": actual_fingerprint, "actual": None}
        elif actual_fingerprint != expected_fingerprint:
            mismatches["source_fingerprint"] = {"expected": expected_fingerprint, "actual": actual_fingerprint}
    pending_fields = _pending_migration_fields(metadata, missing)
    migration_mismatches = _migration_safety_mismatches(
        metadata=metadata,
        expected=expected,
        cache_header=cache_header,
        cache_bytes=cache_bytes,
        actual_source_fingerprint=actual_fingerprint,
        validation_mode=validation_mode,
    )
    migration_pending = bool(pending_fields) and not migration_mismatches
    status_name = STATUS_VALID if not mismatches else STATUS_MIGRATION_PENDING if migration_pending else STATUS_INVALID
    validation_mismatches = {} if migration_pending else mismatches
    validation = _validation_record(
        mode=validation_mode,
        start=validation_start,
        source_fingerprint_scanned=source_fingerprint_scanned,
        mismatches=validation_mismatches,
        result=STATUS_MIGRATION_PENDING if migration_pending else None,
    )
    return {
        "exists": True,
        "valid": status_name == STATUS_VALID,
        "status": status_name,
        "missing": False,
        "migration_pending": migration_pending,
        "metadata_upgrade_supported": migration_pending,
        "reason": "ok" if status_name == STATUS_VALID else status_name if migration_pending else "metadata_mismatch",
        "path": str(data_path),
        "metadata_path": str(meta_path),
        "mismatches": mismatches,
        "migration_mismatches": migration_mismatches,
        "pending_fields": pending_fields,
        "missing_lightweight_metadata": missing,
        "sidecar_schema_version": sidecar_schema_version(metadata),
        "validation_mode": validation_mode,
        "cache_header": cache_header,
        "metadata": metadata,
        "validation": validation,
    }


def upgrade_derived_cache_sidecar(
    *,
    source_path: str | Path,
    cache_path: str | Path,
    modality: str,
    profile: str | None,
    split: str,
    seq_len: int,
    num_pred: int,
    validation_mode: str = "lightweight",
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
    if status.get("valid"):
        return {
            "metadata_upgraded": False,
            "generated": False,
            "rebuilt": False,
            "cache_generated": False,
            "cache_rebuilt": False,
            "cache_path": str(cache_path),
            "metadata": status.get("metadata"),
            "status": status,
        }
    if not status.get("migration_pending"):
        return {
            "metadata_upgraded": False,
            "generated": False,
            "rebuilt": False,
            "cache_generated": False,
            "cache_rebuilt": False,
            "cache_path": str(cache_path),
            "previous_status": status,
            "status": status,
            "reason": status.get("reason", "not_migratable"),
        }

    metadata = dict(status.get("metadata") or {})
    cache_header = status.get("cache_header") or _read_npy_header(Path(cache_path))
    cache_bytes = int(Path(cache_path).stat().st_size)
    upgraded = {
        **metadata,
        **expected,
        "version": CACHE_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "storage_kind": "npy_mmap",
        "layout": "source_contiguous_rows",
        "shape": [int(value) for value in cache_header["shape"]],
        "dtype": str(cache_header["dtype"]),
        "sample_count": int(cache_header["sample_count"]),
        "bytes": cache_bytes,
        "recommended_access_pattern": "sequential_or_locality_ordered_windows",
        "metadata_upgraded_at": datetime.now(timezone.utc).isoformat(),
        "previous_cache_schema_version": status.get("sidecar_schema_version"),
        "validation": {
            **dict(status.get("validation", {})),
            "result": "metadata_upgraded",
            "pending_fields": list(status.get("pending_fields", [])),
            "mismatches": {},
        },
    }
    _atomic_write_json(sidecar_path(cache_path), upgraded)
    upgraded_status = cache_status(cache_path=cache_path, expected=expected, validation_mode=validation_mode)
    return {
        "metadata_upgraded": True,
        "generated": False,
        "rebuilt": False,
        "cache_generated": False,
        "cache_rebuilt": False,
        "cache_path": str(cache_path),
        "metadata": upgraded,
        "previous_status": status,
        "status": upgraded_status,
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
        return {
            "generated": False,
            "rebuilt": False,
            "metadata_upgraded": False,
            "cache_generated": False,
            "cache_rebuilt": False,
            "cache_path": str(cache_path),
            "metadata": status.get("metadata"),
            "status": status,
        }
    if not rebuild and status.get("migration_pending"):
        upgraded = upgrade_derived_cache_sidecar(
            source_path=source_path,
            cache_path=cache_path,
            modality=modality,
            profile=profile,
            split=split,
            seq_len=seq_len,
            num_pred=num_pred,
            validation_mode=validation_mode,
        )
        if upgraded.get("status", {}).get("valid"):
            return upgraded

    had_data = Path(cache_path).exists()
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
    rebuilt = bool(had_data)
    return {
        "generated": True,
        "rebuilt": rebuilt,
        "metadata_upgraded": False,
        "cache_generated": not rebuilt,
        "cache_rebuilt": rebuilt,
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
            try:
                result = ensure_derived_cache(
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
            except Exception as exc:
                result = {
                    "generated": False,
                    "rebuilt": False,
                    "metadata_upgraded": False,
                    "cache_generated": False,
                    "cache_rebuilt": False,
                    "failed": True,
                    "cache_path": str(cache_path),
                    "source_path": str(source),
                    "error": str(exc),
                }
            modality_results.append(result)
        results[modality] = {
            "sources": modality_results,
            "generated": sum(
                1
                for item in modality_results
                if item.get("cache_generated") or (item.get("generated") and not item.get("rebuilt"))
            ),
            "rebuilt": sum(1 for item in modality_results if item.get("cache_rebuilt") or item.get("rebuilt")),
            "metadata_upgraded": sum(1 for item in modality_results if item.get("metadata_upgraded")),
            "valid": sum(1 for item in modality_results if item.get("status", {}).get("valid")),
            "skipped": sum(
                1
                for item in modality_results
                if item.get("status", {}).get("valid") and not item.get("generated") and not item.get("metadata_upgraded")
            ),
            "failed": sum(1 for item in modality_results if item.get("failed")),
            "missing": sum(
                1
                for item in modality_results
                if item.get("status", {}).get("status") == STATUS_MISSING
                or item.get("previous_status", {}).get("status") == STATUS_MISSING
            ),
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
        actual_value = actual.get(key)
        if key == "source_path" and _same_path_value(actual_value, expected_value):
            continue
        if actual_value != expected_value:
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
    result: str | None = None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "duration_seconds": float(time.perf_counter() - start),
        "source_fingerprint_scanned": bool(source_fingerprint_scanned),
        "result": result or ("ok" if not mismatches else "mismatch"),
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
    status: str,
    metadata_path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    missing = status == STATUS_MISSING
    return {
        "exists": exists,
        "valid": valid,
        "status": status,
        "missing": missing,
        "migration_pending": False,
        "metadata_upgrade_supported": False,
        "reason": reason,
        "path": path,
        "metadata_path": metadata_path,
        "mismatches": {"status": {"reason": reason}},
        "migration_mismatches": {"status": {"reason": reason}},
        "pending_fields": [],
        "missing_lightweight_metadata": [],
        "sidecar_schema_version": sidecar_schema_version(metadata),
        "validation_mode": validation_mode,
        "metadata": metadata or {},
        "validation": _validation_record(
            mode=validation_mode,
            start=validation_start,
            source_fingerprint_scanned=False,
            mismatches={"status": {"reason": reason}},
        ),
    }


def sidecar_schema_version(metadata: Any) -> int | None:
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("cache_schema_version")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    version = str(metadata.get("version") or "").strip().lower()
    if version == LEGACY_CACHE_VERSION:
        return 1
    if version == CACHE_VERSION:
        return CACHE_SCHEMA_VERSION
    return None


def summarize_cache_statuses(items: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total": 0,
        "valid": 0,
        "migration_pending": 0,
        "invalid": 0,
        "missing": 0,
        "metadata_upgrade_supported": 0,
        "metadata_upgraded": 0,
        "generated": 0,
        "rebuilt": 0,
        "failed": 0,
        "sidecar_schema_versions": {},
        "pending_fields": {},
        "validation_duration_seconds": 0.0,
        "source_fingerprint_scanned": False,
    }
    for item in _iter_status_items(items):
        summary["total"] += 1
        status = _item_status(item)
        status_name = _status_name(status)
        if status_name in {"valid", "migration_pending", "invalid", "missing"}:
            summary[status_name] += 1
        if bool(status.get("metadata_upgrade_supported") or status.get("migration_pending")):
            summary["metadata_upgrade_supported"] += 1
        if bool(item.get("metadata_upgraded") if isinstance(item, dict) else False):
            summary["metadata_upgraded"] += 1
        if bool(item.get("cache_generated") if isinstance(item, dict) else False):
            summary["generated"] += 1
        elif bool(item.get("generated") if isinstance(item, dict) else False) and not bool(
            item.get("rebuilt") if isinstance(item, dict) else False
        ):
            summary["generated"] += 1
        if bool(item.get("cache_rebuilt") or item.get("rebuilt") if isinstance(item, dict) else False):
            summary["rebuilt"] += 1
        if bool(item.get("failed") if isinstance(item, dict) else False):
            summary["failed"] += 1
        schema_version = status.get("sidecar_schema_version")
        if schema_version is None and isinstance(item, dict):
            schema_version = (item.get("metadata") or {}).get("cache_schema_version")
        key = str(schema_version) if schema_version is not None else "unknown"
        summary["sidecar_schema_versions"][key] = summary["sidecar_schema_versions"].get(key, 0) + 1
        for field in status.get("pending_fields", []) or status.get("missing_lightweight_metadata", []) or []:
            field_key = str(field)
            summary["pending_fields"][field_key] = summary["pending_fields"].get(field_key, 0) + 1
        validation = status.get("validation") if isinstance(status.get("validation"), dict) else {}
        summary["validation_duration_seconds"] += float(validation.get("duration_seconds", 0.0) or 0.0)
        summary["source_fingerprint_scanned"] = bool(
            summary["source_fingerprint_scanned"] or validation.get("source_fingerprint_scanned")
        )
    return summary


def probe_cache_sidecar(cache_path: str | Path) -> dict[str, Any]:
    data_path = Path(cache_path)
    meta_path = sidecar_path(data_path)
    if not data_path.exists():
        return {"status": STATUS_MISSING, "exists": False, "valid": False, "reason": "missing_data", "path": str(data_path)}
    if not meta_path.exists():
        return {"status": STATUS_MISSING, "exists": True, "valid": False, "reason": "missing_metadata", "path": str(data_path)}
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "status": STATUS_INVALID,
            "exists": True,
            "valid": False,
            "reason": f"invalid_metadata_json: {exc}",
            "path": str(data_path),
            "metadata_path": str(meta_path),
        }
    missing = _missing_lightweight_metadata(metadata)
    schema_version = sidecar_schema_version(metadata)
    status = STATUS_VALID if schema_version == CACHE_SCHEMA_VERSION and not missing else STATUS_MIGRATION_PENDING
    return {
        "status": status,
        "exists": True,
        "valid": status == STATUS_VALID,
        "migration_pending": status == STATUS_MIGRATION_PENDING,
        "metadata_upgrade_supported": status == STATUS_MIGRATION_PENDING,
        "reason": "ok" if status == STATUS_VALID else "sidecar_migration_pending",
        "path": str(data_path),
        "metadata_path": str(meta_path),
        "sidecar_schema_version": schema_version,
        "pending_fields": _pending_migration_fields(metadata, missing),
        "missing_lightweight_metadata": missing,
        "metadata": metadata,
    }


def _read_npy_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            shape, fortran_order, dtype = np.lib.format._read_array_header(handle, version)  # type: ignore[attr-defined]
    return {
        "shape": [int(value) for value in shape],
        "dtype": str(np.dtype(dtype)),
        "sample_count": int(shape[0]) if shape else 0,
        "fortran_order": bool(fortran_order),
    }


def _cache_header_mismatches(metadata: dict[str, Any], cache_header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mismatches: dict[str, dict[str, Any]] = {}
    if "shape" in metadata:
        actual_shape = _safe_int_list(metadata.get("shape"))
        if actual_shape != cache_header["shape"]:
            mismatches["shape"] = {"expected": cache_header["shape"], "actual": metadata.get("shape")}
    if "dtype" in metadata:
        try:
            actual_dtype = str(np.dtype(metadata.get("dtype")))
        except (TypeError, ValueError):
            actual_dtype = None
        if actual_dtype != cache_header["dtype"]:
            mismatches["dtype"] = {"expected": cache_header["dtype"], "actual": metadata.get("dtype")}
    if "sample_count" in metadata and _safe_int(metadata.get("sample_count")) != int(cache_header["sample_count"]):
        mismatches["sample_count"] = {
            "expected": int(cache_header["sample_count"]),
            "actual": metadata.get("sample_count"),
        }
    return mismatches


def _pending_migration_fields(metadata: dict[str, Any], missing: list[str]) -> list[str]:
    pending: list[str] = []
    schema_version = sidecar_schema_version(metadata)
    if schema_version != CACHE_SCHEMA_VERSION:
        pending.extend(["version", "cache_schema_version"])
    for field in missing:
        if field in MIGRATION_BACKFILL_METADATA and field not in pending:
            pending.append(field)
    return pending


def _migration_safety_mismatches(
    *,
    metadata: dict[str, Any],
    expected: dict[str, Any],
    cache_header: dict[str, Any],
    cache_bytes: int | None,
    actual_source_fingerprint: str | None,
    validation_mode: str,
) -> dict[str, dict[str, Any]]:
    mismatches: dict[str, dict[str, Any]] = {}
    schema_version = sidecar_schema_version(metadata)
    if schema_version not in {1, CACHE_SCHEMA_VERSION}:
        mismatches["cache_schema_version"] = {"expected": f"1 or {CACHE_SCHEMA_VERSION}", "actual": schema_version}
    for key in MIGRATION_REQUIRED_METADATA:
        if key not in metadata or metadata.get(key) is None:
            mismatches[key] = {"expected": "present", "actual": metadata.get(key)}
    for key in ("modality", "profile", "split", "seq_len", "num_pred"):
        if metadata.get(key) != expected.get(key):
            mismatches[key] = {"expected": expected.get(key), "actual": metadata.get(key)}
    if not _same_path_value(metadata.get("source_path"), expected.get("source_path")):
        mismatches["source_path"] = {"expected": expected.get("source_path"), "actual": metadata.get("source_path")}
    source_fingerprint = metadata.get("source_fingerprint")
    if source_fingerprint in {None, ""}:
        mismatches["source_fingerprint"] = {"expected": "present", "actual": source_fingerprint}
    elif validation_mode == "strong" and actual_source_fingerprint != source_fingerprint:
        mismatches["source_fingerprint"] = {"expected": source_fingerprint, "actual": actual_source_fingerprint}
    header_mismatches = _cache_header_mismatches(metadata, cache_header)
    mismatches.update(header_mismatches)
    if "bytes" in metadata and cache_bytes is not None and _safe_int(metadata.get("bytes")) != int(cache_bytes):
        mismatches["bytes"] = {"expected": int(cache_bytes), "actual": metadata.get("bytes")}
    for key in ("source_key", "source_size_bytes", "source_mtime_ns", "storage_kind", "layout"):
        expected_value = expected.get(key)
        if key == "storage_kind":
            expected_value = "npy_mmap"
        elif key == "layout":
            expected_value = "source_contiguous_rows"
        if key in metadata and metadata.get(key) != expected_value:
            mismatches[key] = {"expected": expected_value, "actual": metadata.get(key)}
    return mismatches


def _same_path_value(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left == right
    try:
        return resolve_path(left) == resolve_path(right)
    except Exception:
        return str(left) == str(right)


def _iter_status_items(items: Any):
    if isinstance(items, dict):
        if _looks_like_status_summary(items):
            return
        if "sources" in items and isinstance(items["sources"], dict):
            yield from _iter_status_items(items["sources"])
        elif "status" in items or "valid" in items or "migration_pending" in items:
            yield items
        else:
            for key, value in items.items():
                if key in {"cache_status_summary", "status_counts", "sidecar_schema_versions", "pending_fields"}:
                    continue
                yield from _iter_status_items(value)
    elif isinstance(items, (list, tuple)):
        for value in items:
            yield from _iter_status_items(value)


def _item_status(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        status = item.get("status")
        if isinstance(status, dict):
            return status
        return item
    return {}


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_int_list(value: Any) -> list[int] | None:
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return None


def _looks_like_status_summary(items: dict[str, Any]) -> bool:
    return (
        "sidecar_schema_versions" in items
        and "total" in items
        and any(key in items for key in ("valid", "migration_pending", "invalid", "missing"))
    )


def _status_name(status: dict[str, Any]) -> str:
    raw = status.get("status")
    if raw in {STATUS_VALID, STATUS_MIGRATION_PENDING, STATUS_INVALID, STATUS_MISSING}:
        return str(raw)
    if bool(status.get("valid")):
        return STATUS_VALID
    if bool(status.get("migration_pending")):
        return STATUS_MIGRATION_PENDING
    if bool(status.get("missing")) or str(status.get("reason", "")).startswith("missing"):
        return STATUS_MISSING
    return STATUS_INVALID


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
    "probe_cache_sidecar",
    "prewarm_multimodal_nf_derived_cache",
    "sidecar_path",
    "sidecar_schema_version",
    "summarize_cache_statuses",
    "upgrade_derived_cache_sidecar",
]
