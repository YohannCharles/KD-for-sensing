from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from kd_sensing.diagnostics.viewer_manifest_schema import _json_ready

def _manifest_output_path(
    *,
    output_path: str | Path | None,
    cache_dir: str | Path | None,
    default_dir: Path,
    cfg: dict[str, Any],
    predictions: str | Path | dict[str, Any] | list[Any] | None,
    quality: str | Path | dict[str, Any] | list[Any] | None,
    gate: str | Path | dict[str, Any] | list[Any] | None,
    sample_limit: int | None,
) -> Path:
    if output_path is not None:
        return Path(output_path).expanduser()
    digest = _cache_digest(cfg, predictions=predictions, quality=quality, gate=gate, sample_limit=sample_limit)[:16]
    root = Path(cache_dir).expanduser() if cache_dir is not None else default_dir / "viewer_cache"
    return root / digest / "samples.json"


def _cached_manifest_result(manifest_path: Path, meta_path: Path, digest: str) -> dict[str, Any] | None:
    if not manifest_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if meta.get("cache_digest") != digest:
        return None
    try:
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(records, list):
        return None
    if not _processed_outputs_exist(records):
        return None
    if not _source_entries_match(meta.get("sources", [])):
        return None
    return {
        "mode": "viewer_dataset_cache",
        "message": "Reused existing Gradio viewer dataset cache.",
        "cache_hit": True,
        "cache_dir": str(manifest_path.parent),
        "manifest_path": str(manifest_path),
        "meta_path": str(meta_path),
        "sample_count": len(records),
        "viewer_command": (
            "conda run -n kd_mm_beam python tools/visualization/gradio_multimodal_viewer.py "
            f"--manifest {manifest_path}"
        ),
    }


def _cache_digest(
    cfg: dict[str, Any],
    *,
    predictions: str | Path | dict[str, Any] | list[Any] | None,
    quality: str | Path | dict[str, Any] | list[Any] | None,
    gate: str | Path | dict[str, Any] | list[Any] | None,
    sample_limit: int | None,
) -> str:
    payload = {
        "cfg": _fingerprint_cfg(cfg),
        "predictions": _external_descriptor(predictions),
        "quality": _external_descriptor(quality),
        "gate": _external_descriptor(gate),
        "sample_limit": sample_limit,
        "cache_version": 6,
    }
    encoded = json.dumps(_json_ready(payload), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fingerprint_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    diagnostics = deepcopy(cfg.get("diagnostics", {}).get("visualization", {}) or {})
    diagnostics.pop("output_dir", None)
    diagnostics.pop("sample_count", None)
    diagnostics.pop("per_seq_sample_count", None)
    diagnostics.pop("preserve_existing_outputs", None)
    return {
        "data": cfg.get("data", {}),
        "model": {
            "modalities": cfg.get("model", {}).get("modalities"),
            "primary": cfg.get("model", {}).get("primary", {}),
        },
        "experiment_task": cfg.get("experiment", {}).get("task"),
        "diagnostics_visualization": diagnostics,
    }


def _external_descriptor(source: str | Path | dict[str, Any] | list[Any] | None) -> Any:
    if source is None:
        return None
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        stat = _path_stat(path)
        return {"path": str(path), "stat": stat}
    return source


def _manifest_meta(
    *,
    digest: str,
    cfg: dict[str, Any],
    manifest_path: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "cache_digest": digest,
        "manifest_path": str(manifest_path),
        "sample_count": len(records),
        "config": _fingerprint_cfg(cfg),
        "sources": _source_entries(records),
    }


def _source_entries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    entries: list[dict[str, Any]] = []
    for record in records:
        extra = record.get("extra", {}) if isinstance(record.get("extra"), dict) else {}
        csv_path = extra.get("csv_path")
        for source_path in extra.get("source_paths", []) if isinstance(extra.get("source_paths"), list) else []:
            if source_path and str(source_path) not in seen:
                seen.add(str(source_path))
                entries.append({"path": str(source_path), **_path_stat_dict(str(source_path))})
        for raw_path in _iter_record_paths(record.get("raw", {})):
            if raw_path and raw_path not in seen:
                seen.add(raw_path)
                entries.append({"path": raw_path, **_path_stat_dict(raw_path)})
        if csv_path and csv_path not in seen:
            seen.add(str(csv_path))
            entries.append({"path": str(csv_path), **_path_stat_dict(str(csv_path))})
    return entries


def _iter_record_paths(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_record_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_record_paths(item)
    elif isinstance(value, str) and value.strip():
        yield value


def _processed_outputs_exist(records: list[dict[str, Any]]) -> bool:
    for record in records:
        processed = record.get("processed", {})
        for path in _iter_record_paths(processed):
            if path and not Path(path).exists():
                return False
    return True


def _source_entries_match(entries: Any) -> bool:
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if not isinstance(entry, dict):
            return False
        path = entry.get("path")
        if not path:
            continue
        current = _path_stat_dict(str(path))
        if current.get("exists") != entry.get("exists"):
            return False
        if not current.get("exists"):
            continue
        if current.get("size") != entry.get("size") or current.get("mtime_ns") != entry.get("mtime_ns"):
            return False
    return True


def _path_stat(path: Path) -> dict[str, Any]:
    return _path_stat_dict(str(path))


def _path_stat_dict(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        return {"exists": False, "size": None, "mtime_ns": None}
    stat = resolved.stat()
    return {"exists": True, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}

__all__ = [
    "_cache_digest",
    "_cached_manifest_result",
    "_manifest_meta",
    "_manifest_output_path",
    "_path_stat_dict",
]
