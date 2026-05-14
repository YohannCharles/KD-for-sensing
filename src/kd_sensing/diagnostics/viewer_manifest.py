from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
import pandas as pd
import torch

from kd_sensing.data.scenes import retarget_deepsense_dataset_config
from kd_sensing.data.transform_ops.image import IMAGENET_RGB_MEAN, IMAGENET_RGB_STD
from kd_sensing.data.transform_ops.lidar import filter_lidar_points, read_lidar_point_cloud
from kd_sensing.diagnostics.visualization.config import (
    apply_visualization_modalities,
    parse_visualization_config,
)
from kd_sensing.diagnostics.visualization.datasets import (
    build_diagnostic_datasets,
    selected_csv_frame_for_dataset,
)
from kd_sensing.diagnostics.visualization.sampling import (
    SampleCandidate,
    collect_candidates,
    select_sample_candidates,
)
from kd_sensing.diagnostics.visualization.stats import (
    modality_statistics,
)
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.modalities import DEFAULT_IMAGE_PROFILE


def export_viewer_manifest(
    cfg: dict[str, Any],
    *,
    output_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    predictions: str | Path | dict[str, Any] | list[Any] | None = None,
    quality: str | Path | dict[str, Any] | list[Any] | None = None,
    gate: str | Path | dict[str, Any] | list[Any] | None = None,
    overwrite: bool = False,
    force_rebuild: bool = False,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Prepare a cached Gradio viewer dataset from existing dataset/config metadata."""

    requested_viz = parse_visualization_config(cfg)
    manifest_path = _manifest_output_path(
        output_path=output_path,
        cache_dir=cache_dir,
        default_dir=requested_viz.output_dir,
        cfg=cfg,
        predictions=predictions,
        quality=quality,
        gate=gate,
        sample_limit=sample_limit,
    )
    cache_root = manifest_path.parent
    meta_path = manifest_path.with_name("manifest_meta.json")
    digest = _cache_digest(
        cfg,
        predictions=predictions,
        quality=quality,
        gate=gate,
        sample_limit=sample_limit,
    )
    if not force_rebuild and not overwrite:
        cached = _cached_manifest_result(manifest_path, meta_path, digest)
        if cached is not None:
            return cached

    prediction_map = _load_external_mapping(predictions)
    quality_map = _load_external_mapping(quality)
    gate_map = _load_external_mapping(gate)

    if requested_viz.compare_scenes is not None:
        records: list[dict[str, Any]] = []
        for scene_id in requested_viz.compare_scenes:
            scene_cfg = deepcopy(cfg)
            retarget_deepsense_dataset_config(scene_cfg.setdefault("data", {}).setdefault("dataset", {}), scene_id)
            scene_cfg.setdefault("diagnostics", {}).setdefault("visualization", {})["compare_scenes"] = None
            records.extend(
                _records_for_single_scene(
                    scene_cfg,
                    prediction_map=prediction_map,
                    quality_map=quality_map,
                    gate_map=gate_map,
                    output_dir=cache_root,
                    sample_limit=sample_limit,
                )
            )
    else:
        records = _records_for_single_scene(
            cfg,
            prediction_map=prediction_map,
            quality_map=quality_map,
            gate_map=gate_map,
            output_dir=cache_root,
            sample_limit=sample_limit,
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_json_ready(records), indent=2, ensure_ascii=False), encoding="utf-8")
    meta = _manifest_meta(digest=digest, cfg=cfg, manifest_path=manifest_path, records=records)
    meta_path.write_text(json.dumps(_json_ready(meta), indent=2, ensure_ascii=False), encoding="utf-8")

    viewer_command = (
        "conda run -n kd_mm_beam python tools/visualization/gradio_multimodal_viewer.py "
        f"--manifest {manifest_path}"
    )
    return {
        "mode": "viewer_dataset_cache",
        "message": "Prepared a reusable Gradio viewer dataset cache from the selected dataset config.",
        "cache_hit": False,
        "cache_dir": str(cache_root),
        "manifest_path": str(manifest_path),
        "meta_path": str(meta_path),
        "sample_count": len(records),
        "viewer_command": viewer_command,
    }


def _records_for_single_scene(
    cfg: dict[str, Any],
    *,
    prediction_map: dict[str, Any],
    quality_map: dict[str, Any],
    gate_map: dict[str, Any],
    output_dir: Path,
    sample_limit: int | None,
) -> list[dict[str, Any]]:
    requested_viz = parse_visualization_config(cfg)
    effective_cfg = apply_visualization_modalities(cfg, requested_viz.modalities)
    enabled_modalities = resolve_enabled_modalities(effective_cfg)
    viz = requested_viz
    datasets = build_diagnostic_datasets(effective_cfg, viz.splits)

    records: list[dict[str, Any]] = []
    for split in viz.splits:
        dataset = datasets[split]
        csv_frame = selected_csv_frame_for_dataset(dataset)
        candidates = collect_candidates(dataset, csv_frame)
        selected, _ = select_sample_candidates(
            candidates,
            sample_count=len(candidates),
            per_seq_sample_count=None,
            seed=viz.seed,
            seq_index=viz.seq_index,
            labels=viz.labels,
        )
        if sample_limit is not None:
            selected = selected[: max(0, int(sample_limit))]
        for candidate in selected:
            row = csv_frame.iloc[candidate.dataset_index]
            sample = dataset[candidate.dataset_index]
            sample_id = _sample_id(dataset, split, candidate)
            record = _manifest_record(
                dataset,
                split=split,
                row=row,
                candidate=candidate,
                sample=sample,
                sample_id=sample_id,
                enabled_modalities=enabled_modalities,
                output_dir=output_dir,
            )
            _attach_prediction_bundle(record, prediction_map, sample_id, candidate.dataset_index)
            _attach_optional(record, "quality", quality_map, sample_id, candidate.dataset_index)
            _attach_optional(record, "gate", gate_map, sample_id, candidate.dataset_index)
            records.append(record)
    return records


def _manifest_record(
    dataset: Any,
    *,
    split: str,
    row: pd.Series,
    candidate: SampleCandidate,
    sample: dict[str, Any],
    sample_id: str,
    enabled_modalities: tuple[str, ...],
    output_dir: Path,
) -> dict[str, Any]:
    data_root = Path(str(getattr(dataset, "data_root", ""))).expanduser()
    asset_dir = output_dir / "viewer_assets" / sample_id
    data_spaces: dict[str, dict[str, str]] = {"raw": {}, "processed": {}}
    raw = _write_raw_assets(row, sample, asset_dir, data_root, dataset=dataset, data_spaces=data_spaces)
    processed = _write_processed_assets(sample, asset_dir, dataset=dataset, data_spaces=data_spaces)
    current_beam = _tensor_int_list(sample.get("input_beam"))
    future_beams = _tensor_int_list(sample.get("target_beam"))
    extra = {
        "dataset_index": int(candidate.dataset_index),
        "csv_row_index": int(candidate.csv_row_index),
        "csv_path": str(getattr(dataset, "root_csv", "")),
        "enabled_modalities": list(enabled_modalities),
        "image_profile": getattr(dataset, "image_profile", None),
        "statistics": modality_statistics(sample),
        "source_paths": _all_source_paths(row, data_root),
        "data_spaces": data_spaces,
    }
    return {
        "sample_id": sample_id,
        "scene_id": getattr(dataset, "scene_id", None),
        "scene_slug": getattr(dataset, "scene_slug", None),
        "split": split,
        "sequence_id": candidate.seq_index,
        "time_index": int(candidate.dataset_index),
        "timestamp": str(candidate.csv_row_index),
        "raw": raw,
        "processed": processed,
        "label": {
            "current_beam": current_beam[-1] if current_beam else None,
            "future_beams": future_beams or ([candidate.future_label] if candidate.future_label is not None else []),
        },
        "extra": extra,
    }


def _raw_paths(row: pd.Series, data_root: Path) -> dict[str, str]:
    return {
        "image": _last_existing_path(row, "camera", data_root),
        "lidar": _last_existing_path(row, "lidar", data_root),
        "radar": _last_existing_path(row, "radar", data_root),
        "gps": _last_existing_path(row, "gps", data_root),
        "mmwave": _last_existing_path(row, "mmwave", data_root),
    }


def _all_source_paths(row: pd.Series, data_root: Path) -> list[str]:
    paths: list[str] = []
    for prefix in ("camera", "radar", "gps", "bs_gps", "lidar", "mmwave", "beam", "future_beam"):
        for value in _all_row_paths(row, prefix, data_root):
            paths.append(value)
            da_path = _radar_da_path(Path(value)) if value.endswith("_RA.npy") else None
            if da_path is not None:
                paths.append(str(da_path))
    seen: set[str] = set()
    unique = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _write_raw_assets(
    row: pd.Series,
    sample: dict[str, Any],
    asset_dir: Path,
    data_root: Path,
    *,
    dataset: Any,
    data_spaces: dict[str, dict[str, str]],
) -> dict[str, str]:
    raw: dict[str, str] = {}
    asset_dir.mkdir(parents=True, exist_ok=True)

    raw["image"] = _last_existing_path(row, "camera", data_root)
    if raw["image"]:
        data_spaces["raw"]["image"] = "source_image"

    lidar_source_path = _last_existing_path(row, "lidar", data_root)
    if lidar_source_path:
        path = asset_dir / "raw_lidar_points.png"
        lidar_space = _save_raw_lidar_preview(
            lidar_source_path,
            path,
            roi=getattr(dataset, "lidar_roi", None),
            bev_size=getattr(dataset, "lidar_bev_size", (224, 224)),
        )
        if lidar_space is not None:
            raw["lidar"] = str(path)
            data_spaces["raw"]["lidar"] = lidar_space
        else:
            raw["lidar"] = lidar_source_path
            data_spaces["raw"]["lidar"] = "source_lidar_unrendered"
    else:
        raw["lidar"] = ""

    radar_path = _last_existing_path(row, "radar", data_root)
    if radar_path:
        path = asset_dir / "raw_radar.png"
        if _save_raw_radar_preview(radar_path, path):
            raw["radar"] = str(path)
            data_spaces["raw"]["radar"] = _radar_source_space(radar_path)
        else:
            raw["radar"] = radar_path
            data_spaces["raw"]["radar"] = "source_radar_unrendered"
    else:
        raw["radar"] = ""

    gps_payload = _raw_gps_payload(_all_row_paths(row, "gps", data_root))
    if gps_payload is not None:
        path = asset_dir / "raw_gps.json"
        path.write_text(json.dumps(_json_ready(gps_payload), ensure_ascii=False), encoding="utf-8")
        raw["gps"] = str(path)
        data_spaces["raw"]["gps"] = "lat_lon"
    else:
        raw["gps"] = _last_existing_path(row, "gps", data_root)

    mmwave_payload = _raw_mmwave_payload(_all_row_paths(row, "mmwave", data_root))
    if mmwave_payload is not None:
        path = asset_dir / "raw_mmwave.json"
        path.write_text(json.dumps(_json_ready(mmwave_payload), ensure_ascii=False), encoding="utf-8")
        raw["mmwave"] = str(path)
        data_spaces["raw"]["mmwave"] = "linear_power"
    else:
        raw["mmwave"] = _last_existing_path(row, "mmwave", data_root)

    return raw


def _write_processed_assets(
    sample: dict[str, Any],
    asset_dir: Path,
    *,
    dataset: Any,
    data_spaces: dict[str, dict[str, str]],
) -> dict[str, str]:
    processed: dict[str, str] = {}
    asset_dir.mkdir(parents=True, exist_ok=True)
    if "image" in sample:
        path = asset_dir / "processed_image.png"
        profile = str(getattr(dataset, "image_profile", DEFAULT_IMAGE_PROFILE) or DEFAULT_IMAGE_PROFILE)
        _save_processed_image(_last_time_frame(sample["image"]), path, profile=profile)
        processed["image"] = str(path)
        data_spaces["processed"]["image"] = profile
    if "lidar" in sample:
        path = asset_dir / "processed_lidar.png"
        _save_lidar_image(sample["lidar"], path)
        processed["lidar"] = str(path)
        data_spaces["processed"]["lidar"] = "dataset_bev"
    if "radar_ra" in sample or "radar_da" in sample:
        path = asset_dir / "processed_radar.png"
        _save_radar_image(sample.get("radar_ra"), sample.get("radar_da"), path)
        processed["radar"] = str(path)
        data_spaces["processed"]["radar"] = "dataset_ra_da"
    if "gps" in sample:
        path = asset_dir / "processed_gps.json"
        gps_payload = _processed_gps_payload(sample["gps"], dataset)
        path.write_text(json.dumps(_json_ready(gps_payload), ensure_ascii=False), encoding="utf-8")
        processed["gps"] = str(path)
        data_spaces["processed"]["gps"] = str(gps_payload.get("feature_space", "gps_features"))
    if "mmwave" in sample:
        path = asset_dir / "processed_mmwave.json"
        payload = _processed_mmwave_payload(sample["mmwave"], dataset)
        path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False), encoding="utf-8")
        processed["mmwave"] = str(path)
        data_spaces["processed"]["mmwave"] = str(payload.get("scale", "mmwave_features"))
    return processed


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
            "teacher": cfg.get("model", {}).get("teacher", {}),
            "student": cfg.get("model", {}).get("student", {}),
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


def _load_external_mapping(source: str | Path | dict[str, Any] | list[Any] | None) -> dict[str, Any]:
    if source is None:
        return {}
    payload: Any
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    else:
        payload = source
    if isinstance(payload, dict):
        if isinstance(payload.get("samples"), list):
            return _mapping_from_records(payload["samples"])
        return {str(key): value for key, value in payload.items()}
    if isinstance(payload, list):
        return _mapping_from_records(payload)
    return {}


def _mapping_from_records(records: Iterable[Any]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        keys = [record.get("sample_id"), record.get("dataset_index"), index]
        for key in keys:
            if key is not None:
                mapping[str(key)] = record
    return mapping


def _attach_optional(record: dict[str, Any], field: str, mapping: dict[str, Any], sample_id: str, dataset_index: int) -> None:
    value = mapping.get(sample_id, mapping.get(str(dataset_index)))
    if isinstance(value, dict) and field in value:
        record[field] = value[field]
    elif value is not None:
        record[field] = value


def _attach_prediction_bundle(
    record: dict[str, Any],
    mapping: dict[str, Any],
    sample_id: str,
    dataset_index: int,
) -> None:
    value = mapping.get(sample_id, mapping.get(str(dataset_index)))
    if value is None:
        return
    if isinstance(value, dict):
        record["prediction"] = value.get("prediction", value)
        for field in ("confidence", "confidence_curves", "beam_distribution"):
            if field in value:
                record[field] = value[field]
        return
    record["prediction"] = value


def _save_raw_lidar_preview(
    lidar_path: str,
    output_path: Path,
    *,
    roi: list[float] | tuple[float, ...] | None,
    bev_size: list[int] | tuple[int, int],
) -> str | None:
    source = Path(lidar_path).expanduser()
    if not source.exists():
        return None
    try:
        if source.suffix.lower() == ".npy":
            array = np.load(source)
            if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
                _save_lidar_image(array, output_path)
                return "precomputed_bev"
        points = read_lidar_point_cloud(Path("/"), str(source))
        _save_lidar_points_topdown(points, output_path, roi=roi, bev_size=bev_size)
        return "point_cloud_topdown"
    except Exception:
        return None


def _save_lidar_points_topdown(
    points: np.ndarray,
    path: Path,
    *,
    roi: list[float] | tuple[float, ...] | None,
    bev_size: list[int] | tuple[int, int],
) -> None:
    height, width = int(bev_size[0]), int(bev_size[1])
    if roi is None:
        finite = np.asarray(points, dtype=np.float32)
        finite = finite[np.isfinite(finite).all(axis=1)] if finite.ndim == 2 else np.empty((0, 4), dtype=np.float32)
        if finite.size:
            x_min, x_max = float(np.min(finite[:, 0])), float(np.max(finite[:, 0]))
            y_min, y_max = float(np.min(finite[:, 1])), float(np.max(finite[:, 1]))
        else:
            x_min, x_max, y_min, y_max = -1.0, 1.0, -1.0, 1.0
        roi_values = (x_min, x_max, y_min, y_max)
        filtered = finite
    else:
        roi_values = tuple(float(value) for value in roi)
        filtered = filter_lidar_points(points, roi=roi_values)
    image = np.zeros((height, width), dtype=np.float32)
    if filtered.size:
        x_min, x_max, y_min, y_max = roi_values[:4]
        x_span = max(float(x_max) - float(x_min), 1e-6)
        y_span = max(float(y_max) - float(y_min), 1e-6)
        cols = np.floor((filtered[:, 0] - float(x_min)) / x_span * width).astype(np.int64)
        rows = height - 1 - np.floor((filtered[:, 1] - float(y_min)) / y_span * height).astype(np.int64)
        cols = np.clip(cols, 0, width - 1)
        rows = np.clip(rows, 0, height - 1)
        np.add.at(image, (rows, cols), 1.0)
        image = np.log1p(image)
    Image.fromarray(_normalize_to_uint8(image)).convert("RGB").save(path)


def _processed_gps_payload(value: Any, dataset: Any) -> dict[str, Any]:
    array = _as_numpy(value)
    feature_space = str(getattr(dataset, "gps_feature_mode", "gps_features") or "gps_features")
    normalized = getattr(dataset, "gps_scaler", None) is not None
    names = _gps_feature_names(feature_space, array)
    return {
        "features": array.tolist(),
        "feature_names": names,
        "feature_space": f"{feature_space}_zscore" if normalized else feature_space,
        "normalized": bool(normalized),
    }


def _gps_feature_names(feature_space: str, array: np.ndarray) -> list[str]:
    last_dim = int(array.shape[-1]) if array.ndim else 1
    if feature_space == "relative_polar" and last_dim == 3:
        return ["distance", "sin_theta", "cos_theta"]
    return [f"feature_{idx}" for idx in range(last_dim)]


def _processed_mmwave_payload(value: Any, dataset: Any) -> dict[str, Any]:
    array = _as_numpy(value)
    normalized = getattr(dataset, "mmwave_scaler", None) is not None and bool(getattr(dataset, "mmwave_normalize", False))
    payload = {"beam_power_seq": array.tolist()} if array.ndim >= 2 else {"beam_power": array.reshape(-1).tolist()}
    payload.update(
        {
            "scale": "z_score" if normalized else "db",
            "units": "standard_deviation" if normalized else "dB",
            "normalized": bool(normalized),
        }
    )
    return payload


def _save_raw_radar_preview(radar_path: str, output_path: Path) -> bool:
    ra_path = Path(radar_path).expanduser()
    if not ra_path.exists():
        return False
    da_path = _radar_da_path(ra_path)
    try:
        ra = np.load(ra_path)
        da = np.load(da_path) if da_path is not None and da_path.exists() else None
    except Exception:
        return False
    _save_radar_image(ra, da, output_path)
    return output_path.exists()


def _radar_source_space(radar_path: str) -> str:
    name = Path(radar_path).name
    if "_RA" in name or "_DA" in name:
        return "precomputed_ra_da"
    return "source_radar_array"


def _radar_da_path(ra_path: Path) -> Path | None:
    text = str(ra_path)
    if "_RA" not in text:
        return None
    return Path(text.replace("_RA", "_DA"))


def _raw_gps_payload(paths: list[str]) -> dict[str, list[float]] | None:
    points: list[list[float]] = []
    for path in paths:
        try:
            values = np.loadtxt(path, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if values.size >= 2 and np.all(np.isfinite(values[:2])):
            points.append([float(values[0]), float(values[1])])
    if not points:
        return None
    return {
        "x": [point[0] for point in points],
        "y": [point[1] for point in points],
        "coordinate_space": "lat_lon",
    }


def _raw_mmwave_payload(paths: list[str]) -> dict[str, Any] | None:
    rows: list[list[float]] = []
    for path in paths:
        try:
            values = np.loadtxt(path, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if values.size > 0 and np.all(np.isfinite(values)):
            rows.append(values.astype(float).tolist())
    if not rows:
        return None
    if len(rows) == 1:
        return {"beam_power": rows[0], "scale": "linear_power", "normalized": False}
    return {"beam_power_seq": rows, "scale": "linear_power", "normalized": False}


def _sample_id(dataset: Any, split: str, candidate: SampleCandidate) -> str:
    scene_slug = str(getattr(dataset, "scene_slug", "scene"))
    seq = str(candidate.seq_index).replace("/", "_").replace("\\", "_")
    return f"{scene_slug}_{split}_idx{candidate.dataset_index:06d}_seq{seq}"


def _sorted_numbered_columns(columns: Iterable[str], prefix: str) -> list[str]:
    selected = []
    for col in columns:
        name = str(col)
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if suffix.isdigit():
            selected.append(name)
    return sorted(selected, key=lambda name: int(name[len(prefix) :]))


def _last_existing_path(row: pd.Series, prefix: str, data_root: Path) -> str:
    values = [str(row[col]) for col in _sorted_numbered_columns(row.index, prefix) if str(row[col]).strip() != "-99"]
    if not values:
        return ""
    return str(_resolve_row_path(values[-1], data_root))


def _all_row_paths(row: pd.Series, prefix: str, data_root: Path) -> list[str]:
    paths = []
    for col in _sorted_numbered_columns(row.index, prefix):
        value = str(row[col]).strip()
        if value == "-99" or not value:
            continue
        paths.append(str(_resolve_row_path(value, data_root)))
    return paths


def _resolve_row_path(value: str, data_root: Path) -> Path:
    text = str(value).strip()
    path = Path(text).expanduser()
    if path.is_absolute() and path.exists():
        return path
    relative = text.lstrip("/")
    return data_root / relative


def _tensor_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    return _as_numpy(value).reshape(-1).astype(int).tolist()


def _last_time_frame(value: Any) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim >= 3:
        return array[-1]
    return array


def _save_scalar_image(value: Any, path: Path) -> None:
    array = _as_numpy(value)
    if array.ndim == 3 and array.shape[-1] in {3, 4}:
        image = _normalize_to_uint8(array[..., :3])
        Image.fromarray(image).save(path)
        return
    if array.ndim == 3:
        array = array[0]
    Image.fromarray(_normalize_to_uint8(array)).save(path)


def _save_processed_image(value: Any, path: Path, *, profile: str) -> None:
    array = _as_numpy(value)
    if profile == "rgb_imagenet" and array.ndim == 3 and array.shape[0] == 3:
        mean = np.asarray(IMAGENET_RGB_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(IMAGENET_RGB_STD, dtype=np.float32).reshape(3, 1, 1)
        rgb = np.clip(array.astype(np.float32) * std + mean, 0.0, 1.0)
        Image.fromarray((np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)).save(path)
        return
    _save_scalar_image(array, path)


def _save_lidar_image(value: Any, path: Path) -> None:
    array = _as_numpy(value)
    if array.ndim == 4:
        array = array[-1]
    if array.ndim == 3:
        channels = [_normalize_float(array[idx]) for idx in range(min(3, array.shape[0]))]
        while len(channels) < 3:
            channels.append(np.zeros_like(channels[0] if channels else array[0], dtype=np.float32))
        image = (np.stack(channels, axis=-1) * 255).astype(np.uint8)
        Image.fromarray(image).save(path)
        return
    Image.fromarray(_normalize_to_uint8(array)).save(path)


def _save_radar_image(ra: Any, da: Any, path: Path) -> None:
    panels = []
    for value in (ra, da):
        if value is None:
            continue
        array = _as_numpy(value)
        if array.ndim >= 3:
            array = array[-1]
        panels.append(_normalize_to_uint8(array))
    if not panels:
        panels = [np.zeros((16, 16), dtype=np.uint8)]
    height = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < height:
            pad = np.zeros((height - panel.shape[0], panel.shape[1]), dtype=np.uint8)
            panel = np.vstack([panel, pad])
        padded.append(panel)
    Image.fromarray(np.hstack(padded)).save(path)


def _normalize_to_uint8(value: Any) -> np.ndarray:
    array = _normalize_float(value)
    return (array * 255).astype(np.uint8)


def _normalize_float(value: Any) -> np.ndarray:
    array = _as_numpy(value).astype(np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    if np.isclose(min_value, max_value):
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - min_value) / (max_value - min_value), 0.0, 1.0)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_ready(value.detach().cpu().numpy())
    return value
