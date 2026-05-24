from __future__ import annotations

import json
import re
import zipfile
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kd_sensing.preprocessing.raymobtime_s008_common import (
    FALLBACK_LINK_QUALITY_DBM,
    HDF5_SUFFIXES,
    RAYMOBTIME_HDF5_CONTEXT,
    RAYMOBTIME_S008_HDF5_FIELD_MAPPING,
    RAY_FEATURE_NAMES,
    _RAY_COLUMN_ALIASES,
    _los_value,
    _sample_id,
    _split_index_files,
    resolve_raymobtime_paths,
)
from kd_sensing.preprocessing.raymobtime_s008_index import build_s008_index

def extract_s008_ray_features(
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    link_target_name: str = "link_power_max_dbm",
) -> dict[str, Any]:
    paths = resolve_raymobtime_paths(data_root=data_root, output_dir=output_dir, cache_dir=cache_dir)
    if not (paths.cache_dir / "index_all_valid.csv").exists():
        build_s008_index(data_root=paths.data_root, cache_dir=paths.cache_dir)
    ray_table = _load_ray_table(paths.ray_zip_path)
    output_paths = {}
    unmatched: list[dict[str, Any]] = []
    split_payloads: dict[str, dict[str, Any]] = {}
    split_quality: dict[str, dict[str, Any]] = {}
    for split, split_file in _split_index_files(paths.cache_dir).items():
        frame = pd.read_csv(split_file)
        no_los, with_los, link_quality, split_unmatched = _features_for_index(frame, ray_table, link_target_name)
        unmatched.extend(split_unmatched)
        quality = _ray_quality_summary(
            split,
            sample_ids=frame["sample_id"].astype(str).to_numpy(),
            link_quality=link_quality,
            unmatched=split_unmatched,
        )
        split_quality[split] = quality
        split_payloads[split] = {
            "frame": frame,
            "no_los": no_los,
            "with_los": with_los,
            "link_quality": link_quality,
        }
    quality_summary = _validate_ray_quality_summary(split_quality)
    for split, payload in split_payloads.items():
        frame = payload["frame"]
        no_los = payload["no_los"]
        with_los = payload["with_los"]
        link_quality = payload["link_quality"]
        path = paths.cache_dir / f"ray_features_{split}.npz"
        np.savez(
            path,
            sample_id=frame["sample_id"].astype(str).to_numpy(),
            ray_features_no_los=no_los.astype(np.float32),
            ray_features_with_los=with_los.astype(np.float32),
            link_quality=link_quality.astype(np.float32),
            feature_names=np.asarray(RAY_FEATURE_NAMES, dtype=object),
            with_los_feature_names=np.asarray((*RAY_FEATURE_NAMES, "LOS"), dtype=object),
            link_target_name=np.asarray(link_target_name),
            link_target_unit=np.asarray("dBm"),
        )
        output_paths[f"ray_features_{split}"] = str(path)
    report_path = paths.cache_dir / "ray_unmatched_report.json"
    report = {
        "unmatched": unmatched,
        "count": len(unmatched),
        "summary": quality_summary,
        "splits": split_quality,
        "hdf5_schema": RAYMOBTIME_S008_HDF5_FIELD_MAPPING,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_paths["unmatched_report"] = str(report_path)
    output_paths["ray_quality_summary"] = quality_summary
    return output_paths



def _load_ray_table(zip_path: Path) -> pd.DataFrame:
    if not zip_path.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            suffix = Path(name).suffix.lower()
            if suffix not in {".csv", ".txt", ".json", *HDF5_SUFFIXES}:
                continue
            raw = archive.read(name)
            frame = _ray_frame_from_bytes(raw, suffix=suffix, source=name)
            if frame is not None and not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    table = _canonicalize_ray_columns(table)
    if "sample_id" not in table.columns and {"EpisodeID", "SceneID", "VehicleArrayID"} <= set(table.columns):
        table["sample_id"] = [
            _sample_id(row.EpisodeID, row.SceneID, row.VehicleArrayID)
            for row in table.itertuples(index=False)
        ]
    return table


def _ray_frame_from_bytes(raw: bytes, *, suffix: str, source: str) -> pd.DataFrame | None:
    if suffix in HDF5_SUFFIXES:
        return _ray_frame_from_hdf5(raw, source=source)
    try:
        if suffix == ".json":
            data = json.loads(raw.decode("utf-8"))
            frame = pd.DataFrame(data if isinstance(data, list) else data.get("rows", data))
        else:
            text = raw.decode("utf-8")
            sep = "," if "," in text.splitlines()[0] else None
            frame = pd.read_csv(StringIO(text), sep=sep)
    except Exception:
        try:
            array = np.load(BytesIO(raw), allow_pickle=True)
        except Exception:
            return None
        if isinstance(array, np.ndarray) and array.ndim == 2:
            frame = pd.DataFrame(array)
        else:
            return None
    frame["_source"] = source
    return frame


def _ray_frame_from_hdf5(raw: bytes, *, source: str) -> pd.DataFrame:
    h5py = _require_h5py(source)
    with h5py.File(BytesIO(raw), "r") as handle:
        frames: list[pd.DataFrame] = []
        has_all_episode_data = "allEpisodeData" in handle
        if has_all_episode_data:
            frames.append(_raymobtime_all_episode_data_frame(handle["allEpisodeData"][()], source=source))
        frames.extend(_generic_hdf5_ray_frames(handle, source=source))
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        if has_all_episode_data:
            return pd.DataFrame(
                columns=[
                    "sample_id",
                    "EpisodeID",
                    "SceneID",
                    "VehicleArrayID",
                    "path_index",
                    "power_dbm",
                    "toa",
                    "elev_aod",
                    "az_aod",
                    "elev_aoa",
                    "az_aoa",
                    "phase",
                    "_source",
                ]
            )
        raise ValueError(
            f"{RAYMOBTIME_HDF5_CONTEXT} could not extract canonical ray rows from {source}. "
            "Expected allEpisodeData or datasets containing power and sample alignment fields."
        )
    frame = pd.concat(frames, ignore_index=True)
    frame["_source"] = source
    return _canonicalize_ray_columns(frame)


def _require_h5py(source: str):
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised with environments missing h5py
        raise ModuleNotFoundError(
            f"{RAYMOBTIME_HDF5_CONTEXT} requires the 'h5py' dependency to read {source}. "
            "Install it in the kd_mm_beam environment, for example: "
            "conda run -n kd_mm_beam python -m pip install h5py"
        ) from exc
    return h5py


def _raymobtime_all_episode_data_frame(array: np.ndarray, *, source: str) -> pd.DataFrame:
    values = np.asarray(array)
    if values.ndim != 4 or values.shape[-1] < 9:
        raise ValueError(
            f"{RAYMOBTIME_HDF5_CONTEXT} expected allEpisodeData shape [SceneID, VehicleArrayID, path, fields>=9] "
            f"for {source}; got {tuple(values.shape)}."
        )
    episode = _episode_id_from_hdf5_source(source)
    scene_count, vehicle_count, path_count, _ = values.shape
    scene_ids = np.repeat(np.arange(scene_count, dtype=np.int64), vehicle_count * path_count)
    vehicle_ids = np.tile(np.repeat(np.arange(vehicle_count, dtype=np.int64), path_count), scene_count)
    path_ids = np.tile(np.arange(path_count, dtype=np.int64), scene_count * vehicle_count)
    flat = values.reshape(-1, values.shape[-1])
    frame = pd.DataFrame(
        {
            "EpisodeID": np.full(flat.shape[0], episode, dtype=np.int64),
            "SceneID": scene_ids,
            "VehicleArrayID": vehicle_ids,
            "path_index": path_ids,
            "power_dbm": flat[:, 0],
            "toa": flat[:, 1],
            "elev_aod": flat[:, 2],
            "az_aod": flat[:, 3],
            "elev_aoa": flat[:, 4],
            "az_aoa": flat[:, 5],
            "phase": flat[:, 8],
            "raymobtime_path_flag": flat[:, 6],
        }
    )
    frame = frame.loc[np.isfinite(pd.to_numeric(frame["power_dbm"], errors="coerce"))].copy()
    frame["_source"] = source
    frame["sample_id"] = [
        _sample_id(row.EpisodeID, row.SceneID, row.VehicleArrayID)
        for row in frame.itertuples(index=False)
    ]
    return frame


def _episode_id_from_hdf5_source(source: str) -> int:
    match = re.search(r"(?:^|[_/-])e(\d+)(?:\D|$)", source)
    if not match:
        raise ValueError(
            f"{RAYMOBTIME_HDF5_CONTEXT} could not infer EpisodeID from HDF5 entry name {source!r}; "
            "expected a filename token like '_e0.hdf5'."
        )
    return int(match.group(1))


def _generic_hdf5_ray_frames(group, *, source: str, group_path: str = "") -> list[pd.DataFrame]:
    h5py = _require_h5py(source)
    frames: list[pd.DataFrame] = []
    datasets: dict[str, np.ndarray] = {}
    for key, item in group.items():
        item_path = f"{group_path}/{key}" if group_path else str(key)
        if isinstance(item, h5py.Dataset):
            value = item[()]
            if getattr(value, "dtype", None) is not None and value.dtype.names:
                frame = _hdf5_structured_array_frame(value, source=source, group_path=item_path)
                if frame is not None and not frame.empty:
                    frames.append(frame)
            else:
                datasets[str(key)] = np.asarray(value)
        elif isinstance(item, h5py.Group):
            frames.extend(_generic_hdf5_ray_frames(item, source=source, group_path=item_path))
    frame = _hdf5_dataset_mapping_frame(datasets, source=source, group_path=group_path)
    if frame is not None and not frame.empty:
        frames.append(frame)
    return frames


def _hdf5_structured_array_frame(array: np.ndarray, *, source: str, group_path: str) -> pd.DataFrame | None:
    if not array.dtype.names:
        return None
    arrays = {name: np.asarray(array[name]) for name in array.dtype.names}
    return _hdf5_dataset_mapping_frame(arrays, source=source, group_path=group_path)


def _hdf5_dataset_mapping_frame(
    datasets: dict[str, np.ndarray],
    *,
    source: str,
    group_path: str,
) -> pd.DataFrame | None:
    if not datasets:
        return None
    canonical: dict[str, np.ndarray] = {}
    raw_names: dict[str, str] = {}
    for name, values in datasets.items():
        canonical_name = _canonical_ray_column_name(name)
        if canonical_name is None:
            continue
        canonical[canonical_name] = np.asarray(values)
        raw_names[canonical_name] = name
    if "power_dbm" not in canonical:
        return None
    power = np.asarray(canonical["power_dbm"])
    if power.ndim == 0:
        return None
    sample_shape = power.shape[:-1] if power.ndim >= 2 else power.shape
    path_count = int(power.shape[-1]) if power.ndim >= 2 else 1
    row_count = int(power.size)
    columns: dict[str, np.ndarray] = {}
    for name, values in canonical.items():
        broadcast = _broadcast_hdf5_values(
            np.asarray(values),
            power_shape=power.shape,
            sample_shape=sample_shape,
            path_count=path_count,
            row_count=row_count,
        )
        if broadcast is not None:
            columns[name] = broadcast
    if "power_dbm" not in columns:
        return None
    if "path_index" not in columns:
        columns["path_index"] = (
            np.tile(np.arange(path_count, dtype=np.int64), row_count // max(path_count, 1))
            if path_count > 1
            else np.arange(row_count, dtype=np.int64)
        )
    frame = pd.DataFrame(columns)
    if "sample_id" not in frame.columns:
        ids = _ids_from_hdf5_context(source=source, group_path=group_path)
        for key, value in ids.items():
            if key not in frame.columns:
                frame[key] = value
    if "sample_id" not in frame.columns and {"EpisodeID", "SceneID", "VehicleArrayID"} <= set(frame.columns):
        frame["sample_id"] = [
            _sample_id(row.EpisodeID, row.SceneID, row.VehicleArrayID)
            for row in frame.itertuples(index=False)
        ]
    frame["_source"] = source
    frame["_hdf5_group"] = group_path
    frame["_hdf5_power_field"] = raw_names.get("power_dbm")
    frame = frame.loc[np.isfinite(pd.to_numeric(frame["power_dbm"], errors="coerce"))].copy()
    return frame


def _broadcast_hdf5_values(
    values: np.ndarray,
    *,
    power_shape: tuple[int, ...],
    sample_shape: tuple[int, ...],
    path_count: int,
    row_count: int,
) -> np.ndarray | None:
    array = np.asarray(values)
    if array.dtype.kind in {"S", "O", "U"}:
        array = _decode_hdf5_strings(array)
    if array.shape == ():
        return np.repeat(array.reshape(1), row_count)
    if tuple(array.shape) == tuple(power_shape):
        return array.reshape(row_count)
    sample_count = int(np.prod(sample_shape)) if sample_shape else 1
    if tuple(array.shape) == tuple(sample_shape):
        return np.repeat(array.reshape(sample_count), max(path_count, 1))
    if array.size == row_count:
        return array.reshape(row_count)
    if sample_count > 0 and array.size == sample_count:
        return np.repeat(array.reshape(sample_count), max(path_count, 1))
    if sample_count == 1 and path_count > 1 and array.size == path_count:
        return array.reshape(path_count)
    if array.size == 1:
        return np.repeat(array.reshape(1), row_count)
    return None


def _decode_hdf5_strings(values: np.ndarray) -> np.ndarray:
    def decode(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    return np.vectorize(decode, otypes=[object])(values)


def _ids_from_hdf5_context(*, source: str, group_path: str) -> dict[str, int]:
    text = f"{source}/{group_path}"
    ids: dict[str, int] = {}
    episode = re.search(r"(?:episode|EpisodeID|episode_id|[_/-]e)(\d+)", text)
    scene = re.search(r"(?:scene|SceneID|scene_id|[_/-]s)(\d+)", text)
    vehicle = re.search(r"(?:vehicle|VehicleArrayID|vehicle_array_id|rx|[_/-]v)(\d+)", text)
    if episode:
        ids["EpisodeID"] = int(episode.group(1))
    if scene:
        ids["SceneID"] = int(scene.group(1))
    if vehicle:
        ids["VehicleArrayID"] = int(vehicle.group(1))
    return ids


def _canonicalize_ray_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in list(result.columns):
        canonical = _canonical_ray_column_name(column)
        if canonical is None or canonical in result.columns:
            continue
        result[canonical] = result[column]
    if "power_dbm" in result.columns:
        result["power_dbm"] = pd.to_numeric(result["power_dbm"], errors="coerce")
    return result


def _canonical_ray_column_name(name: Any) -> str | None:
    token = re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())
    for canonical, aliases in _RAY_COLUMN_ALIASES.items():
        if token == re.sub(r"[^a-z0-9]+", "", canonical.lower()):
            return canonical
        for alias in aliases:
            if token == re.sub(r"[^a-z0-9]+", "", alias.lower()):
                return canonical
    return None


def _features_for_index(
    index: pd.DataFrame,
    ray_table: pd.DataFrame,
    link_target_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    features = []
    with_los = []
    link_quality = []
    unmatched = []
    grouped = ray_table.groupby("sample_id") if "sample_id" in ray_table.columns else None
    for row in index.itertuples(index=False):
        sample_id = str(row.sample_id)
        if grouped is None or sample_id not in grouped.groups:
            vector, link = _empty_ray_feature()
            unmatched.append({"sample_id": sample_id, "reason": "missing_ray_paths"})
        else:
            vector, link = _ray_feature_from_rows(grouped.get_group(sample_id), link_target_name)
        features.append(vector)
        with_los.append(np.concatenate([vector, np.asarray([_los_value(row.LOS)], dtype=np.float32)]))
        link_quality.append(link)
    return (
        np.vstack(features).astype(np.float32),
        np.vstack(with_los).astype(np.float32),
        np.asarray(link_quality, dtype=np.float32),
        unmatched,
    )


def _ray_quality_summary(
    split: str,
    *,
    sample_ids: np.ndarray,
    link_quality: np.ndarray,
    unmatched: list[dict[str, Any]],
) -> dict[str, Any]:
    values = np.asarray(link_quality, dtype=np.float64)
    finite = values[np.isfinite(values)]
    sample_count = int(len(sample_ids))
    unmatched_count = int(len(unmatched))
    fallback_mask = np.isclose(values, FALLBACK_LINK_QUALITY_DBM, atol=1e-6)
    fallback_count = int(np.count_nonzero(fallback_mask))
    summary = {
        "split": split,
        "num_samples": sample_count,
        "matched_ray_paths": int(max(sample_count - unmatched_count, 0)),
        "unmatched_samples": unmatched_count,
        "fallback_link_targets": fallback_count,
        "fallback_ratio": float(fallback_count / max(sample_count, 1)),
        "fallback_value_dbm": float(FALLBACK_LINK_QUALITY_DBM),
        "link_target_distribution": _numeric_distribution(finite),
    }
    return summary


def _numeric_distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "min": None, "mean": None, "max": None, "std": None}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
        "std": float(np.std(finite)),
    }


def _validate_ray_quality_summary(split_quality: dict[str, dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "num_samples": int(sum(int(item.get("num_samples", 0)) for item in split_quality.values())),
        "matched_ray_paths": int(sum(int(item.get("matched_ray_paths", 0)) for item in split_quality.values())),
        "unmatched_samples": int(sum(int(item.get("unmatched_samples", 0)) for item in split_quality.values())),
        "fallback_link_targets": int(sum(int(item.get("fallback_link_targets", 0)) for item in split_quality.values())),
        "fallback_value_dbm": float(FALLBACK_LINK_QUALITY_DBM),
    }
    totals["fallback_ratio"] = float(totals["fallback_link_targets"] / max(totals["num_samples"], 1))
    train = split_quality.get("train", {})
    train_distribution = train.get("link_target_distribution", {}) if isinstance(train, dict) else {}
    train_std = train_distribution.get("std")
    issues: list[str] = []
    if totals["num_samples"] > 0 and totals["matched_ray_paths"] == 0:
        issues.append("all samples are missing ray paths")
    if totals["num_samples"] > 0 and totals["fallback_link_targets"] == totals["num_samples"]:
        issues.append(f"all link_quality targets equal fallback {FALLBACK_LINK_QUALITY_DBM:g} dBm")
    if train.get("num_samples", 0) and train_std is not None and float(train_std) <= 1e-8:
        issues.append("train split link_quality standard deviation is 0")
    summary = {
        **totals,
        "splits": split_quality,
        "status": "failed" if issues else "ok",
        "issues": issues,
    }
    if issues:
        raise ValueError(
            "Raymobtime s008 ray feature cache quality gate failed: "
            + "; ".join(issues)
            + ". Check ray-tracing HDF5 parsing, sample_id alignment, and data_root configuration."
        )
    return summary


def _empty_ray_feature() -> tuple[np.ndarray, float]:
    vector = np.zeros((len(RAY_FEATURE_NAMES),), dtype=np.float32)
    vector[0] = FALLBACK_LINK_QUALITY_DBM
    vector[1] = FALLBACK_LINK_QUALITY_DBM
    return vector, FALLBACK_LINK_QUALITY_DBM


def _ray_feature_from_rows(rows: pd.DataFrame, link_target_name: str) -> tuple[np.ndarray, float]:
    power = _numeric_column(rows, ("power_dbm", "received_power_dbm", "path_power_dbm", "Pr_dBm", "power"))
    toa = _numeric_column(rows, ("toa", "time_of_arrival", "delay", "ToA"))
    valid = np.isfinite(power)
    if power.size == 0 or not np.any(valid):
        return _empty_ray_feature()
    power_v = power[valid]
    strongest = int(np.nanargmax(power))
    toa_v = toa[np.isfinite(toa)] if toa.size else np.asarray([], dtype=np.float64)
    sum_linear = float(np.sum(np.power(10.0, power_v / 10.0)))
    vector = np.asarray(
        [
            float(np.nanmax(power_v)),
            float(np.nanmean(power_v)),
            sum_linear,
            float(power_v.size),
            _safe_min(toa_v),
            _safe_mean(toa_v),
            _row_value(rows, strongest, ("toa", "time_of_arrival", "delay", "ToA")),
            _row_value(rows, strongest, ("elev_aod", "elevation_aod", "departure_elevation")),
            _row_value(rows, strongest, ("az_aod", "azimuth_aod", "departure_azimuth")),
            _row_value(rows, strongest, ("elev_aoa", "elevation_aoa", "arrival_elevation")),
            _row_value(rows, strongest, ("az_aoa", "azimuth_aoa", "arrival_azimuth")),
            _row_value(rows, strongest, ("phase", "path_phase")),
            float(np.nanmax(power_v) - np.nanmin(power_v)) if power_v.size else 0.0,
            float(np.nanmax(toa_v) - np.nanmin(toa_v)) if toa_v.size else 0.0,
        ],
        dtype=np.float32,
    )
    link = float(np.nanmax(power_v)) if link_target_name == "link_power_max_dbm" else float(10.0 * np.log10(max(sum_linear, 1e-12)))
    return vector, link


def _numeric_column(frame: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
    return np.asarray([], dtype=np.float64)


def _row_value(frame: pd.DataFrame, index: int, names: tuple[str, ...]) -> float:
    for name in names:
        if name in frame.columns:
            value = pd.to_numeric(pd.Series([frame.iloc[index][name]]), errors="coerce").iloc[0]
            return 0.0 if pd.isna(value) else float(value)
    return 0.0


def _safe_min(values: np.ndarray) -> float:
    return float(np.nanmin(values)) if values.size else 0.0


def _safe_mean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if values.size else 0.0




__all__ = ["extract_s008_ray_features"]
