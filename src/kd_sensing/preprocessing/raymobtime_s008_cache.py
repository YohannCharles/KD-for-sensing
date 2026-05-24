from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kd_sensing.preprocessing.raymobtime_s008_beam_labels import build_s008_labels
from kd_sensing.preprocessing.raymobtime_s008_common import (
    RAYMOBTIME_S008_HDF5_FIELD_MAPPING,
    _fingerprint,
    _read_json_file,
    _row_selector_for_array,
    _split_index_files,
    _load_split_arrays,
    resolve_raymobtime_paths,
)
from kd_sensing.preprocessing.raymobtime_s008_index import build_s008_index
from kd_sensing.preprocessing.raymobtime_s008_ray_features import extract_s008_ray_features

def build_s008_cache(
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    split_seed: int = 42,
    split_ratios: list[float] | tuple[float, float, float] = (0.7, 0.15, 0.15),
    beam_key: str | None = None,
    num_tx_beams: int | None = 32,
    num_rx_beams: int | None = 8,
    link_target_name: str = "link_power_max_dbm",
) -> dict[str, Any]:
    paths = resolve_raymobtime_paths(data_root=data_root, output_dir=output_dir, cache_dir=cache_dir)
    index_paths = build_s008_index(
        data_root=paths.data_root,
        cache_dir=paths.cache_dir,
        split_seed=split_seed,
        split_ratios=split_ratios,
    )
    label_paths = build_s008_labels(
        data_root=paths.data_root,
        cache_dir=paths.cache_dir,
        beam_key=beam_key,
        num_tx_beams=num_tx_beams,
        num_rx_beams=num_rx_beams,
    )
    ray_paths = extract_s008_ray_features(
        data_root=paths.data_root,
        cache_dir=paths.cache_dir,
        link_target_name=link_target_name,
    )
    output_paths: dict[str, Any] = {**index_paths, **label_paths, **ray_paths}
    for split, split_file in _split_index_files(paths.cache_dir).items():
        frame = pd.read_csv(split_file)
        labels = np.load(paths.cache_dir / f"labels_{split}.npz", allow_pickle=True)
        rays = np.load(paths.cache_dir / f"ray_features_{split}.npz", allow_pickle=True)
        coord = frame.loc[:, ["x", "y", "z"]].to_numpy(dtype=np.float32)
        image = _optional_modality_array(paths.data_root / "baseline_data" / "image_v2_input", frame)
        lidar = _optional_modality_array(paths.data_root / "baseline_data" / "lidar_input", frame)
        cache_path = paths.cache_dir / f"cache_{split}.npz"
        np.savez(
            cache_path,
            sample_id=frame["sample_id"].astype(str).to_numpy(),
            coord=coord,
            image=image,
            lidar=lidar,
            ray=rays["ray_features_no_los"].astype(np.float32),
            ray_features_with_los=rays["ray_features_with_los"].astype(np.float32),
            target_beam=labels["beam_label"].astype(np.int64),
            beam_tx=labels["beam_tx"].astype(np.int64),
            beam_rx=labels["beam_rx"].astype(np.int64),
            los_label=labels["los_label"].astype(np.float32),
            link_quality=rays["link_quality"].astype(np.float32),
            valid_index=frame["valid_index"].to_numpy(dtype=np.int64),
        )
        output_paths[f"cache_{split}"] = str(cache_path)
    metadata_path = paths.cache_dir / "cache_metadata.json"
    metadata = _cache_metadata(paths.cache_dir, link_target_name=link_target_name)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    output_paths["cache_metadata"] = str(metadata_path)
    return output_paths



def _optional_modality_array(path: Path, frame: pd.DataFrame) -> np.ndarray:
    count = len(frame)
    arrays = _load_split_arrays(path, key=None, allow_missing=True)
    if not arrays:
        return np.zeros((count, 1), dtype=np.float32)
    if "all" in arrays:
        first = arrays["all"]
        rows = _row_selector_for_array(first, frame)(frame)
        return first[rows].astype(np.float32)
    return _array_for_source_frame(frame, arrays).astype(np.float32)


def _array_for_source_frame(frame: pd.DataFrame, arrays_by_source: dict[str, np.ndarray]) -> np.ndarray:
    if frame.empty:
        first = next(iter(arrays_by_source.values()))
        return np.empty((0, *first.shape[1:]), dtype=first.dtype)
    if not {"source_split", "source_split_index"} <= set(frame.columns):
        raise ValueError("Raymobtime split index is missing source_split/source_split_index columns.")
    result: np.ndarray | None = None
    for source_split, group in frame.groupby("source_split", sort=False):
        source_key = str(source_split)
        if source_key not in arrays_by_source:
            raise ValueError(
                f"Raymobtime split index references source_split={source_key!r}, "
                f"but available modality splits are {sorted(arrays_by_source)}."
            )
        values = np.asarray(arrays_by_source[source_key])
        positions = group["source_split_index"].to_numpy(dtype=np.int64)
        if positions.size and int(positions.max()) >= len(values):
            raise ValueError(
                f"Raymobtime source_split={source_key!r} references row {int(positions.max())}, "
                f"but modality split has only {len(values)} rows."
            )
        selected = values[positions]
        if result is None:
            result = np.empty((len(frame), *selected.shape[1:]), dtype=selected.dtype)
        result[frame.index.get_indexer(group.index)] = selected
    assert result is not None
    return result


def _cache_metadata(cache_dir: Path, *, link_target_name: str) -> dict[str, Any]:
    ray_report = _read_json_file(cache_dir / "ray_unmatched_report.json")
    ray_summary = ray_report.get("summary", {}) if isinstance(ray_report, dict) else {}
    ray_splits = ray_report.get("splits", {}) if isinstance(ray_report, dict) else {}
    metadata: dict[str, Any] = {
        "dataset": "raymobtime_s008",
        "task_semantics": "current_snapshot_beam_selection",
        "link_target_name": link_target_name,
        "link_target_unit": "dBm",
        "link_target_aggregation": link_target_name.replace("link_power_", "").replace("_dbm", ""),
        "ray_feature_version": 2,
        "ray_quality": ray_summary,
        "hdf5_schema": RAYMOBTIME_S008_HDF5_FIELD_MAPPING,
        "normalization": {},
        "splits": {},
    }
    for split in ("train", "val", "test"):
        path = cache_dir / f"cache_{split}.npz"
        if not path.exists():
            continue
        split_quality = ray_splits.get(split, {}) if isinstance(ray_splits, dict) else {}
        cache = np.load(path, allow_pickle=True)
        metadata["splits"][split] = {
            "num_samples": int(len(cache["sample_id"])),
            "fingerprint": _fingerprint([str(item) for item in cache["sample_id"].tolist()]),
            "ray_quality": split_quality,
        }
    train_path = cache_dir / "cache_train.npz"
    if train_path.exists():
        train = np.load(train_path, allow_pickle=True)
        for key in ("coord", "ray", "link_quality"):
            values = np.asarray(train[key], dtype=np.float32)
            metadata["normalization"][key] = {
                "mean": np.asarray(values.mean(axis=0)).reshape(-1).astype(float).tolist(),
                "std": np.asarray(values.std(axis=0)).reshape(-1).clip(1e-6, None).astype(float).tolist(),
            }
    labels_path = cache_dir / "labels_train.npz"
    if labels_path.exists():
        labels = np.load(labels_path, allow_pickle=True)
        metadata["beam"] = {
            "num_beam_classes": int(np.asarray(labels["num_beam_classes"]).item()),
            "num_tx_beams": int(np.asarray(labels["num_tx_beams"]).item()),
            "num_rx_beams": int(np.asarray(labels["num_rx_beams"]).item()),
        }
    return metadata




__all__ = ["build_s008_cache"]
