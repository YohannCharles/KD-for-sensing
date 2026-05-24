from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from kd_sensing.data.dataset_runtime import SampleIndex, SampleRow, write_index_csv
from kd_sensing.data.layouts import MULTIMODAL_NF_FAMILY
from kd_sensing.preprocessing.multimodal_nf_codebook import _fingerprint, fingerprint_path
from kd_sensing.preprocessing.multimodal_nf_constants import (
    MULTIMODAL_NF_DATASET_TYPE,
    REQUIRED_MULTIMODAL_NF_FIELDS,
)
from kd_sensing.preprocessing.multimodal_nf_hdf5 import (
    _city_from_path,
    _dataset_paths,
    _frame_tokens_from_runs,
    _metadata_row_tokens,
    _require_h5py,
    _resolve_channel_hdf5_files,
    _resolve_hdf5_fields,
    _resolve_optional_hdf5_map,
    _row_tokens,
)
from kd_sensing.preprocessing.multimodal_nf_paths import resolve_multimodal_nf_paths
from kd_sensing.preprocessing.multimodal_nf_splits import _assign_multimodal_nf_splits, _normalize_split

def build_multimodal_nf_index(
    data_root: str | Path | None = None,
    raw_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    channel_path: str | Path | None = None,
    image_path: str | Path | None = None,
    lidar_path: str | Path | None = None,
    split_mode: str = "city",
    train_cities: list[str] | tuple[str, ...] | None = None,
    val_cities: list[str] | tuple[str, ...] | None = None,
    test_cities: list[str] | tuple[str, ...] | None = None,
    split_ratios: list[float] | tuple[float, float, float] = (0.7, 0.15, 0.15),
    split_seed: int = 42,
    seq_len: int = 8,
    num_pred: int | None = None,
    pred_horizon: int | None = None,
) -> dict[str, Any]:
    horizon = _resolve_prediction_horizon(num_pred=num_pred, pred_horizon=pred_horizon)
    paths = resolve_multimodal_nf_paths(data_root=data_root, raw_root=raw_root, cache_dir=cache_dir)
    rows, metadata = build_multimodal_nf_rows(
        data_root=paths.data_root,
        raw_root=paths.raw_root,
        channel_path=channel_path,
        image_path=image_path,
        lidar_path=lidar_path,
        split_mode=split_mode,
        train_cities=train_cities,
        val_cities=val_cities,
        test_cities=test_cities,
        split_ratios=split_ratios,
        split_seed=split_seed,
        seq_len=seq_len,
        num_pred=horizon,
    )
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = _index_prefix(seq_len=seq_len, num_pred=horizon)
    all_index = write_index_csv(paths.cache_dir / f"{prefix}_all.csv", rows)
    split_paths: dict[str, str] = {"index_all": str(all_index)}
    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if _normalize_split(row.split) == split]
        output_name = "val" if split == "validation" else split
        split_paths[f"index_{output_name}"] = str(write_index_csv(paths.cache_dir / f"{prefix}_{output_name}.csv", split_rows))
    metadata_path = paths.cache_dir / f"{prefix}_split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    split_paths["split_metadata"] = str(metadata_path)
    return split_paths


def build_multimodal_nf_rows(
    *,
    data_root: str | Path,
    raw_root: str | Path | None = None,
    channel_path: str | Path | None = None,
    image_path: str | Path | None = None,
    lidar_path: str | Path | None = None,
    split_mode: str = "city",
    train_cities: list[str] | tuple[str, ...] | None = None,
    val_cities: list[str] | tuple[str, ...] | None = None,
    test_cities: list[str] | tuple[str, ...] | None = None,
    split_ratios: list[float] | tuple[float, float, float] = (0.7, 0.15, 0.15),
    split_seed: int = 42,
    seq_len: int = 8,
    num_pred: int = 3,
) -> tuple[list[SampleRow], dict[str, Any]]:
    seq_len = int(seq_len)
    num_pred = int(num_pred)
    if seq_len <= 0 or num_pred <= 0:
        raise ValueError(f"Multimodal-NF sequence windows require positive seq_len and num_pred, got {seq_len}, {num_pred}.")
    paths = resolve_multimodal_nf_paths(data_root=data_root, raw_root=raw_root)
    channel_files = _resolve_channel_hdf5_files(paths, channel_path)
    image_files = _resolve_optional_hdf5_map(paths, image_path, suffix="_img")
    lidar_files = _resolve_optional_hdf5_map(paths, lidar_path, suffix="_lidar")
    h5py = _require_h5py("Multimodal-NF index builder")
    pending_rows: list[dict[str, Any]] = []
    beam_chunks = []
    los_chunks = []
    nf_chunks = []
    frame_items: list[dict[str, Any]] = []
    for channel_file in channel_files:
        with h5py.File(channel_file, "r") as handle:
            resolved = _resolve_hdf5_fields(handle)
            missing = [field for field in REQUIRED_MULTIMODAL_NF_FIELDS if field not in resolved]
            if missing:
                raise ValueError(
                    f"Multimodal-NF HDF5 file {channel_file} missing required fields {missing}. "
                    f"Available datasets: {_dataset_paths(handle)}."
                )
            count = int(handle[resolved["csi"]].shape[0])
            cities = _row_tokens(handle, resolved.get("city"), count, fallback=_city_from_path(channel_file))
            metadata_tokens = _metadata_row_tokens(handle, resolved.get("metadata"), count)
            trajectories = _row_tokens(
                handle,
                resolved.get("trajectory"),
                count,
                fallback_values=metadata_tokens,
                fallback_sequence=True,
                frames_per_traj=20,
            )
            frames = _row_tokens(
                handle,
                resolved.get("frame"),
                count,
                fallback_values=_frame_tokens_from_runs(trajectories) if metadata_tokens is not None else None,
                fallback_frames=True,
                frames_per_traj=20,
            )
            beam_idx = np.asarray(handle[resolved["beam_idx"]][:])
            beam_power = np.asarray(handle[resolved["beam_power"]][:])
            los = np.asarray(handle[resolved["los"]][:])
            nf = np.asarray(handle[resolved["nf"]][:])
        beam_chunks.append(beam_idx)
        los_chunks.append(los)
        nf_chunks.append(nf)
        file_city = _city_from_path(channel_file)
        image_file = image_files.get(file_city)
        lidar_file = lidar_files.get(file_city)
        for idx in range(count):
            city = str(cities[idx])
            trajectory = str(trajectories[idx])
            frame = str(frames[idx])
            frame_items.append(
                {
                    "channel_file": channel_file,
                    "channel_index": int(idx),
                    "global_index": len(frame_items),
                    "file_city": file_city,
                    "city": city,
                    "trajectory": trajectory,
                    "frame": frame,
                    "resolved": resolved,
                    "image_file": image_file,
                    "lidar_file": lidar_file,
                    "beam_top1_triplet": np.asarray(beam_idx[idx, 0]).astype(int).tolist(),
                    "beam_power_top1": float(np.asarray(beam_power[idx, 0]).reshape(())),
                    "los_label": int(np.asarray(los[idx]).reshape(())),
                    "nf_label": int(np.asarray(nf[idx]).reshape(())),
                }
            )
    window_items = _sequence_window_items(frame_items, seq_len=seq_len, num_pred=num_pred)
    splits = _assign_multimodal_nf_splits(
        [str(item["city"]) for item in window_items],
        split_mode=split_mode,
        train_cities=train_cities,
        val_cities=val_cities,
        test_cities=test_cities,
        split_ratios=split_ratios,
        seed=split_seed,
    )
    rows = []
    for row_idx, item in enumerate(window_items):
        city = item["city"]
        trajectory = item["trajectory"]
        frame = item["history_frames"][-1]
        channel_file = item["channel_file"]
        channel_index = int(item["history_indices"][-1])
        target_indices = [int(value) for value in item["target_indices"]]
        sample_id = (
            f"{city}:traj{trajectory}:hist{item['history_frames'][0]}-{item['history_frames'][-1]}"
            f":pred{item['target_frames'][0]}-{item['target_frames'][-1]}"
        )
        refs = {
            "channel_path": str(channel_file),
            "channel_index": channel_index,
            "history_indices": [int(value) for value in item["history_indices"]],
            "target_indices": target_indices,
            "global_index": int(item["target_global_indices"][0]),
            "hdf5_keys": item["resolved"],
        }
        if item["image_file"] is not None:
            refs["image_path"] = str(item["image_file"])
        if item["lidar_file"] is not None:
            refs["lidar_path"] = str(item["lidar_file"])
        rows.append(
            SampleRow(
                sample_id=sample_id,
                split=splits[row_idx],
                dataset_type=MULTIMODAL_NF_DATASET_TYPE,
                family=MULTIMODAL_NF_FAMILY,
                scene_or_city=city,
                trajectory_id=trajectory,
                frame_id=frame,
                resource_refs=refs,
                target_ref={
                    "channel_path": str(channel_file),
                    "channel_index": target_indices[0],
                    "target_indices": target_indices,
                    "beam_idx_key": item["resolved"]["beam_idx"],
                    "beam_power_key": item["resolved"]["beam_power"],
                },
                metadata={
                    "beam_top1_triplet": item["target_beam_top1_triplets"][0],
                    "beam_top1_triplets": item["target_beam_top1_triplets"],
                    "beam_power_top1": item["target_beam_power_top1"][0],
                    "beam_power_top1_horizon": item["target_beam_power_top1"],
                    "los_label": item["target_los_labels"][0],
                    "los_labels": item["target_los_labels"],
                    "nf_label": item["target_nf_labels"][0],
                    "nf_labels": item["target_nf_labels"],
                    "history_frame_ids": item["history_frames"],
                    "target_frame_ids": item["target_frames"],
                },
            )
        )
    beam_idx_all = np.concatenate(beam_chunks, axis=0) if beam_chunks else np.empty((0, 5, 3), dtype=np.int64)
    los_all = np.concatenate([np.asarray(chunk).reshape(-1) for chunk in los_chunks], axis=0) if los_chunks else np.empty((0,))
    nf_all = np.concatenate([np.asarray(chunk).reshape(-1) for chunk in nf_chunks], axis=0) if nf_chunks else np.empty((0,))
    metadata = _index_metadata(
        rows,
        split_mode=split_mode,
        split_ratios=split_ratios,
        split_seed=split_seed,
        data_root=paths.data_root,
        channel_paths=channel_files,
        beam_idx=beam_idx_all,
        los=los_all,
        nf=nf_all,
        seq_len=seq_len,
        num_pred=num_pred,
    )
    return rows, metadata


def load_multimodal_nf_index(path: str | Path, split: str | None = None) -> SampleIndex:
    index = SampleIndex.from_csv(path, storage_kind="hdf5_frame")
    if split is not None:
        return index.for_split(split)
    return index


def _sequence_window_items(frame_items: list[dict[str, Any]], *, seq_len: int, num_pred: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in frame_items:
        key = (str(item["channel_file"]), str(item["city"]), str(item["trajectory"]))
        grouped.setdefault(key, []).append(item)
    windows: list[dict[str, Any]] = []
    for _, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: (_frame_sort_key(item["frame"]), int(item["channel_index"])))
        if len(ordered) < seq_len + num_pred:
            continue
        for start in range(0, len(ordered) - seq_len - num_pred + 1):
            history = ordered[start : start + seq_len]
            future = ordered[start + seq_len : start + seq_len + num_pred]
            last_history = history[-1]
            windows.append(
                {
                    **last_history,
                    "history_indices": [int(item["channel_index"]) for item in history],
                    "history_global_indices": [int(item["global_index"]) for item in history],
                    "history_frames": [str(item["frame"]) for item in history],
                    "target_indices": [int(item["channel_index"]) for item in future],
                    "target_global_indices": [int(item["global_index"]) for item in future],
                    "target_frames": [str(item["frame"]) for item in future],
                    "target_beam_top1_triplets": [item["beam_top1_triplet"] for item in future],
                    "target_beam_power_top1": [float(item["beam_power_top1"]) for item in future],
                    "target_los_labels": [int(item["los_label"]) for item in future],
                    "target_nf_labels": [int(item["nf_label"]) for item in future],
                }
            )
    return windows


def _frame_sort_key(value: Any) -> tuple[int, str]:
    try:
        return int(value), str(value)
    except (TypeError, ValueError):
        return 0, str(value)


def _index_metadata(
    rows: list[SampleRow],
    *,
    split_mode: str,
    split_ratios: Any,
    split_seed: int,
    data_root: Path,
    channel_paths: list[Path],
    beam_idx: np.ndarray,
    los: np.ndarray,
    nf: np.ndarray,
    seq_len: int,
    num_pred: int,
) -> dict[str, Any]:
    by_split: dict[str, list[SampleRow]] = {}
    for row in rows:
        by_split.setdefault(_normalize_split(row.split), []).append(row)
    metadata = {
        "dataset": MULTIMODAL_NF_DATASET_TYPE,
        "family": MULTIMODAL_NF_FAMILY,
        "data_root": str(data_root),
        "split_protocol": "city_level" if str(split_mode).lower() == "city" else "frame_level_debug",
        "task_semantics": "current_frame_near_field_codebook_beam_selection",
        "legacy_task_semantics": "future_near_field_beam_prediction",
        "target_schema": "near_field_3d_codebook_flattened_beam_class",
        "target_schema_aliases": ["near_field_beam_selection"],
        "seq_len": int(seq_len),
        "num_pred": int(num_pred),
        "pred_horizon": int(num_pred),
        "split_mode": split_mode,
        "split_seed": int(split_seed),
        "split_ratios": [float(value) for value in split_ratios],
        "sample_count": int(len(rows)),
        "fingerprint": _fingerprint([row.sample_id for row in rows]),
        "channel_paths": [str(path) for path in channel_paths],
        "channel_fingerprints": {str(path): fingerprint_path(path) for path in channel_paths},
        "splits": {},
    }
    if channel_paths:
        metadata["channel_path"] = str(channel_paths[0])
        metadata["channel_fingerprint"] = fingerprint_path(channel_paths[0])
    for split, split_rows in sorted(by_split.items()):
        indices = [int(row.resource_refs.get("global_index", row.resource_refs["channel_index"])) for row in split_rows]
        metadata["splits"][split] = {
            "num_samples": len(split_rows),
            "cities": sorted({str(row.scene_or_city) for row in split_rows}),
            "beam_label_distribution": _beam_distribution(beam_idx[indices]),
            "los_distribution": _value_counts(np.asarray(los)[indices]),
            "nf_distribution": _value_counts(np.asarray(nf)[indices]),
        }
    return metadata


def _resolve_prediction_horizon(*, num_pred: int | None, pred_horizon: int | None) -> int:
    if pred_horizon is not None:
        return int(pred_horizon)
    if num_pred is not None:
        return int(num_pred)
    return 3


def _index_prefix(*, seq_len: int, num_pred: int) -> str:
    return f"multimodal_nf_seq{int(seq_len)}_pred{int(num_pred)}"


def _beam_distribution(beam_idx: np.ndarray) -> dict[str, int]:
    if beam_idx.size == 0:
        return {}
    top1 = np.asarray(beam_idx)[:, 0, :].reshape(-1, 3)
    counts: dict[str, int] = {}
    for triplet in top1:
        key = ",".join(str(int(value)) for value in triplet)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _value_counts(values: np.ndarray) -> dict[str, int]:
    flat = np.asarray(values).reshape(-1)
    counts: dict[str, int] = {}
    for value in flat:
        key = str(int(value)) if np.issubdtype(np.asarray(value).dtype, np.number) else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


__all__ = ["build_multimodal_nf_index", "build_multimodal_nf_rows", "load_multimodal_nf_index"]
