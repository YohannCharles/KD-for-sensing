from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kd_sensing.preprocessing.raymobtime_s008_common import (
    INDEX_COLUMNS,
    OUTPUT_SPLIT_TO_INDEX_NAME,
    SOURCE_SPLIT_ORDER,
    _fingerprint,
    _load_split_arrays,
    _read_coord_csv,
    _sample_id,
    _split_index_files,
    _value_counts,
    resolve_raymobtime_paths,
)

def build_s008_index(
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    split_seed: int = 42,
    split_ratios: list[float] | tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, Any]:
    paths = resolve_raymobtime_paths(data_root=data_root, output_dir=output_dir, cache_dir=cache_dir)
    frame = _read_coord_csv(paths.csv_path)
    valid = frame.loc[frame["Val"].astype(str).str.upper() == "V"].copy()
    valid = valid.reset_index(names="source_row_index")
    valid["valid_index"] = np.arange(len(valid), dtype=np.int64)
    valid["sample_id"] = [
        _sample_id(row.EpisodeID, row.SceneID, row.VehicleArrayID)
        for row in valid.itertuples(index=False)
    ]
    detected_splits = _detect_source_split_lengths(paths.data_root, total_count=len(valid))
    split_aliases: dict[str, str] = {}
    if detected_splits:
        split_names, source_split_names, source_split_indices = _assign_detected_source_splits(
            len(valid),
            detected_splits["lengths"],
        )
        split_protocol = "raymobtime_s008_official_baseline_split"
        split_source = detected_splits["source"]
        if "validation" in detected_splits["lengths"] and "test" not in detected_splits["lengths"]:
            split_aliases["test"] = "validation"
    else:
        split_names = _assign_splits(len(valid), split_ratios=split_ratios, seed=split_seed)
        source_split_names = np.full(len(valid), "all", dtype=object)
        source_split_indices = valid["valid_index"].to_numpy(dtype=np.int64)
        split_protocol = "raymobtime_s008_snapshot_random"
        split_source = None
    valid["split"] = split_names
    valid["source_split"] = source_split_names
    valid["source_split_index"] = source_split_indices
    valid["coord_row_in_split"] = valid.groupby("split").cumcount()
    index = valid.loc[:, list(INDEX_COLUMNS)].copy()
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    index.to_csv(paths.cache_dir / "index_all_valid.csv", index=False)
    for split, split_name in OUTPUT_SPLIT_TO_INDEX_NAME.items():
        split_frame = _index_frame_for_output_split(index, split, aliases=split_aliases)
        path = paths.cache_dir / f"index_{split_name}.csv"
        split_frame.to_csv(path, index=False)
        output_paths[f"index_{split_name}"] = str(path)
    metadata = _split_metadata(
        index,
        split_seed=split_seed,
        split_ratios=split_ratios,
        data_root=paths.data_root,
        split_protocol=split_protocol,
        split_source=split_source,
        split_aliases=split_aliases,
    )
    metadata_path = paths.cache_dir / "split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    output_paths["index_all_valid"] = str(paths.cache_dir / "index_all_valid.csv")
    output_paths["split_metadata"] = str(metadata_path)
    return output_paths



def _assign_splits(
    count: int,
    *,
    split_ratios: list[float] | tuple[float, float, float],
    seed: int,
) -> np.ndarray:
    ratios = np.asarray(split_ratios, dtype=np.float64)
    if ratios.shape != (3,) or np.any(ratios < 0) or ratios.sum() <= 0:
        raise ValueError("split_ratios must contain three non-negative values for train/validation/test.")
    ratios = ratios / ratios.sum()
    indices = np.arange(count)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    train_end = int(math.floor(count * ratios[0]))
    val_end = train_end + int(math.floor(count * ratios[1]))
    if count >= 3:
        train_end = max(1, min(train_end, count - 2))
        val_end = max(train_end + 1, min(val_end, count - 1))
    names = np.full(count, "test", dtype=object)
    names[indices[:train_end]] = "train"
    names[indices[train_end:val_end]] = "validation"
    return names


def _detect_source_split_lengths(data_root: Path, *, total_count: int) -> dict[str, Any] | None:
    for rel in (
        "baseline_data/coord_input",
        "baseline_data/beam_output",
        "baseline_data/image_v2_input",
        "baseline_data/lidar_input",
    ):
        path = data_root / rel
        if not path.exists():
            continue
        arrays = _load_split_arrays(path, key=None, allow_missing=True)
        lengths = {split: int(len(array)) for split, array in arrays.items() if split != "all"}
        if lengths and sum(lengths.values()) == int(total_count):
            return {"source": rel, "lengths": lengths}
    return None


def _assign_detected_source_splits(
    count: int,
    split_lengths: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_names = np.empty(count, dtype=object)
    source_split_names = np.empty(count, dtype=object)
    source_split_indices = np.empty(count, dtype=np.int64)
    cursor = 0
    for source_split in SOURCE_SPLIT_ORDER:
        length = int(split_lengths.get(source_split, 0))
        if length <= 0:
            continue
        end = cursor + length
        if end > count:
            raise ValueError(
                f"Detected Raymobtime source split lengths exceed valid receiver count: {split_lengths} > {count}."
            )
        split_names[cursor:end] = source_split
        source_split_names[cursor:end] = source_split
        source_split_indices[cursor:end] = np.arange(length, dtype=np.int64)
        cursor = end
    if cursor != count:
        raise ValueError(
            f"Detected Raymobtime source split lengths cover {cursor} rows but valid receiver count is {count}."
        )
    return split_names, source_split_names, source_split_indices


def _index_frame_for_output_split(index: pd.DataFrame, split: str, *, aliases: dict[str, str]) -> pd.DataFrame:
    source = aliases.get(split, split)
    frame = index.loc[index["split"] == source].copy()
    if source != split:
        frame["split"] = split
        frame["coord_row_in_split"] = np.arange(len(frame), dtype=np.int64)
    return frame


def _split_metadata(
    index: pd.DataFrame,
    *,
    split_seed: int,
    split_ratios: Any,
    data_root: Path,
    split_protocol: str,
    split_source: str | None,
    split_aliases: dict[str, str],
) -> dict[str, Any]:
    metadata = {
        "dataset": "raymobtime_s008",
        "task_semantics": "current_snapshot_beam_selection",
        "data_root": str(data_root),
        "split_seed": int(split_seed),
        "split_ratios": [float(value) for value in split_ratios],
        "split_protocol": split_protocol,
        "sample_count": int(len(index)),
        "fingerprint": _fingerprint(index["sample_id"].astype(str).tolist()),
        "splits": {},
    }
    if split_source is not None:
        metadata["split_source"] = split_source
    if split_aliases:
        metadata["split_aliases"] = split_aliases
    for split, frame in index.groupby("split", sort=True):
        metadata["splits"][str(split)] = {
            "num_samples": int(len(frame)),
            "beam_label_distribution": {},
            "los_distribution": _value_counts(frame["LOS"]),
        }
    return metadata



__all__ = ["build_s008_index"]
