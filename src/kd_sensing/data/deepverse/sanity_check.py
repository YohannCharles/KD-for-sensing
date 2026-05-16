from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


def build_sanity_report(
    *,
    rows: list[dict[str, Any]],
    labels: dict[str, np.ndarray],
    split: dict[str, list[str]],
    skip_counts: dict[str, int],
    artifact_paths: dict[str, str],
    radar_features: np.ndarray | None = None,
    split_metadata: dict[str, Any] | None = None,
    blockage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    split_metadata = split_metadata or {}
    blockage = blockage or {}
    beam = labels.get("beam_label", np.asarray([], dtype=np.int64))
    blockage_label = labels.get("blockage_label", np.asarray([], dtype=np.int64))
    blockage_future = labels.get("blockage_labels_future", np.asarray([], dtype=np.int64))
    blockage_valid_mask = labels.get("blockage_valid_mask", np.asarray([], dtype=bool)).astype(bool)
    los_status = labels.get("los_status_future", np.asarray([], dtype=np.int16))
    link_state = labels.get("link_state_future", np.asarray([], dtype=np.int16))
    trajectory = labels.get("trajectory_future", np.asarray([], dtype=np.float32))
    beam_gain = labels.get("beam_gain_future", np.asarray([], dtype=np.float32))
    radar = np.asarray(radar_features) if radar_features is not None else np.asarray([], dtype=np.float32)
    overlap = _raw_frame_overlap(rows, split)

    valid_blockage = (
        blockage_future[blockage_valid_mask]
        if blockage_future.shape == blockage_valid_mask.shape
        else np.asarray([], dtype=np.int64)
    )

    return {
        "sample_count": len(rows),
        "split_counts": {name: len(sample_ids) for name, sample_ids in split.items()},
        "skip_counts": dict(skip_counts),
        "label_distribution": {
            "beam": _counter_dict(beam),
            "blockage": _counter_dict(blockage_label),
            "blockage_valid": _counter_dict(valid_blockage),
            "raw_los_status": _counter_dict(los_status),
            "link_state": _counter_dict(link_state),
        },
        "blockage": blockage,
        "split_protocol": split_metadata,
        "group_counts": split_metadata.get("group_counts_by_split", {}),
        "embargo_span": int(split_metadata.get("embargo_span", 0)),
        "discarded_boundary_windows": int(split_metadata.get("discarded_boundary_windows", 0)),
        "raw_frame_overlap": overlap,
        "missing_modalities": {
            "camera": sum(1 for row in rows if not row.get("camera_paths")),
            "lidar": sum(1 for row in rows if not row.get("lidar_paths")),
            "radar": sum(1 for row in rows if not row.get("radar_feature_history")),
        },
        "checks": {
            "trajectory_has_nan_or_inf": bool(trajectory.size and not np.all(np.isfinite(trajectory))),
            "beam_gain_has_nan_or_inf": bool(beam_gain.size and not np.all(np.isfinite(beam_gain))),
            "radar_feature_has_nan_or_inf": bool(radar.size and not np.all(np.isfinite(radar))),
            "split_covers_all_samples": _split_covers_rows(rows, split),
            "cross_split_raw_frame_overlap_is_zero": overlap["total_overlap_count"] == 0,
        },
        "artifact_paths": {key: str(Path(value)) for key, value in artifact_paths.items()},
    }


def _counter_dict(values: np.ndarray) -> dict[str, int]:
    flat = np.asarray(values).reshape(-1)
    return {str(k): int(v) for k, v in Counter(flat.tolist()).items()}


def _split_covers_rows(rows: list[dict[str, Any]], split: dict[str, list[str]]) -> bool:
    row_ids = {str(row["sample_id"]) for row in rows}
    split_ids = {str(sample_id) for sample_ids in split.values() for sample_id in sample_ids}
    return row_ids == split_ids


def _raw_frame_overlap(rows: list[dict[str, Any]], split: dict[str, list[str]]) -> dict[str, Any]:
    rows_by_id = {str(row["sample_id"]): row for row in rows}
    frames_by_split: dict[str, set[str]] = {}
    for split_name, sample_ids in split.items():
        frames: set[str] = set()
        for sample_id in sample_ids:
            row = rows_by_id.get(str(sample_id))
            if row is None:
                continue
            frames.update(_row_raw_frame_keys(row))
        frames_by_split[split_name] = frames

    pairs: dict[str, dict[str, Any]] = {}
    total = 0
    for left, right in combinations(split.keys(), 2):
        overlap = sorted(frames_by_split[left] & frames_by_split[right])
        total += len(overlap)
        pairs[f"{left}_vs_{right}"] = {
            "count": len(overlap),
            "examples": overlap[:10],
        }
    return {"total_overlap_count": total, "pairs": pairs}


def _row_raw_frame_keys(row: dict[str, Any]) -> list[str]:
    namespace = str(row.get("raw_frame_group_key") or row.get("split_group_key") or row.get("sample_id"))
    frames: list[str] = []
    for column in ("history_indices", "future_indices"):
        payload = row.get(column, "[]")
        if isinstance(payload, str):
            try:
                values = json.loads(payload)
            except json.JSONDecodeError:
                values = []
        else:
            values = payload
        if isinstance(values, list):
            frames.extend(f"{namespace}|t={value}" for value in values)
    return frames
