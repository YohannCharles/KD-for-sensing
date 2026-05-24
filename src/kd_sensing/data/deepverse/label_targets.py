from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kd_sensing.data.deepverse.label_constants import (
    BLOCKAGE_IGNORE_INDEX,
    LINK_STATE_LOS,
    LINK_STATE_NLOS,
    LINK_STATE_UNKNOWN,
)
from kd_sensing.data.deepverse.label_scene import _field, _path_count

def los_to_link_state(los_status: int) -> int:
    status = int(los_status)
    if status == 1:
        return LINK_STATE_LOS
    if status == 0:
        return LINK_STATE_NLOS
    return LINK_STATE_UNKNOWN


def los_to_blockage(los_status: int) -> int:
    link_state = los_to_link_state(los_status)
    if link_state == LINK_STATE_LOS:
        return 0
    if link_state == LINK_STATE_NLOS:
        return 1
    return BLOCKAGE_IGNORE_INDEX


def _link_los_from_path_statuses(path_statuses: Any) -> int:
    statuses = np.asarray(path_statuses).reshape(-1)
    finite_statuses = statuses[np.isfinite(statuses.astype(np.float64, copy=False))]
    if finite_statuses.size == 0:
        return LINK_STATE_UNKNOWN
    if np.any(np.isclose(finite_statuses, 1.0)):
        return 1
    if np.any(np.isclose(finite_statuses, 0.0)):
        return 0
    return LINK_STATE_UNKNOWN


def _parse_ue_file_range(filename: str) -> tuple[int, int] | None:
    stem = Path(filename).stem
    if "_UE_" not in stem:
        return None
    _, range_part = stem.rsplit("_UE_", 1)
    if "-" not in range_part:
        return None
    start_text, end_text = range_part.split("-", 1)
    try:
        return int(start_text), int(end_text)
    except ValueError:
        return None


RADAR_FEATURE_NAMES = [
    "abs_mean",
    "abs_std",
    "abs_max",
    "phase_diff_mean",
    "phase_diff_std",
    "path_count",
]
RADAR_FEATURE_SIZE = len(RADAR_FEATURE_NAMES)

def extract_radar_feature(radar_sample: Any) -> np.ndarray:
    coeffs = _field(radar_sample, "coeffs", "channel", "channels", "H", "h", "tensor", "data")
    if coeffs is None:
        raise KeyError("radar sample does not contain coefficients or tensor data.")
    array = np.asarray(coeffs)
    if array.size == 0:
        raise ValueError("radar coefficients are empty.")
    magnitude = np.abs(array.astype(np.complex64, copy=False)).astype(np.float32)
    phase = np.unwrap(np.angle(array.reshape(-1).astype(np.complex64, copy=False)))
    phase_diff = np.diff(phase).astype(np.float32)
    if phase_diff.size == 0:
        phase_diff = np.zeros(1, dtype=np.float32)
    features = np.asarray(
        [
            float(np.mean(magnitude)),
            float(np.std(magnitude)),
            float(np.max(magnitude)),
            float(np.mean(phase_diff)),
            float(np.std(phase_diff)),
            float(_path_count(_field(radar_sample, "paths", "ray_paths", "rays"))),
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(features)):
        raise ValueError("radar feature contains NaN or Inf.")
    return features

def _infer_num_ant(channel: np.ndarray) -> int:
    if channel.ndim == 0:
        raise ValueError("channel coefficients are scalar.")
    if channel.ndim == 1:
        return int(channel.shape[0])
    if channel.ndim == 2:
        return int(channel.shape[0])
    first = int(channel.shape[0])
    second = int(channel.shape[1])
    if first == 1 and second > 1:
        return second
    return first

def _filter_built_by_sample_ids(built: dict[str, Any], keep_ids: set[str]) -> dict[str, Any]:
    labels = built["labels"]
    mask = np.asarray([str(sample_id) in keep_ids for sample_id in labels["sample_id"]], dtype=bool)
    filtered = dict(built)
    filtered["labels"] = _filter_array_dict(labels, mask)
    for key in ("weak_wireless", "radar_features", "noisy_position"):
        filtered[key] = _filter_array_dict(built[key], mask)
    return filtered


def _filter_array_dict(payload: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    filtered: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        array = np.asarray(value)
        if array.shape[:1] == mask.shape:
            filtered[key] = array[mask]
        else:
            filtered[key] = array
    return filtered


def _blockage_metadata(labels: dict[str, np.ndarray], *, min_class_count: int, min_class_ratio: float) -> dict[str, Any]:
    raw_los = labels.get("los_status_future", np.asarray([], dtype=np.int16))
    blockage = labels.get("blockage_labels_future", np.asarray([], dtype=np.int64))
    valid_mask = labels.get("blockage_valid_mask", np.asarray([], dtype=bool)).astype(bool)
    valid_labels = blockage[valid_mask] if blockage.shape == valid_mask.shape else np.asarray([], dtype=np.int64)
    distribution = _counter_dict(valid_labels)
    raw_distribution = _counter_dict(raw_los)
    total = int(valid_labels.size)
    class_counts = {label: int(np.sum(valid_labels == label)) for label in (0, 1)}
    present_classes = {label for label, count in class_counts.items() if count > 0}
    minority_count = min(class_counts.values()) if total else 0
    minority_ratio = (minority_count / total) if total else 0.0

    reason = ""
    usable = True
    if total == 0:
        usable = False
        reason = "no_valid_blockage_labels"
    elif present_classes != {0, 1}:
        usable = False
        missing = sorted({0, 1} - present_classes)
        reason = f"missing_classes:{','.join(str(value) for value in missing)}"
    elif minority_count < min_class_count:
        usable = False
        reason = "minority_class_count_below_min"
    elif minority_ratio < min_class_ratio:
        usable = False
        reason = "minority_class_ratio_below_min"

    return {
        "usable": usable,
        "reason": reason,
        "ignore_index": BLOCKAGE_IGNORE_INDEX,
        "min_class_count": int(min_class_count),
        "min_class_ratio": float(min_class_ratio),
        "raw_los_status_distribution": raw_distribution,
        "valid_label_distribution": distribution,
        "valid_label_count": total,
        "minority_class_count": int(minority_count),
        "minority_class_ratio": float(minority_ratio),
    }


def _counter_dict(values: np.ndarray) -> dict[str, int]:
    flat = np.asarray(values).reshape(-1)
    return {str(k): int(v) for k, v in Counter(flat.tolist()).items()}


def _stack_label_arrays(labels: dict[str, list[Any]], pred_horizon: int, num_beams: int) -> dict[str, np.ndarray]:
    sample_ids = np.asarray(labels["sample_id"], dtype=str)
    count = len(sample_ids)
    return {
        "sample_id": sample_ids,
        "beam_label": np.asarray(labels["beam_label"], dtype=np.int64),
        "beam_labels_future": _stack_or_empty(labels["beam_labels_future"], (count, pred_horizon), np.int64),
        "blockage_label": np.asarray(labels["blockage_label"], dtype=np.int64),
        "blockage_labels_future": _stack_or_empty(labels["blockage_labels_future"], (count, pred_horizon), np.int64),
        "blockage_valid_mask": _stack_or_empty(labels["blockage_valid_mask"], (count, pred_horizon), bool),
        "trajectory_future": _stack_or_empty(labels["trajectory_future"], (count, pred_horizon, 2), np.float32),
        "los_status_future": _stack_or_empty(labels["los_status_future"], (count, pred_horizon), np.int16),
        "link_state_future": _stack_or_empty(labels["link_state_future"], (count, pred_horizon), np.int16),
        "beam_gain_future": _stack_or_empty(labels["beam_gain_future"], (count, pred_horizon, num_beams), np.float32),
        "valid_mask": _stack_or_empty(labels["valid_mask"], (count, pred_horizon), bool),
    }


def _stack_or_empty(values: Sequence[Any], shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    if values:
        return np.stack(values).astype(dtype)
    return np.empty(shape, dtype=dtype)

def _los_status_source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        payload = row.get("los_status_source_future", "[]")
        try:
            values = json.loads(payload) if isinstance(payload, str) else payload
        except json.JSONDecodeError:
            values = []
        if isinstance(values, list):
            counts.update(str(value) for value in values)
    return dict(counts)

__all__ = [
    "RADAR_FEATURE_NAMES",
    "RADAR_FEATURE_SIZE",
    "extract_radar_feature",
    "los_to_blockage",
    "los_to_link_state",
    "_blockage_metadata",
    "_filter_built_by_sample_ids",
    "_infer_num_ant",
    "_link_los_from_path_statuses",
    "_los_status_source_counts",
    "_parse_ue_file_range",
    "_stack_label_arrays",
    "_stack_or_empty",
]
