from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from PIL import Image
import torch

from kd_sensing.config.io import dump_config
from kd_sensing.data.samples import _select_portion
from kd_sensing.data.scenes import retarget_deepsense_dataset_config
from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.engine.data_factory import build_dataset, prepare_lidar_normalizer
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.engine.run_metadata import dataset_run_metadata
from kd_sensing.modalities import MODALITY_ORDER, dataset_flags_for_modalities, normalize_modalities
from kd_sensing.utils.paths import resolve_path


from kd_sensing.diagnostics.visualization.config import VisualizationConfig, _json_scalar
from kd_sensing.diagnostics.visualization.datasets import selected_csv_frame_for_dataset
from kd_sensing.diagnostics.visualization.sampling import SampleCandidate, collect_candidates, filter_sample_candidates

def tensor_stats(value: Any) -> dict[str, Any]:
    array = _as_numpy(value)
    finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.number) else np.asarray([])
    stats = {
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "nonzero_fraction": None,
    }
    if array.size == 0 or finite.size == 0:
        return stats
    stats.update(
        {
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "nonzero_fraction": float(np.count_nonzero(array) / array.size),
        }
    )
    return stats

def modality_statistics(sample: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    if "image" in sample:
        image_stats = tensor_stats(sample["image"])
        image_stats["mask_density"] = image_stats["nonzero_fraction"]
        stats["image"] = image_stats
    if "radar_ra" in sample or "radar_da" in sample:
        radar: dict[str, Any] = {}
        if "radar_ra" in sample:
            radar["radar_ra"] = tensor_stats(sample["radar_ra"])
        if "radar_da" in sample:
            radar["radar_da"] = tensor_stats(sample["radar_da"])
        stats["radar"] = radar
    if "lidar" in sample:
        lidar = tensor_stats(sample["lidar"])
        lidar["channel_nonzero_fraction"] = _channel_nonzero_fraction(sample["lidar"])
        stats["lidar"] = lidar
    if "gps" in sample:
        gps = tensor_stats(sample["gps"])
        array = _as_numpy(sample["gps"])
        if array.ndim == 2 and array.size:
            gps["per_dimension_min"] = [float(value) for value in np.min(array, axis=0)]
            gps["per_dimension_max"] = [float(value) for value in np.max(array, axis=0)]
        stats["gps"] = gps
    if "mmwave" in sample:
        mmwave = tensor_stats(sample["mmwave"])
        array = _as_numpy(sample["mmwave"])
        if array.ndim == 2 and array.size:
            mmwave["per_time_mean"] = [float(value) for value in np.mean(array, axis=1)]
            mmwave["per_time_std"] = [float(value) for value in np.std(array, axis=1)]
        stats["mmwave"] = mmwave
    return stats

def build_split_stats_report(
    datasets: dict[str, Any],
    *,
    enabled_modalities: tuple[str, ...],
    viz: VisualizationConfig,
    scene_metadata: dict[str, Any],
) -> dict[str, Any]:
    split_stats: dict[str, Any] = {}
    for split, dataset in datasets.items():
        csv_frame = selected_csv_frame_for_dataset(dataset)
        candidates = collect_candidates(dataset, csv_frame)
        split_stats[split] = build_split_statistics(
            dataset,
            candidates,
            enabled_modalities=enabled_modalities,
            seq_index=viz.seq_index,
            labels=viz.labels,
        )

    report = {
        "scene": scene_metadata,
        "enabled_modalities": list(enabled_modalities),
        "filters": {
            "seq_index": list(viz.seq_index) if viz.seq_index is not None else None,
            "labels": list(viz.labels) if viz.labels is not None else None,
        },
        "splits": split_stats,
    }
    if "train" in split_stats and "test" in split_stats:
        report["train_test"] = {
            "future_label_total_variation_distance": _label_total_variation_distance(
                split_stats["train"].get("future_label_distribution", {}),
                split_stats["test"].get("future_label_distribution", {}),
            )
        }
    return report

def build_split_statistics(
    dataset: Any,
    candidates: Iterable[SampleCandidate],
    *,
    enabled_modalities: tuple[str, ...],
    seq_index: tuple[Any, ...] | None = None,
    labels: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    filtered = filter_sample_candidates(candidates, seq_index=seq_index, labels=labels)
    split_accumulator = _empty_modality_accumulator()
    seq_accumulators: dict[str, dict[str, Any]] = {}
    seq_candidates: dict[str, list[SampleCandidate]] = {}

    for candidate in filtered:
        sample = dataset[candidate.dataset_index]
        statistics = modality_statistics(sample)
        _accumulate_modality_statistics(split_accumulator, statistics)

        key = _filter_key(candidate.seq_index)
        seq_accumulators.setdefault(key, _empty_modality_accumulator())
        seq_candidates.setdefault(key, []).append(candidate)
        _accumulate_modality_statistics(seq_accumulators[key], statistics)

    by_seq_index = {}
    for key, group in seq_candidates.items():
        distribution = _candidate_label_distribution(group)
        by_seq_index[key] = {
            "seq_index": group[0].seq_index,
            "candidate_count": len(group),
            "future_label_distribution": distribution,
            "future_label_top_k": _label_top_k(distribution),
            "majority_baseline": _majority_baseline(distribution),
            "modality_statistics": _finalize_modality_accumulator(seq_accumulators[key], enabled_modalities),
        }

    distribution = _candidate_label_distribution(filtered)
    return {
        "dataset": dataset_run_metadata(dataset),
        "candidate_count": len(filtered),
        "seq_index_count": len(seq_candidates),
        "future_label_distribution": distribution,
        "future_label_top_k": _label_top_k(distribution),
        "majority_baseline": _majority_baseline(distribution),
        "modality_statistics": _finalize_modality_accumulator(split_accumulator, enabled_modalities),
        "by_seq_index": by_seq_index,
    }

def _empty_modality_accumulator() -> dict[str, Any]:
    return {
        "image_mask_density": [],
        "radar_ra_std": [],
        "radar_da_std": [],
        "lidar_nonzero_fraction": [],
        "lidar_channel_nonzero_fraction": [],
    }

def _accumulate_modality_statistics(accumulator: dict[str, Any], statistics: dict[str, Any]) -> None:
    image_stats = statistics.get("image")
    if image_stats and image_stats.get("mask_density") is not None:
        accumulator["image_mask_density"].append(float(image_stats["mask_density"]))

    radar_stats = statistics.get("radar", {})
    radar_ra = radar_stats.get("radar_ra", {})
    radar_da = radar_stats.get("radar_da", {})
    if radar_ra.get("std") is not None:
        accumulator["radar_ra_std"].append(float(radar_ra["std"]))
    if radar_da.get("std") is not None:
        accumulator["radar_da_std"].append(float(radar_da["std"]))

    lidar_stats = statistics.get("lidar")
    if lidar_stats and lidar_stats.get("nonzero_fraction") is not None:
        accumulator["lidar_nonzero_fraction"].append(float(lidar_stats["nonzero_fraction"]))
    if lidar_stats and lidar_stats.get("channel_nonzero_fraction"):
        accumulator["lidar_channel_nonzero_fraction"].append(
            [float(value) for value in lidar_stats["channel_nonzero_fraction"]]
        )

def _finalize_modality_accumulator(
    accumulator: dict[str, Any],
    enabled_modalities: tuple[str, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "image" in enabled_modalities:
        result["image"] = {
            "mask_density": _numeric_summary(accumulator["image_mask_density"]),
        }
    if "radar" in enabled_modalities:
        result["radar"] = {
            "radar_ra_std": _numeric_summary(accumulator["radar_ra_std"]),
            "radar_da_std": _numeric_summary(accumulator["radar_da_std"]),
        }
    if "lidar" in enabled_modalities:
        result["lidar"] = {
            "nonzero_fraction": _numeric_summary(accumulator["lidar_nonzero_fraction"]),
            "channel_nonzero_fraction_mean": _mean_vector(accumulator["lidar_channel_nonzero_fraction"]),
        }
    return result

def _numeric_summary(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }

def _mean_vector(values: list[list[float]]) -> list[float]:
    if not values:
        return []
    max_len = max(len(item) for item in values)
    means = []
    for idx in range(max_len):
        column = [item[idx] for item in values if idx < len(item) and np.isfinite(item[idx])]
        means.append(float(np.mean(column)) if column else 0.0)
    return means

def _candidate_label_distribution(candidates: Iterable[SampleCandidate]) -> dict[str, int]:
    counts: dict[int, int] = {}
    for candidate in candidates:
        if candidate.future_label is None:
            continue
        label = int(candidate.future_label)
        counts[label] = counts.get(label, 0) + 1
    return {str(label): counts[label] for label in sorted(counts)}

def _label_top_k(distribution: dict[str, int], *, k: int = 5) -> list[dict[str, Any]]:
    total = sum(int(count) for count in distribution.values())
    if total == 0:
        return []
    ranked = sorted(distribution.items(), key=lambda item: (-int(item[1]), int(item[0])))
    return [
        {
            "label": int(label),
            "count": int(count),
            "fraction": float(int(count) / total),
        }
        for label, count in ranked[:k]
    ]

def _majority_baseline(distribution: dict[str, int]) -> float | None:
    total = sum(int(count) for count in distribution.values())
    if total == 0:
        return None
    return float(max(int(count) for count in distribution.values()) / total)

def _label_total_variation_distance(left: dict[str, int], right: dict[str, int]) -> float | None:
    left_total = sum(int(count) for count in left.values())
    right_total = sum(int(count) for count in right.values())
    if left_total == 0 or right_total == 0:
        return None
    labels = set(left) | set(right)
    distance = 0.0
    for label in labels:
        left_prob = int(left.get(label, 0)) / left_total
        right_prob = int(right.get(label, 0)) / right_total
        distance += abs(left_prob - right_prob)
    return float(0.5 * distance)

def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)

def _channel_nonzero_fraction(value: Any) -> list[float]:
    array = _as_numpy(value)
    if array.ndim == 4:
        by_channel = np.moveaxis(array, 1, 0)
        return [float(np.count_nonzero(channel) / channel.size) if channel.size else 0.0 for channel in by_channel]
    if array.ndim == 3:
        return [float(np.count_nonzero(channel) / channel.size) if channel.size else 0.0 for channel in array]
    return []

def _filter_key(value: Any) -> str:
    return str(_json_scalar(value))

__all__ = [
    'build_split_statistics',
    'build_split_stats_report',
    'modality_statistics',
    'tensor_stats',
]
