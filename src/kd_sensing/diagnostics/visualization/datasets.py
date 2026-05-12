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



def build_diagnostic_datasets(cfg: dict[str, Any], splits: tuple[str, ...]) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    dataset_kwargs: dict[str, Any] = {}
    needs_train = "train" in splits or _needs_train_fit_for_requested_splits(cfg, splits)
    train_dataset = None
    if needs_train:
        try:
            train_dataset = build_dataset(cfg, "train")
            prepare_lidar_normalizer(cfg, train_dataset)
        except Exception as exc:
            raise RuntimeError(
                "Failed to build the train dataset needed for diagnostics. "
                "If only test split is requested, disable GPS/mmWave/LiDAR normalization or provide train data."
            ) from exc
        if getattr(train_dataset, "use_gps", False):
            dataset_kwargs["gps_scaler"] = getattr(train_dataset, "gps_scaler", None)
        if getattr(train_dataset, "use_lidar", False):
            dataset_kwargs["lidar_normalizer"] = getattr(train_dataset, "lidar_normalizer", None)
        if getattr(train_dataset, "use_mmwave", False):
            dataset_kwargs["mmwave_scaler"] = getattr(train_dataset, "mmwave_scaler", None)
        if "train" in splits:
            datasets["train"] = train_dataset

    for split in splits:
        if split == "train":
            continue
        try:
            datasets[split] = build_dataset(cfg, split, **dataset_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build the {split} dataset for diagnostics. "
                "For normalized GPS/LiDAR/mmWave features, diagnostics reuses train-fitted state when possible."
            ) from exc
    return datasets

def selected_csv_frame_for_dataset(dataset: Any) -> pd.DataFrame:
    frame = pd.read_csv(dataset.root_csv, na_values="").fillna(-99)
    metadata = getattr(getattr(dataset, "samples", None), "metadata", {}) or {}
    portion = float(metadata.get("portion", 1.0))
    strategy = str(metadata.get("portion_strategy", "even"))
    seed = int(metadata.get("portion_seed", 42))
    selected, _ = _select_portion(frame, portion=portion, strategy=strategy, seed=seed)
    if len(selected) != len(dataset):
        raise ValueError(
            f"CSV/sample alignment failed for {dataset.root_csv}: selected {len(selected)} rows, "
            f"dataset has {len(dataset)} samples."
        )
    return selected

def scene_metadata_from_datasets(datasets: dict[str, Any]) -> dict[str, Any]:
    if not datasets:
        return {}
    first = next(iter(datasets.values()))
    return {
        "scene_id": getattr(first, "scene_id", None),
        "scene_slug": getattr(first, "scene_slug", None),
    }

def _needs_train_fit_for_requested_splits(cfg: dict[str, Any], splits: tuple[str, ...]) -> bool:
    if not any(split != "train" for split in splits):
        return False
    modalities = resolve_enabled_modalities(cfg)
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if "gps" in modalities and bool(dataset_cfg.get("gps_normalize", True)):
        return True
    if "mmwave" in modalities and bool(dataset_cfg.get("mmwave_normalize", True)):
        return True
    if "lidar" in modalities and _lidar_normalization_needs_train_fit(dataset_cfg):
        return True
    return False

def _lidar_normalization_needs_train_fit(dataset_cfg: dict[str, Any]) -> bool:
    lidar_norm = dataset_cfg.get("lidar_normalization")
    if isinstance(lidar_norm, dict):
        enabled = bool(lidar_norm.get("enabled", False))
        stats_path = lidar_norm.get("stats_path")
        recompute = bool(lidar_norm.get("recompute", False))
    else:
        enabled = bool(dataset_cfg.get("lidar_normalize", False))
        stats_path = None
        recompute = False
    if not enabled:
        return False
    if stats_path and resolve_path(stats_path).exists() and not recompute:
        return False
    return True

__all__ = [
    'build_diagnostic_datasets',
    'scene_metadata_from_datasets',
    'selected_csv_frame_for_dataset',
]
