from __future__ import annotations

from copy import deepcopy
from typing import Any

from torch.utils.data import DataLoader

from kd_sensing.data.scenes import normalize_deepsense_dataset_config
from kd_sensing.engine.cache_policy import apply_cache_policy
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.modalities import dataset_flags_for_modalities
from kd_sensing.registries import DATASETS, import_default_components


def build_dataset(cfg: dict[str, Any], split: str, **extra_dataset_kwargs: Any):
    import_default_components()
    dataset_cfg = deepcopy(cfg["data"]["dataset"])
    normalize_deepsense_dataset_config(dataset_cfg)
    dataset_type = dataset_cfg.get("type")
    dataset_cfg["split"] = split
    enabled_modalities = resolve_enabled_modalities(cfg)
    dataset_cfg["enabled_modalities"] = list(enabled_modalities)
    dataset_cfg.update(dataset_flags_for_modalities(enabled_modalities))
    apply_cache_policy(dataset_cfg, cfg, enabled_modalities)
    if dataset_type not in {"synthetic", "synthetic_sequence"}:
        csv_key = "train_csv_name" if split == "train" else "test_csv_name"
        dataset_cfg["csv_name"] = dataset_cfg.get(csv_key)
    dataset_cfg.update(extra_dataset_kwargs)
    return DATASETS.build(dataset_cfg)


def build_dataloaders(cfg: dict[str, Any]) -> dict[str, DataLoader]:
    loader_cfg = cfg["data"]["dataloader"]
    train_dataset = build_dataset(cfg, "train")
    prepare_lidar_normalizer(cfg, train_dataset)
    dataset_kwargs = {}
    if getattr(train_dataset, "use_gps", False):
        dataset_kwargs["gps_scaler"] = getattr(train_dataset, "gps_scaler", None)
    if getattr(train_dataset, "use_lidar", False):
        dataset_kwargs["lidar_normalizer"] = getattr(train_dataset, "lidar_normalizer", None)
    if getattr(train_dataset, "use_mmwave", False):
        dataset_kwargs["mmwave_scaler"] = getattr(train_dataset, "mmwave_scaler", None)
    test_dataset = build_dataset(cfg, "test", **dataset_kwargs)
    return {
        "train": build_dataloader(train_dataset, loader_cfg, split="train"),
        "test": build_dataloader(test_dataset, loader_cfg, split="test"),
    }


def build_dataloader(dataset: Any, loader_cfg: dict[str, Any], *, split: str) -> DataLoader:
    return DataLoader(dataset, **build_dataloader_kwargs(loader_cfg, split=split))


def build_dataloader_kwargs(loader_cfg: dict[str, Any], *, split: str) -> dict[str, Any]:
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported DataLoader split '{split}'.")
    num_workers = int(loader_cfg.get("num_workers", 0))
    batch_size_key = "train_batch_size" if split == "train" else "test_batch_size"
    drop_last_key = "train_drop_last" if split == "train" else "test_drop_last"
    kwargs: dict[str, Any] = {
        "batch_size": loader_cfg.get(batch_size_key, 3),
        "shuffle": split == "train",
        "num_workers": num_workers,
        "pin_memory": bool(loader_cfg.get("pin_memory", False)),
        "drop_last": bool(loader_cfg.get(drop_last_key, loader_cfg.get("drop_last", False if split == "train" else False))),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(loader_cfg.get("persistent_workers", False))
        prefetch_factor = loader_cfg.get("prefetch_factor")
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def prepare_lidar_normalizer(cfg: dict[str, Any], dataset: Any) -> None:
    if not getattr(dataset, "needs_lidar_streaming_stats", False):
        return
    progress_enabled = cfg.get("output", {}).get("progress", {}).get("enabled", True)
    dataset.fit_lidar_normalizer_streaming(progress_enabled=progress_enabled)


__all__ = [
    "build_dataloader",
    "build_dataloader_kwargs",
    "build_dataloaders",
    "build_dataset",
    "prepare_lidar_normalizer",
]
