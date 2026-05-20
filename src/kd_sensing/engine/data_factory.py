from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from kd_sensing.config.lidar_normalization import canonicalize_lidar_dataset_config
from kd_sensing.data.scenes import normalize_deepsense_dataset_config
from kd_sensing.engine.cache_policy import apply_cache_policy
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.modalities import dataset_flags_for_modalities
from kd_sensing.registries import DATASETS, import_default_components


SNAPSHOT_TRAIN_CSV = "train_seqs_SNAPSHOT_NEXT_FRAME.csv"
SNAPSHOT_VAL_CSV = "val_seqs_SNAPSHOT_NEXT_FRAME.csv"


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
    canonicalize_lidar_dataset_config(dataset_cfg)
    _apply_csi_degradation_seed(dataset_cfg, cfg)
    if dataset_type not in {"synthetic", "synthetic_sequence"}:
        csv_name, dataset_split = _dataset_csv_for_split(dataset_cfg, split)
        dataset_cfg["csv_name"] = csv_name
        dataset_cfg["split"] = dataset_split
        _validate_snapshot_csv_exists(cfg, dataset_cfg, csv_name)
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
    if getattr(train_dataset, "use_csi", False):
        dataset_kwargs["csi_rms_normalizer"] = getattr(train_dataset, "csi_rms_normalizer", None)
    if getattr(train_dataset, "occlusion_target_enabled", False):
        dataset_kwargs["occlusion_target_stats"] = getattr(train_dataset, "occlusion_target_stats", None)
    if getattr(train_dataset, "position_target_enabled", False):
        dataset_kwargs["position_target_scaler"] = getattr(train_dataset, "position_target_scaler", None)
    test_dataset = build_dataset(cfg, "test", **dataset_kwargs)
    return {
        "train": build_dataloader(train_dataset, loader_cfg, split="train"),
        "test": build_dataloader(test_dataset, loader_cfg, split="test"),
    }


def build_dataloader(dataset: Any, loader_cfg: dict[str, Any], *, split: str) -> DataLoader:
    return DataLoader(dataset, **build_dataloader_kwargs(loader_cfg, split=split))


def build_dataloader_kwargs(loader_cfg: dict[str, Any], *, split: str) -> dict[str, Any]:
    settings = resolve_dataloader_split_config(loader_cfg, split=split)
    kwargs: dict[str, Any] = {
        "batch_size": settings["batch_size"],
        "shuffle": settings["shuffle"],
        "num_workers": settings["num_workers"],
        "pin_memory": settings["pin_memory"],
        "drop_last": settings["drop_last"],
    }
    if settings["num_workers"] > 0:
        kwargs["persistent_workers"] = settings["persistent_workers"]
        if settings["prefetch_factor"] is not None:
            kwargs["prefetch_factor"] = settings["prefetch_factor"]
    return kwargs


def resolve_dataloader_split_config(loader_cfg: dict[str, Any], *, split: str) -> dict[str, Any]:
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported DataLoader split '{split}'.")
    num_workers = int(_split_loader_value(loader_cfg, split, "num_workers", 0) or 0)
    prefetch_factor = _split_loader_value(loader_cfg, split, "prefetch_factor", None)
    return {
        "batch_size": int(_split_loader_value(loader_cfg, split, "batch_size", 3) or 3),
        "shuffle": split == "train",
        "num_workers": num_workers,
        "pin_memory": bool(_split_loader_value(loader_cfg, split, "pin_memory", False)),
        "drop_last": bool(_split_loader_value(loader_cfg, split, "drop_last", False)),
        "persistent_workers": bool(_split_loader_value(loader_cfg, split, "persistent_workers", False)),
        "prefetch_factor": int(prefetch_factor) if prefetch_factor is not None else None,
    }


def _split_loader_value(loader_cfg: dict[str, Any], split: str, key: str, default: Any) -> Any:
    split_cfg = loader_cfg.get(split)
    if isinstance(split_cfg, dict) and key in split_cfg:
        return split_cfg[key]
    prefixed_key = f"{split}_{key}"
    if prefixed_key in loader_cfg:
        return loader_cfg[prefixed_key]
    return loader_cfg.get(key, default)


def shutdown_dataloader_workers(dataloader: DataLoader) -> None:
    iterator = getattr(dataloader, "_iterator", None)
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    try:
        dataloader._iterator = None  # type: ignore[attr-defined]
    except Exception:
        pass


def prepare_lidar_normalizer(cfg: dict[str, Any], dataset: Any) -> None:
    if not getattr(dataset, "needs_lidar_streaming_stats", False):
        return
    progress_enabled = cfg.get("output", {}).get("progress", {}).get("enabled", True)
    dataset.fit_lidar_normalizer_streaming(progress_enabled=progress_enabled)


def _dataset_csv_for_split(dataset_cfg: dict[str, Any], split: str) -> tuple[str | None, str]:
    if split == "train":
        return dataset_cfg.get("train_csv_name"), "train"
    val_csv = dataset_cfg.get("val_csv_name")
    if val_csv:
        return val_csv, "validation"
    return dataset_cfg.get("test_csv_name"), split


def _validate_snapshot_csv_exists(cfg: dict[str, Any], dataset_cfg: dict[str, Any], csv_name: str | None) -> None:
    if cfg.get("experiment", {}).get("variant") != "snapshot_next_frame":
        return
    if not csv_name:
        raise FileNotFoundError(
            "Snapshot next-frame baseline requires snapshot CSVs. "
            "Run: python scripts/preprocess.py --config configs/preprocess/sequences_snapshot_next_frame.yaml"
        )
    data_root = dataset_cfg.get("data_root")
    csv_path = Path(str(csv_name))
    if not csv_path.is_absolute():
        csv_path = Path(str(data_root or ".")) / csv_path
    if not csv_path.exists():
        expected = {SNAPSHOT_TRAIN_CSV, SNAPSHOT_VAL_CSV}
        raise FileNotFoundError(
            f"Snapshot next-frame baseline expected {csv_path}. "
            f"Generate {sorted(expected)} first with: "
            "python scripts/preprocess.py --config configs/preprocess/sequences_snapshot_next_frame.yaml"
        )


def _apply_csi_degradation_seed(dataset_cfg: dict[str, Any], cfg: dict[str, Any]) -> None:
    degradation = dataset_cfg.get("csi_degradation")
    if not isinstance(degradation, dict) or "seed" in degradation:
        return
    experiment_seed = cfg.get("experiment", {}).get("seed")
    if experiment_seed is not None:
        degradation["seed"] = int(experiment_seed)


__all__ = [
    "build_dataloader",
    "build_dataloader_kwargs",
    "build_dataloaders",
    "build_dataset",
    "prepare_lidar_normalizer",
    "resolve_dataloader_split_config",
    "shutdown_dataloader_workers",
]
