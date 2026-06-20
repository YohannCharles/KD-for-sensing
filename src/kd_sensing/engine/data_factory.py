from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Subset

from kd_sensing.config.lidar_normalization import canonicalize_lidar_dataset_config
from kd_sensing.data.dataset_descriptors import dataset_descriptor, resolve_dataset_profiles
from kd_sensing.data.scenes import (
    normalize_deepsense_dataset_config,
)
from kd_sensing.engine.cache_policy import apply_cache_policy
from kd_sensing.engine.data_factory_protocols import (
    build_protocol_split_datasets as _build_protocol_split_datasets,
    dataset_scenes_for_split,
    retarget_cfg_for_scene,
)
from kd_sensing.engine.data_factory_scalers import (
    fit_internal_validation_gps_scaler,
    first_dataset,
    harmonize_multi_scene_train_normalizers,
    normalization_kwargs,
    prepare_lidar_normalizer,
)
from kd_sensing.engine.epoch_subsampling import build_epoch_subsample_sampler
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.modalities import dataset_flags_for_modalities
from kd_sensing.registries import DATASETS, import_default_components


SNAPSHOT_TRAIN_CSV = "train_seqs_SNAPSHOT_NEXT_FRAME.csv"
SNAPSHOT_VAL_CSV = "val_seqs_SNAPSHOT_NEXT_FRAME.csv"


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
    if split not in {"train", "test", "validation", "val"}:
        raise ValueError(f"Unsupported DataLoader split '{split}'.")
    normalized_split = "validation" if split == "val" else split
    fallback_split = "test" if normalized_split == "validation" else normalized_split
    num_workers = int(_split_loader_value(loader_cfg, normalized_split, "num_workers", 0, fallback_split=fallback_split) or 0)
    prefetch_factor = _split_loader_value(loader_cfg, normalized_split, "prefetch_factor", None, fallback_split=fallback_split)
    return {
        "batch_size": int(_split_loader_value(loader_cfg, normalized_split, "batch_size", 3, fallback_split=fallback_split) or 3),
        "shuffle": normalized_split == "train",
        "num_workers": num_workers,
        "pin_memory": bool(_split_loader_value(loader_cfg, normalized_split, "pin_memory", False, fallback_split=fallback_split)),
        "drop_last": bool(_split_loader_value(loader_cfg, normalized_split, "drop_last", False, fallback_split=fallback_split)),
        "persistent_workers": bool(_split_loader_value(loader_cfg, normalized_split, "persistent_workers", False, fallback_split=fallback_split)),
        "prefetch_factor": int(prefetch_factor) if prefetch_factor is not None else None,
    }


def _split_loader_value(
    loader_cfg: dict[str, Any],
    split: str,
    key: str,
    default: Any,
    *,
    fallback_split: str | None = None,
) -> Any:
    split_cfg = loader_cfg.get(split)
    if isinstance(split_cfg, dict) and key in split_cfg:
        return split_cfg[key]
    prefixed_key = f"{split}_{key}"
    if prefixed_key in loader_cfg:
        return loader_cfg[prefixed_key]
    if fallback_split and fallback_split != split:
        fallback_cfg = loader_cfg.get(fallback_split)
        if isinstance(fallback_cfg, dict) and key in fallback_cfg:
            return fallback_cfg[key]
        fallback_prefixed_key = f"{fallback_split}_{key}"
        if fallback_prefixed_key in loader_cfg:
            return loader_cfg[fallback_prefixed_key]
    return loader_cfg.get(key, default)


def shutdown_dataloader_workers(dataloader: DataLoader) -> None:
    iterator = getattr(dataloader, "_iterator", None)
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    dataloader._iterator = None  # type: ignore[attr-defined]


def validation_from_train_enabled(cfg: dict[str, Any]) -> bool:
    validation_cfg = cfg.get("data", {}).get("validation_from_train")
    if isinstance(validation_cfg, dict):
        return bool(validation_cfg.get("enabled", False))
    return bool(validation_cfg)


def validation_from_train_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    validation_cfg = cfg.get("data", {}).get("validation_from_train")
    if isinstance(validation_cfg, dict):
        return validation_cfg
    return {"enabled": bool(validation_cfg)}


def has_validation_csv(cfg: dict[str, Any]) -> bool:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    return bool(dataset_cfg.get("val_csv_name"))


def build_train_and_internal_validation_datasets(
    cfg: dict[str, Any],
    *,
    split_dataset_builder: Callable[..., Any],
) -> tuple[Any, Any]:
    raw_cfg = deepcopy(cfg)
    raw_cfg.setdefault("data", {}).setdefault("dataset", {})["gps_normalize"] = False
    full_train = split_dataset_builder(raw_cfg, "train")
    train_dataset, validation_dataset = split_dataset_for_internal_validation(full_train, cfg)
    fit_internal_validation_gps_scaler(train_dataset, validation_dataset)
    return train_dataset, validation_dataset


def split_dataset_for_internal_validation(dataset: Any, cfg: dict[str, Any]) -> tuple[Any, Any]:
    if isinstance(dataset, ConcatDataset):
        train_parts = []
        validation_parts = []
        for offset, component in enumerate(dataset.datasets):
            train_subset, validation_subset = split_single_dataset_for_internal_validation(
                component,
                cfg,
                seed_offset=offset,
            )
            train_parts.append(train_subset)
            validation_parts.append(validation_subset)
        return ConcatDataset(train_parts), ConcatDataset(validation_parts)
    return split_single_dataset_for_internal_validation(dataset, cfg, seed_offset=0)


def split_single_dataset_for_internal_validation(dataset: Any, cfg: dict[str, Any], *, seed_offset: int) -> tuple[Subset, Subset]:
    validation_cfg = validation_from_train_cfg(cfg)
    fraction = float(validation_cfg.get("fraction", validation_cfg.get("val_fraction", 0.1)))
    if not 0.0 < fraction < 1.0:
        raise ValueError("data.validation_from_train.fraction must be between 0 and 1.")
    seed = int(validation_cfg.get("seed", cfg.get("experiment", {}).get("seed", 0))) + int(seed_offset)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset), dtype=np.int64)
    rng.shuffle(indices)
    validation_count = max(1, int(round(float(len(indices)) * fraction)))
    if validation_count >= len(indices):
        validation_count = max(1, len(indices) - 1)
    validation_indices = np.sort(indices[:validation_count]).astype(int).tolist()
    train_indices = np.sort(indices[validation_count:]).astype(int).tolist()
    if not train_indices:
        raise ValueError("Internal validation split left no training samples.")
    train_subset = Subset(dataset, train_indices)
    validation_subset = Subset(dataset, validation_indices)
    annotate_internal_subset(train_subset, role="train", source_dataset=dataset, fraction=fraction, seed=seed)
    annotate_internal_subset(validation_subset, role="validation", source_dataset=dataset, fraction=fraction, seed=seed)
    return train_subset, validation_subset


def annotate_internal_subset(subset: Subset, *, role: str, source_dataset: Any, fraction: float, seed: int) -> None:
    subset.split = role  # type: ignore[attr-defined]
    subset.internal_split = {
        "enabled": True,
        "source_split": "train",
        "role": role,
        "fraction": float(fraction),
        "seed": int(seed),
        "parent_num_samples": int(len(source_dataset)),
    }  # type: ignore[attr-defined]


def build_dataset(cfg: dict[str, Any], split: str, **extra_dataset_kwargs: Any):
    import_default_components()
    dataset_cfg = deepcopy(cfg["data"]["dataset"])
    normalize_deepsense_dataset_config(dataset_cfg)
    dataset_type = dataset_cfg.get("type")
    descriptor = _optional_dataset_descriptor(dataset_type)
    if descriptor is not None and not dataset_cfg.get("data_root"):
        dataset_cfg["data_root"] = descriptor.default_root
    dataset_cfg["split"] = split
    enabled_modalities = resolve_enabled_modalities(cfg)
    dataset_cfg["enabled_modalities"] = list(enabled_modalities)
    if descriptor is not None:
        dataset_cfg["input_profiles"] = resolve_dataset_profiles(dataset_type, enabled_modalities, dataset_cfg)
    dataset_cfg.update(dataset_flags_for_modalities(enabled_modalities))
    apply_cache_policy(dataset_cfg, cfg, enabled_modalities)
    canonicalize_lidar_dataset_config(dataset_cfg)
    _apply_csi_degradation_seed(dataset_cfg, cfg)
    if _uses_csv_split(dataset_type, descriptor):
        csv_name, dataset_split = _dataset_csv_for_split(dataset_cfg, split)
        dataset_cfg["csv_name"] = csv_name
        dataset_cfg["split"] = dataset_split
        _validate_snapshot_csv_exists(cfg, dataset_cfg, csv_name)
    dataset_cfg.update(extra_dataset_kwargs)
    return DATASETS.build(dataset_cfg)


def build_dataloaders(cfg: dict[str, Any]) -> dict[str, DataLoader]:
    loader_cfg = cfg["data"]["dataloader"]
    training_cfg = cfg.get("training", {})
    protocol_splits = build_protocol_split_datasets(cfg)
    if protocol_splits is not None:
        train_dataset = protocol_splits["train"]
        validation_dataset = protocol_splits.get("validation")
        test_dataset = protocol_splits["test"]
    elif validation_from_train_enabled(cfg):
        train_dataset, validation_dataset = build_train_and_internal_validation_datasets(
            cfg,
            split_dataset_builder=build_split_dataset,
        )
        dataset_kwargs = normalization_kwargs(train_dataset)
        test_dataset = build_split_dataset(cfg, "test", **dataset_kwargs)
    else:
        train_dataset = build_split_dataset(cfg, "train")
        validation_dataset = build_split_dataset(cfg, "validation", **normalization_kwargs(train_dataset)) if has_validation_csv(cfg) else None
        dataset_kwargs = normalization_kwargs(train_dataset)
        test_dataset = build_split_dataset(cfg, "test", **dataset_kwargs)
    prepare_lidar_normalizer(cfg, first_dataset(train_dataset))
    dataloaders = {
        "train": build_dataloader(
            train_dataset,
            loader_cfg,
            split="train",
            epoch_subsampling_cfg=training_cfg.get("epoch_subsampling"),
            experiment_seed=cfg.get("experiment", {}).get("seed", 0),
        ),
        "test": build_dataloader(test_dataset, loader_cfg, split="test"),
    }
    if validation_dataset is not None:
        dataloaders["validation"] = build_dataloader(validation_dataset, loader_cfg, split="validation")
    return dataloaders


def build_split_dataset(cfg: dict[str, Any], split: str, **extra_dataset_kwargs: Any):
    protocol_splits = build_protocol_split_datasets(cfg, **extra_dataset_kwargs)
    if protocol_splits is not None:
        split_key = "validation" if split == "val" else split
        if split_key not in protocol_splits:
            raise ValueError(f"Split protocol did not produce split '{split}'.")
        return protocol_splits[split_key]
    scenes = dataset_scenes_for_split(cfg, split)
    if not scenes:
        return build_dataset(cfg, split, **extra_dataset_kwargs)
    if len(scenes) == 1:
        scene_cfg = retarget_cfg_for_scene(cfg, scenes[0])
        return build_dataset(scene_cfg, split, **extra_dataset_kwargs)
    datasets = [build_dataset(retarget_cfg_for_scene(cfg, scene), split, **extra_dataset_kwargs) for scene in scenes]
    if split == "train":
        harmonize_multi_scene_train_normalizers(datasets)
    return ConcatDataset(datasets)


def build_protocol_split_datasets(cfg: dict[str, Any], **extra_dataset_kwargs: Any) -> dict[str, Any] | None:
    return _build_protocol_split_datasets(
        cfg,
        dataset_builder=build_dataset,
        **extra_dataset_kwargs,
    )


def build_dataloader(
    dataset: Any,
    loader_cfg: dict[str, Any],
    *,
    split: str,
    epoch_subsampling_cfg: dict[str, Any] | None = None,
    experiment_seed: int | None = None,
) -> DataLoader:
    kwargs = build_dataloader_kwargs(loader_cfg, split=split)
    if split == "train":
        sampler = build_epoch_subsample_sampler(
            dataset,
            epoch_subsampling_cfg,
            experiment_seed=experiment_seed,
        )
        if sampler is not None:
            kwargs["shuffle"] = False
            kwargs["sampler"] = sampler
    return DataLoader(dataset, **kwargs)


def _dataset_csv_for_split(dataset_cfg: dict[str, Any], split: str) -> tuple[str | None, str]:
    if split == "train":
        return dataset_cfg.get("train_csv_name"), "train"
    if split in {"validation", "val"}:
        val_csv = dataset_cfg.get("val_csv_name")
        if not val_csv:
            raise ValueError("data.dataset.val_csv_name is required when building a validation split.")
        return val_csv, "validation"
    return dataset_cfg.get("test_csv_name"), split


def _optional_dataset_descriptor(dataset_type: Any):
    dataset_key = str(dataset_type or "deepsense6g").strip().lower()
    if dataset_key in {"synthetic", "synthetic_sequence"}:
        return None
    return dataset_descriptor(dataset_key)


def _uses_csv_split(dataset_type: Any, descriptor: Any) -> bool:
    dataset_key = str(dataset_type or "deepsense6g").strip().lower()
    if dataset_key in {"synthetic", "synthetic_sequence"}:
        return False
    return descriptor is None or descriptor.storage_kind == "csv_sequence"


def _validate_snapshot_csv_exists(cfg: dict[str, Any], dataset_cfg: dict[str, Any], csv_name: str | None) -> None:
    if cfg.get("experiment", {}).get("variant") != "snapshot_next_frame":
        return
    if not csv_name:
        raise FileNotFoundError(
            "Snapshot next-frame baseline requires snapshot CSVs. "
            "Run: kd-sensing-preprocess --config configs/preprocess/sequences_snapshot_next_frame.yaml"
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
            "kd-sensing-preprocess --config configs/preprocess/sequences_snapshot_next_frame.yaml"
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
    "build_protocol_split_datasets",
    "build_split_dataset",
    "prepare_lidar_normalizer",
    "resolve_dataloader_split_config",
    "shutdown_dataloader_workers",
]
