from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Subset

from kd_sensing.config.lidar_normalization import canonicalize_lidar_dataset_config
from kd_sensing.data.dataset_descriptors import dataset_descriptor, resolve_dataset_profiles
from kd_sensing.data.scenes import (
    is_deepsense_dataset_type,
    normalize_deepsense_dataset_config,
    retarget_deepsense_dataset_config,
)
from kd_sensing.data.transform_ops.gps import GPSStandardScaler
from kd_sensing.engine.cache_policy import apply_cache_policy
from kd_sensing.engine.epoch_subsampling import build_epoch_subsample_sampler
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.modalities import dataset_flags_for_modalities
from kd_sensing.registries import DATASETS, import_default_components


SNAPSHOT_TRAIN_CSV = "train_seqs_SNAPSHOT_NEXT_FRAME.csv"
SNAPSHOT_VAL_CSV = "val_seqs_SNAPSHOT_NEXT_FRAME.csv"
STRATIFIED_2604_PROTOCOLS = {
    "stratified_80_10_10",
    "deepsense6g_2604_stratified_80_10_10",
    "2604_stratified_80_10_10",
}


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
    elif _validation_from_train_enabled(cfg):
        train_dataset, validation_dataset = _build_train_and_internal_validation_datasets(cfg)
        dataset_kwargs = _normalization_kwargs(train_dataset)
        test_dataset = build_split_dataset(cfg, "test", **dataset_kwargs)
    else:
        train_dataset = build_split_dataset(cfg, "train")
        validation_dataset = build_split_dataset(cfg, "validation", **_normalization_kwargs(train_dataset)) if _has_validation_csv(cfg) else None
        dataset_kwargs = _normalization_kwargs(train_dataset)
        test_dataset = build_split_dataset(cfg, "test", **dataset_kwargs)
    prepare_lidar_normalizer(cfg, _first_dataset(train_dataset))
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
    scenes = _dataset_scenes_for_split(cfg, split)
    if not scenes:
        return build_dataset(cfg, split, **extra_dataset_kwargs)
    if len(scenes) == 1:
        scene_cfg = _retarget_cfg_for_scene(cfg, scenes[0])
        return build_dataset(scene_cfg, split, **extra_dataset_kwargs)
    datasets = [build_dataset(_retarget_cfg_for_scene(cfg, scene), split, **extra_dataset_kwargs) for scene in scenes]
    if split == "train":
        _harmonize_multi_scene_train_normalizers(datasets)
    return ConcatDataset(datasets)


def build_protocol_split_datasets(cfg: dict[str, Any], **extra_dataset_kwargs: Any) -> dict[str, Any] | None:
    if not _stratified_2604_split_enabled(cfg):
        return None
    split_cfg = _stratified_2604_split_cfg(cfg)
    role_scenes = {
        "train": _dataset_scenes_for_protocol_role(cfg, "train"),
        "validation": _dataset_scenes_for_protocol_role(cfg, "validation"),
        "test": _dataset_scenes_for_protocol_role(cfg, "test"),
    }
    all_scenes = _ordered_unique(
        scene
        for scenes in role_scenes.values()
        for scene in scenes
    )
    if not all_scenes:
        raise ValueError("stratified_80_10_10 split requires at least one DeepSense6G scene.")
    source_splits = tuple(str(item) for item in split_cfg.get("source_splits", ("train", "test")))
    scene_subsets: dict[str, dict[Any, Any]] = {"train": {}, "validation": {}, "test": {}}
    for scene_offset, scene in enumerate(all_scenes):
        full_scene = _build_protocol_union_dataset(cfg, scene, source_splits, extra_dataset_kwargs)
        labels = _target_labels_for_dataset(full_scene)
        index_splits = _stratified_indices_by_label(
            labels,
            seed=int(split_cfg["seed"]) + scene_offset,
            validation_fraction=float(split_cfg["validation_fraction"]),
            test_fraction=float(split_cfg["test_fraction"]),
        )
        for role, indices in index_splits.items():
            subset = Subset(full_scene, indices)
            _annotate_protocol_subset(
                subset,
                role=role,
                source_dataset=full_scene,
                scene=scene,
                split_cfg=split_cfg,
                source_splits=source_splits,
                labels=[labels[int(index)] for index in indices],
            )
            scene_subsets[role][scene] = subset
    result: dict[str, Any] = {}
    for role, scenes in role_scenes.items():
        parts = [scene_subsets[role][scene] for scene in scenes]
        result[role] = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    _fit_or_apply_protocol_gps_scaler(
        result["train"],
        result.get("validation"),
        result["test"],
        gps_scaler=extra_dataset_kwargs.get("gps_scaler"),
    )
    return result


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


def _dataset_scenes_for_split(cfg: dict[str, Any], split: str) -> tuple[Any, ...]:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        return ()
    if split == "train":
        raw = dataset_cfg.get("train_scenes")
    elif split in {"validation", "val"}:
        raw = dataset_cfg.get("validation_scenes", dataset_cfg.get("val_scenes", dataset_cfg.get("train_scenes")))
    else:
        raw = dataset_cfg.get("test_scenes", dataset_cfg.get("eval_scenes", dataset_cfg.get("validation_scenes")))
    if raw is None:
        return ()
    if isinstance(raw, (str, int, float)):
        return (raw,)
    return tuple(raw)


def _stratified_2604_split_enabled(cfg: dict[str, Any]) -> bool:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        return False
    protocol = str(dataset_cfg.get("split_protocol") or "").strip().lower()
    return protocol in STRATIFIED_2604_PROTOCOLS


def _stratified_2604_split_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    fractions = dataset_cfg.get("split_fractions") if isinstance(dataset_cfg.get("split_fractions"), dict) else {}
    validation_fraction = float(fractions.get("validation", fractions.get("val", 0.1)))
    test_fraction = float(fractions.get("test", 0.1))
    if validation_fraction <= 0.0 or test_fraction <= 0.0 or validation_fraction + test_fraction >= 1.0:
        raise ValueError("data.dataset.split_fractions must define positive validation/test fractions with train > 0.")
    return {
        "protocol": str(dataset_cfg.get("split_protocol")),
        "strategy": str(dataset_cfg.get("split_strategy") or "stratified_by_target_beam_per_scene"),
        "seed": int(dataset_cfg.get("split_seed", cfg.get("experiment", {}).get("seed", 0))),
        "train_fraction": float(1.0 - validation_fraction - test_fraction),
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "source_splits": tuple(dataset_cfg.get("split_source_splits") or ("train", "test")),
        "label_source": str(dataset_cfg.get("split_label_source") or "future_beam1"),
    }


def _dataset_scenes_for_protocol_role(cfg: dict[str, Any], role: str) -> tuple[Any, ...]:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        return ()
    if role == "train":
        raw = dataset_cfg.get("train_scenes", dataset_cfg.get("scenes", dataset_cfg.get("scene")))
    elif role == "validation":
        raw = dataset_cfg.get(
            "validation_scenes",
            dataset_cfg.get("val_scenes", dataset_cfg.get("train_scenes", dataset_cfg.get("scenes"))),
        )
    else:
        raw = dataset_cfg.get(
            "test_scenes",
            dataset_cfg.get("eval_scenes", dataset_cfg.get("scenes", dataset_cfg.get("train_scenes"))),
        )
    if raw is None:
        raw = dataset_cfg.get("scene")
    if isinstance(raw, (str, int, float)):
        return (raw,)
    return tuple(raw or ())


def _ordered_unique(values) -> tuple[Any, ...]:
    result = []
    seen = set()
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return tuple(result)


def _build_protocol_union_dataset(
    cfg: dict[str, Any],
    scene: Any,
    source_splits: tuple[str, ...],
    extra_dataset_kwargs: dict[str, Any],
) -> Any:
    parts = []
    for source_split in source_splits:
        scene_cfg = _retarget_cfg_for_scene(cfg, scene)
        dataset_cfg = scene_cfg.setdefault("data", {}).setdefault("dataset", {})
        if "gps_scaler" not in extra_dataset_kwargs:
            dataset_cfg["gps_normalize"] = False
        parts.append(build_dataset(scene_cfg, source_split, **extra_dataset_kwargs))
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def _target_labels_for_dataset(dataset: Any) -> list[int]:
    if isinstance(dataset, ConcatDataset):
        labels: list[int] = []
        for component in dataset.datasets:
            labels.extend(_target_labels_for_dataset(component))
        return labels
    if isinstance(dataset, Subset):
        parent_labels = _target_labels_for_dataset(dataset.dataset)
        return [parent_labels[int(index)] for index in dataset.indices]
    labels = []
    samples = getattr(dataset, "samples", None)
    future_beam_paths = getattr(samples, "future_beam_paths", None)
    if future_beam_paths is None:
        raise ValueError("stratified_80_10_10 split requires dataset.samples.future_beam_paths.")
    for idx, paths in enumerate(future_beam_paths):
        if not paths:
            raise ValueError("stratified_80_10_10 split found a sample with no future_beam path.")
        labels.append(int(dataset._target_raw_beam_label_for_index(idx, 0, paths[0])))
    return labels


def _stratified_indices_by_label(
    labels: list[int],
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, list[int]]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[int(label)].append(int(idx))
    rng = np.random.default_rng(int(seed))
    splits = {"train": [], "validation": [], "test": []}
    for label in sorted(by_label):
        indices = np.asarray(by_label[label], dtype=np.int64)
        rng.shuffle(indices)
        total = len(indices)
        test_count = int(round(float(total) * test_fraction)) if total >= 10 else 0
        validation_count = int(round(float(total) * validation_fraction)) if total >= 10 else 0
        if validation_count + test_count >= total:
            overflow = validation_count + test_count - max(0, total - 1)
            while overflow > 0 and validation_count >= test_count and validation_count > 0:
                validation_count -= 1
                overflow -= 1
            while overflow > 0 and test_count > 0:
                test_count -= 1
                overflow -= 1
        test_indices = indices[:test_count]
        validation_indices = indices[test_count : test_count + validation_count]
        train_indices = indices[test_count + validation_count :]
        splits["test"].extend(int(index) for index in test_indices)
        splits["validation"].extend(int(index) for index in validation_indices)
        splits["train"].extend(int(index) for index in train_indices)
    for role, indices in splits.items():
        if not indices:
            raise ValueError(f"stratified_80_10_10 split produced an empty {role} split.")
        splits[role] = sorted(indices)
    return splits


def _annotate_protocol_subset(
    subset: Subset,
    *,
    role: str,
    source_dataset: Any,
    scene: Any,
    split_cfg: dict[str, Any],
    source_splits: tuple[str, ...],
    labels: list[int],
) -> None:
    subset.split = role  # type: ignore[attr-defined]
    label_counts = {str(label): int(labels.count(label)) for label in sorted(set(labels))}
    subset.stratified_split = {  # type: ignore[attr-defined]
        "enabled": True,
        "protocol": split_cfg["protocol"],
        "strategy": split_cfg["strategy"],
        "source_split": "train+test",
        "source_splits": list(source_splits),
        "role": role,
        "scene": scene,
        "seed": int(split_cfg["seed"]),
        "train_fraction": float(split_cfg["train_fraction"]),
        "validation_fraction": float(split_cfg["validation_fraction"]),
        "test_fraction": float(split_cfg["test_fraction"]),
        "label_source": split_cfg["label_source"],
        "parent_num_samples": int(len(source_dataset)),
        "label_count": len(label_counts),
        "label_distribution": label_counts,
    }


def _retarget_cfg_for_scene(cfg: dict[str, Any], scene: Any) -> dict[str, Any]:
    scene_cfg = deepcopy(cfg)
    dataset_cfg = scene_cfg.setdefault("data", {}).setdefault("dataset", {})
    if not is_deepsense_dataset_type(dataset_cfg.get("type", "deepsense6g")):
        raise ValueError("data.dataset.train_scenes/test_scenes are currently supported only for DeepSense6G.")
    retarget_deepsense_dataset_config(dataset_cfg, scene)
    return scene_cfg


def _first_dataset(dataset: Any) -> Any:
    if isinstance(dataset, ConcatDataset):
        datasets = list(getattr(dataset, "datasets", []))
        return _first_dataset(datasets[0]) if datasets else dataset
    if isinstance(dataset, Subset):
        return _first_dataset(dataset.dataset)
    return dataset


def _normalization_kwargs(dataset: Any) -> dict[str, Any]:
    source = _first_dataset(dataset)
    kwargs: dict[str, Any] = {}
    if getattr(source, "use_gps", False):
        kwargs["gps_scaler"] = getattr(source, "gps_scaler", None)
    if getattr(source, "use_lidar", False):
        kwargs["lidar_normalizer"] = getattr(source, "lidar_normalizer", None)
    if getattr(source, "use_mmwave", False):
        kwargs["mmwave_scaler"] = getattr(source, "mmwave_scaler", None)
    if getattr(source, "use_csi", False):
        kwargs["csi_rms_normalizer"] = getattr(source, "csi_rms_normalizer", None)
    if getattr(source, "occlusion_target_enabled", False):
        kwargs["occlusion_target_stats"] = getattr(source, "occlusion_target_stats", None)
    if getattr(source, "position_target_enabled", False):
        kwargs["position_target_scaler"] = getattr(source, "position_target_scaler", None)
    return kwargs


def _harmonize_multi_scene_train_normalizers(datasets: list[Any]) -> None:
    _harmonize_multi_scene_gps_scaler(datasets)


def _validation_from_train_enabled(cfg: dict[str, Any]) -> bool:
    validation_cfg = cfg.get("data", {}).get("validation_from_train")
    if isinstance(validation_cfg, dict):
        return bool(validation_cfg.get("enabled", False))
    return bool(validation_cfg)


def _validation_from_train_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    validation_cfg = cfg.get("data", {}).get("validation_from_train")
    if isinstance(validation_cfg, dict):
        return validation_cfg
    return {"enabled": bool(validation_cfg)}


def _has_validation_csv(cfg: dict[str, Any]) -> bool:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    return bool(dataset_cfg.get("val_csv_name"))


def _build_train_and_internal_validation_datasets(cfg: dict[str, Any]) -> tuple[Any, Any]:
    raw_cfg = deepcopy(cfg)
    raw_cfg.setdefault("data", {}).setdefault("dataset", {})["gps_normalize"] = False
    full_train = build_split_dataset(raw_cfg, "train")
    train_dataset, validation_dataset = _split_dataset_for_internal_validation(full_train, cfg)
    _fit_internal_validation_gps_scaler(train_dataset, validation_dataset)
    return train_dataset, validation_dataset


def _split_dataset_for_internal_validation(dataset: Any, cfg: dict[str, Any]) -> tuple[Any, Any]:
    if isinstance(dataset, ConcatDataset):
        train_parts = []
        validation_parts = []
        for offset, component in enumerate(dataset.datasets):
            train_subset, validation_subset = _split_single_dataset_for_internal_validation(
                component,
                cfg,
                seed_offset=offset,
            )
            train_parts.append(train_subset)
            validation_parts.append(validation_subset)
        return ConcatDataset(train_parts), ConcatDataset(validation_parts)
    return _split_single_dataset_for_internal_validation(dataset, cfg, seed_offset=0)


def _split_single_dataset_for_internal_validation(dataset: Any, cfg: dict[str, Any], *, seed_offset: int) -> tuple[Subset, Subset]:
    validation_cfg = _validation_from_train_cfg(cfg)
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
    _annotate_internal_subset(train_subset, role="train", source_dataset=dataset, fraction=fraction, seed=seed)
    _annotate_internal_subset(validation_subset, role="validation", source_dataset=dataset, fraction=fraction, seed=seed)
    return train_subset, validation_subset


def _annotate_internal_subset(subset: Subset, *, role: str, source_dataset: Any, fraction: float, seed: int) -> None:
    subset.split = role  # type: ignore[attr-defined]
    subset.internal_split = {
        "enabled": True,
        "source_split": "train",
        "role": role,
        "fraction": float(fraction),
        "seed": int(seed),
        "parent_num_samples": int(len(source_dataset)),
    }  # type: ignore[attr-defined]


def _fit_internal_validation_gps_scaler(train_dataset: Any, validation_dataset: Any) -> None:
    _fit_gps_scaler_from_train_dataset(
        train_dataset,
        validation_dataset,
        source="internal_train_subset_streaming_fit",
    )


def _fit_or_apply_protocol_gps_scaler(train_dataset: Any, validation_dataset: Any, test_dataset: Any, *, gps_scaler: Any) -> None:
    if gps_scaler is not None:
        _apply_gps_scaler_to_datasets(gps_scaler, train_dataset, validation_dataset, test_dataset)
        return
    _fit_gps_scaler_from_train_dataset(
        train_dataset,
        validation_dataset,
        test_dataset,
        source="stratified_train_subset_streaming_fit",
    )


def _fit_gps_scaler_from_train_dataset(train_dataset: Any, *apply_datasets: Any, source: str) -> None:
    train_leaves = _leaf_datasets_with_indices(train_dataset)
    gps_train_leaves = [
        (dataset, indices)
        for dataset, indices in train_leaves
        if getattr(dataset, "use_gps", False) and hasattr(dataset, "_gps_features_for_index")
    ]
    if not gps_train_leaves:
        return
    stats_sum = None
    stats_sum_sq = None
    frame_count = 0
    sample_count = 0
    scene_slugs = []
    for dataset, indices in gps_train_leaves:
        scene_slugs.append(str(getattr(dataset, "scene_slug", getattr(dataset, "scene_id", ""))))
        for idx in indices:
            features = np.asarray(dataset._gps_features_for_index(int(idx)), dtype=np.float64)
            if features.ndim != 2:
                raise ValueError(f"GPS features must have shape [T, D], got {features.shape}.")
            batch_sum = features.sum(axis=0)
            batch_sum_sq = np.square(features).sum(axis=0)
            stats_sum = batch_sum if stats_sum is None else stats_sum + batch_sum
            stats_sum_sq = batch_sum_sq if stats_sum_sq is None else stats_sum_sq + batch_sum_sq
            frame_count += int(features.shape[0])
            sample_count += 1
    if frame_count <= 0 or stats_sum is None or stats_sum_sq is None:
        return
    mean = stats_sum / float(frame_count)
    variance = np.maximum(stats_sum_sq / float(frame_count) - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    scaler = GPSStandardScaler(mean_=mean.astype(np.float32), scale_=scale.astype(np.float32))
    metadata = {
        "source": source,
        "sample_count": int(sample_count),
        "frame_count": int(frame_count),
        "scene_slugs": scene_slugs,
        "streaming": True,
        "retains_per_sample_sequence_cache": False,
    }
    _apply_gps_scaler_to_datasets(scaler, train_dataset, *apply_datasets, metadata=metadata)


def _apply_gps_scaler_to_datasets(scaler: Any, *datasets: Any, metadata: dict[str, Any] | None = None) -> None:
    if scaler is None:
        return
    for root_dataset in datasets:
        if root_dataset is None:
            continue
        for dataset, _ in _leaf_datasets_with_indices(root_dataset):
            if not getattr(dataset, "use_gps", False):
                continue
            dataset.gps_normalize = True
            dataset.gps_scaler = scaler
            if metadata is not None:
                dataset.gps_scaler_metadata = dict(metadata)
            elif not getattr(dataset, "gps_scaler_metadata", None):
                dataset.gps_scaler_metadata = {
                    "source": "external_or_checkpoint",
                    "streaming": False,
                }
            if hasattr(dataset, "_gps_feature_cache"):
                dataset._gps_feature_cache.clear()


def _leaf_datasets_with_indices(dataset: Any) -> list[tuple[Any, list[int]]]:
    if isinstance(dataset, ConcatDataset):
        result: list[tuple[Any, list[int]]] = []
        for component in dataset.datasets:
            result.extend(_leaf_datasets_with_indices(component))
        return result
    if isinstance(dataset, Subset):
        if isinstance(dataset.dataset, Subset):
            base_pairs = _leaf_datasets_with_indices(dataset.dataset)
            if len(base_pairs) != 1:
                return base_pairs
            base_dataset, base_indices = base_pairs[0]
            mapped = [base_indices[int(index)] for index in dataset.indices]
            return [(base_dataset, mapped)]
        if isinstance(dataset.dataset, ConcatDataset):
            grouped: dict[int, list[int]] = defaultdict(list)
            cumulative_sizes = list(dataset.dataset.cumulative_sizes)
            for raw_index in dataset.indices:
                global_index = int(raw_index)
                component_idx = bisect_right(cumulative_sizes, global_index)
                previous = cumulative_sizes[component_idx - 1] if component_idx > 0 else 0
                grouped[component_idx].append(global_index - previous)
            result: list[tuple[Any, list[int]]] = []
            for component_idx, local_indices in sorted(grouped.items()):
                component = dataset.dataset.datasets[component_idx]
                if isinstance(component, Subset):
                    base_pairs = _leaf_datasets_with_indices(component)
                    if len(base_pairs) == 1:
                        base_dataset, base_indices = base_pairs[0]
                        mapped = [base_indices[int(index)] for index in local_indices]
                        result.append((base_dataset, mapped))
                    else:
                        result.extend(base_pairs)
                else:
                    result.append((component, [int(index) for index in local_indices]))
            return result
        return [(dataset.dataset, [int(index) for index in dataset.indices])]
    return [(dataset, list(range(len(dataset))))]


def _harmonize_multi_scene_gps_scaler(datasets: list[Any]) -> None:
    gps_datasets = [
        dataset
        for dataset in datasets
        if getattr(dataset, "use_gps", False) and getattr(dataset, "gps_normalize", False)
    ]
    if len(gps_datasets) <= 1:
        return
    total_frames = 0
    weighted_mean = None
    weighted_second = None
    for dataset in gps_datasets:
        scaler = getattr(dataset, "gps_scaler", None)
        metadata = getattr(dataset, "gps_scaler_metadata", {}) or {}
        if scaler is None or scaler.mean_ is None or scaler.scale_ is None:
            return
        frame_count = int(metadata.get("frame_count", 0) or 0)
        if frame_count <= 0:
            return
        mean = np.asarray(scaler.mean_, dtype=np.float64)
        scale = np.asarray(scaler.scale_, dtype=np.float64)
        second = np.square(scale) + np.square(mean)
        weighted_mean = mean * frame_count if weighted_mean is None else weighted_mean + mean * frame_count
        weighted_second = second * frame_count if weighted_second is None else weighted_second + second * frame_count
        total_frames += frame_count
    if total_frames <= 0 or weighted_mean is None or weighted_second is None:
        return
    mean = weighted_mean / float(total_frames)
    variance = np.maximum((weighted_second / float(total_frames)) - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    scaler = GPSStandardScaler(mean_=mean, scale_=scale)
    sample_count = sum(len(dataset) for dataset in gps_datasets)
    scene_slugs = [str(getattr(dataset, "scene_slug", getattr(dataset, "scene_id", ""))) for dataset in gps_datasets]
    metadata = {
        "source": "multi_scene_train_split_streaming_fit",
        "sample_count": int(sample_count),
        "frame_count": int(total_frames),
        "scene_slugs": scene_slugs,
        "streaming": True,
        "retains_per_sample_sequence_cache": False,
    }
    for dataset in gps_datasets:
        dataset.gps_scaler = scaler
        dataset.gps_scaler_metadata = dict(metadata)
        if hasattr(dataset, "_gps_feature_cache"):
            dataset._gps_feature_cache.clear()


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
