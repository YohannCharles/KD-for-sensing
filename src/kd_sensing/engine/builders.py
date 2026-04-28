from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from kd_sensing.registries import DATASETS, DISTILLERS, LOSSES, METRICS, MODELS, import_default_components
from kd_sensing.utils.paths import resolve_path


VALID_MODALITIES = ("image", "radar", "gps", "lidar")


def build_dataset(cfg: dict[str, Any], split: str, **extra_dataset_kwargs: Any):
    import_default_components()
    dataset_cfg = deepcopy(cfg["data"]["dataset"])
    dataset_type = dataset_cfg.get("type")
    dataset_cfg["split"] = split
    enabled_modalities = resolve_enabled_modalities(cfg)
    dataset_cfg["enabled_modalities"] = list(enabled_modalities)
    dataset_cfg["use_gps"] = "gps" in enabled_modalities
    dataset_cfg["use_lidar"] = "lidar" in enabled_modalities
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
    test_dataset = build_dataset(cfg, "test", **dataset_kwargs)
    return {
        "train": build_dataloader(train_dataset, loader_cfg, split="train"),
        "test": build_dataloader(test_dataset, loader_cfg, split="test"),
    }


def resolve_enabled_modalities(cfg: dict[str, Any]) -> tuple[str, ...]:
    task = cfg.get("experiment", {}).get("task", "image")
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if task == "fusion":
        selected = _resolve_fusion_modalities(cfg)
        _validate_dataset_modality_flags(dataset_cfg, selected)
        return selected
    if task not in VALID_MODALITIES:
        raise ValueError(f"Unsupported experiment.task '{task}'.")
    selected = (task,)
    _validate_dataset_modality_flags(dataset_cfg, selected)
    return selected


def _resolve_fusion_modalities(cfg: dict[str, Any]) -> tuple[str, ...]:
    model_cfg = cfg.get("model", {})
    role_modalities = []
    for role in ("teacher", "student"):
        modalities = model_cfg.get(role, {}).get("modalities")
        if modalities:
            role_modalities.append((role, _normalize_modalities(modalities)))
    if not role_modalities:
        return ("image", "radar")
    first_role, selected = role_modalities[0]
    for role, modalities in role_modalities[1:]:
        if modalities != selected:
            raise ValueError(
                "Fusion teacher/student modalities must match unless an explicit cross-modal "
                f"distillation mode is implemented; {first_role}={list(selected)}, {role}={list(modalities)}."
            )
    return selected


def _normalize_modalities(modalities: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    selected = [str(modality) for modality in modalities]
    if not selected:
        raise ValueError("Fusion modalities must contain at least one modality.")
    invalid = [name for name in selected if name not in VALID_MODALITIES]
    if invalid:
        raise ValueError(f"Unknown fusion modalities: {invalid}.")
    if len(set(selected)) != len(selected):
        raise ValueError(f"Fusion modalities must not contain duplicates: {selected}.")
    return tuple(name for name in VALID_MODALITIES if name in set(selected))


def _validate_dataset_modality_flags(dataset_cfg: dict[str, Any], selected: tuple[str, ...]) -> None:
    for modality, key in (("gps", "use_gps"), ("lidar", "use_lidar")):
        if dataset_cfg.get(key, False) and modality not in selected:
            raise ValueError(
                f"data.dataset.{key}=true conflicts with enabled modalities {list(selected)}. "
                f"Add '{modality}' to the task/modalities or disable {key}."
            )


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


def dataset_run_metadata(dataset: Any) -> dict[str, Any]:
    csv_path = getattr(dataset, "root_csv", None)
    csv_name = Path(csv_path).name if csv_path is not None else None
    metadata = {
        "split": getattr(dataset, "split", None),
        "csv_path": str(csv_path) if csv_path is not None else None,
        "csv_name": csv_name,
        "num_samples": len(dataset),
        "enabled_modalities": list(getattr(dataset, "enabled_modalities", [])),
        "split_family": _split_family(csv_name),
    }
    lidar_cache_dir = getattr(dataset, "lidar_cache_dir", None)
    if lidar_cache_dir is not None:
        metadata["lidar_cache_dir"] = str(lidar_cache_dir)
    return metadata


def dataloaders_run_metadata(dataloaders: dict[str, DataLoader]) -> dict[str, Any]:
    return {split: dataset_run_metadata(loader.dataset) for split, loader in dataloaders.items()}


def _split_family(csv_name: str | None) -> str | None:
    if csv_name in {"train_seqs_RA_GPS_LIDAR.csv", "test_seqs_RA_GPS_LIDAR.csv"}:
        return "unified_gps_lidar"
    if csv_name is None:
        return None
    return "configured"


def prepare_lidar_normalizer(cfg: dict[str, Any], dataset: Any) -> None:
    if not getattr(dataset, "needs_lidar_streaming_stats", False):
        return
    progress_enabled = cfg.get("output", {}).get("progress", {}).get("enabled", True)
    dataset.fit_lidar_normalizer_streaming(progress_enabled=progress_enabled)


def _config_uses_gps(cfg: dict[str, Any]) -> bool:
    return "gps" in resolve_enabled_modalities(cfg)


def _config_uses_lidar(cfg: dict[str, Any]) -> bool:
    return "lidar" in resolve_enabled_modalities(cfg)


def build_model(model_cfg: dict[str, Any]):
    import_default_components()
    return MODELS.build(model_cfg)


def build_task_criterion(cfg: dict[str, Any]):
    import_default_components()
    return LOSSES.build(cfg["loss"])


def build_distiller(cfg: dict[str, Any], task_criterion):
    import_default_components()
    return DISTILLERS.build(cfg["distillation"], task_criterion=task_criterion)


def build_metrics(cfg: dict[str, Any]) -> dict[str, Any]:
    import_default_components()
    eval_cfg = cfg.get("evaluation", {})
    return {
        "topk": METRICS.build(
            {
                "type": "topk_accuracy",
                "k_values": eval_cfg.get("k_values", [1, 2, 3, 5, 10]),
            }
        ),
        "dba": METRICS.build(
            {
                "type": "dba",
                "delta": eval_cfg.get("dba_delta", 5),
            }
        ),
    }


def build_optimizer(cfg: dict[str, Any], model) -> torch.optim.Optimizer:
    training_cfg = cfg["training"]
    return torch.optim.Adam(
        model.parameters(),
        lr=training_cfg.get("lr", 7.5e-4),
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )


def build_scheduler(cfg: dict[str, Any], optimizer: torch.optim.Optimizer):
    scheduler_cfg = cfg.get("scheduler", {})
    if scheduler_cfg.get("type", "cosine_warm_restarts") == "none":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=scheduler_cfg.get("T_0", 10),
        T_mult=scheduler_cfg.get("T_mult", 2),
        eta_min=scheduler_cfg.get("eta_min", 1e-6),
    )


def build_device(cfg: dict[str, Any]) -> torch.device:
    requested = cfg.get("experiment", {}).get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_weight_path(cfg: dict[str, Any], weight_name: str | None) -> Path | None:
    if not weight_name:
        return None
    candidate = Path(weight_name).expanduser()
    if candidate.is_absolute():
        return candidate
    weights_dir = cfg.get("paths", {}).get("weights_dir", "All_models")
    return resolve_path(Path(weights_dir) / candidate)
