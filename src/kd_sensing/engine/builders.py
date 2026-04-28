from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from kd_sensing.registries import DATASETS, DISTILLERS, LOSSES, METRICS, MODELS, import_default_components
from kd_sensing.utils.paths import resolve_path


def build_dataset(cfg: dict[str, Any], split: str, **extra_dataset_kwargs: Any):
    import_default_components()
    dataset_cfg = deepcopy(cfg["data"]["dataset"])
    dataset_type = dataset_cfg.get("type")
    dataset_cfg["split"] = split
    if _config_uses_gps(cfg):
        dataset_cfg.setdefault("use_gps", True)
    if _config_uses_lidar(cfg):
        dataset_cfg.setdefault("use_lidar", True)
    if dataset_type not in {"synthetic", "synthetic_sequence"}:
        csv_key = "train_csv_name" if split == "train" else "test_csv_name"
        dataset_cfg["csv_name"] = dataset_cfg.get(csv_key)
    dataset_cfg.update(extra_dataset_kwargs)
    return DATASETS.build(dataset_cfg)


def build_dataloaders(cfg: dict[str, Any]) -> dict[str, DataLoader]:
    loader_cfg = cfg["data"]["dataloader"]
    train_dataset = build_dataset(cfg, "train")
    dataset_kwargs = {}
    if getattr(train_dataset, "use_gps", False):
        dataset_kwargs["gps_scaler"] = getattr(train_dataset, "gps_scaler", None)
    if getattr(train_dataset, "use_lidar", False):
        dataset_kwargs["lidar_normalizer"] = getattr(train_dataset, "lidar_normalizer", None)
    test_dataset = build_dataset(cfg, "test", **dataset_kwargs)
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=loader_cfg.get("train_batch_size", 3),
            shuffle=True,
            num_workers=loader_cfg.get("num_workers", 0),
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=loader_cfg.get("test_batch_size", 3),
            shuffle=False,
            num_workers=loader_cfg.get("num_workers", 0),
        ),
    }


def _config_uses_gps(cfg: dict[str, Any]) -> bool:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if dataset_cfg.get("use_gps", False):
        return True
    task = cfg.get("experiment", {}).get("task", "image")
    if task == "gps":
        return True
    if task != "fusion":
        return False
    model_cfg = cfg.get("model", {})
    for role in ("student", "teacher"):
        modalities = model_cfg.get(role, {}).get("modalities")
        if modalities and "gps" in modalities:
            return True
    return False


def _config_uses_lidar(cfg: dict[str, Any]) -> bool:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if dataset_cfg.get("use_lidar", False):
        return True
    task = cfg.get("experiment", {}).get("task", "image")
    if task == "lidar":
        return True
    if task != "fusion":
        return False
    model_cfg = cfg.get("model", {})
    for role in ("student", "teacher"):
        modalities = model_cfg.get(role, {}).get("modalities")
        if modalities and "lidar" in modalities:
            return True
    return False


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
