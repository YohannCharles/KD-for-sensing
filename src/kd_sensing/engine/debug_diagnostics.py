import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from kd_sensing.engine.objectives.metadata import objective_runtime_metadata


def build_startup_summary(
    cfg: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    device: torch.device,
) -> dict[str, Any]:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    loader_cfg = cfg.get("data", {}).get("dataloader", {})
    model_cfg = cfg.get("model", {})
    primary_cfg = model_cfg.get("primary", {}) if isinstance(model_cfg.get("primary"), dict) else {}
    return {
        "experiment": {
            "name": cfg.get("experiment", {}).get("name"),
            "task": cfg.get("experiment", {}).get("task"),
            "objective": cfg.get("experiment", {}).get("objective"),
            "seed": cfg.get("experiment", {}).get("seed"),
            "device": str(device),
        },
        "objective": objective_runtime_metadata(),
        "data": {
            "modalities": list(primary_cfg.get("modalities") or model_cfg.get("modalities") or []),
            "dataset_type": dataset_cfg.get("type"),
            "dataset_path": dataset_cfg.get("data_root"),
            "train_split": dataset_cfg.get("train_csv_name"),
            "val_split": dataset_cfg.get("val_csv_name"),
            "test_split": dataset_cfg.get("test_csv_name"),
            "seq_len": dataset_cfg.get("seq_len"),
            "num_pred": dataset_cfg.get("num_pred"),
            "num_classes": model_cfg.get("num_classes") or primary_cfg.get("num_classes"),
            "batch_size": {
                "train": loader_cfg.get("train_batch_size"),
                "test": loader_cfg.get("test_batch_size"),
            },
        },
        "optimization": {
            "optimizer": type(optimizer).__name__,
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            "weight_decays": [float(group.get("weight_decay", 0.0)) for group in optimizer.param_groups],
            "scheduler": type(scheduler).__name__ if scheduler is not None else None,
            "loss": cfg.get("loss", {}).get("type"),
            "max_epochs": cfg.get("training", {}).get("epochs"),
        },
        "model": {
            "type": primary_cfg.get("type"),
            "class": model.__class__.__name__,
            "d_model": primary_cfg.get("d_model") or model_cfg.get("d_model"),
        },
        "parameters": module_trainability_report(model),
    }


def module_trainability_report(model: nn.Module) -> dict[str, Any]:
    named_parameters = list(model.named_parameters())
    total = sum(parameter.numel() for _, parameter in named_parameters)
    trainable = sum(parameter.numel() for _, parameter in named_parameters if parameter.requires_grad)
    modules = {}
    for name, module in model.named_children():
        parameters = list(module.parameters())
        module_total = sum(parameter.numel() for parameter in parameters)
        module_trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        modules[name] = {
            "total_params": int(module_total),
            "trainable_params": int(module_trainable),
        }
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(total - trainable),
        "trainable_parameter_names": [name for name, parameter in named_parameters if parameter.requires_grad],
        "modules": modules,
    }


def print_startup_summary(summary: dict[str, Any]) -> None:
    print("[startup_summary] " + json.dumps(summary, sort_keys=True), flush=True)


def write_startup_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    path = run_dir / "startup_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "build_startup_summary",
    "module_trainability_report",
    "print_startup_summary",
    "write_startup_summary",
]
