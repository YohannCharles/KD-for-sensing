from typing import Any

import torch

from kd_sensing.registries import LOSSES, MODELS, import_default_components


def build_model(model_cfg: dict[str, Any]):
    import_default_components()
    return MODELS.build(model_cfg)


def build_task_criterion(cfg: dict[str, Any]):
    import_default_components()
    configured = cfg["loss"]
    loss_cfg = {"type": configured.get("type", "cross_entropy")}
    if loss_cfg["type"] == "focal_loss":
        loss_cfg.update({key: configured[key] for key in ("alpha", "gamma") if key in configured})
    return LOSSES.build(loss_cfg)


def build_optimizer(cfg: dict[str, Any], model) -> torch.optim.Optimizer:
    training_cfg = cfg["training"]
    optimizer_cfg = training_cfg.get("optimizer", {})
    if optimizer_cfg is None:
        optimizer_cfg = {}
    if isinstance(optimizer_cfg, str):
        optimizer_cfg = {"type": optimizer_cfg}
    if not isinstance(optimizer_cfg, dict):
        raise ValueError("training.optimizer must be a mapping when provided.")
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found for optimizer.")
    optimizer_type = str(optimizer_cfg.get("type", "adam")).strip().lower()
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}.get(optimizer_type)
    if optimizer_class is None:
        raise ValueError(f"Unsupported optimizer type {optimizer_type!r}; supported types are 'adam' and 'adamw'.")
    return optimizer_class(
        trainable_params,
        lr=float(training_cfg.get("lr", optimizer_cfg.get("lr", 7.5e-4))),
        weight_decay=float(training_cfg.get("weight_decay", optimizer_cfg.get("weight_decay", 0.0))),
    )


def optimizer_param_group_summary(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    summary = []
    for index, group in enumerate(optimizer.param_groups):
        params = list(group.get("params", []))
        summary.append(
            {
                "index": index,
                "name": str(group.get("name", f"group_{index}")),
                "lr": float(group.get("lr", 0.0)),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "param_count": int(group.get("param_count", _param_count(params))),
            }
        )
    return summary


def _param_count(params: list[torch.nn.Parameter] | tuple[torch.nn.Parameter, ...]) -> int:
    return int(sum(param.numel() for param in params))


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


__all__ = [
    "build_device",
    "build_model",
    "build_optimizer",
    "build_scheduler",
    "build_task_criterion",
    "optimizer_param_group_summary",
]
