from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from kd_sensing.registries import LOSSES, METRICS, MODELS, import_default_components


def build_model(model_cfg: dict[str, Any]):
    import_default_components()
    return MODELS.build(model_cfg)


def build_task_criterion(cfg: dict[str, Any]):
    import_default_components()
    loss_cfg = deepcopy(cfg["loss"])
    for auxiliary_key in (
        "beam_soft",
        "soft_targets",
        "unimodal_aux",
        "auxiliary",
        "multitask",
        "multi_task",
        "objective",
        "selection",
        "selection_multitask",
        "occlusion",
        "position",
        "los",
        "link_quality",
    ):
        loss_cfg.pop(auxiliary_key, None)
    if loss_cfg.get("type") == "cross_entropy":
        loss_cfg.pop("alpha", None)
        loss_cfg.pop("gamma", None)
    return LOSSES.build(loss_cfg)


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
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found for optimizer.")
    return torch.optim.Adam(
        [{"params": trainable_params, "name": "main", "param_count": _param_count(trainable_params)}],
        lr=training_cfg.get("lr", 7.5e-4),
        weight_decay=training_cfg.get("weight_decay", 0.0),
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
    "build_metrics",
    "build_model",
    "build_optimizer",
    "build_scheduler",
    "build_task_criterion",
    "optimizer_param_group_summary",
]
