from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from kd_sensing.registries import DISTILLERS, LOSSES, METRICS, MODELS, import_default_components


def build_model(model_cfg: dict[str, Any]):
    import_default_components()
    return MODELS.build(model_cfg)


def build_task_criterion(cfg: dict[str, Any]):
    import_default_components()
    loss_cfg = deepcopy(cfg["loss"])
    for auxiliary_key in (
        "beam_soft",
        "unimodal_aux",
        "gate",
        "gate_weight",
        "gate_ramp_epochs",
        "uni_weight_warmup",
        "uni_weight_after_warmup",
        "prior_regularization",
        "marf",
    ):
        loss_cfg.pop(auxiliary_key, None)
    if loss_cfg.get("type") == "cross_entropy":
        loss_cfg.pop("alpha", None)
        loss_cfg.pop("gamma", None)
    return LOSSES.build(loss_cfg)


def build_distiller(cfg: dict[str, Any], task_criterion):
    import_default_components()
    model_cfg = cfg.get("model", {})
    modalities = (
        model_cfg.get("modalities")
        or model_cfg.get("student", {}).get("modalities")
        or model_cfg.get("teacher", {}).get("modalities")
    )
    return DISTILLERS.build(
        cfg["distillation"],
        task_criterion=task_criterion,
        num_pred=model_cfg.get("num_pred", 3),
        num_classes=model_cfg.get("num_classes", 64),
        feature_size=model_cfg.get("feature_size", 64),
        modalities=modalities,
    )


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
    param_group_cfg = cfg.get("finetune", {}).get("param_groups", {})
    if param_group_cfg.get("enabled", False):
        groups = _teacher_prior_param_groups(cfg, model, param_group_cfg)
        if not groups:
            raise ValueError("No trainable parameters found for Stage 3 optimizer parameter groups.")
        return torch.optim.Adam(groups, weight_decay=training_cfg.get("weight_decay", 0.0))
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


def _teacher_prior_param_groups(cfg: dict[str, Any], model, param_group_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    training_lr = float(cfg.get("training", {}).get("lr", 7.5e-4))
    strong_modalities = set(str(name) for name in param_group_cfg.get("strong_modalities", ["gps", "mmwave"]))
    groups: dict[str, list[torch.nn.Parameter]] = {
        "fusion": [],
        "head": [],
        "gate": [],
        "strong_encoder": [],
        "weak_encoder": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        role = _parameter_role(name, strong_modalities)
        groups[role].append(param)
    param_groups = []
    lr_by_role = {
        "fusion": float(param_group_cfg.get("fusion_lr", training_lr)),
        "head": float(param_group_cfg.get("head_lr", training_lr)),
        "gate": float(param_group_cfg.get("gate_lr", training_lr)),
        "strong_encoder": float(param_group_cfg.get("strong_encoder_lr", training_lr * 0.2)),
        "weak_encoder": float(param_group_cfg.get("weak_encoder_lr", training_lr * 0.05)),
    }
    for role, params in groups.items():
        if not params:
            continue
        param_groups.append(
            {
                "params": params,
                "name": role,
                "lr": lr_by_role[role],
                "param_count": _param_count(params),
            }
        )
    return param_groups


def _parameter_role(name: str, strong_modalities: set[str]) -> str:
    if name.startswith("encoders."):
        parts = name.split(".")
        modality = parts[1] if len(parts) > 1 else ""
        return "strong_encoder" if modality in strong_modalities else "weak_encoder"
    if name.startswith(("reliability_estimator", "router")):
        return "gate"
    if name.startswith(("prediction_head", "unimodal_head")):
        return "head"
    return "fusion"


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
    "build_distiller",
    "build_metrics",
    "build_model",
    "build_optimizer",
    "build_scheduler",
    "build_task_criterion",
    "optimizer_param_group_summary",
]
