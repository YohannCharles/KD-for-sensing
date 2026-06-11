from __future__ import annotations

from copy import deepcopy
from fnmatch import fnmatchcase
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
    optimizer_cfg = training_cfg.get("optimizer", {})
    if optimizer_cfg is None:
        optimizer_cfg = {}
    if isinstance(optimizer_cfg, str):
        optimizer_cfg = {"type": optimizer_cfg}
    if not isinstance(optimizer_cfg, dict):
        raise ValueError("training.optimizer must be a mapping when provided.")
    trainable_named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable_named_params:
        raise ValueError("No trainable parameters found for optimizer.")
    trainable_params = [param for _, param in trainable_named_params]
    parameter_group_cfgs = optimizer_cfg.get("parameter_groups")
    if parameter_group_cfgs is None:
        return _build_optimizer_instance(
            optimizer_cfg,
            [{"params": trainable_params, "name": "main", "param_count": _param_count(trainable_params)}],
            lr=training_cfg.get("lr", optimizer_cfg.get("lr", 7.5e-4)),
            weight_decay=training_cfg.get("weight_decay", optimizer_cfg.get("weight_decay", 0.0)),
        )
    allow_unmatched_patterns = bool(optimizer_cfg.get("allow_unmatched_patterns", False))
    if "strict" in optimizer_cfg:
        allow_unmatched_patterns = not bool(optimizer_cfg.get("strict", True))
    param_groups = _build_parameter_groups(
        parameter_group_cfgs,
        trainable_named_params=trainable_named_params,
        default_lr=float(optimizer_cfg.get("lr", training_cfg.get("lr", 7.5e-4))),
        default_weight_decay=float(optimizer_cfg.get("weight_decay", training_cfg.get("weight_decay", 0.0))),
        allow_unmatched_patterns=allow_unmatched_patterns,
        require_all_matched=bool(optimizer_cfg.get("require_all_matched", False)),
    )
    return _build_optimizer_instance(
        optimizer_cfg,
        param_groups,
        lr=training_cfg.get("lr", optimizer_cfg.get("lr", 7.5e-4)),
        weight_decay=training_cfg.get("weight_decay", optimizer_cfg.get("weight_decay", 0.0)),
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


def _build_parameter_groups(
    raw_groups: Any,
    *,
    trainable_named_params: list[tuple[str, torch.nn.Parameter]],
    default_lr: float,
    default_weight_decay: float,
    allow_unmatched_patterns: bool,
    require_all_matched: bool,
) -> list[dict[str, Any]]:
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("training.optimizer.parameter_groups must be a non-empty list when provided.")
    grouped_param_ids: set[int] = set()
    grouped_param_names: set[str] = set()
    groups: list[dict[str, Any]] = []
    for index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, dict):
            raise ValueError(f"optimizer parameter group {index} must be a mapping.")
        name = str(raw_group.get("name") or f"group_{index}")
        patterns = raw_group.get("module_patterns", raw_group.get("patterns", raw_group.get("pattern")))
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list) or not patterns:
            raise ValueError(f"optimizer parameter group '{name}' must declare non-empty module_patterns.")
        matched: list[tuple[str, torch.nn.Parameter]] = []
        unmatched_patterns: list[str] = []
        for pattern in [str(item) for item in patterns]:
            pattern_matches = [
                (param_name, param)
                for param_name, param in trainable_named_params
                if _parameter_name_matches(param_name, pattern)
            ]
            if not pattern_matches:
                unmatched_patterns.append(pattern)
            matched.extend(pattern_matches)
        if unmatched_patterns and not allow_unmatched_patterns:
            prefixes = _available_parameter_prefixes(trainable_named_params)
            raise ValueError(
                f"optimizer parameter group '{name}' patterns did not match any trainable parameters: "
                f"{unmatched_patterns}. "
                f"Available parameter prefixes include: {prefixes}."
            )
        unique: list[tuple[str, torch.nn.Parameter]] = []
        seen_names: set[str] = set()
        for param_name, param in matched:
            if param_name in seen_names:
                continue
            seen_names.add(param_name)
            if id(param) in grouped_param_ids:
                raise ValueError(
                    f"optimizer parameter '{param_name}' matched by multiple optimizer parameter groups; "
                    f"current group '{name}', previous grouped parameters include {sorted(grouped_param_names)[:8]}."
                )
            unique.append((param_name, param))
        if not unique:
            raise ValueError(f"optimizer parameter group '{name}' matched no trainable parameters.")
        params = [param for _, param in unique]
        grouped_param_ids.update(id(param) for param in params)
        grouped_param_names.update(param_name for param_name, _ in unique)
        groups.append(
            {
                "params": params,
                "name": name,
                "lr": float(raw_group.get("lr", default_lr)),
                "weight_decay": float(raw_group.get("weight_decay", default_weight_decay)),
                "param_count": _param_count(params),
                "matched_patterns": [str(item) for item in patterns],
                "parameter_names": [param_name for param_name, _ in unique],
            }
        )
    remaining = [
        (param_name, param)
        for param_name, param in trainable_named_params
        if id(param) not in grouped_param_ids
    ]
    if remaining and require_all_matched:
        raise ValueError(
            "training.optimizer.require_all_matched=true but unmatched trainable parameters remain: "
            f"{[name for name, _ in remaining[:12]]}."
        )
    if remaining:
        remaining_params = [param for _, param in remaining]
        groups.append(
            {
                "params": remaining_params,
                "name": "main",
                "lr": default_lr,
                "weight_decay": default_weight_decay,
                "param_count": _param_count(remaining_params),
                "matched_patterns": [],
                "parameter_names": [param_name for param_name, _ in remaining],
            }
        )
    if not groups:
        raise ValueError("No trainable parameters found for optimizer parameter groups.")
    return groups


def _parameter_name_matches(param_name: str, pattern: str) -> bool:
    normalized = pattern.strip()
    if not normalized:
        return False
    return (
        param_name == normalized
        or param_name.startswith(f"{normalized}.")
        or fnmatchcase(param_name, normalized)
        or fnmatchcase(param_name, f"{normalized}.*")
    )


def _available_parameter_prefixes(trainable_named_params: list[tuple[str, torch.nn.Parameter]]) -> list[str]:
    prefixes = sorted({name.split(".")[0] for name, _ in trainable_named_params})
    return prefixes[:20]


def _build_optimizer_instance(
    optimizer_cfg: dict[str, Any],
    param_groups: list[dict[str, Any]],
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    optimizer_type = str(optimizer_cfg.get("type", "adam")).strip().lower()
    if optimizer_type == "adam":
        return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
    if optimizer_type == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer type {optimizer_type!r}; supported types are 'adam' and 'adamw'.")


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
