from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.engine.model_output import ModelOutput


@dataclass(frozen=True)
class AuxiliaryTaskConfig:
    occlusion_enabled: bool
    position_enabled: bool
    occlusion_weight: float
    position_weight: float
    pos_weight: str | float | None


@dataclass(frozen=True)
class AuxiliaryLossResult:
    total: torch.Tensor
    occlusion: torch.Tensor
    position: torch.Tensor
    diagnostics: dict[str, float]


def resolve_auxiliary_task_config(cfg: dict[str, Any]) -> AuxiliaryTaskConfig:
    loss_cfg = cfg.get("loss", {})
    data_cfg = cfg.get("data", {}).get("dataset", {})
    model_cfg = cfg.get("model", {}).get("student", {})
    auxiliary_cfg = _first_mapping(
        loss_cfg.get("auxiliary"),
        loss_cfg.get("multitask"),
        loss_cfg.get("multi_task"),
    )
    dataset_occlusion = _enabled(data_cfg.get("occlusion_target"))
    dataset_position = _enabled(data_cfg.get("position_target"))
    heads_cfg = _first_mapping(model_cfg.get("auxiliary_heads"))
    head_occlusion = _head_enabled(heads_cfg, "occlusion")
    head_position = _head_enabled(heads_cfg, "position")

    occlusion_cfg = _first_mapping(auxiliary_cfg.get("occlusion"), loss_cfg.get("occlusion"))
    position_cfg = _first_mapping(auxiliary_cfg.get("position"), loss_cfg.get("position"))
    auxiliary_enabled = bool(auxiliary_cfg.get("enabled", dataset_occlusion or dataset_position))

    occlusion_enabled = bool(
        auxiliary_enabled
        and (
            _enabled(occlusion_cfg, default=dataset_occlusion or head_occlusion)
            or dataset_occlusion
            or float(auxiliary_cfg.get("occlusion_weight", auxiliary_cfg.get("lambda_occlusion", 0.0))) > 0.0
        )
    )
    position_enabled = bool(
        auxiliary_enabled
        and (
            _enabled(position_cfg, default=dataset_position or head_position)
            or dataset_position
            or float(auxiliary_cfg.get("position_weight", auxiliary_cfg.get("lambda_position", 0.0))) > 0.0
        )
    )

    occlusion_weight = _weight_value(
        occlusion_cfg,
        auxiliary_cfg,
        keys=("weight", "occlusion_weight", "lambda_occlusion"),
        default=1.0,
    )
    position_weight = _weight_value(
        position_cfg,
        auxiliary_cfg,
        keys=("weight", "position_weight", "lambda_position"),
        default=0.01,
    )
    pos_weight = occlusion_cfg.get("pos_weight", auxiliary_cfg.get("pos_weight", None))
    return AuxiliaryTaskConfig(
        occlusion_enabled=occlusion_enabled,
        position_enabled=position_enabled,
        occlusion_weight=occlusion_weight,
        position_weight=position_weight,
        pos_weight=pos_weight,
    )


def auxiliary_tasks_enabled(cfg: dict[str, Any]) -> bool:
    task_cfg = resolve_auxiliary_task_config(cfg)
    return task_cfg.occlusion_enabled or task_cfg.position_enabled


def compute_auxiliary_multitask_loss(
    model_output: ModelOutput,
    targets: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    *,
    reference: torch.Tensor,
) -> AuxiliaryLossResult:
    task_cfg = resolve_auxiliary_task_config(cfg)
    zero = reference.sum() * 0.0
    occlusion_loss = zero
    position_loss = zero
    diagnostics: dict[str, float] = {}

    if task_cfg.occlusion_enabled:
        logits = _diagnostic_tensor(model_output, "occlusion_logits", "model auxiliary output")
        labels = _target_tensor(targets, "occlusion_label", "dataset target")
        valid = targets.get("occlusion_valid")
        if valid is None:
            valid = torch.ones_like(labels, dtype=torch.bool)
        logits, labels, valid = _align_occlusion(logits, labels, valid)
        element_loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
            pos_weight=_resolve_pos_weight(task_cfg.pos_weight, labels, valid),
        )
        occlusion_loss = _masked_mean(element_loss, valid, zero)
        diagnostics["loss/occlusion"] = float(occlusion_loss.detach().cpu().item())
        diagnostics["auxiliary/occlusion_valid"] = float(valid.sum().detach().cpu().item())

    if task_cfg.position_enabled:
        prediction = _diagnostic_tensor(model_output, "position", "model auxiliary output")
        target = _target_tensor(targets, "position_target", "dataset target")
        valid = targets.get("position_valid")
        if valid is None:
            valid = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
        prediction, target, valid = _align_position(prediction, target, valid)
        per_slot = (prediction - target).pow(2).mean(dim=-1)
        position_loss = _masked_mean(per_slot, valid, zero)
        diagnostics["loss/position"] = float(position_loss.detach().cpu().item())
        diagnostics["auxiliary/position_valid"] = float(valid.sum().detach().cpu().item())

    total = task_cfg.occlusion_weight * occlusion_loss + task_cfg.position_weight * position_loss
    diagnostics["loss/multitask_total"] = float(total.detach().cpu().item())
    return AuxiliaryLossResult(
        total=total,
        occlusion=occlusion_loss,
        position=position_loss,
        diagnostics=diagnostics,
    )


def _diagnostic_tensor(model_output: ModelOutput, key: str, source: str) -> torch.Tensor:
    value = model_output.diagnostics.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"Auxiliary loss is enabled but {source} '{key}' is missing.")
    return value


def _target_tensor(targets: dict[str, torch.Tensor], key: str, source: str) -> torch.Tensor:
    value = targets.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"Auxiliary loss is enabled but {source} '{key}' is missing.")
    return value


def _align_occlusion(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"occlusion_logits must have shape [B, H], got {tuple(logits.shape)}.")
    if labels.shape != logits.shape:
        raise ValueError(
            f"occlusion_label shape {tuple(labels.shape)} does not match occlusion_logits {tuple(logits.shape)}."
        )
    if valid.shape != logits.shape:
        raise ValueError(
            f"occlusion_valid shape {tuple(valid.shape)} does not match occlusion_logits {tuple(logits.shape)}."
        )
    return logits, labels.to(dtype=logits.dtype), valid.to(device=logits.device, dtype=torch.bool)


def _align_position(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prediction.ndim != 3 or prediction.shape[-1] != 2:
        raise ValueError(f"position output must have shape [B, H, 2], got {tuple(prediction.shape)}.")
    if target.shape != prediction.shape:
        raise ValueError(f"position_target shape {tuple(target.shape)} does not match output {tuple(prediction.shape)}.")
    if valid.shape != prediction.shape[:2]:
        raise ValueError(f"position_valid shape {tuple(valid.shape)} does not match output {tuple(prediction.shape)}.")
    return prediction, target.to(dtype=prediction.dtype), valid.to(device=prediction.device, dtype=torch.bool)


def _masked_mean(values: torch.Tensor, valid: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    valid_f = valid.to(device=values.device, dtype=values.dtype)
    denom = valid_f.sum()
    if denom.item() <= 0:
        return zero
    return (values * valid_f).sum() / denom.clamp_min(1.0)


def _resolve_pos_weight(spec: str | float | None, labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor | None:
    if spec is None or str(spec).lower() in {"none", "false", "0"}:
        return None
    if isinstance(spec, str) and spec.lower() == "auto":
        valid_labels = labels[valid]
        positives = valid_labels.sum()
        negatives = valid_labels.numel() - positives
        if positives.item() <= 0:
            return torch.ones((), dtype=labels.dtype, device=labels.device)
        return (negatives / positives.clamp_min(1.0)).to(dtype=labels.dtype, device=labels.device)
    return torch.tensor(float(spec), dtype=labels.dtype, device=labels.device)


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _enabled(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return bool(value.get("enabled", value.get("enable", default)))
    return default


def _head_enabled(heads_cfg: dict[str, Any], key: str) -> bool:
    if not heads_cfg:
        return False
    return bool(heads_cfg.get(key, heads_cfg.get(f"{key}_head", heads_cfg.get("enabled", False))))


def _weight_value(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    *,
    keys: tuple[str, ...],
    default: float,
) -> float:
    for key in keys:
        if key in primary:
            return float(primary[key])
    for key in keys:
        if key in secondary:
            return float(secondary[key])
    return float(default)


__all__ = [
    "AuxiliaryLossResult",
    "AuxiliaryTaskConfig",
    "auxiliary_tasks_enabled",
    "compute_auxiliary_multitask_loss",
    "resolve_auxiliary_task_config",
]
