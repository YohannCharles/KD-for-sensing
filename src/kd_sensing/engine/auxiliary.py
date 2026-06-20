from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.engine.model_output import ModelOutput
from kd_sensing.engine.objectives.metadata import AuxiliaryTaskConfig, resolve_auxiliary_task_config


@dataclass(frozen=True)
class AuxiliaryLossResult:
    total: torch.Tensor
    occlusion: torch.Tensor
    position: torch.Tensor
    diagnostics: dict[str, float]


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


__all__ = [
    "AuxiliaryLossResult",
    "AuxiliaryTaskConfig",
    "auxiliary_tasks_enabled",
    "compute_auxiliary_multitask_loss",
    "resolve_auxiliary_task_config",
]
