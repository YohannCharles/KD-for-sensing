from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.losses.circular import circular_soft_ce_loss


@dataclass(frozen=True)
class CameraResidualLossConfig:
    final_circular_soft_ce_weight: float = 1.0
    residual_delta_ce_weight: float = 0.5
    gate_bce_weight: float = 0.1
    good_anchor_kl_weight: float = 0.1
    aux_direct_ce_weight: float = 0.0
    gate_entropy_weight: float = 0.0
    hard_sample_weight: float = 1.0
    good_error_threshold: float = 4.0
    circular_sigma: float = 2.0
    ignore_index: int = -100

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "CameraResidualLossConfig":
        raw = dict(payload or {})
        allowed = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: raw[key] for key in raw if key in allowed})


class CameraResidualLoss(nn.Module):
    def __init__(self, cfg: CameraResidualLossConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.cfg = cfg if isinstance(cfg, CameraResidualLossConfig) else CameraResidualLossConfig.from_mapping(cfg)

    def forward(self, outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        final_logits = outputs["final_logits"]
        device = final_logits.device
        labels = torch.as_tensor(batch["target_label"], device=device, dtype=torch.long).reshape(-1)
        gps_error = torch.as_tensor(batch["gps_error"], device=device, dtype=torch.float32).reshape(-1)
        roles = batch.get("split_role", batch.get("support_query_role", []))
        train_mask = support_role_mask(roles, device=device)
        valid_label = labels.ne(int(self.cfg.ignore_index)) & labels.ge(0)
        train_valid = train_mask & valid_label
        hard_mask = gps_error.ge(float(self.cfg.good_error_threshold))
        final_raw = circular_soft_ce_loss(
            final_logits,
            labels,
            sigma=float(self.cfg.circular_sigma),
            ignore_index=int(self.cfg.ignore_index),
            reduction="none",
        )
        weights = torch.ones_like(final_raw)
        weights = torch.where(hard_mask, weights + float(self.cfg.hard_sample_weight), weights)
        final_ce = _masked_mean(final_raw * weights, train_valid)

        residual_target = torch.as_tensor(batch["residual_delta_class"], device=device, dtype=torch.long).reshape(-1)
        residual_ce_raw = F.cross_entropy(outputs["residual_delta_logits"], residual_target, reduction="none")
        residual_delta_ce = _masked_mean(residual_ce_raw, train_valid)

        gate_target = gate_target_from_gps_error(
            gps_error,
            roles,
            threshold=float(self.cfg.good_error_threshold),
            allow_query=False,
            device=device,
        )
        gate = outputs["correction_gate"].reshape(-1, 1)
        gate_bce_raw = F.binary_cross_entropy(gate, gate_target, reduction="none").reshape(-1)
        gate_bce = _masked_mean(gate_bce_raw, train_valid)

        good_mask = train_valid & gps_error.lt(float(self.cfg.good_error_threshold))
        prior_probs = outputs.get("p_gps")
        if prior_probs is None:
            prior_probs = torch.as_tensor(batch["gps_prior_probs"], device=device, dtype=torch.float32)
        final_log_probs = F.log_softmax(final_logits, dim=-1)
        good_anchor_raw = F.kl_div(final_log_probs, prior_probs.to(device=device, dtype=torch.float32), reduction="none").sum(dim=-1)
        good_anchor_kl = _masked_mean(good_anchor_raw, good_mask)

        aux_direct_ce = final_logits.new_tensor(0.0)
        if "direct_beam_logits" in outputs:
            aux_raw = circular_soft_ce_loss(
                outputs["direct_beam_logits"],
                labels,
                sigma=float(self.cfg.circular_sigma),
                ignore_index=int(self.cfg.ignore_index),
                reduction="none",
            )
            aux_direct_ce = _masked_mean(aux_raw, train_valid)

        entropy = -(gate.clamp_min(1e-8).log() * gate + (1.0 - gate).clamp_min(1e-8).log() * (1.0 - gate)).reshape(-1)
        gate_entropy = _masked_mean(entropy, train_valid)
        total = (
            float(self.cfg.final_circular_soft_ce_weight) * final_ce
            + float(self.cfg.residual_delta_ce_weight) * residual_delta_ce
            + float(self.cfg.gate_bce_weight) * gate_bce
            + float(self.cfg.good_anchor_kl_weight) * good_anchor_kl
            + float(self.cfg.aux_direct_ce_weight) * aux_direct_ce
            + float(self.cfg.gate_entropy_weight) * gate_entropy
        )
        return {
            "loss": total,
            "final_circular_soft_ce": final_ce,
            "residual_delta_ce": residual_delta_ce,
            "gate_bce": gate_bce,
            "good_anchor_kl": good_anchor_kl,
            "aux_direct_ce": aux_direct_ce,
            "gate_entropy": gate_entropy,
            "hard_sample_weight_mean": weights[train_valid].mean() if bool(train_valid.any()) else weights.new_tensor(0.0),
            "train_sample_count": train_valid.to(torch.float32).sum(),
            "good_anchor_sample_count": good_mask.to(torch.float32).sum(),
            "query_label_used_for_training": torch.tensor(False, device=device),
        }


def gate_target_from_gps_error(
    gps_error: torch.Tensor,
    roles: Sequence[str] | torch.Tensor,
    *,
    threshold: float = 4.0,
    allow_query: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    values = torch.as_tensor(gps_error, device=device, dtype=torch.float32).reshape(-1, 1)
    target = values.ge(float(threshold)).to(torch.float32)
    if not allow_query:
        mask = support_role_mask(roles, device=values.device).reshape(-1, 1)
        target = torch.where(mask, target, torch.zeros_like(target))
    return target


def support_role_mask(roles: Sequence[str] | torch.Tensor, *, device: torch.device | None = None) -> torch.Tensor:
    if torch.is_tensor(roles):
        return roles.to(device=device, dtype=torch.bool).reshape(-1)
    return torch.as_tensor([str(role) == "support" for role in roles], dtype=torch.bool, device=device)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    masked = torch.where(mask, values, torch.zeros_like(values))
    return masked.sum() / mask.to(values.dtype).sum().clamp_min(1.0)


__all__ = [
    "CameraResidualLoss",
    "CameraResidualLossConfig",
    "gate_target_from_gps_error",
    "support_role_mask",
]
