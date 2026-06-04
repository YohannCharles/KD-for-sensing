from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.losses.circular import circular_soft_ce_loss


@dataclass(frozen=True)
class ResidualLossConfig:
    final_circular_soft_ce_weight: float = 1.0
    modality_aux_ce_weight: float = 0.1
    gate_bce_weight: float = 0.1
    good_anchor_kl_weight: float = 0.1
    correction_l2_weight: float = 0.001
    hard_sample_weight: float = 1.0
    good_error_threshold: float = 4.0
    circular_sigma: float = 2.0
    ignore_index: int = -100

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ResidualLossConfig":
        raw = dict(payload or {})
        allowed = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in raw.items() if key in allowed})


class ResidualFusionLoss(nn.Module):
    def __init__(self, cfg: ResidualLossConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.cfg = cfg if isinstance(cfg, ResidualLossConfig) else ResidualLossConfig.from_mapping(cfg)

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        labels = torch.as_tensor(batch["target_label"], device=outputs["final_logits"].device, dtype=torch.long)
        gps_error = torch.as_tensor(batch["gps_error"], device=outputs["final_logits"].device, dtype=torch.float32)
        roles = batch.get("support_query_role", [])
        train_mask = support_role_mask(roles, device=labels.device)
        valid_label = labels.ne(int(self.cfg.ignore_index))
        train_valid = train_mask & valid_label
        per_sample_final = circular_soft_ce_loss(
            outputs["final_logits"],
            labels,
            sigma=float(self.cfg.circular_sigma),
            ignore_index=int(self.cfg.ignore_index),
            reduction="none",
        )
        hard_mask = gps_error.ge(float(self.cfg.good_error_threshold))
        weights = torch.ones_like(per_sample_final)
        weights = torch.where(hard_mask, weights + float(self.cfg.hard_sample_weight), weights)
        final_ce = _masked_mean(per_sample_final * weights, train_valid)

        aux_ce = circular_soft_ce_loss(
            outputs["modality_only_logits"],
            labels,
            sigma=float(self.cfg.circular_sigma),
            ignore_index=int(self.cfg.ignore_index),
            reduction="none",
        )
        aux_ce = _masked_mean(aux_ce, train_valid)

        gate_target = gate_target_from_gps_error(
            gps_error,
            roles,
            threshold=float(self.cfg.good_error_threshold),
            allow_query=False,
            device=outputs["correction_gate"].device,
        )
        gate_mask = train_valid.view(-1, 1)
        gate_bce_raw = F.binary_cross_entropy(outputs["correction_gate"], gate_target, reduction="none")
        gate_bce = _masked_mean(gate_bce_raw.squeeze(-1), gate_mask.squeeze(-1))

        good_mask = train_valid & gps_error.lt(float(self.cfg.good_error_threshold))
        prior_probs = torch.as_tensor(batch["gps_prior_probs"], device=outputs["final_logits"].device, dtype=torch.float32)
        final_log_probs = F.log_softmax(outputs["final_logits"], dim=-1)
        anchor_raw = F.kl_div(final_log_probs, prior_probs, reduction="none").sum(dim=-1)
        good_anchor = _masked_mean(anchor_raw, good_mask)

        correction_l2 = outputs["correction_logits"].pow(2).mean()
        total = (
            float(self.cfg.final_circular_soft_ce_weight) * final_ce
            + float(self.cfg.modality_aux_ce_weight) * aux_ce
            + float(self.cfg.gate_bce_weight) * gate_bce
            + float(self.cfg.good_anchor_kl_weight) * good_anchor
            + float(self.cfg.correction_l2_weight) * correction_l2
        )
        return {
            "loss": total,
            "final_ce": final_ce,
            "modality_aux_ce": aux_ce,
            "gate_bce": gate_bce,
            "good_anchor_kl": good_anchor,
            "correction_l2": correction_l2,
            "hard_sample_weight_mean": weights[train_valid].mean() if bool(train_valid.any()) else weights.new_tensor(0.0),
            "train_sample_count": train_valid.to(torch.float32).sum(),
            "query_label_used_for_training": torch.tensor(False, device=labels.device),
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
        return roles.to(device=device, dtype=torch.bool)
    return torch.as_tensor([str(role) == "support" for role in roles], dtype=torch.bool, device=device)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    masked = torch.where(mask, values, torch.zeros_like(values))
    return masked.sum() / mask.to(values.dtype).sum().clamp_min(1.0)

