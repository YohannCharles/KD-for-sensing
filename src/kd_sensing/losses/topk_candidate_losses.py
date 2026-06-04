from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.evaluation.metrics import circular_beam_distance


@dataclass(frozen=True)
class TopKCandidateSelectorLossConfig:
    candidate_soft_ce_weight: float = 1.0
    target_index_ce_weight: float = 0.5
    miss_bce_weight: float = 0.1
    prior_anchor_kl_weight: float = 0.1
    entropy_regularization_weight: float = 0.0
    hard_rank_weight: float = 2.0
    good_error_threshold: float = 4.0
    circular_sigma: float = 2.0
    ignore_index: int = -100
    num_beams: int = 64

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "TopKCandidateSelectorLossConfig":
        raw = dict(payload or {})
        allowed = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: raw[key] for key in raw if key in allowed})


class TopKCandidateSelectorLoss(nn.Module):
    def __init__(self, cfg: TopKCandidateSelectorLossConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.cfg = cfg if isinstance(cfg, TopKCandidateSelectorLossConfig) else TopKCandidateSelectorLossConfig.from_mapping(cfg)

    def forward(self, outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        scores = outputs["final_candidate_scores"]
        device = scores.device
        candidate_beams = torch.as_tensor(batch["candidate_beams"], device=device, dtype=torch.long)
        labels = torch.as_tensor(batch["target_label"], device=device, dtype=torch.long).reshape(-1)
        target_in_topk = torch.as_tensor(batch["target_in_top8"], device=device, dtype=torch.bool).reshape(-1)
        target_index = torch.as_tensor(batch["target_candidate_index"], device=device, dtype=torch.long).reshape(-1)
        nearest_index = torch.as_tensor(batch.get("nearest_candidate_index", target_index), device=device, dtype=torch.long).reshape(-1)
        miss_label = torch.as_tensor(batch.get("miss_label", ~target_in_topk), device=device, dtype=torch.float32).reshape(-1, 1)
        gps_error = torch.as_tensor(batch.get("gps_error", torch.full_like(labels, 999.0)), device=device, dtype=torch.float32).reshape(-1)
        roles = batch.get("split_role", batch.get("support_query_role", []))
        train_mask = support_role_mask(roles, device=device)
        if int(train_mask.numel()) == 0:
            train_mask = torch.ones_like(labels, dtype=torch.bool, device=device)
        valid_label = labels.ne(int(self.cfg.ignore_index)) & labels.ge(0)
        train_valid = train_mask & valid_label

        soft_target = candidate_circular_soft_target(
            candidate_beams,
            labels,
            sigma=float(self.cfg.circular_sigma),
            num_beams=int(self.cfg.num_beams),
            ignore_index=int(self.cfg.ignore_index),
        )
        hard_mask = target_in_topk & target_index.gt(0)
        sample_weight = torch.where(
            hard_mask,
            scores.new_full((scores.shape[0],), float(self.cfg.hard_rank_weight)),
            scores.new_ones((scores.shape[0],)),
        )
        log_probs = F.log_softmax(scores, dim=-1)
        candidate_soft_raw = -(soft_target * log_probs).sum(dim=-1)
        candidate_soft_ce = _masked_mean(candidate_soft_raw * sample_weight, train_valid)

        ce_mask = train_valid & target_in_topk & target_index.ge(0)
        safe_index = torch.where(target_index.ge(0), target_index, torch.zeros_like(target_index))
        index_raw = F.cross_entropy(scores, safe_index, reduction="none")
        target_index_ce = _masked_mean(index_raw * sample_weight, ce_mask)

        miss_logit = outputs["miss_logit"].reshape(-1, 1)
        miss_raw = F.binary_cross_entropy_with_logits(miss_logit, miss_label, reduction="none").reshape(-1)
        miss_bce = _masked_mean(miss_raw, train_valid)

        candidate_probs = outputs.get("candidate_probs", F.softmax(scores, dim=-1))
        gps_candidate_probs = torch.as_tensor(
            batch.get("candidate_probs", candidate_probs.detach()),
            device=device,
            dtype=scores.dtype,
        )
        gps_candidate_probs = gps_candidate_probs / gps_candidate_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        good_mask = train_valid & gps_error.lt(float(self.cfg.good_error_threshold))
        anchor_raw = F.kl_div(
            torch.log(candidate_probs.clamp_min(1e-12)),
            gps_candidate_probs,
            reduction="none",
        ).sum(dim=-1)
        prior_anchor_kl = _masked_mean(anchor_raw, good_mask)

        entropy_raw = -(candidate_probs.clamp_min(1e-12).log() * candidate_probs).sum(dim=-1)
        entropy_regularization = _masked_mean(entropy_raw, train_valid)
        total = (
            float(self.cfg.candidate_soft_ce_weight) * candidate_soft_ce
            + float(self.cfg.target_index_ce_weight) * target_index_ce
            + float(self.cfg.miss_bce_weight) * miss_bce
            + float(self.cfg.prior_anchor_kl_weight) * prior_anchor_kl
            + float(self.cfg.entropy_regularization_weight) * entropy_regularization
        )
        return {
            "loss": total,
            "candidate_soft_ce": candidate_soft_ce,
            "target_index_ce": target_index_ce,
            "miss_bce": miss_bce,
            "prior_anchor_kl": prior_anchor_kl,
            "entropy_regularization": entropy_regularization,
            "hard_sample_weight_mean": sample_weight[train_valid].mean() if bool(train_valid.any()) else scores.new_tensor(0.0),
            "train_sample_count": train_valid.to(torch.float32).sum(),
            "target_index_ce_sample_count": ce_mask.to(torch.float32).sum(),
            "good_anchor_sample_count": good_mask.to(torch.float32).sum(),
            "query_label_used_for_training": torch.tensor(False, device=device),
            "nearest_candidate_index_used_for_miss": nearest_index.ge(0).to(torch.float32).sum(),
        }


def candidate_circular_soft_target(
    candidate_beams: torch.Tensor,
    target: torch.Tensor,
    *,
    sigma: float = 2.0,
    num_beams: int = 64,
    ignore_index: int = -100,
) -> torch.Tensor:
    beams = torch.as_tensor(candidate_beams, dtype=torch.long, device=candidate_beams.device)
    target_tensor = torch.as_tensor(target, dtype=torch.long, device=beams.device).reshape(-1)
    valid = target_tensor.ne(int(ignore_index)) & target_tensor.ge(0)
    safe_target = target_tensor.clamp_min(0).remainder(int(num_beams))
    dist = circular_beam_distance(beams, safe_target.unsqueeze(-1), num_beams=int(num_beams)).to(torch.float32)
    weights = torch.exp(-(dist**2) / (2.0 * max(float(sigma), 1e-6) ** 2))
    weights = weights * valid.unsqueeze(-1).to(weights.dtype)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def support_role_mask(roles: Sequence[str] | torch.Tensor, *, device: torch.device | None = None) -> torch.Tensor:
    if torch.is_tensor(roles):
        return roles.to(device=device, dtype=torch.bool).reshape(-1)
    return torch.as_tensor(
        [
            str(role) in {"support", "source", "source_train", "train", "target_support", "target_support_train"}
            for role in roles
        ],
        dtype=torch.bool,
        device=device,
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    masked = torch.where(mask, values, torch.zeros_like(values))
    return masked.sum() / mask.to(values.dtype).sum().clamp_min(1.0)


__all__ = [
    "TopKCandidateSelectorLoss",
    "TopKCandidateSelectorLossConfig",
    "candidate_circular_soft_target",
    "support_role_mask",
]
