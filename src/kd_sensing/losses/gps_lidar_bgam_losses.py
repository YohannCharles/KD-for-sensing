from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.losses.topk_candidate_losses import candidate_circular_soft_target, support_role_mask


@dataclass(frozen=True)
class GPSLidarBGAMLossConfig:
    candidate_ce_weight: float = 1.0
    candidate_soft_ce_weight: float = 0.5
    nearest_miss_soft_weight: float = 0.25
    full64_ce_weight: float = 0.0
    prior_anchor_kl_weight: float = 0.05
    circular_sigma: float = 2.0
    ignore_index: int = -100
    num_beams: int = 64

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "GPSLidarBGAMLossConfig":
        raw = dict(payload or {})
        allowed = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: raw[key] for key in raw if key in allowed})


class GPSLidarBGAMLoss(nn.Module):
    def __init__(self, cfg: GPSLidarBGAMLossConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.cfg = cfg if isinstance(cfg, GPSLidarBGAMLossConfig) else GPSLidarBGAMLossConfig.from_mapping(cfg)

    def forward(self, outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        scores = outputs["final_candidate_scores"]
        device = scores.device
        candidate_beams = torch.as_tensor(batch["candidate_beams"], device=device, dtype=torch.long)
        labels = torch.as_tensor(batch.get("gt_beam", batch.get("target_label")), device=device, dtype=torch.long).reshape(-1)
        target_index = torch.as_tensor(batch.get("target_candidate_index", torch.full_like(labels, -1)), device=device, dtype=torch.long).reshape(-1)
        if (target_index < 0).any() and labels.numel():
            target_index = _target_index_from_candidates(candidate_beams, labels, target_index, ignore_index=int(self.cfg.ignore_index))
        nearest_index = torch.as_tensor(batch.get("nearest_candidate_index", target_index), device=device, dtype=torch.long).reshape(-1)
        roles = batch.get("split_role", batch.get("support_query_role", []))
        train_mask = support_role_mask(roles, device=device)
        if int(train_mask.numel()) == 0:
            train_mask = torch.ones_like(labels, dtype=torch.bool, device=device)
        valid_label = labels.ne(int(self.cfg.ignore_index)) & labels.ge(0)
        train_valid = train_mask & valid_label
        hit_mask = train_valid & target_index.ge(0)
        miss_mask = train_valid & target_index.lt(0) & nearest_index.ge(0)

        safe_index = torch.where(target_index.ge(0), target_index, torch.zeros_like(target_index))
        ce_raw = F.cross_entropy(scores, safe_index, reduction="none")
        candidate_ce = _masked_mean(ce_raw, hit_mask)

        soft_target = candidate_circular_soft_target(
            candidate_beams,
            labels,
            sigma=float(self.cfg.circular_sigma),
            num_beams=int(self.cfg.num_beams),
            ignore_index=int(self.cfg.ignore_index),
        )
        log_probs = F.log_softmax(scores, dim=-1)
        candidate_soft_ce = _masked_mean(-(soft_target * log_probs).sum(dim=-1), train_valid)

        nearest_safe = torch.where(nearest_index.ge(0), nearest_index, torch.zeros_like(nearest_index))
        nearest_raw = F.cross_entropy(scores, nearest_safe, reduction="none")
        nearest_miss_soft = _masked_mean(nearest_raw, miss_mask)

        full64_ce = scores.new_tensor(0.0)
        if "logits64" in outputs and float(self.cfg.full64_ce_weight) > 0:
            full_raw = F.cross_entropy(outputs["logits64"], labels.clamp_min(0), reduction="none")
            full64_ce = _masked_mean(full_raw, train_valid)

        candidate_probs = outputs.get("candidate_probs", torch.softmax(scores, dim=-1))
        gps_candidate_probs = torch.as_tensor(batch.get("candidate_probs", candidate_probs.detach()), device=device, dtype=scores.dtype)
        gps_candidate_probs = gps_candidate_probs / gps_candidate_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        anchor_raw = F.kl_div(torch.log(candidate_probs.clamp_min(1e-12)), gps_candidate_probs, reduction="none").sum(dim=-1)
        prior_anchor_kl = _masked_mean(anchor_raw, train_valid)

        loss = (
            float(self.cfg.candidate_ce_weight) * candidate_ce
            + float(self.cfg.candidate_soft_ce_weight) * candidate_soft_ce
            + float(self.cfg.nearest_miss_soft_weight) * nearest_miss_soft
            + float(self.cfg.full64_ce_weight) * full64_ce
            + float(self.cfg.prior_anchor_kl_weight) * prior_anchor_kl
        )
        return {
            "loss": loss,
            "candidate_ce": candidate_ce,
            "candidate_soft_ce": candidate_soft_ce,
            "nearest_miss_soft": nearest_miss_soft,
            "full64_ce": full64_ce,
            "prior_anchor_kl": prior_anchor_kl,
            "train_sample_count": train_valid.to(torch.float32).sum(),
            "rerank_ce_sample_count": hit_mask.to(torch.float32).sum(),
            "skipped_rerank_sample_count": (train_valid & target_index.lt(0)).to(torch.float32).sum(),
            "nearest_miss_sample_count": miss_mask.to(torch.float32).sum(),
            "query_label_used_for_training": torch.tensor(False, device=device),
        }


def _target_index_from_candidates(candidate_beams: torch.Tensor, labels: torch.Tensor, current: torch.Tensor, *, ignore_index: int) -> torch.Tensor:
    valid = labels.ne(int(ignore_index)) & labels.ge(0)
    matches = candidate_beams.eq(labels.clamp_min(0).unsqueeze(-1))
    has_match = matches.any(dim=-1) & valid
    found = matches.to(torch.long).argmax(dim=-1)
    return torch.where(current.ge(0), current, torch.where(has_match, found, torch.full_like(current, -1)))


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    return torch.where(mask, values, torch.zeros_like(values)).sum() / mask.to(values.dtype).sum().clamp_min(1.0)


__all__ = ["GPSLidarBGAMLoss", "GPSLidarBGAMLossConfig"]
