from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.evaluation.metrics import circular_window


class BeamCandidateAttentionReranker(nn.Module):
    """Minimal candidate reranker over GPS top-K union local circular beams."""

    def __init__(
        self,
        *,
        num_beams: int = 64,
        gps_topk: int = 16,
        local_radius: int = 8,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        score_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_beams = int(num_beams)
        self.gps_topk = int(gps_topk)
        self.local_radius = int(local_radius)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.score_temperature = float(score_temperature)
        self.image_proj = nn.Linear(self.feature_dim, self.hidden_dim)
        self.beam_embedding = nn.Embedding(self.num_beams, self.hidden_dim)
        self.score_head = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

    def candidate_set(
        self,
        gps_logits: torch.Tensor,
        *,
        gps_pred_top1: torch.Tensor | None = None,
    ) -> list[list[int]]:
        if gps_logits.ndim != 2 or int(gps_logits.shape[-1]) != self.num_beams:
            raise ValueError(f"gps_logits must have shape [B, {self.num_beams}], got {tuple(gps_logits.shape)}.")
        topk = torch.topk(gps_logits, min(self.gps_topk, self.num_beams), dim=-1).indices.detach().cpu()
        top1 = gps_pred_top1.detach().cpu().to(torch.long) if gps_pred_top1 is not None else gps_logits.argmax(dim=-1).detach().cpu()
        candidates: list[list[int]] = []
        for row_idx in range(int(gps_logits.shape[0])):
            ordered: list[int] = []
            for beam in topk[row_idx].tolist():
                _append_unique(ordered, int(beam), self.num_beams)
            for beam in circular_window(int(top1[row_idx]), radius=self.local_radius, num_beams=self.num_beams):
                _append_unique(ordered, int(beam), self.num_beams)
            candidates.append(ordered)
        return candidates

    def forward(
        self,
        *,
        gps_logits: torch.Tensor,
        camera_ae_feature: torch.Tensor,
        gps_pred_top1: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        candidates = self.candidate_set(gps_logits, gps_pred_top1=gps_pred_top1)
        candidate_tensor = _candidate_tensor(candidates, device=gps_logits.device)
        tokens = camera_ae_feature
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(1)
        if tokens.ndim != 3:
            raise ValueError(f"camera_ae_feature must have shape [B, D] or [B, N, D], got {tuple(camera_ae_feature.shape)}.")
        tokens = _fit_feature_dim(tokens, self.feature_dim)
        image_token = self.image_proj(tokens.to(device=gps_logits.device, dtype=gps_logits.dtype)).mean(dim=1)
        query = self.score_head(image_token)
        beam_embed = self.beam_embedding(candidate_tensor.clamp_min(0))
        learned_scores = (beam_embed * query.unsqueeze(1)).sum(dim=-1) / max(self.hidden_dim**0.5, 1.0)
        gps_scores = torch.gather(gps_logits, 1, candidate_tensor.clamp_min(0))
        scores = (learned_scores + gps_scores) / max(self.score_temperature, 1e-6)
        scores = scores.masked_fill(candidate_tensor.lt(0), -1e9)
        order = torch.argsort(scores, dim=-1, descending=True)
        rerank_top1 = torch.gather(candidate_tensor, 1, order[:, :1]).squeeze(-1)
        top3_width = min(3, int(order.shape[-1]))
        rerank_top3 = torch.gather(candidate_tensor, 1, order[:, :top3_width])
        result: dict[str, Any] = {
            "candidates": candidates,
            "candidate_tensor": candidate_tensor,
            "candidate_scores": scores,
            "rerank_top1": rerank_top1,
            "rerank_top3": rerank_top3,
        }
        if target is not None:
            target_flat = target.to(device=gps_logits.device, dtype=torch.long).reshape(-1)
            gps_top = torch.topk(gps_logits, min(self.gps_topk, self.num_beams), dim=-1).indices
            top1 = gps_pred_top1.to(device=gps_logits.device, dtype=torch.long).reshape(-1) if gps_pred_top1 is not None else gps_logits.argmax(dim=-1)
            in_gps_top = (gps_top == target_flat.unsqueeze(-1)).any(dim=-1)
            local = circular_window(top1, radius=self.local_radius, num_beams=self.num_beams)
            in_local = (local == target_flat.unsqueeze(-1)).any(dim=-1)
            in_union = (candidate_tensor == target_flat.unsqueeze(-1)).any(dim=-1)
            top3_hit = (rerank_top3 == target_flat.unsqueeze(-1)).any(dim=-1)
            target_positions = []
            for idx, beams in enumerate(candidates):
                truth = int(target_flat[idx].detach().cpu())
                target_positions.append(beams.index(truth) if truth in beams else -100)
            positions = torch.as_tensor(target_positions, device=gps_logits.device, dtype=torch.long)
            loss = gps_logits.new_tensor(0.0)
            if bool(in_union.any()):
                loss = F.cross_entropy(scores[in_union], positions[in_union])
            result.update(
                {
                    "loss": loss,
                    "loss_mask": in_union,
                    "target_in_gps_top16": in_gps_top,
                    "target_in_local_radius8": in_local,
                    "target_in_union_candidates": in_union,
                    "rerank_top1_hit": rerank_top1.eq(target_flat),
                    "rerank_top3_hit": top3_hit,
                }
            )
        return result


def _candidate_tensor(candidates: list[list[int]], *, device: torch.device) -> torch.Tensor:
    width = max((len(row) for row in candidates), default=0)
    tensor = torch.full((len(candidates), width), -1, device=device, dtype=torch.long)
    for idx, row in enumerate(candidates):
        if row:
            tensor[idx, : len(row)] = torch.as_tensor(row, device=device, dtype=torch.long)
    return tensor


def _fit_feature_dim(value: torch.Tensor, dim: int) -> torch.Tensor:
    current = int(value.shape[-1])
    target = int(dim)
    if current == target:
        return value
    if current > target:
        return value[..., :target]
    pad = value.new_zeros((*value.shape[:-1], target - current))
    return torch.cat([value, pad], dim=-1)


def _append_unique(values: list[int], beam: int, num_beams: int) -> None:
    normalized = int(beam) % int(num_beams)
    if normalized not in values:
        values.append(normalized)


__all__ = ["BeamCandidateAttentionReranker"]
