from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateAttentionSelector(nn.Module):
    """Candidate query tokens attend GPS/camera/image tokens for TopK reranking."""

    def __init__(
        self,
        *,
        topk: int = 8,
        num_beams: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        lambda_init: float = 0.5,
        lambda_max: float = 3.0,
        use_gps_prior_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.topk = int(topk)
        self.num_beams = int(num_beams)
        self.hidden_dim = int(hidden_dim)
        self.lambda_max = float(lambda_max)
        self.use_gps_prior_fusion = bool(use_gps_prior_fusion)
        self.candidate_proj = nn.LazyLinear(self.hidden_dim)
        self.gps_proj = nn.LazyLinear(self.hidden_dim)
        self.camera_proj = nn.LazyLinear(self.hidden_dim)
        self.image_proj = nn.LazyLinear(self.hidden_dim)
        self.attention = nn.MultiheadAttention(self.hidden_dim, int(num_heads), dropout=float(dropout), batch_first=True)
        self.score_head = nn.Linear(self.hidden_dim, 1)
        self.miss_head = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, 1))
        self.lambda_param = nn.Parameter(torch.tensor(_inverse_softplus_or_floor(float(lambda_init)), dtype=torch.float32))

    @property
    def lambda_value(self) -> torch.Tensor:
        return F.softplus(self.lambda_param).clamp(min=0.0, max=self.lambda_max)

    def forward(
        self,
        *,
        candidate_features: torch.Tensor,
        gps_context: torch.Tensor,
        candidate_log_probs: torch.Tensor | None = None,
        candidate_probs: torch.Tensor | None = None,
        camera_ae_feature: torch.Tensor | None = None,
        image_tokens: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if candidate_features.ndim != 3:
            raise ValueError(f"candidate_features must have shape [B, K, F], got {tuple(candidate_features.shape)}.")
        batch, topk, _ = candidate_features.shape
        if int(topk) != self.topk:
            raise ValueError(f"candidate_features TopK dimension must be {self.topk}, got {topk}.")
        if candidate_log_probs is None:
            if candidate_probs is None:
                candidate_log_probs = candidate_features.new_zeros((batch, topk))
            else:
                candidate_log_probs = torch.log(candidate_probs.to(candidate_features.device, candidate_features.dtype).clamp_min(1e-12))
        candidate_log_probs = candidate_log_probs.to(device=candidate_features.device, dtype=candidate_features.dtype)
        candidate_tokens = self.candidate_proj(candidate_features)
        tokens = [self.gps_proj(gps_context.to(device=candidate_features.device, dtype=candidate_features.dtype)).unsqueeze(1)]
        enabled = ["gps_context"]
        if camera_ae_feature is not None:
            camera = camera_ae_feature.to(device=candidate_features.device, dtype=candidate_features.dtype)
            if camera.ndim > 2:
                camera = camera.flatten(start_dim=1)
            tokens.append(self.camera_proj(camera).unsqueeze(1))
            enabled.append("camera_ae")
        if image_tokens is not None:
            image = image_tokens.to(device=candidate_features.device, dtype=candidate_features.dtype)
            if image.ndim == 2:
                image = image.unsqueeze(1)
            if image.ndim != 3:
                raise ValueError(f"image_tokens must have shape [B, D] or [B, N, D], got {tuple(image_tokens.shape)}.")
            tokens.append(self.image_proj(image))
            enabled.append("image")
        memory = torch.cat(tokens, dim=1)
        attended, attn_weights = self.attention(candidate_tokens, memory, memory)
        modality_scores = self.score_head(attended).squeeze(-1)
        lambda_value = self.lambda_value.to(device=candidate_features.device, dtype=candidate_features.dtype)
        if self.use_gps_prior_fusion:
            final_scores = candidate_log_probs + lambda_value.view(1, 1) * modality_scores
        else:
            final_scores = modality_scores
        probs = F.softmax(final_scores, dim=-1)
        miss_logit = self.miss_head(attended.mean(dim=1))
        return {
            "final_candidate_scores": final_scores,
            "modality_candidate_scores": modality_scores,
            "candidate_probs": probs,
            "miss_logit": miss_logit,
            "attention_weights": attn_weights,
            "lambda_value": lambda_value.detach(),
            "diagnostics": {
                "lambda_value": float(lambda_value.detach().cpu()),
                "enabled_modalities": tuple(enabled),
                "use_gps_prior_fusion": self.use_gps_prior_fusion,
            },
        }


def _inverse_softplus_or_floor(value: float) -> float:
    if value <= 0.0:
        return -30.0
    return math.log(math.exp(float(value)) - 1.0)


__all__ = ["CandidateAttentionSelector"]
