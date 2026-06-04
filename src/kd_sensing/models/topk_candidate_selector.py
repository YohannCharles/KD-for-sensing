from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKCandidateSelector(nn.Module):
    """GPS-prior-preserving selector over fixed TopK candidate beams."""

    def __init__(
        self,
        *,
        topk: int = 8,
        num_beams: int = 64,
        candidate_feature_dim: int = 10,
        gps_context_dim: int = 14,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        lambda_init: float = 0.5,
        lambda_max: float = 3.0,
        use_gps_prior_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.topk = int(topk)
        self.num_beams = int(num_beams)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.gps_context_dim = int(gps_context_dim)
        self.hidden_dim = int(hidden_dim)
        self.lambda_max = float(lambda_max)
        self.use_gps_prior_fusion = bool(use_gps_prior_fusion)
        self.candidate_encoder = nn.Sequential(
            nn.LazyLinear(self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.optional_encoder = nn.Sequential(
            nn.LazyLinear(self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.score_head = nn.Linear(self.hidden_dim, 1)
        self.miss_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
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
        image_feature: torch.Tensor | None = None,
        lidar_feature: torch.Tensor | None = None,
        radar_feature: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if candidate_features.ndim != 3:
            raise ValueError(f"candidate_features must have shape [B, K, F], got {tuple(candidate_features.shape)}.")
        batch, topk, _ = candidate_features.shape
        if int(topk) != self.topk:
            raise ValueError(f"candidate_features TopK dimension must be {self.topk}, got {topk}.")
        if gps_context.ndim != 2 or int(gps_context.shape[0]) != int(batch):
            raise ValueError(f"gps_context must have shape [B, D], got {tuple(gps_context.shape)}.")
        if candidate_log_probs is None:
            if candidate_probs is None:
                candidate_log_probs = candidate_features.new_zeros((batch, topk))
            else:
                candidate_log_probs = torch.log(candidate_probs.to(candidate_features.device, candidate_features.dtype).clamp_min(1e-12))
        candidate_log_probs = candidate_log_probs.to(device=candidate_features.device, dtype=candidate_features.dtype)
        gps_expanded = gps_context.to(device=candidate_features.device, dtype=candidate_features.dtype).unsqueeze(1).expand(-1, topk, -1)
        encoded = self.candidate_encoder(torch.cat([candidate_features, gps_expanded], dim=-1))
        optional = _optional_vector(
            camera_ae_feature=camera_ae_feature,
            image_feature=image_feature,
            lidar_feature=lidar_feature,
            radar_feature=radar_feature,
            device=candidate_features.device,
            dtype=candidate_features.dtype,
        )
        enabled_modalities = ["gps_context"]
        if optional is not None:
            optional_encoded = self.optional_encoder(optional).unsqueeze(1)
            encoded = encoded + optional_encoded
            enabled_modalities.extend(_enabled_optional_names(camera_ae_feature, image_feature, lidar_feature, radar_feature))
        modality_scores = self.score_head(encoded).squeeze(-1)
        lambda_value = self.lambda_value.to(device=candidate_features.device, dtype=candidate_features.dtype)
        if self.use_gps_prior_fusion:
            final_scores = candidate_log_probs + lambda_value.view(1, 1) * modality_scores
        else:
            final_scores = modality_scores
        candidate_out_probs = F.softmax(final_scores, dim=-1)
        pooled = encoded.mean(dim=1)
        miss_logit = self.miss_head(pooled)
        return {
            "final_candidate_scores": final_scores,
            "modality_candidate_scores": modality_scores,
            "candidate_probs": candidate_out_probs,
            "miss_logit": miss_logit,
            "lambda_value": lambda_value.detach(),
            "diagnostics": {
                "lambda_value": float(lambda_value.detach().cpu()),
                "enabled_modalities": tuple(enabled_modalities),
                "candidate_score_mean": float(modality_scores.detach().mean().cpu()),
                "candidate_score_std": float(modality_scores.detach().std(unbiased=False).cpu()),
                "use_gps_prior_fusion": self.use_gps_prior_fusion,
            },
        }


def select_final_beams(candidate_beams: torch.Tensor, candidate_probs: torch.Tensor) -> torch.Tensor:
    if candidate_beams.ndim != 2 or candidate_probs.ndim != 2:
        raise ValueError("candidate_beams and candidate_probs must both have shape [B, K].")
    if tuple(candidate_beams.shape) != tuple(candidate_probs.shape):
        raise ValueError(f"Shape mismatch: {tuple(candidate_beams.shape)} vs {tuple(candidate_probs.shape)}.")
    indices = candidate_probs.argmax(dim=-1, keepdim=True)
    return torch.gather(candidate_beams.to(device=candidate_probs.device), 1, indices).squeeze(-1)


def sparse_topk_scores_to_logits(
    candidate_beams: torch.Tensor,
    candidate_scores: torch.Tensor,
    *,
    num_beams: int = 64,
    fill_value: float = -1e9,
) -> torch.Tensor:
    if candidate_beams.ndim != 2 or candidate_scores.ndim != 2:
        raise ValueError("candidate_beams and candidate_scores must both have shape [B, K].")
    if tuple(candidate_beams.shape) != tuple(candidate_scores.shape):
        raise ValueError(f"Shape mismatch: {tuple(candidate_beams.shape)} vs {tuple(candidate_scores.shape)}.")
    logits = candidate_scores.new_full((int(candidate_scores.shape[0]), int(num_beams)), float(fill_value))
    beams = candidate_beams.to(device=candidate_scores.device, dtype=torch.long).remainder(int(num_beams))
    logits.scatter_(1, beams, candidate_scores)
    return logits


def _optional_vector(
    *,
    camera_ae_feature: torch.Tensor | None,
    image_feature: torch.Tensor | None,
    lidar_feature: torch.Tensor | None,
    radar_feature: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    features = []
    for value in (camera_ae_feature, image_feature, lidar_feature, radar_feature):
        if value is None:
            continue
        tensor = value.to(device=device, dtype=dtype)
        if tensor.ndim > 2:
            tensor = tensor.flatten(start_dim=1)
        features.append(tensor)
    if not features:
        return None
    return torch.cat(features, dim=-1)


def _enabled_optional_names(
    camera_ae_feature: torch.Tensor | None,
    image_feature: torch.Tensor | None,
    lidar_feature: torch.Tensor | None,
    radar_feature: torch.Tensor | None,
) -> list[str]:
    names = []
    if camera_ae_feature is not None:
        names.append("camera_ae")
    if image_feature is not None:
        names.append("image")
    if lidar_feature is not None:
        names.append("lidar")
    if radar_feature is not None:
        names.append("radar")
    return names


def _inverse_softplus_or_floor(value: float) -> float:
    if value <= 0.0:
        return -30.0
    return math.log(math.exp(float(value)) - 1.0)


__all__ = [
    "TopKCandidateSelector",
    "select_final_beams",
    "sparse_topk_scores_to_logits",
]
