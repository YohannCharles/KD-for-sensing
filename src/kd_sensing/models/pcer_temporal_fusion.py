from __future__ import annotations

import math

import torch
import torch.nn as nn


PCER_MODES = frozenset(
    (
        "evidence_static",
        "counterfactual_router",
        "evidence_only",
        "block_router",
        "hierarchical_router",
        "mask_residual_router",
    )
)


class TemporalBlockEvidenceRouter(nn.Module):
    """Small shared Router over flattened modality-time prototype evidence."""

    def __init__(
        self,
        *,
        d_model: int,
        num_modalities: int,
        num_timesteps: int,
        hidden_dim: int = 64,
        embedding_dim: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_modalities = int(num_modalities)
        self.num_timesteps = int(num_timesteps)
        embedding = int(embedding_dim)
        if min(self.d_model, self.num_modalities, self.num_timesteps, int(hidden_dim), embedding) <= 0:
            raise ValueError("PCER Router dimensions must be positive.")
        self.modality_embedding = nn.Embedding(self.num_modalities, embedding)
        self.time_embedding = nn.Embedding(self.num_timesteps, embedding)
        input_dim = self.d_model + 2 + 2 * embedding
        self.router = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        block_features: torch.Tensor,
        block_evidence_logits: torch.Tensor,
        availability_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, flat_available = self.score(block_features, block_evidence_logits, availability_mask)
        masked_logits = logits.masked_fill(~flat_available, -torch.inf)
        return masked_logits, masked_block_softmax(masked_logits, flat_available)

    def score(
        self,
        block_features: torch.Tensor,
        block_evidence_logits: torch.Tensor,
        availability_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if block_features.ndim != 4:
            raise ValueError("block_features must have shape [B,T,M,D].")
        batch, timesteps, modalities, width = block_features.shape
        if (timesteps, modalities, width) != (self.num_timesteps, self.num_modalities, self.d_model):
            raise ValueError(
                "PCER block_features shape differs from configured [T,M,D]: "
                f"got {tuple(block_features.shape)}, expected [B,{self.num_timesteps},{self.num_modalities},{self.d_model}]."
            )
        if block_evidence_logits.ndim != 4 or tuple(block_evidence_logits.shape[:3]) != (batch, timesteps, modalities):
            raise ValueError("block_evidence_logits must have shape [B,T,M,K].")
        available = torch.as_tensor(availability_mask, device=block_features.device, dtype=torch.bool)
        if tuple(available.shape) != (batch, timesteps, modalities):
            raise ValueError("availability_mask must have shape [B,T,M].")

        features = block_features.detach()
        evidence = block_evidence_logits.detach()
        probability = torch.softmax(evidence.float(), dim=-1).to(dtype=features.dtype)
        confidence = probability.amax(dim=-1, keepdim=True)
        entropy = -(probability * probability.clamp_min(torch.finfo(probability.dtype).tiny).log()).sum(
            dim=-1, keepdim=True
        ) / math.log(float(probability.shape[-1]))
        modality_ids = torch.arange(modalities, device=features.device).view(1, 1, modalities).expand(batch, timesteps, -1)
        time_ids = torch.arange(timesteps, device=features.device).view(1, timesteps, 1).expand(batch, -1, modalities)
        router_input = torch.cat(
            (
                features,
                confidence,
                entropy,
                self.modality_embedding(modality_ids),
                self.time_embedding(time_ids),
            ),
            dim=-1,
        )
        logits = self.router(router_input).squeeze(-1).reshape(batch, -1)
        flat_available = available.reshape(batch, -1)
        return logits, flat_available


class HierarchicalTemporalBlockRouter(nn.Module):
    """Factor a shared block score into modality alpha and within-modality beta."""

    def __init__(self, **kwargs: int | float) -> None:
        super().__init__()
        self.scorer = TemporalBlockEvidenceRouter(**kwargs)
        self.num_modalities = self.scorer.num_modalities
        self.num_timesteps = self.scorer.num_timesteps

    def forward(
        self,
        block_features: torch.Tensor,
        block_evidence_logits: torch.Tensor,
        availability_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        flat_score, flat_available = self.scorer.score(block_features, block_evidence_logits, availability_mask)
        batch = flat_score.shape[0]
        score = flat_score.reshape(batch, self.num_timesteps, self.num_modalities)
        available = flat_available.reshape_as(score)
        modality_available = available.any(dim=1)
        count = available.sum(dim=1).clamp_min(1)
        alpha_logits = (score.masked_fill(~available, 0.0).sum(dim=1) / count).masked_fill(
            ~modality_available, -torch.inf
        )
        alpha = masked_block_softmax(alpha_logits, modality_available)
        beta_logits = score.permute(0, 2, 1)
        beta_available = available.permute(0, 2, 1)
        beta = torch.softmax(beta_logits.masked_fill(~beta_available, -torch.inf), dim=-1).masked_fill(
            ~beta_available, 0.0
        )
        weights = (alpha.unsqueeze(-1) * beta).permute(0, 2, 1).reshape(batch, -1)
        combined_logits = weights.clamp_min(torch.finfo(weights.dtype).tiny).log().masked_fill(
            ~flat_available, -torch.inf
        )
        return {
            "logits": combined_logits,
            "weights": weights,
            "alpha_logits": alpha_logits,
            "alpha": alpha,
            "beta": beta,
        }


class MaskConditionedResidualRouter(nn.Module):
    """Availability-conditioned prior plus a small zero-mean block residual."""

    def __init__(self, **kwargs: int | float) -> None:
        super().__init__()
        self.scorer = TemporalBlockEvidenceRouter(**kwargs)
        self.num_modalities = self.scorer.num_modalities
        self.num_timesteps = self.scorer.num_timesteps
        hidden = int(kwargs.get("hidden_dim", 64))
        descriptor_dim = 3 * self.num_modalities + self.num_timesteps + 2
        blocks = self.num_modalities * self.num_timesteps
        self.prior = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, blocks),
        )
        self.residual_logit_scale = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))

    def forward(
        self,
        block_features: torch.Tensor,
        block_evidence_logits: torch.Tensor,
        availability_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        dynamic, flat_available = self.scorer.score(block_features, block_evidence_logits, availability_mask)
        batch = dynamic.shape[0]
        available = flat_available.reshape(batch, self.num_timesteps, self.num_modalities)
        descriptor = _availability_descriptor(available)
        prior_logits = self.prior(descriptor).masked_fill(~flat_available, -torch.inf)
        valid = flat_available.to(dtype=dynamic.dtype)
        dynamic_mean = (dynamic * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        residual = (dynamic - dynamic_mean).masked_fill(~flat_available, 0.0)
        scale = torch.sigmoid(self.residual_logit_scale)
        logits = (prior_logits + scale * residual).masked_fill(~flat_available, -torch.inf)
        return {
            "logits": logits,
            "weights": masked_block_softmax(logits, flat_available),
            "prior_logits": prior_logits,
            "prior_weights": masked_block_softmax(prior_logits, flat_available),
            "dynamic_residual": residual,
            "residual_scale": scale,
        }


def _availability_descriptor(available: torch.Tensor) -> torch.Tensor:
    batch, timesteps, modalities = available.shape
    modality_available = available.any(dim=1).float()
    counts = available.float().sum(dim=1) / float(timesteps)
    time_ids = torch.arange(1, timesteps + 1, device=available.device, dtype=torch.float32).view(1, timesteps, 1)
    latest = (available.float() * time_ids).amax(dim=1) / float(timesteps)
    time_available = available.any(dim=2).float()
    latest_sync = available[:, -1].all(dim=1, keepdim=True).float()
    ratio = available.float().mean(dim=(1, 2), keepdim=False).unsqueeze(1)
    return torch.cat((modality_available, counts, latest, time_available, latest_sync, ratio), dim=1)


def static_block_weights(availability_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    available = torch.as_tensor(availability_mask, dtype=torch.bool)
    if available.ndim != 3:
        raise ValueError("availability_mask must have shape [B,T,M].")
    flat = available.reshape(available.shape[0], -1)
    logits = torch.zeros_like(flat, dtype=torch.float32).masked_fill(~flat, -torch.inf)
    return logits, masked_block_softmax(logits, flat)


def masked_block_softmax(logits: torch.Tensor, availability_mask: torch.Tensor) -> torch.Tensor:
    available = torch.as_tensor(availability_mask, device=logits.device, dtype=torch.bool)
    if logits.ndim != 2 or tuple(logits.shape) != tuple(available.shape):
        raise ValueError("block logits and availability_mask must share shape [B,N].")
    if not bool(available.any(dim=1).all().item()):
        raise ValueError("PCER fusion requires at least one available block per sample.")
    weights = torch.softmax(logits.masked_fill(~available, -torch.inf), dim=-1)
    return weights.masked_fill(~available, 0.0)


__all__ = [
    "PCER_MODES",
    "HierarchicalTemporalBlockRouter",
    "MaskConditionedResidualRouter",
    "TemporalBlockEvidenceRouter",
    "masked_block_softmax",
    "static_block_weights",
]
