"""Minimal block quality estimator and reliability fusion for PGCD."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from kd_sensing.models.pcer_temporal_fusion import masked_block_softmax


PGCD_VARIANTS = frozenset(f"c{index}" for index in range(8))


class PrototypeGuidedDegradationRouter(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_modalities: int,
        num_timesteps: int,
        variant: str,
        hidden_dim: int = 64,
        embedding_dim: int = 8,
        dropout: float = 0.0,
        beta_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_modalities = int(num_modalities)
        self.num_timesteps = int(num_timesteps)
        self.variant = str(variant).strip().lower()
        embedding = int(embedding_dim)
        if self.variant not in PGCD_VARIANTS:
            raise ValueError(f"PGCD variant must be one of {sorted(PGCD_VARIANTS)}.")
        if min(self.d_model, self.num_modalities, self.num_timesteps, int(hidden_dim), embedding) <= 0:
            raise ValueError("PGCD Router dimensions must be positive.")
        if not 0.0 <= float(dropout) < 1.0 or float(beta_init) <= 0:
            raise ValueError("PGCD dropout must be in [0,1) and beta_init must be positive.")
        self.prior_logits = nn.Parameter(torch.zeros(self.num_timesteps, self.num_modalities))
        self.modality_embedding = nn.Embedding(self.num_modalities, embedding)
        self.time_embedding = nn.Embedding(self.num_timesteps, embedding)
        input_dim = self.d_model + 3 + 2 * embedding
        self.quality_estimator = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        final = self.quality_estimator[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, -3.0)
        self.raw_beta = nn.Parameter(torch.tensor(_inverse_softplus(float(beta_init)), dtype=torch.float32))

    @property
    def beta_reliability(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_beta)

    def predict(
        self,
        block_features: torch.Tensor,
        block_evidence_logits: torch.Tensor,
        availability_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if block_features.ndim != 4:
            raise ValueError("PGCD block_features must have shape [B,T,M,D].")
        batch, timesteps, modalities, width = block_features.shape
        if (timesteps, modalities, width) != (self.num_timesteps, self.num_modalities, self.d_model):
            raise ValueError("PGCD block feature shape differs from configured [T,M,D].")
        if block_evidence_logits.ndim != 4 or tuple(block_evidence_logits.shape[:3]) != (batch, timesteps, modalities):
            raise ValueError("PGCD block evidence logits must have shape [B,T,M,K].")
        available = torch.as_tensor(availability_mask, device=block_features.device, dtype=torch.bool)
        if tuple(available.shape) != (batch, timesteps, modalities):
            raise ValueError("PGCD availability_mask must have shape [B,T,M].")
        if not bool(available.any(dim=(1, 2)).all().item()):
            raise ValueError("PGCD requires at least one available block per sample.")

        features = block_features.detach()
        evidence = block_evidence_logits.detach()
        probability = torch.softmax(evidence.float(), dim=-1).to(dtype=features.dtype)
        confidence, top2 = probability.topk(2, dim=-1).values.unbind(dim=-1)
        entropy = -(probability * probability.clamp_min(torch.finfo(probability.dtype).tiny).log()).sum(dim=-1)
        entropy = entropy / math.log(float(probability.shape[-1]))
        modality_ids = torch.arange(modalities, device=features.device).reshape(1, 1, modalities).expand(batch, timesteps, -1)
        time_ids = torch.arange(timesteps, device=features.device).reshape(1, timesteps, 1).expand(batch, -1, modalities)
        estimator_input = torch.cat(
            (
                features,
                confidence.unsqueeze(-1),
                entropy.unsqueeze(-1),
                (confidence - top2).unsqueeze(-1),
                self.modality_embedding(modality_ids),
                self.time_embedding(time_ids),
            ),
            dim=-1,
        )
        quality_logits = self.quality_estimator(estimator_input).squeeze(-1)
        predicted_degradation = torch.nn.functional.softplus(quality_logits)
        predicted_reliability = torch.exp(-predicted_degradation).masked_fill(~available, 0.0)
        return {
            "quality_logits": quality_logits.reshape(batch, -1),
            "predicted_degradation": predicted_degradation.reshape(batch, -1),
            "predicted_reliability": predicted_reliability.reshape(batch, -1),
            "confidence": confidence.reshape(batch, -1),
            "entropy": entropy.reshape(batch, -1),
            "margin": (confidence - top2).reshape(batch, -1),
            "availability": available.reshape(batch, -1),
        }

    def forward(
        self,
        block_features: torch.Tensor,
        block_evidence_logits: torch.Tensor,
        availability_mask: torch.Tensor,
        *,
        degradation_override: torch.Tensor | None = None,
        use_dynamic: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        prediction = self.predict(block_features, block_evidence_logits, availability_mask)
        available = prediction["availability"]
        batch = block_features.shape[0]
        prior = self.prior_logits.reshape(1, -1).expand(batch, -1)
        dynamic = self.variant != "c0" if use_dynamic is None else bool(use_dynamic)
        if degradation_override is None:
            degradation = prediction["predicted_degradation"]
        else:
            degradation = torch.as_tensor(
                degradation_override,
                device=block_features.device,
                dtype=block_features.dtype,
            )
            if degradation.ndim == 1:
                degradation = degradation.unsqueeze(0).expand(batch, -1)
            if tuple(degradation.shape) != tuple(available.shape):
                raise ValueError("PGCD degradation override must have shape [N] or [B,N].")
            if bool((degradation < 0).any().item()) or not bool(torch.isfinite(degradation).all().item()):
                raise ValueError("PGCD degradation override must be finite and non-negative.")
        reliability = torch.exp(-degradation).masked_fill(~available, 0.0)
        fusion_logits = prior
        if dynamic:
            fusion_logits = fusion_logits + self.beta_reliability.to(dtype=prior.dtype) * reliability.clamp_min(1e-8).log()
        fusion_logits = fusion_logits.masked_fill(~available, -torch.inf)
        weights = masked_block_softmax(fusion_logits, available)
        prior_weights = masked_block_softmax(prior.masked_fill(~available, -torch.inf), available)
        return {
            **prediction,
            "fusion_degradation": degradation,
            "fusion_reliability": reliability,
            "fusion_logits": fusion_logits,
            "weights": weights,
            "prior_logits": prior.masked_fill(~available, -torch.inf),
            "prior_weights": prior_weights,
            "beta_reliability": self.beta_reliability,
            "dynamic_enabled": torch.tensor(dynamic, device=block_features.device),
        }


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


__all__ = ["PGCD_VARIANTS", "PrototypeGuidedDegradationRouter"]
