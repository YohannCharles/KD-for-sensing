from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.registries import REPRESENTATION_CORES


@REPRESENTATION_CORES.register("masktrain_mean_fusion")
class MaskTrainMeanFusionCore(nn.Module):
    """Plain mask-trained mean fusion used as the non-prototype control."""

    supports_missing_modality_metadata = True

    def __init__(self, d_model: int, modality_count: int, output_dim: int | None = None, **_: Any) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.modality_count = int(modality_count)
        self.output_dim = int(output_dim or d_model)
        if min(self.d_model, self.modality_count, self.output_dim) <= 0:
            raise ValueError("masktrain_mean_fusion dimensions must be positive.")
        self.output_projection = nn.Identity() if self.output_dim == self.d_model else nn.Linear(self.d_model, self.output_dim)
        self.last_fusion_weights: torch.Tensor | None = None

    def forward(self, features: torch.Tensor, *, modality_available: torch.Tensor | None = None) -> torch.Tensor:
        availability = _availability(features, modality_available, self.modality_count, self.d_model)
        weights = availability.to(dtype=features.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.last_fusion_weights = weights.detach()
        return self.output_projection((features * weights.unsqueeze(-1)).sum(dim=1))

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": "masktrain_mean_fusion",
            "model_group": "MaskTrain-CLS",
            "paper_equivalent": False,
            "baseline_scope": "plain_mask_training_control",
            "fusion_type": "availability_normalized_mean",
            "classifier": "shared_linear_beam_head",
            "prototype_learning": False,
            "router_supervision": False,
            "consumes_missing_modality_metadata": True,
        }


@REPRESENTATION_CORES.register("amr_gaussian_uncertainty_fusion")
class AMRGaussianUncertaintyFusionCore(nn.Module):
    """Four-modality local AMR adaptation with Gaussian uncertainty fusion."""

    supports_missing_modality_metadata = True
    supports_reliability_metadata = True

    def __init__(
        self,
        d_model: int,
        modality_count: int,
        latent_dim: int | None = None,
        output_dim: int | None = None,
        dropout: float = 0.1,
        logvar_min: float = -8.0,
        logvar_max: float = 8.0,
        deterministic_inference: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.modality_count = int(modality_count)
        self.latent_dim = int(latent_dim or d_model)
        self.output_dim = int(output_dim or d_model)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.deterministic_inference = bool(deterministic_inference)
        if min(self.d_model, self.modality_count, self.latent_dim, self.output_dim) <= 0:
            raise ValueError("amr_gaussian_uncertainty_fusion dimensions must be positive.")
        self.mu_heads = nn.ModuleList([nn.Linear(self.d_model, self.latent_dim) for _ in range(self.modality_count)])
        self.logvar_heads = nn.ModuleList([nn.Linear(self.d_model, self.latent_dim) for _ in range(self.modality_count)])
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(self.latent_dim, self.output_dim),
        )
        self.last_fusion_weights: torch.Tensor | None = None
        self.last_uncertainty: torch.Tensor | None = None
        self.last_amr_auxiliary: dict[str, torch.Tensor] | None = None

    def forward(self, features: torch.Tensor, *, modality_available: torch.Tensor | None = None) -> torch.Tensor:
        availability = _availability(features, modality_available, self.modality_count, self.d_model)
        mu = torch.stack([head(features[:, index]) for index, head in enumerate(self.mu_heads)], dim=1)
        logvar = torch.stack([head(features[:, index]) for index, head in enumerate(self.logvar_heads)], dim=1)
        logvar = logvar.clamp(self.logvar_min, self.logvar_max)
        latent = mu if (not self.training and self.deterministic_inference) else mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        uncertainty = F.softplus(logvar).mean(dim=-1)
        logits = (-uncertainty).masked_fill(~availability, torch.finfo(uncertainty.dtype).min)
        weights = torch.softmax(logits, dim=1).masked_fill(~availability, 0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
        fused = (latent * weights.unsqueeze(-1)).sum(dim=1)
        self.last_fusion_weights = weights.detach()
        self.last_uncertainty = uncertainty.detach()
        self.last_amr_auxiliary = (
            {"mu": mu, "logvar": logvar, "availability": availability, "fused_mu": (mu * weights.unsqueeze(-1)).sum(dim=1)}
            if self.training
            else None
        )
        return self.output_projection(fused)

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": "amr_gaussian_uncertainty_fusion",
            "model_group": "AMR-Net-4M-Adapted",
            "reproduction_scope": "four_modality_window5_local_adaptation",
            "paper_equivalent": False,
            "paper_alignment": "probabilistic_embedding_and_uncertainty_aware_fusion",
            "adaptations": ["add_radar", "shared_four_modality_encoders", "window5", "feature_level_uncertainty_fusion"],
            "d_model": self.d_model,
            "latent_dim": self.latent_dim,
            "output_dim": self.output_dim,
            "modality_count": self.modality_count,
            "uncertainty": "mean_softplus_log_variance",
            "fusion_type": "masked_uncertainty_softmax",
            "deterministic_inference": self.deterministic_inference,
            "consumes_missing_modality_metadata": True,
            "consumes_reliability_metadata": True,
        }


def _availability(
    features: torch.Tensor,
    modality_available: torch.Tensor | None,
    modality_count: int,
    d_model: int,
) -> torch.Tensor:
    if features.ndim != 4 or int(features.shape[1]) != modality_count or int(features.shape[-1]) != d_model:
        raise ValueError(f"fusion core expects [B,{modality_count},T,{d_model}], got {tuple(features.shape)}.")
    if modality_available is None:
        availability = torch.ones(features.shape[:3], dtype=torch.bool, device=features.device)
    else:
        availability = torch.as_tensor(modality_available, dtype=torch.bool, device=features.device)
        if tuple(availability.shape) != tuple(features.shape[:3]):
            raise ValueError(f"modality_available must match {tuple(features.shape[:3])}, got {tuple(availability.shape)}.")
    return availability


__all__ = ["AMRGaussianUncertaintyFusionCore", "MaskTrainMeanFusionCore"]
