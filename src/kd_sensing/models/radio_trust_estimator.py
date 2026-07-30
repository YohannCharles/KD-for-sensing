"""Target-free sample-level radio trust for prototype evidence fusion."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


TRUST_FEATURE_DIM = 22


def _distribution_statistics(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probability = torch.softmax(logits, dim=-1)
    tiny = torch.finfo(probability.dtype).tiny
    entropy = -(probability * probability.clamp_min(tiny).log()).sum(dim=-1) / math.log(probability.shape[-1])
    top = probability.topk(2, dim=-1).values
    margin = top[:, 0] - top[:, 1]
    concentration = probability.topk(min(5, probability.shape[-1]), dim=-1).values.sum(dim=-1)
    return entropy, margin, concentration


def build_trust_features(
    physical_availability: torch.Tensor,
    sensing_embedding: torch.Tensor,
    sensing_logits: torch.Tensor,
    radio_logits: torch.Tensor,
    csi_quality: torch.Tensor,
    prototypes: torch.Tensor,
    topology_distance: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build the declared target-free sensing, radio, and conflict features."""
    sensing = torch.as_tensor(sensing_logits)
    radio = torch.as_tensor(radio_logits, device=sensing.device, dtype=sensing.dtype)
    physical = torch.as_tensor(physical_availability, device=sensing.device, dtype=torch.bool)
    quality = torch.as_tensor(csi_quality, device=sensing.device, dtype=sensing.dtype)
    embedding = torch.as_tensor(sensing_embedding, device=sensing.device, dtype=sensing.dtype)
    prototype = torch.as_tensor(prototypes, device=sensing.device, dtype=sensing.dtype)
    distance = torch.as_tensor(topology_distance, device=sensing.device, dtype=sensing.dtype)
    batch, beams = sensing.shape
    if radio.shape != sensing.shape or physical.shape != (batch, 4):
        raise ValueError("Trust inputs require matching [B,C] evidence and [B,4] availability.")
    if quality.ndim != 2 or quality.shape[0] != batch or quality.shape[1] < 21:
        raise ValueError("csi_quality must expose the validated 21 quality features.")
    if embedding.shape != (batch, prototype.shape[1]) or distance.shape != (beams, beams):
        raise ValueError("Embedding, prototype, or topology dimensions do not match.")

    p_s = torch.softmax(sensing, dim=-1)
    p_c = torch.softmax(radio, dim=-1)
    sensing_entropy, sensing_margin, sensing_concentration = _distribution_statistics(sensing)
    radio_entropy, radio_margin, _ = _distribution_statistics(radio)
    nearest_similarity = F.normalize(embedding, dim=-1) @ F.normalize(prototype, dim=-1).t()
    nearest_distance = 1.0 - nearest_similarity.max(dim=-1).values

    mixture = 0.5 * (p_s + p_c)
    tiny = torch.finfo(p_s.dtype).tiny
    js = 0.5 * (
        (p_s * (p_s.clamp_min(tiny).log() - mixture.clamp_min(tiny).log())).sum(dim=-1)
        + (p_c * (p_c.clamp_min(tiny).log() - mixture.clamp_min(tiny).log())).sum(dim=-1)
    )
    topology_scale = distance.max().clamp_min(1.0)
    sensing_argmax = sensing.argmax(dim=-1)
    radio_argmax = radio.argmax(dim=-1)
    argmax_distance = distance[sensing_argmax, radio_argmax] / topology_scale
    expected_distance = torch.einsum("bi,ij,bj->b", p_s, distance / topology_scale, p_c)
    difference = radio - sensing

    available_fraction = physical.float().mean(dim=-1)
    missing_severity = ((4.0 - physical.sum(dim=-1).float()) / 3.0).clamp(0.0, 1.0)
    snr_normalized = quality[:, 16]
    valid_ratio = quality[:, 17].clamp(0.0, 1.0)
    log_rms = torch.tanh(quality[:, 18] / 5.0)
    quality_confidence = quality[:, 19].clamp(0.0, 1.0)
    temporal_consistency = quality[:, 20].clamp(0.0, 1.0)
    snr_score = ((snr_normalized + 1.0 / 3.0) * 0.75).clamp(0.0, 1.0)
    quality_score = torch.stack(
        (snr_score, valid_ratio, temporal_consistency, radio_margin, quality_confidence),
        dim=-1,
    ).mean(dim=-1)
    sensing_uncertainty = torch.stack(
        (
            sensing_entropy,
            1.0 - sensing_margin,
            nearest_distance.clamp(0.0, 2.0) * 0.5,
        ),
        dim=-1,
    ).mean(dim=-1)

    features = torch.cat(
        (
            physical.float(),
            available_fraction[:, None],
            missing_severity[:, None],
            sensing_entropy[:, None],
            sensing_margin[:, None],
            nearest_distance[:, None],
            sensing_concentration[:, None],
            snr_normalized[:, None],
            valid_ratio[:, None],
            log_rms[:, None],
            temporal_consistency[:, None],
            quality_confidence[:, None],
            radio_entropy[:, None],
            radio_margin[:, None],
            js[:, None],
            argmax_distance[:, None],
            expected_distance[:, None],
            difference.abs().mean(dim=-1, keepdim=True),
            difference.std(dim=-1, keepdim=True, unbiased=False),
        ),
        dim=-1,
    )
    if features.shape != (batch, TRUST_FEATURE_DIM):
        raise AssertionError(f"Unexpected trust feature shape: {tuple(features.shape)}.")
    return {
        "features": features,
        "missing_severity": missing_severity,
        "quality_score": quality_score,
        "sensing_uncertainty": sensing_uncertainty,
        "sensing_entropy": sensing_entropy,
        "sensing_margin": sensing_margin,
        "radio_entropy": radio_entropy,
        "radio_margin": radio_margin,
        "js_disagreement": js,
        "argmax_topology_distance": argmax_distance,
        "expected_topology_distance": expected_distance,
    }


class RadioTrustEstimator(nn.Module):
    """Bound radio use with interpretable monotone terms plus a small residual."""

    def __init__(
        self,
        *,
        feature_dim: int = TRUST_FEATURE_DIM,
        hidden_dim: int = 32,
        structured: bool = True,
        residual_scale: float = 0.1,
        structured_base_bias: float = -2.0,
        structured_raw_quality: float = 1.5,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.structured = bool(structured)
        self.residual_scale = float(residual_scale)
        self.base_bias = nn.Parameter(torch.tensor(float(structured_base_bias) if self.structured else 0.0))
        self.raw_missing = nn.Parameter(torch.tensor(0.0))
        self.raw_quality = nn.Parameter(torch.tensor(float(structured_raw_quality) if self.structured else 0.0))
        self.raw_uncertainty = nn.Parameter(torch.tensor(0.0))
        self.context = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def forward(
        self,
        statistics: dict[str, torch.Tensor],
        csi_available: torch.Tensor,
        *,
        full_mask: torch.Tensor | None = None,
        rho_floor: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        features = statistics["features"]
        if tuple(features.shape[1:]) != (self.feature_dim,):
            raise ValueError(f"Trust features must have shape [B,{self.feature_dim}].")
        residual = self.residual_scale * torch.tanh(self.context(features).squeeze(-1))
        logit = self.base_bias + residual
        if self.structured:
            logit = (
                logit
                + F.softplus(self.raw_missing) * statistics["missing_severity"]
                + F.softplus(self.raw_quality) * statistics["quality_score"]
                + F.softplus(self.raw_uncertainty) * statistics["sensing_uncertainty"]
            )
        rho = torch.sigmoid(logit)
        floor = float(rho_floor)
        if not 0.0 <= floor < 1.0:
            raise ValueError("rho_floor must be in [0,1).")
        if floor:
            rho = floor + (1.0 - floor) * rho
        active = torch.as_tensor(csi_available, device=rho.device, dtype=torch.bool).reshape(-1)
        if full_mask is not None:
            active = active & ~torch.as_tensor(full_mask, device=rho.device, dtype=torch.bool).reshape(-1)
        rho = rho * active.to(rho)
        return {
            **statistics,
            "rho": rho,
            "rho_logit": logit,
            "context_residual": residual,
        }


__all__ = ["TRUST_FEATURE_DIM", "RadioTrustEstimator", "build_trust_features"]
