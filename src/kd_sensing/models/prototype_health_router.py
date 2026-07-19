from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.losses.beam_prototype_alignment import beam_topology_positions


ROUTER_VARIANTS = frozenset(("patr", "h2r", "core", "unified_hpr"))
FRAME_FEATURE_DIM = 9
TEMPORAL_FEATURE_DIM = FRAME_FEATURE_DIM * 3 + 3
H2R_FEATURE_DIM = 4
CONSENSUS_FEATURE_DIM = 4

_VARIANT_COMPONENTS = {
    "patr": (True, False, False),
    "h2r": (False, True, False),
    "core": (False, False, True),
    "unified_hpr": (True, True, True),
}


@dataclass(frozen=True)
class FrameEvidence:
    cell_mask: torch.Tensor
    frame_probabilities: torch.Tensor
    frame_features: torch.Tensor
    temporal_features: torch.Tensor


@dataclass(frozen=True)
class TemporalPoolOutput:
    features: torch.Tensor
    logits: torch.Tensor
    weights: torch.Tensor


@dataclass(frozen=True)
class ReliabilityRouteOutput:
    gate_logits: torch.Tensor
    weights: torch.Tensor
    prior_weights: torch.Tensor
    residual_logits: torch.Tensor
    modality_features: torch.Tensor
    consensus_features: torch.Tensor
    effective_cell_weights: torch.Tensor


class PrototypeReliabilityRouter(nn.Module):
    """Shared PATR, H2R, CoRe, and Unified-HPR candidate Router."""

    def __init__(
        self,
        *,
        variant: str,
        modality_count: int,
        num_classes: int,
        base_feature_dim: int,
        prior_weights: Sequence[float] | torch.Tensor | None = None,
        topology_id: str = "cyclic_index_v1",
        topology_permutation: Sequence[int] | torch.Tensor | None = None,
        circular: bool = True,
        residual_hidden_dim: int = 64,
        health_hidden_dim: int = 16,
        residual_scale: float = 1.0,
        top_k: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.variant = str(variant).strip().lower()
        if self.variant not in ROUTER_VARIANTS:
            raise ValueError(f"router variant must be one of {sorted(ROUTER_VARIANTS)}, got {variant!r}.")
        self.modality_count = int(modality_count)
        self.num_classes = int(num_classes)
        self.base_feature_dim = int(base_feature_dim)
        self.residual_scale = float(residual_scale)
        self.top_k = int(top_k)
        self.circular = bool(circular)
        if min(self.modality_count, self.num_classes, self.base_feature_dim) <= 0:
            raise ValueError("modality_count, num_classes, and base_feature_dim must be positive.")
        if min(int(residual_hidden_dim), int(health_hidden_dim), self.top_k) <= 0:
            raise ValueError("Router hidden dimensions and top_k must be positive.")
        if self.top_k > self.num_classes:
            raise ValueError("top_k must not exceed num_classes.")
        if not math.isfinite(self.residual_scale) or self.residual_scale < 0.0:
            raise ValueError("residual_scale must be finite and non-negative.")
        topology = str(topology_id).strip().lower()
        expected_circular = topology != "linear_index_v1"
        if self.circular != expected_circular:
            raise ValueError(f"circular={self.circular} conflicts with topology_id={topology!r}.")
        positions = beam_topology_positions(
            self.num_classes,
            topology_id=topology,
            topology_permutation=topology_permutation,
        )
        self.register_buffer("topology_positions", positions, persistent=True)

        prior = _normalized_prior(prior_weights, self.modality_count)
        self.prior_logits = nn.Parameter(prior.log())
        self.use_patr, self.use_h2r, self.use_core = _VARIANT_COMPONENTS[self.variant]
        self.frame_health_head = (
            _zero_initialized_mlp(FRAME_FEATURE_DIM, int(health_hidden_dim), float(dropout))
            if self.use_h2r
            else None
        )
        residual_input_dim = self.base_feature_dim
        residual_input_dim += TEMPORAL_FEATURE_DIM if self.use_patr else 0
        residual_input_dim += H2R_FEATURE_DIM if self.use_h2r else 0
        residual_input_dim += CONSENSUS_FEATURE_DIM if self.use_core else 0
        self.residual_input_dim = residual_input_dim
        self.modality_residual_head = _zero_initialized_mlp(
            residual_input_dim,
            int(residual_hidden_dim),
            float(dropout),
        )

    def prepare(
        self,
        latent_sequence: torch.Tensor,
        frame_logits: torch.Tensor,
        frame_reliability: torch.Tensor,
        cell_mask: torch.Tensor,
    ) -> FrameEvidence:
        """Build detached per-cell and temporal evidence before temporal pooling."""
        _validate_frame_inputs(
            latent_sequence,
            frame_logits,
            frame_reliability,
            cell_mask,
            modality_count=self.modality_count,
            num_classes=self.num_classes,
        )
        latent = latent_sequence.detach()
        logits = frame_logits.detach()
        reliability = frame_reliability.detach()
        mask = cell_mask.to(device=latent.device, dtype=torch.bool)
        probabilities = torch.softmax(logits, dim=-1)
        entropy = -(probabilities * _safe_log(probabilities)).sum(dim=-1, keepdim=True) / math.log(
            self.num_classes
        )
        confidence = probabilities.amax(dim=-1, keepdim=True)
        top2 = logits.topk(2, dim=-1).values
        margin = top2[..., :1] - top2[..., 1:2]
        logit_norm = logits.norm(dim=-1, keepdim=True) / math.sqrt(float(self.num_classes))
        topology_a, topology_b, topology_dispersion = _distribution_topology_statistics(
            probabilities,
            self.topology_positions,
            circular=self.circular,
        )
        latent_mean = _masked_temporal_mean(latent, mask)
        cosine_disagreement = 1.0 - F.cosine_similarity(
            latent,
            latent_mean.unsqueeze(1),
            dim=-1,
            eps=1e-8,
        ).unsqueeze(-1)
        frame_features = torch.cat(
            (
                reliability,
                entropy,
                confidence,
                margin,
                logit_norm,
                topology_a,
                topology_b,
                topology_dispersion,
                cosine_disagreement,
            ),
            dim=-1,
        )
        frame_features = frame_features * mask.unsqueeze(-1).to(dtype=frame_features.dtype)
        mean, std, minimum = _masked_temporal_moments(frame_features, mask)
        distribution_divergence = _temporal_distribution_divergence(probabilities, mask)
        topology_drift = _temporal_topology_drift(
            probabilities,
            mask,
            self.topology_positions,
            circular=self.circular,
        )
        valid_fraction = mask.to(dtype=frame_features.dtype).mean(dim=1).unsqueeze(-1)
        temporal_features = torch.cat(
            (mean, std, minimum, distribution_divergence, topology_drift, valid_fraction),
            dim=-1,
        )
        return FrameEvidence(
            cell_mask=mask,
            frame_probabilities=probabilities,
            frame_features=frame_features,
            temporal_features=temporal_features,
        )

    def temporal_pool(
        self,
        latent_sequence: torch.Tensor,
        evidence: FrameEvidence,
    ) -> TemporalPoolOutput:
        """Pool frames uniformly or with the H2R within-modality health gate."""
        if latent_sequence.ndim != 4 or tuple(latent_sequence.shape[:3]) != tuple(evidence.cell_mask.shape):
            raise ValueError("latent_sequence must be [B,T,M,D] and match evidence.cell_mask.")
        latent = latent_sequence.detach()
        if self.frame_health_head is None:
            logits = torch.zeros_like(evidence.cell_mask, dtype=evidence.frame_features.dtype)
        else:
            logits = self.frame_health_head(evidence.frame_features).squeeze(-1)
        weights = masked_temporal_softmax(logits, evidence.cell_mask)
        features = (latent * weights.unsqueeze(-1).to(dtype=latent.dtype)).sum(dim=1)
        return TemporalPoolOutput(features=features, logits=logits, weights=weights)

    def route(
        self,
        base_features: torch.Tensor,
        unimodal_logits: torch.Tensor,
        evidence: FrameEvidence,
        temporal: TemporalPoolOutput,
        available: torch.Tensor | None = None,
    ) -> ReliabilityRouteOutput:
        """Produce prior-anchored modality weights from the enabled evidence groups."""
        batch_size, modality_count = evidence.cell_mask.shape[0], evidence.cell_mask.shape[2]
        expected_base = (batch_size, modality_count, self.base_feature_dim)
        expected_logits = (batch_size, modality_count, self.num_classes)
        if tuple(base_features.shape) != expected_base:
            raise ValueError(f"base_features must have shape {expected_base}, got {tuple(base_features.shape)}.")
        if tuple(unimodal_logits.shape) != expected_logits:
            raise ValueError(f"unimodal_logits must have shape {expected_logits}, got {tuple(unimodal_logits.shape)}.")
        if tuple(temporal.weights.shape) != tuple(evidence.cell_mask.shape):
            raise ValueError("temporal weights must match evidence.cell_mask.")
        effective_available = evidence.cell_mask.any(dim=1)
        if available is not None:
            supplied = torch.as_tensor(available, device=effective_available.device, dtype=torch.bool)
            if tuple(supplied.shape) != tuple(effective_available.shape):
                raise ValueError("available must have shape [B,M].")
            effective_available = effective_available & supplied
        if not bool(effective_available.any(dim=1).all().item()):
            raise ValueError("PrototypeReliabilityRouter requires one available modality per sample.")

        groups = [base_features.detach()]
        if self.use_patr:
            groups.append(evidence.temporal_features.to(device=base_features.device, dtype=base_features.dtype))
        if self.use_h2r:
            groups.append(
                _health_summary(temporal, evidence.cell_mask).to(
                    device=base_features.device,
                    dtype=base_features.dtype,
                )
            )
        consensus_features = torch.zeros(
            batch_size,
            modality_count,
            CONSENSUS_FEATURE_DIM,
            device=base_features.device,
            dtype=base_features.dtype,
        )
        if self.use_core:
            consensus_features = leave_one_out_consensus_features(
                torch.softmax(unimodal_logits.detach(), dim=-1),
                effective_available,
                self.topology_positions,
                circular=self.circular,
                top_k=self.top_k,
            ).to(dtype=base_features.dtype)
            groups.append(consensus_features)
        modality_features = torch.cat(groups, dim=-1)
        if int(modality_features.shape[-1]) != self.residual_input_dim:
            raise RuntimeError("Candidate Router feature schema does not match its residual head.")
        residual = self.modality_residual_head(modality_features).squeeze(-1)
        residual = _available_center(residual, effective_available)
        bounded_residual = self.residual_scale * torch.tanh(residual)
        prior_logits = self.prior_logits.to(device=residual.device, dtype=residual.dtype).unsqueeze(0).expand_as(residual)
        gate_logits = prior_logits + bounded_residual
        weights = _masked_modality_softmax(gate_logits, effective_available)
        prior_weights = _masked_modality_softmax(prior_logits, effective_available)
        effective_cell_weights = temporal.weights * weights.unsqueeze(1)
        return ReliabilityRouteOutput(
            gate_logits=gate_logits,
            weights=weights,
            prior_weights=prior_weights,
            residual_logits=bounded_residual,
            modality_features=modality_features,
            consensus_features=consensus_features,
            effective_cell_weights=effective_cell_weights,
        )


def masked_temporal_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or tuple(logits.shape) != tuple(mask.shape):
        raise ValueError("temporal logits and mask must share shape [B,T,M].")
    valid = mask.to(device=logits.device, dtype=torch.bool)
    scores = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    weights = torch.softmax(scores, dim=1) * valid.to(dtype=logits.dtype)
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)


def leave_one_out_consensus_features(
    probabilities: torch.Tensor,
    available: torch.Tensor,
    topology_positions: torch.Tensor,
    *,
    circular: bool,
    top_k: int,
) -> torch.Tensor:
    """Return JSD, topology distance, Top-k overlap, and consensus validity."""
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape [B,M,C].")
    batch_size, modality_count, num_classes = probabilities.shape
    usable = torch.as_tensor(available, device=probabilities.device, dtype=torch.bool)
    if tuple(usable.shape) != (batch_size, modality_count):
        raise ValueError("available must have shape [B,M].")
    if tuple(topology_positions.shape) != (num_classes,):
        raise ValueError("topology_positions must have shape [C].")
    if not 0 < int(top_k) <= num_classes:
        raise ValueError("top_k must be in [1,C].")
    probability = probabilities.detach()
    if not bool(torch.isfinite(probability).all().item()) or bool((probability < 0).any().item()):
        raise ValueError("probabilities must be finite and non-negative.")
    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(probability.dtype).tiny
    )
    usable_float = usable.to(dtype=probability.dtype)
    total = (probability * usable_float.unsqueeze(-1)).sum(dim=1, keepdim=True)
    other_count = usable_float.sum(dim=1, keepdim=True) - usable_float
    valid_consensus = usable & other_count.gt(0)
    consensus = (total - probability * usable_float.unsqueeze(-1)) / other_count.clamp_min(1.0).unsqueeze(-1)
    fallback = torch.full_like(consensus, 1.0 / float(num_classes))
    consensus = torch.where(valid_consensus.unsqueeze(-1), consensus, fallback)

    midpoint = 0.5 * (probability + consensus)
    jsd = 0.5 * (
        (probability * (_safe_log(probability) - _safe_log(midpoint))).sum(dim=-1)
        + (consensus * (_safe_log(consensus) - _safe_log(midpoint))).sum(dim=-1)
    ) / math.log(2.0)
    prediction = probability.argmax(dim=-1)
    consensus_prediction = consensus.argmax(dim=-1)
    positions = topology_positions.to(device=probability.device, dtype=probability.dtype)
    distance = _normalized_position_distance(
        positions[prediction],
        positions[consensus_prediction],
        num_classes,
        circular=circular,
    )
    k = int(top_k)
    own_topk = probability.topk(k, dim=-1).indices
    consensus_topk = consensus.topk(k, dim=-1).indices
    overlap = (
        own_topk.unsqueeze(-1).eq(consensus_topk.unsqueeze(-2)).any(dim=-1).to(dtype=probability.dtype).mean(dim=-1)
    )
    output = torch.stack((jsd, distance, overlap, valid_consensus.to(dtype=probability.dtype)), dim=-1)
    return output * valid_consensus.unsqueeze(-1).to(dtype=output.dtype)


def _normalized_prior(value: Sequence[float] | torch.Tensor | None, count: int) -> torch.Tensor:
    prior = torch.ones(count, dtype=torch.float32) if value is None else torch.as_tensor(value, dtype=torch.float32)
    prior = prior.reshape(-1)
    if tuple(prior.shape) != (count,):
        raise ValueError(f"prior_weights must contain {count} values.")
    if not bool(torch.isfinite(prior).all().item()) or bool((prior <= 0).any().item()):
        raise ValueError("prior_weights must be finite and strictly positive.")
    return prior / prior.sum()


def _zero_initialized_mlp(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    module = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)
    return module


def _validate_frame_inputs(
    latent: torch.Tensor,
    logits: torch.Tensor,
    reliability: torch.Tensor,
    mask: torch.Tensor,
    *,
    modality_count: int,
    num_classes: int,
) -> None:
    if latent.ndim != 4:
        raise ValueError("latent_sequence must have shape [B,T,M,D].")
    batch_size, steps, modalities, _ = latent.shape
    if modalities != modality_count:
        raise ValueError(f"latent_sequence must contain {modality_count} modalities.")
    if tuple(logits.shape) != (batch_size, steps, modalities, num_classes):
        raise ValueError("frame_logits must have shape [B,T,M,C].")
    if tuple(reliability.shape) != (batch_size, steps, modalities, 1):
        raise ValueError("frame_reliability must have shape [B,T,M,1].")
    if tuple(mask.shape) != (batch_size, steps, modalities):
        raise ValueError("cell_mask must have shape [B,T,M].")
    if not bool(torch.as_tensor(mask, dtype=torch.bool).any(dim=(1, 2)).all().item()):
        raise ValueError("each sample must retain at least one temporal cell.")
    if not bool(torch.isfinite(latent).all().item()) or not bool(torch.isfinite(logits).all().item()):
        raise ValueError("latent_sequence and frame_logits must be finite.")
    if not bool(torch.isfinite(reliability).all().item()):
        raise ValueError("frame_reliability must be finite.")


def _distribution_topology_statistics(
    probabilities: torch.Tensor,
    positions: torch.Tensor,
    *,
    circular: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    position = positions.to(device=probabilities.device, dtype=probabilities.dtype)
    count = int(position.numel())
    if circular:
        angle = 2.0 * math.pi * position / float(count)
        first = (probabilities * angle.cos()).sum(dim=-1, keepdim=True)
        second = (probabilities * angle.sin()).sum(dim=-1, keepdim=True)
        dispersion = 1.0 - torch.sqrt(first.square() + second.square()).clamp(0.0, 1.0)
        return first, second, dispersion
    normalized = position / float(max(count - 1, 1))
    mean = (probabilities * normalized).sum(dim=-1, keepdim=True)
    variance = (probabilities * (normalized - mean).square()).sum(dim=-1, keepdim=True)
    return mean, torch.zeros_like(mean), variance.clamp_min(0.0).sqrt()


def _masked_temporal_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_temporal_moments(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    count = weights.sum(dim=1).clamp_min(1.0)
    mean = (values * weights).sum(dim=1) / count
    variance = ((values - mean.unsqueeze(1)).square() * weights).sum(dim=1) / count
    minimum = values.masked_fill(~mask.unsqueeze(-1), torch.finfo(values.dtype).max).amin(dim=1)
    minimum = torch.where(mask.any(dim=1).unsqueeze(-1), minimum, torch.zeros_like(minimum))
    # sqrt'(0) is infinite; zero-initialized health logits otherwise create NaN gradients.
    std = variance.clamp_min(torch.finfo(values.dtype).eps).sqrt()
    return mean, std, minimum


def _temporal_distribution_divergence(probabilities: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=probabilities.dtype).unsqueeze(-1)
    mean = (probabilities * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    midpoint = 0.5 * (probabilities + mean.unsqueeze(1))
    jsd = 0.5 * (
        (probabilities * (_safe_log(probabilities) - _safe_log(midpoint))).sum(dim=-1)
        + (mean.unsqueeze(1) * (_safe_log(mean.unsqueeze(1)) - _safe_log(midpoint))).sum(dim=-1)
    ) / math.log(2.0)
    return _masked_scalar_mean(jsd, mask).unsqueeze(-1)


def _temporal_topology_drift(
    probabilities: torch.Tensor,
    mask: torch.Tensor,
    topology_positions: torch.Tensor,
    *,
    circular: bool,
) -> torch.Tensor:
    if probabilities.shape[1] < 2:
        return torch.zeros(
            probabilities.shape[0], probabilities.shape[2], 1, device=probabilities.device, dtype=probabilities.dtype
        )
    prediction = probabilities.argmax(dim=-1)
    positions = topology_positions.to(device=probabilities.device, dtype=probabilities.dtype)[prediction]
    distance = _normalized_position_distance(
        positions[:, 1:],
        positions[:, :-1],
        probabilities.shape[-1],
        circular=circular,
    )
    pair_mask = mask[:, 1:] & mask[:, :-1]
    return _masked_scalar_mean(distance, pair_mask).unsqueeze(-1)


def _normalized_position_distance(
    first: torch.Tensor,
    second: torch.Tensor,
    count: int,
    *,
    circular: bool,
) -> torch.Tensor:
    distance = (first - second).abs()
    if circular:
        distance = torch.minimum(distance, float(count) - distance)
        denominator = max(float(count) / 2.0, 1.0)
    else:
        denominator = float(max(count - 1, 1))
    return distance / denominator


def _masked_scalar_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _health_summary(temporal: TemporalPoolOutput, mask: torch.Tensor) -> torch.Tensor:
    mean, std, minimum = _masked_temporal_moments(temporal.logits.unsqueeze(-1), mask)
    max_weight = temporal.weights.amax(dim=1).unsqueeze(-1)
    return torch.cat((mean, std, minimum, max_weight), dim=-1)


def _available_center(values: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    weights = available.to(device=values.device, dtype=values.dtype)
    mean = (values * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (values - mean) * weights


def _masked_modality_softmax(logits: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    usable = available.to(device=logits.device, dtype=torch.bool)
    masked = logits.masked_fill(~usable, torch.finfo(logits.dtype).min)
    return torch.softmax(masked, dim=1) * usable.to(dtype=logits.dtype)


def _safe_log(values: torch.Tensor) -> torch.Tensor:
    return values.clamp_min(torch.finfo(values.dtype).tiny).log()


__all__ = [
    "CONSENSUS_FEATURE_DIM",
    "FRAME_FEATURE_DIM",
    "H2R_FEATURE_DIM",
    "ROUTER_VARIANTS",
    "TEMPORAL_FEATURE_DIM",
    "FrameEvidence",
    "PrototypeReliabilityRouter",
    "ReliabilityRouteOutput",
    "TemporalPoolOutput",
    "leave_one_out_consensus_features",
    "masked_temporal_softmax",
]
