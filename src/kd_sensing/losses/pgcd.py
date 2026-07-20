"""Beam-topology degradation targets and quality losses for PGCD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.losses.beam_prototype_alignment import beam_topology_positions, make_soft_beam_labels
from kd_sensing.models.pgcd import PGCD_VARIANTS


@dataclass(frozen=True)
class PGCDTargets:
    target: torch.Tensor
    topology_drift: torch.Tensor
    topology_transport: torch.Tensor
    task_degradation: torch.Tensor
    task_degradation_raw: torch.Tensor
    clean_block_loss: torch.Tensor
    corrupted_block_loss: torch.Tensor
    consistency: torch.Tensor


@dataclass(frozen=True)
class PGCDLossResult:
    total: torch.Tensor
    regression: torch.Tensor
    ranking: torch.Tensor
    consistency: torch.Tensor
    targets: PGCDTargets
    diagnostics: dict[str, float]


def beam_topology_distance_matrix(
    num_beams: int,
    *,
    topology_id: str,
    topology_permutation: list[int] | tuple[int, ...] | torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    positions = beam_topology_positions(
        int(num_beams),
        topology_id=str(topology_id),
        topology_permutation=topology_permutation,
        device=device,
    )
    distance = (positions[:, None] - positions[None, :]).abs()
    if str(topology_id).strip().lower() != "linear_index_v1":
        distance = torch.minimum(distance, float(num_beams) - distance)
    return distance / distance.max().clamp_min(1.0)


def topology_transport(
    clean_probability: torch.Tensor,
    corrupted_probability: torch.Tensor,
    distance_matrix: torch.Tensor,
) -> torch.Tensor:
    if clean_probability.shape != corrupted_probability.shape or clean_probability.ndim != 3:
        raise ValueError("PGCD prototype probabilities must share shape [B,N,K].")
    if tuple(distance_matrix.shape) != (clean_probability.shape[-1], clean_probability.shape[-1]):
        raise ValueError("PGCD topology distance matrix must have shape [K,K].")
    return torch.einsum("bnk,kl,bnl->bn", clean_probability, distance_matrix, corrupted_probability)


def debiased_topology_drift(
    clean_probability: torch.Tensor,
    corrupted_probability: torch.Tensor,
    distance_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cross = topology_transport(clean_probability, corrupted_probability, distance_matrix)
    clean_self = topology_transport(clean_probability, clean_probability, distance_matrix)
    corrupted_self = topology_transport(corrupted_probability, corrupted_probability, distance_matrix)
    return (cross - 0.5 * clean_self - 0.5 * corrupted_self).clamp_min(0.0), cross


def pgcd_degradation_targets(
    clean_block_logits: torch.Tensor,
    corrupted_block_logits: torch.Tensor,
    target_beam: torch.Tensor,
    availability_mask: torch.Tensor,
    *,
    severity: torch.Tensor,
    target_mode: str,
    topology_id: str,
    topology_permutation: list[int] | tuple[int, ...] | None = None,
    beam_label_sigma: float = 2.0,
    alpha_drift: float = 0.5,
    alpha_task: float = 0.5,
    task_clip: float = 4.0,
) -> PGCDTargets:
    if clean_block_logits.shape != corrupted_block_logits.shape or clean_block_logits.ndim != 3:
        raise ValueError("PGCD clean/corrupted block logits must share shape [B,N,K].")
    batch, blocks, classes = clean_block_logits.shape
    available = torch.as_tensor(availability_mask, device=corrupted_block_logits.device, dtype=torch.bool)
    severity_tensor = torch.as_tensor(severity, device=corrupted_block_logits.device, dtype=torch.long)
    if tuple(available.shape) != (batch, blocks) or tuple(severity_tensor.shape) != (batch, blocks):
        raise ValueError("PGCD availability and severity must have shape [B,N].")
    labels = target_beam.long()
    if labels.ndim > 1:
        labels = labels[:, 0]
    if tuple(labels.shape) != (batch,):
        raise ValueError("PGCD target_beam must provide one label per sample.")
    if not 0.0 <= float(alpha_drift) <= 1.0 or not 0.0 <= float(alpha_task) <= 1.0:
        raise ValueError("PGCD target alpha values must be in [0,1].")
    if str(target_mode) == "combined" and abs(float(alpha_drift) + float(alpha_task) - 1.0) > 1e-6:
        raise ValueError("Combined PGCD target alpha values must sum to one.")

    distance = beam_topology_distance_matrix(
        classes,
        topology_id=topology_id,
        topology_permutation=topology_permutation,
        device=corrupted_block_logits.device,
    ).to(dtype=torch.float32)
    clean_probability = torch.softmax(clean_block_logits.detach().float(), dim=-1)
    corrupted_probability = torch.softmax(corrupted_block_logits.float(), dim=-1)
    topology_drift, raw_transport = debiased_topology_drift(clean_probability, corrupted_probability, distance)
    soft_target = make_soft_beam_labels(
        labels,
        classes,
        float(beam_label_sigma),
        circular=str(topology_id) != "linear_index_v1",
        topology_id=topology_id,
        topology_permutation=topology_permutation,
    ).to(device=corrupted_block_logits.device, dtype=torch.float32)
    expanded_target = soft_target.unsqueeze(1).expand(batch, blocks, classes)
    clean_loss = -(expanded_target * F.log_softmax(clean_block_logits.detach().float(), dim=-1)).sum(dim=-1)
    corrupted_loss = -(expanded_target * F.log_softmax(corrupted_block_logits.float(), dim=-1)).sum(dim=-1)
    task_raw = corrupted_loss - clean_loss.detach()
    task_degradation = task_raw.clamp(min=0.0, max=float(task_clip))
    normalized_drift = _batch_normalize(topology_drift.detach(), available)
    normalized_task = _batch_normalize(task_degradation.detach(), available)
    mode = str(target_mode).strip().lower()
    if mode == "severity":
        target = severity_tensor.float() / 4.0
    elif mode == "topology":
        target = normalized_drift
    elif mode == "task":
        target = normalized_task
    elif mode == "combined":
        target = float(alpha_drift) * normalized_drift + float(alpha_task) * normalized_task
    elif mode == "none":
        target = torch.zeros_like(normalized_drift)
    else:
        raise ValueError("PGCD target_mode must be none, severity, topology, task, or combined.")
    missing = severity_tensor.eq(4) | ~available
    target = target.masked_fill(missing, 1.0).detach()
    consistency_weight = torch.exp(-target).masked_fill(~available, 0.0)
    consistency = (topology_drift * consistency_weight).sum() / consistency_weight.sum().clamp_min(1.0)
    return PGCDTargets(
        target=target,
        topology_drift=topology_drift.detach(),
        topology_transport=raw_transport.detach(),
        task_degradation=task_degradation.detach(),
        task_degradation_raw=task_raw.detach(),
        clean_block_loss=clean_loss.detach(),
        corrupted_block_loss=corrupted_loss.detach(),
        consistency=consistency,
    )


def pgcd_quality_loss(
    predicted_degradation: torch.Tensor,
    clean_block_logits: torch.Tensor,
    corrupted_block_logits: torch.Tensor,
    target_beam: torch.Tensor,
    availability_mask: torch.Tensor,
    *,
    severity: torch.Tensor,
    corrupted_mask: torch.Tensor,
    variant: str,
    topology_id: str,
    topology_permutation: list[int] | tuple[int, ...] | None = None,
    beam_label_sigma: float = 2.0,
    alpha_drift: float = 0.5,
    alpha_task: float = 0.5,
    task_clip: float = 4.0,
    lambda_quality: float = 0.2,
    lambda_rank: float = 0.1,
    lambda_consistency: float = 0.2,
    rank_margin: float = 0.1,
    rank_target_epsilon: float = 0.02,
) -> PGCDLossResult:
    name = str(variant).strip().lower()
    if name not in PGCD_VARIANTS:
        raise ValueError(f"PGCD variant must be one of {sorted(PGCD_VARIANTS)}.")
    mapping = {
        "c0": ("none", False, False, False),
        "c1": ("severity", True, False, False),
        "c2": ("none", False, False, False),
        "c3": ("topology", True, False, False),
        "c4": ("topology", True, True, False),
        "c5": ("task", True, False, False),
        "c6": ("combined", True, True, False),
        "c7": ("combined", True, True, True),
    }
    target_mode, use_regression, use_ranking, use_consistency = mapping[name]
    targets = pgcd_degradation_targets(
        clean_block_logits,
        corrupted_block_logits,
        target_beam,
        availability_mask,
        severity=severity,
        target_mode=target_mode,
        topology_id=topology_id,
        topology_permutation=topology_permutation,
        beam_label_sigma=beam_label_sigma,
        alpha_drift=alpha_drift,
        alpha_task=alpha_task,
        task_clip=task_clip,
    )
    predicted = predicted_degradation.float()
    available = torch.as_tensor(availability_mask, device=predicted.device, dtype=torch.bool)
    degraded = torch.as_tensor(corrupted_mask, device=predicted.device, dtype=torch.bool)
    regression_mask = available & degraded
    zero = predicted.sum() * 0.0
    regression = (
        F.smooth_l1_loss(predicted[regression_mask], targets.target[regression_mask], reduction="mean")
        if use_regression and bool(regression_mask.any().item())
        else zero
    )
    ranking = (
        pairwise_quality_ranking_loss(
            predicted,
            targets.target,
            severity=torch.as_tensor(severity, device=predicted.device),
            availability=available,
            margin=rank_margin,
            target_epsilon=rank_target_epsilon,
        )
        if use_ranking
        else zero
    )
    consistency = targets.consistency if use_consistency else zero
    total = float(lambda_quality) * regression + float(lambda_rank) * ranking + float(lambda_consistency) * consistency
    selected = available & degraded
    diagnostics = {
        "loss/pgcd_quality": float(regression.detach().cpu().item()),
        "loss/pgcd_rank": float(ranking.detach().cpu().item()),
        "loss/pgcd_consistency": float(consistency.detach().cpu().item()),
        "loss/pgcd_total": float(total.detach().cpu().item()),
        "pgcd/topology_transport_mean": _masked_mean(targets.topology_transport, selected),
        "pgcd/topology_drift_mean": _masked_mean(targets.topology_drift, selected),
        "pgcd/task_degradation_mean": _masked_mean(targets.task_degradation, selected),
        "pgcd/task_degradation_raw_mean": _masked_mean(targets.task_degradation_raw, selected),
        "pgcd/target_mean": _masked_mean(targets.target, selected),
        "pgcd/predicted_degradation_mean": _masked_mean(predicted.detach(), selected),
        "pgcd/regression_active_ratio": float(regression_mask.float().mean().cpu().item()),
    }
    return PGCDLossResult(total, regression, ranking, consistency, targets, diagnostics)


def pairwise_quality_ranking_loss(
    predicted_degradation: torch.Tensor,
    target_degradation: torch.Tensor,
    *,
    severity: torch.Tensor,
    availability: torch.Tensor,
    margin: float,
    target_epsilon: float,
) -> torch.Tensor:
    predicted = predicted_degradation.float()
    target = target_degradation.detach().float()
    severity = severity.long()
    available = availability.bool()
    if predicted.ndim != 2 or target.shape != predicted.shape or severity.shape != predicted.shape or available.shape != predicted.shape:
        raise ValueError("PGCD ranking inputs must share shape [B,N].")
    target_delta = target.unsqueeze(1) - target.unsqueeze(2)
    predicted_delta = predicted.unsqueeze(1) - predicted.unsqueeze(2)
    pair_mask = available.unsqueeze(1) & available.unsqueeze(2)
    pair_mask &= severity.unsqueeze(1).ne(severity.unsqueeze(2))
    pair_mask &= target_delta.abs().gt(float(target_epsilon))
    pair_mask &= torch.triu(torch.ones_like(pair_mask), diagonal=1)
    if not bool(pair_mask.any().item()):
        return predicted.sum() * 0.0
    loss = F.relu(float(margin) - target_delta.sign() * predicted_delta)
    return loss[pair_mask].mean()


def _batch_normalize(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    available = mask.bool()
    if not bool(available.any().item()):
        return torch.zeros_like(value)
    maximum = value.masked_select(available).amax().clamp_min(1e-6)
    return (value / maximum).clamp(0.0, 1.0).masked_fill(~available, 0.0)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    selected = value.detach().float().masked_select(mask.bool())
    return float(selected.mean().cpu().item()) if selected.numel() else 0.0


__all__ = [
    "PGCDLossResult",
    "PGCDTargets",
    "beam_topology_distance_matrix",
    "debiased_topology_drift",
    "pairwise_quality_ranking_loss",
    "pgcd_degradation_targets",
    "pgcd_quality_loss",
    "topology_transport",
]
