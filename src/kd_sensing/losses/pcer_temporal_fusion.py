from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.losses.beam_prototype_alignment import make_soft_beam_labels


def prototype_evidence_consistency_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_mask: torch.Tensor,
    teacher_mask: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if float(temperature) <= 0.0:
        raise ValueError("PCER distill temperature must be positive.")
    student = _prediction_logits(student_logits).float()
    teacher = _prediction_logits(teacher_logits).to(device=student.device, dtype=torch.float32).detach()
    if tuple(student.shape) != tuple(teacher.shape):
        raise ValueError("PCER student and teacher logits must share shape [B,H,K].")
    masked = torch.as_tensor(student_mask, device=student.device, dtype=torch.bool)
    full = torch.as_tensor(teacher_mask, device=student.device, dtype=torch.bool)
    if masked.ndim != 3 or tuple(masked.shape) != tuple(full.shape) or masked.shape[0] != student.shape[0]:
        raise ValueError("PCER student and teacher masks must share shape [B,T,M].")
    if bool((masked & ~full).any().item()):
        raise ValueError("PCER student availability must be a subset of teacher availability.")
    active = (full & ~masked).any(dim=(1, 2))
    scale = float(temperature)
    target = torch.softmax(teacher / scale, dim=-1)
    per_horizon = F.kl_div(
        F.log_softmax(student / scale, dim=-1),
        target,
        reduction="none",
    ).sum(dim=-1) * scale**2
    per_sample = per_horizon.mean(dim=1)
    loss = per_sample[active].mean() if bool(active.any().item()) else student.sum() * 0.0
    return loss, {
        "loss/pcer_mask_consistency": float(loss.detach().cpu().item()),
        "pcer_mask_consistency_active_ratio": float(active.float().mean().cpu().item()),
    }


def counterfactual_router_targets(
    block_evidence_logits: torch.Tensor,
    availability_mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    beam_label_sigma: float,
    circular: bool,
    topology_id: str | None,
    topology_permutation: list[int] | tuple[int, ...] | torch.Tensor | None,
    contribution_temperature: float,
    contribution_clip: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    evidence = block_evidence_logits.detach().float()
    if evidence.ndim != 3:
        raise ValueError("block_evidence_logits must have shape [B,N,K].")
    batch, blocks, classes = evidence.shape
    available = torch.as_tensor(availability_mask, device=evidence.device, dtype=torch.bool)
    if tuple(available.shape) != (batch, blocks):
        raise ValueError("availability_mask must have shape [B,N].")
    count = available.sum(dim=1)
    if not bool(count.gt(1).all().item()):
        raise ValueError("Counterfactual PCER targets require at least two available blocks per sample.")
    if float(contribution_temperature) <= 0.0:
        raise ValueError("PCER contribution temperature must be positive.")
    if contribution_clip is not None and float(contribution_clip) <= 0.0:
        raise ValueError("PCER contribution clip must be positive when set.")

    hard = labels.to(device=evidence.device, dtype=torch.long)
    if hard.ndim > 1:
        hard = hard[:, 0]
    hard = hard.reshape(-1)
    if tuple(hard.shape) != (batch,):
        raise ValueError("PCER labels must have shape [B] or [B,H].")
    target = make_soft_beam_labels(
        hard,
        classes,
        float(beam_label_sigma),
        circular=bool(circular),
        topology_id=topology_id,
        topology_permutation=topology_permutation,
    ).float()

    valid = available.to(dtype=evidence.dtype)
    evidence_sum = (evidence * valid.unsqueeze(-1)).sum(dim=1)
    all_logits = evidence_sum / count.to(dtype=evidence.dtype).unsqueeze(-1)
    loo_logits = (evidence_sum.unsqueeze(1) - evidence) / (count - 1).to(dtype=evidence.dtype).view(batch, 1, 1)
    loss_all = -(target * F.log_softmax(all_logits, dim=-1)).sum(dim=-1)
    loss_without = -(target.unsqueeze(1) * F.log_softmax(loo_logits, dim=-1)).sum(dim=-1)
    contribution = loss_without - loss_all.unsqueeze(1)
    mean = (contribution * valid).sum(dim=1, keepdim=True) / count.to(dtype=evidence.dtype).unsqueeze(1)
    contribution = contribution - mean
    if contribution_clip is not None:
        contribution = contribution.clamp(min=-float(contribution_clip), max=float(contribution_clip))
    masked_contribution = contribution.masked_fill(~available, -torch.inf)
    target_weights = torch.softmax(masked_contribution / float(contribution_temperature), dim=-1)
    target_weights = target_weights.masked_fill(~available, 0.0)
    return target_weights, masked_contribution


def standalone_quality_router_targets(
    block_evidence_logits: torch.Tensor,
    availability_mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    beam_label_sigma: float,
    circular: bool,
    topology_id: str | None,
    topology_permutation: list[int] | tuple[int, ...] | torch.Tensor | None,
    quality_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    evidence = block_evidence_logits.detach().float()
    available = torch.as_tensor(availability_mask, device=evidence.device, dtype=torch.bool)
    if evidence.ndim != 3 or tuple(available.shape) != tuple(evidence.shape[:2]):
        raise ValueError("Standalone quality evidence/availability must have shapes [B,N,K] and [B,N].")
    target = _soft_target(evidence, labels, beam_label_sigma, circular, topology_id, topology_permutation)
    standalone_loss = -(target.unsqueeze(1) * F.log_softmax(evidence, dim=-1)).sum(dim=-1)
    quality = -standalone_loss
    valid = available.to(dtype=quality.dtype)
    count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (quality * valid).sum(dim=1, keepdim=True) / count
    variance = ((quality - mean).square() * valid).sum(dim=1, keepdim=True) / count
    normalized = (quality - mean) / variance.sqrt().clamp_min(1e-6)
    return _masked_target_softmax(normalized, available, quality_temperature), normalized.masked_fill(~available, -torch.inf)


def onpolicy_block_router_targets(
    block_features: torch.Tensor,
    block_evidence_logits: torch.Tensor,
    availability_mask: torch.Tensor,
    predicted_weights: torch.Tensor,
    labels: torch.Tensor,
    *,
    route_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, torch.Tensor]],
    beam_label_sigma: float,
    circular: bool,
    topology_id: str | None,
    topology_permutation: list[int] | tuple[int, ...] | torch.Tensor | None,
    contribution_temperature: float,
    contribution_clip: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = block_features.detach()
    evidence = block_evidence_logits.detach().float()
    available = availability_mask.detach().bool()
    batch, blocks, classes = evidence.shape
    if available.sum(dim=1).min().item() <= 1:
        raise ValueError("On-policy block targets require at least two available blocks.")
    target = _soft_target(evidence, labels, beam_label_sigma, circular, topology_id, topology_permutation)
    full_logits = (predicted_weights.detach().float().unsqueeze(-1) * evidence).sum(dim=1)
    loss_full = -(target * F.log_softmax(full_logits, dim=-1)).sum(dim=-1)
    expanded_available = available.unsqueeze(1).expand(batch, blocks, blocks).clone()
    diagonal = torch.arange(blocks, device=available.device)
    expanded_available[:, diagonal, diagonal] = False
    routed = route_fn(
        features.unsqueeze(1).expand(-1, blocks, -1, -1).reshape(batch * blocks, blocks, features.shape[-1]),
        evidence.unsqueeze(1).expand(-1, blocks, -1, -1).reshape(batch * blocks, blocks, classes),
        expanded_available.reshape(batch * blocks, blocks),
    )
    loo_weights = routed["weights"].reshape(batch, blocks, blocks)
    loo_logits = (loo_weights.unsqueeze(-1) * evidence.unsqueeze(1)).sum(dim=2)
    loss_without = -(target.unsqueeze(1) * F.log_softmax(loo_logits, dim=-1)).sum(dim=-1)
    contribution = loss_without - loss_full.unsqueeze(1)
    return _contribution_target(contribution, available, contribution_temperature, contribution_clip)


def onpolicy_modality_router_targets(
    block_features: torch.Tensor,
    block_evidence_logits: torch.Tensor,
    availability_mask: torch.Tensor,
    predicted_weights: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_timesteps: int,
    num_modalities: int,
    route_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, torch.Tensor]],
    beam_label_sigma: float,
    circular: bool,
    topology_id: str | None,
    topology_permutation: list[int] | tuple[int, ...] | torch.Tensor | None,
    contribution_temperature: float,
    contribution_clip: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = block_features.detach()
    evidence = block_evidence_logits.detach().float()
    available = availability_mask.detach().bool()
    batch, blocks, classes = evidence.shape
    if blocks != int(num_timesteps) * int(num_modalities):
        raise ValueError("On-policy modality target block count differs from T*M.")
    cell = available.reshape(batch, int(num_timesteps), int(num_modalities))
    modality_available = cell.any(dim=1)
    if modality_available.sum(dim=1).min().item() <= 1:
        raise ValueError("On-policy modality targets require at least two available modalities.")
    target = _soft_target(evidence, labels, beam_label_sigma, circular, topology_id, topology_permutation)
    full_logits = (predicted_weights.detach().float().unsqueeze(-1) * evidence).sum(dim=1)
    loss_full = -(target * F.log_softmax(full_logits, dim=-1)).sum(dim=-1)
    expanded = cell.unsqueeze(1).expand(batch, int(num_modalities), -1, -1).clone()
    for modality in range(int(num_modalities)):
        expanded[:, modality, :, modality] = False
    routed = route_fn(
        features.unsqueeze(1).expand(-1, int(num_modalities), -1, -1).reshape(batch * int(num_modalities), blocks, features.shape[-1]),
        evidence.unsqueeze(1).expand(-1, int(num_modalities), -1, -1).reshape(batch * int(num_modalities), blocks, classes),
        expanded.reshape(batch * int(num_modalities), blocks),
    )
    loo_weights = routed["weights"].reshape(batch, int(num_modalities), blocks)
    loo_logits = (loo_weights.unsqueeze(-1) * evidence.unsqueeze(1)).sum(dim=2)
    loss_without = -(target.unsqueeze(1) * F.log_softmax(loo_logits, dim=-1)).sum(dim=-1)
    contribution = loss_without - loss_full.unsqueeze(1)
    return _contribution_target(contribution, modality_available, contribution_temperature, contribution_clip)


def _soft_target(evidence, labels, sigma, circular, topology_id, topology_permutation):
    hard = labels.to(device=evidence.device, dtype=torch.long)
    if hard.ndim > 1:
        hard = hard[:, 0]
    return make_soft_beam_labels(
        hard,
        evidence.shape[-1],
        float(sigma),
        circular=bool(circular),
        topology_id=topology_id,
        topology_permutation=topology_permutation,
    ).float()


def _contribution_target(contribution, available, temperature, clip):
    valid = available.to(dtype=contribution.dtype)
    count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    centered = contribution - (contribution * valid).sum(dim=1, keepdim=True) / count
    if clip is not None:
        centered = centered.clamp(min=-float(clip), max=float(clip))
    return _masked_target_softmax(centered, available, temperature), centered.masked_fill(~available, -torch.inf)


def _masked_target_softmax(values, available, temperature):
    if float(temperature) <= 0:
        raise ValueError("Router target temperature must be positive.")
    target = torch.softmax(values.masked_fill(~available, -torch.inf) / float(temperature), dim=-1)
    return target.masked_fill(~available, 0.0).detach()


def counterfactual_router_loss(
    predicted_weights: torch.Tensor,
    target_weights: torch.Tensor,
    availability_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    predicted = predicted_weights.float()
    target = target_weights.to(device=predicted.device, dtype=torch.float32).detach()
    available = torch.as_tensor(availability_mask, device=predicted.device, dtype=torch.bool)
    if predicted.ndim != 2 or tuple(predicted.shape) != tuple(target.shape) or tuple(predicted.shape) != tuple(available.shape):
        raise ValueError("PCER predicted, target, and availability weights must share shape [B,N].")
    if not bool(torch.isfinite(predicted).all().item()) or not bool(torch.isfinite(target).all().item()):
        raise ValueError("PCER Router weights must be finite.")
    tiny = torch.finfo(predicted.dtype).tiny
    per_sample = (
        target * (target.clamp_min(tiny).log() - predicted.clamp_min(tiny).log())
    ).masked_fill(~available, 0.0).sum(dim=-1)
    loss = per_sample.mean()
    prediction_entropy = -(predicted * predicted.clamp_min(tiny).log()).sum(dim=-1)
    target_entropy = -(target * target.clamp_min(tiny).log()).sum(dim=-1)
    correlation = _pearson(predicted, target, available)
    spearman = _pearson(_rank_rows(predicted), _rank_rows(target), available)
    count = available.sum(dim=-1).float()
    normalized_entropy = target_entropy / count.log().clamp_min(1e-12)
    sorted_target = target.sort(dim=-1, descending=True).values
    return loss, {
        "loss/pcer_route": float(loss.detach().cpu().item()),
        "pcer_router_prediction_entropy": float(prediction_entropy.detach().mean().cpu().item()),
        "pcer_router_target_entropy": float(target_entropy.detach().mean().cpu().item()),
        "pcer_router_target_pearson": float(correlation.detach().mean().cpu().item()),
        "pcer_router_target_spearman": float(spearman.detach().mean().cpu().item()),
        "pcer_router_target_normalized_entropy": float(normalized_entropy.detach().mean().cpu().item()),
        "pcer_router_target_top1_top2_margin": float(
            (sorted_target[:, 0] - sorted_target[:, 1]).detach().mean().cpu().item()
        ),
        "pcer_router_top1_agreement": float(
            predicted.argmax(dim=-1).eq(target.argmax(dim=-1)).float().mean().detach().cpu().item()
        ),
    }


def _pearson(first: torch.Tensor, second: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    weights = available.to(dtype=first.dtype)
    count = weights.sum(dim=-1).clamp_min(1.0)
    first_mean = (first * weights).sum(dim=-1) / count
    second_mean = (second * weights).sum(dim=-1) / count
    first_centered = (first - first_mean.unsqueeze(-1)) * weights
    second_centered = (second - second_mean.unsqueeze(-1)) * weights
    numerator = (first_centered * second_centered).sum(dim=-1)
    denominator = first_centered.square().sum(dim=-1).sqrt() * second_centered.square().sum(dim=-1).sqrt()
    return torch.where(denominator.gt(0), numerator / denominator.clamp_min(1e-12), torch.zeros_like(numerator))


def _rank_rows(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort(dim=-1)
    ranks = torch.empty_like(values)
    rank_values = torch.arange(values.shape[-1], device=values.device, dtype=values.dtype).expand_as(values)
    return ranks.scatter(1, order, rank_values)


def _prediction_logits(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value.unsqueeze(1)
    if value.ndim != 3:
        raise ValueError("PCER prediction logits must have shape [B,K] or [B,H,K].")
    return value


__all__ = [
    "counterfactual_router_loss",
    "counterfactual_router_targets",
    "onpolicy_block_router_targets",
    "onpolicy_modality_router_targets",
    "prototype_evidence_consistency_loss",
    "standalone_quality_router_targets",
]
