"""Loss terms for radio-guided hierarchical beam prototypes."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kd_sensing.models.prototype_fusion_losses import topology_risk


def beam_classification_loss(
    beam_evidence: torch.Tensor,
    targets: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    item = F.cross_entropy(beam_evidence.float(), targets.long(), reduction="none")
    if sample_weights is None:
        return item.mean()
    weight = torch.as_tensor(sample_weights, device=item.device, dtype=torch.float32)
    return (item * weight).sum() / weight.sum().clamp_min(1e-12)


def beam_topology_loss(
    beam_evidence: torch.Tensor,
    targets: torch.Tensor,
    topology_distance: torch.Tensor,
) -> torch.Tensor:
    return topology_risk(beam_evidence.float(), targets.long(), topology_distance.float()).mean()


def prototype_anchor_loss(subprototypes: torch.Tensor, base_prototypes: torch.Tensor) -> torch.Tensor:
    sub = torch.as_tensor(subprototypes).float()
    base = torch.as_tensor(base_prototypes, device=sub.device).float()
    if sub.ndim != 3 or base.shape != sub.shape[::2]:
        expected = (sub.shape[0], sub.shape[-1]) if sub.ndim == 3 else "[beam, dim]"
        raise ValueError(f"base_prototypes must have shape {expected}.")
    mean = F.normalize(sub.mean(dim=1), dim=-1)
    return (1.0 - F.cosine_similarity(mean, F.normalize(base, dim=-1), dim=-1)).mean()


def prototype_diversity_loss(
    subprototypes: torch.Tensor,
    valid_subcluster_mask: torch.Tensor,
    *,
    max_similarity: float = 0.98,
) -> torch.Tensor:
    sub = torch.as_tensor(subprototypes).float()
    valid = torch.as_tensor(valid_subcluster_mask, device=sub.device, dtype=torch.bool)
    if sub.ndim != 3 or sub.shape[1] != 2 or valid.shape != sub.shape[:1]:
        raise ValueError("diversity requires subprototypes [beam,2,dim] and valid mask [beam].")
    if not bool(valid.any()):
        return sub.sum() * 0.0
    similarity = F.cosine_similarity(sub[:, 0], sub[:, 1], dim=-1)
    return F.relu(similarity[valid] - float(max_similarity)).mean()


def adapter_regularization(adapter_residual: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(adapter_residual).float().square().sum(dim=-1).mean()


def full_teacher_consistency_loss(
    missing_evidence: torch.Tensor,
    full_evidence: torch.Tensor,
    targets: torch.Tensor,
    *,
    minimum_margin: float = 0.2,
) -> torch.Tensor:
    missing = torch.as_tensor(missing_evidence).float()
    full = torch.as_tensor(full_evidence, device=missing.device).float().detach()
    labels = torch.as_tensor(targets, device=missing.device, dtype=torch.long)
    if missing.shape != full.shape or missing.ndim != 2 or labels.shape != missing.shape[:1]:
        raise ValueError("evidence and targets must have shapes [B,beam] and [B].")
    full_probability = torch.softmax(full, dim=-1)
    top = full_probability.topk(2, dim=-1)
    confident = full.argmax(dim=-1).eq(labels) & ((top.values[:, 0] - top.values[:, 1]) >= float(minimum_margin))
    if not bool(confident.any()):
        return missing.sum() * 0.0
    item = F.kl_div(
        F.log_softmax(missing[confident], dim=-1),
        full_probability[confident],
        reduction="batchmean",
    )
    return item


__all__ = [
    "adapter_regularization",
    "beam_classification_loss",
    "beam_topology_loss",
    "full_teacher_consistency_loss",
    "prototype_anchor_loss",
    "prototype_diversity_loss",
]
