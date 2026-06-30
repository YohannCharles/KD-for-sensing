from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class BeamPrototypeBank(nn.Module):
    def __init__(self, d_model: int, num_beams: int, *, temperature: float = 0.2) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_beams = int(num_beams)
        self.temperature = float(temperature)
        if self.d_model <= 0 or self.num_beams <= 0 or self.temperature <= 0:
            raise ValueError("d_model, num_beams, and temperature must be positive.")
        self.prototypes = nn.Parameter(torch.randn(self.num_beams, self.d_model) * 0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.d_model:
            raise ValueError(f"features must have shape [B, {self.d_model}], got {tuple(features.shape)}.")
        feature = F.normalize(features, dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        return feature @ proto.t() / self.temperature


def make_soft_beam_labels(
    labels: torch.Tensor,
    num_beams: int,
    sigma: float,
    *,
    circular: bool = True,
) -> torch.Tensor:
    num_beams = int(num_beams)
    sigma = float(sigma)
    if num_beams <= 0 or sigma <= 0:
        raise ValueError("num_beams and sigma must be positive.")
    target = labels.to(dtype=torch.long).reshape(-1)
    if torch.any((target < 0) | (target >= num_beams)):
        raise ValueError(f"labels must be in [0, {num_beams - 1}].")
    beams = torch.arange(num_beams, device=target.device, dtype=torch.float32).view(1, -1)
    center = target.to(dtype=torch.float32).view(-1, 1)
    distance = (beams - center).abs()
    if circular:
        distance = torch.minimum(distance, float(num_beams) - distance)
    weights = torch.exp(-0.5 * (distance / sigma).pow(2))
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def prototype_alignment_loss(
    prototype_bank: BeamPrototypeBank,
    labels: torch.Tensor,
    *,
    fused_features: torch.Tensor,
    modality_features: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    teacher_features: torch.Tensor | None = None,
    beam_label_sigma: float = 1.0,
    beam_label_circular: bool = True,
    lambda_proto: float = 1.0,
    lambda_modality_proto: float = 0.0,
    lambda_teacher_proto: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    hard = labels.to(device=fused_features.device, dtype=torch.long)
    if hard.ndim > 1:
        hard = hard[:, 0]
    target = make_soft_beam_labels(
        hard,
        prototype_bank.num_beams,
        beam_label_sigma,
        circular=beam_label_circular,
    ).to(device=fused_features.device, dtype=fused_features.dtype)
    fused_logits = prototype_bank(fused_features)
    loss_fused = _soft_kl(fused_logits, target)
    zero = fused_logits.sum() * 0.0
    loss_modality = zero
    modality_count = 0
    if modality_features is not None and mask is not None:
        if modality_features.ndim != 3:
            raise ValueError(f"modality_features must have shape [B, M, D], got {tuple(modality_features.shape)}.")
        available = mask.to(device=modality_features.device, dtype=torch.bool)
        if tuple(available.shape) != tuple(modality_features.shape[:2]):
            raise ValueError(f"mask must have shape {tuple(modality_features.shape[:2])}, got {tuple(available.shape)}.")
        modality_count = int(available.sum().detach().cpu().item())
        if modality_count:
            flat_features = modality_features[available]
            flat_target = target.unsqueeze(1).expand(-1, modality_features.shape[1], -1)[available]
            loss_modality = _soft_kl(prototype_bank(flat_features), flat_target)
    loss_teacher = zero
    if teacher_features is not None and float(lambda_teacher_proto) != 0.0:
        loss_teacher = _soft_kl(prototype_bank(teacher_features), target)
    total = (
        float(lambda_proto) * loss_fused
        + float(lambda_modality_proto) * loss_modality
        + float(lambda_teacher_proto) * loss_teacher
    )
    top1, top5 = _topk(fused_logits, hard)
    diagnostics = {
        "loss/prototype_alignment": float(loss_fused.detach().cpu().item()),
        "loss/prototype_modality": float(loss_modality.detach().cpu().item()),
        "loss/prototype_teacher": float(loss_teacher.detach().cpu().item()),
        "prototype/top1": top1,
        "prototype/top5": top5,
        "prototype/sample_count": float(int(hard.numel())),
        "prototype/modality_sample_count": float(modality_count),
        "prototype/target_source_beam_topology": 1.0,
    }
    return total, diagnostics


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    if features.ndim != 2:
        raise ValueError(f"features must have shape [B, D], got {tuple(features.shape)}.")
    labels = labels.to(device=features.device, dtype=torch.long).reshape(-1)
    if labels.numel() != features.shape[0]:
        raise ValueError("labels and features batch size must match.")
    normalized = F.normalize(features, dim=-1)
    logits = normalized @ normalized.t() / float(temperature)
    eye = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    same = labels.view(-1, 1).eq(labels.view(1, -1)) & ~eye
    valid = same.any(dim=1)
    if not bool(valid.any().item()):
        zero = features.sum() * 0.0
        return zero, {"loss/prototype_supcon": 0.0, "prototype/supcon_anchor_count": 0.0}
    logits = logits.masked_fill(eye, torch.finfo(logits.dtype).min)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_anchor = -(log_prob * same.to(dtype=features.dtype)).sum(dim=1) / same.sum(dim=1).clamp_min(1)
    loss = per_anchor[valid].mean()
    return loss, {
        "loss/prototype_supcon": float(loss.detach().cpu().item()),
        "prototype/supcon_anchor_count": float(int(valid.sum().detach().cpu().item())),
    }


def _soft_kl(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")


def _topk(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    top1 = logits.argmax(dim=-1).eq(labels).float().mean()
    k = min(5, int(logits.shape[-1]))
    top5 = logits.topk(k, dim=-1).indices.eq(labels.unsqueeze(-1)).any(dim=-1).float().mean()
    return float(top1.detach().cpu().item()), float(top5.detach().cpu().item())


__all__ = [
    "BeamPrototypeBank",
    "make_soft_beam_labels",
    "prototype_alignment_loss",
    "supervised_contrastive_loss",
]
