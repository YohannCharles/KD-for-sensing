"""Training-only radio guidance for within-beam propagation modes."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def true_beam_mode_scores(mode_scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    scores = torch.as_tensor(mode_scores).float()
    labels = torch.as_tensor(targets, device=scores.device, dtype=torch.long)
    if scores.ndim != 3 or labels.ndim != 1 or scores.shape[0] != labels.shape[0]:
        raise ValueError("mode_scores and targets must have shapes [B,beam,mode] and [B].")
    row = torch.arange(scores.shape[0], device=scores.device)
    return scores[row, labels]


def propagation_mode_distribution(
    mode_scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive.")
    return torch.softmax(true_beam_mode_scores(mode_scores, targets) / float(temperature), dim=-1)


def qualified_teacher_weights(
    sensing_evidence: torch.Tensor,
    csi_evidence: torch.Tensor,
    targets: torch.Tensor,
    *,
    tau_adv: float = 0.5,
    conf_ref: float = 0.3,
    training: bool,
) -> dict[str, torch.Tensor]:
    """Build stop-gradient weights; labels are forbidden outside training."""
    if not training:
        raise RuntimeError("qualified teacher weights are training-only and require targets.")
    if float(tau_adv) <= 0.0 or float(conf_ref) <= 0.0:
        raise ValueError("tau_adv and conf_ref must be positive.")
    sensing = torch.as_tensor(sensing_evidence).float()
    csi = torch.as_tensor(csi_evidence).float()
    labels = torch.as_tensor(targets, device=sensing.device, dtype=torch.long)
    if sensing.shape != csi.shape or sensing.ndim != 2 or labels.shape != sensing.shape[:1]:
        raise ValueError("evidence and targets must have shapes [B,beam] and [B].")

    loss_s = F.cross_entropy(sensing, labels, reduction="none")
    loss_c = F.cross_entropy(csi, labels, reduction="none")
    top_k = min(3, csi.shape[-1])
    qualified = csi.topk(top_k, dim=-1).indices.eq(labels[:, None]).any(dim=-1)
    advantage = torch.sigmoid((loss_s - loss_c) / float(tau_adv))
    probability_c = torch.softmax(csi, dim=-1)
    confidence = (
        probability_c.gather(1, labels[:, None]).squeeze(1) / float(conf_ref)
    ).clamp(0.0, 1.0)
    weight = qualified.to(advantage.dtype) * advantage * confidence
    return {
        "weight": weight.detach(),
        "qualified_rank": qualified.detach(),
        "advantage": advantage.detach(),
        "confidence": confidence.detach(),
        "loss_s": loss_s.detach(),
        "loss_c": loss_c.detach(),
    }


def radio_prototype_distillation_loss(
    sensing_mode_probability: torch.Tensor,
    csi_mode_probability: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    q_s = torch.as_tensor(sensing_mode_probability).float().clamp_min(float(eps))
    q_c = torch.as_tensor(csi_mode_probability).float().detach().clamp_min(float(eps))
    if q_s.shape != q_c.shape or q_s.ndim != 2:
        raise ValueError("mode probabilities must have matching shape [B,mode].")
    item = (q_c * (q_c.log() - q_s.log())).sum(dim=-1)
    if weights is None:
        return item.mean()
    weight = torch.as_tensor(weights, device=item.device, dtype=torch.float32).detach()
    if weight.shape != item.shape:
        raise ValueError("weights must have shape [B].")
    return (weight * item).sum() / (weight.sum() + float(eps))


__all__ = [
    "propagation_mode_distribution",
    "qualified_teacher_weights",
    "radio_prototype_distillation_loss",
    "true_beam_mode_scores",
]
