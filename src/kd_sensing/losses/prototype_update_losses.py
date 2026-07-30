"""Staged losses for CSI-conditioned prototype state updates."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from kd_sensing.models.prototype_fusion_losses import topology_risk


def prototype_update_loss(
    output: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    topology_distance: torch.Tensor,
    *,
    weights: Mapping[str, float],
    low_quality_weight: torch.Tensor | None = None,
    teacher_probability: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    log_posterior = output["log_posterior"].float()
    target = torch.as_tensor(labels, device=log_posterior.device, dtype=torch.long).reshape(-1)
    active = torch.as_tensor(output.get("update_active", torch.ones_like(target)), device=target.device, dtype=torch.bool)
    weight = (
        torch.ones_like(target, dtype=log_posterior.dtype)
        if sample_weight is None
        else torch.as_tensor(sample_weight, device=target.device, dtype=log_posterior.dtype).reshape(-1)
    )
    weight = weight * active.to(weight)
    denominator = weight.sum().clamp_min(1.0)
    task_items = F.nll_loss(log_posterior, target, reduction="none")
    task = (task_items * weight).sum() / denominator
    topology = (topology_risk(log_posterior, target, topology_distance) * weight).sum() / denominator

    q_final = output["q_final"].float()
    q_delta = output["q_delta"].float()
    radius = q_final.shape[1] // 2
    quality = (
        torch.zeros_like(weight)
        if low_quality_weight is None
        else torch.as_tensor(low_quality_weight, device=weight.device, dtype=weight.dtype).reshape(-1).clamp(0.0, 1.0)
    )
    identity_items = -q_final[:, radius].clamp_min(1e-8).log()
    identity = (identity_items * quality * weight).sum() / (quality.mul(weight).sum().clamp_min(1.0))
    offsets = torch.arange(-radius, radius + 1, device=q_delta.device, dtype=q_delta.dtype).abs()
    local = ((q_delta * offsets[None]).sum(dim=-1) * weight).sum() / denominator

    zero = task * 0.0
    likelihood_kd = zero
    if teacher_probability is not None:
        teacher = torch.as_tensor(teacher_probability, device=log_posterior.device).float().detach()
        items = F.kl_div(output["p_c"].float().clamp_min(1e-8).log(), teacher, reduction="none").sum(dim=-1)
        likelihood_kd = (items * weight).sum() / denominator

    p_s = output["p_s"].float().detach()
    p_c = output["p_c"].float().detach()
    preserve_mask = p_s.argmax(dim=-1).eq(target) & p_c.argmax(dim=-1).ne(target) & active
    preserve_items = F.kl_div(output["p_final"].float().clamp_min(1e-8).log(), p_s, reduction="none").sum(dim=-1)
    preserve = preserve_items[preserve_mask].mean() if bool(preserve_mask.any()) else zero
    total = (
        task
        + float(weights.get("topology", 0.0)) * topology
        + float(weights.get("transition_identity", 0.0)) * identity
        + float(weights.get("transition_local", 0.0)) * local
        + float(weights.get("likelihood_kd", 0.0)) * likelihood_kd
        + float(weights.get("preserve", 0.0)) * preserve
    )
    return {
        "total": total,
        "task": task,
        "topology": topology,
        "transition_identity": identity,
        "transition_local": local,
        "likelihood_kd": likelihood_kd,
        "preserve": preserve,
        "preserve_fraction": preserve_mask.float().mean(),
    }


__all__ = ["prototype_update_loss"]
