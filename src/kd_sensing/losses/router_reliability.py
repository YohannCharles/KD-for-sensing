"""Continuous communication utility losses for dynamic reliability routing."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from kd_sensing.losses.beam_prototype_alignment import make_soft_beam_labels


UTILITY_SOURCES = frozenset(("label_topology", "beam_power"))


def expected_router_utility(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    source: str,
    beam_powers: torch.Tensor | None = None,
    beam_temperature: float = 1.0,
    beam_label_sigma: float = 1.0,
    circular: bool = True,
    topology_id: str | None = None,
    topology_permutation: list[int] | tuple[int, ...] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Return expected normalized utility while preserving gradients through logits."""

    if logits.ndim < 2:
        raise ValueError("Router utility logits must have shape [B,...,C].")
    if float(beam_temperature) <= 0.0:
        raise ValueError("Router utility beam_temperature must be positive.")
    source = str(source).strip().lower()
    if source not in UTILITY_SOURCES:
        raise ValueError(f"Router utility source must be one of {sorted(UTILITY_SOURCES)}.")
    batch_size, num_classes = int(logits.shape[0]), int(logits.shape[-1])
    hard = labels.to(device=logits.device, dtype=torch.long)
    if hard.ndim > 1:
        hard = hard[:, 0]
    hard = hard.reshape(-1)
    if hard.shape != (batch_size,):
        raise ValueError(f"Router utility labels must have shape [{batch_size}] or [{batch_size},H].")

    if source == "label_topology":
        reward = make_soft_beam_labels(
            hard,
            num_classes,
            float(beam_label_sigma),
            circular=bool(circular),
            topology_id=topology_id,
            topology_permutation=topology_permutation,
        ).to(device=logits.device, dtype=logits.dtype)
        reward = reward / reward.amax(dim=-1, keepdim=True).clamp_min(torch.finfo(logits.dtype).tiny)
        utility_logits = logits
    else:
        if not torch.is_tensor(beam_powers):
            raise ValueError("beam_power Router utility requires future_beam_power.")
        # AMP logits may be float16, while MMW linear beam powers are typically
        # around 1e-12--1e-8. Normalize and evaluate beam-power utility in
        # float32 so the physical target cannot underflow before division.
        utility_logits = logits.float()
        reward = beam_powers.to(device=logits.device, dtype=torch.float32).detach()
        if reward.shape != (batch_size, num_classes):
            raise ValueError(
                "future_beam_power must have shape "
                f"{(batch_size, num_classes)}, got {tuple(reward.shape)}."
            )
        if not bool(torch.isfinite(reward).all().item()) or bool((reward < 0).any().item()):
            raise ValueError("future_beam_power must contain finite non-negative values.")
        reward = reward / reward.amax(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)

    view_shape = (batch_size,) + (1,) * (logits.ndim - 2) + (num_classes,)
    probabilities = torch.softmax(utility_logits / float(beam_temperature), dim=-1)
    return (probabilities * reward.reshape(view_shape)).sum(dim=-1)


def pairwise_utility_ranking_loss(
    scores: torch.Tensor,
    utility: torch.Tensor,
    mask: torch.Tensor,
    *,
    gap_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank the final axis by detached utility; near ties do not contribute."""

    if scores.shape != utility.shape or scores.shape != mask.shape or scores.ndim < 2:
        raise ValueError("Router ranking scores, utility, and mask must have matching [...,K] shapes.")
    utility = utility.to(device=scores.device, dtype=scores.dtype).detach()
    mask = mask.to(device=scores.device, dtype=torch.bool)
    score_delta = scores.unsqueeze(-1) - scores.unsqueeze(-2)
    utility_delta = utility.unsqueeze(-1) - utility.unsqueeze(-2)
    pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    triangle = torch.triu(torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1)
    active = pair_mask & triangle & utility_delta.abs().gt(float(gap_epsilon))
    if not bool(active.any().item()):
        return scores.sum() * 0.0, active.to(dtype=scores.dtype).mean()
    penalties = F.softplus(-utility_delta.sign() * score_delta) * utility_delta.abs()
    return penalties[active].mean(), active.to(dtype=scores.dtype).mean()


def paired_router_reliability_loss(
    control: Mapping[str, torch.Tensor],
    joint: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    source: str,
    beam_powers: torch.Tensor | None,
    beam_temperature: float,
    beam_label_sigma: float,
    circular: bool,
    topology_id: str | None,
    topology_permutation: list[int] | tuple[int, ...] | torch.Tensor | None,
    quality_weight: float,
    fused_utility_weight: float,
    monotonic_weight: float,
    frame_rank_weight: float,
    residual_anchor_weight: float,
    quality_drop_epsilon: float,
    monotonic_margin_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train a Router from same-availability control/joint counterfactuals."""

    required = {"router_gate_logits", "router_gate_weights", "unimodal_logits", "fused_logits", "available"}
    for name, payload in (("control", control), ("joint", joint)):
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"{name} Router pair is missing fields {missing}.")
    control_available = control["available"].to(dtype=torch.bool)
    joint_available = joint["available"].to(dtype=torch.bool)
    if control_available.shape != joint_available.shape or not torch.equal(control_available, joint_available):
        raise ValueError("Router counterfactual views must have identical availability.")
    control_logits = control["router_gate_logits"]
    joint_logits = joint["router_gate_logits"]
    if control_logits.shape != control_available.shape or joint_logits.shape != control_available.shape:
        raise ValueError("Router gate logits must match [B,M] availability.")

    utility_kwargs: dict[str, Any] = {
        "source": source,
        "beam_powers": beam_powers,
        "beam_temperature": beam_temperature,
        "beam_label_sigma": beam_label_sigma,
        "circular": circular,
        "topology_id": topology_id,
        "topology_permutation": topology_permutation,
    }
    control_utility = expected_router_utility(control["unimodal_logits"].detach(), labels, **utility_kwargs)
    joint_utility = expected_router_utility(joint["unimodal_logits"].detach(), labels, **utility_kwargs)
    available_float = joint_available.to(dtype=joint_logits.dtype)
    quality_target = joint_utility.detach().clamp(0.0, 1.0)
    count = available_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    predicted_quality = joint_logits - (joint_logits * available_float).sum(dim=1, keepdim=True) / count
    centered_target = quality_target - (quality_target * available_float).sum(dim=1, keepdim=True) / count
    quality_per_cell = F.smooth_l1_loss(predicted_quality, centered_target, reduction="none")
    per_modality = (quality_per_cell * available_float).sum(dim=0) / available_float.sum(dim=0).clamp_min(1.0)
    quality_loss = per_modality.mean()

    fused_utility = expected_router_utility(joint["fused_logits"], labels, **utility_kwargs)
    fused_utility_loss = 1.0 - fused_utility.mean()

    utility_drop = (control_utility - joint_utility).detach()
    active = joint_available & utility_drop.abs().gt(float(quality_drop_epsilon))
    score_delta = joint_logits - control_logits.detach()
    margin = float(monotonic_margin_scale) * utility_drop.abs()
    monotonic_penalty = F.relu(utility_drop.sign() * score_delta + margin)
    monotonic_loss = monotonic_penalty[active].mean() if bool(active.any().item()) else score_delta.sum() * 0.0

    frame_rank_loss = joint_logits.sum() * 0.0
    frame_rank_active = joint_logits.new_zeros(())
    if float(frame_rank_weight) != 0.0 and "frame_health_logits" in joint:
        frame_logits = joint.get("frame_unimodal_logits")
        frame_mask = joint.get("cell_mask")
        if not torch.is_tensor(frame_logits) or not torch.is_tensor(frame_mask):
            raise ValueError("Frame reliability supervision requires frame_unimodal_logits and cell_mask.")
        frame_utility = expected_router_utility(frame_logits.detach(), labels, **utility_kwargs)
        frame_rank_loss, frame_rank_active = pairwise_utility_ranking_loss(
            joint["frame_health_logits"].transpose(1, 2),
            frame_utility.transpose(1, 2),
            frame_mask.to(dtype=torch.bool).transpose(1, 2),
            gap_epsilon=float(quality_drop_epsilon),
        )

    residual_anchor = joint_logits.sum() * 0.0
    residual = joint.get("router_residual_logits")
    if torch.is_tensor(residual):
        residual_anchor = (residual.square() * available_float).sum() / available_float.sum().clamp_min(1.0)

    weighted_quality = float(quality_weight) * quality_loss
    weighted_fused = float(fused_utility_weight) * fused_utility_loss
    weighted_monotonic = float(monotonic_weight) * monotonic_loss
    weighted_frame = float(frame_rank_weight) * frame_rank_loss
    weighted_anchor = float(residual_anchor_weight) * residual_anchor
    total = weighted_quality + weighted_fused + weighted_monotonic + weighted_frame + weighted_anchor
    diagnostics = {
        "loss/router_reliability_quality": float(quality_loss.detach().cpu().item()),
        "loss/router_reliability_fused_utility": float(fused_utility_loss.detach().cpu().item()),
        "loss/router_reliability_monotonic": float(monotonic_loss.detach().cpu().item()),
        "loss/router_reliability_frame_rank": float(frame_rank_loss.detach().cpu().item()),
        "loss/router_reliability_residual_anchor": float(residual_anchor.detach().cpu().item()),
        "loss/router_reliability_total": float(total.detach().cpu().item()),
        "router_reliability_joint_utility": float(joint_utility.mean().cpu().item()),
        "router_reliability_control_utility": float(control_utility.mean().cpu().item()),
        "router_reliability_fused_utility": float(fused_utility.detach().mean().cpu().item()),
        "router_reliability_active_ratio": float(active.to(dtype=torch.float32).mean().cpu().item()),
        "router_reliability_violation_ratio": float(
            ((utility_drop.sign() * score_delta.detach()) > 0).logical_and(active).to(dtype=torch.float32).mean().cpu().item()
        ),
        "router_reliability_frame_rank_active_ratio": float(frame_rank_active.detach().cpu().item()),
    }
    return total, diagnostics


__all__ = [
    "UTILITY_SOURCES",
    "expected_router_utility",
    "paired_router_reliability_loss",
    "pairwise_utility_ranking_loss",
]
