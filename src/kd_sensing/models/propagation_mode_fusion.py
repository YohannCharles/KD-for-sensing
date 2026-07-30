"""Inference rules for shared propagation subprototypes."""

from __future__ import annotations

import math

import torch


def aggregate_subprototype_evidence(mode_scores: torch.Tensor, *, tau_sub: float = 1.0) -> torch.Tensor:
    scores = torch.as_tensor(mode_scores).float()
    if scores.ndim < 2 or scores.shape[-1] <= 0 or float(tau_sub) <= 0.0:
        raise ValueError("mode_scores must end in a non-empty mode dimension and tau_sub must be positive.")
    return float(tau_sub) * (
        torch.logsumexp(scores / float(tau_sub), dim=-1) - math.log(scores.shape[-1])
    )


def fixed_beam_evidence_fusion(
    sensing_evidence: torch.Tensor,
    csi_evidence: torch.Tensor,
    *,
    csi_weight: float = 0.5,
    csi_available: bool | torch.Tensor = True,
) -> torch.Tensor:
    sensing = torch.as_tensor(sensing_evidence).float()
    csi = torch.as_tensor(csi_evidence, device=sensing.device).float()
    if sensing.shape != csi.shape or not 0.0 <= float(csi_weight) <= 1.0:
        raise ValueError("evidence shapes must match and csi_weight must be in [0,1].")
    fused = (1.0 - float(csi_weight)) * sensing + float(csi_weight) * csi
    available = torch.as_tensor(csi_available, device=sensing.device, dtype=torch.bool)
    if available.ndim == 0:
        return fused if bool(available) else sensing
    if tuple(available.shape) != tuple(sensing.shape[:-1]):
        raise ValueError("csi_available must be scalar or match evidence batch dimensions.")
    return torch.where(available[..., None], fused, sensing)


def mode_consistent_fusion(
    sensing_mode_scores: torch.Tensor,
    csi_mode_scores: torch.Tensor,
    *,
    csi_weight: float = 0.5,
    tau_sub: float = 1.0,
    csi_available: bool | torch.Tensor = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse matched mode scores first, then aggregate modes into beams."""
    sensing = torch.as_tensor(sensing_mode_scores).float()
    csi = torch.as_tensor(csi_mode_scores, device=sensing.device).float()
    if sensing.ndim != 3 or sensing.shape != csi.shape or not 0.0 <= float(csi_weight) <= 1.0:
        raise ValueError("mode score shapes must match and csi_weight must be in [0,1].")
    fused = (1.0 - float(csi_weight)) * sensing + float(csi_weight) * csi
    available = torch.as_tensor(csi_available, device=sensing.device, dtype=torch.bool)
    if available.ndim == 0:
        selected = fused if bool(available) else sensing
    else:
        if tuple(available.shape) != tuple(sensing.shape[:-2]):
            raise ValueError("csi_available must be scalar or match mode-score batch dimensions.")
        selected = torch.where(available[..., None, None], fused, sensing)
    return aggregate_subprototype_evidence(selected, tau_sub=tau_sub), selected


def full_path_bypass(
    original_full_output: torch.Tensor,
    missing_path_output: torch.Tensor,
    full_path: bool | torch.Tensor,
) -> torch.Tensor:
    """Select original M4 output exactly for Full samples."""
    original = torch.as_tensor(original_full_output)
    candidate = torch.as_tensor(missing_path_output, device=original.device, dtype=original.dtype)
    if original.shape != candidate.shape:
        raise ValueError("original and missing-path outputs must have the same shape.")
    selector = torch.as_tensor(full_path, device=original.device, dtype=torch.bool)
    if selector.ndim == 0:
        return original if bool(selector) else candidate
    if tuple(selector.shape) != tuple(original.shape[:-1]):
        raise ValueError("full_path must be scalar or match output batch dimensions.")
    return torch.where(selector[..., None], original, candidate)


__all__ = [
    "aggregate_subprototype_evidence",
    "fixed_beam_evidence_fusion",
    "full_path_bypass",
    "mode_consistent_fusion",
]
