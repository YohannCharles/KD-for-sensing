"""Numerically stable topology transition and prototype posterior update."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _topology_labels(count: int, labels_by_position: Sequence[int] | torch.Tensor | None, device: torch.device) -> torch.Tensor:
    labels = torch.arange(count, device=device) if labels_by_position is None else torch.as_tensor(labels_by_position, device=device)
    labels = labels.to(dtype=torch.long).reshape(-1)
    if labels.numel() != count or set(labels.tolist()) != set(range(count)):
        raise ValueError("labels_by_position must be a beam-label bijection.")
    return labels


def _linear_shift(probability: torch.Tensor, offset: int) -> torch.Tensor:
    shifted = torch.zeros_like(probability)
    if offset > 0:
        shifted[:, offset:] = probability[:, :-offset]
    elif offset < 0:
        shifted[:, :offset] = probability[:, -offset:]
    else:
        shifted = probability
    return shifted


def topology_transition(
    sensing_prior: torch.Tensor,
    transition_kernel: torch.Tensor,
    *,
    labels_by_position: Sequence[int] | torch.Tensor | None = None,
    circular: bool,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply local displacements along label topology, not arbitrary class coordinates."""
    prior = torch.as_tensor(sensing_prior).float()
    kernel = torch.as_tensor(transition_kernel, device=prior.device).float()
    if prior.ndim != 2 or kernel.ndim != 2 or kernel.shape[0] != prior.shape[0] or kernel.shape[1] % 2 != 1:
        raise ValueError("sensing_prior and transition_kernel must be [B,C] and [B,2K+1].")
    if not bool(torch.isfinite(prior).all()) or not bool(torch.isfinite(kernel).all()) or bool((prior < 0).any()) or bool((kernel < 0).any()):
        raise ValueError("Prior and transition kernel must be finite and non-negative.")
    original_prior = prior
    prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    kernel = kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    labels = _topology_labels(prior.shape[1], labels_by_position, prior.device)
    ordered = prior.index_select(-1, labels)
    radius = kernel.shape[1] // 2
    transitioned = torch.zeros_like(ordered)
    for index, offset in enumerate(range(-radius, radius + 1)):
        shifted = torch.roll(ordered, shifts=offset, dims=-1) if circular else _linear_shift(ordered, offset)
        transitioned = transitioned + kernel[:, index : index + 1] * shifted
    if not circular:
        transitioned = transitioned / transitioned.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    result = torch.zeros_like(transitioned).scatter(1, labels[None].expand(prior.shape[0], -1), transitioned)
    identity = kernel[:, radius].eq(1.0) & kernel.sum(dim=-1).eq(1.0)
    return torch.where(identity[:, None], original_prior, result)


class PrototypePosteriorUpdate(nn.Module):
    """Move a sensing prior, then apply a radio likelihood ratio in FP32 log space."""

    def __init__(
        self,
        *,
        labels_by_position: Sequence[int] | torch.Tensor,
        circular: bool,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if float(eps) <= 0:
            raise ValueError("eps must be positive.")
        labels = _topology_labels(len(labels_by_position), labels_by_position, torch.device("cpu"))
        self.circular = bool(circular)
        self.eps = float(eps)
        self.register_buffer("labels_by_position", labels, persistent=True)

    def forward(
        self,
        sensing_prior: torch.Tensor,
        transition_kernel: torch.Tensor,
        log_likelihood_ratio: torch.Tensor,
        *,
        beta: float | torch.Tensor,
        csi_available: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        device_type = sensing_prior.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            prior = torch.as_tensor(sensing_prior).float()
            log_ratio = torch.as_tensor(log_likelihood_ratio, device=prior.device).float()
            if log_ratio.shape != prior.shape:
                raise ValueError("log_likelihood_ratio must match sensing_prior [B,C].")
            available = (
                torch.ones(prior.shape[0], device=prior.device, dtype=torch.bool)
                if csi_available is None
                else torch.as_tensor(csi_available, device=prior.device, dtype=torch.bool).reshape(-1)
            )
            if available.shape[0] != prior.shape[0]:
                raise ValueError("csi_available must contain one flag per sample.")
            predicted = topology_transition(
                prior,
                transition_kernel,
                labels_by_position=self.labels_by_position,
                circular=self.circular,
                eps=self.eps,
            )
            predicted = torch.where(available[:, None], predicted, prior)
            effective_ratio = torch.where(available[:, None], log_ratio, torch.zeros_like(log_ratio))
            scale = torch.as_tensor(beta, device=prior.device, dtype=torch.float32).clamp_min(0.0)
            unnormalized = predicted.clamp_min(self.eps).log() + scale * effective_ratio
            log_posterior = torch.log_softmax(unnormalized, dim=-1)
            posterior = log_posterior.exp()
            posterior = torch.where(available[:, None], posterior, prior)
            log_posterior = torch.where(available[:, None], log_posterior, prior.clamp_min(self.eps).log())
        return {
            "p_pred": predicted,
            "log_posterior": log_posterior,
            "p_final": posterior,
        }


__all__ = ["PrototypePosteriorUpdate", "topology_transition"]
