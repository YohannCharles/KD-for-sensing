"""Train-prior-corrected likelihood ratios over the shared beam prototypes."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank
from kd_sensing.models.radio_prototype_expert import RadioPrototypeExpert


def estimate_train_beam_prior(
    labels: torch.Tensor,
    *,
    split_role: str,
    num_beams: int = 64,
) -> dict[str, Any]:
    """Estimate a class prior while rejecting validation/test label sources."""
    if str(split_role).strip().lower() != "train":
        raise ValueError("Beam prior estimation is restricted to the train split.")
    target = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
    if target.numel() == 0 or bool(((target < 0) | (target >= int(num_beams))).any()):
        raise ValueError(f"Train labels must be non-empty and lie in [0,{int(num_beams) - 1}].")
    counts = torch.bincount(target, minlength=int(num_beams))
    prior = counts.float() / float(target.numel())
    return {
        "split_role": "train",
        "sample_count": int(target.numel()),
        "counts": counts,
        "prior": prior,
    }


class PrototypeLikelihoodHead(nn.Module):
    """Convert the calibrated radio posterior into a discriminative likelihood ratio."""

    def __init__(
        self,
        radio_expert: RadioPrototypeExpert,
        train_prior: torch.Tensor,
        *,
        eta_prior: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        prior = torch.as_tensor(train_prior, dtype=torch.float32).reshape(-1)
        if prior.numel() != radio_expert.prototype_dim or not bool(torch.isfinite(prior).all()) or bool((prior < 0).any()):
            raise ValueError("train_prior must be one finite non-negative value per shared prototype.")
        if float(prior.sum()) <= 0 or float(eps) <= 0:
            raise ValueError("train_prior must have positive mass and eps must be positive.")
        self.radio_expert = radio_expert
        self.eta_prior = float(eta_prior)
        self.eps = float(eps)
        self.register_buffer("train_prior", prior / prior.sum(), persistent=True)

    def forward(self, c_radio: torch.Tensor, prototype_bank: BeamPrototypeBank) -> dict[str, torch.Tensor]:
        output = self.radio_expert(c_radio, prototype_bank)
        device_type = output["radio_probability"].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            posterior = output["radio_probability"].float()
            log_ratio = posterior.clamp_min(self.eps).log()
            log_ratio = log_ratio - self.eta_prior * self.train_prior.float().clamp_min(self.eps).log()[None]
        return {
            **output,
            "radio_probability": posterior,
            "log_likelihood_ratio": log_ratio,
            "eta_prior": log_ratio.new_tensor(self.eta_prior),
        }


__all__ = ["PrototypeLikelihoodHead", "estimate_train_beam_prior"]
