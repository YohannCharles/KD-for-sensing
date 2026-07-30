"""Propagation-aware subprototypes constrained around a frozen beam bank."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def reproducible_random_residuals(
    base_prototypes: torch.Tensor,
    *,
    num_subprototypes: int = 2,
    radius: float = 0.1,
    seed: int = 1,
) -> torch.Tensor:
    """Create tangent, fixed-radius residuals for the A5 control."""
    base = F.normalize(torch.as_tensor(base_prototypes, dtype=torch.float32), dim=-1)
    if base.ndim != 2 or int(num_subprototypes) <= 0 or float(radius) < 0.0:
        raise ValueError("base_prototypes, num_subprototypes, and radius are invalid.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(
        base.shape[0], int(num_subprototypes), base.shape[1], generator=generator, dtype=torch.float32
    )
    tangent = noise - (noise * base[:, None, :]).sum(dim=-1, keepdim=True) * base[:, None, :]
    return F.normalize(tangent, dim=-1) * float(radius)


class PropagationAwareSubPrototypeBank(nn.Module):
    """Score embeddings against nearby propagation modes for each beam.

    ``base_prototypes`` is stored as a buffer rather than a parameter. Residuals
    use the requested element-wise tanh bound plus an L2 projection, so their
    actual radius cannot exceed ``epsilon`` in 64 dimensions.
    """

    def __init__(
        self,
        base_prototypes: torch.Tensor,
        *,
        num_subprototypes: int = 2,
        epsilon: float = 0.15,
        tau_sub: float = 1.0,
    ) -> None:
        super().__init__()
        base = torch.as_tensor(base_prototypes, dtype=torch.float32)
        if base.ndim != 2:
            raise ValueError("base_prototypes must have shape [num_beams, embedding_dim].")
        if int(num_subprototypes) <= 0:
            raise ValueError("num_subprototypes must be positive.")
        if float(epsilon) <= 0.0 or float(tau_sub) <= 0.0:
            raise ValueError("epsilon and tau_sub must be positive.")

        self.num_beams, self.embedding_dim = map(int, base.shape)
        self.num_subprototypes = int(num_subprototypes)
        self.epsilon = float(epsilon)
        self.tau_sub = float(tau_sub)
        self.register_buffer("base_prototypes", base.detach().clone(), persistent=True)
        self.raw_delta = nn.Parameter(
            torch.zeros(self.num_beams, self.num_subprototypes, self.embedding_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "trainable_beam_mask",
            torch.ones(self.num_beams, dtype=torch.bool),
            persistent=True,
        )
        self.raw_delta.register_hook(
            lambda gradient: gradient * self.trainable_beam_mask[:, None, None].to(gradient)
        )

    def residuals(self) -> torch.Tensor:
        """Return FP32 residuals with a hard L2 radius bound."""
        residual = self.epsilon * torch.tanh(self.raw_delta.float())
        norm = residual.norm(dim=-1, keepdim=True)
        projection = (self.epsilon / norm.clamp_min(1e-12)).clamp(max=1.0)
        return residual * projection

    def subprototypes(self) -> torch.Tensor:
        # The published M4 bank is queried by cosine similarity, so its
        # semantic prototype is unit-normalized before applying epsilon.
        base = F.normalize(self.base_prototypes.float(), dim=-1)[:, None, :]
        return F.normalize(base + self.residuals(), dim=-1)

    @torch.no_grad()
    def initialize_residuals_(self, residuals: torch.Tensor) -> None:
        """Initialize trainable residuals while preserving the hard radius."""
        value = torch.as_tensor(residuals, dtype=torch.float32, device=self.raw_delta.device)
        if tuple(value.shape) != tuple(self.raw_delta.shape):
            raise ValueError(f"residuals must have shape {tuple(self.raw_delta.shape)}.")
        norm = value.norm(dim=-1, keepdim=True)
        value = value * ((self.epsilon * (1.0 - 1e-6)) / norm.clamp_min(1e-12)).clamp(max=1.0)
        scaled = (value / self.epsilon).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
        self.raw_delta.copy_(torch.atanh(scaled))
        self.enforce_trainable_beam_mask_()

    @torch.no_grad()
    def set_trainable_beam_mask_(self, mask: torch.Tensor) -> None:
        value = torch.as_tensor(mask, device=self.raw_delta.device, dtype=torch.bool)
        if tuple(value.shape) != (self.num_beams,):
            raise ValueError(f"trainable beam mask must have shape [{self.num_beams}].")
        self.trainable_beam_mask.copy_(value)
        self.enforce_trainable_beam_mask_()

    @torch.no_grad()
    def enforce_trainable_beam_mask_(self) -> None:
        self.raw_delta.mul_(self.trainable_beam_mask[:, None, None])

    def score(self, embedding: torch.Tensor, *, scale: float | torch.Tensor = 1.0) -> torch.Tensor:
        """Return FP32 cosine scores with shape ``[..., beam, mode]``."""
        feature = torch.as_tensor(embedding).float()
        if feature.shape[-1] != self.embedding_dim:
            raise ValueError(f"embedding last dimension must be {self.embedding_dim}.")
        feature = F.normalize(feature, dim=-1)
        prototype = self.subprototypes()
        scores = torch.einsum("...d,bkd->...bk", feature, prototype)
        return scores * torch.as_tensor(scale, device=scores.device, dtype=torch.float32)

    def aggregate(self, mode_scores: torch.Tensor, *, tau_sub: float | None = None) -> torch.Tensor:
        """Aggregate modes with LogMeanExp, avoiding a fixed K-dependent bias."""
        scores = torch.as_tensor(mode_scores).float()
        if scores.shape[-1] != self.num_subprototypes:
            raise ValueError(f"mode_scores last dimension must be {self.num_subprototypes}.")
        temperature = self.tau_sub if tau_sub is None else float(tau_sub)
        if temperature <= 0.0:
            raise ValueError("tau_sub must be positive.")
        return temperature * (
            torch.logsumexp(scores / temperature, dim=-1) - math.log(self.num_subprototypes)
        )

    def beam_evidence(
        self,
        embedding: torch.Tensor,
        *,
        scale: float | torch.Tensor = 1.0,
        tau_sub: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(embedding, scale=scale)
        return self.aggregate(scores, tau_sub=tau_sub), scores

    def forward(
        self,
        embedding: torch.Tensor,
        *,
        scale: float | torch.Tensor = 1.0,
        tau_sub: float | None = None,
    ) -> torch.Tensor:
        return self.beam_evidence(embedding, scale=scale, tau_sub=tau_sub)[0]


__all__ = ["PropagationAwareSubPrototypeBank", "reproducible_random_residuals"]
