"""Sparse-radio expert that can only decide through a shared prototype bank."""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank


def _inverse_softplus(value: float) -> float:
    if float(value) <= 0:
        raise ValueError("Positive initialization must be greater than zero.")
    return math.log(math.expm1(float(value)))


class PositiveTemperature(nn.Module):
    """A scalar temperature with an always-positive forward value."""

    def __init__(self, initial: float = 1.0, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        if float(initial) <= self.eps:
            raise ValueError("Temperature initialization must exceed eps.")
        self.raw = nn.Parameter(torch.tensor(_inverse_softplus(float(initial) - self.eps)))

    def forward(self) -> torch.Tensor:
        return F.softplus(self.raw) + self.eps


class RadioPrototypeExpert(nn.Module):
    """Map a radio latent to an embedding before querying the frozen M4 bank."""

    def __init__(
        self,
        *,
        radio_dim: int = 128,
        hidden_dim: int = 128,
        prototype_dim: int = 64,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.radio_dim = int(radio_dim)
        self.hidden_dim = int(hidden_dim)
        self.prototype_dim = int(prototype_dim)
        self.input_norm = nn.LayerNorm(self.radio_dim)
        self.hidden_projection = nn.Linear(self.radio_dim, self.hidden_dim)
        self.embedding_projection = nn.Linear(self.hidden_dim, self.prototype_dim)
        self.temperature = PositiveTemperature(float(temperature))

    def forward(self, c_radio: torch.Tensor, prototype_bank: BeamPrototypeBank) -> dict[str, torch.Tensor]:
        radio = torch.as_tensor(c_radio)
        if tuple(radio.shape[1:]) != (self.radio_dim,):
            raise ValueError(f"c_radio must have shape [B,{self.radio_dim}].")
        if prototype_bank.d_model != self.prototype_dim:
            raise ValueError("Radio embedding and shared prototype dimensions do not match.")
        # The audited prototype inverse is ill-conditioned enough that BF16
        # changes radio argmax decisions. Keep this small semantic query FP32.
        with torch.autocast(device_type=radio.device.type, enabled=False):
            radio_fp32 = radio.float()
            hidden = F.gelu(self.hidden_projection(self.input_norm(radio_fp32)))
            embedding = self.embedding_projection(hidden)
            original_logits = prototype_bank(embedding)
            temperature = self.temperature().float()
            logits = original_logits * (float(prototype_bank.temperature) / temperature)
        return {
            "z_radio": embedding,
            "radio_evidence": logits,
            "radio_probability": torch.softmax(logits, dim=-1),
            "radio_temperature": temperature,
        }

    @torch.no_grad()
    def initialize_from_teacher(
        self,
        prototype_bank: BeamPrototypeBank,
        teacher_state: Mapping[str, torch.Tensor],
    ) -> dict[str, float | int]:
        """Factor a training-only CSI head through the frozen prototype matrix."""
        required = {
            "0.weight",
            "0.bias",
            "1.weight",
            "1.bias",
            "3.weight",
            "3.bias",
        }
        if set(teacher_state) != required:
            raise ValueError(f"Teacher head state mismatch: expected={sorted(required)}, got={sorted(teacher_state)}.")
        if self.radio_dim != 128 or self.hidden_dim != 128 or self.prototype_dim != 64:
            raise ValueError("Teacher factorization currently requires the validated 128->128->64 dimensions.")
        prototypes = F.normalize(prototype_bank.prototypes.detach().double(), dim=-1)
        rank = int(torch.linalg.matrix_rank(prototypes).item())
        if rank != self.prototype_dim:
            raise ValueError(f"Shared prototype matrix is rank deficient: rank={rank}.")
        inverse = torch.linalg.inv(prototypes.t())
        teacher_weight = teacher_state["3.weight"].detach().double()
        teacher_bias = teacher_state["3.bias"].detach().double()
        factored_weight = inverse.t() @ teacher_weight
        factored_bias = teacher_bias @ inverse

        self.input_norm.weight.copy_(teacher_state["0.weight"].to(self.input_norm.weight))
        self.input_norm.bias.copy_(teacher_state["0.bias"].to(self.input_norm.bias))
        self.hidden_projection.weight.copy_(teacher_state["1.weight"].to(self.hidden_projection.weight))
        self.hidden_projection.bias.copy_(teacher_state["1.bias"].to(self.hidden_projection.bias))
        self.embedding_projection.weight.copy_(factored_weight.to(self.embedding_projection.weight))
        self.embedding_projection.bias.copy_(factored_bias.to(self.embedding_projection.bias))
        return {
            "prototype_rank": rank,
            "prototype_condition_number": float(torch.linalg.cond(prototypes).item()),
        }


__all__ = ["PositiveTemperature", "RadioPrototypeExpert"]
