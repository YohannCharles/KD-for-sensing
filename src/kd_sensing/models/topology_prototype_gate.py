"""Prototype gates with and without the audited beam topology constraint."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def prototype_gate_input(
    sensing_logits: torch.Tensor,
    radio_logits: torch.Tensor,
) -> torch.Tensor:
    sensing = torch.as_tensor(sensing_logits)
    radio = torch.as_tensor(radio_logits, device=sensing.device, dtype=sensing.dtype)
    if sensing.ndim != 2 or radio.shape != sensing.shape:
        raise ValueError("Prototype gate evidence must have matching [B,C] shapes.")
    p_s = torch.softmax(sensing, dim=-1)
    p_c = torch.softmax(radio, dim=-1)
    disagreement = (p_c - p_s).abs()
    local_disagreement = (disagreement + disagreement.roll(1, dims=-1) + disagreement.roll(-1, dims=-1)) / 3.0
    return torch.stack((sensing, radio, radio - sensing, p_s, p_c, local_disagreement), dim=1)


class TopologyPrototypeGate(nn.Module):
    """A shared local gate operating along an audited cyclic beam order."""

    def __init__(
        self,
        *,
        num_beams: int = 64,
        input_channels: int = 6,
        hidden_channels: int = 16,
        kernel_size: int = 3,
        labels_by_position: Sequence[int] | torch.Tensor | None = None,
        circular: bool = True,
        initial_probability: float = 0.9,
    ) -> None:
        super().__init__()
        self.num_beams = int(num_beams)
        self.input_channels = int(input_channels)
        self.kernel_size = int(kernel_size)
        self.circular = bool(circular)
        probability = float(initial_probability)
        if not 0.0 < probability < 1.0:
            raise ValueError("initial_probability must be in (0,1).")
        if self.kernel_size <= 0 or self.kernel_size % 2 != 1:
            raise ValueError("Topology gate kernel_size must be a positive odd integer.")
        labels = torch.arange(self.num_beams) if labels_by_position is None else torch.as_tensor(labels_by_position)
        labels = labels.to(dtype=torch.long).reshape(-1)
        if labels.numel() != self.num_beams or set(labels.tolist()) != set(range(self.num_beams)):
            raise ValueError("labels_by_position must be a beam-label bijection.")
        self.register_buffer("labels_by_position", labels, persistent=True)
        self.first = nn.Conv1d(self.input_channels, int(hidden_channels), self.kernel_size, padding=0)
        self.second = nn.Conv1d(int(hidden_channels), 1, self.kernel_size, padding=0)
        nn.init.zeros_(self.second.weight)
        nn.init.constant_(self.second.bias, math.log(probability / (1.0 - probability)))

    def _convolve(self, values: torch.Tensor, layer: nn.Conv1d) -> torch.Tensor:
        padding = self.kernel_size // 2
        mode = "circular" if self.circular else "constant"
        return layer(F.pad(values, (padding, padding), mode=mode))

    def forward(self, gate_input: torch.Tensor, rho: torch.Tensor) -> dict[str, torch.Tensor]:
        values = torch.as_tensor(gate_input)
        if values.ndim != 3 or tuple(values.shape[1:]) != (self.input_channels, self.num_beams):
            raise ValueError(f"gate_input must have shape [B,{self.input_channels},{self.num_beams}].")
        trust = torch.as_tensor(rho, device=values.device, dtype=values.dtype).reshape(-1)
        if trust.shape[0] != values.shape[0]:
            raise ValueError("rho must contain one value per sample.")
        labels = self.labels_by_position.to(values.device)
        ordered = values.index_select(-1, labels)
        ordered_logits = self._convolve(F.gelu(self._convolve(ordered, self.first)), self.second).squeeze(1)
        logits = torch.empty_like(ordered_logits)
        logits[:, labels] = ordered_logits
        gate = trust[:, None] * torch.sigmoid(logits)
        return {"prototype_gate": gate, "prototype_gate_logits": logits}


class IndependentPrototypeGate(nn.Module):
    """F3 control: each beam owns unrelated gate weights."""

    def __init__(self, *, num_beams: int = 64, input_channels: int = 6, initial_probability: float = 0.9) -> None:
        super().__init__()
        self.num_beams = int(num_beams)
        self.input_channels = int(input_channels)
        probability = float(initial_probability)
        if not 0.0 < probability < 1.0:
            raise ValueError("initial_probability must be in (0,1).")
        self.weight = nn.Parameter(torch.zeros(self.num_beams, self.input_channels))
        self.bias = nn.Parameter(torch.full((self.num_beams,), math.log(probability / (1.0 - probability))))

    def forward(self, gate_input: torch.Tensor, rho: torch.Tensor) -> dict[str, torch.Tensor]:
        values = torch.as_tensor(gate_input)
        if values.ndim != 3 or tuple(values.shape[1:]) != (self.input_channels, self.num_beams):
            raise ValueError(f"gate_input must have shape [B,{self.input_channels},{self.num_beams}].")
        logits = (values.transpose(1, 2) * self.weight[None]).sum(dim=-1) + self.bias
        trust = torch.as_tensor(rho, device=values.device, dtype=values.dtype).reshape(-1)
        gate = trust[:, None] * torch.sigmoid(logits)
        return {"prototype_gate": gate, "prototype_gate_logits": logits}


__all__ = ["IndependentPrototypeGate", "TopologyPrototypeGate", "prototype_gate_input"]
