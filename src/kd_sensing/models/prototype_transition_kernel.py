"""CSI-conditioned local transition kernels over the audited beam topology."""

from __future__ import annotations

import math

import torch
from torch import nn


TRANSITION_CONTEXT_MODES = ("temporal", "last", "no_delta", "static")


def _logit(probability: float) -> float:
    value = float(probability)
    if not 0.0 < value < 1.0:
        raise ValueError("identity initialization must lie in (0,1).")
    return math.log(value / (1.0 - value))


def transition_context(
    frame_features: torch.Tensor,
    gru_hidden: torch.Tensor,
    *,
    mode: str = "temporal",
) -> torch.Tensor:
    """Build the target-free propagation-change context from ordered CSI frames."""
    frames = torch.as_tensor(frame_features)
    hidden = torch.as_tensor(gru_hidden, device=frames.device, dtype=frames.dtype)
    method = str(mode).lower()
    if method not in TRANSITION_CONTEXT_MODES or method == "static":
        raise ValueError(f"transition_context requires temporal, last, or no_delta mode; got {mode!r}.")
    if frames.ndim != 3 or hidden.shape != (frames.shape[0], frames.shape[2]):
        raise ValueError("frame_features and gru_hidden must have shapes [B,T,D] and [B,D].")
    if method == "last":
        return frames[:, -1]
    if method == "no_delta":
        return hidden
    if frames.shape[1] < 2:
        raise ValueError("Temporal transition context requires at least two ordered CSI frames.")
    delta = frames[:, 1:] - frames[:, :-1]
    return torch.cat(
        (
            hidden,
            frames[:, -1] - frames[:, 0],
            delta.mean(dim=1),
            delta.std(dim=1, unbiased=False),
        ),
        dim=-1,
    )


class PrototypeTransitionKernel(nn.Module):
    """Predict a shared local displacement distribution, mixed with identity."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        radius: int = 3,
        context_mode: str = "temporal",
        identity_initial_mass: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.radius = int(radius)
        self.context_mode = str(context_mode).lower()
        if self.hidden_dim <= 0 or self.radius < 0 or self.context_mode not in TRANSITION_CONTEXT_MODES:
            raise ValueError("Invalid transition hidden_dim, radius, or context_mode.")
        output_dim = 2 * self.radius + 1
        if self.context_mode == "static":
            self.network: nn.Module | None = None
            self.static_logits = nn.Parameter(torch.zeros(output_dim))
            self.context_dim = 0
        else:
            self.static_logits = None
            self.context_dim = self.hidden_dim * (4 if self.context_mode == "temporal" else 1)
            self.network = nn.Sequential(
                nn.Linear(self.context_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, output_dim),
            )
        self.identity_raw = nn.Parameter(torch.tensor(_logit(float(identity_initial_mass)), dtype=torch.float32))
        self.register_buffer("offsets", torch.arange(-self.radius, self.radius + 1), persistent=True)

    def forward(self, frame_features: torch.Tensor, gru_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        frames = torch.as_tensor(frame_features)
        hidden = torch.as_tensor(gru_hidden, device=frames.device, dtype=frames.dtype)
        if frames.ndim != 3 or hidden.shape != (frames.shape[0], self.hidden_dim) or frames.shape[-1] != self.hidden_dim:
            raise ValueError(f"Transition inputs must have shapes [B,T,{self.hidden_dim}] and [B,{self.hidden_dim}].")
        batch = frames.shape[0]
        if self.context_mode == "static":
            assert self.static_logits is not None
            context = frames.new_zeros(batch, 0)
            logits = self.static_logits[None].expand(batch, -1)
        else:
            assert self.network is not None
            context = transition_context(frames, hidden, mode=self.context_mode)
            logits = self.network(context)
        q_delta = torch.softmax(logits.float(), dim=-1)
        gamma = torch.sigmoid(self.identity_raw.float())
        identity = torch.zeros_like(q_delta)
        identity[:, self.radius] = 1.0
        q_final = (1.0 - gamma) * identity + gamma * q_delta
        return {
            "transition_context": context,
            "transition_logits": logits.float(),
            "q_delta": q_delta,
            "q_final": q_final,
            "identity_mass": q_final[:, self.radius],
            "gamma_transition": gamma,
        }


__all__ = ["PrototypeTransitionKernel", "TRANSITION_CONTEXT_MODES", "transition_context"]
