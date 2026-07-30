"""Small residual adapter used only by missing-sensing paths."""

from __future__ import annotations

import torch
import torch.nn as nn


class MissingSensingPrototypeAdapter(nn.Module):
    """Adapt 64-D missing-path features while keeping Full exactly unchanged."""

    def __init__(self, embedding_dim: int = 64, bottleneck_dim: int = 16) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.down = nn.Linear(self.embedding_dim, int(bottleneck_dim))
        self.activation = nn.GELU()
        self.up = nn.Linear(int(bottleneck_dim), self.embedding_dim)
        self.normalization = nn.LayerNorm(self.embedding_dim, elementwise_affine=False)
        self.mix = nn.Parameter(torch.zeros((), dtype=torch.float32))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def adapter_residual(self, feature: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(feature)
        if value.shape[-1] != self.embedding_dim:
            raise ValueError(f"feature last dimension must be {self.embedding_dim}.")
        return self.up(self.activation(self.down(value)))

    def forward(
        self,
        feature: torch.Tensor,
        missing: bool | torch.Tensor,
        *,
        return_residual: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        value = torch.as_tensor(feature)
        residual = self.adapter_residual(value)
        normalized = self.normalization(value + residual)
        candidate = value + self.mix.to(value.dtype) * (normalized - value)

        missing_mask = torch.as_tensor(missing, device=value.device, dtype=torch.bool)
        if missing_mask.ndim == 0:
            output = candidate if bool(missing_mask) else value
        else:
            if tuple(missing_mask.shape) != tuple(value.shape[:-1]):
                raise ValueError("missing must be scalar or match feature batch dimensions.")
            output = torch.where(missing_mask[..., None], candidate, value)
        return (output, residual) if return_residual else output


__all__ = ["MissingSensingPrototypeAdapter"]
