"""Masked context encoder over physically available sensing modalities."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class AvailableModalityContext(nn.Module):
    """Encode four sensing slots without allowing missing tokens into the pool."""

    def __init__(
        self,
        *,
        input_dim: int = 64,
        hidden_dim: int = 128,
        num_modalities: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_modalities = int(num_modalities)
        if self.hidden_dim % int(num_heads):
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.modality_embedding = nn.Parameter(torch.randn(self.num_modalities, self.hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=self.hidden_dim * 2,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.pool_query = nn.Parameter(torch.randn(self.hidden_dim) / math.sqrt(self.hidden_dim))
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, tokens: torch.Tensor, availability: torch.Tensor) -> dict[str, torch.Tensor]:
        values = torch.as_tensor(tokens)
        mask = torch.as_tensor(availability, device=values.device, dtype=torch.bool)
        expected = (values.shape[0], self.num_modalities, self.input_dim)
        if values.ndim != 3 or tuple(values.shape) != expected:
            raise ValueError(f"tokens must have shape [B,{self.num_modalities},{self.input_dim}].")
        if tuple(mask.shape) != tuple(values.shape[:2]) or not bool(mask.any(dim=1).all()):
            raise ValueError("availability must be non-empty [B,num_modalities].")

        projected = self.input_projection(values) + self.modality_embedding[None]
        encoded = self.encoder(projected, src_key_padding_mask=~mask)
        encoded = encoded * mask.unsqueeze(-1).to(encoded)
        scores = (encoded * self.pool_query).sum(dim=-1) / math.sqrt(self.hidden_dim)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        pooled = self.output_norm((weights.unsqueeze(-1) * encoded).sum(dim=1))
        return {
            "available_tokens": encoded,
            "available_mask": mask,
            "pool_weights": weights,
            "z_available": pooled,
        }


__all__ = ["AvailableModalityContext"]
