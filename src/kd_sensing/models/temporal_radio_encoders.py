"""Fair temporal encoders for sparse-radio ablations."""

from __future__ import annotations

import torch
from torch import nn


TEMPORAL_RADIO_METHODS = ("last", "mean", "gru", "lstm", "tcn", "transformer")


class _MeanMLP(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        width = hidden_dim * 4
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, hidden_dim),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        pooled = frames.mean(dim=1)
        return pooled + self.network(pooled)


class _RecurrentEncoder(nn.Module):
    def __init__(self, kind: str, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        recurrent = nn.GRU if kind == "gru" else nn.LSTM
        self.network = recurrent(
            hidden_dim,
            hidden_dim,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.network(frames)[0][:, -1]


class _TemporalConv(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        width = hidden_dim * 2
        self.network = nn.Sequential(
            nn.Conv1d(hidden_dim, width, 3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, hidden_dim, 3, padding=1),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        residual = frames[:, -1]
        encoded = self.network(frames.transpose(1, 2)).transpose(1, 2)[:, -1]
        return self.norm(residual + encoded)


class _TemporalTransformer(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, heads: int, dropout: float, maximum_frames: int) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, maximum_frames, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.network = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[1] > self.position.shape[1]:
            raise ValueError("Temporal Transformer received more frames than configured.")
        encoded = self.network(frames + self.position[:, : frames.shape[1]])
        return self.norm(encoded[:, -1])


class TemporalRadioEncoder(nn.Module):
    """Map frame-level radio features ``[B,T,D]`` to ``[B,D]``."""

    def __init__(
        self,
        method: str,
        *,
        hidden_dim: int = 128,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        maximum_frames: int = 5,
    ) -> None:
        super().__init__()
        self.method = str(method).lower()
        self.hidden_dim = int(hidden_dim)
        if self.method not in TEMPORAL_RADIO_METHODS:
            raise ValueError(f"Unknown temporal radio encoder: {method}.")
        if self.method == "last":
            self.encoder: nn.Module = nn.Identity()
        elif self.method == "mean":
            self.encoder = _MeanMLP(self.hidden_dim, float(dropout))
        elif self.method in {"gru", "lstm"}:
            self.encoder = _RecurrentEncoder(self.method, self.hidden_dim, int(layers), float(dropout))
        elif self.method == "tcn":
            self.encoder = _TemporalConv(self.hidden_dim, float(dropout))
        else:
            self.encoder = _TemporalTransformer(
                self.hidden_dim,
                int(layers),
                int(heads),
                float(dropout),
                int(maximum_frames),
            )

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        frames = torch.as_tensor(frame_features)
        if frames.ndim != 3 or frames.shape[-1] != self.hidden_dim or frames.shape[1] < 1:
            raise ValueError(f"frame_features must have shape [B,T,{self.hidden_dim}].")
        if self.method == "last":
            return frames[:, -1]
        return self.encoder(frames)


__all__ = ["TEMPORAL_RADIO_METHODS", "TemporalRadioEncoder"]
