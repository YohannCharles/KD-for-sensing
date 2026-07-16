import math

import torch
import torch.nn as nn

from kd_sensing.registries import MODELS


class GpsFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_feature: int,
        gps_input_size: int = 3,
        hidden_size: int = 64,
        dropout: float = 0.1,
        normalized_feature_jitter_std: float = 0.0,
    ):
        super().__init__()
        if gps_input_size <= 0:
            raise ValueError(f"gps_input_size ({gps_input_size}) must be positive.")
        if not math.isfinite(float(normalized_feature_jitter_std)) or float(normalized_feature_jitter_std) < 0.0:
            raise ValueError("normalized_feature_jitter_std must be a finite non-negative value.")
        self.gps_input_size = gps_input_size
        self.output_dim = int(n_feature)
        self.hidden_size = int(hidden_size)
        self.dropout = float(dropout)
        self.normalized_feature_jitter_std = float(normalized_feature_jitter_std)
        self.net = nn.Sequential(
            nn.Linear(gps_input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.output_dim),
        )

    def forward(self, gps_batch: torch.Tensor) -> torch.Tensor:
        if gps_batch.ndim != 3:
            raise ValueError(f"GPS input must have shape [B, T, D], got {tuple(gps_batch.shape)}.")
        batch_size, seq_len, feature_dim = gps_batch.shape
        if feature_dim != self.gps_input_size:
            raise ValueError(
                f"GPS input feature_dim ({feature_dim}) must equal gps_input_size ({self.gps_input_size})."
            )
        if self.training and self.normalized_feature_jitter_std:
            gps_batch = gps_batch + torch.randn_like(gps_batch) * self.normalized_feature_jitter_std
        features = self.net(gps_batch.reshape(batch_size * seq_len, feature_dim))
        return features.view(batch_size, seq_len, -1)

    def training_strategy_metadata(self) -> dict[str, object]:
        return {
            "encoder": "gps_mlp",
            "output_dim": self.output_dim,
            "gps_input_size": self.gps_input_size,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "normalized_feature_jitter_std": self.normalized_feature_jitter_std,
            "jitter_mode": "training_only_normalized_features" if self.normalized_feature_jitter_std else "disabled",
        }
