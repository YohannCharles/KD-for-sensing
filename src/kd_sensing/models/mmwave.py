from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.registries import MODELS


MMWAVE_INPUT_SIZE = 64


def _resolve_feature_size(n_feature: int | None, feature_size: int | None) -> int:
    value = n_feature if n_feature is not None else feature_size
    if value is None:
        raise ValueError("MmWaveFeatureExtractor requires n_feature or feature_size.")
    return int(value)


def _validate_mmwave_input_size(mmwave_input_size: int) -> int:
    value = int(mmwave_input_size)
    if value != MMWAVE_INPUT_SIZE:
        raise ValueError(f"mmwave_input_size ({value}) must equal {MMWAVE_INPUT_SIZE}.")
    return value


def _validate_mmwave_gru_params(
    feature_size: int,
    gru_params: list[int] | tuple[int, int, int],
) -> tuple[int, int, int]:
    if len(gru_params) != 3:
        raise ValueError("gru_params must contain [input_size, hidden_size, num_layers].")
    gru_input_size, gru_hidden_size, gru_num_layers = gru_params
    if int(gru_input_size) != int(feature_size):
        raise ValueError(
            f"gru_input_size ({gru_input_size}) must equal feature_size ({feature_size})"
        )
    return int(gru_input_size), int(gru_hidden_size), int(gru_num_layers)


@MODELS.register("mmwave_feature_extractor")
class MmWaveFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_feature: int | None = None,
        *,
        feature_size: int | None = None,
        mmwave_input_size: int = MMWAVE_INPUT_SIZE,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        output_size = _resolve_feature_size(n_feature, feature_size)
        self.mmwave_input_size = _validate_mmwave_input_size(mmwave_input_size)
        self.net = nn.Sequential(
            nn.Linear(self.mmwave_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, mmwave_batch: torch.Tensor) -> torch.Tensor:
        if mmwave_batch.ndim != 3:
            raise ValueError(f"mmWave input must have shape [B, T, 64], got {tuple(mmwave_batch.shape)}.")
        batch_size, seq_len, feature_dim = mmwave_batch.shape
        if int(feature_dim) != self.mmwave_input_size:
            raise ValueError(
                f"mmWave input feature_dim ({feature_dim}) must equal mmwave_input_size "
                f"({self.mmwave_input_size})."
            )
        features = self.net(mmwave_batch.reshape(batch_size * seq_len, feature_dim))
        return features.view(batch_size, seq_len, -1)


@MODELS.register("mmwave_teacher")
class MmWaveModalityNet(nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_classes: int,
        gru_params: list[int] | tuple[int, int, int],
        mmwave_input_size: int = MMWAVE_INPUT_SIZE,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.name = "MmWaveModalityNet"
        self.mmwave_input_size = _validate_mmwave_input_size(mmwave_input_size)
        gru_input_size, gru_hidden_size, gru_num_layers = _validate_mmwave_gru_params(
            feature_size,
            gru_params,
        )
        self.feature_extraction = MmWaveFeatureExtractor(
            feature_size=feature_size,
            mmwave_input_size=self.mmwave_input_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(gru_input_size)
        self.GRU = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=0.5 if gru_num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.temporal_attention = nn.Sequential(
            nn.Linear(gru_hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, mmwave_batch: torch.Tensor):
        features = self.feature_extraction(mmwave_batch)
        seq_len = int(features.shape[1])
        features = self.layer_norm(features)
        seq_out, _ = self.GRU(features)
        attn_weights = F.softmax(self.temporal_attention(seq_out), dim=1)
        context_vector = torch.sum(seq_out * attn_weights, dim=1)
        enhanced_seq_out = seq_out + context_vector.unsqueeze(1).expand(-1, seq_len, -1)
        pred = self.classifier(enhanced_seq_out)
        return pred, features, enhanced_seq_out


@MODELS.register("mmwave_student")
class MmWaveStudentModalityNet(nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_classes: int,
        gru_params: list[int] | tuple[int, int, int],
        mmwave_input_size: int = MMWAVE_INPUT_SIZE,
        width_multiplier: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.name = "MmWaveStudentModalityNet"
        self.mmwave_input_size = _validate_mmwave_input_size(mmwave_input_size)
        gru_input_size, gru_hidden_size, gru_num_layers = _validate_mmwave_gru_params(
            feature_size,
            gru_params,
        )
        hidden_size = max(int(64 * float(width_multiplier)), 16)
        self.feature_extraction = MmWaveFeatureExtractor(
            feature_size=feature_size,
            mmwave_input_size=self.mmwave_input_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(gru_input_size)
        self.GRU = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=0.3 if gru_num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, mmwave_batch: torch.Tensor):
        features = self.feature_extraction(mmwave_batch)
        features = self.layer_norm(features)
        seq_out, _ = self.GRU(features)
        pred = self.classifier(seq_out)
        return pred, features, seq_out
