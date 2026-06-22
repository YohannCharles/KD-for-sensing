import torch
import torch.nn as nn

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

MODELS.register_removed(
    "mmwave_feature_extractor",
    "Use encoders.mmwave.type='mmwave_mlp' in model.primary.type='modular_sequence', or import MmWaveFeatureExtractor directly.",
)
MODELS.register_removed(
    "mmwave_strong",
    "Use model.primary.type='modular_sequence' with encoders.mmwave.type='mmwave_mlp', representation_core.type='single_gru', and heads.beam.type='beam_head'.",
)
MODELS.register_removed(
    "mmwave_lightweight",
    "Use configs/mmwave/lightweight.yaml with model.primary.type='modular_sequence' and encoders.mmwave.type='mmwave_mlp'.",
)
MODELS.register_removed("mmwave_teacher", "Use configs/mmwave/strong.yaml with model.primary.type='modular_sequence'.")
MODELS.register_removed("mmwave_student", "Use configs/mmwave/lightweight.yaml with model.primary.type='modular_sequence'.")
