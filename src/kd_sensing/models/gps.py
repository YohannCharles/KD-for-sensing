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
    ):
        super().__init__()
        if gps_input_size <= 0:
            raise ValueError(f"gps_input_size ({gps_input_size}) must be positive.")
        self.gps_input_size = gps_input_size
        self.net = nn.Sequential(
            nn.Linear(gps_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_feature),
        )

    def forward(self, gps_batch: torch.Tensor) -> torch.Tensor:
        if gps_batch.ndim != 3:
            raise ValueError(f"GPS input must have shape [B, T, D], got {tuple(gps_batch.shape)}.")
        batch_size, seq_len, feature_dim = gps_batch.shape
        if feature_dim != self.gps_input_size:
            raise ValueError(
                f"GPS input feature_dim ({feature_dim}) must equal gps_input_size ({self.gps_input_size})."
            )
        features = self.net(gps_batch.reshape(batch_size * seq_len, feature_dim))
        return features.view(batch_size, seq_len, -1)

MODELS.register_removed(
    "gps_strong",
    "Use model.primary.type='modular_sequence' with encoders.gps.type='gps_mlp', representation_core.type='single_gru', and heads.beam.type='beam_head'.",
)
MODELS.register_removed(
    "gps_lightweight",
    "Use configs/gps/lightweight.yaml with model.primary.type='modular_sequence' and encoders.gps.type='gps_mlp'.",
)
MODELS.register_removed("gps_teacher", "Use configs/gps/strong.yaml with model.primary.type='modular_sequence'.")
MODELS.register_removed("gps_student", "Use configs/gps/lightweight.yaml with model.primary.type='modular_sequence'.")
