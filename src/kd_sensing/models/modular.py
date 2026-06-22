from typing import Any

import torch
import torch.nn as nn

from kd_sensing.modalities import (
    MODALITY_ORDER,
    REMOVED_IMAGE_ENCODERS,
    image_profile_spec,
    normalize_modalities,
    resolve_image_profile,
    validate_image_encoder_profile,
)
from kd_sensing.models.auxiliary_heads import TemporalAuxiliaryHeads
from kd_sensing.models.csi_encoder import PilotDualViewCSIEncoder
import kd_sensing.models.geometry_prior  # noqa: F401
from kd_sensing.models.gps import GpsFeatureExtractor
from kd_sensing.models.image_encoders import ResNet18ImageEncoder
from kd_sensing.models.lidar import LidarFeatureExtractor
from kd_sensing.models.mmwave import MMWAVE_INPUT_SIZE, MmWaveFeatureExtractor
from kd_sensing.models.radar import RadarFeatureExtractor
from kd_sensing.registries import ENCODERS, HEADS, MODELS, PROJECTORS, REPRESENTATION_CORES


def _resolve_dim(value: int | None, *fallbacks: int | None, default: int = 64) -> int:
    for candidate in (value, *fallbacks):
        if candidate is not None:
            resolved = int(candidate)
            if resolved <= 0:
                raise ValueError(f"dimension must be positive, got {resolved}.")
            return resolved
    return int(default)


@ENCODERS.register("radar_cnn")
class RadarCNNEncoder(RadarFeatureExtractor):
    def __init__(
        self,
        output_dim: int | None = None,
        *,
        feature_size: int | None = None,
        d_model: int | None = None,
        radar_channels: int = 2,
        in_channels: int | None = None,
        **_: Any,
    ):
        self.output_dim = _resolve_dim(output_dim, feature_size, d_model)
        super().__init__(self.output_dim, in_channels=int(in_channels or radar_channels))


@ENCODERS.register("gps_mlp")
class GpsMLPEncoder(GpsFeatureExtractor):
    def __init__(
        self,
        output_dim: int | None = None,
        *,
        feature_size: int | None = None,
        d_model: int | None = None,
        gps_input_size: int = 3,
        hidden_size: int = 64,
        dropout: float = 0.1,
        **_: Any,
    ):
        self.output_dim = _resolve_dim(output_dim, feature_size, d_model)
        super().__init__(self.output_dim, gps_input_size=gps_input_size, hidden_size=hidden_size, dropout=dropout)


@ENCODERS.register("lidar_cnn")
class LidarCNNEncoder(LidarFeatureExtractor):
    def __init__(
        self,
        output_dim: int | None = None,
        *,
        feature_size: int | None = None,
        d_model: int | None = None,
        lidar_channels: int = 3,
        in_channels: int | None = None,
        **_: Any,
    ):
        self.output_dim = _resolve_dim(output_dim, feature_size, d_model)
        super().__init__(self.output_dim, in_channels=int(in_channels or lidar_channels))


class PointCloudMLPEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int | None = None,
        *,
        feature_size: int | None = None,
        d_model: int | None = None,
        input_profile: str | None = None,
        hidden_size: int = 64,
        dropout: float = 0.1,
        **_: Any,
    ) -> None:
        super().__init__()
        self.output_dim = _resolve_dim(output_dim, feature_size, d_model)
        self.input_profile = input_profile
        hidden = int(hidden_size)
        self.point_mlp = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.output_dim),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(self.output_dim * 2),
            nn.Linear(self.output_dim * 2, self.output_dim),
        )

    def forward(self, lidar_batch: torch.Tensor) -> torch.Tensor:
        if lidar_batch.ndim != 4 or int(lidar_batch.shape[-1]) != 3:
            raise ValueError(
                "point_cloud_mlp expects LiDAR point cloud input [B, T, P, 3], "
                f"got {tuple(lidar_batch.shape)}."
            )
        points = lidar_batch.to(dtype=torch.float32)
        encoded = self.point_mlp(points)
        mean = encoded.mean(dim=2)
        max_values = encoded.max(dim=2).values
        return self.projection(torch.cat([mean, max_values], dim=-1))


@ENCODERS.register("mmwave_mlp")
class MmWaveMLPEncoder(MmWaveFeatureExtractor):
    def __init__(
        self,
        output_dim: int | None = None,
        *,
        feature_size: int | None = None,
        d_model: int | None = None,
        mmwave_input_size: int = MMWAVE_INPUT_SIZE,
        csi_train_rms: float = 1.0,
        hidden_size: int = 128,
        dropout: float = 0.1,
        **_: Any,
    ):
        self.output_dim = _resolve_dim(output_dim, feature_size, d_model)
        super().__init__(
            feature_size=self.output_dim,
            mmwave_input_size=mmwave_input_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )


@PROJECTORS.register("linear")
class LinearProjector(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.0, layer_norm: bool = True, **_: Any):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(d_model)
        layers: list[nn.Module] = []
        if bool(layer_norm):
            layers.append(nn.LayerNorm(self.input_dim))
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(self.input_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim not in {3, 4}:
            raise ValueError(f"Projector input must have shape [B, T, D] or [B, T, K, D], got {tuple(features.shape)}.")
        return self.net(features)


@PROJECTORS.register("identity")
class IdentityProjector(nn.Module):
    def __init__(self, input_dim: int, d_model: int | None = None, **_: Any):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(input_dim if d_model is None else d_model)
        if self.input_dim != self.output_dim:
            raise ValueError(
                f"identity projector requires input_dim == d_model, got {self.input_dim} and {self.output_dim}."
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim not in {3, 4}:
            raise ValueError(f"Projector input must have shape [B, T, D] or [B, T, K, D], got {tuple(features.shape)}.")
        return features


@REPRESENTATION_CORES.register("single_gru")
class SingleGRUCore(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_size: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        **_: Any,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.output_dim = int(hidden_size or d_model)
        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.output_dim,
            num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"single_gru core expects [B, T, D], got {tuple(features.shape)}.")
        output, _ = self.gru(features)
        return output


@REPRESENTATION_CORES.register("early_concat_gru")
class EarlyConcatGRUCore(nn.Module):
    def __init__(
        self,
        d_model: int,
        modality_count: int,
        hidden_size: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        **_: Any,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.modality_count = int(modality_count)
        self.output_dim = int(hidden_size or d_model)
        self.gru = nn.GRU(
            input_size=self.d_model * self.modality_count,
            hidden_size=self.output_dim,
            num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"early_concat_gru core expects [B, K, T, D], got {tuple(features.shape)}.")
        batch_size, modality_count, seq_len, d_model = features.shape
        if int(modality_count) != self.modality_count or int(d_model) != self.d_model:
            raise ValueError(
                "early_concat_gru core received incompatible features: "
                f"expected K={self.modality_count}, D={self.d_model}, got {tuple(features.shape)}."
            )
        concat = features.permute(0, 2, 1, 3).reshape(batch_size, seq_len, modality_count * d_model)
        output, _ = self.gru(concat)
        return output


@REPRESENTATION_CORES.register("snapshot_frame")
class SnapshotFrameCore(nn.Module):
    def __init__(
        self,
        d_model: int,
        modality_count: int | None = None,
        hidden_size: int | None = None,
        output_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        **_: Any,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.modality_count = None if modality_count is None else int(modality_count)
        self.output_dim = int(output_dim or hidden_size or d_model)
        if self.d_model <= 0 or self.output_dim <= 0:
            raise ValueError("snapshot_frame dimensions must be positive.")
        self.dropout = nn.Dropout(float(dropout))
        self.activation = str(activation)
        self.single_norm = nn.LayerNorm(self.d_model)
        self.single_projection = _snapshot_projection(
            self.d_model,
            self.output_dim,
            dropout=float(dropout),
            activation=activation,
        )
        self._multi_fusion: nn.Module | None = (
            self._build_multi_fusion(self.modality_count, dropout=float(dropout), activation=activation)
            if self.modality_count is not None and self.modality_count > 1
            else None
        )
        if self._multi_fusion is not None:
            self.add_module("multi_fusion", self._multi_fusion)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 3:
            batch_size, seq_len, d_model = features.shape
            self._validate_time_dim(seq_len, features.shape)
            if int(d_model) != self.d_model:
                raise ValueError(
                    f"snapshot_frame core expected D={self.d_model}, got {tuple(features.shape)}."
                )
            current = self.single_norm(features)
            return self.single_projection(current)
        if features.ndim == 4:
            batch_size, modality_count, seq_len, d_model = features.shape
            self._validate_time_dim(seq_len, features.shape)
            if int(d_model) != self.d_model:
                raise ValueError(
                    f"snapshot_frame core expected D={self.d_model}, got {tuple(features.shape)}."
                )
            if self.modality_count is not None and int(modality_count) != self.modality_count:
                raise ValueError(
                    "snapshot_frame core received incompatible modality count: "
                    f"expected K={self.modality_count}, got {tuple(features.shape)}."
                )
            fusion = self._multi_fusion_module(int(modality_count), features.device)
            current_tokens = features[:, :, 0, :].reshape(batch_size, modality_count * self.d_model)
            fused = fusion(current_tokens)
            return fused.unsqueeze(1)
        raise ValueError(f"snapshot_frame core expects [B, 1, D] or [B, K, 1, D], got {tuple(features.shape)}.")

    @staticmethod
    def _validate_time_dim(seq_len: int, shape: torch.Size) -> None:
        if int(seq_len) != 1:
            raise ValueError(
                "snapshot_frame baseline requires seq_len=1 and num_pred=1; "
                f"received time dimension T={int(seq_len)} from input shape {tuple(shape)}."
            )

    def _multi_fusion_module(self, modality_count: int, device: torch.device) -> nn.Module:
        input_dim = int(modality_count) * self.d_model
        if self._multi_fusion is None:
            self._multi_fusion = self._build_multi_fusion(
                modality_count,
                dropout=float(self.dropout.p),
                activation=self.activation,
            )
            self.add_module("multi_fusion", self._multi_fusion)
        first_linear = next(module for module in self._multi_fusion.modules() if isinstance(module, nn.Linear))
        if int(first_linear.in_features) != input_dim:
            raise ValueError(
                "snapshot_frame core was initialized for a different modality count; "
                f"expected fused input {first_linear.in_features}, got {input_dim}."
            )
        return self._multi_fusion.to(device)

    def _build_multi_fusion(self, modality_count: int, *, dropout: float, activation: str) -> nn.Module:
        input_dim = int(modality_count) * self.d_model
        hidden_dim = max(input_dim, self.output_dim)
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            _snapshot_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.output_dim),
        )


@REPRESENTATION_CORES.register("token_aware_transformer")
@REPRESENTATION_CORES.register("token_transformer")
class TokenTransformerCore(nn.Module):
    def __init__(
        self,
        d_model: int,
        modality_count: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        **_: Any,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.modality_count = int(modality_count)
        if self.d_model % int(num_heads) != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({num_heads}).")
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dropout=float(dropout),
            dim_feedforward=max(self.d_model * 4, 64),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.output_dim = self.d_model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"token_transformer core expects [B, K, T, D], got {tuple(features.shape)}.")
        batch_size, modality_count, seq_len, d_model = features.shape
        tokens = features.permute(0, 2, 1, 3).reshape(batch_size, seq_len * modality_count, d_model)
        memory = self.transformer(tokens)
        return memory.view(batch_size, seq_len, modality_count, d_model).mean(dim=2)


@REPRESENTATION_CORES.register("amber_lite_missing_modality_transformer")
class AmberLiteMissingModalityTransformerCore(nn.Module):
    supports_missing_modality_metadata = True
    supports_reliability_metadata = True

    def __init__(
        self,
        d_model: int,
        modality_count: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        output_dim: int | None = None,
        mask_token_strategy: str = "learned_per_modality",
        **_: Any,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.modality_count = int(modality_count)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.max_seq_len = int(max_seq_len)
        self.output_dim = int(output_dim or d_model)
        self.mask_token_strategy = str(mask_token_strategy)
        if self.d_model <= 0 or self.modality_count <= 0 or self.output_dim <= 0:
            raise ValueError("amber_lite_missing_modality_transformer dimensions must be positive.")
        if self.num_heads <= 0 or self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({num_heads}).")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")
        if self.mask_token_strategy not in {"learned_per_modality", "learned_shared"}:
            raise ValueError(
                "mask_token_strategy must be 'learned_per_modality' or 'learned_shared', "
                f"got {mask_token_strategy!r}."
            )

        token_count = self.modality_count if self.mask_token_strategy == "learned_per_modality" else 1
        self.mask_tokens = nn.Parameter(torch.zeros(token_count, self.d_model))
        self.modality_embedding = nn.Embedding(self.modality_count, self.d_model)
        self.time_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(float(dropout))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dropout=float(dropout),
            dim_feedforward=max(self.d_model * 4, 64),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_projection = (
            nn.Identity() if self.output_dim == self.d_model else nn.Linear(self.d_model, self.output_dim)
        )
        nn.init.trunc_normal_(self.mask_tokens, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        *,
        modality_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                "amber_lite_missing_modality_transformer requires multimodal [B, K, T, D] input, "
                f"got {tuple(features.shape)}."
            )
        batch_size, modality_count, seq_len, d_model = features.shape
        if int(modality_count) != self.modality_count:
            raise ValueError(
                "amber_lite_missing_modality_transformer received incompatible modality count: "
                f"expected K={self.modality_count}, got K={int(modality_count)}."
            )
        if int(d_model) != self.d_model:
            raise ValueError(
                "amber_lite_missing_modality_transformer received incompatible feature dimension: "
                f"expected D={self.d_model}, got D={int(d_model)}."
            )
        if int(seq_len) > self.max_seq_len:
            raise ValueError(
                "amber_lite_missing_modality_transformer received too many history steps: "
                f"T={int(seq_len)} exceeds max_seq_len={self.max_seq_len}."
            )

        if modality_available is not None:
            availability = _coerce_core_availability_mask(
                modality_available,
                features=features,
                core_name="amber_lite_missing_modality_transformer",
            )
            mask_tokens = self._mask_tokens(features.device, features.dtype)
            features = torch.where(availability.unsqueeze(-1), features, mask_tokens)

        time_ids = torch.arange(int(seq_len), device=features.device)
        time = self.time_embedding(time_ids).view(1, 1, int(seq_len), self.d_model)
        modality_ids = torch.arange(self.modality_count, device=features.device)
        modality = self.modality_embedding(modality_ids).view(1, self.modality_count, 1, self.d_model)
        tokens = self.input_dropout(self.input_norm(features + time + modality))
        tokens = tokens.permute(0, 2, 1, 3).contiguous().view(
            batch_size,
            int(seq_len) * self.modality_count,
            self.d_model,
        )
        memory = self.transformer(tokens)
        fused = memory.view(batch_size, int(seq_len), self.modality_count, self.d_model).mean(dim=2)
        return self.output_projection(self.output_norm(fused))

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": "amber_lite_missing_modality_transformer",
            "d_model": self.d_model,
            "output_dim": self.output_dim,
            "modality_count": self.modality_count,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "max_seq_len": self.max_seq_len,
            "mask_token_strategy": self.mask_token_strategy,
            "consumes_reliability_metadata": True,
            "consumes_missing_modality_metadata": True,
            "missing_metadata_fields": [
                "image_valid_mask",
                "radar_valid_mask",
                "gps_valid_mask",
                "lidar_valid_mask",
            ],
        }

    def _mask_tokens(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tokens = self.mask_tokens
        if self.mask_token_strategy == "learned_shared":
            tokens = tokens.expand(self.modality_count, -1)
        return tokens.to(device=device, dtype=dtype).view(1, self.modality_count, 1, self.d_model)


@REPRESENTATION_CORES.register("next_beam_query_transformer")
class NextBeamQueryTransformerCore(nn.Module):
    def __init__(
        self,
        d_model: int,
        modality_count: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        output_dim: int | None = None,
        **_: Any,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.modality_count = int(modality_count)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.max_seq_len = int(max_seq_len)
        self.output_dim = int(output_dim or d_model)
        if self.d_model <= 0 or self.modality_count <= 0 or self.output_dim <= 0:
            raise ValueError("next_beam_query_transformer dimensions must be positive.")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads}).")

        self.modality_embedding = nn.Embedding(self.modality_count, self.d_model)
        self.time_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        self.next_beam_query = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.input_norm = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(float(dropout))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dropout=float(dropout),
            dim_feedforward=max(self.d_model * 4, 64),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_projection = (
            nn.Identity() if self.output_dim == self.d_model else nn.Linear(self.d_model, self.output_dim)
        )
        nn.init.trunc_normal_(self.next_beam_query, std=0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                "next_beam_query_transformer core requires multimodal [B, K, T, D] input, "
                f"got {tuple(features.shape)}."
            )
        batch_size, modality_count, seq_len, d_model = features.shape
        if int(modality_count) != self.modality_count:
            raise ValueError(
                "next_beam_query_transformer core received incompatible modality count: "
                f"expected K={self.modality_count}, got K={int(modality_count)} from shape {tuple(features.shape)}."
            )
        if int(d_model) != self.d_model:
            raise ValueError(
                "next_beam_query_transformer core received incompatible feature dimension: "
                f"expected D={self.d_model}, got D={int(d_model)} from shape {tuple(features.shape)}."
            )
        if int(seq_len) > self.max_seq_len:
            raise ValueError(
                "next_beam_query_transformer core received too many history steps: "
                f"T={int(seq_len)} exceeds max_seq_len={self.max_seq_len}."
            )

        time_ids = torch.arange(int(seq_len), device=features.device)
        time = self.time_embedding(time_ids).view(1, 1, int(seq_len), self.d_model)
        modality_ids = torch.arange(self.modality_count, device=features.device)
        modality = self.modality_embedding(modality_ids).view(1, self.modality_count, 1, self.d_model)
        tokens = self.input_dropout(self.input_norm(features + time + modality))
        tokens = tokens.permute(0, 2, 1, 3).contiguous().view(batch_size, int(seq_len) * self.modality_count, self.d_model)
        query = self.next_beam_query.expand(batch_size, -1, -1)
        query = self.input_dropout(self.input_norm(query))
        memory = self.transformer(torch.cat([tokens, query], dim=1))
        query_hidden = self.output_norm(memory[:, -1, :])
        return self.output_projection(query_hidden).unsqueeze(1)


@REPRESENTATION_CORES.register("feature_consistency_gate")
@REPRESENTATION_CORES.register("jepa_feature_consistency_gate")
class FeatureConsistencyGateCore(nn.Module):
    def __init__(
        self,
        d_model: int,
        modality_count: int,
        output_dim: int | None = None,
        image_index: int = 0,
        gps_index: int = 1,
        history_window: int = 4,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.modality_count = int(modality_count)
        self.output_dim = int(output_dim or d_model)
        self.image_index = int(image_index)
        self.gps_index = int(gps_index)
        self.history_window = int(history_window)
        if self.d_model <= 0 or self.modality_count <= 0 or self.output_dim <= 0:
            raise ValueError("feature_consistency_gate dimensions must be positive.")
        if not 0 <= self.image_index < self.modality_count:
            raise ValueError(
                "feature_consistency_gate image_index must select an enabled modality, "
                f"got image_index={self.image_index}, modality_count={self.modality_count}."
            )
        if not 0 <= self.gps_index < self.modality_count:
            raise ValueError(
                "feature_consistency_gate gps_index must select an enabled modality, "
                f"got gps_index={self.gps_index}, modality_count={self.modality_count}."
            )
        if self.history_window <= 0:
            raise ValueError(f"feature_consistency_gate history_window must be positive, got {history_window}.")
        hidden = int(hidden_dim or max(self.d_model * 2, 32))
        self.gps_residual_projection = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(self.d_model * 5),
            nn.Linear(self.d_model * 5, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, 3),
        )
        self.output_projection = (
            nn.Identity() if self.output_dim == self.d_model else nn.Linear(self.d_model, self.output_dim)
        )
        self.last_feature_consistency_diagnostics: dict[str, Any] | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                "feature_consistency_gate core requires multimodal [B, K, T, D] input, "
                f"got {tuple(features.shape)}."
            )
        batch_size, modality_count, seq_len, d_model = features.shape
        if int(modality_count) != self.modality_count:
            raise ValueError(
                "feature_consistency_gate received incompatible modality count: "
                f"expected K={self.modality_count}, got K={int(modality_count)} from shape {tuple(features.shape)}."
            )
        if int(d_model) != self.d_model:
            raise ValueError(
                "feature_consistency_gate received incompatible feature dimension: "
                f"expected D={self.d_model}, got D={int(d_model)} from shape {tuple(features.shape)}."
            )
        current = features[:, self.image_index, :, :]
        gps = features[:, self.gps_index, :, :]
        predicted, availability, source_ranges = self._predict_from_history(current)
        gps_residual = current + self.gps_residual_projection(gps - current)
        gate_input = torch.cat(
            [
                current,
                predicted,
                gps_residual,
                current - predicted,
                gps - current,
            ],
            dim=-1,
        )
        weights = torch.softmax(self.gate(gate_input), dim=-1)
        fused = (
            weights[..., 0:1] * current
            + weights[..., 1:2] * predicted
            + weights[..., 2:3] * gps_residual
        )
        self.last_feature_consistency_diagnostics = {
            "type": "feature_consistency_gate",
            "branch_availability": {
                "current": True,
                "temporal_predicted": bool(availability.any().item()),
                "gps_residual": True,
            },
            "history_window": self.history_window,
            "history_source_range": source_ranges,
            "insufficient_history_count": int((~availability).sum().item()),
            "gate_weight_mean": weights.detach().mean(dim=(0, 1)).cpu().tolist(),
            "latent_consistency": {
                "current_predicted_l2": float((current - predicted).detach().pow(2).mean().sqrt().cpu().item()),
                "gps_residual_l2": float((gps - current).detach().pow(2).mean().sqrt().cpu().item()),
            },
            "condition_id_consumed": False,
            "blocked_condition_fields": [
                "c_idx",
                "d_idx",
                "predictive_condition_id",
                "gps_condition",
                "image_condition",
            ],
        }
        return self.output_projection(fused)

    def _predict_from_history(
        self,
        current: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[int] | None]]:
        batch_size, seq_len, _ = current.shape
        predicted = torch.zeros_like(current)
        availability = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=current.device)
        source_ranges: list[list[int] | None] = [None for _ in range(seq_len)]
        for step in range(seq_len):
            start = max(0, step - self.history_window)
            end = step
            if end > start:
                predicted[:, step, :] = current[:, start:end, :].mean(dim=1)
                availability[:, step] = True
                source_ranges[step] = [start, end - 1]
            else:
                predicted[:, step, :] = current[:, step, :]
        return predicted, availability, source_ranges

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": "feature_consistency_gate",
            "d_model": self.d_model,
            "output_dim": self.output_dim,
            "modality_count": self.modality_count,
            "image_index": self.image_index,
            "gps_index": self.gps_index,
            "history_window": self.history_window,
            "consumes_reliability_metadata": False,
            "forbidden_condition_fields": [
                "c_idx",
                "d_idx",
                "predictive_condition_id",
                "gps_condition",
                "image_condition",
            ],
        }


@HEADS.register("beam_head")
@HEADS.register("beam")
class BeamClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.0, **_: Any):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(self.input_dim, self.num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"beam head expects [B, T, D], got {tuple(features.shape)}.")
        return self.net(features)


@MODELS.register("modular_sequence")
class ModularSequenceModel(nn.Module):
    def __init__(
        self,
        *,
        modalities: list[str] | tuple[str, ...] | None = None,
        encoders: dict[str, Any] | None = None,
        projectors: dict[str, Any] | None = None,
        representation_core: dict[str, Any] | None = None,
        heads: dict[str, Any] | None = None,
        feature_size: int = 64,
        d_model: int | None = None,
        num_classes: int = 64,
        num_pred: int = 3,
        image_profile: str | None = None,
        image_channels: int | None = None,
        radar_channels: int = 2,
        gps_input_size: int = 3,
        lidar_channels: int = 3,
        mmwave_input_size: int = MMWAVE_INPUT_SIZE,
        csi_train_rms: float = 1.0,
        auxiliary_heads: bool | dict[str, Any] | None = None,
        paper_metadata: dict[str, Any] | None = None,
        geometry_prior: bool | dict[str, Any] | None = None,
        logit_fusion: dict[str, Any] | None = None,
        geometry_prior_fusion: dict[str, Any] | None = None,
        reranker: bool | dict[str, Any] | None = None,
        **_: Any,
    ):
        super().__init__()
        self.supports_modality_kwargs = True
        self.modalities = normalize_modalities(tuple(modalities or ("image",)), context="modular sequence modalities")
        self.feature_size = int(feature_size)
        self.d_model = int(d_model or feature_size)
        self.num_classes = int(num_classes)
        self.num_pred = int(num_pred)
        self.image_profile = resolve_image_profile(image_profile)
        self.image_channels = int(image_channels or image_profile_spec(self.image_profile).channels)
        self.paper_metadata = dict(paper_metadata or {})

        self.encoders = nn.ModuleDict()
        self.projectors = nn.ModuleDict()
        self.encoder_output_dims: dict[str, int] = {}
        self.encoder_configs: dict[str, dict[str, Any]] = {}
        self.projector_configs: dict[str, dict[str, Any]] = {}

        encoder_cfgs = dict(encoders or {})
        projector_cfgs = dict(projectors or {})
        for modality in self.modalities:
            encoder_cfg = self._encoder_config(
                modality,
                encoder_cfgs.get(modality),
                radar_channels=radar_channels,
                gps_input_size=gps_input_size,
                lidar_channels=lidar_channels,
                mmwave_input_size=mmwave_input_size,
                csi_train_rms=csi_train_rms,
            )
            self._validate_modality_encoder_profile(modality, encoder_cfg)
            self.encoder_configs[modality] = dict(encoder_cfg)
            encoder = ENCODERS.build(encoder_cfg)
            self.encoders[modality] = encoder
            raw_dim = int(getattr(encoder, "output_dim", encoder_cfg.get("output_dim", self.feature_size)))
            self.encoder_output_dims[modality] = raw_dim
            projector_cfg = self._projector_config(projector_cfgs.get(modality), input_dim=raw_dim)
            self.projector_configs[modality] = dict(projector_cfg)
            self.projectors[modality] = PROJECTORS.build(projector_cfg)

        self._validate_encoder_context_dependencies()
        core_cfg = self._core_config(representation_core)
        self.representation_core_config = dict(core_cfg)
        self.representation_core = REPRESENTATION_CORES.build(core_cfg)
        core_output_dim = int(getattr(self.representation_core, "output_dim", self.d_model))
        head_cfgs = dict(heads or {})
        beam_cfg = dict(head_cfgs.get("beam") or head_cfgs.get("beam_head") or {"type": "beam_head"})
        beam_cfg.setdefault("input_dim", core_output_dim)
        beam_cfg.setdefault("num_classes", self.num_classes)
        self.head_configs = {"beam": dict(beam_cfg)}
        self.heads = nn.ModuleDict({"beam": HEADS.build(beam_cfg)})
        self.geometry_prior_config: dict[str, Any] = _optional_component_config(
            geometry_prior,
            default_type="gps_geometry_prior",
        )
        self.geometry_prior: nn.Module | None = None
        self.geometry_prior_fusion_config: dict[str, Any] = {}
        self.geometry_prior_fusion: nn.Module | None = None
        if self.geometry_prior_config:
            if "gps" not in self.modalities:
                raise ValueError("model.primary.geometry_prior.enabled=true requires 'gps' in model.primary.modalities.")
            prior_cfg = dict(self.geometry_prior_config)
            prior_cfg.setdefault("num_classes", self.num_classes)
            prior_cfg.setdefault("num_pred", self.num_pred)
            prior_cfg.setdefault("history_window", self.num_pred)
            prior_cfg.setdefault("gps_source_window", self.num_pred)
            self.geometry_prior_config = prior_cfg
            self.geometry_prior = HEADS.build(prior_cfg)
            fusion_cfg = _optional_component_config(
                logit_fusion or geometry_prior_fusion or prior_cfg.get("fusion"),
                default_type="geometry_prior_logit_fusion",
                default_enabled=True,
            )
            fusion_cfg.setdefault("num_classes", self.num_classes)
            fusion_cfg.setdefault("mode", prior_cfg.get("mode", "assistive"))
            self.geometry_prior_fusion_config = fusion_cfg
            self.geometry_prior_fusion = HEADS.build(fusion_cfg)
        self.reranker_config: dict[str, Any] = _optional_component_config(
            reranker,
            default_type="safe_residual_beam_reranker",
        )
        self.reranker: nn.Module | None = None
        if self.reranker_config:
            rerank_cfg = dict(self.reranker_config)
            rerank_cfg.setdefault("num_classes", self.num_classes)
            self.reranker_config = rerank_cfg
            self.reranker = HEADS.build(rerank_cfg)
        self.auxiliary_heads = TemporalAuxiliaryHeads(
            core_output_dim,
            num_pred=self.num_pred,
            auxiliary_heads=auxiliary_heads,
            dropout=float(beam_cfg.get("dropout", 0.0)),
        )

    def forward(
        self,
        image_batch: torch.Tensor | None = None,
        radar_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
        mmwave_batch: torch.Tensor | None = None,
        csi_batch: torch.Tensor | None = None,
        image_valid_mask: torch.Tensor | None = None,
        radar_valid_mask: torch.Tensor | None = None,
        image_observability_score: torch.Tensor | None = None,
        gps_valid_mask: torch.Tensor | None = None,
        lidar_valid_mask: torch.Tensor | None = None,
        gps_delay_steps: torch.Tensor | None = None,
        image_dropout_mask: torch.Tensor | None = None,
        radar_dropout_mask: torch.Tensor | None = None,
        gps_dropout_mask: torch.Tensor | None = None,
        lidar_dropout_mask: torch.Tensor | None = None,
        gps_counterfactual_mask: torch.Tensor | None = None,
        benchmark_condition_metadata: dict[str, Any] | None = None,
        image_degradation_metadata: dict[str, Any] | None = None,
        missing_modality_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del image_degradation_metadata, missing_modality_metadata
        raw_inputs = {
            "image": image_batch,
            "radar": radar_batch,
            "gps": gps_batch,
            "lidar": lidar_batch,
            "mmwave": mmwave_batch,
            "csi": csi_batch,
        }
        reliability_inputs = {
            "image_valid_mask": image_valid_mask,
            "radar_valid_mask": radar_valid_mask,
            "image_observability_score": image_observability_score,
            "gps_valid_mask": gps_valid_mask,
            "lidar_valid_mask": lidar_valid_mask,
            "gps_delay_steps": gps_delay_steps,
            "image_dropout_mask": image_dropout_mask,
            "radar_dropout_mask": radar_dropout_mask,
            "gps_dropout_mask": gps_dropout_mask,
            "lidar_dropout_mask": lidar_dropout_mask,
            "gps_counterfactual_mask": gps_counterfactual_mask,
            "benchmark_condition_metadata": benchmark_condition_metadata,
        }
        modality_valid_inputs = {
            "image": image_valid_mask,
            "radar": radar_valid_mask,
            "gps": gps_valid_mask,
            "lidar": lidar_valid_mask,
            "mmwave": None,
            "csi": None,
        }
        modality_dropout_inputs = {
            "image": image_dropout_mask,
            "radar": radar_dropout_mask,
            "gps": gps_dropout_mask,
            "lidar": lidar_dropout_mask,
            "mmwave": None,
            "csi": None,
        }
        encoded: dict[str, torch.Tensor] = {}
        projected: dict[str, torch.Tensor] = {}
        encoder_auxiliary_features: dict[str, dict[str, torch.Tensor]] = {}
        encoder_runtime_metadata: dict[str, Any] = {}
        batch_size = None
        seq_len = None
        pending = list(self.modalities)
        while pending:
            progressed = False
            for modality in list(pending):
                encoder = self.encoders[modality]
                dependencies = _encoder_context_dependencies(encoder)
                source = _encoder_context_source(encoder)
                if not _encoder_dependencies_satisfied(dependencies, source=source, encoded=encoded, projected=projected):
                    continue
                tensor = raw_inputs[modality]
                if tensor is None:
                    raise ValueError(f"Modular sequence model requires '{modality}' input because it is enabled.")
                context_kwargs = _encoder_context_kwargs(
                    encoder,
                    modality=modality,
                    raw_tensor=tensor,
                    dependencies=dependencies,
                    raw_inputs=raw_inputs,
                    encoded=encoded,
                    projected=projected,
                )
                context_kwargs.update(_encoder_reliability_kwargs(encoder, modality=modality, reliability_inputs=reliability_inputs))
                features = encoder(tensor, **context_kwargs) if context_kwargs else encoder(tensor)
                temporal_aux_metadata = getattr(encoder, "last_temporal_auxiliary_metadata", None)
                if isinstance(temporal_aux_metadata, dict) and bool(temporal_aux_metadata.get("enabled", False)):
                    current_latent = getattr(encoder, "last_current_latent", None)
                    predicted_latent = getattr(encoder, "last_temporal_predicted_latent", None)
                    if isinstance(current_latent, torch.Tensor) and isinstance(predicted_latent, torch.Tensor):
                        encoder_auxiliary_features[modality] = {
                            "current_latent": current_latent,
                            "temporal_predicted_latent": predicted_latent,
                        }
                    encoder_runtime_metadata[modality] = {
                        "temporal_auxiliary": temporal_aux_metadata,
                    }
                predictive_diagnostics = getattr(encoder, "last_predictive_gps_query_diagnostics", None)
                if isinstance(predictive_diagnostics, dict):
                    encoder_runtime_metadata.setdefault(modality, {})["predictive_gps_query"] = predictive_diagnostics
                visual_token_diagnostics = getattr(encoder, "last_visual_token_diagnostics", None)
                if isinstance(visual_token_diagnostics, dict) and visual_token_diagnostics:
                    encoder_runtime_metadata.setdefault(modality, {})["visual_tokens"] = visual_token_diagnostics
                batch_size, seq_len = _check_temporal_features(features, modality, batch_size, seq_len)
                encoded[modality] = features
                projected_features = self.projectors[modality](features)
                _check_projected_features(projected_features, modality, self.d_model)
                projected[modality] = projected_features
                pending.remove(modality)
                progressed = True
            if not progressed:
                unmet = {
                    modality: _unmet_context_dependencies(
                        _encoder_context_dependencies(self.encoders[modality]),
                        source=_encoder_context_source(self.encoders[modality]),
                        encoded=encoded,
                        projected=projected,
                    )
                    for modality in pending
                }
                raise ValueError(
                    "Unable to satisfy modular sequence encoder condition dependencies; "
                    f"pending modalities={pending}, unmet dependencies={unmet}. "
                    "Check for missing condition modalities or circular dependencies."
                )
        ordered = [projected[modality] for modality in self.modalities]
        has_token_features = any(features.ndim == 4 for features in ordered)
        if has_token_features:
            token_pieces = [features if features.ndim == 4 else features.unsqueeze(2) for features in ordered]
            token_features = torch.cat(token_pieces, dim=2)
            core_input = token_features.permute(0, 2, 1, 3).contiguous()
            availability_mask = _core_input_availability_mask(
                projected,
                self.modalities,
                valid_masks=modality_valid_inputs,
                dropout_masks=modality_dropout_inputs,
                token_features=True,
            )
            input_features = torch.cat(
                [features.mean(dim=2) if features.ndim == 4 else features for features in ordered],
                dim=-1,
            )
        elif len(ordered) == 1:
            core_input = ordered[0]
            availability_mask = _core_input_availability_mask(
                projected,
                self.modalities,
                valid_masks=modality_valid_inputs,
                dropout_masks=modality_dropout_inputs,
                token_features=False,
            )
            input_features = core_input
        else:
            stacked = torch.stack(ordered, dim=1)
            core_input = stacked
            availability_mask = _core_input_availability_mask(
                projected,
                self.modalities,
                valid_masks=modality_valid_inputs,
                dropout_masks=modality_dropout_inputs,
                token_features=False,
            )
            input_features = torch.cat(ordered, dim=-1)
        if bool(getattr(self.representation_core, "supports_missing_modality_metadata", False)):
            if core_input.ndim == 3:
                core_input = core_input.unsqueeze(1)
            output_features = self.representation_core(core_input, modality_available=availability_mask)
        else:
            output_features = self.representation_core(core_input)
        image_logits = self.heads["beam"](output_features)
        logits = image_logits
        geometry_prior_payload: dict[str, Any] | None = None
        geometry_fusion_payload: dict[str, Any] | None = None
        rerank_payload: dict[str, Any] | None = None
        if self.geometry_prior is not None and self.geometry_prior_fusion is not None:
            geometry_prior_payload = self.geometry_prior(
                gps_batch,
                target_time=int(image_logits.shape[1]),
                gps_valid_mask=gps_valid_mask,
                gps_delay_steps=gps_delay_steps,
                gps_counterfactual_mask=gps_counterfactual_mask,
            )
            geometry_fusion_payload = self.geometry_prior_fusion(
                image_logits=image_logits,
                prior_logits=geometry_prior_payload["logits"],
                prior_distribution=geometry_prior_payload.get("distribution"),
                prior_availability_mask=geometry_prior_payload.get("availability_mask"),
                image_valid_mask=image_valid_mask,
                image_observability_score=image_observability_score,
                gps_valid_mask=gps_valid_mask,
                gps_delay_steps=gps_delay_steps,
                gps_counterfactual_mask=gps_counterfactual_mask,
            )
            logits = geometry_fusion_payload["logits"]
        if self.reranker is not None:
            rerank_payload = self.reranker(
                anchor_logits=image_logits,
                geometry_prior_logits=geometry_prior_payload["logits"] if geometry_prior_payload is not None else None,
                image_observability_score=image_observability_score,
                gps_valid_mask=gps_valid_mask,
                gps_delay_steps=gps_delay_steps,
                gps_counterfactual_mask=gps_counterfactual_mask,
            )
            logits = rerank_payload["logits"]
        output = {
            "logits": logits,
            "input_features": input_features,
            "output_features": output_features,
            "modalities": self.modalities,
            "modality_features": projected,
            "encoder_features": encoded,
            "image_profile": self.image_profile,
        }
        if bool(getattr(self.representation_core, "supports_missing_modality_metadata", False)):
            output["missing_modality_metadata"] = _missing_modality_output_metadata(
                availability_mask,
                modalities=self.modalities,
            )
        if has_token_features:
            output["token_features"] = core_input
        if geometry_prior_payload is not None and geometry_fusion_payload is not None:
            fusion_diagnostics = dict(geometry_fusion_payload.get("diagnostics", {}))
            output.update(
                {
                    "anchor_logits": image_logits,
                    "image_logits": image_logits,
                    "geometry_prior_logits": geometry_prior_payload["logits"],
                    "geometry_prior_distribution": geometry_prior_payload["distribution"],
                    "geometry_prior_entropy": geometry_prior_payload["entropy"],
                    "geometry_prior_topk_indices": geometry_prior_payload["topk_indices"],
                    "geometry_prior_topk_probabilities": geometry_prior_payload["topk_probabilities"],
                    "geometry_prior_availability_mask": geometry_prior_payload["availability_mask"],
                    "geometry_prior_unavailable_reason": geometry_prior_payload["unavailable_reason"],
                    "geometry_prior_diagnostics": {
                        "entropy": geometry_prior_payload["entropy"],
                        "topk_indices": geometry_prior_payload["topk_indices"],
                        "availability_mask": geometry_prior_payload["availability_mask"],
                        "unavailable_reason": geometry_prior_payload["unavailable_reason"],
                        "metadata": geometry_prior_payload["metadata"],
                    },
                    "geometry_prior_fusion_diagnostics": fusion_diagnostics,
                    "branch_weights": fusion_diagnostics.get("branch_weights"),
                }
            )
        elif rerank_payload is not None:
            output["anchor_logits"] = image_logits
        if rerank_payload is not None:
            rerank_diagnostics = dict(rerank_payload.get("diagnostics", {}))
            output.update(
                {
                    "rerank_logits": rerank_diagnostics.get("rerank_logits", rerank_payload["logits"]),
                    "safe_rerank_diagnostics": rerank_diagnostics,
                    "candidate_ids": rerank_diagnostics.get("candidate_ids"),
                    "candidate_source_mask": rerank_diagnostics.get("candidate_source_mask"),
                    "selected_source": rerank_diagnostics.get("selected_source"),
                    "target_rank_delta": rerank_diagnostics.get("target_rank_delta"),
                    "fallback_reason": rerank_diagnostics.get("fallback_reason_code"),
                    "gate_confidence": rerank_diagnostics.get("gate_confidence"),
                    "condition_id_consumed": False,
                }
            )
        if encoder_auxiliary_features:
            output["encoder_auxiliary_features"] = encoder_auxiliary_features
        if encoder_runtime_metadata:
            output["runtime_metadata"] = {"encoder_temporal_auxiliary": encoder_runtime_metadata}
            predictive_runtime = {
                modality: metadata["predictive_gps_query"]
                for modality, metadata in encoder_runtime_metadata.items()
                if isinstance(metadata, dict) and isinstance(metadata.get("predictive_gps_query"), dict)
            }
            if predictive_runtime:
                output["predictive_gps_query_diagnostics"] = predictive_runtime
        feature_consistency_diagnostics = getattr(self.representation_core, "last_feature_consistency_diagnostics", None)
        if isinstance(feature_consistency_diagnostics, dict):
            output["feature_consistency_diagnostics"] = feature_consistency_diagnostics
        output.update(self.auxiliary_heads(output_features))
        return output

    def training_strategy_metadata(self) -> dict[str, Any]:
        encoders: dict[str, Any] = {}
        conditioned: dict[str, Any] = {}
        reliability_consumers: list[str] = []
        for modality, encoder in self.encoders.items():
            metadata = _component_training_strategy_metadata(
                encoder,
                self.encoder_configs.get(modality, {}),
                role="encoder",
            )
            dependencies = _encoder_context_dependencies(encoder)
            source = _encoder_context_source(encoder)
            if dependencies:
                metadata = {
                    **metadata,
                    "required_context_modalities": list(dependencies),
                    "context_feature_source": source,
                    "context_feature_kwargs": dict(getattr(encoder, "context_feature_kwargs", {}) or {}),
                }
                conditioned[modality] = {
                    "required_context_modalities": list(dependencies),
                    "context_feature_source": source,
                    "context_feature_kwargs": metadata["context_feature_kwargs"],
                }
            if _component_consumes_reliability_metadata(encoder, metadata):
                reliability_consumers.append(f"encoders.{modality}")
            encoders[modality] = metadata
        projectors = {
            modality: _component_training_strategy_metadata(
                projector,
                self.projector_configs.get(modality, {}),
                role="projector",
            )
            for modality, projector in self.projectors.items()
        }
        core_metadata = _component_training_strategy_metadata(
            self.representation_core,
            self.representation_core_config,
            role="representation_core",
        )
        missing_metadata_consumers: list[str] = []
        if _component_consumes_reliability_metadata(self.representation_core, core_metadata):
            reliability_consumers.append("representation_core")
        if bool(core_metadata.get("consumes_missing_modality_metadata", False)) or bool(
            getattr(self.representation_core, "supports_missing_modality_metadata", False)
        ):
            missing_metadata_consumers.append("representation_core")
        heads = {
            name: _component_training_strategy_metadata(
                head,
                self.head_configs.get(name, {}),
                role="head",
            )
            for name, head in self.heads.items()
        }
        for name, metadata in heads.items():
            if _component_consumes_reliability_metadata(self.heads[name], metadata):
                reliability_consumers.append(f"heads.{name}")
        geometry_prior_metadata: dict[str, Any] | None = None
        if self.geometry_prior is not None:
            geometry_prior_metadata = _component_training_strategy_metadata(
                self.geometry_prior,
                self.geometry_prior_config,
                role="geometry_prior",
            )
            if _component_consumes_reliability_metadata(self.geometry_prior, geometry_prior_metadata):
                reliability_consumers.append("geometry_prior")
        geometry_fusion_metadata: dict[str, Any] | None = None
        if self.geometry_prior_fusion is not None:
            geometry_fusion_metadata = _component_training_strategy_metadata(
                self.geometry_prior_fusion,
                self.geometry_prior_fusion_config,
                role="logit_fusion",
            )
            if _component_consumes_reliability_metadata(self.geometry_prior_fusion, geometry_fusion_metadata):
                reliability_consumers.append("geometry_prior_logit_fusion")
        reranker_metadata: dict[str, Any] | None = None
        if self.reranker is not None:
            reranker_metadata = _component_training_strategy_metadata(
                self.reranker,
                self.reranker_config,
                role="safe_residual_reranker",
            )
            if _component_consumes_reliability_metadata(self.reranker, reranker_metadata):
                reliability_consumers.append("safe_residual_reranker")
        core_type = str(self.representation_core_config.get("type", self.representation_core.__class__.__name__))
        metadata = {
            "type": "modular_sequence",
            "architecture_category": "component_baseline",
            "model_group": "safe_residual_beam_rerank_fusion"
            if reranker_metadata
            else "geometry_prior_beam_fusion"
            if geometry_prior_metadata
            else "modular_sequence",
            "modalities": list(self.modalities),
            "enabled_modalities": list(self.modalities),
            "d_model": self.d_model,
            "encoders": encoders,
            "projectors": projectors,
            "conditioned_encoders": conditioned,
            "representation_core_type": core_type,
            "representation_core_class": self.representation_core.__class__.__name__,
            "representation_core": core_metadata,
            "heads": heads,
            "geometry_prior": geometry_prior_metadata
            or {
                "enabled": False,
                "mode": "disabled",
            },
            "geometry_prior_mode": (geometry_prior_metadata or {}).get("prior_mode", "disabled"),
            "fusion_mode": (geometry_fusion_metadata or {}).get("fusion_mode", core_type),
            "logit_fusion": geometry_fusion_metadata or {"enabled": False, "mode": "disabled"},
            "safe_residual_reranker": reranker_metadata
            or {
                "enabled": False,
                "mode": "disabled",
            },
            "reranker": reranker_metadata
            or {
                "enabled": False,
                "mode": "disabled",
            },
            "loss_mode": "config_resolved",
            "teacher_guidance_mode": "config_resolved",
            "curriculum_mode": "config_resolved",
            "consumes_reliability_metadata": bool(reliability_consumers),
            "reliability_metadata_consumers": reliability_consumers,
            "reliability_metadata": {
                "consumed": bool(reliability_consumers),
                "consumers": list(reliability_consumers),
                "fields": [
                    "image_valid_mask",
                    "image_observability_score",
                    "gps_valid_mask",
                    "gps_delay_steps",
                    "gps_counterfactual_mask",
                    "benchmark_condition_metadata",
                ],
            },
            "consumes_missing_modality_metadata": bool(missing_metadata_consumers),
            "missing_modality_metadata": {
                "consumed": bool(missing_metadata_consumers),
                "consumers": list(missing_metadata_consumers),
                "fields": [f"{modality}_valid_mask" for modality in self.modalities],
                "dropout_fields": [f"{modality}_dropout_mask" for modality in self.modalities],
                "mask_token_strategy": core_metadata.get("mask_token_strategy", "none"),
            },
        }
        if self.paper_metadata:
            metadata.update(self.paper_metadata)
        return metadata

    def _encoder_config(
        self,
        modality: str,
        raw_cfg: Any,
        *,
        radar_channels: int,
        gps_input_size: int,
        lidar_channels: int,
        mmwave_input_size: int,
        csi_train_rms: float,
    ) -> dict[str, Any]:
        if raw_cfg is None:
            raw_cfg = {"type": _default_encoder_type(modality, self.image_profile)}
        if isinstance(raw_cfg, str):
            raw_cfg = {"type": raw_cfg}
        if not isinstance(raw_cfg, dict):
            raise ValueError(f"Encoder config for modality '{modality}' must be a dict or string.")
        cfg = dict(raw_cfg)
        cfg.setdefault("output_dim", self.feature_size)
        if modality == "image":
            cfg.setdefault("image_profile", self.image_profile)
            cfg.setdefault("image_channels", self.image_channels)
        elif modality == "radar":
            cfg.setdefault("radar_channels", radar_channels)
        elif modality == "gps":
            cfg.setdefault("gps_input_size", gps_input_size)
        elif modality == "lidar":
            cfg.setdefault("lidar_channels", lidar_channels)
        elif modality == "mmwave":
            cfg.setdefault("mmwave_input_size", mmwave_input_size)
        elif modality == "csi":
            cfg.setdefault("train_rms", csi_train_rms)
        return cfg

    def _projector_config(self, raw_cfg: Any, *, input_dim: int) -> dict[str, Any]:
        if raw_cfg is None:
            raw_cfg = {"type": "linear"}
        if isinstance(raw_cfg, str):
            raw_cfg = {"type": raw_cfg}
        if not isinstance(raw_cfg, dict):
            raise ValueError("Projector config must be a dict or string.")
        cfg = dict(raw_cfg)
        cfg.setdefault("input_dim", input_dim)
        cfg.setdefault("d_model", self.d_model)
        return cfg

    def _core_config(self, raw_cfg: dict[str, Any] | None) -> dict[str, Any]:
        if raw_cfg is None:
            raw_cfg = {"type": "single_gru" if len(self.modalities) == 1 else "early_concat_gru"}
        cfg = dict(raw_cfg)
        cfg.setdefault("d_model", self.d_model)
        cfg.setdefault("modality_count", len(self.modalities))
        return cfg

    def _validate_modality_encoder_profile(self, modality: str, encoder_cfg: dict[str, Any]) -> None:
        if modality != "image":
            return
        encoder_name = str(encoder_cfg.get("type"))
        if encoder_name == "resnet18_imagenet_rgb" or encoder_name.startswith("tinyvit_"):
            validate_image_encoder_profile(
                encoder_name=encoder_name,
                image_profile=self.image_profile,
                expected_channels=3,
                actual_channels=encoder_cfg.get("image_channels", self.image_channels),
            )
        elif encoder_name in REMOVED_IMAGE_ENCODERS:
            raise ValueError(
                f"Removed image encoder '{encoder_name}' is no longer supported. "
                "Use 'resnet18_imagenet_rgb' with image_profile 'rgb_imagenet'."
            )

    def _validate_encoder_context_dependencies(self) -> None:
        enabled = set(self.modalities)
        for modality, encoder in self.encoders.items():
            dependencies = _encoder_context_dependencies(encoder)
            missing = [dependency for dependency in dependencies if dependency not in enabled]
            if missing:
                raise ValueError(
                    f"Encoder for modality '{modality}' requires condition modalities {missing}, "
                    f"but enabled model.primary.modalities are {list(self.modalities)}."
                )
            if modality in dependencies:
                raise ValueError(
                    f"Encoder for modality '{modality}' cannot depend on its own condition feature."
                )


def _default_encoder_type(modality: str, image_profile: str) -> str:
    if modality == "image":
        return "resnet18_imagenet_rgb"
    return {
        "radar": "radar_cnn",
        "gps": "gps_mlp",
        "lidar": "lidar_cnn",
        "mmwave": "mmwave_mlp",
        "csi": "pilot_dual_view_csi",
    }[modality]


def _optional_component_config(
    raw_cfg: Any,
    *,
    default_type: str,
    default_enabled: bool = False,
) -> dict[str, Any]:
    if raw_cfg in (None, False, "", "none"):
        if not default_enabled:
            return {}
        raw_cfg = {}
    if raw_cfg is True:
        raw_cfg = {}
    if isinstance(raw_cfg, str):
        raw_cfg = {"type": raw_cfg}
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"Optional component config must be a mapping, string, bool, or null, got {type(raw_cfg).__name__}.")
    cfg = dict(raw_cfg)
    enabled = cfg.pop("enabled", default_enabled or bool(cfg))
    if not enabled:
        return {}
    cfg.setdefault("type", default_type)
    return cfg


def _encoder_context_dependencies(encoder: nn.Module) -> tuple[str, ...]:
    raw = getattr(encoder, "required_context_modalities", ())
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = (raw,)
    return tuple(str(item) for item in raw)


def _encoder_context_source(encoder: nn.Module) -> str:
    source = str(getattr(encoder, "context_feature_source", "projected")).strip().lower()
    if source == "none":
        return source
    if source not in {"projected", "encoded", "raw"}:
        raise ValueError(
            "Encoder requested unsupported condition feature source "
            f"{source!r}; supported sources are 'projected', 'encoded', and 'raw'."
        )
    return source


def _component_training_strategy_metadata(
    component: nn.Module,
    cfg: dict[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    raw = component.training_strategy_metadata() if hasattr(component, "training_strategy_metadata") else {}
    metadata = dict(raw) if isinstance(raw, dict) else {}
    registry_type = cfg.get("type")
    if registry_type not in (None, ""):
        registry_type = str(registry_type)
        metadata.setdefault("type", registry_type)
        metadata.setdefault("registry_type", registry_type)
        if role == "encoder":
            metadata.setdefault("encoder", registry_type)
        elif role == "projector":
            metadata.setdefault("projector", registry_type)
        elif role == "representation_core":
            metadata.setdefault("core", registry_type)
        elif role == "head":
            metadata.setdefault("head", registry_type)
    metadata.setdefault("class", component.__class__.__name__)
    metadata.setdefault("component_role", role)
    if "consumes_reliability_metadata" not in metadata:
        metadata["consumes_reliability_metadata"] = _component_consumes_reliability_metadata(component, metadata)
    return metadata


def _component_consumes_reliability_metadata(component: nn.Module, metadata: dict[str, Any] | None = None) -> bool:
    metadata = metadata or {}
    for key in (
        "consumes_reliability_metadata",
        "supports_reliability_metadata",
        "supports_observability_metadata",
        "consumes_missing_modality_metadata",
    ):
        if key in metadata:
            return bool(metadata.get(key))
    temporal_fallback = metadata.get("temporal_fallback")
    if isinstance(temporal_fallback, dict) and bool(temporal_fallback.get("enabled", False)):
        return True
    return bool(
        getattr(component, "consumes_reliability_metadata", False)
        or getattr(component, "supports_reliability_metadata", False)
        or getattr(component, "supports_observability_metadata", False)
        or getattr(component, "supports_missing_modality_metadata", False)
    )


def _encoder_dependencies_satisfied(
    dependencies: tuple[str, ...],
    *,
    source: str,
    encoded: dict[str, torch.Tensor],
    projected: dict[str, torch.Tensor],
) -> bool:
    if source == "raw":
        return True
    if source == "none":
        return not dependencies
    if source == "encoded":
        return all(dependency in encoded for dependency in dependencies)
    if source == "projected":
        return all(dependency in projected for dependency in dependencies)
    return False


def _unmet_context_dependencies(
    dependencies: tuple[str, ...],
    *,
    source: str,
    encoded: dict[str, torch.Tensor],
    projected: dict[str, torch.Tensor],
) -> list[str]:
    if source == "raw":
        return []
    if source == "none":
        return [] if not dependencies else list(dependencies)
    if source == "encoded":
        return [dependency for dependency in dependencies if dependency not in encoded]
    if source == "projected":
        return [dependency for dependency in dependencies if dependency not in projected]
    return list(dependencies)


def _encoder_context_kwargs(
    encoder: nn.Module,
    *,
    modality: str,
    raw_tensor: torch.Tensor,
    dependencies: tuple[str, ...],
    raw_inputs: dict[str, torch.Tensor | None],
    encoded: dict[str, torch.Tensor],
    projected: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not dependencies:
        return {}
    source = _encoder_context_source(encoder)
    kwarg_names = getattr(encoder, "context_feature_kwargs", {})
    if not isinstance(kwarg_names, dict):
        kwarg_names = {}
    context_kwargs: dict[str, torch.Tensor] = {}
    for dependency in dependencies:
        if source == "projected":
            feature = projected[dependency]
        elif source == "encoded":
            feature = encoded[dependency]
        elif source == "raw":
            feature = raw_inputs.get(dependency)
            if feature is None:
                raise ValueError(
                    f"Encoder for modality '{modality}' requested raw condition feature from '{dependency}', "
                    "but that raw batch input is missing."
                )
        else:
            raise ValueError(
                f"Encoder for modality '{modality}' requested unsupported condition feature source "
                f"{source!r}; supported sources are 'projected', 'encoded', and 'raw'."
            )
        _check_condition_feature_shape(
            modality=modality,
            dependency=dependency,
            raw_tensor=raw_tensor,
            condition_features=feature,
            source=source,
        )
        kwarg = str(kwarg_names.get(dependency, f"{dependency}_condition_features"))
        context_kwargs[kwarg] = feature
    return context_kwargs


def _encoder_reliability_kwargs(
    encoder: nn.Module,
    *,
    modality: str,
    reliability_inputs: dict[str, Any],
) -> dict[str, Any]:
    if modality != "image":
        return {}
    if not bool(getattr(encoder, "supports_observability_metadata", False)):
        return {}
    return {key: value for key, value in reliability_inputs.items() if value is not None}


def _check_condition_feature_shape(
    *,
    modality: str,
    dependency: str,
    raw_tensor: torch.Tensor,
    condition_features: torch.Tensor,
    source: str,
) -> None:
    if source != "raw" and condition_features.ndim != 3:
        raise ValueError(
            f"Condition feature for modality '{dependency}' must have shape [B, T, D], "
            f"got {tuple(condition_features.shape)} while encoding '{modality}'."
        )
    if source == "raw" and condition_features.ndim < 2:
        raise ValueError(
            f"Raw condition feature for modality '{dependency}' must expose batch/time dimensions, "
            f"got {tuple(condition_features.shape)} while encoding '{modality}'."
        )
    if raw_tensor.ndim < 2:
        raise ValueError(
            f"Modular sequence input for modality '{modality}' must expose batch/time dimensions, "
            f"got {tuple(raw_tensor.shape)}."
        )
    raw_batch_time = tuple(int(value) for value in raw_tensor.shape[:2])
    condition_batch_time = tuple(int(value) for value in condition_features.shape[:2])
    if raw_batch_time != condition_batch_time:
        raise ValueError(
            "Condition feature batch/time dimensions must match the conditioned encoder input; "
            f"modality '{modality}' input shape {tuple(raw_tensor.shape)}, "
            f"condition modality '{dependency}' feature shape {tuple(condition_features.shape)}."
        )


def _snapshot_projection(input_dim: int, output_dim: int, *, dropout: float, activation: str) -> nn.Module:
    hidden_dim = max(int(input_dim), int(output_dim))
    return nn.Sequential(
        nn.Linear(int(input_dim), hidden_dim),
        _snapshot_activation(activation),
        nn.Dropout(float(dropout)),
        nn.Linear(hidden_dim, int(output_dim)),
    )


def _snapshot_activation(name: str) -> nn.Module:
    normalized = str(name).lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "tanh":
        return nn.Tanh()
    return nn.GELU()


def _core_input_availability_mask(
    projected: dict[str, torch.Tensor],
    modalities: tuple[str, ...],
    *,
    valid_masks: dict[str, torch.Tensor | None],
    dropout_masks: dict[str, torch.Tensor | None],
    token_features: bool,
) -> torch.Tensor | None:
    pieces: list[torch.Tensor] = []
    for modality in modalities:
        features = projected[modality]
        mask = _modality_availability_from_inputs(
            modality,
            features,
            valid_mask=valid_masks.get(modality),
            dropout_mask=dropout_masks.get(modality),
        )
        if token_features:
            token_count = int(features.shape[2]) if features.ndim == 4 else 1
            mask = mask.unsqueeze(2).expand(-1, -1, token_count)
        pieces.append(mask)
    if not pieces:
        return None
    if token_features:
        return torch.cat(pieces, dim=2).permute(0, 2, 1).contiguous()
    return torch.stack(pieces, dim=1)


def _modality_availability_from_inputs(
    modality: str,
    features: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None,
    dropout_mask: torch.Tensor | None,
) -> torch.Tensor:
    batch_size, seq_len = int(features.shape[0]), int(features.shape[1])
    if valid_mask is not None:
        mask = _coerce_temporal_mask(
            valid_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=features.device,
            name=f"{modality}_valid_mask",
        )
    elif dropout_mask is not None:
        mask = ~_coerce_temporal_mask(
            dropout_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=features.device,
            name=f"{modality}_dropout_mask",
        )
    else:
        mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=features.device)
    return mask


def _coerce_temporal_mask(
    mask: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    value = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if value.ndim == 1:
        value = value.unsqueeze(1)
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [B, T] or [B], got {tuple(value.shape)}.")
    if int(value.shape[0]) != int(batch_size):
        raise ValueError(f"{name} batch size must be {batch_size}, got {tuple(value.shape)}.")
    if int(value.shape[1]) == int(seq_len):
        return value
    if int(value.shape[1]) == 1:
        return value.expand(-1, int(seq_len))
    raise ValueError(f"{name} time dimension must be 1 or {seq_len}, got {tuple(value.shape)}.")


def _coerce_core_availability_mask(
    mask: torch.Tensor,
    *,
    features: torch.Tensor,
    core_name: str,
) -> torch.Tensor:
    value = torch.as_tensor(mask, dtype=torch.bool, device=features.device)
    expected = tuple(int(item) for item in features.shape[:3])
    if value.ndim != 3 or tuple(int(item) for item in value.shape) != expected:
        raise ValueError(f"{core_name} modality_available must have shape {expected}, got {tuple(value.shape)}.")
    return value


def _missing_modality_output_metadata(
    availability_mask: torch.Tensor | None,
    *,
    modalities: tuple[str, ...],
) -> dict[str, Any]:
    if availability_mask is None:
        return {"available": True, "modalities": list(modalities), "missing_counts": {}}
    missing = ~availability_mask.detach()
    counts: dict[str, int] = {}
    for index, modality in enumerate(modalities):
        if index >= int(missing.shape[1]):
            break
        counts[modality] = int(missing[:, index, :].sum().cpu().item())
    return {
        "available": True,
        "modalities": list(modalities),
        "availability_mask": availability_mask,
        "missing_counts": counts,
        "provenance": "input_valid_or_dropout_masks",
    }


def _check_temporal_features(
    features: torch.Tensor,
    modality: str,
    batch_size: int | None,
    seq_len: int | None,
) -> tuple[int, int]:
    if features.ndim not in {3, 4}:
        raise ValueError(
            f"{modality} encoder output must have shape [B, T, D] or [B, T, K, D], got {tuple(features.shape)}."
        )
    current_batch, current_seq = int(features.shape[0]), int(features.shape[1])
    if batch_size is not None and (current_batch != batch_size or current_seq != seq_len):
        raise ValueError(
            "Modular sequence modalities must share batch/time dimensions; "
            f"modality '{modality}' produced shape {tuple(features.shape)}, "
            f"expected batch={batch_size}, time={seq_len}."
        )
    return current_batch, current_seq


def _check_projected_features(features: torch.Tensor, modality: str, d_model: int) -> None:
    if features.ndim not in {3, 4} or int(features.shape[-1]) != int(d_model):
        raise ValueError(
            f"{modality} projector output must have shape [B, T, {int(d_model)}] or "
            f"[B, T, K, {int(d_model)}], got {tuple(features.shape)}."
        )


MODELS.register_removed("modular_sequence_model", "Use 'modular_sequence'.")
ENCODERS.register_removed(
    "point_cloud_mlp",
    "Use 'lidar_cnn' for current LiDAR BEV configs; point cloud input is not a current registry surface.",
)
REPRESENTATION_CORES.register_removed(
    "jepa_token_transformer",
    "Use 'token_transformer' or 'token_aware_transformer'.",
)


__all__ = [
    "BeamClassificationHead",
    "AmberLiteMissingModalityTransformerCore",
    "EarlyConcatGRUCore",
    "GpsMLPEncoder",
    "IdentityProjector",
    "LidarCNNEncoder",
    "LinearProjector",
    "MmWaveMLPEncoder",
    "ModularSequenceModel",
    "NextBeamQueryTransformerCore",
    "PilotDualViewCSIEncoder",
    "PointCloudMLPEncoder",
    "RadarCNNEncoder",
    "ResNet18ImageEncoder",
    "SingleGRUCore",
    "SnapshotFrameCore",
    "TokenTransformerCore",
]
