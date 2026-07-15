import os
from typing import Any

import torch
import torch.nn as nn

from kd_sensing.modalities import (
    MODALITY_ORDER,
    image_profile_spec,
    normalize_modalities,
    resolve_image_profile,
)
from kd_sensing.models.auxiliary_heads import TemporalAuxiliaryHeads
import kd_sensing.models.amber_full  # noqa: F401
from kd_sensing.models.gps import GpsFeatureExtractor
from kd_sensing.models.image_encoders import ResNet18ImageEncoder
from kd_sensing.models.lidar import LidarFeatureExtractor
from kd_sensing.models.modular_config import (
    normalize_beam_head_config,
    normalize_core_config,
    normalize_encoder_config,
    normalize_projector_config,
    validate_encoder_context_dependencies,
    validate_modality_encoder_profile,
)
from kd_sensing.models.modular_forward import (
    assemble_core_input_stage,
    assemble_model_output_stage,
    coerce_core_availability_mask as _coerce_core_availability_mask,
    collect_forward_inputs,
    component_consumes_reliability_metadata as _component_consumes_reliability_metadata,
    component_training_strategy_metadata as _component_training_strategy_metadata,
    encoder_context_dependencies as _encoder_context_dependencies,
    encoder_context_source as _encoder_context_source,
    run_core_head_stage,
    run_encoder_projector_stage,
)
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

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": "token_aware_transformer",
            "d_model": self.d_model,
            "modality_count": self.modality_count,
            "token_readout_type": "legacy_uniform_mean",
            "token_readout_trainable": False,
            "readout_trainable_params": 0,
            "readout_aggregation": "mean_dim_2_after_transformer",
            "k_tokens": self.modality_count,
        }



@REPRESENTATION_CORES.register("amr_lite")
@REPRESENTATION_CORES.register("amr_lite_masked_gate")
class AmrLiteMaskedGateCore(nn.Module):
    supports_missing_modality_metadata = True

    def __init__(
        self,
        d_model: int,
        modality_count: int,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        dropout: float = 0.0,
        imputation_type: str = "learnable_token",
        **_: Any,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.modality_count = int(modality_count)
        self.hidden_dim = int(hidden_dim or max(self.d_model, 32))
        self.output_dim = int(output_dim or d_model)
        self.imputation_type = str(imputation_type)
        if self.d_model <= 0 or self.modality_count <= 0 or self.output_dim <= 0:
            raise ValueError("amr_lite dimensions must be positive.")
        if self.imputation_type not in {"zero", "mean_feature", "learnable_token"}:
            raise ValueError("amr_lite imputation_type must be zero, mean_feature, or learnable_token.")
        self.imputation_tokens = nn.Parameter(torch.zeros(self.modality_count, self.d_model))
        self.gate = nn.Sequential(
            nn.LayerNorm(self.d_model + 1),
            nn.Linear(self.d_model + 1, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )
        self.output_projection = (
            nn.Identity() if self.output_dim == self.d_model else nn.Linear(self.d_model, self.output_dim)
        )
        self.last_amr_lite_gate_stats: list[dict[str, Any]] = []
        nn.init.trunc_normal_(self.imputation_tokens, std=0.02)

    def forward(self, features: torch.Tensor, *, modality_available: torch.Tensor | None = None) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"amr_lite core expects [B, K, T, D], got {tuple(features.shape)}.")
        batch_size, modality_count, seq_len, d_model = features.shape
        if int(modality_count) != self.modality_count or int(d_model) != self.d_model:
            raise ValueError(
                "amr_lite received incompatible features: "
                f"expected K={self.modality_count}, D={self.d_model}, got {tuple(features.shape)}."
            )
        availability = _coerce_core_availability_mask(
            modality_available,
            features=features,
            core_name="amr_lite",
        ) if modality_available is not None else torch.ones(
            batch_size,
            self.modality_count,
            seq_len,
            dtype=torch.bool,
            device=features.device,
        )
        imputed = self._impute(features, availability)
        gate_input = torch.cat([imputed, availability.to(dtype=imputed.dtype).unsqueeze(-1)], dim=-1)
        weights = torch.softmax(self.gate(gate_input).squeeze(-1), dim=1)
        fused = (weights.unsqueeze(-1) * imputed).sum(dim=1)
        self.last_amr_lite_gate_stats = _gate_stats_by_missing_count(weights, availability)
        return self.output_projection(fused)

    def _impute(self, features: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        if self.imputation_type == "zero":
            replacement = torch.zeros_like(features)
        elif self.imputation_type == "learnable_token":
            replacement = self.imputation_tokens.to(device=features.device, dtype=features.dtype).view(
                1,
                self.modality_count,
                1,
                self.d_model,
            )
        else:
            visible = availability.to(dtype=features.dtype).unsqueeze(-1)
            denom = visible.sum(dim=1, keepdim=True).clamp_min(1.0)
            replacement = (features * visible).sum(dim=1, keepdim=True) / denom
        return torch.where(availability.unsqueeze(-1), features, replacement)

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": "amr_lite",
            "d_model": self.d_model,
            "output_dim": self.output_dim,
            "modality_count": self.modality_count,
            "hidden_dim": self.hidden_dim,
            "imputation_type": self.imputation_type,
            "consumes_missing_modality_metadata": True,
            "gate": "mask_aware_softmax_modality_gate",
        }


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


def _modular_missing_availability_overrides(
    modalities: tuple[str, ...],
    *,
    missing_mask: torch.Tensor | None,
    modality_mask: torch.Tensor | None,
    available_modalities: list[str] | tuple[str, ...] | torch.Tensor | None,
) -> dict[str, torch.Tensor | None]:
    overrides: dict[str, torch.Tensor | None] = {}
    for mask, name in ((missing_mask, "missing_mask"), (modality_mask, "modality_mask")):
        if mask is None:
            continue
        for modality, values in _availability_map_from_tensor(mask, modalities, name=name).items():
            overrides[modality] = _merge_availability_mask(overrides.get(modality), values)
    if torch.is_tensor(available_modalities):
        for modality, values in _availability_map_from_tensor(
            available_modalities,
            modalities,
            name="available_modalities",
        ).items():
            overrides[modality] = _merge_availability_mask(overrides.get(modality), values)
    elif available_modalities is not None:
        for modality, values in _availability_map_from_names(available_modalities, modalities).items():
            overrides[modality] = _merge_availability_mask(overrides.get(modality), values)
    return overrides


def _merge_availability_mask(current: torch.Tensor | None, values: torch.Tensor) -> torch.Tensor:
    if current is None:
        return values
    base = torch.as_tensor(current, dtype=torch.bool)
    other = values.to(device=base.device, dtype=torch.bool)
    return base & other


def _availability_map_from_tensor(
    mask: torch.Tensor,
    modalities: tuple[str, ...],
    *,
    name: str,
) -> dict[str, torch.Tensor]:
    value = torch.as_tensor(mask, dtype=torch.bool)
    if value.ndim == 1:
        if int(value.numel()) != len(modalities):
            raise ValueError(f"{name} length must match model modalities {list(modalities)}, got {int(value.numel())}.")
        value = value.unsqueeze(0)
    elif value.ndim == 2:
        if int(value.shape[1]) != len(modalities):
            raise ValueError(f"{name} shape must be [B,{len(modalities)}], got {tuple(value.shape)}.")
    else:
        raise ValueError(f"{name} must have shape [K] or [B,K], got {tuple(value.shape)}.")
    if not bool(value.any(dim=1).all().item()):
        empty = (~value.any(dim=1)).nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        raise ValueError(f"{name} leaves no available modalities for sample indices {empty}.")
    return {modality: value[:, index] for index, modality in enumerate(modalities)}


def _availability_map_from_names(
    available_modalities: list[str] | tuple[str, ...],
    modalities: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    available = {str(item) for item in available_modalities}
    unknown = sorted(available - set(modalities))
    if unknown:
        raise ValueError(f"available_modalities contains unknown modalities {unknown}; model modalities={list(modalities)}.")
    if not available:
        raise ValueError("available_modalities must keep at least one modality.")
    row = torch.tensor([modality in available for modality in modalities], dtype=torch.bool)
    return {modality: row[index : index + 1] for index, modality in enumerate(modalities)}


def _maybe_log_missing_mask_debug(
    model: nn.Module,
    modalities: tuple[str, ...],
    overrides: dict[str, torch.Tensor | None],
    *,
    metadata: dict[str, Any] | None,
) -> None:
    if not overrides:
        return
    metadata = metadata if isinstance(metadata, dict) else {}
    debug_requested = bool(metadata.get("debug_missing_mask") or metadata.get("debug") or os.environ.get("KD_SENSING_DEBUG_MISSING_MASK"))
    if not debug_requested or bool(getattr(model, "_missing_mask_debug_logged", False)):
        return
    applied = [modality for modality in modalities if overrides.get(modality) is not None]
    mask = torch.stack([torch.as_tensor(overrides[modality], dtype=torch.bool).flatten()[0] for modality in modalities])
    available = [modality for modality, keep in zip(modalities, mask.tolist()) if bool(keep)]
    missing = [modality for modality, keep in zip(modalities, mask.tolist()) if not bool(keep)]
    pattern = str(metadata.get("pattern") or metadata.get("pattern_name") or "samplewise")
    print(f"[MissingMask] pattern={pattern} available={available} missing={missing}")
    print(f"[MissingMask] applied_modalities={applied}")
    setattr(model, "_missing_mask_debug_logged", True)


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
        auxiliary_heads: bool | dict[str, Any] | None = None,
        paper_metadata: dict[str, Any] | None = None,
        **extra: Any,
    ):
        super().__init__()
        retired_options = sorted(
            key for key in extra if key in {"geometry_prior", "logit_fusion", "geometry_prior_fusion", "reranker"}
        )
        if retired_options:
            raise ValueError(f"Unsupported modular_sequence options: {retired_options}.")
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
            encoder_cfg = normalize_encoder_config(
                modality,
                encoder_cfgs.get(modality),
                image_profile=self.image_profile,
                image_channels=self.image_channels,
                feature_size=self.feature_size,
                radar_channels=radar_channels,
                gps_input_size=gps_input_size,
                lidar_channels=lidar_channels,
            )
            validate_modality_encoder_profile(
                modality,
                encoder_cfg,
                image_profile=self.image_profile,
                image_channels=self.image_channels,
            )
            self.encoder_configs[modality] = dict(encoder_cfg)
            encoder = ENCODERS.build(encoder_cfg)
            self.encoders[modality] = encoder
            raw_dim = int(getattr(encoder, "output_dim", encoder_cfg.get("output_dim", self.feature_size)))
            self.encoder_output_dims[modality] = raw_dim
            projector_cfg = normalize_projector_config(projector_cfgs.get(modality), input_dim=raw_dim, d_model=self.d_model)
            self.projector_configs[modality] = dict(projector_cfg)
            self.projectors[modality] = PROJECTORS.build(projector_cfg)

        validate_encoder_context_dependencies(self.encoders, self.modalities)
        core_cfg = normalize_core_config(
            representation_core,
            modalities=self.modalities,
            encoder_configs=self.encoder_configs,
            d_model=self.d_model,
        )
        self.representation_core_config = dict(core_cfg)
        self.representation_core = REPRESENTATION_CORES.build(core_cfg)
        core_output_dim = int(getattr(self.representation_core, "output_dim", self.d_model))
        head_cfgs = dict(heads or {})
        beam_cfg = normalize_beam_head_config(head_cfgs, core_output_dim=core_output_dim, num_classes=self.num_classes)
        self.head_configs = {"beam": dict(beam_cfg)}
        self.heads = nn.ModuleDict({"beam": HEADS.build(beam_cfg)})
        self.auxiliary_heads = TemporalAuxiliaryHeads(
            core_output_dim,
            num_pred=self.num_pred,
            auxiliary_heads=auxiliary_heads,
            dropout=float(beam_cfg.get("dropout", 0.0)),
        )
        self._missing_mask_debug_logged = False

    def forward(
        self,
        image_batch: torch.Tensor | None = None,
        radar_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
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
        temporal_mask: torch.Tensor | None = None,
        modality_temporal_mask: torch.Tensor | None = None,
        missing_mask: torch.Tensor | None = None,
        missing_modality_metadata: dict[str, Any] | None = None,
        available_modalities: list[str] | tuple[str, ...] | torch.Tensor | None = None,
        modality_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        del temporal_mask, modality_temporal_mask
        inputs = collect_forward_inputs(
            image_batch=image_batch,
            radar_batch=radar_batch,
            gps_batch=gps_batch,
            lidar_batch=lidar_batch,
            image_valid_mask=image_valid_mask,
            radar_valid_mask=radar_valid_mask,
            image_observability_score=image_observability_score,
            gps_valid_mask=gps_valid_mask,
            lidar_valid_mask=lidar_valid_mask,
            gps_delay_steps=gps_delay_steps,
            image_dropout_mask=image_dropout_mask,
            radar_dropout_mask=radar_dropout_mask,
            gps_dropout_mask=gps_dropout_mask,
            lidar_dropout_mask=lidar_dropout_mask,
        )
        modality_availability_overrides = _modular_missing_availability_overrides(
            self.modalities,
            missing_mask=missing_mask,
            modality_mask=modality_mask,
            available_modalities=available_modalities,
        )
        _maybe_log_missing_mask_debug(
            self,
            self.modalities,
            modality_availability_overrides,
            metadata=missing_modality_metadata,
        )
        encoder_stage = run_encoder_projector_stage(
            self,
            inputs.raw_inputs,
            inputs.reliability_inputs,
        )
        core_stage = assemble_core_input_stage(
            self,
            encoder_stage.projected,
            modality_valid_inputs=inputs.modality_valid_inputs,
            modality_dropout_inputs=inputs.modality_dropout_inputs,
            modality_availability_overrides=modality_availability_overrides,
        )
        output_features, image_logits = run_core_head_stage(self, core_stage.core_input, core_stage.availability_mask)
        return assemble_model_output_stage(
            self,
            logits=image_logits,
            input_features=core_stage.input_features,
            output_features=output_features,
            core_input=core_stage.core_input,
            availability_mask=core_stage.availability_mask,
            has_token_features=core_stage.has_token_features,
            encoded=encoder_stage.encoded,
            projected=encoder_stage.projected,
            encoder_runtime_metadata=encoder_stage.encoder_runtime_metadata,
            missing_modality_metadata_input=missing_modality_metadata,
        )

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
        core_type = str(self.representation_core_config.get("type", self.representation_core.__class__.__name__))
        token_readout_type = str(core_metadata.get("token_readout_type", "frame_feature"))
        metadata = {
            "type": "modular_sequence",
            "architecture_category": "component_baseline",
            "model_group": "modular_sequence",
            "modalities": list(self.modalities),
            "enabled_modalities": list(self.modalities),
            "d_model": self.d_model,
            "encoders": encoders,
            "projectors": projectors,
            "conditioned_encoders": conditioned,
            "representation_core_type": core_type,
            "representation_core_class": self.representation_core.__class__.__name__,
            "representation_core": core_metadata,
            "token_readout_type": token_readout_type,
            "token_readout_trainable": bool(core_metadata.get("token_readout_trainable", False)),
            "readout_trainable_params": int(core_metadata.get("readout_trainable_params", 0) or 0),
            "k_tokens": core_metadata.get("k_tokens"),
            "heads": heads,
            "loss_mode": "config_resolved",
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


def _gate_stats_by_missing_count(weights: torch.Tensor, availability: torch.Tensor) -> list[dict[str, Any]]:
    missing_counts = (~availability).sum(dim=1)
    rows: list[dict[str, Any]] = []
    for count in sorted(int(value) for value in torch.unique(missing_counts).detach().cpu().tolist()):
        selected = missing_counts == count
        if not selected.any():
            continue
        for modality_index in range(int(weights.shape[1])):
            values = weights[:, modality_index, :][selected]
            rows.append(
                {
                    "pattern": f"missing_count_{count}",
                    "modality": f"modality_{modality_index}",
                    "mean_gate": float(values.detach().mean().cpu().item()),
                    "std_gate": float(values.detach().std(unbiased=False).cpu().item()),
                }
            )
    return rows


__all__ = [
    "BeamClassificationHead",
    "AmberLiteMissingModalityTransformerCore",
    "AmrLiteMaskedGateCore",
    "EarlyConcatGRUCore",
    "GpsMLPEncoder",
    "IdentityProjector",
    "LidarCNNEncoder",
    "LinearProjector",
    "ModularSequenceModel",
    "NextBeamQueryTransformerCore",
    "RadarCNNEncoder",
    "ResNet18ImageEncoder",
    "SingleGRUCore",
    "SnapshotFrameCore",
    "TokenTransformerCore",
]
