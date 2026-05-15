from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from kd_sensing.modalities import MODALITY_ORDER, normalize_modalities
from kd_sensing.models.auxiliary_heads import resolve_auxiliary_heads
from kd_sensing.models.gps import GpsFeatureExtractor
from kd_sensing.models.image import ImageFeatureExtractor
from kd_sensing.models.lidar import LidarFeatureExtractor
from kd_sensing.models.mmwave import MMWAVE_INPUT_SIZE, MmWaveFeatureExtractor
from kd_sensing.models.radar import RadarFeatureExtractor
from kd_sensing.registries import MODELS


@MODELS.register("cls_token_transformer_fusion")
class CLSTokenTransformerFusionNet(nn.Module):
    supports_force_modality_mask = True

    def __init__(
        self,
        *,
        feature_size: int,
        num_classes: int,
        num_pred: int = 3,
        modalities: list[str] | tuple[str, ...] | None = None,
        d_model: int | None = None,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        image_channels: int = 3,
        radar_channels: int = 2,
        gps_input_size: int = 3,
        lidar_channels: int = 3,
        mmwave_input_size: int = MMWAVE_INPUT_SIZE,
        auxiliary_heads: bool | dict[str, Any] | None = None,
        **_: Any,
    ):
        super().__init__()
        self.name = "CLSTokenTransformerFusionNet"
        self.modalities = normalize_modalities(
            tuple(modalities or MODALITY_ORDER),
            context="CLS-token transformer fusion modalities",
        )
        self.modality_count = len(self.modalities)
        self.feature_size = int(feature_size)
        self.d_model = int(d_model or feature_size)
        self.num_classes = int(num_classes)
        self.num_pred = int(num_pred)
        self.horizon = self.num_pred
        self.max_seq_len = int(max_seq_len)
        self.cls_type_id = len(MODALITY_ORDER)

        if self.num_pred <= 0:
            raise ValueError(f"num_pred must be positive, got {num_pred}.")
        if int(num_layers) <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        if int(num_heads) <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if self.d_model % int(num_heads) != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({num_heads}).")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")

        self.encoders = nn.ModuleDict()
        self.feature_projections = nn.ModuleDict()
        for modality in self.modalities:
            self.encoders[modality] = _build_modality_encoder(
                modality,
                self.feature_size,
                image_channels=image_channels,
                radar_channels=radar_channels,
                gps_input_size=gps_input_size,
                lidar_channels=lidar_channels,
                mmwave_input_size=mmwave_input_size,
            )
            self.feature_projections[modality] = (
                nn.Identity() if self.feature_size == self.d_model else nn.Linear(self.feature_size, self.d_model)
            )

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.token_type_embedding = nn.Embedding(len(MODALITY_ORDER) + 1, self.d_model)
        self.time_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(float(dropout))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=max(self.d_model * 4, 64),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.output_norm = nn.LayerNorm(self.d_model)
        self.prediction_head = nn.Linear(self.d_model, self.num_pred * self.num_classes)
        self.auxiliary_heads = resolve_auxiliary_heads(auxiliary_heads)
        self.occlusion_head = (
            nn.Linear(self.d_model, self.num_pred) if self.auxiliary_heads["occlusion"] else None
        )
        self.position_head = (
            nn.Linear(self.d_model, self.num_pred * 2) if self.auxiliary_heads["position"] else None
        )

    def forward(
        self,
        image_batch: torch.Tensor | None = None,
        radar_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
        mmwave_batch: torch.Tensor | None = None,
        force_modality_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | tuple[str, ...]]:
        raw_inputs = {
            "image": image_batch,
            "radar": radar_batch,
            "gps": gps_batch,
            "lidar": lidar_batch,
            "mmwave": mmwave_batch,
        }
        modality_features = []
        batch_size = None
        seq_len = None
        for modality in self.modalities:
            tensor = raw_inputs[modality]
            if tensor is None:
                raise ValueError(
                    f"CLS-token transformer fusion requires '{modality}' input because it is enabled."
                )
            features = self.encoders[modality](tensor)
            features = self.feature_projections[modality](features)
            batch_size, seq_len = _check_temporal_features(features, modality, batch_size, seq_len)
            modality_features.append(features)
        assert batch_size is not None and seq_len is not None
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}.")

        stacked_features = torch.stack(modality_features, dim=1)
        effective_mask = _effective_modality_mask(
            batch_size,
            self.modality_count,
            device=stacked_features.device,
            force_modality_mask=force_modality_mask,
        )
        if torch.any(~effective_mask.any(dim=1)):
            raise ValueError("force_modality_mask leaves no available modalities for at least one sample.")

        embedded_tokens = self._embed_modality_tokens(stacked_features)
        token_padding_mask = ~effective_mask.unsqueeze(-1).expand(batch_size, self.modality_count, seq_len)
        diagnostic_tokens = embedded_tokens.masked_fill(token_padding_mask.unsqueeze(-1), 0.0)
        flat_tokens = _serialize_time_first(embedded_tokens)
        flat_padding_mask = _serialize_mask_time_first(token_padding_mask)
        cls = self.cls_token.expand(batch_size, -1, -1)
        cls_type_ids = torch.full(
            (batch_size, 1),
            self.cls_type_id,
            dtype=torch.long,
            device=stacked_features.device,
        )
        cls = self.input_dropout(self.input_norm(cls + self.token_type_embedding(cls_type_ids)))
        transformer_input = torch.cat([cls, flat_tokens], dim=1)
        cls_padding = torch.zeros(batch_size, 1, dtype=torch.bool, device=flat_padding_mask.device)
        padding_mask = torch.cat([cls_padding, flat_padding_mask], dim=1)
        memory = self.transformer(transformer_input, src_key_padding_mask=padding_mask)
        cls_hidden = self.output_norm(memory[:, 0, :])
        logits = self.prediction_head(cls_hidden).view(batch_size, self.num_pred, self.num_classes)
        input_features = _available_timewise_mean(diagnostic_tokens, effective_mask)
        output_features = cls_hidden.unsqueeze(1).expand(-1, self.num_pred, -1).contiguous()

        output: dict[str, torch.Tensor | tuple[str, ...]] = {
            "logits": logits,
            "input_features": input_features,
            "output_features": output_features,
            "token_features": diagnostic_tokens,
            "modalities": self.modalities,
            "effective_modality_mask": effective_mask,
            "fusion_memory": memory,
            "token_padding_mask": token_padding_mask,
            "serialized_token_padding_mask": flat_padding_mask,
            "cls_features": cls_hidden,
        }
        if self.occlusion_head is not None:
            output["occlusion_logits"] = self.occlusion_head(cls_hidden).view(batch_size, self.num_pred)
        if self.position_head is not None:
            output["position"] = self.position_head(cls_hidden).view(batch_size, self.num_pred, 2)
        return output

    def _embed_modality_tokens(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, modality_count, seq_len, _ = features.shape
        time_ids = torch.arange(seq_len, device=features.device)
        time = self.time_embedding(time_ids).view(1, 1, seq_len, self.d_model)
        type_ids = torch.tensor(
            [MODALITY_ORDER.index(name) for name in self.modalities],
            dtype=torch.long,
            device=features.device,
        )
        token_type = self.token_type_embedding(type_ids).view(1, modality_count, 1, self.d_model)
        tokens = self.input_norm(features + time + token_type)
        return self.input_dropout(tokens)


def _build_modality_encoder(
    modality: str,
    feature_size: int,
    *,
    image_channels: int,
    radar_channels: int,
    gps_input_size: int,
    lidar_channels: int,
    mmwave_input_size: int,
) -> nn.Module:
    if modality == "image":
        return ImageFeatureExtractor(feature_size, image_channels)
    if modality == "radar":
        return RadarFeatureExtractor(feature_size, radar_channels)
    if modality == "gps":
        return GpsFeatureExtractor(feature_size, gps_input_size=gps_input_size)
    if modality == "lidar":
        return LidarFeatureExtractor(feature_size, in_channels=lidar_channels)
    if modality == "mmwave":
        return MmWaveFeatureExtractor(feature_size=feature_size, mmwave_input_size=mmwave_input_size)
    available = ", ".join(MODALITY_ORDER)
    raise ValueError(f"Unknown CLS-token transformer fusion modality '{modality}'. Available modalities: {available}.")


def _check_temporal_features(
    features: torch.Tensor,
    modality: str,
    batch_size: int | None,
    seq_len: int | None,
) -> tuple[int, int]:
    if features.ndim != 3:
        raise ValueError(f"{modality} features must have shape [B, T, D], got {tuple(features.shape)}.")
    current_batch = int(features.shape[0])
    current_seq = int(features.shape[1])
    if batch_size is not None and (batch_size != current_batch or seq_len != current_seq):
        raise ValueError("Enabled CLS-token transformer fusion modalities must share batch and sequence dimensions.")
    return current_batch, current_seq


def _effective_modality_mask(
    batch_size: int,
    modality_count: int,
    *,
    device: torch.device,
    force_modality_mask: torch.Tensor | None,
) -> torch.Tensor:
    mask = torch.ones(batch_size, modality_count, dtype=torch.bool, device=device)
    if force_modality_mask is None:
        return mask
    forced = force_modality_mask.to(device=device, dtype=torch.bool)
    if forced.ndim == 1:
        if forced.shape[0] != modality_count:
            raise ValueError(f"force_modality_mask shape must be [K] or [B, K], got {tuple(forced.shape)}.")
        forced = forced.unsqueeze(0).expand(batch_size, -1)
    if forced.shape != mask.shape:
        raise ValueError(f"force_modality_mask shape must be {tuple(mask.shape)}, got {tuple(forced.shape)}.")
    return mask & forced


def _serialize_time_first(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 4:
        raise ValueError(f"tokens must have shape [B, K, T, D], got {tuple(tokens.shape)}.")
    batch_size, modality_count, seq_len, d_model = tokens.shape
    return tokens.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len * modality_count, d_model)


def _serialize_mask_time_first(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape [B, K, T], got {tuple(mask.shape)}.")
    batch_size, modality_count, seq_len = mask.shape
    return mask.permute(0, 2, 1).contiguous().view(batch_size, seq_len * modality_count)


def _available_timewise_mean(tokens: torch.Tensor, effective_mask: torch.Tensor) -> torch.Tensor:
    valid = effective_mask.to(device=tokens.device, dtype=tokens.dtype).view(tokens.shape[0], tokens.shape[1], 1, 1)
    counts = valid.sum(dim=1).clamp_min(1.0)
    return (tokens * valid).sum(dim=1) / counts


__all__ = [
    "CLSTokenTransformerFusionNet",
]
