from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class SharedTemporalTransformer(nn.Module):
    """Apply one temporal encoder independently to every sensing modality."""

    def __init__(
        self,
        *,
        d_model: int = 64,
        num_modalities: int = 4,
        seq_length: int = 5,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        norm_first: bool = True,
        causal: bool = False,
        adapter_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_modalities = int(num_modalities)
        self.seq_length = int(seq_length)
        if min(self.d_model, self.num_modalities, self.seq_length, int(num_layers), int(num_heads)) <= 0:
            raise ValueError("Temporal Transformer dimensions must be positive.")
        if self.d_model % int(num_heads):
            raise ValueError("Temporal Transformer d_model must be divisible by num_heads.")
        if int(dim_feedforward) <= 0 or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dim_feedforward must be positive and dropout must be in [0, 1).")
        if bool(causal):
            raise ValueError("PCPF-T uses all five historical frames and does not support causal attention.")

        self.input_norms = nn.ModuleList(nn.LayerNorm(self.d_model) for _ in range(self.num_modalities))
        self.input_adapters = nn.ModuleList(
            nn.Linear(self.d_model, self.d_model) if adapter_enabled else nn.Identity()
            for _ in range(self.num_modalities)
        )
        self.time_embedding = nn.Parameter(torch.zeros(self.seq_length, self.d_model))
        self.modality_embedding = nn.Parameter(torch.zeros(self.num_modalities, self.d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(num_layers),
            norm=nn.LayerNorm(self.d_model),
            enable_nested_tensor=False,
        )
        nn.init.normal_(self.time_embedding, std=0.02)
        nn.init.normal_(self.modality_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    @property
    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def forward(self, features: torch.Tensor, temporal_mask: torch.Tensor) -> dict[str, Any]:
        expected = (self.seq_length, self.num_modalities, self.d_model)
        if features.ndim != 4 or tuple(features.shape[1:]) != expected:
            raise ValueError(f"features must have shape [B,{expected[0]},{expected[1]},{expected[2]}].")
        mask = torch.as_tensor(temporal_mask, device=features.device, dtype=torch.bool)
        if tuple(mask.shape) != tuple(features.shape[:3]):
            raise ValueError("temporal_mask must have shape [B,T,M] matching features.")

        adapted = torch.stack(
            [
                self.input_adapters[index](self.input_norms[index](features[:, :, index]))
                for index in range(self.num_modalities)
            ],
            dim=2,
        )
        adapted = (
            adapted
            + self.time_embedding.view(1, self.seq_length, 1, self.d_model)
            + self.modality_embedding.view(1, 1, self.num_modalities, self.d_model)
        )
        batch_size = int(features.shape[0])
        flattened = adapted.permute(0, 2, 1, 3).reshape(
            batch_size * self.num_modalities, self.seq_length, self.d_model
        )
        modality_for_row = self.modality_embedding.unsqueeze(0).expand(batch_size, -1, -1).reshape(
            batch_size * self.num_modalities, self.d_model
        )
        cls = self.cls_token.expand(batch_size * self.num_modalities, -1, -1)
        cls = cls + modality_for_row.unsqueeze(1)
        transformer_input = torch.cat([cls, flattened], dim=1)

        frame_padding = ~mask.permute(0, 2, 1).reshape(batch_size * self.num_modalities, self.seq_length)
        cls_padding = torch.zeros(
            batch_size * self.num_modalities,
            1,
            device=features.device,
            dtype=torch.bool,
        )
        encoded = self.encoder(
            transformer_input,
            src_key_padding_mask=torch.cat([cls_padding, frame_padding], dim=1),
        )
        available = mask.any(dim=1)
        available_float = available.to(dtype=encoded.dtype)
        cls_features = encoded[:, 0].reshape(batch_size, self.num_modalities, self.d_model)
        cls_features = cls_features * available_float.unsqueeze(-1)
        frame_features = encoded[:, 1:].reshape(
            batch_size, self.num_modalities, self.seq_length, self.d_model
        ).permute(0, 2, 1, 3)
        frame_features = frame_features * mask.unsqueeze(-1).to(dtype=encoded.dtype)
        return {
            "temporal_token_features": frame_features,
            "temporal_cls_features": cls_features,
            "temporal_attention_valid_fraction": mask.to(dtype=torch.float32).mean(dim=1),
            "available_modalities": available,
            "temporal_pooling_type": "shared_temporal_transformer",
            "temporal_pooling_param_count": self.parameter_count,
        }


__all__ = ["SharedTemporalTransformer"]
