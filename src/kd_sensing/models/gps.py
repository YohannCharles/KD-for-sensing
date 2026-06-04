from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.engine.gps_coarse_anchor import GpsCoarseAnchorConfig, GpsCoarseHead
from kd_sensing.models.auxiliary_heads import (
    TemporalAuxiliaryHeads,
    temporal_output_with_optional_auxiliary,
)
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


def _validate_gps_gru_params(
    feature_size: int,
    gru_params: list[int] | tuple[int, int, int],
) -> tuple[int, int, int]:
    if len(gru_params) != 3:
        raise ValueError("gru_params must contain [input_size, hidden_size, num_layers].")
    gru_input_size, gru_hidden_size, gru_num_layers = gru_params
    if gru_input_size != feature_size:
        raise ValueError(
            f"gru_input_size ({gru_input_size}) must equal feature_size ({feature_size})"
        )
    return int(gru_input_size), int(gru_hidden_size), int(gru_num_layers)


@MODELS.register("gps_strong")
class GpsModalityNet(nn.Module):
    def __init__(
        self,
        gps_input_size: int,
        feature_size: int,
        num_classes: int,
        gru_params: list[int] | tuple[int, int, int],
        num_pred: int = 3,
        auxiliary_heads: bool | dict | None = None,
        coarse_anchor: bool | dict | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.name = "GpsModalityNet"
        gru_input_size, gru_hidden_size, gru_num_layers = _validate_gps_gru_params(
            feature_size,
            gru_params,
        )
        self.feature_extraction = GpsFeatureExtractor(
            feature_size,
            gps_input_size=gps_input_size,
            hidden_size=64,
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
        self.auxiliary_heads = TemporalAuxiliaryHeads(
            gru_hidden_size,
            num_pred=num_pred,
            auxiliary_heads=auxiliary_heads,
            dropout=dropout,
        )
        self.coarse_anchor_cfg = _resolve_coarse_anchor_config(
            coarse_anchor,
            num_classes=num_classes,
            num_pred=num_pred,
            dropout=dropout,
        )
        self.coarse_anchor_head = (
            GpsCoarseHead(
                gru_hidden_size,
                num_classes=self.coarse_anchor_cfg.num_classes,
                group_size=self.coarse_anchor_cfg.group_size,
                hidden_size=self.coarse_anchor_cfg.hidden_size,
                dropout=self.coarse_anchor_cfg.dropout,
                beam_auxiliary=self.coarse_anchor_cfg.beam_auxiliary,
            )
            if self.coarse_anchor_cfg.enabled
            else None
        )

    def forward(self, gps_batch: torch.Tensor):
        _, seq_len, _ = gps_batch.shape
        features = self.feature_extraction(gps_batch)
        features = self.layer_norm(features)
        seq_out, _ = self.GRU(features)
        attn_weights = F.softmax(self.temporal_attention(seq_out), dim=1)
        context_vector = torch.sum(seq_out * attn_weights, dim=1)
        enhanced_seq_out = seq_out + context_vector.unsqueeze(1).expand(-1, seq_len, -1)
        pred = self.classifier(enhanced_seq_out)
        if self.coarse_anchor_head is not None:
            anchor = self.coarse_anchor_head(enhanced_seq_out)
            return {
                "logits": pred,
                "input_features": features,
                "output_features": enhanced_seq_out,
                **anchor.to_model_output(),
                "coarse_anchor": {
                    "enabled": True,
                    "anchor_source": "gps_neural_coarse",
                    "num_classes": self.coarse_anchor_cfg.num_classes,
                    "group_size": self.coarse_anchor_cfg.group_size,
                    "loss_weight": self.coarse_anchor_cfg.loss_weight,
                    "beam_auxiliary_weight": self.coarse_anchor_cfg.beam_auxiliary_weight,
                },
            }
        return temporal_output_with_optional_auxiliary(
            logits=pred,
            input_features=features,
            output_features=enhanced_seq_out,
            auxiliary_heads=self.auxiliary_heads,
        )


@MODELS.register("gps_lightweight")
class GpsStudentModalityNet(nn.Module):
    def __init__(
        self,
        gps_input_size: int,
        feature_size: int,
        num_classes: int,
        gru_params: list[int] | tuple[int, int, int],
        num_pred: int = 3,
        auxiliary_heads: bool | dict | None = None,
        coarse_anchor: bool | dict | None = None,
        width_multiplier: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.name = "GpsStudentModalityNet"
        gru_input_size, gru_hidden_size, gru_num_layers = _validate_gps_gru_params(
            feature_size,
            gru_params,
        )
        hidden_size = max(int(32 * width_multiplier), 8)
        self.feature_extraction = GpsFeatureExtractor(
            feature_size,
            gps_input_size=gps_input_size,
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
        self.auxiliary_heads = TemporalAuxiliaryHeads(
            gru_hidden_size,
            num_pred=num_pred,
            auxiliary_heads=auxiliary_heads,
            dropout=dropout,
        )
        self.coarse_anchor_cfg = _resolve_coarse_anchor_config(
            coarse_anchor,
            num_classes=num_classes,
            num_pred=num_pred,
            dropout=dropout,
        )
        self.coarse_anchor_head = (
            GpsCoarseHead(
                gru_hidden_size,
                num_classes=self.coarse_anchor_cfg.num_classes,
                group_size=self.coarse_anchor_cfg.group_size,
                hidden_size=self.coarse_anchor_cfg.hidden_size,
                dropout=self.coarse_anchor_cfg.dropout,
                beam_auxiliary=self.coarse_anchor_cfg.beam_auxiliary,
            )
            if self.coarse_anchor_cfg.enabled
            else None
        )

    def forward(self, gps_batch: torch.Tensor):
        features = self.feature_extraction(gps_batch)
        features = self.layer_norm(features)
        seq_out, _ = self.GRU(features)
        pred = self.classifier(seq_out)
        if self.coarse_anchor_head is not None:
            anchor = self.coarse_anchor_head(seq_out)
            return {
                "logits": pred,
                "input_features": features,
                "output_features": seq_out,
                **anchor.to_model_output(),
                "coarse_anchor": {
                    "enabled": True,
                    "anchor_source": "gps_neural_coarse",
                    "num_classes": self.coarse_anchor_cfg.num_classes,
                    "group_size": self.coarse_anchor_cfg.group_size,
                    "loss_weight": self.coarse_anchor_cfg.loss_weight,
                    "beam_auxiliary_weight": self.coarse_anchor_cfg.beam_auxiliary_weight,
                },
            }
        return temporal_output_with_optional_auxiliary(
            logits=pred,
            input_features=features,
            output_features=seq_out,
            auxiliary_heads=self.auxiliary_heads,
        )


def _resolve_coarse_anchor_config(
    coarse_anchor: bool | dict | None,
    *,
    num_classes: int,
    num_pred: int,
    dropout: float,
) -> GpsCoarseAnchorConfig:
    if not coarse_anchor:
        return GpsCoarseAnchorConfig(enabled=False, anchor_source="gps_neural_coarse", num_classes=num_classes)
    payload = dict(coarse_anchor) if isinstance(coarse_anchor, dict) else {}
    payload.setdefault("enabled", True)
    payload.setdefault("anchor_source", "gps_neural_coarse")
    payload.setdefault("num_classes", num_classes)
    payload.setdefault("horizon", num_pred)
    payload.setdefault("dropout", dropout)
    return GpsCoarseAnchorConfig.from_mapping(payload)


MODELS.register_removed("gps_teacher", "Use 'gps_strong'.")
MODELS.register_removed("gps_student", "Use 'gps_lightweight'.")

GpsStrongModalityNet = GpsModalityNet
GpsLightweightModalityNet = GpsStudentModalityNet
