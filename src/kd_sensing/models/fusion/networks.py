from __future__ import annotations

import torch
import torch.nn as nn

from kd_sensing.models.gps import GpsFeatureExtractor
from kd_sensing.models.lidar import LidarFeatureExtractor
from kd_sensing.models.mmwave import MMWAVE_INPUT_SIZE, MmWaveFeatureExtractor
from kd_sensing.models.radar import RadarFeatureExtractor
from kd_sensing.modalities import MODALITY_ORDER, normalize_modalities
from kd_sensing.registries import MODELS


VALID_FUSION_MODALITIES = MODALITY_ORDER


def _normalize_modalities(modalities: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    selected = ("image", "radar") if modalities is None else modalities
    try:
        return normalize_modalities(selected, context="fusion modalities")
    except ValueError as exc:
        message = str(exc)
        if "Unknown modalities" in message:
            raise ValueError(message.replace("Unknown modalities in fusion modalities", "Unknown fusion modalities")) from exc
        raise


def _require_tensor(tensor: torch.Tensor | None, modality: str) -> torch.Tensor:
    if tensor is None:
        raise ValueError(f"Fusion model requires '{modality}' input because it is enabled in modalities.")
    return tensor


class FusionImageFeatureExtractor(nn.Module):
    def __init__(self, n_feature: int, in_channel: int = 1):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channel, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_feature),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, channels, height, width = x.size()
        spatial_features = []
        for t in range(seq_length):
            frame_features = self.cnn_layers(x[:, t, :, :, :])
            frame_features = self.flatten(frame_features)
            spatial_features.append(self.fc_layer(frame_features))
        return torch.stack(spatial_features, dim=1)


@MODELS.register("fusion_teacher")
class FusionModalityNet(nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_classes: int,
        gru_params: list[int] | tuple[int, int, int],
        image_channels: int = 1,
        radar_channels: int = 2,
        gps_input_size: int = 3,
        lidar_channels: int = 3,
        mmwave_input_size: int = MMWAVE_INPUT_SIZE,
        num_heads: int = 8,
        modalities: list[str] | tuple[str, ...] | None = None,
    ):
        super().__init__()
        self.name = "FusionModalityNet"
        self.modalities = _normalize_modalities(modalities)
        gru_input_size, gru_hidden_size, gru_num_layers = gru_params
        if gru_input_size != feature_size:
            raise ValueError(
                f"gru_input_size ({gru_input_size}) must equal feature_size ({feature_size})"
            )
        if "image" in self.modalities:
            self.image_feature_extractor = FusionImageFeatureExtractor(feature_size, image_channels)
        if "radar" in self.modalities:
            self.radar_feature_extractor = RadarFeatureExtractor(feature_size, radar_channels)
        if "gps" in self.modalities:
            self.gps_feature_extractor = GpsFeatureExtractor(feature_size, gps_input_size)
        if "lidar" in self.modalities:
            self.lidar_feature_extractor = LidarFeatureExtractor(feature_size, lidar_channels)
        if "mmwave" in self.modalities:
            self.mmwave_feature_extractor = MmWaveFeatureExtractor(
                feature_size=feature_size,
                mmwave_input_size=mmwave_input_size,
            )
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_size * len(self.modalities), feature_size),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.GRU = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=0.5,
            batch_first=True,
        )
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=gru_hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(gru_input_size)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        image_batch: torch.Tensor | None = None,
        radar_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
        mmwave_batch: torch.Tensor | None = None,
    ):
        modality_features = []
        batch_size = None
        seq_len = None
        if "image" in self.modalities:
            image_features = self.image_feature_extractor(_require_tensor(image_batch, "image"))
            batch_size, seq_len = _check_temporal_features(
                image_features,
                "image",
                batch_size,
                seq_len,
            )
            modality_features.append(image_features)
        if "radar" in self.modalities:
            radar_features = self.radar_feature_extractor(_require_tensor(radar_batch, "radar"))
            batch_size, seq_len = _check_temporal_features(
                radar_features,
                "radar",
                batch_size,
                seq_len,
            )
            modality_features.append(radar_features)
        if "gps" in self.modalities:
            gps_features = self.gps_feature_extractor(_require_tensor(gps_batch, "gps"))
            batch_size, seq_len = _check_temporal_features(
                gps_features,
                "gps",
                batch_size,
                seq_len,
            )
            modality_features.append(gps_features)
        if "lidar" in self.modalities:
            lidar_features = self.lidar_feature_extractor(_require_tensor(lidar_batch, "lidar"))
            batch_size, seq_len = _check_temporal_features(
                lidar_features,
                "lidar",
                batch_size,
                seq_len,
            )
            modality_features.append(lidar_features)
        if "mmwave" in self.modalities:
            mmwave_features = self.mmwave_feature_extractor(_require_tensor(mmwave_batch, "mmwave"))
            batch_size, seq_len = _check_temporal_features(
                mmwave_features,
                "mmwave",
                batch_size,
                seq_len,
            )
            modality_features.append(mmwave_features)
        fused_features = torch.cat(modality_features, dim=2)
        features = self.fusion_layer(fused_features)
        features = self.layer_norm(features)
        seq_out, _ = self.GRU(features)
        attn_output, _ = self.multihead_attention(seq_out, seq_out, seq_out)
        enhanced_seq_out = attn_output + seq_out
        pred = self.classifier(enhanced_seq_out)
        return pred, features, enhanced_seq_out


@MODELS.register("fusion_student")
class StudentModalityNet(nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_classes: int,
        gru_params: list[int] | tuple[int, int, int],
        image_channels: int = 1,
        radar_channels: int = 2,
        gps_input_size: int = 3,
        lidar_channels: int = 3,
        mmwave_input_size: int = MMWAVE_INPUT_SIZE,
        modalities: list[str] | tuple[str, ...] | None = None,
    ):
        super().__init__()
        self.name = "StudentModalityNet"
        self.modalities = _normalize_modalities(modalities)
        gru_input_size, gru_hidden_size, gru_num_layers = gru_params
        if gru_input_size != feature_size:
            raise ValueError(
                f"gru_input_size ({gru_input_size}) must equal feature_size ({feature_size})"
            )

        def ds_conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        branch_dims = []
        if "image" in self.modalities:
            self.image_cnn_layers = nn.Sequential(
                nn.Conv2d(image_channels, 12, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(12),
                nn.ReLU(inplace=True),
                ds_conv_block(12, 16, stride=2),
                ds_conv_block(16, 24, stride=2),
                ds_conv_block(24, 40, stride=2),
                ds_conv_block(40, 96, stride=2),
            )
            self.image_global_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.image_global_max_pool = nn.AdaptiveMaxPool2d(1)
            branch_dims.append(96 * 2)
        if "radar" in self.modalities:
            self.radar_cnn_layers = nn.Sequential(
                nn.Conv2d(radar_channels, 12, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(12),
                nn.ReLU(inplace=True),
                ds_conv_block(12, 16, stride=2),
                ds_conv_block(16, 24, stride=2),
                ds_conv_block(24, 96, stride=2),
            )
            self.radar_global_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.radar_global_max_pool = nn.AdaptiveMaxPool2d(1)
            branch_dims.append(96 * 2)
        if "gps" in self.modalities:
            self.gps_projection = nn.Sequential(
                nn.Linear(gps_input_size, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(64, 96),
                nn.ReLU(inplace=True),
            )
            branch_dims.append(96)
        if "lidar" in self.modalities:
            self.lidar_cnn_layers = nn.Sequential(
                nn.Conv2d(lidar_channels, 12, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(12),
                nn.ReLU(inplace=True),
                ds_conv_block(12, 16, stride=2),
                ds_conv_block(16, 24, stride=2),
                ds_conv_block(24, 96, stride=2),
            )
            self.lidar_global_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.lidar_global_max_pool = nn.AdaptiveMaxPool2d(1)
            branch_dims.append(96 * 2)
        if "mmwave" in self.modalities:
            if int(mmwave_input_size) != MMWAVE_INPUT_SIZE:
                raise ValueError(f"mmwave_input_size ({mmwave_input_size}) must equal {MMWAVE_INPUT_SIZE}.")
            self.mmwave_input_size = int(mmwave_input_size)
            self.mmwave_projection = nn.Sequential(
                nn.Linear(self.mmwave_input_size, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(64, 96),
                nn.ReLU(inplace=True),
            )
            branch_dims.append(96)
        self.fusion_layer = nn.Sequential(
            nn.Linear(sum(branch_dims), 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, feature_size),
        )
        self.GRU = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=0.5 if gru_num_layers > 1 else 0,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(gru_input_size)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        image_batch: torch.Tensor | None = None,
        radar_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
        mmwave_batch: torch.Tensor | None = None,
        beam=None,
    ):
        batch_size = None
        seq_len = None
        pooled_features = []

        if "image" in self.modalities:
            image_batch = _require_tensor(image_batch, "image")
            batch_size, seq_len, channels, height, width = _check_sequence_tensor(
                image_batch,
                "image",
                batch_size,
                seq_len,
                expected_ndim=5,
            )
            img = image_batch.reshape(batch_size * seq_len, channels, height, width)
            img_feat = self.image_cnn_layers(img)
            pooled_features.extend(
                [
                    self.image_global_avg_pool(img_feat).flatten(1),
                    self.image_global_max_pool(img_feat).flatten(1),
                ]
            )
        if "radar" in self.modalities:
            radar_batch = _require_tensor(radar_batch, "radar")
            batch_size, seq_len, radar_channels, radar_height, radar_width = _check_sequence_tensor(
                radar_batch,
                "radar",
                batch_size,
                seq_len,
                expected_ndim=5,
            )
            rad = radar_batch.reshape(batch_size * seq_len, radar_channels, radar_height, radar_width)
            rad_feat = self.radar_cnn_layers(rad)
            pooled_features.extend(
                [
                    self.radar_global_avg_pool(rad_feat).flatten(1),
                    self.radar_global_max_pool(rad_feat).flatten(1),
                ]
            )
        if "gps" in self.modalities:
            gps_batch = _require_tensor(gps_batch, "gps")
            batch_size, seq_len, gps_dim = _check_sequence_tensor(
                gps_batch,
                "gps",
                batch_size,
                seq_len,
                expected_ndim=3,
            )
            gps_flat = gps_batch.reshape(batch_size * seq_len, gps_dim)
            pooled_features.append(self.gps_projection(gps_flat))
        if "lidar" in self.modalities:
            lidar_batch = _require_tensor(lidar_batch, "lidar")
            batch_size, seq_len, lidar_channels, lidar_height, lidar_width = _check_sequence_tensor(
                lidar_batch,
                "lidar",
                batch_size,
                seq_len,
                expected_ndim=5,
            )
            lidar = lidar_batch.reshape(batch_size * seq_len, lidar_channels, lidar_height, lidar_width)
            lidar_feat = self.lidar_cnn_layers(lidar)
            pooled_features.extend(
                [
                    self.lidar_global_avg_pool(lidar_feat).flatten(1),
                    self.lidar_global_max_pool(lidar_feat).flatten(1),
                ]
            )
        if "mmwave" in self.modalities:
            mmwave_batch = _require_tensor(mmwave_batch, "mmwave")
            batch_size, seq_len, mmwave_dim = _check_sequence_tensor(
                mmwave_batch,
                "mmwave",
                batch_size,
                seq_len,
                expected_ndim=3,
            )
            if int(mmwave_dim) != self.mmwave_input_size:
                raise ValueError(
                    f"mmWave input feature_dim ({mmwave_dim}) must equal mmwave_input_size "
                    f"({self.mmwave_input_size})."
                )
            mmwave_flat = mmwave_batch.reshape(batch_size * seq_len, mmwave_dim)
            pooled_features.append(self.mmwave_projection(mmwave_flat))

        fused = torch.cat(pooled_features, dim=1)
        fused_features = self.fusion_layer(fused).view(batch_size, seq_len, -1)
        features = self.layer_norm(fused_features)
        seq_out, _ = self.GRU(features)
        pred = self.classifier(seq_out)
        return pred, features, seq_out


def _check_temporal_features(
    features: torch.Tensor,
    modality: str,
    batch_size: int | None,
    seq_len: int | None,
) -> tuple[int, int]:
    if features.ndim != 3:
        raise ValueError(f"{modality} features must have shape [B, T, D], got {tuple(features.shape)}.")
    current_batch, current_seq = int(features.shape[0]), int(features.shape[1])
    if batch_size is not None and (current_batch != batch_size or current_seq != seq_len):
        raise ValueError("Enabled fusion modalities must share batch and sequence dimensions.")
    return current_batch, current_seq


def _check_sequence_tensor(
    tensor: torch.Tensor,
    modality: str,
    batch_size: int | None,
    seq_len: int | None,
    *,
    expected_ndim: int,
) -> tuple[int, ...]:
    if tensor.ndim != expected_ndim:
        raise ValueError(f"{modality} input must have {expected_ndim} dimensions, got {tuple(tensor.shape)}.")
    current_batch, current_seq = int(tensor.shape[0]), int(tensor.shape[1])
    if batch_size is not None and (current_batch != batch_size or current_seq != seq_len):
        raise ValueError("Enabled fusion modalities must share batch and sequence dimensions.")
    return tuple(int(dim) for dim in tensor.shape)

