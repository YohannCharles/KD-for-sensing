from __future__ import annotations

import torch
import torch.nn as nn

from kd_sensing.registries import MODELS


def _validate_lidar_gru_params(
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


def _ds_conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
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


@MODELS.register("lidar_feature_extractor")
class LidarFeatureExtractor(nn.Module):
    def __init__(self, n_feature: int, in_channels: int = 3):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(96, 48, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(96, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc_layer = nn.Sequential(
            nn.Linear(96 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_feature),
        )

    def forward(self, lidar_batch: torch.Tensor) -> torch.Tensor:
        if lidar_batch.ndim != 5:
            raise ValueError(f"LiDAR input must have shape [B, T, C, H, W], got {tuple(lidar_batch.shape)}.")
        batch_size, seq_len, channels, height, width = lidar_batch.shape
        lidar = lidar_batch.reshape(batch_size * seq_len, channels, height, width)
        lidar_feat = self.cnn_layers(lidar)
        lidar_feat = lidar_feat * self.channel_attention(lidar_feat)
        lidar_feat = lidar_feat * self.spatial_attention(lidar_feat)
        pooled = torch.cat(
            [
                self.global_avg_pool(lidar_feat).flatten(1),
                self.global_max_pool(lidar_feat).flatten(1),
            ],
            dim=1,
        )
        features = self.fc_layer(pooled)
        return features.view(batch_size, seq_len, -1)


@MODELS.register("lidar_teacher")
class LidarModalityNet(nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_classes: int,
        gru_params: list[int] | tuple[int, int, int],
        lidar_channels: int = 3,
        num_heads: int = 8,
    ):
        super().__init__()
        self.name = "LidarModalityNet"
        gru_input_size, gru_hidden_size, gru_num_layers = _validate_lidar_gru_params(
            feature_size,
            gru_params,
        )
        if num_heads <= 0:
            raise ValueError(f"num_heads ({num_heads}) must be positive.")
        if gru_hidden_size % num_heads != 0:
            raise ValueError(
                f"gru_hidden_size ({gru_hidden_size}) must be divisible by num_heads ({num_heads}) "
                "for lidar_teacher multihead attention."
            )
        self.feature_extraction = LidarFeatureExtractor(feature_size, in_channels=lidar_channels)
        self.layer_norm = nn.LayerNorm(gru_input_size)
        self.GRU = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=0.5 if gru_num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=gru_hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, lidar_batch: torch.Tensor):
        features = self.feature_extraction(lidar_batch)
        features = self.layer_norm(features)
        seq_out, _ = self.GRU(features)
        attn_output, _ = self.multihead_attention(seq_out, seq_out, seq_out)
        enhanced_seq_out = attn_output + seq_out
        pred = self.classifier(enhanced_seq_out)
        return pred, features, enhanced_seq_out


@MODELS.register("lidar_student")
class LidarStudentModalityNet(nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_classes: int,
        gru_params: list[int] | tuple[int, int, int],
        lidar_channels: int = 3,
        width_multiplier: float = 1.0,
    ):
        super().__init__()
        self.name = "LidarStudentModalityNet"
        gru_input_size, gru_hidden_size, gru_num_layers = _validate_lidar_gru_params(
            feature_size,
            gru_params,
        )
        c1 = max(int(12 * width_multiplier), 8)
        c2 = max(int(16 * width_multiplier), 8)
        c3 = max(int(24 * width_multiplier), 8)
        c4 = max(int(96 * width_multiplier), 16)
        self.lidar_cnn_layers = nn.Sequential(
            nn.Conv2d(lidar_channels, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            _ds_conv_block(c1, c2, stride=2),
            _ds_conv_block(c2, c3, stride=2),
            _ds_conv_block(c3, c4, stride=2),
        )
        self.lidar_global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.lidar_global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.feature_projection = nn.Sequential(
            nn.Linear(c4 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, feature_size),
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
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, lidar_batch: torch.Tensor):
        if lidar_batch.ndim != 5:
            raise ValueError(f"LiDAR input must have shape [B, T, C, H, W], got {tuple(lidar_batch.shape)}.")
        batch_size, seq_len, channels, height, width = lidar_batch.shape
        lidar = lidar_batch.reshape(batch_size * seq_len, channels, height, width)
        lidar_feat = self.lidar_cnn_layers(lidar)
        pooled = torch.cat(
            [
                self.lidar_global_avg_pool(lidar_feat).flatten(1),
                self.lidar_global_max_pool(lidar_feat).flatten(1),
            ],
            dim=1,
        )
        projected = self.feature_projection(pooled).view(batch_size, seq_len, -1)
        features = self.layer_norm(projected)
        seq_out, _ = self.GRU(features)
        pred = self.classifier(seq_out)
        return pred, features, seq_out
