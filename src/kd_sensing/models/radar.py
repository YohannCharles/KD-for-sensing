from __future__ import annotations

import torch
import torch.nn as nn

from kd_sensing.registries import MODELS


@MODELS.register("radar_feature_extractor")
class RadarFeatureExtractor(nn.Module):
    def __init__(self, n_feature: int, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 8 * 4, 512),
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
        frames = x.reshape(batch_size * seq_length, channels, height, width)
        frame_features = self.net(frames)
        frame_features = self.flatten(frame_features)
        frame_features = self.fc_layer(frame_features)
        return frame_features.view(batch_size, seq_length, -1)


@MODELS.register("radar_student")
class RadarStudentNet(nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_classes: int,
        gru_params: list[int] | tuple[int, int, int],
        radar_channels: int = 2,
    ):
        super().__init__()
        self.name = "RadarStudentNet"
        if len(gru_params) != 3:
            raise ValueError("gru_params must contain [input_size, hidden_size, num_layers].")
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
        self.feature_projection = nn.Sequential(
            nn.Linear(96 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, feature_size),
        )
        self.layer_norm = nn.LayerNorm(gru_input_size)
        self.GRU = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=0.5 if gru_num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, radar_batch: torch.Tensor):
        batch_size, seq_len, channels, height, width = radar_batch.shape
        radar = radar_batch.reshape(batch_size * seq_len, channels, height, width)
        radar_feat = self.radar_cnn_layers(radar)
        radar_avg = self.radar_global_avg_pool(radar_feat).flatten(1)
        radar_max = self.radar_global_max_pool(radar_feat).flatten(1)
        pooled = torch.cat([radar_avg, radar_max], dim=1)
        projected = self.feature_projection(pooled).view(batch_size, seq_len, -1)
        features = self.layer_norm(projected)
        seq_out, _ = self.GRU(features)
        pred = self.classifier(seq_out)
        return pred, features, seq_out


@MODELS.register("radar_teacher")
class RadarTeacherNet(nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_classes: int,
        gru_params: list[int] | tuple[int, int, int],
        radar_channels: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        self.name = "RadarTeacherNet"
        if len(gru_params) != 3:
            raise ValueError("gru_params must contain [input_size, hidden_size, num_layers].")
        gru_input_size, gru_hidden_size, gru_num_layers = gru_params
        if gru_input_size != feature_size:
            raise ValueError(
                f"gru_input_size ({gru_input_size}) must equal feature_size ({feature_size})"
            )
        if num_heads <= 0:
            raise ValueError(f"num_heads ({num_heads}) must be positive.")
        if gru_hidden_size % num_heads != 0:
            raise ValueError(
                f"gru_hidden_size ({gru_hidden_size}) must be divisible by num_heads ({num_heads}) "
                "for radar_teacher multihead attention."
            )

        self.radar_feature_extractor = RadarFeatureExtractor(feature_size, radar_channels)
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

    def forward(self, radar_batch: torch.Tensor):
        features = self.radar_feature_extractor(radar_batch)
        features = self.layer_norm(features)
        seq_out, _ = self.GRU(features)
        attn_output, _ = self.multihead_attention(seq_out, seq_out, seq_out)
        enhanced_seq_out = attn_output + seq_out
        pred = self.classifier(enhanced_seq_out)
        return pred, features, enhanced_seq_out
