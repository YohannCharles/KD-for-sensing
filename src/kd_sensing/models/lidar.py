import torch
import torch.nn as nn

from kd_sensing.registries import MODELS


class LidarFeatureExtractor(nn.Module):
    def __init__(self, n_feature: int, in_channels: int = 3):
        super().__init__()
        self.in_channels = int(in_channels)
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
        if int(channels) != self.in_channels:
            raise ValueError(
                f"LiDAR input channel count must be {self.in_channels} for this encoder, got {int(channels)}."
            )
        if int(height) <= 0 or int(width) <= 0:
            raise ValueError(f"LiDAR input spatial size must be positive, got {int(height)}x{int(width)}.")
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

MODELS.register_removed(
    "lidar_feature_extractor",
    "Use encoders.lidar.type='lidar_cnn' in model.primary.type='modular_sequence', or import LidarFeatureExtractor directly.",
)
MODELS.register_removed(
    "lidar_strong",
    "Use model.primary.type='modular_sequence' with encoders.lidar.type='lidar_cnn', representation_core.type='single_gru', and heads.beam.type='beam_head'.",
)
MODELS.register_removed(
    "lidar_lightweight",
    "Use configs/lidar/lightweight.yaml with model.primary.type='modular_sequence' and encoders.lidar.type='lidar_cnn'.",
)
MODELS.register_removed("lidar_teacher", "Use configs/lidar/strong.yaml with model.primary.type='modular_sequence'.")
MODELS.register_removed("lidar_student", "Use configs/lidar/lightweight.yaml with model.primary.type='modular_sequence'.")
