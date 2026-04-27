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
        batch_size, seq_length, _, _, _ = x.size()
        spatial_features = []
        for t in range(seq_length):
            frame_features = self.net(x[:, t, :, :, :])
            frame_features = self.flatten(frame_features)
            spatial_features.append(self.fc_layer(frame_features))
        return torch.stack(spatial_features, dim=1)


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
        num_heads: int = 8,
    ):
        super().__init__()
        self.name = "FusionModalityNet"
        gru_input_size, gru_hidden_size, gru_num_layers = gru_params
        if gru_input_size != feature_size:
            raise ValueError(
                f"gru_input_size ({gru_input_size}) must equal feature_size ({feature_size})"
            )
        self.image_feature_extractor = FusionImageFeatureExtractor(feature_size, image_channels)
        self.radar_feature_extractor = RadarFeatureExtractor(feature_size, radar_channels)
        self.fusion_layer = nn.Sequential(
            nn.Linear(64 + 64, feature_size),
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

    def forward(self, image_batch: torch.Tensor, radar_batch: torch.Tensor):
        image_features = self.image_feature_extractor(image_batch)
        radar_features = self.radar_feature_extractor(radar_batch)
        fused_features = torch.cat([image_features, radar_features], dim=2)
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
    ):
        super().__init__()
        self.name = "StudentModalityNet"
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

        self.image_cnn_layers = nn.Sequential(
            nn.Conv2d(image_channels, 12, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            ds_conv_block(12, 16, stride=2),
            ds_conv_block(16, 24, stride=2),
            ds_conv_block(24, 40, stride=2),
            ds_conv_block(40, 96, stride=2),
        )
        self.radar_cnn_layers = nn.Sequential(
            nn.Conv2d(radar_channels, 12, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            ds_conv_block(12, 16, stride=2),
            ds_conv_block(16, 24, stride=2),
            ds_conv_block(24, 96, stride=2),
        )
        self.image_global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.image_global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.radar_global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.radar_global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fusion_layer = nn.Sequential(
            nn.Linear(96 * 4, 128),
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

    def forward(self, image_batch: torch.Tensor, radar_batch: torch.Tensor, beam=None):
        batch_size, seq_len, channels, height, width = image_batch.shape
        radar_batch_size, radar_seq_len, radar_channels, radar_height, radar_width = radar_batch.shape
        if batch_size != radar_batch_size or seq_len != radar_seq_len:
            raise ValueError("Image and radar batches must share batch and sequence dimensions.")
        img = image_batch.reshape(batch_size * seq_len, channels, height, width)
        rad = radar_batch.reshape(batch_size * seq_len, radar_channels, radar_height, radar_width)
        img_feat = self.image_cnn_layers(img)
        rad_feat = self.radar_cnn_layers(rad)
        img_avg = self.image_global_avg_pool(img_feat).flatten(1)
        img_max = self.image_global_max_pool(img_feat).flatten(1)
        rad_avg = self.radar_global_avg_pool(rad_feat).flatten(1)
        rad_max = self.radar_global_max_pool(rad_feat).flatten(1)
        fused = torch.cat([img_avg, img_max, rad_avg, rad_max], dim=1)
        fused_features = self.fusion_layer(fused).view(batch_size, seq_len, -1)
        features = self.layer_norm(fused_features)
        seq_out, _ = self.GRU(features)
        pred = self.classifier(seq_out)
        return pred, features, seq_out

