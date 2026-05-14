from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.modalities import image_profile_spec, validate_image_encoder_profile
from kd_sensing.registries import MODELS


class ImageFeatureExtractor(nn.Module):
    def __init__(self, n_feature: int, in_channel: int = 3):
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
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
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
        frames = x.reshape(batch_size * seq_length, channels, height, width)
        frame_features = self.cnn_layers(frames)
        frame_features = frame_features * self.channel_attention(frame_features)
        frame_features = frame_features * self.spatial_attention(frame_features)
        frame_features = self.flatten(frame_features)
        frame_features = self.fc_layer(frame_features)
        return frame_features.view(batch_size, seq_length, -1)


@MODELS.register("image_teacher")
class ImageModalityNet(nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_classes: int,
        gru_params: list[int] | tuple[int, int, int],
        image_channels: int = 3,
        image_profile: str | None = None,
        **_: object,
    ):
        super().__init__()
        self.name = "ImageModalityNet"
        _validate_image_profile_channels(self.name, image_profile, image_channels)
        gru_input_size, gru_hidden_size, gru_num_layers = gru_params
        if gru_input_size != feature_size:
            raise ValueError(
                f"gru_input_size ({gru_input_size}) must equal feature_size ({feature_size})"
            )
        self.feature_extraction = ImageFeatureExtractor(feature_size, in_channel=int(image_channels))
        self.GRU = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=0.8 if gru_num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.temporal_attention = nn.Sequential(
            nn.Linear(gru_hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
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

    def forward(self, image_batch: torch.Tensor):
        _, seq_len, _, _, _ = image_batch.size()
        features = self.feature_extraction(image_batch)
        features = self.layer_norm(features)
        seq_out, _ = self.GRU(features)
        attn_weights = F.softmax(self.temporal_attention(seq_out), dim=1)
        context_vector = torch.sum(seq_out * attn_weights, dim=1)
        enhanced_seq_out = seq_out + context_vector.unsqueeze(1).expand(-1, seq_len, -1)
        pred = self.classifier(enhanced_seq_out)
        return pred, features, enhanced_seq_out


@MODELS.register("image_student")
class ImageStudentModalityNet(nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_classes: int,
        gru_params: list[int] | tuple[int, int, int],
        width_multiplier: float = 1.5,
        image_channels: int = 3,
        image_profile: str | None = None,
    ):
        super().__init__()
        self.name = "ImageStudentModalityNet"
        _validate_image_profile_channels(self.name, image_profile, image_channels)
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

        c1 = int(12 * width_multiplier)
        c2 = int(24 * width_multiplier)
        c3 = int(48 * width_multiplier)
        c4 = int(96 * width_multiplier)
        self.image_cnn_layers = nn.Sequential(
            nn.Conv2d(image_channels, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ds_conv_block(c1, c2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ds_conv_block(c2, c3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ds_conv_block(c3, c4),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        reduction = max(c4 // 2, 1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c4, reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, c4, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c4, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )
        self.image_global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fusion_layer = nn.Sequential(
            nn.Linear(c4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feature_size),
        )
        self.GRU = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=0.8 if gru_num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.temporal_attention = nn.Sequential(
            nn.Linear(gru_hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.layer_norm = nn.LayerNorm(gru_input_size)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, image_batch: torch.Tensor):
        batch_size, seq_len, channels, height, width = image_batch.shape
        img = image_batch.reshape(batch_size * seq_len, channels, height, width)
        img_feat = self.image_cnn_layers(img)
        img_feat = img_feat * self.channel_attention(img_feat)
        img_feat = img_feat * self.spatial_attention(img_feat)
        img_pooled = self.image_global_max_pool(img_feat).flatten(1)
        fused_features = self.fusion_layer(img_pooled).view(batch_size, seq_len, -1)
        features = self.layer_norm(fused_features)
        seq_out, _ = self.GRU(features)
        attn_weights = F.softmax(self.temporal_attention(seq_out), dim=1)
        context_vector = torch.sum(seq_out * attn_weights, dim=1)
        enhanced_seq_out = seq_out + context_vector.unsqueeze(1).expand(-1, seq_len, -1)
        pred = self.classifier(enhanced_seq_out)
        return pred, features, enhanced_seq_out


def _validate_image_profile_channels(encoder_name: str, image_profile: str | None, image_channels: int) -> None:
    if image_profile is None:
        return
    spec = image_profile_spec(image_profile)
    validate_image_encoder_profile(
        encoder_name=encoder_name,
        image_profile=image_profile,
        expected_channels=spec.channels,
        actual_channels=image_channels,
    )
