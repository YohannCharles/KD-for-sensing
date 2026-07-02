import torch
import torch.nn as nn

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
