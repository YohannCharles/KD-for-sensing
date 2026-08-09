import torch
import torch.nn as nn


def _dual_radar_branch() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
    )


class RadarDualBranchFeatureExtractor(nn.Module):
    """Encode range-angle and Doppler-angle maps before fusing them."""

    def __init__(self, n_feature: int, in_channels: int = 2):
        super().__init__()
        self.in_channels = int(in_channels)
        if self.in_channels != 2:
            raise ValueError(f"Dual-branch Radar encoder requires exactly 2 channels, got {self.in_channels}.")
        self.ra_branch = _dual_radar_branch()
        self.da_branch = _dual_radar_branch()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fusion = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 32 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_feature),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Radar input must have shape [B, T, C, H, W], got {tuple(x.shape)}.")
        batch_size, seq_length, channels, height, width = x.size()
        if int(channels) != self.in_channels:
            raise ValueError(
                f"Radar input channel count must be {self.in_channels} for this encoder, got {int(channels)}."
            )
        if (int(height), int(width)) != (128, 64):
            raise ValueError(f"Radar input spatial size must be 128x64, got {int(height)}x{int(width)}.")

        frames = x.reshape(batch_size * seq_length, channels, height, width)
        ra_features = self.pool(self.ra_branch(frames[:, 0:1]))
        da_features = self.pool(self.da_branch(frames[:, 1:2]))
        fused = self.fusion(torch.cat([ra_features, da_features], dim=1))
        return fused.view(batch_size, seq_length, -1)



class RadarFeatureExtractor(nn.Module):
    def __init__(self, n_feature: int, in_channels: int = 1):
        super().__init__()
        self.in_channels = int(in_channels)
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
        if x.ndim != 5:
            raise ValueError(f"Radar input must have shape [B, T, C, H, W], got {tuple(x.shape)}.")
        batch_size, seq_length, channels, height, width = x.size()
        if int(channels) != self.in_channels:
            raise ValueError(
                f"Radar input channel count must be {self.in_channels} for this encoder, got {int(channels)}."
            )
        if (int(height), int(width)) != (128, 64):
            raise ValueError(f"Radar input spatial size must be 128x64, got {int(height)}x{int(width)}.")
        frames = x.reshape(batch_size * seq_length, channels, height, width)
        frame_features = self.net(frames)
        frame_features = self.flatten(frame_features)
        frame_features = self.fc_layer(frame_features)
        return frame_features.view(batch_size, seq_length, -1)
