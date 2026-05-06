from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def validate_sequence_image_tensor(tensor: torch.Tensor, modality: str) -> tuple[int, int, int, int, int]:
    if tensor.ndim != 5:
        raise ValueError(f"{modality} input must have shape [B, T, C, H, W], got {tuple(tensor.shape)}.")
    batch_size, seq_len, channels, height, width = (int(dim) for dim in tensor.shape)
    if channels <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"{modality} input must have positive C/H/W dimensions, got {tuple(tensor.shape)}.")
    return batch_size, seq_len, channels, height, width


def flatten_sequence_frames(tensor: torch.Tensor, modality: str) -> tuple[torch.Tensor, int, int]:
    batch_size, seq_len, channels, height, width = validate_sequence_image_tensor(tensor, modality)
    return tensor.reshape(batch_size * seq_len, channels, height, width), batch_size, seq_len


def restore_sequence_features(features: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"Frame features must have shape [B*T, D], got {tuple(features.shape)}.")
    return features.view(batch_size, seq_len, -1)


def build_resnet18_feature_extractor(
    *,
    in_channels: int = 3,
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise ModuleNotFoundError("M2BeamLLM ResNet encoders require torchvision to be installed.") from exc

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    if int(in_channels) != 3:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            int(in_channels),
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                if int(in_channels) == 1:
                    model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                else:
                    repeats = (int(in_channels) + 2) // 3
                    expanded = old_conv.weight.repeat(1, repeats, 1, 1)[:, : int(in_channels)]
                    expanded = expanded * (3.0 / float(in_channels))
                    model.conv1.weight.copy_(expanded)
    model.fc = nn.Identity()
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    return model


class M2BeamLLMImageEncoder(nn.Module):
    def __init__(
        self,
        feature_size: int,
        image_channels: int = 1,
        image_channel_adapter: str = "repeat",
        pretrained: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.image_channels = int(image_channels)
        self.image_channel_adapter = str(image_channel_adapter)
        if self.image_channel_adapter not in {"repeat", "rgb", "strict"}:
            raise ValueError("image_channel_adapter must be one of repeat, rgb, or strict.")
        self.backbone = build_resnet18_feature_extractor(
            in_channels=3,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        self.projection = nn.Sequential(nn.Linear(512, feature_size), nn.ReLU(inplace=True))
        self.register_buffer("imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        frames, batch_size, seq_len = flatten_sequence_frames(image_batch, "Image")
        if frames.shape[1] != self.image_channels:
            raise ValueError(
                f"Image input channels ({frames.shape[1]}) must equal configured image_channels "
                f"({self.image_channels})."
            )
        frames = self._adapt_to_rgb(frames.float())
        if frames.shape[-2:] != (224, 224):
            frames = F.interpolate(frames, size=(224, 224), mode="bilinear", align_corners=False)
        frames = (frames - self.imagenet_mean.to(frames.dtype)) / self.imagenet_std.to(frames.dtype)
        features = self.projection(self.backbone(frames))
        return restore_sequence_features(features, batch_size, seq_len)

    def _adapt_to_rgb(self, frames: torch.Tensor) -> torch.Tensor:
        channels = int(frames.shape[1])
        if channels == 3:
            return frames
        if channels == 1 and self.image_channel_adapter == "repeat":
            return frames.repeat(1, 3, 1, 1)
        if channels == 1 and self.image_channel_adapter == "rgb":
            return frames.repeat(1, 3, 1, 1)
        raise ValueError(
            "M2BeamLLM image encoder requires RGB input or explicit single-channel repeat adapter; "
            f"got {channels} channels with adapter '{self.image_channel_adapter}'."
        )


class M2BeamLLMRadarEncoder(nn.Module):
    def __init__(
        self,
        feature_size: int,
        radar_channels: int = 2,
        radar_input_mode: str = "ra_map",
    ):
        super().__init__()
        self.radar_channels = int(radar_channels)
        self.radar_input_mode = str(radar_input_mode)
        if self.radar_input_mode not in {"ra_map", "raw_fft"}:
            raise ValueError("radar_input_mode must be one of ra_map or raw_fft.")
        self.cnn = nn.Sequential(
            nn.Conv2d(self.radar_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(nn.Linear(64, 128), nn.ReLU(inplace=True), nn.Linear(128, feature_size), nn.ReLU(inplace=True))

    def forward(self, radar_batch: torch.Tensor | None = None, *, raw_radar: torch.Tensor | None = None) -> torch.Tensor:
        if self.radar_input_mode == "raw_fft":
            if raw_radar is None:
                raise ValueError("M2BeamLLM radar raw_fft path requires raw radar input; no raw radar tensor was provided.")
            radar_batch = self._raw_fft_to_ra_map(raw_radar)
        if radar_batch is None:
            raise ValueError("M2BeamLLM radar encoder requires radar_batch for ra_map mode.")
        frames, batch_size, seq_len = flatten_sequence_frames(radar_batch, "Radar")
        if int(frames.shape[1]) != self.radar_channels:
            raise ValueError(
                f"Radar input channels ({frames.shape[1]}) must equal radar_channels ({self.radar_channels})."
            )
        features = self.cnn(frames.float()).flatten(1)
        features = self.projection(features)
        return restore_sequence_features(features, batch_size, seq_len)

    def _raw_fft_to_ra_map(self, raw_radar: torch.Tensor) -> torch.Tensor:
        if raw_radar.ndim < 5:
            raise ValueError(
                "M2BeamLLM radar raw_fft path requires raw radar input with shape [B, T, C, ...]."
            )
        raw = raw_radar.float()
        range_fft = torch.fft.fft(raw, dim=-2)
        range_fft = range_fft - range_fft.mean(dim=-2, keepdim=True)
        angle_fft = torch.fft.fftshift(torch.fft.fft(range_fft, dim=-1), dim=-1)
        magnitude = torch.log1p(torch.abs(angle_fft))
        while magnitude.ndim > 5:
            magnitude = magnitude.mean(dim=2)
        if magnitude.ndim != 5:
            raise ValueError(
                "M2BeamLLM radar raw_fft path produced an invalid RA map; expected [B, T, C, H, W]."
            )
        channels = int(magnitude.shape[2])
        if channels < self.radar_channels:
            repeats = (self.radar_channels + channels - 1) // channels
            magnitude = magnitude.repeat(1, 1, repeats, 1, 1)
        return magnitude[:, :, : self.radar_channels]


class M2BeamLLMLidarEncoder(nn.Module):
    def __init__(
        self,
        feature_size: int,
        lidar_channels: int = 1,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        expected_size: Sequence[int] = (256, 256),
        allow_resize: bool = True,
    ):
        super().__init__()
        self.lidar_channels = int(lidar_channels)
        self.expected_size = (int(expected_size[0]), int(expected_size[1]))
        self.allow_resize = bool(allow_resize)
        self.backbone = build_resnet18_feature_extractor(
            in_channels=self.lidar_channels,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        self.projection = nn.Sequential(nn.Linear(512, feature_size), nn.ReLU(inplace=True))

    def forward(self, lidar_batch: torch.Tensor) -> torch.Tensor:
        frames, batch_size, seq_len = flatten_sequence_frames(lidar_batch, "LiDAR")
        if int(frames.shape[1]) != self.lidar_channels:
            raise ValueError(
                f"M2BeamLLM LiDAR encoder expected lidar_channels={self.lidar_channels}, "
                f"got {frames.shape[1]}. Set lidar_channels explicitly when using non-histogram BEV input."
            )
        if tuple(int(dim) for dim in frames.shape[-2:]) != self.expected_size:
            if not self.allow_resize:
                raise ValueError(
                    f"M2BeamLLM LiDAR histogram input must be {self.expected_size}, got {tuple(frames.shape[-2:])}."
                )
            frames = F.interpolate(frames.float(), size=self.expected_size, mode="bilinear", align_corners=False)
        features = self.projection(self.backbone(frames.float()))
        return restore_sequence_features(features, batch_size, seq_len)


class M2BeamLLMGpsEncoder(nn.Module):
    def __init__(
        self,
        feature_size: int,
        gps_input_size: int = 2,
        hidden_dims: Sequence[int] = (32, 64),
    ):
        super().__init__()
        if int(gps_input_size) != 2:
            raise ValueError("M2BeamLLM GPS encoder expects two-dimensional GPS coordinates.")
        first_hidden, second_hidden = (int(hidden_dims[0]), int(hidden_dims[1]))
        self.gps_input_size = int(gps_input_size)
        self.net = nn.Sequential(
            nn.Linear(self.gps_input_size, first_hidden),
            nn.LayerNorm(first_hidden),
            nn.GELU(),
            nn.Linear(first_hidden, second_hidden),
            nn.LayerNorm(second_hidden),
            nn.GELU(),
            nn.Linear(second_hidden, feature_size),
        )

    def forward(self, gps_batch: torch.Tensor) -> torch.Tensor:
        if gps_batch.ndim != 3:
            raise ValueError(f"GPS input must have shape [B, T, 2], got {tuple(gps_batch.shape)}.")
        batch_size, seq_len, feature_dim = (int(dim) for dim in gps_batch.shape)
        if feature_dim != self.gps_input_size:
            raise ValueError(
                f"M2BeamLLM GPS input feature_dim ({feature_dim}) must equal gps_input_size ({self.gps_input_size})."
            )
        features = self.net(gps_batch.reshape(batch_size * seq_len, feature_dim).float())
        return features.view(batch_size, seq_len, -1)
