from typing import Any

import torch
import torch.nn as nn


class CameraAutoEncoder(nn.Module):
    """Tiny convolutional autoencoder for DeepSense6G camera frames."""

    def __init__(
        self,
        *,
        latent_dim: int = 128,
        image_channels: int = 3,
        base_channels: int = 32,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.image_channels = int(image_channels)
        self.base_channels = int(base_channels)
        self.image_size = int(image_size)
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(self.image_channels, self.base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channels),
            nn.GELU(),
            nn.Conv2d(self.base_channels, self.base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channels * 2),
            nn.GELU(),
            nn.Conv2d(self.base_channels * 2, self.base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channels * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        encoded_dim = self.base_channels * 4 * 4 * 4
        self.to_latent = nn.Linear(encoded_dim, self.latent_dim)
        self.from_latent = nn.Linear(self.latent_dim, encoded_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.base_channels * 4, self.base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channels * 2),
            nn.GELU(),
            nn.ConvTranspose2d(self.base_channels * 2, self.base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channels),
            nn.GELU(),
            nn.ConvTranspose2d(self.base_channels, self.base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channels),
            nn.GELU(),
            nn.ConvTranspose2d(self.base_channels, self.image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder_cnn(image)
        return self.to_latent(features.flatten(start_dim=1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.from_latent(latent).view(latent.shape[0], self.base_channels * 4, 4, 4)
        reconstruction = self.decoder(hidden)
        if reconstruction.shape[-2:] != (self.image_size, self.image_size):
            reconstruction = nn.functional.interpolate(
                reconstruction,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return reconstruction

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(image)
        reconstruction = self.decode(latent)
        return {"reconstruction": reconstruction, "latent": latent}

    def metadata(self) -> dict[str, Any]:
        return {
            "model": "CameraAutoEncoder",
            "latent_dim": self.latent_dim,
            "image_channels": self.image_channels,
            "base_channels": self.base_channels,
            "image_size": self.image_size,
            "pretrained_weights": False,
        }


__all__ = ["CameraAutoEncoder"]
