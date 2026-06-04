from __future__ import annotations

import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(
        self,
        *,
        encoder: str = "tiny_cnn",
        output_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = str(encoder)
        self.output_dim = int(output_dim)
        if self.encoder == "tiny_cnn":
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.GELU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, self.output_dim),
                nn.GELU(),
            )
        elif self.encoder == "torchvision_resnet18":
            try:
                from torchvision.models import resnet18
            except Exception as exc:  # pragma: no cover - depends on optional torchvision.
                raise RuntimeError("torchvision_resnet18 encoder requires torchvision to be installed.") from exc
            model = resnet18(weights=None)
            in_features = int(model.fc.in_features)
            model.fc = nn.Linear(in_features, self.output_dim)
            self.net = model
        else:
            raise ValueError("ImageEncoder encoder must be tiny_cnn or torchvision_resnet18.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"ImageEncoder expects [B, C, H, W], got {tuple(x.shape)}.")
        if int(x.shape[1]) not in {1, 3}:
            raise ValueError(f"ImageEncoder expects 1 or 3 channels, got {int(x.shape[1])}.")
        if int(x.shape[1]) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.net(x.to(dtype=torch.float32))


class ArrayEncoder(nn.Module):
    def __init__(self, *, output_dim: int = 64) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.flat = nn.Sequential(nn.Flatten(start_dim=1), nn.LazyLinear(self.output_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in {2, 3, 4}:
            raise ValueError(
                "ArrayEncoder supports flat [B, D], 2D map [B, H, W], or 3D map [B, C, H, W]; "
                f"got {tuple(x.shape)}."
            )
        return self.flat(x.to(dtype=torch.float32))


class TabularEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 9,
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(output_dim)),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or int(x.shape[-1]) != self.input_dim:
            raise ValueError(
                f"TabularEncoder expects [B, {self.input_dim}] GPS context features, got {tuple(x.shape)}."
            )
        return self.net(x.to(dtype=torch.float32))

