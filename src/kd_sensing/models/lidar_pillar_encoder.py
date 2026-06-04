from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn


class LidarBEVSpatialEncoder(nn.Module):
    """Small CNN that preserves a spatial BEV feature map for BGAM."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = int(in_channels)
        for width in channels:
            width = int(width)
            layers.extend(
                [
                    nn.Conv2d(current, width, kernel_size=3, padding=1),
                    nn.BatchNorm2d(width),
                    nn.GELU(),
                ]
            )
            if float(dropout) > 0:
                layers.append(nn.Dropout2d(float(dropout)))
            current = width
        self.out_channels = current
        self.net = nn.Sequential(*layers)

    def forward(self, lidar_bev: torch.Tensor) -> torch.Tensor:
        if lidar_bev.ndim != 4:
            raise ValueError(f"lidar_bev must have shape [B,C,H,W], got {tuple(lidar_bev.shape)}.")
        return self.net(lidar_bev.to(dtype=torch.float32))


class SimplePillarEncoder(nn.Module):
    """Pure PyTorch raw point cloud to fixed-size six-channel pseudo-image."""

    def __init__(
        self,
        *,
        bev_size: Sequence[int] = (64, 64),
        roi: Sequence[float] = (-30.0, 30.0, -30.0, 30.0, -3.0, 5.0),
    ) -> None:
        super().__init__()
        self.bev_size = (int(bev_size[0]), int(bev_size[1]))
        self.roi = tuple(float(value) for value in roi)

    def forward(self, raw_points: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(raw_points):
            if raw_points.ndim == 2:
                return self.pillarize(raw_points).unsqueeze(0)
            if raw_points.ndim == 3:
                return torch.stack([self.pillarize(raw_points[idx]) for idx in range(int(raw_points.shape[0]))], dim=0)
        return torch.stack([self.pillarize(points) for points in raw_points], dim=0)

    def pillarize(self, points: torch.Tensor) -> torch.Tensor:
        height, width = self.bev_size
        device = points.device if torch.is_tensor(points) else torch.device("cpu")
        pseudo = torch.zeros((6, height, width), dtype=torch.float32, device=device)
        if points.numel() == 0:
            return pseudo
        pts = points.to(device=device, dtype=torch.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"raw point cloud must have shape [N,3] or [N,4], got {tuple(pts.shape)}.")
        if pts.shape[1] == 3:
            pts = torch.cat([pts, torch.ones((pts.shape[0], 1), device=device, dtype=pts.dtype)], dim=1)
        x_min, x_max, y_min, y_max = self.roi[:4]
        z_min, z_max = self.roi[4:] if len(self.roi) >= 6 else (-1e9, 1e9)
        mask = (
            pts[:, 0].ge(x_min)
            & pts[:, 0].le(x_max)
            & pts[:, 1].ge(y_min)
            & pts[:, 1].le(y_max)
            & pts[:, 2].ge(z_min)
            & pts[:, 2].le(z_max)
        )
        pts = pts[mask]
        if pts.numel() == 0:
            return pseudo
        x_span = max(float(x_max - x_min), 1e-6)
        y_span = max(float(y_max - y_min), 1e-6)
        cols = torch.clamp(((pts[:, 0] - x_min) / x_span * width).floor().long(), 0, width - 1)
        rows = torch.clamp(height - 1 - ((pts[:, 1] - y_min) / y_span * height).floor().long(), 0, height - 1)
        flat = rows * width + cols
        count = torch.bincount(flat, minlength=height * width).to(torch.float32)
        count_safe = count.clamp_min(1.0)
        sum_z = torch.zeros(height * width, dtype=torch.float32, device=device).scatter_add_(0, flat, pts[:, 2])
        sum_i = torch.zeros(height * width, dtype=torch.float32, device=device).scatter_add_(0, flat, pts[:, 3])
        max_z = torch.full((height * width,), -torch.inf, dtype=torch.float32, device=device)
        max_z.scatter_reduce_(0, flat, pts[:, 2], reduce="amax", include_self=True)
        x_centers = x_min + (cols.to(torch.float32) + 0.5) / width * x_span
        y_centers = y_min + (height - rows.to(torch.float32) - 0.5) / height * y_span
        sum_x_offset = torch.zeros(height * width, dtype=torch.float32, device=device).scatter_add_(0, flat, pts[:, 0] - x_centers)
        sum_y_offset = torch.zeros(height * width, dtype=torch.float32, device=device).scatter_add_(0, flat, pts[:, 1] - y_centers)
        nonempty = count.gt(0)
        pseudo[0] = (torch.log1p(count) / torch.log1p(count.max().clamp_min(1.0))).reshape(height, width)
        pseudo[1] = torch.where(nonempty, sum_z / count_safe, torch.zeros_like(sum_z)).reshape(height, width)
        pseudo[2] = torch.where(nonempty, max_z, torch.zeros_like(max_z)).reshape(height, width)
        pseudo[3] = torch.where(nonempty, sum_i / count_safe, torch.zeros_like(sum_i)).reshape(height, width)
        pseudo[4] = torch.where(nonempty, sum_x_offset / count_safe, torch.zeros_like(sum_x_offset)).reshape(height, width)
        pseudo[5] = torch.where(nonempty, sum_y_offset / count_safe, torch.zeros_like(sum_y_offset)).reshape(height, width)
        return pseudo


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def lidar_quality_summary(bev_batches: Sequence[torch.Tensor], *, roi: Sequence[float], bev_size: Sequence[int], cache_path: str = "") -> dict[str, Any]:
    if not bev_batches:
        return {
            "sample_count": 0,
            "nonempty_rate": 0.0,
            "zero_value_ratio": 0.0,
            "channel_mean": [],
            "channel_std": [],
            "roi": [float(value) for value in roi],
            "bev_size": [int(value) for value in bev_size],
            "cache_path": str(cache_path),
            "lidar_input_degradation_risk": True,
        }
    tensor = torch.cat([batch.detach().cpu().to(torch.float32) for batch in bev_batches], dim=0)
    nonempty = tensor.flatten(start_dim=1).abs().sum(dim=1).gt(0)
    zero_ratio = float(tensor.eq(0).to(torch.float32).mean().item())
    channel_mean = tensor.mean(dim=(0, 2, 3)).tolist()
    channel_std = tensor.std(dim=(0, 2, 3), unbiased=False).tolist()
    risk = bool(float(nonempty.to(torch.float32).mean().item()) < 0.5 or zero_ratio > 0.98 or max(channel_std or [0.0]) < 1e-8)
    return {
        "sample_count": int(tensor.shape[0]),
        "nonempty_rate": float(nonempty.to(torch.float32).mean().item()),
        "zero_value_ratio": zero_ratio,
        "channel_mean": [float(value) for value in channel_mean],
        "channel_std": [float(value) for value in channel_std],
        "roi": [float(value) for value in roi],
        "bev_size": [int(value) for value in bev_size],
        "cache_path": str(cache_path),
        "lidar_input_degradation_risk": risk,
    }


__all__ = [
    "LidarBEVSpatialEncoder",
    "SimplePillarEncoder",
    "freeze_module",
    "lidar_quality_summary",
]
