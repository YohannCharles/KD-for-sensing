"""Deterministic inference-only sensor corruptions for fixed evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CorruptionSpec:
    name: str
    severity: int


CORRUPTION_GRID = tuple(
    CorruptionSpec(name, severity)
    for name in ("gps_noise", "image_occlusion", "image_blur", "radar_noise", "lidar_sparsify")
    for severity in (1, 2, 3)
)

ORACLE_GAP_CORRUPTION_GRID = tuple(
    CorruptionSpec(name, severity)
    for name in ("image_occlusion", "radar_noise", "lidar_sparsify", "gps_noise")
    for severity in (1, 2, 3)
)

CORRUPTION_PARAMETERS = {
    "image_occlusion": {"unit": "area_fraction", "values": (0.10, 0.25, 0.40)},
    "radar_noise": {"unit": "target_snr_db", "values": (20.0, 10.0, 0.0)},
    "lidar_sparsify": {"unit": "retain_probability", "values": (0.75, 0.50, 0.25)},
    "gps_noise": {"unit": "xy_sigma_m", "values": (1.0, 3.0, 10.0)},
}


def apply_inference_corruption(
    batch: dict[str, Any],
    spec: CorruptionSpec,
    *,
    seed: int,
    batch_index: int,
    gps_scaler_mean: Any | None = None,
    gps_scaler_scale: Any | None = None,
    selector: Any | None = None,
) -> dict[str, Any]:
    """Apply one deterministic corruption without changing availability masks.

    When provided, ``selector`` must be ``[B,T]`` and only those temporal cells
    receive the corruption. Omitting it preserves the original whole-batch path.
    """
    severity = int(spec.severity)
    if severity not in {1, 2, 3}:
        raise ValueError("corruption severity must be 1, 2, or 3")
    # Severity reuses the same random draw, so stronger levels are paired perturbations.
    generator = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(batch_index) * 97)
    if spec.name == "gps_noise":
        value = _tensor(batch, "gps")
        mean, scale = _gps_scaler_tensors(value, gps_scaler_mean, gps_scaler_scale)
        physical = value * scale + mean
        radius = physical[..., 0].clamp_min(torch.finfo(value.dtype).eps)
        direction_norm = torch.linalg.vector_norm(physical[..., 1:3], dim=-1).clamp_min(
            torch.finfo(value.dtype).eps
        )
        sin_angle = physical[..., 1] / direction_norm
        cos_angle = physical[..., 2] / direction_norm
        xy = torch.stack((radius * cos_angle, radius * sin_angle), dim=-1)
        sigma_m = float(CORRUPTION_PARAMETERS["gps_noise"]["values"][severity - 1])
        xy = xy + torch.randn(xy.shape, generator=generator, dtype=value.dtype) * sigma_m
        noisy_radius = torch.linalg.vector_norm(xy, dim=-1).clamp_min(torch.finfo(value.dtype).eps)
        noisy_physical = torch.stack(
            (noisy_radius, xy[..., 1] / noisy_radius, xy[..., 0] / noisy_radius), dim=-1
        )
        corrupted = (noisy_physical - mean) / scale
        batch["gps"] = _select_temporal_cells(value, corrupted, selector)
    elif spec.name == "image_occlusion":
        original = _tensor(batch, "image")
        value = original.clone()
        fraction = float(CORRUPTION_PARAMETERS["image_occlusion"]["values"][severity - 1])
        height, width = value.shape[-2:]
        side_fraction = math.sqrt(fraction)
        block_h, block_w = max(1, round(height * side_fraction)), max(1, round(width * side_fraction))
        flat = value.reshape(-1, *value.shape[-3:])
        for item in flat:
            center_y, center_x = torch.rand(2, generator=generator).tolist()
            top = min(max(round(center_y * height - block_h / 2), 0), height - block_h)
            left = min(max(round(center_x * width - block_w / 2), 0), width - block_w)
            item[..., top : top + block_h, left : left + block_w] = 0
        batch["image"] = _select_temporal_cells(original, value, selector)
    elif spec.name == "image_blur":
        value = _tensor(batch, "image")
        kernel = (3, 7, 11)[severity - 1]
        flat = value.reshape(-1, *value.shape[-3:])
        corrupted = F.avg_pool2d(flat, kernel, stride=1, padding=kernel // 2).reshape_as(value)
        batch["image"] = _select_temporal_cells(value, corrupted, selector)
    elif spec.name == "radar_noise":
        target_snr_db = float(CORRUPTION_PARAMETERS["radar_noise"]["values"][severity - 1])
        noise_ratio = 10.0 ** (-target_snr_db / 20.0)
        for key in ("radar_ra", "radar_da"):
            value = _tensor(batch, key)
            std = value.float().flatten(1).std(dim=1, unbiased=False).view(-1, *([1] * (value.ndim - 1)))
            noise = torch.randn(value.shape, generator=generator, dtype=value.dtype)
            corrupted = value + noise * std.to(value.dtype) * noise_ratio
            batch[key] = _select_temporal_cells(value, corrupted, selector)
    elif spec.name == "lidar_sparsify":
        value = _tensor(batch, "lidar")
        keep_probability = float(CORRUPTION_PARAMETERS["lidar_sparsify"]["values"][severity - 1])
        keep_shape = (*value.shape[:-3], 1, *value.shape[-2:])
        keep = torch.rand(keep_shape, generator=generator) < keep_probability
        corrupted = value * keep.to(dtype=value.dtype)
        batch["lidar"] = _select_temporal_cells(value, corrupted, selector)
    else:
        raise ValueError(f"Unsupported inference corruption {spec.name!r}.")
    return batch


def _gps_scaler_tensors(value: torch.Tensor, mean: Any | None, scale: Any | None) -> tuple[torch.Tensor, torch.Tensor]:
    if mean is None or scale is None:
        raise ValueError("gps_noise requires the checkpoint's gps_scaler_mean and gps_scaler_scale")
    mean_tensor = torch.as_tensor(mean, dtype=value.dtype)
    scale_tensor = torch.as_tensor(scale, dtype=value.dtype)
    if mean_tensor.ndim == 1:
        mean_tensor = mean_tensor.reshape(*([1] * (value.ndim - 1)), 3)
        scale_tensor = scale_tensor.reshape(*([1] * (value.ndim - 1)), 3)
    elif mean_tensor.ndim == 2 and mean_tensor.shape == (value.shape[0], 3):
        mean_tensor = mean_tensor.reshape(value.shape[0], *([1] * (value.ndim - 2)), 3)
        scale_tensor = scale_tensor.reshape(value.shape[0], *([1] * (value.ndim - 2)), 3)
    else:
        raise ValueError("GPS scaler mean/scale must be [3] or batch-aligned [B,3].")
    if mean_tensor.shape != scale_tensor.shape or bool((scale_tensor <= 0).any()):
        raise ValueError("GPS scaler mean/scale must have matching shape and positive scale")
    return mean_tensor, scale_tensor


def _tensor(batch: dict[str, Any], key: str) -> torch.Tensor:
    value = batch.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"Inference corruption requires tensor batch field {key!r}.")
    return value


def _select_temporal_cells(
    original: torch.Tensor,
    corrupted: torch.Tensor,
    selector: Any | None,
) -> torch.Tensor:
    if selector is None:
        return corrupted
    if original.ndim < 2 or corrupted.shape != original.shape:
        raise ValueError("Selective corruption requires matching tensors with batch and time dimensions.")
    selected = torch.as_tensor(selector, device=original.device, dtype=torch.bool)
    if tuple(selected.shape) != tuple(original.shape[:2]):
        raise ValueError(
            f"corruption selector must have shape {tuple(original.shape[:2])}, got {tuple(selected.shape)}."
        )
    expanded = selected.reshape(*selected.shape, *([1] * (original.ndim - 2)))
    return torch.where(expanded, corrupted, original)


__all__ = [
    "CORRUPTION_GRID",
    "CORRUPTION_PARAMETERS",
    "ORACLE_GAP_CORRUPTION_GRID",
    "CorruptionSpec",
    "apply_inference_corruption",
]
