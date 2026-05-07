from __future__ import annotations

import torch

from kd_sensing.modalities import batch_input_keys_for_modalities, normalize_modalities


def normalize_batch(batch) -> dict[str, torch.Tensor]:
    if isinstance(batch, dict):
        return batch
    if isinstance(batch, (tuple, list)) and len(batch) == 5:
        image, radar_ra, radar_da, input_beam, target_beam = batch
        return {
            "image": image,
            "radar_ra": radar_ra,
            "radar_da": radar_da,
            "input_beam": input_beam,
            "target_beam": target_beam,
        }
    raise TypeError(f"Unsupported batch type: {type(batch).__name__}")


def prepare_labels(
    batch: dict[str, torch.Tensor],
    *,
    num_pred: int,
    downsample_ratio: int,
    device: torch.device,
    non_blocking: bool = False,
) -> torch.Tensor:
    beam_downsampled = torch.floor(batch["input_beam"].float() / downsample_ratio).to(torch.int64)
    label_downsampled = torch.floor(batch["target_beam"].float() / downsample_ratio).to(torch.int64)
    return torch.cat(
        [beam_downsampled[..., -1:], label_downsampled[:, :num_pred]],
        dim=-1,
    ).to(device, non_blocking=non_blocking)


def prepare_image_inputs(
    batch: dict[str, torch.Tensor],
    *,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    non_blocking: bool = False,
) -> torch.Tensor:
    if "image" not in batch:
        raise ValueError("Image input is required but batch does not contain an 'image' field.")
    image = batch["image"].to(device, non_blocking=non_blocking)
    if image.ndim == 4:
        image = image.unsqueeze(2)
    image = image[:, 1 - seq_length :, ...]
    batch_size, _, channels, height, width = image.shape
    zeros = torch.zeros(
        batch_size,
        num_pred,
        channels,
        height,
        width,
        dtype=image.dtype,
        device=device,
    )
    return torch.cat([image, zeros], dim=1)


def prepare_fusion_inputs(
    batch: dict[str, torch.Tensor],
    *,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    modalities: list[str] | tuple[str, ...] | None = None,
    non_blocking: bool = False,
) -> dict[str, torch.Tensor]:
    selected = normalize_modalities(tuple(modalities or ("image", "radar")), context="fusion batch modalities")
    input_keys = batch_input_keys_for_modalities(selected)
    preparers = {
        "image": prepare_image_inputs,
        "radar": prepare_radar_inputs,
        "gps": prepare_gps_inputs,
        "lidar": prepare_lidar_inputs,
        "mmwave": prepare_mmwave_inputs,
    }
    inputs: dict[str, torch.Tensor] = {}
    for modality in selected:
        inputs[input_keys[modality]] = preparers[modality](
            batch,
            seq_length=seq_length,
            num_pred=num_pred,
            device=device,
            non_blocking=non_blocking,
        )
    return inputs


def prepare_gps_inputs(
    batch: dict[str, torch.Tensor],
    *,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    non_blocking: bool = False,
) -> torch.Tensor:
    if "gps" not in batch:
        raise ValueError("GPS input is required but batch does not contain a 'gps' field.")
    gps = batch["gps"].to(device, non_blocking=non_blocking)
    if gps.ndim == 2:
        gps = gps.unsqueeze(0)
    gps = gps[:, -seq_length:, :]
    batch_size, _, feature_dim = gps.shape
    pad_steps = max(num_pred - 1, 0)
    zeros = torch.zeros(
        batch_size,
        pad_steps,
        feature_dim,
        dtype=gps.dtype,
        device=device,
    )
    return torch.cat([gps, zeros], dim=1)


def prepare_radar_inputs(
    batch: dict[str, torch.Tensor],
    *,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    non_blocking: bool = False,
) -> torch.Tensor:
    if "radar_ra" not in batch or "radar_da" not in batch:
        raise ValueError("Radar input is required but batch does not contain 'radar_ra' and 'radar_da' fields.")
    radar_ra = batch["radar_ra"].to(device, non_blocking=non_blocking)
    radar_da = batch["radar_da"].to(device, non_blocking=non_blocking)
    if radar_ra.ndim == 4:
        radar_ra = radar_ra.unsqueeze(2)
    if radar_da.ndim == 4:
        radar_da = radar_da.unsqueeze(2)
    radar_ra = radar_ra[:, -seq_length:, ...]
    radar_da = radar_da[:, -seq_length:, ...]
    batch_size, _, channels, height, width = radar_ra.shape
    pad_steps = max(num_pred - 1, 0)
    zeros_ra = torch.zeros(
        batch_size,
        pad_steps,
        channels,
        height,
        width,
        dtype=radar_ra.dtype,
        device=device,
    )
    zeros_da = torch.zeros_like(zeros_ra)
    radar_ra = torch.cat([radar_ra, zeros_ra], dim=1)
    radar_da = torch.cat([radar_da, zeros_da], dim=1)
    return torch.cat([radar_ra, radar_da], dim=2)


def prepare_lidar_inputs(
    batch: dict[str, torch.Tensor],
    *,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    non_blocking: bool = False,
) -> torch.Tensor:
    if "lidar" not in batch:
        raise ValueError("LiDAR input is required but batch does not contain a 'lidar' field.")
    lidar = batch["lidar"].to(device, non_blocking=non_blocking)
    if lidar.ndim == 4:
        lidar = lidar.unsqueeze(2)
    if lidar.ndim != 5:
        raise ValueError(f"LiDAR input must have shape [B, T, C, H, W], got {tuple(lidar.shape)}.")
    lidar = lidar[:, -seq_length:, ...]
    batch_size, _, channels, height, width = lidar.shape
    pad_steps = max(num_pred - 1, 0)
    zeros = torch.zeros(
        batch_size,
        pad_steps,
        channels,
        height,
        width,
        dtype=lidar.dtype,
        device=device,
    )
    return torch.cat([lidar, zeros], dim=1)


def prepare_mmwave_inputs(
    batch: dict[str, torch.Tensor],
    *,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    non_blocking: bool = False,
) -> torch.Tensor:
    if "mmwave" not in batch:
        raise ValueError("mmWave input is required but batch does not contain a 'mmwave' field.")
    mmwave = batch["mmwave"].to(device, non_blocking=non_blocking)
    if mmwave.ndim == 2:
        mmwave = mmwave.unsqueeze(0)
    if mmwave.ndim != 3:
        raise ValueError(f"mmWave input must have shape [B, T, 64], got {tuple(mmwave.shape)}.")
    mmwave = mmwave[:, -seq_length:, :]
    batch_size, _, feature_dim = mmwave.shape
    pad_steps = max(num_pred - 1, 0)
    zeros = torch.zeros(
        batch_size,
        pad_steps,
        feature_dim,
        dtype=mmwave.dtype,
        device=device,
    )
    return torch.cat([mmwave, zeros], dim=1)


def forward_model(
    model,
    task: str,
    image_batch: torch.Tensor | None = None,
    radar_batch: torch.Tensor | None = None,
    gps_batch: torch.Tensor | None = None,
    lidar_batch: torch.Tensor | None = None,
    mmwave_batch: torch.Tensor | None = None,
    force_modality_mask: torch.Tensor | None = None,
    force_reliability_gate: torch.Tensor | float | None = None,
    gate_temperature: float | torch.Tensor | None = None,
):
    if task == "fusion":
        kwargs = {
            "image_batch": image_batch,
            "radar_batch": radar_batch,
            "gps_batch": gps_batch,
            "lidar_batch": lidar_batch,
            "mmwave_batch": mmwave_batch,
        }
        if force_modality_mask is not None:
            if not getattr(model, "supports_force_modality_mask", False):
                raise ValueError("force_modality_mask is only supported by models that opt in to modality masks.")
            kwargs["force_modality_mask"] = force_modality_mask
        if force_reliability_gate is not None:
            if not getattr(model, "supports_reliability_controls", False):
                raise ValueError("force_reliability_gate is only supported by CRAF-style models.")
            kwargs["force_reliability_gate"] = force_reliability_gate
        if gate_temperature is not None:
            if not getattr(model, "supports_reliability_controls", False):
                raise ValueError("gate_temperature is only supported by CRAF-style models.")
            kwargs["gate_temperature"] = gate_temperature
        return model(**kwargs)
    if task == "radar":
        radar_input = radar_batch if radar_batch is not None else image_batch
        if radar_input is None:
            raise ValueError("Radar task requires radar_batch")
        return model(radar_input)
    if task == "gps":
        if gps_batch is None:
            raise ValueError("GPS task requires gps_batch")
        return model(gps_batch)
    if task == "lidar":
        if lidar_batch is None:
            raise ValueError("LiDAR task requires lidar_batch")
        return model(lidar_batch)
    if task == "mmwave":
        if mmwave_batch is None:
            raise ValueError("mmWave task requires mmwave_batch")
        return model(mmwave_batch)
    if image_batch is None:
        raise ValueError("Image task requires image_batch")
    return model(image_batch)
