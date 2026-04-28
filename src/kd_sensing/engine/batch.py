from __future__ import annotations

import torch


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
) -> torch.Tensor:
    beam_downsampled = torch.floor(batch["input_beam"].float() / downsample_ratio).to(torch.int64)
    label_downsampled = torch.floor(batch["target_beam"].float() / downsample_ratio).to(torch.int64)
    return torch.cat(
        [beam_downsampled[..., -1:], label_downsampled[:, :num_pred]],
        dim=-1,
    ).to(device)


def prepare_image_inputs(
    batch: dict[str, torch.Tensor],
    *,
    seq_length: int,
    num_pred: int,
    device: torch.device,
) -> torch.Tensor:
    if "image" not in batch:
        raise ValueError("Image input is required but batch does not contain an 'image' field.")
    image = batch["image"].to(device)
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
) -> dict[str, torch.Tensor]:
    selected = tuple(modalities or ("image", "radar"))
    inputs: dict[str, torch.Tensor] = {}
    if "image" in selected:
        inputs["image_batch"] = prepare_image_inputs(
            batch,
            seq_length=seq_length,
            num_pred=num_pred,
            device=device,
        )
    if "radar" in selected:
        inputs["radar_batch"] = prepare_radar_inputs(
            batch,
            seq_length=seq_length,
            num_pred=num_pred,
            device=device,
        )
    if "gps" in selected:
        inputs["gps_batch"] = prepare_gps_inputs(
            batch,
            seq_length=seq_length,
            num_pred=num_pred,
            device=device,
        )
    if "lidar" in selected:
        inputs["lidar_batch"] = prepare_lidar_inputs(
            batch,
            seq_length=seq_length,
            num_pred=num_pred,
            device=device,
        )
    return inputs


def prepare_gps_inputs(
    batch: dict[str, torch.Tensor],
    *,
    seq_length: int,
    num_pred: int,
    device: torch.device,
) -> torch.Tensor:
    if "gps" not in batch:
        raise ValueError("GPS input is required but batch does not contain a 'gps' field.")
    gps = batch["gps"].to(device)
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
) -> torch.Tensor:
    if "radar_ra" not in batch or "radar_da" not in batch:
        raise ValueError("Radar input is required but batch does not contain 'radar_ra' and 'radar_da' fields.")
    radar_ra = batch["radar_ra"].to(device)
    radar_da = batch["radar_da"].to(device)
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
) -> torch.Tensor:
    if "lidar" not in batch:
        raise ValueError("LiDAR input is required but batch does not contain a 'lidar' field.")
    lidar = batch["lidar"].to(device)
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


def forward_model(
    model,
    task: str,
    image_batch: torch.Tensor | None = None,
    radar_batch: torch.Tensor | None = None,
    gps_batch: torch.Tensor | None = None,
    lidar_batch: torch.Tensor | None = None,
):
    if task == "fusion":
        return model(
            image_batch=image_batch,
            radar_batch=radar_batch,
            gps_batch=gps_batch,
            lidar_batch=lidar_batch,
        )
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
    if image_batch is None:
        raise ValueError("Image task requires image_batch")
    return model(image_batch)
