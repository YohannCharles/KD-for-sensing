from typing import Any

import torch


def normalize_batch(batch: Any) -> dict[str, Any]:
    if not isinstance(batch, dict):
        raise TypeError(f"Four-modality training requires a mapping batch, got {type(batch).__name__}.")
    return batch


def prepare_labels(
    batch: dict[str, Any],
    *,
    num_pred: int,
    device: torch.device,
    non_blocking: bool = False,
) -> torch.Tensor:
    if "target_beam" not in batch:
        raise ValueError("Four-modality batch is missing target_beam.")
    labels = torch.as_tensor(batch["target_beam"], dtype=torch.long).to(device=device, non_blocking=non_blocking)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    if labels.ndim != 2:
        raise ValueError(f"target_beam must have shape [B, H], got {tuple(labels.shape)}.")
    return labels[:, :num_pred]


def prepare_fusion_inputs(
    batch: dict[str, Any],
    *,
    seq_length: int,
    device: torch.device,
    non_blocking: bool = False,
    modalities: tuple[str, ...] = ("image", "radar", "gps", "lidar"),
) -> dict[str, Any]:
    canonical = ("image", "radar", "gps", "lidar")
    pcpf_sparse_csi = (*canonical, "csi")
    names = tuple(str(value) for value in modalities)
    if (
        not names
        or len(set(names)) != len(names)
        or (names != pcpf_sparse_csi and bool(set(names) - set(canonical)))
    ):
        raise ValueError("Fusion modalities must be a unique non-empty canonical subset.")
    inputs: dict[str, Any] = {}
    if "image" in names:
        image = _sequence(batch, "image", seq_length, device, non_blocking)
        if image.ndim != 5:
            raise ValueError("image must have shape [B,T,C,H,W].")
        inputs["image_batch"] = image
    if "radar" in names:
        radar_ra = _sequence(batch, "radar_ra", seq_length, device, non_blocking)
        radar_da = _sequence(batch, "radar_da", seq_length, device, non_blocking)
        if radar_ra.ndim == 4:
            radar_ra = radar_ra.unsqueeze(2)
        if radar_da.ndim == 4:
            radar_da = radar_da.unsqueeze(2)
        if radar_ra.ndim != 5 or radar_da.ndim != 5 or radar_ra.shape != radar_da.shape:
            raise ValueError("Radar RA/DA inputs must share shape [B,T,C,H,W].")
        if tuple(radar_ra.shape[-2:]) != (128, 64):
            raise ValueError("Radar maps must have spatial shape [128, 64].")
        inputs["radar_batch"] = torch.cat([radar_ra, radar_da], dim=2)
    if "gps" in names:
        gps = _sequence(batch, "gps", seq_length, device, non_blocking)
        if gps.ndim != 3:
            raise ValueError("gps must have shape [B,T,F].")
        inputs["gps_batch"] = gps
    if "lidar" in names:
        lidar = _sequence(batch, "lidar", seq_length, device, non_blocking)
        if lidar.ndim != 5:
            raise ValueError("lidar must have shape [B,T,C,H,W].")
        inputs["lidar_batch"] = lidar
    if "csi" in names:
        csi = _sequence(batch, "csi", seq_length, device, non_blocking)
        pattern_ids = _sequence(batch, "csi_pattern_ids", seq_length, device, non_blocking).long()
        pilot_mask = _sequence(batch, "csi_pilot_mask", seq_length, device, non_blocking).bool()
        if not torch.is_complex(csi) or tuple(csi.shape[1:]) != (seq_length, 2, 2):
            raise ValueError(f"csi must be complex [B,{seq_length},2,2].")
        if tuple(pattern_ids.shape) != tuple(csi.shape[:3]) or tuple(pilot_mask.shape) != tuple(csi.shape):
            raise ValueError("csi_pattern_ids/csi_pilot_mask must have shapes [B,T,2] and [B,T,2,2].")
        inputs.update(
            csi_batch=csi,
            csi_pattern_ids=pattern_ids,
            csi_pilot_mask=pilot_mask,
            csi_frequency_positions=_batch_value(batch, "csi_frequency_positions", device, non_blocking),
            csi_frequency_ids=_batch_value(batch, "csi_frequency_ids", device, non_blocking).long(),
        )
        if "csi_snr_db" in batch:
            inputs["csi_snr_db"] = _batch_value(batch, "csi_snr_db", device, non_blocking)
        if "csi_snr_available" in batch:
            inputs["csi_snr_available"] = _batch_value(
                batch,
                "csi_snr_available",
                device,
                non_blocking,
            ).bool()
    selected = [canonical.index(name) for name in names] if "csi" not in names else []
    for key in ("temporal_mask", "modality_temporal_mask", "available_modalities"):
        if key in batch:
            value = torch.as_tensor(batch[key], device=device, dtype=torch.bool)
            if key != "temporal_mask" and value.shape[-1] == len(canonical):
                if "csi" in names:
                    csi_valid = torch.as_tensor(batch.get("csi_valid_mask"), device=device, dtype=torch.bool)
                    if key == "available_modalities":
                        csi_valid = csi_valid.any(dim=1)
                    value = torch.cat([value, csi_valid.unsqueeze(-1)], dim=-1)
                elif len(names) != len(canonical):
                    value = value[..., selected]
            inputs[key] = value
    return inputs


def forward_model(model, *, force_modality_mask: torch.Tensor | None = None, **inputs: Any):
    if force_modality_mask is not None:
        if not getattr(model, "supports_force_modality_mask", False):
            raise ValueError("Model does not support force_modality_mask.")
        inputs["force_modality_mask"] = force_modality_mask
    return model(**inputs)


def _sequence(batch: dict[str, Any], key: str, seq_length: int, device: torch.device, non_blocking: bool) -> torch.Tensor:
    if key not in batch:
        raise ValueError(f"Four-modality batch is missing {key}.")
    value = torch.as_tensor(batch[key]).to(device=device, non_blocking=non_blocking)
    if value.ndim < 3:
        raise ValueError(f"{key} must include batch and time dimensions, got {tuple(value.shape)}.")
    value = value[:, -seq_length:]
    if value.shape[1] == seq_length:
        return value
    if value.shape[1] == 0:
        raise ValueError(f"{key} has no timesteps.")
    pad = value[:, :1].expand(-1, seq_length - value.shape[1], *([-1] * (value.ndim - 2)))
    return torch.cat([pad, value], dim=1)


def _batch_value(
    batch: dict[str, Any],
    key: str,
    device: torch.device,
    non_blocking: bool,
) -> torch.Tensor:
    if key not in batch:
        raise ValueError(f"Fusion batch is missing {key}.")
    return torch.as_tensor(batch[key]).to(device=device, non_blocking=non_blocking)
