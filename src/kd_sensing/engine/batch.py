from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.modalities import batch_input_keys_for_modalities, image_profile_spec, normalize_modalities


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
    label_downsampled = torch.floor(batch["target_beam"].float() / downsample_ratio).to(torch.int64)
    return label_downsampled[:, :num_pred].to(device, non_blocking=non_blocking)


def prepare_auxiliary_targets(
    batch: dict[str, torch.Tensor],
    *,
    num_pred: int,
    device: torch.device,
    non_blocking: bool = False,
) -> dict[str, torch.Tensor]:
    targets: dict[str, torch.Tensor] = {}
    if "occlusion_label" in batch:
        occlusion = batch["occlusion_label"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
        if occlusion.ndim != 2:
            raise ValueError(f"occlusion_label must have shape [B, H], got {tuple(occlusion.shape)}.")
        targets["occlusion_label"] = occlusion[:, :num_pred]
        valid = batch.get("occlusion_valid")
        if valid is None:
            valid_t = torch.ones_like(targets["occlusion_label"], dtype=torch.bool, device=device)
        else:
            valid_t = valid.to(device=device, dtype=torch.bool, non_blocking=non_blocking)
            if valid_t.ndim != 2:
                raise ValueError(f"occlusion_valid must have shape [B, H], got {tuple(valid_t.shape)}.")
            valid_t = valid_t[:, :num_pred]
        targets["occlusion_valid"] = valid_t
    if "position_target" in batch:
        position = batch["position_target"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
        if position.ndim != 3 or position.shape[-1] != 2:
            raise ValueError(f"position_target must have shape [B, H, 2], got {tuple(position.shape)}.")
        targets["position_target"] = position[:, :num_pred, :]
        valid = batch.get("position_valid")
        if valid is None:
            valid_t = torch.ones(position.shape[:2], dtype=torch.bool, device=device)
        else:
            valid_t = valid.to(device=device, dtype=torch.bool, non_blocking=non_blocking)
            if valid_t.ndim != 2:
                raise ValueError(f"position_valid must have shape [B, H], got {tuple(valid_t.shape)}.")
            valid_t = valid_t[:, :num_pred]
        targets["position_valid"] = valid_t
    if "los_label" in batch:
        los = batch["los_label"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
        if los.ndim == 1:
            los = los.unsqueeze(1)
        if los.ndim != 2:
            raise ValueError(f"los_label must have shape [B, H], got {tuple(los.shape)}.")
        targets["los_label"] = los[:, :num_pred]
    if "link_quality" in batch:
        link = batch["link_quality"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
        if link.ndim == 1:
            link = link.unsqueeze(1)
        if link.ndim != 2:
            raise ValueError(f"link_quality must have shape [B, H], got {tuple(link.shape)}.")
        targets["link_quality"] = link[:, :num_pred]
    return targets


def prepare_radio_semantic_labels(
    batch: dict[str, torch.Tensor],
    *,
    num_pred: int,
    device: torch.device,
    ignore_index: int = -100,
    non_blocking: bool = False,
) -> torch.Tensor | None:
    if "radio_semantic_label" not in batch:
        return None
    labels = batch["radio_semantic_label"].to(device=device, dtype=torch.long, non_blocking=non_blocking)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    if labels.ndim != 2:
        raise ValueError(f"radio_semantic_label must have shape [B, H], got {tuple(labels.shape)}.")
    labels = labels[:, :num_pred]
    available = batch.get("radio_semantic_available")
    if available is not None:
        mask = available.to(device=device, dtype=torch.bool, non_blocking=non_blocking)
        if mask.ndim == 1:
            mask = mask.unsqueeze(1)
        labels = torch.where(mask[:, : labels.shape[1]], labels, torch.full_like(labels, int(ignore_index)))
    return labels


SENSITIVE_TARGET_FIELDS = (
    "target_beam",
    "beam",
    "beam_power",
    "csi",
    "channel",
    "channel_path",
    "path_params",
    "path_descriptor",
    "path_semantic_label",
    "radio_semantic_label",
)


def assert_sensitive_fields_allowed(
    batch: dict[str, Any],
    *,
    split: str,
    label_budget: int | None,
    fields: tuple[str, ...] | list[str],
    allow_labeled_target_path_supervision: bool = False,
    hint: str = "Use source supervision, evaluation-only diagnostics, or enable the labeled target supervision option.",
) -> None:
    budget = int(label_budget or 0)
    split_text = str(split or batch.get("split") or "target").lower()
    unlabeled = budget <= 0 or "unlabeled" in split_text
    for field in fields:
        if field not in batch:
            continue
        is_path_field = field in {"path_params", "path_descriptor", "path_semantic_label"}
        if not unlabeled and (not is_path_field or allow_labeled_target_path_supervision):
            continue
        raise RuntimeError(
            "Target sensitive field access blocked: "
            f"split={split_text}, field={field}, label_budget={budget}. {hint}"
        )


def prepare_path_semantic_labels(
    batch: dict[str, torch.Tensor],
    *,
    num_pred: int,
    device: torch.device,
    ignore_index: int = -100,
    non_blocking: bool = False,
) -> torch.Tensor | None:
    if "path_semantic_label" not in batch:
        return None
    labels = batch["path_semantic_label"].to(device=device, dtype=torch.long, non_blocking=non_blocking)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    if labels.ndim != 2:
        raise ValueError(f"path_semantic_label must have shape [B, H], got {tuple(labels.shape)}.")
    labels = labels[:, :num_pred]
    valid = batch.get("path_valid")
    if valid is not None:
        mask = valid.to(device=device, dtype=torch.bool, non_blocking=non_blocking)
        if mask.ndim == 1:
            mask = mask.unsqueeze(1)
        labels = torch.where(mask[:, : labels.shape[1]], labels, torch.full_like(labels, int(ignore_index)))
    return labels


def prepare_path_descriptors(
    batch: dict[str, torch.Tensor],
    *,
    num_pred: int,
    device: torch.device,
    non_blocking: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if "path_descriptor" not in batch:
        return None
    descriptor = batch["path_descriptor"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    if descriptor.ndim == 2:
        descriptor = descriptor.unsqueeze(1)
    if descriptor.ndim != 3:
        raise ValueError(f"path_descriptor must have shape [B, H, D], got {tuple(descriptor.shape)}.")
    descriptor = descriptor[:, :num_pred, :]
    valid = batch.get("path_valid")
    if valid is None:
        mask = torch.isfinite(descriptor).all(dim=-1)
    else:
        mask = valid.to(device=device, dtype=torch.bool, non_blocking=non_blocking)
        if mask.ndim == 1:
            mask = mask.unsqueeze(1)
        mask = mask[:, : descriptor.shape[1]]
    return descriptor, mask


def prepare_beam_power_targets(
    batch: dict[str, torch.Tensor],
    *,
    num_pred: int,
    device: torch.device,
    non_blocking: bool = False,
) -> torch.Tensor | None:
    if "beam_power" not in batch:
        return None
    power = batch["beam_power"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    if power.ndim == 2:
        power = power.unsqueeze(1)
    if power.ndim != 3:
        raise ValueError(f"beam_power must have shape [B, H, C], got {tuple(power.shape)}.")
    return power[:, :num_pred, :]


def prepare_image_inputs(
    batch: dict[str, torch.Tensor],
    *,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    image_profile: str | None = None,
    non_blocking: bool = False,
) -> torch.Tensor:
    if "image" not in batch:
        raise ValueError("Image input is required but batch does not contain an 'image' field.")
    profile = image_profile_spec(image_profile)
    image = batch["image"].to(device, non_blocking=non_blocking)
    if image.ndim != 5:
        raise ValueError(f"Image input must have shape [B, T, C, H, W], got {tuple(image.shape)}.")
    if int(image.shape[2]) != int(profile.channels):
        raise ValueError(
            f"Image input for profile '{profile.name}' must have {profile.channels} channels, "
            f"got {int(image.shape[2])}."
        )
    image = image[:, -seq_length:, ...]
    target_size = tuple(int(value) for value in profile.default_size)
    if tuple(int(value) for value in image.shape[-2:]) != target_size:
        image = _resize_image_sequence(image, target_size)
    pad_steps = max(int(num_pred) - 1, 0)
    batch_size, _, channels, height, width = image.shape
    zeros = torch.zeros(
        batch_size,
        pad_steps,
        channels,
        height,
        width,
        dtype=image.dtype,
        device=device,
    )
    return torch.cat([image, zeros], dim=1)


def _resize_image_sequence(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    batch_size, seq_len, channels, height, width = image.shape
    frames = image.reshape(batch_size * seq_len, channels, height, width)
    resized = F.interpolate(frames, size=size, mode="bilinear", align_corners=False)
    return resized.reshape(batch_size, seq_len, channels, size[0], size[1])


def prepare_fusion_inputs(
    batch: dict[str, torch.Tensor],
    *,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    modalities: list[str] | tuple[str, ...] | None = None,
    image_profile: str | None = None,
    input_profiles: dict[str, str] | None = None,
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
        "csi": prepare_csi_inputs,
        "coord": prepare_coord_inputs,
        "ray": prepare_ray_inputs,
    }
    inputs: dict[str, torch.Tensor] = {}
    for modality in selected:
        if modality == "image":
            inputs[input_keys[modality]] = prepare_image_inputs(
                batch,
                seq_length=seq_length,
                num_pred=num_pred,
                device=device,
                image_profile=image_profile,
                non_blocking=non_blocking,
            )
            continue
        inputs[input_keys[modality]] = preparers[modality](
            batch,
            seq_length=seq_length,
            num_pred=num_pred,
            device=device,
            profile=(input_profiles or {}).get(modality),
            non_blocking=non_blocking,
        )
    if "geometry" in batch:
        inputs["geometry_batch"] = prepare_geometry_inputs(
            batch,
            seq_length=seq_length,
            num_pred=num_pred,
            device=device,
            non_blocking=non_blocking,
        )
    if "geometry_mask" in batch:
        inputs["geometry_mask"] = prepare_geometry_mask(
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
    profile: str | None = None,
    non_blocking: bool = False,
) -> torch.Tensor:
    if "gps" not in batch:
        raise ValueError("GPS input is required but batch does not contain a 'gps' field.")
    gps = batch["gps"].to(device, non_blocking=non_blocking)
    if gps.ndim == 2:
        gps = gps.unsqueeze(0)
    if gps.ndim != 3:
        profile_text = f" for profile '{profile}'" if profile else ""
        raise ValueError(f"GPS input{profile_text} must have shape [B, T, F], got {tuple(gps.shape)}.")
    if profile == "uav_xyz_snapshot" and int(gps.shape[-1]) != 3:
        raise ValueError(f"GPS input profile 'uav_xyz_snapshot' requires [B, T, 3], got {tuple(gps.shape)}.")
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
    profile: str | None = None,
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
    profile: str | None = None,
    non_blocking: bool = False,
) -> torch.Tensor:
    if "lidar" not in batch:
        raise ValueError("LiDAR input is required but batch does not contain a 'lidar' field.")
    lidar = batch["lidar"].to(device, non_blocking=non_blocking)
    if profile == "point_cloud_xyz_10000":
        if lidar.ndim == 3:
            lidar = lidar.unsqueeze(0)
        if lidar.ndim != 4 or int(lidar.shape[-1]) != 3:
            raise ValueError(
                "LiDAR input profile 'point_cloud_xyz_10000' requires shape [B, T, P, 3], "
                f"got {tuple(lidar.shape)}."
            )
        lidar = lidar[:, -seq_length:, ...]
        batch_size, _, point_count, coord_dim = lidar.shape
        pad_steps = max(num_pred - 1, 0)
        zeros = torch.zeros(
            batch_size,
            pad_steps,
            point_count,
            coord_dim,
            dtype=lidar.dtype,
            device=device,
        )
        return torch.cat([lidar, zeros], dim=1)
    if lidar.ndim == 4:
        lidar = lidar.unsqueeze(2)
    if lidar.ndim == 6:
        lidar = lidar[:, -seq_length:, ...]
        batch_size, _, channels, depth, height, width = lidar.shape
        pad_steps = max(num_pred - 1, 0)
        zeros = torch.zeros(
            batch_size,
            pad_steps,
            channels,
            depth,
            height,
            width,
            dtype=lidar.dtype,
            device=device,
        )
        return torch.cat([lidar, zeros], dim=1)
    if lidar.ndim != 5:
        raise ValueError(
            f"LiDAR input must have shape [B, T, C, H, W] or [B, T, C, D, H, W], got {tuple(lidar.shape)}."
        )
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
    profile: str | None = None,
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


def prepare_csi_inputs(
    batch: dict[str, torch.Tensor],
    *,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    profile: str | None = None,
    non_blocking: bool = False,
) -> torch.Tensor:
    if "csi" not in batch:
        raise ValueError("CSI input is required but batch does not contain a 'csi' field.")
    csi = batch["csi"].to(device, non_blocking=non_blocking)
    if csi.ndim == 4:
        csi = csi.unsqueeze(0)
    if csi.ndim not in {5, 6}:
        expected = "[B, T, M, K, 2]" if profile == "xl_mimo_nf" else "[B, T, Nsc, Nant, 2]"
        raise ValueError(
            f"CSI input profile '{profile or 'pilot_dual_view'}' must have shape {expected}, got {tuple(csi.shape)}."
        )
    if profile == "xl_mimo_nf" and (csi.ndim != 5 or int(csi.shape[-1]) != 2):
        raise ValueError(f"CSI input profile 'xl_mimo_nf' requires [B, T, M, K, 2], got {tuple(csi.shape)}.")
    csi = csi[:, -seq_length:, ...]
    pad_steps = max(num_pred - 1, 0)
    pad_shape = (int(csi.shape[0]), pad_steps, *tuple(csi.shape[2:]))
    zeros = torch.zeros(pad_shape, dtype=csi.dtype, device=device)
    return torch.cat([csi, zeros], dim=1)


def prepare_coord_inputs(
    batch: dict[str, torch.Tensor],
    *,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    profile: str | None = None,
    non_blocking: bool = False,
) -> torch.Tensor:
    return _prepare_snapshot_vector_input(
        batch,
        sample_key="coord",
        display_name="coord",
        seq_length=seq_length,
        num_pred=num_pred,
        device=device,
        non_blocking=non_blocking,
    )


def prepare_ray_inputs(
    batch: dict[str, torch.Tensor],
    *,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    profile: str | None = None,
    non_blocking: bool = False,
) -> torch.Tensor:
    return _prepare_snapshot_vector_input(
        batch,
        sample_key="ray",
        display_name="ray",
        seq_length=seq_length,
        num_pred=num_pred,
        device=device,
        non_blocking=non_blocking,
    )


def prepare_geometry_inputs(
    batch: dict[str, torch.Tensor],
    *,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    non_blocking: bool = False,
) -> torch.Tensor:
    return _prepare_snapshot_vector_input(
        batch,
        sample_key="geometry",
        display_name="geometry",
        seq_length=seq_length,
        num_pred=num_pred,
        device=device,
        non_blocking=non_blocking,
    )


def prepare_geometry_mask(
    batch: dict[str, torch.Tensor],
    *,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    non_blocking: bool = False,
) -> torch.Tensor:
    if "geometry_mask" not in batch:
        raise ValueError("geometry_mask input is required but batch does not contain a 'geometry_mask' field.")
    mask = batch["geometry_mask"].to(device=device, dtype=torch.bool, non_blocking=non_blocking)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"geometry_mask must have shape [B, T, F], got {tuple(mask.shape)}.")
    mask = mask[:, -seq_length:, :]
    batch_size, _, feature_dim = mask.shape
    pad_steps = max(num_pred - 1, 0)
    zeros = torch.zeros(batch_size, pad_steps, feature_dim, dtype=torch.bool, device=device)
    return torch.cat([mask, zeros], dim=1)


def _prepare_snapshot_vector_input(
    batch: dict[str, torch.Tensor],
    *,
    sample_key: str,
    display_name: str,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    non_blocking: bool,
) -> torch.Tensor:
    if sample_key not in batch:
        raise ValueError(f"{display_name} input is required but batch does not contain a '{sample_key}' field.")
    value = batch[sample_key].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    if value.ndim == 2:
        value = value.unsqueeze(1)
    if value.ndim != 3:
        raise ValueError(f"{display_name} input must have shape [B, T, F], got {tuple(value.shape)}.")
    value = value[:, -seq_length:, :]
    batch_size, _, feature_dim = value.shape
    pad_steps = max(num_pred - 1, 0)
    zeros = torch.zeros(batch_size, pad_steps, feature_dim, dtype=value.dtype, device=device)
    return torch.cat([value, zeros], dim=1)


def forward_model(
    model,
    task: str,
    image_batch: torch.Tensor | None = None,
    radar_batch: torch.Tensor | None = None,
    gps_batch: torch.Tensor | None = None,
    lidar_batch: torch.Tensor | None = None,
    mmwave_batch: torch.Tensor | None = None,
    csi_batch: torch.Tensor | None = None,
    coord_batch: torch.Tensor | None = None,
    ray_batch: torch.Tensor | None = None,
    geometry_batch: torch.Tensor | None = None,
    geometry_mask: torch.Tensor | None = None,
    force_modality_mask: torch.Tensor | None = None,
    force_reliability_gate: torch.Tensor | float | None = None,
    gate_temperature: float | torch.Tensor | None = None,
    **extra_model_kwargs,
):
    if task == "fusion":
        kwargs = {
            "image_batch": image_batch,
            "radar_batch": radar_batch,
            "gps_batch": gps_batch,
            "lidar_batch": lidar_batch,
            "mmwave_batch": mmwave_batch,
            "csi_batch": csi_batch,
            "coord_batch": coord_batch,
            "ray_batch": ray_batch,
            "geometry_batch": geometry_batch,
            "geometry_mask": geometry_mask,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
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
        kwargs.update({key: value for key, value in extra_model_kwargs.items() if value is not None})
        return model(**kwargs)
    if task == "radar":
        radar_input = radar_batch if radar_batch is not None else image_batch
        if radar_input is None:
            raise ValueError("Radar task requires radar_batch")
        if getattr(model, "supports_modality_kwargs", False):
            return model(radar_batch=radar_input)
        return model(radar_input)
    if task == "gps":
        if gps_batch is None:
            raise ValueError("GPS task requires gps_batch")
        if getattr(model, "supports_modality_kwargs", False):
            return model(gps_batch=gps_batch)
        return model(gps_batch)
    if task == "lidar":
        if lidar_batch is None:
            raise ValueError("LiDAR task requires lidar_batch")
        if getattr(model, "supports_modality_kwargs", False):
            return model(lidar_batch=lidar_batch)
        return model(lidar_batch)
    if task == "mmwave":
        if mmwave_batch is None:
            raise ValueError("mmWave task requires mmwave_batch")
        if getattr(model, "supports_modality_kwargs", False):
            return model(mmwave_batch=mmwave_batch)
        return model(mmwave_batch)
    if task == "csi":
        if csi_batch is None:
            raise ValueError("CSI task requires csi_batch")
        if getattr(model, "supports_modality_kwargs", False):
            return model(csi_batch=csi_batch)
        return model(csi_batch)
    if task == "coord":
        if coord_batch is None:
            raise ValueError("coord task requires coord_batch")
        if getattr(model, "supports_modality_kwargs", False):
            return model(coord_batch=coord_batch)
        return model(coord_batch)
    if task == "ray":
        if ray_batch is None:
            raise ValueError("ray task requires ray_batch")
        if getattr(model, "supports_modality_kwargs", False):
            return model(ray_batch=ray_batch)
        return model(ray_batch)
    if image_batch is None:
        raise ValueError("Image task requires image_batch")
    if getattr(model, "supports_modality_kwargs", False):
        return model(image_batch=image_batch)
    return model(image_batch)
