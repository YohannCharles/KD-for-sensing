import inspect
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from kd_sensing.engine.batch_targets import (
    SENSITIVE_TARGET_FIELDS,
    assert_sensitive_fields_allowed,
    prepare_auxiliary_targets,
    prepare_beam_power_targets,
    prepare_beamspace_power_targets,
    prepare_labels,
    prepare_path_descriptors,
    prepare_path_semantic_labels,
    prepare_radio_semantic_labels,
    prepare_soft_beam_targets,
)
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
    gps_input_seq_len: int | None = None,
    num_pred: int,
    device: torch.device,
    modalities: list[str] | tuple[str, ...] | None = None,
    image_profile: str | None = None,
    input_profiles: dict[str, str] | None = None,
    include_reliability_metadata: bool = False,
    include_missing_modality_metadata: bool = False,
    strict_reliability_metadata: bool = True,
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
        if modality == "gps":
            inputs[input_keys[modality]] = prepare_gps_inputs(
                batch,
                seq_length=seq_length,
                input_seq_length=gps_input_seq_len,
                num_pred=num_pred,
                device=device,
                profile=(input_profiles or {}).get(modality),
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
    if include_reliability_metadata:
        inputs.update(
            prepare_reliability_metadata_inputs(
                batch,
                seq_length=seq_length,
                num_pred=num_pred,
            device=device,
            modalities=selected,
            include_missing_modality_metadata=include_missing_modality_metadata,
            strict=strict_reliability_metadata,
            non_blocking=non_blocking,
        )
        )
    return inputs


def prepare_reliability_metadata_inputs(
    batch: dict[str, Any],
    *,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    modalities: list[str] | tuple[str, ...],
    include_missing_modality_metadata: bool = False,
    strict: bool = True,
    non_blocking: bool = False,
) -> dict[str, Any]:
    selected = normalize_modalities(tuple(modalities), context="reliability metadata modalities")
    inputs: dict[str, Any] = {}
    specs: dict[str, tuple[str, torch.dtype, bool]] = {}
    if "image" in selected:
        specs.update(
            {
                "image_valid_mask": ("image_valid_mask", torch.bool, True),
                "image_observability_score": ("image_observability_score", torch.float32, True),
                "image_dropout_mask": ("image_dropout_mask", torch.bool, False),
                "image_burst_dropout_mask": ("image_burst_dropout_mask", torch.bool, False),
            }
        )
    if "gps" in selected:
        specs.update(
            {
                "gps_valid_mask": ("gps_valid_mask", torch.bool, True),
                "gps_delay_steps": ("gps_delay_steps", torch.float32, True),
                "gps_dropout_mask": ("gps_dropout_mask", torch.bool, False),
            }
        )
    if include_missing_modality_metadata:
        for modality in selected:
            specs.setdefault(f"{modality}_valid_mask", (f"{modality}_valid_mask", torch.bool, False))
            specs.setdefault(f"{modality}_dropout_mask", (f"{modality}_dropout_mask", torch.bool, False))
        for key in ("temporal_mask", "modality_temporal_mask", "available_modalities"):
            if key in batch:
                inputs[key] = torch.as_tensor(batch[key], device=device, dtype=torch.bool)
    for key, (output_key, dtype, required) in specs.items():
        if key not in batch:
            if strict and required:
                raise ValueError(
                    f"Model config declares observability-aware fusion, but required reliability metadata "
                    f"'{key}' is missing from the batch."
                )
            continue
        inputs[output_key] = _prepare_temporal_metadata_input(
            batch[key],
            key=key,
            seq_length=seq_length,
            num_pred=num_pred,
            device=device,
            dtype=dtype,
            pad_value=_metadata_pad_value(key),
            non_blocking=non_blocking,
        )
    return inputs


def model_cfg_consumes_reliability_metadata(model_cfg: Mapping[str, Any] | None) -> bool:
    if not isinstance(model_cfg, Mapping):
        return False
    for key in ("requires_reliability_metadata", "consume_reliability_metadata", "observability_aware"):
        if bool(model_cfg.get(key, False)):
            return True
    if model_cfg_consumes_missing_modality_metadata(model_cfg):
        return True
    fusion = model_cfg.get("observability_aware_fusion", model_cfg.get("reliability_metadata"))
    if isinstance(fusion, Mapping):
        return bool(fusion.get("enabled", True))
    if fusion not in (None, False, "", "none"):
        return bool(fusion)
    model_type = str(model_cfg.get("type", "")).strip().lower()
    return "observability_aware" in model_type


def reliability_metadata_strict(model_cfg: Mapping[str, Any] | None) -> bool:
    if not isinstance(model_cfg, Mapping):
        return True
    if model_cfg_consumes_missing_modality_metadata(model_cfg):
        missing_cfg = model_cfg.get("missing_modality_metadata", {})
        if isinstance(missing_cfg, Mapping) and "strict" in missing_cfg:
            return bool(missing_cfg.get("strict"))
        return bool(model_cfg.get("strict_missing_modality_metadata", False))
    fusion = model_cfg.get("observability_aware_fusion", model_cfg.get("reliability_metadata"))
    if isinstance(fusion, Mapping):
        return bool(fusion.get("strict", fusion.get("require_fields", True)))
    return bool(model_cfg.get("strict_reliability_metadata", True))


def model_cfg_consumes_missing_modality_metadata(model_cfg: Mapping[str, Any] | None) -> bool:
    if not isinstance(model_cfg, Mapping):
        return False
    if bool(model_cfg.get("consume_missing_modality_metadata", False)):
        return True
    raw = model_cfg.get("missing_modality_metadata")
    if isinstance(raw, Mapping):
        if bool(raw.get("enabled", True)):
            return True
    elif raw not in (None, False, "", "none"):
        return bool(raw)
    core = model_cfg.get("representation_core")
    core_type = str(core.get("type", "")) if isinstance(core, Mapping) else str(core or "")
    return core_type == "amber_lite_missing_modality_transformer"


def _prepare_temporal_metadata_input(
    raw: Any,
    *,
    key: str,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    dtype: torch.dtype,
    pad_value: bool | float,
    non_blocking: bool,
) -> torch.Tensor:
    value = torch.as_tensor(raw).to(device=device, dtype=dtype, non_blocking=non_blocking)
    if value.ndim == 1:
        value = value.unsqueeze(1)
    if value.ndim != 2:
        raise ValueError(f"{key} must have shape [B, T] or [B], got {tuple(value.shape)}.")
    value = value[:, -int(seq_length) :]
    value = _left_pad_metadata(value, int(seq_length), pad_value=pad_value)
    pad_steps = max(int(num_pred) - 1, 0)
    if pad_steps <= 0:
        return value
    pad = torch.full(
        (int(value.shape[0]), pad_steps),
        pad_value,
        dtype=value.dtype,
        device=device,
    )
    return torch.cat([value, pad], dim=1)


def _left_pad_metadata(value: torch.Tensor, seq_length: int, *, pad_value: bool | float) -> torch.Tensor:
    if int(value.shape[1]) >= int(seq_length):
        return value
    pad_steps = int(seq_length) - int(value.shape[1])
    pad = torch.full((int(value.shape[0]), pad_steps), pad_value, dtype=value.dtype, device=value.device)
    return torch.cat([pad, value], dim=1)


def _metadata_pad_value(key: str) -> bool | float:
    if key.endswith("valid_mask"):
        return True
    if key.endswith("dropout_mask"):
        return False
    if key == "image_observability_score":
        return 1.0
    if key == "gps_delay_steps":
        return 0.0
    return 0.0


def prepare_gps_inputs(
    batch: dict[str, torch.Tensor],
    *,
    seq_length: int,
    input_seq_length: int | None = None,
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
    window = int(input_seq_length) if input_seq_length is not None else int(seq_length)
    if window <= 0:
        raise ValueError("GPS input_seq_length must be positive when provided.")
    gps = gps[:, -window:, :]
    gps = _left_pad_temporal_sequence(gps, seq_length)
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


def _left_pad_temporal_sequence(value: torch.Tensor, seq_length: int) -> torch.Tensor:
    if int(value.shape[1]) >= int(seq_length):
        return value
    if int(value.shape[1]) <= 0:
        raise ValueError("Temporal input must contain at least one timestep before left padding.")
    pad_steps = int(seq_length) - int(value.shape[1])
    earliest = value[:, :1, ...].expand(-1, pad_steps, *([-1] * (value.ndim - 2)))
    return torch.cat([earliest, value], dim=1)


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
    if radar_ra.ndim != 5:
        raise ValueError(
            f"Radar RA input must have shape [B, T, H, W] or [B, T, C, H, W], got {tuple(radar_ra.shape)}."
        )
    if radar_da.ndim != 5:
        raise ValueError(
            f"Radar DA input must have shape [B, T, H, W] or [B, T, C, H, W], got {tuple(radar_da.shape)}."
        )
    if radar_ra.shape[:2] != radar_da.shape[:2] or radar_ra.shape[-2:] != radar_da.shape[-2:]:
        raise ValueError(
            "Radar RA/DA inputs must share batch, time, height, and width dimensions; "
            f"got RA {tuple(radar_ra.shape)} and DA {tuple(radar_da.shape)}."
        )
    if int(radar_ra.shape[-2]) != 128 or int(radar_ra.shape[-1]) != 64:
        profile_text = f" for profile '{profile}'" if profile else ""
        raise ValueError(
            f"Radar input{profile_text} must use RA/DA maps with shape [B, T, C, 128, 64], "
            f"got RA {tuple(radar_ra.shape)}."
        )
    if int(radar_da.shape[-2]) != 128 or int(radar_da.shape[-1]) != 64:
        profile_text = f" for profile '{profile}'" if profile else ""
        raise ValueError(
            f"Radar input{profile_text} must use RA/DA maps with shape [B, T, C, 128, 64], "
            f"got DA {tuple(radar_da.shape)}."
        )
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
    if torch.is_tensor(batch.get("csi_input")):
        csi = batch["csi_input"].to(device, non_blocking=non_blocking)
    elif "csi_target" in batch:
        raise ValueError("CSI model input must come from 'csi_input'; refusing to pass 'csi_target' to model forward.")
    elif "csi" in batch:
        csi = batch["csi"].to(device, non_blocking=non_blocking)
    else:
        raise ValueError("CSI input is required but batch does not contain a 'csi_input' field.")
    if csi.ndim == 4:
        csi = csi.unsqueeze(0)
    if csi.ndim not in {5, 6}:
        raise ValueError(
            f"CSI input profile '{profile or 'pilot_dual_view'}' must have shape [B, T, Nsc, Nant, 2], "
            f"got {tuple(csi.shape)}."
        )
    csi = csi[:, -seq_length:, ...]
    pad_steps = max(num_pred - 1, 0)
    pad_shape = (int(csi.shape[0]), pad_steps, *tuple(csi.shape[2:]))
    zeros = torch.zeros(pad_shape, dtype=csi.dtype, device=device)
    return torch.cat([csi, zeros], dim=1)


def forward_model(
    model,
    task: str,
    image_batch: torch.Tensor | None = None,
    radar_batch: torch.Tensor | None = None,
    gps_batch: torch.Tensor | None = None,
    lidar_batch: torch.Tensor | None = None,
    mmwave_batch: torch.Tensor | None = None,
    csi_batch: torch.Tensor | None = None,
    force_modality_mask: torch.Tensor | None = None,
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
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        if force_modality_mask is not None:
            if not getattr(model, "supports_force_modality_mask", False):
                raise ValueError("force_modality_mask is only supported by models that opt in to modality masks.")
            kwargs["force_modality_mask"] = force_modality_mask
        kwargs.update(_filter_supported_model_kwargs(model, extra_model_kwargs))
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
    if image_batch is None:
        raise ValueError("Image task requires image_batch")
    if getattr(model, "supports_modality_kwargs", False):
        return model(image_batch=image_batch)
    return model(image_batch)



def _filter_supported_model_kwargs(model, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    materialized = {key: value for key, value in kwargs.items() if value is not None}
    if not materialized:
        return {}
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError, AttributeError):
        return materialized
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return materialized
    allowed = set(signature.parameters)
    return {key: value for key, value in materialized.items() if key in allowed}
