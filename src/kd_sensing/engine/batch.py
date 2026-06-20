import inspect
from typing import Any, Mapping

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


def prepare_soft_beam_targets(
    batch: dict[str, torch.Tensor],
    *,
    num_pred: int,
    num_classes: int,
    device: torch.device,
    downsample_ratio: int = 1,
    enabled: bool = True,
    non_blocking: bool = False,
) -> torch.Tensor | None:
    if not enabled:
        return None
    target_key = _soft_beam_target_key(batch)
    if target_key is None:
        return None
    targets = batch[target_key].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    if targets.ndim == 2:
        targets = targets.unsqueeze(1)
    if targets.ndim != 3:
        raise ValueError(f"target_beam_distribution must have shape [B, H, C], got {tuple(targets.shape)}.")
    targets = targets[:, :num_pred, :]
    downsample_ratio = int(downsample_ratio or 1)
    if int(targets.shape[-1]) != int(num_classes) and downsample_ratio > 1:
        source_classes = int(targets.shape[-1])
        class_ids = torch.arange(source_classes, device=device, dtype=torch.long)
        group_ids = torch.div(class_ids, downsample_ratio, rounding_mode="floor").clamp_max(int(num_classes) - 1)
        downsampled = torch.zeros(*targets.shape[:2], int(num_classes), dtype=targets.dtype, device=device)
        targets = downsampled.scatter_add(
            -1,
            group_ids.view(1, 1, -1).expand(targets.shape[0], targets.shape[1], -1),
            targets,
        )
    if int(targets.shape[-1]) != int(num_classes):
        raise ValueError(
            f"target_beam_distribution class dimension must match num_classes={num_classes}, "
            f"got {int(targets.shape[-1])}."
        )
    mask = _soft_beam_target_mask(batch, target_key)
    content_valid = torch.isfinite(targets).all(dim=-1) & targets.sum(dim=-1).gt(0)
    if mask is None:
        valid = content_valid
    else:
        valid = mask.to(device=device, dtype=torch.bool, non_blocking=non_blocking)
        if valid.ndim == 1:
            valid = valid.unsqueeze(1)
        if valid.ndim != 2:
            raise ValueError(f"target_beam_distribution_mask must have shape [B, H], got {tuple(valid.shape)}.")
        valid = valid[:, : targets.shape[1]]
        valid = valid & content_valid
    row_sum = targets.sum(dim=-1, keepdim=True)
    normalized = torch.where(row_sum.gt(0), targets / row_sum.clamp_min(1e-12), torch.zeros_like(targets))
    normalized = torch.where(valid.unsqueeze(-1), normalized, torch.zeros_like(normalized))
    if "target_beam" not in batch:
        return normalized
    hard_labels = torch.floor(batch["target_beam"].float() / downsample_ratio).to(torch.long)
    hard_labels = hard_labels.to(device=device, non_blocking=non_blocking)
    if hard_labels.ndim == 1:
        hard_labels = hard_labels.unsqueeze(1)
    if hard_labels.ndim != 2:
        raise ValueError(f"target_beam must have shape [B, H], got {tuple(hard_labels.shape)}.")
    hard_labels = hard_labels[:, : targets.shape[1]]
    hard_valid = hard_labels.ge(0) & hard_labels.lt(int(num_classes))
    fallback_rows = (~valid) & hard_valid
    if not torch.any(fallback_rows):
        return normalized
    fallback = F.one_hot(hard_labels.clamp_min(0), num_classes=int(num_classes)).to(normalized.dtype)
    return torch.where(fallback_rows.unsqueeze(-1), fallback, normalized)


def _soft_beam_target_key(batch: dict[str, torch.Tensor]) -> str | None:
    return "target_beam_distribution" if "target_beam_distribution" in batch else None


def _soft_beam_target_mask(batch: dict[str, torch.Tensor], target_key: str):
    preferred = f"{target_key}_mask"
    for key in ("target_beam_distribution_mask", preferred):
        if key in batch:
            return batch[key]
    return None


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
    "target_beam_distribution",
    "beam",
    "beam_power",
    "beamspace_power_label",
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
    allow_labeled_target_radio_supervision: bool = False,
    hint: str = "Use source supervision, evaluation-only diagnostics, or enable the labeled target supervision option.",
) -> None:
    budget = int(label_budget or 0)
    split_text = str(split or batch.get("split") or "target").lower()
    is_source = split_text.startswith("source")
    is_target_test = "target_test" in split_text or split_text.endswith("test")
    labeled_subset = (budget > 0) and ("unlabeled" not in split_text) and not is_target_test
    if is_source:
        return
    for field in fields:
        if field not in batch:
            continue
        is_path_field = field in {"path_params", "path_descriptor", "path_semantic_label"}
        is_radio_field = field == "radio_semantic_label"
        is_beam_field = field in {"target_beam", "target_beam_distribution", "beam"}
        allowed = False
        if labeled_subset and is_beam_field:
            allowed = True
        elif labeled_subset and is_path_field and allow_labeled_target_path_supervision:
            allowed = True
        elif labeled_subset and is_radio_field and allow_labeled_target_radio_supervision:
            allowed = True
        if allowed:
            continue
        raise RuntimeError(
            "Target sensitive field access blocked: "
            f"split={split_text}, field={field}, label_budget={budget}, "
            f"labeled_subset={labeled_subset}. {hint}"
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


def prepare_beamspace_power_targets(
    batch: dict[str, torch.Tensor],
    *,
    num_pred: int,
    device: torch.device,
    non_blocking: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if "beamspace_power_label" not in batch:
        return None
    target = batch["beamspace_power_label"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    if target.ndim == 2:
        target = target.unsqueeze(1)
    if target.ndim != 3:
        raise ValueError(f"beamspace_power_label must have shape [B, H, C], got {tuple(target.shape)}.")
    target = target[:, :num_pred, :]
    mask_raw = batch.get("beamspace_power_available")
    if mask_raw is None:
        mask = torch.isfinite(target).all(dim=-1) & target.sum(dim=-1).gt(0)
    else:
        mask = mask_raw.to(device=device, dtype=torch.bool, non_blocking=non_blocking)
        if mask.ndim == 1:
            mask = mask.unsqueeze(1)
        if mask.ndim != 2:
            raise ValueError(f"beamspace_power_available must have shape [B, H], got {tuple(mask.shape)}.")
        mask = mask[:, : target.shape[1]]
        mask = mask & torch.isfinite(target).all(dim=-1) & target.sum(dim=-1).gt(0)
    row_sum = target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    target = torch.where(mask.unsqueeze(-1), target / row_sum, torch.zeros_like(target))
    return target, mask


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
    if "gps_bev_xy" in batch:
        inputs["gps_bev_xy_batch"] = prepare_gps_bev_xy_inputs(
            batch,
            seq_length=seq_length,
            num_pred=num_pred,
            device=device,
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
                "gps_counterfactual_mask": ("gps_counterfactual_mask", torch.bool, False),
            }
        )
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
    condition = _benchmark_condition_metadata(batch)
    if condition:
        inputs["benchmark_condition_metadata"] = condition
    if "image_degradation_metadata" in batch:
        inputs["image_degradation_metadata"] = batch["image_degradation_metadata"]
    return inputs


def model_cfg_consumes_reliability_metadata(model_cfg: Mapping[str, Any] | None) -> bool:
    if not isinstance(model_cfg, Mapping):
        return False
    for key in ("requires_reliability_metadata", "consume_reliability_metadata", "observability_aware"):
        if bool(model_cfg.get(key, False)):
            return True
    if _model_cfg_has_predictive_gps_query_pooler(model_cfg):
        return True
    if _model_cfg_has_geometry_prior(model_cfg):
        return True
    if _model_cfg_has_safe_reranker(model_cfg):
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
    fusion = model_cfg.get("observability_aware_fusion", model_cfg.get("reliability_metadata"))
    if isinstance(fusion, Mapping):
        return bool(fusion.get("strict", fusion.get("require_fields", True)))
    if _model_cfg_has_predictive_gps_query_pooler(model_cfg):
        return bool(model_cfg.get("strict_reliability_metadata", False))
    if _model_cfg_has_geometry_prior(model_cfg):
        return bool(model_cfg.get("strict_reliability_metadata", False))
    if _model_cfg_has_safe_reranker(model_cfg):
        return bool(model_cfg.get("strict_reliability_metadata", False))
    return bool(model_cfg.get("strict_reliability_metadata", True))


def _model_cfg_has_predictive_gps_query_pooler(model_cfg: Mapping[str, Any]) -> bool:
    encoders = model_cfg.get("encoders")
    if not isinstance(encoders, Mapping):
        return False
    for encoder in encoders.values():
        if not isinstance(encoder, Mapping):
            continue
        pooler = encoder.get("pooler")
        pooler_type = str(pooler.get("type", "")) if isinstance(pooler, Mapping) else str(encoder.get("pooling", ""))
        if pooler_type.strip().lower() in {
            "predictive_gps_query",
            "predictive_gps_query++",
            "predictive_gps_query_plus_plus",
            "gps_query_plus_plus",
        }:
            return True
    return False


def _model_cfg_has_geometry_prior(model_cfg: Mapping[str, Any]) -> bool:
    raw = model_cfg.get("geometry_prior")
    if raw is True:
        return True
    if isinstance(raw, Mapping):
        return bool(raw.get("enabled", True))
    return False


def _model_cfg_has_safe_reranker(model_cfg: Mapping[str, Any]) -> bool:
    raw = model_cfg.get("reranker")
    if raw is True:
        return True
    if isinstance(raw, Mapping):
        return bool(raw.get("enabled", True))
    return False


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


def _benchmark_condition_metadata(batch: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(batch.get("benchmark_condition_metadata"), Mapping):
        return dict(batch["benchmark_condition_metadata"])
    metadata = batch.get("metadata")
    if isinstance(metadata, Mapping):
        perturbation = metadata.get("benchmark_perturbation")
        if isinstance(perturbation, Mapping):
            return dict(perturbation)
    difficulty = batch.get("difficulty")
    if isinstance(difficulty, Mapping):
        return {
            "difficulty_condition": difficulty.get("condition"),
            "difficulty_profile_digest": difficulty.get("profile_digest"),
        }
    return {}


def prepare_gps_bev_xy_inputs(
    batch: dict[str, torch.Tensor],
    *,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    non_blocking: bool = False,
) -> torch.Tensor:
    if "gps_bev_xy" not in batch:
        raise ValueError("GPS BEV XY input is required but batch does not contain a 'gps_bev_xy' field.")
    xy = batch["gps_bev_xy"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    if xy.ndim == 2:
        xy = xy.unsqueeze(0)
    if xy.ndim != 3 or int(xy.shape[-1]) != 2:
        raise ValueError(f"gps_bev_xy must have shape [B, T, 2], got {tuple(xy.shape)}.")
    xy = xy[:, -seq_length:, :]
    xy = _left_pad_temporal_sequence(xy, seq_length)
    batch_size, _, feature_dim = xy.shape
    pad_steps = max(num_pred - 1, 0)
    zeros = torch.zeros(batch_size, pad_steps, feature_dim, dtype=xy.dtype, device=device)
    return torch.cat([xy, zeros], dim=1)


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
    if "csi" not in batch:
        raise ValueError("CSI input is required but batch does not contain a 'csi' field.")
    csi = batch["csi"].to(device, non_blocking=non_blocking)
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
    geometry_batch: torch.Tensor | None = None,
    geometry_mask: torch.Tensor | None = None,
    gps_bev_xy_batch: torch.Tensor | None = None,
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
            "geometry_batch": geometry_batch,
            "geometry_mask": geometry_mask,
            "gps_bev_xy_batch": gps_bev_xy_batch,
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
