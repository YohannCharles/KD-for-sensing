from typing import Any

import torch
import torch.nn.functional as F


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


__all__ = [
    "SENSITIVE_TARGET_FIELDS",
    "assert_sensitive_fields_allowed",
    "prepare_auxiliary_targets",
    "prepare_beam_power_targets",
    "prepare_beamspace_power_targets",
    "prepare_labels",
    "prepare_path_descriptors",
    "prepare_path_semantic_labels",
    "prepare_radio_semantic_labels",
    "prepare_soft_beam_targets",
]
