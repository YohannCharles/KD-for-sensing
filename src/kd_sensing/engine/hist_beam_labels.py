from __future__ import annotations

import torch


def validate_hist_grouping(num_classes: int, group_size: int) -> int:
    classes = int(num_classes)
    group = int(group_size)
    if classes <= 0 or group <= 0 or classes % group != 0:
        raise ValueError(
            f"num_classes ({classes}) must be divisible by group_size ({group}) for HiST-Beam labels."
        )
    return classes // group


def hist_beam_labels(
    labels: torch.Tensor,
    *,
    num_classes: int = 64,
    group_size: int = 8,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_hist_grouping(num_classes, group_size)
    if labels.ndim != 2:
        raise ValueError(f"HiST-Beam labels must have shape [B, H], got {tuple(labels.shape)}.")
    target = labels.to(torch.long)
    valid = target.ne(ignore_index)
    if torch.any(valid & ((target < 0) | (target >= int(num_classes)))):
        raise ValueError(f"HiST-Beam labels must be in [0, {num_classes}) or ignore_index={ignore_index}.")
    safe = target.masked_fill(~valid, 0)
    coarse = torch.div(safe, int(group_size), rounding_mode="floor")
    fine = safe.remainder(int(group_size))
    coarse = coarse.masked_fill(~valid, ignore_index)
    fine = fine.masked_fill(~valid, ignore_index)
    return coarse, fine


def ensure_horizon_shape(name: str, tensor: torch.Tensor, labels: torch.Tensor) -> None:
    if tensor.shape[:2] != labels.shape[:2]:
        raise ValueError(
            f"{name} horizon shape {tuple(tensor.shape[:2])} must match labels shape {tuple(labels.shape[:2])}."
        )


__all__ = ["ensure_horizon_shape", "hist_beam_labels", "validate_hist_grouping"]
