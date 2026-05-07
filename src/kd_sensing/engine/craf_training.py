from __future__ import annotations

import torch
import torch.nn.functional as F


def generate_modality_dropout_mask(
    available_mask: torch.Tensor,
    *,
    drop_prob: float = 0.0,
    min_keep: int = 1,
) -> torch.Tensor:
    """Return a boolean keep mask for modality dropout."""

    available = available_mask.to(torch.bool)
    if available.ndim != 2:
        raise ValueError(f"available_mask must have shape [B, K], got {tuple(available.shape)}.")
    if not 0.0 <= float(drop_prob) <= 1.0:
        raise ValueError(f"drop_prob must be in [0, 1], got {drop_prob}.")
    keep = available & (torch.rand(available.shape, device=available.device) >= float(drop_prob))
    min_keep = max(int(min_keep), 0)
    if min_keep > 0:
        keep = _enforce_min_keep(keep, available, min_keep=min_keep)
    return keep


def generate_counterfactual_drop_masks(
    available_mask: torch.Tensor,
    *,
    mode: str = "sample_one",
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build counterfactual keep masks and one-hot dropped-modality masks."""

    available = available_mask.to(torch.bool)
    if available.ndim != 2:
        raise ValueError(f"available_mask must have shape [B, K], got {tuple(available.shape)}.")
    batch_size, modality_count = available.shape
    if mode == "sample_one":
        keep = available.clone()
        dropped = torch.zeros_like(available)
        for row in range(batch_size):
            candidates = torch.nonzero(available[row], as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            choice = candidates[torch.randint(candidates.numel(), (1,), device=available.device).item()]
            keep[row, choice] = False
            dropped[row, choice] = True
        return [(keep, dropped)]
    if mode == "leave_one_out":
        masks: list[tuple[torch.Tensor, torch.Tensor]] = []
        for modality_idx in range(modality_count):
            dropped = torch.zeros_like(available)
            dropped[:, modality_idx] = available[:, modality_idx]
            if not torch.any(dropped):
                continue
            keep = available.clone()
            keep[:, modality_idx] = False
            masks.append((keep, dropped))
        return masks
    raise ValueError("counterfactual mode must be 'sample_one' or 'leave_one_out'.")


def loss_delta_to_gate_target(
    full_loss: torch.Tensor,
    drop_loss: torch.Tensor,
    dropped_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    target_floor: float = 0.0,
    target_ceiling: float = 1.0,
) -> torch.Tensor:
    """Map per-sample loss degradation to modality gate targets."""

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    if full_loss.shape != drop_loss.shape:
        raise ValueError("full_loss and drop_loss must have the same shape.")
    dropped = dropped_mask.to(torch.bool)
    if dropped.ndim != 2 or dropped.shape[0] != full_loss.shape[0]:
        raise ValueError("dropped_mask must have shape [B, K] aligned with full_loss.")
    contribution = torch.sigmoid((drop_loss - full_loss) / float(temperature))
    contribution = contribution.clamp(float(target_floor), float(target_ceiling))
    return dropped.to(contribution.dtype) * contribution.unsqueeze(1)


def masked_gate_mse_loss(
    reliability: torch.Tensor,
    target: torch.Tensor,
    supervision_mask: torch.Tensor,
) -> torch.Tensor:
    mask = supervision_mask.to(torch.bool)
    if not torch.any(mask):
        return reliability.sum() * 0.0
    return F.mse_loss(reliability[mask], target.detach()[mask])


def _enforce_min_keep(keep: torch.Tensor, available: torch.Tensor, *, min_keep: int) -> torch.Tensor:
    keep = keep.clone()
    for row in range(keep.shape[0]):
        available_indices = torch.nonzero(available[row], as_tuple=False).flatten()
        if available_indices.numel() == 0:
            continue
        current = int(keep[row].sum().item())
        required = min(min_keep, int(available_indices.numel()))
        if current >= required:
            continue
        dropped_available = available_indices[~keep[row, available_indices]]
        if dropped_available.numel() == 0:
            continue
        restore_order = torch.randperm(dropped_available.numel(), device=keep.device)
        restore = dropped_available[restore_order[: required - current]]
        keep[row, restore] = True
    return keep
