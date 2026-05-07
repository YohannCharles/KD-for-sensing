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


def generate_context_marginal_masks(
    available_mask: torch.Tensor,
    *,
    num_samples: int = 1,
    min_keep: int = 1,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Sample context masks A and paired masks A union {m} for marginal contribution."""

    available = available_mask.to(torch.bool)
    if available.ndim != 2:
        raise ValueError(f"available_mask must have shape [B, K], got {tuple(available.shape)}.")
    num_samples = max(int(num_samples), 0)
    if num_samples == 0:
        return []
    min_keep = max(int(min_keep), 0)
    masks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for _ in range(num_samples):
        context = torch.zeros_like(available)
        with_target = torch.zeros_like(available)
        target = torch.zeros_like(available)
        for row in range(available.shape[0]):
            candidates = torch.nonzero(available[row], as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            target_idx = candidates[torch.randint(candidates.numel(), (1,), device=available.device).item()]
            target[row, target_idx] = True
            context_candidates = candidates[candidates != target_idx]
            required = min(min_keep, int(context_candidates.numel()))
            if context_candidates.numel() > 0:
                max_keep = int(context_candidates.numel())
                keep_count = required
                if max_keep > required:
                    keep_count = int(
                        torch.randint(
                            required,
                            max_keep + 1,
                            (1,),
                            device=available.device,
                        ).item()
                    )
                order = torch.randperm(context_candidates.numel(), device=available.device)
                context[row, context_candidates[order[:keep_count]]] = True
            with_target[row] = context[row] | target[row]
        if torch.any(target):
            masks.append((context, with_target, target))
    return masks


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


def loss_delta_to_binary_gate_target(
    delta: torch.Tensor,
    supervision_mask: torch.Tensor,
    *,
    ignore_delta_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map CE deltas to binary reliability targets and an ignore-band valid mask."""

    if delta.ndim != 1:
        raise ValueError(f"delta must have shape [B], got {tuple(delta.shape)}.")
    mask = supervision_mask.to(torch.bool)
    if mask.ndim != 2 or mask.shape[0] != delta.shape[0]:
        raise ValueError("supervision_mask must have shape [B, K] aligned with delta.")
    eps = float(ignore_delta_eps)
    if eps < 0.0:
        raise ValueError(f"ignore_delta_eps must be non-negative, got {ignore_delta_eps}.")
    valid = delta.abs().gt(eps).unsqueeze(1) & mask
    target = delta.gt(eps).to(delta.dtype).unsqueeze(1).expand_as(mask)
    return target * mask.to(delta.dtype), valid


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
