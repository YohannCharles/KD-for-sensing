from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross entropy over [B, H, C] logits with per-sample means."""

    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, H, C], got {tuple(logits.shape)}.")
    if labels.shape != logits.shape[:2]:
        raise ValueError(f"labels must have shape {tuple(logits.shape[:2])}, got {tuple(labels.shape)}.")

    batch_size, horizon, num_classes = logits.shape
    flat_loss = F.cross_entropy(
        logits.reshape(batch_size * horizon, num_classes),
        labels.reshape(batch_size * horizon),
        ignore_index=ignore_index,
        reduction="none",
    ).view(batch_size, horizon)
    valid = labels.ne(ignore_index)
    valid_count = valid.sum(dim=1).clamp_min(1)
    per_sample = (flat_loss * valid).sum(dim=1) / valid_count
    total_valid = valid.sum().clamp_min(1)
    scalar = (flat_loss * valid).sum() / total_valid
    return scalar, per_sample


def counterfactual_sequence_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return CE-only per-sample loss for counterfactual delta targets."""

    _, per_sample = sequence_cross_entropy(logits, labels, ignore_index=ignore_index)
    return per_sample


def beam_soft_label_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    sigma: float = 2.0,
    circular: bool = True,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Soft-label loss where nearby beam classes receive non-zero target mass."""

    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    if logits.ndim == 2:
        flat_logits = logits
        flat_labels = labels.reshape(-1)
    elif logits.ndim == 3:
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1)
    else:
        raise ValueError(f"logits must have shape [N, C] or [B, H, C], got {tuple(logits.shape)}.")

    valid = flat_labels.ne(ignore_index)
    if not torch.any(valid):
        return flat_logits.sum() * 0.0

    valid_logits = flat_logits[valid]
    valid_labels = flat_labels[valid].to(torch.long)
    num_classes = int(valid_logits.shape[-1])
    class_ids = torch.arange(num_classes, device=valid_logits.device, dtype=valid_logits.dtype)
    label_values = valid_labels.to(valid_logits.dtype).unsqueeze(1)
    distances = torch.abs(class_ids.unsqueeze(0) - label_values)
    if circular:
        distances = torch.minimum(distances, num_classes - distances)
    targets = torch.exp(-0.5 * (distances / float(sigma)) ** 2)
    targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_probs = F.log_softmax(valid_logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()
