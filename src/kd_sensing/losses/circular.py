from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def circular_soft_target(
    target: torch.Tensor,
    *,
    num_beams: int = 64,
    sigma: float = 2.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Gaussian soft labels in circular beam space."""
    beams = _positive_num_beams(num_beams)
    sigma_value = max(float(sigma), 1e-6)
    target_tensor = torch.as_tensor(target, dtype=torch.long)
    classes = torch.arange(beams, device=target_tensor.device, dtype=torch.long)
    valid = target_tensor.ne(int(ignore_index)) & target_tensor.ge(0) & target_tensor.lt(beams)
    safe_target = target_tensor.clamp(min=0).remainder(beams)
    diff = torch.abs(classes.view(*((1,) * target_tensor.ndim), beams) - safe_target.unsqueeze(-1))
    dist = torch.minimum(diff, torch.as_tensor(beams, device=diff.device, dtype=diff.dtype) - diff).to(torch.float32)
    weights = torch.exp(-(dist**2) / (2.0 * sigma_value * sigma_value))
    weights = weights * valid.unsqueeze(-1).to(weights.dtype)
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return weights / denom


def circular_soft_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    sigma: float = 2.0,
    class_weight: torch.Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    soft = circular_soft_target(
        target.to(device=logits.device),
        num_beams=int(logits.shape[-1]),
        sigma=float(sigma),
        ignore_index=int(ignore_index),
    )
    loss = _weighted_soft_ce(logits, soft, class_weight=class_weight)
    valid = target.to(device=logits.device).ne(int(ignore_index))
    return _reduce(loss, valid, reduction=reduction)


def focal_circular_soft_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    sigma: float = 2.0,
    gamma: float = 2.0,
    class_weight: torch.Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    soft = circular_soft_target(
        target.to(device=logits.device),
        num_beams=int(logits.shape[-1]),
        sigma=float(sigma),
        ignore_index=int(ignore_index),
    )
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    pt = (probs * soft).sum(dim=-1).clamp_min(1e-8)
    per_class = -soft * log_probs
    if class_weight is not None:
        per_class = per_class * class_weight.to(device=logits.device, dtype=per_class.dtype).view(
            *((1,) * (per_class.ndim - 1)),
            -1,
        )
    loss = ((1.0 - pt) ** float(gamma)) * per_class.sum(dim=-1)
    valid = target.to(device=logits.device).ne(int(ignore_index))
    return _reduce(loss, valid, reduction=reduction)


def class_balanced_weights(
    labels,
    *,
    num_classes: int = 64,
    mode: str = "none",
    beta: float = 0.999,
    normalize: str = "mean_one",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute class weights and metadata from a training label histogram."""
    beams = _positive_num_beams(num_classes)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
    valid = labels_tensor.ge(0) & labels_tensor.lt(beams)
    histogram = torch.bincount(labels_tensor[valid], minlength=beams).to(torch.float32)
    mode_key = str(mode or "none").strip().lower()
    if mode_key == "none":
        weights = torch.ones(beams, dtype=torch.float32)
    elif mode_key == "inverse_freq":
        weights = torch.where(histogram > 0, 1.0 / histogram.clamp_min(1.0), torch.zeros_like(histogram))
    elif mode_key == "inverse_sqrt_freq":
        weights = torch.where(histogram > 0, 1.0 / torch.sqrt(histogram.clamp_min(1.0)), torch.zeros_like(histogram))
    elif mode_key == "effective_num":
        beta_value = min(max(float(beta), 0.0), 0.999999)
        effective = 1.0 - torch.pow(torch.full_like(histogram, beta_value), histogram)
        weights = torch.where(histogram > 0, (1.0 - beta_value) / effective.clamp_min(1e-12), torch.zeros_like(histogram))
    else:
        raise ValueError("loss.class_weight must be one of none, inverse_freq, inverse_sqrt_freq, or effective_num.")
    if str(normalize or "mean_one") == "mean_one":
        active = weights[histogram > 0]
        if active.numel() > 0:
            weights = weights / active.mean().clamp_min(1e-12)
    metadata = {
        "class_weight_mode": mode_key,
        "beta": float(beta),
        "normalize": str(normalize or "mean_one"),
        "fit_split": "train",
        "label_histogram": [int(item) for item in histogram.to(torch.long).tolist()],
        "weights": [float(item) for item in weights.tolist()],
        "num_valid_labels": int(valid.sum().item()),
    }
    return weights, metadata


def _weighted_soft_ce(
    logits: torch.Tensor,
    soft: torch.Tensor,
    *,
    class_weight: torch.Tensor | None,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    per_class = -soft * log_probs
    if class_weight is not None:
        per_class = per_class * class_weight.to(device=logits.device, dtype=per_class.dtype).view(
            *((1,) * (per_class.ndim - 1)),
            -1,
        )
    return per_class.sum(dim=-1)


def _reduce(loss: torch.Tensor, valid: torch.Tensor, *, reduction: str) -> torch.Tensor:
    valid = valid.to(device=loss.device, dtype=torch.bool)
    masked = torch.where(valid, loss, torch.zeros_like(loss))
    if reduction == "none":
        return masked
    if reduction == "sum":
        return masked.sum()
    if reduction != "mean":
        raise ValueError("reduction must be one of none, sum, or mean.")
    return masked.sum() / valid.to(masked.dtype).sum().clamp_min(1.0)


def _positive_num_beams(num_beams: int) -> int:
    beams = int(num_beams)
    if beams <= 0:
        raise ValueError(f"num_beams must be positive, got {num_beams}.")
    return beams
