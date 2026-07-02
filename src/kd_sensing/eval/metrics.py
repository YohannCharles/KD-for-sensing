import math

import torch
import torch.nn.functional as F


def reliability_error_stats(
    logits: torch.Tensor,
    target: torch.Tensor,
    global_reliability: torch.Tensor | None = None,
    modality_reliability: torch.Tensor | None = None,
    missing_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, K], got {tuple(logits.shape)}.")
    target = target.reshape(-1).to(device=logits.device, dtype=torch.long)
    probs = F.softmax(logits, dim=1)
    confidence, pred = probs.max(dim=1)
    correct = pred.eq(target)
    stats = {
        "mean_confidence": _mean(confidence),
        "mean_global_reliability": math.nan,
        "mean_global_reliability_correct": math.nan,
        "mean_global_reliability_wrong": math.nan,
        "mean_modality_reliability": math.nan,
        "mean_available_modality_reliability": math.nan,
    }
    if global_reliability is not None:
        rel = _sample_vector(global_reliability, logits.shape[0], logits.device)
        stats["mean_global_reliability"] = _mean(rel)
        stats["mean_global_reliability_correct"] = _mean(rel[correct])
        stats["mean_global_reliability_wrong"] = _mean(rel[~correct])
    if modality_reliability is not None:
        rel_m = modality_reliability.to(device=logits.device, dtype=torch.float32).squeeze(-1)
        stats["mean_modality_reliability"] = _mean(rel_m)
        if missing_mask is not None:
            available = missing_mask.to(device=logits.device, dtype=torch.bool)
            if available.shape == rel_m.shape:
                stats["mean_available_modality_reliability"] = _mean(rel_m[available])
    return stats


def expected_calibration_error(logits: torch.Tensor, target: torch.Tensor, n_bins: int = 15) -> float:
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, K], got {tuple(logits.shape)}.")
    target = target.reshape(-1).to(device=logits.device, dtype=torch.long)
    probs = F.softmax(logits, dim=1)
    confidence, pred = probs.max(dim=1)
    correct = pred.eq(target).float()
    ece = logits.new_tensor(0.0)
    for lower in torch.linspace(0, 1, int(n_bins) + 1, device=logits.device)[:-1]:
        upper = lower + (1.0 / int(n_bins))
        in_bin = (confidence > lower) & (confidence <= upper)
        if torch.any(in_bin):
            ece = ece + in_bin.float().mean() * torch.abs(confidence[in_bin].mean() - correct[in_bin].mean())
    return float(ece.detach().cpu().item())


def _sample_vector(value: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    tensor = value.to(device=device, dtype=torch.float32)
    while tensor.ndim > 1:
        tensor = tensor.mean(dim=-1)
    if tensor.numel() != batch_size:
        raise ValueError(f"global_reliability must reduce to shape [{batch_size}], got {tuple(tensor.shape)}.")
    return tensor.reshape(batch_size)


def _mean(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return math.nan
    finite = value.detach().to(dtype=torch.float32)
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        return math.nan
    return float(finite.mean().cpu().item())
