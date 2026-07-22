from typing import Any

import torch
import torch.nn.functional as F


def bcacl_auxiliary_loss(
    *,
    features: torch.Tensor,
    private_logits: torch.Tensor,
    shared_logits: torch.Tensor,
    labels: torch.Tensor,
    observed_mask: torch.Tensor,
    lambda_private: float = 1.0,
    lambda_shared: float = 1.0,
    private_modality_weights: torch.Tensor | None = None,
    shared_modality_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    observed = _observed_mask(observed_mask)
    targets = _labels(labels, batch_size=features.shape[0])
    valid = observed & targets.ne(-100).unsqueeze(1)
    zero = features.sum() * 0.0
    private_loss, private_per_modality, private_correct = _masked_classification_loss(
        private_logits, targets, valid, zero, modality_weights=private_modality_weights
    )
    shared_loss, shared_per_modality, shared_correct = _masked_classification_loss(
        shared_logits, targets, valid, zero, modality_weights=shared_modality_weights
    )
    private_weighted = float(lambda_private) * private_loss
    shared_weighted = float(lambda_shared) * shared_loss
    total = private_weighted + shared_weighted
    return {
        "loss": total,
        "loss_private": private_loss,
        "loss_shared": shared_loss,
        "loss_private_weighted": private_weighted,
        "loss_shared_weighted": shared_weighted,
        "private_per_modality": private_per_modality,
        "shared_per_modality": shared_per_modality,
        "private_correct": private_correct,
        "shared_correct": shared_correct,
        "observed_counts": valid.sum(dim=0),
    }


def _masked_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    zero: torch.Tensor,
    *,
    modality_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    modality_count = int(valid.shape[1])
    if logits.ndim != 3 or logits.shape[:2] != valid.shape:
        raise ValueError("BCACL classification logits must have shape [B,M,K].")
    per_item = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.unsqueeze(1).expand_as(valid).reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(valid)
    if modality_weights is None:
        total = (per_item * valid).sum() / valid.sum().clamp_min(1)
    else:
        weights = modality_weights.to(device=per_item.device, dtype=per_item.dtype)
        if weights.shape != (modality_count,):
            raise ValueError("BCACL modality weights must have shape [M].")
        item_weights = valid.to(per_item) * weights.unsqueeze(0)
        total = (per_item * item_weights).sum() / item_weights.sum().clamp_min(1.0)
    losses = zero.new_zeros(modality_count)
    correct = torch.zeros(modality_count, device=valid.device, dtype=torch.long)
    predictions = logits.argmax(dim=-1)
    for modality in range(modality_count):
        rows = valid[:, modality]
        losses[modality] = (per_item[:, modality] * rows).sum() / rows.sum().clamp_min(1)
        correct[modality] = (predictions[:, modality].eq(targets) & rows).sum()
    return total, losses, correct


def _observed_mask(value: torch.Tensor) -> torch.Tensor:
    mask = value.to(dtype=torch.bool)
    if mask.ndim != 2 or not bool(mask.any(dim=1).all().item()):
        raise ValueError("BCACL observed_mask must be [B,M] with at least one observed modality per sample.")
    return mask


def _labels(value: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    labels = value[:, 0] if value.ndim == 2 else value
    if labels.ndim != 1 or labels.shape[0] != batch_size:
        raise ValueError("BCACL labels must have shape [B] or [B,H].")
    return labels.to(dtype=torch.long)


__all__ = ["bcacl_auxiliary_loss"]
