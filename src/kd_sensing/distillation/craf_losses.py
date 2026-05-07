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


def prior_regularization_loss(
    gate: torch.Tensor,
    prior: torch.Tensor,
    modality_mask: torch.Tensor | None = None,
    *,
    loss_type: str = "mse",
) -> torch.Tensor:
    """Regularize gate values toward teacher priors on available modalities only."""

    if gate.ndim != 2:
        raise ValueError(f"gate must have shape [B, K], got {tuple(gate.shape)}.")
    if prior.ndim == 1:
        prior = prior.view(1, -1).expand_as(gate)
    elif prior.shape != gate.shape:
        raise ValueError(f"prior must have shape [K] or {tuple(gate.shape)}, got {tuple(prior.shape)}.")
    prior = prior.to(device=gate.device, dtype=gate.dtype)
    if modality_mask is None:
        mask = torch.ones_like(gate, dtype=torch.bool)
    else:
        mask = modality_mask.to(device=gate.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.view(1, -1).expand_as(gate)
        if mask.shape != gate.shape:
            raise ValueError(f"modality_mask must have shape [K] or {tuple(gate.shape)}, got {tuple(mask.shape)}.")
    if not torch.any(mask):
        return gate.sum() * 0.0
    diff = gate[mask] - prior[mask].detach()
    normalized_type = str(loss_type).lower()
    if normalized_type == "mse":
        return torch.mean(diff.square())
    if normalized_type == "l1":
        return torch.mean(diff.abs())
    raise ValueError("prior regularization loss_type must be 'mse' or 'l1'.")


def reliability_weighted_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    reliability: torch.Tensor,
    *,
    modalities: list[str] | tuple[str, ...],
    use_modalities: list[str] | tuple[str, ...] | None = None,
    temperature: float = 3.0,
    modality_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL loss over unimodal logits weighted by CRAF reliability gates."""

    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 4:
        raise ValueError(
            "student_logits and teacher_logits must share shape [B, K, H, C]; "
            f"got {tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}."
        )
    if reliability.shape != student_logits.shape[:2]:
        raise ValueError("reliability must have shape [B, K] aligned with logits.")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    modality_names = [str(name) for name in modalities]
    selected = set(str(name) for name in (use_modalities or modality_names))
    selector = torch.tensor(
        [name in selected for name in modality_names],
        device=student_logits.device,
        dtype=torch.bool,
    ).view(1, -1)
    if modality_mask is None:
        mask = selector.expand_as(reliability)
    else:
        mask = modality_mask.to(device=student_logits.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.view(1, -1).expand_as(reliability)
        mask = mask & selector
    if not torch.any(mask):
        return student_logits.sum() * 0.0
    student_log_probs = F.log_softmax(student_logits / float(temperature), dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach() / float(temperature), dim=-1)
    per_slot = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    per_modality = per_slot.mean(dim=-1) * (float(temperature) ** 2)
    weights = reliability.to(dtype=per_modality.dtype).detach().masked_fill(~mask, 0.0)
    denominator = weights.sum().clamp_min(1e-12)
    return (per_modality * weights).sum() / denominator
