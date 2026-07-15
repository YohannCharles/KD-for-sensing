from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F


def amber_cma_analogue_loss(
    fused_features: torch.Tensor,
    modality_features: torch.Tensor,
    availability: torch.Tensor,
    sample_ids: Sequence[Any] | None,
    *,
    temperature: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align available modality anchors with same-sample fused candidates."""

    if fused_features.ndim != 2:
        raise ValueError(f"fused_features must have shape [B, D], got {tuple(fused_features.shape)}.")
    if modality_features.ndim != 3:
        raise ValueError(f"modality_features must have shape [B, M, D], got {tuple(modality_features.shape)}.")
    batch_size = int(fused_features.shape[0])
    if int(modality_features.shape[0]) != batch_size or int(modality_features.shape[-1]) != int(
        fused_features.shape[-1]
    ):
        raise ValueError(
            "fused_features and modality_features must share batch and feature dimensions, "
            f"got {tuple(fused_features.shape)} and {tuple(modality_features.shape)}."
        )
    if modality_features.device != fused_features.device:
        raise ValueError("fused_features and modality_features must be on the same device.")
    available = availability.to(device=modality_features.device, dtype=torch.bool)
    if tuple(available.shape) != tuple(modality_features.shape[:2]):
        raise ValueError(
            f"availability must have shape {tuple(modality_features.shape[:2])}, got {tuple(available.shape)}."
        )
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("AMBER CMA analogue temperature must be positive.")
    identities = _validated_identities(sample_ids, batch_size)
    identity_codes, unique_count = _identity_codes(identities, fused_features.device)

    anchor_rows = available.nonzero(as_tuple=False)[:, 0]
    anchor_count = int(anchor_rows.numel())
    diagnostics = {
        "amber_cma/anchor_count": float(anchor_count),
        "amber_cma/unique_sample_count": float(unique_count),
        "amber_cma/duplicate_candidate_count": float(batch_size - unique_count),
    }
    if anchor_count == 0:
        zero = (fused_features.sum() + modality_features.sum()) * 0.0
        return zero, {**diagnostics, "loss/amber_cma_raw": 0.0, "amber_cma/positive_candidate_mean": 0.0}

    anchors = F.normalize(modality_features[available], dim=-1)
    candidates = F.normalize(fused_features, dim=-1)
    logits = anchors @ candidates.t() / temperature
    positives = identity_codes[anchor_rows].unsqueeze(1).eq(identity_codes.unsqueeze(0))
    positive_logsumexp = torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)
    loss = (torch.logsumexp(logits, dim=1) - positive_logsumexp).mean()
    diagnostics.update(
        {
            "loss/amber_cma_raw": float(loss.detach().cpu().item()),
            "amber_cma/positive_candidate_mean": float(positives.sum(dim=1).float().mean().cpu().item()),
        }
    )
    return loss, diagnostics


def _validated_identities(sample_ids: Sequence[Any] | None, batch_size: int) -> tuple[str, ...]:
    if sample_ids is None or isinstance(sample_ids, (str, bytes)) or len(sample_ids) != batch_size:
        actual = 0 if sample_ids is None else len(sample_ids)
        raise ValueError(
            "AMBER CMA analogue requires one stable sample identity per batch item; "
            f"expected {batch_size}, got {actual}."
        )
    identities = tuple("" if item is None else str(item).strip() for item in sample_ids)
    if any(not item for item in identities):
        raise ValueError("AMBER CMA analogue requires non-empty stable sample identities.")
    return identities


def _identity_codes(identities: tuple[str, ...], device: torch.device) -> tuple[torch.Tensor, int]:
    mapping: dict[str, int] = {}
    codes = [mapping.setdefault(identity, len(mapping)) for identity in identities]
    return torch.tensor(codes, dtype=torch.long, device=device), len(mapping)


__all__ = ["amber_cma_analogue_loss"]
