from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ModalitySubsetSpec:
    name: str
    mask: torch.Tensor
    modalities: tuple[str, ...]


class ModalitySubsetSampler:
    """Prior-driven modality subset sampler with no dataset-specific modality assumptions."""

    def __init__(
        self,
        modalities: Iterable[str],
        prior: dict[str, float] | list[float] | tuple[float, ...] | torch.Tensor | None = None,
        *,
        top_prior_k: int = 2,
        min_keep: int = 1,
        random_keep_prob: float = 0.5,
    ):
        self.modalities = tuple(str(name) for name in modalities)
        if not self.modalities:
            raise ValueError("ModalitySubsetSampler requires at least one modality.")
        self.modality_count = len(self.modalities)
        self.prior = _resolve_prior(prior, self.modalities)
        self.top_prior_k = max(1, int(top_prior_k))
        self.min_keep = max(1, int(min_keep))
        self.random_keep_prob = min(max(float(random_keep_prob), 0.0), 1.0)

    def sample(
        self,
        mode: str,
        *,
        available_mask: torch.Tensor | None = None,
        batch_size: int | None = None,
        device: torch.device | None = None,
    ) -> ModalitySubsetSpec:
        mode = str(mode)
        available = self._available(available_mask, batch_size=batch_size, device=device)
        if mode == "all":
            mask = available.clone()
        elif mode == "top_prior":
            mask = self._top_prior_mask(available, k=self.top_prior_k)
        elif mode == "single_best_prior":
            mask = self._top_prior_mask(available, k=1)
        elif mode == "random":
            mask = self._random_mask(available, include_top=False)
        elif mode == "random_with_top_prior":
            mask = self._random_mask(available, include_top=True)
        elif mode == "drop_one":
            mask = self._drop_one_mask(available)
        else:
            raise ValueError(
                "Unsupported MARF subset mode "
                f"'{mode}'. Expected all, top_prior, single_best_prior, random, random_with_top_prior, or drop_one."
            )
        mask = _ensure_min_keep(mask & available, available, self.min_keep)
        return ModalitySubsetSpec(mode, mask, self._modalities_from_mask(mask))

    def explicit(self, name: str, selected: Iterable[str], *, device: torch.device | None = None) -> ModalitySubsetSpec:
        selected_set = {str(item) for item in selected}
        mask = torch.tensor([name in selected_set for name in self.modalities], dtype=torch.bool, device=device)
        return ModalitySubsetSpec(str(name), mask, tuple(name for name in self.modalities if name in selected_set))

    def low_prior(
        self,
        *,
        name: str = "low_prior_only",
        k: int | None = None,
        device: torch.device | None = None,
    ) -> ModalitySubsetSpec:
        keep = int(k or max(self.modality_count - self.top_prior_k, 1))
        order = torch.argsort(self.prior, descending=False)
        selected = order[: min(max(keep, 1), self.modality_count)]
        mask = torch.zeros(self.modality_count, dtype=torch.bool, device=device)
        mask[selected.to(device=mask.device)] = True
        return ModalitySubsetSpec(str(name), mask, self._modalities_from_mask(mask))

    def _available(
        self,
        available_mask: torch.Tensor | None,
        *,
        batch_size: int | None,
        device: torch.device | None,
    ) -> torch.Tensor:
        if available_mask is None:
            rows = int(batch_size or 1)
            return torch.ones(rows, self.modality_count, dtype=torch.bool, device=device)
        available = available_mask.to(device=device, dtype=torch.bool)
        if available.ndim == 1:
            return available.view(1, -1)
        if available.ndim != 2 or available.shape[1] != self.modality_count:
            raise ValueError(
                f"available_mask must have shape [K] or [B, K] with K={self.modality_count}, "
                f"got {tuple(available.shape)}."
            )
        return available

    def _top_prior_mask(self, available: torch.Tensor, *, k: int) -> torch.Tensor:
        prior = self.prior.to(device=available.device)
        mask = torch.zeros_like(available)
        for row_idx in range(available.shape[0]):
            candidates = torch.nonzero(available[row_idx], as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            count = min(max(int(k), self.min_keep), int(candidates.numel()))
            selected = candidates[torch.argsort(prior[candidates], descending=True)[:count]]
            mask[row_idx, selected] = True
        return mask

    def _random_mask(self, available: torch.Tensor, *, include_top: bool) -> torch.Tensor:
        prior = self.prior.to(device=available.device)
        mask = torch.zeros_like(available)
        for row_idx in range(available.shape[0]):
            candidates = torch.nonzero(available[row_idx], as_tuple=False).flatten().tolist()
            if not candidates:
                continue
            selected: set[int] = set()
            if include_top:
                top = max(candidates, key=lambda idx: float(prior[idx].item()))
                selected.add(int(top))
            for idx in candidates:
                if random.random() < self.random_keep_prob:
                    selected.add(int(idx))
            min_keep = min(self.min_keep, len(candidates))
            while len(selected) < min_keep:
                selected.add(int(random.choice(candidates)))
            for idx in selected:
                mask[row_idx, idx] = True
        return mask

    def _drop_one_mask(self, available: torch.Tensor) -> torch.Tensor:
        mask = available.clone()
        for row_idx in range(available.shape[0]):
            candidates = torch.nonzero(available[row_idx], as_tuple=False).flatten().tolist()
            if len(candidates) <= self.min_keep:
                continue
            drop_idx = int(random.choice(candidates))
            mask[row_idx, drop_idx] = False
        return mask

    def _modalities_from_mask(self, mask: torch.Tensor) -> tuple[str, ...]:
        row = mask[0] if mask.ndim == 2 else mask
        return tuple(name for name, keep in zip(self.modalities, row.detach().cpu().tolist()) if keep)


def marf_residual_norm_loss(
    residual_delta: torch.Tensor,
    residual_weights: torch.Tensor | None = None,
    modality_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if residual_delta.ndim != 4:
        raise ValueError(f"residual_delta must have shape [B, H, K, D], got {tuple(residual_delta.shape)}.")
    per_delta = residual_delta.square().mean(dim=-1)
    mask = _bhk_mask(per_delta, residual_weights=residual_weights, modality_mask=modality_mask)
    if not torch.any(mask):
        return residual_delta.sum() * 0.0
    if residual_weights is not None:
        weights = residual_weights.detach().to(device=per_delta.device, dtype=per_delta.dtype).masked_fill(~mask, 0.0)
        denom = weights.sum().clamp_min(1e-12)
        return (per_delta * weights).sum() / denom
    return per_delta[mask].mean()


def marf_anchor_prior_regularization_loss(
    anchor_weights: torch.Tensor,
    prior: torch.Tensor,
    modality_mask: torch.Tensor | None = None,
    *,
    loss_type: str = "mse",
) -> torch.Tensor:
    if anchor_weights.ndim != 3:
        raise ValueError(f"anchor_weights must have shape [B, H, K], got {tuple(anchor_weights.shape)}.")
    prior_values = _prior_batch(prior, anchor_weights)
    if modality_mask is None:
        available = torch.ones(anchor_weights.shape[0], anchor_weights.shape[2], dtype=torch.bool, device=anchor_weights.device)
    else:
        available = modality_mask.to(device=anchor_weights.device, dtype=torch.bool)
        if available.ndim == 1:
            available = available.view(1, -1).expand(anchor_weights.shape[0], -1)
    if available.shape != anchor_weights.shape[::2]:
        raise ValueError(
            f"modality_mask must have shape [K] or [B, K], got {tuple(available.shape)} for anchor weights."
        )
    prior_dist = prior_values.masked_fill(~available, 0.0)
    prior_dist = prior_dist / prior_dist.sum(dim=1, keepdim=True).clamp_min(1e-12)
    target = prior_dist.unsqueeze(1).expand_as(anchor_weights)
    mask = available.unsqueeze(1).expand_as(anchor_weights)
    if not torch.any(mask):
        return anchor_weights.sum() * 0.0
    diff = anchor_weights[mask] - target.detach()[mask]
    if str(loss_type).lower() == "mse":
        return diff.square().mean()
    if str(loss_type).lower() == "l1":
        return diff.abs().mean()
    raise ValueError("MARF prior regularization loss_type must be 'mse' or 'l1'.")


def marf_anchor_entropy(anchor_weights: torch.Tensor, modality_mask: torch.Tensor | None = None) -> torch.Tensor:
    if anchor_weights.ndim != 3:
        raise ValueError(f"anchor_weights must have shape [B, H, K], got {tuple(anchor_weights.shape)}.")
    probs = anchor_weights.clamp_min(1e-12)
    entropy = -(probs * probs.log()).sum(dim=-1)
    if modality_mask is None:
        return entropy.mean()
    available = modality_mask.to(device=anchor_weights.device, dtype=torch.bool)
    if available.ndim == 1:
        available = available.view(1, -1).expand(anchor_weights.shape[0], -1)
    valid = available.any(dim=1).view(-1, 1).expand_as(entropy)
    if not torch.any(valid):
        return anchor_weights.sum() * 0.0
    return entropy[valid].mean()


def all_to_subset_kl_loss(
    subset_logits: torch.Tensor,
    all_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    temperature: float = 3.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    if subset_logits.shape != all_logits.shape or subset_logits.ndim != 3:
        raise ValueError(
            "subset_logits and all_logits must share shape [B, H, C], "
            f"got {tuple(subset_logits.shape)} and {tuple(all_logits.shape)}."
        )
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    student_log_probs = F.log_softmax(subset_logits / float(temperature), dim=-1)
    teacher_probs = F.softmax(all_logits.detach() / float(temperature), dim=-1)
    per_slot = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    if labels is not None:
        valid = labels.to(device=subset_logits.device).ne(ignore_index)
        if valid.shape != subset_logits.shape[:2]:
            raise ValueError(f"labels must have shape {tuple(subset_logits.shape[:2])}, got {tuple(labels.shape)}.")
        if not torch.any(valid):
            return subset_logits.sum() * 0.0
        per_slot = per_slot[valid]
    return per_slot.mean() * (float(temperature) ** 2)


def _resolve_prior(
    prior: dict[str, float] | list[float] | tuple[float, ...] | torch.Tensor | None,
    modalities: tuple[str, ...],
) -> torch.Tensor:
    if prior is None:
        return torch.full((len(modalities),), 1.0 / max(len(modalities), 1), dtype=torch.float32)
    if torch.is_tensor(prior):
        values = prior.detach().float().flatten().cpu()
    elif isinstance(prior, dict):
        values = torch.tensor([float(prior.get(name, 0.0)) for name in modalities], dtype=torch.float32)
    else:
        values = torch.tensor([float(value) for value in prior], dtype=torch.float32)
    if values.numel() != len(modalities):
        raise ValueError(f"prior must contain {len(modalities)} values, got {values.numel()}.")
    return values


def _ensure_min_keep(mask: torch.Tensor, available: torch.Tensor, min_keep: int) -> torch.Tensor:
    result = mask.clone()
    for row_idx in range(result.shape[0]):
        candidates = torch.nonzero(available[row_idx], as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        keep = min(max(int(min_keep), 1), int(candidates.numel()))
        if int(result[row_idx].sum().item()) >= keep:
            continue
        missing = candidates[~result[row_idx, candidates]]
        add = missing[: keep - int(result[row_idx].sum().item())]
        result[row_idx, add] = True
    return result


def _bhk_mask(
    values: torch.Tensor,
    *,
    residual_weights: torch.Tensor | None,
    modality_mask: torch.Tensor | None,
) -> torch.Tensor:
    mask = torch.ones_like(values, dtype=torch.bool)
    if residual_weights is not None:
        if residual_weights.shape != values.shape:
            raise ValueError(f"residual_weights must have shape {tuple(values.shape)}, got {tuple(residual_weights.shape)}.")
        mask = mask & residual_weights.to(device=values.device).gt(0)
    if modality_mask is not None:
        available = modality_mask.to(device=values.device, dtype=torch.bool)
        if available.ndim == 1:
            available = available.view(1, 1, -1).expand_as(values)
        elif available.ndim == 2:
            available = available.unsqueeze(1).expand_as(values)
        if available.shape != values.shape:
            raise ValueError(f"modality_mask must have shape [K] or [B, K], got {tuple(modality_mask.shape)}.")
        mask = mask & available
    return mask


def _prior_batch(prior: torch.Tensor, anchor_weights: torch.Tensor) -> torch.Tensor:
    if prior.ndim == 1:
        return prior.to(device=anchor_weights.device, dtype=anchor_weights.dtype).view(1, -1).expand(
            anchor_weights.shape[0],
            -1,
        )
    if prior.ndim == 2:
        if prior.shape != (anchor_weights.shape[0], anchor_weights.shape[2]):
            raise ValueError(
                f"prior must have shape [K] or [B, K], got {tuple(prior.shape)} for anchor weights."
            )
        return prior.to(device=anchor_weights.device, dtype=anchor_weights.dtype)
    raise ValueError(f"prior must have shape [K] or [B, K], got {tuple(prior.shape)}.")


__all__ = [
    "ModalitySubsetSampler",
    "ModalitySubsetSpec",
    "all_to_subset_kl_loss",
    "marf_anchor_entropy",
    "marf_anchor_prior_regularization_loss",
    "marf_residual_norm_loss",
]
