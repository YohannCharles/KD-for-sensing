from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from kd_sensing.engine.runtime import run_model_step
from kd_sensing.engine.training_extensions import BatchState, ExtensionContext, LossBundle, TrainingExtension


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


class MarfTrainingExtension(TrainingExtension):
    name = "marf"

    def after_forward(
        self,
        context: ExtensionContext,
        state: Any,
        batch_state: BatchState,
    ) -> LossBundle | None:
        del state
        losses = compute_marf_extra_losses(
            context.cfg,
            context.student_model,
            context.task,
            batch_state.batch,
            model_cfg=context.model_cfg["student"],
            seq_length=context.seq_length_student,
            num_pred=context.num_pred,
            num_classes=context.num_classes,
            labels=batch_state.labels,
            student_outputs=batch_state.student_logits,
            diagnostics=batch_state.student_output.diagnostics,
            task_criterion=context.task_criterion,
            device=context.device,
            non_blocking=context.non_blocking,
        )
        return LossBundle(
            total=losses["total"],
            components={
                "marf_residual_norm": losses["residual_norm"],
                "marf_prior_regularization": losses["prior_regularization"],
                "marf_anchor_entropy": losses["anchor_entropy"],
                "marf_subset_ce": losses["subset_ce"],
                "marf_subset_kd": losses["subset_kd"],
            },
            diagnostics=dict(losses.get("_diagnostics", {})),
        )


def compute_marf_extra_losses(
    cfg: dict,
    model,
    task: str,
    batch: dict[str, torch.Tensor],
    *,
    model_cfg: dict,
    seq_length: int,
    num_pred: int,
    num_classes: int,
    labels: torch.Tensor,
    student_outputs: torch.Tensor,
    diagnostics: dict,
    task_criterion,
    device: torch.device,
    non_blocking: bool,
) -> dict[str, torch.Tensor | dict[str, float]]:
    zero = student_outputs.sum() * 0.0
    scalar_diagnostics: dict[str, float] = {}
    losses = {
        "total": zero,
        "residual_norm": zero,
        "prior_regularization": zero,
        "anchor_entropy": zero,
        "subset_ce": zero,
        "subset_kd": zero,
        "_diagnostics": scalar_diagnostics,
    }
    if not getattr(model, "supports_marf_routing", False):
        return losses

    scalar_diagnostics.update(marf_scalar_diagnostics(diagnostics))
    loss_cfg = cfg.get("loss", {}).get("marf", {})
    residual_cfg = loss_cfg.get("residual_norm", {})
    residual_weight = float(residual_cfg.get("weight", 0.0))
    residual_enabled = bool(residual_cfg.get("enabled", residual_weight > 0.0)) and residual_weight > 0.0
    scalar_diagnostics["loss/marf_residual_norm_weight"] = residual_weight if residual_enabled else 0.0
    if residual_enabled and torch.is_tensor(diagnostics.get("residual_delta")):
        losses["residual_norm"] = marf_residual_norm_loss(
            diagnostics["residual_delta"],
            diagnostics.get("residual_weights"),
            diagnostics.get("effective_modality_mask"),
        )
        losses["total"] = losses["total"] + residual_weight * losses["residual_norm"]
        scalar_diagnostics["loss/marf_residual_norm"] = float(losses["residual_norm"].detach().cpu().item())

    prior_cfg = loss_cfg.get("prior_regularization", cfg.get("loss", {}).get("prior_regularization", {}))
    prior_weight = float(prior_cfg.get("weight", 0.0))
    prior_enabled = bool(prior_cfg.get("enabled", prior_weight > 0.0)) and prior_weight > 0.0
    scalar_diagnostics["loss/marf_prior_regularization_weight"] = prior_weight if prior_enabled else 0.0
    if prior_enabled and torch.is_tensor(diagnostics.get("anchor_weights")) and torch.is_tensor(diagnostics.get("prior")):
        losses["prior_regularization"] = marf_anchor_prior_regularization_loss(
            diagnostics["anchor_weights"],
            diagnostics["prior"],
            diagnostics.get("effective_modality_mask"),
            loss_type=str(prior_cfg.get("loss_type", "mse")),
        )
        losses["total"] = losses["total"] + prior_weight * losses["prior_regularization"]
        scalar_diagnostics["loss/marf_prior_regularization"] = float(
            losses["prior_regularization"].detach().cpu().item()
        )

    entropy_cfg = loss_cfg.get("anchor_entropy", {})
    entropy_weight = float(entropy_cfg.get("weight", 0.0))
    entropy_enabled = bool(entropy_cfg.get("enabled", entropy_weight > 0.0)) and entropy_weight > 0.0
    scalar_diagnostics["loss/marf_anchor_entropy_weight"] = entropy_weight if entropy_enabled else 0.0
    if entropy_enabled and torch.is_tensor(diagnostics.get("anchor_weights")):
        entropy_value = marf_anchor_entropy(diagnostics["anchor_weights"], diagnostics.get("effective_modality_mask"))
        losses["anchor_entropy"] = entropy_value
        sign = -1.0 if bool(entropy_cfg.get("maximize", True)) else 1.0
        losses["total"] = losses["total"] + sign * entropy_weight * entropy_value
        scalar_diagnostics["loss/marf_anchor_entropy"] = float(entropy_value.detach().cpu().item())

    subset_cfg = cfg.get("training", {}).get("subset_training", {})
    subset_enabled = bool(subset_cfg.get("enabled", False))
    if not subset_enabled:
        scalar_diagnostics["loss/marf_subset_ce"] = 0.0
        scalar_diagnostics["loss/marf_subset_kd"] = 0.0
        return losses
    if task != "fusion":
        raise ValueError("training.subset_training.enabled=true requires experiment.task=fusion.")
    if not getattr(model, "supports_force_modality_mask", False):
        raise ValueError("training.subset_training.enabled=true requires force_modality_mask support.")

    modes = subset_cfg.get("modes") or subset_cfg.get("subsets") or []
    if isinstance(modes, str):
        modes = [modes]
    if not modes:
        return losses
    available = diagnostics.get("effective_modality_mask")
    if not torch.is_tensor(available):
        available = torch.ones(
            labels.shape[0],
            len(model_cfg.get("modalities", getattr(model, "modalities", ("image", "radar")))),
            dtype=torch.bool,
            device=device,
        )
    prior = marf_prior_vector(diagnostics, available.shape[1], device=device)
    sampler = ModalitySubsetSampler(
        model_cfg.get("modalities", getattr(model, "modalities", ("image", "radar"))),
        prior,
        top_prior_k=int(subset_cfg.get("top_prior_k", 2)),
        min_keep=int(subset_cfg.get("min_keep", 1)),
        random_keep_prob=float(subset_cfg.get("random_keep_prob", 0.5)),
    )
    ce_weight = float(subset_cfg.get("ce_weight", loss_cfg.get("subset_ce", {}).get("weight", 0.0)))
    kd_weight = float(subset_cfg.get("kd_weight", loss_cfg.get("subset_kd", {}).get("weight", 0.0)))
    temperature = float(subset_cfg.get("temperature", loss_cfg.get("subset_kd", {}).get("temperature", 3.0)))
    ignore_index = int(subset_cfg.get("ignore_index", -100))
    subset_ce_losses = []
    subset_kd_losses = []
    max_subsets = int(subset_cfg.get("max_subsets_per_batch", len(modes)))
    for mode in list(modes)[: max(max_subsets, 0)]:
        spec = sampler.sample(str(mode), available_mask=available.detach(), device=device)
        if not torch.any(spec.mask):
            continue
        if str(mode) == "all" and torch.equal(spec.mask, available.detach()):
            subset_outputs = student_outputs
        else:
            subset_step = run_model_step(
                model,
                task,
                batch,
                model_cfg=model_cfg,
                seq_length=seq_length,
                num_pred=num_pred,
                device=device,
                non_blocking=non_blocking,
                force_modality_mask=spec.mask,
            )
            subset_outputs = subset_step.logits
        if ce_weight > 0.0:
            subset_ce_losses.append(task_criterion(subset_outputs.reshape(-1, num_classes), labels.flatten()))
        if kd_weight > 0.0:
            subset_kd_losses.append(
                all_to_subset_kl_loss(
                    subset_outputs,
                    student_outputs.detach(),
                    labels,
                    temperature=temperature,
                    ignore_index=ignore_index,
                )
            )
    if subset_ce_losses:
        losses["subset_ce"] = torch.stack(subset_ce_losses).mean()
        losses["total"] = losses["total"] + ce_weight * losses["subset_ce"]
    if subset_kd_losses:
        losses["subset_kd"] = torch.stack(subset_kd_losses).mean()
        losses["total"] = losses["total"] + kd_weight * losses["subset_kd"]
    scalar_diagnostics["loss/marf_subset_ce_weight"] = ce_weight if subset_ce_losses else 0.0
    scalar_diagnostics["loss/marf_subset_kd_weight"] = kd_weight if subset_kd_losses else 0.0
    scalar_diagnostics["loss/marf_subset_ce"] = float(losses["subset_ce"].detach().cpu().item())
    scalar_diagnostics["loss/marf_subset_kd"] = float(losses["subset_kd"].detach().cpu().item())
    return losses


def marf_scalar_diagnostics(diagnostics: dict) -> dict[str, float]:
    anchor = diagnostics.get("anchor_weights")
    residual = diagnostics.get("residual_weights")
    prior = diagnostics.get("prior")
    mask = diagnostics.get("effective_modality_mask")
    modalities = diagnostics.get("modalities")
    if not torch.is_tensor(anchor) or anchor.ndim != 3:
        return {}
    modality_names = _diagnostic_modalities(modalities, anchor.shape[-1])
    if torch.is_tensor(mask):
        available = mask.detach().to(device=anchor.device, dtype=torch.bool)
    else:
        available = torch.ones(anchor.shape[0], anchor.shape[-1], dtype=torch.bool, device=anchor.device)
    scalars: dict[str, float] = {}
    for idx, modality in enumerate(modality_names):
        modality_mask = available[:, idx]
        if torch.any(modality_mask):
            values = anchor[:, :, idx][modality_mask]
            scalars[f"marf/anchor_mean/{modality}"] = float(values.detach().float().mean().cpu().item())
            for horizon_idx in range(anchor.shape[1]):
                horizon_values = anchor[:, horizon_idx, idx][modality_mask]
                scalars[f"marf/anchor_h{horizon_idx}/{modality}"] = float(
                    horizon_values.detach().float().mean().cpu().item()
                )
            if torch.is_tensor(residual) and residual.ndim == 3:
                residual_values = residual[:, :, idx][modality_mask]
                scalars[f"marf/residual_mean/{modality}"] = float(
                    residual_values.detach().float().mean().cpu().item()
                )
        if torch.is_tensor(prior):
            prior_values = prior[:, idx] if prior.ndim == 2 else prior[idx].view(1).expand(anchor.shape[0])
            scalars[f"marf/prior/{modality}"] = float(prior_values.detach().float().mean().cpu().item())
    return scalars


def marf_prior_vector(diagnostics: dict, modality_count: int, *, device: torch.device) -> torch.Tensor:
    prior = diagnostics.get("prior")
    if torch.is_tensor(prior):
        values = prior.detach()
        if values.ndim == 2:
            values = values.mean(dim=0)
        return values.to(device=device, dtype=torch.float32).flatten()
    return torch.full((int(modality_count),), 1.0 / max(int(modality_count), 1), dtype=torch.float32, device=device)


def _diagnostic_modalities(modalities, modality_count: int) -> list[str]:
    if not isinstance(modalities, (tuple, list)) or len(modalities) != modality_count:
        return [f"modality_{idx}" for idx in range(modality_count)]
    return [str(modality) for modality in modalities]


_compute_marf_extra_losses = compute_marf_extra_losses


__all__ = [
    "MarfTrainingExtension",
    "ModalitySubsetSampler",
    "ModalitySubsetSpec",
    "all_to_subset_kl_loss",
    "compute_marf_extra_losses",
    "marf_anchor_entropy",
    "marf_anchor_prior_regularization_loss",
    "marf_prior_vector",
    "marf_residual_norm_loss",
    "marf_scalar_diagnostics",
]
