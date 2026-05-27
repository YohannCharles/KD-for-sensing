from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.engine.hist_beam_labels import ensure_horizon_shape, hist_beam_labels


@dataclass(frozen=True)
class HistBeamLossResult:
    total: torch.Tensor
    hierarchical: torch.Tensor
    coarse: torch.Tensor
    fine: torch.Tensor
    flat: torch.Tensor
    orthogonality: torch.Tensor
    shared_scene: torch.Tensor
    private_scene: torch.Tensor
    diagnostics: dict[str, float]


def hist_beam_enabled(cfg: dict[str, Any], model_output: Any | None = None) -> bool:
    hist_cfg = cfg.get("hist_beam")
    if isinstance(hist_cfg, dict) and hist_cfg.get("enabled") is not False:
        return True
    model_cfg = cfg.get("model", {}).get("student", {})
    if model_cfg.get("type") == "hist_beam_fusion":
        return True
    if isinstance(model_output, dict) and "coarse_logits" in model_output and "fine_logits" in model_output:
        return True
    return False


def compute_hist_beam_loss(
    output: dict[str, Any],
    labels: torch.Tensor,
    *,
    cfg: dict[str, Any] | None = None,
    scene_labels: torch.Tensor | None = None,
    num_classes: int | None = None,
    group_size: int | None = None,
    ignore_index: int = -100,
) -> HistBeamLossResult:
    cfg = cfg or {}
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    model_cfg = cfg.get("model", {}).get("student", {}) if isinstance(cfg.get("model"), dict) else {}
    reference = _reference_tensor(output)
    num_classes = int(num_classes or hist_cfg.get("num_classes", model_cfg.get("num_classes", reference.shape[-1])))
    group_size = int(group_size or hist_cfg.get("group_size", model_cfg.get("group_size", 8)))
    weights = _loss_weights(hist_cfg, model_cfg)
    zero = reference.sum() * 0.0

    flat_logits = _first_tensor(output, ("flat_logits", "beam_logits", "logits"))
    coarse_logits = _tensor(output, "coarse_logits")
    fine_logits = _tensor(output, "fine_logits")
    if coarse_logits is None or fine_logits is None:
        flat = _flat_ce(flat_logits, labels, num_classes=num_classes, ignore_index=ignore_index) if flat_logits is not None else zero
        diagnostics = {"hist/loss_flat": _scalar(flat), "hist/loss_total": _scalar(flat)}
        return HistBeamLossResult(flat, zero, zero, zero, flat, zero, zero, zero, diagnostics)

    ensure_horizon_shape("coarse_logits", coarse_logits, labels)
    ensure_horizon_shape("fine_logits", fine_logits, labels)
    coarse_target, fine_target = hist_beam_labels(
        labels,
        num_classes=num_classes,
        group_size=group_size,
        ignore_index=ignore_index,
    )
    coarse_loss = F.cross_entropy(
        coarse_logits.reshape(-1, coarse_logits.shape[-1]),
        coarse_target.reshape(-1),
        ignore_index=ignore_index,
    )
    fine_loss = _fine_loss_for_true_group(fine_logits, coarse_target, fine_target, ignore_index=ignore_index)
    hierarchical = coarse_loss + fine_loss
    flat = (
        _flat_ce(flat_logits, labels, num_classes=num_classes, ignore_index=ignore_index)
        if flat_logits is not None and weights["flat"] > 0
        else zero
    )
    orth = _orthogonality_loss(
        _tensor(output, "shared_representation"),
        _tensor(output, "private_representation"),
        zero,
    )
    shared_scene = _scene_ce(_tensor(output, "shared_scene_logits"), scene_labels, zero)
    private_scene = _scene_ce(_tensor(output, "private_scene_logits"), scene_labels, zero)
    total = (
        weights["hierarchical"] * hierarchical
        + weights["flat"] * flat
        + weights["orthogonality"] * orth
        + weights["scene_confusion"] * shared_scene
        + weights["scene_private"] * private_scene
    )
    diagnostics = {
        "hist/loss_total": _scalar(total),
        "hist/loss_hierarchical": _scalar(hierarchical),
        "hist/loss_coarse": _scalar(coarse_loss),
        "hist/loss_fine": _scalar(fine_loss),
        "hist/loss_flat": _scalar(flat),
        "hist/loss_orthogonality": _scalar(orth),
        "hist/loss_shared_scene": _scalar(shared_scene),
        "hist/loss_private_scene": _scalar(private_scene),
    }
    return HistBeamLossResult(
        total=total,
        hierarchical=hierarchical,
        coarse=coarse_loss,
        fine=fine_loss,
        flat=flat,
        orthogonality=orth,
        shared_scene=shared_scene,
        private_scene=private_scene,
        diagnostics=diagnostics,
    )


def entropy_minimization_loss(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean()


def prototype_consistency_loss(
    representation: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    confidence_threshold: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if representation.ndim == 3:
        representation = representation[:, 0, :]
    if prototypes.ndim != 2:
        raise ValueError(f"prototypes must have shape [G, D], got {tuple(prototypes.shape)}.")
    rep = F.normalize(representation, dim=-1)
    proto = F.normalize(prototypes.to(device=rep.device, dtype=rep.dtype), dim=-1)
    scores = rep @ proto.t()
    assignment = torch.softmax(scores, dim=-1)
    confidence, groups = assignment.max(dim=-1)
    mask = confidence >= float(confidence_threshold)
    if not torch.any(mask):
        loss = scores.sum() * 0.0
        return loss, {"prototype_coverage": 0.0, "prototype_used": 0.0}
    selected = proto[groups[mask]]
    loss = 1.0 - F.cosine_similarity(rep[mask], selected, dim=-1).mean()
    return loss, {
        "prototype_coverage": float(mask.float().mean().detach().cpu().item()),
        "prototype_used": float(mask.sum().detach().cpu().item()),
    }


def _fine_loss_for_true_group(
    fine_logits: torch.Tensor,
    coarse_target: torch.Tensor,
    fine_target: torch.Tensor,
    *,
    ignore_index: int,
) -> torch.Tensor:
    valid = coarse_target.ne(ignore_index) & fine_target.ne(ignore_index)
    if not torch.any(valid):
        return fine_logits.sum() * 0.0
    flat_fine = fine_logits.reshape(-1, fine_logits.shape[-2], fine_logits.shape[-1])
    flat_coarse = coarse_target.reshape(-1)
    flat_fine_target = fine_target.reshape(-1)
    flat_valid = valid.reshape(-1)
    selected = flat_fine[flat_valid, flat_coarse[flat_valid], :]
    return F.cross_entropy(selected, flat_fine_target[flat_valid], ignore_index=ignore_index)


def _flat_ce(logits: torch.Tensor, labels: torch.Tensor, *, num_classes: int, ignore_index: int) -> torch.Tensor:
    ensure_horizon_shape("flat_logits", logits, labels)
    return F.cross_entropy(logits.reshape(-1, num_classes), labels.reshape(-1), ignore_index=ignore_index)


def _orthogonality_loss(shared: torch.Tensor | None, private: torch.Tensor | None, zero: torch.Tensor) -> torch.Tensor:
    if shared is None or private is None:
        return zero
    if shared.ndim == 3:
        shared = shared.reshape(-1, shared.shape[-1])
    if private.ndim == 3:
        private = private.reshape(-1, private.shape[-1])
    return torch.mean(F.cosine_similarity(shared, private, dim=-1).pow(2))


def _scene_ce(logits: torch.Tensor | None, labels: torch.Tensor | None, zero: torch.Tensor) -> torch.Tensor:
    if logits is None or labels is None:
        return zero
    labels = labels.to(device=logits.device, dtype=torch.long)
    if labels.ndim > 1:
        labels = labels.reshape(labels.shape[0], -1)[:, 0]
    if logits.shape[0] != labels.shape[0]:
        return zero
    return F.cross_entropy(logits, labels)


def _loss_weights(hist_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> dict[str, float]:
    weights = hist_cfg.get("loss_weights") if isinstance(hist_cfg.get("loss_weights"), dict) else {}
    if not weights and isinstance(model_cfg.get("loss_weights"), dict):
        weights = model_cfg["loss_weights"]
    return {
        "hierarchical": float(weights.get("hierarchical", weights.get("lambda_hier", 1.0))),
        "flat": float(weights.get("flat", weights.get("lambda_flat", 0.2))),
        "orthogonality": float(weights.get("orthogonality", weights.get("lambda_orth", 0.01))),
        "scene_confusion": float(weights.get("scene_confusion", weights.get("lambda_scene_c", 0.05))),
        "scene_private": float(weights.get("scene_private", weights.get("lambda_scene_s", 0.05))),
    }


def _tensor(output: dict[str, Any], key: str) -> torch.Tensor | None:
    value = output.get(key)
    return value if torch.is_tensor(value) else None


def _first_tensor(output: dict[str, Any], keys: tuple[str, ...]) -> torch.Tensor | None:
    for key in keys:
        value = _tensor(output, key)
        if value is not None:
            return value
    return None


def _reference_tensor(output: dict[str, Any]) -> torch.Tensor:
    for key in ("logits", "beam_logits", "flat_logits", "coarse_logits"):
        value = output.get(key)
        if torch.is_tensor(value):
            return value
    raise ValueError("HiST-Beam loss requires at least one tensor output.")


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


__all__ = [
    "HistBeamLossResult",
    "compute_hist_beam_loss",
    "entropy_minimization_loss",
    "hist_beam_enabled",
    "prototype_consistency_loss",
]
