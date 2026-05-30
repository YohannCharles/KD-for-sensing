from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.engine.hist_beam_labels import ensure_horizon_shape, hist_beam_labels
from kd_sensing.engine.hist_beam_residuals import circular_residual_labels, residual_target_enabled


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
    angular_smoothing: torch.Tensor
    geometry_consistency: torch.Tensor
    radio_semantic: torch.Tensor
    path_semantic: torch.Tensor
    path_regression: torch.Tensor
    diagnostics: dict[str, Any]


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
    radio_semantic_labels: torch.Tensor | None = None,
    path_semantic_labels: torch.Tensor | None = None,
    path_descriptors: torch.Tensor | None = None,
    path_descriptor_mask: torch.Tensor | None = None,
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
    if residual_target_enabled(cfg):
        residual_logits = _tensor(output, "residual_logits")
        last_beam = _tensor(output, "last_beam")
        if residual_logits is None:
            raise RuntimeError("history-anchored residual training requires residual_logits in model output.")
        if last_beam is None:
            raise RuntimeError("history-anchored residual training requires input_beam or last_beam in model output.")
        ensure_horizon_shape("residual_logits", residual_logits, labels)
        residual_labels = circular_residual_labels(
            labels,
            last_beam,
            num_classes=int(residual_logits.shape[-1]),
            ignore_index=ignore_index,
            future_field="target_beam",
            last_field="last_beam",
        )
        residual = F.cross_entropy(
            residual_logits.reshape(-1, residual_logits.shape[-1]),
            residual_labels.reshape(-1),
            ignore_index=ignore_index,
        )
        lambda_absolute_aux = float(
            hist_cfg.get(
                "lambda_absolute_aux",
                (hist_cfg.get("history_anchor", {}) if isinstance(hist_cfg.get("history_anchor"), dict) else {}).get(
                    "lambda_absolute_aux",
                    0.0,
                ),
            )
            or 0.0
        )
        absolute_aux = (
            _flat_ce(_first_tensor(output, ("beam_logits", "absolute_beam_logits", "logits")), labels, num_classes=num_classes, ignore_index=ignore_index)
            if lambda_absolute_aux > 0
            else zero
        )
        orth = _orthogonality_loss(
            _tensor(output, "shared_representation"),
            _tensor(output, "private_representation"),
            zero,
        )
        shared_scene = _scene_ce(_tensor(output, "shared_scene_logits"), scene_labels, zero)
        private_scene = _scene_ce(_tensor(output, "private_scene_logits"), scene_labels, zero)
        geometry, geometry_diag = multimodal_geometry_consistency_loss(output, zero=zero)
        radio, radio_diag = radio_semantic_ce_loss(
            _tensor(output, "radio_logits"),
            radio_semantic_labels,
            zero=zero,
            ignore_index=ignore_index,
        )
        path, path_diag = path_semantic_ce_loss(
            _tensor(output, "path_logits"),
            path_semantic_labels,
            zero=zero,
            ignore_index=ignore_index,
        )
        path_reg, path_reg_diag = path_descriptor_regression_loss(
            _tensor(output, "path_attr_pred"),
            path_descriptors,
            path_descriptor_mask,
            zero=zero,
        )
        total = (
            residual
            + lambda_absolute_aux * absolute_aux
            + weights["orthogonality"] * orth
            + weights["scene_confusion"] * shared_scene
            + weights["scene_private"] * private_scene
            + weights["geometry_consistency"] * geometry
            + weights["radio_semantic"] * radio
            + weights["path_semantic"] * path
            + weights["path_regression"] * path_reg
        )
        pred = residual_logits.argmax(dim=-1)
        valid = residual_labels.ne(ignore_index)
        residual_acc = (
            float((pred[valid] == residual_labels[valid]).float().mean().detach().cpu().item())
            if torch.any(valid)
            else 0.0
        )
        diagnostics = {
            "hist/loss_total": _scalar(total),
            "hist/loss_residual": _scalar(residual),
            "hist/loss_absolute_aux": _scalar(absolute_aux),
            "hist/lambda_absolute_aux": float(lambda_absolute_aux),
            "hist/residual_accuracy": residual_acc,
            "hist/residual_target_enabled": 1.0,
            "hist/loss_orthogonality": _scalar(orth),
            "hist/loss_shared_scene": _scalar(shared_scene),
            "hist/loss_private_scene": _scalar(private_scene),
            "hist/loss_geometry_consistency": _scalar(geometry),
            "hist/loss_radio_semantic": _scalar(radio),
            "hist/loss_path_semantic": _scalar(path),
            "hist/loss_path_regression": _scalar(path_reg),
        }
        diagnostics.update(geometry_diag)
        diagnostics.update(radio_diag)
        diagnostics.update(path_diag)
        diagnostics.update(path_reg_diag)
        return HistBeamLossResult(
            total=total,
            hierarchical=residual,
            coarse=zero,
            fine=zero,
            flat=absolute_aux,
            orthogonality=orth,
            shared_scene=shared_scene,
            private_scene=private_scene,
            angular_smoothing=zero,
            geometry_consistency=geometry,
            radio_semantic=radio,
            path_semantic=path,
            path_regression=path_reg,
            diagnostics=diagnostics,
        )

    if coarse_logits is None or fine_logits is None:
        flat = _flat_ce(flat_logits, labels, num_classes=num_classes, ignore_index=ignore_index) if flat_logits is not None else zero
        radio, radio_diag = radio_semantic_ce_loss(
            _tensor(output, "radio_logits"),
            radio_semantic_labels,
            zero=zero,
            ignore_index=ignore_index,
        )
        path, path_diag = path_semantic_ce_loss(
            _tensor(output, "path_logits"),
            path_semantic_labels,
            zero=zero,
            ignore_index=ignore_index,
        )
        path_reg, path_reg_diag = path_descriptor_regression_loss(
            _tensor(output, "path_attr_pred"),
            path_descriptors,
            path_descriptor_mask,
            zero=zero,
        )
        total = flat + weights["radio_semantic"] * radio + weights["path_semantic"] * path + weights["path_regression"] * path_reg
        diagnostics = {
            "hist/loss_flat": _scalar(flat),
            "hist/loss_radio_semantic": _scalar(radio),
            "hist/loss_path_semantic": _scalar(path),
            "hist/loss_path_regression": _scalar(path_reg),
            "hist/loss_total": _scalar(total),
        }
        diagnostics.update(radio_diag)
        diagnostics.update(path_diag)
        diagnostics.update(path_reg_diag)
        return HistBeamLossResult(total, zero, zero, zero, flat, zero, zero, zero, zero, zero, radio, path, path_reg, diagnostics)

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
    angular, angular_diag = angular_smoothing_loss(
        _first_tensor(output, ("beam_log_probs", "beam_logits", "logits")),
        labels,
        cfg=hist_cfg,
        num_classes=num_classes,
        ignore_index=ignore_index,
        zero=zero,
    )
    geometry, geometry_diag = multimodal_geometry_consistency_loss(output, zero=zero)
    radio, radio_diag = radio_semantic_ce_loss(
        _tensor(output, "radio_logits"),
        radio_semantic_labels,
        zero=zero,
        ignore_index=ignore_index,
    )
    path, path_diag = path_semantic_ce_loss(
        _tensor(output, "path_logits"),
        path_semantic_labels,
        zero=zero,
        ignore_index=ignore_index,
    )
    path_reg, path_reg_diag = path_descriptor_regression_loss(
        _tensor(output, "path_attr_pred"),
        path_descriptors,
        path_descriptor_mask,
        zero=zero,
    )
    total = (
        weights["hierarchical"] * hierarchical
        + weights["flat"] * flat
        + weights["orthogonality"] * orth
        + weights["scene_confusion"] * shared_scene
        + weights["scene_private"] * private_scene
        + weights["angular_smoothing"] * angular
        + weights["geometry_consistency"] * geometry
        + weights["radio_semantic"] * radio
        + weights["path_semantic"] * path
        + weights["path_regression"] * path_reg
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
        "hist/loss_angular_smoothing": _scalar(angular),
        "hist/loss_geometry_consistency": _scalar(geometry),
        "hist/loss_radio_semantic": _scalar(radio),
        "hist/loss_path_semantic": _scalar(path),
        "hist/loss_path_regression": _scalar(path_reg),
    }
    diagnostics.update(angular_diag)
    diagnostics.update(geometry_diag)
    diagnostics.update(radio_diag)
    diagnostics.update(path_diag)
    diagnostics.update(path_reg_diag)
    return HistBeamLossResult(
        total=total,
        hierarchical=hierarchical,
        coarse=coarse_loss,
        fine=fine_loss,
        flat=flat,
        orthogonality=orth,
        shared_scene=shared_scene,
        private_scene=private_scene,
        angular_smoothing=angular,
        geometry_consistency=geometry,
        radio_semantic=radio,
        path_semantic=path,
        path_regression=path_reg,
        diagnostics=diagnostics,
    )


def radio_semantic_ce_loss(
    radio_logits: torch.Tensor | None,
    radio_labels: torch.Tensor | None,
    *,
    zero: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if radio_logits is None:
        return zero, {
            "hist/radio_available": 0.0,
            "hist/radio_coverage": 0.0,
            "hist/radio_unavailable_reason": "radio_logits_missing",
        }
    if radio_labels is None:
        return radio_logits.sum() * 0.0, {
            "hist/radio_available": 0.0,
            "hist/radio_coverage": 0.0,
            "hist/radio_unavailable_reason": "radio_semantic_label_missing",
            "hist/radio_num_classes": int(radio_logits.shape[-1]),
        }
    labels = radio_labels.to(device=radio_logits.device, dtype=torch.long)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    ensure_horizon_shape("radio_logits", radio_logits, labels)
    valid = labels.ne(ignore_index) & labels.ge(0) & labels.lt(radio_logits.shape[-1])
    if not torch.any(valid):
        return radio_logits.sum() * 0.0, {
            "hist/radio_available": 0.0,
            "hist/radio_coverage": 0.0,
            "hist/radio_unavailable_reason": "radio_semantic_label_missing",
            "hist/radio_num_classes": int(radio_logits.shape[-1]),
        }
    safe_labels = torch.where(valid, labels, torch.full_like(labels, int(ignore_index)))
    loss = F.cross_entropy(
        radio_logits.reshape(-1, radio_logits.shape[-1]),
        safe_labels.reshape(-1),
        ignore_index=ignore_index,
    )
    pred = radio_logits.argmax(dim=-1)
    accuracy = (pred[valid] == labels[valid]).float().mean()
    return loss, {
        "hist/radio_available": 1.0,
        "hist/radio_coverage": float(valid.float().mean().detach().cpu().item()),
        "hist/radio_valid_count": float(valid.sum().detach().cpu().item()),
        "hist/radio_accuracy": float(accuracy.detach().cpu().item()),
        "hist/radio_num_classes": int(radio_logits.shape[-1]),
    }


def path_semantic_ce_loss(
    path_logits: torch.Tensor | None,
    path_labels: torch.Tensor | None,
    *,
    zero: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if path_logits is None:
        return zero, {
            "hist/path_available": 0.0,
            "hist/path_coverage": 0.0,
            "hist/path_unavailable_reason": "path_logits_missing",
        }
    if path_labels is None:
        return path_logits.sum() * 0.0, {
            "hist/path_available": 0.0,
            "hist/path_coverage": 0.0,
            "hist/path_unavailable_reason": "path_semantic_label_missing",
            "hist/path_num_classes": int(path_logits.shape[-1]),
        }
    labels = path_labels.to(device=path_logits.device, dtype=torch.long)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    ensure_horizon_shape("path_logits", path_logits, labels)
    valid = labels.ne(ignore_index) & labels.ge(0) & labels.lt(path_logits.shape[-1])
    if not torch.any(valid):
        return path_logits.sum() * 0.0, {
            "hist/path_available": 0.0,
            "hist/path_coverage": 0.0,
            "hist/path_unavailable_reason": "path_semantic_label_missing",
            "hist/path_num_classes": int(path_logits.shape[-1]),
        }
    safe_labels = torch.where(valid, labels, torch.full_like(labels, int(ignore_index)))
    loss = F.cross_entropy(
        path_logits.reshape(-1, path_logits.shape[-1]),
        safe_labels.reshape(-1),
        ignore_index=ignore_index,
    )
    pred = path_logits.argmax(dim=-1)
    accuracy = (pred[valid] == labels[valid]).float().mean()
    return loss, {
        "hist/path_available": 1.0,
        "hist/path_coverage": float(valid.float().mean().detach().cpu().item()),
        "hist/path_valid_count": float(valid.sum().detach().cpu().item()),
        "hist/path_accuracy": float(accuracy.detach().cpu().item()),
        "hist/path_num_classes": int(path_logits.shape[-1]),
    }


def path_descriptor_regression_loss(
    path_attr_pred: torch.Tensor | None,
    path_descriptors: torch.Tensor | None,
    path_descriptor_mask: torch.Tensor | None,
    *,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if path_attr_pred is None:
        return zero, {
            "hist/path_regression_available": 0.0,
            "hist/path_regression_coverage": 0.0,
            "hist/path_regression_unavailable_reason": "path_attr_pred_missing",
        }
    if path_descriptors is None:
        return path_attr_pred.sum() * 0.0, {
            "hist/path_regression_available": 0.0,
            "hist/path_regression_coverage": 0.0,
            "hist/path_regression_unavailable_reason": "path_descriptor_missing",
        }
    target = path_descriptors.to(device=path_attr_pred.device, dtype=path_attr_pred.dtype)
    if target.ndim == 2:
        target = target.unsqueeze(1)
    if target.shape != path_attr_pred.shape:
        return path_attr_pred.sum() * 0.0, {
            "hist/path_regression_available": 0.0,
            "hist/path_regression_coverage": 0.0,
            "hist/path_regression_unavailable_reason": f"path_descriptor_shape_mismatch:{tuple(target.shape)}!={tuple(path_attr_pred.shape)}",
        }
    if path_descriptor_mask is None:
        valid = torch.isfinite(target).all(dim=-1)
    else:
        valid = path_descriptor_mask.to(device=path_attr_pred.device, dtype=torch.bool)
        if valid.ndim == 1:
            valid = valid.unsqueeze(1)
    if not torch.any(valid):
        return path_attr_pred.sum() * 0.0, {
            "hist/path_regression_available": 0.0,
            "hist/path_regression_coverage": 0.0,
            "hist/path_regression_unavailable_reason": "path_descriptor_missing",
        }
    loss = F.smooth_l1_loss(path_attr_pred[valid], target[valid])
    mse = F.mse_loss(path_attr_pred[valid], target[valid])
    return loss, {
        "hist/path_regression_available": 1.0,
        "hist/path_regression_coverage": float(valid.float().mean().detach().cpu().item()),
        "hist/path_regression_valid_count": float(valid.sum().detach().cpu().item()),
        "hist/path_descriptor_mse": float(mse.detach().cpu().item()),
    }


def entropy_minimization_loss(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean()


def prototype_consistency_loss(
    representation: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    confidence_threshold: float = 0.0,
    coarse_logits: torch.Tensor | None = None,
    counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if representation.ndim == 3:
        representation = representation[:, 0, :]
    if prototypes.ndim != 2:
        raise ValueError(f"prototypes must have shape [G, D], got {tuple(prototypes.shape)}.")
    rep = F.normalize(representation, dim=-1)
    proto = F.normalize(prototypes.to(device=rep.device, dtype=rep.dtype), dim=-1)
    if coarse_logits is not None:
        if coarse_logits.ndim == 3:
            coarse_scores = coarse_logits[:, 0, :]
        else:
            coarse_scores = coarse_logits
        assignment = torch.softmax(coarse_scores.to(device=rep.device, dtype=rep.dtype), dim=-1)
        scores = assignment
    else:
        scores = rep @ proto.t()
        assignment = torch.softmax(scores, dim=-1)
    confidence, groups = assignment.max(dim=-1)
    mask = confidence >= float(confidence_threshold)
    if counts is not None:
        available = counts.to(device=rep.device).reshape(-1).gt(0)
        mask = mask & available[groups]
    if not torch.any(mask):
        loss = scores.sum() * 0.0
        return loss, {
            "prototype_coverage": 0.0,
            "prototype_used": 0.0,
            "prototype_confidence_mean": float(confidence.detach().mean().cpu().item()) if confidence.numel() else 0.0,
        }
    selected = proto[groups[mask]]
    loss = 1.0 - F.cosine_similarity(rep[mask], selected, dim=-1).mean()
    return loss, {
        "prototype_coverage": float(mask.float().mean().detach().cpu().item()),
        "prototype_used": float(mask.sum().detach().cpu().item()),
        "prototype_confidence_mean": float(confidence[mask].detach().mean().cpu().item()),
    }


def angular_smoothing_loss(
    logits: torch.Tensor | None,
    labels: torch.Tensor,
    *,
    cfg: dict[str, Any],
    num_classes: int,
    ignore_index: int,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    smoothing_cfg = cfg.get("angular_smoothing") if isinstance(cfg.get("angular_smoothing"), dict) else {}
    if not smoothing_cfg or smoothing_cfg.get("enabled") is not True or logits is None:
        return zero, {"hist/angular_smoothing_coverage": 0.0}
    sigma = float(smoothing_cfg.get("sigma", smoothing_cfg.get("temperature", 1.5)))
    topology = str(smoothing_cfg.get("topology", "linear")).lower()
    ensure_horizon_shape("angular_logits", logits, labels)
    valid = labels.ne(ignore_index)
    if not torch.any(valid):
        return zero, {"hist/angular_smoothing_coverage": 0.0, "hist/angular_smoothing_topology": 0.0}
    targets = _soft_angular_targets(
        labels[valid].to(dtype=torch.long),
        num_classes=int(num_classes),
        sigma=sigma,
        topology=topology,
        device=logits.device,
        dtype=logits.dtype,
    )
    log_probs = F.log_softmax(logits[valid], dim=-1)
    loss = F.kl_div(log_probs, targets, reduction="batchmean")
    return loss, {
        "hist/angular_smoothing_coverage": float(valid.float().mean().detach().cpu().item()),
        "hist/angular_smoothing_sigma": sigma,
        "hist/angular_smoothing_valid": float(valid.sum().detach().cpu().item()),
        "hist/angular_smoothing_circular": 1.0 if topology == "circular" else 0.0,
    }


def multimodal_geometry_consistency_loss(
    output: dict[str, Any],
    *,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    diagnostics = output.get("geometry_diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("enabled") is not True:
        return zero, {"hist/geometry_consistency_coverage": 0.0}
    geometry = _tensor(output, "geometry_representation")
    shared = _tensor(output, "shared_geometry_representation")
    if shared is None:
        shared = _tensor(output, "shared_representation")
    if geometry is None or shared is None:
        return zero, {
            "hist/geometry_consistency_coverage": float(diagnostics.get("coverage", 0.0) or 0.0),
            "hist/geometry_consistency_available": 0.0,
        }
    if geometry.ndim == 3:
        geometry = geometry[:, 0, :]
    if shared.ndim == 3:
        shared = shared[:, 0, :]
    if geometry.shape != shared.shape:
        return zero, {
            "hist/geometry_consistency_coverage": float(diagnostics.get("coverage", 0.0) or 0.0),
            "hist/geometry_consistency_available": 0.0,
        }
    loss = 1.0 - F.cosine_similarity(F.normalize(geometry, dim=-1), F.normalize(shared, dim=-1), dim=-1).mean()
    return loss, {
        "hist/geometry_consistency_coverage": float(diagnostics.get("coverage", 0.0) or 0.0),
        "hist/geometry_consistency_available": 1.0,
    }


def _soft_angular_targets(
    labels: torch.Tensor,
    *,
    num_classes: int,
    sigma: float,
    topology: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    classes = torch.arange(int(num_classes), device=device, dtype=dtype).view(1, -1)
    center = labels.to(device=device, dtype=dtype).view(-1, 1)
    distance = torch.abs(classes - center)
    if topology == "circular":
        distance = torch.minimum(distance, float(num_classes) - distance)
    weights = torch.exp(-0.5 * (distance / max(float(sigma), 1e-6)).pow(2))
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


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
        "angular_smoothing": float(weights.get("angular_smoothing", weights.get("lambda_ang", 0.0))),
        "geometry_consistency": float(weights.get("geometry_consistency", weights.get("lambda_geom", 0.0))),
        "radio_semantic": float(weights.get("radio_semantic", weights.get("lambda_radio", 0.0))),
        "path_semantic": float(weights.get("path_semantic", weights.get("lambda_path", 0.3))),
        "path_regression": float(weights.get("path_regression", weights.get("lambda_path_reg", 0.05))),
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
    "angular_smoothing_loss",
    "entropy_minimization_loss",
    "hist_beam_enabled",
    "multimodal_geometry_consistency_loss",
    "path_descriptor_regression_loss",
    "path_semantic_ce_loss",
    "prototype_consistency_loss",
    "radio_semantic_ce_loss",
]
