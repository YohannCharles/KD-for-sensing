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
    beamspace_power_labels: torch.Tensor | None = None,
    beamspace_power_mask: torch.Tensor | None = None,
    current_epoch: int | None = None,
    v7_adaptation: bool = False,
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
    if _v8_enabled(hist_cfg, model_cfg, output):
        return _compute_v8_hist_beam_loss(
            output,
            labels,
            cfg=cfg,
            weights=weights,
            num_classes=num_classes,
            ignore_index=ignore_index,
            zero=zero,
        )
    if _v7_enabled(hist_cfg, model_cfg, output):
        return _compute_v7_hist_beam_loss(
            output,
            labels,
            cfg=cfg,
            weights=weights,
            beamspace_power_labels=beamspace_power_labels,
            beamspace_power_mask=beamspace_power_mask,
            current_epoch=current_epoch,
            v7_adaptation=v7_adaptation,
            num_classes=num_classes,
            ignore_index=ignore_index,
            zero=zero,
        )
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
        return HistBeamLossResult(
            total=total,
            hierarchical=zero,
            coarse=zero,
            fine=zero,
            flat=flat,
            angular_smoothing=zero,
            geometry_consistency=zero,
            radio_semantic=radio,
            path_semantic=path,
            path_regression=path_reg,
            diagnostics=diagnostics,
        )

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


def _flat_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    ensure_horizon_shape("flat_logits", logits, labels)
    weight = None
    if class_weights is not None:
        weight = class_weights.to(device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits.reshape(-1, num_classes), labels.reshape(-1), weight=weight, ignore_index=ignore_index)


def _loss_weights(hist_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> dict[str, float]:
    weights = hist_cfg.get("loss_weights") if isinstance(hist_cfg.get("loss_weights"), dict) else {}
    if not weights and isinstance(model_cfg.get("loss_weights"), dict):
        weights = model_cfg["loss_weights"]
    retired_keys = {
        "orthogonality",
        "scene_confusion",
        "scene_private",
        "lambda_orth",
        "lambda_scene_c",
        "lambda_scene_s",
    }
    present_retired = sorted(key for key in retired_keys if key in weights)
    if present_retired:
        raise ValueError(
            "Legacy HiST-Beam shared/private decoupling loss weights are retired "
            f"and must be removed from the training config: {present_retired}. "
            "Use current baseline losses such as hierarchical, flat, radio_semantic, "
            "path_semantic, geometry_consistency, v7_*, v8_* or v9_*."
        )
    return {
        "hierarchical": float(weights.get("hierarchical", weights.get("lambda_hier", 1.0))),
        "flat": float(weights.get("flat", weights.get("lambda_flat", 0.2))),
        "angular_smoothing": float(weights.get("angular_smoothing", weights.get("lambda_ang", 0.0))),
        "geometry_consistency": float(weights.get("geometry_consistency", weights.get("lambda_geom", 0.0))),
        "radio_semantic": float(weights.get("radio_semantic", weights.get("lambda_radio", 0.0))),
        "path_semantic": float(weights.get("path_semantic", weights.get("lambda_path", 0.3))),
        "path_regression": float(weights.get("path_regression", weights.get("lambda_path_reg", 0.05))),
        "v7_shared_ce": float(weights.get("v7_shared_ce", weights.get("shared_ce", 1.0))),
        "v7_final_ce": float(weights.get("v7_final_ce", weights.get("final_ce", 1.0))),
        "v7_bsp_kl": float(weights.get("v7_bsp_kl", weights.get("bsp_kl", weights.get("beamspace_kl", 1.0)))),
        "v7_phys_kl": float(weights.get("v7_phys_kl", weights.get("phys_kl", weights.get("physical_kl", 1.0)))),
        "v7_res_l2": float(weights.get("v7_res_l2", weights.get("residual_l2", 0.01))),
        "v7_gate_l1": float(weights.get("v7_gate_l1", weights.get("gate_l1", 0.001))),
        "v7_diff": float(weights.get("v7_diff", weights.get("difference", 0.01))),
        "v8_final_ce": float(weights.get("v8_final_ce", weights.get("final_ce", 1.0))),
        "v8_prior_smooth": float(weights.get("v8_prior_smooth", weights.get("prior_smooth", 0.001))),
        "v8_sector_ce": float(weights.get("v8_sector_ce", weights.get("sector_ce", 0.2))),
        "v8_offset_ce": float(weights.get("v8_offset_ce", weights.get("offset_ce", 0.2))),
        "v9_widened_prior_marginal_kl": float(
            weights.get("v9_widened_prior_marginal_kl", weights.get("widened_prior_marginal_kl", 0.0))
        ),
    }


def gaussian_smooth_beam_prior(
    labels: torch.Tensor | list[int] | tuple[int, ...] | None,
    num_beams: int,
    sigma: float = 1.5,
    eps: float = 1e-4,
    device: torch.device | None = None,
) -> torch.Tensor:
    device = device or (labels.device if torch.is_tensor(labels) else torch.device("cpu"))
    classes = int(num_beams)
    if classes <= 0:
        raise ValueError(f"num_beams must be positive, got {classes}.")
    flat = _labels_to_1d(labels, device=device)
    valid = flat[flat.ge(0) & flat.lt(classes)]
    if valid.numel() == 0:
        return torch.full((classes,), 1.0 / classes, dtype=torch.float32, device=device)
    bins = torch.arange(classes, device=device, dtype=torch.float32).view(1, -1)
    centers = valid.to(dtype=torch.float32).view(-1, 1)
    weights = torch.exp(-0.5 * ((bins - centers) / max(float(sigma), 1e-6)).pow(2))
    prior = weights.sum(dim=0) + float(eps)
    return prior / prior.sum().clamp_min(1e-12)


def make_beam_soft_labels(
    labels: torch.Tensor,
    num_beams: int,
    sigma: float = 1.0,
    eps: float = 1e-8,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    classes = int(num_beams)
    if classes <= 0:
        raise ValueError(f"num_beams must be positive, got {classes}.")
    target = labels.to(dtype=torch.long)
    device = target.device
    out = torch.zeros(*target.shape, classes, device=device, dtype=torch.float32)
    valid = target.ne(ignore_index) & target.ge(0) & target.lt(classes)
    if not torch.any(valid):
        return out
    bins = torch.arange(classes, device=device, dtype=torch.float32).view(1, -1)
    centers = target[valid].to(dtype=torch.float32).view(-1, 1)
    weights = torch.exp(-0.5 * ((bins - centers) / max(float(sigma), 1e-6)).pow(2)) + float(eps)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    out[valid] = weights
    return out


def _compute_v8_hist_beam_loss(
    output: dict[str, Any],
    labels: torch.Tensor,
    *,
    cfg: dict[str, Any],
    weights: dict[str, float],
    num_classes: int,
    ignore_index: int,
    zero: torch.Tensor,
) -> HistBeamLossResult:
    logits_final = _first_tensor(output, ("logits_final", "beam_logits", "logits"))
    if logits_final is None:
        raise RuntimeError("V8 HiST-Beam loss requires logits_final, beam_logits, or logits.")
    ensure_horizon_shape("v8_logits_final", logits_final, labels)
    v8_cfg = _v8_loss_cfg(cfg, output)
    use_soft = bool(v8_cfg.get("use_soft_beam_label", True))
    labels_on_device = labels.to(device=logits_final.device)
    valid = labels_on_device.ne(ignore_index) & labels_on_device.ge(0) & labels_on_device.lt(int(num_classes))
    if use_soft and torch.any(valid):
        sigma = float(v8_cfg.get("soft_label_sigma", 1.0))
        soft_targets = make_beam_soft_labels(labels_on_device, num_classes, sigma=sigma, ignore_index=ignore_index).to(
            device=logits_final.device,
            dtype=logits_final.dtype,
        )
        log_probs = F.log_softmax(logits_final, dim=-1)
        final_loss = -(soft_targets[valid] * log_probs[valid]).sum(dim=-1).mean()
        final_loss_name = "soft_ce"
    elif torch.any(valid):
        final_loss = _flat_ce(logits_final, labels_on_device, num_classes=num_classes, ignore_index=ignore_index)
        final_loss_name = "hard_ce"
    else:
        final_loss = logits_final.sum() * 0.0
        final_loss_name = "unavailable"

    prior_bias = _tensor(output, "target_prior_bias")
    if prior_bias is not None and float(v8_cfg.get("loss_prior_smooth_weight", weights["v8_prior_smooth"])) > 0.0:
        bias = prior_bias[0, 0, :] if prior_bias.ndim == 3 else prior_bias.reshape(-1)
        prior_smooth = (bias[1:] - bias[:-1]).pow(2).mean() if bias.numel() > 1 else bias.sum() * 0.0
    else:
        prior_smooth = zero

    sector_logits = _tensor(output, "sector_logits")
    offset_logits = _tensor(output, "offset_logits")
    sector_loss, sector_diag = _v8_sector_loss(
        sector_logits,
        labels_on_device,
        sector_size=int(v8_cfg.get("sector_size", 8)),
        num_classes=num_classes,
        ignore_index=ignore_index,
        zero=zero,
    )
    offset_loss, offset_diag = _v8_offset_loss(
        offset_logits,
        labels_on_device,
        sector_size=int(v8_cfg.get("sector_size", 8)),
        num_classes=num_classes,
        ignore_index=ignore_index,
        zero=zero,
    )
    prior_weight = float(v8_cfg.get("loss_prior_smooth_weight", weights["v8_prior_smooth"]))
    marginal_kl, marginal_diag = _v9_widened_prior_marginal_kl(
        logits_final,
        v8_cfg,
        weight=float(weights.get("v9_widened_prior_marginal_kl", 0.0)),
        zero=zero,
    )
    total = (
        weights["v8_final_ce"] * final_loss
        + prior_weight * prior_smooth
        + weights["v8_sector_ce"] * sector_loss
        + weights["v8_offset_ce"] * offset_loss
        + marginal_kl
    )
    diagnostics = {
        "hist/loss_total": _scalar(total),
        "hist/v8/loss_final_soft_ce": _scalar(final_loss) if final_loss_name == "soft_ce" else 0.0,
        "hist/v8/loss_final_hard_ce": _scalar(final_loss) if final_loss_name == "hard_ce" else 0.0,
        "hist/v8/loss_final_type": final_loss_name,
        "hist/v8/loss_prior_smooth": _scalar(prior_smooth),
        "hist/v8/loss_sector_ce": _scalar(sector_loss),
        "hist/v8/loss_offset_ce": _scalar(offset_loss),
        "hist/v8/final_ce_weight": float(weights["v8_final_ce"]),
        "hist/v8/prior_smooth_weight": prior_weight,
        "hist/v8/sector_ce_weight": float(weights["v8_sector_ce"]),
        "hist/v8/offset_ce_weight": float(weights["v8_offset_ce"]),
        "hist/v8/soft_label_enabled": 1.0 if use_soft else 0.0,
        "hist/v8/valid_label_count": float(valid.sum().detach().cpu().item()),
        "hist/v8/target_physical_oracle_used": 0.0,
        "hist/v9/anti_collapse_loss": _scalar(marginal_kl),
        "hist/v9/anti_collapse_weight": float(weights.get("v9_widened_prior_marginal_kl", 0.0)),
        "loss/beam_soft_target": _scalar(final_loss) if final_loss_name == "soft_ce" else 0.0,
        "loss/beam_smoothing": _scalar(final_loss) if final_loss_name == "soft_ce" else 0.0,
    }
    diagnostics.update(sector_diag)
    diagnostics.update(offset_diag)
    diagnostics.update(marginal_diag)
    return HistBeamLossResult(
        total=total,
        hierarchical=final_loss,
        coarse=sector_loss,
        fine=offset_loss,
        flat=final_loss,
        angular_smoothing=zero,
        geometry_consistency=zero,
        radio_semantic=zero,
        path_semantic=zero,
        path_regression=zero,
        diagnostics=diagnostics,
    )


def widened_target_prior(
    support_prior: torch.Tensor,
    *,
    sigma: float = 3.0,
    temperature: float = 1.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    prior = support_prior.to(dtype=torch.float32).reshape(-1).clamp_min(float(eps))
    prior = prior / prior.sum().clamp_min(float(eps))
    classes = int(prior.numel())
    bins = torch.arange(classes, device=prior.device, dtype=prior.dtype)
    distance = (bins.view(-1, 1) - bins.view(1, -1)).abs()
    kernel = torch.exp(-0.5 * (distance / max(float(sigma), 1e-6)).pow(2))
    widened = kernel @ prior
    if float(temperature) != 1.0:
        widened = widened.clamp_min(float(eps)).pow(1.0 / max(float(temperature), 1e-6))
    widened = widened.clamp_min(float(eps))
    return widened / widened.sum().clamp_min(float(eps))


def prediction_marginal_kl_loss(logits: torch.Tensor, target_prior: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    p_bar = probs.reshape(-1, probs.shape[-1]).mean(dim=0).clamp_min(1e-12)
    p_bar = p_bar / p_bar.sum().clamp_min(1e-12)
    target = target_prior.to(device=logits.device, dtype=logits.dtype).reshape(-1).clamp_min(1e-12)
    target = target / target.sum().clamp_min(1e-12)
    return torch.sum(p_bar * (torch.log(p_bar) - torch.log(target)))


def _v9_widened_prior_marginal_kl(
    logits: torch.Tensor,
    cfg: dict[str, Any],
    *,
    weight: float,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not bool(cfg.get("use_widened_prior_marginal_kl", False)) or float(weight) <= 0.0:
        return zero, {
            "hist/v9/widened_prior_marginal_kl_enabled": 0.0,
            "hist/v9/widened_prior_target": "disabled",
        }
    support = cfg.get("support_prior")
    if support is None:
        return logits.sum() * 0.0, {
            "hist/v9/widened_prior_marginal_kl_enabled": 0.0,
            "hist/v9/widened_prior_unavailable_reason": "support_prior_missing",
            "hist/v9/widened_prior_target": "widened_support_prior",
        }
    support_t = torch.as_tensor(support, device=logits.device, dtype=logits.dtype).reshape(-1)
    widened = widened_target_prior(
        support_t,
        sigma=float(cfg.get("widened_prior_sigma", 3.0)),
        temperature=float(cfg.get("widened_prior_temperature", 1.5)),
    ).to(device=logits.device, dtype=logits.dtype)
    loss = prediction_marginal_kl_loss(logits, widened)
    return float(weight) * loss, {
        "hist/v9/widened_prior_marginal_kl_enabled": 1.0,
        "hist/v9/widened_prior_marginal_kl": _scalar(loss),
        "hist/v9/widened_prior_target": "widened_support_prior",
        "hist/v9/widened_prior_sigma": float(cfg.get("widened_prior_sigma", 3.0)),
        "hist/v9/widened_prior_temperature": float(cfg.get("widened_prior_temperature", 1.5)),
    }


def _compute_v7_hist_beam_loss(
    output: dict[str, Any],
    labels: torch.Tensor,
    *,
    cfg: dict[str, Any],
    weights: dict[str, float],
    beamspace_power_labels: torch.Tensor | None,
    beamspace_power_mask: torch.Tensor | None,
    current_epoch: int | None,
    v7_adaptation: bool,
    num_classes: int,
    ignore_index: int,
    zero: torch.Tensor,
) -> HistBeamLossResult:
    logits_shared = _tensor(output, "logits_shared")
    logits_final = _first_tensor(output, ("logits_final", "beam_logits", "logits"))
    delta = _tensor(output, "delta_logits_private")
    alpha = _tensor(output, "alpha")
    pred_phys = _tensor(output, "pred_beamspace_power")
    if logits_shared is None or logits_final is None:
        raise RuntimeError("V7 HiST-Beam loss requires logits_shared and logits_final outputs.")
    ensure_horizon_shape("logits_shared", logits_shared, labels)
    ensure_horizon_shape("logits_final", logits_final, labels)
    warmup_epochs = int(cfg.get("training", {}).get("shared_warmup_epochs", cfg.get("hist_beam", {}).get("shared_warmup_epochs", 0)) or 0)
    warmup_active = (current_epoch is not None) and int(current_epoch) < warmup_epochs
    effective_final = logits_shared if warmup_active else logits_final
    class_weights, class_balance_diag = _v7_class_balance_weights(
        labels,
        cfg=cfg,
        num_classes=num_classes,
        ignore_index=ignore_index,
        zero=zero,
        v7_adaptation=v7_adaptation,
    )
    shared_ce = _flat_ce(logits_shared, labels, num_classes=num_classes, ignore_index=ignore_index, class_weights=class_weights)
    final_ce = (
        _flat_ce(effective_final, labels, num_classes=num_classes, ignore_index=ignore_index, class_weights=class_weights)
        if not warmup_active
        else zero
    )
    bsp_kl, bsp_diag = _beamspace_kl(
        logits_shared,
        beamspace_power_labels,
        beamspace_power_mask,
        temperature=float(_v7_loss_cfg(cfg).get("temperature", _v7_loss_cfg(cfg).get("kl_temperature", 1.0))),
        zero=zero,
    )
    phys_kl, phys_diag = _physical_head_kl(pred_phys, beamspace_power_labels, beamspace_power_mask, zero=zero)
    if v7_adaptation:
        bsp_kl = zero
        phys_kl = zero
        bsp_diag = {"hist/v7/bsp_available": 0.0, "hist/v7/bsp_unavailable_reason": "target_physical_oracle_not_used_for_training"}
        phys_diag = {"hist/v7/phys_available": 0.0, "hist/v7/phys_unavailable_reason": "target_physical_oracle_not_used_for_training"}
    res_l2 = delta.pow(2).mean() if torch.is_tensor(delta) and not warmup_active else zero
    gate_l1 = alpha.abs().mean() if torch.is_tensor(alpha) and not warmup_active else zero
    diff = _v7_difference_loss(
        _tensor(output, "shared_representation"),
        _tensor(output, "private_representation"),
        zero=zero,
    ) if not warmup_active else zero
    if warmup_active:
        res_l2 = zero
        gate_l1 = zero
        diff = zero
    if v7_adaptation:
        total = (
            weights["v7_final_ce"] * _flat_ce(logits_final, labels, num_classes=num_classes, ignore_index=ignore_index, class_weights=class_weights)
            + weights["v7_res_l2"] * res_l2
            + weights["v7_gate_l1"] * gate_l1
        )
    elif warmup_active:
        total = (
            weights["v7_shared_ce"] * shared_ce
            + weights["v7_bsp_kl"] * bsp_kl
            + weights["v7_phys_kl"] * phys_kl
        )
    else:
        total = (
            weights["v7_shared_ce"] * shared_ce
            + weights["v7_final_ce"] * final_ce
            + weights["v7_bsp_kl"] * bsp_kl
            + weights["v7_phys_kl"] * phys_kl
            + weights["v7_res_l2"] * res_l2
            + weights["v7_gate_l1"] * gate_l1
            + weights["v7_diff"] * diff
        )
    diagnostics = {
        "hist/loss_total": _scalar(total),
        "hist/v7/loss_shared_ce": _scalar(shared_ce),
        "hist/v7/loss_final_ce": _scalar(final_ce),
        "hist/v7/loss_bsp_kl": _scalar(bsp_kl),
        "hist/v7/loss_phys_kl": _scalar(phys_kl),
        "hist/v7/loss_res_l2": _scalar(res_l2),
        "hist/v7/loss_gate_l1": _scalar(gate_l1),
        "hist/v7/loss_diff": _scalar(diff),
        "hist/v7/warmup_active": 1.0 if warmup_active else 0.0,
        "hist/v7/target_adaptation_loss": 1.0 if v7_adaptation else 0.0,
    }
    diagnostics.update(bsp_diag)
    diagnostics.update(phys_diag)
    diagnostics.update(class_balance_diag)
    return HistBeamLossResult(
        total=total,
        hierarchical=shared_ce,
        coarse=zero,
        fine=zero,
        flat=final_ce,
        angular_smoothing=zero,
        geometry_consistency=zero,
        radio_semantic=zero,
        path_semantic=zero,
        path_regression=zero,
        diagnostics=diagnostics,
    )


def _beamspace_kl(
    logits: torch.Tensor,
    target: torch.Tensor | None,
    mask: torch.Tensor | None,
    *,
    temperature: float,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    prepared = _prepare_bsp_target(logits, target, mask)
    if prepared is None:
        return logits.sum() * 0.0, {
            "hist/v7/bsp_available": 0.0,
            "hist/v7/bsp_coverage": 0.0,
            "hist/v7/bsp_unavailable_reason": "beamspace_power_label_missing",
        }
    target_t, valid = prepared
    if not torch.any(valid):
        return logits.sum() * 0.0, {
            "hist/v7/bsp_available": 0.0,
            "hist/v7/bsp_coverage": 0.0,
            "hist/v7/bsp_unavailable_reason": "beamspace_power_label_unavailable",
        }
    temp = max(float(temperature), 1e-6)
    log_probs = F.log_softmax(logits[valid] / temp, dim=-1)
    loss = F.kl_div(log_probs, target_t[valid], reduction="batchmean") * (temp * temp)
    return loss, {
        "hist/v7/bsp_available": 1.0,
        "hist/v7/bsp_coverage": float(valid.float().mean().detach().cpu().item()),
        "hist/v7/bsp_valid_count": float(valid.sum().detach().cpu().item()),
    }


def _physical_head_kl(
    pred: torch.Tensor | None,
    target: torch.Tensor | None,
    mask: torch.Tensor | None,
    *,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if pred is None:
        return zero, {"hist/v7/phys_available": 0.0, "hist/v7/phys_unavailable_reason": "pred_beamspace_power_missing"}
    prepared = _prepare_bsp_target(pred, target, mask)
    if prepared is None:
        return pred.sum() * 0.0, {"hist/v7/phys_available": 0.0, "hist/v7/phys_unavailable_reason": "beamspace_power_label_missing"}
    target_t, valid = prepared
    if not torch.any(valid):
        return pred.sum() * 0.0, {"hist/v7/phys_available": 0.0, "hist/v7/phys_unavailable_reason": "beamspace_power_label_unavailable"}
    probs = pred.clamp_min(1e-12)
    log_probs = torch.log(probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12))
    loss = F.kl_div(log_probs[valid], target_t[valid], reduction="batchmean")
    return loss, {
        "hist/v7/phys_available": 1.0,
        "hist/v7/phys_coverage": float(valid.float().mean().detach().cpu().item()),
    }


def _prepare_bsp_target(
    reference: torch.Tensor,
    target: torch.Tensor | None,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if target is None:
        return None
    target_t = target.to(device=reference.device, dtype=reference.dtype)
    if target_t.ndim == 2:
        target_t = target_t.unsqueeze(1)
    if target_t.ndim != 3 or target_t.shape[:2] != reference.shape[:2] or target_t.shape[-1] != reference.shape[-1]:
        raise ValueError(f"beamspace_power_label shape {tuple(target_t.shape)} does not match logits {tuple(reference.shape)}.")
    valid = torch.isfinite(target_t).all(dim=-1) & target_t.sum(dim=-1).gt(0)
    if mask is not None:
        mask_t = mask.to(device=reference.device, dtype=torch.bool)
        if mask_t.ndim == 1:
            mask_t = mask_t.unsqueeze(1)
        valid = valid & mask_t[:, : target_t.shape[1]]
    normalized = target_t / target_t.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return normalized, valid


def _v7_difference_loss(shared: torch.Tensor | None, private: torch.Tensor | None, *, zero: torch.Tensor) -> torch.Tensor:
    if shared is None or private is None:
        return zero
    if shared.ndim == 3:
        shared = shared.reshape(-1, shared.shape[-1])
    if private.ndim == 3:
        private = private.reshape(-1, private.shape[-1])
    if shared.shape[0] < 2 or private.shape[0] < 2:
        return zero
    return torch.mean(F.cosine_similarity(shared, private, dim=-1).pow(2))


def _v7_loss_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    value = hist_cfg.get("v7_loss", hist_cfg.get("v7", {}))
    return value if isinstance(value, dict) else {}


def _v7_class_balance_weights(
    labels: torch.Tensor,
    *,
    cfg: dict[str, Any],
    num_classes: int,
    ignore_index: int,
    zero: torch.Tensor,
    v7_adaptation: bool,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    balance_cfg = hist_cfg.get("class_balance") if isinstance(hist_cfg.get("class_balance"), dict) else {}
    if not bool(balance_cfg.get("enabled", False)):
        return None, {"hist/v7/class_balance_enabled": 0.0}
    if v7_adaptation and not bool(balance_cfg.get("target_adaptation", balance_cfg.get("adaptation", False))):
        return None, {"hist/v7/class_balance_enabled": 0.0, "hist/v7/class_balance_skipped_reason": "target_adaptation_disabled"}
    if not v7_adaptation and not bool(balance_cfg.get("source_training", True)):
        return None, {"hist/v7/class_balance_enabled": 0.0, "hist/v7/class_balance_skipped_reason": "source_training_disabled"}
    valid = labels.reshape(-1).to(device=zero.device, dtype=torch.long)
    valid = valid[valid.ne(ignore_index) & valid.ge(0) & valid.lt(int(num_classes))]
    if valid.numel() == 0:
        return None, {"hist/v7/class_balance_enabled": 0.0, "hist/v7/class_balance_skipped_reason": "no_valid_labels"}
    counts = torch.bincount(valid, minlength=int(num_classes)).to(device=zero.device, dtype=zero.dtype)
    present = counts.gt(0)
    if int(present.sum().detach().cpu().item()) <= 1:
        return None, {
            "hist/v7/class_balance_enabled": 0.0,
            "hist/v7/class_balance_skipped_reason": "single_present_class",
            "hist/v7/class_balance_present_classes": float(present.sum().detach().cpu().item()),
        }
    mode = str(balance_cfg.get("mode", "inverse_sqrt")).strip().lower()
    power = 0.5 if mode in {"inverse_sqrt", "sqrt", "inv_sqrt"} else 1.0
    if "power" in balance_cfg:
        power = float(balance_cfg["power"])
    mean_count = counts[present].mean().clamp_min(1.0)
    weights = torch.ones(int(num_classes), device=zero.device, dtype=zero.dtype)
    weights[present] = (mean_count / counts[present].clamp_min(1.0)).pow(float(power))
    weights[present] = weights[present] / weights[present].mean().clamp_min(1e-12)
    max_weight = balance_cfg.get("max_weight")
    if max_weight is not None:
        weights[present] = weights[present].clamp(max=float(max_weight))
        weights[present] = weights[present] / weights[present].mean().clamp_min(1e-12)
    min_weight = balance_cfg.get("min_weight")
    if min_weight is not None:
        weights[present] = weights[present].clamp(min=float(min_weight))
        weights[present] = weights[present] / weights[present].mean().clamp_min(1e-12)
    return weights, {
        "hist/v7/class_balance_enabled": 1.0,
        "hist/v7/class_balance_mode": mode,
        "hist/v7/class_balance_power": float(power),
        "hist/v7/class_balance_present_classes": float(present.sum().detach().cpu().item()),
        "hist/v7/class_balance_min_weight": float(weights[present].min().detach().cpu().item()),
        "hist/v7/class_balance_max_weight": float(weights[present].max().detach().cpu().item()),
    }


def _v7_enabled(hist_cfg: dict[str, Any], model_cfg: dict[str, Any], output: dict[str, Any]) -> bool:
    variant = str(hist_cfg.get("variant", model_cfg.get("variant", ""))).strip().lower()
    if variant in {"v7_shared_physical_private_residual", "shared_physical_private_residual"}:
        return True
    meta = output.get("hist_beam")
    if isinstance(meta, dict) and bool(meta.get("v7_shared_physical_private_residual", False)):
        return True
    return torch.is_tensor(output.get("logits_shared")) and torch.is_tensor(output.get("delta_logits_private"))


def _v8_enabled(hist_cfg: dict[str, Any], model_cfg: dict[str, Any], output: dict[str, Any]) -> bool:
    variant = str(hist_cfg.get("variant", model_cfg.get("variant", ""))).strip().lower()
    if variant in {"v8_target_prior_head", "v9_input_conditioned_target_adaptation"}:
        return True
    meta = output.get("hist_beam")
    if isinstance(meta, dict) and bool(meta.get("v8_target_prior_head", False)):
        return True
    if isinstance(meta, dict) and bool(meta.get("v9_input_conditioned_target_adaptation", False)):
        return True
    return torch.is_tensor(output.get("target_logits")) and torch.is_tensor(output.get("target_prior_bias"))


def _v8_loss_cfg(cfg: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    v8 = hist_cfg.get("v8") if isinstance(hist_cfg.get("v8"), dict) else {}
    v9 = hist_cfg.get("v9") if isinstance(hist_cfg.get("v9"), dict) else {}
    meta = output.get("hist_beam") if isinstance(output.get("hist_beam"), dict) else {}
    support_prior = v9.get("support_prior")
    if support_prior is None:
        prior_bias = _tensor(output, "target_prior_bias")
        if prior_bias is not None:
            bias = prior_bias[0, 0, :] if prior_bias.ndim == 3 else prior_bias.reshape(-1)
            support_prior = torch.softmax(bias.detach().to(dtype=torch.float32), dim=-1).cpu().tolist()
    return {
        "use_soft_beam_label": v8.get("use_soft_beam_label", meta.get("v8_use_soft_beam_label", True)),
        "soft_label_sigma": v8.get("soft_label_sigma", meta.get("v8_soft_label_sigma", 1.0)),
        "loss_prior_smooth_weight": v8.get("loss_prior_smooth_weight", meta.get("v8_loss_prior_smooth_weight", 0.001)),
        "sector_size": v8.get("sector_size", meta.get("v8_sector_size", hist_cfg.get("group_size", 8))),
        "use_widened_prior_marginal_kl": v9.get(
            "use_widened_prior_marginal_kl",
            meta.get("v9_use_widened_prior_marginal_kl", False),
        ),
        "widened_prior_sigma": v9.get("widened_prior_sigma", meta.get("v9_widened_prior_sigma", 3.0)),
        "widened_prior_temperature": v9.get(
            "widened_prior_temperature",
            meta.get("v9_widened_prior_temperature", 1.5),
        ),
        "support_prior": support_prior,
    }


def _v8_sector_loss(
    sector_logits: torch.Tensor | None,
    labels: torch.Tensor,
    *,
    sector_size: int,
    num_classes: int,
    ignore_index: int,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if sector_logits is None:
        return zero, {"hist/v8/sector_available": 0.0, "hist/v8/sector_unavailable_reason": "sector_logits_missing"}
    ensure_horizon_shape("sector_logits", sector_logits, labels)
    labels_t = labels.to(device=sector_logits.device, dtype=torch.long)
    target = torch.div(labels_t, int(sector_size), rounding_mode="floor")
    valid = labels_t.ne(ignore_index) & labels_t.ge(0) & labels_t.lt(int(num_classes)) & target.lt(sector_logits.shape[-1])
    if not torch.any(valid):
        return sector_logits.sum() * 0.0, {
            "hist/v8/sector_available": 0.0,
            "hist/v8/sector_unavailable_reason": "no_valid_sector_labels",
        }
    safe_target = torch.where(valid, target, torch.full_like(target, int(ignore_index)))
    loss = F.cross_entropy(
        sector_logits.reshape(-1, sector_logits.shape[-1]),
        safe_target.reshape(-1),
        ignore_index=ignore_index,
    )
    return loss, {
        "hist/v8/sector_available": 1.0,
        "hist/v8/sector_coverage": float(valid.float().mean().detach().cpu().item()),
    }


def _v8_offset_loss(
    offset_logits: torch.Tensor | None,
    labels: torch.Tensor,
    *,
    sector_size: int,
    num_classes: int,
    ignore_index: int,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if offset_logits is None:
        return zero, {"hist/v8/offset_available": 0.0, "hist/v8/offset_unavailable_reason": "offset_logits_missing"}
    ensure_horizon_shape("offset_logits", offset_logits, labels)
    labels_t = labels.to(device=offset_logits.device, dtype=torch.long)
    target = labels_t.remainder(int(sector_size))
    valid = (
        labels_t.ne(ignore_index)
        & labels_t.ge(0)
        & labels_t.lt(int(num_classes))
        & target.ge(0)
        & target.lt(offset_logits.shape[-1])
    )
    if not torch.any(valid):
        return offset_logits.sum() * 0.0, {
            "hist/v8/offset_available": 0.0,
            "hist/v8/offset_unavailable_reason": "no_valid_offset_labels",
        }
    safe_target = torch.where(valid, target, torch.full_like(target, int(ignore_index)))
    loss = F.cross_entropy(
        offset_logits.reshape(-1, offset_logits.shape[-1]),
        safe_target.reshape(-1),
        ignore_index=ignore_index,
    )
    return loss, {
        "hist/v8/offset_available": 1.0,
        "hist/v8/offset_coverage": float(valid.float().mean().detach().cpu().item()),
    }


def _labels_to_1d(labels: torch.Tensor | list[int] | tuple[int, ...] | None, *, device: torch.device) -> torch.Tensor:
    if labels is None:
        return torch.empty(0, dtype=torch.long, device=device)
    if torch.is_tensor(labels):
        return labels.detach().to(device=device, dtype=torch.long).reshape(-1)
    return torch.as_tensor(list(labels), dtype=torch.long, device=device).reshape(-1)


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
    "gaussian_smooth_beam_prior",
    "hist_beam_enabled",
    "make_beam_soft_labels",
    "multimodal_geometry_consistency_loss",
    "path_descriptor_regression_loss",
    "path_semantic_ce_loss",
    "prediction_marginal_kl_loss",
    "prototype_consistency_loss",
    "radio_semantic_ce_loss",
    "widened_target_prior",
]
