from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

import kd_sensing.engine.objectives.metadata as _objective_metadata
from kd_sensing.engine.auxiliary import compute_auxiliary_multitask_loss
from kd_sensing.engine.model_output import ModelOutput


@dataclass(frozen=True)
class PredictionTargets:
    labels: torch.Tensor
    occlusion_label: torch.Tensor | None = None
    occlusion_valid: torch.Tensor | None = None
    position_target: torch.Tensor | None = None
    position_valid: torch.Tensor | None = None
    los_label: torch.Tensor | None = None
    link_quality: torch.Tensor | None = None

    def as_auxiliary_dict(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        if self.occlusion_label is not None:
            result["occlusion_label"] = self.occlusion_label
        if self.occlusion_valid is not None:
            result["occlusion_valid"] = self.occlusion_valid
        if self.position_target is not None:
            result["position_target"] = self.position_target
        if self.position_valid is not None:
            result["position_valid"] = self.position_valid
        return result


@dataclass(frozen=True)
class PredictionLossBundle:
    total: torch.Tensor
    primary: torch.Tensor
    beam: torch.Tensor
    occlusion: torch.Tensor
    position: torch.Tensor
    multitask_total: torch.Tensor
    diagnostics: dict[str, float]
    los: torch.Tensor | None = None
    link_quality: torch.Tensor | None = None
    selection_multitask_total: torch.Tensor | None = None


def prepare_prediction_targets(
    *,
    labels: torch.Tensor,
    auxiliary_targets: dict[str, torch.Tensor],
    cfg: dict[str, Any],
) -> PredictionTargets:
    objective = resolve_prediction_objective(cfg)
    targets = PredictionTargets(
        labels=labels,
        occlusion_label=auxiliary_targets.get("occlusion_label"),
        occlusion_valid=auxiliary_targets.get("occlusion_valid"),
        position_target=auxiliary_targets.get("position_target"),
        position_valid=auxiliary_targets.get("position_valid"),
        los_label=auxiliary_targets.get("los_label"),
        link_quality=auxiliary_targets.get("link_quality"),
    )
    if objective in {"occlusion", "multitask"}:
        _require_tensor(targets.occlusion_label, "occlusion_label", objective)
        _require_tensor(targets.occlusion_valid, "occlusion_valid", objective)
    if objective in {"position", "multitask"}:
        _require_tensor(targets.position_target, "position_target", objective)
        _require_tensor(targets.position_valid, "position_valid", objective)
    if objective == "selection_multitask":
        _require_tensor(targets.los_label, "los_label", objective)
        _require_tensor(targets.link_quality, "link_quality", objective)
    if objective == "current_los_classification":
        _require_tensor(targets.los_label, "los_label", objective)
    if objective == "current_link_quality":
        _require_tensor(targets.link_quality, "link_quality", objective)
    return targets


def compute_prediction_loss(
    model_output: ModelOutput,
    targets: PredictionTargets,
    cfg: dict[str, Any],
    *,
    reference: torch.Tensor,
    beam_total_loss: torch.Tensor,
    beam_task_loss: torch.Tensor | None = None,
) -> PredictionLossBundle:
    objective = resolve_prediction_objective(cfg)
    zero = reference.sum() * 0.0
    beam_component = beam_total_loss
    beam_primary = beam_task_loss if beam_task_loss is not None else beam_total_loss
    occlusion_loss = zero
    position_loss = zero
    multitask_total = zero
    los_loss = zero
    link_quality_loss = zero
    selection_multitask_total = zero
    diagnostics = {"loss/beam": float(beam_primary.detach().cpu().item())}

    if objective == "near_field_beam_selection":
        metadata = _near_field_codebook_metadata(cfg)
        expected_classes = int(metadata["num_beam_classes"])
        actual_classes = int(reference.shape[-1])
        if actual_classes != expected_classes:
            raise ValueError(
                "near_field_beam_selection model output class count does not match codebook metadata: "
                f"model_output_classes={actual_classes}, codebook_shape={metadata['shape']}, "
                f"expected_num_classes={expected_classes}."
            )
        diagnostics = {
            "loss/near_field_beam_selection": float(beam_primary.detach().cpu().item()),
            "loss/primary": float(beam_primary.detach().cpu().item()),
        }
        return PredictionLossBundle(
            total=beam_component,
            primary=beam_primary,
            beam=beam_primary,
            occlusion=zero,
            position=zero,
            multitask_total=zero,
            diagnostics=diagnostics,
            los=zero,
            link_quality=zero,
            selection_multitask_total=zero,
        )

    if objective == "current_beam_selection":
        diagnostics = {
            "loss/beam_selection": float(beam_primary.detach().cpu().item()),
            "loss/primary": float(beam_primary.detach().cpu().item()),
        }
        return PredictionLossBundle(
            total=beam_component,
            primary=beam_primary,
            beam=beam_primary,
            occlusion=zero,
            position=zero,
            multitask_total=zero,
            diagnostics=diagnostics,
            los=zero,
            link_quality=zero,
            selection_multitask_total=zero,
        )

    if objective == "current_los_classification":
        los_loss = _los_loss(model_output, targets, cfg)
        diagnostics = {
            "loss/los": float(los_loss.detach().cpu().item()),
            "loss/primary": float(los_loss.detach().cpu().item()),
        }
        return PredictionLossBundle(
            total=los_loss,
            primary=los_loss,
            beam=beam_primary,
            occlusion=zero,
            position=zero,
            multitask_total=zero,
            diagnostics=diagnostics,
            los=los_loss,
            link_quality=None,
            selection_multitask_total=None,
        )

    if objective == "current_link_quality":
        link_quality_loss = _link_quality_loss(model_output, targets, cfg)
        diagnostics = {
            "loss/link_quality": float(link_quality_loss.detach().cpu().item()),
            "loss/primary": float(link_quality_loss.detach().cpu().item()),
        }
        return PredictionLossBundle(
            total=link_quality_loss,
            primary=link_quality_loss,
            beam=beam_primary,
            occlusion=zero,
            position=zero,
            multitask_total=zero,
            diagnostics=diagnostics,
            los=None,
            link_quality=link_quality_loss,
            selection_multitask_total=None,
        )

    if objective == "beam":
        auxiliary_loss = compute_auxiliary_multitask_loss(
            model_output,
            targets.as_auxiliary_dict(),
            cfg,
            reference=reference,
        )
        total = beam_component + auxiliary_loss.total
        auxiliary_diagnostics = dict(auxiliary_loss.diagnostics)
        if "loss/occlusion" not in auxiliary_diagnostics and "loss/position" not in auxiliary_diagnostics:
            auxiliary_diagnostics.pop("loss/multitask_total", None)
        diagnostics.update(auxiliary_diagnostics)
        diagnostics["loss/beam"] = float(beam_primary.detach().cpu().item())
        diagnostics["loss/primary"] = float(beam_primary.detach().cpu().item())
        return PredictionLossBundle(
            total=total,
            primary=beam_primary,
            beam=beam_primary,
            occlusion=auxiliary_loss.occlusion,
            position=auxiliary_loss.position,
            multitask_total=auxiliary_loss.total,
            diagnostics=diagnostics,
            los=zero,
            link_quality=zero,
            selection_multitask_total=zero,
        )

    if objective in {"occlusion", "multitask"}:
        occlusion_loss = _occlusion_loss(model_output, targets, cfg, zero)
        diagnostics["loss/occlusion"] = float(occlusion_loss.detach().cpu().item())

    if objective in {"position", "multitask"}:
        position_loss = _position_loss(model_output, targets, cfg, zero)
        diagnostics["loss/position"] = float(position_loss.detach().cpu().item())

    if objective == "selection_multitask":
        los_loss = _los_loss(model_output, targets, cfg)
        link_quality_loss = _link_quality_loss(model_output, targets, cfg)
        weights = selection_multitask_loss_weights(cfg)
        selection_multitask_total = (
            weights["beam_selection"] * beam_primary
            + weights["los"] * los_loss
            + weights["link_quality"] * link_quality_loss
        )
        diagnostics = {
            "loss/beam_selection": float(beam_primary.detach().cpu().item()),
            "loss/los": float(los_loss.detach().cpu().item()),
            "loss/link_quality": float(link_quality_loss.detach().cpu().item()),
            "loss/selection_multitask_total": float(selection_multitask_total.detach().cpu().item()),
            "loss/primary": float(selection_multitask_total.detach().cpu().item()),
            "objective/weight_beam_selection": float(weights["beam_selection"]),
            "objective/weight_los": float(weights["los"]),
            "objective/weight_link_quality": float(weights["link_quality"]),
        }
        return PredictionLossBundle(
            total=selection_multitask_total,
            primary=selection_multitask_total,
            beam=beam_primary,
            occlusion=zero,
            position=zero,
            multitask_total=zero,
            diagnostics=diagnostics,
            los=los_loss,
            link_quality=link_quality_loss,
            selection_multitask_total=selection_multitask_total,
        )

    if objective == "occlusion":
        primary = occlusion_loss
        total = primary
    elif objective == "position":
        primary = position_loss
        total = primary
    else:
        weights = multitask_loss_weights(cfg)
        multitask_total = (
            weights["beam"] * beam_component
            + weights["occlusion"] * occlusion_loss
            + weights["position"] * position_loss
        )
        primary = multitask_total
        total = multitask_total
        diagnostics["loss/multitask_total"] = float(multitask_total.detach().cpu().item())
        diagnostics["objective/weight_beam"] = float(weights["beam"])
        diagnostics["objective/weight_occlusion"] = float(weights["occlusion"])
        diagnostics["objective/weight_position"] = float(weights["position"])

    diagnostics["loss/primary"] = float(primary.detach().cpu().item())
    return PredictionLossBundle(
        total=total,
        primary=primary,
        beam=beam_primary,
        occlusion=occlusion_loss,
        position=position_loss,
        multitask_total=multitask_total,
        diagnostics=diagnostics,
        los=zero,
        link_quality=zero,
        selection_multitask_total=zero,
    )


PREDICTION_OBJECTIVES = _objective_metadata.PREDICTION_OBJECTIVES
PredictionObjectiveSpec = _objective_metadata.PredictionObjectiveSpec
configure_objective_defaults = _objective_metadata.configure_objective_defaults
default_primary_metric = _objective_metadata.default_primary_metric
multitask_loss_weights = _objective_metadata.multitask_loss_weights
normalize_objective_metric = _objective_metadata.normalize_objective_metric
objective_available_metrics = _objective_metadata.objective_available_metrics
objective_enabled_heads = _objective_metadata.objective_enabled_heads
objective_enabled_targets = _objective_metadata.objective_enabled_targets
objective_history_fields = _objective_metadata.objective_history_fields
objective_metric_mode = _objective_metadata.objective_metric_mode
objective_optional_history_fields = _objective_metadata.objective_optional_history_fields
objective_requires_occlusion = _objective_metadata.objective_requires_occlusion
objective_requires_position = _objective_metadata.objective_requires_position
objective_runtime_metadata = _objective_metadata.objective_runtime_metadata
objective_spec = _objective_metadata.objective_spec
objective_tensorboard_scalars = _objective_metadata.objective_tensorboard_scalars
produced_metric_names = _objective_metadata.produced_metric_names
resolve_prediction_objective = _objective_metadata.resolve_prediction_objective
selection_multitask_loss_weights = _objective_metadata.selection_multitask_loss_weights
validate_objective_metric_available = _objective_metadata.validate_objective_metric_available


def _occlusion_loss(
    model_output: ModelOutput,
    targets: PredictionTargets,
    cfg: dict[str, Any],
    zero: torch.Tensor,
) -> torch.Tensor:
    logits = _diagnostic_tensor(model_output, "occlusion_logits", "model output")
    labels = _require_tensor(targets.occlusion_label, "occlusion_label", resolve_prediction_objective(cfg))
    valid = _require_tensor(targets.occlusion_valid, "occlusion_valid", resolve_prediction_objective(cfg))
    if logits.ndim != 2:
        raise ValueError(f"occlusion_logits must have shape [B, H], got {tuple(logits.shape)}.")
    if labels.shape != logits.shape:
        raise ValueError(
            f"occlusion_label shape {tuple(labels.shape)} does not match occlusion_logits {tuple(logits.shape)}."
        )
    if valid.shape != logits.shape:
        raise ValueError(
            f"occlusion_valid shape {tuple(valid.shape)} does not match occlusion_logits {tuple(logits.shape)}."
        )
    labels = labels.to(device=logits.device, dtype=logits.dtype)
    valid = valid.to(device=logits.device, dtype=torch.bool)
    loss_cfg = _objective_loss_cfg(cfg, "occlusion")
    element_loss = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
        pos_weight=_resolve_pos_weight(loss_cfg.get("pos_weight"), labels, valid),
    )
    return _masked_mean(element_loss, valid, zero)


def _position_loss(
    model_output: ModelOutput,
    targets: PredictionTargets,
    cfg: dict[str, Any],
    zero: torch.Tensor,
) -> torch.Tensor:
    prediction = _diagnostic_tensor(model_output, "position", "model output")
    target = _require_tensor(targets.position_target, "position_target", resolve_prediction_objective(cfg))
    valid = _require_tensor(targets.position_valid, "position_valid", resolve_prediction_objective(cfg))
    if prediction.ndim != 3 or prediction.shape[-1] != 2:
        raise ValueError(f"position output must have shape [B, H, 2], got {tuple(prediction.shape)}.")
    if target.shape != prediction.shape:
        raise ValueError(f"position_target shape {tuple(target.shape)} does not match output {tuple(prediction.shape)}.")
    if valid.shape != prediction.shape[:2]:
        raise ValueError(f"position_valid shape {tuple(valid.shape)} does not match output {tuple(prediction.shape)}.")
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    valid = valid.to(device=prediction.device, dtype=torch.bool)
    loss_cfg = _objective_loss_cfg(cfg, "position")
    loss_type = str(loss_cfg.get("type", "mse")).lower()
    if loss_type in {"smooth_l1", "huber"}:
        beta = float(loss_cfg.get("beta", loss_cfg.get("smooth_l1_beta", 1.0)))
        per_coord = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
        per_slot = per_coord.mean(dim=-1)
    elif loss_type in {"mse", "l2"}:
        per_slot = (prediction - target).pow(2).mean(dim=-1)
    else:
        raise ValueError("loss.position.type must be one of mse or smooth_l1.")
    return _masked_mean(per_slot, valid, zero)


def _los_loss(model_output: ModelOutput, targets: PredictionTargets, cfg: dict[str, Any]) -> torch.Tensor:
    logits = _diagnostic_tensor(model_output, "los_logits", "model output")
    labels = _require_tensor(targets.los_label, "los_label", resolve_prediction_objective(cfg))
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    if logits.ndim != 2:
        raise ValueError(f"los_logits must have shape [B, H], got {tuple(logits.shape)}.")
    if labels.shape != logits.shape:
        raise ValueError(f"los_label shape {tuple(labels.shape)} does not match los_logits {tuple(logits.shape)}.")
    labels = labels.to(device=logits.device, dtype=logits.dtype)
    loss_cfg = _objective_loss_cfg(cfg, "los")
    return F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=_resolve_pos_weight(loss_cfg.get("pos_weight"), labels, torch.ones_like(labels, dtype=torch.bool)),
    )


def _link_quality_loss(model_output: ModelOutput, targets: PredictionTargets, cfg: dict[str, Any]) -> torch.Tensor:
    prediction = _diagnostic_tensor(model_output, "link_quality", "model output")
    target = _require_tensor(targets.link_quality, "link_quality", resolve_prediction_objective(cfg))
    if prediction.ndim == 1:
        prediction = prediction.unsqueeze(1)
    if prediction.ndim != 2:
        raise ValueError(f"link_quality output must have shape [B, H], got {tuple(prediction.shape)}.")
    if target.shape != prediction.shape:
        raise ValueError(
            f"link_quality target shape {tuple(target.shape)} does not match output {tuple(prediction.shape)}."
        )
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    loss_cfg = _objective_loss_cfg(cfg, "link_quality")
    beta = float(loss_cfg.get("beta", loss_cfg.get("smooth_l1_beta", 1.0)))
    return F.smooth_l1_loss(prediction, target, beta=beta)


def _objective_loss_cfg(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    loss_cfg = cfg.get("loss", {})
    objective_cfg = _mapping(loss_cfg.get("objective"))
    auxiliary_cfg = _mapping(loss_cfg.get("auxiliary"))
    return {
        **_mapping(auxiliary_cfg.get(name)),
        **_mapping(loss_cfg.get(name)),
        **_mapping(objective_cfg.get(name)),
    }


def _near_field_codebook_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    metadata = dataset_cfg.get("codebook_metadata")
    if isinstance(metadata, dict) and "shape" in metadata:
        shape = [int(value) for value in metadata["shape"]]
        return {
            "shape": shape,
            "num_beam_classes": int(metadata.get("num_beam_classes", math.prod(shape))),
        }
    shape = dataset_cfg.get("codebook_shape")
    if shape is not None:
        values = [int(value) for value in shape]
        return {"shape": values, "num_beam_classes": int(math.prod(values))}
    profile = str(dataset_cfg.get("codebook_profile", "")).lower()
    if profile == "dense":
        values = [90, 45, 16]
        return {"shape": values, "num_beam_classes": int(math.prod(values))}
    if profile == "small":
        values = [20, 20, 10]
        return {"shape": values, "num_beam_classes": int(math.prod(values))}
    raise ValueError(
        "experiment.objective='near_field_beam_selection' requires Multimodal-NF codebook metadata. "
        "Configure data.dataset.codebook_shape, codebook_profile, codebook_path, or build the dataset first "
        "so codebook_metadata can be reused."
    )


def _diagnostic_tensor(model_output: ModelOutput, key: str, source: str) -> torch.Tensor:
    value = model_output.diagnostics.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"Prediction objective requires {source} '{key}', but it is missing.")
    return value


def _require_tensor(value: torch.Tensor | None, key: str, objective: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise ValueError(f"experiment.objective '{objective}' requires batch target '{key}', but it is missing.")
    return value


def _masked_mean(values: torch.Tensor, valid: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    valid_f = valid.to(device=values.device, dtype=values.dtype)
    denom = valid_f.sum()
    if denom.item() <= 0:
        return zero
    return (values * valid_f).sum() / denom.clamp_min(1.0)


def _resolve_pos_weight(spec: Any, labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor | None:
    if spec is None or str(spec).lower() in {"none", "false", "0"}:
        return None
    if isinstance(spec, str) and spec.lower() == "auto":
        valid_labels = labels[valid]
        positives = valid_labels.sum()
        negatives = valid_labels.numel() - positives
        if positives.item() <= 0:
            return torch.ones((), dtype=labels.dtype, device=labels.device)
        return (negatives / positives.clamp_min(1.0)).to(dtype=labels.dtype, device=labels.device)
    return torch.tensor(float(spec), dtype=labels.dtype, device=labels.device)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}



__all__ = [
    "PREDICTION_OBJECTIVES",
    "PredictionLossBundle",
    "PredictionObjectiveSpec",
    "PredictionTargets",
    "compute_prediction_loss",
    "configure_objective_defaults",
    "default_primary_metric",
    "multitask_loss_weights",
    "normalize_objective_metric",
    "objective_available_metrics",
    "objective_enabled_heads",
    "objective_enabled_targets",
    "objective_history_fields",
    "objective_metric_mode",
    "objective_optional_history_fields",
    "objective_requires_occlusion",
    "objective_requires_position",
    "objective_runtime_metadata",
    "objective_spec",
    "objective_tensorboard_scalars",
    "prepare_prediction_targets",
    "produced_metric_names",
    "resolve_prediction_objective",
    "selection_multitask_loss_weights",
    "validate_objective_metric_available",
]
