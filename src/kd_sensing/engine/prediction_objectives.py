from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

import kd_sensing.engine.objectives.metadata as _objective_metadata
from kd_sensing.engine.auxiliary import compute_auxiliary_multitask_loss
from kd_sensing.engine.model_output import ModelOutput
from kd_sensing.losses.amber_full import amber_full_auxiliary_loss_from_output
from kd_sensing.losses.amr_net import amr_net_loss_from_output


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
    jepa: torch.Tensor | None = None


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
        predictive_latent_loss, predictive_latent_diagnostics = _predictive_latent_auxiliary_loss(
            model_output,
            cfg,
            zero,
        )
        rerank_loss, rerank_diagnostics = _safe_rerank_auxiliary_loss(
            model_output,
            targets,
            cfg,
            zero,
        )
        amr_loss, amr_diagnostics = amr_net_loss_from_output(model_output, targets.labels, cfg)
        amber_loss, amber_diagnostics = amber_full_auxiliary_loss_from_output(model_output, cfg, zero)
        total = beam_component + auxiliary_loss.total + predictive_latent_loss + rerank_loss + amr_loss + amber_loss
        auxiliary_diagnostics = dict(auxiliary_loss.diagnostics)
        if "loss/occlusion" not in auxiliary_diagnostics and "loss/position" not in auxiliary_diagnostics:
            auxiliary_diagnostics.pop("loss/multitask_total", None)
        diagnostics.update(auxiliary_diagnostics)
        diagnostics.update(predictive_latent_diagnostics)
        diagnostics.update(rerank_diagnostics)
        diagnostics.update(amr_diagnostics)
        diagnostics.update(amber_diagnostics)
        diagnostics["loss/beam"] = float(beam_primary.detach().cpu().item())
        diagnostics["loss/primary"] = float(beam_primary.detach().cpu().item())
        return PredictionLossBundle(
            total=total,
            primary=beam_primary,
            beam=beam_primary,
            occlusion=auxiliary_loss.occlusion,
            position=auxiliary_loss.position,
            multitask_total=auxiliary_loss.total + predictive_latent_loss + rerank_loss + amber_loss,
            diagnostics=diagnostics,
            los=zero,
            link_quality=zero,
            selection_multitask_total=zero,
            jepa=predictive_latent_loss,
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


def dba_aware_loss_config(cfg: dict[str, Any]) -> dict[str, Any]:
    loss_cfg = cfg.get("loss", {}) if isinstance(cfg.get("loss"), dict) else {}
    raw = loss_cfg.get("dba_aware", loss_cfg.get("beam_topology_smoothing", {}))
    if raw is True:
        raw = {"enabled": True}
    if not isinstance(raw, dict):
        return {"enabled": False}
    resolved = dict(raw)
    resolved.setdefault("enabled", False)
    resolved.setdefault("mode", "circular_gaussian")
    resolved.setdefault("sigma", 2.0)
    resolved.setdefault("temperature", 1.0)
    resolved.setdefault("distance_mode", "circular")
    resolved.setdefault("class_order", "circular")
    return resolved


def build_dba_aware_soft_targets(
    labels: torch.Tensor,
    *,
    num_classes: int,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor | None, dict[str, float]]:
    loss_cfg = dba_aware_loss_config(cfg)
    if not bool(loss_cfg.get("enabled", False)):
        return None, {}
    if labels.ndim != 2:
        raise ValueError(f"DBA-aware beam loss expects hard labels [B, H], got {tuple(labels.shape)}.")
    num_classes = int(num_classes)
    if num_classes <= 0:
        raise ValueError(f"DBA-aware beam loss requires positive num_classes, got {num_classes}.")
    sigma = float(loss_cfg.get("sigma", 2.0))
    temperature = float(loss_cfg.get("temperature", 1.0))
    if sigma <= 0.0:
        raise ValueError(f"loss.dba_aware.sigma must be positive, got {sigma}.")
    if temperature <= 0.0:
        raise ValueError(f"loss.dba_aware.temperature must be positive, got {temperature}.")
    classes = torch.arange(num_classes, dtype=torch.float32, device=labels.device).view(1, 1, num_classes)
    target = labels.to(dtype=torch.float32).unsqueeze(-1)
    valid = labels.ge(0) & labels.lt(num_classes)
    distance_mode = str(loss_cfg.get("distance_mode", "circular")).lower()
    raw_distance = (classes - target).abs()
    if distance_mode == "circular":
        distance = torch.minimum(raw_distance, float(num_classes) - raw_distance)
    elif distance_mode == "linear":
        distance = raw_distance
    else:
        raise ValueError("loss.dba_aware.distance_mode must be 'circular' or 'linear'.")
    mode = str(loss_cfg.get("mode", "circular_gaussian")).lower()
    if mode in {"circular_gaussian", "gaussian", "beam_topology_smoothing"}:
        logits = -0.5 * (distance / sigma).pow(2) / temperature
    elif mode in {"distance_aware_ce", "laplace"}:
        logits = -(distance / sigma) / temperature
    else:
        raise ValueError("loss.dba_aware.mode must be circular_gaussian or distance_aware_ce.")
    targets = torch.softmax(logits, dim=-1)
    targets = torch.where(valid.unsqueeze(-1), targets, torch.zeros_like(targets))
    sample_count = int(valid.sum().detach().cpu().item())
    diagnostics = {
        "loss/beam_dba_aware_enabled": 1.0,
        "loss/beam_dba_aware_sigma": sigma,
        "loss/beam_dba_aware_temperature": temperature,
        "loss/beam_dba_aware_sample_count": float(sample_count),
    }
    return targets, diagnostics


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


def _predictive_latent_auxiliary_loss(
    model_output: ModelOutput,
    cfg: dict[str, Any],
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_cfg = cfg.get("loss", {}) if isinstance(cfg.get("loss"), dict) else {}
    auxiliary_cfg = _mapping(loss_cfg.get("auxiliary"))
    objective_cfg = _mapping(loss_cfg.get("objective"))
    aux_cfg = {
        **_mapping(auxiliary_cfg.get("predictive_latent")),
        **_mapping(auxiliary_cfg.get("predictive_latent_auxiliary")),
        **_mapping(loss_cfg.get("predictive_latent_auxiliary")),
        **_mapping(objective_cfg.get("predictive_latent_auxiliary")),
    }
    enabled = bool(aux_cfg.get("enabled", False))
    weight = float(aux_cfg.get("weight", aux_cfg.get("lambda", 0.0)) or 0.0)
    if not enabled or weight <= 0.0:
        return zero, {}
    features = model_output.diagnostics.get("encoder_auxiliary_features")
    if not isinstance(features, dict) or not features:
        if bool(aux_cfg.get("allow_missing", False)):
            return zero, {"loss/predictive_latent_auxiliary": 0.0, "predictive_latent_auxiliary/sample_count": 0.0}
        raise ValueError("Predictive latent auxiliary loss requires encoder_auxiliary_features in model output.")
    modality = str(aux_cfg.get("modality", "image"))
    if modality not in features:
        if len(features) == 1 and bool(aux_cfg.get("allow_single_available_modality", True)):
            modality = next(iter(features))
        else:
            raise ValueError(f"Predictive latent auxiliary loss requires modality '{modality}' auxiliary features.")
    payload = features.get(modality)
    if not isinstance(payload, dict):
        raise ValueError(f"Predictive latent auxiliary features for modality '{modality}' must be a mapping.")
    prediction_key = str(aux_cfg.get("prediction_key", "temporal_predicted_latent"))
    target_key = str(aux_cfg.get("target_key", "current_latent"))
    predicted = payload.get(prediction_key)
    target = payload.get(target_key)
    if not torch.is_tensor(predicted) or not torch.is_tensor(target):
        raise ValueError(
            "Predictive latent auxiliary loss requires tensor features "
            f"'{prediction_key}' and '{target_key}' for modality '{modality}'."
        )
    if predicted.shape != target.shape:
        raise ValueError(
            "Predictive latent auxiliary predicted/target shapes must match, "
            f"got {tuple(predicted.shape)} and {tuple(target.shape)}."
        )
    detach_target = bool(aux_cfg.get("detach_target", True))
    target_for_loss = target.detach() if detach_target else target
    pred_for_loss = predicted
    if bool(aux_cfg.get("normalize_latent", aux_cfg.get("latent_normalize", False))):
        pred_for_loss = F.normalize(pred_for_loss, dim=-1)
        target_for_loss = F.normalize(target_for_loss, dim=-1)
    finite = torch.isfinite(pred_for_loss).all(dim=-1) & torch.isfinite(target_for_loss).all(dim=-1)
    sample_count = int(finite.sum().detach().cpu().item())
    if sample_count <= 0:
        if bool(aux_cfg.get("allow_empty", False)):
            return zero, {"loss/predictive_latent_auxiliary": 0.0, "predictive_latent_auxiliary/sample_count": 0.0}
        raise ValueError("Predictive latent auxiliary loss has no finite samples.")
    loss_type = str(aux_cfg.get("type", "mse")).strip().lower()
    if loss_type in {"smooth_l1", "huber"}:
        per_dim = F.smooth_l1_loss(
            pred_for_loss,
            target_for_loss,
            reduction="none",
            beta=float(aux_cfg.get("beta", 1.0)),
        )
    elif loss_type == "mse":
        per_dim = (pred_for_loss - target_for_loss).pow(2)
    else:
        raise ValueError("Predictive latent auxiliary loss type must be mse, smooth_l1, or huber.")
    loss = per_dim.mean(dim=-1)[finite].mean() * weight
    return loss, {
        "loss/predictive_latent_auxiliary": float(loss.detach().cpu().item()),
        "predictive_latent_auxiliary/weight": float(weight),
        "predictive_latent_auxiliary/sample_count": float(sample_count),
        "predictive_latent_auxiliary/target_detached": float(detach_target),
    }


def _safe_rerank_auxiliary_loss(
    model_output: ModelOutput,
    targets: PredictionTargets,
    cfg: dict[str, Any],
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_cfg = cfg.get("loss", {}) if isinstance(cfg.get("loss"), dict) else {}
    objective_cfg = _mapping(loss_cfg.get("objective"))
    raw = {
        **_mapping(loss_cfg.get("rerank")),
        **_mapping(loss_cfg.get("safe_rerank")),
        **_mapping(loss_cfg.get("safe_residual_rerank")),
        **_mapping(objective_cfg.get("rerank")),
        **_mapping(objective_cfg.get("safe_rerank")),
    }
    enabled = bool(raw.get("enabled", False))
    total_weight = float(raw.get("weight", raw.get("lambda", 0.0)) or 0.0)
    if not enabled or total_weight <= 0.0:
        return zero, {}
    diagnostics = model_output.diagnostics
    candidate_ids = diagnostics.get("candidate_ids")
    rerank_logits = diagnostics.get("rerank_logits", model_output.logits)
    anchor_logits = diagnostics.get("anchor_logits")
    if not torch.is_tensor(candidate_ids) or not torch.is_tensor(rerank_logits) or not torch.is_tensor(anchor_logits):
        if bool(raw.get("allow_missing", False)):
            return zero, {"loss/rerank_total": 0.0, "rerank_loss/skipped_samples": 0.0}
        raise ValueError("Safe rerank loss requires candidate_ids, rerank_logits and anchor_logits diagnostics.")
    labels = targets.labels.to(device=rerank_logits.device, dtype=torch.long)
    labels = _match_time(labels, int(rerank_logits.shape[1]))
    candidate_ids = _match_time(candidate_ids.to(device=rerank_logits.device, dtype=torch.long), int(rerank_logits.shape[1]))
    anchor_logits = _match_time(anchor_logits.to(device=rerank_logits.device), int(rerank_logits.shape[1]))
    valid_candidates = candidate_ids.ge(0)
    safe_ids = candidate_ids.clamp_min(0)
    candidate_logits = torch.gather(rerank_logits, 2, safe_ids).float()
    candidate_logits = candidate_logits.masked_fill(~valid_candidates, -1e9)
    valid_labels = labels.ge(0) & labels.lt(int(rerank_logits.shape[-1]))
    safe_labels = labels.clamp(0, int(rerank_logits.shape[-1]) - 1)
    target_in_candidates = candidate_ids.eq(labels.unsqueeze(-1)) & valid_candidates & valid_labels.unsqueeze(-1)
    covered = target_in_candidates.any(dim=-1) & valid_labels
    sample_count = int(valid_labels.sum().detach().cpu().item())
    coverage_count = int(covered.sum().detach().cpu().item())
    skipped = sample_count - coverage_count

    candidate_ce = zero
    if coverage_count > 0:
        local_targets = target_in_candidates.to(dtype=torch.long).argmax(dim=-1)
        candidate_ce = F.cross_entropy(candidate_logits[covered], local_targets[covered])

    pairwise_margin = zero
    pair_weight = float(raw.get("pairwise_margin_weight", raw.get("margin_weight", 0.0)) or 0.0)
    margin = float(raw.get("margin", 0.1))
    if pair_weight > 0.0 and sample_count > 0:
        target_score = torch.gather(rerank_logits, 2, safe_labels.unsqueeze(-1)).squeeze(-1)
        anchor_top = anchor_logits.argmax(dim=-1)
        anchor_score = torch.gather(rerank_logits, 2, anchor_top.unsqueeze(-1)).squeeze(-1)
        pairwise_margin = F.relu(margin - (target_score - anchor_score))[valid_labels].mean()

    no_regret = zero
    no_regret_weight = float(raw.get("no_regret_weight", raw.get("consistency_weight", 0.0)) or 0.0)
    anchor_top = anchor_logits.argmax(dim=-1)
    if no_regret_weight > 0.0:
        protect = anchor_top.eq(labels) & valid_labels
        dba_threshold = raw.get("anchor_dba_threshold")
        if dba_threshold is not None:
            protect = protect | (
                _circular_distance(anchor_top, safe_labels, int(rerank_logits.shape[-1])).le(float(dba_threshold))
                & valid_labels
            )
        if bool(protect.any().detach().cpu().item()):
            no_regret = F.cross_entropy(rerank_logits[protect], anchor_top[protect])

    ce_weight = float(raw.get("candidate_ce_weight", raw.get("ce_weight", 1.0)) or 0.0)
    loss = total_weight * (ce_weight * candidate_ce + pair_weight * pairwise_margin + no_regret_weight * no_regret)
    residual_scale = diagnostics.get("residual_scale")
    residual_scale_mean = float(residual_scale.detach().float().mean().cpu().item()) if torch.is_tensor(residual_scale) else 0.0
    return loss, {
        "loss/rerank_total": float(loss.detach().cpu().item()),
        "loss/rerank_candidate_ce": float((candidate_ce * ce_weight * total_weight).detach().cpu().item()),
        "loss/rerank_pairwise_margin": float((pairwise_margin * pair_weight * total_weight).detach().cpu().item()),
        "loss/no_regret_consistency": float((no_regret * no_regret_weight * total_weight).detach().cpu().item()),
        "rerank_loss/candidate_coverage": float(coverage_count / max(sample_count, 1)),
        "rerank_loss/skipped_samples": float(skipped),
        "rerank_loss/anchor_correct_count": float((anchor_top.eq(labels) & valid_labels).sum().detach().cpu().item()),
        "rerank_loss/residual_scale_mean": residual_scale_mean,
        "rerank_loss/weight": float(total_weight),
    }


def _match_time(value: torch.Tensor, target_steps: int) -> torch.Tensor:
    if value.ndim < 2:
        return value
    if int(value.shape[1]) == int(target_steps):
        return value
    if int(value.shape[1]) > int(target_steps):
        return value[:, -int(target_steps) :, ...]
    pad_shape = (int(value.shape[0]), int(target_steps) - int(value.shape[1]), *tuple(value.shape[2:]))
    pad_value = -1 if value.dtype in (torch.long, torch.int64, torch.int32, torch.int16, torch.int8) else 0
    pad = torch.full(pad_shape, pad_value, dtype=value.dtype, device=value.device)
    return torch.cat([pad, value], dim=1)


def _circular_distance(prediction: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    absolute = (prediction.to(dtype=torch.long) - target.to(dtype=torch.long)).abs()
    return torch.minimum(absolute, torch.as_tensor(num_classes, dtype=absolute.dtype, device=absolute.device) - absolute)


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
    "build_dba_aware_soft_targets",
    "dba_aware_loss_config",
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
