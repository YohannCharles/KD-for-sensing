from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.engine.auxiliary import (
    compute_auxiliary_multitask_loss,
    resolve_auxiliary_task_config,
)
from kd_sensing.engine.model_output import ModelOutput


PREDICTION_OBJECTIVES = ("beam", "occlusion", "position", "multitask")

_DEFAULT_METRICS: dict[str, tuple[str, str]] = {
    "beam": ("val_adba", "max"),
    "occlusion": ("val_occlusion_blocked_f1", "max"),
    "position": ("val_position_rmse", "min"),
    "multitask": ("val_multitask_loss", "min"),
}

_BASE_AVAILABLE_METRICS = ("val_loss",)
_BEAM_AVAILABLE_METRICS = (
    "val_acc",
    "val_adba",
    "val_atop3",
    "val_atop5",
    "val_top1_avg",
    "val_top3_avg",
    "val_top5_avg",
)
_OCCLUSION_AVAILABLE_METRICS = (
    "val_occlusion_accuracy",
    "val_occlusion_blocked_f1",
)
_POSITION_AVAILABLE_METRICS = (
    "val_position_rmse",
    "val_position_mae",
)
_MULTITASK_AVAILABLE_METRICS = (
    *_BEAM_AVAILABLE_METRICS,
    *_OCCLUSION_AVAILABLE_METRICS,
    *_POSITION_AVAILABLE_METRICS,
    "val_multitask_loss",
)

_METRIC_ALIASES: dict[str, str] = {
    "adba": "val_adba",
    "dba": "val_adba",
    "val_adba": "val_adba",
    "val_dba": "val_adba",
    "dba/val_adba": "val_adba",
    "beam/adba": "val_adba",
    "beam/dba": "val_adba",
    "beam/val_adba": "val_adba",
    "beam/val_dba": "val_adba",
    "top1": "val_acc",
    "val_top1": "val_acc",
    "val_acc": "val_acc",
    "val_acc_top1": "val_acc",
    "top1_val_acc": "val_acc",
    "accuracy/val": "val_acc",
    "accuracy/val_top1": "val_acc",
    "beam/accuracy_val": "val_acc",
    "beam/val_top1": "val_acc",
    "beam/val_acc": "val_acc",
    "val/acc_top1": "val_acc",
    "val/top1": "val_acc",
    "loss": "val_loss",
    "val_loss": "val_loss",
    "loss/val": "val_loss",
    "occlusion": "val_occlusion_blocked_f1",
    "occlusion_f1": "val_occlusion_blocked_f1",
    "blocked_f1": "val_occlusion_blocked_f1",
    "val_occlusion_blocked_f1": "val_occlusion_blocked_f1",
    "occlusion/blocked_f1": "val_occlusion_blocked_f1",
    "position": "val_position_rmse",
    "position_rmse": "val_position_rmse",
    "val_position_rmse": "val_position_rmse",
    "position/rmse": "val_position_rmse",
    "multitask": "val_multitask_loss",
    "multitask_loss": "val_multitask_loss",
    "val_multitask_loss": "val_multitask_loss",
    "loss/multitask_total": "val_multitask_loss",
}

_METRIC_MODES: dict[str, str] = {
    "val_loss": "min",
    "val_acc": "max",
    "val_adba": "max",
    "val_atop3": "max",
    "val_atop5": "max",
    "val_top1_avg": "max",
    "val_top3_avg": "max",
    "val_top5_avg": "max",
    "val_occlusion_accuracy": "max",
    "val_occlusion_blocked_f1": "max",
    "val_position_rmse": "min",
    "val_position_mae": "min",
    "val_multitask_loss": "min",
}

_HISTORY_FIELDS: tuple[str, ...] = (
    "train_loss",
    "train_task_loss",
    "train_objective_loss",
    "train_distill_loss",
    "train_beam_soft_loss",
    "train_unimodal_loss",
    "train_counterfactual_loss",
    "train_prior_regularization_loss",
    "train_reliability_kd_loss",
    "train_occlusion_loss",
    "train_position_loss",
    "train_multitask_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_atop3",
    "val_atop5",
    "val_adba",
    "val_occlusion_accuracy",
    "val_occlusion_blocked_f1",
    "val_position_rmse",
    "val_position_mae",
    "val_multitask_loss",
    "val_primary_metric",
    "learning_rates",
)

_OPTIONAL_HISTORY_FIELDS = {
    "train_occlusion_loss",
    "train_position_loss",
    "train_multitask_loss",
    "val_occlusion_accuracy",
    "val_occlusion_blocked_f1",
    "val_position_rmse",
    "val_position_mae",
    "val_multitask_loss",
}

_COMMON_TENSORBOARD_SCALARS: tuple[tuple[str, str], ...] = (
    ("loss/train_objective", "train_objective_loss"),
    ("objective/val_primary_metric", "val_primary_metric"),
    ("loss/train_beam_soft", "train_beam_soft_loss"),
    ("loss/train_unimodal_aux", "train_unimodal_loss"),
    ("loss/train_counterfactual_gate", "train_counterfactual_loss"),
    ("loss/train_prior_regularization", "train_prior_regularization_loss"),
    ("loss/train_reliability_kd", "train_reliability_kd_loss"),
)

_BEAM_TENSORBOARD_SCALARS: tuple[tuple[str, str], ...] = (
    ("beam/accuracy_train", "train_acc"),
    ("beam/accuracy_val", "val_acc"),
    ("beam/val_atop3", "val_atop3"),
    ("beam/val_atop5", "val_atop5"),
    ("beam/val_adba", "val_adba"),
)

_AUXILIARY_TENSORBOARD_SCALARS: tuple[tuple[str, str], ...] = (
    ("loss/multitask_total", "train_multitask_loss"),
    ("loss/val_multitask_total", "val_multitask_loss"),
    ("loss/occlusion", "train_occlusion_loss"),
    ("loss/position", "train_position_loss"),
    ("occlusion/accuracy", "val_occlusion_accuracy"),
    ("occlusion/blocked_f1", "val_occlusion_blocked_f1"),
    ("position/rmse", "val_position_rmse"),
    ("position/mae", "val_position_mae"),
)

_OBJECTIVE_AVAILABLE_METRICS: dict[str, tuple[str, ...]] = {
    "beam": (*_BASE_AVAILABLE_METRICS, *_BEAM_AVAILABLE_METRICS),
    "occlusion": (*_BASE_AVAILABLE_METRICS, *_OCCLUSION_AVAILABLE_METRICS),
    "position": (*_BASE_AVAILABLE_METRICS, *_POSITION_AVAILABLE_METRICS),
    "multitask": (*_BASE_AVAILABLE_METRICS, *_MULTITASK_AVAILABLE_METRICS),
}


@dataclass(frozen=True)
class PredictionObjectiveSpec:
    name: str
    required_targets: tuple[str, ...]
    required_outputs: tuple[str, ...]
    primary_loss_name: str
    default_metric: str
    default_metric_mode: str
    available_metrics: tuple[str, ...]
    metric_aliases: dict[str, str]
    metric_modes: dict[str, str]
    history_fields: tuple[str, ...]
    tensorboard_scalars: tuple[tuple[str, str], ...]
    runtime_metadata: dict[str, Any]


@dataclass(frozen=True)
class PredictionTargets:
    labels: torch.Tensor
    occlusion_label: torch.Tensor | None = None
    occlusion_valid: torch.Tensor | None = None
    position_target: torch.Tensor | None = None
    position_valid: torch.Tensor | None = None

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


def resolve_prediction_objective(cfg: dict[str, Any]) -> str:
    raw = cfg.get("experiment", {}).get("objective", "beam")
    objective = str(raw).strip().lower()
    if objective not in PREDICTION_OBJECTIVES:
        supported = ", ".join(PREDICTION_OBJECTIVES)
        raise ValueError(f"experiment.objective must be one of: {supported}; got '{raw}'.")
    return objective


def objective_spec(cfg_or_objective: dict[str, Any] | str) -> PredictionObjectiveSpec:
    objective = (
        resolve_prediction_objective(cfg_or_objective)
        if isinstance(cfg_or_objective, dict)
        else str(cfg_or_objective).strip().lower()
    )
    if objective not in PREDICTION_OBJECTIVES:
        supported = ", ".join(PREDICTION_OBJECTIVES)
        raise ValueError(f"Unknown prediction objective '{objective}'. Supported objectives: {supported}.")
    metric, mode = _DEFAULT_METRICS[objective]
    required_targets: tuple[str, ...]
    required_outputs: tuple[str, ...]
    primary_loss: str
    if objective == "beam":
        required_targets = ("beam",)
        required_outputs = ("logits",)
        primary_loss = "beam"
    elif objective == "occlusion":
        required_targets = ("occlusion",)
        required_outputs = ("occlusion_logits",)
        primary_loss = "occlusion"
    elif objective == "position":
        required_targets = ("position",)
        required_outputs = ("position",)
        primary_loss = "position"
    else:
        required_targets = ("beam", "occlusion", "position")
        required_outputs = ("logits", "occlusion_logits", "position")
        primary_loss = "multitask_total"
    return PredictionObjectiveSpec(
        name=objective,
        required_targets=required_targets,
        required_outputs=required_outputs,
        primary_loss_name=primary_loss,
        default_metric=metric,
        default_metric_mode=mode,
        available_metrics=_OBJECTIVE_AVAILABLE_METRICS[objective],
        metric_aliases=dict(_METRIC_ALIASES),
        metric_modes=dict(_METRIC_MODES),
        history_fields=_HISTORY_FIELDS,
        tensorboard_scalars=_tensorboard_scalars_for_objective(objective),
        runtime_metadata={
            "default_metric": metric,
            "default_metric_mode": mode,
        },
    )


def default_primary_metric(objective: str) -> tuple[str, str]:
    spec = objective_spec(objective)
    return spec.default_metric, spec.default_metric_mode


def normalize_objective_metric(metric: object, *, objective: str = "beam") -> str:
    spec = objective_spec(objective)
    raw = spec.default_metric if metric is None else str(metric).strip()
    key = raw.lower().replace("-", "_")
    normalized = spec.metric_aliases.get(key, key)
    if normalized not in spec.metric_modes:
        supported = ", ".join(sorted(spec.metric_aliases))
        raise ValueError(f"Unsupported early stopping metric '{raw}'. Supported aliases: {supported}.")
    return normalized


def objective_metric_mode(metric: str, mode: object | None = None) -> str:
    if mode is None:
        return _METRIC_MODES.get(metric, "max")
    normalized = str(mode).strip().lower()
    if normalized not in {"min", "max"}:
        raise ValueError(f"training.early_stopping_mode must be 'min' or 'max', got '{mode}'.")
    return normalized


def objective_history_fields(cfg_or_objective: dict[str, Any] | str, *, include_compat: bool = True) -> tuple[str, ...]:
    spec = objective_spec(cfg_or_objective)
    if include_compat:
        return spec.history_fields
    available = set(spec.available_metrics)
    fields = []
    for key in spec.history_fields:
        if key.startswith("val_") and key not in available and key != "val_primary_metric":
            continue
        fields.append(key)
    return tuple(fields)


def objective_optional_history_fields() -> set[str]:
    return set(_OPTIONAL_HISTORY_FIELDS)


def objective_tensorboard_scalars(cfg_or_objective: dict[str, Any] | str) -> tuple[tuple[str, str], ...]:
    return objective_spec(cfg_or_objective).tensorboard_scalars


def produced_metric_names(metrics: dict[str, Any]) -> set[str]:
    available = set(metrics.get("available_metrics", [])) if isinstance(metrics.get("available_metrics"), list) else set()
    if "loss" in metrics:
        available.add("val_loss")
    if "topk" in metrics and "1" in metrics.get("topk", {}):
        available.add("val_acc")
    if "dba" in metrics:
        available.add("val_adba")
    for key, value in metrics.items():
        if key.startswith("val_") and _finite_number(value):
            available.add(key)
    return available


def objective_available_metrics(
    cfg_or_objective: dict[str, Any] | str,
    metrics: dict[str, Any] | None = None,
) -> list[str]:
    spec = objective_spec(cfg_or_objective)
    allowed = set(spec.available_metrics)
    if metrics is None:
        return sorted(allowed)
    produced = produced_metric_names(metrics)
    filtered = {
        name
        for name in produced
        if name in allowed or _is_allowed_pattern_metric(name, objective=spec.name)
    }
    return sorted(filtered)


def validate_objective_metric_available(metrics: dict[str, Any], metric: str) -> None:
    available = set(metrics.get("available_metrics", []))
    if not available:
        objective = metrics.get("objective", {}).get("name") if isinstance(metrics.get("objective"), dict) else "beam"
        available = set(objective_available_metrics(str(objective), metrics))
    if metric in available:
        return
    objective = metrics.get("objective", {}).get("name") if isinstance(metrics.get("objective"), dict) else None
    objective_text = f" for experiment.objective='{objective}'" if objective else ""
    available_text = ", ".join(sorted(available)) if available else "none"
    if metric == "val_adba" and "val_adba" not in produced_metric_names(metrics):
        reason = " because DBA/ADBA was not produced"
    else:
        reason = ""
    raise ValueError(
        f"Early stopping metric '{metric}' is not available in validation metrics{objective_text}{reason}. "
        f"Available metrics: {available_text}. Configure training.early_stopping_metric to a metric produced "
        "by the current objective."
    )


def configure_objective_defaults(
    cfg: dict[str, Any],
    *,
    explicit_early_stopping_metric: bool = False,
    explicit_early_stopping_mode: bool = False,
) -> None:
    experiment = cfg.setdefault("experiment", {})
    experiment["objective"] = resolve_prediction_objective(cfg)
    metric, mode = default_primary_metric(experiment["objective"])
    training = cfg.setdefault("training", {})
    if not explicit_early_stopping_metric:
        training["early_stopping_metric"] = metric
    if not explicit_early_stopping_mode:
        training["early_stopping_mode"] = mode


def objective_requires_occlusion(cfg: dict[str, Any]) -> bool:
    return resolve_prediction_objective(cfg) in {"occlusion", "multitask"}


def objective_requires_position(cfg: dict[str, Any]) -> bool:
    return resolve_prediction_objective(cfg) in {"position", "multitask"}


def objective_enabled_targets(cfg: dict[str, Any]) -> list[str]:
    return list(objective_spec(cfg).required_targets)


def objective_enabled_heads(cfg: dict[str, Any]) -> list[str]:
    heads = []
    for output in objective_spec(cfg).required_outputs:
        if output == "logits":
            heads.append("beam")
        elif output == "occlusion_logits":
            heads.append("occlusion")
        elif output == "position":
            heads.append("position")
    return heads


def objective_runtime_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    spec = objective_spec(cfg)
    return {
        "name": spec.name,
        "primary_loss": spec.primary_loss_name,
        "primary_metric": spec.default_metric,
        "primary_metric_mode": spec.default_metric_mode,
        "available_metrics": list(spec.available_metrics),
        "metric_aliases": dict(spec.metric_aliases),
        "metric_modes": dict(spec.metric_modes),
        "history_fields": list(spec.history_fields),
        "tensorboard_scalars": [
            {"tag": tag, "history_key": history_key}
            for tag, history_key in spec.tensorboard_scalars
        ],
        "enabled_targets": list(spec.required_targets),
        "enabled_heads": objective_enabled_heads(cfg),
        "loss_weights": multitask_loss_weights(cfg),
        **spec.runtime_metadata,
    }


def _tensorboard_scalars_for_objective(objective: str) -> tuple[tuple[str, str], ...]:
    scalars = list(_COMMON_TENSORBOARD_SCALARS)
    if objective in {"beam", "multitask"}:
        scalars.extend(_BEAM_TENSORBOARD_SCALARS)
    scalars.extend(_AUXILIARY_TENSORBOARD_SCALARS)
    return tuple(scalars)


def _finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value == value and value not in (float("inf"), float("-inf"))
    return False


def _is_allowed_pattern_metric(name: str, *, objective: str) -> bool:
    if objective not in {"beam", "multitask"}:
        return False
    return name.startswith(("val_top1_", "val_top3_", "val_top5_"))


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
    )
    if objective in {"occlusion", "multitask"}:
        _require_tensor(targets.occlusion_label, "occlusion_label", objective)
        _require_tensor(targets.occlusion_valid, "occlusion_valid", objective)
    if objective in {"position", "multitask"}:
        _require_tensor(targets.position_target, "position_target", objective)
        _require_tensor(targets.position_valid, "position_valid", objective)
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
    diagnostics = {"loss/beam": float(beam_primary.detach().cpu().item())}

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
        )

    if objective in {"occlusion", "multitask"}:
        occlusion_loss = _occlusion_loss(model_output, targets, cfg, zero)
        diagnostics["loss/occlusion"] = float(occlusion_loss.detach().cpu().item())

    if objective in {"position", "multitask"}:
        position_loss = _position_loss(model_output, targets, cfg, zero)
        diagnostics["loss/position"] = float(position_loss.detach().cpu().item())

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
    )


def multitask_loss_weights(cfg: dict[str, Any]) -> dict[str, float]:
    loss_cfg = cfg.get("loss", {})
    objective_cfg = _mapping(loss_cfg.get("objective"))
    weights_cfg = _mapping(objective_cfg.get("weights"))
    multitask_cfg = _mapping(loss_cfg.get("multitask") or loss_cfg.get("multi_task"))
    auxiliary_cfg = _mapping(loss_cfg.get("auxiliary"))
    return {
        "beam": _weight_from_configs(
            ("beam", "beam_weight", "lambda_beam"),
            weights_cfg,
            objective_cfg,
            multitask_cfg,
            default=1.0,
        ),
        "occlusion": _weight_from_configs(
            ("occlusion", "occlusion_weight", "lambda_occlusion"),
            weights_cfg,
            objective_cfg,
            multitask_cfg,
            auxiliary_cfg,
            default=resolve_auxiliary_task_config(cfg).occlusion_weight,
        ),
        "position": _weight_from_configs(
            ("position", "position_weight", "lambda_position"),
            weights_cfg,
            objective_cfg,
            multitask_cfg,
            auxiliary_cfg,
            default=resolve_auxiliary_task_config(cfg).position_weight,
        ),
    }


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


def _objective_loss_cfg(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    loss_cfg = cfg.get("loss", {})
    objective_cfg = _mapping(loss_cfg.get("objective"))
    auxiliary_cfg = _mapping(loss_cfg.get("auxiliary"))
    return {
        **_mapping(auxiliary_cfg.get(name)),
        **_mapping(loss_cfg.get(name)),
        **_mapping(objective_cfg.get(name)),
    }


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


def _weight_from_configs(
    keys: tuple[str, ...],
    *configs: dict[str, Any],
    default: float,
) -> float:
    for cfg in configs:
        for key in keys:
            if key in cfg:
                value = cfg[key]
                if isinstance(value, dict):
                    if "weight" in value:
                        return float(value["weight"])
                    continue
                return float(value)
    return float(default)


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
    "validate_objective_metric_available",
]
