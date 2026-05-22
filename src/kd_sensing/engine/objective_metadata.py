from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PREDICTION_OBJECTIVES = (
    "beam",
    "occlusion",
    "position",
    "multitask",
    "current_beam_selection",
    "current_los_classification",
    "current_link_quality",
    "selection_multitask",
)

_DEFAULT_METRICS: dict[str, tuple[str, str]] = {
    "beam": ("val_adba", "max"),
    "occlusion": ("val_occlusion_blocked_f1", "max"),
    "position": ("val_position_rmse", "min"),
    "multitask": ("val_multitask_loss", "min"),
    "current_beam_selection": ("val_beam_top1", "max"),
    "current_los_classification": ("val_los_f1", "max"),
    "current_link_quality": ("val_link_mae", "min"),
    "selection_multitask": ("val_selection_multitask_loss", "min"),
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
_CURRENT_BEAM_AVAILABLE_METRICS = (
    "val_beam_top1",
    "val_beam_top3",
    "val_beam_top5",
)
_CURRENT_LOS_AVAILABLE_METRICS = (
    "val_los_accuracy",
    "val_los_f1",
    "val_los_auc",
)
_CURRENT_LINK_AVAILABLE_METRICS = (
    "val_link_mae",
    "val_link_rmse",
    "val_link_r2",
)
_SELECTION_MULTITASK_AVAILABLE_METRICS = (
    *_CURRENT_BEAM_AVAILABLE_METRICS,
    "val_los_accuracy",
    "val_los_f1",
    "val_los_auc",
    "val_link_mae",
    "val_link_rmse",
    "val_link_r2",
    "val_selection_multitask_loss",
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
    "current_beam_selection": "val_beam_top1",
    "beam_selection": "val_beam_top1",
    "beam_top1": "val_beam_top1",
    "val_beam_top1": "val_beam_top1",
    "beam_top3": "val_beam_top3",
    "val_beam_top3": "val_beam_top3",
    "beam_top5": "val_beam_top5",
    "val_beam_top5": "val_beam_top5",
    "current_los_classification": "val_los_f1",
    "los": "val_los_f1",
    "los_classification": "val_los_f1",
    "selection_multitask": "val_selection_multitask_loss",
    "selection_multitask_loss": "val_selection_multitask_loss",
    "val_selection_multitask_loss": "val_selection_multitask_loss",
    "loss/selection_multitask_total": "val_selection_multitask_loss",
    "los_accuracy": "val_los_accuracy",
    "val_los_accuracy": "val_los_accuracy",
    "los_f1": "val_los_f1",
    "val_los_f1": "val_los_f1",
    "los_auc": "val_los_auc",
    "val_los_auc": "val_los_auc",
    "link_mae": "val_link_mae",
    "val_link_mae": "val_link_mae",
    "current_link_quality": "val_link_mae",
    "link_quality": "val_link_mae",
    "link_quality_regression": "val_link_mae",
    "link_rmse": "val_link_rmse",
    "val_link_rmse": "val_link_rmse",
    "link_r2": "val_link_r2",
    "val_link_r2": "val_link_r2",
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
    "val_beam_top1": "max",
    "val_beam_top3": "max",
    "val_beam_top5": "max",
    "val_los_accuracy": "max",
    "val_los_f1": "max",
    "val_los_auc": "max",
    "val_link_mae": "min",
    "val_link_rmse": "min",
    "val_link_r2": "max",
    "val_selection_multitask_loss": "min",
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

_SELECTION_HISTORY_FIELDS: tuple[str, ...] = (
    *_HISTORY_FIELDS[:-2],
    "train_los_loss",
    "train_link_quality_loss",
    "train_selection_multitask_loss",
    "val_beam_top1",
    "val_beam_top3",
    "val_beam_top5",
    "val_los_accuracy",
    "val_los_f1",
    "val_los_auc",
    "val_link_mae",
    "val_link_rmse",
    "val_link_r2",
    "val_selection_multitask_loss",
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
    "train_los_loss",
    "train_link_quality_loss",
    "train_selection_multitask_loss",
    "val_beam_top1",
    "val_beam_top3",
    "val_beam_top5",
    "val_los_accuracy",
    "val_los_f1",
    "val_los_auc",
    "val_link_mae",
    "val_link_rmse",
    "val_link_r2",
    "val_selection_multitask_loss",
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

_SELECTION_TENSORBOARD_SCALARS: tuple[tuple[str, str], ...] = (
    ("loss/selection_multitask_total", "train_selection_multitask_loss"),
    ("loss/val_selection_multitask_total", "val_selection_multitask_loss"),
    ("loss/los", "train_los_loss"),
    ("loss/link_quality", "train_link_quality_loss"),
    ("beam/val_top1", "val_beam_top1"),
    ("beam/val_top3", "val_beam_top3"),
    ("beam/val_top5", "val_beam_top5"),
    ("los/accuracy", "val_los_accuracy"),
    ("los/f1", "val_los_f1"),
    ("los/auc", "val_los_auc"),
    ("link/mae", "val_link_mae"),
    ("link/rmse", "val_link_rmse"),
    ("link/r2", "val_link_r2"),
)

_OBJECTIVE_AVAILABLE_METRICS: dict[str, tuple[str, ...]] = {
    "beam": (*_BASE_AVAILABLE_METRICS, *_BEAM_AVAILABLE_METRICS),
    "occlusion": (*_BASE_AVAILABLE_METRICS, *_OCCLUSION_AVAILABLE_METRICS),
    "position": (*_BASE_AVAILABLE_METRICS, *_POSITION_AVAILABLE_METRICS),
    "multitask": (*_BASE_AVAILABLE_METRICS, *_MULTITASK_AVAILABLE_METRICS),
    "current_beam_selection": (*_BASE_AVAILABLE_METRICS, *_CURRENT_BEAM_AVAILABLE_METRICS),
    "current_los_classification": (*_BASE_AVAILABLE_METRICS, *_CURRENT_LOS_AVAILABLE_METRICS),
    "current_link_quality": (*_BASE_AVAILABLE_METRICS, *_CURRENT_LINK_AVAILABLE_METRICS),
    "selection_multitask": (*_BASE_AVAILABLE_METRICS, *_SELECTION_MULTITASK_AVAILABLE_METRICS),
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
class AuxiliaryTaskConfig:
    occlusion_enabled: bool
    position_enabled: bool
    occlusion_weight: float
    position_weight: float
    pos_weight: str | float | None


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
    elif objective == "current_beam_selection":
        required_targets = ("target_beam",)
        required_outputs = ("logits",)
        primary_loss = "beam_selection"
    elif objective == "current_los_classification":
        required_targets = ("los_label",)
        required_outputs = ("los_logits",)
        primary_loss = "los"
    elif objective == "current_link_quality":
        required_targets = ("link_quality",)
        required_outputs = ("link_quality",)
        primary_loss = "link_quality"
    elif objective == "occlusion":
        required_targets = ("occlusion",)
        required_outputs = ("occlusion_logits",)
        primary_loss = "occlusion"
    elif objective == "position":
        required_targets = ("position",)
        required_outputs = ("position",)
        primary_loss = "position"
    elif objective == "multitask":
        required_targets = ("beam", "occlusion", "position")
        required_outputs = ("logits", "occlusion_logits", "position")
        primary_loss = "multitask_total"
    else:
        required_targets = ("target_beam", "los_label", "link_quality")
        required_outputs = ("logits", "los_logits", "link_quality")
        primary_loss = "selection_multitask_total"
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
        history_fields=_SELECTION_HISTORY_FIELDS
        if objective in {
            "current_beam_selection",
            "current_los_classification",
            "current_link_quality",
            "selection_multitask",
        }
        else _HISTORY_FIELDS,
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
    for key in (
        "beam_top1",
        "beam_top3",
        "beam_top5",
        "los_accuracy",
        "los_f1",
        "los_auc",
        "link_mae",
        "link_rmse",
        "link_r2",
    ):
        if key in metrics and _finite_number(metrics[key]):
            available.add(f"val_{key}")
    if "selection_multitask_loss" in metrics and _finite_number(metrics["selection_multitask_loss"]):
        available.add("val_selection_multitask_loss")
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
            objective = resolve_prediction_objective(cfg)
            heads.append("beam_selection" if objective in {"current_beam_selection", "selection_multitask"} else "beam")
        elif output == "occlusion_logits":
            heads.append("occlusion")
        elif output == "position":
            heads.append("position")
        elif output == "los_logits":
            heads.append("los")
        elif output == "link_quality":
            heads.append("link_quality")
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
        "loss_weights": _runtime_loss_weights(cfg, spec.name),
        **spec.runtime_metadata,
    }


def multitask_loss_weights(cfg: dict[str, Any]) -> dict[str, float]:
    loss_cfg = cfg.get("loss", {})
    objective_cfg = _mapping(loss_cfg.get("objective"))
    weights_cfg = _mapping(objective_cfg.get("weights"))
    multitask_cfg = _mapping(loss_cfg.get("multitask") or loss_cfg.get("multi_task"))
    auxiliary_cfg = _mapping(loss_cfg.get("auxiliary"))
    auxiliary_defaults = resolve_auxiliary_task_config(cfg)
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
            default=auxiliary_defaults.occlusion_weight,
        ),
        "position": _weight_from_configs(
            ("position", "position_weight", "lambda_position"),
            weights_cfg,
            objective_cfg,
            multitask_cfg,
            auxiliary_cfg,
            default=auxiliary_defaults.position_weight,
        ),
    }


def selection_multitask_loss_weights(cfg: dict[str, Any]) -> dict[str, float]:
    loss_cfg = cfg.get("loss", {})
    objective_cfg = _mapping(loss_cfg.get("objective"))
    weights_cfg = _mapping(objective_cfg.get("weights") or objective_cfg.get("loss_weights"))
    selection_cfg = _mapping(loss_cfg.get("selection_multitask") or loss_cfg.get("selection"))
    return {
        "beam_selection": _weight_from_configs(
            ("beam_selection", "beam", "beam_weight", "lambda_beam"),
            weights_cfg,
            objective_cfg,
            selection_cfg,
            default=1.0,
        ),
        "los": _weight_from_configs(
            ("los", "los_weight", "lambda_los"),
            weights_cfg,
            objective_cfg,
            selection_cfg,
            default=0.5,
        ),
        "link_quality": _weight_from_configs(
            ("link_quality", "link", "link_weight", "lambda_link"),
            weights_cfg,
            objective_cfg,
            selection_cfg,
            default=0.2,
        ),
    }


def resolve_auxiliary_task_config(cfg: dict[str, Any]) -> AuxiliaryTaskConfig:
    loss_cfg = cfg.get("loss", {})
    data_cfg = cfg.get("data", {}).get("dataset", {})
    model_cfg = cfg.get("model", {}).get("student", {})
    auxiliary_cfg = _first_mapping(
        loss_cfg.get("auxiliary"),
        loss_cfg.get("multitask"),
        loss_cfg.get("multi_task"),
    )
    dataset_occlusion = _enabled(data_cfg.get("occlusion_target"))
    dataset_position = _enabled(data_cfg.get("position_target"))
    heads_cfg = _first_mapping(model_cfg.get("auxiliary_heads"))
    head_occlusion = _head_enabled(heads_cfg, "occlusion")
    head_position = _head_enabled(heads_cfg, "position")

    occlusion_cfg = _first_mapping(auxiliary_cfg.get("occlusion"), loss_cfg.get("occlusion"))
    position_cfg = _first_mapping(auxiliary_cfg.get("position"), loss_cfg.get("position"))
    auxiliary_enabled = bool(auxiliary_cfg.get("enabled", dataset_occlusion or dataset_position))

    occlusion_enabled = bool(
        auxiliary_enabled
        and (
            _enabled(occlusion_cfg, default=dataset_occlusion or head_occlusion)
            or dataset_occlusion
            or float(auxiliary_cfg.get("occlusion_weight", auxiliary_cfg.get("lambda_occlusion", 0.0))) > 0.0
        )
    )
    position_enabled = bool(
        auxiliary_enabled
        and (
            _enabled(position_cfg, default=dataset_position or head_position)
            or dataset_position
            or float(auxiliary_cfg.get("position_weight", auxiliary_cfg.get("lambda_position", 0.0))) > 0.0
        )
    )

    occlusion_weight = _weight_value(
        occlusion_cfg,
        auxiliary_cfg,
        keys=("weight", "occlusion_weight", "lambda_occlusion"),
        default=1.0,
    )
    position_weight = _weight_value(
        position_cfg,
        auxiliary_cfg,
        keys=("weight", "position_weight", "lambda_position"),
        default=0.01,
    )
    pos_weight = occlusion_cfg.get("pos_weight", auxiliary_cfg.get("pos_weight", None))
    return AuxiliaryTaskConfig(
        occlusion_enabled=occlusion_enabled,
        position_enabled=position_enabled,
        occlusion_weight=occlusion_weight,
        position_weight=position_weight,
        pos_weight=pos_weight,
    )


def _runtime_loss_weights(cfg: dict[str, Any], objective: str) -> dict[str, float]:
    if objective == "selection_multitask":
        return selection_multitask_loss_weights(cfg)
    if objective in {"current_beam_selection", "current_los_classification", "current_link_quality"}:
        return {}
    return multitask_loss_weights(cfg)


def _tensorboard_scalars_for_objective(objective: str) -> tuple[tuple[str, str], ...]:
    scalars = list(_COMMON_TENSORBOARD_SCALARS)
    if objective in {"beam", "multitask"}:
        scalars.extend(_BEAM_TENSORBOARD_SCALARS)
    if objective in {
        "current_beam_selection",
        "current_los_classification",
        "current_link_quality",
        "selection_multitask",
    }:
        scalars.extend(_SELECTION_TENSORBOARD_SCALARS)
    else:
        scalars.extend(_AUXILIARY_TENSORBOARD_SCALARS)
    return tuple(scalars)


def _finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value == value and value not in (float("inf"), float("-inf"))
    return False


def _is_allowed_pattern_metric(name: str, *, objective: str) -> bool:
    if objective in {"current_beam_selection", "selection_multitask"}:
        return name.startswith(("val_beam_top",))
    if objective not in {"beam", "multitask"}:
        return False
    return name.startswith(("val_top1_", "val_top3_", "val_top5_"))


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _enabled(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return bool(value.get("enabled", value.get("enable", default)))
    return default


def _head_enabled(heads_cfg: dict[str, Any], key: str) -> bool:
    if not heads_cfg:
        return False
    return bool(heads_cfg.get(key, heads_cfg.get(f"{key}_head", heads_cfg.get("enabled", False))))


def _weight_value(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    *,
    keys: tuple[str, ...],
    default: float,
) -> float:
    for key in keys:
        if key in primary:
            return float(primary[key])
    for key in keys:
        if key in secondary:
            return float(secondary[key])
    return float(default)


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
    "AuxiliaryTaskConfig",
    "PredictionObjectiveSpec",
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
    "produced_metric_names",
    "resolve_auxiliary_task_config",
    "resolve_prediction_objective",
    "selection_multitask_loss_weights",
    "validate_objective_metric_available",
]
