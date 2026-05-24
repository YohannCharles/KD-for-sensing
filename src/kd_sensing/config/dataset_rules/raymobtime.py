from __future__ import annotations

from typing import Any

from kd_sensing.config.normalization import RAYMOBTIME_SELECTION_MODEL_TYPES
from kd_sensing.engine.objectives.metadata import resolve_prediction_objective


def validate_raymobtime_config(cfg: dict[str, Any]) -> None:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if dataset_cfg.get("type") != "raymobtime_s008":
        return
    objective = resolve_prediction_objective(cfg)
    if objective not in {
        "current_beam_selection",
        "current_los_classification",
        "current_link_quality",
        "selection_multitask",
    }:
        raise ValueError(
            "data.dataset.type='raymobtime_s008' requires experiment.objective to be "
            "'current_beam_selection', 'current_los_classification', "
            "'current_link_quality', or 'selection_multitask'."
        )
    forbidden = find_forbidden_raymobtime_keys(cfg)
    if forbidden:
        keys = ", ".join(forbidden)
        raise ValueError(
            "Raymobtime s008 only supports current snapshot beam selection. "
            f"Remove future/transition configuration keys: {keys}."
        )
    model_cfg = cfg.get("model", {}).get("student", {})
    model_type = str(model_cfg.get("type", ""))
    if model_type not in RAYMOBTIME_SELECTION_MODEL_TYPES:
        raise ValueError(
            "Raymobtime s008 requires model.student.type to be simple_concat_multitask_selection "
            "or task_aware_gated_multitask_selection."
        )
    if cfg.get("distillation", {}).get("type", "no_kd") != "no_kd":
        raise ValueError("Raymobtime s008 selection configs must use distillation.type='no_kd'.")
    cfg.setdefault("experiment", {})["task_semantics"] = "current_snapshot_beam_selection"
    cfg["experiment"]["uses_history_window"] = False
    cfg["experiment"]["uses_temporal_core"] = False


def find_forbidden_raymobtime_keys(cfg: dict[str, Any]) -> list[str]:
    forbidden_tokens = (
        "future_beam",
        "beam_prediction_horizon",
        "beam_tracking",
        "los_transition",
        "beam_switch",
    )
    found: list[str] = []

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                dotted = f"{prefix}.{key}" if prefix else str(key)
                if any(token in str(key) for token in forbidden_tokens):
                    found.append(dotted)
                visit(item, dotted)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{prefix}[{index}]")

    visit(cfg, "")
    return sorted(found)
