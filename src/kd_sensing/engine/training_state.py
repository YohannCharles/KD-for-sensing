from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kd_sensing.engine.prediction_objectives import (
    normalize_objective_metric,
    objective_metric_mode,
    validate_objective_metric_available,
)


@dataclass
class TrainingState:
    start_epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_top1: float = float("-inf")
    best_top1_epoch: int = 0
    best_early_stopping_value: float = float("-inf")
    best_early_stopping_epoch: int = 0
    registry_checkpoint: dict[str, Any] | None = None
    epochs_without_improvement: int = 0
    checkpoint_loads: list[dict[str, Any] | None] = field(default_factory=list)
    history: dict[str, list] = field(default_factory=dict)
    epoch_logs: list[dict[str, Any]] = field(default_factory=list)

    def apply_resume_checkpoint(
        self,
        checkpoint: dict[str, Any],
        *,
        early_stopping_metric: str,
        early_stopping_mode: str,
        objective: str,
    ) -> tuple[str, str]:
        self.start_epoch = int(checkpoint.get("epoch", self.start_epoch))
        self.best_val_loss = float(checkpoint.get("best_val_loss", checkpoint.get("test_loss", self.best_val_loss)))
        self.best_val_top1 = float(checkpoint.get("best_val_top1", self.best_val_top1))
        self.best_top1_epoch = int(checkpoint.get("best_top1_epoch", self.best_top1_epoch))
        if "early_stopping_metric" in checkpoint:
            early_stopping_metric = normalize_early_stopping_metric(
                checkpoint["early_stopping_metric"],
                objective=objective,
            )
        if "early_stopping_mode" in checkpoint:
            early_stopping_mode = resolve_early_stopping_mode(
                early_stopping_metric,
                checkpoint["early_stopping_mode"],
            )
        self.best_early_stopping_value = float(
            checkpoint.get(
                "best_early_stopping_value",
                legacy_early_stopping_value(
                    checkpoint,
                    early_stopping_metric,
                    self.best_early_stopping_value,
                ),
            )
        )
        self.best_early_stopping_epoch = int(
            checkpoint.get(
                "best_early_stopping_epoch",
                legacy_early_stopping_epoch(
                    checkpoint,
                    early_stopping_metric,
                    self.best_early_stopping_epoch,
                ),
            )
        )
        self.registry_checkpoint = checkpoint.get("checkpoint_registry", self.registry_checkpoint)
        self.epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        return early_stopping_metric, early_stopping_mode


def normalize_early_stopping_metric(metric: object, *, objective: str = "beam") -> str:
    return normalize_objective_metric(metric, objective=objective)


def resolve_early_stopping_mode(metric: str, mode: object | None) -> str:
    return objective_metric_mode(metric, mode)


def configure_early_stopping(training_cfg: dict, objective: str = "beam") -> tuple[str, str]:
    metric = normalize_early_stopping_metric(training_cfg.get("early_stopping_metric"), objective=objective)
    mode = resolve_early_stopping_mode(metric, training_cfg.get("early_stopping_mode"))
    training_cfg["early_stopping_metric"] = metric
    training_cfg["early_stopping_mode"] = mode
    return metric, mode


def initial_early_stopping_value(mode: str) -> float:
    return float("inf") if mode == "min" else float("-inf")


def early_stopping_min_epoch(total_epochs: int) -> int:
    total_epochs = max(int(total_epochs), 0)
    return (total_epochs + 1) // 2


def early_stopping_improved(current: float, best: float, *, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return current < best - min_delta
    if mode == "max":
        return current > best + min_delta
    raise ValueError(f"Unsupported early stopping mode '{mode}'.")


def early_stopping_metric_value(epoch_log: dict, metric: str) -> float:
    if metric not in epoch_log:
        raise ValueError(
            f"Early stopping metric '{metric}' is not available in validation metrics. "
            "Ensure validation produces DBA/ADBA metrics or configure "
            "training.early_stopping_metric to another supported metric such as val_loss."
        )
    value = epoch_log[metric]
    if value is None:
        raise ValueError(
            f"Early stopping metric '{metric}' is None. Configure a supported metric with a numeric value."
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Early stopping metric '{metric}' must be numeric, got {value!r}.") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"Early stopping metric '{metric}' must be finite, got {numeric}.")
    return numeric


def validate_early_stopping_source_available(val_metrics: dict, metric: str) -> None:
    validate_objective_metric_available(val_metrics, metric)


def available_early_stopping_metrics(val_metrics: dict) -> set[str]:
    available = val_metrics.get("available_metrics", [])
    return set(available) if isinstance(available, list) else set()


def legacy_early_stopping_value(checkpoint: dict, metric: str, default: float) -> float:
    if metric == "val_loss":
        return float(checkpoint.get("best_val_loss", checkpoint.get("test_loss", default)))
    if metric == "val_acc":
        return float(checkpoint.get("best_val_top1", default))
    return default


def legacy_early_stopping_epoch(checkpoint: dict, metric: str, default: int) -> int:
    if metric == "val_acc":
        return int(checkpoint.get("best_top1_epoch", default))
    return default


def early_stopping_state(
    *,
    metric: str,
    mode: str,
    best_value: float,
    best_epoch: int,
    epochs_without_improvement: int,
) -> dict:
    return {
        "metric": metric,
        "mode": mode,
        "best_value": float(best_value),
        "best_epoch": int(best_epoch),
        "epochs_without_improvement": int(epochs_without_improvement),
    }
