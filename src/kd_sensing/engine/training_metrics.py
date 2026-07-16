from typing import Any

import numpy as np

from kd_sensing.engine.training_extensions import EpochDiagnosticsAccumulator
from kd_sensing.evaluation.horizon_selection import aggregate_topk_and_dba


HISTORY_FIELDS = (
    "train_loss",
    "train_task_loss",
    "train_auxiliary_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_atop3",
    "val_atop5",
    "val_adba",
    "learning_rates",
)


class EpochMetricsRecorder:
    def __init__(self, *, objective_metadata: dict[str, Any]) -> None:
        self.objective_metadata = objective_metadata
        self.history = {key: [] for key in HISTORY_FIELDS}
        self.epoch_logs: list[dict[str, Any]] = []
        self.running: dict[str, float] = {}
        self.numerators: dict[str, float] = {}
        self.denominators: dict[str, float] = {}
        self.diagnostics = EpochDiagnosticsAccumulator()
        self.batch_count = 0

    def start_epoch(self, current_lr: float) -> None:
        keys = ("loss", "task_loss", "auxiliary_loss", "beam_loss", "acc")
        self.running = {key: 0.0 for key in keys}
        self.numerators = {key: 0.0 for key in keys}
        self.denominators = {key: 0.0 for key in keys}
        self.diagnostics = EpochDiagnosticsAccumulator()
        self.batch_count = 0
        self.history["learning_rates"].append(float(current_lr))

    def update_batch(self, result, step: int) -> dict[str, float]:
        self.batch_count = step + 1
        numerators = result.metric_numerators
        denominators = result.metric_denominators
        if float(denominators.get("loss", 0.0)) <= 0.0:
            raise ValueError("Training loss has zero effective observations.")
        for key in self.running:
            denominator = float(denominators.get(key, 0.0))
            if denominator < 0.0:
                raise ValueError(f"Training metric '{key}' has a negative observation count.")
            self.numerators[key] += float(numerators.get(key, 0.0))
            self.denominators[key] += denominator
            if self.denominators[key]:
                self.running[key] = self.numerators[key] / self.denominators[key]
        self.diagnostics.update(result.scalar_diagnostics)
        return self.progress_metrics()

    def progress_metrics(self) -> dict[str, float]:
        return {"loss": self.running["loss"], "task": self.running["task_loss"], "acc": self.running["acc"]}

    def finish_epoch(
        self,
        *,
        epoch: int,
        total_epochs: int,
        val_metrics: dict[str, Any] | None,
        current_lr: float,
        optimizer_groups: list[dict[str, Any]],
        health_metrics: dict[str, Any] | None = None,
        extension_metrics: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], float | None, float | None]:
        validation = val_metrics or {}
        curves = aggregate_validation_metrics(validation) if val_metrics is not None else _empty_validation_curves()
        val_loss = float(validation["loss"]) if val_metrics is not None else None
        val_acc = float(curves["val_acc"]) if val_metrics is not None else None
        self.history["train_loss"].append(self.running["loss"])
        self.history["train_task_loss"].append(self.running["task_loss"])
        self.history["train_auxiliary_loss"].append(self.running["auxiliary_loss"])
        self.history["train_acc"].append(self.running["acc"])
        for key, value in {"val_loss": val_loss, **curves}.items():
            self.history[key].append(np.nan if value is None else float(value))
        log: dict[str, Any] = {
            "epoch": int(epoch) + 1,
            "total_epochs": int(total_epochs),
            "train_batches": self.batch_count,
            "objective": "beam",
            "primary_loss": self.objective_metadata["primary_loss"],
            "train_loss": self.running["loss"],
            "train_task_loss": self.running["task_loss"],
            "train_auxiliary_loss": self.running["auxiliary_loss"],
            "train_acc": self.running["acc"],
            "train_loss_observation_count": _observation_count(self.denominators["loss"]),
            "train_accuracy_observation_count": _observation_count(self.denominators["acc"]),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_atop3": curves["val_atop3"],
            "val_atop5": curves["val_atop5"],
            "val_adba": curves["val_adba"],
            "learning_rate": float(current_lr),
            "validation_metrics": validation if val_metrics is not None else None,
        }
        if health_metrics:
            log.update(health_metrics)
        if extension_metrics:
            log.update(extension_metrics)
        log.update(self.diagnostics.mean())
        for group in optimizer_groups:
            log[f"optimizer/lr/{group['name']}"] = float(group["lr"])
            log[f"optimizer/params/{group['name']}"] = float(group["param_count"])
        self.epoch_logs.append(log)
        return log, val_loss, val_acc


def aggregate_validation_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    aggregate = aggregate_topk_and_dba(metrics)
    return {
        "val_acc": aggregate["top1"],
        "val_atop3": aggregate["top3"],
        "val_atop5": aggregate["top5"],
        "val_adba": aggregate["adba"],
    }


def training_outputs_payload(history: dict[str, list], objective_metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    payload = {key: np.asarray(values) for key, values in history.items()}
    payload.update(
        {
            "objective": np.asarray(objective_metadata["name"]),
            "primary_loss": np.asarray(objective_metadata["primary_loss"]),
            "loss_weight_names": np.asarray(["beam"], dtype=object),
            "loss_weights": np.asarray([1.0], dtype=float),
        }
    )
    return payload


def _empty_validation_curves() -> dict[str, float | None]:
    return {"val_acc": None, "val_atop3": None, "val_atop5": None, "val_adba": None}


def _observation_count(value: float) -> int | float:
    rounded = round(value)
    return int(rounded) if abs(value - rounded) < 1e-9 else value
