from copy import deepcopy
from typing import Any

import numpy as np

from kd_sensing.engine.objectives.metadata import (
    objective_history_fields,
    objective_optional_history_fields,
)
from kd_sensing.engine.tensorboard_logging import finite_float_or_none
from kd_sensing.engine.training_extensions import EpochDiagnosticsAccumulator
from kd_sensing.engine.training_state import early_stopping_metric_value
from kd_sensing.evaluation.horizon_selection import aggregate_topk_and_dba, metric_horizons_from_metrics, selected_horizon_mean
from kd_sensing.evaluation.lidar_diagnostics import lidar_preprocessing_metadata_from_dataset


class EpochMetricsRecorder:
    def __init__(
        self,
        *,
        objective: str,
        objective_metadata: dict[str, Any],
        early_stopping_metric: str,
        early_stopping_mode: str,
    ) -> None:
        self.objective = objective
        self.objective_metadata = objective_metadata
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_mode = early_stopping_mode
        self.history = {field: [] for field in objective_history_fields(objective, include_compat=True)}
        self.epoch_logs: list[dict[str, Any]] = []
        self.running: dict[str, float] = {}
        self.metric_numerators: dict[str, float] = {}
        self.metric_denominators: dict[str, float] = {}
        self.epoch_diagnostics = EpochDiagnosticsAccumulator()
        self.batch_count = 0

    def update_early_stopping(self, *, metric: str, mode: str) -> None:
        self.early_stopping_metric = metric
        self.early_stopping_mode = mode

    def start_epoch(self, current_lr: float) -> None:
        self.running = {
            "loss": 0.0,
            "task_loss": 0.0,
            "beam_soft_loss": 0.0,
            "unimodal_loss": 0.0,
            "occlusion_loss": 0.0,
            "position_loss": 0.0,
            "multitask_loss": 0.0,
            "los_loss": 0.0,
            "link_quality_loss": 0.0,
            "selection_multitask_loss": 0.0,
            "jepa_loss": 0.0,
            "acc": 0.0,
        }
        self.metric_numerators = {key: 0.0 for key in self.running}
        self.metric_denominators = {key: 0.0 for key in self.running}
        self.batch_count = 0
        self.epoch_diagnostics = EpochDiagnosticsAccumulator()
        self.history["learning_rates"].append(float(current_lr))

    def update_batch(self, result, step: int) -> dict[str, float]:
        self.batch_count = step + 1
        numerators = getattr(result, "metric_numerators", None)
        denominators = getattr(result, "metric_denominators", None)
        if not isinstance(numerators, dict) or not isinstance(denominators, dict):
            raise TypeError("Batch result must provide metric_numerators and metric_denominators mappings.")
        if float(denominators.get("loss", 0.0)) <= 0.0:
            raise ValueError("Training loss has zero effective observations.")
        for key in self.running:
            numerator = float(numerators.get(key, 0.0))
            denominator = float(denominators.get(key, 0.0))
            if denominator < 0.0:
                raise ValueError(f"Training metric '{key}' has a negative observation count.")
            self.metric_numerators[key] += numerator
            self.metric_denominators[key] += denominator
            if self.metric_denominators[key] > 0.0:
                self.running[key] = self.metric_numerators[key] / self.metric_denominators[key]
        self.epoch_diagnostics.update(result.scalar_diagnostics)
        return self.progress_metrics()

    def _metric_value(self, key: str) -> float | None:
        if self.metric_denominators.get(key, 0.0) <= 0.0:
            return None
        return float(self.running[key])

    def progress_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.running["loss"]),
            "task": float(self.running["task_loss"]),
            "acc": float(self.running["acc"]),
        }

    def finish_epoch(
        self,
        *,
        epoch: int,
        total_epochs: int,
        val_metrics: dict[str, Any] | None,
        current_lr: float,
        optimizer_groups: list[dict[str, Any]],
        train_lidar_quality=None,
        train_dataset=None,
        epoch_subsampling: dict[str, Any] | None = None,
        health_metrics: dict[str, Any] | None = None,
        extension_metrics: dict[str, Any] | None = None,
        model_selection_enabled: bool = True,
    ) -> tuple[dict[str, Any], float | None, float | None, float | None]:
        has_validation = val_metrics is not None
        val_metrics = val_metrics or {}
        val_loss = float(val_metrics["loss"]) if has_validation else None
        total = val_metrics.get("total", [])
        horizons = metric_horizons_from_metrics(val_metrics, num_pred=len(total))
        top1 = _topk_values(val_metrics.get("topk"), 1)
        val_acc = mean_valid_slots(top1, total, horizons=horizons) if top1 else (0.0 if has_validation else None)
        validation_curve_metrics = (
            aggregate_validation_metrics(val_metrics)
            if has_validation
            else {"val_atop3": None, "val_atop5": None, "val_adba": None, "val_beam_dba": None}
        )

        val_occlusion_accuracy = finite_float_or_none(val_metrics.get("val_occlusion_accuracy"))
        val_occlusion_blocked_f1 = finite_float_or_none(val_metrics.get("val_occlusion_blocked_f1"))
        val_position_rmse = finite_float_or_none(val_metrics.get("val_position_rmse"))
        val_position_mae = finite_float_or_none(val_metrics.get("val_position_mae"))
        val_multitask_loss = finite_float_or_none(val_metrics.get("val_multitask_loss"))
        val_beam_top1 = finite_float_or_none(val_metrics.get("val_beam_top1"))
        val_beam_top3 = finite_float_or_none(val_metrics.get("val_beam_top3"))
        val_beam_top5 = finite_float_or_none(val_metrics.get("val_beam_top5"))
        val_beam_dba = finite_float_or_none(val_metrics.get("val_beam_dba"))
        val_los_accuracy = finite_float_or_none(val_metrics.get("val_los_accuracy"))
        val_los_f1 = finite_float_or_none(val_metrics.get("val_los_f1"))
        val_los_auc = finite_float_or_none(val_metrics.get("val_los_auc"))
        val_link_mae = finite_float_or_none(val_metrics.get("val_link_mae"))
        val_link_rmse = finite_float_or_none(val_metrics.get("val_link_rmse"))
        val_link_r2 = finite_float_or_none(val_metrics.get("val_link_r2"))
        val_selection_multitask_loss = finite_float_or_none(val_metrics.get("val_selection_multitask_loss"))
        val_jepa_loss = finite_float_or_none(val_metrics.get("val_jepa_loss"))
        val_jepa_mask_target_ratio = finite_float_or_none(val_metrics.get("val_jepa_mask_target_ratio"))
        val_jepa_mask_context_ratio = finite_float_or_none(val_metrics.get("val_jepa_mask_context_ratio"))
        val_jepa_ema_decay = finite_float_or_none(val_metrics.get("val_jepa_ema_decay"))
        active_occlusion = val_occlusion_accuracy is not None or val_occlusion_blocked_f1 is not None
        active_position = val_position_rmse is not None or val_position_mae is not None
        active_selection = (
            val_los_accuracy is not None
            or val_los_f1 is not None
            or val_link_mae is not None
            or val_selection_multitask_loss is not None
        )
        train_occlusion_loss = self._metric_value("occlusion_loss") if (
            self.objective in {"occlusion", "multitask"} or active_occlusion
        ) else None
        train_position_loss = (
            self._metric_value("position_loss") if self.objective in {"position", "multitask"} or active_position else None
        )
        train_multitask_loss = (
            self._metric_value("multitask_loss")
            if self.objective == "multitask" or (self.objective == "beam" and (active_occlusion or active_position))
            else None
        )
        train_los_loss = self._metric_value("los_loss") if self.objective == "selection_multitask" or active_selection else None
        train_link_quality_loss = (
            self._metric_value("link_quality_loss") if self.objective == "selection_multitask" or active_selection else None
        )
        train_selection_multitask_loss = (
            self._metric_value("selection_multitask_loss") if self.objective == "selection_multitask" else None
        )
        train_jepa_loss = self._metric_value("jepa_loss") if self.objective == "gps_conditioned_jepa" else None

        self.history["train_loss"].append(float(self.running["loss"]))
        self.history["train_task_loss"].append(float(self.running["task_loss"]))
        self.history["train_objective_loss"].append(float(self.running["task_loss"]))
        self.history["train_beam_soft_loss"].append(float(self.running["beam_soft_loss"]))
        self.history["train_unimodal_loss"].append(float(self.running["unimodal_loss"]))
        append_history(self.history, "train_jepa_loss", train_jepa_loss)
        append_history(self.history, "train_occlusion_loss", train_occlusion_loss)
        append_history(self.history, "train_position_loss", train_position_loss)
        append_history(self.history, "train_multitask_loss", train_multitask_loss)
        append_history(self.history, "train_los_loss", train_los_loss)
        append_history(self.history, "train_link_quality_loss", train_link_quality_loss)
        append_history(self.history, "train_selection_multitask_loss", train_selection_multitask_loss)
        append_history(self.history, "train_acc", float(self.running["acc"]))
        self.history["val_loss"].append(float(val_loss) if val_loss is not None else np.nan)
        append_history(self.history, "val_acc", val_acc if val_acc is not None else np.nan)
        append_history(
            self.history,
            "val_atop3",
            validation_curve_metrics["val_atop3"] if validation_curve_metrics["val_atop3"] is not None else np.nan,
        )
        append_history(
            self.history,
            "val_atop5",
            validation_curve_metrics["val_atop5"] if validation_curve_metrics["val_atop5"] is not None else np.nan,
        )
        append_history(
            self.history,
            "val_adba",
            validation_curve_metrics["val_adba"] if validation_curve_metrics["val_adba"] is not None else np.nan,
        )
        append_history(self.history, "val_occlusion_accuracy", val_occlusion_accuracy)
        append_history(self.history, "val_occlusion_blocked_f1", val_occlusion_blocked_f1)
        append_history(self.history, "val_position_rmse", val_position_rmse)
        append_history(self.history, "val_position_mae", val_position_mae)
        append_history(self.history, "val_multitask_loss", val_multitask_loss)
        append_history(self.history, "val_beam_top1", val_beam_top1)
        append_history(self.history, "val_beam_top3", val_beam_top3)
        append_history(self.history, "val_beam_top5", val_beam_top5)
        append_history(self.history, "val_beam_dba", val_beam_dba)
        append_history(self.history, "val_los_accuracy", val_los_accuracy)
        append_history(self.history, "val_los_f1", val_los_f1)
        append_history(self.history, "val_los_auc", val_los_auc)
        append_history(self.history, "val_link_mae", val_link_mae)
        append_history(self.history, "val_link_rmse", val_link_rmse)
        append_history(self.history, "val_link_r2", val_link_r2)
        append_history(self.history, "val_selection_multitask_loss", val_selection_multitask_loss)
        append_history(self.history, "val_jepa_loss", val_jepa_loss)
        append_history(self.history, "val_jepa_mask_target_ratio", val_jepa_mask_target_ratio)
        append_history(self.history, "val_jepa_mask_context_ratio", val_jepa_mask_context_ratio)
        append_history(self.history, "val_jepa_ema_decay", val_jepa_ema_decay)

        early_stopping_candidates = dict(validation_curve_metrics)
        if val_loss is not None:
            early_stopping_candidates["val_loss"] = float(val_loss)
        if val_acc is not None:
            early_stopping_candidates["val_acc"] = val_acc
        for key, value in {
            "val_occlusion_accuracy": val_occlusion_accuracy,
            "val_occlusion_blocked_f1": val_occlusion_blocked_f1,
            "val_position_rmse": val_position_rmse,
            "val_position_mae": val_position_mae,
            "val_multitask_loss": val_multitask_loss,
            "val_beam_top1": val_beam_top1,
            "val_beam_top3": val_beam_top3,
            "val_beam_top5": val_beam_top5,
            "val_beam_dba": val_beam_dba,
            "val_los_accuracy": val_los_accuracy,
            "val_los_f1": val_los_f1,
            "val_los_auc": val_los_auc,
            "val_link_mae": val_link_mae,
            "val_link_rmse": val_link_rmse,
            "val_link_r2": val_link_r2,
            "val_selection_multitask_loss": val_selection_multitask_loss,
            "val_jepa_loss": val_jepa_loss,
            "val_jepa_mask_target_ratio": val_jepa_mask_target_ratio,
            "val_jepa_mask_context_ratio": val_jepa_mask_context_ratio,
            "val_jepa_ema_decay": val_jepa_ema_decay,
        }.items():
            if value is not None:
                early_stopping_candidates[key] = value
        primary_metric_value = (
            early_stopping_metric_value(early_stopping_candidates, self.early_stopping_metric)
            if model_selection_enabled
            else None
        )
        self.history["val_primary_metric"].append(
            float(primary_metric_value) if primary_metric_value is not None else np.nan
        )
        epoch_log = {
            "epoch": epoch + 1,
            "total_epochs": total_epochs,
            "train_batches": self.batch_count,
            "objective": self.objective,
            "primary_loss": self.objective_metadata["primary_loss"],
            "primary_metric": self.early_stopping_metric,
            "primary_metric_mode": self.early_stopping_mode,
            "enabled_targets": self.objective_metadata["enabled_targets"],
            "enabled_heads": self.objective_metadata["enabled_heads"],
            "loss_weights": self.objective_metadata.get("loss_weights", {}),
            "train_loss": float(self.running["loss"]),
            "train_task_loss": float(self.running["task_loss"]),
            "train_objective_loss": float(self.running["task_loss"]),
            "train_beam_soft_loss": float(self.running["beam_soft_loss"]),
            "loss/beam_soft_target": float(self.running["beam_soft_loss"]),
            "train_unimodal_loss": float(self.running["unimodal_loss"]),
            "train_occlusion_loss": train_occlusion_loss,
            "train_position_loss": train_position_loss,
            "train_multitask_loss": train_multitask_loss,
            "train_los_loss": train_los_loss,
            "train_link_quality_loss": train_link_quality_loss,
            "train_selection_multitask_loss": train_selection_multitask_loss,
            "train_jepa_loss": train_jepa_loss,
            "loss/occlusion": train_occlusion_loss,
            "loss/position": train_position_loss,
            "loss/multitask_total": train_multitask_loss,
            "loss/los": train_los_loss,
            "loss/link_quality": train_link_quality_loss,
            "loss/selection_multitask_total": train_selection_multitask_loss,
            "loss/jepa": train_jepa_loss,
            "train_acc": float(self.running["acc"]),
            "train_loss_observation_count": _observation_count(self.metric_denominators.get("loss", 0.0)),
            "train_task_loss_observation_count": _observation_count(
                self.metric_denominators.get("task_loss", 0.0)
            ),
            "train_beam_soft_loss_observation_count": _observation_count(
                self.metric_denominators.get("beam_soft_loss", 0.0)
            ),
            "train_occlusion_loss_observation_count": _observation_count(
                self.metric_denominators.get("occlusion_loss", 0.0)
            ),
            "train_position_loss_observation_count": _observation_count(
                self.metric_denominators.get("position_loss", 0.0)
            ),
            "train_accuracy_observation_count": _observation_count(self.metric_denominators.get("acc", 0.0)),
            "val_loss": float(val_loss) if val_loss is not None else None,
            "val_acc": val_acc,
            "val_atop3": validation_curve_metrics["val_atop3"],
            "val_atop5": validation_curve_metrics["val_atop5"],
            "val_adba": validation_curve_metrics["val_adba"],
            "val_occlusion_accuracy": val_occlusion_accuracy,
            "val_occlusion_blocked_f1": val_occlusion_blocked_f1,
            "val_position_rmse": val_position_rmse,
            "val_position_mae": val_position_mae,
            "val_multitask_loss": val_multitask_loss,
            "val_beam_top1": val_beam_top1,
            "val_beam_top3": val_beam_top3,
            "val_beam_top5": val_beam_top5,
            "val_los_accuracy": val_los_accuracy,
            "val_los_f1": val_los_f1,
            "val_los_auc": val_los_auc,
            "val_link_mae": val_link_mae,
            "val_link_rmse": val_link_rmse,
            "val_link_r2": val_link_r2,
            "val_selection_multitask_loss": val_selection_multitask_loss,
            "val_jepa_loss": val_jepa_loss,
            "val_jepa_mask_target_ratio": val_jepa_mask_target_ratio,
            "val_jepa_mask_context_ratio": val_jepa_mask_context_ratio,
            "val_jepa_ema_decay": val_jepa_ema_decay,
            "val_primary_metric": float(primary_metric_value) if primary_metric_value is not None else None,
            "learning_rate": float(current_lr),
            "validation_metrics": deepcopy(val_metrics) if has_validation else None,
        }
        if epoch_subsampling:
            epoch_log.update(epoch_subsampling)
        if train_lidar_quality is not None:
            epoch_log["lidar_input_quality_train"] = train_lidar_quality.finalize(
                split=getattr(train_dataset, "split", "train"),
                preprocessing=lidar_preprocessing_metadata_from_dataset(train_dataset),
            )
        for group in optimizer_groups:
            group_name = group["name"]
            epoch_log[f"optimizer/lr/{group_name}"] = float(group["lr"])
            epoch_log[f"optimizer/params/{group_name}"] = float(group["param_count"])
        if health_metrics:
            epoch_log.update(health_metrics)
        if has_validation:
            epoch_log.update(validation_subset_epoch_scalars(val_metrics))
        epoch_log.update(self.epoch_diagnostics.mean())
        if extension_metrics:
            epoch_log.update(extension_metrics)
        epoch_log = prune_epoch_log_for_objective(epoch_log, self.history, self.objective)
        self.epoch_logs.append(epoch_log)
        return epoch_log, val_loss, val_acc, primary_metric_value


def mean_valid_slots(values, totals, horizons=None) -> float:
    values_arr = np.asarray(values, dtype=float)
    totals_arr = np.asarray(totals, dtype=float)
    length = min(values_arr.size, totals_arr.size)
    if length == 0:
        return 0.0

    values_arr = values_arr[:length]
    valid_slots = totals_arr[:length] > 0
    if horizons is not None:
        selected = np.zeros((length,), dtype=bool)
        for horizon in horizons:
            index = int(horizon) - 1
            if 0 <= index < length:
                selected[index] = True
        valid_slots &= selected
    if not np.any(valid_slots):
        return 0.0
    return float(np.mean(values_arr[valid_slots]))


def aggregate_validation_metrics(val_metrics: dict) -> dict[str, float]:
    aggregated = aggregate_topk_and_dba(val_metrics)
    return {
        "val_atop3": aggregated["top3"],
        "val_atop5": aggregated["top5"],
        "val_adba": aggregated["adba"],
        "val_beam_dba": float(val_metrics.get("val_beam_dba", 0.0) or 0.0),
    }


def _topk_values(topk: Any, k: int) -> list[float]:
    if isinstance(topk, dict):
        values = topk.get(str(k), topk.get(k, []))
    elif isinstance(topk, (list, tuple)):
        index = int(k) - 1
        values = topk[index] if 0 <= index < len(topk) else []
    else:
        values = []
    if isinstance(values, np.ndarray):
        return [float(value) for value in values.tolist()]
    if isinstance(values, (list, tuple)):
        return [float(value) for value in values]
    if values is None:
        return []
    return [float(values)]


def validation_subset_epoch_scalars(val_metrics: dict) -> dict[str, float]:
    subset_metrics = val_metrics.get("modality_subsets")
    if not isinstance(subset_metrics, dict):
        return {}
    scalars: dict[str, float] = {}
    parent_horizons = val_metrics.get("metric_horizons")
    parent_horizon_source = val_metrics.get("metric_horizon_source")
    for subset_name, metrics in subset_metrics.items():
        if not isinstance(metrics, dict):
            continue
        if parent_horizons is not None and "metric_horizons" not in metrics:
            metrics = dict(metrics)
            metrics["metric_horizons"] = parent_horizons
            if parent_horizon_source is not None and "metric_horizon_source" not in metrics:
                metrics["metric_horizon_source"] = parent_horizon_source
        prefix = f"val/subset/{subset_name}"
        total = metrics.get("total", [])
        horizons = metric_horizons_from_metrics(metrics, num_pred=len(total))
        scalars[f"{prefix}/loss"] = float(metrics.get("loss", 0.0))
        topk = metrics.get("topk", {})
        aggregated = aggregate_topk_and_dba(metrics)
        scalars[f"{prefix}/top1"] = aggregated["top1"]
        scalars[f"{prefix}/atop3"] = aggregated["top3"]
        scalars[f"{prefix}/atop5"] = aggregated["top5"]
        scalars[f"{prefix}/adba"] = aggregated["adba"]
        scalars[f"{prefix}/top3"] = selected_horizon_mean(topk.get("3", []), total, horizons=horizons)
        scalars[f"{prefix}/top5"] = selected_horizon_mean(topk.get("5", []), total, horizons=horizons)
    return scalars


def first_valid_slot(values, totals) -> float:
    values_arr = np.asarray(values, dtype=float)
    totals_arr = np.asarray(totals, dtype=float)
    length = min(values_arr.size, totals_arr.size)
    if length == 0:
        return 0.0
    for idx in range(length):
        if totals_arr[idx] > 0:
            return float(values_arr[idx])
    return 0.0


def training_outputs_payload(
    history: dict[str, list],
    objective_metadata: dict,
    early_stopping_metric: str,
    early_stopping_mode: str,
) -> dict[str, np.ndarray]:
    payload = {key: history_array(key, value) for key, value in history.items()}
    payload["objective"] = np.asarray(objective_metadata["name"])
    payload["primary_loss"] = np.asarray(objective_metadata["primary_loss"])
    payload["primary_metric"] = np.asarray(early_stopping_metric)
    payload["primary_metric_mode"] = np.asarray(early_stopping_mode)
    payload["enabled_targets"] = np.asarray(objective_metadata["enabled_targets"], dtype=object)
    payload["enabled_heads"] = np.asarray(objective_metadata["enabled_heads"], dtype=object)
    if objective_metadata["name"] == "gps_conditioned_jepa":
        weight_names = ("jepa",)
    elif objective_metadata["name"] == "selection_multitask":
        weight_names = ("beam_selection", "los", "link_quality")
    else:
        weight_names = ("beam", "occlusion", "position")
    weights = objective_metadata.get("loss_weights", {})
    payload["loss_weight_names"] = np.asarray(weight_names, dtype=object)
    payload["loss_weights"] = np.asarray([float(weights.get(name, np.nan)) for name in weight_names], dtype=float)
    return payload


OPTIONAL_HISTORY_KEYS = objective_optional_history_fields()


def prune_epoch_log_for_objective(epoch_log: dict[str, Any], history: dict[str, list], objective: str) -> dict[str, Any]:
    if objective not in {
        "current_beam_selection",
        "current_los_classification",
        "current_link_quality",
        "selection_multitask",
        "gps_conditioned_jepa",
    }:
        return epoch_log
    allowed_history = set(history)
    pruned = dict(epoch_log)
    metric_keys = {
        "val_acc",
        "val_atop3",
        "val_atop5",
        "val_adba",
        "val_occlusion_accuracy",
        "val_occlusion_blocked_f1",
        "val_position_rmse",
        "val_position_mae",
        "val_multitask_loss",
        "val_beam_top1",
        "val_beam_top3",
        "val_beam_top5",
        "val_beam_dba",
        "val_los_accuracy",
        "val_los_f1",
        "val_los_auc",
        "val_link_mae",
        "val_link_rmse",
        "val_link_r2",
        "val_selection_multitask_loss",
        "val_jepa_loss",
        "val_jepa_mask_target_ratio",
        "val_jepa_mask_context_ratio",
        "val_jepa_ema_decay",
    }
    loss_keys = {
        "train_occlusion_loss": "train_occlusion_loss",
        "train_position_loss": "train_position_loss",
        "train_multitask_loss": "train_multitask_loss",
        "train_los_loss": "train_los_loss",
        "train_link_quality_loss": "train_link_quality_loss",
        "train_selection_multitask_loss": "train_selection_multitask_loss",
        "loss/occlusion": "train_occlusion_loss",
        "loss/position": "train_position_loss",
        "loss/multitask_total": "train_multitask_loss",
        "loss/los": "train_los_loss",
        "loss/link_quality": "train_link_quality_loss",
        "loss/selection_multitask_total": "train_selection_multitask_loss",
        "loss/beam_soft_target": "train_beam_soft_loss",
        "loss/jepa": "train_jepa_loss",
    }
    for key in metric_keys:
        if key not in allowed_history and key in pruned:
            pruned.pop(key, None)
    for key, history_key in loss_keys.items():
        if history_key not in allowed_history:
            pruned.pop(key, None)
    return pruned


def append_history(history: dict[str, list], key: str, value) -> None:
    if key in history:
        history[key].append(value)


def _loss_item(values: dict[str, Any], key: str) -> float:
    value = values.get(key)
    if value is None:
        return 0.0
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _observation_count(value: float) -> int | float:
    rounded = round(float(value))
    return int(rounded) if abs(float(value) - rounded) < 1e-9 else float(value)


def history_array(key: str, values: list) -> np.ndarray:
    if key not in OPTIONAL_HISTORY_KEYS:
        return np.asarray(values)
    return np.asarray([np.nan if finite_float_or_none(value) is None else float(value) for value in values], dtype=float)


def checkpoint_task_metrics(epoch_log: dict) -> dict[str, float]:
    keys = (
        "val_acc",
        "val_adba",
        "val_loss",
        "val_occlusion_accuracy",
        "val_occlusion_blocked_f1",
        "val_position_rmse",
        "val_position_mae",
        "val_multitask_loss",
        "val_beam_top1",
        "val_beam_top3",
        "val_beam_top5",
        "val_beam_dba",
        "val_los_accuracy",
        "val_los_f1",
        "val_los_auc",
        "val_link_mae",
        "val_link_rmse",
        "val_link_r2",
        "val_selection_multitask_loss",
        "val_jepa_loss",
        "val_jepa_mask_target_ratio",
        "val_jepa_mask_context_ratio",
        "val_jepa_ema_decay",
    )
    return {
        key: float(epoch_log[key])
        for key in keys
        if key in epoch_log and isinstance(epoch_log[key], (int, float))
    }
