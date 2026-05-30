from __future__ import annotations

from pathlib import Path
from typing import Any

from kd_sensing.modalities import MODALITY_ORDER


def teacher_metrics_from_training(
    cfg: dict[str, Any],
    history: dict[str, Any],
    epoch_logs: list[dict[str, Any]],
    *,
    best_selected_epoch: int | None = None,
    selection_metric: str = "early_stopping",
    selection_mode: str = "early_stopping",
    checkpoint: str = "checkpoints/best.pth",
    best_top1_epoch: int | None = None,
) -> dict[str, Any] | None:
    task = str(cfg.get("experiment", {}).get("task", "image"))
    if task == "fusion":
        return None
    if task not in MODALITY_ORDER:
        return None
    if epoch_logs:
        if best_selected_epoch and 1 <= best_selected_epoch <= len(epoch_logs):
            best_idx = int(best_selected_epoch) - 1
        else:
            best_idx = max(
                range(len(epoch_logs)),
                key=lambda idx: float(
                    epoch_logs[idx].get("val_primary_metric", epoch_logs[idx].get("val_acc", 0.0))
                ),
            )
        best = epoch_logs[best_idx]
        selected_epoch = int(best.get("epoch", best_idx + 1))
        metrics = {
            "modality": task,
            "best_epoch": selected_epoch,
            "selected_epoch": selected_epoch,
            "selection_metric": str(selection_metric),
            "selection_mode": str(selection_mode),
            "checkpoint": checkpoint,
            "checkpoint_path": checkpoint,
            "checkpoint_source": "objective-checkpoint"
            if Path(str(checkpoint)).name != "best_top1.pth"
            else "top1-checkpoint",
            "val_acc_top1": float(best.get("val_acc", 0.0)),
            "val_acc_top3": float(best.get("val_atop3", 0.0)),
            "val_acc_top5": float(best.get("val_atop5", 0.0)),
            "val_adba": float(best.get("val_adba", 0.0)),
            "train_acc_top1": float(best.get("train_acc", 0.0)),
        }
        if best_top1_epoch and 1 <= best_top1_epoch <= len(epoch_logs):
            top1 = epoch_logs[int(best_top1_epoch) - 1]
            metrics["top1_epoch"] = int(top1.get("epoch", int(best_top1_epoch)))
            metrics["top1_checkpoint"] = "checkpoints/best_top1.pth"
            metrics["top1_val_acc"] = float(top1.get("val_acc", 0.0))
        metrics.update(_extended_validation_metrics(best))
        return metrics
    val_acc = history.get("val_acc") or []
    if not val_acc:
        return None
    primary = history.get("val_primary_metric") or []
    values = primary if primary else val_acc
    best_idx = max(range(len(values)), key=lambda idx: float(values[idx]))
    return {
        "modality": task,
        "best_epoch": best_idx + 1,
        "selected_epoch": best_idx + 1,
        "selection_metric": str(selection_metric if primary else "val_acc_top1"),
        "selection_mode": str(selection_mode if primary else "legacy_top1"),
        "checkpoint": checkpoint if primary else "checkpoints/best_top1.pth",
        "checkpoint_path": checkpoint if primary else "checkpoints/best_top1.pth",
        "checkpoint_source": "objective-checkpoint" if primary else "top1-checkpoint",
        "val_acc_top1": float(val_acc[best_idx]),
        "val_acc_top3": _history_value(history, "val_atop3", best_idx),
        "val_acc_top5": _history_value(history, "val_atop5", best_idx),
        "val_adba": _history_value(history, "val_adba", best_idx),
        "train_acc_top1": _history_value(history, "train_acc", best_idx),
    }


def _extended_validation_metrics(epoch_log: dict[str, Any]) -> dict[str, Any]:
    validation = epoch_log.get("validation_metrics")
    if not isinstance(validation, dict):
        return {}
    topk = validation.get("topk") if isinstance(validation.get("topk"), dict) else {}
    per_horizon = {
        "top1": _list_of_floats(topk.get("1", [])),
        "top3": _list_of_floats(topk.get("3", [])),
        "top5": _list_of_floats(topk.get("5", [])),
        "dba": _list_of_floats(validation.get("dba", [])),
        "total": [int(value) for value in validation.get("total", [])],
    }
    averages = {
        "top1": float(validation.get("val_top1_avg", 0.0)),
        "top3": float(validation.get("val_top3_avg", 0.0)),
        "top5": float(validation.get("val_top5_avg", 0.0)),
        "adba": float(epoch_log.get("val_adba", 0.0)),
    }
    extended: dict[str, Any] = {
        "per_horizon": per_horizon,
        "averages": averages,
    }
    for key in ("degradation_baselines", "degradation_risk", "lidar_input_quality"):
        if key in validation:
            extended[key] = validation[key]
    if "lidar_input_quality_train" in epoch_log:
        extended["lidar_input_quality_train"] = epoch_log["lidar_input_quality_train"]
    return extended


def _list_of_floats(values: Any) -> list[float]:
    return [float(value) for value in (values or [])]


def _history_value(history: dict[str, Any], key: str, index: int) -> float:
    values = history.get(key) or []
    if index < len(values):
        return float(values[index])
    return 0.0
