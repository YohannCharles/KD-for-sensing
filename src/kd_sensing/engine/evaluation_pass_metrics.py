from typing import Any

import numpy as np
import torch

from kd_sensing.engine.data_factory_scalers import shared_dataset_attribute
from kd_sensing.engine.objectives.metadata import objective_available_metrics
from kd_sensing.evaluation.horizon_selection import (
    horizon_indices,
    metric_horizon_source_from_config,
    metric_horizons_from_config,
)
from kd_sensing.evaluation.metrics import (
    calculate_current_beam_dba,
    calculate_dba_score,
    calculate_link_metrics,
    calculate_los_metrics,
    calculate_occlusion_metrics,
    calculate_position_rmse,
    calculate_topk_accuracy,
)


def metrics_from_outputs(
    loss: float,
    outputs: torch.Tensor,
    labels: torch.Tensor,
    cfg: dict[str, Any],
    *,
    objective: str,
) -> dict[str, Any]:
    num_label_horizons = int(labels.shape[1]) if labels.ndim > 1 else 1
    if objective in {"current_beam_selection", "selection_multitask"}:
        metric_horizons = (1,)
    else:
        metric_horizons = metric_horizons_from_config(cfg, num_pred=num_label_horizons)
    topk_acc, total = calculate_topk_accuracy(
        outputs,
        labels,
        cfg.get("evaluation", {}).get("k_values", [1, 2, 3, 5, 10]),
    )
    metrics = {
        "loss": float(loss),
        "topk": {str(k): v.tolist() for k, v in topk_acc.items()},
        "total": total.tolist(),
        "metric_horizons": list(metric_horizons),
        "metric_horizon_indices": list(horizon_indices(metric_horizons)),
        "metric_horizon_source": metric_horizon_source_from_config(cfg),
        "label_space": str(cfg.get("evaluation", {}).get("label_space", "64_beam")),
        "beam_shift": int(cfg.get("evaluation", {}).get("beam_shift", 0)),
        "metric_profile": str(cfg.get("evaluation", {}).get("metric_profile", "64_beam_circular_topk")),
        "dba_distance_mode": str(cfg.get("evaluation", {}).get("dba_distance_mode", "circular")),
        "circular_beam_distance": bool(
            cfg.get("evaluation", {}).get(
                "circular_beam_distance",
                cfg.get("evaluation", {}).get("dba_distance_mode", "circular") == "circular",
            )
        ),
    }
    if objective in {"current_beam_selection", "selection_multitask"}:
        metrics.update(_flat_current_beam_metrics(topk_acc, total))
        beam_dba = calculate_current_beam_dba(
            outputs,
            labels,
            cfg.get("evaluation", {}).get("dba_delta", 5),
            distance_mode=cfg.get("evaluation", {}).get("dba_distance_mode", "circular"),
        )
        metrics["beam_dba_current"] = beam_dba
        metrics["val_beam_dba"] = beam_dba
    elif objective in {"current_los_classification", "current_link_quality"}:
        pass
    else:
        metrics.update(_flat_future_topk_metrics(topk_acc, total, metric_horizons=metric_horizons))
        dba_score = calculate_dba_score(
            outputs,
            labels,
            cfg.get("evaluation", {}).get("dba_delta", 5),
            distance_mode=cfg.get("evaluation", {}).get("dba_distance_mode", "circular"),
        )
        metrics["dba"] = dba_score.tolist()
    return metrics


def beam_metrics_by_los_bucket(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    los_labels: torch.Tensor,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    if outputs.ndim == 2:
        outputs = outputs.unsqueeze(1)
    if los_labels.ndim > 1:
        los_labels = los_labels[:, 0]
    los_labels = los_labels.detach().cpu().to(torch.float32).reshape(-1)
    if outputs.shape[0] != labels.shape[0] or labels.shape[0] != los_labels.shape[0]:
        return {}

    buckets: dict[str, Any] = {}
    for label_value, label_name in ((0, "NLOS"), (1, "LOS")):
        mask = los_labels >= 0.5 if label_value == 1 else los_labels < 0.5
        sample_count = int(mask.sum().item())
        if sample_count == 0:
            continue
        bucket_metrics = metrics_from_outputs(
            0.0,
            outputs[mask],
            labels[mask],
            cfg,
            objective="current_beam_selection",
        )
        bucket_metrics["sample_count"] = sample_count
        bucket_metrics["los_label"] = label_value
        bucket_metrics["los_bucket"] = label_name
        buckets[f"LOS={label_value}"] = bucket_metrics
    return buckets


def auxiliary_metrics_from_outputs(
    dataloader,
    *,
    occlusion_logits: torch.Tensor | None,
    occlusion_labels: torch.Tensor | None,
    occlusion_valid: torch.Tensor | None,
    position_outputs: torch.Tensor | None,
    position_targets: torch.Tensor | None,
    position_valid: torch.Tensor | None,
    los_logits: torch.Tensor | None,
    los_labels: torch.Tensor | None,
    link_outputs: torch.Tensor | None,
    link_targets: torch.Tensor | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if occlusion_logits is not None and occlusion_labels is not None:
        metrics.update(calculate_occlusion_metrics(occlusion_logits, occlusion_labels, occlusion_valid))
    if position_outputs is not None and position_targets is not None:
        dataset = getattr(dataloader, "dataset", None)
        scaler = (
            shared_dataset_attribute(
                dataset,
                "position_target_scaler",
                enabled=lambda leaf: bool(
                    getattr(leaf, "position_target_scaler", None) is not None
                    or (
                        getattr(leaf, "position_target_enabled", False)
                        and getattr(leaf, "position_target_normalize", False)
                    )
                ),
            )
            if dataset is not None
            else None
        )
        mean = getattr(scaler, "mean_", None)
        scale = getattr(scaler, "scale_", None)
        metrics.update(
            calculate_position_rmse(
                position_outputs,
                position_targets,
                position_valid,
                mean=mean,
                scale=scale,
            )
        )
    if los_logits is not None and los_labels is not None:
        metrics.update(calculate_los_metrics(los_logits, los_labels))
    if link_outputs is not None and link_targets is not None:
        metrics.update(calculate_link_metrics(link_outputs, link_targets))
    return metrics


def attach_objective_metrics(
    metrics: dict[str, Any],
    auxiliary_metrics: dict[str, float],
    *,
    objective: str,
    val_occlusion_loss: float,
    val_position_loss: float,
    val_multitask_loss: float,
    val_los_loss: float,
    val_link_quality_loss: float,
    val_selection_multitask_loss: float,
) -> None:
    auxiliary: dict[str, float] = dict(auxiliary_metrics)
    has_occlusion = int(auxiliary_metrics.get("occlusion_total", 0)) > 0
    has_position = int(auxiliary_metrics.get("position_total", 0)) > 0

    if has_occlusion:
        auxiliary["loss_occlusion"] = float(val_occlusion_loss)
        metrics["loss/occlusion"] = auxiliary["loss_occlusion"]
        if "occlusion_accuracy" in auxiliary_metrics:
            metrics["val_occlusion_accuracy"] = float(auxiliary_metrics["occlusion_accuracy"])
        if "occlusion_blocked_f1" in auxiliary_metrics:
            metrics["val_occlusion_blocked_f1"] = float(auxiliary_metrics["occlusion_blocked_f1"])

    if has_position:
        auxiliary["loss_position"] = float(val_position_loss)
        metrics["loss/position"] = auxiliary["loss_position"]
        if "position_rmse" in auxiliary_metrics:
            metrics["val_position_rmse"] = float(auxiliary_metrics["position_rmse"])
        if "position_mae" in auxiliary_metrics:
            metrics["val_position_mae"] = float(auxiliary_metrics["position_mae"])

    if objective == "multitask":
        auxiliary["loss_multitask_total"] = float(val_multitask_loss)
        metrics["loss/multitask_total"] = auxiliary["loss_multitask_total"]
        metrics["val_multitask_loss"] = auxiliary["loss_multitask_total"]

    has_los = int(auxiliary_metrics.get("los_total", 0)) > 0
    has_link = int(auxiliary_metrics.get("link_total", 0)) > 0
    if has_los:
        if objective in {"current_los_classification", "selection_multitask"}:
            auxiliary["loss_los"] = float(val_los_loss)
            metrics["loss/los"] = auxiliary["loss_los"]
            for key in ("los_accuracy", "los_f1", "los_auc"):
                metrics[key] = auxiliary_metrics.get(key)
                metrics[f"val_{key}"] = auxiliary_metrics.get(key)
            metrics["los_auc_available"] = bool(auxiliary_metrics.get("los_auc_available", False))
            if auxiliary_metrics.get("los_auc_unavailable_reason"):
                metrics["los_auc_unavailable_reason"] = auxiliary_metrics["los_auc_unavailable_reason"]
    if has_link:
        if objective in {"current_link_quality", "selection_multitask"}:
            auxiliary["loss_link_quality"] = float(val_link_quality_loss)
            metrics["loss/link_quality"] = auxiliary["loss_link_quality"]
            for key in ("link_mae", "link_rmse", "link_r2"):
                metrics[key] = float(auxiliary_metrics[key])
                metrics[f"val_{key}"] = float(auxiliary_metrics[key])
    if objective == "selection_multitask":
        auxiliary["loss_selection_multitask_total"] = float(val_selection_multitask_loss)
        metrics["loss/selection_multitask_total"] = auxiliary["loss_selection_multitask_total"]
        metrics["selection_multitask_loss"] = auxiliary["loss_selection_multitask_total"]
        metrics["val_selection_multitask_loss"] = auxiliary["loss_selection_multitask_total"]

    if auxiliary:
        metrics["auxiliary"] = auxiliary


def available_metrics(objective: str, metrics: dict[str, Any]) -> list[str]:
    return objective_available_metrics(objective, metrics)


def _flat_future_topk_metrics(topk_acc: dict[int, object], total, *, metric_horizons: tuple[int, ...]) -> dict[str, float]:
    scalars: dict[str, float] = {}
    total_arr = torch.as_tensor(total, dtype=torch.float32).cpu().numpy()
    horizon_names = [f"t{idx + 1}" for idx in range(len(total_arr))]
    selected = np.zeros((len(total_arr),), dtype=bool)
    for index in horizon_indices(metric_horizons):
        if 0 <= index < len(selected):
            selected[index] = True
    for k in (1, 3, 5):
        if k not in topk_acc:
            continue
        values = torch.as_tensor(topk_acc[k], dtype=torch.float32).cpu().numpy()
        length = min(len(values), len(total_arr))
        valid = (total_arr[:length] > 0) & selected[:length]
        for idx in range(length):
            scalars[f"val_top{k}_{horizon_names[idx]}"] = float(values[idx])
        scalars[f"val_top{k}_avg"] = float(values[:length][valid].mean()) if valid.any() else 0.0
    return scalars


def _flat_current_beam_metrics(topk_acc: dict[int, object], total) -> dict[str, float]:
    scalars: dict[str, float] = {}
    total_arr = torch.as_tensor(total, dtype=torch.float32).cpu().numpy()
    valid = total_arr > 0
    for k, name in ((1, "beam_top1"), (3, "beam_top3"), (5, "beam_top5")):
        if k not in topk_acc:
            continue
        values = torch.as_tensor(topk_acc[k], dtype=torch.float32).cpu().numpy()
        length = min(len(values), len(total_arr))
        if length == 0:
            value = 0.0
        else:
            value = float(values[:length][valid[:length]].mean()) if valid[:length].any() else 0.0
        scalars[name] = value
        scalars[f"val_{name}"] = value
    return scalars


__all__ = [
    "attach_objective_metrics",
    "auxiliary_metrics_from_outputs",
    "available_metrics",
    "beam_metrics_by_los_bucket",
    "metrics_from_outputs",
]
