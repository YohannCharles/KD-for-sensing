from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from kd_sensing.engine.debug_diagnostics import set_csi_debug_batch_source
from kd_sensing.engine.modality_resolution import config_uses_lidar, resolve_enabled_modalities
from kd_sensing.engine.prediction_objectives import (
    compute_prediction_loss,
    objective_available_metrics,
    objective_runtime_metadata,
    prepare_prediction_targets,
    resolve_prediction_objective,
)
from kd_sensing.engine.runtime import (
    autocast_context,
    prepare_task_auxiliary_targets,
    prepare_task_batch,
    prepare_task_labels,
    resolve_amp_settings,
    run_model_step,
    transfer_non_blocking,
)
from kd_sensing.evaluation.lidar_diagnostics import (
    LidarQualityAccumulator,
    degradation_baselines_from_labels,
    lidar_degradation_report,
    lidar_preprocessing_metadata_from_dataset,
)
from kd_sensing.evaluation.metrics import (
    calculate_dba_score,
    calculate_link_metrics,
    calculate_los_metrics,
    calculate_occlusion_metrics,
    calculate_position_rmse,
    calculate_topk_accuracy,
)


@dataclass(frozen=True)
class EvaluationPassResult:
    metrics: dict[str, Any]
    outputs: torch.Tensor
    labels: torch.Tensor
    input_beams: torch.Tensor | None
    objective_metadata: dict[str, Any]
    enabled_modalities: tuple[str, ...]
    saw_lidar: bool


def run_evaluation_pass(
    model,
    dataloader,
    cfg: dict[str, Any],
    criterion,
    device: torch.device,
    *,
    force_modality_mask: torch.Tensor | None = None,
) -> EvaluationPassResult:
    model.eval()
    objective = resolve_prediction_objective(cfg)
    objective_metadata = objective_runtime_metadata(cfg)
    enabled_modalities = resolve_enabled_modalities(cfg)
    task = cfg["experiment"].get("task", "image")
    model_cfg = cfg["model"]
    num_pred = model_cfg.get("num_pred", 3)
    downsample_ratio = model_cfg.get("downsample_ratio", 1)
    seq_length = model_cfg.get("seq_length_student", 8)
    num_classes = model_cfg.get("num_classes", 64)
    non_blocking = transfer_non_blocking(cfg)
    amp_enabled, amp_dtype = resolve_amp_settings(cfg, device)
    val_loss = 0.0
    val_occlusion_loss = 0.0
    val_position_loss = 0.0
    val_multitask_loss = 0.0
    val_los_loss = 0.0
    val_link_quality_loss = 0.0
    val_selection_multitask_loss = 0.0
    all_outputs = []
    all_labels = []
    all_input_beams = []
    all_occlusion_logits = []
    all_occlusion_labels = []
    all_occlusion_valid = []
    all_position_outputs = []
    all_position_targets = []
    all_position_valid = []
    all_los_logits = []
    all_los_labels = []
    all_link_outputs = []
    all_link_targets = []
    lidar_quality = LidarQualityAccumulator()
    saw_lidar = False

    with torch.no_grad():
        for batch in dataloader:
            batch = prepare_task_batch(batch)
            if "input_beam" in batch:
                all_input_beams.append(batch["input_beam"].detach().cpu())
            if "lidar" in batch:
                saw_lidar = True
                lidar_quality.update(batch["lidar"], raw_lidar=batch.get("lidar_raw"))
            labels = prepare_task_labels(
                batch,
                num_pred=num_pred,
                downsample_ratio=downsample_ratio,
                device=device,
                non_blocking=non_blocking,
            )
            auxiliary_targets = prepare_task_auxiliary_targets(
                batch,
                num_pred=num_pred,
                device=device,
                non_blocking=non_blocking,
            )
            prediction_targets = prepare_prediction_targets(
                labels=labels,
                auxiliary_targets=auxiliary_targets,
                cfg=cfg,
            )
            with autocast_context(amp_enabled, device, amp_dtype):
                set_csi_debug_batch_source(model, "val")
                step = run_model_step(
                    model,
                    task,
                    batch,
                    model_cfg=cfg["model"]["student"],
                    seq_length=seq_length,
                    num_pred=num_pred,
                    device=device,
                    non_blocking=non_blocking,
                    force_modality_mask=force_modality_mask,
                )
                outputs = step.logits
                beam_loss = criterion(outputs.reshape(-1, num_classes), labels.flatten())
                prediction_loss = compute_prediction_loss(
                    step.model_output,
                    prediction_targets,
                    cfg,
                    reference=outputs,
                    beam_total_loss=beam_loss,
                    beam_task_loss=beam_loss,
                )
                loss = prediction_loss.total
            val_loss += loss.item()
            val_occlusion_loss += prediction_loss.occlusion.item()
            val_position_loss += prediction_loss.position.item()
            val_multitask_loss += prediction_loss.multitask_total.item()
            if prediction_loss.los is not None:
                val_los_loss += prediction_loss.los.item()
            if prediction_loss.link_quality is not None:
                val_link_quality_loss += prediction_loss.link_quality.item()
            if prediction_loss.selection_multitask_total is not None:
                val_selection_multitask_loss += prediction_loss.selection_multitask_total.item()
            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            if "occlusion_logits" in step.model_output.diagnostics and "occlusion_label" in auxiliary_targets:
                all_occlusion_logits.append(step.model_output.diagnostics["occlusion_logits"].detach().cpu())
                all_occlusion_labels.append(auxiliary_targets["occlusion_label"].detach().cpu())
                all_occlusion_valid.append(auxiliary_targets["occlusion_valid"].detach().cpu())
            if "position" in step.model_output.diagnostics and "position_target" in auxiliary_targets:
                all_position_outputs.append(step.model_output.diagnostics["position"].detach().cpu())
                all_position_targets.append(auxiliary_targets["position_target"].detach().cpu())
                all_position_valid.append(auxiliary_targets["position_valid"].detach().cpu())
            if "los_logits" in step.model_output.diagnostics and "los_label" in auxiliary_targets:
                all_los_logits.append(step.model_output.diagnostics["los_logits"].detach().cpu())
                all_los_labels.append(auxiliary_targets["los_label"].detach().cpu())
            if "link_quality" in step.model_output.diagnostics and "link_quality" in auxiliary_targets:
                all_link_outputs.append(step.model_output.diagnostics["link_quality"].detach().cpu())
                all_link_targets.append(auxiliary_targets["link_quality"].detach().cpu())

    outputs_t = torch.cat(all_outputs, dim=0)
    labels_t = torch.cat(all_labels, dim=0)
    auxiliary_metrics = _auxiliary_metrics_from_outputs(
        dataloader,
        occlusion_logits=torch.cat(all_occlusion_logits, dim=0) if all_occlusion_logits else None,
        occlusion_labels=torch.cat(all_occlusion_labels, dim=0) if all_occlusion_labels else None,
        occlusion_valid=torch.cat(all_occlusion_valid, dim=0) if all_occlusion_valid else None,
        position_outputs=torch.cat(all_position_outputs, dim=0) if all_position_outputs else None,
        position_targets=torch.cat(all_position_targets, dim=0) if all_position_targets else None,
        position_valid=torch.cat(all_position_valid, dim=0) if all_position_valid else None,
        los_logits=torch.cat(all_los_logits, dim=0) if all_los_logits else None,
        los_labels=torch.cat(all_los_labels, dim=0) if all_los_labels else None,
        link_outputs=torch.cat(all_link_outputs, dim=0) if all_link_outputs else None,
        link_targets=torch.cat(all_link_targets, dim=0) if all_link_targets else None,
    )
    metrics = _metrics_from_outputs(val_loss / max(len(dataloader), 1), outputs_t, labels_t, cfg, objective=objective)
    _attach_objective_metrics(
        metrics,
        auxiliary_metrics,
        objective=objective,
        dataloader_len=len(dataloader),
        val_occlusion_loss=val_occlusion_loss,
        val_position_loss=val_position_loss,
        val_multitask_loss=val_multitask_loss,
        val_los_loss=val_los_loss,
        val_link_quality_loss=val_link_quality_loss,
        val_selection_multitask_loss=val_selection_multitask_loss,
    )
    metrics["objective"] = objective_metadata
    metrics["available_metrics"] = objective_available_metrics(objective, metrics)
    metrics["enabled_modalities"] = list(enabled_modalities)
    dataset = getattr(dataloader, "dataset", None)
    if dataset is not None and hasattr(dataset, "raymobtime_metadata"):
        raymobtime_metadata = dataset.raymobtime_metadata()
        metrics["task_semantics"] = raymobtime_metadata.get("task_semantics")
        metrics["link_target"] = {
            "name": raymobtime_metadata.get("link_target_name"),
            "unit": raymobtime_metadata.get("link_target_unit"),
            "aggregation": getattr(dataset, "cache_metadata", {}).get("link_target_aggregation"),
        }

    input_beams_t = torch.cat(all_input_beams, dim=0) if all_input_beams else None
    baselines = degradation_baselines_from_labels(
        labels_t,
        input_beams=input_beams_t,
        num_classes=num_classes,
        downsample_ratio=downsample_ratio,
    )
    metrics["degradation_baselines"] = baselines
    if saw_lidar:
        quality_summary = lidar_quality.finalize(
            split=getattr(dataset, "split", None),
            preprocessing=lidar_preprocessing_metadata_from_dataset(dataset),
        )
        metrics["lidar_input_quality"] = quality_summary
        metrics["degradation_risk"] = lidar_degradation_report(metrics, baselines, quality_summary)
    elif config_uses_lidar(cfg):
        metrics["degradation_risk"] = lidar_degradation_report(metrics, baselines, None)

    return EvaluationPassResult(
        metrics=metrics,
        outputs=outputs_t,
        labels=labels_t,
        input_beams=input_beams_t,
        objective_metadata=objective_metadata,
        enabled_modalities=enabled_modalities,
        saw_lidar=saw_lidar,
    )


def _metrics_from_outputs(
    loss: float,
    outputs: torch.Tensor,
    labels: torch.Tensor,
    cfg: dict[str, Any],
    *,
    objective: str,
) -> dict[str, Any]:
    topk_acc, total = calculate_topk_accuracy(
        outputs,
        labels,
        cfg.get("evaluation", {}).get("k_values", [1, 2, 3, 5, 10]),
    )
    metrics = {
        "loss": float(loss),
        "topk": {str(k): v.tolist() for k, v in topk_acc.items()},
        "total": total.tolist(),
    }
    metrics.update(_flat_future_topk_metrics(topk_acc, total))
    metrics.update(_flat_current_beam_metrics(topk_acc, total))
    if objective not in {
        "current_beam_selection",
        "current_los_classification",
        "current_link_quality",
        "selection_multitask",
    }:
        dba_score = calculate_dba_score(
            outputs,
            labels,
            cfg.get("evaluation", {}).get("dba_delta", 5),
        )
        metrics["dba"] = dba_score.tolist()
    return metrics


def _auxiliary_metrics_from_outputs(
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
        scaler = getattr(getattr(dataloader, "dataset", None), "position_target_scaler", None)
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


def _flat_future_topk_metrics(topk_acc: dict[int, object], total) -> dict[str, float]:
    scalars: dict[str, float] = {}
    total_arr = torch.as_tensor(total, dtype=torch.float32).cpu().numpy()
    horizon_names = [f"t{idx + 1}" for idx in range(len(total_arr))]
    for k in (1, 3, 5):
        if k not in topk_acc:
            continue
        values = torch.as_tensor(topk_acc[k], dtype=torch.float32).cpu().numpy()
        length = min(len(values), len(total_arr))
        valid = total_arr[:length] > 0
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


def _attach_objective_metrics(
    metrics: dict[str, Any],
    auxiliary_metrics: dict[str, float],
    *,
    objective: str,
    dataloader_len: int,
    val_occlusion_loss: float,
    val_position_loss: float,
    val_multitask_loss: float,
    val_los_loss: float,
    val_link_quality_loss: float,
    val_selection_multitask_loss: float,
) -> None:
    auxiliary: dict[str, float] = dict(auxiliary_metrics)
    batches = max(dataloader_len, 1)
    has_occlusion = int(auxiliary_metrics.get("occlusion_total", 0)) > 0
    has_position = int(auxiliary_metrics.get("position_total", 0)) > 0

    if has_occlusion:
        auxiliary["loss_occlusion"] = float(val_occlusion_loss / batches)
        metrics["loss/occlusion"] = auxiliary["loss_occlusion"]
        if "occlusion_accuracy" in auxiliary_metrics:
            metrics["val_occlusion_accuracy"] = float(auxiliary_metrics["occlusion_accuracy"])
        if "occlusion_blocked_f1" in auxiliary_metrics:
            metrics["val_occlusion_blocked_f1"] = float(auxiliary_metrics["occlusion_blocked_f1"])

    if has_position:
        auxiliary["loss_position"] = float(val_position_loss / batches)
        metrics["loss/position"] = auxiliary["loss_position"]
        if "position_rmse" in auxiliary_metrics:
            metrics["val_position_rmse"] = float(auxiliary_metrics["position_rmse"])
        if "position_mae" in auxiliary_metrics:
            metrics["val_position_mae"] = float(auxiliary_metrics["position_mae"])

    if objective == "multitask":
        auxiliary["loss_multitask_total"] = float(val_multitask_loss / batches)
        metrics["loss/multitask_total"] = auxiliary["loss_multitask_total"]
        metrics["val_multitask_loss"] = auxiliary["loss_multitask_total"]

    has_los = int(auxiliary_metrics.get("los_total", 0)) > 0
    has_link = int(auxiliary_metrics.get("link_total", 0)) > 0
    if has_los:
        if objective in {"current_los_classification", "selection_multitask"}:
            auxiliary["loss_los"] = float(val_los_loss / batches)
            metrics["loss/los"] = auxiliary["loss_los"]
        for key in ("los_accuracy", "los_f1", "los_auc"):
            metrics[key] = auxiliary_metrics.get(key)
            metrics[f"val_{key}"] = auxiliary_metrics.get(key)
        metrics["los_auc_available"] = bool(auxiliary_metrics.get("los_auc_available", False))
        if auxiliary_metrics.get("los_auc_unavailable_reason"):
            metrics["los_auc_unavailable_reason"] = auxiliary_metrics["los_auc_unavailable_reason"]
    if has_link:
        if objective in {"current_link_quality", "selection_multitask"}:
            auxiliary["loss_link_quality"] = float(val_link_quality_loss / batches)
            metrics["loss/link_quality"] = auxiliary["loss_link_quality"]
        for key in ("link_mae", "link_rmse", "link_r2"):
            metrics[key] = float(auxiliary_metrics[key])
            metrics[f"val_{key}"] = float(auxiliary_metrics[key])
    if objective == "selection_multitask":
        auxiliary["loss_selection_multitask_total"] = float(val_selection_multitask_loss / batches)
        metrics["loss/selection_multitask_total"] = auxiliary["loss_selection_multitask_total"]
        metrics["selection_multitask_loss"] = auxiliary["loss_selection_multitask_total"]
        metrics["val_selection_multitask_loss"] = auxiliary["loss_selection_multitask_total"]

    if auxiliary:
        metrics["auxiliary"] = auxiliary


__all__ = ["EvaluationPassResult", "run_evaluation_pass"]
