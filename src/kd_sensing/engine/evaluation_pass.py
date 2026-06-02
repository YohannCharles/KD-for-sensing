from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from kd_sensing.engine.batch import (
    prepare_beamspace_power_targets,
    prepare_beam_power_targets,
    prepare_history_anchor_inputs,
    prepare_path_descriptors,
    prepare_path_semantic_labels,
    prepare_radio_semantic_labels,
)
from kd_sensing.engine.debug_diagnostics import set_csi_debug_batch_source
from kd_sensing.engine.hist_beam_history_anchor import history_anchor_run_metadata
from kd_sensing.engine.hist_beam_image_only import filter_image_only_batch, image_only_protocol_enabled, image_only_run_metadata
from kd_sensing.engine.modality_resolution import config_uses_lidar, resolve_enabled_modalities
from kd_sensing.engine.objectives.metadata import (
    objective_available_metrics,
    objective_runtime_metadata,
    resolve_prediction_objective,
)
from kd_sensing.engine.hist_beam_prototypes import load_source_prototypes
from kd_sensing.engine.hist_beam_residuals import (
    history_anchor_enabled,
    num_delta_classes_from_config,
    residual_target_enabled,
)
from kd_sensing.engine.prediction_objectives import (
    compute_prediction_loss,
    prepare_prediction_targets,
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
    calculate_current_beam_dba,
    calculate_dba_score,
    calculate_link_metrics,
    calculate_los_metrics,
    calculate_occlusion_metrics,
    calculate_position_rmse,
    calculate_topk_accuracy,
)
from kd_sensing.evaluation.hist_beam_outputs import (
    beam_power_metrics,
    beam_histogram_metrics,
    calculate_hist_beam_metrics,
    markov_delta_baseline_metrics,
    path_descriptor_regression_metrics,
    path_semantic_metrics,
    radio_semantic_metrics,
)
from kd_sensing.evaluation.horizon_selection import (
    horizon_indices,
    metric_horizon_source_from_config,
    metric_horizons_from_config,
)


@dataclass(frozen=True)
class EvaluationPassResult:
    metrics: dict[str, Any]
    outputs: torch.Tensor
    labels: torch.Tensor
    input_beams: torch.Tensor | None
    last_beams: torch.Tensor | None
    residual_logits: torch.Tensor | None
    residual_labels: torch.Tensor | None
    radio_logits: torch.Tensor | None
    radio_labels: torch.Tensor | None
    path_logits: torch.Tensor | None
    path_labels: torch.Tensor | None
    path_attr_pred: torch.Tensor | None
    path_descriptors: torch.Tensor | None
    path_valid: torch.Tensor | None
    beam_power: torch.Tensor | None
    shared_logits: torch.Tensor | None
    target_logits: torch.Tensor | None
    target_prior_bias: torch.Tensor | None
    prototype_logits: torch.Tensor | None
    alpha: torch.Tensor | None
    delta_logits_private: torch.Tensor | None
    pred_beamspace_power: torch.Tensor | None
    beamspace_power_label: torch.Tensor | None
    beamspace_power_mask: torch.Tensor | None
    metadata: list[dict[str, Any]]
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
    hist_forward_kwargs = _hist_forward_kwargs(cfg, device)
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
    all_last_beams = []
    all_residual_logits = []
    all_residual_labels = []
    all_metadata: list[dict[str, Any]] = []
    all_occlusion_logits = []
    all_occlusion_labels = []
    all_occlusion_valid = []
    all_position_outputs = []
    all_position_targets = []
    all_position_valid = []
    all_los_logits = []
    all_los_labels = []
    all_los_bucket_labels = []
    all_link_outputs = []
    all_link_targets = []
    all_radio_logits = []
    all_radio_labels = []
    all_path_logits = []
    all_path_labels = []
    all_path_attr_pred = []
    all_path_descriptors = []
    all_path_valid = []
    all_beam_power = []
    all_shared_logits = []
    all_target_logits = []
    all_target_prior_bias = []
    all_prototype_logits = []
    all_alpha = []
    all_delta_logits_private = []
    all_pred_beamspace_power = []
    all_beamspace_power_label = []
    all_beamspace_power_mask = []
    lidar_quality = LidarQualityAccumulator()
    saw_lidar = False

    with torch.no_grad():
        for batch in dataloader:
            batch = filter_image_only_batch(prepare_task_batch(batch), cfg, stage="target_test")
            all_metadata.extend(_metadata_rows_from_batch(batch.get("metadata")))
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
            radio_labels = prepare_radio_semantic_labels(
                batch,
                num_pred=num_pred,
                device=device,
                non_blocking=non_blocking,
            )
            path_labels = prepare_path_semantic_labels(
                batch,
                num_pred=num_pred,
                device=device,
                non_blocking=non_blocking,
            )
            path_targets = prepare_path_descriptors(
                batch,
                num_pred=num_pred,
                device=device,
                non_blocking=non_blocking,
            )
            beam_power = prepare_beam_power_targets(
                batch,
                num_pred=num_pred,
                device=device,
                non_blocking=non_blocking,
            )
            beamspace_targets = prepare_beamspace_power_targets(
                batch,
                num_pred=num_pred,
                device=device,
                non_blocking=non_blocking,
            )
            if radio_labels is not None:
                all_radio_labels.append(radio_labels.detach().cpu())
            if path_labels is not None:
                all_path_labels.append(path_labels.detach().cpu())
            if path_targets is not None:
                all_path_descriptors.append(path_targets[0].detach().cpu())
                all_path_valid.append(path_targets[1].detach().cpu())
            if beam_power is not None:
                all_beam_power.append(beam_power.detach().cpu())
            if beamspace_targets is not None:
                all_beamspace_power_label.append(beamspace_targets[0].detach().cpu())
                all_beamspace_power_mask.append(beamspace_targets[1].detach().cpu())
            if "los_label" in auxiliary_targets:
                all_los_bucket_labels.append(auxiliary_targets["los_label"].detach().cpu())
            prediction_targets = prepare_prediction_targets(
                labels=labels,
                auxiliary_targets=auxiliary_targets,
                cfg=cfg,
            )
            batch_history_kwargs = prepare_history_anchor_inputs(
                batch,
                num_pred=num_pred,
                num_classes=num_delta_classes_from_config(cfg, default=num_classes),
                downsample_ratio=downsample_ratio,
                device=device,
                enabled=history_anchor_enabled(cfg),
                non_blocking=non_blocking,
                sample_ids=_sample_ids_from_metadata(batch.get("metadata")),
            )
            if "last_beam_batch" in batch_history_kwargs:
                all_last_beams.append(batch_history_kwargs["last_beam_batch"].detach().cpu())
            if "residual_labels" in batch_history_kwargs:
                all_residual_labels.append(batch_history_kwargs["residual_labels"].detach().cpu())
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
                    extra_model_kwargs={
                        **hist_forward_kwargs,
                        **{key: value for key, value in batch_history_kwargs.items() if key != "residual_labels"},
                    },
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
            radio_logits = step.model_output.diagnostics.get("radio_logits")
            if torch.is_tensor(radio_logits):
                all_radio_logits.append(radio_logits.detach().cpu())
            path_logits = step.model_output.diagnostics.get("path_logits")
            if torch.is_tensor(path_logits):
                all_path_logits.append(path_logits.detach().cpu())
            path_attr_pred = step.model_output.diagnostics.get("path_attr_pred")
            if torch.is_tensor(path_attr_pred):
                all_path_attr_pred.append(path_attr_pred.detach().cpu())
            residual_logits = step.model_output.diagnostics.get("residual_logits")
            if torch.is_tensor(residual_logits):
                all_residual_logits.append(residual_logits.detach().cpu())
            for key, bucket in (
                ("logits_shared", all_shared_logits),
                ("target_logits", all_target_logits),
                ("target_prior_bias", all_target_prior_bias),
                ("prototype_logits", all_prototype_logits),
                ("alpha", all_alpha),
                ("delta_logits_private", all_delta_logits_private),
                ("pred_beamspace_power", all_pred_beamspace_power),
            ):
                value = step.model_output.diagnostics.get(key)
                if torch.is_tensor(value):
                    bucket.append(value.detach().cpu())

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
    radio_logits_t = torch.cat(all_radio_logits, dim=0) if all_radio_logits else None
    radio_labels_t = torch.cat(all_radio_labels, dim=0) if all_radio_labels else None
    path_logits_t = torch.cat(all_path_logits, dim=0) if all_path_logits else None
    path_labels_t = torch.cat(all_path_labels, dim=0) if all_path_labels else None
    path_attr_pred_t = torch.cat(all_path_attr_pred, dim=0) if all_path_attr_pred else None
    path_descriptors_t = torch.cat(all_path_descriptors, dim=0) if all_path_descriptors else None
    path_valid_t = torch.cat(all_path_valid, dim=0) if all_path_valid else None
    beam_power_t = torch.cat(all_beam_power, dim=0) if all_beam_power else None
    shared_logits_t = torch.cat(all_shared_logits, dim=0) if all_shared_logits else None
    target_logits_t = torch.cat(all_target_logits, dim=0) if all_target_logits else None
    target_prior_bias_t = torch.cat(all_target_prior_bias, dim=0) if all_target_prior_bias else None
    prototype_logits_t = torch.cat(all_prototype_logits, dim=0) if all_prototype_logits else None
    alpha_t = torch.cat(all_alpha, dim=0) if all_alpha else None
    delta_logits_private_t = torch.cat(all_delta_logits_private, dim=0) if all_delta_logits_private else None
    pred_beamspace_power_t = torch.cat(all_pred_beamspace_power, dim=0) if all_pred_beamspace_power else None
    beamspace_power_label_t = torch.cat(all_beamspace_power_label, dim=0) if all_beamspace_power_label else None
    beamspace_power_mask_t = torch.cat(all_beamspace_power_mask, dim=0) if all_beamspace_power_mask else None
    input_beams_t = torch.cat(all_input_beams, dim=0) if all_input_beams else None
    last_beams_t = torch.cat(all_last_beams, dim=0) if all_last_beams else None
    residual_logits_t = torch.cat(all_residual_logits, dim=0) if all_residual_logits else None
    residual_labels_t = torch.cat(all_residual_labels, dim=0) if all_residual_labels else None
    if _hist_beam_metrics_enabled(cfg):
        metrics.update(radio_semantic_metrics(radio_logits_t, radio_labels_t))
        metrics.update(path_semantic_metrics(path_logits_t, path_labels_t))
        metrics.update(path_descriptor_regression_metrics(path_attr_pred_t, path_descriptors_t, path_valid_t))
        pred_beams = outputs_t.argmax(dim=-1)
        metrics.update(
            beam_power_metrics(
                pred_beams.reshape(-1),
                labels_t.reshape(-1),
                beam_power_t.reshape(-1, beam_power_t.shape[-1]) if beam_power_t is not None else None,
            )
        )
        metrics.update(beam_histogram_metrics(labels_t, outputs_t, num_classes=num_classes, prefix="target_test"))
        if shared_logits_t is not None:
            metrics.update(_v7_evaluation_metrics(
                final_logits=outputs_t,
                shared_logits=shared_logits_t,
                labels=labels_t,
                beam_power=beam_power_t,
                alpha=alpha_t,
                delta_logits_private=delta_logits_private_t,
                pred_beamspace_power=pred_beamspace_power_t,
                beamspace_power_label=beamspace_power_label_t,
                beamspace_power_mask=beamspace_power_mask_t,
                k_values=cfg.get("evaluation", {}).get("k_values", [1, 3, 5]),
            ))
    if objective in {"current_beam_selection", "selection_multitask"} and all_los_bucket_labels:
        metrics["los_buckets"] = _beam_metrics_by_los_bucket(
            outputs_t,
            labels_t,
            torch.cat(all_los_bucket_labels, dim=0),
            cfg,
        )
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
    if residual_target_enabled(cfg):
        metrics.update(_residual_evaluation_metrics(residual_logits_t, residual_labels_t, outputs_t, labels_t, cfg))
    markov_reference = cfg.get("hist_beam", {}).get("markov_baseline", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    if isinstance(markov_reference, dict) and markov_reference.get("enabled", True) is not False:
        metrics.update(
            markov_delta_baseline_metrics(
                markov_reference.get("last_beams"),
                markov_reference.get("labels"),
                last_beams_t,
                labels_t,
                num_classes=num_classes,
                k_values=cfg.get("evaluation", {}).get("k_values", [1, 3, 5]),
                smoothing=float(markov_reference.get("smoothing", 1.0)),
                train_split=str(markov_reference.get("split", "source_train")),
            )
        )
    metrics.update(history_anchor_run_metadata(cfg))
    if image_only_protocol_enabled(cfg):
        metrics.update(image_only_run_metadata(cfg, stage="target_test"))
        metrics["target_test_label_usage"] = "evaluation_only"
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
        last_beams=last_beams_t,
        residual_logits=residual_logits_t,
        residual_labels=residual_labels_t,
        radio_logits=radio_logits_t,
        radio_labels=radio_labels_t,
        path_logits=path_logits_t,
        path_labels=path_labels_t,
        path_attr_pred=path_attr_pred_t,
        path_descriptors=path_descriptors_t,
        path_valid=path_valid_t,
        beam_power=beam_power_t,
        shared_logits=shared_logits_t,
        target_logits=target_logits_t,
        target_prior_bias=target_prior_bias_t,
        prototype_logits=prototype_logits_t,
        alpha=alpha_t,
        delta_logits_private=delta_logits_private_t,
        pred_beamspace_power=pred_beamspace_power_t,
        beamspace_power_label=beamspace_power_label_t,
        beamspace_power_mask=beamspace_power_mask_t,
        metadata=all_metadata,
        objective_metadata=objective_metadata,
        enabled_modalities=enabled_modalities,
        saw_lidar=saw_lidar,
    )


def _hist_forward_kwargs(cfg: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    proto_cfg = hist_cfg.get("prototype", {}) if isinstance(hist_cfg.get("prototype"), dict) else {}
    proto_path = (
        hist_cfg.get("source_prototype_path")
        or proto_cfg.get("source_prototype_path")
        or proto_cfg.get("path")
        or proto_cfg.get("artifact_path")
    )
    if not proto_path:
        return {}
    try:
        artifact = load_source_prototypes(proto_path, map_location=device)
    except Exception:
        return {}
    proto_type = str(hist_cfg.get("proto_type", proto_cfg.get("proto_type", (artifact.get("metadata") or {}).get("proto_type", "")))).strip().lower()
    kwargs: dict[str, torch.Tensor] = {}
    if proto_type == "path":
        path = artifact.get("mu_path_c", artifact.get("shared_prototypes"))
        counts = artifact.get("count_path", artifact.get("counts"))
        if torch.is_tensor(path):
            kwargs["path_prototypes"] = path.to(device=device)
        if torch.is_tensor(counts):
            kwargs["path_prototype_counts"] = counts.to(device=device)
    elif proto_type == "radio_semantic":
        radio = artifact.get("mu_radio_c", artifact.get("shared_prototypes"))
        counts = artifact.get("count_radio", artifact.get("counts"))
        if torch.is_tensor(radio):
            kwargs["radio_prototypes"] = radio.to(device=device)
        if torch.is_tensor(counts):
            kwargs["radio_prototype_counts"] = counts.to(device=device)
    return kwargs


def _metrics_from_outputs(
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
    }
    if objective in {"current_beam_selection", "selection_multitask"}:
        metrics.update(_flat_current_beam_metrics(topk_acc, total))
        beam_dba = calculate_current_beam_dba(
            outputs,
            labels,
            cfg.get("evaluation", {}).get("dba_delta", 5),
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
        )
        metrics["dba"] = dba_score.tolist()
    if _hist_beam_metrics_enabled(cfg):
        hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
        model_cfg = cfg.get("model", {}).get("student", {})
        metrics.update(
            calculate_hist_beam_metrics(
                outputs,
                labels,
                group_size=int(hist_cfg.get("group_size", model_cfg.get("group_size", 8))),
                num_classes=int(model_cfg.get("num_classes", cfg.get("model", {}).get("num_classes", outputs.shape[-1]))),
            )
        )
        metrics.setdefault("power_metrics_available", False)
        metrics.setdefault("power_metrics_unavailable_reason", "beam_power_vector_missing")
    return metrics


def _hist_beam_metrics_enabled(cfg: dict[str, Any]) -> bool:
    hist_cfg = cfg.get("hist_beam")
    if isinstance(hist_cfg, dict) and hist_cfg.get("enabled") is not False:
        return True
    return cfg.get("model", {}).get("student", {}).get("type") == "hist_beam_fusion"


def _v7_evaluation_metrics(
    *,
    final_logits: torch.Tensor,
    shared_logits: torch.Tensor,
    labels: torch.Tensor,
    beam_power: torch.Tensor | None,
    alpha: torch.Tensor | None,
    delta_logits_private: torch.Tensor | None,
    pred_beamspace_power: torch.Tensor | None,
    beamspace_power_label: torch.Tensor | None,
    beamspace_power_mask: torch.Tensor | None,
    k_values: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    shared_topk, shared_total = calculate_topk_accuracy(shared_logits, labels, k_values)
    final_topk, final_total = calculate_topk_accuracy(final_logits, labels, k_values)
    metrics: dict[str, Any] = {
        "shared_topk": {str(k): v.tolist() for k, v in shared_topk.items()},
        "final_topk": {str(k): v.tolist() for k, v in final_topk.items()},
        "shared_total": shared_total.tolist(),
        "final_total": final_total.tolist(),
    }
    for k in (1, 3):
        if k in shared_topk:
            metrics[f"shared_top{k}"] = _topk_average(shared_topk[k], shared_total)
        if k in final_topk:
            metrics[f"final_top{k}"] = _topk_average(final_topk[k], final_total)
    shared_power = beam_power_metrics(
        shared_logits.argmax(dim=-1).reshape(-1),
        labels.reshape(-1),
        beam_power.reshape(-1, beam_power.shape[-1]) if beam_power is not None else None,
    )
    final_power = beam_power_metrics(
        final_logits.argmax(dim=-1).reshape(-1),
        labels.reshape(-1),
        beam_power.reshape(-1, beam_power.shape[-1]) if beam_power is not None else None,
    )
    metrics.update(_prefix_metrics(shared_power, "shared", rename={"normalized_received_power": "nrp"}))
    metrics.update(_prefix_metrics(final_power, "final", rename={"normalized_received_power": "nrp"}))
    if alpha is not None:
        metrics["alpha_mean"] = float(alpha.float().mean().item())
        metrics["alpha_std"] = float(alpha.float().std(unbiased=False).item())
    if delta_logits_private is not None:
        metrics["delta_norm"] = float(delta_logits_private.float().norm(dim=-1).mean().item())
    if pred_beamspace_power is not None and beamspace_power_label is not None:
        phys = _physical_kl_metric(pred_beamspace_power, beamspace_power_label, beamspace_power_mask)
        metrics.update(phys)
    return metrics


def _topk_average(values: torch.Tensor, total: torch.Tensor) -> float:
    values_t = torch.as_tensor(values, dtype=torch.float32)
    total_t = torch.as_tensor(total, dtype=torch.float32)
    valid = total_t.gt(0)
    return float(values_t[valid].mean().item()) if torch.any(valid) else 0.0


def _prefix_metrics(metrics: dict[str, Any], prefix: str, *, rename: dict[str, str] | None = None) -> dict[str, Any]:
    renamed = rename or {}
    return {f"{prefix}_{renamed.get(key, key)}": value for key, value in metrics.items()}


def _physical_kl_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> dict[str, Any]:
    pred_t = pred.to(torch.float32)
    target_t = target.to(torch.float32)
    valid = torch.isfinite(target_t).all(dim=-1) & target_t.sum(dim=-1).gt(0)
    if mask is not None:
        valid = valid & mask.to(torch.bool)
    if not torch.any(valid):
        return {"phys_kl_available": False, "phys_kl_unavailable_reason": "beamspace_power_label_unavailable"}
    target_t = target_t / target_t.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    pred_t = pred_t.clamp_min(1e-12)
    pred_t = pred_t / pred_t.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    kl = torch.sum(target_t[valid] * (torch.log(target_t[valid].clamp_min(1e-12)) - torch.log(pred_t[valid])), dim=-1)
    return {
        "phys_kl_available": True,
        "phys_kl": float(kl.mean().item()),
        "phys_kl_coverage": float(valid.float().mean().item()),
    }


def _metadata_rows_from_batch(metadata: Any) -> list[dict[str, Any]]:
    if metadata is None:
        return []
    if isinstance(metadata, list):
        return [dict(item) for item in metadata if isinstance(item, dict)]
    if not isinstance(metadata, dict):
        return []
    length = 0
    for value in metadata.values():
        if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0:
            length = max(length, int(value.shape[0]))
        elif isinstance(value, (list, tuple)):
            length = max(length, len(value))
        else:
            length = max(length, 1)
    rows: list[dict[str, Any]] = []
    for index in range(length):
        row = {}
        for key, value in metadata.items():
            row[key] = _metadata_value_at(value, index)
        rows.append(row)
    return rows


def _metadata_value_at(value: Any, index: int) -> Any:
    if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0:
        item = value[index]
        return item.item() if hasattr(item, "item") else item
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    return value


def _sample_ids_from_metadata(metadata: Any) -> list[str] | None:
    rows = _metadata_rows_from_batch(metadata)
    if not rows:
        return None
    return [str(row.get("sample_id", index)) for index, row in enumerate(rows)]


def _residual_evaluation_metrics(
    residual_logits: torch.Tensor | None,
    residual_labels: torch.Tensor | None,
    outputs: torch.Tensor,
    labels: torch.Tensor,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    if residual_logits is None or residual_labels is None:
        return {
            "residual_metrics_available": False,
            "residual_metrics_unavailable_reason": "residual_logits_or_labels_missing",
            "history_anchor_enabled": True,
            "residual_target_enabled": True,
        }
    if residual_logits.ndim == 2:
        residual_logits = residual_logits.unsqueeze(1)
    if residual_labels.ndim == 1:
        residual_labels = residual_labels.unsqueeze(1)
    valid = residual_labels.ge(0) & residual_labels.lt(residual_logits.shape[-1])
    if not torch.any(valid):
        return {
            "residual_metrics_available": False,
            "residual_metrics_unavailable_reason": "no_valid_residual_labels",
            "history_anchor_enabled": True,
            "residual_target_enabled": True,
        }
    residual_pred = residual_logits.argmax(dim=-1)
    residual_accuracy = (residual_pred[valid] == residual_labels[valid]).float().mean()
    topk_acc, total = calculate_topk_accuracy(
        outputs,
        labels,
        cfg.get("evaluation", {}).get("k_values", [1, 3, 5]),
    )
    result = {
        "residual_metrics_available": True,
        "residual_metrics_unavailable_reason": None,
        "history_anchor_enabled": True,
        "residual_target_enabled": True,
        "residual_accuracy": float(residual_accuracy.detach().cpu().item()),
        "residual_total": int(valid.sum().detach().cpu().item()),
        "reconstructed_absolute_topk": {str(k): v.tolist() for k, v in topk_acc.items()},
        "reconstructed_absolute_total": total.tolist(),
    }
    for key in (1, 3, 5):
        if key in topk_acc:
            values = torch.as_tensor(topk_acc[key], dtype=torch.float32)
            count = torch.as_tensor(total, dtype=torch.float32)
            mask = count.gt(0)
            result[f"reconstructed_absolute_top{key}_avg"] = float(values[mask].mean().item()) if torch.any(mask) else 0.0
            result[f"val_reconstructed_absolute_top{key}_avg"] = result[f"reconstructed_absolute_top{key}_avg"]
    result["val_residual_accuracy"] = result["residual_accuracy"]
    return result


def _beam_metrics_by_los_bucket(
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
        bucket_metrics = _metrics_from_outputs(
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
