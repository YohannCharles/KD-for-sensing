from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from kd_sensing.engine.debug_diagnostics import set_csi_debug_batch_source
from kd_sensing.engine.modality_resolution import config_uses_lidar, resolve_enabled_modalities
from kd_sensing.engine.objectives.metadata import objective_runtime_metadata, resolve_prediction_objective
from kd_sensing.engine.evaluation_pass_metrics import (
    attach_objective_metrics,
    auxiliary_metrics_from_outputs,
    available_metrics,
    beam_metrics_by_los_bucket,
    metrics_from_outputs,
)
from kd_sensing.engine.evaluation_pass_runtime import (
    evaluation_split_name,
    metadata_rows_from_batch,
    prepare_evaluation_batch,
)
from kd_sensing.engine.prediction_objectives import (
    compute_prediction_loss,
    prepare_prediction_targets,
)
from kd_sensing.engine.runtime import (
    autocast_context,
    prepare_task_auxiliary_targets,
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
from kd_sensing.losses.jepa import jepa_loss_from_output
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
    metadata: list[dict[str, Any]]
    objective_metadata: dict[str, Any]
    enabled_modalities: tuple[str, ...]
    saw_lidar: bool


@dataclass
class _EvaluationPassState:
    loss: float = 0.0
    loss_observations: int = 0
    occlusion_loss: float = 0.0
    occlusion_loss_observations: int = 0
    position_loss: float = 0.0
    position_loss_observations: int = 0
    multitask_loss: float = 0.0
    multitask_loss_observations: int = 0
    los_loss: float = 0.0
    los_loss_observations: int = 0
    link_quality_loss: float = 0.0
    link_quality_loss_observations: int = 0
    selection_multitask_loss: float = 0.0
    selection_multitask_loss_observations: int = 0
    outputs: list[torch.Tensor] = field(default_factory=list)
    labels: list[torch.Tensor] = field(default_factory=list)
    input_beams: list[torch.Tensor] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)
    occlusion_logits: list[torch.Tensor] = field(default_factory=list)
    occlusion_labels: list[torch.Tensor] = field(default_factory=list)
    occlusion_valid: list[torch.Tensor] = field(default_factory=list)
    position_outputs: list[torch.Tensor] = field(default_factory=list)
    position_targets: list[torch.Tensor] = field(default_factory=list)
    position_valid: list[torch.Tensor] = field(default_factory=list)
    los_logits: list[torch.Tensor] = field(default_factory=list)
    los_labels: list[torch.Tensor] = field(default_factory=list)
    los_bucket_labels: list[torch.Tensor] = field(default_factory=list)
    link_outputs: list[torch.Tensor] = field(default_factory=list)
    link_targets: list[torch.Tensor] = field(default_factory=list)
    lidar_quality: LidarQualityAccumulator = field(default_factory=LidarQualityAccumulator)
    saw_lidar: bool = False


@dataclass(frozen=True)
class _EvaluationBatchTargets:
    labels: torch.Tensor
    auxiliary_targets: dict[str, torch.Tensor]
    prediction_targets: dict[str, Any]


@dataclass(frozen=True)
class _EvaluationBatchStep:
    outputs: torch.Tensor
    diagnostics: Mapping[str, Any]
    prediction_loss: Any
    loss: torch.Tensor


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
    seq_length = model_cfg.get("seq_length", 8)
    num_classes = model_cfg.get("num_classes", 64)
    non_blocking = transfer_non_blocking(cfg)
    amp_enabled, amp_dtype = resolve_amp_settings(cfg, device)
    if objective == "gps_conditioned_jepa":
        return _run_jepa_evaluation_pass(
            model,
            dataloader,
            cfg,
            device,
            task=task,
            model_cfg=model_cfg,
            num_pred=num_pred,
            seq_length=seq_length,
            non_blocking=non_blocking,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            objective_metadata=objective_metadata,
            enabled_modalities=enabled_modalities,
        )
    state = _EvaluationPassState()
    difficulty_seed = int(cfg.get("experiment", {}).get("seed", 0))
    split_name = evaluation_split_name(dataloader, cfg)

    with torch.no_grad():
        for step_index, batch in enumerate(dataloader):
            batch = prepare_evaluation_batch(
                batch,
                cfg=cfg,
                split_name=split_name,
                difficulty_seed=difficulty_seed,
                step_index=step_index,
            )
            _record_evaluation_batch_metadata(state, batch)
            targets = _prepare_evaluation_targets(
                batch,
                cfg=cfg,
                num_pred=num_pred,
                downsample_ratio=downsample_ratio,
                device=device,
                non_blocking=non_blocking,
            )
            if "los_label" in targets.auxiliary_targets:
                state.los_bucket_labels.append(targets.auxiliary_targets["los_label"].detach().cpu())
            with autocast_context(amp_enabled, device, amp_dtype):
                step = _run_supervised_evaluation_step(
                    model,
                    batch,
                    targets,
                    cfg=cfg,
                    criterion=criterion,
                    task=task,
                    seq_length=seq_length,
                    num_pred=num_pred,
                    num_classes=num_classes,
                    device=device,
                    non_blocking=non_blocking,
                    force_modality_mask=force_modality_mask,
                )
            _record_evaluation_batch_outputs(
                state,
                outputs=step.outputs,
                labels=targets.labels,
                diagnostics=step.diagnostics,
                auxiliary_targets=targets.auxiliary_targets,
                prediction_loss=step.prediction_loss,
                loss=step.loss,
                objective=objective,
            )
    return _finalize_supervised_evaluation_result(
        state,
        dataloader=dataloader,
        cfg=cfg,
        objective=objective,
        objective_metadata=objective_metadata,
        enabled_modalities=enabled_modalities,
        num_classes=num_classes,
        downsample_ratio=downsample_ratio,
    )


def _prepare_evaluation_batch(
    batch: Mapping[str, Any],
    *,
    cfg: dict[str, Any],
    split_name: str,
    difficulty_seed: int,
    step_index: int,
) -> dict[str, Any]:
    return prepare_evaluation_batch(
        batch,
        cfg=cfg,
        split_name=split_name,
        difficulty_seed=difficulty_seed,
        step_index=step_index,
    )


def _prepare_evaluation_targets(
    batch: Mapping[str, Any],
    *,
    cfg: dict[str, Any],
    num_pred: int,
    downsample_ratio: int,
    device: torch.device,
    non_blocking: bool,
) -> _EvaluationBatchTargets:
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
    return _EvaluationBatchTargets(
        labels=labels,
        auxiliary_targets=auxiliary_targets,
        prediction_targets=prediction_targets,
    )


def _run_supervised_evaluation_step(
    model,
    batch: Mapping[str, Any],
    targets: _EvaluationBatchTargets,
    *,
    cfg: dict[str, Any],
    criterion,
    task: str,
    seq_length: int,
    num_pred: int,
    num_classes: int,
    device: torch.device,
    non_blocking: bool,
    force_modality_mask: torch.Tensor | None,
) -> _EvaluationBatchStep:
    set_csi_debug_batch_source(model, "val")
    step = run_model_step(
        model,
        task,
        batch,
        model_cfg=cfg["model"]["primary"],
        seq_length=seq_length,
        num_pred=num_pred,
        device=device,
        non_blocking=non_blocking,
        force_modality_mask=force_modality_mask,
    )
    outputs = step.logits
    beam_loss = criterion(outputs.reshape(-1, num_classes), targets.labels.flatten())
    prediction_loss = compute_prediction_loss(
        step.model_output,
        targets.prediction_targets,
        cfg,
        reference=outputs,
        beam_total_loss=beam_loss,
        beam_task_loss=beam_loss,
    )
    return _EvaluationBatchStep(
        outputs=outputs,
        diagnostics=step.model_output.diagnostics,
        prediction_loss=prediction_loss,
        loss=prediction_loss.total,
    )


def _record_evaluation_batch_metadata(state: _EvaluationPassState, batch: Mapping[str, Any]) -> None:
    state.metadata.extend(metadata_rows_from_batch(batch.get("metadata")))
    if "input_beam" in batch:
        state.input_beams.append(batch["input_beam"].detach().cpu())
    if "lidar" in batch:
        state.saw_lidar = True
        state.lidar_quality.update(batch["lidar"], raw_lidar=batch.get("lidar_raw"))


def _record_evaluation_batch_outputs(
    state: _EvaluationPassState,
    *,
    outputs: torch.Tensor,
    labels: torch.Tensor,
    diagnostics: Mapping[str, Any],
    auxiliary_targets: Mapping[str, torch.Tensor],
    prediction_loss,
    loss: torch.Tensor,
    objective: str,
) -> None:
    observations = _effective_loss_observations(objective, labels, auxiliary_targets)
    if observations <= 0:
        raise ValueError(
            f"Evaluation objective '{objective}' has zero valid loss observations in a batch; "
            "refusing to report an unavailable validation loss."
        )
    state.loss += loss.item() * observations
    state.loss_observations += observations
    occlusion_observations = _valid_mask_count(auxiliary_targets.get("occlusion_valid"))
    position_observations = _valid_mask_count(auxiliary_targets.get("position_valid"))
    state.occlusion_loss += prediction_loss.occlusion.item() * occlusion_observations
    state.occlusion_loss_observations += occlusion_observations
    state.position_loss += prediction_loss.position.item() * position_observations
    state.position_loss_observations += position_observations
    if objective == "multitask":
        state.multitask_loss += prediction_loss.multitask_total.item() * observations
        state.multitask_loss_observations += observations
    if prediction_loss.los is not None:
        los_observations = _target_count(auxiliary_targets.get("los_label"))
        state.los_loss += prediction_loss.los.item() * los_observations
        state.los_loss_observations += los_observations
    if prediction_loss.link_quality is not None:
        link_observations = _target_count(auxiliary_targets.get("link_quality"))
        state.link_quality_loss += prediction_loss.link_quality.item() * link_observations
        state.link_quality_loss_observations += link_observations
    if prediction_loss.selection_multitask_total is not None:
        state.selection_multitask_loss += prediction_loss.selection_multitask_total.item() * observations
        state.selection_multitask_loss_observations += observations

    state.outputs.append(outputs.detach().cpu())
    state.labels.append(labels.detach().cpu())
    if "occlusion_logits" in diagnostics and "occlusion_label" in auxiliary_targets:
        state.occlusion_logits.append(diagnostics["occlusion_logits"].detach().cpu())
        state.occlusion_labels.append(auxiliary_targets["occlusion_label"].detach().cpu())
        state.occlusion_valid.append(auxiliary_targets["occlusion_valid"].detach().cpu())
    if "position" in diagnostics and "position_target" in auxiliary_targets:
        state.position_outputs.append(diagnostics["position"].detach().cpu())
        state.position_targets.append(auxiliary_targets["position_target"].detach().cpu())
        state.position_valid.append(auxiliary_targets["position_valid"].detach().cpu())
    if "los_logits" in diagnostics and "los_label" in auxiliary_targets:
        state.los_logits.append(diagnostics["los_logits"].detach().cpu())
        state.los_labels.append(auxiliary_targets["los_label"].detach().cpu())
    if "link_quality" in diagnostics and "link_quality" in auxiliary_targets:
        state.link_outputs.append(diagnostics["link_quality"].detach().cpu())
        state.link_targets.append(auxiliary_targets["link_quality"].detach().cpu())


def _effective_loss_observations(
    objective: str,
    labels: torch.Tensor,
    auxiliary_targets: Mapping[str, torch.Tensor],
) -> int:
    if objective == "occlusion":
        valid = auxiliary_targets.get("occlusion_valid")
        return int(valid.to(torch.bool).sum().item()) if valid is not None else 0
    if objective == "position":
        valid = auxiliary_targets.get("position_valid")
        return int(valid.to(torch.bool).sum().item()) if valid is not None else 0
    if objective == "current_los_classification":
        target = auxiliary_targets.get("los_label")
        return int(target.numel()) if target is not None else 0
    if objective == "current_link_quality":
        target = auxiliary_targets.get("link_quality")
        return int(target.numel()) if target is not None else 0
    return int(labels.ne(-100).sum().item())


def _valid_mask_count(value: torch.Tensor | None) -> int:
    return int(value.to(torch.bool).sum().item()) if value is not None else 0


def _target_count(value: torch.Tensor | None) -> int:
    return int(value.numel()) if value is not None else 0


def _weighted_loss_mean(total: float, observations: int) -> float:
    return float(total / observations) if observations > 0 else 0.0


def _cat_or_none(values: list[torch.Tensor]) -> torch.Tensor | None:
    return torch.cat(values, dim=0) if values else None


def _finalize_supervised_evaluation_result(
    state: _EvaluationPassState,
    *,
    dataloader,
    cfg: dict[str, Any],
    objective: str,
    objective_metadata: dict[str, Any],
    enabled_modalities: tuple[str, ...],
    num_classes: int,
    downsample_ratio: int,
) -> EvaluationPassResult:
    outputs_t = torch.cat(state.outputs, dim=0)
    labels_t = torch.cat(state.labels, dim=0)
    auxiliary_metrics = auxiliary_metrics_from_outputs(
        dataloader,
        occlusion_logits=_cat_or_none(state.occlusion_logits),
        occlusion_labels=_cat_or_none(state.occlusion_labels),
        occlusion_valid=_cat_or_none(state.occlusion_valid),
        position_outputs=_cat_or_none(state.position_outputs),
        position_targets=_cat_or_none(state.position_targets),
        position_valid=_cat_or_none(state.position_valid),
        los_logits=_cat_or_none(state.los_logits),
        los_labels=_cat_or_none(state.los_labels),
        link_outputs=_cat_or_none(state.link_outputs),
        link_targets=_cat_or_none(state.link_targets),
    )
    if state.loss_observations <= 0:
        raise ValueError("Evaluation pass has zero valid loss observations.")
    metrics = metrics_from_outputs(
        state.loss / state.loss_observations,
        outputs_t,
        labels_t,
        cfg,
        objective=objective,
    )
    metrics["loss_observation_count"] = int(state.loss_observations)
    input_beams_t = _cat_or_none(state.input_beams)
    if objective in {"current_beam_selection", "selection_multitask"} and state.los_bucket_labels:
        metrics["los_buckets"] = beam_metrics_by_los_bucket(
            outputs_t,
            labels_t,
            torch.cat(state.los_bucket_labels, dim=0),
            cfg,
        )
    attach_objective_metrics(
        metrics,
        auxiliary_metrics,
        objective=objective,
        val_occlusion_loss=_weighted_loss_mean(state.occlusion_loss, state.occlusion_loss_observations),
        val_position_loss=_weighted_loss_mean(state.position_loss, state.position_loss_observations),
        val_multitask_loss=_weighted_loss_mean(state.multitask_loss, state.multitask_loss_observations),
        val_los_loss=_weighted_loss_mean(state.los_loss, state.los_loss_observations),
        val_link_quality_loss=_weighted_loss_mean(state.link_quality_loss, state.link_quality_loss_observations),
        val_selection_multitask_loss=_weighted_loss_mean(
            state.selection_multitask_loss,
            state.selection_multitask_loss_observations,
        ),
    )
    metrics["objective"] = objective_metadata
    metrics["available_metrics"] = available_metrics(objective, metrics)
    metrics["enabled_modalities"] = list(enabled_modalities)
    dataset = getattr(dataloader, "dataset", None)
    baselines = degradation_baselines_from_labels(
        labels_t,
        input_beams=input_beams_t,
        num_classes=num_classes,
        downsample_ratio=downsample_ratio,
    )
    metrics["degradation_baselines"] = baselines
    if state.saw_lidar:
        quality_summary = state.lidar_quality.finalize(
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
        metadata=state.metadata,
        objective_metadata=objective_metadata,
        enabled_modalities=enabled_modalities,
        saw_lidar=state.saw_lidar,
    )


def _run_jepa_evaluation_pass(
    model,
    dataloader,
    cfg: dict[str, Any],
    device: torch.device,
    *,
    task: str,
    model_cfg: dict[str, Any],
    num_pred: int,
    seq_length: int,
    non_blocking: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    objective_metadata: dict[str, Any],
    enabled_modalities: tuple[str, ...],
) -> EvaluationPassResult:
    val_loss_sum = 0.0
    val_loss_observations = 0
    all_outputs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_metadata: list[dict[str, Any]] = []
    diagnostic_sums: dict[str, float] = {}
    diagnostic_counts: dict[str, int] = {}
    difficulty_seed = int(cfg.get("experiment", {}).get("seed", 0))
    split_name = evaluation_split_name(dataloader, cfg)

    with torch.no_grad():
        for step_index, batch in enumerate(dataloader):
            batch = prepare_evaluation_batch(
                batch,
                cfg=cfg,
                split_name=split_name,
                difficulty_seed=difficulty_seed,
                step_index=step_index,
            )
            missing = [key for key in ("image", "gps") if key not in batch]
            if missing:
                raise ValueError(
                    "gps_conditioned_jepa objective requires image and GPS fields during validation; "
                    f"missing: {', '.join(missing)}."
                )
            all_metadata.extend(metadata_rows_from_batch(batch.get("metadata")))
            with autocast_context(amp_enabled, device, amp_dtype):
                set_csi_debug_batch_source(model, "val")
                step = run_model_step(
                    model,
                    task,
                    batch,
                    model_cfg=cfg["model"]["primary"],
                    seq_length=seq_length,
                    num_pred=num_pred,
                    device=device,
                    non_blocking=non_blocking,
                    extra_model_kwargs={
                        "jepa_epoch": 0,
                        "jepa_step": int(step_index),
                    },
                )
                result = jepa_loss_from_output(step.model_output, cfg)
            loss_value = float(result.loss.detach().cpu().item())
            observations = int(result.diagnostics.get("jepa/valid_target_tokens", 0))
            if observations <= 0:
                raise ValueError("JEPA evaluation produced zero valid target tokens.")
            val_loss_sum += loss_value * observations
            val_loss_observations += observations
            _accumulate_scalar_diagnostics(diagnostic_sums, diagnostic_counts, result.diagnostics)
            _accumulate_scalar_diagnostics(diagnostic_sums, diagnostic_counts, step.model_output.diagnostics)
            all_outputs.append(step.logits.detach().cpu())
            all_labels.append(torch.zeros(step.logits.shape[0], num_pred, dtype=torch.long))

    if val_loss_observations <= 0:
        raise ValueError("JEPA evaluation pass has zero valid target tokens.")
    loss = float(val_loss_sum / val_loss_observations)
    averaged = {
        key: float(value / max(diagnostic_counts.get(key, 0), 1))
        for key, value in diagnostic_sums.items()
        if diagnostic_counts.get(key, 0) > 0
    }
    metrics: dict[str, Any] = {
        "loss": loss,
        "val_loss": loss,
        "val_jepa_loss": loss,
        "topk": {},
        "total": [0 for _ in range(max(int(num_pred), 1))],
        "metric_horizons": list(metric_horizons_from_config(cfg, num_pred=max(int(num_pred), 1))),
        "metric_horizon_indices": list(horizon_indices(metric_horizons_from_config(cfg, num_pred=max(int(num_pred), 1)))),
        "metric_horizon_source": metric_horizon_source_from_config(cfg),
        "objective": objective_metadata,
        "enabled_modalities": list(enabled_modalities),
        "loss_observation_count": int(val_loss_observations),
    }
    if "jepa/mask_target_ratio" in averaged:
        metrics["val_jepa_mask_target_ratio"] = averaged["jepa/mask_target_ratio"]
    if "jepa/mask_context_ratio" in averaged:
        metrics["val_jepa_mask_context_ratio"] = averaged["jepa/mask_context_ratio"]
    if "jepa/ema_decay" in averaged:
        metrics["val_jepa_ema_decay"] = averaged["jepa/ema_decay"]
    metrics["jepa"] = averaged
    metrics["available_metrics"] = available_metrics("gps_conditioned_jepa", metrics)
    outputs_t = torch.cat(all_outputs, dim=0) if all_outputs else torch.empty(0, max(int(num_pred), 1), 1)
    labels_t = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, max(int(num_pred), 1), dtype=torch.long)
    return EvaluationPassResult(
        metrics=metrics,
        outputs=outputs_t,
        labels=labels_t,
        input_beams=None,
        metadata=all_metadata,
        objective_metadata=objective_metadata,
        enabled_modalities=enabled_modalities,
        saw_lidar=False,
    )


def _accumulate_scalar_diagnostics(
    sums: dict[str, float],
    counts: dict[str, int],
    diagnostics: dict[str, Any],
) -> None:
    for key, value in diagnostics.items():
        if isinstance(value, (int, float)):
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
        elif torch.is_tensor(value) and value.numel() == 1:
            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu().item())
            counts[key] = counts.get(key, 0) + 1


def _metrics_from_outputs(
    loss: float,
    outputs: torch.Tensor,
    labels: torch.Tensor,
    cfg: dict[str, Any],
    *,
    objective: str,
) -> dict[str, Any]:
    return metrics_from_outputs(loss, outputs, labels, cfg, objective=objective)


def _metadata_rows_from_batch(metadata: Any) -> list[dict[str, Any]]:
    return metadata_rows_from_batch(metadata)


def _evaluation_split_name(dataloader, cfg: Mapping[str, Any]) -> str:
    return evaluation_split_name(dataloader, cfg)


def _beam_metrics_by_los_bucket(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    los_labels: torch.Tensor,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    return beam_metrics_by_los_bucket(outputs, labels, los_labels, cfg)


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
    return auxiliary_metrics_from_outputs(
        dataloader,
        occlusion_logits=occlusion_logits,
        occlusion_labels=occlusion_labels,
        occlusion_valid=occlusion_valid,
        position_outputs=position_outputs,
        position_targets=position_targets,
        position_valid=position_valid,
        los_logits=los_logits,
        los_labels=los_labels,
        link_outputs=link_outputs,
        link_targets=link_targets,
    )


def _attach_objective_metrics(
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
    attach_objective_metrics(
        metrics,
        auxiliary_metrics,
        objective=objective,
        val_occlusion_loss=val_occlusion_loss,
        val_position_loss=val_position_loss,
        val_multitask_loss=val_multitask_loss,
        val_los_loss=val_los_loss,
        val_link_quality_loss=val_link_quality_loss,
        val_selection_multitask_loss=val_selection_multitask_loss,
    )


__all__ = ["EvaluationPassResult", "run_evaluation_pass"]
