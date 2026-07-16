from dataclasses import dataclass, field
from typing import Any

import torch

from kd_sensing.engine.evaluation_pass_runtime import (
    metadata_rows_from_batch,
    prepare_evaluation_batch,
)
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.engine.objectives.metadata import (
    objective_available_metrics,
    objective_runtime_metadata,
    resolve_prediction_objective,
)
from kd_sensing.engine.prediction_objectives import compute_prediction_loss, prepare_prediction_targets
from kd_sensing.engine.runtime import (
    autocast_context,
    prepare_task_labels,
    resolve_amp_settings,
    run_model_step,
    transfer_non_blocking,
)
from kd_sensing.evaluation.horizon_selection import (
    aggregate_topk_and_dba,
    horizon_indices,
    metric_horizon_source_from_config,
    metric_horizons_from_config,
)
from kd_sensing.evaluation.metrics import calculate_dba_score, calculate_topk_accuracy


@dataclass(frozen=True)
class EvaluationPassResult:
    metrics: dict[str, Any]
    outputs: torch.Tensor
    labels: torch.Tensor
    metadata: list[dict[str, Any]]
    objective_metadata: dict[str, Any]
    enabled_modalities: tuple[str, ...]


@dataclass
class _EvaluationState:
    loss_sum: float = 0.0
    observations: int = 0
    outputs: list[torch.Tensor] = field(default_factory=list)
    labels: list[torch.Tensor] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)


def run_evaluation_pass(
    model,
    dataloader,
    cfg: dict[str, Any],
    criterion,
    device: torch.device,
    *,
    force_modality_mask: torch.Tensor | None = None,
) -> EvaluationPassResult:
    objective = resolve_prediction_objective(cfg)
    model.eval()
    model_cfg = cfg["model"]
    num_pred = int(model_cfg.get("num_pred", 1))
    num_classes = int(model_cfg.get("num_classes", 64))
    non_blocking = transfer_non_blocking(cfg)
    amp_enabled, amp_dtype = resolve_amp_settings(cfg, device)
    state = _EvaluationState()

    with torch.no_grad():
        for raw_batch in dataloader:
            batch = prepare_evaluation_batch(raw_batch)
            labels = prepare_task_labels(
                batch,
                num_pred=num_pred,
                device=device,
                non_blocking=non_blocking,
            )
            observations = int(labels.ne(-100).sum().item())
            if observations <= 0:
                raise ValueError("Evaluation has zero valid beam-label observations.")
            with autocast_context(amp_enabled, device, amp_dtype):
                step = run_model_step(
                    model,
                    cfg["experiment"].get("task", "fusion"),
                    batch,
                    seq_length=int(model_cfg.get("seq_length", 5)),
                    num_pred=num_pred,
                    device=device,
                    non_blocking=non_blocking,
                    force_modality_mask=force_modality_mask,
                )
                beam_loss = criterion(step.logits.reshape(-1, num_classes), labels.flatten())
                loss = compute_prediction_loss(
                    step.model_output,
                    prepare_prediction_targets(labels=labels, auxiliary_targets={}, cfg=cfg),
                    cfg,
                    reference=step.logits,
                    beam_total_loss=beam_loss,
                    beam_task_loss=beam_loss,
                ).total
            state.loss_sum += float(loss.detach().cpu()) * observations
            state.observations += observations
            state.outputs.append(step.logits.detach().cpu())
            state.labels.append(labels.detach().cpu())
            state.metadata.extend(metadata_rows_from_batch(batch.get("metadata")))

    if not state.outputs or state.observations <= 0:
        raise ValueError("Evaluation pass produced no valid beam predictions.")
    outputs = torch.cat(state.outputs)
    labels = torch.cat(state.labels)
    metrics = _metrics_from_outputs(state.loss_sum / state.observations, outputs, labels, cfg, objective=objective)
    metrics.update(
        {
            "loss_observation_count": state.observations,
            "objective": objective_runtime_metadata(cfg),
            "enabled_modalities": list(resolve_enabled_modalities(cfg)),
        }
    )
    metrics["available_metrics"] = objective_available_metrics(objective, metrics)
    return EvaluationPassResult(
        metrics=metrics,
        outputs=outputs,
        labels=labels,
        metadata=state.metadata,
        objective_metadata=objective_runtime_metadata(cfg),
        enabled_modalities=resolve_enabled_modalities(cfg),
    )


def _metrics_from_outputs(
    loss: float,
    outputs: torch.Tensor,
    labels: torch.Tensor,
    cfg: dict[str, Any],
    *,
    objective: str = "beam",
) -> dict[str, Any]:
    if objective != "beam":
        raise ValueError("Only beam evaluation is retained.")
    horizons = metric_horizons_from_config(cfg, num_pred=int(labels.shape[1]))
    topk, total = calculate_topk_accuracy(outputs, labels, cfg.get("evaluation", {}).get("k_values", [1, 3, 5]))
    dba = calculate_dba_score(
        outputs,
        labels,
        cfg.get("evaluation", {}).get("dba_delta", 5),
        distance_mode=cfg.get("evaluation", {}).get("dba_distance_mode", "circular"),
    )
    metrics: dict[str, Any] = {
        "loss": float(loss),
        "topk": {str(k): values.tolist() for k, values in topk.items()},
        "total": total.tolist(),
        "dba": dba.tolist(),
        "metric_horizons": list(horizons),
        "metric_horizon_indices": list(horizon_indices(horizons)),
        "metric_horizon_source": metric_horizon_source_from_config(cfg),
        "label_space": str(cfg.get("evaluation", {}).get("label_space", "64_beam")),
        "beam_shift": int(cfg.get("evaluation", {}).get("beam_shift", 0)),
        "metric_profile": str(cfg.get("evaluation", {}).get("metric_profile", "64_beam_circular_topk")),
        "dba_distance_mode": str(cfg.get("evaluation", {}).get("dba_distance_mode", "circular")),
    }
    aggregate = aggregate_topk_and_dba(metrics)
    metrics.update(
        {
            "val_loss": float(loss),
            "val_acc": aggregate["top1"],
            "val_atop3": aggregate["top3"],
            "val_atop5": aggregate["top5"],
            "val_adba": aggregate["adba"],
            "val_top1_avg": aggregate["top1"],
            "val_top3_avg": aggregate["top3"],
            "val_top5_avg": aggregate["top5"],
        }
    )
    for k, values in topk.items():
        for index, value in enumerate(values, start=1):
            metrics[f"val_top{k}_t{index}"] = float(value)
    return metrics


__all__ = ["EvaluationPassResult", "run_evaluation_pass"]
