from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from kd_sensing.engine.evaluation_pass_runtime import (
    metadata_rows_from_batch,
    prepare_evaluation_batch,
)
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.engine.objectives.metadata import (
    objective_available_metrics,
    objective_runtime_metadata,
)
from kd_sensing.engine.prediction_objectives import compute_prediction_loss
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
from kd_sensing.evaluation.metrics import calculate_dba_score, calculate_topk_accuracy, circular_beam_distance


@dataclass(frozen=True)
class EvaluationPassResult:
    metrics: dict[str, Any]
    outputs: torch.Tensor | None
    labels: torch.Tensor | None
    metadata: list[dict[str, Any]]
    objective_metadata: dict[str, Any]
    enabled_modalities: tuple[str, ...]


@dataclass
class _EvaluationState:
    loss_sum: torch.Tensor | None = None
    observations: torch.Tensor | None = None
    totals: torch.Tensor | None = None
    topk_correct: dict[int, torch.Tensor] = field(default_factory=dict)
    dba_sum: torch.Tensor | None = None
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
    capture_outputs: bool = False,
    batch_transform: Callable[[Any], Any] | None = None,
) -> EvaluationPassResult:
    model.eval()
    model_cfg = cfg["model"]
    num_pred = int(model_cfg.get("num_pred", 1))
    num_classes = int(model_cfg.get("num_classes", 64))
    non_blocking = transfer_non_blocking(cfg)
    amp_enabled, amp_dtype = resolve_amp_settings(cfg, device)
    state = _EvaluationState()

    with torch.no_grad():
        for raw_batch in dataloader:
            if batch_transform is not None:
                raw_batch = batch_transform(raw_batch)
            batch = prepare_evaluation_batch(raw_batch)
            labels = prepare_task_labels(
                batch,
                num_pred=num_pred,
                device=device,
                non_blocking=non_blocking,
            )
            valid = labels.ne(-100)
            observations = valid.sum()
            if int(observations.item()) <= 0:
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
                validation_loss = getattr(model, "compute_validation_loss", None)
                if callable(validation_loss):
                    loss = validation_loss(step.model_output, labels, cfg)
                else:
                    beam_loss = criterion(step.logits.reshape(-1, num_classes), labels.flatten())
                    loss = compute_prediction_loss(
                        step.model_output,
                        cfg,
                        reference=step.logits,
                        beam_total_loss=beam_loss,
                        beam_task_loss=beam_loss,
                    ).total
            _accumulate_beam_metrics(state, step.logits.detach(), labels, cfg)
            weighted_loss = loss.detach() * observations.to(dtype=loss.dtype)
            state.loss_sum = weighted_loss if state.loss_sum is None else state.loss_sum + weighted_loss
            state.observations = observations if state.observations is None else state.observations + observations
            if capture_outputs:
                state.outputs.append(step.logits.detach().cpu())
                state.labels.append(labels.detach().cpu())
                state.metadata.extend(metadata_rows_from_batch(batch.get("metadata")))

    if state.observations is None or state.totals is None or int(state.observations.item()) <= 0:
        raise ValueError("Evaluation pass produced no valid beam predictions.")
    assert state.loss_sum is not None and state.dba_sum is not None
    metrics = _metrics_from_accumulators(
        loss=float((state.loss_sum / state.observations.clamp_min(1)).item()),
        totals=state.totals,
        topk_correct=state.topk_correct,
        dba_sum=state.dba_sum,
        cfg=cfg,
    )
    metrics.update(
        {
            "loss_observation_count": int(state.observations.item()),
            "objective": objective_runtime_metadata(),
            "enabled_modalities": list(resolve_enabled_modalities(cfg)),
            "prediction_capture": bool(capture_outputs),
        }
    )
    metrics["available_metrics"] = objective_available_metrics(metrics)
    outputs = torch.cat(state.outputs) if capture_outputs and state.outputs else None
    labels = torch.cat(state.labels) if capture_outputs and state.labels else None
    return EvaluationPassResult(
        metrics=metrics,
        outputs=outputs,
        labels=labels,
        metadata=state.metadata,
        objective_metadata=objective_runtime_metadata(),
        enabled_modalities=resolve_enabled_modalities(cfg),
    )


def _accumulate_beam_metrics(
    state: _EvaluationState,
    outputs: torch.Tensor,
    labels: torch.Tensor,
    cfg: dict[str, Any],
) -> None:
    if outputs.ndim != 3 or labels.ndim != 2 or outputs.shape[:2] != labels.shape:
        raise ValueError("Evaluation outputs and labels must have shapes [B, H, C] and [B, H].")
    valid = labels.ne(-100)
    totals = valid.sum(dim=0)
    k_values = tuple(int(value) for value in cfg.get("evaluation", {}).get("k_values", [1, 3, 5]))
    if not k_values or any(value <= 0 for value in k_values):
        raise ValueError("evaluation.k_values must contain positive integers.")
    max_k = min(max(k_values), int(outputs.shape[-1]))
    predictions = outputs.topk(max_k, dim=-1).indices
    matches = predictions.eq(labels.unsqueeze(-1)) & valid.unsqueeze(-1)
    topk_correct = {
        k: matches[..., : min(k, max_k)].any(dim=-1).sum(dim=0)
        for k in k_values
    }
    dba_predictions = outputs.topk(min(3, int(outputs.shape[-1])), dim=-1).indices
    distance_mode = str(cfg.get("evaluation", {}).get("dba_distance_mode", "circular")).strip().lower()
    if distance_mode in {"circular", "wrap", "wrapped"}:
        distances = circular_beam_distance(dba_predictions, labels.unsqueeze(-1), num_beams=int(outputs.shape[-1]))
    elif distance_mode in {"linear", "official", "beambench", "non_circular", "noncircular"}:
        distances = (dba_predictions.to(torch.long) - labels.unsqueeze(-1).to(torch.long)).abs()
    else:
        raise ValueError("evaluation.dba_distance_mode must be 'circular' or 'linear'.")
    delta = max(float(cfg.get("evaluation", {}).get("dba_delta", 5)), 1e-8)
    progressive = 1.0 - torch.cummin((distances.to(torch.float32) / delta).clamp(max=1.0), dim=-1).values
    dba_sum = (progressive.mean(dim=-1) * valid).sum(dim=0)
    state.totals = totals if state.totals is None else state.totals + totals
    state.dba_sum = dba_sum if state.dba_sum is None else state.dba_sum + dba_sum
    for k, value in topk_correct.items():
        state.topk_correct[k] = value if k not in state.topk_correct else state.topk_correct[k] + value


def _metrics_from_accumulators(
    *,
    loss: float,
    totals: torch.Tensor,
    topk_correct: dict[int, torch.Tensor],
    dba_sum: torch.Tensor,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    horizons = metric_horizons_from_config(cfg, num_pred=int(totals.numel()))
    total = totals.detach().cpu().numpy()
    topk = {
        int(k): (value.to(torch.float32) / totals.clamp_min(1)).detach().cpu().numpy()
        for k, value in topk_correct.items()
    }
    dba = (dba_sum.to(torch.float32) / totals.clamp_min(1)).detach().cpu().numpy()
    distance_mode = str(cfg.get("evaluation", {}).get("dba_distance_mode", "circular"))
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
        "metric_profile": _metric_profile(cfg, distance_mode=distance_mode),
        "dba_distance_mode": distance_mode,
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


def _metrics_from_outputs(
    loss: float,
    outputs: torch.Tensor,
    labels: torch.Tensor,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    horizons = metric_horizons_from_config(cfg, num_pred=int(labels.shape[1]))
    topk, total = calculate_topk_accuracy(outputs, labels, cfg.get("evaluation", {}).get("k_values", [1, 3, 5]))
    dba = calculate_dba_score(
        outputs,
        labels,
        cfg.get("evaluation", {}).get("dba_delta", 5),
        distance_mode=cfg.get("evaluation", {}).get("dba_distance_mode", "circular"),
    )
    distance_mode = str(cfg.get("evaluation", {}).get("dba_distance_mode", "circular"))
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
        "metric_profile": _metric_profile(cfg, distance_mode=distance_mode),
        "dba_distance_mode": distance_mode,
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


def _metric_profile(cfg: dict[str, Any], *, distance_mode: str) -> str:
    profile = str(cfg.get("evaluation", {}).get("metric_profile", ""))
    legacy_profiles = {"64_beam_circular_topk", "64_beam_linear_topk"}
    if profile and profile not in legacy_profiles:
        return profile
    label_space = str(cfg.get("evaluation", {}).get("label_space", "64_beam"))
    beam_count = label_space.split("_", 1)[0] if label_space.endswith("_beam") else "64"
    return f"{beam_count}_beam_{distance_mode}_topk_progressive_top3_dba_v1"


__all__ = ["EvaluationPassResult", "run_evaluation_pass"]
