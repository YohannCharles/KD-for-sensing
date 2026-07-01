import math
from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.engine.model_output import adapt_model_output
from kd_sensing.engine.runtime import prepare_task_batch, run_model_step
from kd_sensing.eval.metrics import expected_calibration_error, reliability_error_stats, topk_accuracy
from kd_sensing.evaluation.metrics import beam_classification_circular_summary
from kd_sensing.eval.missing_patterns import (
    get_default_missing_patterns,
    make_fixed_missing_mask,
    sample_eval_random_missing_mask,
)


def evaluate_missing_matrix(
    model,
    dataloader,
    device,
    modalities: list[str],
    patterns: dict[str, list[int]] | None = None,
    random_missing: list[float] | None = None,
    prediction_index: int | str = "last",
    max_batches: int | None = None,
    cfg: dict[str, Any] | None = None,
) -> list[dict]:
    device = torch.device(device)
    model.to(device)
    model.eval()
    fixed_patterns = patterns or get_default_missing_patterns(modalities)
    results = []
    with torch.no_grad():
        for name, pattern in fixed_patterns.items():
            results.append(
                _evaluate_pattern(
                    model,
                    dataloader,
                    device,
                    pattern_name=name,
                    pattern=pattern,
                    prediction_index=prediction_index,
                    max_batches=max_batches,
                    cfg=cfg,
                )
            )
        for p_missing in random_missing or []:
            pattern_name = f"random_{float(p_missing):g}"
            results.append(
                _evaluate_pattern(
                    model,
                    dataloader,
                    device,
                    pattern_name=pattern_name,
                    pattern=None,
                    num_modalities=len(modalities),
                    random_p=float(p_missing),
                    prediction_index=prediction_index,
                    max_batches=max_batches,
                    cfg=cfg,
                )
            )
    if "avg_missing" not in {row.get("pattern") for row in results}:
        avg = _average_missing_results(results)
        if avg is not None:
            results.append(avg)
    return results


def _evaluate_pattern(
    model,
    dataloader,
    device: torch.device,
    *,
    pattern_name: str,
    pattern: list[int] | None,
    prediction_index: int | str,
    max_batches: int | None,
    cfg: dict[str, Any] | None,
    num_modalities: int | None = None,
    random_p: float | None = None,
) -> dict[str, Any]:
    accumulator = _Accumulator()
    mask_label = ",".join(str(int(value)) for value in pattern) if pattern is not None else f"random_{random_p:g}"
    for batch_index, raw_batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        batch_size = _batch_size(raw_batch)
        missing_mask = (
            make_fixed_missing_mask(batch_size, pattern, device=device)
            if pattern is not None
            else sample_eval_random_missing_mask(batch_size, int(num_modalities), float(random_p), device=device)
        )
        logits, target, diagnostics = _forward_batch(
            model,
            raw_batch,
            missing_mask,
            device,
            prediction_index=prediction_index,
            cfg=cfg,
        )
        metrics = {
            "loss": float(F.cross_entropy(logits, target).detach().cpu().item()),
            **topk_accuracy(logits, target, topk=(1, 3, 5)),
            **_beam_error_metrics(logits, target, cfg),
            **reliability_error_stats(
                logits,
                target,
                global_reliability=_diagnostic_tensor(diagnostics, "global_reliability"),
                modality_reliability=_diagnostic_tensor(diagnostics, "modality_reliability"),
                missing_mask=missing_mask,
            ),
            "ece": expected_calibration_error(logits, target),
        }
        accumulator.update(metrics, int(target.numel()))
    return {
        "pattern": pattern_name,
        "mask": mask_label,
        "num_samples": accumulator.num_samples,
        "sample_count": accumulator.num_samples,
        "count": accumulator.num_samples,
        **accumulator.mean(),
    }


def _forward_batch(
    model,
    raw_batch,
    missing_mask: torch.Tensor,
    device: torch.device,
    *,
    prediction_index: int | str,
    cfg: dict[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if cfg is not None:
        model_cfg = cfg["model"]["primary"]
        num_pred = int(model_cfg.get("num_pred", cfg.get("model", {}).get("num_pred", 1)))
        step = run_model_step(
            model,
            cfg.get("experiment", {}).get("task", "image"),
            raw_batch,
            model_cfg=model_cfg,
            seq_length=int(model_cfg.get("seq_length", cfg.get("model", {}).get("seq_length", 8))),
            num_pred=num_pred,
            downsample_ratio=int(model_cfg.get("downsample_ratio", cfg.get("model", {}).get("downsample_ratio", 1))),
            device=device,
            extra_model_kwargs={"missing_mask": missing_mask},
        )
        return (
            _select_prediction(step.logits, prediction_index),
            _select_target(step.labels, prediction_index),
            step.model_output.diagnostics,
        )

    batch = _move_batch(prepare_task_batch(raw_batch), device)
    output = _direct_model_call(model, batch, missing_mask)
    model_output = adapt_model_output(output)
    return (
        _select_prediction(model_output.logits, prediction_index),
        _select_target(_extract_target(batch), prediction_index),
        model_output.diagnostics,
    )


def _direct_model_call(model, batch: dict[str, Any], missing_mask: torch.Tensor):
    try:
        return model(batch, missing_mask=missing_mask)
    except TypeError:
        return model(missing_mask=missing_mask, **batch)


def _select_prediction(logits: torch.Tensor, prediction_index: int | str) -> torch.Tensor:
    if logits.ndim == 2:
        return logits
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, K] or [B, T, K], got {tuple(logits.shape)}.")
    index = _prediction_index(prediction_index, logits.shape[1])
    return logits[:, index, :]


def _select_target(target: torch.Tensor | None, prediction_index: int | str) -> torch.Tensor:
    if target is None:
        raise ValueError("target labels are required for evaluation.")
    target = target.to(dtype=torch.long)
    if target.ndim <= 1:
        return target.reshape(-1)
    index = _prediction_index(prediction_index, target.shape[1])
    return target[:, index].reshape(-1)


def _prediction_index(value: int | str, length: int) -> int:
    if value == "last":
        return length - 1
    if value == "first":
        return 0
    index = int(value)
    if index < 0:
        index += length
    if index < 0 or index >= length:
        raise ValueError(f"prediction_index {value!r} out of range for {length} predictions.")
    return index


def _extract_target(batch: dict[str, Any]) -> torch.Tensor:
    for key in ("target", "target_beam", "beam_label", "label", "labels", "y"):
        value = batch.get(key)
        if torch.is_tensor(value):
            return value
    raise ValueError("Batch must contain one of target, target_beam, beam_label, label, labels, or y.")


def _batch_size(batch: Any) -> int:
    if isinstance(batch, dict):
        for value in batch.values():
            if torch.is_tensor(value) and value.ndim > 0:
                return int(value.shape[0])
    if isinstance(batch, (list, tuple)) and batch:
        return _batch_size(batch[0])
    raise ValueError("Could not infer batch size from dataloader batch.")


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _diagnostic_tensor(diagnostics: dict[str, Any], key: str) -> torch.Tensor | None:
    value = diagnostics.get(key)
    return value if torch.is_tensor(value) else None


class _Accumulator:
    def __init__(self) -> None:
        self.num_samples = 0
        self.sums: dict[str, float] = {}

    def update(self, metrics: dict[str, float], count: int) -> None:
        self.num_samples += int(count)
        for key, value in metrics.items():
            if isinstance(value, float) and math.isfinite(value):
                self.sums[key] = self.sums.get(key, 0.0) + value * int(count)

    def mean(self) -> dict[str, float]:
        keys = [
            "loss",
            "top1",
            "top3",
            "top5",
            "adba",
            "mae",
            "mean_confidence",
            "mean_global_reliability",
            "mean_global_reliability_correct",
            "mean_global_reliability_wrong",
            "mean_modality_reliability",
            "mean_available_modality_reliability",
            "ece",
        ]
        return {
            key: (self.sums[key] / self.num_samples if self.num_samples and key in self.sums else math.nan)
            for key in keys
        }


def _beam_error_metrics(logits: torch.Tensor, target: torch.Tensor, cfg: dict[str, Any] | None) -> dict[str, float]:
    eval_cfg = (cfg or {}).get("evaluation", {}) if isinstance(cfg, dict) else {}
    summary = beam_classification_circular_summary(
        logits,
        target,
        num_beams=int(logits.shape[-1]),
        dba_delta=float(eval_cfg.get("dba_delta", 5)),
    )
    return {
        "adba": float(summary.get("DBA", math.nan)),
        "mae": float(summary.get("mean_circular_error", math.nan)),
    }


def _average_missing_results(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [
        row
        for row in results
        if str(row.get("pattern", "")) != "full"
        and (
            str(row.get("pattern", "")).startswith("missing_")
            or str(row.get("pattern", "")).startswith("only_")
            or str(row.get("pattern", "")) == "non_gps_only"
        )
    ]
    total = sum(int(row.get("num_samples", row.get("sample_count", 0)) or 0) for row in rows)
    if not rows or total <= 0:
        return None
    sample_count = max(int(row.get("num_samples", row.get("sample_count", 0)) or 0) for row in rows)
    averaged = {
        "pattern": "avg_missing",
        "mask": "aggregate",
        "num_samples": sample_count,
        "sample_count": sample_count,
        "count": sample_count,
    }
    for key in (
        "loss",
        "top1",
        "top3",
        "top5",
        "adba",
        "mae",
        "mean_confidence",
        "mean_global_reliability",
        "mean_global_reliability_correct",
        "mean_global_reliability_wrong",
        "mean_modality_reliability",
        "mean_available_modality_reliability",
        "ece",
    ):
        value = _weighted_mean(rows, key)
        if value is not None:
            averaged[key] = value
    return averaged


def _weighted_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    numerator = 0.0
    denominator = 0
    for row in rows:
        value = row.get(key)
        count = int(row.get("num_samples", row.get("sample_count", 0)) or 0)
        if count <= 0 or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        numerator += float(value) * count
        denominator += count
    return numerator / denominator if denominator else None
