from __future__ import annotations

from typing import Any, Mapping

import torch


def evaluate_jepa_msac_predictions(
    predictions: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    beam_power_reference: Any | None = None,
    representation_latents: Any | None = None,
    augmented_latents: Any | None = None,
) -> dict[str, Any]:
    summary = {
        "task_metrics": task_metric_summary(predictions, targets, beam_power_reference=beam_power_reference),
        "representation_metrics": representation_quality_summary(representation_latents, augmented_latents),
    }
    return summary


def task_metric_summary(
    predictions: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    beam_power_reference: Any | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "predicted_location" in predictions and "future_location" in targets:
        pred = _tensor(predictions["predicted_location"])
        target = _tensor(targets["future_location"], device=pred.device)
        error = torch.linalg.norm(pred - target, dim=-1)
        result["ADE"] = _metric(float(error.mean().item()), "min", "meters", int(error.numel()), "aggregate")
        result["FDE"] = _metric(float(error[:, -1].mean().item()), "min", "meters", int(error.shape[0]), "final")
        result["ADE_horizon"] = _horizon(error, "min", "meters")
    if "beam_logits" in predictions and "future_beam" in targets:
        logits = _tensor(predictions["beam_logits"])
        labels = _tensor(targets["future_beam"], device=logits.device).long()
        result["Top-1"] = _topk_metric(logits, labels, k=1)
        result["Top-3"] = _topk_metric(logits, labels, k=3)
        result["Top-1_horizon"] = _topk_horizon(logits, labels, k=1)
        result["Top-3_horizon"] = _topk_horizon(logits, labels, k=3)
        result["L1-RSRP diff"] = _rsrp_diff(logits, labels, beam_power_reference)
    if "scalar_rssi" in predictions and "future_rssi_scalar" in targets:
        pred = _tensor(predictions["scalar_rssi"])
        target = _tensor(targets["future_rssi_scalar"], device=pred.device)
        diff = pred - target
        result["RSSI RMSE"] = _metric(float(torch.sqrt((diff**2).mean()).item()), "min", "dB", int(diff.numel()), "aggregate")
        result["RSSI MAE"] = _metric(float(diff.abs().mean().item()), "min", "dB", int(diff.numel()), "aggregate")
        result["RSSI RMSE_horizon"] = _horizon(torch.sqrt((diff**2).mean(dim=0)), "min", "dB")
        result["RSSI MAE_horizon"] = _horizon(diff.abs().mean(dim=0), "min", "dB")
    return result


def representation_quality_summary(latents: Any | None, augmented_latents: Any | None = None) -> dict[str, Any]:
    if latents is None:
        return {
            "RRankMe": _unavailable("latent matrix unavailable"),
            "RLDA": _unavailable("latent matrix unavailable"),
        }
    matrix = _tensor(latents).reshape(-1, _tensor(latents).shape[-1]).float()
    rrankme = _rrankme(matrix)
    summary = {
        "RRankMe": _metric(rrankme, "max", "effective_rank", int(matrix.shape[0]), "aggregate"),
        "latent_dimension": int(matrix.shape[-1]),
        "sample_count": int(matrix.shape[0]),
        "stability": "svd_entropy_eps_1e-12",
    }
    if augmented_latents is None:
        summary["RLDA"] = _unavailable("augmentation view unavailable")
        summary["augmentation_count"] = 0
    else:
        aug = _tensor(augmented_latents, device=matrix.device).reshape(-1, matrix.shape[-1]).float()
        count = min(matrix.shape[0], aug.shape[0])
        distance = torch.linalg.norm(matrix[:count] - aug[:count], dim=-1)
        scale = torch.linalg.norm(matrix[:count], dim=-1).clamp_min(1e-12)
        value = float((1.0 / (1.0 + distance / scale)).mean().item())
        summary["RLDA"] = _metric(value, "max", "relative_similarity", int(count), "aggregate")
        summary["augmentation_count"] = int(aug.shape[0])
    return summary


def _metric(value: float, direction: str, unit: str, sample_count: int, horizon: str) -> dict[str, Any]:
    return {
        "available": True,
        "value": float(value),
        "direction": direction,
        "unit": unit,
        "sample_count": int(sample_count),
        "horizon_aggregation": horizon,
    }


def _unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "value": None, "reason": str(reason)}


def _tensor(value: Any, *, device: torch.device | None = None) -> torch.Tensor:
    tensor = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
    tensor = tensor.float() if not tensor.dtype.is_floating_point else tensor
    return tensor.to(device=device) if device is not None else tensor


def _topk_metric(logits: torch.Tensor, labels: torch.Tensor, *, k: int) -> dict[str, Any]:
    hits = logits.topk(min(int(k), logits.shape[-1]), dim=-1).indices.eq(labels.unsqueeze(-1)).any(dim=-1).float()
    return _metric(float(hits.mean().item()), "max", "accuracy", int(hits.numel()), "aggregate")


def _topk_horizon(logits: torch.Tensor, labels: torch.Tensor, *, k: int) -> list[dict[str, Any]]:
    hits = logits.topk(min(int(k), logits.shape[-1]), dim=-1).indices.eq(labels.unsqueeze(-1)).any(dim=-1).float()
    return [_metric(float(hits[:, idx].mean().item()), "max", "accuracy", int(hits.shape[0]), f"t+{idx + 1}") for idx in range(hits.shape[1])]


def _horizon(values: torch.Tensor, direction: str, unit: str) -> list[dict[str, Any]]:
    if values.ndim == 2:
        values = values.mean(dim=0)
    return [_metric(float(values[idx].item()), direction, unit, 1, f"t+{idx + 1}") for idx in range(values.shape[0])]


def _rsrp_diff(logits: torch.Tensor, labels: torch.Tensor, beam_power_reference: Any | None) -> dict[str, Any]:
    if beam_power_reference is None:
        return _unavailable("beam-power reference unavailable")
    reference = _tensor(beam_power_reference, device=logits.device)
    pred_beam = logits.argmax(dim=-1)
    pred_power = reference.gather(-1, pred_beam.unsqueeze(-1)).squeeze(-1)
    target_power = reference.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return _metric(float((pred_power - target_power).abs().mean().item()), "min", "dB", int(labels.numel()), "aggregate")


def _rrankme(matrix: torch.Tensor) -> float:
    if matrix.numel() == 0:
        return 0.0
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    probs = singular_values / singular_values.sum().clamp_min(1e-12)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    return float(torch.exp(entropy).item())


__all__ = [
    "evaluate_jepa_msac_predictions",
    "representation_quality_summary",
    "task_metric_summary",
]
