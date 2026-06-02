from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from kd_sensing.baselines.gps_window.types import GpsWindowPrediction, GpsWindowSample


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return target


def write_predictions_csv(
    path: str | Path,
    samples: list[GpsWindowSample],
    predictions: list[GpsWindowPrediction],
    *,
    labels: torch.Tensor,
    top_k: int = 5,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sample_id",
            "scenario",
            "split",
            "horizon",
            "true_beam",
            "predicted_beam",
            "topk_predicted_beams",
            "center_beam",
            "gps_coverage",
            "fallback_status",
            "score_max",
            "score_margin",
            "diagnostics_json",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample, pred, label_row in zip(samples, predictions, labels.detach().cpu().tolist()):
            scores = pred.scores.detach().cpu()
            top_values = torch.topk(scores, k=min(int(top_k), scores.shape[-1]), dim=-1).indices.tolist()
            for horizon_idx, truth in enumerate(label_row):
                row_scores = scores[horizon_idx]
                top2 = torch.topk(row_scores, k=min(2, row_scores.numel())).values
                margin = float(top2[0] - top2[1]) if top2.numel() > 1 else 0.0
                writer.writerow(
                    {
                        "sample_id": sample.sample_id,
                        "scenario": sample.scenario,
                        "split": sample.split,
                        "horizon": horizon_idx + 1,
                        "true_beam": int(truth),
                        "predicted_beam": int(torch.argmax(row_scores).item()),
                        "topk_predicted_beams": json.dumps(top_values[horizon_idx]),
                        "center_beam": int(pred.center_beams[horizon_idx]),
                        "gps_coverage": float(pred.gps_coverage),
                        "fallback_status": pred.fallback_status,
                        "score_max": float(row_scores.max().item()),
                        "score_margin": margin,
                        "diagnostics_json": json.dumps(_json_safe(pred.diagnostics), sort_keys=True),
                    }
                )
    return target


def prediction_histogram(predicted: torch.Tensor, labels: torch.Tensor, *, num_classes: int) -> dict[str, Any]:
    pred = predicted.detach().cpu().reshape(-1).to(torch.long)
    truth = labels.detach().cpu().reshape(-1).to(torch.long)
    valid = truth.ge(0)
    pred_hist = torch.bincount(pred[valid], minlength=int(num_classes)).tolist() if torch.any(valid) else [0] * int(num_classes)
    true_hist = torch.bincount(truth[valid], minlength=int(num_classes)).tolist() if torch.any(valid) else [0] * int(num_classes)
    return {
        "predicted_beam_histogram": [int(item) for item in pred_hist],
        "true_beam_histogram": [int(item) for item in true_hist],
        "total": int(valid.sum().item()),
    }


def collapse_diagnostics(predicted: torch.Tensor, labels: torch.Tensor, *, num_classes: int) -> dict[str, Any]:
    hist = prediction_histogram(predicted, labels, num_classes=num_classes)
    total = max(int(hist["total"]), 1)
    max_count = max(hist["predicted_beam_histogram"]) if hist["predicted_beam_histogram"] else 0
    active = sum(1 for item in hist["predicted_beam_histogram"] if int(item) > 0)
    return {
        "top_predicted_fraction": float(max_count / total),
        "active_predicted_beams": int(active),
        "collapsed": bool(max_count / total >= 0.5),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value

