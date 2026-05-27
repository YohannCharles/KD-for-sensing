from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import torch

from kd_sensing.engine.hist_beam_labels import hist_beam_labels


def calculate_hist_beam_metrics(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    group_size: int = 8,
    num_classes: int | None = None,
) -> dict[str, Any]:
    if outputs.ndim == 2:
        outputs = outputs.unsqueeze(1)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    classes = int(num_classes or outputs.shape[-1])
    coarse_true, fine_true = hist_beam_labels(labels, num_classes=classes, group_size=group_size)
    pred = torch.argmax(outputs, dim=-1)
    coarse_pred, fine_pred = hist_beam_labels(pred, num_classes=classes, group_size=group_size)
    valid = labels.ne(-100)
    total = int(valid.sum().item())
    if total == 0:
        return {
            "coarse_accuracy": 0.0,
            "fine_offset_accuracy": 0.0,
            "hist_total": 0,
        }
    coarse_acc = (coarse_true[valid] == coarse_pred[valid]).float().mean().item()
    fine_acc = (fine_true[valid] == fine_pred[valid]).float().mean().item()
    return {
        "coarse_accuracy": float(coarse_acc),
        "fine_offset_accuracy": float(fine_acc),
        "hist_total": total,
    }


def beam_power_metrics(
    predicted_beams: torch.Tensor,
    true_beams: torch.Tensor,
    power_vectors: torch.Tensor | None,
) -> dict[str, Any]:
    if power_vectors is None:
        return {
            "power_metrics_available": False,
            "power_metrics_unavailable_reason": "beam_power_vector_missing",
        }
    if power_vectors.ndim != 2:
        raise ValueError(f"power_vectors must have shape [N, C], got {tuple(power_vectors.shape)}.")
    pred = predicted_beams.detach().cpu().reshape(-1).to(torch.long)
    truth = true_beams.detach().cpu().reshape(-1).to(torch.long)
    power = power_vectors.detach().cpu().to(torch.float32)
    valid = truth.ge(0) & truth.lt(power.shape[-1]) & pred.ge(0) & pred.lt(power.shape[-1])
    if not torch.any(valid):
        return {
            "power_metrics_available": False,
            "power_metrics_unavailable_reason": "no_valid_beam_indices",
        }
    chosen = power[torch.arange(power.shape[0])[valid], pred[valid]]
    optimal = power[torch.arange(power.shape[0])[valid], truth[valid]]
    eps = 1e-8
    normalized = (chosen + eps) / (optimal + eps)
    loss_db = -10.0 * torch.log10(normalized.clamp_min(eps))
    return {
        "power_metrics_available": True,
        "normalized_received_power": float(normalized.mean().item()),
        "beam_power_loss_db": float(loss_db.mean().item()),
    }


def write_hist_beam_predictions(
    path: str | Path,
    outputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    metadata: Iterable[dict[str, Any]] | None = None,
    group_size: int = 8,
    top_k: int = 5,
    variant_metadata: dict[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if outputs.ndim == 2:
        outputs = outputs.unsqueeze(1)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    top_values = torch.topk(outputs, k=min(int(top_k), outputs.shape[-1]), dim=-1).indices.detach().cpu()
    pred = torch.argmax(outputs, dim=-1).detach().cpu()
    labels_cpu = labels.detach().cpu()
    coarse_true, fine_true = hist_beam_labels(labels_cpu, num_classes=outputs.shape[-1], group_size=group_size)
    coarse_pred, fine_pred = hist_beam_labels(pred, num_classes=outputs.shape[-1], group_size=group_size)
    metadata_list = list(metadata or [{} for _ in range(labels_cpu.shape[0])])
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "scene",
                "horizon",
                "true_beam",
                "pred_beam",
                "predicted_beam",
                "topk_predictions",
                "coarse_true",
                "coarse_pred",
                "fine_true",
                "fine_pred",
                "split",
                "variant_metadata",
            ],
        )
        writer.writeheader()
        for row_idx in range(labels_cpu.shape[0]):
            meta = metadata_list[row_idx] if row_idx < len(metadata_list) else {}
            for horizon in range(labels_cpu.shape[1]):
                writer.writerow(
                    {
                        "sample_id": meta.get("sample_id", f"sample_{row_idx}"),
                        "scene": meta.get("scene_slug", meta.get("scene_id")),
                        "horizon": horizon,
                        "true_beam": int(labels_cpu[row_idx, horizon].item()),
                        "pred_beam": int(pred[row_idx, horizon].item()),
                        "predicted_beam": int(pred[row_idx, horizon].item()),
                        "topk_predictions": json.dumps(
                            [int(item) for item in top_values[row_idx, horizon].tolist()]
                        ),
                        "coarse_true": int(coarse_true[row_idx, horizon].item()),
                        "coarse_pred": int(coarse_pred[row_idx, horizon].item()),
                        "fine_true": int(fine_true[row_idx, horizon].item()),
                        "fine_pred": int(fine_pred[row_idx, horizon].item()),
                        "split": meta.get("split", (variant_metadata or {}).get("split", "test")),
                        "variant_metadata": json.dumps(variant_metadata or {}, sort_keys=True),
                    }
                )
    return target


def summarize_loso_runs(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record.get("summary_type", "source_only"),
            record.get("fold"),
            record.get("target_scene"),
            record.get("variant"),
            record.get("budget"),
        )
        grouped[key].append(record)
    summaries = []
    for (summary_type, fold, target_scene, variant, budget), items in grouped.items():
        metric_names = sorted(
            {
                name
                for item in items
                for name, value in item.get("metrics", {}).items()
                if isinstance(value, (int, float))
            }
        )
        row = {
            "summary_type": summary_type,
            "fold": fold,
            "target_scene": target_scene,
            "variant": variant,
            "budget": budget,
            "run_count": len(items),
            "run_paths": [item.get("run_path") for item in items],
            "seeds": [item.get("seed") for item in items],
        }
        for metric in metric_names:
            values = [float(item["metrics"][metric]) for item in items if metric in item.get("metrics", {})]
            if values:
                row[f"{metric}_mean"] = float(mean(values))
        summaries.append(row)
    return {"rows": summaries, "run_count": len(records)}


def write_loso_summary(path: str | Path, records: list[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = summarize_loso_runs(records)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return target


__all__ = [
    "beam_power_metrics",
    "calculate_hist_beam_metrics",
    "summarize_loso_runs",
    "write_hist_beam_predictions",
    "write_loso_summary",
]
