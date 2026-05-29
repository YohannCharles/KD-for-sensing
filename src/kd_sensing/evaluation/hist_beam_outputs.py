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
    angular_error = torch.abs(pred[valid].to(torch.float32) - labels[valid].to(torch.float32)).mean().item()
    return {
        "coarse_accuracy": float(coarse_acc),
        "fine_offset_accuracy": float(fine_acc),
        "mean_angular_error": float(angular_error),
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
        "power_metrics_unavailable_reason": None,
        "normalized_received_power": float(normalized.mean().item()),
        "beam_power_loss_db": float(loss_db.mean().item()),
    }


def radio_semantic_metrics(
    radio_logits: torch.Tensor | None,
    radio_labels: torch.Tensor | None,
    *,
    ignore_index: int = -100,
) -> dict[str, Any]:
    if radio_logits is None:
        return {
            "radio_metrics_available": False,
            "radio_metrics_unavailable_reason": "radio_logits_missing",
        }
    if radio_labels is None:
        return {
            "radio_metrics_available": False,
            "radio_metrics_unavailable_reason": "radio_semantic_label_missing",
        }
    if radio_logits.ndim == 2:
        radio_logits = radio_logits.unsqueeze(1)
    if radio_labels.ndim == 1:
        radio_labels = radio_labels.unsqueeze(1)
    valid = radio_labels.ne(ignore_index) & radio_labels.ge(0) & radio_labels.lt(radio_logits.shape[-1])
    if not torch.any(valid):
        return {
            "radio_metrics_available": False,
            "radio_metrics_unavailable_reason": "radio_semantic_label_missing",
            "radio_semantic_coverage": 0.0,
        }
    pred = radio_logits.argmax(dim=-1)
    accuracy = (pred[valid] == radio_labels[valid]).float().mean()
    return {
        "radio_metrics_available": True,
        "radio_metrics_unavailable_reason": None,
        "radio_semantic_accuracy": float(accuracy.item()),
        "radio_semantic_coverage": float(valid.float().mean().item()),
        "radio_semantic_total": int(valid.sum().item()),
    }


def path_semantic_metrics(
    path_logits: torch.Tensor | None,
    path_labels: torch.Tensor | None,
    *,
    path_assignment: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> dict[str, Any]:
    if path_logits is None and path_assignment is None:
        return {
            "path_metrics_available": False,
            "path_metrics_unavailable_reason": "path_logits_missing",
        }
    if path_labels is None:
        return {
            "path_metrics_available": False,
            "path_metrics_unavailable_reason": "path_semantic_label_missing",
        }
    scores = path_logits if path_logits is not None else path_assignment
    assert scores is not None
    if scores.ndim == 2:
        scores = scores.unsqueeze(1)
    if path_labels.ndim == 1:
        path_labels = path_labels.unsqueeze(1)
    valid = path_labels.ne(ignore_index) & path_labels.ge(0) & path_labels.lt(scores.shape[-1])
    if not torch.any(valid):
        return {
            "path_metrics_available": False,
            "path_metrics_unavailable_reason": "path_semantic_label_missing",
            "path_semantic_coverage": 0.0,
        }
    pred = scores.argmax(dim=-1)
    confidence = torch.softmax(scores, dim=-1).max(dim=-1).values if path_logits is not None else scores.max(dim=-1).values
    accuracy = (pred[valid] == path_labels[valid]).float().mean()
    histogram = torch.bincount(path_labels[valid].reshape(-1).to(torch.long), minlength=int(scores.shape[-1]))
    return {
        "path_metrics_available": True,
        "path_metrics_unavailable_reason": None,
        "path_semantic_accuracy": float(accuracy.item()),
        "path_semantic_coverage": float(valid.float().mean().item()),
        "path_semantic_total": int(valid.sum().item()),
        "prototype_assignment_confidence": float(confidence[valid].float().mean().item()),
        "prototype_coverage_per_class": [int(item) for item in histogram.tolist()],
        "source_target_path_class_histogram": [int(item) for item in histogram.tolist()],
    }


def path_descriptor_regression_metrics(
    path_attr_pred: torch.Tensor | None,
    path_descriptors: torch.Tensor | None,
    path_valid: torch.Tensor | None = None,
) -> dict[str, Any]:
    if path_attr_pred is None:
        return {
            "path_descriptor_metrics_available": False,
            "path_descriptor_unavailable_reason": "path_attr_pred_missing",
        }
    if path_descriptors is None:
        return {
            "path_descriptor_metrics_available": False,
            "path_descriptor_unavailable_reason": "path_descriptor_missing",
        }
    pred = path_attr_pred.detach().cpu().to(torch.float32)
    target = path_descriptors.detach().cpu().to(torch.float32)
    if pred.ndim == 2:
        pred = pred.unsqueeze(1)
    if target.ndim == 2:
        target = target.unsqueeze(1)
    if pred.shape != target.shape:
        return {
            "path_descriptor_metrics_available": False,
            "path_descriptor_unavailable_reason": f"path_descriptor_shape_mismatch:{tuple(pred.shape)}!={tuple(target.shape)}",
        }
    if path_valid is None:
        valid = torch.isfinite(target).all(dim=-1)
    else:
        valid = path_valid.detach().cpu().to(torch.bool)
        if valid.ndim == 1:
            valid = valid.unsqueeze(1)
    if not torch.any(valid):
        return {
            "path_descriptor_metrics_available": False,
            "path_descriptor_unavailable_reason": "path_descriptor_missing",
            "path_descriptor_coverage": 0.0,
        }
    mse = torch.mean((pred[valid] - target[valid]) ** 2)
    return {
        "path_descriptor_metrics_available": True,
        "path_descriptor_unavailable_reason": None,
        "path_descriptor_regression_mse": float(mse.item()),
        "path_descriptor_coverage": float(valid.float().mean().item()),
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
    radio_logits: torch.Tensor | None = None,
    radio_labels: torch.Tensor | None = None,
    path_logits: torch.Tensor | None = None,
    path_labels: torch.Tensor | None = None,
    path_assignment: torch.Tensor | None = None,
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
    radio_pred = None
    if radio_logits is not None:
        if radio_logits.ndim == 2:
            radio_logits = radio_logits.unsqueeze(1)
        radio_pred = radio_logits.detach().cpu().argmax(dim=-1)
    radio_labels_cpu = None
    if radio_labels is not None:
        radio_labels_cpu = radio_labels.detach().cpu()
        if radio_labels_cpu.ndim == 1:
            radio_labels_cpu = radio_labels_cpu.unsqueeze(1)
    path_scores = path_logits if path_logits is not None else path_assignment
    path_pred = None
    if path_scores is not None:
        if path_scores.ndim == 2:
            path_scores = path_scores.unsqueeze(1)
        path_pred = path_scores.detach().cpu().argmax(dim=-1)
    path_labels_cpu = None
    if path_labels is not None:
        path_labels_cpu = path_labels.detach().cpu()
        if path_labels_cpu.ndim == 1:
            path_labels_cpu = path_labels_cpu.unsqueeze(1)
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
                "radio_true",
                "radio_pred",
                "radio_unavailable_reason",
                "path_true",
                "path_pred",
                "path_unavailable_reason",
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
                        "radio_true": _radio_value(radio_labels_cpu, row_idx, horizon),
                        "radio_pred": _radio_value(radio_pred, row_idx, horizon),
                        "radio_unavailable_reason": _radio_unavailable_reason(meta, radio_labels_cpu, row_idx, horizon),
                        "path_true": _radio_value(path_labels_cpu, row_idx, horizon),
                        "path_pred": _radio_value(path_pred, row_idx, horizon),
                        "path_unavailable_reason": _path_unavailable_reason(meta, path_labels_cpu, row_idx, horizon),
                        "fine_true": int(fine_true[row_idx, horizon].item()),
                        "fine_pred": int(fine_pred[row_idx, horizon].item()),
                        "split": meta.get("split", (variant_metadata or {}).get("split", "test")),
                        "variant_metadata": json.dumps(variant_metadata or {}, sort_keys=True),
                    }
                )
    return target


def _radio_value(values: torch.Tensor | None, row_idx: int, horizon: int) -> int | str:
    if values is None or row_idx >= values.shape[0] or horizon >= values.shape[1]:
        return ""
    value = int(values[row_idx, horizon].item())
    return "" if value < 0 else value


def _radio_unavailable_reason(meta: dict[str, Any], labels: torch.Tensor | None, row_idx: int, horizon: int) -> str:
    if labels is not None and row_idx < labels.shape[0] and horizon < labels.shape[1] and int(labels[row_idx, horizon].item()) >= 0:
        return ""
    reason = meta.get("radio_unavailable_reason") or meta.get("radio_semantic_unavailable_reason")
    if isinstance(reason, (list, tuple)):
        return str(reason[horizon]) if horizon < len(reason) else ""
    return str(reason or "radio_semantic_label_missing")


def _path_unavailable_reason(meta: dict[str, Any], labels: torch.Tensor | None, row_idx: int, horizon: int) -> str:
    if labels is not None and row_idx < labels.shape[0] and horizon < labels.shape[1] and int(labels[row_idx, horizon].item()) >= 0:
        return ""
    reason = meta.get("path_unavailable_reason") or meta.get("path_semantic_unavailable_reason")
    if isinstance(reason, (list, tuple)):
        return str(reason[horizon]) if horizon < len(reason) else ""
    return str(reason or "path_semantic_label_missing")


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
    "path_descriptor_regression_metrics",
    "path_semantic_metrics",
    "radio_semantic_metrics",
    "summarize_loso_runs",
    "write_hist_beam_predictions",
    "write_loso_summary",
]
