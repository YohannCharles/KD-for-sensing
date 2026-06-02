from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import torch

from kd_sensing.engine.hist_beam_labels import hist_beam_labels
from kd_sensing.evaluation.hist_beam_residuals import residual_topk_to_absolute


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
    last_beams: torch.Tensor | None = None,
    residual_logits: torch.Tensor | None = None,
    residual_labels: torch.Tensor | None = None,
    shared_logits: torch.Tensor | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if outputs.ndim == 2:
        outputs = outputs.unsqueeze(1)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    top_values = torch.topk(outputs, k=min(int(top_k), outputs.shape[-1]), dim=-1).indices.detach().cpu()
    pred = torch.argmax(outputs, dim=-1).detach().cpu()
    shared_pred = None
    shared_top_values = None
    if shared_logits is not None:
        if shared_logits.ndim == 2:
            shared_logits = shared_logits.unsqueeze(1)
        shared_top_values = torch.topk(shared_logits, k=min(int(top_k), shared_logits.shape[-1]), dim=-1).indices.detach().cpu()
        shared_pred = torch.argmax(shared_logits, dim=-1).detach().cpu()
    labels_cpu = labels.detach().cpu()
    last_beams_cpu = _ensure_prediction_horizon(last_beams.detach().cpu(), labels_cpu.shape[1]) if last_beams is not None else None
    residual_labels_cpu = (
        _ensure_prediction_horizon(residual_labels.detach().cpu(), labels_cpu.shape[1])
        if residual_labels is not None
        else None
    )
    residual_pred = None
    residual_topk = None
    residual_absolute_topk = None
    if residual_logits is not None and last_beams is not None:
        residual_logits_t = residual_logits.detach().cpu()
        if residual_logits_t.ndim == 2:
            residual_logits_t = residual_logits_t.unsqueeze(1)
        residual_pred = residual_logits_t.argmax(dim=-1)
        residual_top = residual_topk_to_absolute(residual_logits_t, last_beams_cpu, k=top_k)
        residual_topk = residual_top["residual_topk"]
        residual_absolute_topk = residual_top["absolute_topk"]
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
                "final_predicted_beam",
                "shared_predicted_beam",
                "final_topk",
                "shared_topk",
                "last_beam",
                "true_residual",
                "pred_residual",
                "topk_residual",
                "topk_reconstructed_beam",
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
                        "final_predicted_beam": int(pred[row_idx, horizon].item()),
                        "shared_predicted_beam": _optional_tensor_value(shared_pred, row_idx, horizon),
                        "final_topk": json.dumps([int(item) for item in top_values[row_idx, horizon].tolist()]),
                        "shared_topk": _optional_tensor_list(shared_top_values, row_idx, horizon),
                        "last_beam": _optional_tensor_value(last_beams_cpu, row_idx, horizon),
                        "true_residual": _optional_tensor_value(residual_labels_cpu, row_idx, horizon),
                        "pred_residual": _optional_tensor_value(residual_pred, row_idx, horizon),
                        "topk_residual": _optional_tensor_list(residual_topk, row_idx, horizon),
                        "topk_reconstructed_beam": _optional_tensor_list(residual_absolute_topk, row_idx, horizon),
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


def markov_delta_baseline_metrics(
    train_last_beams: torch.Tensor | None,
    train_labels: torch.Tensor | None,
    eval_last_beams: torch.Tensor | None,
    eval_labels: torch.Tensor,
    *,
    num_classes: int,
    k_values: Iterable[int] = (1, 3, 5),
    smoothing: float = 1.0,
    train_split: str = "source_train",
) -> dict[str, Any]:
    if train_last_beams is None or train_labels is None or eval_last_beams is None:
        return {
            "markov_delta_baseline_available": False,
            "markov_delta_unavailable_reason": "history_or_training_labels_missing",
        }
    classes = int(num_classes)
    train_last = _ensure_prediction_horizon(train_last_beams.detach().cpu(), _horizon(train_labels))
    train_future = _ensure_prediction_horizon(train_labels.detach().cpu(), train_last.shape[1])
    eval_last = _ensure_prediction_horizon(eval_last_beams.detach().cpu(), _horizon(eval_labels))
    eval_future = _ensure_prediction_horizon(eval_labels.detach().cpu(), eval_last.shape[1])
    train_valid = train_future.ge(0) & train_future.lt(classes) & train_last.ge(0) & train_last.lt(classes)
    eval_valid = eval_future.ge(0) & eval_future.lt(classes) & eval_last.ge(0) & eval_last.lt(classes)
    if not torch.any(train_valid) or not torch.any(eval_valid):
        return {
            "markov_delta_baseline_available": False,
            "markov_delta_unavailable_reason": "no_valid_markov_samples",
            "markov_delta_train_split": train_split,
            "markov_delta_smoothing": float(smoothing),
        }
    delta = (train_future[train_valid] - train_last[train_valid]).remainder(classes)
    counts = torch.bincount(delta.reshape(-1).to(torch.long), minlength=classes).to(torch.float32)
    probs = counts + float(smoothing)
    order = torch.argsort(probs, descending=True)
    predictions = (eval_last.unsqueeze(-1) + order.view(1, 1, classes)).remainder(classes)
    result: dict[str, Any] = {
        "markov_delta_baseline_available": True,
        "markov_delta_unavailable_reason": None,
        "markov_delta_train_split": train_split,
        "markov_delta_train_samples": int(train_valid.sum().item()),
        "markov_delta_eval_samples": int(eval_valid.sum().item()),
        "markov_delta_smoothing": float(smoothing),
        "markov_delta_histogram": [int(item) for item in counts.to(torch.long).tolist()],
        "markov_delta_top_residuals": [int(item) for item in order[: min(5, classes)].tolist()],
    }
    for k in k_values:
        kk = min(int(k), classes)
        hit = predictions[:, :, :kk].eq(eval_future.unsqueeze(-1)).any(dim=-1) & eval_valid
        result[f"markov_delta_top{k}"] = float(hit[eval_valid].float().mean().item())
    return result


def beam_histogram_metrics(
    labels: torch.Tensor,
    outputs: torch.Tensor | None = None,
    *,
    num_classes: int,
    prefix: str = "target_test",
) -> dict[str, Any]:
    label_hist = _beam_histogram(labels, num_classes=num_classes)
    result: dict[str, Any] = {f"{prefix}_true_beam_histogram": label_hist}
    if outputs is not None:
        pred = outputs.argmax(dim=-1) if outputs.ndim >= 2 else outputs
        result[f"{prefix}_predicted_beam_histogram"] = _beam_histogram(pred, num_classes=num_classes)
    return result


def prediction_histogram_payload(
    labels: torch.Tensor,
    outputs: torch.Tensor,
    *,
    num_classes: int,
    top_k: int = 8,
) -> dict[str, Any]:
    if outputs.ndim == 2:
        outputs = outputs.unsqueeze(1)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    pred = outputs.argmax(dim=-1)
    valid = labels.ne(-100) & labels.ge(0) & labels.lt(int(num_classes))
    true_hist = _beam_histogram(labels, num_classes=num_classes)
    pred_hist = _beam_histogram(torch.where(valid, pred, torch.full_like(pred, -100)), num_classes=num_classes)
    if torch.any(valid):
        error = torch.abs(pred[valid].to(torch.float32) - labels[valid].to(torch.float32))
        mean_abs = float(error.mean().item())
        within_1 = float(error.le(1).float().mean().item())
        within_2 = float(error.le(2).float().mean().item())
        within_3 = float(error.le(3).float().mean().item())
        top_counts = torch.bincount(pred[valid].detach().cpu().reshape(-1).to(torch.long), minlength=int(num_classes))
        total_pred = float(top_counts.sum().item())
        top_values = torch.topk(top_counts, k=min(5, int(num_classes))).values.to(torch.float32)
    else:
        mean_abs = 0.0
        within_1 = 0.0
        within_2 = 0.0
        within_3 = 0.0
        total_pred = 0.0
        top_values = torch.zeros(min(5, int(num_classes)), dtype=torch.float32)
    unique_pred = int(sum(1 for value in pred_hist if int(value) > 0))
    ratio = lambda n: float(top_values[: min(n, top_values.numel())].sum().item() / max(total_pred, 1.0))
    return {
        "true_hist": true_hist,
        "pred_hist": pred_hist,
        "true_top_beams": _top_hist_beams(true_hist, top_k=top_k),
        "pred_top_beams": _top_hist_beams(pred_hist, top_k=top_k),
        "unique_pred_beams": unique_pred,
        "top1_pred_beam_ratio": ratio(1),
        "top2_pred_beam_ratio": ratio(2),
        "top5_pred_beam_ratio": ratio(5),
        "mean_abs_beam_error": mean_abs,
        "within_1_acc": within_1,
        "within_2_acc": within_2,
        "within_3_acc": within_3,
        "valid_count": int(valid.sum().item()),
    }


def histogram_kl(lhs_hist: Iterable[float] | torch.Tensor, rhs_hist: Iterable[float] | torch.Tensor, *, eps: float = 1e-12) -> float:
    lhs = torch.as_tensor(list(lhs_hist) if not torch.is_tensor(lhs_hist) else lhs_hist, dtype=torch.float64).reshape(-1)
    rhs = torch.as_tensor(list(rhs_hist) if not torch.is_tensor(rhs_hist) else rhs_hist, dtype=torch.float64).reshape(-1)
    if lhs.numel() != rhs.numel():
        raise ValueError(f"histogram sizes must match, got {lhs.numel()} and {rhs.numel()}.")
    if lhs.numel() == 0:
        return 0.0
    lhs = lhs.clamp_min(0)
    rhs = rhs.clamp_min(0)
    lhs = lhs + float(eps)
    rhs = rhs + float(eps)
    lhs = lhs / lhs.sum().clamp_min(float(eps))
    rhs = rhs / rhs.sum().clamp_min(float(eps))
    return float(torch.sum(lhs * (torch.log(lhs) - torch.log(rhs))).item())


def collapse_diagnostics_payload(
    labels: torch.Tensor,
    outputs: torch.Tensor,
    *,
    num_classes: int,
    support_prior: Iterable[float] | torch.Tensor | None = None,
    target_logits: torch.Tensor | None = None,
    target_prior_bias: torch.Tensor | None = None,
    prototype_logits: torch.Tensor | None = None,
    beta_prior_initial: float | None = None,
    beta_prior_final: float | None = None,
    beta_prior_effective: float | None = None,
    prototype_metadata: dict[str, Any] | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    final_payload = prediction_histogram_payload(labels, outputs, num_classes=num_classes, top_k=top_k)
    true_hist = final_payload["true_hist"]
    pred_hist = final_payload["pred_hist"]
    if support_prior is None and target_prior_bias is not None:
        bias = target_prior_bias[0, 0, :] if target_prior_bias.ndim == 3 else target_prior_bias.reshape(-1)
        support_prior = torch.softmax(bias.detach().cpu().to(torch.float32), dim=-1)
    support = (
        [float(item) for item in torch.as_tensor(support_prior, dtype=torch.float32).reshape(-1).tolist()]
        if support_prior is not None
        else [1.0 / max(int(num_classes), 1) for _ in range(int(num_classes))]
    )
    payload: dict[str, Any] = {
        "support_prior_hist": support,
        "true_hist": true_hist,
        "pred_hist": pred_hist,
        "true_top_beams": final_payload["true_top_beams"],
        "pred_top_beams": final_payload["pred_top_beams"],
        "unique_pred_beams": int(sum(1 for value in pred_hist if int(value) > 0)),
        "kl_pred_support": histogram_kl(pred_hist, support),
        "kl_true_support": histogram_kl(true_hist, support),
        "kl_pred_true": histogram_kl(pred_hist, true_hist),
        "mean_abs_beam_error": final_payload["mean_abs_beam_error"],
        "within_3_acc": final_payload["within_3_acc"],
        "beta_prior_initial": beta_prior_initial,
        "beta_prior_final": beta_prior_final,
        "beta_prior_effective": beta_prior_effective,
        "per_true_beam_confusion": _per_true_beam_confusion(labels, outputs, num_classes=num_classes, top_k=top_k),
        "branches": {
            "final": _branch_metrics(outputs, labels, num_classes=num_classes, top_k=top_k),
        },
        "prototype": dict(prototype_metadata or {}),
        "evaluation_only_target_test_label": True,
    }
    if target_logits is not None:
        payload["branches"]["target_logits_only"] = _branch_metrics(target_logits, labels, num_classes=num_classes, top_k=top_k)
    if target_prior_bias is not None:
        payload["branches"]["prior_only"] = _branch_metrics(target_prior_bias, labels, num_classes=num_classes, top_k=top_k)
    if target_logits is not None and target_prior_bias is not None:
        payload["branches"]["target_logits_plus_prior"] = _branch_metrics(target_logits + target_prior_bias, labels, num_classes=num_classes, top_k=top_k)
    if prototype_logits is not None:
        payload["branches"]["prototype_only"] = _branch_metrics(prototype_logits, labels, num_classes=num_classes, top_k=top_k)
        if target_logits is not None and target_prior_bias is not None:
            payload["branches"]["target_prior_plus_prototype"] = _branch_metrics(
                target_logits + target_prior_bias + prototype_logits,
                labels,
                num_classes=num_classes,
                top_k=top_k,
            )
            payload["[v9-sector] top predicted beams before proto"] = payload["branches"][
                "target_logits_plus_prior"
            ]["pred_top_beams"]
            payload["[v9-sector] top predicted beams after proto"] = payload["branches"][
                "target_prior_plus_prototype"
            ]["pred_top_beams"]
    return payload


def write_collapse_diagnostics(
    path: str | Path,
    labels: torch.Tensor,
    outputs: torch.Tensor,
    *,
    num_classes: int,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = collapse_diagnostics_payload(labels, outputs, num_classes=num_classes, **kwargs)
    if metadata:
        payload["metadata"] = dict(metadata)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return target


def write_prediction_histogram(
    path: str | Path,
    labels: torch.Tensor,
    outputs: torch.Tensor,
    *,
    num_classes: int,
    top_k: int = 8,
    metadata: dict[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = prediction_histogram_payload(labels, outputs, num_classes=num_classes, top_k=top_k)
    if metadata:
        payload["metadata"] = dict(metadata)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return target


def write_confusion_by_true_beam(
    path: str | Path,
    labels: torch.Tensor,
    outputs: torch.Tensor,
    *,
    num_classes: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if outputs.ndim == 2:
        outputs = outputs.unsqueeze(1)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    pred = outputs.argmax(dim=-1)
    valid = labels.ne(-100) & labels.ge(0) & labels.lt(int(num_classes)) & pred.ge(0) & pred.lt(int(num_classes))
    confusion: dict[str, dict[str, int]] = {}
    if torch.any(valid):
        true_values = labels[valid].detach().cpu().to(torch.long)
        pred_values = pred[valid].detach().cpu().to(torch.long)
        for true_beam, pred_beam in zip(true_values.tolist(), pred_values.tolist()):
            true_key = str(int(true_beam))
            pred_key = str(int(pred_beam))
            bucket = confusion.setdefault(true_key, {})
            bucket[pred_key] = int(bucket.get(pred_key, 0) + 1)
    payload: dict[str, Any] = {"confusion_by_true_beam": confusion}
    if metadata:
        payload["metadata"] = dict(metadata)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return target


def source_prior_collapse_diagnostics(
    *,
    source_histogram: Iterable[int] | None,
    target_true_histogram: Iterable[int] | None,
    predicted_histogram: Iterable[int] | None,
    top_fraction_threshold: float = 0.5,
) -> dict[str, Any]:
    if source_histogram is None or target_true_histogram is None or predicted_histogram is None:
        return {
            "source_prior_collapse_available": False,
            "source_prior_collapse": False,
            "source_prior_collapse_unavailable_reason": "histogram_missing",
        }
    source = torch.tensor(list(source_histogram), dtype=torch.float32)
    target = torch.tensor(list(target_true_histogram), dtype=torch.float32)
    pred = torch.tensor(list(predicted_histogram), dtype=torch.float32)
    if source.numel() == 0 or target.numel() == 0 or pred.numel() == 0 or float(pred.sum().item()) <= 0:
        return {
            "source_prior_collapse_available": False,
            "source_prior_collapse": False,
            "source_prior_collapse_unavailable_reason": "empty_histogram",
        }
    source_top = int(source.argmax().item())
    target_top = int(target.argmax().item()) if float(target.sum().item()) > 0 else None
    pred_top = int(pred.argmax().item())
    pred_top_fraction = float((pred.max() / pred.sum().clamp_min(1.0)).item())
    collapse = pred_top == source_top and target_top is not None and source_top != target_top and pred_top_fraction >= float(top_fraction_threshold)
    return {
        "source_prior_collapse_available": True,
        "source_prior_collapse": bool(collapse),
        "source_prior_collapse_source_top_beam": source_top,
        "source_prior_collapse_target_top_beam": target_top,
        "source_prior_collapse_predicted_top_beam": pred_top,
        "source_prior_collapse_predicted_top_fraction": pred_top_fraction,
        "source_prior_collapse_threshold": float(top_fraction_threshold),
    }


def _branch_metrics(outputs: torch.Tensor, labels: torch.Tensor, *, num_classes: int, top_k: int) -> dict[str, Any]:
    payload = prediction_histogram_payload(labels, outputs, num_classes=num_classes, top_k=top_k)
    result = {
        "pred_hist": payload["pred_hist"],
        "pred_top_beams": payload["pred_top_beams"],
        "unique_pred_beams": int(sum(1 for value in payload["pred_hist"] if int(value) > 0)),
        "within_3_acc": payload["within_3_acc"],
        "mean_abs_beam_error": payload["mean_abs_beam_error"],
    }
    if outputs.ndim == 2:
        outputs = outputs.unsqueeze(1)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    valid = labels.ne(-100) & labels.ge(0) & labels.lt(int(num_classes))
    top = torch.topk(outputs, k=min(max(int(top_k), 1), int(num_classes)), dim=-1).indices
    for k in (1, 3, 5):
        kk = min(k, top.shape[-1])
        hit = top[:, :, :kk].eq(labels.unsqueeze(-1)) & valid.unsqueeze(-1)
        result[f"top{k}"] = float(hit.any(dim=-1)[valid].float().mean().item()) if torch.any(valid) else 0.0
    return result


def _per_true_beam_confusion(labels: torch.Tensor, outputs: torch.Tensor, *, num_classes: int, top_k: int) -> list[dict[str, Any]]:
    if outputs.ndim == 2:
        outputs = outputs.unsqueeze(1)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    pred = outputs.argmax(dim=-1)
    valid = labels.ne(-100) & labels.ge(0) & labels.lt(int(num_classes))
    true_hist = _beam_histogram(labels, num_classes=num_classes)
    top_true = [item["beam"] for item in _top_hist_beams(true_hist, top_k=top_k) if item["count"] > 0]
    rows: list[dict[str, Any]] = []
    for beam in top_true:
        mask = valid & labels.eq(int(beam))
        if not torch.any(mask):
            continue
        pred_values = pred[mask].detach().cpu().to(torch.long)
        pred_hist = torch.bincount(pred_values, minlength=int(num_classes))
        rows.append(
            {
                "true_beam": int(beam),
                "count": int(mask.sum().item()),
                "pred_top_beams": _top_hist_beams([int(item) for item in pred_hist.tolist()], top_k=top_k),
            }
        )
    return rows


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


def _ensure_prediction_horizon(values: torch.Tensor, horizon: int) -> torch.Tensor:
    if values.ndim == 1:
        return values.unsqueeze(1).expand(-1, int(horizon))
    if values.ndim == 2:
        if values.shape[1] == int(horizon):
            return values
        if values.shape[1] == 1:
            return values.expand(-1, int(horizon))
    raise ValueError(f"prediction tensor must have shape [B], [B, 1], or [B, H], got {tuple(values.shape)}.")


def _optional_tensor_value(values: torch.Tensor | None, row_idx: int, horizon: int) -> int | str:
    if values is None or row_idx >= values.shape[0] or horizon >= values.shape[1]:
        return ""
    value = int(values[row_idx, horizon].item())
    return "" if value < 0 else value


def _optional_tensor_list(values: torch.Tensor | None, row_idx: int, horizon: int) -> str:
    if values is None or row_idx >= values.shape[0] or horizon >= values.shape[1]:
        return ""
    return json.dumps([int(item) for item in values[row_idx, horizon].tolist()])


def _horizon(values: torch.Tensor) -> int:
    return int(values.shape[1]) if values.ndim > 1 else 1


def _beam_histogram(values: torch.Tensor, *, num_classes: int) -> list[int]:
    tensor = values.detach().cpu().to(torch.long)
    valid = tensor.ge(0) & tensor.lt(int(num_classes))
    if not torch.any(valid):
        return [0 for _ in range(int(num_classes))]
    return [int(item) for item in torch.bincount(tensor[valid].reshape(-1), minlength=int(num_classes)).tolist()]


def _top_hist_beams(hist: Iterable[int], *, top_k: int) -> list[dict[str, int]]:
    counts = torch.as_tensor(list(hist), dtype=torch.long)
    if counts.numel() == 0:
        return []
    k = min(int(top_k), int(counts.numel()))
    values, indices = torch.topk(counts, k=k)
    return [{"beam": int(index.item()), "count": int(value.item())} for value, index in zip(values, indices)]


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
    "beam_histogram_metrics",
    "calculate_hist_beam_metrics",
    "collapse_diagnostics_payload",
    "histogram_kl",
    "markov_delta_baseline_metrics",
    "path_descriptor_regression_metrics",
    "path_semantic_metrics",
    "prediction_histogram_payload",
    "radio_semantic_metrics",
    "source_prior_collapse_diagnostics",
    "summarize_loso_runs",
    "write_hist_beam_predictions",
    "write_collapse_diagnostics",
    "write_prediction_histogram",
    "write_loso_summary",
]
