from __future__ import annotations

import numpy as np
import torch

from kd_sensing.registries import METRICS

DBA_TOP_K = 3


def calculate_topk_accuracy(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    k_values: list[int] | tuple[int, ...] = (1, 2, 3, 5, 10),
):
    num_pred = labels.shape[1]
    max_k = min(max(k_values), outputs.shape[-1])
    topk_correct = {k: np.zeros((num_pred,)) for k in k_values}
    total = torch.sum(labels != -100, dim=0).cpu().numpy()
    _, idx = torch.topk(outputs, max_k, dim=-1)
    idx_np = idx.cpu().numpy()
    labels_np = labels.cpu().numpy()
    for i in range(labels_np.shape[1]):
        for j in range(labels_np.shape[0]):
            for k in k_values:
                kk = min(k, max_k)
                topk_correct[k][i] += np.isin(labels_np[j, i], idx_np[j, i, :kk])
    return {k: topk_correct[k] / (total + 1e-8) for k in k_values}, total


def calculate_dba_score(outputs: torch.Tensor, labels: torch.Tensor, delta: float = 5):
    num_pred = labels.shape[1]
    dba_sum = np.zeros((num_pred,))
    valid_count = np.zeros((num_pred,))
    k = min(DBA_TOP_K, outputs.shape[-1])
    _, idx = torch.topk(outputs, k, dim=-1)
    idx_np = idx.cpu().numpy()
    labels_np = labels.cpu().numpy()
    for t in range(labels_np.shape[1]):
        for b in range(labels_np.shape[0]):
            gt = labels_np[b, t]
            if gt == -100:
                continue
            preds = idx_np[b, t, :k]
            norm_dists = np.minimum(np.abs(preds - gt) / delta, 1.0)
            y_by_k = 1.0 - np.minimum.accumulate(norm_dists)
            dba_sum[t] += np.mean(y_by_k)
            valid_count[t] += 1
    valid_count[valid_count == 0] = 1
    return dba_sum / valid_count


def calculate_occlusion_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> dict[str, float]:
    logits = logits.detach().cpu()
    labels = labels.detach().cpu().to(torch.float32)
    if valid is None:
        valid = torch.ones_like(labels, dtype=torch.bool)
    else:
        valid = valid.detach().cpu().to(torch.bool)
    if logits.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError("occlusion logits, labels, and valid mask must share shape [B, H].")
    mask = valid.numpy().astype(bool)
    total = int(mask.sum())
    if total == 0:
        return {
            "occlusion_accuracy": 0.0,
            "occlusion_blocked_f1": 0.0,
            "occlusion_total": 0,
            "occlusion_positive": 0,
        }
    pred = (torch.sigmoid(logits).numpy() >= 0.5).astype(np.int64)
    target = (labels.numpy() >= 0.5).astype(np.int64)
    pred_v = pred[mask]
    target_v = target[mask]
    tp = int(((pred_v == 1) & (target_v == 1)).sum())
    fp = int(((pred_v == 1) & (target_v == 0)).sum())
    fn = int(((pred_v == 0) & (target_v == 1)).sum())
    correct = int((pred_v == target_v).sum())
    denom = 2 * tp + fp + fn
    f1 = float(0.0 if denom == 0 else (2 * tp) / denom)
    return {
        "occlusion_accuracy": float(correct / max(total, 1)),
        "occlusion_blocked_f1": f1,
        "occlusion_total": total,
        "occlusion_positive": int((target_v == 1).sum()),
    }


def calculate_los_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    logits = logits.detach().cpu()
    labels = labels.detach().cpu().to(torch.float32)
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    if logits.shape != labels.shape:
        raise ValueError("LOS logits and labels must share shape [B, H].")
    probs = torch.sigmoid(logits).numpy().reshape(-1)
    target = (labels.numpy().reshape(-1) >= 0.5).astype(np.int64)
    pred = (probs >= 0.5).astype(np.int64)
    total = int(target.size)
    if total == 0:
        return {
            "los_accuracy": 0.0,
            "los_f1": 0.0,
            "los_auc": None,
            "los_auc_available": False,
            "los_auc_unavailable_reason": "empty_split",
            "los_total": 0,
            "los_positive": 0,
        }
    tp = int(((pred == 1) & (target == 1)).sum())
    fp = int(((pred == 1) & (target == 0)).sum())
    fn = int(((pred == 0) & (target == 1)).sum())
    correct = int((pred == target).sum())
    denom = 2 * tp + fp + fn
    auc, auc_available, reason = _binary_auc(probs, target)
    return {
        "los_accuracy": float(correct / max(total, 1)),
        "los_f1": float(0.0 if denom == 0 else (2 * tp) / denom),
        "los_auc": auc,
        "los_auc_available": auc_available,
        "los_auc_unavailable_reason": reason,
        "los_total": total,
        "los_positive": int((target == 1).sum()),
    }


def calculate_link_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = prediction.detach().cpu().to(torch.float32)
    truth = target.detach().cpu().to(torch.float32)
    if pred.ndim == 1:
        pred = pred.unsqueeze(1)
    if truth.ndim == 1:
        truth = truth.unsqueeze(1)
    if pred.shape != truth.shape:
        raise ValueError("link prediction and target must share shape [B, H].")
    pred_np = pred.numpy().reshape(-1).astype(np.float64)
    target_np = truth.numpy().reshape(-1).astype(np.float64)
    if pred_np.size == 0:
        return {"link_mae": 0.0, "link_rmse": 0.0, "link_r2": 0.0, "link_total": 0}
    error = pred_np - target_np
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    denom = float(np.sum((target_np - target_np.mean()) ** 2))
    r2 = 0.0 if denom <= 1e-12 else float(1.0 - np.sum(error ** 2) / denom)
    return {"link_mae": mae, "link_rmse": rmse, "link_r2": r2, "link_total": int(pred_np.size)}


def _binary_auc(probabilities: np.ndarray, target: np.ndarray) -> tuple[float | None, bool, str | None]:
    positives = int((target == 1).sum())
    negatives = int((target == 0).sum())
    if positives == 0 or negatives == 0:
        return None, False, "single_class_split"
    order = np.argsort(probabilities)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(probabilities) + 1, dtype=np.float64)
    pos_ranks = ranks[target == 1].sum()
    auc = (pos_ranks - positives * (positives + 1) / 2.0) / max(positives * negatives, 1)
    return float(auc), True, None


def calculate_position_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> dict[str, float]:
    pred_np = prediction.detach().cpu().numpy().astype(np.float64)
    target_np = target.detach().cpu().numpy().astype(np.float64)
    if valid is None:
        valid_np = np.ones(pred_np.shape[:2], dtype=bool)
    else:
        valid_np = valid.detach().cpu().numpy().astype(bool)
    if pred_np.shape != target_np.shape or pred_np.ndim != 3 or pred_np.shape[-1] != 2:
        raise ValueError("position prediction and target must share shape [B, H, 2].")
    if valid_np.shape != pred_np.shape[:2]:
        raise ValueError("position valid mask must have shape [B, H].")
    if mean is not None and scale is not None:
        mean_np = np.asarray(mean, dtype=np.float64).reshape(1, 1, 2)
        scale_np = np.asarray(scale, dtype=np.float64).reshape(1, 1, 2)
        pred_np = pred_np * scale_np + mean_np
        target_np = target_np * scale_np + mean_np
    total = int(valid_np.sum())
    if total == 0:
        return {"position_rmse": 0.0, "position_mae": 0.0, "position_total": 0}
    squared = ((pred_np - target_np) ** 2).sum(axis=-1)
    absolute = np.abs(pred_np - target_np).sum(axis=-1)
    return {
        "position_rmse": float(np.sqrt(squared[valid_np].mean())),
        "position_mae": float(absolute[valid_np].mean()),
        "position_total": total,
    }


@METRICS.register("topk_accuracy")
class TopKAccuracyMetric:
    def __init__(self, k_values: list[int] | tuple[int, ...] = (1, 2, 3, 5, 10)):
        self.k_values = list(k_values)

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        return calculate_topk_accuracy(outputs, labels, self.k_values)


@METRICS.register("dba")
class DBAMetric:
    def __init__(self, delta: float = 5):
        self.delta = delta

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        return calculate_dba_score(outputs, labels, self.delta)
