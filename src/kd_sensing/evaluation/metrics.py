from __future__ import annotations

import numpy as np
import torch

from kd_sensing.registries import METRICS

DBA_TOP_K = 3


def circular_beam_distance(
    prediction,
    target,
    *,
    num_beams: int = 64,
):
    """Circular class distance for scalar, numpy, or torch beam ids."""
    beams = _positive_num_beams(num_beams)
    if torch.is_tensor(prediction) or torch.is_tensor(target):
        pred = prediction if torch.is_tensor(prediction) else torch.as_tensor(prediction)
        truth = target if torch.is_tensor(target) else torch.as_tensor(target, device=pred.device)
        pred = pred.to(dtype=torch.long).remainder(beams)
        truth = truth.to(device=pred.device, dtype=torch.long).remainder(beams)
        absolute = torch.abs(pred - truth)
        return torch.minimum(absolute, torch.as_tensor(beams, device=absolute.device, dtype=absolute.dtype) - absolute)
    pred_np = np.asarray(prediction, dtype=np.int64) % beams
    truth_np = np.asarray(target, dtype=np.int64) % beams
    absolute_np = np.abs(pred_np - truth_np)
    distance = np.minimum(absolute_np, beams - absolute_np)
    if distance.ndim == 0:
        return int(distance.item())
    return distance


def signed_circular_beam_residual(
    prediction,
    target,
    *,
    num_beams: int = 64,
):
    """Signed shortest residual target - prediction in circular beam space."""
    beams = _positive_num_beams(num_beams)
    if torch.is_tensor(prediction) or torch.is_tensor(target):
        pred = prediction if torch.is_tensor(prediction) else torch.as_tensor(prediction)
        truth = target if torch.is_tensor(target) else torch.as_tensor(target, device=pred.device)
        diff = (truth.to(device=pred.device, dtype=torch.long) - pred.to(dtype=torch.long)).remainder(beams)
        half = beams // 2
        return torch.where(diff > half, diff - beams, diff)
    diff_np = (np.asarray(target, dtype=np.int64) - np.asarray(prediction, dtype=np.int64)) % beams
    half_np = beams // 2
    residual = np.where(diff_np > half_np, diff_np - beams, diff_np)
    if np.asarray(residual).ndim == 0:
        return int(np.asarray(residual).item())
    return residual


def signed_circular_residual(
    target,
    prediction,
    *,
    num_beams: int = 64,
):
    """Signed shortest residual target - prediction in circular beam space.

    This is the residual-correction oriented argument order.  The older
    signed_circular_beam_residual helper keeps prediction, target order.
    """
    return signed_circular_beam_residual(prediction, target, num_beams=num_beams)


def residual_to_delta_class(residual, *, radius: int = 8):
    """Map signed circular residuals to local delta class ids plus overflow."""
    radius_value = _non_negative_radius(radius)
    overflow = 2 * radius_value + 1
    if torch.is_tensor(residual):
        values = residual.to(dtype=torch.long)
        local = values.ge(-radius_value) & values.le(radius_value)
        classes = values + radius_value
        return torch.where(local, classes, torch.full_like(classes, overflow))
    values_np = np.asarray(residual, dtype=np.int64)
    classes_np = values_np + radius_value
    classes_np = np.where(np.abs(values_np) <= radius_value, classes_np, overflow)
    if classes_np.ndim == 0:
        return int(classes_np.item())
    return classes_np


def delta_class_to_residual(class_id, *, radius: int = 8, overflow_value=None):
    """Map local delta class ids back to signed residuals; overflow maps to None by default."""
    radius_value = _non_negative_radius(radius)
    overflow = 2 * radius_value + 1
    if torch.is_tensor(class_id):
        values = class_id.to(dtype=torch.long)
        residual = values - radius_value
        if overflow_value is None:
            overflow_tensor = torch.full_like(residual, -10_000)
        else:
            overflow_tensor = torch.full_like(residual, int(overflow_value))
        return torch.where(values.eq(overflow), overflow_tensor, residual)
    values_np = np.asarray(class_id, dtype=np.int64)
    residual_np = values_np - radius_value
    if overflow_value is None:
        if values_np.ndim == 0:
            return None if int(values_np.item()) == overflow else int(residual_np.item())
        result = residual_np.astype(object)
        result[values_np == overflow] = None
        return result
    residual_np = np.where(values_np == overflow, int(overflow_value), residual_np)
    if residual_np.ndim == 0:
        return int(residual_np.item())
    return residual_np


def residual_delta_class_count(*, radius: int = 8) -> int:
    """Number of residual delta classes including overflow."""
    radius_value = _non_negative_radius(radius)
    return 2 * radius_value + 2


def circular_shift_beam(
    prediction,
    delta,
    *,
    num_beams: int = 64,
):
    """Shift beam ids by a signed circular delta."""
    beams = _positive_num_beams(num_beams)
    if torch.is_tensor(prediction) or torch.is_tensor(delta):
        pred = prediction if torch.is_tensor(prediction) else torch.as_tensor(prediction)
        shift = delta if torch.is_tensor(delta) else torch.as_tensor(delta, device=pred.device)
        return (pred.to(dtype=torch.long) + shift.to(device=pred.device, dtype=torch.long)).remainder(beams)
    shifted = (np.asarray(prediction, dtype=np.int64) + np.asarray(delta, dtype=np.int64)) % beams
    if np.asarray(shifted).ndim == 0:
        return int(np.asarray(shifted).item())
    return shifted


def circular_window(
    center,
    *,
    radius: int,
    num_beams: int = 64,
):
    """Return circular beam windows centered on center without duplicate ids."""
    beams = _positive_num_beams(num_beams)
    radius_value = max(int(radius), 0)
    offsets = list(range(-radius_value, radius_value + 1))
    if torch.is_tensor(center):
        base = center.to(dtype=torch.long)
        offset_tensor = torch.as_tensor(offsets, device=base.device, dtype=torch.long)
        return (base.unsqueeze(-1) + offset_tensor).remainder(beams)
    center_np = np.asarray(center, dtype=np.int64)
    window = (np.expand_dims(center_np, axis=-1) + np.asarray(offsets, dtype=np.int64)) % beams
    if center_np.ndim == 0:
        return [int(item) for item in np.unique(window.reshape(-1))]
    return window


def gps_good_bad_label(error, *, threshold: float = 4.0):
    """Return GPS-good and GPS-bad boolean labels from circular error."""
    if torch.is_tensor(error):
        values = error.to(dtype=torch.float32)
        good = values < float(threshold)
        return good, ~good
    values_np = np.asarray(error, dtype=np.float64)
    good_np = values_np < float(threshold)
    bad_np = np.logical_not(good_np)
    if values_np.ndim == 0:
        return bool(good_np.item()), bool(bad_np.item())
    return good_np, bad_np


def circular_topk_min_distance(
    outputs,
    target,
    *,
    k: int = 3,
    num_beams: int | None = None,
):
    """Minimum circular distance between target and top-k predictions."""
    if torch.is_tensor(outputs) or torch.is_tensor(target):
        out = outputs if torch.is_tensor(outputs) else torch.as_tensor(outputs)
        truth = target if torch.is_tensor(target) else torch.as_tensor(target, device=out.device)
        beams = int(num_beams or out.shape[-1])
        kk = max(1, min(int(k), int(out.shape[-1])))
        topk = torch.topk(out, kk, dim=-1).indices
        truth = truth.to(device=out.device, dtype=torch.long)
        distances = circular_beam_distance(topk, truth.unsqueeze(-1), num_beams=beams)
        return distances.min(dim=-1).values
    out_np = np.asarray(outputs)
    beams = int(num_beams or out_np.shape[-1])
    kk = max(1, min(int(k), int(out_np.shape[-1])))
    topk_np = np.argsort(out_np, axis=-1)[..., -kk:]
    truth_np = np.asarray(target, dtype=np.int64)
    distances = circular_beam_distance(topk_np, np.expand_dims(truth_np, axis=-1), num_beams=beams)
    return np.min(distances, axis=-1)


def dba_from_circular_distances(distances, *, delta: float = 5.0):
    values = np.asarray(distances, dtype=np.float64)
    if values.size == 0:
        return 0.0
    normalized = np.minimum(values / max(float(delta), 1e-8), 1.0)
    return float(np.mean(1.0 - normalized))


def dba_zero_ratio(distances) -> float:
    values = np.asarray(distances, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.mean(values == 0))


def beam_classification_circular_summary(
    outputs,
    labels,
    *,
    num_beams: int | None = None,
    dba_delta: float = 5.0,
    topk: tuple[int, ...] = (1, 3, 5),
    ignore_index: int = -100,
) -> dict[str, float | int]:
    """Single-horizon hard-label summary using circular beam error."""
    if torch.is_tensor(outputs):
        scores = outputs.detach().cpu()
    else:
        scores = torch.as_tensor(outputs)
    if torch.is_tensor(labels):
        target = labels.detach().cpu().to(torch.long)
    else:
        target = torch.as_tensor(labels, dtype=torch.long)
    if scores.ndim == 3 and scores.shape[1] == 1:
        scores = scores[:, 0, :]
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    if scores.ndim != 2:
        raise ValueError(f"outputs must have shape [N, C] or [N, 1, C], got {tuple(scores.shape)}.")
    if target.ndim != 1 or int(target.shape[0]) != int(scores.shape[0]):
        raise ValueError(f"labels must have shape [N] or [N, 1], got {tuple(target.shape)}.")
    beams = int(num_beams or scores.shape[-1])
    valid = target.ne(int(ignore_index)) & target.ge(0) & target.lt(beams)
    valid_count = int(valid.sum().item())
    result: dict[str, float | int] = {
        "sample_count": int(target.numel()),
        "valid_label_count": valid_count,
    }
    if valid_count == 0:
        result.update(
            {
                "DBA": 0.0,
                "DBA_zero_ratio": 0.0,
                "mean_circular_error": 0.0,
                "median_circular_error": 0.0,
                "exact_acc": 0.0,
                "pm1_acc": 0.0,
                "pm2_acc": 0.0,
                "pm4_acc": 0.0,
                "top1": 0.0,
                "top3": 0.0,
                "top5": 0.0,
            }
        )
        return result
    scores_v = scores[valid]
    target_v = target[valid]
    pred = scores_v.argmax(dim=-1)
    distances = circular_beam_distance(pred, target_v, num_beams=beams).numpy()
    result.update(
        {
            "DBA": dba_from_circular_distances(distances, delta=dba_delta),
            "DBA_zero_ratio": dba_zero_ratio(distances),
            "mean_circular_error": float(np.mean(distances)),
            "median_circular_error": float(np.median(distances)),
            "exact_acc": float(np.mean(distances == 0)),
            "pm1_acc": float(np.mean(distances <= 1)),
            "pm2_acc": float(np.mean(distances <= 2)),
            "pm4_acc": float(np.mean(distances <= 4)),
        }
    )
    for kk in topk:
        key = f"top{int(kk)}"
        top_dist = circular_topk_min_distance(scores_v, target_v, k=int(kk), num_beams=beams)
        result[key] = float(torch.mean(top_dist.eq(0).to(torch.float32)).item())
    for key in ("top1", "top3", "top5"):
        result.setdefault(key, 0.0)
    return result


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
    num_classes = int(outputs.shape[-1])
    dba_sum = np.zeros((num_pred,))
    valid_count = np.zeros((num_pred,))
    k = min(DBA_TOP_K, num_classes)
    _, idx = torch.topk(outputs, k, dim=-1)
    idx_np = idx.cpu().numpy()
    labels_np = labels.cpu().numpy()
    for t in range(labels_np.shape[1]):
        for b in range(labels_np.shape[0]):
            gt = labels_np[b, t]
            if gt == -100:
                continue
            preds = idx_np[b, t, :k]
            distances = _circular_class_distance(preds, int(gt), num_classes=num_classes)
            norm_dists = np.minimum(distances / delta, 1.0)
            y_by_k = 1.0 - np.minimum.accumulate(norm_dists)
            dba_sum[t] += np.mean(y_by_k)
            valid_count[t] += 1
    valid_count[valid_count == 0] = 1
    return dba_sum / valid_count


def calculate_beam_group_metrics(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    group_size: int = 8,
    num_classes: int | None = None,
) -> dict[str, object]:
    if outputs.ndim == 2:
        outputs = outputs.unsqueeze(1)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    classes = int(num_classes or outputs.shape[-1])
    group = int(group_size)
    if group <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    if classes % group != 0:
        raise ValueError(f"num_classes ({classes}) must be divisible by group_size ({group}).")
    pred = torch.argmax(outputs, dim=-1)
    valid = labels.ne(-100) & labels.ge(0) & labels.lt(classes)
    total = int(valid.sum().item())
    if total == 0:
        return {
            "coarse_accuracy": 0.0,
            "fine_offset_accuracy": 0.0,
            "beam_group_total": 0,
        }
    safe_labels = labels.clamp_min(0)
    coarse_true = torch.div(safe_labels, group, rounding_mode="floor")
    fine_true = safe_labels.remainder(group)
    coarse_pred = torch.div(pred.clamp_min(0), group, rounding_mode="floor")
    fine_pred = pred.clamp_min(0).remainder(group)
    coarse_acc = (coarse_true[valid] == coarse_pred[valid]).float().mean().item()
    fine_acc = (fine_true[valid] == fine_pred[valid]).float().mean().item()
    angular_error = torch.abs(pred[valid].to(torch.float32) - labels[valid].to(torch.float32)).mean().item()
    return {
        "coarse_accuracy": float(coarse_acc),
        "fine_offset_accuracy": float(fine_acc),
        "mean_angular_error": float(angular_error),
        "beam_group_total": total,
    }


def beam_power_metrics(
    predicted_beams: torch.Tensor,
    true_beams: torch.Tensor,
    power_vectors: torch.Tensor | None,
) -> dict[str, object]:
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
    count = min(int(pred.numel()), int(truth.numel()), int(power.shape[0]))
    pred = pred[:count]
    truth = truth[:count]
    power = power[:count]
    valid = truth.ge(0) & truth.lt(power.shape[-1]) & pred.ge(0) & pred.lt(power.shape[-1])
    if not torch.any(valid):
        return {
            "power_metrics_available": False,
            "power_metrics_unavailable_reason": "no_valid_beam_indices",
        }
    row_index = torch.arange(power.shape[0])[valid]
    chosen = power[row_index, pred[valid]]
    optimal = power[row_index, truth[valid]]
    eps = 1e-8
    normalized = (chosen + eps) / (optimal + eps)
    loss_db = -10.0 * torch.log10(normalized.clamp_min(eps))
    return {
        "power_metrics_available": True,
        "power_metrics_unavailable_reason": None,
        "normalized_received_power": float(normalized.mean().item()),
        "beam_power_loss_db": float(loss_db.mean().item()),
    }


def calculate_current_beam_dba(outputs: torch.Tensor, labels: torch.Tensor, delta: float = 5) -> float:
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    if outputs.ndim == 2:
        outputs = outputs.unsqueeze(1)
    if labels.ndim != 2 or labels.shape[1] != 1:
        raise ValueError("current beam DBA labels must have shape [B, 1] or [B].")
    if outputs.ndim != 3 or outputs.shape[1] != 1:
        raise ValueError("current beam DBA outputs must have shape [B, 1, C] or [B, C].")
    return float(calculate_dba_score(outputs, labels, delta=delta)[0])


def _circular_class_distance(preds: np.ndarray, truth: int, *, num_classes: int) -> np.ndarray:
    return np.asarray(circular_beam_distance(preds, int(truth), num_beams=int(num_classes)), dtype=np.int64)


def _positive_num_beams(num_beams: int) -> int:
    beams = int(num_beams)
    if beams <= 0:
        raise ValueError(f"num_beams must be positive, got {num_beams}.")
    return beams


def _non_negative_radius(radius: int) -> int:
    value = int(radius)
    if value < 0:
        raise ValueError(f"radius must be non-negative, got {radius}.")
    return value


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
