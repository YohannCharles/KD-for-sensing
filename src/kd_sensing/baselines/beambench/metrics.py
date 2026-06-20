from typing import Iterable

import numpy as np
import torch

from kd_sensing.evaluation.metrics import circular_topk_min_distance, dba_from_circular_distances


def official_dba_score(
    topk_predictions,
    labels,
    *,
    max_k: int = 3,
    delta: float = 5.0,
    prediction_beam_shift: int = 0,
    label_beam_shift: int = 0,
    ignore_index: int = -100,
) -> float:
    """BeamBench/DeepSense non-circular top-k DBA over beam id predictions."""
    preds = _as_topk_array(topk_predictions) - int(prediction_beam_shift)
    truth = np.asarray(labels, dtype=np.int64).reshape(-1) - int(label_beam_shift)
    if preds.shape[0] != truth.shape[0]:
        raise ValueError("topk_predictions and labels must have the same sample count.")
    valid = truth != int(ignore_index)
    preds = preds[valid]
    truth = truth[valid]
    if truth.size == 0:
        return 0.0
    kk = max(1, min(int(max_k), int(preds.shape[-1])))
    yk = np.zeros(kk, dtype=np.float64)
    for k_index in range(kk):
        distances = np.abs(preds[:, : k_index + 1] - truth[:, None]) / max(float(delta), 1e-8)
        clipped = np.minimum(distances, 1.0)
        yk[k_index] = 1.0 - float(np.mean(np.min(clipped, axis=1)))
    return float(np.mean(yk))


def official_topk_accuracy(
    topk_predictions,
    labels,
    *,
    k: int = 3,
    prediction_beam_shift: int = 0,
    label_beam_shift: int = 0,
    ignore_index: int = -100,
) -> float:
    preds = _as_topk_array(topk_predictions) - int(prediction_beam_shift)
    truth = np.asarray(labels, dtype=np.int64).reshape(-1) - int(label_beam_shift)
    if preds.shape[0] != truth.shape[0]:
        raise ValueError("topk_predictions and labels must have the same sample count.")
    valid = truth != int(ignore_index)
    preds = preds[valid]
    truth = truth[valid]
    if truth.size == 0:
        return 0.0
    kk = max(1, min(int(k), int(preds.shape[-1])))
    return float(np.mean(np.any(preds[:, :kk] == truth[:, None], axis=1)))


def beambench_metric_summary_from_logits(
    logits,
    labels,
    *,
    num_beams: int | None = None,
    topk: Iterable[int] = (1, 3, 5),
    dba_delta: float = 5.0,
    label_beam_shift: int = 0,
    circular: bool = True,
    ignore_index: int = -100,
) -> dict[str, float | int | bool | str]:
    scores = logits.detach().cpu() if torch.is_tensor(logits) else torch.as_tensor(logits)
    if scores.ndim != 2:
        raise ValueError(f"logits must have shape [N, C], got {tuple(scores.shape)}.")
    label_tensor = labels.detach().cpu().to(torch.long) if torch.is_tensor(labels) else torch.as_tensor(labels, dtype=torch.long)
    label_tensor = label_tensor.reshape(-1)
    if int(label_tensor.numel()) != int(scores.shape[0]):
        raise ValueError("labels must have shape [N].")
    beams = int(num_beams or scores.shape[-1])
    normalized_labels = label_tensor - int(label_beam_shift)
    valid = normalized_labels.ne(int(ignore_index)) & normalized_labels.ge(0) & normalized_labels.lt(beams)
    valid_count = int(valid.sum().item())
    max_k = max(1, min(max(int(item) for item in topk), int(scores.shape[-1])))
    topk_indices = torch.topk(scores, max_k, dim=-1).indices.numpy()
    labels_np = normalized_labels.numpy()
    report: dict[str, float | int | bool | str] = {
        "sample_count": int(label_tensor.numel()),
        "valid_label_count": valid_count,
        "num_beams": beams,
        "label_beam_shift": int(label_beam_shift),
        "official_dba_delta": float(dba_delta),
        "official_dba_circular": False,
        "circular_metrics_enabled": bool(circular),
    }
    if valid_count == 0:
        for kk in topk:
            report[f"official_top{int(kk)}_acc"] = 0.0
            if circular:
                report[f"circular_top{int(kk)}_acc"] = 0.0
        report["official_top3_dba"] = 0.0
        if circular:
            report["circular_top3_dba"] = 0.0
        return report
    for kk in topk:
        report[f"official_top{int(kk)}_acc"] = official_topk_accuracy(topk_indices, labels_np, k=int(kk))
    report["official_top3_dba"] = official_dba_score(topk_indices, labels_np, max_k=min(3, max_k), delta=dba_delta)
    if circular:
        for kk in topk:
            dist = circular_topk_min_distance(scores[valid], normalized_labels[valid], k=int(kk), num_beams=beams)
            report[f"circular_top{int(kk)}_acc"] = float(torch.mean(dist.eq(0).to(torch.float32)).item())
        top3_dist = circular_topk_min_distance(scores[valid], normalized_labels[valid], k=min(3, max_k), num_beams=beams)
        report["circular_top3_dba"] = dba_from_circular_distances(top3_dist.numpy(), delta=dba_delta)
    return report


def _as_topk_array(values) -> np.ndarray:
    if torch.is_tensor(values):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"topk_predictions must have shape [N, K], got {array.shape}.")
    return array.astype(np.int64)
