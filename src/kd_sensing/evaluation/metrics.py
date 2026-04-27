from __future__ import annotations

import numpy as np
import torch

from kd_sensing.registries import METRICS


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
    dba_score = np.zeros((num_pred,))
    valid_count = np.zeros((num_pred,))
    k = min(3, outputs.shape[-1])
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
            dba_score[t] += np.min(norm_dists)
            valid_count[t] += 1
    valid_count[valid_count == 0] = 1
    return 1 - (dba_score / valid_count)


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

