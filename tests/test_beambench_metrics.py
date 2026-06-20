import numpy as np
import pytest
import torch

from kd_sensing.baselines.beambench.metrics import (
    beambench_metric_summary_from_logits,
    official_dba_score,
    official_topk_accuracy,
)


def test_beambench_perfect_prediction_reaches_maximum_metrics():
    logits = torch.full((3, 8), -10.0)
    labels = torch.tensor([0, 3, 7])
    logits[0, 0] = 5.0
    logits[1, 3] = 5.0
    logits[2, 7] = 5.0

    summary = beambench_metric_summary_from_logits(logits, labels, num_beams=8, topk=(1, 3), dba_delta=5)

    assert summary["official_top1_acc"] == pytest.approx(1.0)
    assert summary["official_top3_acc"] == pytest.approx(1.0)
    assert summary["official_top3_dba"] == pytest.approx(1.0)
    assert summary["circular_top3_dba"] == pytest.approx(1.0)


def test_official_dba_decreases_as_beam_distance_grows():
    labels = np.asarray([10, 10, 10])
    near = np.asarray([[10, 11, 12], [9, 10, 11], [11, 12, 13]])
    far = np.asarray([[25, 26, 27], [0, 1, 2], [31, 32, 33]])

    assert official_dba_score(far, labels, max_k=3) <= official_dba_score(near, labels, max_k=3)


def test_topk_hit_counts_without_top1_hit():
    predictions = np.asarray([[1, 2, 3], [4, 5, 6]])
    labels = np.asarray([3, 4])

    assert official_topk_accuracy(predictions, labels, k=1) == pytest.approx(0.5)
    assert official_topk_accuracy(predictions, labels, k=3) == pytest.approx(1.0)
