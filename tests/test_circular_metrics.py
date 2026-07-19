import numpy as np
import pytest
import torch

from kd_sensing.evaluation.metrics import (
    beam_classification_circular_summary,
    beam_power_communication_summary,
    calculate_dba_score,
    calculate_topk_accuracy,
    circular_beam_distance,
    circular_topk_min_distance,
)


def test_beam_power_communication_summary_uses_full_power_vector() -> None:
    logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, 0.0, 1.0]])
    powers = torch.tensor([[1.0, 4.0, 2.0], [1.0, 4.0, 2.0]])

    metrics = beam_power_communication_summary(logits, powers)

    assert metrics["normalized_gain"] == pytest.approx(0.625)
    assert metrics["gain_loss_db"] == pytest.approx(3.0102999)
    assert metrics["spectral_efficiency_ratio_10db"] < 1.0
    assert metrics["spectral_efficiency_loss_10db"] > 0.0


def test_circular_beam_distance_wraps_scalar_numpy_and_torch():
    assert circular_beam_distance(0, 63, num_beams=64) == 1
    assert np.array_equal(circular_beam_distance(np.array([0, 63]), np.array([63, 0]), num_beams=64), [1, 1])
    assert torch.equal(circular_beam_distance(torch.tensor([0, 63]), torch.tensor([63, 0]), num_beams=64), torch.tensor([1, 1]))


def test_topk_and_dba_follow_wrapped_beam_distance():
    logits = torch.full((2, 1, 64), -10.0)
    logits[0, 0, 0], logits[0, 0, 63] = 8.0, 7.0
    logits[1, 0, 63], logits[1, 0, 0] = 8.0, 7.0
    labels = torch.tensor([[63], [0]])

    topk, total = calculate_topk_accuracy(logits, labels, (1, 2))
    assert total.tolist() == [2]
    assert topk[1].tolist() == [0.0]
    assert topk[2].tolist() == [1.0]
    assert circular_topk_min_distance(logits[:, 0], labels[:, 0], k=2, num_beams=64).tolist() == [0, 0]
    assert calculate_dba_score(logits, labels, distance_mode="circular")[0] > calculate_dba_score(logits, labels, distance_mode="linear")[0]


def test_matrix_summary_keeps_only_current_beam_fields():
    logits = torch.full((1, 64), -10.0)
    logits[0, 63] = 8.0
    labels = torch.tensor([0])

    circular = beam_classification_circular_summary(logits, labels, num_beams=64)
    linear = beam_classification_circular_summary(logits, labels, num_beams=64, distance_mode="linear")
    assert set(circular) == {"DBA", "mean_error", "within_1", "within_3", "top1", "top3", "top5"}
    assert circular["mean_error"] == 1.0
    assert circular["within_3"] == 1.0
    assert linear["mean_error"] == 63.0
    assert linear["within_3"] == 0.0


def test_topk_distance_respects_linear_mode_at_beam_wrap_boundary():
    logits = torch.full((1, 64), -10.0)
    logits[0, 63], logits[0, 2], logits[0, 3] = 8.0, 7.0, 6.0
    labels = torch.tensor([0])

    assert circular_topk_min_distance(logits, labels, k=3, num_beams=64, distance_mode="circular").tolist() == [1]
    assert circular_topk_min_distance(logits, labels, k=3, num_beams=64, distance_mode="linear").tolist() == [2]
    assert beam_classification_circular_summary(logits, labels, num_beams=64, distance_mode="circular")["top3"] == 0.0
    assert beam_classification_circular_summary(logits, labels, num_beams=64, distance_mode="linear")["top3"] == 0.0
