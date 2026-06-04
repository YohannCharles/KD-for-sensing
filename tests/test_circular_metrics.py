from __future__ import annotations

import numpy as np
import torch

from kd_sensing.evaluation.metrics import (
    beam_classification_circular_summary,
    circular_beam_distance,
    circular_shift_beam,
    circular_topk_min_distance,
    circular_window,
    gps_good_bad_label,
    signed_circular_residual,
)
from kd_sensing.data.deepsense6g_residual import circular_gaussian_prior_from_top1
from kd_sensing.losses.circular import (
    circular_soft_ce_loss,
    circular_soft_target,
    class_balanced_weights,
    focal_circular_soft_ce_loss,
)


def test_circular_beam_distance_wraparound_scalar_numpy_and_torch():
    assert circular_beam_distance(0, 63, num_beams=64) == 1
    assert circular_beam_distance(63, 1, num_beams=64) == 2
    assert np.array_equal(
        circular_beam_distance(np.array([0, 2, 63]), np.array([63, 62, 0]), num_beams=64),
        np.array([1, 4, 1]),
    )
    torch_dist = circular_beam_distance(torch.tensor([0, 63]), torch.tensor([63, 0]), num_beams=64)
    assert torch.equal(torch_dist, torch.tensor([1, 1]))


def test_topk_min_distance_and_circular_summary_fields():
    logits = torch.full((3, 64), -10.0)
    logits[0, 0] = 8.0
    logits[0, 63] = 7.0
    logits[1, 63] = 8.0
    logits[1, 0] = 7.0
    logits[2, 10] = 8.0
    logits[2, 12] = 7.0
    labels = torch.tensor([63, 0, 12])

    top2_distance = circular_topk_min_distance(logits, labels, k=2, num_beams=64)
    assert torch.equal(top2_distance, torch.tensor([0, 0, 0]))

    summary = beam_classification_circular_summary(logits, labels, num_beams=64, dba_delta=5)
    assert summary["valid_label_count"] == 3
    assert summary["DBA_zero_ratio"] == 0.0
    assert summary["pm2_acc"] == 1.0
    assert summary["top3"] == 1.0
    assert "mean_circular_error" in summary
    assert "median_circular_error" in summary


def test_circular_soft_target_normalizes_and_wraps_boundary():
    target = torch.tensor([0])
    soft = circular_soft_target(target, num_beams=64, sigma=1.5)

    assert tuple(soft.shape) == (1, 64)
    assert torch.isclose(soft.sum(), torch.tensor(1.0), atol=1e-6)
    assert soft[0, 63] > soft[0, 62]
    assert soft[0, 63] == soft[0, 1]


def test_circular_soft_ce_focal_and_class_weights_are_finite():
    logits = torch.randn(4, 64)
    labels = torch.tensor([0, 0, 3, 63])
    weights, metadata = class_balanced_weights(labels, num_classes=64, mode="effective_num", beta=0.9)

    assert weights.shape == (64,)
    assert metadata["class_weight_mode"] == "effective_num"
    assert metadata["label_histogram"][0] == 2
    assert metadata["fit_split"] == "train"

    ce = circular_soft_ce_loss(logits, labels, sigma=2.0, class_weight=weights)
    focal = focal_circular_soft_ce_loss(logits, labels, sigma=2.0, gamma=2.0, class_weight=weights)
    assert torch.isfinite(ce)
    assert torch.isfinite(focal)


def test_residual_circular_helpers_wrap_shift_window_and_good_bad():
    assert signed_circular_residual(1, 63, num_beams=64) == 2
    assert signed_circular_residual(63, 1, num_beams=64) == -2
    assert circular_shift_beam(63, 2, num_beams=64) == 1
    assert circular_window(0, radius=2, num_beams=64) == [0, 1, 2, 62, 63]
    good, bad = gps_good_bad_label(np.array([0, 3, 4]), threshold=4)
    assert np.array_equal(good, np.array([True, True, False]))
    assert np.array_equal(bad, np.array([False, False, True]))


def test_fallback_gaussian_prior_uses_top1_not_target_label():
    logits_from_top1, probs_from_top1 = circular_gaussian_prior_from_top1(63, num_beams=64, sigma=2.0)
    logits_with_different_target, _ = circular_gaussian_prior_from_top1(63, num_beams=64, sigma=2.0)

    assert logits_from_top1.argmax() == 63
    assert np.isclose(probs_from_top1.sum(), 1.0)
    assert np.array_equal(logits_from_top1, logits_with_different_target)
