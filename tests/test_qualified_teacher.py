import pytest
import torch

from kd_sensing.models.radio_guided_prototype_distillation import (
    propagation_mode_distribution,
    qualified_teacher_weights,
    radio_prototype_distillation_loss,
)


def test_qualified_teacher_weight_is_stop_gradient() -> None:
    sensing = torch.randn(4, 8, requires_grad=True)
    csi = torch.randn(4, 8, requires_grad=True)
    result = qualified_teacher_weights(sensing, csi, torch.arange(4), training=True)
    assert not result["weight"].requires_grad


def test_qualified_teacher_is_forbidden_during_validation_or_inference() -> None:
    with pytest.raises(RuntimeError, match="training-only"):
        qualified_teacher_weights(torch.randn(2, 4), torch.randn(2, 4), torch.tensor([0, 1]), training=False)


def test_distillation_only_compares_modes_inside_true_beam() -> None:
    scores = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    probability = propagation_mode_distribution(scores, torch.tensor([0, 2]))
    expected = torch.softmax(torch.stack((scores[0, 0], scores[1, 2])), dim=-1)
    assert torch.equal(probability, expected)


def test_zero_teacher_coverage_produces_finite_zero_loss() -> None:
    q_s = torch.softmax(torch.randn(3, 2), dim=-1)
    q_c = torch.softmax(torch.randn(3, 2), dim=-1)
    loss = radio_prototype_distillation_loss(q_s, q_c, weights=torch.zeros(3))
    assert torch.isfinite(loss)
    assert float(loss) == 0.0
