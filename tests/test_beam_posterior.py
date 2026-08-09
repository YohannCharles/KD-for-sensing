import pytest
import torch

from kd_sensing.models.beam_posterior import beam_posterior_statistics


def _one_hot(index: int, value: float = 1.0) -> torch.Tensor:
    result = torch.zeros(1, 64)
    result[0, index] = value
    return result


def test_one_hot_posterior_has_zero_spread_and_stable_top_l() -> None:
    stats = beam_posterior_statistics(_one_hot(63), top_l=7)

    assert stats["beam_map"].tolist() == [63]
    assert stats["beam_circular_mean_label"].tolist() == [63]
    assert stats["beam_variance"].item() == pytest.approx(0.0)
    assert stats["beam_spread"].item() == pytest.approx(0.0)
    assert stats["beam_circular_variance"].item() == pytest.approx(0.0)
    assert stats["beam_normalized_entropy"].item() == pytest.approx(0.0)
    assert stats["beam_top_indices"].shape == (1, 7)
    assert stats["beam_top_indices"][0, 0].item() == 63


def test_circular_boundary_treats_zero_and_63_as_neighbors() -> None:
    probability = torch.zeros(1, 64)
    probability[0, 63] = 0.6
    probability[0, 0] = 0.4
    stats = beam_posterior_statistics(probability, top_l=2)

    assert stats["beam_map"].item() == 63
    assert stats["beam_circular_mean"].item() > 63.0 or stats["beam_circular_mean"].item() < 1.0
    assert stats["beam_variance"].item() == pytest.approx(0.4, abs=1e-5)
    assert stats["beam_spread"].item() == pytest.approx(0.4**0.5, abs=1e-5)


def test_uniform_posterior_falls_back_to_map_and_has_max_entropy() -> None:
    probability = torch.full((2, 64), 1.0 / 64.0)
    stats = beam_posterior_statistics(probability)

    assert stats["beam_map"].tolist() == [0, 0]
    assert stats["beam_circular_mean_label"].tolist() == [0, 0]
    assert torch.allclose(stats["beam_circular_variance"], torch.ones(2))
    assert torch.allclose(stats["beam_normalized_entropy"], torch.ones(2), atol=1e-6)
    assert torch.isfinite(stats["beam_variance"]).all()


def test_top_l_breaks_probability_ties_by_lower_beam_index() -> None:
    probability = torch.full((1, 64), 1.0 / 64.0)
    stats = beam_posterior_statistics(probability, top_l=5)

    assert stats["beam_top_indices"].tolist() == [[0, 1, 2, 3, 4]]
    assert torch.allclose(stats["beam_top_probabilities"], torch.full((1, 5), 1.0 / 64.0))


@pytest.mark.parametrize(
    "probability, message",
    [
        (torch.ones(1, 63), "exactly 64"),
        (torch.full((1, 64), -1.0), "non-negative"),
        (torch.full((1, 64), 0.5), "sum to one"),
        (torch.full((1, 64), float("nan")), "finite"),
    ],
)
def test_invalid_posterior_is_rejected(probability: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        beam_posterior_statistics(probability)


def test_topology_positions_are_validated_and_used() -> None:
    probability = _one_hot(0)
    positions = torch.arange(64).flip(0)
    stats = beam_posterior_statistics(probability, topology_positions=positions)

    assert stats["beam_map"].item() == 0
    assert stats["beam_circular_mean"].item() == pytest.approx(63.0)

    with pytest.raises(ValueError, match="permutation"):
        beam_posterior_statistics(probability, topology_positions=torch.zeros(64))
