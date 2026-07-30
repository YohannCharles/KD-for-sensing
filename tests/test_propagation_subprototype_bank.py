import math

import torch

from kd_sensing.losses.hierarchical_prototype_losses import prototype_diversity_loss
from kd_sensing.models.propagation_subprototype_bank import (
    PropagationAwareSubPrototypeBank,
    reproducible_random_residuals,
)


def _bank() -> PropagationAwareSubPrototypeBank:
    return PropagationAwareSubPrototypeBank(torch.eye(4), num_subprototypes=2, epsilon=0.15)


def test_subprototype_shape() -> None:
    assert _bank().subprototypes().shape == (4, 2, 4)


def test_residual_radius_does_not_exceed_epsilon() -> None:
    bank = _bank()
    with torch.no_grad():
        bank.raw_delta.fill_(100.0)
    assert bool((bank.residuals().norm(dim=-1) <= bank.epsilon + 1e-6).all())


def test_base_prototype_bank_is_frozen() -> None:
    bank = _bank()
    assert "base_prototypes" not in dict(bank.named_parameters())
    assert not bank.base_prototypes.requires_grad


def test_logmeanexp_has_no_fixed_k_bias() -> None:
    bank = _bank()
    one = torch.tensor([[[2.5]]]).expand(1, 4, 1)
    two = torch.tensor([[[2.5, 2.5]]]).expand(1, 4, 2)
    assert torch.equal(bank.aggregate(two), one.squeeze(-1))
    assert math.isclose(float(bank.aggregate(two)[0, 0]), 2.5)


def test_sensing_and_csi_scoring_share_the_same_subprototypes() -> None:
    bank = _bank()
    feature = torch.randn(3, 4)
    sensing = bank.score(feature, scale=2.0)
    csi = bank.score(feature, scale=2.0)
    assert torch.equal(sensing, csi)


def test_invalid_clusters_do_not_contribute_diversity() -> None:
    bank = _bank()
    loss = prototype_diversity_loss(bank.subprototypes(), torch.zeros(4, dtype=torch.bool))
    assert float(loss.detach()) == 0.0


def test_invalid_clusters_remain_at_base_after_optimizer_step() -> None:
    bank = _bank()
    bank.set_trainable_beam_mask_(torch.tensor([True, False, True, False]))
    optimizer = torch.optim.AdamW(bank.parameters(), lr=0.1, weight_decay=0.1)
    loss = bank(torch.randn(8, 4)).sum()
    loss.backward()
    assert torch.equal(bank.raw_delta.grad[1], torch.zeros_like(bank.raw_delta.grad[1]))
    optimizer.step()
    bank.enforce_trainable_beam_mask_()
    assert torch.equal(bank.residuals()[1], torch.zeros_like(bank.residuals()[1]))
    assert torch.equal(bank.subprototypes()[1, 0], bank.subprototypes()[1, 1])


def test_a5_random_initialization_is_reproducible_and_radius_matched() -> None:
    first = reproducible_random_residuals(torch.eye(4), radius=0.1, seed=5)
    second = reproducible_random_residuals(torch.eye(4), radius=0.1, seed=5)
    other = reproducible_random_residuals(torch.eye(4), radius=0.1, seed=6)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert torch.allclose(first.norm(dim=-1), torch.full((4, 2), 0.1))
