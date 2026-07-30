from __future__ import annotations

import torch

from kd_sensing.diagnostics.prototype_deformation import (
    MASKS,
    additive_deformation,
    benjamini_hochberg,
    estimate_centers,
    mask_metadata,
    normalize,
    prototype_logits,
    spherical_exp_map,
    spherical_log_map,
    validate_mask_contract,
)
from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank


def test_spherical_exp_log_round_trip() -> None:
    generator = torch.Generator().manual_seed(17)
    base = normalize(torch.randn(32, 64, generator=generator))
    target = normalize(torch.randn(32, 64, generator=generator))
    reconstructed = spherical_exp_map(base, spherical_log_map(base, target))
    torch.testing.assert_close(reconstructed, target, atol=2e-5, rtol=2e-5)


def test_mask_contract_is_complete_unique_and_uses_internal_slot_order() -> None:
    validate_mask_contract()
    rows = mask_metadata()
    assert len(rows) == 15
    assert MASKS["full"] == (1, 1, 1, 1)
    assert MASKS["missing_lidar"] == (1, 0, 1, 1)
    assert MASKS["radar_only"] == (0, 0, 1, 0)
    assert {row["group"] for row in rows} == {"Full", "Single", "Two", "Three"}


def test_additive_fit_reads_only_single_missing_shifts() -> None:
    shifts = torch.randn(15, 3, 5, generator=torch.Generator().manual_seed(23))
    first = additive_deformation(shifts)
    shifts[5:] = 10_000.0
    second = additive_deformation(shifts)
    torch.testing.assert_close(first, second)


def test_empirical_center_estimation_is_train_tensor_only_and_finite() -> None:
    features = torch.randn(40, 15, 8, generator=torch.Generator().manual_seed(31))
    labels = torch.arange(40) % 4
    prototypes = torch.randn(4, 8, generator=torch.Generator().manual_seed(37))
    centers = estimate_centers(features, labels, prototypes, kappa=20, num_beams=4)
    assert centers["spherical"].shape == (15, 4, 8)
    assert torch.isfinite(centers["shrinkage"]).all()
    torch.testing.assert_close(centers["spherical"].norm(dim=-1), torch.ones(15, 4))


def test_scoring_matches_frozen_prototype_bank_formula() -> None:
    bank = BeamPrototypeBank(7, 5, temperature=0.1)
    features = torch.randn(11, 7, generator=torch.Generator().manual_seed(41))
    expected = bank(features)
    actual = prototype_logits(features, bank.prototypes.detach(), temperature=0.1)
    torch.testing.assert_close(actual, expected)


def test_benjamini_hochberg_is_monotone_in_sorted_p_values() -> None:
    p = torch.tensor([0.04, 0.001, 0.02, 0.8])
    q = benjamini_hochberg(p)
    order = p.numpy().argsort()
    assert (q[order][1:] >= q[order][:-1]).all()
