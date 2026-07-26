import csv
from pathlib import Path

import numpy as np
import torch

from kd_sensing.baselines.full_pool_candidate12 import (
    Candidate12Model,
    assignment_diagnostics,
    capacity_constrained_assignment,
    load_signed_angle_order,
    motion_mixture,
    noncircular_shift,
    pamr_candidate_gate,
    remix_loss,
    signed_offset_targets,
)


def _order() -> tuple[int, ...]:
    return tuple([32, *range(31, -1, -1), *range(63, 32, -1)])


def test_signed_order_is_loaded_from_physical_angle_not_label_index(tmp_path: Path):
    path = tmp_path / "topology_table.csv"
    rows = [{"label": label, "principal_local_angle_deg": position - 32} for position, label in enumerate(_order())]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert load_signed_angle_order(path) == _order()


def test_non_circular_shift_has_no_endpoint_wrap_and_zero_is_exact():
    order = _order()
    probability = torch.full((2, 64), 1 / 64)
    assert torch.equal(noncircular_shift(probability, 0, order), probability)
    plus = noncircular_shift(probability, 1, order)
    minus = noncircular_shift(probability, -1, order)
    assert plus[:, order[0]].eq(0).all()
    assert minus[:, order[-1]].eq(0).all()
    assert torch.allclose(plus.sum(-1), torch.ones(2))
    assert torch.allclose(minus.sum(-1), torch.ones(2))


def test_motion_mixture_is_normalized_and_local():
    anchor = torch.softmax(torch.randn(3, 64), -1)
    shift_logits = torch.randn(3, 7)
    result = motion_mixture(anchor, shift_logits, _order())
    assert result.shape == (3, 64)
    assert torch.allclose(result.sum(-1), torch.ones(3), atol=1e-6)
    assert (result >= 0).all()


def test_far_signed_residual_is_excluded_instead_of_clipped():
    order = _order()
    anchor = torch.zeros(2, 64)
    anchor[0, order[0]] = 1
    anchor[1, order[20]] = 1
    labels = torch.tensor([order[10], order[22]])
    target, valid, raw = signed_offset_targets(anchor, labels, order, radius=3)
    assert raw.tolist() == [10, 2]
    assert valid.tolist() == [False, True]
    assert target[1].item() == 5


def test_risk_assignment_is_deterministic_unique_and_capacity_bounded():
    generator = np.random.default_rng(2026)
    scores = generator.normal(size=(1000, 4))
    ids = [f"sample-{index:04d}" for index in range(1000)]
    first = capacity_constrained_assignment(scores, ids)
    second = capacity_constrained_assignment(scores, ids)
    assert np.array_equal(first, second)
    counts = np.bincount(first, minlength=4)
    assert counts.sum() == 1000
    assert np.all(counts >= 150)
    assert np.all(counts <= 400)


def test_assignment_diagnostics_use_topology_risk_and_margin_percentiles():
    generator = np.random.default_rng(7)
    logits = generator.normal(size=(12, 4, 64))
    features = generator.normal(size=(12, 4, 8))
    prototypes = generator.normal(size=(64, 8))
    labels = np.arange(12) % 64
    distance = np.abs(np.arange(64)[:, None] - np.arange(64)[None, :]) / 63
    result = assignment_diagnostics(logits, features, prototypes, labels, distance, [f"s-{i}" for i in range(12)])
    assert set(result) == {"kl_uniform", "risk", "margin", "margin_hardness", "risk_rank", "margin_rank", "combined_hardness"}
    assert result["combined_hardness"].shape == (12, 4)
    assert np.isfinite(result["combined_hardness"]).all()


def test_remix_prototype_is_a_detached_anchor():
    model = Candidate12Model(d_model=64, seq_len=5)
    features = torch.randn(3, 4, 64, requires_grad=True)
    output = {"modality_features": features, "unimodal_logits": torch.randn(3, 4, 64)}
    loss, _ = remix_loss(model, output, torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]))
    loss.backward()
    assert features.grad is not None
    assert model.prototype_bank.prototypes.grad is None


def test_pamr_gate_requires_sample_specific_dynamic_gain():
    criteria = {
        "full_top1_plus_0_5pp": True,
        "within3_plus_0_5pp_or_mae_minus_0_05": True,
        "corrected_exceeds_introduced": True,
        "dynamic_beats_mean_0_3pp": False,
        "dynamic_beats_shuffle_0_3pp": False,
        "oracle_beats_anchor_1pp": True,
        "distance_gt5_non_increasing": True,
        "two_of_three_weather_nonworse": True,
    }
    assert sum(criteria.values()) == 6
    assert not pamr_candidate_gate(criteria)
    criteria["dynamic_beats_mean_0_3pp"] = True
    criteria["dynamic_beats_shuffle_0_3pp"] = True
    assert pamr_candidate_gate(criteria)
