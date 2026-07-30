from __future__ import annotations

from pathlib import Path

import torch
import yaml

from kd_sensing.models.mask_conditioned_prototype_compensation import (
    MaskConditionedPrototypeCompensation,
    TSPC_AVAILABLE_COUNTS,
    TSPC_MASK_NAMES,
)
from tools.run_temporal_sparse_prototype_compensation import balanced_epoch_schedule


ROOT = Path(__file__).resolve().parents[1]


def _evidence(batch: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    sensing = torch.randn(batch, 64, generator=generator)
    radio = torch.randn(batch, 64, generator=generator)
    base = torch.softmax(sensing, dim=-1)
    return sensing, radio, base


def test_tspc_parameter_budgets_and_initial_f1_equivalence():
    sensing, radio, base = _evidence(14)
    mask_ids = torch.arange(14)
    available = torch.ones(14, dtype=torch.bool)
    expected = torch.softmax((sensing / 0.8) + 0.5 * ((radio / 0.2) - (sensing / 0.8)), dim=-1)
    expected_counts = {"M0": 1, "M1": 3, "M2": 14, "M3": 17}
    for method, count in expected_counts.items():
        model = MaskConditionedPrototypeCompensation(
            method,
            initial_weight=0.5,
            sensing_temperature=0.8,
            radio_temperature=0.2,
        )
        actual = model(sensing, radio, mask_ids, available, base_probability=base)["final_probability"]
        assert model.trainable_parameter_count == count
        assert torch.equal(model.lambda_table(), torch.full((14,), 0.5))
        assert torch.max(torch.abs(actual - expected)).item() < 1e-6


def test_tspc_full_and_csi_off_are_exact_probability_bypasses():
    sensing, radio, base = _evidence(6)
    model = MaskConditionedPrototypeCompensation("M3", sensing_temperature=0.7, radio_temperature=0.3)
    full = model(
        sensing[:3],
        radio[:3],
        torch.full((3,), -1),
        torch.ones(3, dtype=torch.bool),
        base_probability=base[:3],
    )
    off = model(
        sensing[3:],
        radio[3:],
        torch.arange(3),
        torch.zeros(3, dtype=torch.bool),
        base_probability=base[3:],
    )
    assert torch.equal(full["final_probability"], base[:3])
    assert torch.equal(off["final_probability"], base[3:])
    assert torch.count_nonzero(full["lambda"]) == 0
    assert torch.count_nonzero(off["lambda"]) == 0


def test_tspc_lambda_uses_only_mask_and_regularizes_hierarchy():
    model = MaskConditionedPrototypeCompensation("M3")
    model.alpha_count.data.copy_(torch.tensor((2.0, 0.0, -2.0)))
    model.delta_mask.data.copy_(torch.linspace(-0.2, 0.2, 14))
    mask_ids = torch.tensor((0, 0, 2, 2, 8, 8))
    values = model.lambdas(mask_ids)
    assert values[0] == values[1]
    assert values[2] == values[3]
    assert values[4] == values[5]
    terms = model.regularization()
    assert float(terms["delta"].detach()) > 0
    assert float(terms["group"].detach()) >= 0
    assert float(terms["severity"].detach()) == 0


def test_tspc_mask_order_and_beam_conditioned_schedule_are_balanced():
    assert len(TSPC_MASK_NAMES) == 14
    assert TSPC_AVAILABLE_COUNTS == (3, 1, 3, 1, 3, 1, 3, 1, 2, 2, 2, 2, 2, 2)
    labels = torch.arange(64).repeat_interleave(29)
    order, masks = balanced_epoch_schedule(labels, epoch=4, seed=1)
    assert torch.equal(torch.sort(order).values, torch.arange(len(labels)))
    for beam in range(64):
        counts = torch.bincount(masks[labels == beam], minlength=14)
        assert int(counts.max() - counts.min()) <= 1


def test_tspc_config_reports_frame_and_window_re_separately():
    config = yaml.safe_load(
        (ROOT / "tools/configs/temporal_sparse_prototype_compensation.yaml").read_text(encoding="utf-8")
    )
    allocations = config["source"]["allocations"]
    assert allocations["temporal_2x2"]["re_per_frame"] == 4
    assert allocations["temporal_2x2"]["history_frames"] == 5
    assert allocations["temporal_2x2"]["re_window"] == 20
    for name in ("single_5x4", "single_4x5"):
        assert allocations[name]["re_per_frame"] == 20
        assert allocations[name]["history_frames"] == 1
        assert allocations[name]["re_window"] == 20
    assert config["protocol"]["outer_test_enabled"] is False
