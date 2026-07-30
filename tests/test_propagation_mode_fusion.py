import pytest
import torch

from kd_sensing.models.propagation_mode_fusion import (
    aggregate_subprototype_evidence,
    fixed_beam_evidence_fusion,
    mode_consistent_fusion,
)


def test_csi_off_is_exact_sensing_fallback() -> None:
    sensing = torch.randn(5, 64)
    csi = torch.randn(5, 64)
    assert torch.equal(
        fixed_beam_evidence_fusion(sensing, csi, csi_available=False),
        sensing.float(),
    )


def test_a7_fuses_modes_before_beam_aggregation() -> None:
    sensing = torch.tensor([[[4.0, 0.0], [1.0, 1.0]]])
    csi = torch.tensor([[[0.0, 4.0], [1.0, 1.0]]])
    evidence, fused_modes = mode_consistent_fusion(sensing, csi, csi_weight=0.5)
    expected_modes = 0.5 * sensing + 0.5 * csi
    expected = aggregate_subprototype_evidence(expected_modes)
    assert torch.equal(fused_modes, expected_modes)
    assert torch.equal(evidence, expected)


def test_a6_independent_classifier_logits_cannot_enter_a7_mode_fusion() -> None:
    with pytest.raises(ValueError):
        mode_consistent_fusion(torch.randn(3, 64), torch.randn(3, 64))


def test_a7_csi_off_keeps_sensing_modes_before_aggregation() -> None:
    sensing = torch.randn(3, 64, 2)
    csi = torch.randn(3, 64, 2)
    evidence, selected = mode_consistent_fusion(sensing, csi, csi_available=False)
    assert torch.equal(selected, sensing.float())
    assert torch.equal(evidence, aggregate_subprototype_evidence(sensing))
