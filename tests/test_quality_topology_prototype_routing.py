from __future__ import annotations

from itertools import combinations
from unittest.mock import Mock

import torch
import torch.nn as nn

from kd_sensing.baselines.mmw_trajectory import ABTC_METHOD, TrajectoryBaselineModel
from kd_sensing.baselines.quality_topology_prototype_routing import QualityTopologyPrototypeRoutingModel
from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank
from kd_sensing.models.dynamic_prototype_fusion import DynamicPrototypeFusion
from kd_sensing.models.prototype_fusion_losses import missing_monotonic_loss, quality_monotonic_loss
from kd_sensing.models.radio_prototype_expert import PositiveTemperature, RadioPrototypeExpert
from kd_sensing.models.radio_trust_estimator import RadioTrustEstimator, build_trust_features
from kd_sensing.models.topology_prototype_gate import TopologyPrototypeGate, prototype_gate_input
from tools.run_quality_topology_prototype_routing import (
    _cached_radio_view,
    _low_re_quality_response,
)


def _masks() -> torch.Tensor:
    values = []
    for count in range(1, 5):
        values.extend(tuple(index in available for index in range(4)) for available in combinations(range(4), count))
    return torch.tensor(values, dtype=torch.bool)


def _quality(batch: int, *, snr: float = 10.0, valid: float = 1.0, consistency: float = 1.0) -> torch.Tensor:
    value = torch.zeros(batch, 21)
    value[:, 16] = snr / 30.0
    value[:, 17] = valid
    value[:, 19] = torch.sigmoid(torch.tensor((snr + 5.0) / 2.0)) * valid
    value[:, 20] = consistency
    return value


def _fusion(method: str = "F5") -> DynamicPrototypeFusion:
    return DynamicPrototypeFusion(
        method,
        labels_by_position=tuple(range(64)),
        gate_hidden_channels=4,
        trust_hidden_dim=8,
    )


def test_positive_temperature_stays_positive():
    temperature = PositiveTemperature(0.1)
    assert float(temperature().detach()) > 0
    temperature.raw.data.fill_(-100.0)
    assert float(temperature().detach()) > 0


def test_cached_radio_view_accepts_generic_low_re_budget():
    radio = {
        "c_radio_2x2": torch.randn(3, 128),
        "csi_quality_2x2": torch.randn(3, 21),
        "csi_available_2x2": torch.ones(3, dtype=torch.bool),
    }
    view = _cached_radio_view(radio, "2x2")
    assert view["c_radio"] is radio["c_radio_2x2"]
    assert view["csi_quality"] is radio["csi_quality_2x2"]
    assert view["csi_available"] is radio["csi_available_2x2"]


def test_low_re_quality_gate_rejects_constant_trust(tmp_path):
    (tmp_path / "snr_summary.csv").write_text(
        "method,snr_db,rho_mean\nF2,30,0.42427\nF2,-10,0.42426\n",
        encoding="utf-8",
    )
    (tmp_path / "dropout_summary.csv").write_text(
        "method,pilot_dropout,rho_mean\nF2,0.0,0.42427\nF2,0.5,0.42426\n",
        encoding="utf-8",
    )
    result = _low_re_quality_response(tmp_path, "F2", minimum_delta=0.05)
    assert result["passed"] is False
    assert result["snr_rho_delta"] < 0.05
    assert result["dropout_rho_delta"] < 0.05


def test_training_teacher_factorization_preserves_argmax_through_shared_bank():
    torch.manual_seed(1)
    bank = BeamPrototypeBank(64, 64, temperature=0.1)
    teacher = nn.Sequential(
        nn.LayerNorm(128),
        nn.Linear(128, 128),
        nn.GELU(),
        nn.Linear(128, 64),
    ).eval()
    expert = RadioPrototypeExpert().eval()
    audit = expert.initialize_from_teacher(bank, teacher.state_dict())
    radio = torch.randn(128, 128)
    expected = teacher(radio)
    actual = expert(radio, bank)["radio_evidence"]
    assert audit["prototype_rank"] == 64
    assert not bool(expected.argmax(dim=-1).ne(actual.argmax(dim=-1)).any())
    assert not any("classifier" in name for name in expert.state_dict())


def test_radio_prototype_query_has_fp32_precision_fence_inside_autocast():
    torch.manual_seed(11)
    bank = BeamPrototypeBank(64, 64, temperature=0.1)
    expert = RadioPrototypeExpert().eval()
    radio = torch.randn(16, 128)
    expected = expert(radio, bank)["radio_evidence"]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = expert(radio, bank)["radio_evidence"]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_structured_trust_initially_increases_with_missingness_and_real_quality():
    torch.manual_seed(2)
    bank = BeamPrototypeBank(64, 64)
    embedding = torch.randn(4, 64)
    sensing = bank(embedding)
    radio = sensing + torch.randn_like(sensing) * 0.1
    distance = torch.minimum(
        (torch.arange(64)[:, None] - torch.arange(64)[None]).abs(),
        64 - (torch.arange(64)[:, None] - torch.arange(64)[None]).abs(),
    ).float()
    estimator = RadioTrustEstimator(hidden_dim=8, structured=True).eval()
    severe = torch.tensor([[1, 0, 0, 0]] * 4, dtype=torch.bool)
    mild = torch.tensor([[1, 1, 1, 0]] * 4, dtype=torch.bool)
    clean = _quality(4, snr=20.0, valid=1.0, consistency=1.0)
    degraded = _quality(4, snr=-10.0, valid=0.25, consistency=0.1)
    severe_stats = build_trust_features(severe, embedding, sensing, radio, clean, bank.prototypes, distance)
    mild_stats = build_trust_features(mild, embedding, sensing, radio, clean, bank.prototypes, distance)
    degraded_stats = build_trust_features(severe, embedding, sensing, radio, degraded, bank.prototypes, distance)
    available = torch.ones(4, dtype=torch.bool)
    rho_severe = estimator(severe_stats, available)["rho"]
    rho_mild = estimator(mild_stats, available)["rho"]
    rho_degraded = estimator(degraded_stats, available)["rho"]
    assert bool((rho_severe > rho_mild).all())
    assert bool((rho_severe > rho_degraded).all())
    assert float(missing_monotonic_loss(rho_severe, rho_mild).detach()) == 0.0
    assert float(quality_monotonic_loss(rho_severe, rho_degraded).detach()) == 0.0


def test_topology_gate_connects_endpoint_beams_with_circular_padding():
    gate = TopologyPrototypeGate(num_beams=64, hidden_channels=2, kernel_size=3, labels_by_position=range(64))
    gate.first.weight.data.fill_(1.0)
    gate.first.bias.data.zero_()
    gate.second.weight.data.fill_(1.0)
    gate.second.bias.data.zero_()
    values = torch.zeros(1, 6, 64, requires_grad=True)
    output = gate(values, torch.ones(1))["prototype_gate"]
    output[0, 0].backward()
    assert float(values.grad[0, :, 63].abs().sum()) > 0
    assert gate.first.weight.shape[-1] == 3
    assert not any(parameter.shape[0] == 64 for parameter in gate.parameters())


def test_topology_gate_uses_declared_uniform_radio_prior():
    gate = TopologyPrototypeGate(num_beams=64, hidden_channels=2, initial_probability=0.9).eval()
    output = gate(torch.randn(3, 6, 64), torch.ones(3))["prototype_gate"]
    torch.testing.assert_close(output, torch.full_like(output, 0.9), rtol=1e-6, atol=1e-6)


def test_csi_unavailable_exactly_restores_uncalibrated_missing_m4_evidence():
    torch.manual_seed(3)
    bank = BeamPrototypeBank(64, 64)
    model = _fusion().eval()
    embedding = torch.randn(3, 64)
    sensing = bank(embedding)
    output = model(
        embedding,
        sensing,
        torch.randn(3, 128),
        _quality(3),
        torch.zeros(3, dtype=torch.bool),
        torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool),
        bank,
        torch.ones(64, 64),
    )
    torch.testing.assert_close(output["final_evidence"], sensing, rtol=0, atol=0)
    assert not bool(output["rho"].count_nonzero())
    assert not bool(output["prototype_gate"].count_nonzero())


def test_full_batch_bypasses_radio_and_fusion_exactly():
    torch.manual_seed(4)
    base = TrajectoryBaselineModel(ABTC_METHOD, dropout=0.0).eval()
    fusion = _fusion().eval()
    fusion.forward = Mock(side_effect=AssertionError("fusion must not run"))
    radio = nn.Identity()
    radio.forward = Mock(side_effect=AssertionError("radio must not run"))
    model = QualityTopologyPrototypeRoutingModel(
        base,
        fusion,
        topology_distance=torch.ones(64, 64),
        radio_encoder=radio,
        freeze_radio=False,
    ).eval()
    sequence = torch.randn(2, 5, 4, 64)
    full = torch.ones(2, 4, dtype=torch.bool)
    expected = base.forward_tokens(model._token_mapping(sequence), availability=full)["logits"].softmax(dim=-1)
    actual = model(sequence, full, radio_inputs={"unused": torch.ones(2)})
    torch.testing.assert_close(actual["probabilities"], expected, rtol=0, atol=0)
    assert actual["fusion_bypassed"] is True
    assert not bool(actual["radio_called"].any())
    assert not bool(actual["pilot_re"].any())
    assert fusion.forward.call_count == 0
    assert radio.forward.call_count == 0


def test_mixed_all_masks_forward_backward_preserves_full_and_has_no_classifier_path():
    torch.manual_seed(5)
    base = TrajectoryBaselineModel(ABTC_METHOD, dropout=0.0).eval()
    fusion = _fusion().train()
    model = QualityTopologyPrototypeRoutingModel(
        base,
        fusion,
        topology_distance=torch.ones(64, 64),
        radio_encoder=None,
    ).train()
    masks = _masks()
    sequence = torch.randn(len(masks), 5, 4, 64)
    radio = {
        "c_radio": torch.randn(len(masks), 128),
        "csi_quality": _quality(len(masks)),
        "csi_available": torch.ones(len(masks), dtype=torch.bool),
    }
    expected_full = base.forward_tokens(model._token_mapping(sequence), availability=masks)["logits"][-1:].softmax(dim=-1)
    output = model(sequence, masks, radio_output=radio)
    torch.testing.assert_close(output["probabilities"][-1:], expected_full, rtol=0, atol=0)
    output["logits"][:-1].sum().backward()
    assert any(parameter.grad is not None for parameter in fusion.parameters() if parameter.requires_grad)
    assert not any("classifier" in name for name in fusion.state_dict())
    assert int(output["pilot_re"][-1]) == 0


def test_prototype_gate_input_has_declared_six_channels():
    values = prototype_gate_input(torch.randn(2, 64), torch.randn(2, 64))
    assert values.shape == (2, 6, 64)
