import torch
from torch import nn

from kd_sensing.baselines.sparse_pilot_transition import SparsePilotTransitionModel
from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank


class _SensingStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 8
        self.prototype_bank = BeamPrototypeBank(8, num_beams=8)

    def forward(self, features, missing_mask=None):
        del missing_mask
        state = self.prototype_bank.describe(features)
        return {
            "logits": self.prototype_bank(features).unsqueeze(1),
            "output_features": features,
            "prototype_state": state,
        }


class _MustNotRun(nn.Module):
    def forward(self, *_args, **_kwargs):
        raise AssertionError("disabled CSI branch must not run the sparse pilot encoder")


def test_two_stage_wrapper_backpropagates_only_into_new_modules():
    torch.manual_seed(2)
    base = _SensingStub()
    model = SparsePilotTransitionModel(
        base,
        topology_positions=torch.arange(8),
        num_candidate_patterns=6,
        num_selected_patterns=2,
        csi_hidden_dim=16,
        csi_quality_dim=4,
        topology_radius=2,
    )
    sensing = model.sensing_forward({"features": torch.randn(3, 8)})
    result = model.forward_with_candidates(
        sensing,
        torch.randn(3, 6, 4, dtype=torch.complex64),
        frequency_positions=torch.arange(4),
        snr_db=torch.tensor([0.0, 10.0, 20.0]),
        generator=torch.Generator().manual_seed(3),
    )
    torch.nn.functional.nll_loss(result["logits"], torch.tensor([0, 1, 2])).backward()
    assert result["selected_y"].shape == (3, 2, 4)
    assert all(parameter.grad is None for parameter in base.parameters())
    assert model.selector.pilot_logits.grad is not None


def test_two_stage_wrapper_csi_off_is_exact_u0_fallback():
    model = SparsePilotTransitionModel(
        _SensingStub(),
        topology_positions=torch.arange(8),
        num_candidate_patterns=4,
        num_selected_patterns=2,
        csi_hidden_dim=16,
        csi_quality_dim=4,
    )
    sensing = model.sensing_forward({"features": torch.randn(2, 8)})
    result = model.forward_selected(
        sensing,
        torch.randn(2, 2, 3, dtype=torch.complex64),
        pattern_ids=torch.tensor([[0, 1], [2, 3]]),
        frequency_positions=torch.arange(3),
        pilot_mask=torch.ones(2, 2, 3, dtype=torch.bool),
        snr_db=10.0,
        csi_available=torch.zeros(2, dtype=torch.bool),
    )
    assert torch.equal(result["p_final"], sensing["p0"])
    assert torch.equal(result["alpha"], torch.zeros(2))


def test_disabled_csi_switch_bypasses_encoder_and_exactly_reproduces_u0():
    model = SparsePilotTransitionModel(
        _SensingStub(),
        topology_positions=torch.arange(8),
        num_candidate_patterns=4,
        num_selected_patterns=2,
        csi_hidden_dim=16,
        csi_quality_dim=4,
        use_sparse_pilot_csi=False,
        use_prototype_pilot_lookup=False,
        use_dual_route_transition=False,
        use_csi_reliability_gate=False,
    )
    model.csi_encoder = _MustNotRun()
    sensing = model.sensing_forward({"features": torch.randn(2, 8)})
    result = model.forward_with_candidates(
        sensing,
        torch.randn(2, 4, 3, dtype=torch.complex64),
        frequency_positions=torch.arange(3),
        snr_db=10.0,
    )
    assert result["pattern_ids"].tolist() == [[0, 1], [0, 1]]
    assert torch.equal(result["p_final"], sensing["p0"])
    assert torch.equal(result["alpha"], torch.zeros(2))


def test_csi_only_fallback_is_directly_supervised_and_receives_availability():
    torch.manual_seed(7)
    model = SparsePilotTransitionModel(
        _SensingStub(),
        topology_positions=torch.arange(8),
        num_candidate_patterns=4,
        num_selected_patterns=2,
        csi_hidden_dim=16,
        csi_quality_dim=4,
        use_csi_only_fallback=True,
        use_availability_gate=True,
    )
    sensing = model.sensing_forward(
        {"features": torch.randn(2, 8)},
        missing_mask=torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.bool),
    )
    result = model.forward_selected(
        sensing,
        torch.randn(2, 2, 3, dtype=torch.complex64),
        pattern_ids=torch.tensor([[0, 1], [2, 3]]),
        frequency_positions=torch.arange(3),
        pilot_mask=torch.ones(2, 2, 3, dtype=torch.bool),
        snr_db=10.0,
    )
    torch.nn.functional.nll_loss(result["q_csi"].clamp_min(1e-12).log(), torch.tensor([0, 1])).backward()
    assert sensing["sensing_availability"].tolist() == [0.25, 1.0]
    assert torch.equal(result["q_fallback"], result["q_csi"])
    assert model.transition.csi_query.weight.grad is not None
