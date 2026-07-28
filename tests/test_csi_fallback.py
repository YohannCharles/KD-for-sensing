import torch

from kd_sensing.models.prototype_transition import PrototypeTransition
from kd_sensing.models.sparse_pilot_encoder import SparsePilotEncoder


def test_sparse_encoder_consumes_complex_phase_and_reports_low_snr_confidence():
    encoder = SparsePilotEncoder(num_candidate_patterns=8, hidden_dim=32, num_heads=4, num_layers=2, quality_dim=8)
    observations = torch.randn(3, 2, 4, dtype=torch.complex64)
    result = encoder(
        observations,
        torch.tensor([[0, 1], [2, 3], [4, 5]]),
        torch.linspace(-1.0, 1.0, 4),
        torch.ones(3, 2, 4, dtype=torch.bool),
        torch.tensor([-100.0, 0.0, 30.0]),
    )
    assert result["csi_feature"].shape == (3, 32)
    assert result["quality_confidence"][0] < 1e-8
    assert result["quality_confidence"][2] > result["quality_confidence"][1]


def test_csi_missing_is_exact_probability_fallback():
    torch.manual_seed(4)
    batch, prototypes, dim = 3, 8, 16
    transition = PrototypeTransition(
        sensing_dim=dim,
        csi_dim=12,
        quality_dim=7,
        prototype_dim=dim,
        topology_radius=2,
    )
    p0 = torch.softmax(torch.randn(batch, prototypes), dim=-1)
    output = transition(
        torch.randn(batch, dim),
        p0,
        torch.randn(prototypes, dim),
        torch.randn(batch, 12),
        torch.randn(batch, 7),
        torch.arange(prototypes),
        csi_available=torch.zeros(batch, dtype=torch.bool),
        quality_confidence=torch.ones(batch),
    )
    assert torch.equal(output["alpha"], torch.zeros(batch))
    assert torch.equal(output["p_final"], p0)
    assert torch.allclose(output["p_final"].sum(dim=-1), torch.ones(batch))


def test_all_pilot_dropout_forces_encoder_unavailable():
    encoder = SparsePilotEncoder(num_candidate_patterns=4, hidden_dim=16, num_heads=4, quality_dim=4)
    result = encoder(
        torch.zeros(2, 2, 3, dtype=torch.complex64),
        torch.tensor([[0, 1], [2, 3]]),
        torch.arange(3),
        torch.zeros(2, 2, 3, dtype=torch.bool),
        -10.0,
    )
    assert not bool(result["csi_available"].any())
    assert torch.equal(result["quality_confidence"], torch.zeros(2))
