import torch

from kd_sensing.models.pcpf_temporal_risk import analytic_fusion_weights


def test_analytic_fusion_masks_missing_and_normalizes_rows() -> None:
    risk = torch.tensor([[0.2, 0.3, 1000.0, 0.5], [1000.0, 0.1, 1000.0, 1000.0]])
    available = torch.tensor([[True, True, False, True], [False, True, False, False]])

    weights = analytic_fusion_weights(
        risk=risk,
        available=available,
        static_capability=torch.tensor([1.0, 0.8, 0.5, 0.4]),
        tau=torch.tensor(0.05),
        max_log_adjustment=0.75,
    )

    assert weights.dtype == torch.float32
    assert weights[~available].eq(0).all()
    assert weights[1, 1].item() == 1.0
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
    assert torch.isfinite(weights).all()


def test_dynamic_log_odds_stay_within_static_anchor_bound() -> None:
    risk = torch.tensor([[0.0, 100.0, 0.5]])
    available = torch.ones_like(risk, dtype=torch.bool)
    capability = torch.tensor([0.8, 0.2, 0.5])
    gamma = 0.6

    weights = analytic_fusion_weights(
        risk=risk,
        available=available,
        static_capability=capability,
        tau=0.01,
        max_log_adjustment=gamma,
    )

    dynamic_log_odds = torch.log(weights[0, :, None] / weights[0, None, :])
    static_log_odds = torch.log(capability[:, None] / capability[None, :])
    assert (dynamic_log_odds - static_log_odds).abs().max().item() <= 2.0 * gamma + 1e-6
