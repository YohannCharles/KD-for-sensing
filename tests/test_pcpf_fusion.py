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
    )

    assert weights.dtype == torch.float32
    assert weights[~available].eq(0).all()
    assert weights[1, 1].item() == 1.0
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
    assert torch.isfinite(weights).all()
