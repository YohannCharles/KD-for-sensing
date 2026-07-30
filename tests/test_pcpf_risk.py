import math

import torch

from kd_sensing.losses.pcpf_temporal_risk import topology_risk_target
from kd_sensing.models.pcpf_temporal_risk import topology_risk_components


def _circular_distribution(indices: list[int]) -> torch.Tensor:
    probability = torch.zeros(1, len(indices), 1, 64)
    for index, beam in enumerate(indices):
        probability[0, index, 0, beam] = 1.0
    return probability


def test_topology_risk_target_detaches_and_wraps_zero_sixty_three() -> None:
    logits = torch.full((1, 1, 64), -20.0, requires_grad=True)
    with torch.no_grad():
        logits[0, 0, 63] = 20.0
    probability = torch.softmax(logits, dim=-1)

    target = topology_risk_target(probability, torch.tensor([[0]]), torch.tensor([[True]]))

    assert not target.requires_grad
    torch.testing.assert_close(target, torch.tensor([[1.0 / 32.0]]), atol=1e-5, rtol=0)


def test_temporal_residual_does_not_treat_wrap_as_full_circle_jump() -> None:
    frames = _circular_distribution([62, 63, 0, 1, 2])
    features = torch.zeros(1, 5, 1, 64)
    features[0, :, 0] = frames[0, :, 0]
    probability = frames[:, -1].expand(-1, 1, -1)
    result = topology_risk_components(
        mu=features[:, -1],
        logvar=torch.full((1, 1, 64), -4.0),
        frame_features=features,
        temporal_mask=torch.ones(1, 5, 1, dtype=torch.bool),
        probabilities=probability,
        prototypes=torch.eye(64),
        prototype_temperature=0.01,
        topology_positions=torch.arange(64, dtype=torch.float32),
    )

    assert result["temp_valid"].item()
    assert result["components"][0, 0, 2].item() < 1e-5


def test_risk_components_are_fp32_and_handle_short_history_and_single_modality() -> None:
    mu = torch.randn(2, 4, 64, dtype=torch.bfloat16)
    logvar = torch.full_like(mu, -4.0)
    frames = torch.randn(2, 5, 4, 64, dtype=torch.bfloat16)
    mask = torch.zeros(2, 5, 4, dtype=torch.bool)
    mask[:, :2, 0] = True
    probability = torch.zeros(2, 4, 64)
    probability[:, 0] = torch.softmax(torch.randn(2, 64), dim=-1)

    result = topology_risk_components(
        mu=mu,
        logvar=logvar,
        frame_features=frames,
        temporal_mask=mask,
        probabilities=probability,
        prototypes=torch.randn(64, 64, dtype=torch.bfloat16),
        prototype_temperature=0.2,
        topology_positions=torch.arange(64, dtype=torch.float32),
    )

    assert result["components"].dtype == torch.float32
    assert result["components"][:, 0, 2].eq(0).all()
    assert result["components"][:, 0, 3].eq(0).all()
    assert torch.isfinite(result["components"]).all()
    assert math.isfinite(result["components"][:, 0, 0].mean().item())
