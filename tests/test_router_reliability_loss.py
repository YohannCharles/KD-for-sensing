import pytest
import torch

from kd_sensing.losses.router_reliability import (
    expected_router_utility,
    fused_router_decision_loss,
    paired_router_reliability_loss,
    pairwise_utility_ranking_loss,
)


@pytest.mark.parametrize("objective", ("joint_hard_ce", "power_soft_ce", "power_top1_margin"))
def test_fused_decision_objectives_preserve_router_gradient(objective: str) -> None:
    logits = torch.zeros(2, 4, requires_grad=True)
    labels = torch.tensor([0, 1])
    powers = torch.tensor(
        [[3.0e-8, 2.0e-9, 1.0e-10, 1.0e-12], [1.0e-10, 4.0e-8, 2.0e-9, 1.0e-12]]
    )
    loss, active = fused_router_decision_loss(
        logits,
        labels,
        objective=objective,
        expected_utility=torch.ones(2),
        beam_powers=powers,
        margin_scale=0.5,
        gap_epsilon=0.01,
    )
    loss.backward()
    assert active.item() > 0.0
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0


def test_power_decision_objective_rejects_empty_power_row() -> None:
    with pytest.raises(ValueError, match="at least one positive"):
        fused_router_decision_loss(
            torch.zeros(1, 4),
            torch.tensor([0]),
            objective="power_soft_ce",
            expected_utility=torch.ones(1),
            beam_powers=torch.zeros(1, 4),
            margin_scale=0.5,
            gap_epsilon=0.01,
        )


def test_label_topology_utility_respects_cycle_endpoint() -> None:
    labels = torch.tensor([0])
    logits_near = torch.full((1, 64), -20.0)
    logits_far = logits_near.clone()
    logits_near[0, 63] = 20.0
    logits_far[0, 32] = 20.0
    near = expected_router_utility(
        logits_near,
        labels,
        source="label_topology",
        topology_id="ula_dft_phase_cycle_v1",
        circular=True,
    )
    far = expected_router_utility(
        logits_far,
        labels,
        source="label_topology",
        topology_id="ula_dft_phase_cycle_v1",
        circular=True,
    )
    assert near.item() > far.item()


def test_beam_power_utility_validates_target() -> None:
    with pytest.raises(ValueError, match="future_beam_power"):
        expected_router_utility(torch.zeros(2, 4), torch.zeros(2, dtype=torch.long), source="beam_power")
    with pytest.raises(ValueError, match="non-negative"):
        expected_router_utility(
            torch.zeros(1, 4),
            torch.zeros(1, dtype=torch.long),
            source="beam_power",
            beam_powers=torch.tensor([[1.0, -1.0, 0.0, 0.0]]),
        )


def test_beam_power_utility_preserves_small_linear_power_under_half_logits() -> None:
    logits = torch.tensor([[12.0, -12.0, -12.0, -12.0]], dtype=torch.float16, requires_grad=True)
    powers = torch.tensor([[3.0e-8, 2.0e-10, 1.0e-11, 1.0e-12]], dtype=torch.float32)
    utility = expected_router_utility(
        logits,
        torch.tensor([0]),
        source="beam_power",
        beam_powers=powers,
    )
    assert utility.dtype == torch.float32
    assert utility.item() == pytest.approx(1.0, abs=1.0e-6)
    utility.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_pairwise_ranking_ignores_near_ties_and_has_gradient() -> None:
    scores = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    utility = torch.tensor([[0.9, 0.2, 0.19]])
    loss, active = pairwise_utility_ranking_loss(
        scores,
        utility,
        torch.ones_like(scores, dtype=torch.bool),
        gap_epsilon=0.05,
    )
    loss.backward()
    assert active.item() > 0.0
    assert scores.grad is not None and scores.grad.abs().sum().item() > 0.0


def test_paired_loss_does_not_require_corruption_metadata() -> None:
    batch, modalities, classes = 2, 4, 8
    control_gate = torch.zeros(batch, modalities, requires_grad=True)
    joint_gate = torch.zeros(batch, modalities, requires_grad=True)
    control_unimodal = torch.randn(batch, modalities, classes)
    joint_unimodal = control_unimodal.clone()
    joint_unimodal[:, 0] = torch.roll(joint_unimodal[:, 0], shifts=3, dims=-1)
    available = torch.ones(batch, modalities, dtype=torch.bool)
    control = {
        "router_gate_logits": control_gate,
        "router_gate_weights": torch.softmax(control_gate, dim=1),
        "unimodal_logits": control_unimodal,
        "fused_logits": control_unimodal.mean(dim=1),
        "available": available,
    }
    weights = torch.softmax(joint_gate, dim=1)
    joint = {
        "router_gate_logits": joint_gate,
        "router_gate_weights": weights,
        "unimodal_logits": joint_unimodal,
        "fused_logits": (weights.unsqueeze(-1) * joint_unimodal.detach()).sum(dim=1),
        "available": available,
    }
    loss, diagnostics = paired_router_reliability_loss(
        control,
        joint,
        torch.tensor([0, 1]),
        source="label_topology",
        beam_powers=None,
        beam_temperature=1.0,
        beam_label_sigma=1.0,
        circular=True,
        topology_id="cyclic_index_v1",
        topology_permutation=None,
        fused_decision_objective="expected_utility",
        fused_decision_margin=0.5,
        quality_weight=0.1,
        fused_utility_weight=0.2,
        monotonic_weight=0.1,
        frame_rank_weight=0.0,
        residual_anchor_weight=0.0,
        quality_drop_epsilon=0.001,
        monotonic_margin_scale=0.25,
    )
    loss.backward()
    assert joint_gate.grad is not None and torch.isfinite(joint_gate.grad).all()
    assert diagnostics["loss/router_reliability_total"] >= 0.0
