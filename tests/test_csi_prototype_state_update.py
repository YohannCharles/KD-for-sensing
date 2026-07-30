from __future__ import annotations

import inspect

import pytest
import torch
import yaml

from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank
from kd_sensing.losses.prototype_update_losses import prototype_update_loss
from kd_sensing.models.csi_conditioned_prototype_update import CSIConditionedPrototypeUpdate
from kd_sensing.models.prototype_likelihood_head import PrototypeLikelihoodHead, estimate_train_beam_prior
from kd_sensing.models.prototype_posterior_update import PrototypePosteriorUpdate, topology_transition
from kd_sensing.models.prototype_transition_kernel import PrototypeTransitionKernel, transition_context
from kd_sensing.models.radio_prototype_expert import RadioPrototypeExpert
from tools.run_csi_prototype_state_update import (
    MASK_NAMES,
    _diagnostic_sample_rows,
    _gather_mask_tensor,
    _load_config,
    _retained_m4_evidence,
)


def _bank() -> BeamPrototypeBank:
    bank = BeamPrototypeBank(64, 64, temperature=0.1)
    with torch.no_grad():
        bank.prototypes.copy_(torch.eye(64))
    bank.prototypes.requires_grad_(False)
    return bank


def _model(*, transition: bool = True, likelihood: bool = True) -> CSIConditionedPrototypeUpdate:
    expert = RadioPrototypeExpert(radio_dim=128, hidden_dim=128, prototype_dim=64, temperature=0.2)
    head = PrototypeLikelihoodHead(expert, torch.arange(1, 65, dtype=torch.float32), eta_prior=1.0)
    kernel = PrototypeTransitionKernel(hidden_dim=128, radius=3, identity_initial_mass=0.1)
    return CSIConditionedPrototypeUpdate(
        head,
        kernel,
        labels_by_position=tuple(range(64)),
        circular_topology=True,
        sensing_temperature=0.8,
        transition_enabled=transition,
        likelihood_enabled=likelihood,
    )


def test_train_prior_is_normalized_and_rejects_non_train_labels():
    labels = torch.arange(64).repeat_interleave(torch.arange(1, 65))
    result = estimate_train_beam_prior(labels, split_role="train")
    assert result["sample_count"] == int(torch.arange(1, 65).sum())
    assert result["counts"].tolist() == list(range(1, 65))
    assert torch.allclose(result["prior"].sum(), torch.tensor(1.0))
    for role in ("validation", "test", "outer_test"):
        with pytest.raises(ValueError, match="train split"):
            estimate_train_beam_prior(labels, split_role=role)


def test_likelihood_prior_correction_uses_calibrated_radio_probability_once():
    expert = RadioPrototypeExpert(radio_dim=128, hidden_dim=128, prototype_dim=64, temperature=0.2)
    prior = torch.arange(1, 65, dtype=torch.float32)
    head = PrototypeLikelihoodHead(expert, prior, eta_prior=0.5, eps=1e-8)
    radio = torch.randn(3, 128)
    output = head(radio, _bank())
    expected = output["radio_probability"].clamp_min(1e-8).log()
    expected = expected - 0.5 * head.train_prior.clamp_min(1e-8).log()[None]
    assert output["radio_probability"].dtype == torch.float32
    assert torch.allclose(output["radio_probability"].sum(dim=-1), torch.ones(3), atol=1e-6)
    assert torch.equal(output["log_likelihood_ratio"], expected)


def test_retained_m4_probability_recovers_single_temperature_prior():
    logits = torch.randn(7, 64)
    probability = torch.softmax(logits, dim=-1)
    temperature = 0.8330621
    recovered = torch.softmax(_retained_m4_evidence(probability) / temperature, dim=-1)
    expected = torch.softmax(logits / temperature, dim=-1)
    assert torch.allclose(recovered, expected, atol=1e-6)


def test_diagnostic_samples_are_unique_random_sample_mask_pairs():
    selected = _diagnostic_sample_rows(samples=6365, masks=14, limit=200, seed=760002)
    flat = [mask_id * 6365 + int(row) for mask_id, rows in selected.items() for row in rows]
    assert len(flat) == len(set(flat)) == 200
    assert len([rows for rows in selected.values() if len(rows)]) > 1
    assert min(flat) >= 0 and max(flat) < 6365 * 14


def test_transition_context_uses_ordered_frame_deltas():
    frames = torch.arange(2 * 5 * 128, dtype=torch.float32).reshape(2, 5, 128)
    hidden = torch.randn(2, 128)
    context = transition_context(frames, hidden)
    delta = frames[:, 1:] - frames[:, :-1]
    expected = torch.cat((hidden, frames[:, -1] - frames[:, 0], delta.mean(1), delta.std(1, unbiased=False)), dim=-1)
    assert context.shape == (2, 512)
    assert torch.equal(context, expected)
    assert not torch.equal(context, transition_context(frames.flip(1), hidden))


@pytest.mark.parametrize("mode,context_dim", (("temporal", 512), ("last", 128), ("no_delta", 128), ("static", 0)))
def test_transition_kernel_is_local_normalized_and_identity_biased(mode: str, context_dim: int):
    kernel = PrototypeTransitionKernel(context_mode=mode, radius=3, identity_initial_mass=0.1)
    output = kernel(torch.randn(4, 5, 128), torch.randn(4, 128))
    assert output["transition_context"].shape == (4, context_dim)
    assert output["q_delta"].shape == output["q_final"].shape == (4, 7)
    assert torch.allclose(output["q_delta"].sum(dim=-1), torch.ones(4), atol=1e-7)
    assert torch.allclose(output["q_final"].sum(dim=-1), torch.ones(4), atol=1e-7)
    assert bool((output["identity_mass"] >= 0.9).all())


def test_circular_and_linear_topology_transitions_have_expected_boundaries():
    prior = torch.zeros(1, 64)
    prior[0, 63] = 1.0
    shift_right = torch.tensor([[0.0, 0.0, 1.0]])
    circular = topology_transition(prior, shift_right, labels_by_position=range(64), circular=True)
    assert circular[0, 0] == 1.0

    linear_prior = torch.zeros(1, 64)
    linear_prior[0, 0] = 1.0
    linear = topology_transition(linear_prior, shift_right, labels_by_position=range(64), circular=False)
    assert linear[0, 1] == 1.0
    truncated = topology_transition(linear_prior, torch.tensor([[0.5, 0.5, 0.0]]), circular=False)
    assert torch.allclose(truncated.sum(dim=-1), torch.ones(1))
    assert truncated[0, 0] == 1.0


def test_identity_transition_and_posterior_update_are_stable_fp32():
    prior = torch.softmax(torch.randn(5, 64), dim=-1)
    identity = torch.zeros(5, 7)
    identity[:, 3] = 1.0
    moved = topology_transition(prior, identity, labels_by_position=range(64), circular=True)
    assert torch.equal(moved, prior)
    updater = PrototypePosteriorUpdate(labels_by_position=tuple(range(64)), circular=True)
    output = updater(prior, identity, torch.randn(5, 64).mul(1000).half(), beta=2.0)
    assert output["p_final"].dtype == torch.float32
    assert torch.isfinite(output["log_posterior"]).all()
    assert torch.allclose(output["p_pred"].sum(dim=-1), torch.ones(5), atol=1e-6)
    assert torch.allclose(output["p_final"].sum(dim=-1), torch.ones(5), atol=1e-6)


def test_csi_off_and_full_rows_are_exact_bypasses():
    model = _model()
    evidence = torch.randn(4, 64)
    base = torch.softmax(torch.randn(4, 64), dim=-1)
    output = model(
        evidence,
        torch.randn(4, 128),
        torch.randn(4, 5, 128),
        _bank(),
        torch.zeros(4, dtype=torch.bool),
        full=torch.tensor((False, False, True, True)),
        full_probability=base,
    )
    expected_prior = torch.softmax(evidence.float() / 0.8, dim=-1)
    assert torch.equal(output["p_final"][:2], expected_prior[:2])
    assert torch.equal(output["p_final"][2:], base[2:])
    assert torch.count_nonzero(output["pilot_re"]) == 0
    assert torch.equal(output["q_final"], torch.nn.functional.one_hot(torch.full((4,), 3), 7).float())


def test_composed_update_normalizes_prior_and_backpropagates_transition_loss():
    model = _model(likelihood=False)
    evidence = torch.randn(6, 64)
    output = model(
        evidence,
        torch.randn(6, 128),
        torch.randn(6, 5, 128),
        _bank(),
        torch.ones(6, dtype=torch.bool),
    )
    assert torch.allclose(output["p_s"].sum(dim=-1), torch.ones(6), atol=1e-7)
    assert torch.allclose(output["p_pred"].sum(dim=-1), torch.ones(6), atol=1e-6)
    distance = torch.arange(64).sub(torch.arange(64)[:, None]).abs().float()
    distance = torch.minimum(distance, 64.0 - distance)
    terms = prototype_update_loss(
        output,
        torch.arange(6),
        distance,
        weights={"topology": 0.2, "transition_identity": 0.05, "transition_local": 0.01},
        low_quality_weight=torch.linspace(0, 1, 6),
    )
    terms["total"].backward()
    assert any(parameter.grad is not None for parameter in model.transition_kernel.parameters())


def test_forward_contract_exposes_no_future_channel_input():
    parameters = inspect.signature(CSIConditionedPrototypeUpdate.forward).parameters
    assert all("future" not in name and "target" not in name for name in parameters)


def test_runner_rotates_all_14_missing_masks_without_collapsing_rows():
    assert len(MASK_NAMES) == 14
    records = {f"z_{name}": torch.full((14, 64), float(mask_id)) for mask_id, name in enumerate(MASK_NAMES)}
    gathered = _gather_mask_tensor(records, "z", torch.arange(14), torch.arange(14))
    assert gathered.shape == (14, 64)
    assert torch.equal(gathered[:, 0], torch.arange(14, dtype=torch.float32))


def test_runner_rejects_outer_test_enabled_config(tmp_path):
    config = {
        "protocol": {"outer_test_enabled": True},
        "pilot": {"re_per_frame": 4, "history_frames": 5, "re_window": 20, "budget": "2x2"},
    }
    path = tmp_path / "outer-enabled.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="outer_test_enabled=false"):
        _load_config(path)
