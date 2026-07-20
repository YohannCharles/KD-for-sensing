import pytest
import torch
import torch.nn.functional as F

from kd_sensing.losses.beam_prototype_alignment import make_soft_beam_labels
from kd_sensing.losses.pcer_temporal_fusion import counterfactual_router_targets
from kd_sensing.models.pcer_temporal_fusion import masked_block_softmax


def _targets(evidence: torch.Tensor, availability: torch.Tensor, label: int = 0):
    return counterfactual_router_targets(
        evidence,
        availability,
        torch.tensor([label]),
        beam_label_sigma=0.2,
        circular=True,
        topology_id="cyclic_index_v1",
        topology_permutation=None,
        contribution_temperature=0.5,
        contribution_clip=None,
    )


def _peaked(blocks: int, classes: int, peaks: list[int], strengths: list[float] | None = None) -> torch.Tensor:
    evidence = torch.zeros(1, blocks, classes)
    for index, peak in enumerate(peaks):
        evidence[0, index, peak] = strengths[index] if strengths is not None else 12.0
    return evidence


def test_unique_correct_block_has_largest_contribution_and_target() -> None:
    evidence = _peaked(4, 8, [0, 4, 4, 4])
    target, contribution = _targets(evidence, torch.ones(1, 4, dtype=torch.bool))
    assert contribution.argmax(dim=-1).item() == 0
    assert target.argmax(dim=-1).item() == 0


def test_unique_harmful_block_has_smallest_contribution_and_target() -> None:
    evidence = _peaked(4, 8, [0, 4, 0, 0], [10.0, 30.0, 10.0, 10.0])
    target, contribution = _targets(evidence, torch.ones(1, 4, dtype=torch.bool))
    assert contribution.argmin(dim=-1).item() == 1
    assert contribution[0, 1] < 0
    assert target.argmin(dim=-1).item() == 1
    assert target[0, 1] < 0.05


def test_redundant_blocks_produce_uniform_target() -> None:
    evidence = _peaked(4, 8, [0, 0, 0, 0])
    target, contribution = _targets(evidence, torch.ones(1, 4, dtype=torch.bool))
    assert torch.allclose(contribution, torch.zeros_like(contribution), atol=1e-6)
    assert torch.allclose(target, torch.full_like(target, 0.25), atol=1e-6)


def test_availability_masks_target_and_prediction_on_both_sides() -> None:
    evidence = _peaked(4, 8, [0, 4, 4, 4])
    availability = torch.tensor([[False, True, True, True]])
    target, _ = _targets(evidence, availability)
    prediction = masked_block_softmax(torch.tensor([[20.0, 3.0, 2.0, 1.0]]), availability)
    assert target[0, 0].item() == pytest.approx(0.0)
    assert prediction[0, 0].item() == pytest.approx(0.0)
    assert target.sum().item() == pytest.approx(1.0)
    assert prediction.sum().item() == pytest.approx(1.0)


def test_time_major_flatten_mask_leave_one_out_and_reshape_keep_indices() -> None:
    timesteps, modalities, classes = 3, 2, 5
    identifiers = torch.arange(timesteps * modalities).reshape(timesteps, modalities)
    evidence_tm = torch.zeros(1, timesteps, modalities, classes)
    for time in range(timesteps):
        for modality in range(modalities):
            flat_index = int(identifiers[time, modality])
            evidence_tm[0, time, modality, flat_index % classes] = float(flat_index + 1)
    availability_tm = torch.tensor([[[True, False], [True, True], [False, True]]])
    flat_evidence = evidence_tm.reshape(1, timesteps * modalities, classes)
    flat_availability = availability_tm.reshape(1, timesteps * modalities)
    target, contribution = _targets(flat_evidence, flat_availability, label=2)

    valid = flat_availability.float()
    count = valid.sum(dim=1)
    evidence_sum = (flat_evidence * valid.unsqueeze(-1)).sum(dim=1)
    all_logits = evidence_sum / count.unsqueeze(-1)
    loo_logits = (evidence_sum.unsqueeze(1) - flat_evidence) / (count - 1).view(1, 1, 1)
    soft_label = make_soft_beam_labels(
        torch.tensor([2]), classes, 0.2, circular=True, topology_id="cyclic_index_v1"
    )
    loss_all = -(soft_label * F.log_softmax(all_logits, dim=-1)).sum(dim=-1)
    raw = -(soft_label.unsqueeze(1) * F.log_softmax(loo_logits, dim=-1)).sum(dim=-1) - loss_all.unsqueeze(1)
    centered = raw - (raw * valid).sum(dim=1, keepdim=True) / count.unsqueeze(1)

    assert identifiers.reshape(-1).tolist() == [0, 1, 2, 3, 4, 5]
    assert flat_availability.tolist() == [[True, False, True, True, False, True]]
    assert torch.equal(flat_evidence.reshape_as(evidence_tm), evidence_tm)
    assert torch.allclose(contribution[flat_availability], centered[flat_availability], atol=1e-6)
    assert torch.equal(target.masked_select(~flat_availability), torch.zeros(2))
