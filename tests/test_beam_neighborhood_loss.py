import torch
import torch.nn.functional as F

from kd_sensing.losses.beam import BeamNeighborhoodCrossEntropyLoss, LabelSmoothingCrossEntropyLoss
from kd_sensing.losses.u_mask_beam_jepa import u_mask_beam_jepa_loss


def test_beam_neighborhood_loss_soft_targets_are_circular_and_normalized():
    criterion = BeamNeighborhoodCrossEntropyLoss(sigma=1.5, circular=True, mix_ce=0.5)
    logits = torch.zeros(3, 8)
    targets = torch.tensor([0, 3, -100])

    soft = criterion.soft_targets(logits, targets)

    assert torch.allclose(soft[:2].sum(dim=-1), torch.ones(2))
    assert soft[2].sum().item() == 0.0
    assert soft[0, 7] == soft[0, 1]
    assert soft[0, 1] > soft[0, 4]


def test_beam_neighborhood_loss_mix_ce_matches_definition():
    criterion = BeamNeighborhoodCrossEntropyLoss(sigma=1.0, circular=False, mix_ce=0.25)
    logits = torch.tensor([[3.0, 0.0, -1.0, -2.0], [0.0, 2.0, 1.0, -1.0]], requires_grad=True)
    targets = torch.tensor([0, 2])

    loss = criterion(logits, targets)
    soft = criterion.soft_targets(logits, targets)
    hard = F.cross_entropy(logits, targets)
    soft_loss = -(soft * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    assert torch.allclose(loss, 0.75 * hard + 0.25 * soft_loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_label_smoothing_ce_matches_torch_cross_entropy():
    logits = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 2.0]], requires_grad=True)
    targets = torch.tensor([0, 2])
    criterion = LabelSmoothingCrossEntropyLoss(smoothing=0.05)

    loss = criterion(logits, targets)
    expected = F.cross_entropy(logits, targets, label_smoothing=0.05)

    assert torch.allclose(loss, expected)


def test_u_mask_loss_can_consume_beam_neighborhood_criterion():
    logits = torch.tensor([[[2.0, 0.0, -1.0, -2.0]], [[0.0, 1.0, 2.0, -1.0]]], requires_grad=True)
    labels = torch.tensor([[0], [2]])
    criterion = BeamNeighborhoodCrossEntropyLoss(sigma=1.0, circular=True, mix_ce=0.5)

    result = u_mask_beam_jepa_loss(
        {"logits": logits},
        labels,
        use_teacher=False,
        use_jepa_loss=False,
        beam_criterion=criterion,
    )

    expected = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    assert torch.allclose(result["loss_beam"], expected)
