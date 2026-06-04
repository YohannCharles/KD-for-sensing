from __future__ import annotations

import torch

from kd_sensing.losses.topk_candidate_losses import (
    TopKCandidateSelectorLoss,
    candidate_circular_soft_target,
)


def test_topk_candidate_loss_components_masks_anchor_and_hard_rank_weighting():
    candidate_beams = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 4, 6, 8, 10, 12, 14]])
    target = torch.tensor([2, 3])
    soft = candidate_circular_soft_target(candidate_beams, target, sigma=1.5, num_beams=16)
    assert soft.shape == (2, 8)
    assert torch.isclose(soft.sum(dim=-1), torch.ones(2), atol=1e-6).all()
    assert soft[1].argmax().item() in {1, 2}

    outputs = {
        "final_candidate_scores": torch.randn(2, 8),
        "candidate_probs": torch.softmax(torch.randn(2, 8), dim=-1),
        "miss_logit": torch.randn(2, 1),
    }
    batch = {
        "candidate_beams": candidate_beams,
        "candidate_probs": torch.softmax(torch.randn(2, 8), dim=-1),
        "target_label": target,
        "target_in_top8": torch.tensor([True, False]),
        "target_candidate_index": torch.tensor([2, -1]),
        "nearest_candidate_index": torch.tensor([2, 1]),
        "miss_label": torch.tensor([0.0, 1.0]),
        "gps_error": torch.tensor([1.0, 6.0]),
        "support_query_role": ["support", "query_test"],
    }
    loss = TopKCandidateSelectorLoss({"hard_rank_weight": 2.0, "num_beams": 16})(outputs, batch)

    assert torch.isfinite(loss["loss"])
    assert torch.isfinite(loss["candidate_soft_ce"])
    assert torch.isfinite(loss["target_index_ce"])
    assert torch.isfinite(loss["miss_bce"])
    assert torch.isfinite(loss["prior_anchor_kl"])
    assert loss["train_sample_count"].item() == 1
    assert loss["target_index_ce_sample_count"].item() == 1
    assert loss["query_label_used_for_training"].item() is False
    assert loss["hard_sample_weight_mean"].item() == 2.0
