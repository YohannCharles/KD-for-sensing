from __future__ import annotations

import torch

from kd_sensing.models.topk_candidate_selector import (
    TopKCandidateSelector,
    select_final_beams,
    sparse_topk_scores_to_logits,
)


def test_topk_candidate_selector_forward_probs_sparse_logits_and_zero_lambda_ranking():
    candidate_features = torch.randn(2, 8, 10)
    gps_context = torch.randn(2, 14)
    candidate_probs = torch.softmax(torch.randn(2, 8), dim=-1)
    candidate_beams = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0]])
    model = TopKCandidateSelector(topk=8, num_beams=16, hidden_dim=32)

    out = model(candidate_features=candidate_features, gps_context=gps_context, candidate_probs=candidate_probs)

    assert out["final_candidate_scores"].shape == (2, 8)
    assert out["modality_candidate_scores"].shape == (2, 8)
    assert out["candidate_probs"].shape == (2, 8)
    assert out["miss_logit"].shape == (2, 1)
    assert torch.allclose(out["candidate_probs"].sum(dim=-1), torch.ones(2), atol=1e-6)
    selected = select_final_beams(candidate_beams, out["candidate_probs"])
    assert selected.shape == (2,)

    sparse = sparse_topk_scores_to_logits(candidate_beams, out["final_candidate_scores"], num_beams=16)
    assert sparse.shape == (2, 16)
    assert sparse[0, 8] < -1e8
    assert torch.isclose(sparse[0, 0], out["final_candidate_scores"][0, 0])

    gps_locked = TopKCandidateSelector(topk=8, num_beams=16, hidden_dim=32, lambda_init=0.0)
    locked = gps_locked(candidate_features=candidate_features, gps_context=gps_context, candidate_probs=candidate_probs)
    assert torch.equal(locked["candidate_probs"].argmax(dim=-1), candidate_probs.argmax(dim=-1))
