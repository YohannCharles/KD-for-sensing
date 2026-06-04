from __future__ import annotations

import torch

from kd_sensing.models.beam_candidate_attention import BeamCandidateAttentionReranker


def test_beam_candidate_attention_candidate_union_and_recall_flags():
    reranker = BeamCandidateAttentionReranker(num_beams=8, gps_topk=1, local_radius=2, feature_dim=4, hidden_dim=8)
    logits = torch.full((2, 8), -10.0)
    logits[0, 7] = 5.0
    logits[1, 3] = 5.0
    feature = torch.randn(2, 4)
    target = torch.tensor([1, 7])

    out = reranker(gps_logits=logits, camera_ae_feature=feature, target=target)

    assert set(out["candidates"][0]) >= {5, 6, 7, 0, 1}
    assert all(0 <= beam < 8 for beam in out["candidates"][0])
    assert out["candidate_scores"].shape[0] == 2
    assert out["target_in_gps_top16"].tolist() == [False, False]
    assert out["target_in_local_radius8"].tolist() == [True, False]
    assert out["target_in_union_candidates"].tolist() == [True, False]
    assert torch.isfinite(out["loss"])
