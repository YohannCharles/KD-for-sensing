from __future__ import annotations

import torch

from kd_sensing.models.deepsense6g_residual_fusion import GPSAnchoredTopKReranker


def test_topk_reranker_candidate_wraparound_and_loss_mask():
    reranker = GPSAnchoredTopKReranker(num_beams=8, gps_top_k=1, local_radius=2, modality_top_m=1)
    logits = torch.full((2, 8), -10.0)
    logits[0, 7] = 5.0
    logits[1, 3] = 5.0
    target = torch.tensor([1, 7])

    out = reranker(logits, target=target)

    assert set(out["candidates"][0]) >= {5, 6, 7, 0, 1}
    assert all(0 <= beam < 8 for beam in out["candidates"][0])
    assert out["target_in_union_candidates"].tolist() == [True, False]
    assert out["loss_mask"].tolist() == [True, False]
    assert torch.isfinite(out["loss"])
