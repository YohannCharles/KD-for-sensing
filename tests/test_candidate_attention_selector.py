from __future__ import annotations

import torch

from kd_sensing.models.candidate_attention_selector import CandidateAttentionSelector


def test_candidate_attention_selector_camera_gps_and_image_tokens_shape():
    model = CandidateAttentionSelector(topk=8, num_beams=16, hidden_dim=32, num_heads=4)
    out = model(
        candidate_features=torch.randn(2, 8, 10),
        gps_context=torch.randn(2, 14),
        candidate_probs=torch.softmax(torch.randn(2, 8), dim=-1),
        camera_ae_feature=torch.randn(2, 12),
        image_tokens=torch.randn(2, 3, 20),
    )

    assert out["final_candidate_scores"].shape == (2, 8)
    assert out["candidate_probs"].shape == (2, 8)
    assert out["miss_logit"].shape == (2, 1)
    assert torch.allclose(out["candidate_probs"].sum(dim=-1), torch.ones(2), atol=1e-6)
    assert "camera_ae" in out["diagnostics"]["enabled_modalities"]
    assert "image" in out["diagnostics"]["enabled_modalities"]
