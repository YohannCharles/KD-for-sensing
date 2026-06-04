from __future__ import annotations

import torch

from kd_sensing.models.deepsense6g_residual_fusion import GPSAnchoredResidualFusion


def test_gps_anchored_residual_fusion_forward_formula_and_scale():
    model = GPSAnchoredResidualFusion(
        num_beams=8,
        gps_context_dim=9,
        hidden_dim=16,
        dropout=0.0,
        correction_scale_init=0.5,
        correction_scale_max=3.0,
        use_gate=True,
        use_anchor=True,
    )
    prior = torch.randn(3, 8)
    context = torch.randn(3, 9)

    out = model(gps_prior_logits=prior, gps_context_features=context)

    assert out["final_logits"].shape == (3, 8)
    assert out["correction_logits"].shape == (3, 8)
    assert out["modality_only_logits"].shape == (3, 8)
    assert out["correction_gate"].shape == (3, 1)
    assert out["correction_strength"].shape == (3, 1)
    expected = prior + out["correction_strength"] * out["correction_logits"]
    assert torch.allclose(out["final_logits"], expected, atol=1e-6)
    assert torch.isclose(model.correction_scale, torch.tensor(0.5), atol=1e-5)
    assert float(model.correction_scale.detach()) <= 3.0
    assert "prior_entropy" in out["diagnostics"]
