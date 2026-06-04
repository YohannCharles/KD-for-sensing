from __future__ import annotations

import torch

from kd_sensing.models.camera_residual_fusion import CameraGPSResidualFusion, synthesize_correction_distribution


def test_camera_residual_fusion_forward_and_gated_formula():
    model = CameraGPSResidualFusion(
        num_beams=8,
        gps_context_dim=9,
        camera_feature_dim=16,
        hidden_dim=24,
        delta_radius=2,
        dropout=0.0,
        gate_bias_init=-2.0,
    )
    prior = torch.randn(3, 8)
    pred = torch.tensor([7, 3, 0])
    context = torch.randn(3, 9)
    feature = torch.randn(3, 16)

    out = model(gps_prior_logits=prior, gps_pred_top1=pred, gps_context=context, camera_ae_feature=feature)

    assert out["residual_delta_logits"].shape == (3, 6)
    assert out["correction_gate"].shape == (3, 1)
    assert out["p_corr"].shape == (3, 8)
    assert out["final_logits"].shape == (3, 8)
    assert torch.allclose(out["p_corr"].sum(dim=-1), torch.ones(3), atol=1e-6)
    expected = (1.0 - out["correction_gate"]) * out["p_gps"] + out["correction_gate"] * out["p_corr"]
    assert torch.allclose(out["final_logits"].exp(), expected / expected.sum(dim=-1, keepdim=True), atol=1e-6)
    assert torch.all(model.gate_head.bias.detach() < 0)


def test_synthesize_correction_distribution_wraps_delta():
    logits = torch.full((1, 6), -10.0)
    logits[0, 4] = 10.0  # radius=2, class 4 -> delta +2
    out = synthesize_correction_distribution(logits, torch.tensor([7]), num_beams=8, delta_radius=2)

    assert int(out.argmax(dim=-1).item()) == 1
    assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-6)
