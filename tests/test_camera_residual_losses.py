from __future__ import annotations

import torch

from kd_sensing.losses.camera_residual_losses import CameraResidualLoss, gate_target_from_gps_error


def test_camera_residual_loss_masks_query_and_anchors_good_samples():
    outputs = {
        "final_logits": torch.randn(3, 8, requires_grad=True),
        "residual_delta_logits": torch.randn(3, 6, requires_grad=True),
        "correction_gate": torch.sigmoid(torch.randn(3, 1, requires_grad=True)),
        "p_gps": torch.softmax(torch.randn(3, 8), dim=-1),
        "direct_beam_logits": torch.randn(3, 8, requires_grad=True),
    }
    batch = {
        "target_label": torch.tensor([1, 2, 3]),
        "gps_error": torch.tensor([1.0, 4.0, 9.0]),
        "residual_delta_class": torch.tensor([3, 5, 2]),
        "gps_prior_probs": outputs["p_gps"],
        "split_role": ["support", "support", "query_test"],
    }
    loss_fn = CameraResidualLoss({"hard_sample_weight": 2.0, "good_error_threshold": 4.0, "circular_sigma": 1.0})

    result = loss_fn(outputs, batch)
    gate_target = gate_target_from_gps_error(batch["gps_error"], batch["split_role"], threshold=4.0)

    assert torch.isfinite(result["loss"])
    assert result["hard_sample_weight_mean"] > 1.0
    assert result["good_anchor_sample_count"].item() == 1.0
    assert gate_target.squeeze(-1).tolist() == [0.0, 1.0, 0.0]
    assert bool(result["query_label_used_for_training"]) is False
