from __future__ import annotations

import torch

from kd_sensing.losses.residual import ResidualFusionLoss, gate_target_from_gps_error


def test_residual_loss_masks_query_gate_and_weights_hard_samples():
    outputs = {
        "final_logits": torch.randn(3, 8, requires_grad=True),
        "modality_only_logits": torch.randn(3, 8, requires_grad=True),
        "correction_logits": torch.randn(3, 8, requires_grad=True),
        "correction_gate": torch.sigmoid(torch.randn(3, 1, requires_grad=True)),
    }
    prior = torch.softmax(torch.randn(3, 8), dim=-1)
    batch = {
        "target_label": torch.tensor([1, 2, 3]),
        "gps_error": torch.tensor([1.0, 4.0, 9.0]),
        "gps_prior_probs": prior,
        "support_query_role": ["support", "support", "query_test"],
    }
    loss_fn = ResidualFusionLoss(
        {
            "hard_sample_weight": 2.0,
            "good_error_threshold": 4.0,
            "circular_sigma": 1.0,
        }
    )

    result = loss_fn(outputs, batch)
    gate_target = gate_target_from_gps_error(batch["gps_error"], batch["support_query_role"], threshold=4.0)

    assert torch.isfinite(result["loss"])
    assert result["hard_sample_weight_mean"] > 1.0
    assert torch.isfinite(result["good_anchor_kl"])
    assert gate_target.squeeze(-1).tolist() == [0.0, 1.0, 0.0]
    assert bool(result["query_label_used_for_training"]) is False
