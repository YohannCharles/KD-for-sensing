from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from kd_sensing.losses.gps_lidar_bgam_losses import GPSLidarBGAMLoss
from kd_sensing.models.gps_lidar_bgam import GPSGuidedBGAM, GPSPriorEncoder, LidarBEVCrossAttention, save_debug_masks
from kd_sensing.models.gps_lidar_bgam_model import GPSLidarBGAMBeamPredictor


def test_bgam_masks_attention_encoder_debug_and_no_label_leakage(tmp_path: Path):
    bev = torch.ones(2, 4, 8, 8)
    theta = torch.tensor([0.0, math.pi / 2])
    bgam = GPSGuidedBGAM(roi=(-4, 4, -4, 4, -1, 2), bev_size=(8, 8), sigma=0.3, hard_half_width=0.4)

    soft = bgam(bev, theta_gps=theta, mode="single_soft")
    assert soft["mask"].shape == (2, 1, 8, 8)
    assert soft["mask"][0].max() > 0.85

    hard = bgam(bev, theta_gps=theta, mode="single_hard")
    assert set(torch.unique(hard["mask"]).tolist()) <= {0.0, 1.0}

    beams = torch.tensor([[0, 2, 4], [1, 3, 5]])
    probs = torch.tensor([[0.6, 0.3, 0.1], [0.5, 0.25, 0.25]])
    angles = torch.linspace(-math.pi / 2, math.pi / 2, 8)
    union = bgam(bev, theta_gps=theta, gps_topk_beams=beams, gps_topk_probs=probs, beam_angles=angles, mode="topk_union_soft")
    per = bgam(bev, theta_gps=theta, gps_topk_beams=beams, gps_topk_probs=probs, beam_angles=angles, mode="topk_per_candidate")
    assert union["mask"].shape == (2, 1, 8, 8)
    assert per["candidate_masked_bev_feat"].shape == (2, 3, 4, 8, 8)
    history_beams = torch.tensor([[0, 1, 2], [2, 3, 4]])
    history_probs = torch.tensor([[0.6, 0.3, 0.1], [0.5, 0.3, 0.2]])
    history_entropy = torch.tensor([[0.2, 0.4, 0.6], [0.3, 0.4, 0.5]])
    history_valid = torch.tensor([[True, True, False], [True, True, True]])
    hist_soft = bgam(
        bev,
        theta_gps=theta,
        history_pseudo_beams=history_beams,
        history_pseudo_probs=history_probs,
        history_pseudo_entropy=history_entropy,
        history_valid_mask=history_valid,
        beam_angles=angles,
        mode="history_pseudo_soft",
    )
    hist_union = bgam(
        bev,
        theta_gps=theta,
        gps_topk_beams=beams,
        gps_topk_probs=probs,
        history_pseudo_beams=history_beams,
        history_pseudo_probs=history_probs,
        history_pseudo_entropy=history_entropy,
        history_valid_mask=history_valid,
        beam_angles=angles,
        mode="history_pseudo_topk_union",
    )
    assert hist_soft["mask"].shape == (2, 1, 8, 8)
    assert hist_union["mask_source"] == "history_pseudo_topk_union"

    attn = LidarBEVCrossAttention(in_channels=4, d_model=16, num_heads=4, num_queries=2)
    assert attn(soft["masked_bev_feat"]).shape == (2, 16)

    gps = GPSPriorEncoder(d_model=16, hidden_dim=16)
    assert gps(theta_gps=theta, distance_to_rsu=torch.ones(2), candidate_probs=probs).shape == (2, 16)

    debug = save_debug_masks(soft["mask"], output_dir=tmp_path / "debug", sample_ids=["a", "b"], theta_gps=theta, mode="single_soft", sigma=soft["sigma"])
    metadata = json.loads(Path(debug["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata[0]["gt_beam_used_as_mask_source"] is False

    changed_label = bgam(bev, theta_gps=theta, mode="single_soft")["mask"]
    assert torch.allclose(soft["mask"], changed_label)


def test_gps_lidar_bgam_predictor_loss_forward_backward_and_no_future_label_input():
    model = GPSLidarBGAMBeamPredictor(
        topk=4,
        num_beams=8,
        d_model=16,
        hidden_dim=16,
        lidar_in_channels=3,
        lidar_channels=(8,),
        roi=(-4, 4, -4, 4, -1, 2),
        bev_size=(8, 8),
        bgam_mode="topk_per_candidate",
        attention_heads=4,
    )
    batch = {
        "candidate_beams": torch.tensor([[0, 2, 3, 4], [0, 2, 4, 6]]),
        "candidate_probs": torch.tensor([[0.5, 0.2, 0.2, 0.1], [0.5, 0.2, 0.2, 0.1]]),
        "candidate_log_probs": torch.log(torch.tensor([[0.5, 0.2, 0.2, 0.1], [0.5, 0.2, 0.2, 0.1]])),
        "theta_gps": torch.tensor([0.0, 0.2]),
        "distance_to_rsu": torch.tensor([1.0, 2.0]),
        "gps_entropy": torch.tensor([0.5, 1.0]),
        "lidar_bev": torch.randn(2, 3, 8, 8),
        "gt_beam": torch.tensor([2, 7]),
        "target_label": torch.tensor([2, 7]),
        "target_candidate_index": torch.tensor([1, -1]),
        "nearest_candidate_index": torch.tensor([1, 3]),
        "support_query_role": ["support", "support"],
    }
    outputs = model(
        candidate_beams=batch["candidate_beams"],
        candidate_probs=batch["candidate_probs"],
        candidate_log_probs=batch["candidate_log_probs"],
        theta_gps=batch["theta_gps"],
        distance_to_rsu=batch["distance_to_rsu"],
        gps_entropy=batch["gps_entropy"],
        lidar_bev=batch["lidar_bev"],
        beam_angles=torch.linspace(-math.pi / 2, math.pi / 2, 8),
    )
    assert outputs["final_candidate_scores"].shape == (2, 4)
    assert outputs["candidate_probs"].shape == (2, 4)
    assert outputs["selected_beam"].shape == (2,)
    history_outputs = model(
        candidate_beams=batch["candidate_beams"],
        candidate_probs=batch["candidate_probs"],
        candidate_log_probs=batch["candidate_log_probs"],
        theta_gps=batch["theta_gps"],
        distance_to_rsu=batch["distance_to_rsu"],
        gps_entropy=batch["gps_entropy"],
        lidar_bev=batch["lidar_bev"],
        history_pseudo_beams=torch.tensor([[0, 2, 3], [0, 4, 6]]),
        history_pseudo_probs=torch.tensor([[0.6, 0.3, 0.1], [0.5, 0.3, 0.2]]),
        history_pseudo_entropy=torch.tensor([[0.2, 0.4, 0.5], [0.3, 0.5, 0.7]]),
        history_valid_mask=torch.tensor([[True, True, True], [True, True, False]]),
        beam_angles=torch.linspace(-math.pi / 2, math.pi / 2, 8),
        bgam_mode="history_pseudo_per_candidate",
    )
    assert history_outputs["diagnostics"]["mask_source"] == "history_pseudo_per_candidate"
    loss = GPSLidarBGAMLoss({"num_beams": 8})(outputs, batch)
    assert torch.isfinite(loss["loss"])
    assert loss["rerank_ce_sample_count"].item() == 1
    assert loss["skipped_rerank_sample_count"].item() == 1
    loss["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)

    try:
        model(
            candidate_beams=batch["candidate_beams"],
            candidate_probs=batch["candidate_probs"],
            candidate_log_probs=batch["candidate_log_probs"],
            theta_gps=batch["theta_gps"],
            distance_to_rsu=batch["distance_to_rsu"],
            lidar_bev=batch["lidar_bev"],
            gt_beam=batch["gt_beam"],
        )
    except ValueError as exc:
        assert "future label" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("gt_beam must not be accepted as a BGAM input")
