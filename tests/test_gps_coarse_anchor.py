from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from kd_sensing.baselines.gps_window.geometry import angle_to_beam
from kd_sensing.baselines.gps_window.types import GpsWindowSample
from kd_sensing.engine.gps_coarse_anchor import (
    GpsCoarseAnchorConfig,
    build_geometry_anchor,
    coarse_labels_from_beam,
    compute_gps_coarse_anchor_loss,
    geometry_anchor_metrics,
    gps_anchor_tensors_from_batch,
    run_gps_coarse_anchor_evaluation,
)
from kd_sensing.evaluation.metrics import calculate_dba_score
from kd_sensing.models.gps import GpsStudentModalityNet


def test_anchor_contract_shapes_group_validation_and_metadata():
    cfg = GpsCoarseAnchorConfig.from_mapping(
        {"enabled": True, "num_classes": 8, "group_size": 2, "horizon": 1}
    )
    sample = GpsWindowSample(
        sample_id="s0",
        scenario="scene",
        split="target_test",
        history_geometry=({"available": True, "relative_azimuth": 90.0},),
        target_beams=(2,),
    )
    anchor = build_geometry_anchor([sample], cfg, calibration_samples=[], evaluation_split="target_test")
    assert tuple(anchor.coarse_logits.shape) == (1, 1, 4)
    assert tuple(anchor.center_beam.shape) == (1, 1)
    assert tuple(anchor.confidence.shape) == (1, 1)
    assert tuple(anchor.beam_scores.shape) == (1, 1, 8)
    assert anchor.metadata["anchor_source"] == "geometry_calibrated"
    assert anchor.metadata["used_target_test_for_calibration"] is False

    with pytest.raises(ValueError, match="num_classes .* group_size"):
        GpsCoarseAnchorConfig.from_mapping({"num_classes": 7, "group_size": 3})


def test_geometry_anchor_boresight_direction_offset_scores_and_confidence():
    cfg = GpsCoarseAnchorConfig.from_mapping(
        {
            "enabled": True,
            "num_classes": 8,
            "group_size": 2,
            "horizon": 1,
            "boresight_angle_degrees": 90.0,
            "beam_offset": 1,
            "score_width": 1.0,
            "neighbor_top_k": 3,
        }
    )
    sample = GpsWindowSample(
        sample_id="s0",
        scenario="scene",
        split="target_test",
        history_geometry=({"available": True, "relative_azimuth": 90.0},),
        target_beams=(1,),
    )
    anchor = build_geometry_anchor([sample], cfg, calibration_samples=[], evaluation_split="target_test")
    assert int(anchor.center_beam[0, 0].item()) == 1
    assert int(anchor.coarse_logits.argmax(dim=-1)[0, 0].item()) == 0
    assert float(anchor.confidence[0, 0].item()) > 0.0
    metrics = geometry_anchor_metrics(anchor, torch.tensor([[1]]), cfg=cfg)
    assert metrics["center_beam_top1"] == 1.0
    assert metrics["coarse_accuracy"] == 1.0
    assert metrics["dba_avg"] == 1.0
    assert metrics["residual_preview"]["diagnostic_only"] is True


def test_angle_lookup_anchor_uses_calibration_angle_label_pairs():
    cfg = GpsCoarseAnchorConfig.from_mapping(
        {
            "enabled": True,
            "algorithm": "angle_lookup",
            "num_classes": 8,
            "group_size": 2,
            "horizon": 1,
            "angle_lookup_k": 1,
            "score_width": 1.0,
            "neighbor_top_k": 3,
        }
    )
    calibration = [
        GpsWindowSample(
            sample_id="cal0",
            scenario="scene",
            split="target_adapt_support",
            history_geometry=({"available": True, "relative_azimuth": 10.0},),
            target_beams=(3,),
        ),
        GpsWindowSample(
            sample_id="cal1",
            scenario="scene",
            split="target_adapt_support",
            history_geometry=({"available": True, "relative_azimuth": 80.0},),
            target_beams=(5,),
        ),
    ]
    sample = GpsWindowSample(
        sample_id="s0",
        scenario="scene",
        split="target_test",
        history_geometry=({"available": True, "relative_azimuth": 12.0},),
        target_beams=(3,),
    )

    anchor = build_geometry_anchor([sample], cfg, calibration_samples=calibration, evaluation_split="target_test")

    assert int(anchor.center_beam[0, 0].item()) == 3
    assert anchor.metadata["anchor_algorithm"] == "angle_lookup"
    assert anchor.metadata["calibration_state"]["angle_lookup_sample_count"] == 2
    metrics = geometry_anchor_metrics(anchor, torch.tensor([[3]]), cfg=cfg)
    assert metrics["center_beam_top1"] == 1.0
    assert metrics["dba_avg"] == 1.0


def test_dba_treats_beam_codebook_boundary_as_circular():
    outputs = torch.full((1, 1, 64), -10.0)
    outputs[0, 0, 0] = 10.0
    labels = torch.tensor([[63]])

    dba = calculate_dba_score(outputs, labels, delta=5)

    assert dba[0] > 0.0


def test_oracle_guard_marks_target_test_calibration_and_forbidden_fields():
    cfg = GpsCoarseAnchorConfig.from_mapping(
        {
            "enabled": True,
            "num_classes": 8,
            "group_size": 2,
            "used_fields": ["gps", "future_beam_power_argmax"],
        }
    )
    sample = GpsWindowSample(
        sample_id="s0",
        scenario="scene",
        split="target_test",
        history_geometry=({"available": True, "relative_azimuth": 0.0},),
        target_beams=(0,),
    )
    anchor = build_geometry_anchor([sample], cfg, calibration_samples=[sample], calibration_split="target_test")
    guard = anchor.metadata["oracle_guard"]
    assert guard["eligible_for_main_claim"] is False
    assert "future_beam_power_argmax" in guard["used_target_oracle_fields"]
    assert guard["used_target_test_for_calibration"] is True


def test_gps_neural_head_default_compatibility_and_loss():
    legacy = GpsStudentModalityNet(
        gps_input_size=3,
        feature_size=8,
        num_classes=8,
        gru_params=[8, 8, 1],
        num_pred=2,
    )
    x = torch.randn(2, 4, 3)
    legacy_out = legacy(x)
    assert isinstance(legacy_out, tuple)

    model = GpsStudentModalityNet(
        gps_input_size=3,
        feature_size=8,
        num_classes=8,
        gru_params=[8, 8, 1],
        num_pred=2,
        coarse_anchor={
            "enabled": True,
            "num_classes": 8,
            "group_size": 2,
            "hidden_size": 8,
            "beam_auxiliary": True,
            "beam_auxiliary_weight": 0.1,
        },
    )
    out = model(x)
    assert isinstance(out, dict)
    assert tuple(out["gps_anchor_coarse_logits"].shape) == (2, 4, 4)
    cfg = GpsCoarseAnchorConfig.from_mapping(
        {"enabled": True, "anchor_source": "gps_neural_coarse", "num_classes": 8, "group_size": 2}
    )
    labels = torch.tensor([[0, 3], [7, -100]])
    anchor = {
        "coarse_logits": out["gps_anchor_coarse_logits"][:, -2:, :],
        "beam_scores": out["gps_anchor_beam_scores"][:, -2:, :],
    }
    loss, diagnostics = compute_gps_coarse_anchor_loss(anchor, labels, cfg)
    assert torch.isfinite(loss)
    assert diagnostics["gps_anchor/loss_total"] >= 0.0
    assert torch.equal(coarse_labels_from_beam(labels, num_classes=8, group_size=2), torch.tensor([[0, 1], [3, -100]]))


def test_gps_anchor_batch_adapter_requires_fields():
    batch = {
        "gps_anchor_coarse_logits": torch.zeros(2, 1, 4),
        "gps_anchor_center_beam": torch.tensor([[1], [2]]),
        "gps_anchor_confidence": torch.ones(2, 1),
        "gps_anchor_residual_anchor_beam": torch.tensor([[1], [2]]),
    }
    tensors = gps_anchor_tensors_from_batch(batch, num_pred=1, device=torch.device("cpu"))
    assert tuple(tensors["gps_anchor_coarse_logits"].shape) == (2, 1, 4)
    with pytest.raises(ValueError, match="gps_anchor.enabled=true"):
        gps_anchor_tensors_from_batch({}, num_pred=1, device=torch.device("cpu"))


def test_gps_coarse_anchor_runner_writes_metrics_and_predictions(tmp_path: Path):
    data_root = tmp_path / "MMW" / "sunny"
    for scene in ("source_scene", "target_scene"):
        split_dir = data_root / "Prepared" / scene / "splits" / "l5p3_group_safe"
        split_dir.mkdir(parents=True)
        _write_rows(split_dir / "train.csv", scene=scene, count=2)
        _write_rows(split_dir / "test.csv", scene=scene, count=3)
    out_dir = tmp_path / "out"
    result = run_gps_coarse_anchor_evaluation(
        {
            "data": {
                "data_root": str(data_root),
                "split_tag": "l5p3_group_safe",
                "source_scenes": ["source_scene"],
                "target_scenes": ["target_scene"],
            },
            "coarse_anchor": {
                "enabled": True,
                "num_classes": 8,
                "group_size": 2,
                "horizon": 1,
            },
        },
        execute=True,
        output_dir=out_dir,
    )
    assert result["mode"] == "execute"
    metrics = json.loads((out_dir / "target_scene" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["metrics"]["valid_label_count"] == 3
    assert "dba_avg" in metrics["metrics"]
    assert "dba_by_horizon" in metrics["metrics"]
    assert metrics["metadata"]["used_target_test_for_calibration"] is False
    predictions_path = out_dir / "target_scene" / "predictions.csv"
    assert predictions_path.exists()
    with predictions_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert rows[0]["predicted_beam"] == rows[0]["pred_beam"]
    assert rows[0]["final_predicted_beam"] == rows[0]["predicted_beam"]
    assert rows[0]["topk_predictions"]
    assert rows[0]["anchor_center_beam"]


def test_gps_coarse_anchor_runner_supports_target_adapt_calibration(tmp_path: Path):
    data_root = tmp_path / "MMW" / "sunny"
    split_dir = data_root / "Prepared" / "target_scene" / "splits" / "l5p3_group_safe"
    split_dir.mkdir(parents=True)
    _write_rows(split_dir / "train.csv", scene="target_scene", count=4, label_offset=2)
    _write_rows(split_dir / "test.csv", scene="target_scene", count=4, label_offset=2)

    out_dir = tmp_path / "out"
    result = run_gps_coarse_anchor_evaluation(
        {
            "data": {
                "data_root": str(data_root),
                "split_tag": "l5p3_group_safe",
                "target_scenes": ["target_scene"],
            },
            "coarse_anchor": {
                "enabled": True,
                "num_classes": 8,
                "group_size": 2,
                "horizon": 1,
                "calibration_mode": "target_adapt",
                "auto_calibrate_beam_mapping": True,
                "calibration_holdout_fraction": 0.5,
                "calibration_holdout_strategy": "angle_coverage",
            },
        },
        execute=True,
        output_dir=out_dir,
    )
    scene = result["scene_results"][0]
    assert scene["calibration_split"] == "target_adapt_support_fit"
    assert scene["selection_split"] == "target_adapt_support_selection"
    assert scene["support_fit_sample_count"] == 2
    assert scene["support_selection_sample_count"] == 2
    assert scene["calibration_holdout"]["holdout_strategy"] == "angle_coverage"
    assert scene["calibration_holdout"]["fit_protects_angle_extrema"] is True
    metrics = json.loads((out_dir / "target_scene" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["metrics"]["center_beam_top1"] == 1.0
    assert metrics["metrics"]["dba_avg"] == 1.0
    assert metrics["metadata"]["calibration_sample_count"] == 2
    assert metrics["metadata"]["used_target_test_for_calibration"] is False
    assert metrics["metadata"]["effective_beam_offset"] == 2


def _write_rows(path: Path, *, scene: str, count: int, label_offset: int = 0) -> None:
    rows = []
    for idx in range(count):
        angle = float(idx * 45.0)
        label = int((angle_to_beam(angle, num_classes=8) + int(label_offset)) % 8)
        rows.append(
            {
                "sample_id": f"{scene}:{idx}",
                "target_sample_id": f"{scene}:{idx + 1}",
                "scene_slug": scene,
                "geometry1": json.dumps({"available": True, "relative_azimuth": angle}),
                "future_beam_label1": label,
                "beam_label": label,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
