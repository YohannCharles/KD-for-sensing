from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from kd_sensing.data.deepsense6g_topk_candidate_manifest import ratio_tag
from kd_sensing.engine.deepsense6g_gps_lidar_bgam import run_deepsense6g_gps_lidar_bgam


def test_gps_lidar_bgam_runner_cpu_smoke_manifest_train_eval_outputs(tmp_path: Path):
    output_root = tmp_path / "bgam"
    manifest_dir = output_root / ratio_tag(0.15) / "mapping_disabled" / "manifest"
    manifest_dir.mkdir(parents=True)
    rows = []
    for idx in range(6):
        rows.append(_row(tmp_path, idx, "support", target=2 if idx % 2 == 0 else 3))
    for idx in range(3):
        rows.append(_row(tmp_path, idx + 100, "query_test", target=2 if idx % 2 == 0 else 3))
    _write_csv(manifest_dir / "gps_lidar_bgam_manifest.csv", rows)
    cfg = {
        "data": {"support_ratio": 0.15, "label_space": "mapping_disabled", "num_beams": 8, "topk": 4},
        "candidate": {"topk": 4, "num_beams": 8},
        "geometry": {"beam_angle_convention": "dft_ula_broadside_approximation"},
        "lidar": {"bev_size": [8, 8], "roi": [-4, 4, -4, 4, -1, 2], "input_channels": 3, "missing_policy": "zeros"},
        "bgam": {"sigma": 0.4, "hard_half_width": 0.5, "debug_masks": {"enabled": True, "max_samples": 2}},
        "model": {
            "topk": 4,
            "num_beams": 8,
            "d_model": 16,
            "hidden_dim": 16,
            "dropout": 0.0,
            "lidar": {"in_channels": 3, "channels": [8], "freeze_lidar_encoder": False},
            "cross_attention": {"num_heads": 4, "num_queries": 1},
        },
        "loss": {"num_beams": 8, "prior_anchor_kl_weight": 0.0},
        "train": {"seed": 3, "epochs": 1, "batch_size": 2, "lr": 0.001, "max_train_samples": 6, "max_eval_samples": 3},
        "eval": {"batch_size": 2, "max_samples": 3},
        "ablation": {"enabled": ["gps_only", "gps_lidar_topk_per_candidate_rerank"], "trainable": ["gps_lidar_topk_per_candidate_rerank"]},
        "metrics": {"dba_delta": 5.0},
        "outputs": {"root": str(output_root), "manifest_dir": "manifest", "write_config_snapshot": True},
    }

    result = run_deepsense6g_gps_lidar_bgam(cfg, debug_masks=True)
    result_dir = Path(result["result_dir"])

    assert (result_dir / "metrics.json").exists()
    assert (result_dir / "summary_overall.csv").exists()
    assert (result_dir / "summary_by_scene.csv").exists()
    assert (result_dir / "summary_by_bgam_mode.csv").exists()
    assert (result_dir / "predictions.csv").exists()
    assert (result_dir / "run_metadata.json").exists()
    assert (result_dir / "resolved_config.yaml").exists()
    assert result["checkpoint_paths"]
    predictions = _read_csv(result_dir / "predictions.csv")
    assert {
        "sample_id",
        "scene",
        "gt_beam",
        "gps_top1",
        "history_pseudo_top1",
        "history_pseudo_entropy_mean",
        "pred_beam",
        "gps_topk",
        "model_topk",
        "correct",
        "target_in_topk",
        "bgam_mode",
        "uses_oracle_history_label",
        "beam_angle_source",
    } <= set(predictions[0])
    assert len(predictions) == 6


def _row(tmp_path: Path, idx: int, role: str, *, target: int) -> dict[str, str]:
    bev_path = tmp_path / f"bev_{idx}.npy"
    np.save(bev_path, np.ones((3, 8, 8), dtype=np.float32) * (1.0 + idx))
    beams = [0, 2, 3, 4]
    probs = [0.5, 0.2, 0.2, 0.1]
    target_idx = beams.index(target)
    row = {
        "scene": "scenario31",
        "sample_id": f"scenario31:{role}:{idx}",
        "support_query_role": role,
        "split_role": role,
        "target_label": str(target),
        "gt_beam": str(target),
        "gps_top1": "0",
        "gps_top1_prob": "0.5",
        "gps_top2_prob": "0.2",
        "gps_top1_top2_margin": "0.3",
        "gps_entropy": "1.0",
        "theta_gps": "0.0",
        "distance_to_rsu": "1.0",
        "lidar_bev_cache_path": str(bev_path),
        "lidar_path": str(bev_path),
        "lidar_available": "True",
        "beam_angle_source": "dft_ula_approximation",
        "target_in_top8": "True",
        "target_candidate_index": str(target_idx),
        "nearest_candidate_index": str(target_idx),
    }
    for cand_idx, (beam, prob) in enumerate(zip(beams, probs)):
        row[f"cand{cand_idx}_beam"] = str(beam)
        row[f"cand{cand_idx}_logit"] = str(np.log(prob))
        row[f"cand{cand_idx}_prob"] = str(prob)
        row[f"cand{cand_idx}_rank"] = str(cand_idx + 1)
        row[f"cand{cand_idx}_dist_to_gps_top1"] = str(beam)
    return row


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]
