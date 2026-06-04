from __future__ import annotations

import csv
import math
from pathlib import Path

from kd_sensing.data.deepsense6g_topk_candidate_manifest import MANIFEST_NAME, ratio_tag
from kd_sensing.engine.deepsense6g_top8_selector import run_deepsense6g_top8_selector


def test_top8_selector_runner_trains_mlp_without_query_leakage(tmp_path: Path):
    output_root = tmp_path / "selector"
    manifest_dir = output_root / ratio_tag(0.15) / "mapping_disabled" / "manifest"
    manifest_dir.mkdir(parents=True)
    _write_csv(
        manifest_dir / MANIFEST_NAME,
        [
            _row(idx, "support", target=2 if idx % 2 == 0 else 3)
            for idx in range(6)
        ]
        + [
            _row(idx + 100, "query_test", target=2 if idx % 2 == 0 else 3)
            for idx in range(3)
        ],
    )
    cfg = {
        "data": {"support_ratio": 0.15, "label_space": "mapping_disabled", "num_beams": 8},
        "candidate": {"topk": 4},
        "model": {"hidden_dim": 16, "dropout": 0.0, "lambda_init": 0.2, "lambda_max": 1.0},
        "loss": {"num_beams": 8, "prior_anchor_kl_weight": 0.0},
        "train": {"seed": 7, "epochs": 2, "batch_size": 2, "lr": 0.01, "patience": 2},
        "ablation": {"enabled": ["gps_context_only_selector"]},
        "metrics": {"dba_delta": 5.0},
        "outputs": {"root": str(output_root), "manifest_dir": "manifest", "write_config_snapshot": False},
    }

    result = run_deepsense6g_top8_selector(cfg)

    result_dir = Path(result["result_dir"])
    history = _read_csv(result_dir / "training_history.csv")
    predictions = _read_csv(result_dir / "predictions.csv")
    metadata = (result_dir / "run_metadata.json").read_text(encoding="utf-8")

    assert result["training_history_rows"] >= 1
    assert any(row["event"] == "epoch" for row in history)
    assert len(predictions) == 3
    assert {row["trained_model_used"] for row in predictions} == {"True"}
    assert '"query_label_used_for_training": false' in metadata


def _row(idx: int, role: str, *, target: int) -> dict[str, str]:
    beams = [0, 2, 3, 4]
    probs = [0.55, 0.2, 0.15, 0.1]
    target_idx = beams.index(target)
    row = {
        "scene": "scenario31",
        "sample_id": f"scenario31:{role}:{idx}",
        "timestamp": str(idx),
        "frame_id": str(idx),
        "split": "train" if role == "support" else "test",
        "support_query_role": role,
        "split_role": role,
        "target_label": str(target),
        "gps_top1": "0",
        "gps_pred_top1": "0",
        "gps_top1_prob": str(probs[0]),
        "gps_top2_prob": str(probs[1]),
        "gps_top1_top2_margin": str(probs[0] - probs[1]),
        "gps_entropy": "1.0",
        "gps_circular_error": str(min(abs(target), 8 - abs(target))),
        "gps_signed_residual": str(target),
        "theta_degrees": "0.0",
        "theta": "0.0",
        "range": "1.0",
        "E": str(float(idx % 3)),
        "N": str(float(idx % 2)),
        "heading_degrees": "0.0",
        "heading": "0.0",
        "speed": "0.0",
        "image_path": "",
        "image_exists": "False",
        "camera_ae_feature_row_index": "-1",
        "camera_ae_feature_path": "",
        "camera_ae_feature_available": "False",
        "lidar_feature_path": "",
        "lidar_path": "",
        "lidar_feature_available": "False",
        "radar_feature_path": "",
        "radar_path": "",
        "radar_feature_available": "False",
        "gps_logits_row_index": str(idx),
        "gps_logits_source": "gps_logits.npy",
        "gps_protocol": "target_adapt_beambench",
        "gps_ablation": "test",
        "target_in_top8": "True",
        "target_candidate_index": str(target_idx),
        "nearest_candidate_index": str(target_idx),
        "nearest_candidate_error": "0",
        "top8_oracle_error": "0",
        "top8_oracle_beam": str(target),
        "top8_miss": "False",
    }
    for cand_idx, (beam, prob) in enumerate(zip(beams, probs)):
        row[f"cand{cand_idx}_beam"] = str(beam)
        row[f"cand{cand_idx}_logit"] = str(math.log(prob))
        row[f"cand{cand_idx}_prob"] = str(prob)
        row[f"cand{cand_idx}_rank"] = str(cand_idx + 1)
        row[f"cand{cand_idx}_dist_to_gps_top1"] = str(min(abs(beam), 8 - abs(beam)))
    return row


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]
