from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from kd_sensing.data.deepsense6g_residual import (
    FALLBACK_PRIOR_SOURCE,
    ResidualManifestDataset,
    build_residual_manifest,
)


def test_residual_manifest_fields_roles_fallback_and_gps_context_dataset(tmp_path: Path):
    gps_root = tmp_path / "gps"
    gps_dir = gps_root / "r15" / "mapping_disabled"
    gps_dir.mkdir(parents=True)
    data_root = tmp_path / "DeepSense6G"
    scene_root = data_root / "scenario31"
    scene_root.mkdir(parents=True)
    _write_deepsense_csv(scene_root / "train_seqs_RA_GPS_LIDAR.csv", split="train")
    _write_deepsense_csv(scene_root / "test_seqs_RA_GPS_LIDAR.csv", split="test")
    _write_csv(
        gps_dir / "summary_overall.csv",
        [
            {
                "protocol": "target_adapt_beambench",
                "ablation": "branch_mixture_circular",
                "DBA": "0.5",
                "mean_circular_error": "2.0",
            }
        ],
    )
    _write_csv(
        gps_dir / "predictions.csv",
        [
            {
                "sample_id": "scenario31:test:0:mmWave_power_1",
                "scene": "scenario31",
                "split": "test",
                "protocol": "target_adapt_beambench",
                "ablation": "branch_mixture_circular",
                "true_beam": "1",
                "predicted_beam": "63",
                "final_predicted_beam": "63",
                "topk_predictions": json.dumps([63, 0, 1, 2, 3]),
                "circular_error": "2",
                "signed_residual": "2",
                "theta_degrees": "10",
                "E": "1.0",
                "N": "2.0",
                "support_query_role": "query_test",
            }
        ],
    )
    _write_csv(
        gps_dir / "support_manifest.csv",
        [
            {
                "protocol": "target_adapt_beambench",
                "target_scene": "scenario31",
                "label_space": "mapping_disabled",
                "role": "support",
                "sample_id": "scenario31:train:0:mmWave_power_0",
                "scene": "scenario31",
                "split": "train",
                "target_label": "3",
            }
        ],
    )
    cfg = {
        "data": {
            "data_root": str(data_root),
            "gps_sweep_root": str(gps_root),
            "label_space": "mapping_disabled",
            "support_ratio": 0.15,
            "num_beams": 64,
        },
        "residual": {"gps_ablation": "best", "gps_prior_fallback_sigma": 2.0},
        "outputs": {"root": str(tmp_path / "residual")},
    }

    result = build_residual_manifest(cfg)
    manifest = Path(result["manifest_path"])
    rows = _read_csv(manifest)

    assert result["support_count"] == 1
    assert result["query_count"] == 1
    assert {"scene", "sample_id", "support_query_role", "gps_pred_top1", "gps_prior_source"} <= set(rows[0])
    assert {row["support_query_role"] for row in rows} == {"support", "query_test"}
    query = [row for row in rows if row["support_query_role"] == "query_test"][0]
    assert query["gps_prior_source"] == FALLBACK_PRIOR_SOURCE
    assert query["gps_pred_top1"] == "63"
    assert result["modality_availability"]["image"]["available"] is False

    dataset = ResidualManifestDataset(manifest, enabled_modalities=("gps_context",), num_beams=64)
    item = dataset[0]
    assert item["gps_context_features"].shape[0] == 9
    assert item["gps_prior_logits"].shape[0] == 64


def test_residual_manifest_uses_exported_logits_for_support_and_test_query(tmp_path: Path):
    gps_root = tmp_path / "gps"
    gps_dir = gps_root / "r15" / "mapping_disabled"
    gps_dir.mkdir(parents=True)
    data_root = tmp_path / "DeepSense6G"
    scene_root = data_root / "scenario31"
    scene_root.mkdir(parents=True)
    _write_deepsense_csv(scene_root / "train_seqs_RA_GPS_LIDAR.csv", split="train", count=2)
    _write_deepsense_csv(scene_root / "test_seqs_RA_GPS_LIDAR.csv", split="test", count=1)
    _write_csv(
        gps_dir / "summary_overall.csv",
        [
            {
                "protocol": "target_adapt_beambench",
                "ablation": "branch_mixture_circular",
                "DBA": "0.5",
                "mean_circular_error": "2.0",
            }
        ],
    )
    _write_csv(
        gps_dir / "predictions.csv",
        [
            {
                "sample_id": "scenario31:test:0:mmWave_power_1",
                "scene": "scenario31",
                "split": "test",
                "protocol": "target_adapt_beambench",
                "ablation": "branch_mixture_circular",
                "true_beam": "1",
                "predicted_beam": "4",
                "final_predicted_beam": "4",
                "topk_predictions": json.dumps([4, 3, 5, 2, 6]),
                "circular_error": "3",
                "signed_residual": "-3",
                "theta_degrees": "10",
                "E": "1.0",
                "N": "2.0",
                "support_query_role": "query_test",
            }
        ],
    )
    _write_csv(
        gps_dir / "support_manifest.csv",
        [
            {
                "protocol": "target_adapt_beambench",
                "target_scene": "scenario31",
                "label_space": "mapping_disabled",
                "role": "support",
                "sample_id": "scenario31:train:0:mmWave_power_0",
                "scene": "scenario31",
                "split": "train",
                "target_label": "3",
            },
            {
                "protocol": "target_adapt_beambench",
                "target_scene": "scenario31",
                "label_space": "mapping_disabled",
                "role": "query",
                "sample_id": "scenario31:train:1:mmWave_power_1",
                "scene": "scenario31",
                "split": "train",
                "target_label": "5",
            },
        ],
    )
    logits = np.full((2, 64), -8.0, dtype=np.float32)
    logits[0, 3] = 4.0
    logits[1, 4] = 4.0
    np.save(gps_dir / "gps_logits.npy", logits)
    _write_csv(
        gps_dir / "gps_logits_index.csv",
        [
            {
                "row_index": "0",
                "sample_id": "scenario31:train:0:mmWave_power_0",
                "scene": "scenario31",
                "split": "train",
                "protocol": "target_adapt_beambench",
                "ablation": "branch_mixture_circular",
                "support_query_role": "support",
            },
            {
                "row_index": "1",
                "sample_id": "scenario31:test:0:mmWave_power_1",
                "scene": "scenario31",
                "split": "test",
                "protocol": "target_adapt_beambench",
                "ablation": "branch_mixture_circular",
                "support_query_role": "query_test",
            },
        ],
    )
    cfg = {
        "data": {
            "data_root": str(data_root),
            "gps_sweep_root": str(gps_root),
            "label_space": "mapping_disabled",
            "support_ratio": 0.15,
            "num_beams": 64,
        },
        "residual": {"gps_ablation": "best", "gps_prior_fallback_sigma": 2.0},
        "outputs": {"root": str(tmp_path / "residual")},
    }

    result = build_residual_manifest(cfg)
    rows = _read_csv(Path(result["manifest_path"]))

    assert result["support_count"] == 1
    assert result["query_count"] == 1
    assert len(rows) == 2
    assert {row["sample_id"] for row in rows} == {
        "scenario31:train:0:mmWave_power_0",
        "scenario31:test:0:mmWave_power_1",
    }
    assert {row["gps_prior_source"] for row in rows} == {"gps_logits.npy"}
    support = [row for row in rows if row["support_query_role"] == "support"][0]
    assert support["gps_pred_top1"] == "3"


def _write_deepsense_csv(path: Path, *, split: str, count: int = 1) -> None:
    _write_csv(
        path,
        [
            {
                "camera8": "./missing/image.jpg",
                "radar8": "./missing/radar.npy",
                "lidar8": "./missing/lidar.ply",
                "future_beam1": f"./unit1/mmWave_data/mmWave_power_{idx if split == 'train' else idx + 1}.txt",
                "seq_index": str(idx),
            }
            for idx in range(count)
        ],
    )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]
