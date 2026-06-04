from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from kd_sensing.data.beam_label_space import label_space_metadata
from kd_sensing.data.deepsense6g_topk_candidate_manifest import (
    STRICT_LOGITS_ERROR,
    build_topk_candidate_manifest,
    load_strict_gps_logits,
)


def test_topk_candidate_manifest_fields_strict_logits_and_no_top5_truncation(tmp_path: Path):
    gps_root = tmp_path / "gps"
    gps_dir = gps_root / "r15" / "mapping_disabled"
    gps_dir.mkdir(parents=True)
    data_root = tmp_path / "DeepSense6G"
    scene_root = data_root / "scenario31"
    scene_root.mkdir(parents=True)
    _write_deepsense_csv(scene_root / "train_seqs_RA_GPS_LIDAR.csv", split="train")
    _write_deepsense_csv(scene_root / "test_seqs_RA_GPS_LIDAR.csv", split="test")
    _write_csv(
        gps_dir / "predictions.csv",
        [
            {
                "sample_id": "scenario31:test:0:mmWave_power_1",
                "scene": "scenario31",
                "split": "test",
                "protocol": "target_adapt_beambench",
                "ablation": "branch",
                "label_space": "mapping_disabled",
                "true_beam": "9",
                "predicted_beam": "0",
                "final_predicted_beam": "0",
                "topk_predictions": json.dumps([0, 1, 2, 3, 4]),
                "support_query_role": "query_test",
                "theta_degrees": "10",
                "E": "1.0",
                "N": "2.0",
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
    logits = np.full((2, 16), -20.0, dtype=np.float32)
    logits[0, [3, 0, 1, 2, 4, 5, 6, 7]] = [8, 7, 6, 5, 4, 3, 2, 1]
    logits[1, [0, 1, 2, 3, 4, 9, 5, 6]] = [9, 8, 7, 6, 5, 4, 3, 2]
    np.save(gps_dir / "gps_logits.npy", logits)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    np.save(feature_dir / "features.npy", np.ones((2, 4), dtype=np.float32))
    _write_csv(
        feature_dir / "features_index.csv",
        [
            {
                "row_index": "0",
                "scene": "scenario31",
                "sample_id": "scenario31:train:0:mmWave_power_0",
                "split_role": "support",
            },
            {
                "row_index": "1",
                "scene": "scenario31",
                "sample_id": "scenario31:test:0:mmWave_power_1",
                "split_role": "query_test",
            },
        ],
    )
    _write_csv(
        gps_dir / "gps_logits_index.csv",
        [
            {
                "row_index": "0",
                "sample_id": "scenario31:train:0:mmWave_power_0",
                "scene": "scenario31",
                "split": "train",
                "protocol": "target_adapt_beambench",
                "ablation": "branch",
                "support_query_role": "support",
            },
            {
                "row_index": "1",
                "sample_id": "scenario31:test:0:mmWave_power_1",
                "scene": "scenario31",
                "split": "test",
                "protocol": "target_adapt_beambench",
                "ablation": "branch",
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
            "num_beams": 16,
            "scenes": ["scenario31"],
        },
        "candidate": {"topk": 8, "gps_protocol": "target_adapt_beambench", "gps_ablation": "branch"},
        "image": {
            "ae_feature_path": str(feature_dir / "features.npy"),
            "ae_feature_index_path": str(feature_dir / "features_index.csv"),
        },
        "outputs": {"root": str(tmp_path / "selector")},
    }

    result = build_topk_candidate_manifest(cfg)
    rows = _read_csv(Path(result["manifest_path"]))
    query = [row for row in rows if row["support_query_role"] == "query_test"][0]
    beams = [int(query[f"cand{idx}_beam"]) for idx in range(8)]

    assert result["support_count"] == 1
    assert result["query_count"] == 1
    assert {"scene", "sample_id", "gps_top1", "target_in_top8", "top8_oracle_beam"} <= set(query)
    assert 9 in beams
    assert 9 not in json.loads(query.get("topk_predictions", "[]") or "[]")
    assert query["target_in_top8"] in {"True", "true", "1"}
    assert query["target_candidate_index"] == "5"
    assert query["camera_ae_feature_available"] in {"True", "true", "1"}
    assert query["camera_ae_feature_row_index"] == "1"
    assert query["camera_ae_feature_path"] == str(feature_dir / "features.npy")
    assert Path(result["top8_recall_summary_path"]).exists()


def test_topk_candidate_manifest_rejects_mapping_fingerprint_mismatch(tmp_path: Path):
    gps_root = tmp_path / "gps"
    gps_dir = gps_root / "r15" / "mapping_enabled"
    gps_dir.mkdir(parents=True)
    data_root = tmp_path / "DeepSense6G"
    scene_root = data_root / "scenario31"
    scene_root.mkdir(parents=True)
    _write_deepsense_csv(scene_root / "train_seqs_RA_GPS_LIDAR.csv", split="train")
    _write_deepsense_csv(scene_root / "test_seqs_RA_GPS_LIDAR.csv", split="test")
    np.save(gps_dir / "gps_logits.npy", np.zeros((1, 8), dtype=np.float32))
    _write_csv(
        gps_dir / "gps_logits_index.csv",
        [
            {
                "row_index": "0",
                "sample_id": "scenario31:test:0:mmWave_power_1",
                "scene": "scenario31",
                "split": "test",
                "protocol": "target_adapt_beambench",
                "ablation": "branch",
                "support_query_role": "query_test",
                "label_space": "mapping_enabled",
                "beam_label_space": "wrong_space",
                "beam_label_mapping_fingerprint": "badfingerprint",
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
                "ablation": "branch",
                "label_space": "mapping_enabled",
                "beam_label_space": "mapped",
                "beam_label_mapping_fingerprint": "also_bad",
                "true_beam": "0",
                "support_query_role": "query_test",
            }
        ],
    )
    cfg = {
        "data": {
            "data_root": str(data_root),
            "gps_sweep_root": str(gps_root),
            "label_space": "mapping_enabled",
            "label_spaces": {
                "mapping_enabled": {"enabled": True, "label_space": "mapped", "num_classes": 8, "offset": 1},
                "mapping_disabled": {"enabled": False},
            },
            "support_ratio": 0.15,
            "num_beams": 8,
            "scenes": ["scenario31"],
        },
        "candidate": {"topk": 4, "gps_protocol": "target_adapt_beambench", "gps_ablation": "branch"},
        "outputs": {"root": str(tmp_path / "selector")},
    }

    expected = label_space_metadata(cfg["data"], "mapping_enabled", num_beams=8)
    assert expected["beam_label_mapping_fingerprint"] != "badfingerprint"
    with pytest.raises(ValueError, match="mapping fingerprint|beam label-space"):
        build_topk_candidate_manifest(cfg)


def test_strict_logits_loader_fails_without_saved_logits(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="save-logits"):
        load_strict_gps_logits(tmp_path, num_beams=16)
    with pytest.raises(FileNotFoundError, match=STRICT_LOGITS_ERROR.split(".")[0]):
        load_strict_gps_logits(tmp_path, num_beams=16)


def _write_deepsense_csv(path: Path, *, split: str) -> None:
    _write_csv(
        path,
        [
            {
                "camera8": "./missing/image.jpg",
                "radar8": "./missing/radar.npy",
                "lidar8": "./missing/lidar.npy",
                "future_beam1": f"./unit1/mmWave_data/mmWave_power_{idx if split == 'train' else idx + 1}.txt",
                "seq_index": str(idx),
            }
            for idx in range(1)
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
