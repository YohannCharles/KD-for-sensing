from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from kd_sensing.data.deepsense6g_camera_residual import (
    CameraResidualManifestDataset,
    build_camera_residual_manifest,
)


def test_camera_manifest_fields_image_availability_and_query_mask(tmp_path: Path):
    gps_root = tmp_path / "gps"
    gps_dir = gps_root / "r15" / "mapping_disabled"
    gps_dir.mkdir(parents=True)
    data_root = tmp_path / "DeepSense6G"
    scene_root = data_root / "scenario31"
    image_dir = scene_root / "camera"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(120, 20, 10)).save(image_dir / "mmWave_power_0.jpg")
    _write_deepsense_csv(scene_root / "train_seqs_RA_GPS_LIDAR.csv", split="train", image="./camera/mmWave_power_0.jpg")
    _write_deepsense_csv(scene_root / "test_seqs_RA_GPS_LIDAR.csv", split="test", image="./missing/image.jpg")
    _write_csv(
        gps_dir / "summary_overall.csv",
        [{"protocol": "target_adapt_beambench", "ablation": "branch_mixture_circular", "DBA": "0.5"}],
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
                "topk_predictions": json.dumps([63, 0, 1]),
                "circular_error": "2",
                "signed_residual": "2",
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
        "residual": {"gps_ablation": "best", "gps_prior_fallback_sigma": 2.0, "delta_radius": 8},
        "outputs": {"root": str(tmp_path / "camera"), "analysis_root": str(tmp_path / "camera")},
    }

    result = build_camera_residual_manifest(cfg)
    rows = _read_csv(Path(result["manifest_path"]))

    assert result["support_count"] == 1
    assert result["query_count"] == 1
    assert {"split_role", "image_exists", "ae_feature_row_index", "gps_residual_delta_class"} <= set(rows[0])
    support = [row for row in rows if row["split_role"] == "support"][0]
    query = [row for row in rows if row["split_role"] == "query_test"][0]
    assert support["image_exists"] == "True"
    assert query["image_exists"] == "False"

    eval_dataset = CameraResidualManifestDataset(result["manifest_path"], include_query_labels=False)
    query_item = eval_dataset[[row["split_role"] for row in rows].index("query_test")]
    assert int(query_item["target_label"]) == -100

    train_dataset = CameraResidualManifestDataset(result["manifest_path"], training_only=True)
    assert len(train_dataset) == 1

    ae_dataset = CameraResidualManifestDataset(result["manifest_path"], stage="ae_training")
    assert len(ae_dataset) == 1
    assert ae_dataset[0]["image"].shape == (3, 64, 64)


def _write_deepsense_csv(path: Path, *, split: str, image: str) -> None:
    _write_csv(
        path,
        [
            {
                "camera8": image,
                "future_beam1": f"./unit1/mmWave_data/mmWave_power_{0 if split == 'train' else 1}.txt",
                "seq_index": "0",
            }
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
