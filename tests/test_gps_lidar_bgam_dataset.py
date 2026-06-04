from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from kd_sensing.data.deepsense6g_gps_lidar_bgam_dataset import GPSLidarBGAMDataset, collate_gps_lidar_bgam_batch
from kd_sensing.data.deepsense6g_gps_lidar_bgam_manifest import build_gps_lidar_bgam_manifest


def test_gps_lidar_bgam_manifest_dataset_lazy_lidar_and_query_normalizer(tmp_path: Path):
    bev_path = tmp_path / "bev.npy"
    raw_path = tmp_path / "points.npy"
    np.save(bev_path, np.ones((3, 8, 8), dtype=np.float32))
    np.save(raw_path, np.asarray([[0.0, 0.0, 0.5, 1.0], [1.0, 0.0, 0.2, 0.5]], dtype=np.float32))
    source = tmp_path / "top8.csv"
    rows = [_row(0, "support", bev_path, raw_path, target=2), _row(1, "query_test", bev_path, raw_path, target=3)]
    _write_csv(source, rows)
    cfg = {
        "data": {
            "top8_manifest_path": str(source),
            "support_ratio": 0.15,
            "label_space": "mapping_disabled",
            "num_beams": 8,
            "column_mapping": {"user_x": "ux", "user_y": "uy", "rsu_x": "rx", "rsu_y": "ry", "rsu_yaw": "yaw"},
        },
        "candidate": {"topk": 4, "num_beams": 8},
        "geometry": {"coordinate_frame": "local_xy", "yaw_unit": "radians", "yaw_zero_axis": "x"},
        "lidar": {"bev_size": [8, 8], "roi": [-4, 4, -4, 4, -1, 2], "input_channels": 3, "missing_policy": "zeros"},
        "outputs": {"root": str(tmp_path / "out"), "manifest_dir": "manifest"},
    }

    result = build_gps_lidar_bgam_manifest(cfg)
    manifest = Path(result["manifest_path"])
    enriched = _read_csv(manifest)

    assert result["row_count"] == 2
    assert "theta_gps" in enriched[0]
    assert "history_pseudo_beams" in enriched[0]
    assert "history_valid_mask" in enriched[0]
    assert Path(result["pseudo_history_summary_path"]).exists()
    assert enriched[0]["lidar_available"] in {"True", "true", "1"}
    assert json.loads((Path(result["metadata_path"])).read_text(encoding="utf-8"))["query_label_used_for_training"] is False

    gps_only = GPSLidarBGAMDataset(manifest, topk=4, num_beams=8, lidar_cfg=cfg["lidar"], load_lidar=False)
    item = gps_only[0]
    assert "lidar_bev" not in item
    assert item["history_pseudo_beams"].shape == (8,)
    assert str(item["history_valid_mask"].dtype) == "torch.bool"
    assert gps_only.normalizer.metadata["excluded_target_query_count"] == 1

    with_lidar = GPSLidarBGAMDataset(manifest, topk=4, num_beams=8, lidar_cfg=cfg["lidar"], load_lidar=True)
    lidar_item = with_lidar[0]
    assert lidar_item["lidar_bev"].shape == (3, 8, 8)

    raw_cfg = dict(cfg["lidar"])
    raw_cfg["profile"] = "pillar6"
    raw_dataset = GPSLidarBGAMDataset(manifest, topk=4, num_beams=8, lidar_cfg=raw_cfg, load_lidar=True, lidar_profile="pillar6")
    batch = collate_gps_lidar_bgam_batch([raw_dataset[0], raw_dataset[1]])
    assert isinstance(batch["raw_points"], list)
    assert len(batch["raw_points"]) == 2

    missing = dict(rows[0])
    missing["sample_id"] = "missing"
    missing["lidar_bev_cache_path"] = ""
    missing["lidar_path"] = ""
    missing_source = tmp_path / "missing.csv"
    _write_csv(missing_source, [missing])
    cfg["data"]["top8_manifest_path"] = str(missing_source)
    missing_result = build_gps_lidar_bgam_manifest(cfg)
    missing_row = _read_csv(Path(missing_result["manifest_path"]))[0]
    assert "missing_" in missing_row["lidar_missing_reason"]

    mutated_rows = [dict(row) for row in rows]
    mutated_rows[1]["target_label"] = "7"
    mutated_rows[1]["gt_beam"] = "7"
    mutated_source = tmp_path / "mutated.csv"
    _write_csv(mutated_source, mutated_rows)
    cfg["data"]["top8_manifest_path"] = str(mutated_source)
    cfg["outputs"]["root"] = str(tmp_path / "mutated_out")
    mutated_result = build_gps_lidar_bgam_manifest(cfg)
    mutated_enriched = _read_csv(Path(mutated_result["manifest_path"]))
    assert mutated_enriched[1]["history_pseudo_beams"] == enriched[1]["history_pseudo_beams"]
    assert mutated_enriched[1]["history_pseudo_probs"] == enriched[1]["history_pseudo_probs"]


def _row(idx: int, role: str, bev_path: Path, raw_path: Path, *, target: int) -> dict[str, str]:
    beams = [0, 2, 3, 4]
    probs = [0.5, 0.2, 0.2, 0.1]
    row = {
        "scene": "scenario31",
        "sample_id": f"scenario31:{role}:{idx}",
        "support_query_role": role,
        "split_role": role,
        "target_label": str(target),
        "gps_top1": "0",
        "gps_top1_prob": "0.5",
        "gps_top2_prob": "0.2",
        "gps_top1_top2_margin": "0.3",
        "gps_entropy": "1.0",
        "ux": "1.0",
        "uy": "0.0",
        "rx": "0.0",
        "ry": "0.0",
        "yaw": "0.0",
        "lidar_bev_cache_path": str(bev_path),
        "lidar_path": str(raw_path),
    }
    for cand_idx, (beam, prob) in enumerate(zip(beams, probs)):
        row[f"cand{cand_idx}_beam"] = str(beam)
        row[f"cand{cand_idx}_logit"] = str(np.log(prob))
        row[f"cand{cand_idx}_prob"] = str(prob)
        row[f"cand{cand_idx}_rank"] = str(cand_idx + 1)
        row[f"cand{cand_idx}_dist_to_gps_top1"] = str(beam)
    row["target_candidate_index"] = str(beams.index(target))
    row["nearest_candidate_index"] = row["target_candidate_index"]
    return row


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]
