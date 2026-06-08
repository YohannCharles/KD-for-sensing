from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def create_mock_dataset(
    root: str | Path,
    *,
    rows: int = 12,
    num_beams: int = 64,
    scene: str = "MOCK-S31",
) -> dict[str, str | int | bool]:
    base = Path(root)
    for name in ("camera", "radar", "lidar", "gps", "beam"):
        (base / name).mkdir(parents=True, exist_ok=True)
    records = []
    for idx in range(int(rows)):
        label = int((idx * 7) % int(num_beams))
        camera_rel = f"camera/mock_camera_{idx:03d}.txt"
        radar_rel = f"radar/mock_radar_{idx:03d}.npy"
        lidar_rel = f"lidar/mock_lidar_{idx:03d}.pcd"
        bs_rel = f"gps/mock_bs_{idx:03d}.txt"
        ue1_rel = f"gps/mock_ue1_{idx:03d}.txt"
        ue2_rel = f"gps/mock_ue2_{idx:03d}.txt"
        beam_rel = f"beam/mock_beam_{idx:03d}.txt"
        (base / camera_rel).write_text(f"MOCK camera placeholder {idx}\n", encoding="utf-8")
        np.save(base / radar_rel, np.full((2, 4), float(idx), dtype=np.float32))
        _write_mock_pcd(base / lidar_rel, idx)
        np.savetxt(base / bs_rel, np.asarray([42.0, -71.0], dtype=np.float32))
        np.savetxt(base / ue1_rel, np.asarray([42.0 + idx * 1e-4, -71.0], dtype=np.float32))
        np.savetxt(base / ue2_rel, np.asarray([42.0 + idx * 1e-4, -71.0 + 1e-4], dtype=np.float32))
        one_hot = np.zeros(int(num_beams), dtype=np.float32)
        one_hot[label] = 1.0
        np.savetxt(base / beam_rel, one_hot)
        records.append(
            {
                "unit1_rgb_5": camera_rel,
                "unit1_radar_5": radar_rel,
                "unit1_lidar_5": lidar_rel,
                "unit1_loc": bs_rel,
                "unit2_loc_1": ue1_rel,
                "unit2_loc_2": ue2_rel,
                "future_beam1": beam_rel,
                "label": label,
                "scene": scene,
                "sample": f"MOCK-{idx:03d}",
                "seq": idx // 3,
                "timestamp": 1_700_000_000 + idx,
                "mock_feature_0": float(idx) / max(float(rows - 1), 1.0),
                "mock_feature_1": float(np.sin(idx)),
                "mock_feature_2": float(np.cos(idx)),
                "mock_data": True,
            }
        )
    csv_path = base / "ml_challenge_mock_multi_modal.csv"
    pd.DataFrame.from_records(records).to_csv(csv_path, index=False)
    metadata = {
        "mock_data": True,
        "dataset_family": "BeamBench/DeepSense6G",
        "num_rows": int(rows),
        "num_beams": int(num_beams),
        "csv": csv_path.name,
        "warning": "MOCK dataset for code-path smoke tests only; do not compare as a real baseline.",
    }
    (base / "MOCK_DATASET_MARKER.txt").write_text("MOCK BeamBench smoke dataset\n", encoding="utf-8")
    (base / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "mock_data": True,
        "data_root": str(base),
        "csv": str(csv_path),
        "row_count": int(rows),
        "num_beams": int(num_beams),
    }


def _write_mock_pcd(path: Path, idx: int) -> None:
    content = "\n".join(
        [
            "# .PCD v0.7 - Point Cloud Data file format",
            "# MOCK LiDAR placeholder",
            "VERSION 0.7",
            "FIELDS x y z",
            "SIZE 4 4 4",
            "TYPE F F F",
            "COUNT 1 1 1",
            "WIDTH 1",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            "POINTS 1",
            "DATA ascii",
            f"{float(idx):.3f} 0.0 0.0",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
