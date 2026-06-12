from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kd_sensing.baselines.beambench.dataset_check import check_dataset, resolve_csv_fields
from kd_sensing.baselines.beambench.official import plan_official_classical_evaluation


def test_resolve_csv_fields_covers_official_and_sequence_aliases():
    fields = resolve_csv_fields(
        [
            "unit1_rgb_5",
            "unit1_lidar_5",
            "unit1_radar_5",
            "unit1_loc",
            "unit2_loc_1",
            "label",
            "future_beam_label1",
            "scene",
            "sample",
            "seq_index",
            "timestamp",
        ]
    )

    assert fields.camera_columns == ("unit1_rgb_5",)
    assert fields.lidar_columns == ("unit1_lidar_5",)
    assert fields.radar_columns == ("unit1_radar_5",)
    assert "unit1_loc" in fields.gps_columns
    assert "unit2_loc_1" in fields.gps_columns
    assert set(fields.label_columns) == {"label", "future_beam_label1"}
    assert fields.scene_columns == ("scene",)
    assert fields.sequence_columns == ("seq_index",)


def test_check_dataset_reports_missing_paths_invalid_labels_and_identifiers(tmp_path: Path):
    _write_file(tmp_path / "camera" / "a.txt")
    _write_file(tmp_path / "camera" / "b.txt")
    _write_file(tmp_path / "lidar" / "a.pcd")
    (tmp_path / "radar").mkdir()
    np.save(tmp_path / "radar" / "a.npy", np.zeros((2, 2), dtype=np.float32))
    np.save(tmp_path / "radar" / "b.npy", np.zeros((2, 2), dtype=np.float32))
    _write_gps(tmp_path / "gps" / "bs.txt")
    _write_gps(tmp_path / "gps" / "ue1.txt")
    _write_gps(tmp_path / "gps" / "ue2.txt")
    _write_beam(tmp_path / "beam" / "beam0.txt", 2)
    _write_beam(tmp_path / "beam" / "beam1.txt", 4)
    frame = pd.DataFrame(
        [
            {
                "unit1_rgb_5": "camera/a.txt",
                "unit1_lidar_5": "lidar/a.pcd",
                "unit1_radar_5": "radar/a.npy",
                "unit1_loc": "gps/bs.txt",
                "unit2_loc_1": "gps/ue1.txt",
                "unit2_loc_2": "gps/ue2.txt",
                "future_beam1": "beam/beam0.txt",
                "label": 2,
                "scene": 31,
                "sample": "s0",
                "seq": 0,
                "timestamp": 100,
            },
            {
                "unit1_rgb_5": "camera/b.txt",
                "unit1_lidar_5": "lidar/missing.pcd",
                "unit1_radar_5": "radar/b.npy",
                "unit1_loc": "gps/bs.txt",
                "unit2_loc_1": "gps/ue1.txt",
                "unit2_loc_2": "gps/ue2.txt",
                "future_beam1": "beam/beam1.txt",
                "label": 64,
                "scene": 31,
                "sample": "s1",
                "seq": 0,
                "timestamp": 101,
            },
        ]
    )
    csv = tmp_path / "mock.csv"
    frame.to_csv(csv, index=False)

    report = check_dataset(tmp_path, csv, num_beams=64, beam_shift=0)

    assert report["ok"] is False
    assert report["sensor_files"]["lidar"]["missing_count"] == 1
    assert report["labels"]["invalid_count"] == 1
    assert report["labels"]["raw_max"] == 64
    assert report["identifiers"]["scene"]["available"] is True
    assert report["identifiers"]["sample"]["unique_counts"]["sample"] == 2
    assert report["identifiers"]["sequence"]["available"] is True


def test_check_dataset_missing_csv_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        check_dataset(tmp_path, "missing.csv")


def test_plan_official_classical_evaluation_checks_required_artifacts(tmp_path: Path):
    official_root = tmp_path / "BeamBench"
    data_root = tmp_path / "raw_data" / "test"
    model_dir = official_root / "results" / "models"
    official_root.mkdir(parents=True)
    data_root.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    (official_root / "classical.py").write_text("print('mock')\n", encoding="utf-8")
    (data_root / "ml_challenge_test_multi_modal.csv").write_text("unit2_loc_1,unit2_loc_2,unit1_loc\n", encoding="utf-8")
    np.savetxt(model_dir / "classic.npy", np.zeros(4, dtype=np.float32))

    report = plan_official_classical_evaluation(
        official_root=official_root,
        data_folder=data_root,
        output_dir=tmp_path / "plan",
    )

    assert report["mode"] == "official_classical_evaluation"
    assert "classical.py" in report["command_text"]
    assert report["prediction_path"].endswith("results/topk/classical.csv")
    assert report["blocked"] is True
    assert any("classic_angle.npy" in reason for reason in report["blocked_reasons"])
    assert (tmp_path / "plan" / "official_classical_plan.json").exists()


def _write_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("MOCK\n", encoding="utf-8")


def _write_gps(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray([42.0, -71.0], dtype=np.float32))


def _write_beam(path: Path, label: int, num_beams: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.zeros(num_beams, dtype=np.float32)
    values[int(label)] = 1.0
    np.savetxt(path, values)
