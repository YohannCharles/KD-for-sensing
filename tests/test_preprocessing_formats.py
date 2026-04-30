from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.data.transforms import read_lidar_point_cloud  # noqa: E402
from kd_sensing.preprocessing.csv import process_radar_and_create_new_csv  # noqa: E402


def test_radar_fft_csv_reads_npy_inputs(tmp_path: Path):
    radar_dir = tmp_path / "unit1" / "radar_data"
    radar_dir.mkdir(parents=True)
    raw = (np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5) / 100.0).astype(np.complex64)
    np.save(radar_dir / "radar_data_1.npy", raw)
    csv_path = tmp_path / "scenario.csv"
    csv_path.write_text("index,unit1_radar,seq_index\n1,./unit1/radar_data/radar_data_1.npy,1\n", encoding="utf-8")

    frame = process_radar_and_create_new_csv(csv_path, tmp_path, output_suffix="RA", fft_tuple=(4, 8, 6))

    output_path = tmp_path / "unit1" / "radar_data_RA" / "radar_data_1_RA.npy"
    assert output_path.exists()
    assert np.load(output_path).shape == (8, 4)
    assert frame.loc[0, "unit1_radar"] == "/unit1/radar_data_RA/radar_data_1_RA.npy"


def test_lidar_reader_reads_ascii_ply(tmp_path: Path):
    ply_path = tmp_path / "cloud.ply"
    ply_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property double x",
                "property double y",
                "property double z",
                "property ushort intensity",
                "end_header",
                "1.0 2.0 3.0 4",
                "5.0 6.0 7.0 8",
            ]
        ),
        encoding="utf-8",
    )

    points = read_lidar_point_cloud(tmp_path, "cloud.ply")

    assert points.shape == (2, 4)
    assert points.dtype == np.float32
    np.testing.assert_allclose(points[0], [1.0, 2.0, 3.0, 4.0])
