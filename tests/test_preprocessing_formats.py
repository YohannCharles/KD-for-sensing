from pathlib import Path

import numpy as np

from kd_sensing.data.transform_ops.lidar import read_lidar_point_cloud


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
