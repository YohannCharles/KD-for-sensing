from pathlib import Path

import numpy as np
import pytest

from kd_sensing.data.transform_ops.lidar import build_lidar_bev, read_lidar_point_cloud
from kd_sensing.data.transform_ops.io import joined_resource


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


def test_resource_join_rejects_symlink_outside_data_root(tmp_path: Path):
    outside = tmp_path.parent / f"{tmp_path.name}_outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = tmp_path / "outside_link.txt"
    link.symlink_to(outside)

    try:
        with np.testing.assert_raises_regex(ValueError, "escapes data_root"):
            joined_resource(tmp_path, link.name)
    finally:
        outside.unlink(missing_ok=True)


def test_lidar_augmentation_rejects_precomputed_bev(tmp_path: Path):
    np.save(tmp_path / "bev.npy", np.zeros((3, 8, 8), dtype=np.float32))

    with pytest.raises(ValueError, match="precomputed BEV"):
        build_lidar_bev(tmp_path, "bev.npy", augment=True)
