from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kd_sensing.data.beam_soft_targets import SoftBeamLabelConfig
from kd_sensing.data.datasets.deepsense6g import DeepSense6GDataset
from kd_sensing.data.datasets.deepsense6g_cache_paths import (
    build_deepsense6g_sample_cache,
    resolve_dataset_cache_base,
    resolve_image_cache_dir,
    resolve_lidar_cache_dir,
)
from kd_sensing.data.datasets.deepsense6g_columns import validate_required_columns
from kd_sensing.data.datasets.deepsense6g_contract import (
    normalize_beam_target_source,
    resolve_target_beam_paths,
    validate_beam_target_source_contract,
)
from kd_sensing.data.datasets.deepsense6g_label_adapters import (
    deepsense6g_soft_beam_distribution,
    deepsense6g_soft_beam_label_domain,
    deepsense6g_soft_beam_num_classes,
    read_deepsense6g_beam_label,
)
from kd_sensing.data.datasets.deepsense6g_gps_contract import (
    PAPER_SCENE_CENTER_ANGLES_RAD,
    normalize_gps_bev_xy_source,
    normalize_gps_feature_mode,
    resolve_gps_angle_offset,
)
from kd_sensing.data.layouts import deepsense6g_image_cache_root, deepsense6g_lidar_bev_cache_root
from kd_sensing.data.transform_ops.lidar import parameterized_lidar_cache_dir
from kd_sensing.utils.paths import resolve_path


def test_gps_contract_defaults_scene_calibration_and_invalid_values() -> None:
    assert normalize_gps_feature_mode(None) == "relative_polar"
    assert normalize_gps_feature_mode(" PAPER_DISTANCE_ANGLE ") == "paper_distance_angle"
    assert normalize_gps_bev_xy_source(None) == "history_relative_xy"

    angle, source = resolve_gps_angle_offset(
        gps_feature_mode="paper_distance_angle",
        scene_id=31,
        explicit_value=None,
        source=None,
    )

    assert angle == PAPER_SCENE_CENTER_ANGLES_RAD[31]
    assert source == "paper_scene_default"
    assert resolve_gps_angle_offset(
        gps_feature_mode="relative_polar",
        scene_id=31,
        explicit_value=1.0,
        source="explicit",
    ) == (None, "not_applicable")

    with pytest.raises(ValueError, match="Supported modes"):
        normalize_gps_feature_mode("legacy_raw")
    with pytest.raises(ValueError, match="gps_angle_offset_source"):
        resolve_gps_angle_offset(
            gps_feature_mode="paper_distance_angle",
            scene_id=31,
            explicit_value=None,
            source="unknown",
        )
    with pytest.raises(ValueError, match="gps_bev_xy_source"):
        normalize_gps_bev_xy_source("raw_world_xy")


def test_target_source_contract_preserves_tableiii_current_semantics() -> None:
    assert normalize_beam_target_source("beam-last") == "current"
    assert normalize_beam_target_source("future_beam1") == "future"

    input_paths = ["beam_0.txt", "beam_1.txt", "beam_2.txt"]
    future_paths = ["future_0.txt", "future_1.txt", "future_2.txt"]

    assert resolve_target_beam_paths(input_paths, future_paths, source="current", num_pred=2) == [
        "beam_1.txt",
        "beam_2.txt",
    ]
    assert resolve_target_beam_paths(input_paths, future_paths, source="future", num_pred=2) == [
        "future_0.txt",
        "future_1.txt",
    ]

    with pytest.raises(ValueError, match="num_pred <= seq_len"):
        validate_beam_target_source_contract("current", num_pred=2, seq_len=1)
    with pytest.raises(ValueError, match="beam_target_source"):
        normalize_beam_target_source("previous")


def test_column_contract_reports_missing_required_columns() -> None:
    with pytest.raises(ValueError, match=r"gps is enabled .* contains 1 gps1..gpsN columns; expected at least 2"):
        validate_required_columns(
            Path("mock.csv"),
            ("gps",),
            camera_cols=[],
            radar_cols=[],
            gps_cols=["gps1"],
            bs_gps_cols=["bs_gps1", "bs_gps2"],
            future_gps_cols=[],
            future_bs_gps_cols=[],
            lidar_cols=[],
            mmwave_cols=[],
            csi_cols=[],
            beam_cols=["beam1"],
            future_beam_cols=["future_beam1"],
            seq_len=1,
            gps_seq_len=2,
            num_pred=1,
            include_position_targets=False,
            include_history_position_targets=False,
        )


def test_cache_path_helpers_match_existing_layout(tmp_path: Path) -> None:
    data_root = tmp_path / "scenario31"
    assert resolve_dataset_cache_base(data_root, "local_cache") == data_root / "local_cache"
    assert resolve_dataset_cache_base(data_root, "outputs/cache/custom") == resolve_path("outputs/cache/custom")
    assert resolve_image_cache_dir(scene_id=31, data_root=data_root, image_cache_dir=None) == resolve_path(
        deepsense6g_image_cache_root(31)
    )

    expected_lidar = parameterized_lidar_cache_dir(
        resolve_path(deepsense6g_lidar_bev_cache_root(31)),
        bev_size=(32, 48),
        roi=(-1.0, 1.0, -2.0, 2.0),
        fov_degrees=(-45.0, 45.0),
        remove_ground=True,
        ground_z_threshold=0.2,
        background_path="bg.npy",
        background_distance_threshold=0.3,
    )

    assert (
        resolve_lidar_cache_dir(
            scene_id=31,
            data_root=data_root,
            lidar_cache_dir=None,
            lidar_bev_size=(32, 48),
            lidar_roi=(-1.0, 1.0, -2.0, 2.0),
            lidar_fov_degrees=(-45.0, 45.0),
            lidar_remove_ground=True,
            lidar_ground_z_threshold=0.2,
            lidar_background_path="bg.npy",
            lidar_background_distance_threshold=0.3,
        )
        == expected_lidar
    )


def test_sample_cache_and_soft_label_helpers_are_synthetic_contracts(tmp_path: Path) -> None:
    assert build_deepsense6g_sample_cache(None, split="train") == (None, False)
    assert build_deepsense6g_sample_cache({"enabled": False}, split="train") == (None, False)
    with pytest.raises(ValueError, match="requires sample_cache.path"):
        build_deepsense6g_sample_cache(True, split="train")
    with pytest.raises(ValueError, match="supports only 'lmdb'"):
        build_deepsense6g_sample_cache({"enabled": True, "backend": "sqlite", "path": "cache"}, split="train")

    cfg = SoftBeamLabelConfig(enabled=True, domain="auto", source="gaussian", num_classes=8, cache=True)
    assert deepsense6g_soft_beam_label_domain("train", cfg) == "source"
    assert deepsense6g_soft_beam_label_domain("test", cfg) == "target"
    mapping = SimpleNamespace(enabled=True, num_classes=8, fingerprint="synthetic", reorder_distribution=lambda x, axis=-1: x)
    assert deepsense6g_soft_beam_num_classes([3, 4], configured=None, beam_label_mapping=mapping) == 8

    beam_file = tmp_path / "beam.txt"
    values = np.zeros((8,), dtype=np.float32)
    values[3] = 1.0
    np.savetxt(beam_file, values)
    assert read_deepsense6g_beam_label(tmp_path, "beam.txt") == 3
    dist, from_power = deepsense6g_soft_beam_distribution(
        data_root=tmp_path,
        rel_path="beam.txt",
        label=3,
        cfg=cfg,
        num_classes=8,
        split="train",
        cache={},
        beam_label_mapping=mapping,
    )
    assert from_power is False
    assert dist.shape == (8,)
    assert int(np.argmax(dist)) == 3
    assert float(dist.sum()) == pytest.approx(1.0)


def test_dataset_uses_current_beam_target_without_changing_sample_contract(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    _write_minimal_deepsense_row(csv_path, current_label=7, future_label=22)

    current_dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        scene=31,
        seq_len=1,
        num_pred=1,
        enabled_modalities=["gps"],
        use_gps=True,
        gps_feature_mode="paper_distance_angle",
        gps_normalize=False,
        beam_target_source="current",
        return_metadata=True,
    )
    future_dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        scene=31,
        seq_len=1,
        num_pred=1,
        enabled_modalities=["gps"],
        use_gps=True,
        gps_feature_mode="paper_distance_angle",
        gps_normalize=False,
        beam_target_source="future",
        return_metadata=True,
    )

    current_sample = current_dataset[0]
    future_sample = future_dataset[0]

    assert set(current_sample) == {
        "input_beam",
        "target_beam",
        "gps",
        "metadata",
        "history_indices",
        "target_index",
    }
    assert int(current_sample["target_beam"][0]) == 7
    assert int(future_sample["target_beam"][0]) == 22
    assert current_sample["metadata"]["beam_target_source"] == "current"
    assert current_sample["metadata"]["target_beam_path"] == "beam_current.txt"
    assert future_sample["metadata"]["target_beam_path"] == "beam_future.txt"


def _write_minimal_deepsense_row(csv_path: Path, *, current_label: int, future_label: int) -> None:
    _write_gps(csv_path.parent / "gps_ue.txt", lat=33.0, lon=-111.0)
    _write_gps(csv_path.parent / "gps_bs.txt", lat=33.0, lon=-111.001)
    _write_beam(csv_path.parent / "beam_current.txt", label=current_label)
    _write_beam(csv_path.parent / "beam_future.txt", label=future_label)
    csv_path.write_text(
        "gps1,bs_gps1,beam1,future_beam1,seq_index\n"
        "gps_ue.txt,gps_bs.txt,beam_current.txt,beam_future.txt,0\n",
        encoding="utf-8",
    )


def _write_gps(path: Path, *, lat: float, lon: float) -> None:
    path.write_text(f"{lat:.8f}\n{lon:.8f}\n", encoding="utf-8")


def _write_beam(path: Path, *, label: int) -> None:
    values = np.zeros((64,), dtype=np.float32)
    values[int(label)] = 1.0
    np.savetxt(path, values)
