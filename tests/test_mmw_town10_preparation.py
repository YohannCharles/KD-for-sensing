from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.data.datasets.mmw import MMWDataset  # noqa: E402
from kd_sensing.data.mmw.preparation import (  # noqa: E402
    MMWPreparationConfig,
    derive_beam_power_from_file,
    prepare_town10_skybridge,
    validate_zip_inputs,
)
from kd_sensing.engine.data_factory import build_dataset  # noqa: E402


def test_mmw_zip_prepare_manifest_sequences_split_and_reports(tmp_path: Path):
    sensor_zip, channel_zip = _write_mmw_zip_pair(tmp_path, frames=12)
    config = MMWPreparationConfig(
        sensor_zip=sensor_zip,
        channel_zip=channel_zip,
        output_root=tmp_path / "dataset",
        seq_len=8,
        pred_len=3,
        split_seed=7,
    )

    result = prepare_town10_skybridge(config)

    prepared = config.prepared_root
    manifest = pd.read_csv(prepared / "manifests" / "frame_manifest.csv")
    all_sequences = pd.read_csv(prepared / "splits" / "all_sequences.csv")
    train = pd.read_csv(prepared / "splits" / "train.csv")
    test = pd.read_csv(prepared / "splits" / "test.csv")
    metadata = json.loads((prepared / "metadata.json").read_text(encoding="utf-8"))
    report = json.loads((prepared / "sanity_report.json").read_text(encoding="utf-8"))
    split_metadata = json.loads((prepared / "splits" / "split_metadata.json").read_text(encoding="utf-8"))

    assert result["frames"] == 12
    assert result["windows"] == 2
    assert len(manifest) == 12
    assert {"agent", "frame_id", "camera0", "lidar", "gps", "channel_path", "beam_power_path"} <= set(manifest.columns)
    assert all_sequences.loc[0, "beam8"].endswith("000007.txt")
    assert all_sequences.loc[0, "future_beam1"].endswith("000008.txt")
    assert all_sequences.loc[1, "future_beam1"].endswith("000009.txt")
    assert set(train["seq_index"]).isdisjoint(set(test["seq_index"]))
    assert split_metadata["train_window_count"] == len(train)
    assert metadata["channel_to_beam"]["num_beams"] == 64
    assert metadata["channel_to_beam"]["mappings"][0]["algorithm_version"]
    assert report["valid_frame_count"] == 12
    assert report["window_count"] == 2
    assert (prepared / "beam_power" / "cav_0" / "000000.txt").exists()


def test_mmw_zip_validation_reports_absolute_missing_paths(tmp_path: Path):
    config = MMWPreparationConfig(
        sensor_zip=tmp_path / "missing_sensor.zip",
        channel_zip=tmp_path / "missing_channel.zip",
    )

    with pytest.raises(FileNotFoundError, match=str((tmp_path / "missing_sensor.zip").resolve())):
        validate_zip_inputs(config)


def test_mmw_channel_to_beam_rejects_invalid_dimensions_and_nan(tmp_path: Path):
    valid = tmp_path / "valid_paths.npy"
    invalid = tmp_path / "invalid_paths.npy"
    nan = tmp_path / "nan_paths.npy"
    channel = np.zeros(64, dtype=np.complex64)
    channel[5] = 2.0 + 0.0j
    np.save(valid, channel)
    np.save(invalid, np.ones(63, dtype=np.complex64))
    bad = channel.copy()
    bad[0] = np.nan + 0.0j
    np.save(nan, bad)

    power, meta = derive_beam_power_from_file(valid)

    assert power.shape == (64,)
    assert np.isfinite(power).all()
    assert int(np.argmax(power)) == 5
    assert meta["source_channel_field"] == "array"
    with pytest.raises(ValueError, match="incompatible|expected|does not match"):
        derive_beam_power_from_file(invalid)
    with pytest.raises(ValueError, match="NaN or Inf"):
        derive_beam_power_from_file(nan)


def test_mmw_dataset_loads_mmwave_only_and_image_fusion_lazily(tmp_path: Path):
    sensor_zip, channel_zip = _write_mmw_zip_pair(tmp_path, frames=12)
    config = MMWPreparationConfig(
        sensor_zip=sensor_zip,
        channel_zip=channel_zip,
        output_root=tmp_path / "dataset",
        seq_len=8,
        pred_len=3,
    )
    prepare_town10_skybridge(config)
    train_csv = config.prepared_root / "splits" / "train.csv"

    mmwave_only = MMWDataset(
        data_root=str(config.condition_root),
        scene=config.scenario,
        csv_name=str(train_csv),
        split="train",
        seq_len=8,
        num_pred=3,
        enabled_modalities=["mmwave"],
        mmwave_normalize=False,
    )
    sample = mmwave_only[0]

    assert set(sample) == {"input_beam", "target_beam", "mmwave"}
    assert sample["input_beam"].shape == (8,)
    assert sample["target_beam"].shape == (3,)
    assert sample["mmwave"].shape == (8, 64)
    assert sample["mmwave"].dtype == torch.float32

    frame = pd.read_csv(train_csv)
    frame.loc[:, "lidar1"] = "missing_disabled_lidar.pcd"
    corrupt_csv = config.prepared_root / "splits" / "train_missing_disabled.csv"
    frame.to_csv(corrupt_csv, index=False)
    lazy = MMWDataset(
        data_root=str(config.condition_root),
        scene=config.scenario,
        csv_name=str(corrupt_csv),
        split="train",
        seq_len=8,
        num_pred=3,
        enabled_modalities=["mmwave"],
        mmwave_normalize=False,
    )
    assert lazy[0]["mmwave"].shape == (8, 64)

    fusion = MMWDataset(
        data_root=str(config.condition_root),
        scene=config.scenario,
        csv_name=str(train_csv),
        split="train",
        seq_len=8,
        num_pred=3,
        enabled_modalities=["image", "mmwave"],
        image_size=[8, 8],
        mmwave_normalize=False,
    )
    fusion_sample = fusion[0]
    assert fusion_sample["image"].shape == (8, 3, 8, 8)
    assert fusion_sample["mmwave"].shape == (8, 64)


def test_mmw_dataset_factory_registers_scene_defaults(tmp_path: Path):
    sensor_zip, channel_zip = _write_mmw_zip_pair(tmp_path, frames=12)
    config = MMWPreparationConfig(sensor_zip=sensor_zip, channel_zip=channel_zip, output_root=tmp_path / "dataset")
    prepare_town10_skybridge(config)
    cfg = {
        "experiment": {"task": "mmwave"},
        "data": {
            "dataset": {
                "type": "mmw",
                "condition": "sunny",
                "scene": "Town10_skybridge_seed24",
                "data_root": str(config.condition_root),
                "seq_len": 8,
                "num_pred": 3,
                "mmwave_normalize": False,
            },
            "dataloader": {},
        },
    }

    dataset = build_dataset(cfg, "train")

    assert isinstance(dataset, MMWDataset)
    assert dataset.scene_slug == "Town10_skybridge_seed24"
    assert dataset[0]["mmwave"].shape == (8, 64)


def _write_mmw_zip_pair(tmp_path: Path, *, frames: int) -> tuple[Path, Path]:
    source = tmp_path / "source"
    sensor_root = source / "sensor" / "Town10" / "Town10_skybridge_seed24" / "cav_0"
    rsu_root = source / "sensor" / "Town10" / "Town10_skybridge_seed24" / "rsu_0"
    channel_root = source / "channel" / "Town10" / "Town10_skybridge_seed24" / "cav_0"
    sensor_root.mkdir(parents=True)
    rsu_root.mkdir(parents=True)
    channel_root.mkdir(parents=True)
    for idx in range(frames):
        frame_id = f"{idx:06d}"
        (sensor_root / f"{frame_id}.yaml").write_text("x: 1\n", encoding="utf-8")
        (sensor_root / f"{frame_id}.pcd").write_text("VERSION .7\n", encoding="utf-8")
        for camera_idx in range(4):
            image = Image.new("RGB", (2, 2), color=(idx, camera_idx, 10))
            image.save(sensor_root / f"{frame_id}_camera{camera_idx}.png")
        (rsu_root / f"{frame_id}.yaml").write_text("rsu: true\n", encoding="utf-8")
        (rsu_root / f"{frame_id}.pcd").write_text("VERSION .7\n", encoding="utf-8")
        (rsu_root / f"{frame_id}_radar.json").write_text("{}", encoding="utf-8")
        channel = np.zeros(64, dtype=np.complex64)
        channel[idx % 64] = complex(idx + 1, 0)
        np.save(channel_root / f"{frame_id}_paths.npy", channel)
    sensor_zip = tmp_path / "Town10_skybridge_seed24.zip"
    channel_zip = tmp_path / "Town10.zip"
    _zip_dir(source / "sensor", sensor_zip)
    _zip_dir(source / "channel", channel_zip)
    return sensor_zip, channel_zip


def _zip_dir(root: Path, target: Path) -> None:
    with zipfile.ZipFile(target, "w") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(root))
