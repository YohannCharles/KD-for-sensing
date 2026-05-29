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
    ChannelFile,
    MMWPreparationConfig,
    build_prepared_artifacts,
    derive_beam_power_from_file,
    load_preparation_config,
    prepare_town10_skybridge,
    validate_zip_inputs,
)
from kd_sensing.data.mmw.radio_semantic import RadioSemanticLabelBuilder  # noqa: E402
from kd_sensing.engine.data_factory import build_dataset  # noqa: E402
from kd_sensing.preprocessing.mmw_radar import generate_mmw_radar_maps  # noqa: E402


def test_radio_semantic_label_builder_peak_spread_fallback_and_invalid():
    builder = RadioSemanticLabelBuilder.from_config(
        {"mode": "peak_spread", "group_size": 8, "num_spread_bins": 3, "entropy_thresholds": [0.35, 0.65]}
    )
    narrow = np.zeros(64, dtype=np.float32)
    narrow[10] = 10.0
    spread = np.ones(64, dtype=np.float32)
    spread[10] = 1.1

    narrow_result = builder.derive(beam_power=narrow, beam_label=10)
    spread_result = builder.derive(beam_power=spread, beam_label=10)
    fallback = builder.derive(beam_power=None, beam_label=18)
    bad = narrow.copy()
    bad[0] = np.nan
    invalid = builder.derive(beam_power=bad, beam_label=18)

    assert narrow_result.label == 3
    assert narrow_result.diagnostics["best_beam"] == 10
    assert spread_result.label == 5
    assert fallback.label == 2
    assert fallback.diagnostics["radio_semantic_mode"] == "coarse"
    assert fallback.diagnostics["fallback_reason"] == "missing_beam_power"
    assert invalid.label is None
    assert invalid.diagnostics["unavailable_reason"] == "invalid_power_vector_nonfinite"
    counts = builder.class_counts([narrow_result.label, spread_result.label, None])
    assert counts["available_count"] == 2


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
    assert {
        "condition",
        "town",
        "sensor_scenario",
        "channel_scenario",
        "channel_agent",
        "sample_id",
        "agent",
        "frame_id",
        "camera0",
        "lidar",
        "gps",
        "channel_path",
        "beam_power_path",
        "coarse_sector",
        "relative_geometry_json",
    } <= set(manifest.columns)
    assert all_sequences.loc[0, "beam8"].endswith("000007.txt")
    assert all_sequences.loc[0, "future_beam1"].endswith("000008.txt")
    assert all_sequences.loc[1, "future_beam1"].endswith("000009.txt")
    assert set(train["seq_index"]).isdisjoint(set(test["seq_index"]))
    assert split_metadata["train_window_count"] == len(train)
    assert metadata["channel_to_beam"]["num_beams"] == 64
    assert metadata["channel_to_beam"]["mappings"][0]["algorithm_version"]
    assert metadata["radio_semantic"]["radio_semantic_mode"] == "peak_spread"
    assert metadata["radio_semantic"]["class_counts"]["available_count"] == 12
    assert report["valid_frame_count"] == 12
    assert report["window_count"] == 2
    assert (prepared / "beam_power" / "cav_0" / "000000.txt").exists()
    assert (config.condition_root / "data_availability.json").exists()


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


def test_mmw_radar_maps_preprocessor_materializes_radar_and_gps_columns(tmp_path: Path):
    root = tmp_path / "MMW" / "sunny"
    scene = "Town10_test_seed0"
    prepared = root / "Prepared" / scene
    splits = prepared / "splits"
    manifests = prepared / "manifests"
    radar_root = root / "Sensor_Data" / scene / "rsu_1"
    radar_root.mkdir(parents=True)
    splits.mkdir(parents=True)
    manifests.mkdir(parents=True)
    (radar_root / "000001.json").write_text(
        json.dumps(
            [
                {"depth": 10.0, "azimuth": -0.1, "velocity": -1.0, "altitude": 0.0},
                {"depth": 20.0, "azimuth": 0.2, "velocity": 2.0, "altitude": 0.1},
            ]
        ),
        encoding="utf-8",
    )
    rsu_json = json.dumps({"agents": {"rsu_1": {"radar": f"Sensor_Data/{scene}/rsu_1/000001.json"}}})
    pd.DataFrame([{"frame_id": "000001", "rsu_json": rsu_json}]).to_csv(
        manifests / "frame_manifest.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "seq_index": 0,
                "beam1": f"Prepared/{scene}/beam_power/cav_1/000001.txt",
                "future_beam1": f"Prepared/{scene}/beam_power/cav_1/000001.txt",
                "gps1": f"Sensor_Data/{scene}/cav_1/000001.yaml",
            }
        ]
    ).to_csv(splits / "train.csv", index=False)

    report = generate_mmw_radar_maps(data_root=root, scenes=[scene], progress=False)

    ra = np.load(prepared / "derived" / "radar_maps" / "rsu_1" / "000001_RA.npy")
    da = np.load(prepared / "derived" / "radar_maps" / "rsu_1" / "000001_DA.npy")
    train_with_columns = pd.read_csv(splits / "train_with_radar_with_bs_gps.csv")
    assert report["generated"] == 1
    assert ra.shape == (128, 64)
    assert da.shape == (128, 64)
    assert float(ra.max()) > 0.0
    assert train_with_columns.loc[0, "radar1"].endswith("000001_RA.npy")
    assert train_with_columns.loc[0, "bs_gps1"] == f"Sensor_Data/{scene}/rsu_1/000001.yaml"


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

    geometry = MMWDataset(
        data_root=str(config.condition_root),
        scene=config.scenario,
        csv_name=str(train_csv),
        split="train",
        seq_len=8,
        num_pred=3,
        enabled_modalities=["mmwave"],
        return_geometry=True,
        return_metadata=True,
        mmwave_normalize=False,
    )
    geometry_sample = geometry[0]
    assert geometry_sample["geometry"].shape == (8, 8)
    assert geometry_sample["geometry_mask"].shape == (8, 8)
    assert geometry_sample["geometry_mask"].any()
    assert geometry_sample["metadata"]["dataset_family"] == "MMW"

    radio = MMWDataset(
        data_root=str(config.condition_root),
        scene=config.scenario,
        csv_name=str(train_csv),
        split="train",
        seq_len=8,
        num_pred=3,
        enabled_modalities=["mmwave"],
        radio_semantic={"enabled": True, "mode": "peak_spread", "group_size": 8},
        mmwave_normalize=False,
    )
    radio_sample = radio[0]
    assert "radio_semantic_label" in radio_sample
    assert "beam_power" in radio_sample
    assert "channel_path" not in radio_sample
    assert "csi" not in radio_sample
    assert radio_sample["radio_semantic_label"].shape == (3,)
    assert radio_sample["beam_power"].shape == (3, 64)
    assert radio_sample["radio_semantic_available"].all()
    assert radio_sample["sample_id"]
    assert radio_sample["domain_metadata"]["dataset_family"] == "MMW"


def test_mmw_dataset_rebuilds_incomplete_derived_bs_gps_csv(tmp_path: Path):
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
    derived_csv = config.prepared_root / "splits" / "train_with_bs_gps.csv"
    source_frame = pd.read_csv(train_csv)
    source_rows = len(source_frame)
    header = source_frame.head(0).copy()
    for idx in range(1, 9):
        header[f"bs_gps{idx}"] = []
    header.to_csv(derived_csv, index=False)

    dataset = MMWDataset(
        data_root=str(config.condition_root),
        scene=config.scenario,
        csv_name=str(train_csv),
        split="train",
        seq_len=8,
        num_pred=3,
        enabled_modalities=["gps", "mmwave"],
        gps_normalize=False,
        mmwave_normalize=False,
    )

    rebuilt = pd.read_csv(derived_csv)
    assert dataset.root_csv == derived_csv.resolve()
    assert len(dataset) == source_rows
    assert len(rebuilt) == source_rows
    assert {"bs_gps1", "bs_gps8"} <= set(rebuilt.columns)


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


def test_mmw_alias_matching_records_channel_agent_and_scenario(tmp_path: Path):
    sensor_zip, channel_zip = _write_mmw_zip_pair(
        tmp_path,
        frames=12,
        sensor_scenario="Town10_skybridge_seed24",
        channel_scenario="Town10_skybridge",
        agents=("cav_1",),
        rsu_agent="rsu_1",
    )
    config = MMWPreparationConfig(
        sensor_zip=sensor_zip,
        channel_zip=channel_zip,
        output_root=tmp_path / "dataset",
        channel_scenario_aliases={"Town10_skybridge_seed24": "Town10_skybridge"},
    )

    prepare_town10_skybridge(config)

    manifest = pd.read_csv(config.prepared_root / "manifests" / "frame_manifest.csv")
    metadata = json.loads((config.prepared_root / "metadata.json").read_text(encoding="utf-8"))
    report = json.loads((config.prepared_root / "sanity_report.json").read_text(encoding="utf-8"))
    assert set(manifest["agent"]) == {"cav_1"}
    assert set(manifest["channel_agent"]) == {"cav_1"}
    assert set(manifest["channel_scenario"]) == {"Town10_skybridge"}
    assert metadata["scenario_alias"]["channel_scenario"] == "Town10_skybridge"
    assert report["scenario_alias"]["matched_frame_count"] == 12


def test_mmw_preparation_rejects_channel_agent_mismatch(tmp_path: Path):
    sensor_zip, channel_zip = _write_mmw_zip_pair(tmp_path, frames=12)
    config = MMWPreparationConfig(sensor_zip=sensor_zip, channel_zip=channel_zip, output_root=tmp_path / "dataset")
    prepare_town10_skybridge(config)
    from kd_sensing.data.mmw.preparation import index_sensor_frames

    sensor_index = index_sensor_frames(config.sensor_root, town=config.town, scenario=config.scenario)
    bad_channel = {
        ("cav_0", "000000"): ChannelFile(
            path=config.channel_root / "Town10" / config.scenario / "cav_0" / "000000_paths.npy",
            agent="cav_2",
            frame_id="000000",
            scenario=config.scenario,
        )
    }

    with pytest.raises(ValueError, match="agent mismatches|no valid sequence"):
        build_prepared_artifacts(
            config,
            {"cav_0": {"000000": sensor_index["cav_0"]["000000"]}},
            bad_channel,
            zip_info=validate_zip_inputs(config),
        )


def test_mmw_preparation_config_overrides_download_paths(tmp_path: Path):
    sensor_zip, channel_zip = _write_mmw_zip_pair(tmp_path, frames=12)
    config_path = tmp_path / "mmw.yaml"
    config_path.write_text(
        "mmw:\n"
        "  sensor_zip: missing_sensor.zip\n"
        "  channel_zip: missing_channel.zip\n"
        "  output_root: dataset\n",
        encoding="utf-8",
    )

    config = load_preparation_config(
        config_path,
        [
            f"mmw.sensor_zip={sensor_zip}",
            f"mmw.channel_zip={channel_zip}",
            f"mmw.output_root={tmp_path / 'dataset'}",
        ],
    )

    assert config.sensor_zip == sensor_zip
    assert config.channel_zip == channel_zip
    result = prepare_town10_skybridge(config, dry_run=True)
    assert result["status"] == "dry_run"


def _write_mmw_zip_pair(
    tmp_path: Path,
    *,
    frames: int,
    sensor_scenario: str = "Town10_skybridge_seed24",
    channel_scenario: str = "Town10_skybridge_seed24",
    agents: tuple[str, ...] = ("cav_0",),
    rsu_agent: str = "rsu_0",
) -> tuple[Path, Path]:
    source = tmp_path / "source"
    sensor_roots = [source / "sensor" / "Town10" / sensor_scenario / agent for agent in agents]
    rsu_root = source / "sensor" / "Town10" / sensor_scenario / rsu_agent
    channel_roots = [source / "channel" / "Town10" / channel_scenario / agent for agent in agents]
    for sensor_root in sensor_roots:
        sensor_root.mkdir(parents=True)
    rsu_root.mkdir(parents=True)
    for channel_root in channel_roots:
        channel_root.mkdir(parents=True)
    for idx in range(frames):
        frame_id = f"{idx:06d}"
        for sensor_root in sensor_roots:
            (sensor_root / f"{frame_id}.yaml").write_text(_cav_yaml(idx), encoding="utf-8")
            (sensor_root / f"{frame_id}.pcd").write_text("VERSION .7\n", encoding="utf-8")
            for camera_idx in range(4):
                image = Image.new("RGB", (2, 2), color=(idx, camera_idx, 10))
                image.save(sensor_root / f"{frame_id}_camera{camera_idx}.png")
        (rsu_root / f"{frame_id}.yaml").write_text(_rsu_yaml(), encoding="utf-8")
        (rsu_root / f"{frame_id}.pcd").write_text("VERSION .7\n", encoding="utf-8")
        (rsu_root / f"{frame_id}.json").write_text("{}", encoding="utf-8")
        for channel_root in channel_roots:
            channel = np.zeros(64, dtype=np.complex64)
            channel[idx % 64] = complex(idx + 1, 0)
            np.save(channel_root / f"{frame_id}_paths.npy", channel)
    sensor_zip = tmp_path / f"{sensor_scenario}.zip"
    channel_zip = tmp_path / "Town10.zip"
    _zip_dir(source / "sensor", sensor_zip)
    _zip_dir(source / "channel", channel_zip)
    return sensor_zip, channel_zip


def _cav_yaml(idx: int) -> str:
    return (
        "actor: cav_0\n"
        f"frame: {idx}\n"
        "sensors:\n"
        "  vehicle_pose:\n"
        "    location: {x: 1.0, y: 2.0, z: 0.0}\n"
        "    rotation: {yaw: 10.0, pitch: 0.0, roll: 0.0}\n"
        "  vehicle_speed:\n"
        "    speed: {x: 1.0, y: 0.0, z: 0.0}\n"
        "  GPS:\n"
        "    location: {x: 1.0, y: 2.0, z: 0.0}\n"
        "vehicles:\n"
        "  0: {bp_id: static.prop.trafficwarning}\n"
    )


def _rsu_yaml() -> str:
    return (
        "actor: rsu_0\n"
        "sensors:\n"
        "  rsu_pose:\n"
        "    location: {x: 0.0, y: 0.0, z: 0.0}\n"
        "    rotation: {yaw: 0.0, pitch: 0.0, roll: 0.0}\n"
        "vehicles:\n"
        "  0: {bp_id: static.prop.trafficwarning}\n"
    )


def _zip_dir(root: Path, target: Path) -> None:
    with zipfile.ZipFile(target, "w") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(root))
