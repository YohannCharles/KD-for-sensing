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
    GROUP_SAFE_TIME_BLOCK,
    MMWPreparationConfig,
    PreparedFrame,
    build_prepared_artifacts,
    build_sequence_rows,
    build_sequence_splits_from_manifest,
    compute_split_leakage_diagnostics,
    derive_beam_power_from_file,
    load_preparation_config,
    prepare_town10_skybridge,
    split_sequence_rows,
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


def test_mmw_prepare_split_tag_writes_isolated_sequence_splits(tmp_path: Path):
    sensor_zip, channel_zip = _write_mmw_zip_pair(tmp_path, frames=12)
    config = MMWPreparationConfig(
        sensor_zip=sensor_zip,
        channel_zip=channel_zip,
        output_root=tmp_path / "dataset",
        seq_len=5,
        pred_len=6,
        split_tag="l5p6",
    )

    result = prepare_town10_skybridge(config)

    split_dir = config.prepared_root / "splits" / "l5p6"
    train = pd.read_csv(split_dir / "train.csv")
    metadata = json.loads((config.prepared_root / "metadata_l5p6.json").read_text(encoding="utf-8"))
    report = json.loads((config.prepared_root / "sanity_report_l5p6.json").read_text(encoding="utf-8"))
    assert result["artifacts"]["train_csv"].endswith("splits/l5p6/train.csv")
    assert "future_beam6" in train.columns
    assert metadata["seq_len"] == 5
    assert metadata["pred_len"] == 6
    assert metadata["split_tag"] == "l5p6"
    assert report["split_tag"] == "l5p6"


def test_mmw_public_sequence_split_utility_writes_metadata(tmp_path: Path):
    sensor_zip, channel_zip = _write_mmw_zip_pair(tmp_path, frames=12)
    config = MMWPreparationConfig(
        sensor_zip=sensor_zip,
        channel_zip=channel_zip,
        output_root=tmp_path / "dataset",
        seq_len=8,
        pred_len=3,
    )
    prepare_town10_skybridge(config)

    result = build_sequence_splits_from_manifest(
        data_root=config.condition_root,
        scene=config.scenario,
        seq_len=5,
        pred_len=6,
        split_tag="l5p6",
        split_seed=17,
        train_ratio=0.5,
    )

    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert result["outputs"]["train_csv"].endswith("splits/l5p6/train.csv")
    assert Path(result["outputs"]["train_csv"]).exists()
    assert metadata["public_utility"] == "kd_sensing.data.mmw.preparation.build_sequence_splits_from_manifest"
    assert metadata["manifest_path"].endswith("manifests/frame_manifest.csv")
    assert metadata["split_seed"] == 17
    assert metadata["train_ratio"] == pytest.approx(0.5)
    assert metadata["seq_len"] == 5
    assert metadata["num_pred"] == 6
    assert metadata["condition"] == "sunny"
    assert metadata["scenario"] == config.scenario
    assert metadata["manifest_rows"] == 12
    assert metadata["window_count"] == result["windows"]
    assert metadata["outputs"]["metadata"] == result["metadata_path"]


def test_mmw_group_safe_split_metadata_has_no_frame_overlap_or_guard_violations(tmp_path: Path):
    sensor_zip, channel_zip = _write_mmw_zip_pair(tmp_path, frames=64)
    config = MMWPreparationConfig(
        sensor_zip=sensor_zip,
        channel_zip=channel_zip,
        output_root=tmp_path / "dataset",
        seq_len=5,
        pred_len=3,
        split_seed=7,
        split_tag="l5p3_group_safe",
    )

    result = prepare_town10_skybridge(config)

    split_dir = config.prepared_root / "splits" / "l5p3_group_safe"
    train = pd.read_csv(split_dir / "train.csv")
    test = pd.read_csv(split_dir / "test.csv")
    split_metadata = json.loads((split_dir / "split_metadata.json").read_text(encoding="utf-8"))
    diagnostics = split_metadata["leakage_diagnostics"]

    assert result["windows"] == 57
    assert split_metadata["split_strategy"] == GROUP_SAFE_TIME_BLOCK
    assert split_metadata["split_protocol"] == "mmw_sequence_split_v2"
    assert split_metadata["strict_validation_eligible"] is True
    assert split_metadata["eligibility_reasons"] == []
    assert split_metadata["guard_band_frames"] >= config.seq_len + config.pred_len - 1
    assert len(split_metadata["train_groups"]) > 0
    assert len(split_metadata["test_groups"]) > 0
    assert set(train["seq_index"]).isdisjoint(set(test["seq_index"]))
    assert diagnostics["train_test_frame_overlap_count"] == 0
    assert diagnostics["guard_band_violations"] == 0
    assert diagnostics["test_window_max_frame_overlap"]["max"] < diagnostics["window_length_frames"]
    assert split_metadata["label_distribution"]["train"]
    assert split_metadata["group_key_fields"] == [
        "condition",
        "town",
        "sensor_scenario",
        "agent",
        "contiguous_segment_id",
        "time_block_id",
    ]


def test_mmw_sequence_split_rejects_unsupported_strategy():
    rows = _overlapping_window_rows(12, seq_len=3, pred_len=2)

    with pytest.raises(ValueError, match="Unsupported MMW split_strategy"):
        split_sequence_rows(
            rows,
            seed=0,
            train_ratio=0.5,
            strategy="random_window",
            seq_len=3,
            pred_len=2,
        )


def test_mmw_sequence_rows_record_stable_group_and_window_metadata():
    frames = [
        PreparedFrame(
            condition="sunny",
            town="Town10",
            sensor_scenario="Town10_fixture",
            agent="cav_0",
            frame_id=f"{idx:06d}",
            camera0=f"camera/{idx:06d}.png",
            lidar=f"lidar/{idx:06d}.pcd",
            gps=f"gps/{idx:06d}.yaml",
            beam_power_path=f"beam/{idx:06d}.txt",
            beam_label=idx % 4,
        )
        for idx in [0, 1, 2, 3, 4, 7, 8, 9, 10, 11]
    ]

    rows, non_contiguous = build_sequence_rows(frames, seq_len=3, pred_len=2)

    assert non_contiguous == 1
    assert {row["contiguous_segment_id"] for row in rows} == {
        "sunny:Town10:Town10_fixture:cav_0:segment_0000",
        "sunny:Town10:Town10_fixture:cav_0:segment_0001",
    }
    first = rows[0]
    assert json.loads(first["history_frame_ids_json"]) == ["000000", "000001", "000002"]
    assert json.loads(first["future_frame_ids_json"]) == ["000003", "000004"]
    assert json.loads(first["window_frame_ids_json"]) == ["000000", "000001", "000002", "000003", "000004"]
    assert json.loads(first["future_label_sequence_json"]) == [3, 0]
    assert first["future_label_sequence_key"] == "3,0"
    assert first["window_start_frame"] == "000000"
    assert first["window_end_frame"] == "000004"


def test_mmw_leakage_diagnostics_cover_overlap_adjacency_and_future_reuse():
    rows = _overlapping_window_rows(12, seq_len=3, pred_len=2)
    train_rows = [rows[index] for index in (0, 1, 4, 5)]
    test_rows = [rows[index] for index in (2, 3, 6, 7)]
    diagnostics = compute_split_leakage_diagnostics(
        train_rows,
        test_rows,
        seq_len=3,
        pred_len=2,
    )

    assert diagnostics["train_test_frame_overlap_count"] > 0
    assert diagnostics["test_window_max_frame_overlap"]["max"] > 0
    assert diagnostics["adjacent_window_cross_split_ratio"] > 0.0
    assert diagnostics["future_label_sequence_reuse_ratio"] > 0.0


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

    assert {"input_beam", "target_beam", "mmwave", "sample_id", "domain_metadata"} <= set(sample)
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


def test_mmw_sensor_assisted_sample_shapes_and_metadata(tmp_path: Path):
    sensor_zip, channel_zip = _write_mmw_zip_pair(tmp_path, frames=12, rsu_agent="rsu_1")
    config = MMWPreparationConfig(
        sensor_zip=sensor_zip,
        channel_zip=channel_zip,
        output_root=tmp_path / "dataset",
        seq_len=5,
        pred_len=3,
    )
    prepare_town10_skybridge(config)
    _materialize_sensor_assisted_fixture(config, frames=12)
    train_csv = config.prepared_root / "splits" / "train.csv"

    dataset = MMWDataset(
        data_root=str(config.condition_root),
        scene=config.scenario,
        csv_name=str(train_csv),
        split="train",
        seq_len=5,
        num_pred=3,
        enabled_modalities=["image", "gps", "lidar", "radar"],
        image_size=[8, 8],
        gps_normalize=False,
        lidar_bev_size=[8, 8],
        lidar_normalize=False,
        return_metadata=True,
        return_modality_availability=True,
    )
    sample = dataset[0]

    assert sample["image"].shape == (5, 3, 8, 8)
    assert sample["gps"].shape == (5, 3)
    assert sample["lidar"].shape == (5, 3, 8, 8)
    assert sample["radar_ra"].shape == (5, 128, 64)
    assert sample["radar_da"].shape == (5, 128, 64)
    assert sample["target_beam"].shape == (3,)
    assert "mmwave" not in sample
    assert sample["metadata"]["condition"] == "sunny"
    assert sample["metadata"]["town"] == "Town10"
    assert sample["metadata"]["scenario"] == config.scenario
    assert sample["metadata"]["sample_id"]
    assert sample["metadata"]["modality_availability"]["1"]["cav"]["gps"] is True
    assert sample["domain_metadata"]["scenario"] == config.scenario


def test_mmw_sensor_assisted_missing_radar_maps_error_is_actionable(tmp_path: Path):
    sensor_zip, channel_zip = _write_mmw_zip_pair(tmp_path, frames=12, rsu_agent="rsu_1")
    config = MMWPreparationConfig(
        sensor_zip=sensor_zip,
        channel_zip=channel_zip,
        output_root=tmp_path / "dataset",
        seq_len=5,
        pred_len=3,
    )
    prepare_town10_skybridge(config)

    with pytest.raises(ValueError) as excinfo:
        MMWDataset(
            data_root=str(config.condition_root),
            scene=config.scenario,
            csv_name=str(config.prepared_root / "splits" / "train.csv"),
            split="train",
            seq_len=5,
            num_pred=3,
            enabled_modalities=["radar"],
        )

    message = str(excinfo.value)
    assert "kd-sensing-preprocess" in message
    assert "configs/preprocess/mmw_radar_maps.yaml" in message


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


def _materialize_sensor_assisted_fixture(config: MMWPreparationConfig, *, frames: int) -> None:
    rsu_root = config.condition_root / "Sensor_Data" / config.scenario / "rsu_1"
    rsu_root.mkdir(parents=True, exist_ok=True)
    for idx in range(frames):
        (rsu_root / f"{idx:06d}.yaml").write_text(_rsu_yaml(), encoding="utf-8")
    for pcd_path in config.condition_root.rglob("*.pcd"):
        _write_valid_ascii_pcd(pcd_path)
    radar_root = config.prepared_root / "derived" / "radar_maps" / "rsu_1"
    radar_root.mkdir(parents=True, exist_ok=True)
    for idx in range(frames):
        ra = np.zeros((128, 64), dtype=np.float32)
        da = np.zeros((128, 64), dtype=np.float32)
        ra[idx % 128, idx % 64] = float(idx + 1)
        da[idx % 128, idx % 64] = float(idx + 1) / 2.0
        np.save(radar_root / f"{idx:06d}_RA.npy", ra)
        np.save(radar_root / f"{idx:06d}_DA.npy", da)


def _overlapping_window_rows(count: int, *, seq_len: int, pred_len: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    window_len = int(seq_len) + int(pred_len)
    for idx in range(count):
        future = [idx % 2, (idx + 1) % 2]
        rows.append(
            {
                "seq_index": idx,
                "condition": "sunny",
                "town": "Town10",
                "sensor_scenario": "Town10_fixture",
                "agent": "cav_0",
                "contiguous_segment_id": "seg0",
                "window_start_frame": f"{idx:06d}",
                "window_end_frame": f"{idx + window_len - 1:06d}",
                "future_end_frame": f"{idx + window_len - 1:06d}",
                "window_frame_ids_json": json.dumps([f"{frame:06d}" for frame in range(idx, idx + window_len)]),
                "future_label_sequence_json": json.dumps(future),
                "future_label_sequence_key": ",".join(str(label) for label in future),
            }
        )
    return rows


def _write_valid_ascii_pcd(path: Path) -> None:
    path.write_text(
        "VERSION .7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        "WIDTH 2\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        "POINTS 2\n"
        "DATA ascii\n"
        "0.0 0.0 0.0 1.0\n"
        "1.0 1.0 0.5 0.8\n",
        encoding="utf-8",
    )


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
