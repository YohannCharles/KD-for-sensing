import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from kd_sensing.data.datasets.mmw import MMWDataset
from kd_sensing.data.mmw.preparation_splits import build_sequence_splits_from_manifest, compute_split_identity_audit
from kd_sensing.engine.evaluation_pass_runtime import sample_ids_from_batch


def test_mmw_dataset_accepts_prepared_four_sensor_csv(tmp_path: Path):
    fields = [
        f"{prefix}{index}"
        for prefix, count in (("camera", 5), ("radar", 5), ("gps", 5), ("bs_gps", 5), ("lidar", 5), ("future_beam_label", 1))
        for index in range(1, count + 1)
    ]
    fields.append("sample_id")
    row = {
        **{f"camera{index}": f"assets/camera{index}.png" for index in range(1, 6)},
        **{f"radar{index}": f"assets/radar{index}_RA.npy" for index in range(1, 6)},
        **{f"gps{index}": f"assets/gps{index}.yaml" for index in range(1, 6)},
        **{f"bs_gps{index}": f"assets/bs_gps{index}.yaml" for index in range(1, 6)},
        **{f"lidar{index}": "assets/lidar.npy" for index in range(1, 6)},
    }
    row["future_beam_label1"] = 63
    for value in row.values():
        if not isinstance(value, str):
            continue
        path = tmp_path / value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        if "_RA" in value:
            (tmp_path / value.replace("_RA", "_DA")).touch()
    row["sample_id"] = "shared-sample"
    for name in ("train.csv", "test.csv"):
        with (tmp_path / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)

    dataset = MMWDataset(
        condition="sunny",
        scene="Town03_5wayroad_seed28",
        data_root=str(tmp_path),
        train_csv_name="train.csv",
        test_csv_name="test.csv",
        split="train",
    )

    assert dataset.enabled_modalities == ("image", "radar", "gps", "lidar")
    assert len(dataset) == 1
    assert dataset.samples.gps_paths is not None
    assert dataset.samples.lidar_paths is not None
    metadata_sample = dataset._with_metadata(0, {})
    assert metadata_sample["sample_id"] == "shared-sample"
    assert metadata_sample["metadata"]["stable_sample_id"] == "mmw:sunny:Town03_5wayroad_seed28:train:shared-sample"
    assert sample_ids_from_batch({"metadata": {"stable_sample_id": [metadata_sample["metadata"]["stable_sample_id"]]}}) == [
        "mmw:sunny:Town03_5wayroad_seed28:train:shared-sample"
    ]


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("future_beam_label1", 64, "[0, 63]"),
        ("camera1", "../outside.png", "invalid camera1 path"),
    ],
)
def test_mmw_dataset_rejects_invalid_label_and_root_escaping_resource(
    tmp_path: Path,
    column: str,
    value: object,
    message: str,
):
    fields = [
        f"{prefix}{index}"
        for prefix, count in (("camera", 1), ("radar", 1), ("gps", 1), ("bs_gps", 1), ("lidar", 1), ("future_beam_label", 1))
        for index in range(1, count + 1)
    ]
    row = {
        "camera1": "camera.png",
        "radar1": "radar_RA.npy",
        "gps1": "gps.yaml",
        "bs_gps1": "bs_gps.yaml",
        "lidar1": "lidar.npy",
        "future_beam_label1": 0,
    }
    row[column] = value
    for path in ("camera.png", "radar_RA.npy", "radar_DA.npy", "gps.yaml", "bs_gps.yaml", "lidar.npy"):
        (tmp_path / path).touch()
    (tmp_path / "outside.png").touch()
    for name in ("train.csv", "test.csv"):
        with (tmp_path / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)

    with pytest.raises(ValueError, match=message):
        MMWDataset(
            condition="sunny",
            scene="Town03_5wayroad_seed28",
            data_root=str(tmp_path),
            train_csv_name="train.csv",
            test_csv_name="test.csv",
            split="train",
            seq_len=1,
        )


def test_mmw_dataset_rejects_missing_derived_radar_da_map(tmp_path: Path):
    fields = ["camera1", "radar1", "gps1", "bs_gps1", "lidar1", "future_beam_label1"]
    row = {
        "camera1": "camera.png",
        "radar1": "radar_RA.npy",
        "gps1": "gps.yaml",
        "bs_gps1": "bs_gps.yaml",
        "lidar1": "lidar.npy",
        "future_beam_label1": 0,
    }
    for path in ("camera.png", "radar_RA.npy", "gps.yaml", "bs_gps.yaml", "lidar.npy"):
        (tmp_path / path).touch()
    for name in ("train.csv", "test.csv"):
        with (tmp_path / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)

    with pytest.raises(FileNotFoundError, match=r"radar1 \(_DA\)"):
        MMWDataset(
            condition="sunny",
            scene="Town03_5wayroad_seed28",
            data_root=str(tmp_path),
            train_csv_name="train.csv",
            test_csv_name="test.csv",
            split="train",
            seq_len=1,
        )


def test_mmw_dataset_rejects_non_contiguous_columns_before_loading_resources(tmp_path: Path):
    fields = ["camera1", "camera3", "radar1", "gps1", "bs_gps1", "lidar1", "future_beam_label1"]
    row = {field: "missing" for field in fields}
    row["future_beam_label1"] = 0
    for name in ("train.csv", "test.csv"):
        with (tmp_path / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)

    with pytest.raises(ValueError, match="non-contiguous camera"):
        MMWDataset(
            condition="sunny",
            scene="Town03_5wayroad_seed28",
            data_root=str(tmp_path),
            train_csv_name="train.csv",
            test_csv_name="test.csv",
            split="train",
            seq_len=1,
        )


def test_mmw_dataset_requires_materialized_four_sensor_columns(tmp_path: Path):
    fields = [f"{prefix}{index}" for prefix, count in (("camera", 5), ("gps", 5), ("lidar", 5), ("future_beam_label", 1)) for index in range(1, count + 1)]
    row = {field: f"assets/{field}" for field in fields}
    row["future_beam_label1"] = 63
    for name in ("train.csv", "test.csv"):
        with (tmp_path / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)

    with pytest.raises(ValueError, match="radar1"):
        MMWDataset(
            condition="sunny",
            scene="Town03_5wayroad_seed28",
            data_root=str(tmp_path),
            train_csv_name="train.csv",
            test_csv_name="test.csv",
            split="train",
        )


def test_mmw_dataset_loads_prepared_four_sensor_sample(tmp_path: Path):
    fields = [f"{prefix}{index}" for prefix, count in (("camera", 5), ("radar", 5), ("gps", 5), ("bs_gps", 5), ("lidar", 5), ("future_beam_label", 1)) for index in range(1, count + 1)]
    for index in range(1, 6):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / f"camera{index}.png")
        np.save(tmp_path / f"radar{index}_RA.npy", np.zeros((128, 64), dtype=np.float32))
        np.save(tmp_path / f"radar{index}_DA.npy", np.zeros((128, 64), dtype=np.float32))
        (tmp_path / f"gps{index}.yaml").write_text("sensors:\n  GPS:\n    location: {x: 2.0, y: 1.0}\n", encoding="utf-8")
        (tmp_path / f"bs_gps{index}.yaml").write_text("sensors:\n  GPS:\n    location: {x: 0.0, y: 0.0}\n", encoding="utf-8")
    np.save(tmp_path / "lidar.npy", np.zeros((3, 224, 224), dtype=np.float32))
    row = {
        **{f"camera{index}": f"camera{index}.png" for index in range(1, 6)},
        **{f"radar{index}": f"radar{index}_RA.npy" for index in range(1, 6)},
        **{f"gps{index}": f"gps{index}.yaml" for index in range(1, 6)},
        **{f"bs_gps{index}": f"bs_gps{index}.yaml" for index in range(1, 6)},
        **{f"lidar{index}": "lidar.npy" for index in range(1, 6)},
        "future_beam_label1": 63,
    }
    for name in ("train.csv", "test.csv"):
        with (tmp_path / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)

    sample = MMWDataset(
        condition="sunny",
        scene="Town03_5wayroad_seed28",
        data_root=str(tmp_path),
        train_csv_name="train.csv",
        test_csv_name="test.csv",
        split="train",
    )[0]

    assert sample["image"].shape == (5, 3, 224, 224)
    assert sample["radar_ra"].shape == sample["radar_da"].shape == (5, 128, 64)
    assert sample["gps"].shape == (5, 3)
    assert sample["lidar"].shape == (5, 3, 224, 224)
    assert sample["target_beam"].tolist() == [63]
    assert "input_beam" not in sample


def test_mmw_dataset_requires_explicit_domain(tmp_path: Path):
    with pytest.raises(ValueError, match="condition and scene"):
        MMWDataset(data_root=str(tmp_path), train_csv_name="train.csv", test_csv_name="test.csv", split="train")


def test_split_identity_audit_fails_closed_on_shared_resource_or_missing_identity() -> None:
    train = [{"sample_id": "train", "target_sample_id": "target-train", "contiguous_segment_id": "segment", "camera1": "a.png"}]
    validation = [{"sample_id": "validation", "target_sample_id": "target-validation", "contiguous_segment_id": "segment", "camera1": "a.png"}]

    overlap = compute_split_identity_audit(train, validation, train_group_ids=["train-group"], test_group_ids=["validation-group"])
    assert overlap["status"] == "failed"
    assert overlap["resource_reference_overlap_count"] == 1

    missing = compute_split_identity_audit(
        [{"sample_id": "train", "contiguous_segment_id": "segment", "camera1": "a.png"}],
        validation,
    )
    assert missing["status"] == "failed"
    assert missing["missing_fields"]["train"] == ["target_sample_id"]


def test_manifest_split_builder_writes_t2_sequence_inputs(tmp_path: Path):
    scene = "Town03_5wayroad_seed28"
    manifest = tmp_path / "Prepared" / scene / "manifests" / "frame_manifest.csv"
    manifest.parent.mkdir(parents=True)
    fields = ["condition", "town", "sensor_scenario", "channel_scenario", "agent", "frame_id", "sample_id", "camera0", "lidar", "gps", "beam_power_path", "beam_label"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                "condition": "sunny",
                "town": "Town03",
                "sensor_scenario": scene,
                "channel_scenario": scene,
                "agent": "cav_1",
                "frame_id": f"{index:06d}",
                "sample_id": f"sample-{index}",
                "camera0": f"camera/{index}.png",
                "lidar": f"lidar/{index}.npy",
                "gps": f"gps/{index}.yaml",
                "beam_power_path": f"Prepared/{scene}/beam_power/{index}.txt",
                "beam_label": index % 64,
            }
            for index in range(24)
        )

    result = build_sequence_splits_from_manifest(
        data_root=tmp_path,
        scene=scene,
        seq_len=3,
        pred_len=1,
        train_ratio=0.5,
        block_size_frames=8,
    )

    assert result["windows"] > 0
    assert result["train_rows"] > 0
    assert result["test_rows"] > 0
    assert result["strict_validation_eligible"]
    assert Path(result["outputs"]["train_csv"]).exists()
