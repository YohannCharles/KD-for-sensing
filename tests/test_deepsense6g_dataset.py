import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from kd_sensing.config import load_config
from kd_sensing.data.datasets.deepsense6g import DeepSense6GDataset
from kd_sensing.engine.batch import prepare_fusion_inputs, prepare_labels
from kd_sensing.engine.data_factory import build_dataset
from kd_sensing.engine.data_factory_scalers import fit_gps_scaler, gps_scaler_kwargs
from kd_sensing.engine.run_metadata import dataset_run_metadata, prediction_setup_metadata


def _write_row(root: Path, stem: str, *, latitude: float, label: int, label_size: int = 64) -> dict[str, str]:
    Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(root / f"{stem}.png")
    np.save(root / f"{stem}_RA.npy", np.zeros((128, 64), dtype=np.float32))
    np.save(root / f"{stem}_DA.npy", np.zeros((128, 64), dtype=np.float32))
    np.savetxt(root / f"{stem}_gps.txt", [latitude, 1.0])
    np.savetxt(root / f"{stem}_bs_gps.txt", [0.0, 0.0])
    power = np.zeros(label_size, dtype=np.float32)
    power[min(label, label_size - 1)] = 1.0
    np.savetxt(root / f"{stem}_future_beam.txt", power)
    return {
        "seq_index": stem,
        "camera1": f"{stem}.png",
        "radar1": f"{stem}_RA.npy",
        "gps1": f"{stem}_gps.txt",
        "bs_gps1": f"{stem}_bs_gps.txt",
        "lidar1": "lidar.npy",
        "future_beam1": f"{stem}_future_beam.txt",
    }


def _write_scene31(root: Path, *, label_size: int = 64) -> Path:
    root.mkdir()
    np.save(root / "lidar.npy", np.zeros((3, 224, 224), dtype=np.float32))
    fields = ["seq_index", "camera1", "radar1", "gps1", "bs_gps1", "lidar1", "future_beam1"]
    train_rows = [
        _write_row(root, "train_a", latitude=1.0, label=37, label_size=label_size),
        _write_row(root, "train_b", latitude=2.0, label=12, label_size=label_size),
    ]
    test_rows = [_write_row(root, "test", latitude=3.0, label=5, label_size=label_size)]
    for name, rows in (("train.csv", train_rows), ("test.csv", test_rows)):
        with (root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return root


def _dataset(root: Path, split: str) -> DeepSense6GDataset:
    return DeepSense6GDataset(
        scene=31,
        data_root=str(root),
        train_csv_name="train.csv",
        test_csv_name="test.csv",
        split=split,
        seq_len=1,
        num_pred=1,
        enabled_modalities=("image", "radar", "gps", "lidar"),
        use_gps=True,
        use_lidar=True,
    )


def test_deepsense6g_scene31_emits_four_sensor_batch_for_current_runtime(tmp_path: Path) -> None:
    dataset = _dataset(_write_scene31(tmp_path / "scenario31"), "train")

    sample = dataset[0]
    assert sample["image"].shape == (1, 3, 224, 224)
    assert sample["radar_ra"].shape == sample["radar_da"].shape == (1, 128, 64)
    assert sample["gps"].shape == (1, 3)
    assert sample["lidar"].shape == (1, 3, 224, 224)
    assert sample["target_beam"].tolist() == [37]
    assert "input_beam" not in sample

    batch = next(iter(DataLoader(dataset, batch_size=1)))
    labels = prepare_labels(batch, num_pred=1, device=torch.device("cpu"))
    fusion = prepare_fusion_inputs(batch, seq_length=1, device=torch.device("cpu"))
    assert labels.tolist() == [[37]]
    assert fusion["image_batch"].shape == (1, 1, 3, 224, 224)
    assert fusion["radar_batch"].shape == (1, 1, 2, 128, 64)
    assert fusion["gps_batch"].shape == (1, 1, 3)
    assert fusion["lidar_batch"].shape == (1, 1, 3, 224, 224)
    metadata = dataset_run_metadata(dataset)
    assert metadata["dataset_family"] == "DeepSense6G"
    assert metadata["scene_id"] == 31


def test_deepsense6g_rejects_unsupported_scene_and_non_64_beam_label(tmp_path: Path) -> None:
    root = _write_scene31(tmp_path / "scenario31", label_size=63)

    with pytest.raises(ValueError):
        DeepSense6GDataset(
            scene=9,
            data_root=str(root),
            train_csv_name="train.csv",
            test_csv_name="test.csv",
            split="train",
            seq_len=1,
            num_pred=1,
        )
    with pytest.raises(ValueError):
        _dataset(root, "train")[0]


def test_deepsense6g_test_split_reuses_train_gps_scaler(tmp_path: Path) -> None:
    root = _write_scene31(tmp_path / "scenario31")
    train = _dataset(root, "train")
    test = _dataset(root, "test")

    fit_gps_scaler(train, test, source="test")

    assert test.gps_scaler is train.gps_scaler
    assert gps_scaler_kwargs(train) == {"gps_scaler": train.gps_scaler}
    expected = train.gps_scaler.transform(test._gps_features_for_index(0))
    np.testing.assert_allclose(test[0]["gps"].numpy(), expected)


def test_deepsense6g_t2_config_builds_dataset_through_factory(tmp_path: Path) -> None:
    root = _write_scene31(tmp_path / "scenario31")
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/deepsense6g/t2.yaml")
    cfg["data"]["dataset"].update(
        data_root=str(root),
        train_csv_name="train.csv",
        test_csv_name="test.csv",
        seq_len=1,
    )

    dataset = build_dataset(cfg, "train")

    assert isinstance(dataset, DeepSense6GDataset)
    assert len(dataset) == 2
    setup = prediction_setup_metadata(cfg)
    assert setup["dataset_type"] == "deepsense6g"
    assert setup["scene"] == 31


def test_deepsense6g_config_rejects_legacy_scene_and_unscaled_gps() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/deepsense6g/t2.yaml"

    with pytest.raises(ValueError, match="scene"):
        load_config(path, overrides=["data.dataset.scene=9"])
    with pytest.raises(ValueError, match="GPS normalization"):
        load_config(path, overrides=["data.dataset.gps_normalize=false"])
