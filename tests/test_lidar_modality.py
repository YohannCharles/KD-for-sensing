from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.data.datasets.scenario9 import Scenario9Dataset  # noqa: E402
from kd_sensing.data.transforms import (  # noqa: E402
    LidarBEVNormalizer,
    build_lidar_bev,
    filter_lidar_points,
    lidar_points_to_bev,
    read_lidar_point_cloud,
)
from kd_sensing.engine.batch import forward_model, prepare_fusion_inputs, prepare_lidar_inputs  # noqa: E402
from kd_sensing.models.fusion import FusionModalityNet, StudentModalityNet  # noqa: E402
from kd_sensing.models.lidar import LidarFeatureExtractor, LidarModalityNet, LidarStudentModalityNet  # noqa: E402
from kd_sensing.registries import MODELS  # noqa: E402

import kd_sensing.models  # noqa: E402,F401


LIDAR_CONFIGS = [
    "configs/lidar/no_kd.yaml",
    "configs/lidar/teacher_no_kd.yaml",
    "configs/lidar/student_no_kd.yaml",
    "configs/lidar/logits_kd.yaml",
    "configs/lidar/rkd.yaml",
]

LIDAR_FUSION_CONFIGS = [
    "configs/fusion/radar_lidar_no_kd.yaml",
    "configs/fusion/all_modalities_lidar_no_kd.yaml",
    *[
        f"configs/fusion/{slug}_{mode}.yaml"
        for slug in [
            "image_lidar",
            "radar_lidar",
            "gps_lidar",
            "image_radar_lidar",
            "image_gps_lidar",
            "radar_gps_lidar",
            "image_radar_gps_lidar",
        ]
        for mode in ["teacher_no_kd", "student_no_kd", "logits_kd", "rkd"]
    ],
]


def test_lidar_point_cloud_reader_filters_invalid_and_builds_bev(tmp_path: Path):
    point_file = tmp_path / "cloud.txt"
    point_file.write_text(
        "\n".join(
            [
                "1.0 0.0 0.2 0.5",
                "1.2 0.1 0.3 0.8",
                "nan 0.0 0.0 1.0",
                "8.0 8.0 0.0 1.0",
            ]
        ),
        encoding="utf-8",
    )

    points = read_lidar_point_cloud(tmp_path, "cloud.txt")
    filtered = filter_lidar_points(points, roi=[0.0, 2.0, -1.0, 1.0, -1.0, 1.0])
    bev = lidar_points_to_bev(filtered, bev_size=[16, 16], roi=[0.0, 2.0, -1.0, 1.0, -1.0, 1.0])
    empty = build_lidar_bev(
        tmp_path,
        "cloud.txt",
        bev_size=[16, 16],
        roi=[20.0, 30.0, 20.0, 30.0, -1.0, 1.0],
    )

    assert points.shape[1] == 4
    assert np.isfinite(points).all()
    assert filtered.shape[0] == 2
    assert bev.shape == (3, 16, 16)
    assert bev.dtype == np.float32
    assert float(bev.sum()) > 0.0
    assert empty.shape == (3, 16, 16)
    assert float(empty.sum()) == 0.0


def test_lidar_dataset_returns_lidar_tensor_and_keeps_old_behavior(tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_dataset_fixture(tmp_path, csv_path)

    lidar_dataset = Scenario9Dataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=3,
        num_pred=1,
        fft_tuple=[4, 8, 6],
        clipped_range=4,
        use_lidar=True,
        lidar_bev_size=[16, 16],
        lidar_roi=[0.0, 2.0, -1.0, 1.0, -1.0, 1.0],
        lidar_normalize=False,
    )
    old_dataset = Scenario9Dataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=3,
        num_pred=1,
        fft_tuple=[4, 8, 6],
        clipped_range=4,
        use_lidar=False,
    )

    sample = lidar_dataset[0]
    old_sample = old_dataset[0]

    assert sample["lidar"].shape == (3, 3, 16, 16)
    assert sample["lidar"].dtype == torch.float32
    assert "lidar" not in old_sample


def test_lidar_normalizer_fits_train_and_reuses_for_test(tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_dataset_fixture(tmp_path, csv_path)

    train_dataset = Scenario9Dataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=3,
        num_pred=1,
        fft_tuple=[4, 8, 6],
        clipped_range=4,
        use_lidar=True,
        lidar_bev_size=[16, 16],
    )
    test_dataset = Scenario9Dataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="test",
        seq_len=3,
        num_pred=1,
        fft_tuple=[4, 8, 6],
        clipped_range=4,
        use_lidar=True,
        lidar_bev_size=[16, 16],
        lidar_normalizer=train_dataset.lidar_normalizer,
    )

    assert isinstance(train_dataset.lidar_normalizer, LidarBEVNormalizer)
    assert test_dataset.lidar_normalizer is train_dataset.lidar_normalizer
    with pytest.raises(ValueError, match="requires a train-fitted lidar_normalizer"):
        Scenario9Dataset(
            data_root=str(tmp_path),
            csv_name=str(csv_path),
            split="test",
            seq_len=3,
            num_pred=1,
            fft_tuple=[4, 8, 6],
            clipped_range=4,
            use_lidar=True,
            lidar_bev_size=[16, 16],
        )


def test_lidar_models_forward_contracts_and_param_validation():
    extractor = MODELS.build({"type": "lidar_feature_extractor", "n_feature": 64, "in_channels": 3})
    assert isinstance(extractor, LidarFeatureExtractor)
    with torch.no_grad():
        features = extractor(torch.randn(2, 10, 3, 32, 32))
    assert features.shape == (2, 10, 64)

    for model_type, expected_cls in [
        ("lidar_teacher", LidarModalityNet),
        ("lidar_student", LidarStudentModalityNet),
    ]:
        model = MODELS.build(
            {
                "type": model_type,
                "feature_size": 64,
                "num_classes": 64,
                "gru_params": [64, 64, 2],
                "lidar_channels": 3,
            }
        )
        assert isinstance(model, expected_cls)
        model.eval()
        with torch.no_grad():
            pred, input_features, output_features = model(torch.randn(2, 10, 3, 32, 32))
        assert pred.shape == (2, 10, 64)
        assert input_features.shape == (2, 10, 64)
        assert output_features.shape == (2, 10, 64)

    with pytest.raises(ValueError, match="gru_params must contain"):
        MODELS.build(
            {
                "type": "lidar_student",
                "feature_size": 64,
                "num_classes": 64,
                "gru_params": [64, 64],
            }
        )
    with pytest.raises(ValueError, match="must equal feature_size"):
        MODELS.build(
            {
                "type": "lidar_teacher",
                "feature_size": 64,
                "num_classes": 64,
                "gru_params": [32, 64, 2],
            }
        )


def test_lidar_batch_and_fusion_paths():
    batch = {
        "lidar": torch.randn(2, 8, 3, 16, 16),
        "input_beam": torch.zeros(2, 8, dtype=torch.long),
        "target_beam": torch.zeros(2, 3, dtype=torch.long),
    }
    lidar_input = prepare_lidar_inputs(batch, seq_length=8, num_pred=3, device=torch.device("cpu"))
    fusion_inputs = prepare_fusion_inputs(
        batch,
        seq_length=8,
        num_pred=3,
        device=torch.device("cpu"),
        modalities=["lidar"],
    )
    model = LidarStudentModalityNet(feature_size=64, num_classes=64, gru_params=[64, 64, 2])
    fusion_model = StudentModalityNet(
        feature_size=64,
        num_classes=64,
        gru_params=[64, 64, 2],
        modalities=["lidar"],
    )
    teacher = FusionModalityNet(
        feature_size=64,
        num_classes=64,
        gru_params=[64, 64, 2],
        modalities=["lidar"],
    )
    with torch.no_grad():
        pred, _, _ = forward_model(model, "lidar", lidar_batch=lidar_input)
        fusion_pred, _, _ = fusion_model(**fusion_inputs)

    assert lidar_input.shape == (2, 10, 3, 16, 16)
    assert sorted(fusion_inputs) == ["lidar_batch"]
    assert pred.shape == (2, 10, 64)
    assert fusion_pred.shape == (2, 10, 64)
    with pytest.raises(ValueError, match="requires 'lidar' input"):
        teacher()


@pytest.mark.parametrize("config_path", LIDAR_CONFIGS)
def test_lidar_configs_build(config_path: str):
    cfg = load_config(ROOT / config_path)
    model = MODELS.build(cfg["model"]["student"])

    assert cfg["experiment"]["task"] == "lidar"
    assert cfg["data"]["dataset"]["use_lidar"] is True
    assert cfg["model"]["teacher"]["gru_params"] == [64, 64, 2]
    assert cfg["model"]["student"]["gru_params"] == [64, 64, 2]
    assert isinstance(model, (LidarModalityNet, LidarStudentModalityNet))


@pytest.mark.parametrize("config_path", LIDAR_FUSION_CONFIGS)
def test_lidar_fusion_configs_build(config_path: str):
    cfg = load_config(ROOT / config_path)
    teacher = MODELS.build(cfg["model"]["teacher"])
    student = MODELS.build(cfg["model"]["student"])

    assert cfg["experiment"]["task"] == "fusion"
    assert "lidar" in cfg["model"]["teacher"]["modalities"]
    assert "lidar" in cfg["model"]["student"]["modalities"]
    assert cfg["model"]["teacher"]["modalities"] == cfg["model"]["student"]["modalities"]
    assert cfg["data"]["dataset"]["use_lidar"] is True
    assert cfg["data"]["dataset"]["lidar_bev_size"] == [224, 224]
    assert cfg["data"]["dataset"]["lidar_roi"] == [-30.0, 30.0, -30.0, 30.0, -3.0, 5.0]
    assert cfg["data"]["dataset"]["lidar_normalize"] is True
    assert cfg["model"]["teacher"]["lidar_channels"] == 3
    assert cfg["model"]["student"]["lidar_channels"] == 3
    assert isinstance(teacher, FusionModalityNet)
    assert isinstance(student, (FusionModalityNet, StudentModalityNet))


def _write_dataset_fixture(root: Path, csv_path: Path) -> None:
    for idx in range(3):
        Image.fromarray(np.full((8, 8, 3), idx * 30, dtype=np.uint8)).save(root / f"camera_{idx}.jpg")
        np.save(root / f"radar_{idx}_RA.npy", np.ones((4, 4), dtype=np.float32) * idx)
        np.save(root / f"radar_{idx}_DA.npy", np.ones((6, 4), dtype=np.float32) * idx)
        beam = np.zeros(64, dtype=np.float32)
        beam[idx] = 1.0
        np.savetxt(root / f"beam_{idx}.txt", beam)
        lidar = np.array(
            [
                [1.0 + idx * 0.1, 0.0, 0.2, 0.5],
                [1.2 + idx * 0.1, 0.2, 0.3, 0.8],
            ],
            dtype=np.float32,
        )
        np.savetxt(root / f"lidar_{idx}.txt", lidar)
    future = np.zeros(64, dtype=np.float32)
    future[4] = 1.0
    np.savetxt(root / "future_0.txt", future)

    columns = (
        [f"camera{i}" for i in range(1, 4)]
        + [f"radar{i}" for i in range(1, 4)]
        + [f"lidar{i}" for i in range(1, 4)]
        + [f"beam{i}" for i in range(1, 4)]
        + ["future_beam1", "seq_index"]
    )
    values = (
        [f"camera_{idx}.jpg" for idx in range(3)]
        + [f"radar_{idx}_RA.npy" for idx in range(3)]
        + [f"lidar_{idx}.txt" for idx in range(3)]
        + [f"beam_{idx}.txt" for idx in range(3)]
        + ["future_0.txt", "1"]
    )
    csv_path.write_text(",".join(columns) + "\n" + ",".join(values) + "\n", encoding="utf-8")
