from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.data.datasets.scenario9 import Scenario9Dataset  # noqa: E402
from kd_sensing.data.transforms import GPSMinMaxScaler, lidar_points_to_m2beamllm_histogram  # noqa: E402
from kd_sensing.models.fusion import FusionModalityNet, StudentModalityNet  # noqa: E402
from kd_sensing.models.m2beamllm_encoders import (  # noqa: E402
    M2BeamLLMGpsEncoder,
    M2BeamLLMImageEncoder,
    M2BeamLLMLidarEncoder,
    M2BeamLLMRadarEncoder,
)
from kd_sensing.models.mmwave import MmWaveFeatureExtractor  # noqa: E402
from kd_sensing.registries import MODELS  # noqa: E402

import kd_sensing.models  # noqa: E402,F401


def test_m2beamllm_encoder_shapes_and_raw_fft_error():
    encoders_and_inputs = [
        (M2BeamLLMImageEncoder(64, image_channels=1, pretrained=False), torch.rand(1, 2, 1, 112, 112)),
        (M2BeamLLMRadarEncoder(64, radar_channels=2), torch.rand(1, 2, 2, 128, 64)),
        (M2BeamLLMLidarEncoder(64, lidar_channels=1, pretrained=False), torch.rand(1, 2, 1, 256, 256)),
        (M2BeamLLMGpsEncoder(64, gps_input_size=2), torch.rand(1, 2, 2)),
    ]

    for encoder, tensor in encoders_and_inputs:
        encoder.eval()
        with torch.no_grad():
            features = encoder(tensor)
        assert features.shape == (1, 2, 64)

    raw_encoder = M2BeamLLMRadarEncoder(64, radar_channels=2, radar_input_mode="raw_fft")
    with pytest.raises(ValueError, match="raw radar input"):
        raw_encoder(torch.rand(1, 2, 2, 128, 64))


@pytest.mark.parametrize(
    ("model_type", "input_tensor"),
    [
        ("m2beamllm_image_teacher", torch.rand(1, 2, 1, 112, 112)),
        ("m2beamllm_image_student", torch.rand(1, 2, 1, 112, 112)),
        ("m2beamllm_radar_teacher", torch.rand(1, 2, 2, 128, 64)),
        ("m2beamllm_radar_student", torch.rand(1, 2, 2, 128, 64)),
        ("m2beamllm_gps_teacher", torch.rand(1, 2, 2)),
        ("m2beamllm_gps_student", torch.rand(1, 2, 2)),
        ("m2beamllm_lidar_teacher", torch.rand(1, 2, 1, 256, 256)),
        ("m2beamllm_lidar_student", torch.rand(1, 2, 1, 256, 256)),
    ],
)
def test_m2beamllm_single_modality_forward_contracts(model_type: str, input_tensor: torch.Tensor):
    cfg = {
        "type": model_type,
        "feature_size": 64,
        "num_classes": 64,
        "gru_params": [64, 64, 1],
    }
    if "image" in model_type:
        cfg.update({"image_channels": 1, "m2beamllm_pretrained": False})
    if "radar" in model_type:
        cfg.update({"radar_channels": 2})
    if "gps" in model_type:
        cfg.update({"gps_input_size": 2})
    if "lidar" in model_type:
        cfg.update({"lidar_channels": 1, "m2beamllm_pretrained": False})
    model = MODELS.build(cfg)
    model.eval()

    with torch.no_grad():
        pred, features, output_features = model(input_tensor)

    assert pred.shape == (1, 2, 64)
    assert features.shape == (1, 2, 64)
    assert output_features.shape == (1, 2, 64)
    assert model.GRU.input_size == 64


def test_m2beamllm_fusion_profile_keeps_mmwave_extractor():
    model = StudentModalityNet(
        feature_size=64,
        num_classes=64,
        gru_params=[64, 64, 1],
        modalities=["image", "radar", "gps", "lidar", "mmwave"],
        encoder_profile="m2beamllm",
        image_channels=1,
        radar_channels=2,
        gps_input_size=2,
        lidar_channels=1,
    )
    model.eval()

    assert isinstance(model.mmwave_feature_extractor, MmWaveFeatureExtractor)
    assert isinstance(model.image_feature_extractor, M2BeamLLMImageEncoder)

    with torch.no_grad():
        pred, features, output_features = model(
            image_batch=torch.rand(1, 2, 1, 112, 112),
            radar_batch=torch.rand(1, 2, 2, 128, 64),
            gps_batch=torch.rand(1, 2, 2),
            lidar_batch=torch.rand(1, 2, 1, 256, 256),
            mmwave_batch=torch.rand(1, 2, 64),
        )

    assert pred.shape == (1, 2, 64)
    assert features.shape == (1, 2, 64)
    assert output_features.shape == (1, 2, 64)


def test_m2beamllm_gps_minmax_scaler_fits_train_and_reuses_for_test(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    train_gps, train_bs = _write_gps_files(tmp_path, "train", 33.0, -111.0)
    test_gps, test_bs = _write_gps_files(tmp_path, "test", 34.0, -112.0)
    _write_beam_files(tmp_path)
    _write_sequence_csv(train_csv, train_gps, train_bs, seq_index=1)
    _write_sequence_csv(test_csv, test_gps, test_bs, seq_index=2)

    train_dataset = Scenario9Dataset(
        data_root=str(tmp_path),
        csv_name=str(train_csv),
        split="train",
        enabled_modalities=["gps"],
        use_gps=True,
        gps_feature_mode="m2beamllm_minmax",
    )
    test_dataset = Scenario9Dataset(
        data_root=str(tmp_path),
        csv_name=str(test_csv),
        split="test",
        enabled_modalities=["gps"],
        use_gps=True,
        gps_feature_mode="m2beamllm_minmax",
        gps_scaler=train_dataset.gps_scaler,
    )

    assert isinstance(train_dataset.gps_scaler, GPSMinMaxScaler)
    assert test_dataset.gps_scaler is train_dataset.gps_scaler
    assert train_dataset[0]["gps"].shape == (8, 2)
    with pytest.raises(ValueError, match="requires a train-fitted gps_scaler"):
        Scenario9Dataset(
            data_root=str(tmp_path),
            csv_name=str(test_csv),
            split="test",
            enabled_modalities=["gps"],
            use_gps=True,
            gps_feature_mode="m2beamllm_minmax",
        )


def test_lidar_m2beamllm_histogram_clips_and_normalizes_counts():
    points = np.array(
        [[0.5, 0.5, 0.0, 1.0] for _ in range(7)] + [[1.5, 0.5, 0.0, 1.0]],
        dtype=np.float32,
    )
    histogram = lidar_points_to_m2beamllm_histogram(
        points,
        histogram_size=[4, 4],
        roi=[0.0, 2.0, 0.0, 2.0, -1.0, 1.0],
    )

    assert histogram.shape == (1, 4, 4)
    assert histogram.max() == 1.0
    assert 0.0 < histogram.sum() <= 2.0


def test_m2beamllm_example_configs_load_and_default_configs_stay_old():
    for config_path in [
        "configs/m2beamllm/image_no_kd.yaml",
        "configs/m2beamllm/radar_no_kd.yaml",
        "configs/m2beamllm/gps_no_kd.yaml",
        "configs/m2beamllm/lidar_no_kd.yaml",
        "configs/m2beamllm/fusion_image_radar_gps_lidar_no_kd.yaml",
        "configs/m2beamllm/fusion_all_modalities_no_kd.yaml",
    ]:
        cfg = load_config(ROOT / config_path)
        teacher = MODELS.build(cfg["model"]["teacher"])
        student = MODELS.build(cfg["model"]["student"])
        assert teacher.GRU.input_size == 64
        assert student.GRU.input_size == 64

    old_image = MODELS.build(load_config(ROOT / "configs/image/student_no_kd.yaml")["model"]["student"])
    old_fusion = MODELS.build(load_config(ROOT / "configs/fusion/image_radar_student_no_kd.yaml")["model"]["student"])
    assert old_image.__class__.__name__ == "ImageStudentModalityNet"
    assert isinstance(old_fusion, StudentModalityNet)
    assert getattr(old_fusion, "encoder_profile", None) is None


def test_fusion_teacher_accepts_m2beamllm_profile():
    teacher = FusionModalityNet(
        feature_size=64,
        num_classes=64,
        gru_params=[64, 64, 1],
        modalities=["image", "radar", "gps", "lidar"],
        encoder_profile="m2beamllm",
        image_channels=1,
        radar_channels=2,
        gps_input_size=2,
        lidar_channels=1,
    )
    assert isinstance(teacher.image_feature_extractor, M2BeamLLMImageEncoder)


def _write_gps_files(root: Path, prefix: str, lat: float, lon: float) -> tuple[list[str], list[str]]:
    gps_dir = root / "gps"
    gps_dir.mkdir(exist_ok=True)
    gps_paths = []
    bs_paths = []
    for idx in range(8):
        gps_name = f"gps/{prefix}_ue_{idx}.txt"
        bs_name = f"gps/{prefix}_bs_{idx}.txt"
        (root / gps_name).write_text(f"{lat + idx * 0.001}\n{lon + idx * 0.001}\n", encoding="utf-8")
        (root / bs_name).write_text(f"{lat}\n{lon}\n", encoding="utf-8")
        gps_paths.append(gps_name)
        bs_paths.append(bs_name)
    return gps_paths, bs_paths


def _write_sequence_csv(path: Path, gps_paths: list[str], bs_paths: list[str], seq_index: int) -> None:
    columns = (
        [f"camera{i}" for i in range(1, 9)]
        + [f"radar{i}" for i in range(1, 9)]
        + [f"gps{i}" for i in range(1, 9)]
        + [f"bs_gps{i}" for i in range(1, 9)]
        + [f"beam{i}" for i in range(1, 9)]
        + [f"future_beam{i}" for i in range(1, 4)]
        + ["seq_index"]
    )
    values = (
        [f"camera_{idx}.jpg" for idx in range(8)]
        + [f"radar_{idx}.npy" for idx in range(8)]
        + gps_paths
        + bs_paths
        + [f"beam_{idx}.txt" for idx in range(8)]
        + [f"future_beam_{idx}.txt" for idx in range(3)]
        + [str(seq_index)]
    )
    path.write_text(",".join(columns) + "\n" + ",".join(values) + "\n", encoding="utf-8")


def _write_beam_files(root: Path) -> None:
    for idx in range(8):
        beam = np.zeros(64, dtype=np.float32)
        beam[idx] = 1.0
        np.savetxt(root / f"beam_{idx}.txt", beam)
    for idx in range(3):
        future = np.zeros(64, dtype=np.float32)
        future[idx + 10] = 1.0
        np.savetxt(root / f"future_beam_{idx}.txt", future)
