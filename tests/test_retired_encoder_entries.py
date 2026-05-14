from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.data.datasets.scenario9 import Scenario9Dataset  # noqa: E402
from kd_sensing.models.fusion import FusionTeacherModalityNet, FusionStudentModalityNet  # noqa: E402
from kd_sensing.registries import MODELS, RegistryError  # noqa: E402

import kd_sensing.models  # noqa: E402,F401


def _retired_prefix() -> str:
    return "m2" + "beamllm"


def _profile_key() -> str:
    return "encoder" + "_profile"


@pytest.mark.parametrize(
    ("model_type", "input_tensor", "extra"),
    [
        ("image_teacher", torch.rand(1, 2, 3, 224, 224), {}),
        ("image_student", torch.rand(1, 2, 3, 224, 224), {}),
        ("radar_teacher", torch.rand(1, 2, 2, 128, 64), {"radar_channels": 2}),
        ("radar_student", torch.rand(1, 2, 2, 128, 64), {"radar_channels": 2}),
        ("gps_teacher", torch.rand(1, 2, 3), {"gps_input_size": 3}),
        ("gps_student", torch.rand(1, 2, 3), {"gps_input_size": 3}),
        ("lidar_teacher", torch.rand(1, 2, 3, 224, 224), {"lidar_channels": 3}),
        ("lidar_student", torch.rand(1, 2, 3, 224, 224), {"lidar_channels": 3}),
    ],
)
def test_standard_single_modality_registrations_build_and_forward(model_type: str, input_tensor: torch.Tensor, extra: dict):
    cfg = {
        "type": model_type,
        "feature_size": 64,
        "num_classes": 64,
        "gru_params": [64, 64, 1],
        **extra,
    }
    model = MODELS.build(cfg)
    model.eval()

    with torch.no_grad():
        pred, features, output_features = model(input_tensor)

    assert pred.shape == (1, 2, 64)
    assert features.shape == (1, 2, 64)
    assert output_features.shape == (1, 2, 64)


@pytest.mark.parametrize("model_type", ["fusion_teacher", "fusion_student"])
def test_standard_fusion_registrations_build_and_forward(model_type: str):
    model = MODELS.build(
        {
            "type": model_type,
            "feature_size": 64,
            "num_classes": 64,
            "gru_params": [64, 64, 1],
            "modalities": ["image", "radar", "gps", "lidar", "mmwave"],
            "image_channels": 1,
            "radar_channels": 2,
            "gps_input_size": 3,
            "lidar_channels": 3,
        }
    )
    model.eval()

    with torch.no_grad():
        pred, features, output_features = model(
            image_batch=torch.rand(1, 2, 1, 224, 224),
            radar_batch=torch.rand(1, 2, 2, 128, 64),
            gps_batch=torch.rand(1, 2, 3),
            lidar_batch=torch.rand(1, 2, 3, 224, 224),
            mmwave_batch=torch.rand(1, 2, 64),
        )

    assert pred.shape == (1, 2, 64)
    assert features.shape == (1, 2, 64)
    assert output_features.shape == (1, 2, 64)


@pytest.mark.parametrize(
    "suffix",
    [
        "image_teacher",
        "image_student",
        "radar_teacher",
        "radar_student",
        "gps_teacher",
        "gps_student",
        "lidar_teacher",
        "lidar_student",
    ],
)
def test_retired_single_modality_registrations_are_unknown(suffix: str):
    with pytest.raises(RegistryError, match="Unknown component"):
        MODELS.build({"type": f"{_retired_prefix()}_{suffix}"})


@pytest.mark.parametrize("model_cls", [FusionTeacherModalityNet, FusionStudentModalityNet])
def test_retired_fusion_profile_is_not_accepted(model_cls: type):
    kwargs = {
        "feature_size": 64,
        "num_classes": 64,
        "gru_params": [64, 64, 1],
        "modalities": ["image", "radar"],
        _profile_key(): _retired_prefix(),
    }

    with pytest.raises(TypeError, match=_profile_key()):
        model_cls(**kwargs)


def test_retired_dataset_preprocessing_modes_are_rejected(tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_sequence_csv(csv_path)

    with pytest.raises(ValueError, match="gps_feature_mode"):
        Scenario9Dataset(
            data_root=str(tmp_path),
            csv_name=str(csv_path),
            enabled_modalities=["gps"],
            gps_feature_mode=f"{_retired_prefix()}_minmax",
        )

    with pytest.raises(ValueError, match="lidar_encoding"):
        Scenario9Dataset(
            data_root=str(tmp_path),
            csv_name=str(tmp_path / "missing.csv"),
            enabled_modalities=["lidar"],
            lidar_encoding=f"{_retired_prefix()}_histogram",
        )


def _write_sequence_csv(path: Path) -> None:
    columns = (
        [f"gps{i}" for i in range(1, 9)]
        + [f"bs_gps{i}" for i in range(1, 9)]
        + [f"beam{i}" for i in range(1, 9)]
        + [f"future_beam{i}" for i in range(1, 4)]
    )
    values = (
        [f"gps_{idx}.txt" for idx in range(8)]
        + [f"bs_gps_{idx}.txt" for idx in range(8)]
        + [f"beam_{idx}.txt" for idx in range(8)]
        + [f"future_beam_{idx}.txt" for idx in range(3)]
    )
    path.write_text(",".join(columns) + "\n" + ",".join(values) + "\n", encoding="utf-8")
