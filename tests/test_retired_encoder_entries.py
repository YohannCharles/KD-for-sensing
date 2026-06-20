import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
from kd_sensing.data.datasets.deepsense6g import DeepSense6GDataset  # noqa: E402
from kd_sensing.models.fusion import FusionStrongModalityNet, FusionLightweightModalityNet  # noqa: E402
from kd_sensing.models.gps import GpsLightweightModalityNet, GpsStrongModalityNet  # noqa: E402
from kd_sensing.models.image import ImageLightweightModalityNet, ImageStrongModalityNet  # noqa: E402
from kd_sensing.models.lidar import LidarLightweightModalityNet, LidarStrongModalityNet  # noqa: E402
from kd_sensing.models.radar import RadarLightweightModalityNet, RadarStrongModalityNet  # noqa: E402
from kd_sensing.registries import MODELS, RegistryError  # noqa: E402

import kd_sensing.models  # noqa: E402,F401


def _retired_prefix() -> str:
    return "m2" + "beamllm"


def _profile_key() -> str:
    return "encoder" + "_profile"


@pytest.mark.parametrize(
    ("model_cls", "input_tensor", "extra"),
    [
        (ImageStrongModalityNet, torch.rand(1, 2, 3, 224, 224), {"image_channels": 3}),
        (ImageLightweightModalityNet, torch.rand(1, 2, 3, 224, 224), {"image_channels": 3}),
        (RadarStrongModalityNet, torch.rand(1, 2, 2, 128, 64), {"radar_channels": 2}),
        (RadarLightweightModalityNet, torch.rand(1, 2, 2, 128, 64), {"radar_channels": 2}),
        (GpsStrongModalityNet, torch.rand(1, 2, 3), {"gps_input_size": 3}),
        (GpsLightweightModalityNet, torch.rand(1, 2, 3), {"gps_input_size": 3}),
        (LidarStrongModalityNet, torch.rand(1, 2, 3, 224, 224), {"lidar_channels": 3}),
        (LidarLightweightModalityNet, torch.rand(1, 2, 3, 224, 224), {"lidar_channels": 3}),
    ],
)
def test_legacy_single_modality_classes_still_forward_when_directly_instantiated(
    model_cls: type,
    input_tensor: torch.Tensor,
    extra: dict,
):
    model = model_cls(feature_size=64, num_classes=64, gru_params=[64, 64, 1], **extra)
    model.eval()

    with torch.no_grad():
        pred, features, output_features = model(input_tensor)

    assert pred.shape == (1, 2, 64)
    assert features.shape == (1, 2, 64)
    assert output_features.shape == (1, 2, 64)


@pytest.mark.parametrize("model_type", ["fusion_strong", "fusion_lightweight"])
def test_legacy_fusion_classes_still_forward_when_directly_instantiated(model_type: str):
    model_cls = FusionStrongModalityNet if model_type == "fusion_strong" else FusionLightweightModalityNet
    model = model_cls(
        feature_size=64,
        num_classes=64,
        gru_params=[64, 64, 1],
        modalities=["image", "radar", "gps", "lidar", "mmwave"],
        image_channels=1,
        radar_channels=2,
        gps_input_size=3,
        lidar_channels=3,
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
        "image_strong",
        "image_lightweight",
        "radar_strong",
        "radar_lightweight",
        "gps_strong",
        "gps_lightweight",
        "lidar_strong",
        "lidar_lightweight",
    ],
)
def test_retired_single_modality_registrations_are_unknown(suffix: str):
    with pytest.raises(RegistryError, match="Unknown component"):
        MODELS.build({"type": f"{_retired_prefix()}_{suffix}"})


@pytest.mark.parametrize("model_cls", [FusionStrongModalityNet, FusionLightweightModalityNet])
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
        DeepSense6GDataset(
            data_root=str(tmp_path),
            csv_name=str(csv_path),
            enabled_modalities=["gps"],
            gps_feature_mode=f"{_retired_prefix()}_minmax",
        )

    with pytest.raises(ValueError, match="lidar_encoding"):
        DeepSense6GDataset(
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
