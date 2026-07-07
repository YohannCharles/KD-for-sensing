from pathlib import Path

import pytest

from kd_sensing.data.datasets.deepsense6g import DeepSense6GDataset
from kd_sensing.registries import MODELS, RegistryError

import kd_sensing.models  # noqa: F401


def _retired_prefix() -> str:
    return "m2" + "beamllm"


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
