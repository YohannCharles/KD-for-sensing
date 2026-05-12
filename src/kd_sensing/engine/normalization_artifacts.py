from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from kd_sensing.data.transform_ops.normalization import LidarBEVNormalizer, MmWaveStandardScaler, load_gps_scaler


def save_normalization_artifacts(dataloaders: dict[str, DataLoader], run_dir: str | Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    train_dataset = dataloaders.get("train").dataset if dataloaders.get("train") is not None else None
    if train_dataset is None:
        return artifacts
    artifact_dir = Path(run_dir) / "artifacts"

    gps_scaler = getattr(train_dataset, "gps_scaler", None)
    if gps_scaler is not None:
        gps_path = artifact_dir / "gps_scaler.npz"
        gps_scaler.save(gps_path)
        artifacts["gps_scaler"] = str(gps_path)

    lidar_normalizer = getattr(train_dataset, "lidar_normalizer", None)
    if lidar_normalizer is not None and getattr(train_dataset, "lidar_normalize", False):
        lidar_path = artifact_dir / "lidar_normalizer.npz"
        lidar_normalizer.save(lidar_path)
        artifacts["lidar_normalizer"] = str(lidar_path)

    mmwave_scaler = getattr(train_dataset, "mmwave_scaler", None)
    if mmwave_scaler is not None and getattr(train_dataset, "mmwave_normalize", False):
        mmwave_path = artifact_dir / "mmwave_scaler.npz"
        mmwave_scaler.save(mmwave_path)
        artifacts["mmwave_scaler"] = str(mmwave_path)
    return artifacts


def load_normalization_artifacts(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    artifacts = metadata.get("normalization_artifacts") or {}
    dataset_kwargs: dict[str, Any] = {}
    gps_path = artifacts.get("gps_scaler")
    if gps_path:
        path = Path(gps_path)
        if not path.exists():
            raise FileNotFoundError(f"GPS scaler artifact not found: {path}")
        dataset_kwargs["gps_scaler"] = load_gps_scaler(path)
    lidar_path = artifacts.get("lidar_normalizer")
    if lidar_path:
        path = Path(lidar_path)
        if not path.exists():
            raise FileNotFoundError(f"LiDAR normalizer artifact not found: {path}")
        dataset_kwargs["lidar_normalizer"] = LidarBEVNormalizer.load(path)
    mmwave_path = artifacts.get("mmwave_scaler")
    if mmwave_path:
        path = Path(mmwave_path)
        if not path.exists():
            raise FileNotFoundError(f"mmWave scaler artifact not found: {path}")
        dataset_kwargs["mmwave_scaler"] = MmWaveStandardScaler.load(path)
    return dataset_kwargs


__all__ = ["load_normalization_artifacts", "save_normalization_artifacts"]
