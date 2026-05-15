from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from kd_sensing.data.transform_ops.normalization import (
    LidarBEVNormalizer,
    MmWaveStandardScaler,
    OcclusionTargetStats,
    PositionTargetStandardScaler,
    load_gps_scaler,
)


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

    occlusion_stats = getattr(train_dataset, "occlusion_target_stats", None)
    if occlusion_stats is not None and getattr(train_dataset, "occlusion_target_enabled", False):
        occlusion_path = artifact_dir / "occlusion_target_stats.json"
        occlusion_stats.save(occlusion_path)
        artifacts["occlusion_target_stats"] = str(occlusion_path)

    position_scaler = getattr(train_dataset, "position_target_scaler", None)
    if (
        position_scaler is not None
        and getattr(train_dataset, "position_target_enabled", False)
        and getattr(train_dataset, "position_target_normalize", False)
    ):
        position_path = artifact_dir / "position_target_scaler.npz"
        position_scaler.save(position_path)
        artifacts["position_target_scaler"] = str(position_path)
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
    occlusion_path = artifacts.get("occlusion_target_stats")
    if occlusion_path:
        path = Path(occlusion_path)
        if not path.exists():
            raise FileNotFoundError(f"Occlusion target stats artifact not found: {path}")
        dataset_kwargs["occlusion_target_stats"] = OcclusionTargetStats.load(path)
    position_path = artifacts.get("position_target_scaler")
    if position_path:
        path = Path(position_path)
        if not path.exists():
            raise FileNotFoundError(f"Position target scaler artifact not found: {path}")
        dataset_kwargs["position_target_scaler"] = PositionTargetStandardScaler.load(path)
    return dataset_kwargs


__all__ = ["load_normalization_artifacts", "save_normalization_artifacts"]
