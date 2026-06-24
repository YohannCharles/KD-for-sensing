import hashlib
import json
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from kd_sensing.config.canonical import SNAPSHOT_VARIANT
from kd_sensing.data.split_metadata import split_metadata_summary_for_csv
from kd_sensing.data.transform_ops.csi import CSIRMSNormalizer
from kd_sensing.data.transform_ops.gps import PositionTargetStandardScaler, load_gps_scaler
from kd_sensing.data.transform_ops.lidar import LidarBEVNormalizer
from kd_sensing.data.transform_ops.mmwave import MmWaveStandardScaler, OcclusionTargetStats


def save_normalization_artifacts(dataloaders: dict[str, DataLoader], run_dir: str | Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
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

    csi_normalizer = getattr(train_dataset, "csi_rms_normalizer", None)
    if csi_normalizer is not None and getattr(train_dataset, "csi_train_rms", False):
        csi_path = artifact_dir / "csi_rms_normalizer.npz"
        csi_normalizer.save(csi_path)
        artifacts["csi_rms_normalizer"] = str(csi_path)

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
    if artifacts:
        artifacts["metadata"] = split_dependent_artifact_metadata(train_dataset)
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
    csi_path = artifacts.get("csi_rms_normalizer")
    if csi_path:
        path = Path(csi_path)
        if not path.exists():
            raise FileNotFoundError(f"CSI RMS normalizer artifact not found: {path}")
        dataset_kwargs["csi_rms_normalizer"] = CSIRMSNormalizer.load(path)
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


def split_dependent_artifact_metadata(train_dataset: Any) -> dict[str, Any]:
    csv_path = getattr(train_dataset, "root_csv", None)
    split_metadata = {}
    if csv_path is not None:
        split_metadata = split_metadata_summary_for_csv(csv_path, split="train")
    payload = {
        "source_csv_path": str(csv_path) if csv_path is not None else None,
        "split_metadata_path": split_metadata.get("path"),
        "split_protocol": split_metadata.get("split_protocol"),
        "split_seed": split_metadata.get("split_seed"),
        "in_len": split_metadata.get("in_len"),
        "out_len": split_metadata.get("out_len"),
        "seq_len": int(getattr(train_dataset, "seq_len", 0) or 0),
        "num_pred": int(getattr(train_dataset, "num_pred", 0) or 0),
        "enabled_modalities": list(getattr(train_dataset, "enabled_modalities", [])),
        "num_samples": len(train_dataset),
    }
    payload["split_fingerprint"] = _fingerprint(payload)
    return payload


def validate_normalization_artifact_fingerprint(cfg: dict[str, Any], metadata: dict[str, Any] | None) -> None:
    if cfg.get("experiment", {}).get("variant") != SNAPSHOT_VARIANT:
        return
    if not metadata:
        return
    artifacts = metadata.get("normalization_artifacts") or {}
    if not _has_split_dependent_artifact(artifacts):
        return
    artifact_metadata = artifacts.get("metadata")
    if not isinstance(artifact_metadata, dict):
        raise ValueError(
            "Snapshot next-frame evaluation refuses split-dependent normalization artifacts without "
            "snapshot split fingerprint metadata."
        )
    if artifact_metadata.get("split_protocol") != "snapshot_next_frame_balanced_seq":
        raise ValueError(
            "Snapshot next-frame evaluation refuses normalization artifacts fitted on a non-snapshot split; "
            f"got split_protocol={artifact_metadata.get('split_protocol')!r}."
        )
    if int(artifact_metadata.get("seq_len") or 0) != 1 or int(artifact_metadata.get("num_pred") or 0) != 1:
        raise ValueError(
            "Snapshot next-frame evaluation requires normalization artifacts fitted with seq_len=1 and num_pred=1."
        )
    checkpoint_splits = metadata.get("split_metadata") or {}
    train_split = checkpoint_splits.get("train") if isinstance(checkpoint_splits, dict) else None
    if isinstance(train_split, dict):
        train_protocol = train_split.get("split_protocol")
        if train_protocol and train_protocol != artifact_metadata.get("split_protocol"):
            raise ValueError(
                "Normalization artifact split fingerprint does not match the checkpoint train split metadata."
            )


def _has_split_dependent_artifact(artifacts: dict[str, Any]) -> bool:
    return any(
        artifacts.get(key)
        for key in (
            "gps_scaler",
            "lidar_normalizer",
            "mmwave_scaler",
            "csi_rms_normalizer",
            "occlusion_target_stats",
            "position_target_scaler",
        )
    )


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


__all__ = [
    "load_normalization_artifacts",
    "save_normalization_artifacts",
    "split_dependent_artifact_metadata",
    "validate_normalization_artifact_fingerprint",
]
