import hashlib
import json
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from kd_sensing.data.transform_ops.gps import load_gps_scaler
from kd_sensing.engine.data_factory_groups import leaf_datasets_with_indices
from kd_sensing.engine.data_factory_scalers import shared_dataset_attribute


def save_normalization_artifacts(dataloaders: dict[str, DataLoader], run_dir: str | Path) -> dict[str, Any]:
    train_loader = dataloaders.get("train")
    train_dataset = train_loader.dataset if train_loader is not None else None
    if train_dataset is None:
        return {}
    scaler = _shared_gps_scaler(train_dataset)
    if scaler is None:
        return {}
    path = Path(run_dir) / "artifacts" / "gps_scaler.npz"
    scaler.save(path)
    return {"gps_scaler": str(path), "metadata": split_dependent_artifact_metadata(train_dataset)}


def load_normalization_artifacts(metadata: dict[str, Any] | None) -> dict[str, Any]:
    artifacts = (metadata or {}).get("normalization_artifacts") or {}
    if not artifacts:
        return {}
    _validate_artifact_metadata_integrity(artifacts)
    path = artifacts.get("gps_scaler")
    return {"gps_scaler": load_gps_scaler(_existing_path(path, "GPS scaler"))} if path else {}


def split_dependent_artifact_metadata(train_dataset: Any) -> dict[str, Any]:
    sources = [
        {
            "domain_id": getattr(leaf, "domain_id", None),
            "scene": getattr(leaf, "scene_slug", getattr(leaf, "scene_id", None)),
            "source_csv_path": str(getattr(leaf, "root_csv", "")) or None,
            "sample_count": len(indices),
        }
        for leaf, indices in leaf_datasets_with_indices(train_dataset)
    ]
    if not sources:
        raise ValueError("GPS normalization artifact provenance requires a train dataset.")
    modes = {
        str(getattr(leaf, "gps_feature_mode", "relative_polar"))
        for leaf, _ in leaf_datasets_with_indices(train_dataset)
        if getattr(leaf, "use_gps", False)
    }
    if len(modes) > 1:
        raise ValueError(f"Pooled GPS datasets must use one gps_feature_mode, got {sorted(modes)}.")
    payload = {
        "schema_version": 1,
        "fit_split": "train",
        "normalization_modalities": ["gps"],
        "gps_feature_mode": next(iter(modes), None),
        "effective_sample_count": sum(int(item["sample_count"]) for item in sources),
        "source_components": sources,
    }
    payload["normalization_fingerprint"] = _fingerprint(payload)
    return payload


def validate_normalization_artifact_fingerprint(cfg: dict[str, Any], metadata: dict[str, Any] | None) -> None:
    artifacts = (metadata or {}).get("normalization_artifacts") or {}
    if not artifacts:
        return
    _validate_artifact_metadata_integrity(artifacts)
    if artifacts.get("gps_scaler"):
        expected = str(cfg.get("data", {}).get("dataset", {}).get("gps_feature_mode", "relative_polar"))
        recorded = artifacts["metadata"].get("gps_feature_mode")
        if recorded != expected:
            raise ValueError(f"GPS normalization artifact feature mode {recorded!r} does not match evaluation config {expected!r}.")


def _shared_gps_scaler(dataset: Any) -> Any:
    return shared_dataset_attribute(
        dataset,
        "gps_scaler",
        enabled=lambda leaf: bool(getattr(leaf, "use_gps", False) and getattr(leaf, "gps_normalize", False)),
    )


def _existing_path(value: str | Path, name: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{name} artifact not found: {path}")
    return path


def _validate_artifact_metadata_integrity(artifacts: dict[str, Any]) -> None:
    metadata = artifacts.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("fit_split") != "train":
        raise ValueError("Evaluation refuses GPS normalization artifacts without train-fit provenance metadata.")
    if int(metadata.get("effective_sample_count") or 0) <= 0:
        raise ValueError("GPS normalization artifact effective_sample_count must be positive.")
    payload = {key: value for key, value in metadata.items() if key != "normalization_fingerprint"}
    if metadata.get("normalization_fingerprint") != _fingerprint(payload):
        raise ValueError("GPS normalization artifact fingerprint does not match its provenance metadata.")
    if metadata.get("normalization_modalities") != ["gps"]:
        raise ValueError("GPS normalization artifact modalities do not match saved artifacts.")


def _fingerprint(payload: Any) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


__all__ = ["load_normalization_artifacts", "save_normalization_artifacts", "split_dependent_artifact_metadata", "validate_normalization_artifact_fingerprint"]
