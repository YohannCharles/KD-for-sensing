import hashlib
import json
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from kd_sensing.data.split_metadata import split_metadata_summary_for_csv
from kd_sensing.data.transform_ops.gps import load_gps_scaler
from kd_sensing.data.transform_ops.lidar import LidarBEVNormalizer
from kd_sensing.engine.data_factory_groups import leaf_datasets_with_indices
from kd_sensing.engine.data_factory_scalers import shared_dataset_attribute


def save_normalization_artifacts(dataloaders: dict[str, DataLoader], run_dir: str | Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    train_dataset = dataloaders.get("train").dataset if dataloaders.get("train") is not None else None
    if train_dataset is None:
        return artifacts
    metadata = normalization_artifact_metadata(dataloaders)
    artifact_dir = Path(run_dir) / "artifacts"

    gps_scaler = _shared_artifact(train_dataset, "gps_scaler")
    if gps_scaler is not None:
        gps_path = artifact_dir / "gps_scaler.npz"
        gps_scaler.save(gps_path)
        artifacts["gps_scaler"] = str(gps_path)

    lidar_normalizer = _shared_artifact(train_dataset, "lidar_normalizer")
    if lidar_normalizer is not None:
        lidar_path = artifact_dir / "lidar_normalizer.npz"
        lidar_normalizer.save(lidar_path)
        artifacts["lidar_normalizer"] = str(lidar_path)

    if artifacts:
        artifacts["metadata"] = metadata
    return artifacts


def normalization_artifact_metadata(dataloaders: dict[str, DataLoader]) -> dict[str, Any]:
    train_loader = dataloaders.get("train")
    train_dataset = train_loader.dataset if train_loader is not None else None
    if train_dataset is None:
        return {}
    artifact_attributes = (
        ("gps_scaler", "gps_scaler"),
        ("lidar_normalizer", "lidar_normalizer"),
    )
    modalities = tuple(
        artifact_name
        for artifact_name, attribute in artifact_attributes
        if _shared_artifact(train_dataset, attribute) is not None
    )
    if not modalities:
        return {}
    return split_dependent_artifact_metadata(train_dataset, modalities=modalities)


def load_normalization_artifacts(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    artifacts = metadata.get("normalization_artifacts") or {}
    if _has_split_dependent_artifact(artifacts):
        _validate_artifact_metadata_integrity(artifacts)
    artifact_metadata = artifacts.get("metadata") if isinstance(artifacts.get("metadata"), dict) else {}
    dataset_kwargs: dict[str, Any] = {}
    gps_path = artifacts.get("gps_scaler")
    if gps_path:
        path = Path(gps_path)
        if not path.exists():
            raise FileNotFoundError(f"GPS scaler artifact not found: {path}")
        scaler = load_gps_scaler(path)
        metadata_mode = artifact_metadata.get("gps_feature_mode")
        scaler_mode = getattr(scaler, "feature_mode_", None)
        if metadata_mode is not None and scaler_mode is not None and str(metadata_mode) != str(scaler_mode):
            raise ValueError(
                f"GPS scaler artifact mode {scaler_mode!r} does not match metadata mode {metadata_mode!r}."
            )
        if scaler_mode is None and metadata_mode is not None:
            scaler.feature_mode_ = str(metadata_mode)
        dataset_kwargs["gps_scaler"] = scaler
    lidar_path = artifacts.get("lidar_normalizer")
    if lidar_path:
        path = Path(lidar_path)
        if not path.exists():
            raise FileNotFoundError(f"LiDAR normalizer artifact not found: {path}")
        dataset_kwargs["lidar_normalizer"] = LidarBEVNormalizer.load(path)
    return dataset_kwargs


def split_dependent_artifact_metadata(
    train_dataset: Any,
    *,
    modalities: tuple[str, ...] = (),
) -> dict[str, Any]:
    source_components = _normalization_source_components(train_dataset)
    if not source_components:
        raise ValueError("Normalization artifact provenance requires at least one train dataset leaf.")
    source_paths = [item["source_csv_path"] for item in source_components]
    split_protocol = _common_component_value(source_components, "split_protocol")
    split_seed = _common_component_value(source_components, "split_seed")
    in_len = _common_component_value(source_components, "in_len")
    out_len = _common_component_value(source_components, "out_len")
    seq_len = _common_leaf_value(train_dataset, "seq_len", default=0)
    num_pred = _common_leaf_value(train_dataset, "num_pred", default=0)
    enabled_modalities = _common_leaf_value(train_dataset, "enabled_modalities", default=[])
    gps_feature_mode = _common_leaf_value(
        train_dataset,
        "gps_feature_mode",
        default=None,
        enabled=lambda leaf: bool(getattr(leaf, "use_gps", False)),
    )
    payload = {
        "schema_version": 2,
        "fit_split": "train",
        "domain_policy": "shared",
        "normalization_modalities": [_artifact_modality(item) for item in modalities],
        "feature_mode": gps_feature_mode,
        "source_csv_path": source_paths[0] if len(source_paths) == 1 else None,
        "split_metadata_path": source_components[0]["split_metadata_path"] if len(source_components) == 1 else None,
        "split_protocol": split_protocol,
        "split_seed": split_seed,
        "in_len": in_len,
        "out_len": out_len,
        "seq_len": int(seq_len or 0),
        "num_pred": int(num_pred or 0),
        "enabled_modalities": list(enabled_modalities or []),
        "num_samples": sum(int(item["effective_sample_count"]) for item in source_components),
        "effective_sample_count": sum(int(item["effective_sample_count"]) for item in source_components),
        "gps_feature_mode": gps_feature_mode,
        "gps_angle_frame": _common_leaf_value(
            train_dataset,
            "gps_angle_frame",
            default=None,
            enabled=lambda leaf: bool(getattr(leaf, "use_gps", False)),
        ),
        "gps_yaw_source": _common_leaf_value(
            train_dataset,
            "gps_yaw_source",
            default=None,
            enabled=lambda leaf: bool(getattr(leaf, "use_gps", False)),
        ),
        "gps_yaw_validation_policy": _common_leaf_value(
            train_dataset,
            "gps_yaw_validation_policy",
            default=None,
            enabled=lambda leaf: bool(getattr(leaf, "use_gps", False)),
        ),
        "gps_yaw_validation": _common_leaf_value(
            train_dataset,
            "gps_yaw_validation",
            default=None,
            enabled=lambda leaf: bool(getattr(leaf, "use_gps", False)),
        ),
        "source_component_count": len(source_components),
        "source_components": source_components,
    }
    fingerprint = _fingerprint(payload)
    payload["normalization_fingerprint"] = fingerprint
    payload["split_fingerprint"] = fingerprint
    return payload


def _shared_artifact(dataset: Any, attribute: str) -> Any:
    return shared_dataset_attribute(
        dataset,
        attribute,
        enabled=lambda leaf: _normalization_enabled(leaf, attribute),
    )


def _normalization_enabled(leaf: Any, attribute: str) -> bool:
    if attribute == "gps_scaler":
        return bool(getattr(leaf, "use_gps", False) and getattr(leaf, "gps_normalize", False))
    if attribute == "lidar_normalizer":
        return bool(getattr(leaf, "use_lidar", False) and getattr(leaf, "lidar_normalize", False))
    return False


def _normalization_source_components(dataset: Any) -> list[dict[str, Any]]:
    components = []
    for leaf, indices in leaf_datasets_with_indices(dataset):
        csv_path = getattr(leaf, "root_csv", None)
        path = Path(csv_path) if csv_path is not None else None
        split_metadata = split_metadata_summary_for_csv(path, split="train") if path is not None else {}
        components.append(
            {
                "domain_id": getattr(leaf, "domain_id", None),
                "scene_id": getattr(leaf, "scene_id", None),
                "scene_slug": getattr(leaf, "scene_slug", None),
                "source_csv_path": str(path) if path is not None else None,
                "source_csv_sha256": _file_sha256(path) if path is not None and path.exists() else None,
                "sample_count": len(indices),
                "effective_sample_count": len(indices),
                "effective_indices_fingerprint": _fingerprint([int(index) for index in indices]),
                "split_metadata_path": split_metadata.get("path"),
                "split_protocol": split_metadata.get("split_protocol"),
                "split_seed": split_metadata.get("split_seed"),
                "in_len": split_metadata.get("in_len"),
                "out_len": split_metadata.get("out_len"),
                "gps_feature_mode": getattr(leaf, "gps_feature_mode", None),
                "gps_angle_frame": getattr(leaf, "gps_angle_frame", None),
                "gps_yaw_source": getattr(leaf, "gps_yaw_source", None),
                "gps_yaw_validation_policy": getattr(leaf, "gps_yaw_validation_policy", None),
                "gps_yaw_validation": getattr(leaf, "gps_yaw_validation", None),
            }
        )
    return components


def _common_component_value(components: list[dict[str, Any]], key: str) -> Any:
    return _one_common_value([component.get(key) for component in components], key)


def _common_leaf_value(
    dataset: Any,
    attribute: str,
    *,
    default: Any,
    enabled=None,
) -> Any:
    values = [
        getattr(leaf, attribute, default)
        for leaf, _ in leaf_datasets_with_indices(dataset)
        if enabled is None or enabled(leaf)
    ]
    if not values:
        return default
    return _one_common_value(values, attribute)


def _one_common_value(values: list[Any], name: str) -> Any:
    normalized = {json.dumps(value, sort_keys=True, default=str) for value in values}
    if len(normalized) > 1:
        raise ValueError(f"Pooled normalization leaves have incompatible {name}: {values}.")
    return values[0] if values else None


def _artifact_modality(artifact: str) -> str:
    return {
        "gps_scaler": "gps",
        "lidar_normalizer": "lidar",
    }.get(str(artifact), str(artifact))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_normalization_artifact_fingerprint(cfg: dict[str, Any], metadata: dict[str, Any] | None) -> None:
    if not metadata:
        return
    artifacts = metadata.get("normalization_artifacts") or {}
    if not _has_split_dependent_artifact(artifacts):
        return
    _validate_artifact_metadata_integrity(artifacts)
    artifact_metadata = artifacts.get("metadata")
    assert isinstance(artifact_metadata, dict)
    if artifacts.get("gps_scaler"):
        expected_mode = str(
            cfg.get("data", {}).get("dataset", {}).get("gps_feature_mode", "relative_polar")
        )
        artifact_mode = artifact_metadata.get("gps_feature_mode") or artifact_metadata.get("feature_mode")
        if not artifact_mode:
            raise ValueError("GPS normalization artifact is missing gps_feature_mode provenance.")
        if str(artifact_mode) != expected_mode:
            raise ValueError(
                f"GPS normalization artifact feature mode {artifact_mode!r} does not match "
                f"evaluation config {expected_mode!r}."
            )


def _validate_artifact_metadata_integrity(artifacts: dict[str, Any]) -> None:
    artifact_metadata = artifacts.get("metadata")
    if not isinstance(artifact_metadata, dict):
        raise ValueError(
            "Evaluation refuses split-dependent normalization artifacts without train-fit provenance metadata."
        )
    if artifact_metadata.get("fit_split") != "train":
        raise ValueError(
            f"Normalization artifact fit_split must be 'train', got {artifact_metadata.get('fit_split')!r}."
        )
    if int(artifact_metadata.get("effective_sample_count") or 0) <= 0:
        raise ValueError("Normalization artifact effective_sample_count must be positive.")
    if artifact_metadata.get("domain_policy") not in {"shared", "per_domain"}:
        raise ValueError("Normalization artifact domain_policy must be 'shared' or 'per_domain'.")
    if artifact_metadata.get("domain_policy") == "per_domain" and not isinstance(artifacts.get("per_domain"), dict):
        raise ValueError("Per-domain normalization requires an explicit per_domain artifact mapping.")
    if "feature_mode" not in artifact_metadata:
        raise ValueError("Normalization artifact is missing feature_mode provenance.")
    expected_modalities = sorted(
        _artifact_modality(key)
        for key in (
            "gps_scaler",
            "lidar_normalizer",
        )
        if artifacts.get(key)
    )
    recorded_modalities = sorted(str(item) for item in artifact_metadata.get("normalization_modalities", []))
    if recorded_modalities != expected_modalities:
        raise ValueError(
            f"Normalization artifact modalities {recorded_modalities} do not match artifacts {expected_modalities}."
        )
    recorded = artifact_metadata.get("normalization_fingerprint")
    if not recorded:
        raise ValueError("Normalization artifact is missing normalization_fingerprint provenance.")
    fingerprint_payload = {
        key: value
        for key, value in artifact_metadata.items()
        if key not in {"normalization_fingerprint", "split_fingerprint"}
    }
    expected = _fingerprint(fingerprint_payload)
    if str(recorded) != expected:
        raise ValueError("Normalization artifact fingerprint does not match its provenance metadata.")


def _has_split_dependent_artifact(artifacts: dict[str, Any]) -> bool:
    return any(
        artifacts.get(key)
        for key in (
            "gps_scaler",
            "lidar_normalizer",
        )
    )


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


__all__ = [
    "load_normalization_artifacts",
    "normalization_artifact_metadata",
    "save_normalization_artifacts",
    "split_dependent_artifact_metadata",
    "validate_normalization_artifact_fingerprint",
]
