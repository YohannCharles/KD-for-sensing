import csv
import hashlib
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from kd_sensing.data.transform_ops.gps import load_gps_scaler
from kd_sensing.data.mmw.trajectory_protocol import (
    TRAJECTORY_PROTOCOL_MODE,
    split_cache_identity,
    validate_split_cache_identity,
)
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
    metadata = split_dependent_artifact_metadata(train_dataset)
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                **metadata,
                "artifact": str(path),
                "artifact_sha256": _sha256_file(path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"gps_scaler": str(path), "metadata": metadata, "metadata_sidecar": str(sidecar)}


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
        "source_split": "train",
        "normalization_modalities": ["gps"],
        "gps_feature_mode": next(iter(modes), None),
        "effective_sample_count": sum(int(item["sample_count"]) for item in sources),
        "sample_id_hash": _sample_id_hash(train_dataset),
        "creation_command": shlex.join(sys.argv),
        "creation_time": datetime.now(timezone.utc).isoformat(),
        "source_components": sources,
    }
    protocol = getattr(train_dataset, "data_protocol_identity", None)
    if isinstance(protocol, dict):
        payload.update(split_cache_identity(protocol))
        payload.update(
            split_manifest=protocol.get("split_manifest"),
            protocol_fingerprint=protocol.get("protocol_fingerprint"),
        )
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
    protocol = cfg.get("data_protocol")
    if isinstance(protocol, dict) and protocol.get("mode") == TRAJECTORY_PROTOCOL_MODE:
        report = json.loads(Path(str(protocol.get("audit_report", ""))).read_text(encoding="utf-8"))
        recorded = artifacts["metadata"]
        expected_source_split = str(protocol.get("train_role", "train"))
        if recorded.get("source_split") != expected_source_split:
            raise ValueError(
                "MMW GPS normalization artifact source split does not match the protocol train role."
            )
        if recorded.get("sample_id_hash") != report.get("train_sample_id_hash"):
            raise ValueError("MMW GPS normalization artifact sample identity does not match the split audit.")
        if int(recorded.get("effective_sample_count", 0)) != int(report.get("train_sample_count", -1)):
            raise ValueError("MMW GPS normalization artifact sample count does not match the split audit.")
        expected_identity = {
            "split_protocol": protocol.get("protocol_id"),
            "protocol_version": protocol.get("protocol_version"),
            "split_seed": protocol.get("split_seed"),
            "block_size": protocol.get("block_size"),
            "split_manifest_hash": protocol.get("split_manifest_hash"),
            "data_source_hash": protocol.get("data_source_hash"),
            "window_config_hash": protocol.get("window_config_hash"),
            "weather_binding": protocol.get("weather_binding"),
            "split_manifest": protocol.get("split_manifest", protocol.get("path")),
            "protocol_fingerprint": protocol.get("protocol_fingerprint"),
        }
        validate_split_cache_identity(recorded, protocol)
        if any(recorded.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("MMW GPS normalization artifact does not match the bound split manifest identity.")


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
    sidecar = artifacts.get("metadata_sidecar")
    if sidecar:
        recorded = json.loads(_existing_path(sidecar, "GPS scaler metadata sidecar").read_text(encoding="utf-8"))
        if any(recorded.get(key) != value for key, value in metadata.items()):
            raise ValueError("GPS normalization sidecar does not match checkpoint provenance metadata.")
        if recorded.get("artifact_sha256") != _sha256_file(_existing_path(artifacts.get("gps_scaler"), "GPS scaler")):
            raise ValueError("GPS normalization artifact SHA256 does not match its sidecar.")


def _fingerprint(payload: Any) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _sample_id_hash(dataset: Any) -> str:
    values = []
    for leaf, indices in leaf_datasets_with_indices(dataset):
        rows = getattr(getattr(leaf, "samples", None), "rows", None)
        if not isinstance(rows, list):
            rows = _csv_rows(Path(str(getattr(leaf, "root_csv", ""))))
        for index in indices:
            row = rows[int(index)] if int(index) < len(rows) else {}
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id and str(getattr(leaf, "schema_identity", {}).get("dataset_family", "")) == "DeepSense6G":
                scene = str(getattr(leaf, "scene_slug", "")).strip()
                split = str(getattr(leaf, "split", "")).strip()
                seq_index = str(row.get("seq_index", "")).strip() or str(int(index))
                if scene and split:
                    sample_id = f"deepsense6g:{scene}:{split}:{seq_index}"
            if not sample_id:
                raise ValueError("GPS normalization provenance requires a sample_id for every train sample.")
            values.append(sample_id)
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["load_normalization_artifacts", "save_normalization_artifacts", "split_dependent_artifact_metadata", "validate_normalization_artifact_fingerprint"]
