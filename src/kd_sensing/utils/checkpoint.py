import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import warnings

import torch


class CheckpointLoadError(RuntimeError):
    """Raised when checkpoint contents do not match the target model."""


def load_torch_payload(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    trusted_local: bool = False,
) -> Any:
    checkpoint_path = Path(path).expanduser()
    if trusted_local:
        if not checkpoint_path.is_file():
            raise CheckpointLoadError(f"Trusted-local checkpoint does not exist: {checkpoint_path}")
        warnings.warn(
            f"Loading trusted-local checkpoint with unsafe pickle enabled: {checkpoint_path}",
            RuntimeWarning,
            stacklevel=2,
        )
    try:
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=not trusted_local,
        )
    except TypeError as exc:  # pragma: no cover - only unsupported PyTorch versions.
        raise CheckpointLoadError(
            "Safe checkpoint loading requires a PyTorch version that supports weights_only."
        ) from exc


def save_checkpoint(state: dict[str, Any], save_path: str | Path, filename: str = "checkpoint.pth") -> Path:
    directory = Path(save_path)
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / filename
    fd, tmp_name = tempfile.mkstemp(prefix=f".{filepath.name}.tmp-", dir=directory)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(state, tmp_path)
        _fsync_file(tmp_path)
        os.replace(tmp_path, filepath)
        _fsync_directory(directory)
    finally:
        tmp_path.unlink(missing_ok=True)
    return filepath


def checkpoint_sidecar_path(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    return path.with_suffix(path.suffix + ".json")


def checkpoint_file_digest(path: str | Path) -> tuple[str, int]:
    checkpoint_path = Path(path)
    digest = hashlib.sha256()
    size = 0
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def write_checkpoint_sidecar(checkpoint_path: str | Path, metadata: dict[str, Any]) -> Path:
    sidecar = checkpoint_sidecar_path(checkpoint_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{sidecar.name}.", suffix=".tmp", dir=sidecar.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_json_ready(metadata), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, sidecar)
        _fsync_directory(sidecar.parent)
    finally:
        tmp_path.unlink(missing_ok=True)
    return sidecar


def publish_checkpoint(
    state: dict[str, Any],
    save_path: str | Path,
    filename: str = "checkpoint.pth",
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Atomically publish a checkpoint followed by its integrity sidecar."""
    filepath = save_checkpoint(state, save_path, filename)
    digest, size = checkpoint_file_digest(filepath)
    completed = {
        **(metadata or {}),
        "path": str(filepath),
        "publish_complete": True,
        "checkpoint_sha256": digest,
        "checkpoint_size_bytes": int(size),
        "checkpoint_schema_version": state.get("checkpoint_schema_version"),
        "checkpoint_role": state.get("checkpoint_role"),
    }
    if "selection" in state:
        completed.setdefault("selection", state["selection"])
    sidecar = write_checkpoint_sidecar(filepath, completed)
    completed["sidecar_path"] = str(sidecar)
    return filepath, completed


def publish_checkpoint_copy(
    source_checkpoint: str | Path,
    target_checkpoint: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Publish a verified checkpoint copy without exposing a partial target."""
    source = Path(source_checkpoint)
    payload = load_torch_payload(source, map_location="cpu")
    validate_checkpoint_publication(source, payload=payload)
    target = Path(target_checkpoint)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(source, tmp_path)
        _fsync_file(tmp_path)
        os.replace(tmp_path, target)
        _fsync_directory(target.parent)
    finally:
        tmp_path.unlink(missing_ok=True)
    digest, size = checkpoint_file_digest(target)
    completed = {
        **(metadata or {}),
        "path": str(target),
        "publish_complete": True,
        "checkpoint_sha256": digest,
        "checkpoint_size_bytes": int(size),
        "checkpoint_schema_version": payload.get("checkpoint_schema_version") if isinstance(payload, dict) else None,
        "checkpoint_role": payload.get("checkpoint_role") if isinstance(payload, dict) else None,
    }
    if isinstance(payload, dict) and "selection" in payload:
        completed.setdefault("selection", payload["selection"])
    sidecar = write_checkpoint_sidecar(target, completed)
    completed["sidecar_path"] = str(sidecar)
    return target, completed


def validate_checkpoint_publication(
    checkpoint_path: str | Path,
    *,
    payload: Any | None = None,
) -> dict[str, Any]:
    """Validate the sidecar completion marker for current-schema checkpoints."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise CheckpointLoadError(f"Checkpoint does not exist: {path}")
    if payload is None:
        payload = load_torch_payload(path, map_location="cpu")
    current = isinstance(payload, dict) and "checkpoint_schema_version" in payload
    sidecar = checkpoint_sidecar_path(path)
    if not sidecar.is_file():
        if current:
            raise CheckpointLoadError(f"Current checkpoint is missing its completion sidecar: {sidecar}")
        return {
            "path": str(path),
            "publish_complete": False,
            "integrity_verified": False,
            "checkpoint_schema_version": None,
            "checkpoint_role": payload.get("checkpoint_role") if isinstance(payload, dict) else None,
        }
    try:
        with sidecar.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        if current:
            raise CheckpointLoadError(f"Current checkpoint has an unreadable completion sidecar: {sidecar}") from exc
        return {
            "path": str(path),
            "publish_complete": False,
            "integrity_verified": False,
            "checkpoint_schema_version": None,
        }
    if not isinstance(metadata, dict):
        if current:
            raise CheckpointLoadError(f"Current checkpoint sidecar must contain a mapping: {sidecar}")
        return {"path": str(path), "publish_complete": False, "integrity_verified": False}

    integrity_fields = ("checkpoint_sha256", "checkpoint_size_bytes")
    has_integrity = all(field in metadata for field in integrity_fields)
    if current:
        missing = [
            field
            for field in (
                "publish_complete",
                "checkpoint_sha256",
                "checkpoint_size_bytes",
                "checkpoint_schema_version",
                "checkpoint_role",
            )
            if field not in metadata
        ]
        if missing:
            raise CheckpointLoadError(
                f"Current checkpoint sidecar is incomplete for {path}; missing fields: {missing}."
            )
        if metadata.get("publish_complete") is not True:
            raise CheckpointLoadError(f"Current checkpoint publication is incomplete: {path}")
        if metadata.get("checkpoint_schema_version") != payload.get("checkpoint_schema_version"):
            raise CheckpointLoadError(f"Checkpoint schema version does not match sidecar for {path}.")
        if metadata.get("checkpoint_role") != payload.get("checkpoint_role"):
            raise CheckpointLoadError(f"Checkpoint role does not match sidecar for {path}.")
    if has_integrity:
        digest, size = checkpoint_file_digest(path)
        if int(metadata["checkpoint_size_bytes"]) != size:
            raise CheckpointLoadError(f"Checkpoint size does not match completion sidecar for {path}.")
        if str(metadata["checkpoint_sha256"]) != digest:
            raise CheckpointLoadError(f"Checkpoint digest does not match completion sidecar for {path}.")
        metadata = {**metadata, "integrity_verified": True}
    else:
        metadata = {**metadata, "integrity_verified": False}
    return metadata


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    *,
    strict: bool = True,
    role: str = "resume",
    map_location: str | torch.device = "cpu",
):
    checkpoint_path = Path(path)
    load_result = load_model_state(
        checkpoint_path,
        model,
        role=role,
        map_location=map_location,
        strict=strict,
    )
    checkpoint = load_result["checkpoint"]
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    checkpoint["_load_info"] = checkpoint_load_summary(load_result)
    return checkpoint


def load_model_state(
    path: str | Path,
    model,
    *,
    role: str = "model",
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    checkpoint = load_torch_payload(checkpoint_path, map_location=map_location)
    publication = validate_checkpoint_publication(checkpoint_path, payload=checkpoint)
    state_dict = _extract_state_dict(checkpoint)
    ignored_keys = sorted(key for key in state_dict if _is_stats_key(key))
    state_dict = {key: value for key, value in state_dict.items() if not _is_stats_key(key)}
    try:
        incompatible = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise CheckpointLoadError(
            f"Failed to load {role} checkpoint {checkpoint_path}: {exc}"
        ) from exc
    missing_keys = sorted(incompatible.missing_keys)
    unexpected_keys = sorted(incompatible.unexpected_keys)
    result = {
        "checkpoint": checkpoint,
        "path": str(checkpoint_path),
        "role": role,
        "strict": bool(strict),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "ignored_keys": ignored_keys,
        "publication": publication,
    }
    if strict and (missing_keys or unexpected_keys):
        raise CheckpointLoadError(_format_mismatch(result))
    return result


def checkpoint_load_summary(load_result: dict[str, Any] | None) -> dict[str, Any] | None:
    if load_result is None:
        return None
    return {
        "path": load_result["path"],
        "role": load_result["role"],
        "strict": load_result["strict"],
        "missing_keys": list(load_result["missing_keys"]),
        "unexpected_keys": list(load_result["unexpected_keys"]),
        "ignored_keys": list(load_result["ignored_keys"]),
    }


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise CheckpointLoadError(f"Checkpoint payload must be a state dict, got {type(state_dict).__name__}.")
    return state_dict


def _format_mismatch(load_result: dict[str, Any]) -> str:
    parts = [
        f"Checkpoint mismatch while loading {load_result['role']} from {load_result['path']}.",
    ]
    if load_result["missing_keys"]:
        parts.append(f"Missing keys: {load_result['missing_keys']}.")
    if load_result["unexpected_keys"]:
        parts.append(f"Unexpected keys: {load_result['unexpected_keys']}.")
    return " ".join(parts)


def _is_stats_key(key: str) -> bool:
    return key.endswith("total_ops") or key.endswith("total_params")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value
