"""Explicit model-only checkpoint initialization and calibration freezing."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import torch

from kd_sensing.engine.training_resume import CHECKPOINT_SCHEMA_VERSION
from kd_sensing.utils.checkpoint import (
    CheckpointLoadError,
    checkpoint_file_digest,
    load_torch_payload,
    validate_checkpoint_publication,
)
from kd_sensing.utils.paths import resolve_path


_FROZEN_PREFIXES_ATTRIBUTE = "_kd_sensing_frozen_module_prefixes"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def initialize_model_from_checkpoint(
    model: torch.nn.Module,
    training_cfg: Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any] | None:
    """Load model weights without restoring any training trajectory state."""
    raw = training_cfg.get("initialization_checkpoint")
    if raw in (None, False):
        return None
    if training_cfg.get("resume"):
        raise ValueError("training.initialization_checkpoint and training.resume are mutually exclusive.")
    if int(training_cfg.get("start_epoch", 0)) != 0:
        raise ValueError("training.initialization_checkpoint requires training.start_epoch=0.")
    if not isinstance(raw, Mapping):
        raise ValueError("training.initialization_checkpoint must be a mapping.")

    request = dict(raw)
    path = _checkpoint_path(request.get("path"))
    expected_sha256 = _expected_sha256(request.get("sha256"))
    expected_role = _required_string(request, "role")
    expected_schema = _expected_schema(request.get("checkpoint_schema_version", CHECKPOINT_SCHEMA_VERSION))
    required_prefixes = _prefixes(request.get("required_prefixes"), field="required_prefixes", required=True)
    allowed_missing_prefixes = _prefixes(
        request.get("allowed_missing_prefixes", ()),
        field="allowed_missing_prefixes",
        required=False,
    )
    freeze_prefixes = _prefixes(request.get("freeze_prefixes", ()), field="freeze_prefixes", required=False)
    _reject_overlapping_prefixes(required_prefixes, allowed_missing_prefixes)

    actual_sha256, checkpoint_size = checkpoint_file_digest(path)
    if actual_sha256 != expected_sha256:
        raise CheckpointLoadError(
            f"Initialization checkpoint SHA256 mismatch for {path}: "
            f"expected={expected_sha256}, actual={actual_sha256}."
        )
    payload = load_torch_payload(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise CheckpointLoadError(f"Initialization checkpoint payload must be a mapping: {path}.")
    publication = validate_checkpoint_publication(path, payload=payload)
    if payload.get("checkpoint_role") != expected_role:
        raise CheckpointLoadError(
            f"Initialization checkpoint role mismatch for {path}: "
            f"expected={expected_role!r}, actual={payload.get('checkpoint_role')!r}."
        )
    if payload.get("checkpoint_schema_version") != expected_schema:
        raise CheckpointLoadError(
            f"Initialization checkpoint schema mismatch for {path}: "
            f"expected={expected_schema!r}, actual={payload.get('checkpoint_schema_version')!r}."
        )
    source_state = payload.get("state_dict")
    if not isinstance(source_state, Mapping):
        raise CheckpointLoadError(f"Initialization checkpoint state_dict must be a mapping: {path}.")
    if not all(isinstance(key, str) for key in source_state):
        raise CheckpointLoadError(f"Initialization checkpoint state_dict keys must be strings: {path}.")

    target_state = model.state_dict()
    source_keys = set(source_state)
    target_keys = set(target_state)
    _validate_required_prefixes(required_prefixes, source_keys=source_keys, target_keys=target_keys, path=path)
    unexpected_keys = sorted(source_keys - target_keys)
    if unexpected_keys:
        raise CheckpointLoadError(
            f"Initialization checkpoint has unexpected state_dict keys for {path}: {unexpected_keys}."
        )
    missing_keys = sorted(target_keys - source_keys)
    _validate_allowed_missing_prefixes(
        allowed_missing_prefixes,
        missing_keys=missing_keys,
        source_keys=source_keys,
        target_keys=target_keys,
        path=path,
    )
    _validate_shapes(source_state, target_state, path=path)
    frozen_parameter_names = _validate_freeze_prefixes(model, freeze_prefixes)

    incompatible = model.load_state_dict(dict(source_state), strict=False)
    loaded_missing = sorted(incompatible.missing_keys)
    loaded_unexpected = sorted(incompatible.unexpected_keys)
    if loaded_missing != missing_keys or loaded_unexpected:
        raise CheckpointLoadError(
            "Initialization checkpoint load result changed after preflight: "
            f"missing={loaded_missing}, expected_missing={missing_keys}, unexpected={loaded_unexpected}."
        )
    freeze_model_prefixes(model, freeze_prefixes)
    return {
        "path": str(path),
        "role": "initialization",
        "strict": True,
        "source_checkpoint_role": expected_role,
        "checkpoint_schema_version": expected_schema,
        "source_sha256": actual_sha256,
        "checkpoint_size_bytes": int(checkpoint_size),
        "publication_integrity_verified": bool(publication.get("integrity_verified", False)),
        "loaded_key_count": len(source_keys),
        "loaded_keys": sorted(source_keys),
        "missing_keys": missing_keys,
        "unexpected_keys": [],
        "ignored_keys": [],
        "required_prefixes": list(required_prefixes),
        "allowed_missing_prefixes": list(allowed_missing_prefixes),
        "freeze_prefixes": list(freeze_prefixes),
        "frozen_parameter_count": len(frozen_parameter_names),
    }


def freeze_model_prefixes(model: torch.nn.Module, prefixes: tuple[str, ...]) -> None:
    """Freeze matching parameters and keep matching modules in evaluation mode."""
    if not prefixes:
        setattr(model, _FROZEN_PREFIXES_ATTRIBUTE, ())
        return
    for name, parameter in model.named_parameters():
        if _matches_any_prefix(name, prefixes):
            parameter.requires_grad_(False)
    setattr(model, _FROZEN_PREFIXES_ATTRIBUTE, tuple(prefixes))
    enforce_frozen_module_eval(model)


def enforce_frozen_module_eval(model: torch.nn.Module) -> None:
    """Re-apply eval mode after the training loop calls ``model.train()``."""
    prefixes = tuple(getattr(model, _FROZEN_PREFIXES_ATTRIBUTE, ()))
    if not prefixes:
        return
    for name, module in model.named_modules():
        if name and _matches_any_prefix(name, prefixes):
            module.eval()


def _checkpoint_path(value: Any) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("training.initialization_checkpoint.path must be a non-empty path.")
    path = resolve_path(value)
    if path is None or not path.is_file():
        raise CheckpointLoadError(f"Initialization checkpoint does not exist: {path}.")
    return path


def _expected_sha256(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError("training.initialization_checkpoint.sha256 must be a 64-character hexadecimal digest.")
    return normalized


def _required_string(request: Mapping[str, Any], field: str) -> str:
    value = request.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"training.initialization_checkpoint.{field} must be a non-empty string.")
    return value.strip()


def _expected_schema(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("training.initialization_checkpoint.checkpoint_schema_version must be an integer.")
    try:
        schema = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "training.initialization_checkpoint.checkpoint_schema_version must be an integer."
        ) from exc
    if schema < 0:
        raise ValueError("training.initialization_checkpoint.checkpoint_schema_version must be non-negative.")
    return schema


def _prefixes(value: Any, *, field: str, required: bool) -> tuple[str, ...]:
    if value is None:
        value = ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"training.initialization_checkpoint.{field} must be a list of dotted prefixes.")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"training.initialization_checkpoint.{field} contains an empty prefix.")
        prefix = item.strip().rstrip(".")
        if not prefix or "*" in prefix:
            raise ValueError(
                f"training.initialization_checkpoint.{field} requires literal dotted prefixes, got {item!r}."
            )
        normalized.append(prefix)
    if required and not normalized:
        raise ValueError(f"training.initialization_checkpoint.{field} must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"training.initialization_checkpoint.{field} contains duplicate prefixes.")
    return tuple(normalized)


def _reject_overlapping_prefixes(required: tuple[str, ...], allowed_missing: tuple[str, ...]) -> None:
    overlap = [
        (left, right)
        for left in required
        for right in allowed_missing
        if _matches_prefix(left, right) or _matches_prefix(right, left)
    ]
    if overlap:
        raise ValueError(
            "training.initialization_checkpoint required_prefixes and allowed_missing_prefixes overlap: "
            f"{overlap}."
        )


def _validate_required_prefixes(
    prefixes: tuple[str, ...],
    *,
    source_keys: set[str],
    target_keys: set[str],
    path: Path,
) -> None:
    for prefix in prefixes:
        source_matches = {key for key in source_keys if _matches_prefix(key, prefix)}
        target_matches = {key for key in target_keys if _matches_prefix(key, prefix)}
        if not source_matches or not target_matches:
            raise CheckpointLoadError(
                f"Initialization required prefix {prefix!r} is absent from "
                f"{'source' if not source_matches else 'target'} state_dict for {path}."
            )
        missing_required = sorted(target_matches - source_matches)
        if missing_required:
            raise CheckpointLoadError(
                f"Initialization checkpoint is incomplete under required prefix {prefix!r}: {missing_required}."
            )


def _validate_allowed_missing_prefixes(
    prefixes: tuple[str, ...],
    *,
    missing_keys: list[str],
    source_keys: set[str],
    target_keys: set[str],
    path: Path,
) -> None:
    disallowed = [key for key in missing_keys if not _matches_any_prefix(key, prefixes)]
    if disallowed:
        raise CheckpointLoadError(
            f"Initialization checkpoint has non-allowlisted missing keys for {path}: {disallowed}."
        )
    for prefix in prefixes:
        if any(_matches_prefix(key, prefix) for key in source_keys):
            raise CheckpointLoadError(
                f"Initialization allowed-missing prefix {prefix!r} is partially present in source checkpoint {path}."
            )
        if not any(_matches_prefix(key, prefix) for key in target_keys):
            raise CheckpointLoadError(
                f"Initialization allowed-missing prefix {prefix!r} is absent from target model for {path}."
            )
        if not any(_matches_prefix(key, prefix) for key in missing_keys):
            raise CheckpointLoadError(
                f"Initialization allowed-missing prefix {prefix!r} does not match a missing key for {path}."
            )


def _validate_shapes(source: Mapping[str, Any], target: Mapping[str, Any], *, path: Path) -> None:
    mismatches = []
    for key in sorted(set(source) & set(target)):
        source_shape = getattr(source[key], "shape", None)
        target_shape = getattr(target[key], "shape", None)
        if source_shape is None or target_shape is None or tuple(source_shape) != tuple(target_shape):
            mismatches.append((key, source_shape, target_shape))
    if mismatches:
        raise CheckpointLoadError(f"Initialization checkpoint shape mismatch for {path}: {mismatches}.")


def _validate_freeze_prefixes(model: torch.nn.Module, prefixes: tuple[str, ...]) -> list[str]:
    parameter_names = [
        name for name, _ in model.named_parameters() if _matches_any_prefix(name, prefixes)
    ]
    module_names = [name for name, _ in model.named_modules() if name and _matches_any_prefix(name, prefixes)]
    for prefix in prefixes:
        if not any(_matches_prefix(name, prefix) for name in (*parameter_names, *module_names)):
            raise ValueError(f"training.initialization_checkpoint.freeze_prefixes has no model match: {prefix!r}.")
    return parameter_names


def _matches_any_prefix(key: str, prefixes: tuple[str, ...]) -> bool:
    return any(_matches_prefix(key, prefix) for prefix in prefixes)


def _matches_prefix(key: str, prefix: str) -> bool:
    return key == prefix or key.startswith(f"{prefix}.")


__all__ = [
    "enforce_frozen_module_eval",
    "freeze_model_prefixes",
    "initialize_model_from_checkpoint",
]
