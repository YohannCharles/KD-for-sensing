from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointLoadError(RuntimeError):
    """Raised when checkpoint contents do not match the target model."""


def save_checkpoint(state: dict[str, Any], save_path: str | Path, filename: str = "checkpoint.pth") -> Path:
    directory = Path(save_path)
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / filename
    torch.save(state, filepath)
    return filepath


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
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
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
