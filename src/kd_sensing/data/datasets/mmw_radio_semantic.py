from pathlib import Path
from typing import Any

import numpy as np
import torch


def _radio_semantic_config(value: bool | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, dict):
        payload = dict(value)
        payload["enabled"] = bool(payload.get("enabled", payload.get("enable", False)))
        return payload
    if value is True:
        return {"enabled": True}
    return {"enabled": False}

def _path_semantic_config(value: bool | dict[str, Any] | None, *, field_map: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(value, dict):
        payload = dict(value)
        payload["enabled"] = bool(payload.get("enabled", payload.get("enable", False)))
    elif value is True:
        payload = {"enabled": True}
    else:
        payload = {"enabled": False}
    if field_map:
        payload.setdefault("field_map", field_map)
    payload.setdefault("mode", "kmeans_path_descriptor")
    payload.setdefault("num_path_classes", 24)
    payload.setdefault("fit_on_source_only", True)
    payload.setdefault("fallback_if_missing", "radio_power")
    payload.setdefault("use_path_regression", True)
    return payload

def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value

def _optional_row_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text or text in {"-99", "nan", "None"}:
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return int(parsed)

def _collate_safe_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, dict):
        return {str(key): _collate_safe_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_collate_safe_value(item) for item in value]
    if isinstance(value, list):
        return [_collate_safe_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value

def _beam_power_for_horizon(value: Any, horizon: int) -> np.ndarray | None:
    if not torch.is_tensor(value):
        return None
    tensor = value.detach().cpu()
    if tensor.ndim == 1:
        return tensor.numpy()
    if tensor.ndim >= 2 and horizon < tensor.shape[0]:
        return tensor[horizon].reshape(-1).numpy()
    return None

def _radio_label_for_horizon(sample: dict[str, Any], horizon: int) -> int | None:
    value = sample.get("radio_semantic_label")
    if not torch.is_tensor(value):
        return None
    labels = value.detach().cpu().reshape(-1)
    if horizon >= labels.numel():
        return None
    label = int(labels[horizon].item())
    return label if label >= 0 else None


__all__ = [
    "_beam_power_for_horizon",
    "_collate_safe_value",
    "_json_scalar",
    "_optional_row_int",
    "_path_semantic_config",
    "_radio_label_for_horizon",
    "_radio_semantic_config",
]
