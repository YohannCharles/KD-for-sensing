from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from kd_sensing.data.transform_ops.io import joined_resource


def resolve_last_input_channel_ref(
    row: Mapping[str, Any],
    history_channel_paths: Sequence[str],
    *,
    data_root: str | Path,
    seq_len: int,
    num_pred: int,
) -> dict[str, str]:
    resolved = resolve_input_channel_refs(
        row,
        history_channel_paths,
        data_root=data_root,
        seq_len=seq_len,
        num_pred=num_pred,
    )
    return {
        "channel_ref": resolved["channel_history_refs"][-1],
        "pilot_frame_id": resolved["history_frame_ids"][-1],
        "last_input_frame_id": resolved["last_input_frame_id"],
        "target_frame_id": resolved["target_frame_id"],
    }


def resolve_input_channel_refs(
    row: Mapping[str, Any],
    history_channel_paths: Sequence[str],
    *,
    data_root: str | Path,
    seq_len: int,
    num_pred: int,
) -> dict[str, Any]:
    history = _frame_ids(row, "history_frame_ids_json")
    future = _frame_ids(row, "future_frame_ids_json")
    channels = [str(value).strip() for value in history_channel_paths]
    if len(history) < int(seq_len) or len(channels) < int(seq_len) or len(future) < int(num_pred):
        raise ValueError("Sparse pilot row lacks the configured history/target frame identities.")
    history = history[-int(seq_len) :]
    channels = channels[-int(seq_len) :]
    try:
        history_numbers = [int(value) for value in history]
        target_number = int(future[0])
    except ValueError as exc:
        raise ValueError("MMW sparse pilot frame ids must be numeric.") from exc
    if any(right != left + 1 for left, right in zip(history_numbers, history_numbers[1:], strict=False)):
        raise ValueError("MMW sparse pilot history frame ids must be consecutive and increasing.")
    if history_numbers[-1] >= target_number:
        raise ValueError(
            "Sparse pilot temporal leakage: require every history frame before target, "
            f"got last_input={history[-1]}, target={future[0]}."
        )

    resolved: list[str] = []
    for frame_id, reference in zip(history, channels, strict=True):
        pilot = _channel_frame_id(reference)
        if pilot != frame_id:
            raise ValueError(f"Pilot channel frame {pilot!r} does not match history frame {frame_id!r}.")
        path = joined_resource(data_root, reference)
        if path.suffix.lower() != ".npz" or not path.name.endswith("_paths.npz"):
            raise ValueError(f"Sparse pilot channel_ref must be a *_paths.npz file, got {reference!r}.")
        if not path.is_file():
            raise FileNotFoundError(f"Sparse pilot channel_ref is missing: {path}")
        resolved.append(str(path.resolve()))
    return {
        "channel_history_refs": resolved,
        "history_frame_ids": history,
        "last_input_frame_id": history[-1],
        "target_frame_id": future[0],
    }


def _frame_ids(row: Mapping[str, Any], key: str) -> list[str]:
    try:
        values = json.loads(str(row.get(key, "")))
    except json.JSONDecodeError as exc:
        raise ValueError(f"MMW sparse pilot row has invalid {key}.") from exc
    if not isinstance(values, list) or not values:
        raise ValueError(f"MMW sparse pilot row has no {key}.")
    return [str(value) for value in values]


def _channel_frame_id(reference: str) -> str:
    stem = Path(reference.replace("\\", "/")).stem
    if not stem.endswith("_paths"):
        raise ValueError(f"Channel reference does not end in _paths.npz: {reference!r}.")
    return stem[: -len("_paths")]


__all__ = ["resolve_input_channel_refs", "resolve_last_input_channel_ref"]
