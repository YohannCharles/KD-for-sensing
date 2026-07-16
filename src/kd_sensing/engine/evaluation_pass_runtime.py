from typing import Any, Mapping

import torch

from kd_sensing.engine.runtime import prepare_task_batch


def prepare_evaluation_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    return prepare_task_batch(batch)


def metadata_rows_from_batch(metadata: Any) -> list[dict[str, Any]]:
    if metadata is None:
        return []
    if isinstance(metadata, list):
        return [dict(item) for item in metadata if isinstance(item, dict)]
    if not isinstance(metadata, dict):
        return []
    size = _metadata_batch_size(metadata)
    return [
        {key: _metadata_value_at(value, index, batch_size=size) for key, value in metadata.items()}
        for index in range(size)
    ]


def sample_ids_from_batch(batch: Mapping[str, Any]) -> list[str]:
    rows = metadata_rows_from_batch(batch.get("metadata"))
    ids = [
        str(row.get("stable_sample_id") or row["sample_id"])
        for row in rows
        if (row.get("stable_sample_id") or row.get("sample_id")) not in (None, "")
    ]
    if ids:
        return ids
    value = batch.get("sample_id")
    if value is None:
        return []
    if torch.is_tensor(value):
        return [str(item.item()) for item in value.reshape(-1)]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _metadata_batch_size(metadata: Mapping[str, Any]) -> int:
    for value in metadata.values():
        size = _metadata_batch_length(value)
        if size:
            return size
    return 1


def _metadata_batch_length(value: Any) -> int:
    if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0:
        return int(value.shape[0])
    return len(value) if isinstance(value, (list, tuple)) else 0


def _metadata_value_at(value: Any, index: int, *, batch_size: int) -> Any:
    if isinstance(value, dict):
        return {key: _metadata_value_at(item, index, batch_size=batch_size) for key, item in value.items()}
    if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0:
        item = value[index] if int(value.shape[0]) == batch_size else value
        return item.item() if hasattr(item, "item") else item.tolist() if hasattr(item, "tolist") else item
    if isinstance(value, (list, tuple)):
        return value[index] if len(value) == batch_size else list(value)
    return value


__all__ = ["metadata_rows_from_batch", "prepare_evaluation_batch", "sample_ids_from_batch"]
