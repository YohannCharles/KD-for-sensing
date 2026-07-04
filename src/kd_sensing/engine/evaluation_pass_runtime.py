from typing import Any, Mapping

import torch

from kd_sensing.data.difficulty import DifficultyContext, apply_configured_difficulty
from kd_sensing.engine.runtime import prepare_task_batch


def prepare_evaluation_batch(
    batch: Mapping[str, Any],
    *,
    cfg: dict[str, Any],
    split_name: str,
    difficulty_seed: int,
    step_index: int,
) -> dict[str, Any]:
    prepared = prepare_task_batch(batch)
    return apply_evaluation_difficulty(prepared, cfg, split_name, difficulty_seed, step_index)


def metadata_rows_from_batch(metadata: Any) -> list[dict[str, Any]]:
    if metadata is None:
        return []
    if isinstance(metadata, list):
        return [dict(item) for item in metadata if isinstance(item, dict)]
    if not isinstance(metadata, dict):
        return []
    length = _metadata_batch_size(metadata)
    rows: list[dict[str, Any]] = []
    for index in range(length):
        row = {}
        for key, value in metadata.items():
            row[key] = _metadata_value_at(value, index, batch_size=length)
        rows.append(row)
    return rows


def _metadata_batch_size(metadata: dict[str, Any]) -> int:
    for key in ("dataset_index", "sample_id", "target_beam_path", "input_beam_path"):
        if key in metadata:
            length = _metadata_batch_length(metadata[key])
            if length > 0:
                return length
    length = 0
    for value in metadata.values():
        length = max(length, _metadata_batch_length(value))
    return max(length, 1)


def _metadata_batch_length(value: Any) -> int:
    if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0:
        return int(value.shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    return 0


def _metadata_value_at(value: Any, index: int, *, batch_size: int) -> Any:
    if isinstance(value, dict):
        return {key: _metadata_value_at(item, index, batch_size=batch_size) for key, item in value.items()}
    if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0:
        if int(value.shape[0]) == batch_size:
            item = value[index]
            return item.item() if hasattr(item, "item") else item
        return value.tolist() if hasattr(value, "tolist") else value
    if isinstance(value, (list, tuple)):
        if len(value) == batch_size:
            return value[index]
        if value and all(_metadata_batch_length(item) == batch_size for item in value):
            return [_metadata_value_at(item, index, batch_size=batch_size) for item in value]
        return list(value)
    return value


def evaluation_split_name(dataloader, cfg: Mapping[str, Any]) -> str:
    dataset = getattr(dataloader, "dataset", None)
    split = getattr(dataset, "split", None)
    if split:
        return str(split)
    configured = cfg.get("evaluation", {}).get("split") if isinstance(cfg.get("evaluation"), Mapping) else None
    return str(configured or cfg.get("protocol", {}).get("split", "test"))


def apply_evaluation_difficulty(
    batch: dict[str, Any],
    cfg: Mapping[str, Any],
    split_name: str,
    seed: int,
    step_index: int,
) -> dict[str, Any]:
    return apply_configured_difficulty(
        batch,
        cfg,
        DifficultyContext(
            stage="evaluation",
            split=split_name,
            seed=seed,
            step=step_index,
            sample_ids=tuple(sample_ids_from_batch(batch)),
        ),
    ).batch


def sample_ids_from_batch(batch: Mapping[str, Any]) -> list[str]:
    rows = metadata_rows_from_batch(batch.get("metadata"))
    ids = [str(row.get("sample_id", "")) for row in rows if row.get("sample_id") not in (None, "")]
    if ids:
        return ids
    value = batch.get("sample_id", batch.get("sample_ids"))
    if value is None:
        return []
    if torch.is_tensor(value):
        return [str(item.item() if hasattr(item, "item") else item) for item in value.reshape(-1)]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


__all__ = [
    "apply_evaluation_difficulty",
    "evaluation_split_name",
    "metadata_rows_from_batch",
    "prepare_evaluation_batch",
    "sample_ids_from_batch",
]
