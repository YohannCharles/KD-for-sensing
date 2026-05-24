from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

def _load_external_mapping(source: str | Path | dict[str, Any] | list[Any] | None) -> dict[str, Any]:
    if source is None:
        return {}
    payload: Any
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    else:
        payload = source
    if isinstance(payload, dict):
        if isinstance(payload.get("samples"), list):
            return _mapping_from_records(payload["samples"])
        return {str(key): value for key, value in payload.items()}
    if isinstance(payload, list):
        return _mapping_from_records(payload)
    return {}


def _mapping_from_records(records: Iterable[Any]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        keys = [record.get("sample_id"), record.get("dataset_index"), index]
        for key in keys:
            if key is not None:
                mapping[str(key)] = record
    return mapping


def _attach_optional(record: dict[str, Any], field: str, mapping: dict[str, Any], sample_id: str, dataset_index: int) -> None:
    value = mapping.get(sample_id, mapping.get(str(dataset_index)))
    if isinstance(value, dict) and field in value:
        record[field] = value[field]
    elif value is not None:
        record[field] = value


def _attach_prediction_bundle(
    record: dict[str, Any],
    mapping: dict[str, Any],
    sample_id: str,
    dataset_index: int,
) -> None:
    value = mapping.get(sample_id, mapping.get(str(dataset_index)))
    if value is None:
        return
    if isinstance(value, dict):
        record["prediction"] = value.get("prediction", value)
        for field in ("confidence", "confidence_curves", "beam_distribution"):
            if field in value:
                record[field] = value[field]
        return
    record["prediction"] = value


__all__ = ["_attach_optional", "_attach_prediction_bundle", "_load_external_mapping"]
