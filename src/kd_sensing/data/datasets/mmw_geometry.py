import json
from typing import Any

import pandas as pd
import torch

from kd_sensing.data.datasets.mmw_columns import _numbered_columns


def _parse_geometry_cell(value: Any) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}

def _parse_availability_cell(value: Any) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}

def _float_or_zero(value: Any) -> tuple[float, bool]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0, False
    if not torch.isfinite(torch.tensor(numeric)):
        return 0.0, False
    return numeric, True

def _numbered_json_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return _numbered_columns(frame.columns, prefix)

def _value_for_field(payload: dict[str, Any], field: str) -> tuple[float, bool]:
    if not payload.get("available", False):
        return 0.0, False
    return _float_or_zero(payload.get(field))

def _empty_geometry(seq_len: int, field_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((seq_len, field_count), dtype=torch.float32),
        torch.zeros((seq_len, field_count), dtype=torch.bool),
    )

def _row_at(frame: pd.DataFrame, idx: int):
    if idx < 0 or idx >= len(frame):
        return None
    return frame.iloc[idx]

def _row_first(row, keys: tuple[str, ...]) -> str:
    if row is None:
        return ""
    for key in keys:
        if key in row:
            text = str(row[key]).strip()
            if text and text != "-99":
                return text
    return ""

def _availability_json_from_row(row, seq_len: int) -> dict[str, Any]:
    values = {}
    for idx in range(1, seq_len + 1):
        key = f"modality_availability{idx}"
        if key in row:
            values[str(idx)] = _parse_availability_cell(row[key])
    return values

def _geometry_json_from_row(row, seq_len: int) -> list[dict[str, Any]]:
    values = []
    for idx in range(1, seq_len + 1):
        key = f"geometry{idx}"
        values.append(_parse_geometry_cell(row[key]) if key in row else {})
    return values

def _geometry_tensor_from_payloads(
    payloads: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not payloads:
        return _empty_geometry(0, len(fields))
    values = torch.zeros((len(payloads), len(fields)), dtype=torch.float32)
    mask = torch.zeros((len(payloads), len(fields)), dtype=torch.bool)
    for row_idx, payload in enumerate(payloads):
        for col_idx, field in enumerate(fields):
            numeric, ok = _value_for_field(payload, field)
            values[row_idx, col_idx] = float(numeric)
            mask[row_idx, col_idx] = bool(ok)
    return values, mask


__all__ = [
    "_availability_json_from_row",
    "_geometry_json_from_row",
    "_geometry_tensor_from_payloads",
    "_parse_availability_cell",
    "_parse_geometry_cell",
]
