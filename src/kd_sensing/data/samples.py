import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kd_sensing.data.transform_ops.io import joined_resource


MMW_BEAM_CLASS_COUNT = 64
MMW_RESOURCE_VALIDATION_WORKERS_PER_DOMAIN = 6


@dataclass
class SequenceSamples:
    rgb_paths: list[list[str]]
    radar_paths: list[list[str]]
    gps_paths: list[list[str]] | None = None
    bs_gps_paths: list[list[str]] | None = None
    lidar_paths: list[list[str]] | None = None
    metadata: dict | None = None
    rows: list[dict[str, Any]] | None = None


def create_samples(
    csv_path: str | Path,
    portion: float = 1.0,
    *,
    data_root: str | Path | None = None,
    enabled_modalities: list[str] | tuple[str, ...] | set[str] | None = None,
    seq_len: int | None = None,
    gps_source_seq_len: int | None = None,
    num_pred: int | None = None,
    portion_strategy: str = "even",
    portion_seed: int = 42,
) -> SequenceSamples:
    frame = pd.read_csv(csv_path, na_values="").fillna(-99)
    if frame.empty:
        raise ValueError(f"MMW CSV {Path(csv_path)} has no rows.")
    modalities = tuple(enabled_modalities or ("image", "radar", "gps", "lidar"))
    columns = {
        name: _numbered_columns(frame.columns, name)
        for name in ("camera", "radar", "gps", "bs_gps", "lidar", "future_beam_label")
    }
    row_columns = [
        column
        for column in frame.columns
        if column
        in {
            "condition",
            "town",
            "sensor_scenario",
            "sample_id",
            "target_sample_id",
            "trajectory_group_id",
        }
        or str(column).startswith("future_beam")
    ]
    _validate_columns(
        csv_path,
        modalities,
        columns,
        seq_len=int(seq_len or 1),
        gps_seq_len=int(gps_source_seq_len or seq_len or 1),
        num_pred=int(num_pred or 1),
    )
    _validate_rows(
        csv_path,
        frame,
        modalities=modalities,
        columns=columns,
        data_root=data_root,
    )
    selected, metadata = _select_portion(frame, portion=portion, strategy=portion_strategy, seed=portion_seed)
    return SequenceSamples(
        rgb_paths=selected[columns["camera"]].values.tolist() if "image" in modalities else [],
        radar_paths=selected[columns["radar"]].values.tolist() if "radar" in modalities else [],
        gps_paths=selected[columns["gps"]].values.tolist() if "gps" in modalities else None,
        bs_gps_paths=selected[columns["bs_gps"]].values.tolist() if "gps" in modalities else None,
        lidar_paths=selected[columns["lidar"]].values.tolist() if "lidar" in modalities else None,
        metadata=metadata,
        rows=selected[row_columns].to_dict(orient="records"),
    )


def _numbered_columns(columns, prefix: str) -> list[str]:
    indexed = sorted(
        (
            (int(text[len(prefix) :]), text)
            for column in columns
            if (text := str(column)).startswith(prefix) and text[len(prefix) :].isdigit()
        ),
        key=lambda item: item[0],
    )
    if indexed and [number for number, _ in indexed] != list(range(1, len(indexed) + 1)):
        raise ValueError(f"MMW CSV has non-contiguous {prefix}1..{prefix}N columns.")
    return [name for _, name in indexed]


def _validate_columns(
    csv_path: str | Path,
    modalities: tuple[str, ...],
    columns: dict[str, list[str]],
    *,
    seq_len: int,
    gps_seq_len: int,
    num_pred: int,
) -> None:
    required = {"future_beam_label": num_pred}
    required.update({"camera": seq_len} if "image" in modalities else {})
    required.update({"radar": seq_len} if "radar" in modalities else {})
    required.update({"gps": gps_seq_len, "bs_gps": gps_seq_len} if "gps" in modalities else {})
    required.update({"lidar": seq_len} if "lidar" in modalities else {})
    for name, minimum in required.items():
        if len(columns[name]) < minimum:
            raise ValueError(
                f"MMW CSV {Path(csv_path)} needs at least {minimum} {name}1..{name}N columns, "
                f"found {len(columns[name])}."
            )


def _validate_rows(
    csv_path: str | Path,
    frame: pd.DataFrame,
    *,
    modalities: tuple[str, ...],
    columns: dict[str, list[str]],
    data_root: str | Path | None,
) -> None:
    resource_prefixes = []
    if "image" in modalities:
        resource_prefixes.append("camera")
    if "radar" in modalities:
        resource_prefixes.append("radar")
    if "gps" in modalities:
        resource_prefixes.extend(("gps", "bs_gps"))
    if "lidar" in modalities:
        resource_prefixes.append("lidar")
    resource_columns = [column for prefix in resource_prefixes for column in columns[prefix]]
    label_columns = columns["future_beam_label"]
    validation_tasks: dict[str, tuple[int, str, str]] = {}
    for column in resource_columns:
        texts = frame[column].astype(str).str.strip()
        missing = texts.eq("") | texts.str.lower().isin({"-99", "-99.0", "nan", "none"})
        if bool(missing.any()):
            row_index = missing.index[missing][0]
            raise ValueError(f"MMW CSV {Path(csv_path)} row {row_index} is missing {column}.")
        if column.startswith("radar"):
            invalid_radar = ~texts.str.contains("_RA", regex=False)
            if bool(invalid_radar.any()):
                row_index = invalid_radar.index[invalid_radar][0]
                raise ValueError(
                    f"MMW CSV {Path(csv_path)} row {row_index} {column} must reference an _RA map so its _DA map can be verified."
                )
        if data_root is None:
            continue
        unique = pd.DataFrame({"row_index": frame.index, "value": texts}).drop_duplicates("value", keep="first")
        for row_index, value in unique.itertuples(index=False, name=None):
            validation_tasks.setdefault(value, (row_index, column, value))
            if column.startswith("radar"):
                da_value = value.replace("_RA", "_DA")
                validation_tasks.setdefault(da_value, (row_index, f"{column} (_DA)", da_value))

    if data_root is not None and validation_tasks:
        tasks = list(validation_tasks.values())
        workers = _resource_validation_worker_count(len(tasks))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mmw-resource") as executor:
            list(
                executor.map(
                    lambda task: _validate_resource(data_root, csv_path, task[0], task[1], task[2]),
                    tasks,
                )
            )

    for column in label_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        valid = numeric.notna() & np.isfinite(numeric) & numeric.mod(1).eq(0) & numeric.between(0, MMW_BEAM_CLASS_COUNT - 1)
        invalid = ~valid
        if bool(invalid.any()):
            row_index = invalid.index[invalid][0]
            _validate_beam_label(csv_path, row_index, column, frame.at[row_index, column])


def _resource_validation_worker_count(task_count: int) -> int:
    return max(1, min(task_count, MMW_RESOURCE_VALIDATION_WORKERS_PER_DOMAIN, os.cpu_count() or 1))


def _validate_resource(data_root: str | Path, csv_path: str | Path, row_index: int, column: str, value: str) -> None:
    try:
        path = joined_resource(data_root, value)
    except ValueError as exc:
        raise ValueError(f"MMW CSV {Path(csv_path)} row {row_index} has invalid {column} path {value!r}: {exc}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"MMW CSV {Path(csv_path)} row {row_index} {column} artifact is missing: {path}")


def _validate_beam_label(csv_path: str | Path, row_index: int, column: str, value: object) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MMW CSV {Path(csv_path)} row {row_index} has invalid {column}={value!r}.") from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or not 0 <= numeric < MMW_BEAM_CLASS_COUNT:
        raise ValueError(
            f"MMW CSV {Path(csv_path)} row {row_index} {column} must be an integer in [0, {MMW_BEAM_CLASS_COUNT - 1}], got {value!r}."
        )


def _cell_text(value: object) -> str | None:
    text = str(value).strip()
    return text if text and text.lower() not in {"-99", "-99.0", "nan", "none"} else None


def _select_portion(frame: pd.DataFrame, *, portion: float, strategy: str, seed: int) -> tuple[pd.DataFrame, dict]:
    if portion <= 0:
        raise ValueError(f"portion must be positive, got {portion}.")
    total = len(frame)
    if portion >= 1.0 or total == 0:
        indices = list(range(total))
        effective_strategy = "all"
    else:
        count = min(max(1, int(total * portion)), total)
        effective_strategy = strategy or "even"
        if effective_strategy == "head":
            indices = list(range(count))
        elif effective_strategy == "random":
            indices = sorted(int(index) for index in np.random.default_rng(int(seed)).choice(total, size=count, replace=False))
        elif effective_strategy in {"even", "deterministic_even", "stratified_even"}:
            indices = _even_indices(total, count)
            effective_strategy = "even"
        else:
            raise ValueError(f"Unsupported portion_strategy '{strategy}'.")
    selected = frame.iloc[indices]
    metadata = {
        "total_rows": int(total),
        "selected_rows": int(len(selected)),
        "portion": float(portion),
        "portion_strategy": effective_strategy,
        "portion_seed": int(seed),
    }
    if "seq_index" in selected:
        metadata.update(
            seq_index_min=_safe_scalar(selected["seq_index"].min()),
            seq_index_max=_safe_scalar(selected["seq_index"].max()),
            seq_index_count=int(selected["seq_index"].nunique()),
        )
    return selected, metadata


def _even_indices(total: int, count: int) -> list[int]:
    if count >= total:
        return list(range(total))
    if count == 1:
        return [total // 2]
    indices = sorted({int(round(value)) for value in np.linspace(0, total - 1, count)})
    candidate = 0
    while len(indices) < count:
        if candidate not in indices:
            indices.append(candidate)
        candidate += 1
    return sorted(indices[:count])


def _safe_scalar(value):
    return value.item() if hasattr(value, "item") else value


__all__ = ["SequenceSamples", "create_samples"]
