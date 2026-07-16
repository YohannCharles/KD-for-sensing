from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
    enabled_modalities: list[str] | tuple[str, ...] | set[str] | None = None,
    seq_len: int | None = None,
    gps_source_seq_len: int | None = None,
    num_pred: int | None = None,
    portion_strategy: str = "even",
    portion_seed: int = 42,
) -> SequenceSamples:
    frame = pd.read_csv(csv_path, na_values="").fillna(-99)
    selected, metadata = _select_portion(frame, portion=portion, strategy=portion_strategy, seed=portion_seed)
    modalities = tuple(enabled_modalities or ("image", "radar", "gps", "lidar"))
    columns = {
        name: _numbered_columns(frame.columns, name)
        for name in ("camera", "radar", "gps", "bs_gps", "lidar", "future_beam_label")
    }
    row_columns = [
        column
        for column in selected.columns
        if column in {"condition", "town", "sensor_scenario", "sample_id", "target_sample_id"}
        or str(column).startswith("future_beam_label")
    ]
    _validate_columns(
        csv_path,
        modalities,
        columns,
        seq_len=int(seq_len or 1),
        gps_seq_len=int(gps_source_seq_len or seq_len or 1),
        num_pred=int(num_pred or 1),
    )
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
    return sorted(
        (str(column) for column in columns if str(column).startswith(prefix) and str(column)[len(prefix) :].isdigit()),
        key=lambda name: int(name[len(prefix) :]),
    )


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
