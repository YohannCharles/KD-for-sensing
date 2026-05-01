from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kd_sensing.data.split_metadata import SPLIT_METADATA_PROTOCOL, default_split_metadata_path
from kd_sensing.registries import PREPROCESSORS
from kd_sensing.utils.paths import resolve_path


@dataclass(frozen=True)
class SequenceColumnPlan:
    base_columns: list[str]
    gps_source_column: str
    bs_gps_column: str
    lidar_source_column: str
    mmwave_source_column: str


@dataclass(frozen=True)
class SequenceSplit:
    train_seq_index: list[Any]
    test_seq_index: list[Any]


def generate_sequence_data(
    csv_path: str | Path,
    data_root: str | Path,
    output_suffix: str,
    in_len: int,
    out_len: int,
    training_set_pct: float = 0.8,
    include_gps: bool = False,
    gps_column: str = "unit2_loc_cal",
    gps_fallback_column: str | None = "unit2_loc",
    bs_gps_column: str = "unit1_loc",
    include_lidar: bool = False,
    lidar_column: str = "unit1_lidar",
    lidar_fallback_column: str | None = "unit1_lidar_SCR",
    include_mmwave: bool = False,
    mmwave_column: str = "unit1_pwr_60ghz",
    mmwave_fallback_column: str | None = None,
    split_strategy: str = SPLIT_METADATA_PROTOCOL,
    split_seed: int = 42,
    min_test_sequences: int | None = None,
    test_sequence_count: int | None = None,
    split_metadata_path: str | Path | None = None,
) -> tuple[Path, Path]:
    csv_path = resolve_path(csv_path)
    data_root = resolve_path(data_root)
    all_data = pd.read_csv(csv_path)
    if split_strategy != SPLIT_METADATA_PROTOCOL:
        raise ValueError(
            f"Unsupported sequence split_strategy '{split_strategy}'. "
            f"This workflow supports only '{SPLIT_METADATA_PROTOCOL}'."
        )
    plan = resolve_sequence_column_plan(
        all_data,
        csv_path=csv_path,
        include_gps=include_gps,
        gps_column=gps_column,
        gps_fallback_column=gps_fallback_column,
        bs_gps_column=bs_gps_column,
        include_lidar=include_lidar,
        lidar_column=lidar_column,
        lidar_fallback_column=lidar_fallback_column,
        include_mmwave=include_mmwave,
        mmwave_column=mmwave_column,
        mmwave_fallback_column=mmwave_fallback_column,
    )
    all_seqs = build_sequence_windows(
        all_data,
        plan,
        in_len=in_len,
        out_len=out_len,
        include_gps=include_gps,
        include_lidar=include_lidar,
        include_mmwave=include_mmwave,
    )
    split = select_balanced_sequence_split(
        all_seqs,
        training_set_pct=training_set_pct,
        split_seed=split_seed,
        min_test_sequences=min_test_sequences,
        test_sequence_count=test_sequence_count,
        data_root=data_root,
    )
    train_path = data_root / f"train_seqs{output_suffix}.csv"
    test_path = data_root / f"test_seqs{output_suffix}.csv"
    data_root.mkdir(parents=True, exist_ok=True)
    train_frame = all_seqs[all_seqs["seq_index"].isin(split.train_seq_index)]
    test_frame = all_seqs[all_seqs["seq_index"].isin(split.test_seq_index)]
    train_frame.to_csv(train_path, index=False)
    test_frame.to_csv(test_path, index=False)
    metadata_path = Path(split_metadata_path) if split_metadata_path is not None else default_split_metadata_path(train_path)
    if not metadata_path.is_absolute():
        metadata_path = data_root / metadata_path
    write_split_metadata(
        metadata_path,
        source_csv_path=csv_path,
        data_root=data_root,
        train_path=train_path,
        test_path=test_path,
        all_windows=all_seqs,
        train_frame=train_frame,
        test_frame=test_frame,
        split=split,
        training_set_pct=training_set_pct,
        split_seed=split_seed,
        min_test_sequences=min_test_sequences,
        requested_test_sequence_count=test_sequence_count,
    )
    return train_path, test_path


def resolve_sequence_column_plan(
    all_data: pd.DataFrame,
    *,
    csv_path: Path,
    include_gps: bool,
    gps_column: str,
    gps_fallback_column: str | None,
    bs_gps_column: str,
    include_lidar: bool,
    lidar_column: str,
    lidar_fallback_column: str | None,
    include_mmwave: bool,
    mmwave_column: str,
    mmwave_fallback_column: str | None,
) -> SequenceColumnPlan:
    base_columns = ["unit1_rgb", "unit1_radar", "unit1_pwr_60ghz", "seq_index"]
    _require_columns(all_data, csv_path, base_columns)
    gps_source_column = gps_column
    lidar_source_column = lidar_column
    mmwave_source_column = mmwave_column
    if include_gps:
        gps_source_column = _resolve_source_column(
            all_data,
            csv_path=csv_path,
            column=gps_column,
            fallback_column=gps_fallback_column,
            label="GPS",
        )
        if bs_gps_column not in all_data.columns:
            raise ValueError(f"BS GPS column '{bs_gps_column}' not found in {csv_path}.")
        base_columns.extend([gps_source_column, bs_gps_column])
    if include_lidar:
        lidar_source_column = _resolve_source_column(
            all_data,
            csv_path=csv_path,
            column=lidar_column,
            fallback_column=lidar_fallback_column,
            label="LiDAR",
        )
        base_columns.append(lidar_source_column)
    if include_mmwave:
        mmwave_source_column = _resolve_source_column(
            all_data,
            csv_path=csv_path,
            column=mmwave_column,
            fallback_column=mmwave_fallback_column,
            label="mmWave",
        )
        if mmwave_source_column not in base_columns:
            base_columns.append(mmwave_source_column)
    return SequenceColumnPlan(
        base_columns=base_columns,
        gps_source_column=gps_source_column,
        bs_gps_column=bs_gps_column,
        lidar_source_column=lidar_source_column,
        mmwave_source_column=mmwave_source_column,
    )


def build_sequence_windows(
    all_data: pd.DataFrame,
    plan: SequenceColumnPlan,
    *,
    in_len: int,
    out_len: int,
    include_gps: bool,
    include_lidar: bool,
    include_mmwave: bool,
) -> pd.DataFrame:
    all_seq_idx = all_data["seq_index"].unique()
    rows = []
    for seq_idx in all_seq_idx:
        seq = all_data.loc[all_data["seq_index"] == seq_idx, plan.base_columns].reset_index(drop=True)
        start = 0
        while start + in_len + out_len <= seq.shape[0]:
            image = seq["unit1_rgb"].iloc[start : start + in_len].tolist()
            radar = seq["unit1_radar"].iloc[start : start + in_len].tolist()
            gps = seq[plan.gps_source_column].iloc[start : start + in_len].tolist() if include_gps else []
            bs_gps = seq[plan.bs_gps_column].iloc[start : start + in_len].tolist() if include_gps else []
            lidar = seq[plan.lidar_source_column].iloc[start : start + in_len].tolist() if include_lidar else []
            mmwave = seq[plan.mmwave_source_column].iloc[start : start + in_len].tolist() if include_mmwave else []
            in_beam = seq["unit1_pwr_60ghz"].iloc[start : start + in_len].tolist()
            out_beam = seq["unit1_pwr_60ghz"].iloc[start + in_len : start + in_len + out_len].tolist()
            rows.append(image + radar + gps + bs_gps + lidar + mmwave + in_beam + out_beam + [seq_idx])
            start += 1
    return pd.DataFrame(
        rows,
        columns=sequence_window_columns(
            in_len,
            out_len,
            include_gps=include_gps,
            include_lidar=include_lidar,
            include_mmwave=include_mmwave,
        ),
    )


def sequence_window_columns(
    in_len: int,
    out_len: int,
    *,
    include_gps: bool,
    include_lidar: bool,
    include_mmwave: bool,
) -> list[str]:
    return (
        [f"camera{i}" for i in range(1, in_len + 1)]
        + [f"radar{i}" for i in range(1, in_len + 1)]
        + ([f"gps{i}" for i in range(1, in_len + 1)] if include_gps else [])
        + ([f"bs_gps{i}" for i in range(1, in_len + 1)] if include_gps else [])
        + ([f"lidar{i}" for i in range(1, in_len + 1)] if include_lidar else [])
        + ([f"mmwave{i}" for i in range(1, in_len + 1)] if include_mmwave else [])
        + [f"beam{i}" for i in range(1, in_len + 1)]
        + [f"future_beam{i}" for i in range(1, out_len + 1)]
        + ["seq_index"]
    )


def select_balanced_sequence_split(
    windows: pd.DataFrame,
    *,
    training_set_pct: float,
    split_seed: int,
    min_test_sequences: int | None = None,
    test_sequence_count: int | None = None,
    data_root: str | Path | None = None,
) -> SequenceSplit:
    if "seq_index" not in windows.columns:
        raise ValueError("Sequence windows must contain a seq_index column.")
    seq_values = _sorted_values(windows["seq_index"].drop_duplicates().tolist())
    test_count = _resolve_test_sequence_count(
        total_sequences=len(seq_values),
        training_set_pct=training_set_pct,
        min_test_sequences=min_test_sequences,
        test_sequence_count=test_sequence_count,
    )
    if test_count == 0:
        return SequenceSplit(train_seq_index=seq_values, test_seq_index=[])

    stats = _sequence_split_stats(windows, seq_values, data_root=data_root)
    target_ratio = max(0.0, min(1.0, 1.0 - float(training_set_pct)))
    if test_sequence_count is not None and target_ratio == 0.0:
        target_ratio = test_count / len(seq_values)
    rng = np.random.default_rng(int(split_seed))
    tie_order = list(seq_values)
    rng.shuffle(tie_order)
    tie_rank = {seq_idx: rank for rank, seq_idx in enumerate(tie_order)}
    selected: list[Any] = []
    selected_windows = 0
    selected_counts: Counter = Counter()

    while len(selected) < test_count:
        remaining = [seq_idx for seq_idx in seq_values if seq_idx not in selected]
        best_seq = min(
            remaining,
            key=lambda seq_idx: (
                round(
                    _candidate_split_score(
                        seq_idx,
                        stats=stats,
                        selected_windows=selected_windows,
                        selected_counts=selected_counts,
                        target_ratio=target_ratio,
                    ),
                    12,
                ),
                tie_rank[seq_idx],
            ),
        )
        selected.append(best_seq)
        selected_windows += stats["per_seq"][best_seq]["window_count"]
        selected_counts.update(stats["per_seq"][best_seq]["label_counts"])

    test_seq = _sorted_values(selected)
    test_set = set(test_seq)
    train_seq = [seq_idx for seq_idx in seq_values if seq_idx not in test_set]
    return SequenceSplit(train_seq_index=train_seq, test_seq_index=test_seq)


def write_split_metadata(
    metadata_path: str | Path,
    *,
    source_csv_path: Path,
    data_root: Path,
    train_path: Path,
    test_path: Path,
    all_windows: pd.DataFrame,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    split: SequenceSplit,
    training_set_pct: float,
    split_seed: int,
    min_test_sequences: int | None,
    requested_test_sequence_count: int | None,
) -> Path:
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    label_distribution = {
        "all": label_distribution_summary(all_windows, data_root=data_root),
        "train": label_distribution_summary(train_frame, data_root=data_root),
        "test": label_distribution_summary(test_frame, data_root=data_root),
    }
    payload = {
        "split_protocol": SPLIT_METADATA_PROTOCOL,
        "split_seed": int(split_seed),
        "training_set_pct": float(training_set_pct),
        "min_test_sequences": None if min_test_sequences is None else int(min_test_sequences),
        "requested_test_sequence_count": (
            None if requested_test_sequence_count is None else int(requested_test_sequence_count)
        ),
        "source_csv_path": str(source_csv_path),
        "data_root": str(data_root),
        "output_csv_paths": {
            "train": str(train_path),
            "test": str(test_path),
        },
        "sequence_counts": {
            "total": int(len(split.train_seq_index) + len(split.test_seq_index)),
            "train": int(len(split.train_seq_index)),
            "test": int(len(split.test_seq_index)),
        },
        "window_counts": {
            "total": int(len(all_windows)),
            "train": int(len(train_frame)),
            "test": int(len(test_frame)),
        },
        "seq_index": {
            "train": _json_ready(split.train_seq_index),
            "test": _json_ready(split.test_seq_index),
        },
        "label_distribution": label_distribution,
        "splits": {
            "train": {
                "csv_path": str(train_path),
                "num_samples": int(len(train_frame)),
                "sequence_count": int(len(split.train_seq_index)),
                "seq_index": _json_ready(split.train_seq_index),
                "label_distribution": label_distribution["train"],
            },
            "test": {
                "csv_path": str(test_path),
                "num_samples": int(len(test_frame)),
                "sequence_count": int(len(split.test_seq_index)),
                "seq_index": _json_ready(split.test_seq_index),
                "label_distribution": label_distribution["test"],
            },
        },
    }
    metadata_path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    return metadata_path


def label_distribution_summary(frame: pd.DataFrame, *, data_root: str | Path | None = None) -> dict[str, Any]:
    if frame.empty:
        return {"columns": {}, "num_samples": 0}
    cache: dict[str, Any] = {}
    columns = [column for column in frame.columns if column.startswith("future_beam")]
    beam_columns = [column for column in frame.columns if column.startswith("beam")]
    if beam_columns:
        columns.insert(0, beam_columns[-1])
    summary = {}
    for column in columns:
        labels = [_label_key(value, data_root=data_root, cache=cache) for value in frame[column].tolist()]
        counts = Counter(labels)
        summary[column] = {
            "total": int(len(labels)),
            "counts": {str(label): int(count) for label, count in sorted(counts.items(), key=lambda item: str(item[0]))},
            "top": [
                {"label": str(label), "count": int(count)}
                for label, count in counts.most_common(10)
            ],
        }
    return {"columns": summary, "num_samples": int(len(frame))}


def _sequence_split_stats(
    windows: pd.DataFrame,
    seq_values: list[Any],
    *,
    data_root: str | Path | None,
) -> dict[str, Any]:
    label_column = "future_beam1" if "future_beam1" in windows.columns else None
    cache: dict[str, Any] = {}
    per_seq = {}
    total_counts: Counter = Counter()
    for seq_idx in seq_values:
        seq_frame = windows[windows["seq_index"] == seq_idx]
        label_counts: Counter = Counter()
        if label_column is not None:
            label_counts.update(_label_key(value, data_root=data_root, cache=cache) for value in seq_frame[label_column])
        per_seq[seq_idx] = {
            "window_count": int(len(seq_frame)),
            "label_counts": label_counts,
        }
        total_counts.update(label_counts)
    return {
        "per_seq": per_seq,
        "total_windows": int(len(windows)),
        "total_counts": total_counts,
    }


def _candidate_split_score(
    seq_idx: Any,
    *,
    stats: dict[str, Any],
    selected_windows: int,
    selected_counts: Counter,
    target_ratio: float,
) -> float:
    candidate = stats["per_seq"][seq_idx]
    candidate_windows = selected_windows + candidate["window_count"]
    total_windows = max(1, stats["total_windows"])
    window_error = abs((candidate_windows / total_windows) - target_ratio)
    candidate_counts = selected_counts + candidate["label_counts"]
    label_error = _distribution_distance(candidate_counts, stats["total_counts"])
    return window_error + label_error


def _distribution_distance(candidate_counts: Counter, target_counts: Counter) -> float:
    candidate_total = sum(candidate_counts.values())
    target_total = sum(target_counts.values())
    if candidate_total == 0 or target_total == 0:
        return 0.0
    labels = set(candidate_counts) | set(target_counts)
    return sum(
        abs(candidate_counts.get(label, 0) / candidate_total - target_counts.get(label, 0) / target_total)
        for label in labels
    ) / 2.0


def _resolve_test_sequence_count(
    *,
    total_sequences: int,
    training_set_pct: float,
    min_test_sequences: int | None,
    test_sequence_count: int | None,
) -> int:
    training_pct = float(training_set_pct)
    if training_pct <= 0.0 or training_pct > 1.0:
        raise ValueError(f"training_set_pct must be in the range (0, 1], got {training_set_pct}.")
    if total_sequences <= 1:
        requested = [value for value in (min_test_sequences, test_sequence_count) if value not in (None, 0)]
        if requested:
            raise ValueError("At least two seq_index values are required to create a non-empty test split.")
        return 0
    max_test = total_sequences - 1
    min_count = 0 if min_test_sequences is None else int(min_test_sequences)
    if min_count < 0:
        raise ValueError(f"min_test_sequences must be non-negative, got {min_test_sequences}.")
    if min_count > max_test:
        raise ValueError(
            f"min_test_sequences={min_count} would leave no training sequences; "
            f"maximum allowed for {total_sequences} sequences is {max_test}."
        )
    if test_sequence_count is not None:
        requested_count = int(test_sequence_count)
        if requested_count < 0:
            raise ValueError(f"test_sequence_count must be non-negative, got {test_sequence_count}.")
        if requested_count > max_test:
            raise ValueError(
                f"test_sequence_count={requested_count} would leave no training sequences; "
                f"maximum allowed for {total_sequences} sequences is {max_test}."
            )
        if requested_count < min_count:
            raise ValueError(
                f"test_sequence_count={requested_count} conflicts with min_test_sequences={min_count}."
            )
        return requested_count

    test_ratio = 1.0 - training_pct
    if test_ratio <= 0.0:
        derived_count = 0
    else:
        derived_count = max(1, int(round(total_sequences * test_ratio)))
    return min(max(derived_count, min_count), max_test)


def _resolve_source_column(
    all_data: pd.DataFrame,
    *,
    csv_path: Path,
    column: str,
    fallback_column: str | None,
    label: str,
) -> str:
    if column in all_data.columns:
        return column
    if fallback_column and fallback_column in all_data.columns:
        return fallback_column
    raise ValueError(
        f"{label} column '{column}' not found in {csv_path}; "
        f"fallback '{fallback_column}' is also unavailable."
    )


def _require_columns(all_data: pd.DataFrame, csv_path: Path, columns: list[str]) -> None:
    missing = [column for column in columns if column not in all_data.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}.")


def _label_key(value: Any, *, data_root: str | Path | None, cache: dict[str, Any]) -> Any:
    key = str(value)
    if key in cache:
        return cache[key]
    if key.strip() in {"", "-99", "nan", "None"}:
        cache[key] = "missing"
        return cache[key]
    path = Path(key)
    if data_root is not None and not path.is_absolute():
        path = Path(data_root) / key.lstrip("/")
    try:
        values = np.loadtxt(path)
        values = np.asarray(values)
        if values.size == 0:
            label: Any = "empty"
        else:
            label = int(np.argmax(values))
    except Exception:
        label = f"path:{key}"
    cache[key] = label
    return label


def _sorted_values(values: list[Any]) -> list[Any]:
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda value: str(value))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(_json_ready(key)): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


@PREPROCESSORS.register("sequence_csv")
class SequencePreprocessor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self):
        return generate_sequence_data(**self.kwargs)
