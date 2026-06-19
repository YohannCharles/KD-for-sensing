from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kd_sensing.data.split_metadata import SPLIT_METADATA_PROTOCOL
from kd_sensing.preprocessing.sequence_splits import SequenceSplit, label_key


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
    protocol: str = SPLIT_METADATA_PROTOCOL,
    eval_name: str = "test",
    in_len: int | None = None,
    out_len: int | None = None,
    enabled_columns: list[str] | None = None,
    include_position_targets: bool = False,
) -> Path:
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    label_distribution = {
        "all": label_distribution_summary(all_windows, data_root=data_root),
        "train": label_distribution_summary(train_frame, data_root=data_root),
        eval_name: label_distribution_summary(test_frame, data_root=data_root),
    }
    payload = {
        "split_protocol": protocol,
        "in_len": None if in_len is None else int(in_len),
        "out_len": None if out_len is None else int(out_len),
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
            eval_name: str(test_path),
        },
        "enabled_columns": list(enabled_columns or all_windows.columns),
        "include_position_targets": bool(include_position_targets),
        "sequence_counts": {
            "total": int(len(split.train_seq_index) + len(split.test_seq_index)),
            "train": int(len(split.train_seq_index)),
            eval_name: int(len(split.test_seq_index)),
        },
        "window_counts": {
            "total": int(len(all_windows)),
            "train": int(len(train_frame)),
            eval_name: int(len(test_frame)),
        },
        "seq_index": {
            "train": json_ready(split.train_seq_index),
            eval_name: json_ready(split.test_seq_index),
        },
        "label_distribution": label_distribution,
        "splits": {
            "train": {
                "csv_path": str(train_path),
                "num_samples": int(len(train_frame)),
                "sequence_count": int(len(split.train_seq_index)),
                "seq_index": json_ready(split.train_seq_index),
                "label_distribution": label_distribution["train"],
            },
            eval_name: {
                "csv_path": str(test_path),
                "num_samples": int(len(test_frame)),
                "sequence_count": int(len(split.test_seq_index)),
                "seq_index": json_ready(split.test_seq_index),
                "label_distribution": label_distribution[eval_name],
            },
        },
    }
    if eval_name == "validation":
        payload["output_csv_paths"]["val"] = str(test_path)
        payload["sequence_counts"]["val"] = int(len(split.test_seq_index))
        payload["window_counts"]["val"] = int(len(test_frame))
        payload["seq_index"]["val"] = json_ready(split.test_seq_index)
        payload["label_distribution"]["val"] = label_distribution[eval_name]
        payload["splits"]["val"] = dict(payload["splits"][eval_name])
    else:
        payload["output_csv_paths"]["test"] = str(test_path)
        payload["sequence_counts"]["test"] = int(len(split.test_seq_index))
        payload["window_counts"]["test"] = int(len(test_frame))
        payload["seq_index"]["test"] = json_ready(split.test_seq_index)
    metadata_path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
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
        labels = [label_key(value, data_root=data_root, cache=cache) for value in frame[column].tolist()]
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


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(json_ready(key)): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "json_ready",
    "label_distribution_summary",
    "write_split_metadata",
]
