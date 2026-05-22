from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from kd_sensing.diagnostics.visualization.config import _json_scalar

@dataclass(frozen=True)
class SampleCandidate:
    dataset_index: int
    csv_row_index: int
    seq_index: Any
    future_label: int | None

def collect_candidates(dataset: Any, csv_frame: Any) -> list[SampleCandidate]:
    future_cols = _sorted_numbered_columns(csv_frame.columns, "future_beam")
    candidates = []
    for dataset_index, (csv_row_index, row) in enumerate(csv_frame.iterrows()):
        future_label = None
        if future_cols:
            path = row[future_cols[0]]
            if str(path).strip() != "-99":
                future_label = int(dataset._beam_label(str(path)))
        candidates.append(
            SampleCandidate(
                dataset_index=dataset_index,
                csv_row_index=int(csv_row_index),
                seq_index=_json_scalar(row["seq_index"]) if "seq_index" in row else None,
                future_label=future_label,
            )
        )
    return candidates

def select_sample_candidates(
    candidates: Iterable[SampleCandidate],
    *,
    sample_count: int,
    per_seq_sample_count: int | None = None,
    seed: int,
    seq_index: tuple[Any, ...] | None = None,
    labels: tuple[int, ...] | None = None,
) -> tuple[list[SampleCandidate], dict[str, Any]]:
    all_candidates = list(candidates)
    filtered = filter_sample_candidates(all_candidates, seq_index=seq_index, labels=labels)
    if per_seq_sample_count is not None:
        selected, by_seq_index = _select_per_seq_candidates(
            filtered,
            per_seq_sample_count=int(per_seq_sample_count),
            seed=int(seed),
        )
        requested = int(per_seq_sample_count) * len(by_seq_index)
        return selected, {
            "seed": int(seed),
            "requested_count": requested,
            "per_seq_sample_count": int(per_seq_sample_count),
            "candidate_count": len(filtered),
            "actual_count": len(selected),
            "seq_index_filter": list(seq_index) if seq_index is not None else None,
            "label_filter": list(labels) if labels is not None else None,
            "by_seq_index": by_seq_index,
            "selected_dataset_indices": [candidate.dataset_index for candidate in selected],
            "selected_csv_row_indices": [candidate.csv_row_index for candidate in selected],
        }

    requested = int(sample_count)
    if requested >= len(filtered):
        selected = list(filtered)
    elif requested == 0:
        selected = []
    else:
        rng = np.random.default_rng(int(seed))
        positions = rng.choice(len(filtered), size=requested, replace=False)
        selected = [filtered[int(pos)] for pos in positions]

    return selected, {
        "seed": int(seed),
        "requested_count": requested,
        "per_seq_sample_count": None,
        "candidate_count": len(filtered),
        "actual_count": len(selected),
        "seq_index_filter": list(seq_index) if seq_index is not None else None,
        "label_filter": list(labels) if labels is not None else None,
        "selected_dataset_indices": [candidate.dataset_index for candidate in selected],
        "selected_csv_row_indices": [candidate.csv_row_index for candidate in selected],
    }

def filter_sample_candidates(
    candidates: Iterable[SampleCandidate],
    *,
    seq_index: tuple[Any, ...] | None = None,
    labels: tuple[int, ...] | None = None,
) -> list[SampleCandidate]:
    seq_filter = {_filter_key(value) for value in seq_index} if seq_index is not None else None
    label_filter = {int(value) for value in labels} if labels is not None else None
    return [
        candidate
        for candidate in candidates
        if (seq_filter is None or _filter_key(candidate.seq_index) in seq_filter)
        and (label_filter is None or candidate.future_label in label_filter)
    ]

def _select_per_seq_candidates(
    candidates: list[SampleCandidate],
    *,
    per_seq_sample_count: int,
    seed: int,
) -> tuple[list[SampleCandidate], dict[str, Any]]:
    groups: dict[str, list[SampleCandidate]] = {}
    seq_values: dict[str, Any] = {}
    for candidate in candidates:
        key = _filter_key(candidate.seq_index)
        groups.setdefault(key, []).append(candidate)
        seq_values.setdefault(key, candidate.seq_index)

    rng = np.random.default_rng(int(seed))
    selected: list[SampleCandidate] = []
    summary: dict[str, Any] = {}
    for key, group in groups.items():
        if per_seq_sample_count >= len(group):
            chosen = list(group)
        elif per_seq_sample_count == 0:
            chosen = []
        else:
            positions = rng.choice(len(group), size=per_seq_sample_count, replace=False)
            chosen = [group[int(pos)] for pos in positions]
        selected.extend(chosen)
        summary[key] = {
            "seq_index": seq_values[key],
            "requested_count": int(per_seq_sample_count),
            "candidate_count": len(group),
            "actual_count": len(chosen),
            "selected_dataset_indices": [candidate.dataset_index for candidate in chosen],
            "selected_csv_row_indices": [candidate.csv_row_index for candidate in chosen],
        }
    return selected, summary

def _sorted_numbered_columns(columns: Iterable[str], prefix: str) -> list[str]:
    selected = []
    for col in columns:
        if not str(col).startswith(prefix):
            continue
        suffix = str(col)[len(prefix) :]
        if suffix.isdigit():
            selected.append(str(col))
    return sorted(selected, key=lambda name: int(name[len(prefix) :]))

def _filter_key(value: Any) -> str:
    return str(_json_scalar(value))

__all__ = [
    'SampleCandidate',
    'collect_candidates',
    'filter_sample_candidates',
    'select_sample_candidates',
]
