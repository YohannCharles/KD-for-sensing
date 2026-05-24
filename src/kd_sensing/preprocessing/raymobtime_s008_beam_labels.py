from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kd_sensing.preprocessing.raymobtime_s008_common import (
    SOURCE_SPLIT_ORDER,
    _load_np_arrays,
    _load_split_arrays,
    _los_series_to_float,
    _row_selector_for_array,
    _split_index_files,
    resolve_raymobtime_paths,
)
from kd_sensing.preprocessing.raymobtime_s008_index import build_s008_index

def normalize_beam_labels(
    values: np.ndarray,
    *,
    num_tx_beams: int | None = None,
    num_rx_beams: int | None = None,
) -> dict[str, np.ndarray | int]:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim == 1:
        labels = array.astype(np.int64)
        rx_count = int(num_rx_beams or 1)
        tx_count = int(num_tx_beams or (int(labels.max(initial=0)) // max(rx_count, 1) + 1))
        beam_tx = labels // max(rx_count, 1)
        beam_rx = labels % max(rx_count, 1)
        num_classes = int(max(int(labels.max(initial=0)) + 1, tx_count * rx_count))
    elif array.ndim == 2 and int(array.shape[1]) == 2:
        pair = array.astype(np.int64)
        beam_tx = pair[:, 0]
        beam_rx = pair[:, 1]
        tx_count = int(num_tx_beams or int(beam_tx.max(initial=0)) + 1)
        rx_count = int(num_rx_beams or int(beam_rx.max(initial=0)) + 1)
        labels = beam_tx * max(rx_count, 1) + beam_rx
        num_classes = int(tx_count * rx_count)
    elif array.ndim == 3:
        scores = np.abs(array) if np.iscomplexobj(array) else array
        n, dim0, dim1 = scores.shape
        if num_tx_beams is not None and num_rx_beams is not None:
            expected = (int(num_tx_beams), int(num_rx_beams))
            reversed_expected = (int(num_rx_beams), int(num_tx_beams))
            if (dim0, dim1) == expected:
                tx_count, rx_count = expected
            elif (dim0, dim1) == reversed_expected:
                scores = np.swapaxes(scores, 1, 2)
                tx_count, rx_count = expected
            else:
                raise ValueError(
                    "Raymobtime beam score matrix shape does not match configured beam dimensions: "
                    f"got [N, {dim0}, {dim1}], expected [N, {expected[0]}, {expected[1]}] "
                    f"or [N, {reversed_expected[0]}, {reversed_expected[1]}]."
                )
        else:
            tx_count = int(num_tx_beams or dim0)
            rx_count = int(num_rx_beams or dim1)
        flat = scores.reshape(n, -1)
        labels = np.argmax(flat, axis=1).astype(np.int64)
        beam_tx = labels // max(rx_count, 1)
        beam_rx = labels % max(rx_count, 1)
        num_classes = int(tx_count * rx_count)
    else:
        raise ValueError(
            "Raymobtime beam_output must have shape [N], [N, 2], or [N, Tx, Rx]; "
            f"got {tuple(array.shape)}."
        )
    return {
        "beam_label": labels.astype(np.int64),
        "beam_tx": beam_tx.astype(np.int64),
        "beam_rx": beam_rx.astype(np.int64),
        "num_beam_classes": int(num_classes),
        "num_tx_beams": int(tx_count),
        "num_rx_beams": int(rx_count),
    }


def build_s008_labels(
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    beam_key: str | None = None,
    num_tx_beams: int | None = 32,
    num_rx_beams: int | None = 8,
) -> dict[str, Any]:
    paths = resolve_raymobtime_paths(data_root=data_root, output_dir=output_dir, cache_dir=cache_dir)
    if not (paths.cache_dir / "index_all_valid.csv").exists():
        build_s008_index(data_root=paths.data_root, cache_dir=paths.cache_dir)
    valid_index = pd.read_csv(paths.cache_dir / "index_all_valid.csv")
    beam_arrays = _load_split_arrays(paths.data_root / "baseline_data" / "beam_output", key=beam_key)
    if "all" in beam_arrays:
        normalized_all = normalize_beam_labels(
            beam_arrays["all"],
            num_tx_beams=num_tx_beams,
            num_rx_beams=num_rx_beams,
        )
        normalized_by_source: dict[str, dict[str, np.ndarray | int]] | None = None
        labels = np.asarray(normalized_all["beam_label"])
        row_selector = _row_selector_for_array(labels, valid_index)
        metadata = _beam_metadata(normalized_all)
    else:
        normalized_all = None
        normalized_by_source = {
            split: normalize_beam_labels(array, num_tx_beams=num_tx_beams, num_rx_beams=num_rx_beams)
            for split, array in beam_arrays.items()
        }
        row_selector = None
        metadata = _merged_beam_metadata(normalized_by_source)
    output_paths: dict[str, Any] = {}
    for split, split_file in _split_index_files(paths.cache_dir).items():
        frame = pd.read_csv(split_file)
        if normalized_by_source is None:
            assert normalized_all is not None and row_selector is not None
            rows = row_selector(frame)
            beam_label = np.asarray(normalized_all["beam_label"])[rows]
            beam_tx = np.asarray(normalized_all["beam_tx"])[rows]
            beam_rx = np.asarray(normalized_all["beam_rx"])[rows]
        else:
            beam_label = _values_for_source_frame(frame, normalized_by_source, "beam_label")
            beam_tx = _values_for_source_frame(frame, normalized_by_source, "beam_tx")
            beam_rx = _values_for_source_frame(frame, normalized_by_source, "beam_rx")
        path = paths.cache_dir / f"labels_{split}.npz"
        np.savez(
            path,
            sample_id=frame["sample_id"].astype(str).to_numpy(),
            valid_index=frame["valid_index"].to_numpy(dtype=np.int64),
            beam_label=beam_label.astype(np.int64),
            beam_tx=beam_tx.astype(np.int64),
            beam_rx=beam_rx.astype(np.int64),
            los_label=_los_series_to_float(frame["LOS"]),
            **metadata,
        )
        output_paths[f"labels_{split}"] = str(path)
    output_paths["beam_metadata"] = metadata
    return output_paths



def _beam_metadata(normalized: dict[str, np.ndarray | int]) -> dict[str, int]:
    return {
        "num_beam_classes": int(normalized["num_beam_classes"]),
        "num_tx_beams": int(normalized["num_tx_beams"]),
        "num_rx_beams": int(normalized["num_rx_beams"]),
    }


def _merged_beam_metadata(normalized_by_source: dict[str, dict[str, np.ndarray | int]]) -> dict[str, int]:
    metadata_values = [_beam_metadata(value) for value in normalized_by_source.values()]
    if not metadata_values:
        raise ValueError("Raymobtime beam output contains no split arrays.")
    first = metadata_values[0]
    for metadata in metadata_values[1:]:
        if metadata != first:
            raise ValueError(f"Raymobtime beam split metadata is inconsistent: {metadata_values}.")
    return first


def _values_for_source_frame(
    frame: pd.DataFrame,
    normalized_by_source: dict[str, dict[str, np.ndarray | int]],
    key: str,
) -> np.ndarray:
    if frame.empty:
        dtype = np.asarray(next(iter(normalized_by_source.values()))[key]).dtype
        return np.asarray([], dtype=dtype)
    if not {"source_split", "source_split_index"} <= set(frame.columns):
        raise ValueError("Raymobtime split index is missing source_split/source_split_index columns.")
    result: np.ndarray | None = None
    for source_split, group in frame.groupby("source_split", sort=False):
        source_key = str(source_split)
        if source_key not in normalized_by_source:
            raise ValueError(
                f"Raymobtime split index references source_split={source_key!r}, "
                f"but available beam splits are {sorted(normalized_by_source)}."
            )
        values = np.asarray(normalized_by_source[source_key][key])
        positions = group["source_split_index"].to_numpy(dtype=np.int64)
        if positions.size and int(positions.max()) >= len(values):
            raise ValueError(
                f"Raymobtime source_split={source_key!r} references row {int(positions.max())}, "
                f"but beam split has only {len(values)} rows."
            )
        selected = values[positions]
        if result is None:
            result = np.empty((len(frame), *selected.shape[1:]), dtype=selected.dtype)
        result[frame.index.get_indexer(group.index)] = selected
    assert result is not None
    return result




__all__ = ["build_s008_labels", "normalize_beam_labels"]
