from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from kd_sensing.data.datasets.deepsense6g_columns import sorted_numbered_columns, validate_required_columns


@dataclass
class SequenceSamples:
    rgb_paths: list[list[str]]
    radar_paths: list[list[str]]
    input_beam_paths: list[list[str]]
    future_beam_paths: list[list[str]]
    gps_paths: list[list[str]] | None = None
    bs_gps_paths: list[list[str]] | None = None
    future_gps_paths: list[list[str]] | None = None
    future_bs_gps_paths: list[list[str]] | None = None
    lidar_paths: list[list[str]] | None = None
    mmwave_paths: list[list[str]] | None = None
    csi_paths: list[list[str]] | None = None
    metadata: dict | None = None


def create_samples(
    csv_path: str | Path,
    portion: float = 1.0,
    *,
    enabled_modalities: list[str] | tuple[str, ...] | set[str] | None = None,
    seq_len: int | None = None,
    gps_seq_len: int | None = None,
    gps_source_seq_len: int | None = None,
    num_pred: int | None = None,
    portion_strategy: str = "even",
    portion_seed: int = 42,
    include_position_targets: bool = False,
    include_history_position_targets: bool = False,
) -> SequenceSamples:
    frame = pd.read_csv(csv_path, na_values="").fillna(-99)
    selected_frame, metadata = _select_portion(
        frame,
        portion=portion,
        strategy=portion_strategy,
        seed=portion_seed,
    )
    selected_modalities = tuple(enabled_modalities or ("image", "radar"))
    data_samples_rgb = []
    data_samples_radar = []
    data_samples_gps = []
    data_samples_bs_gps = []
    data_samples_future_gps = []
    data_samples_future_bs_gps = []
    data_samples_lidar = []
    data_samples_mmwave = []
    data_samples_csi = []
    pred_beam = []
    inp_beam = []
    camera_cols = sorted_numbered_columns(frame.columns, "camera")
    radar_cols = sorted_numbered_columns(frame.columns, "radar")
    future_beam_cols = sorted_numbered_columns(frame.columns, "future_beam")
    beam_cols = sorted_numbered_columns(frame.columns, "beam")
    gps_cols = sorted_numbered_columns(frame.columns, "gps")
    bs_gps_cols = sorted_numbered_columns(frame.columns, "bs_gps")
    future_gps_cols = sorted_numbered_columns(frame.columns, "future_gps")
    future_bs_gps_cols = sorted_numbered_columns(frame.columns, "future_bs_gps")
    lidar_cols = sorted_numbered_columns(frame.columns, "lidar")
    mmwave_cols = sorted_numbered_columns(frame.columns, "mmwave")
    csi_cols = sorted_numbered_columns(frame.columns, "csi")
    validate_required_columns(
        csv_path,
        selected_modalities,
        camera_cols=camera_cols,
        radar_cols=radar_cols,
        gps_cols=gps_cols,
        bs_gps_cols=bs_gps_cols,
        future_gps_cols=future_gps_cols,
        future_bs_gps_cols=future_bs_gps_cols,
        lidar_cols=lidar_cols,
        mmwave_cols=mmwave_cols,
        csi_cols=csi_cols,
        beam_cols=beam_cols,
        future_beam_cols=future_beam_cols,
        seq_len=seq_len,
        gps_seq_len=gps_source_seq_len if gps_source_seq_len is not None else gps_seq_len,
        num_pred=num_pred,
        include_position_targets=include_position_targets,
        include_history_position_targets=include_history_position_targets,
    )
    needs_history_gps = "gps" in selected_modalities or include_history_position_targets
    for _, row in selected_frame.iterrows():
        if "image" in selected_modalities:
            data_samples_rgb.append(row[camera_cols].tolist())
        if "radar" in selected_modalities:
            data_samples_radar.append(row[radar_cols].tolist())
        if needs_history_gps:
            data_samples_gps.append(row[gps_cols].tolist())
            data_samples_bs_gps.append(row[bs_gps_cols].tolist())
        if include_position_targets:
            data_samples_future_gps.append(row[future_gps_cols].tolist())
            data_samples_future_bs_gps.append(row[future_bs_gps_cols].tolist())
        if "lidar" in selected_modalities:
            data_samples_lidar.append(row[lidar_cols].tolist())
        if "mmwave" in selected_modalities:
            data_samples_mmwave.append(row[mmwave_cols].tolist())
        if "csi" in selected_modalities:
            data_samples_csi.append(row[csi_cols].tolist())
        pred_beam.append(row[future_beam_cols].tolist())
        inp_beam.append(row[beam_cols].tolist())
    return SequenceSamples(
        rgb_paths=data_samples_rgb,
        radar_paths=data_samples_radar,
        input_beam_paths=inp_beam,
        future_beam_paths=pred_beam,
        gps_paths=data_samples_gps or None,
        bs_gps_paths=data_samples_bs_gps or None,
        future_gps_paths=data_samples_future_gps or None,
        future_bs_gps_paths=data_samples_future_bs_gps or None,
        lidar_paths=data_samples_lidar or None,
        mmwave_paths=data_samples_mmwave or None,
        csi_paths=data_samples_csi or None,
        metadata=metadata,
    )


def _select_portion(
    frame: pd.DataFrame,
    *,
    portion: float,
    strategy: str,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    if portion <= 0:
        raise ValueError(f"portion must be positive, got {portion}.")
    total = len(frame)
    if portion >= 1.0 or total == 0:
        selected = frame
        selected_indices = list(range(total))
        effective_strategy = "all"
    else:
        selected_count = max(1, int(total * portion))
        selected_count = min(selected_count, total)
        effective_strategy = strategy or "even"
        if effective_strategy == "head":
            selected_indices = list(range(selected_count))
        elif effective_strategy == "random":
            rng = np.random.default_rng(int(seed))
            selected_indices = sorted(int(idx) for idx in rng.choice(total, size=selected_count, replace=False))
        elif effective_strategy in {"even", "deterministic_even", "stratified_even"}:
            selected_indices = _even_indices(total, selected_count)
            effective_strategy = "even"
        else:
            raise ValueError(f"Unsupported portion_strategy '{strategy}'.")
        selected = frame.iloc[selected_indices]
    metadata = {
        "total_rows": int(total),
        "selected_rows": int(len(selected)),
        "portion": float(portion),
        "portion_strategy": effective_strategy,
        "portion_seed": int(seed),
    }
    if "seq_index" in frame.columns and len(selected) > 0:
        seq_values = selected["seq_index"]
        metadata.update(
            {
                "seq_index_min": _safe_scalar(seq_values.min()),
                "seq_index_max": _safe_scalar(seq_values.max()),
                "seq_index_count": int(seq_values.nunique()),
            }
        )
    return selected, metadata


def _even_indices(total: int, selected_count: int) -> list[int]:
    if selected_count >= total:
        return list(range(total))
    if selected_count == 1:
        return [total // 2]
    raw = np.linspace(0, total - 1, selected_count)
    indices = sorted({int(round(value)) for value in raw})
    candidate = 0
    while len(indices) < selected_count:
        if candidate not in indices:
            indices.append(candidate)
        candidate += 1
    return sorted(indices[:selected_count])


def _safe_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value
