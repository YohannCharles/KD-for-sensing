from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class SequenceSamples:
    rgb_paths: list[list[str]]
    radar_paths: list[list[str]]
    input_beam_paths: list[list[str]]
    future_beam_paths: list[list[str]]
    gps_paths: list[list[str]] | None = None
    bs_gps_paths: list[list[str]] | None = None
    lidar_paths: list[list[str]] | None = None


def create_samples(
    csv_path: str | Path,
    portion: float = 1.0,
    *,
    enabled_modalities: list[str] | tuple[str, ...] | set[str] | None = None,
    seq_len: int | None = None,
    num_pred: int | None = None,
) -> SequenceSamples:
    frame = pd.read_csv(csv_path, na_values="").fillna(-99)
    num_data = int(len(frame) * portion)
    selected_modalities = tuple(enabled_modalities or ("image", "radar"))
    data_samples_rgb = []
    data_samples_radar = []
    data_samples_gps = []
    data_samples_bs_gps = []
    data_samples_lidar = []
    pred_beam = []
    inp_beam = []
    camera_cols = _sorted_numbered_columns(frame.columns, "camera")
    radar_cols = _sorted_numbered_columns(frame.columns, "radar")
    future_beam_cols = _sorted_numbered_columns(frame.columns, "future_beam")
    beam_cols = _sorted_numbered_columns(frame.columns, "beam")
    gps_cols = _sorted_numbered_columns(frame.columns, "gps")
    bs_gps_cols = _sorted_numbered_columns(frame.columns, "bs_gps")
    lidar_cols = _sorted_numbered_columns(frame.columns, "lidar")
    _validate_required_columns(
        csv_path,
        selected_modalities,
        camera_cols=camera_cols,
        radar_cols=radar_cols,
        gps_cols=gps_cols,
        bs_gps_cols=bs_gps_cols,
        lidar_cols=lidar_cols,
        beam_cols=beam_cols,
        future_beam_cols=future_beam_cols,
        seq_len=seq_len,
        num_pred=num_pred,
    )
    for _, row in frame.head(num_data).iterrows():
        if "image" in selected_modalities:
            data_samples_rgb.append(row[camera_cols].tolist())
        if "radar" in selected_modalities:
            data_samples_radar.append(row[radar_cols].tolist())
        if "gps" in selected_modalities:
            data_samples_gps.append(row[gps_cols].tolist())
            data_samples_bs_gps.append(row[bs_gps_cols].tolist())
        if "lidar" in selected_modalities:
            data_samples_lidar.append(row[lidar_cols].tolist())
        pred_beam.append(row[future_beam_cols].tolist())
        inp_beam.append(row[beam_cols].tolist())
    return SequenceSamples(
        rgb_paths=data_samples_rgb,
        radar_paths=data_samples_radar,
        input_beam_paths=inp_beam,
        future_beam_paths=pred_beam,
        gps_paths=data_samples_gps or None,
        bs_gps_paths=data_samples_bs_gps or None,
        lidar_paths=data_samples_lidar or None,
    )


def _validate_required_columns(
    csv_path: str | Path,
    enabled_modalities: tuple[str, ...],
    *,
    camera_cols: list[str],
    radar_cols: list[str],
    gps_cols: list[str],
    bs_gps_cols: list[str],
    lidar_cols: list[str],
    beam_cols: list[str],
    future_beam_cols: list[str],
    seq_len: int | None,
    num_pred: int | None,
) -> None:
    path = Path(csv_path)
    minimum_seq = int(seq_len) if seq_len is not None else 1
    minimum_pred = int(num_pred) if num_pred is not None else 1
    requirements = {
        "beam": (beam_cols, minimum_seq, "beam1..beamN"),
        "future_beam": (future_beam_cols, minimum_pred, "future_beam1..future_beamN"),
    }
    if "image" in enabled_modalities:
        requirements["image"] = (camera_cols, minimum_seq, "camera1..cameraN")
    if "radar" in enabled_modalities:
        requirements["radar"] = (radar_cols, minimum_seq, "radar1..radarN")
    if "gps" in enabled_modalities:
        requirements["gps"] = (gps_cols, minimum_seq, "gps1..gpsN")
        requirements["bs_gps"] = (bs_gps_cols, minimum_seq, "bs_gps1..bs_gpsN")
    if "lidar" in enabled_modalities:
        requirements["lidar"] = (lidar_cols, minimum_seq, "lidar1..lidarN")
    for name, (columns, minimum, expected) in requirements.items():
        if len(columns) < minimum:
            raise ValueError(
                f"{name} is enabled but {path} contains {len(columns)} {expected} columns; "
                f"expected at least {minimum}."
            )


def _sorted_numbered_columns(columns, prefix: str) -> list[str]:
    selected = []
    for col in columns:
        if not col.startswith(prefix):
            continue
        suffix = col[len(prefix) :]
        if suffix.isdigit():
            selected.append(col)
    return sorted(selected, key=lambda name: int(name[len(prefix) :]))
