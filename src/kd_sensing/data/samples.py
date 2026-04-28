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


def create_samples(csv_path: str | Path, portion: float = 1.0) -> SequenceSamples:
    frame = pd.read_csv(csv_path, na_values="").fillna(-99)
    num_data = int(len(frame) * portion)
    data_samples_rgb = []
    data_samples_radar = []
    data_samples_gps = []
    data_samples_bs_gps = []
    pred_beam = []
    inp_beam = []
    future_beam_cols = _sorted_numbered_columns(frame.columns, "future_beam")
    gps_cols = _sorted_numbered_columns(frame.columns, "gps")
    bs_gps_cols = _sorted_numbered_columns(frame.columns, "bs_gps")
    for _, row in frame.head(num_data).iterrows():
        data_samples_rgb.append(row["camera1":"camera8"].tolist())
        data_samples_radar.append(row["radar1":"radar8"].tolist())
        if gps_cols:
            data_samples_gps.append(row[gps_cols].tolist())
        if bs_gps_cols:
            data_samples_bs_gps.append(row[bs_gps_cols].tolist())
        pred_beam.append(row[future_beam_cols].tolist())
        inp_beam.append(row["beam1":"beam8"].tolist())
    return SequenceSamples(
        rgb_paths=data_samples_rgb,
        radar_paths=data_samples_radar,
        input_beam_paths=inp_beam,
        future_beam_paths=pred_beam,
        gps_paths=data_samples_gps or None,
        bs_gps_paths=data_samples_bs_gps or None,
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
