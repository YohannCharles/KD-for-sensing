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


def create_samples(csv_path: str | Path, portion: float = 1.0) -> SequenceSamples:
    frame = pd.read_csv(csv_path, na_values="").fillna(-99)
    num_data = int(len(frame) * portion)
    data_samples_rgb = []
    data_samples_radar = []
    pred_beam = []
    inp_beam = []
    future_beam_cols = [col for col in frame.columns if col.startswith("future_beam")]
    future_beam_cols.sort()
    for _, row in frame.head(num_data).iterrows():
        data_samples_rgb.append(row["camera1":"camera8"].tolist())
        data_samples_radar.append(row["radar1":"radar8"].tolist())
        pred_beam.append(row[future_beam_cols].tolist())
        inp_beam.append(row["beam1":"beam8"].tolist())
    return SequenceSamples(
        rgb_paths=data_samples_rgb,
        radar_paths=data_samples_radar,
        input_beam_paths=inp_beam,
        future_beam_paths=pred_beam,
    )

