from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

from kd_sensing.preprocessing.radar import Doppler_Angle, Radar_Cube, Range_Angle, Range_Doppler
from kd_sensing.registries import PREPROCESSORS
from kd_sensing.utils.paths import resolve_path


def process_radar_and_create_new_csv(
    csv_path: str | Path,
    data_root: str | Path,
    output_csv_path: str | Path | None = None,
    output_suffix: str = "FFT",
    test_mode: bool = False,
    test_portion: float = 0.01,
    fft_tuple: list[int] | tuple[int, int, int] = (64, 256, 128),
) -> pd.DataFrame:
    csv_path = resolve_path(csv_path)
    data_root = resolve_path(data_root)
    frame = pd.read_csv(csv_path)
    if test_mode:
        frame = frame.head(max(1, int(len(frame) * test_portion)))
    radar_columns = [col for col in frame.columns if "radar" in col.lower() and "unit" in col.lower()]
    fft_output_dir = data_root / "unit1" / f"radar_data_{output_suffix}"
    fft_output_dir.mkdir(parents=True, exist_ok=True)
    processed_files: dict[str, Path] = {}
    frame_new = frame.copy()
    for radar_col in radar_columns:
        for idx in tqdm(range(len(frame)), desc=f"Processing {radar_col}"):
            radar_path = frame.loc[idx, radar_col]
            if pd.isna(radar_path) or radar_path == -99:
                continue
            original_filename = os.path.basename(radar_path)
            name_without_ext = os.path.splitext(original_filename)[0]
            new_filename = f"{name_without_ext}_{output_suffix}.npy"
            new_filepath = fft_output_dir / new_filename
            if original_filename not in processed_files:
                full_radar_path = data_root / str(radar_path).lstrip("./").lstrip("/")
                try:
                    smp_radar = loadmat(full_radar_path)["data"]
                    radar_cube = Radar_Cube(smp_radar, fft_tuple, remove_mean=True)
                    if output_suffix == "RA":
                        np.save(new_filepath, Range_Angle(radar_cube, mean=True, log_scale=True))
                    elif output_suffix == "RD":
                        np.save(new_filepath, Range_Doppler(radar_cube, mean=True, log_scale=True))
                    elif output_suffix == "DA":
                        np.save(new_filepath, Doppler_Angle(radar_cube, mean=True, log_scale=True))
                    else:
                        np.save(new_filepath, radar_cube)
                    processed_files[original_filename] = new_filepath
                except Exception as exc:
                    print(f"Error processing {full_radar_path}: {exc}")
                    continue
            frame_new.loc[idx, radar_col] = f"/unit1/radar_data_{output_suffix}/{new_filename}"
    if output_csv_path is None:
        output_csv_path = csv_path.with_name(f"{csv_path.stem}_{output_suffix}.csv")
    frame_new.to_csv(resolve_path(output_csv_path), index=False)
    return frame_new


@PREPROCESSORS.register("radar_fft_csv")
class CSVFFTPreprocessor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self):
        return process_radar_and_create_new_csv(**self.kwargs)

