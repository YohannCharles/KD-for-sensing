from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kd_sensing.registries import PREPROCESSORS
from kd_sensing.utils.paths import resolve_path


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
) -> tuple[Path, Path]:
    csv_path = resolve_path(csv_path)
    data_root = resolve_path(data_root)
    all_data = pd.read_csv(csv_path)
    all_seq_idx = all_data["seq_index"].unique()
    all_seq_split = []
    base_columns = ["unit1_rgb", "unit1_radar", "unit1_pwr_60ghz", "seq_index"]
    gps_source_column = gps_column
    lidar_source_column = lidar_column
    mmwave_source_column = mmwave_column
    if include_gps:
        if gps_source_column not in all_data.columns:
            if gps_fallback_column and gps_fallback_column in all_data.columns:
                gps_source_column = gps_fallback_column
            else:
                raise ValueError(
                    f"GPS column '{gps_column}' not found in {csv_path}; "
                    f"fallback '{gps_fallback_column}' is also unavailable."
                )
        if bs_gps_column not in all_data.columns:
            raise ValueError(f"BS GPS column '{bs_gps_column}' not found in {csv_path}.")
        base_columns.extend([gps_source_column, bs_gps_column])
    if include_lidar:
        if lidar_source_column not in all_data.columns:
            if lidar_fallback_column and lidar_fallback_column in all_data.columns:
                lidar_source_column = lidar_fallback_column
            else:
                raise ValueError(
                    f"LiDAR column '{lidar_column}' not found in {csv_path}; "
                    f"fallback '{lidar_fallback_column}' is also unavailable."
                )
        base_columns.append(lidar_source_column)
    if include_mmwave:
        if mmwave_source_column not in all_data.columns:
            if mmwave_fallback_column and mmwave_fallback_column in all_data.columns:
                mmwave_source_column = mmwave_fallback_column
            else:
                raise ValueError(
                    f"mmWave column '{mmwave_column}' not found in {csv_path}; "
                    f"fallback '{mmwave_fallback_column}' is also unavailable."
                )
        if mmwave_source_column not in base_columns:
            base_columns.append(mmwave_source_column)
    for seq_idx in all_seq_idx:
        tmp = all_data[all_data["seq_index"] == seq_idx]
        all_seq_split.append(tmp[base_columns])
    all_seqs = []
    for seq in all_seq_split:
        start = 0
        while start + in_len + out_len <= seq.shape[0]:
            image = seq["unit1_rgb"][start : start + in_len].tolist()
            radar = seq["unit1_radar"][start : start + in_len].tolist()
            gps = seq[gps_source_column][start : start + in_len].tolist() if include_gps else []
            bs_gps = seq[bs_gps_column][start : start + in_len].tolist() if include_gps else []
            lidar = seq[lidar_source_column][start : start + in_len].tolist() if include_lidar else []
            mmwave = seq[mmwave_source_column][start : start + in_len].tolist() if include_mmwave else []
            in_beam = seq["unit1_pwr_60ghz"][start : start + in_len].tolist()
            out_beam = seq["unit1_pwr_60ghz"][start + in_len : start + in_len + out_len].tolist()
            seq_idx = seq["seq_index"][0:1].tolist()
            all_seqs.append(image + radar + gps + bs_gps + lidar + mmwave + in_beam + out_beam + seq_idx)
            start += 1
    col_names = (
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
    all_seqs = pd.DataFrame(all_seqs, columns=col_names)
    ind_select = int(training_set_pct * all_seq_idx.shape[0])
    train_seq_idx = np.sort(all_seq_idx[:ind_select])
    test_seq_idx = np.sort(all_seq_idx[ind_select:])
    train_path = data_root / f"train_seqs{output_suffix}.csv"
    test_path = data_root / f"test_seqs{output_suffix}.csv"
    all_seqs[all_seqs["seq_index"].isin(train_seq_idx)].to_csv(train_path, index=False)
    all_seqs[all_seqs["seq_index"].isin(test_seq_idx)].to_csv(test_path, index=False)
    return train_path, test_path


@PREPROCESSORS.register("sequence_csv")
class SequencePreprocessor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self):
        return generate_sequence_data(**self.kwargs)
