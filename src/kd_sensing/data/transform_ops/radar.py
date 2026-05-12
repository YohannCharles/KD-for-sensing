from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from kd_sensing.data.transform_ops.io import joined_resource


def load_radar_maps(
    data_root: str | Path,
    radar_paths: list[str],
    seq_len: int,
    fft_tuple: list[int] | tuple[int, int, int] = (64, 256, 128),
    clipped_range: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    radar_val_range_angle = np.zeros((seq_len, clipped_range, fft_tuple[0]))
    radar_val_doppler_angle = np.zeros((seq_len, fft_tuple[2], fft_tuple[0]))
    for i, rel_path in enumerate(radar_paths[-seq_len:]):
        range_angle_map = np.load(joined_resource(data_root, rel_path))
        radar_val_range_angle[i, ...] = range_angle_map[:clipped_range, ...]
        doppler_rel_path = str(rel_path).replace("_RA", "_DA")
        radar_val_doppler_angle[i, ...] = np.load(joined_resource(data_root, doppler_rel_path))
    return (
        torch.tensor(radar_val_range_angle, dtype=torch.float32),
        torch.tensor(radar_val_doppler_angle, dtype=torch.float32),
    )


__all__ = ["load_radar_maps"]
