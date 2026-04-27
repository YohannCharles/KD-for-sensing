from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import io
from skimage.color import rgb2gray


def build_image_transform(image_size: list[int] | tuple[int, int] = (224, 224)):
    height, width = tuple(image_size)

    def transform(array):
        image = Image.fromarray(array)
        return image.resize((width, height))

    return transform


def joined_resource(data_root: str | Path, rel_path: str) -> Path:
    return Path(data_root) / str(rel_path).lstrip("/")


def load_motion_masks(
    data_root: str | Path,
    rgb_paths: list[str],
    seq_len: int,
    transform=None,
) -> torch.Tensor:
    transform = transform or build_image_transform()
    image_val = np.zeros((seq_len, 224, 224))
    image_motion_masks = np.zeros((seq_len - 1, 224, 224))
    for i, rel_path in enumerate(rgb_paths[-seq_len:]):
        img = transform(io.imread(joined_resource(data_root, rel_path)))
        img = rgb2gray(np.asarray(img))
        image_val[i, ...] = gaussian_filter(img, sigma=1)
        if i >= 1:
            diff = np.abs(image_val[i, ...] - image_val[i - 1, ...])
            max_pixel_value = np.max(diff)
            threshold_value = 0.1 * max_pixel_value
            image_motion_masks[i - 1, ...] = (diff > threshold_value).astype(np.uint8)
    return torch.tensor(image_motion_masks, dtype=torch.float32)


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
