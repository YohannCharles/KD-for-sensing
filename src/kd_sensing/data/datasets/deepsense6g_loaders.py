from __future__ import annotations

from collections import OrderedDict

import torch

from kd_sensing.data.datasets.deepsense6g_cache_paths import (
    resolve_image_cache_dir,
    resolve_lidar_cache_dir_from_state,
)
from kd_sensing.data.transform_ops.lidar import load_lidar_background_points


class DeepSense6GModalityLoader:
    def __init__(self, dataset):
        self.dataset = dataset

    def load_image(self, idx: int) -> torch.Tensor:
        ds = self.dataset
        if ds.transform is None:
            raise ValueError("Image modality is enabled but image transform is unavailable.")
        return ds._load_rgb_imagenet_frames(ds.samples.rgb_paths[idx][-ds.seq_len :])

    def load_radar(self, idx: int):
        ds = self.dataset
        return ds._load_radar_maps(ds.samples.radar_paths[idx][-ds.seq_len :])

    def load_gps(self, idx: int) -> torch.Tensor:
        ds = self.dataset
        gps_features = ds._gps_features_for_index(idx)
        if ds.gps_scaler is not None:
            gps_features = ds.gps_scaler.transform(gps_features)
        return torch.tensor(gps_features, dtype=torch.float32)

    def load_gps_bev_xy(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.dataset._gps_bev_xy_for_index(idx), dtype=torch.float32)

    def load_mmwave(self, idx: int) -> torch.Tensor:
        ds = self.dataset
        mmwave_features = ds._mmwave_features_for_index(idx)
        if ds.mmwave_scaler is not None:
            mmwave_features = ds.mmwave_scaler.transform(mmwave_features)
        return torch.tensor(mmwave_features, dtype=torch.float32)

    def load_csi(self, idx: int) -> torch.Tensor:
        ds = self.dataset
        csi = ds._csi_for_index(idx)
        return torch.tensor(csi, dtype=torch.float32)

    def load_lidar_pair(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ds = self.dataset
        lidar_bev = ds._lidar_bev_for_index(
            idx,
            augment=ds.split == "train" and ds.lidar_augment,
        )
        raw_lidar_bev = lidar_bev
        if ds.lidar_normalize:
            if ds.lidar_normalizer is None:
                raise ValueError(
                    "LiDAR normalization is enabled but no normalizer is available. "
                    "Use build_dataloaders/evaluate, provide lidar_normalization.stats_path, "
                    "or disable LiDAR normalization."
                )
            lidar_bev = ds.lidar_normalizer.transform(lidar_bev)
        return (
            torch.tensor(raw_lidar_bev, dtype=torch.float32),
            torch.tensor(lidar_bev, dtype=torch.float32),
        )

    def load_lidar(self, idx: int) -> torch.Tensor:
        return self.load_lidar_pair(idx)[1]


def configure_deepsense6g_resource_readers(
    dataset,
    *,
    enabled_modalities,
    image_cache_dir,
    image_use_cache: bool,
    image_write_cache: bool,
    image_cache_policy,
    image_size,
    lidar_cache_dir,
    lidar_background_path,
) -> None:
    dataset.image_cache_policy = str(
        image_cache_policy or dataset._policy_from_cache_flags(image_use_cache, image_write_cache)
    )
    dataset.image_cache_dir = (
        resolve_image_cache_dir(
            scene_id=dataset.scene_id,
            data_root=dataset.data_root,
            image_cache_dir=image_cache_dir,
        )
        if "image" in enabled_modalities
        else None
    )
    dataset.image_cache = dataset._build_image_cache() if "image" in enabled_modalities else None
    dataset.lidar_cache_dir = resolve_lidar_cache_dir_from_state(dataset, lidar_cache_dir) if dataset.use_lidar else None
    dataset.lidar_background_points = (
        load_lidar_background_points(dataset.data_root, lidar_background_path) if dataset.use_lidar else None
    )
    dataset._lidar_bev_cache: OrderedDict[int, object] = OrderedDict()
    if "image" not in enabled_modalities:
        dataset.image_cache_dir = None
        dataset.image_cache = None
        dataset.image_cache_policy = "off"
    dataset.transform = dataset._build_image_transform(image_size) if "image" in enabled_modalities else None
    dataset.modality_loader = DeepSense6GModalityLoader(dataset)


__all__ = ["DeepSense6GModalityLoader", "configure_deepsense6g_resource_readers"]
