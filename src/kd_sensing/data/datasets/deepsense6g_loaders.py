from __future__ import annotations

import torch


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

    def load_mmwave(self, idx: int) -> torch.Tensor:
        ds = self.dataset
        mmwave_features = ds._mmwave_features_for_index(idx)
        if ds.mmwave_scaler is not None:
            mmwave_features = ds.mmwave_scaler.transform(mmwave_features)
        return torch.tensor(mmwave_features, dtype=torch.float32)

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


__all__ = ["DeepSense6GModalityLoader"]
