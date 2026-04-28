from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from kd_sensing.data.samples import create_samples
from kd_sensing.data.transforms import (
    GPSStandardScaler,
    DEFAULT_LIDAR_BEV_SIZE,
    DEFAULT_LIDAR_ROI,
    LidarBEVNormalizer,
    LidarBEVStreamingStats,
    SUPPORTED_GPS_FEATURE_MODE,
    build_image_transform,
    joined_resource,
    load_lidar_background_points,
    load_lidar_bev_sequence,
    load_gps_feature_sequence,
    load_motion_masks,
    load_radar_maps,
)
from kd_sensing.registries import DATASETS
from kd_sensing.utils.paths import resolve_path


@DATASETS.register("scenario9")
class Scenario9Dataset(Dataset):
    """Scenario 9 sequence dataset with standardized batch field names."""

    def __init__(
        self,
        data_root: str,
        csv_name: str | None = None,
        root_csv: str | None = None,
        split: str = "train",
        train_csv_name: str = "train_seqs_RA.csv",
        test_csv_name: str = "test_seqs_RA.csv",
        seq_len: int = 8,
        num_pred: int = 3,
        image_size: list[int] | tuple[int, int] = (224, 224),
        fft_tuple: list[int] | tuple[int, int, int] = (64, 256, 128),
        clipped_range: int = 128,
        portion: float = 1.0,
        use_gps: bool = False,
        gps_feature_mode: str = SUPPORTED_GPS_FEATURE_MODE,
        gps_normalize: bool = True,
        gps_smooth_window: int = 3,
        gps_scaler: GPSStandardScaler | None = None,
        use_lidar: bool = False,
        lidar_bev_size: list[int] | tuple[int, int] = DEFAULT_LIDAR_BEV_SIZE,
        lidar_roi: list[float] | tuple[float, ...] = DEFAULT_LIDAR_ROI,
        lidar_fov_degrees: list[float] | tuple[float, float] | None = None,
        lidar_remove_ground: bool = False,
        lidar_ground_z_threshold: float = 0.1,
        lidar_background_path: str | None = None,
        lidar_background_distance_threshold: float = 0.2,
        lidar_cache_dir: str | None = None,
        lidar_use_cache: bool = False,
        lidar_write_cache: bool = False,
        lidar_normalize: bool = False,
        lidar_normalization: dict[str, Any] | None = None,
        lidar_normalizer: LidarBEVNormalizer | None = None,
        lidar_memory_cache: bool | dict[str, Any] = False,
        lidar_augment: bool = False,
        lidar_point_dropout: float = 0.0,
        lidar_jitter_std: float = 0.0,
        **_: object,
    ):
        self.data_root = resolve_path(data_root)
        selected_csv = root_csv or csv_name
        if selected_csv is None:
            selected_csv = train_csv_name if split == "train" else test_csv_name
        self.root_csv = Path(selected_csv)
        if not self.root_csv.is_absolute():
            self.root_csv = self.data_root / self.root_csv
        self.seq_len = seq_len
        self.num_pred = num_pred
        self.fft_tuple = tuple(fft_tuple)
        self.clipped_range = clipped_range
        self.split = split
        self.use_gps = use_gps
        self.gps_feature_mode = gps_feature_mode
        self.gps_normalize = gps_normalize
        self.gps_smooth_window = gps_smooth_window
        self.gps_scaler = gps_scaler
        self._gps_feature_cache: dict[int, np.ndarray] = {}
        self.use_lidar = use_lidar
        self.lidar_bev_size = tuple(lidar_bev_size)
        self.lidar_roi = tuple(lidar_roi)
        self.lidar_fov_degrees = tuple(lidar_fov_degrees) if lidar_fov_degrees is not None else None
        self.lidar_remove_ground = lidar_remove_ground
        self.lidar_ground_z_threshold = lidar_ground_z_threshold
        self.lidar_background_distance_threshold = lidar_background_distance_threshold
        self.lidar_use_cache = lidar_use_cache
        self.lidar_write_cache = lidar_write_cache
        self.lidar_normalization = self._resolve_lidar_normalization(lidar_normalize, lidar_normalization)
        self.lidar_normalize = self.lidar_normalization["enabled"]
        self.lidar_normalization_mode = self.lidar_normalization["mode"]
        self.lidar_stats_path = self._resolve_lidar_stats_path(self.lidar_normalization.get("stats_path"))
        self.lidar_stats_recompute = bool(self.lidar_normalization.get("recompute", False))
        self.lidar_normalizer = lidar_normalizer
        self.lidar_memory_cache_enabled, self.lidar_memory_cache_max_items = self._resolve_lidar_memory_cache(
            lidar_memory_cache
        )
        self.lidar_augment = lidar_augment
        self.lidar_point_dropout = lidar_point_dropout
        self.lidar_jitter_std = lidar_jitter_std
        self.lidar_cache_dir = self._resolve_lidar_cache_dir(lidar_cache_dir)
        self.lidar_background_points = load_lidar_background_points(self.data_root, lidar_background_path)
        self._lidar_bev_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.transform = build_image_transform(image_size)
        self.samples = create_samples(self.root_csv, portion=portion)
        if self.use_gps:
            self._ensure_gps_columns()
            self._prepare_gps_scaler()
        if self.use_lidar:
            self._ensure_lidar_columns()
            self._prepare_lidar_normalizer_from_config()

    def __len__(self) -> int:
        return len(self.samples.rgb_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        samples_rgb = self.samples.rgb_paths[idx][-self.seq_len :]
        samples_radar = self.samples.radar_paths[idx][-self.seq_len :]
        beam_paths = self.samples.input_beam_paths[idx][-self.seq_len :]
        future_beam_paths = self.samples.future_beam_paths[idx][: self.num_pred]

        image = load_motion_masks(self.data_root, samples_rgb, self.seq_len, self.transform)
        radar_ra, radar_da = load_radar_maps(
            self.data_root,
            samples_radar,
            self.seq_len,
            self.fft_tuple,
            self.clipped_range,
        )

        input_beam = [
            int(np.argmax(np.loadtxt(joined_resource(self.data_root, beam_path))))
            for beam_path in beam_paths
        ]
        target_beam = [
            int(np.argmax(np.loadtxt(joined_resource(self.data_root, beam_path))))
            for beam_path in future_beam_paths
        ]
        sample = {
            "image": image,
            "radar_ra": radar_ra,
            "radar_da": radar_da,
            "input_beam": torch.tensor(input_beam, dtype=torch.int64),
            "target_beam": torch.tensor(target_beam, dtype=torch.int64).squeeze(),
        }
        if self.use_gps:
            gps_features = self._gps_features_for_index(idx)
            if self.gps_scaler is not None:
                gps_features = self.gps_scaler.transform(gps_features)
            sample["gps"] = torch.tensor(gps_features, dtype=torch.float32)
        if self.use_lidar:
            lidar_bev = self._lidar_bev_for_index(
                idx,
                augment=self.split == "train" and self.lidar_augment,
            )
            if self.lidar_normalize:
                if self.lidar_normalizer is None:
                    raise ValueError(
                        "LiDAR normalization is enabled but no normalizer is available. "
                        "Use build_dataloaders/evaluate, provide lidar_normalization.stats_path, "
                        "or disable LiDAR normalization."
                    )
                lidar_bev = self.lidar_normalizer.transform(lidar_bev)
            sample["lidar"] = torch.tensor(lidar_bev, dtype=torch.float32)
        return sample

    def _ensure_gps_columns(self) -> None:
        if self.samples.gps_paths is None:
            raise ValueError(
                f"GPS is enabled but {self.root_csv} does not contain gps1..gpsN columns. "
                "Regenerate sequence CSVs with include_gps: true."
            )
        if self.gps_feature_mode != SUPPORTED_GPS_FEATURE_MODE:
            raise ValueError(
                f"Unsupported gps_feature_mode '{self.gps_feature_mode}'. "
                f"This change only supports '{SUPPORTED_GPS_FEATURE_MODE}'."
            )
        if self.samples.bs_gps_paths is None:
            raise ValueError(
                f"gps_feature_mode '{self.gps_feature_mode}' requires bs_gps1..bs_gpsN columns in {self.root_csv}."
            )

    def _prepare_gps_scaler(self) -> None:
        if not self.gps_normalize:
            self.gps_scaler = None
            return
        if self.gps_scaler is not None:
            return
        if self.split != "train":
            raise ValueError(
                "GPS normalization for non-train split requires a train-fitted gps_scaler. "
                "Use build_dataloaders/evaluate so the train scaler can be reused."
            )
        all_features = [self._gps_features_for_index(idx) for idx in range(len(self))]
        stacked = np.concatenate(all_features, axis=0)
        self.gps_scaler = GPSStandardScaler().fit(stacked)

    def _gps_features_for_index(self, idx: int) -> np.ndarray:
        if idx not in self._gps_feature_cache:
            if self.samples.gps_paths is None:
                raise ValueError("GPS paths are unavailable for this dataset.")
            bs_paths = self.samples.bs_gps_paths[idx] if self.samples.bs_gps_paths is not None else None
            self._gps_feature_cache[idx] = load_gps_feature_sequence(
                self.data_root,
                self.samples.gps_paths[idx],
                bs_paths,
                seq_len=self.seq_len,
                mode=self.gps_feature_mode,
                smooth_window=self.gps_smooth_window,
            )
        return self._gps_feature_cache[idx]

    def _resolve_lidar_cache_dir(self, lidar_cache_dir: str | None) -> Path | None:
        if not lidar_cache_dir:
            return None
        path = Path(lidar_cache_dir).expanduser()
        if path.is_absolute():
            return path
        return self.data_root / path

    def _resolve_lidar_stats_path(self, stats_path: str | None) -> Path | None:
        if not stats_path:
            return None
        return resolve_path(stats_path)

    def _resolve_lidar_normalization(
        self,
        legacy_enabled: bool,
        config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if config is None:
            enabled = bool(legacy_enabled)
            mode = "streaming_stats" if enabled else "none"
            stats_path = None
            recompute = False
        else:
            enabled = bool(config.get("enabled", False))
            mode = str(config.get("mode", "streaming_stats" if enabled else "none"))
            stats_path = config.get("stats_path")
            recompute = bool(config.get("recompute", False))
        if not enabled:
            mode = "none"
        if mode not in {"none", "streaming_stats"}:
            raise ValueError(f"Unsupported LiDAR normalization mode '{mode}'.")
        return {
            "enabled": enabled,
            "mode": mode,
            "stats_path": stats_path,
            "recompute": recompute,
        }

    def _resolve_lidar_memory_cache(self, config: bool | dict[str, Any]) -> tuple[bool, int | None]:
        if isinstance(config, dict):
            enabled = bool(config.get("enabled", False))
            max_items = config.get("max_items")
        else:
            enabled = bool(config)
            max_items = None
        if max_items is not None:
            max_items = int(max_items)
            if max_items <= 0:
                raise ValueError("lidar_memory_cache.max_items must be positive when provided.")
        return enabled, max_items

    def _ensure_lidar_columns(self) -> None:
        if self.samples.lidar_paths is None:
            raise ValueError(
                f"LiDAR is enabled but {self.root_csv} does not contain lidar1..lidarN columns. "
                "Regenerate sequence CSVs with include_lidar: true."
            )

    def _prepare_lidar_normalizer_from_config(self) -> None:
        if not self.lidar_normalize:
            self.lidar_normalizer = None
            return
        if self.lidar_normalizer is not None:
            return
        if self.lidar_stats_path is not None and self.lidar_stats_path.exists() and not self.lidar_stats_recompute:
            self.lidar_normalizer = LidarBEVNormalizer.load(self.lidar_stats_path)
            return
        if self.split != "train":
            raise ValueError(
                "LiDAR normalization for non-train split requires a train-fitted lidar_normalizer "
                "or an existing lidar_normalization.stats_path. Use build_dataloaders/evaluate so "
                "the train normalizer can be reused."
            )

    @property
    def needs_lidar_streaming_stats(self) -> bool:
        return (
            self.use_lidar
            and self.lidar_normalize
            and self.lidar_normalization_mode == "streaming_stats"
            and self.lidar_normalizer is None
            and self.split == "train"
        )

    def fit_lidar_normalizer_streaming(self, *, progress_enabled: bool = False) -> LidarBEVNormalizer:
        if not self.needs_lidar_streaming_stats:
            if self.lidar_normalizer is None:
                raise ValueError("LiDAR streaming stats were requested but the dataset is not fit-ready.")
            return self.lidar_normalizer
        stats = LidarBEVStreamingStats()
        iterator = range(len(self))
        if progress_enabled:
            iterator = tqdm(iterator, desc="LiDAR stats", unit="sample")
        for idx in iterator:
            stats.update(self._lidar_bev_for_index(idx, augment=False))
        self.lidar_normalizer = stats.finalize()
        if self.lidar_stats_path is not None:
            self.lidar_normalizer.save(self.lidar_stats_path)
        return self.lidar_normalizer

    def _lidar_bev_for_index(self, idx: int, *, augment: bool) -> np.ndarray:
        if not augment and self.lidar_memory_cache_enabled and idx in self._lidar_bev_cache:
            self._lidar_bev_cache.move_to_end(idx)
            return self._lidar_bev_cache[idx]
        if self.samples.lidar_paths is None:
            raise ValueError("LiDAR paths are unavailable for this dataset.")
        bev = load_lidar_bev_sequence(
            self.data_root,
            self.samples.lidar_paths[idx],
            seq_len=self.seq_len,
            bev_size=self.lidar_bev_size,
            roi=self.lidar_roi,
            fov_degrees=self.lidar_fov_degrees,
            remove_ground=self.lidar_remove_ground,
            ground_z_threshold=self.lidar_ground_z_threshold,
            background_points=self.lidar_background_points,
            background_distance_threshold=self.lidar_background_distance_threshold,
            cache_dir=self.lidar_cache_dir,
            use_cache=self.lidar_use_cache,
            write_cache=self.lidar_write_cache and not augment,
            augment=augment,
            point_dropout=self.lidar_point_dropout,
            jitter_std=self.lidar_jitter_std,
        )
        if not augment and self.lidar_memory_cache_enabled:
            self._lidar_bev_cache[idx] = bev
            self._lidar_bev_cache.move_to_end(idx)
            if self.lidar_memory_cache_max_items is not None:
                while len(self._lidar_bev_cache) > self.lidar_memory_cache_max_items:
                    self._lidar_bev_cache.popitem(last=False)
        return bev
