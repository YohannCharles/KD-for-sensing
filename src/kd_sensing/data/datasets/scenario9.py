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
    MMWAVE_POWER_DIM,
    MmWaveStandardScaler,
    SUPPORTED_GPS_FEATURE_MODE,
    build_image_transform,
    joined_resource,
    load_lidar_background_points,
    load_lidar_bev_sequence,
    load_gps_feature_sequence,
    load_mmwave_feature_sequence,
    load_motion_masks,
    load_radar_maps,
    parameterized_image_motion_cache_dir,
    parameterized_lidar_cache_dir,
)
from kd_sensing.registries import DATASETS
from kd_sensing.utils.paths import resolve_path


VALID_MODALITIES = ("image", "radar", "gps", "lidar", "mmwave")


@DATASETS.register("scenario9")
class Scenario9Dataset(Dataset):
    """Scenario 9 sequence dataset with standardized batch field names."""

    def __init__(
        self,
        data_root: str,
        csv_name: str | None = None,
        root_csv: str | None = None,
        split: str = "train",
        train_csv_name: str = "train_seqs_RA_GPS_LIDAR.csv",
        test_csv_name: str = "test_seqs_RA_GPS_LIDAR.csv",
        seq_len: int = 8,
        num_pred: int = 3,
        image_size: list[int] | tuple[int, int] = (224, 224),
        image_motion_cache_dir: str | None = None,
        image_motion_use_cache: bool = False,
        image_motion_write_cache: bool = False,
        image_motion_cache_policy: str | None = None,
        image_motion_cache_version: str = "v1",
        image_motion_gaussian_sigma: float = 1.0,
        image_motion_threshold_ratio: float = 0.1,
        image_motion_threshold_strategy: str = "relative_max",
        image_motion_grayscale: str = "rgb2gray",
        fft_tuple: list[int] | tuple[int, int, int] = (64, 256, 128),
        clipped_range: int = 128,
        portion: float = 1.0,
        beam_label_cache: bool | str = "lazy",
        use_gps: bool = False,
        gps_feature_mode: str = SUPPORTED_GPS_FEATURE_MODE,
        gps_normalize: bool = True,
        gps_scaler: GPSStandardScaler | None = None,
        use_mmwave: bool = False,
        mmwave_normalize: bool = True,
        mmwave_scaler: MmWaveStandardScaler | None = None,
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
        lidar_cache_policy: str | None = None,
        lidar_normalize: bool = False,
        lidar_normalization: dict[str, Any] | None = None,
        lidar_normalizer: LidarBEVNormalizer | None = None,
        lidar_memory_cache: bool | dict[str, Any] = False,
        lidar_augment: bool = False,
        lidar_point_dropout: float = 0.0,
        lidar_jitter_std: float = 0.0,
        enabled_modalities: list[str] | tuple[str, ...] | None = None,
        portion_strategy: str = "even",
        portion_seed: int = 42,
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
        self.image_size = tuple(image_size)
        self.image_motion_use_cache = bool(image_motion_use_cache)
        self.image_motion_write_cache = bool(image_motion_write_cache)
        self.image_motion_cache_policy = image_motion_cache_policy
        self.image_motion_cache_version = str(image_motion_cache_version)
        self.image_motion_gaussian_sigma = float(image_motion_gaussian_sigma)
        self.image_motion_threshold_ratio = float(image_motion_threshold_ratio)
        self.image_motion_threshold_strategy = str(image_motion_threshold_strategy)
        self.image_motion_grayscale = str(image_motion_grayscale)
        self.image_motion_cache_dir: Path | None = None
        self.fft_tuple = tuple(fft_tuple)
        self.clipped_range = clipped_range
        self.split = split
        self.enabled_modalities = self._resolve_enabled_modalities(enabled_modalities, use_gps, use_lidar, use_mmwave)
        if "image" in self.enabled_modalities:
            self.image_motion_cache_dir = self._resolve_image_motion_cache_dir(image_motion_cache_dir)
        self.beam_label_cache_mode = self._resolve_beam_label_cache(beam_label_cache)
        self._beam_label_cache: dict[str, int] = {}
        self.use_gps = "gps" in self.enabled_modalities
        self.gps_feature_mode = gps_feature_mode
        self.gps_normalize = gps_normalize
        self.gps_scaler = gps_scaler
        self._gps_feature_cache: dict[int, np.ndarray] = {}
        self.use_mmwave = "mmwave" in self.enabled_modalities
        self.mmwave_normalize = bool(mmwave_normalize)
        self.mmwave_scaler = mmwave_scaler
        self._mmwave_feature_cache: dict[int, np.ndarray] = {}
        self.use_lidar = "lidar" in self.enabled_modalities
        self.lidar_bev_size = tuple(lidar_bev_size)
        self.lidar_roi = tuple(lidar_roi)
        self.lidar_fov_degrees = tuple(lidar_fov_degrees) if lidar_fov_degrees is not None else None
        self.lidar_remove_ground = lidar_remove_ground
        self.lidar_ground_z_threshold = lidar_ground_z_threshold
        self.lidar_background_distance_threshold = lidar_background_distance_threshold
        self.lidar_background_path = lidar_background_path
        self.lidar_use_cache = bool(lidar_use_cache)
        self.lidar_write_cache = bool(lidar_write_cache)
        self.lidar_cache_policy = lidar_cache_policy
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
        self.lidar_cache_dir = self._resolve_lidar_cache_dir(lidar_cache_dir) if self.use_lidar else None
        self.lidar_background_points = (
            load_lidar_background_points(self.data_root, lidar_background_path) if self.use_lidar else None
        )
        self._lidar_bev_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.transform = build_image_transform(image_size) if "image" in self.enabled_modalities else None
        self.samples = create_samples(
            self.root_csv,
            portion=portion,
            enabled_modalities=self.enabled_modalities,
            seq_len=seq_len,
            num_pred=num_pred,
            portion_strategy=portion_strategy,
            portion_seed=portion_seed,
        )
        if self.beam_label_cache_mode == "eager":
            self._prepare_beam_label_cache()
        if self.use_gps:
            self._ensure_gps_columns()
            self._prepare_gps_scaler()
        if self.use_mmwave:
            self._ensure_mmwave_columns()
            self._prepare_mmwave_scaler()
        if self.use_lidar:
            self._ensure_lidar_columns()
            self._prepare_lidar_normalizer_from_config()

    def __len__(self) -> int:
        return len(self.samples.input_beam_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        beam_paths = self.samples.input_beam_paths[idx][-self.seq_len :]
        future_beam_paths = self.samples.future_beam_paths[idx][: self.num_pred]

        input_beam = [
            self._beam_label(beam_path)
            for beam_path in beam_paths
        ]
        target_beam = [
            self._beam_label(beam_path)
            for beam_path in future_beam_paths
        ]
        sample = {
            "input_beam": torch.tensor(input_beam, dtype=torch.int64),
            "target_beam": torch.tensor(target_beam, dtype=torch.int64),
        }
        if "image" in self.enabled_modalities:
            if self.transform is None:
                raise ValueError("Image modality is enabled but image transform is unavailable.")
            samples_rgb = self.samples.rgb_paths[idx][-self.seq_len :]
            sample["image"] = load_motion_masks(
                self.data_root,
                samples_rgb,
                self.seq_len,
                self.transform,
                image_size=self.image_size,
                gaussian_sigma=self.image_motion_gaussian_sigma,
                threshold_ratio=self.image_motion_threshold_ratio,
                threshold_strategy=self.image_motion_threshold_strategy,
                grayscale=self.image_motion_grayscale,
                cache_dir=self.image_motion_cache_dir,
                use_cache=self.image_motion_use_cache,
                write_cache=self.image_motion_write_cache,
            )
        if "radar" in self.enabled_modalities:
            samples_radar = self.samples.radar_paths[idx][-self.seq_len :]
            radar_ra, radar_da = load_radar_maps(
                self.data_root,
                samples_radar,
                self.seq_len,
                self.fft_tuple,
                self.clipped_range,
            )
            sample["radar_ra"] = radar_ra
            sample["radar_da"] = radar_da
        if self.use_gps:
            gps_features = self._gps_features_for_index(idx)
            if self.gps_scaler is not None:
                gps_features = self.gps_scaler.transform(gps_features)
            sample["gps"] = torch.tensor(gps_features, dtype=torch.float32)
        if self.use_mmwave:
            mmwave_features = self._mmwave_features_for_index(idx)
            if self.mmwave_scaler is not None:
                mmwave_features = self.mmwave_scaler.transform(mmwave_features)
            sample["mmwave"] = torch.tensor(mmwave_features, dtype=torch.float32)
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

    def _resolve_enabled_modalities(
        self,
        enabled_modalities: list[str] | tuple[str, ...] | None,
        use_gps: bool,
        use_lidar: bool,
        use_mmwave: bool,
    ) -> tuple[str, ...]:
        if enabled_modalities is None:
            selected = ["image", "radar"]
            if use_gps:
                selected.append("gps")
            if use_lidar:
                selected.append("lidar")
            if use_mmwave:
                selected.append("mmwave")
        else:
            selected = [str(modality) for modality in enabled_modalities]
        if not selected:
            raise ValueError("Scenario9Dataset requires at least one enabled modality.")
        invalid = [name for name in selected if name not in VALID_MODALITIES]
        if invalid:
            raise ValueError(f"Unknown Scenario 9 modalities: {invalid}.")
        if len(set(selected)) != len(selected):
            raise ValueError(f"Scenario 9 modalities must not contain duplicates: {selected}.")
        return tuple(name for name in VALID_MODALITIES if name in set(selected))

    def _resolve_image_motion_cache_dir(self, image_motion_cache_dir: str | None) -> Path | None:
        if not image_motion_cache_dir:
            return None
        path = Path(image_motion_cache_dir).expanduser()
        base = path if path.is_absolute() else self.data_root / path
        return parameterized_image_motion_cache_dir(
            base,
            image_size=self.image_size,
            gaussian_sigma=self.image_motion_gaussian_sigma,
            threshold_ratio=self.image_motion_threshold_ratio,
            threshold_strategy=self.image_motion_threshold_strategy,
            grayscale=self.image_motion_grayscale,
            cache_version=self.image_motion_cache_version,
        )

    def _resolve_beam_label_cache(self, config: bool | str) -> str:
        if isinstance(config, bool):
            return "eager" if config else "off"
        mode = str(config).lower()
        if mode in {"true", "yes", "on"}:
            return "eager"
        if mode in {"false", "no", "off", "none"}:
            return "off"
        if mode not in {"eager", "lazy"}:
            raise ValueError("beam_label_cache must be one of eager, lazy, off, true, or false.")
        return mode

    def _prepare_beam_label_cache(self) -> None:
        unique_paths = {
            str(path)
            for paths in [*self.samples.input_beam_paths, *self.samples.future_beam_paths]
            for path in paths
            if str(path).strip() and str(path).strip() != "-99"
        }
        for beam_path in sorted(unique_paths):
            self._beam_label_cache[beam_path] = self._read_beam_label(beam_path)

    def _beam_label(self, beam_path: str) -> int:
        key = str(beam_path)
        if self.beam_label_cache_mode != "off" and key in self._beam_label_cache:
            return self._beam_label_cache[key]
        label = self._read_beam_label(key)
        if self.beam_label_cache_mode != "off":
            self._beam_label_cache[key] = label
        return label

    def _read_beam_label(self, beam_path: str) -> int:
        path = joined_resource(self.data_root, beam_path)
        try:
            values = np.loadtxt(path)
        except Exception as exc:
            raise ValueError(f"Failed to read beam label file {path}: {exc}") from exc
        values = np.asarray(values)
        if values.size == 0:
            raise ValueError(f"Beam label file {path} is empty.")
        return int(np.argmax(values))

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
            )
        return self._gps_feature_cache[idx]

    def _ensure_mmwave_columns(self) -> None:
        if self.samples.mmwave_paths is None:
            raise ValueError(
                f"mmWave is enabled but {self.root_csv} does not contain mmwave1..mmwaveN columns. "
                "Regenerate sequence CSVs with include_mmwave: true."
            )

    def _prepare_mmwave_scaler(self) -> None:
        if not self.mmwave_normalize:
            self.mmwave_scaler = None
            return
        if self.mmwave_scaler is not None:
            return
        if self.split != "train":
            raise ValueError(
                "mmWave normalization for non-train split requires a train-fitted mmwave_scaler. "
                "Use build_dataloaders/evaluate with a checkpoint that records mmwave_scaler, "
                "provide mmwave_scaler explicitly, or disable mmWave normalization."
            )
        all_features = [self._mmwave_features_for_index(idx) for idx in range(len(self))]
        stacked = np.concatenate(all_features, axis=0)
        self.mmwave_scaler = MmWaveStandardScaler().fit(stacked)

    def _mmwave_features_for_index(self, idx: int) -> np.ndarray:
        if idx not in self._mmwave_feature_cache:
            if self.samples.mmwave_paths is None:
                raise ValueError("mmWave paths are unavailable for this dataset.")
            self._mmwave_feature_cache[idx] = load_mmwave_feature_sequence(
                self.data_root,
                self.samples.mmwave_paths[idx],
                seq_len=self.seq_len,
                expected_dim=MMWAVE_POWER_DIM,
            )
        return self._mmwave_feature_cache[idx]

    def _resolve_lidar_cache_dir(self, lidar_cache_dir: str | None) -> Path | None:
        if not lidar_cache_dir:
            return None
        path = Path(lidar_cache_dir).expanduser()
        if path.is_absolute():
            base = path
        else:
            base = self.data_root / path
        return parameterized_lidar_cache_dir(
            base,
            bev_size=self.lidar_bev_size,
            roi=self.lidar_roi,
            fov_degrees=self.lidar_fov_degrees,
            remove_ground=self.lidar_remove_ground,
            ground_z_threshold=self.lidar_ground_z_threshold,
            background_path=self.lidar_background_path,
            background_distance_threshold=self.lidar_background_distance_threshold,
        )

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
