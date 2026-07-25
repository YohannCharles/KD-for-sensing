import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from kd_sensing.data.layouts import mmw_condition_layout
from kd_sensing.data.samples import create_samples
from kd_sensing.data.transform_ops.gps import GPSStandardScaler, load_gps_coordinate_cache, load_gps_feature_sequence
from kd_sensing.data.transform_ops.image import build_rgb_imagenet_transform, load_rgb_imagenet_frames
from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.data.transform_ops.lidar import (
    load_lidar_bev_sequence,
    parameterized_lidar_cache_dir,
    validate_lidar_cache_metadata,
)
from kd_sensing.data.transform_ops.radar import load_radar_maps
from kd_sensing.modalities import normalize_modalities, resolve_image_profile
from kd_sensing.registries import DATASETS


@DATASETS.register("mmw")
class MMWDataset(Dataset):
    """Four-sensor MMW sequence dataset used by U0 and retained baselines."""

    def __init__(
        self,
        *,
        condition: str | None = None,
        scene: str | None = None,
        data_root: str | None = None,
        train_csv_name: str | None = None,
        test_csv_name: str | None = None,
        val_csv_name: str | None = None,
        split: str = "train",
        seq_len: int = 5,
        num_pred: int = 1,
        portion: float = 1.0,
        portion_strategy: str = "even",
        portion_seed: int = 42,
        image_profile: str | None = "rgb_imagenet",
        image_size: list[int] | tuple[int, int] = (224, 224),
        frame_cache_root: str | None = None,
        frame_cache_strict: bool = False,
        gps_coordinate_cache_root: str | None = None,
        fft_tuple: list[int] | tuple[int, int, int] = (64, 256, 128),
        clipped_range: int = 128,
        use_gps: bool = True,
        gps_feature_mode: str = "relative_polar",
        gps_normalize: bool = True,
        gps_scaler: GPSStandardScaler | None = None,
        use_lidar: bool = True,
        lidar_bev_size: list[int] | tuple[int, int] = (224, 224),
        lidar_roi: list[float] | tuple[float, ...] = (-30.0, 30.0, -30.0, 30.0, -3.0, 5.0),
        lidar_fov_degrees: list[float] | tuple[float, float] | None = None,
        lidar_remove_ground: bool = False,
        lidar_ground_z_threshold: float = 0.1,
        lidar_background_distance_threshold: float = 0.2,
        lidar_augment: bool = False,
        lidar_point_dropout: float = 0.0,
        lidar_jitter_std: float = 0.0,
        include_router_utility_targets: bool = False,
        include_router_corruption_metadata: bool = False,
        enabled_modalities: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if not str(condition or "").strip() or not str(scene or "").strip():
            raise ValueError("MMW condition and scene must be explicit.")
        layout = mmw_condition_layout(str(condition))
        self.condition = layout.condition
        self.scene_slug = str(scene).strip()
        self.scene_id = self.scene_slug
        self.data_root = Path(data_root or layout.root)
        self.split = str(split)
        prepared = Path("Prepared") / self.scene_slug / "splits"
        csv_name = {
            "train": train_csv_name or str(prepared / "train.csv"),
            "validation": val_csv_name,
            "test": test_csv_name or str(prepared / "test.csv"),
        }.get(self.split)
        if csv_name is None:
            raise ValueError(f"Unsupported MMW split: {split}.")
        csv_path = Path(csv_name)
        self.root_csv = csv_path if csv_path.is_absolute() else joined_resource(self.data_root, csv_name)
        if not self.root_csv.exists():
            raise FileNotFoundError(f"MMW prepared split CSV is missing: {self.root_csv}")
        self.seq_len = int(seq_len)
        self.num_pred = int(num_pred)
        if self.seq_len <= 0 or self.num_pred <= 0:
            raise ValueError("MMW seq_len and num_pred must be positive.")
        requested = normalize_modalities(
            tuple(enabled_modalities or ("image", "radar", "gps", "lidar")),
            context="MMW enabled modalities",
        )
        unsupported = sorted(set(requested) - {"image", "radar", "gps", "lidar"})
        if unsupported:
            raise ValueError(f"MMW retained surface supports only image/radar/gps/lidar, got {unsupported}.")
        self.enabled_modalities = requested
        self.use_gps = bool(use_gps and "gps" in requested)
        self.use_lidar = bool(use_lidar and "lidar" in requested)
        self.image_profile = resolve_image_profile(image_profile)
        self.image_size = tuple(int(value) for value in image_size)
        self.transform = build_rgb_imagenet_transform(self.image_size) if "image" in requested else None
        self.fft_tuple = tuple(int(value) for value in fft_tuple)
        self.clipped_range = int(clipped_range)
        self.gps_feature_mode = str(gps_feature_mode)
        self.gps_normalize = bool(gps_normalize)
        self.gps_scaler = gps_scaler
        self.include_router_utility_targets = bool(include_router_utility_targets)
        self.include_router_corruption_metadata = bool(include_router_corruption_metadata)
        self._beam_power_cache: dict[str, torch.Tensor] = {}
        self.lidar_bev_size = tuple(int(value) for value in lidar_bev_size)
        self.lidar_roi = tuple(float(value) for value in lidar_roi)
        self.lidar_fov_degrees = tuple(lidar_fov_degrees) if lidar_fov_degrees is not None else None
        self.lidar_remove_ground = bool(lidar_remove_ground)
        self.lidar_ground_z_threshold = float(lidar_ground_z_threshold)
        self.lidar_background_distance_threshold = float(lidar_background_distance_threshold)
        self.lidar_augment = bool(lidar_augment)
        self.lidar_point_dropout = float(lidar_point_dropout)
        self.lidar_jitter_std = float(lidar_jitter_std)
        self._lidar_augmentation_seed = int(portion_seed)
        self._lidar_epoch = 0
        self.lidar_stats_path = None
        self.frame_cache_strict = bool(frame_cache_strict)
        if self.frame_cache_strict and frame_cache_root is None:
            raise ValueError("frame_cache_strict=true requires frame_cache_root.")
        condition_cache_root = Path(frame_cache_root) / self.condition if frame_cache_root is not None else None
        self.image_cache_dir = condition_cache_root / "image_derived" if condition_cache_root is not None else None
        self.lidar_cache_dir = (
            parameterized_lidar_cache_dir(
                condition_cache_root / "lidar_bev",
                bev_size=self.lidar_bev_size,
                roi=self.lidar_roi,
                fov_degrees=self.lidar_fov_degrees,
                remove_ground=self.lidar_remove_ground,
                ground_z_threshold=self.lidar_ground_z_threshold,
                background_distance_threshold=self.lidar_background_distance_threshold,
            )
            if condition_cache_root is not None
            else None
        )
        if self.frame_cache_strict:
            if self.lidar_augment:
                raise ValueError("Strict LiDAR BEV cache cannot be combined with train-time point augmentation.")
            if self.image_cache_dir is None or not self.image_cache_dir.is_dir():
                raise FileNotFoundError(f"Strict RGB cache directory is missing: {self.image_cache_dir}")
            assert self.lidar_cache_dir is not None
            validate_lidar_cache_metadata(
                self.lidar_cache_dir,
                bev_size=self.lidar_bev_size,
                roi=self.lidar_roi,
                fov_degrees=self.lidar_fov_degrees,
                remove_ground=self.lidar_remove_ground,
                ground_z_threshold=self.lidar_ground_z_threshold,
                background_distance_threshold=self.lidar_background_distance_threshold,
            )
        self.gps_coordinate_cache_path = (
            Path(gps_coordinate_cache_root) / f"{self.condition}__{self.scene_slug}.npz"
            if gps_coordinate_cache_root is not None
            else None
        )
        if self.gps_coordinate_cache_path is not None:
            self._gps_persistent_coordinate_cache = load_gps_coordinate_cache(self.gps_coordinate_cache_path)
        else:
            self._gps_persistent_coordinate_cache = {}
        self._gps_feature_cache: dict[str, np.ndarray] = dict(self._gps_persistent_coordinate_cache)
        self.samples = create_samples(
            self.root_csv,
            portion=float(portion),
            data_root=self.data_root,
            portion_strategy=portion_strategy,
            portion_seed=int(portion_seed),
            enabled_modalities=self.enabled_modalities,
            seq_len=self.seq_len,
            gps_source_seq_len=self.seq_len,
            num_pred=self.num_pred,
        )
        self.schema_identity = {
            "dataset_family": "MMW",
            "modalities": list(self.enabled_modalities),
            "seq_len": self.seq_len,
            "num_pred": self.num_pred,
        }

    def __len__(self) -> int:
        return len(self.samples.rows or [])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "target_beam": torch.tensor(
                [self._target_beam_label(idx, horizon) for horizon in range(self.num_pred)],
                dtype=torch.long,
            )
        }
        if "image" in self.enabled_modalities:
            sample["image"] = load_rgb_imagenet_frames(
                self.data_root,
                self.samples.rgb_paths[idx][-self.seq_len :],
                self.seq_len,
                self.transform,
                image_size=self.image_size,
                cache_dir=self.image_cache_dir,
                strict_cache=self.frame_cache_strict,
            )
        if "radar" in self.enabled_modalities:
            sample["radar_ra"], sample["radar_da"] = load_radar_maps(
                self.data_root,
                self.samples.radar_paths[idx][-self.seq_len :],
                self.seq_len,
                self.fft_tuple,
                self.clipped_range,
            )
        if self.use_gps:
            gps = self._gps_features_for_index(idx)
            sample["gps"] = torch.tensor(self.gps_scaler.transform(gps) if self.gps_scaler is not None else gps, dtype=torch.float32)
        if self.use_lidar:
            lidar = self._lidar_bev_for_index(idx, augment=self.split == "train" and self.lidar_augment)
            sample["lidar"] = torch.tensor(lidar, dtype=torch.float32)
        if self.include_router_utility_targets:
            sample["future_beam_power"] = self._future_beam_power(idx)
        if self.include_router_utility_targets or self.include_router_corruption_metadata:
            if self.gps_scaler is None or self.gps_scaler.mean_ is None or self.gps_scaler.scale_ is None:
                raise ValueError("Router GPS metadata requires the train-fit GPS scaler.")
            sample["gps_scaler_mean"] = torch.as_tensor(self.gps_scaler.mean_, dtype=torch.float32)
            sample["gps_scaler_scale"] = torch.as_tensor(self.gps_scaler.scale_, dtype=torch.float32)
        return self._with_metadata(idx, sample)

    def _future_beam_power(self, idx: int) -> torch.Tensor:
        relative = _row_text(self._row(idx), "future_beam1")
        if relative is None:
            raise ValueError(f"MMW prepared row {idx} is missing future_beam1.")
        path = str(joined_resource(self.data_root, relative).resolve())
        cached = self._beam_power_cache.get(path)
        if cached is None:
            try:
                values = torch.as_tensor(np.loadtxt(path), dtype=torch.float32).reshape(-1)
            except Exception as exc:
                raise ValueError(f"Failed to load future beam power vector {path}: {exc}") from exc
            if values.numel() != 64 or not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
                raise ValueError(f"Future beam power must contain 64 finite non-negative values: {path}")
            cached = values
            self._beam_power_cache[path] = cached
        return cached.clone()

    def _target_beam_label(self, idx: int, horizon: int) -> int:
        label = _row_label(self._row(idx), f"future_beam_label{horizon + 1}")
        if label is None:
            raise ValueError(f"MMW prepared row {idx} is missing future_beam_label{horizon + 1}.")
        return label

    def _row(self, idx: int) -> dict[str, Any]:
        rows = self.samples.rows or []
        return rows[idx] if 0 <= idx < len(rows) else {}

    def _with_metadata(self, idx: int, sample: dict[str, Any]) -> dict[str, Any]:
        row = self._row(idx)
        metadata = {
            key: text
            for key in ("condition", "town", "sensor_scenario", "sample_id", "target_sample_id")
            if (text := _row_text(row, key)) is not None
        }
        source_sample_id = str(
            metadata.get("sample_id") or metadata.get("target_sample_id") or f"{self.scene_slug}:{idx}"
        )
        metadata.update(dataset_family="MMW", condition=self.condition, scenario=self.scene_slug)
        future_beam_path = _row_text(row, "future_beam1")
        if future_beam_path is not None:
            metadata["future_beam_path"] = str(joined_resource(self.data_root, future_beam_path).resolve())
        metadata["source_sample_id"] = source_sample_id
        metadata["stable_sample_id"] = f"mmw:{self.condition}:{self.scene_slug}:{self.split}:{source_sample_id}"
        sample["metadata"] = metadata
        sample["sample_id"] = source_sample_id
        sample["domain_metadata"] = {
            "dataset_family": "MMW",
            "condition": self.condition,
            "town": metadata.get("town", ""),
            "scenario": self.scene_slug,
            "scene_slug": self.scene_slug,
            "split": self.split,
        }
        return sample

    def _gps_features_for_index(self, idx: int) -> np.ndarray:
        if self.samples.gps_paths is None or self.samples.bs_gps_paths is None:
            raise ValueError("MMW GPS paths are unavailable for an enabled GPS modality.")
        if self.gps_coordinate_cache_path is not None and self.frame_cache_strict:
            required = self.samples.gps_paths[idx][-self.seq_len :] + self.samples.bs_gps_paths[idx][-self.seq_len :]
            missing = [path for path in required if path not in self._gps_feature_cache]
            if missing:
                raise FileNotFoundError(
                    f"Strict GPS coordinate cache miss in {self.gps_coordinate_cache_path}: {missing[0]}"
                )
        return load_gps_feature_sequence(
            self.data_root,
            self.samples.gps_paths[idx],
            self.samples.bs_gps_paths[idx],
            seq_len=self.seq_len,
            mode=self.gps_feature_mode,
            frame_feature_cache=self._gps_feature_cache,
        )

    def _lidar_bev_for_index(self, idx: int, *, augment: bool) -> np.ndarray:
        if self.samples.lidar_paths is None:
            raise ValueError("MMW LiDAR paths are unavailable for an enabled LiDAR modality.")
        return load_lidar_bev_sequence(
            self.data_root,
            self.samples.lidar_paths[idx],
            seq_len=self.seq_len,
            bev_size=self.lidar_bev_size,
            roi=self.lidar_roi,
            fov_degrees=self.lidar_fov_degrees,
            remove_ground=self.lidar_remove_ground,
            ground_z_threshold=self.lidar_ground_z_threshold,
            background_distance_threshold=self.lidar_background_distance_threshold,
            augment=augment,
            point_dropout=self.lidar_point_dropout,
            jitter_std=self.lidar_jitter_std,
            rng=self._lidar_rng(idx) if augment else None,
            cache_dir=self.lidar_cache_dir,
            strict_cache=self.frame_cache_strict,
        )

    def reset_gps_feature_cache(self) -> None:
        self._gps_feature_cache = dict(self._gps_persistent_coordinate_cache)

    def set_epoch(self, epoch: int) -> None:
        self._lidar_epoch = int(epoch)

    def _lidar_rng(self, idx: int) -> np.random.Generator:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        sample_id = str(self.samples.rows[idx].get("sample_id") or self.samples.rows[idx].get("target_sample_id") or idx)
        payload = f"{self._lidar_augmentation_seed}:{self._lidar_epoch}:{worker_id}:{self.split}:{sample_id}".encode("utf-8")
        return np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


def _row_label(row: dict[str, Any], key: str) -> int | None:
    try:
        value = int(row.get(key))
    except (TypeError, ValueError):
        return None
    return value if 0 <= value < 64 else None


def _row_text(row: dict[str, Any], key: str) -> str | None:
    value = str(row.get(key, "")).strip()
    return value if value and value not in {"-99", "nan", "None"} else None


__all__ = ["MMWDataset"]
