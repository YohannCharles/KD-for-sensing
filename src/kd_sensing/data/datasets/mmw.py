from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from kd_sensing.data.datasets.mmw_family_adapter import MMWFamilyAdapter, prepare_mmw_family_init
from kd_sensing.data.samples import create_samples
from kd_sensing.data.transform_ops.gps import GPSStandardScaler, load_gps_feature_sequence
from kd_sensing.data.transform_ops.image import build_rgb_imagenet_transform, load_rgb_imagenet_frames
from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.data.transform_ops.lidar import LidarBEVNormalizer, load_lidar_bev_sequence
from kd_sensing.data.transform_ops.radar import load_radar_maps
from kd_sensing.modalities import normalize_modalities, resolve_image_profile
from kd_sensing.registries import DATASETS


@DATASETS.register("mmw")
class MMWDataset(Dataset):
    """Four-sensor MMW sequence dataset used by T2 and its retained baselines."""

    def __init__(
        self,
        *,
        condition: str = "sunny",
        scene: str | None = "town10_skybridge_seed24",
        scene_id: str | None = None,
        scene_slug: str | None = None,
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
        lidar_normalize: bool = False,
        lidar_normalizer: LidarBEVNormalizer | None = None,
        lidar_augment: bool = False,
        lidar_point_dropout: float = 0.0,
        lidar_jitter_std: float = 0.0,
        enabled_modalities: list[str] | tuple[str, ...] | None = None,
        beam_label_calibration: bool | dict[str, Any] | None = None,
        return_metadata: bool = False,
        **extra: Any,
    ) -> None:
        self._reject_retired_inputs(extra)
        init = prepare_mmw_family_init(
            condition=condition,
            scene=scene,
            scene_id=scene_id,
            scene_slug=scene_slug,
            data_root=data_root,
            train_csv_name=train_csv_name,
            test_csv_name=test_csv_name,
            val_csv_name=val_csv_name,
            beam_label_calibration=beam_label_calibration,
            kwargs={},
        )
        self.condition = init.condition
        self.scene_id = init.scenario
        self.scene_slug = init.scenario
        self.data_root = Path(init.root)
        self.split = "validation" if split == "val" else str(split)
        csv_name = {
            "train": init.train_csv_name,
            "validation": init.val_csv_name or init.test_csv_name,
            "test": init.test_csv_name,
        }.get(self.split)
        if csv_name is None:
            raise ValueError(f"Unsupported MMW split: {split}.")
        self.root_csv = joined_resource(self.data_root, csv_name)
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
            raise ValueError(f"MMW T2 surface supports only image/radar/gps/lidar, got {unsupported}.")
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
        self._gps_feature_cache: dict[str, np.ndarray] = {}
        self.lidar_bev_size = tuple(int(value) for value in lidar_bev_size)
        self.lidar_roi = tuple(float(value) for value in lidar_roi)
        self.lidar_fov_degrees = tuple(lidar_fov_degrees) if lidar_fov_degrees is not None else None
        self.lidar_remove_ground = bool(lidar_remove_ground)
        self.lidar_ground_z_threshold = float(lidar_ground_z_threshold)
        self.lidar_background_distance_threshold = float(lidar_background_distance_threshold)
        self.lidar_normalize = bool(lidar_normalize)
        self.lidar_normalizer = lidar_normalizer
        self.lidar_augment = bool(lidar_augment)
        self.lidar_point_dropout = float(lidar_point_dropout)
        self.lidar_jitter_std = float(lidar_jitter_std)
        self.lidar_stats_path = None
        self.return_metadata = bool(return_metadata)
        self.beam_label_mapping = init.beam_label_mapping
        self._beam_label_cache: dict[str, int] = {}
        self.samples = create_samples(
            self.root_csv,
            portion=float(portion),
            portion_strategy=portion_strategy,
            portion_seed=int(portion_seed),
            enabled_modalities=self.enabled_modalities,
            seq_len=self.seq_len,
            gps_source_seq_len=self.seq_len,
            num_pred=self.num_pred,
        )
        self.family_adapter = MMWFamilyAdapter(self, condition=init.condition, scenario=init.scenario)
        self.schema_identity = {
            "dataset_family": "MMW",
            "modalities": list(self.enabled_modalities),
            "seq_len": self.seq_len,
            "num_pred": self.num_pred,
        }

    @staticmethod
    def _reject_retired_inputs(extra: dict[str, Any]) -> None:
        retired = {
            key: value
            for key, value in extra.items()
            if key in {"use_csi", "use_mmwave", "physics_supervision", "radio_semantic", "path_semantic", "physical_label"}
            and bool(value)
        }
        if retired:
            raise ValueError(f"Retired MMW inputs are not supported: {', '.join(sorted(retired))}.")

    def __len__(self) -> int:
        return len(self.samples.input_beam_paths)

    @property
    def needs_lidar_streaming_stats(self) -> bool:
        return bool(self.use_lidar and self.lidar_normalize and self.lidar_normalizer is None)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        beam_paths = self.samples.input_beam_paths[idx][-self.seq_len :]
        target_paths = self.samples.future_beam_paths[idx][: self.num_pred]
        raw_input = [self._raw_beam_label(path) for path in beam_paths]
        raw_target = [self.family_adapter.target_raw_beam_label_for_index(idx, horizon, path) for horizon, path in enumerate(target_paths)]
        sample: dict[str, Any] = {
            "input_beam": torch.tensor([self._map_beam_label(value) for value in raw_input], dtype=torch.long),
            "target_beam": torch.tensor([self._map_beam_label(value) for value in raw_target], dtype=torch.long),
            "history_indices": torch.arange(len(beam_paths), dtype=torch.long),
            "target_index": torch.tensor(len(beam_paths), dtype=torch.long),
        }
        if "image" in self.enabled_modalities:
            sample["image"] = load_rgb_imagenet_frames(
                self.data_root,
                self.samples.rgb_paths[idx][-self.seq_len :],
                self.seq_len,
                self.transform,
                image_size=self.image_size,
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
            sample["lidar_raw"] = torch.tensor(lidar, dtype=torch.float32)
            if self.lidar_normalizer is not None:
                lidar = self.lidar_normalizer.transform(lidar)
            sample["lidar"] = torch.tensor(lidar, dtype=torch.float32)
        return self.family_adapter.augment_sample(idx, sample)

    def _raw_beam_label(self, beam_path: str) -> int:
        key = str(beam_path)
        if key not in self._beam_label_cache:
            path = joined_resource(self.data_root, key)
            try:
                values = np.asarray(np.loadtxt(path))
            except Exception as exc:
                raise ValueError(f"Failed to read MMW beam label file {path}: {exc}") from exc
            if values.size == 0:
                raise ValueError(f"MMW beam label file {path} is empty.")
            self._beam_label_cache[key] = int(np.argmax(values))
        return self._beam_label_cache[key]

    def _map_beam_label(self, raw_label: int) -> int:
        return self.beam_label_mapping.map_label(int(raw_label))

    def _gps_features_for_index(self, idx: int) -> np.ndarray:
        if self.samples.gps_paths is None or self.samples.bs_gps_paths is None:
            raise ValueError("MMW GPS paths are unavailable for an enabled GPS modality.")
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
        )


__all__ = ["MMWDataset"]
