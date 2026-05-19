from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from kd_sensing.data.samples import create_samples
from kd_sensing.data.scenes import resolve_deepsense_scene
from kd_sensing.data.transform_ops.gps import (
    GPSStandardScaler,
    PositionTargetStandardScaler,
    SUPPORTED_GPS_FEATURE_MODE,
    load_gps_feature_sequence,
)
from kd_sensing.data.transform_ops.csi import CSIRMSNormalizer, load_csi_sequence
from kd_sensing.data.transform_ops.image import (
    build_rgb_imagenet_transform,
    load_rgb_imagenet_frames,
)
from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.data.transform_ops.lidar import (
    DEFAULT_LIDAR_BEV_SIZE,
    DEFAULT_LIDAR_ROI,
    LidarBEVNormalizer,
    LidarBEVStreamingStats,
    load_lidar_background_points,
    load_lidar_bev_sequence,
    parameterized_lidar_cache_dir,
)
from kd_sensing.data.transform_ops.mmwave import (
    MMWAVE_POWER_DIM,
    MmWaveStandardScaler,
    OcclusionTargetStats,
    load_mmwave_feature_sequence,
)
from kd_sensing.data.transform_ops.radar import load_radar_maps
from kd_sensing.data.datasets.deepsense6g_loaders import DeepSense6GModalityLoader
from kd_sensing.data.datasets.deepsense6g_targets import (
    DeepSense6GTargetProvider,
    coerce_occlusion_stats,
    resolve_occlusion_target_config,
    resolve_position_target_config,
)
from kd_sensing.modalities import MODALITY_ORDER, image_profile_spec, normalize_modalities, resolve_image_profile
from kd_sensing.registries import DATASETS
from kd_sensing.utils.paths import resolve_path


VALID_MODALITIES = MODALITY_ORDER
REMOVED_IMAGE_OPTION_PREFIX = "image_" + "motion_"


@DATASETS.register("deepsense6g")
class DeepSense6GDataset(Dataset):
    """DeepSense6G sequence dataset with standardized batch field names."""

    def __init__(
        self,
        data_root: str | None = None,
        csv_name: str | None = None,
        root_csv: str | None = None,
        split: str = "train",
        train_csv_name: str | None = "train_seqs_RA_GPS_LIDAR.csv",
        val_csv_name: str | None = None,
        test_csv_name: str | None = "test_seqs_RA_GPS_LIDAR.csv",
        scene: str | int | None = None,
        scene_id: str | int | None = None,
        scene_slug: str | None = None,
        seq_len: int = 8,
        num_pred: int = 3,
        image_profile: str | None = None,
        image_size: list[int] | tuple[int, int] = (224, 224),
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
        use_csi: bool = False,
        csi_train_rms: bool = True,
        csi_rms_normalizer: CSIRMSNormalizer | float | dict[str, Any] | None = None,
        occlusion_target: bool | dict[str, Any] | None = None,
        occlusion_target_stats: OcclusionTargetStats | dict[str, Any] | None = None,
        position_target: bool | dict[str, Any] | None = None,
        position_target_scaler: PositionTargetStandardScaler | None = None,
        use_lidar: bool = False,
        lidar_encoding: str = "bev",
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
        return_metadata: bool = False,
        portion_strategy: str = "even",
        portion_seed: int = 42,
        **extra: object,
    ):
        removed_keys = sorted(key for key in extra if str(key).startswith(REMOVED_IMAGE_OPTION_PREFIX))
        if removed_keys:
            keys = ", ".join(str(key) for key in removed_keys)
            raise ValueError(
                f"Removed image motion dataset option(s): {keys}. "
                "Use RGB/ImageNet image input; image motion cache is no longer supported."
            )
        scene_value = scene if scene is not None else scene_id if scene_id is not None else scene_slug
        self.scene = resolve_deepsense_scene(scene_value)
        self.scene_id = self.scene.scene_id
        self.scene_slug = self.scene.scene_slug
        if data_root is None:
            data_root = self.scene.default_data_root
        self.data_root = resolve_path(data_root)
        selected_csv = root_csv or csv_name
        if selected_csv is None:
            if split == "train":
                default_csv = self.scene.default_train_csv_name
                configured_csv = train_csv_name
            elif split in {"val", "validation"}:
                default_csv = val_csv_name or test_csv_name or self.scene.default_test_csv_name
                configured_csv = val_csv_name or test_csv_name
            else:
                default_csv = self.scene.default_test_csv_name
                configured_csv = test_csv_name
            selected_csv = configured_csv or default_csv
        self.root_csv = Path(selected_csv)
        if not self.root_csv.is_absolute():
            self.root_csv = self.data_root / self.root_csv
        self.seq_len = seq_len
        self.num_pred = num_pred
        self.image_profile = resolve_image_profile(image_profile)
        self.image_profile_spec = image_profile_spec(self.image_profile)
        self.image_size = tuple(image_size)
        self.fft_tuple = tuple(fft_tuple)
        self.clipped_range = clipped_range
        self.split = split
        self.enabled_modalities = self._resolve_enabled_modalities(
            enabled_modalities,
            use_gps,
            use_lidar,
            use_mmwave,
            use_csi,
        )
        self.return_metadata = bool(return_metadata)
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
        self.use_csi = "csi" in self.enabled_modalities
        self.csi_train_rms = bool(csi_train_rms)
        self.csi_rms_normalizer = self._coerce_csi_rms_normalizer(csi_rms_normalizer)
        self._csi_cache: dict[int, np.ndarray] = {}
        self.occlusion_target_config = resolve_occlusion_target_config(occlusion_target)
        self.occlusion_target_enabled = bool(self.occlusion_target_config["enabled"])
        self.position_target_config = resolve_position_target_config(position_target)
        self.position_target_enabled = bool(self.position_target_config["enabled"])
        self.position_target_source = str(self.position_target_config["source"])
        self.position_target_normalize = bool(self.position_target_config["normalize"])
        self.use_lidar = "lidar" in self.enabled_modalities
        self.lidar_encoding = str(lidar_encoding)
        if self.lidar_encoding != "bev":
            raise ValueError("lidar_encoding must be 'bev'.")
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
        self.transform = self._build_image_transform(image_size) if "image" in self.enabled_modalities else None
        self.samples = create_samples(
            self.root_csv,
            portion=portion,
            enabled_modalities=self.enabled_modalities,
            seq_len=seq_len,
            num_pred=num_pred,
            portion_strategy=portion_strategy,
            portion_seed=portion_seed,
            include_position_targets=(
                self.position_target_enabled and self.position_target_source == "future_gps_local_xy"
            ),
            include_history_position_targets=(
                self.position_target_enabled and self.position_target_source == "last_input_gps_local_xy"
            ),
        )
        self.target_provider = DeepSense6GTargetProvider(
            self,
            occlusion_stats=occlusion_target_stats,
            position_scaler=position_target_scaler,
        )
        self.modality_loader = DeepSense6GModalityLoader(self)
        if self.beam_label_cache_mode == "eager":
            self._prepare_beam_label_cache()
        if self.occlusion_target_enabled:
            self._prepare_occlusion_target_stats()
        if self.use_gps:
            self._ensure_gps_columns()
            self._prepare_gps_scaler()
        if self.position_target_enabled:
            self._ensure_position_target_columns()
            self._prepare_position_target_scaler()
        if self.use_mmwave:
            self._ensure_mmwave_columns()
            self._prepare_mmwave_scaler()
        if self.use_csi:
            self._ensure_csi_columns()
            self._prepare_csi_rms_normalizer()
        if self.use_lidar:
            self._ensure_lidar_columns()
            self._prepare_lidar_normalizer_from_config()

    def __len__(self) -> int:
        return len(self.samples.input_beam_paths)

    @property
    def occlusion_target_stats(self) -> OcclusionTargetStats | None:
        return self.target_provider.occlusion_target_stats

    @occlusion_target_stats.setter
    def occlusion_target_stats(self, value: OcclusionTargetStats | dict[str, Any] | None) -> None:
        self.target_provider.occlusion_target_stats = coerce_occlusion_stats(value)

    @property
    def position_target_scaler(self) -> PositionTargetStandardScaler | None:
        return self.target_provider.position_target_scaler

    @position_target_scaler.setter
    def position_target_scaler(self, value: PositionTargetStandardScaler | None) -> None:
        self.target_provider.position_target_scaler = value

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample, _ = self._getitem_with_timing(idx, collect_timing=False)
        return sample

    def profile_getitem_components(self, idx: int) -> dict[str, float]:
        _, timings = self._getitem_with_timing(idx, collect_timing=True)
        return timings

    def _getitem_with_timing(self, idx: int, *, collect_timing: bool) -> tuple[dict[str, Any], dict[str, float]]:
        timings = {
            name: 0.0
            for name in ("targets", "auxiliary_targets", "image", "radar", "gps", "mmwave", "csi", "lidar")
        }

        def record(name: str, fn):
            if not collect_timing:
                return fn()
            start = time.perf_counter()
            try:
                return fn()
            finally:
                timings[name] += time.perf_counter() - start

        beam_paths = self.samples.input_beam_paths[idx][-self.seq_len :]
        future_beam_paths = self.samples.future_beam_paths[idx][: self.num_pred]

        def build_beam_targets() -> tuple[list[int], list[int]]:
            input_beam = [self._beam_label(beam_path) for beam_path in beam_paths]
            target_beam = [self._beam_label(beam_path) for beam_path in future_beam_paths]
            return input_beam, target_beam

        input_beam, target_beam = record("targets", build_beam_targets)
        sample = {
            "input_beam": torch.tensor(input_beam, dtype=torch.int64),
            "target_beam": torch.tensor(target_beam, dtype=torch.int64),
        }

        def build_auxiliary_targets() -> dict[str, torch.Tensor]:
            values: dict[str, torch.Tensor] = {}
            if self.occlusion_target_enabled:
                occlusion_label, occlusion_valid = self.target_provider.occlusion_targets_for_paths(future_beam_paths)
                values["occlusion_label"] = torch.tensor(occlusion_label, dtype=torch.float32)
                values["occlusion_valid"] = torch.tensor(occlusion_valid, dtype=torch.bool)
            if self.position_target_enabled:
                position_target, position_valid = self.target_provider.position_targets_for_index(idx)
                if self.position_target_scaler is not None and self.position_target_normalize:
                    scaled = position_target.copy()
                    valid = position_valid.astype(bool)
                    if np.any(valid):
                        scaled[valid] = self.position_target_scaler.transform(scaled[valid])
                    position_target = scaled
                values["position_target"] = torch.tensor(position_target, dtype=torch.float32)
                values["position_valid"] = torch.tensor(position_valid, dtype=torch.bool)
            return values

        if self.occlusion_target_enabled or self.position_target_enabled:
            sample.update(record("auxiliary_targets", build_auxiliary_targets))
        if "image" in self.enabled_modalities:
            sample["image"] = record("image", lambda: self.modality_loader.load_image(idx))
        if "radar" in self.enabled_modalities:
            radar_ra, radar_da = record("radar", lambda: self.modality_loader.load_radar(idx))
            sample["radar_ra"] = radar_ra
            sample["radar_da"] = radar_da
        if self.use_gps:
            sample["gps"] = record("gps", lambda: self.modality_loader.load_gps(idx))
        if self.use_mmwave:
            sample["mmwave"] = record("mmwave", lambda: self.modality_loader.load_mmwave(idx))
        if self.use_csi:
            sample["csi"] = record("csi", lambda: self.modality_loader.load_csi(idx))
        if self.use_lidar:
            lidar_raw, lidar_model_input = record("lidar", lambda: self.modality_loader.load_lidar_pair(idx))
            sample["lidar_raw"] = lidar_raw
            sample["lidar"] = lidar_model_input
        if self.return_metadata:
            sample["metadata"] = self._metadata_for_index(idx, beam_paths, future_beam_paths)
        return sample, timings

    def _metadata_for_index(self, idx: int, beam_paths: list[str], future_beam_paths: list[str]) -> dict[str, Any]:
        last_beam_path = str(beam_paths[-1]) if beam_paths else ""
        first_future_beam_path = str(future_beam_paths[0]) if future_beam_paths else ""
        key = "|".join(
            [
                self.scene_slug,
                self.split,
                str(idx),
                last_beam_path,
                first_future_beam_path,
            ]
        )
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        metadata: dict[str, Any] = {
            "dataset_index": int(idx),
            "sample_id": f"{self.scene_slug}:{self.split}:{idx}:{digest}",
            "scene_id": self.scene_id,
            "scene_slug": self.scene_slug,
            "split": self.split,
            "root_csv": str(self.root_csv),
            "input_beam_path": last_beam_path,
            "target_beam_path": first_future_beam_path,
        }
        if "image" in self.enabled_modalities:
            metadata["image_profile"] = self.image_profile
            metadata["processed_image_source"] = "rgb_imagenet"
        self._add_path_metadata(metadata, "image_path", getattr(self.samples, "rgb_paths", None), idx)
        self._add_path_metadata(metadata, "radar_path", getattr(self.samples, "radar_paths", None), idx)
        self._add_path_metadata(metadata, "gps_path", getattr(self.samples, "gps_paths", None), idx)
        self._add_path_metadata(metadata, "lidar_path", getattr(self.samples, "lidar_paths", None), idx)
        self._add_path_metadata(metadata, "mmwave_path", getattr(self.samples, "mmwave_paths", None), idx)
        self._add_path_metadata(metadata, "csi_path", getattr(self.samples, "csi_paths", None), idx)
        seq_id, frame_idx = self._parse_sequence_position(
            first_future_beam_path or last_beam_path or metadata.get("mmwave_path", "")
        )
        if seq_id is not None:
            metadata["seq_id"] = seq_id
        if frame_idx is not None:
            metadata["frame_idx"] = int(frame_idx)
        return metadata

    @staticmethod
    def _add_path_metadata(metadata: dict[str, Any], key: str, paths: list[list[str]] | None, idx: int) -> None:
        if not paths or idx >= len(paths) or not paths[idx]:
            return
        metadata[key] = str(paths[idx][-1])

    @staticmethod
    def _parse_sequence_position(path: str) -> tuple[str | None, int | None]:
        text = str(path)
        seq_id = None
        frame_idx = None
        seq_match = re.search(r"(?:^|[/_-])seq(?:uence)?[_-]?([A-Za-z0-9]+)", text, flags=re.IGNORECASE)
        if seq_match:
            seq_id = seq_match.group(1)
        frame_match = re.search(
            r"(?:frame|frm|camera|radar|beam|gps|lidar|mmwave|pwr)[_-]?(\d+)",
            Path(text).stem,
            flags=re.IGNORECASE,
        )
        if frame_match:
            frame_idx = int(frame_match.group(1))
        return seq_id, frame_idx

    def _resolve_enabled_modalities(
        self,
        enabled_modalities: list[str] | tuple[str, ...] | None,
        use_gps: bool,
        use_lidar: bool,
        use_mmwave: bool,
        use_csi: bool,
    ) -> tuple[str, ...]:
        if enabled_modalities is None:
            selected = ["image", "radar"]
            if use_gps:
                selected.append("gps")
            if use_lidar:
                selected.append("lidar")
            if use_mmwave:
                selected.append("mmwave")
            if use_csi:
                selected.append("csi")
        else:
            selected = [str(modality) for modality in enabled_modalities]
        return normalize_modalities(selected, context="DeepSense6G modalities")

    def _build_image_transform(self, image_size: list[int] | tuple[int, int]):
        return build_rgb_imagenet_transform(image_size)

    def _load_rgb_imagenet_frames(self, paths: list[str]) -> torch.Tensor:
        return load_rgb_imagenet_frames(
            self.data_root,
            paths,
            self.seq_len,
            self.transform,
            image_size=self.image_size,
        )

    def _load_radar_maps(self, paths: list[str]):
        return load_radar_maps(
            self.data_root,
            paths,
            self.seq_len,
            self.fft_tuple,
            self.clipped_range,
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

    def _prepare_occlusion_target_stats(self) -> None:
        self.target_provider.prepare_occlusion_target_stats()

    def _occlusion_targets_for_paths(self, future_beam_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
        return self.target_provider.occlusion_targets_for_paths(future_beam_paths)

    def _max_power_for_path(self, rel_path: str) -> float | None:
        return self.target_provider.max_power_for_path(rel_path)

    def _ensure_position_target_columns(self) -> None:
        self.target_provider.ensure_position_target_columns()

    def _prepare_position_target_scaler(self) -> None:
        self.target_provider.prepare_position_target_scaler()

    def _position_targets_for_index(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.target_provider.position_targets_for_index(idx)

    @staticmethod
    def _valid_resource_path(path: object) -> bool:
        text = str(path).strip()
        return bool(text) and text != "-99"

    def auxiliary_target_metadata(self) -> dict[str, Any]:
        return self.target_provider.auxiliary_target_metadata()

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

    def _ensure_csi_columns(self) -> None:
        if self.samples.csi_paths is None:
            raise ValueError(
                f"CSI is enabled but {self.root_csv} does not contain csi1..csiN columns. "
                "Regenerate sequence CSVs with CSI export enabled."
            )

    def _prepare_csi_rms_normalizer(self) -> None:
        if not self.csi_train_rms:
            self.csi_rms_normalizer = None
            return
        if self.csi_rms_normalizer is not None:
            return
        if self.split != "train":
            raise ValueError(
                "CSI RMS normalization for non-train split requires a train-fitted csi_rms_normalizer. "
                "Use build_dataloaders/evaluate so the train normalizer can be reused."
            )
        stats = None
        from kd_sensing.data.transform_ops.csi import CSIRMSStreamingStats

        stats = CSIRMSStreamingStats()
        for idx in range(len(self)):
            stats.update(self._csi_for_index(idx))
        self.csi_rms_normalizer = stats.finalize()

    def _csi_for_index(self, idx: int) -> np.ndarray:
        if idx not in self._csi_cache:
            if self.samples.csi_paths is None:
                raise ValueError("CSI paths are unavailable for this dataset.")
            self._csi_cache[idx] = load_csi_sequence(
                self.data_root,
                self.samples.csi_paths[idx],
                seq_len=self.seq_len,
            )
        return self._csi_cache[idx]

    @staticmethod
    def _coerce_csi_rms_normalizer(value: CSIRMSNormalizer | float | dict[str, Any] | None) -> CSIRMSNormalizer | None:
        if value is None:
            return None
        if isinstance(value, CSIRMSNormalizer):
            return value
        if isinstance(value, dict):
            return CSIRMSNormalizer(rms=float(value["rms"]), sample_count=int(value.get("sample_count", 0)))
        return CSIRMSNormalizer(rms=float(value), sample_count=0)

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
        enabled_flag: bool,
        config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if config is None:
            enabled = bool(enabled_flag)
            mode = "streaming_stats" if enabled else "none"
            stats_path = None
            recompute = False
        else:
            enabled = bool(config.get("enabled", False))
            if bool(enabled_flag) != enabled:
                raise ValueError(
                    "Conflicting LiDAR normalization config: "
                    f"lidar_normalize={bool(enabled_flag)!r} but "
                    f"lidar_normalization.enabled={enabled!r}. "
                    "Choose raw BEV with both fields disabled, or choose an explicit "
                    "streaming stats profile with both fields enabled."
                )
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
