import hashlib
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from kd_sensing.data.samples import create_samples
from kd_sensing.data.scenes import resolve_deepsense_scene
from kd_sensing.data.sample_cache import LmdbSampleCache, sample_cache_path_for_split
from kd_sensing.data.beam_soft_targets import (
    SoftBeamLabelConfig,
    read_beam_power_vector,
    soft_distribution_from_power_or_label,
)
from kd_sensing.data.beam_label_calibration import BeamLabelMapping, resolve_beam_label_mapping
from kd_sensing.data.transform_ops.gps import (
    GPS_FEATURE_DIMS,
    GPSStandardScaler,
    PositionTargetStandardScaler,
    SUPPORTED_GPS_FEATURE_MODE,
    load_gps_feature_sequence,
    load_relative_xy_sequence,
)
from kd_sensing.data.transform_ops.csi import (
    CSIDegradationConfig,
    CSIRMSNormalizer,
    load_csi_sequence,
)
from kd_sensing.data.transform_ops.image import (
    build_rgb_imagenet_transform,
    load_rgb_imagenet_frames,
)
from kd_sensing.data.transform_ops.image_cache import (
    IMAGE_DERIVED_CACHE_VERSION,
    ImageDerivedCache,
    ImageDerivedCacheConfig,
)
from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.data.transform_ops.lidar import (
    DEFAULT_LIDAR_BEV_SIZE,
    DEFAULT_LIDAR_ROI,
    LidarBEVNormalizer,
    LidarBEVStreamingStats,
    load_lidar_bev_sequence,
)
from kd_sensing.data.transform_ops.mmwave import (
    MMWAVE_POWER_DIM,
    MmWaveStandardScaler,
    OcclusionTargetStats,
    load_mmwave_feature_sequence,
)
from kd_sensing.data.transform_ops.radar import load_radar_maps
from kd_sensing.data.datasets.deepsense6g_loaders import configure_deepsense6g_resource_readers
from kd_sensing.data.datasets.deepsense6g_targets import (
    configure_deepsense6g_target_state,
    coerce_occlusion_stats,
    prepare_deepsense6g_targets,
)
from kd_sensing.data.datasets.deepsense6g_columns import (
    ensure_enabled_contract_columns,
)
from kd_sensing.data.datasets.deepsense6g_contract import (
    add_path_metadata,
    normalize_beam_target_source,
    parse_sequence_position,
    resolve_beam_label_cache_mode,
    resolve_enabled_modalities,
    resolve_sequence_csv_path,
    resolve_target_beam_paths,
    validate_beam_target_source_contract,
)
from kd_sensing.data.datasets.deepsense6g_gps_contract import (
    normalize_gps_feature_mode,
    resolve_gps_angle_offset,
    resolve_gps_source_seq_len,
)
from kd_sensing.data.datasets.deepsense6g_sample_assembly import (
    build_auxiliary_target_tensors,
    build_beam_target_tensors,
)
from kd_sensing.data.datasets.deepsense6g_scalers import _StreamingFeatureStats
from kd_sensing.data.datasets.deepsense6g_scalers import (
    configure_deepsense6g_scaler_state,
    prepare_deepsense6g_scalers_and_normalizers,
)
from kd_sensing.modalities import MODALITY_ORDER, image_profile_spec, resolve_image_profile
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
        gps_seq_len: int | None = None,
        gps_source_seq_len: int | None = None,
        num_pred: int = 3,
        image_profile: str | None = None,
        image_size: list[int] | tuple[int, int] = (224, 224),
        image_cache_dir: str | None = None,
        image_use_cache: bool = False,
        image_write_cache: bool = False,
        image_cache_policy: str | None = None,
        image_cache_transform_version: str = IMAGE_DERIVED_CACHE_VERSION,
        fft_tuple: list[int] | tuple[int, int, int] = (64, 256, 128),
        clipped_range: int = 128,
        portion: float = 1.0,
        beam_label_cache: bool | str = "lazy",
        use_gps: bool = False,
        gps_feature_mode: str = SUPPORTED_GPS_FEATURE_MODE,
        gps_angle_offset_rad: float | None = None,
        gps_angle_offset_source: str | None = None,
        gps_normalize: bool = True,
        gps_scaler: GPSStandardScaler | None = None,
        use_gps_bev_xy: bool = False,
        gps_bev_xy_source: str = "history_relative_xy",
        gps_bev_roi: list[float] | tuple[float, ...] | None = None,
        use_mmwave: bool = False,
        mmwave_normalize: bool = True,
        mmwave_scaler: MmWaveStandardScaler | None = None,
        use_csi: bool = False,
        csi_train_rms: bool = True,
        csi_rms_normalizer: CSIRMSNormalizer | float | dict[str, Any] | None = None,
        csi_degradation: CSIDegradationConfig | dict[str, Any] | bool | None = None,
        occlusion_target: bool | dict[str, Any] | None = None,
        occlusion_target_stats: OcclusionTargetStats | dict[str, Any] | None = None,
        position_target: bool | dict[str, Any] | None = None,
        position_target_scaler: PositionTargetStandardScaler | None = None,
        soft_beam_labels: bool | dict[str, Any] | None = None,
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
        beam_label_mapping: BeamLabelMapping | None = None,
        beam_target_source: str = "future",
        sample_cache: dict[str, Any] | bool | None = None,
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
        self.root_csv = resolve_sequence_csv_path(
            self.data_root,
            self.scene,
            root_csv=root_csv,
            csv_name=csv_name,
            split=split,
            train_csv_name=train_csv_name,
            val_csv_name=val_csv_name,
            test_csv_name=test_csv_name,
        )
        self.seq_len = int(seq_len)
        self.gps_source_seq_len = resolve_gps_source_seq_len(
            seq_len=self.seq_len,
            gps_seq_len=gps_seq_len,
            gps_source_seq_len=gps_source_seq_len,
        )
        self.gps_seq_len = self.gps_source_seq_len
        self.num_pred = int(num_pred)
        self.image_profile = resolve_image_profile(image_profile)
        self.image_profile_spec = image_profile_spec(self.image_profile)
        self.image_size = tuple(image_size)
        self.image_cache_transform_version = str(image_cache_transform_version)
        self.fft_tuple = tuple(fft_tuple)
        self.clipped_range = clipped_range
        self.split = split
        self.enabled_modalities = resolve_enabled_modalities(
            enabled_modalities,
            use_gps=use_gps,
            use_lidar=use_lidar,
            use_mmwave=use_mmwave,
            use_csi=use_csi,
        )
        self.return_metadata = bool(return_metadata)
        self.beam_target_source = normalize_beam_target_source(beam_target_source)
        self.sample_cache, self.sample_cache_write_on_miss = self._build_sample_cache(sample_cache)
        validate_beam_target_source_contract(self.beam_target_source, num_pred=num_pred, seq_len=seq_len)
        self.beam_label_mapping = beam_label_mapping or resolve_beam_label_mapping(None, scene=self.scene_slug)
        self.beam_label_cache_mode = resolve_beam_label_cache_mode(beam_label_cache)
        self.beam_label_cache_metadata = {
            "cache_mode": self.beam_label_cache_mode,
            **self.beam_label_mapping.metadata(),
        }
        self._beam_label_cache: dict[str, int] = {}
        self.use_gps = "gps" in self.enabled_modalities
        self.gps_feature_mode = normalize_gps_feature_mode(gps_feature_mode)
        self.gps_angle_offset_rad, self.gps_angle_offset_source = resolve_gps_angle_offset(
            gps_feature_mode=self.gps_feature_mode,
            scene_id=self.scene_id,
            explicit_value=gps_angle_offset_rad,
            source=gps_angle_offset_source,
        )
        configure_deepsense6g_scaler_state(
            self,
            gps_normalize=gps_normalize,
            gps_scaler=gps_scaler,
            use_gps_bev_xy=use_gps_bev_xy,
            gps_bev_xy_source=gps_bev_xy_source,
            gps_bev_roi=gps_bev_roi,
            mmwave_normalize=mmwave_normalize,
            mmwave_scaler=mmwave_scaler,
            csi_train_rms=csi_train_rms,
            csi_rms_normalizer=csi_rms_normalizer,
            csi_degradation=csi_degradation,
        )
        configure_deepsense6g_target_state(
            self,
            occlusion_target=occlusion_target,
            occlusion_target_stats=occlusion_target_stats,
            position_target=position_target,
            position_target_scaler=position_target_scaler,
            soft_beam_labels=soft_beam_labels,
        )
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
        configure_deepsense6g_resource_readers(
            self,
            enabled_modalities=self.enabled_modalities,
            image_cache_dir=image_cache_dir,
            image_use_cache=image_use_cache,
            image_write_cache=image_write_cache,
            image_cache_policy=image_cache_policy,
            image_size=image_size,
            lidar_cache_dir=lidar_cache_dir,
            lidar_background_path=lidar_background_path,
        )
        self.samples = create_samples(
            self.root_csv,
            portion=portion,
            enabled_modalities=self.enabled_modalities,
            seq_len=self.seq_len,
            gps_source_seq_len=self.gps_source_seq_len,
            num_pred=self.num_pred,
            portion_strategy=portion_strategy,
            portion_seed=portion_seed,
            include_position_targets=(
                self.position_target_enabled and self.position_target_source == "future_gps_local_xy"
            ),
            include_history_position_targets=(
                self.use_gps_bev_xy
                or (self.position_target_enabled and self.position_target_source == "last_input_gps_local_xy")
            ),
        )
        prepare_deepsense6g_targets(self)
        ensure_enabled_contract_columns(
            root_csv=self.root_csv,
            samples=self.samples,
            use_gps=self.use_gps,
            use_gps_bev_xy=self.use_gps_bev_xy,
            use_mmwave=self.use_mmwave,
            use_csi=self.use_csi,
            use_lidar=self.use_lidar,
            gps_feature_mode=self.gps_feature_mode,
            supported_gps_modes=GPS_FEATURE_DIMS,
        )
        prepare_deepsense6g_scalers_and_normalizers(self)

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
        if self.sample_cache is not None:
            key = self._sample_cache_key(idx)
            cached = self.sample_cache.get(key)
            if cached is not None:
                return cached
        sample, _ = self._getitem_with_timing(idx, collect_timing=False)
        if self.sample_cache is not None and self.sample_cache_write_on_miss:
            self.sample_cache.put(self._sample_cache_key(idx), sample)
        return sample

    def profile_getitem_components(self, idx: int) -> dict[str, float]:
        _, timings = self._getitem_with_timing(idx, collect_timing=True)
        return timings

    def _sample_cache_key(self, idx: int) -> str:
        return f"{self.split}:{int(idx)}"

    def _build_sample_cache(self, cfg: dict[str, Any] | bool | None) -> tuple[LmdbSampleCache | None, bool]:
        if not cfg:
            return None, False
        if cfg is True:
            raise ValueError("data.dataset.sample_cache=true requires sample_cache.path.")
        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
            return None, False
        if str(cfg.get("backend", "lmdb")) != "lmdb":
            raise ValueError("data.dataset.sample_cache.backend currently supports only 'lmdb'.")
        raw_path = cfg.get("path")
        if not raw_path:
            raise ValueError("data.dataset.sample_cache.path is required when sample cache is enabled.")
        path = sample_cache_path_for_split(raw_path, self.split)
        return (
            LmdbSampleCache(
                path,
                readonly=not bool(cfg.get("write_on_miss", False)),
                map_size_gb=float(cfg.get("map_size_gb", 64.0)),
                readahead=bool(cfg.get("readahead", True)),
            ),
            bool(cfg.get("write_on_miss", False)),
        )

    def _getitem_with_timing(self, idx: int, *, collect_timing: bool) -> tuple[dict[str, Any], dict[str, float]]:
        timings = {
            name: 0.0
            for name in ("targets", "auxiliary_targets", "image", "radar", "gps", "gps_bev_xy", "mmwave", "csi", "lidar")
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
        target_beam_paths = self._target_beam_paths(beam_paths, future_beam_paths)

        sample, beam_metadata = record(
            "targets",
            lambda: build_beam_target_tensors(self, idx, beam_paths, target_beam_paths),
        )
        input_beam = beam_metadata["input_beam"]
        target_beam = beam_metadata["target_beam"]
        raw_input_beam = beam_metadata["raw_input_beam"]
        raw_target_beam = beam_metadata["raw_target_beam"]
        if self.soft_beam_label_config.enabled:
            distributions, mask = record(
                "targets",
                lambda: self._soft_beam_targets_for_paths(target_beam_paths, target_beam),
            )
            sample["target_beam_distribution"] = torch.tensor(distributions, dtype=torch.float32)
            sample["target_beam_distribution_mask"] = torch.tensor(mask, dtype=torch.bool)

        if self.occlusion_target_enabled or self.position_target_enabled:
            sample.update(
                record("auxiliary_targets", lambda: build_auxiliary_target_tensors(self, idx, future_beam_paths))
            )
        if "image" in self.enabled_modalities:
            sample["image"] = record("image", lambda: self.modality_loader.load_image(idx))
        if "radar" in self.enabled_modalities:
            radar_ra, radar_da = record("radar", lambda: self.modality_loader.load_radar(idx))
            sample["radar_ra"] = radar_ra
            sample["radar_da"] = radar_da
        if self.use_gps:
            sample["gps"] = record("gps", lambda: self.modality_loader.load_gps(idx))
        if self.use_gps_bev_xy:
            sample["gps_bev_xy"] = record("gps_bev_xy", lambda: self.modality_loader.load_gps_bev_xy(idx))
        if self.use_mmwave:
            sample["mmwave"] = record("mmwave", lambda: self.modality_loader.load_mmwave(idx))
        if self.use_csi:
            sample["csi"] = record("csi", lambda: self.modality_loader.load_csi(idx))
        if self.use_lidar:
            lidar_raw, lidar_model_input = record("lidar", lambda: self.modality_loader.load_lidar_pair(idx))
            sample["lidar_raw"] = lidar_raw
            sample["lidar"] = lidar_model_input
        if self.return_metadata:
            sample["metadata"] = self._metadata_for_index(idx, beam_paths, target_beam_paths, future_beam_paths)
            self._add_beam_label_metadata(
                sample["metadata"],
                idx=idx,
                target_beam_paths=target_beam_paths,
                input_beam=input_beam,
                target_beam=target_beam,
                raw_input_beam=raw_input_beam,
                raw_target_beam=raw_target_beam,
            )
        return sample, timings

    def _metadata_for_index(
        self,
        idx: int,
        beam_paths: list[str],
        target_beam_paths: list[str],
        future_beam_paths: list[str],
    ) -> dict[str, Any]:
        last_beam_path = str(beam_paths[-1]) if beam_paths else ""
        first_target_beam_path = str(target_beam_paths[0]) if target_beam_paths else ""
        first_future_beam_path = str(future_beam_paths[0]) if future_beam_paths else ""
        key = "|".join(
            [
                self.scene_slug,
                self.split,
                str(idx),
                last_beam_path,
                first_target_beam_path,
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
            "beam_target_source": self.beam_target_source,
            "seq_len": int(self.seq_len),
            "gps_seq_len": int(self.gps_source_seq_len),
            "gps_source_seq_len": int(self.gps_source_seq_len),
            "input_beam_path": last_beam_path,
            "target_beam_path": first_target_beam_path,
            "future_beam_path": first_future_beam_path,
        }
        if "image" in self.enabled_modalities:
            metadata["image_profile"] = self.image_profile
            metadata["processed_image_source"] = "rgb_imagenet"
        add_path_metadata(metadata, "image_path", getattr(self.samples, "rgb_paths", None), idx)
        add_path_metadata(metadata, "radar_path", getattr(self.samples, "radar_paths", None), idx)
        add_path_metadata(metadata, "gps_path", getattr(self.samples, "gps_paths", None), idx)
        add_path_metadata(metadata, "lidar_path", getattr(self.samples, "lidar_paths", None), idx)
        add_path_metadata(metadata, "mmwave_path", getattr(self.samples, "mmwave_paths", None), idx)
        add_path_metadata(metadata, "csi_path", getattr(self.samples, "csi_paths", None), idx)
        if self.use_csi and self.csi_degradation.enabled:
            metadata["csi_degradation"] = self._csi_degradation_metadata_for_index(idx)
        if self.use_gps_bev_xy:
            metadata["gps_bev_xy_source"] = self.gps_bev_xy_source
            metadata["gps_bev_roi"] = [float(value) for value in self.gps_bev_roi]
        seq_id, frame_idx = parse_sequence_position(first_target_beam_path or last_beam_path or metadata.get("mmwave_path", ""))
        if seq_id is not None:
            metadata["seq_id"] = seq_id
        if frame_idx is not None:
            metadata["frame_idx"] = int(frame_idx)
        return metadata

    def _target_beam_paths(self, beam_paths: list[str], future_beam_paths: list[str]) -> list[str]:
        return resolve_target_beam_paths(
            beam_paths,
            future_beam_paths,
            source=self.beam_target_source,
            num_pred=self.num_pred,
        )

    def _build_image_transform(self, image_size: list[int] | tuple[int, int]):
        return build_rgb_imagenet_transform(image_size)

    def _load_rgb_imagenet_frames(self, paths: list[str]) -> torch.Tensor:
        return load_rgb_imagenet_frames(
            self.data_root,
            paths,
            self.seq_len,
            self.transform,
            image_size=self.image_size,
            image_cache=self.image_cache,
        )

    def image_cache_metadata(self) -> dict[str, Any]:
        if self.image_cache is None:
            return {
                "policy": "off",
                "enabled": False,
                "accessed": False,
            }
        summary = self.image_cache.summary()
        summary["enabled"] = self.image_cache_policy != "off"
        summary["accessed"] = bool(summary["hits"] or summary["misses"] or summary["generated"])
        return summary

    def _build_image_cache(self) -> ImageDerivedCache | None:
        if self.image_cache_policy == "off" or self.image_cache_dir is None:
            return None
        return ImageDerivedCache(
            ImageDerivedCacheConfig(
                cache_dir=self.image_cache_dir,
                policy=self.image_cache_policy,
                image_profile=self.image_profile,
                image_size=(int(self.image_size[0]), int(self.image_size[1])),
                transform_version=self.image_cache_transform_version,
            )
        )

    @staticmethod
    def _policy_from_cache_flags(use_cache: bool, write_cache: bool) -> str:
        if use_cache and write_cache:
            return "auto"
        if use_cache:
            return "read_only"
        if write_cache:
            return "rebuild"
        return "off"

    def _load_radar_maps(self, paths: list[str]):
        return load_radar_maps(
            self.data_root,
            paths,
            self.seq_len,
            self.fft_tuple,
            self.clipped_range,
        )

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
        return self._map_beam_label(self._raw_beam_label(beam_path))

    def _raw_beam_label(self, beam_path: str) -> int:
        key = str(beam_path)
        if self.beam_label_cache_mode != "off" and key in self._beam_label_cache:
            return self._beam_label_cache[key]
        label = self._read_beam_label(key)
        if self.beam_label_cache_mode != "off":
            self._beam_label_cache[key] = label
        return label

    def _input_raw_beam_label_for_index(self, idx: int, horizon: int, beam_path: str) -> int:
        return self._raw_beam_label(beam_path)

    def _target_raw_beam_label_for_index(self, idx: int, horizon: int, beam_path: str) -> int:
        return self._raw_beam_label(beam_path)

    def _input_beam_label_source_for_index(self, idx: int, horizon: int, beam_path: str) -> str:
        return "beam_power_argmax"

    def _target_beam_label_source_for_index(self, idx: int, horizon: int, beam_path: str) -> str:
        return "beam_power_argmax"

    def _map_beam_label(self, raw_label: int) -> int:
        return self.beam_label_mapping.map_label(int(raw_label))

    def _add_beam_label_metadata(
        self,
        metadata: dict[str, Any],
        *,
        idx: int,
        target_beam_paths: list[str],
        input_beam: list[int],
        target_beam: list[int],
        raw_input_beam: list[int],
        raw_target_beam: list[int],
    ) -> None:
        mapping_metadata = self.beam_label_mapping.metadata()
        metadata.update(mapping_metadata)
        metadata["raw_input_beam"] = [int(value) for value in raw_input_beam]
        metadata["raw_target_beam"] = [int(value) for value in raw_target_beam]
        metadata["calibrated_input_beam"] = [int(value) for value in input_beam]
        metadata["calibrated_target_beam"] = [int(value) for value in target_beam]
        metadata["input_beam_label_source"] = [
            self._input_beam_label_source_for_index(idx, horizon, beam_path)
            for horizon, beam_path in enumerate(self.samples.input_beam_paths[idx][-self.seq_len :])
        ]
        metadata["target_beam_label_source"] = [
            self._target_beam_label_source_for_index(idx, horizon, beam_path)
            for horizon, beam_path in enumerate(target_beam_paths)
        ]

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

    def _soft_beam_targets_for_paths(
        self,
        future_beam_paths: list[str],
        hard_labels: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.soft_beam_label_config
        num_classes = self._soft_beam_num_classes(hard_labels)
        distributions: list[np.ndarray] = []
        masks: list[bool] = []
        for horizon, rel_path in enumerate(future_beam_paths[: self.num_pred]):
            label = hard_labels[horizon] if horizon < len(hard_labels) else -100
            if label < 0:
                distributions.append(np.zeros(num_classes, dtype=np.float32))
                masks.append(False)
                continue
            distribution, power_available = self._soft_beam_distribution_for_path(
                rel_path,
                label,
                cfg=cfg,
                num_classes=num_classes,
            )
            distributions.append(distribution)
            masks.append(True)
        while len(distributions) < int(self.num_pred):
            distributions.append(np.zeros(num_classes, dtype=np.float32))
            masks.append(False)
        return np.stack(distributions, axis=0), np.asarray(masks, dtype=bool)

    def _soft_beam_distribution_for_path(
        self,
        rel_path: object,
        label: int,
        *,
        cfg: SoftBeamLabelConfig,
        num_classes: int,
    ) -> tuple[np.ndarray, bool]:
        key = str(rel_path or "").strip()
        domain = self._soft_beam_label_domain(cfg)
        source = cfg.source if domain == "source" else cfg.target_source
        circular = True if domain == "target" else cfg.circular
        if domain == "target" and source != "gaussian":
            raise ValueError("target-domain soft beam labels must use circular Gaussian targets.")
        cache_key = (
            f"{domain}|{key}|{label}|{num_classes}|{source}|{cfg.sigma}|{circular}|{cfg.temperature}|"
            f"{self.beam_label_mapping.fingerprint}"
        )
        if cfg.cache and cache_key in self._soft_beam_distribution_cache:
            return self._soft_beam_distribution_cache[cache_key]
        power = None
        # Target adaptation must not use target-side power/RSS oracle profiles; use only hard labels plus
        # codebook adjacency through circular Gaussian smoothing.
        if (
            domain == "source"
            and source in {"power", "rss", "power_or_gaussian", "rss_or_gaussian"}
            and key
            and key != "-99"
        ):
            power = read_beam_power_vector(joined_resource(self.data_root, key), num_classes=num_classes)
        result = soft_distribution_from_power_or_label(
            power,
            int(label),
            num_classes=num_classes,
            source=source,
            sigma=cfg.sigma,
            circular=circular,
            temperature=cfg.temperature,
            epsilon=cfg.epsilon,
        )
        if result[1] and self.beam_label_mapping.enabled:
            result = (
                self.beam_label_mapping.reorder_distribution(result[0], axis=-1).astype(np.float32),
                result[1],
            )
        if cfg.cache:
            self._soft_beam_distribution_cache[cache_key] = result
        return result

    def _soft_beam_label_domain(self, cfg: SoftBeamLabelConfig) -> str:
        if cfg.domain in {"source", "target"}:
            return cfg.domain
        split_text = str(self.split or "").strip().lower()
        if split_text.startswith("target") or split_text in {"test", "val", "validation"}:
            return "target"
        return "source"

    def _soft_beam_num_classes(self, hard_labels: list[int]) -> int:
        configured = self.soft_beam_label_config.num_classes
        if configured is not None:
            return int(configured)
        if self.beam_label_mapping.enabled:
            return int(self.beam_label_mapping.num_classes)
        if hard_labels:
            return max(64, max(int(value) for value in hard_labels) + 1)
        return 64

    def _prepare_gps_scaler(self) -> None:
        if not self.gps_normalize:
            self.gps_scaler = None
            return
        if self.gps_scaler is not None:
            self.gps_scaler_metadata = {
                "source": "provided",
                "sample_count": len(self),
                "streaming": False,
                "retains_per_sample_sequence_cache": False,
            }
            return
        if self.split != "train":
            raise ValueError(
                "GPS normalization for non-train split requires a train-fitted gps_scaler. "
                "Use build_dataloaders/evaluate so the train scaler can be reused."
            )
        stats = _StreamingFeatureStats()
        for idx in range(len(self)):
            if self.samples.gps_paths is None:
                raise ValueError("GPS paths are unavailable for this dataset.")
            bs_paths = self.samples.bs_gps_paths[idx] if self.samples.bs_gps_paths is not None else None
            stats.update(
                load_gps_feature_sequence(
                    self.data_root,
                    self.samples.gps_paths[idx],
                    bs_paths,
                    seq_len=self.gps_source_seq_len,
                    mode=self.gps_feature_mode,
                    angle_offset_rad=self.gps_angle_offset_rad,
                    frame_feature_cache=self._gps_frame_feature_cache,
                )
            )
        mean, scale = stats.finalize()
        self.gps_scaler = GPSStandardScaler(mean_=mean, scale_=scale)
        self._gps_feature_cache.clear()
        self.gps_scaler_metadata = {
            "source": "train_split_streaming_fit",
            "sample_count": len(self),
            "frame_count": int(stats.count),
            "streaming": True,
            "retains_per_sample_sequence_cache": False,
        }

    def _gps_features_for_index(self, idx: int) -> np.ndarray:
        if idx not in self._gps_feature_cache:
            if self.samples.gps_paths is None:
                raise ValueError("GPS paths are unavailable for this dataset.")
            bs_paths = self.samples.bs_gps_paths[idx] if self.samples.bs_gps_paths is not None else None
            self._gps_feature_cache[idx] = load_gps_feature_sequence(
                self.data_root,
                self.samples.gps_paths[idx],
                bs_paths,
                seq_len=self.gps_source_seq_len,
                mode=self.gps_feature_mode,
                angle_offset_rad=self.gps_angle_offset_rad,
                frame_feature_cache=self._gps_frame_feature_cache,
            )
        return self._gps_feature_cache[idx]

    def _gps_bev_xy_for_index(self, idx: int) -> np.ndarray:
        if idx not in self._gps_bev_xy_cache:
            if self.samples.gps_paths is None or self.samples.bs_gps_paths is None:
                raise ValueError("GPS BEV XY paths are unavailable for this dataset.")
            self._gps_bev_xy_cache[idx] = load_relative_xy_sequence(
                self.data_root,
                self.samples.gps_paths[idx],
                self.samples.bs_gps_paths[idx],
                seq_len=self.gps_source_seq_len,
            )
        return self._gps_bev_xy_cache[idx]

    def _prepare_mmwave_scaler(self) -> None:
        if not self.mmwave_normalize:
            self.mmwave_scaler = None
            return
        if self.mmwave_scaler is not None:
            self.mmwave_scaler_metadata = {
                "source": "provided",
                "sample_count": len(self),
                "streaming": False,
                "retains_per_sample_sequence_cache": False,
            }
            return
        if self.split != "train":
            raise ValueError(
                "mmWave normalization for non-train split requires a train-fitted mmwave_scaler. "
                "Use build_dataloaders/evaluate with a checkpoint that records mmwave_scaler, "
                "provide mmwave_scaler explicitly, or disable mmWave normalization."
            )
        stats = _StreamingFeatureStats()
        for idx in range(len(self)):
            if self.samples.mmwave_paths is None:
                raise ValueError("mmWave paths are unavailable for this dataset.")
            stats.update(
                load_mmwave_feature_sequence(
                    self.data_root,
                    self.samples.mmwave_paths[idx],
                    seq_len=self.seq_len,
                    expected_dim=MMWAVE_POWER_DIM,
                    frame_feature_cache=self._mmwave_frame_feature_cache,
                )
            )
        mean, scale = stats.finalize()
        self.mmwave_scaler = MmWaveStandardScaler(mean_=mean, scale_=scale)
        self._mmwave_feature_cache.clear()
        self.mmwave_scaler_metadata = {
            "source": "train_split_streaming_fit",
            "sample_count": len(self),
            "frame_count": int(stats.count),
            "streaming": True,
            "retains_per_sample_sequence_cache": False,
        }

    def _mmwave_features_for_index(self, idx: int) -> np.ndarray:
        if idx not in self._mmwave_feature_cache:
            if self.samples.mmwave_paths is None:
                raise ValueError("mmWave paths are unavailable for this dataset.")
            self._mmwave_feature_cache[idx] = load_mmwave_feature_sequence(
                self.data_root,
                self.samples.mmwave_paths[idx],
                seq_len=self.seq_len,
                expected_dim=MMWAVE_POWER_DIM,
                frame_feature_cache=self._mmwave_frame_feature_cache,
            )
        return self._mmwave_feature_cache[idx]

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
            stats.update(self._clean_csi_for_index(idx))
        self.csi_rms_normalizer = stats.finalize()

    def _csi_for_index(self, idx: int) -> np.ndarray:
        if self.csi_degradation.enabled:
            return self._degraded_csi_for_index(idx)
        return self._clean_csi_for_index(idx)

    def _clean_csi_for_index(self, idx: int) -> np.ndarray:
        if idx not in self._csi_clean_cache:
            if self.samples.csi_paths is None:
                raise ValueError("CSI paths are unavailable for this dataset.")
            self._csi_clean_cache[idx] = load_csi_sequence(
                self.data_root,
                self.samples.csi_paths[idx],
                seq_len=self.seq_len,
            )
        return self._csi_clean_cache[idx]

    def _degraded_csi_for_index(self, idx: int) -> np.ndarray:
        if idx not in self._csi_degraded_cache:
            if self.samples.csi_paths is None:
                raise ValueError("CSI paths are unavailable for this dataset.")
            diagnostics: dict[str, Any] = {}
            self._csi_degraded_cache[idx] = load_csi_sequence(
                self.data_root,
                self.samples.csi_paths[idx],
                seq_len=self.seq_len,
                degradation=self.csi_degradation,
                split=self.split,
                sample_index=idx,
                sample_key=self._csi_sample_key(idx),
                diagnostics=diagnostics,
            )
            self._csi_degradation_diagnostics[idx] = diagnostics
        return self._csi_degraded_cache[idx]

    def _csi_sample_key(self, idx: int) -> str:
        if self.samples.csi_paths is None:
            return f"{self.scene_slug}:{self.split}:{idx}"
        return "|".join([self.scene_slug, self.split, str(idx), *map(str, self.samples.csi_paths[idx])])

    def _csi_degradation_metadata_for_index(self, idx: int) -> dict[str, Any]:
        diagnostics = self._csi_degradation_diagnostics.get(idx)
        if diagnostics is None:
            return {
                "enabled": bool(self.csi_degradation.enabled),
                "profile": self.csi_degradation.profile,
                "resolved_parameters": self.csi_degradation.to_dict(),
                "seed": self.csi_degradation.seed,
            }
        return {
            "enabled": bool(diagnostics.get("enabled", self.csi_degradation.enabled)),
            "profile": diagnostics.get("profile", self.csi_degradation.profile),
            "resolved_parameters": diagnostics.get("resolved_parameters", self.csi_degradation.to_dict()),
            "seed": diagnostics.get("seed", self.csi_degradation.seed),
            "sample_seed": diagnostics.get("sample_seed"),
            "temporal_shift": diagnostics.get("temporal_shift", 0),
            "temporal_fill_mode": diagnostics.get("temporal_fill_mode", self.csi_degradation.temporal_fill_mode),
            "skipped_operators": list(diagnostics.get("skipped_operators", [])),
        }

    def csi_degradation_metadata(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.csi_degradation.enabled),
            "profile": self.csi_degradation.profile,
            "resolved_parameters": self.csi_degradation.to_dict(),
            "seed": self.csi_degradation.seed,
        }

    @staticmethod
    def _coerce_csi_rms_normalizer(value: CSIRMSNormalizer | float | dict[str, Any] | None) -> CSIRMSNormalizer | None:
        if value is None:
            return None
        if isinstance(value, CSIRMSNormalizer):
            return value
        if isinstance(value, dict):
            return CSIRMSNormalizer(rms=float(value["rms"]), sample_count=int(value.get("sample_count", 0)))
        return CSIRMSNormalizer(rms=float(value), sample_count=0)

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
