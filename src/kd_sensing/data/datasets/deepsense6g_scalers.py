from __future__ import annotations

import numpy as np

from kd_sensing.data.datasets.deepsense6g_gps_contract import normalize_gps_bev_xy_source
from kd_sensing.data.transform_ops.csi import resolve_csi_degradation_config


class _StreamingFeatureStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum: np.ndarray | None = None
        self.sum_sq: np.ndarray | None = None

    def update(self, features: np.ndarray) -> None:
        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"Streaming feature stats expect [N, D] features, got {array.shape}.")
        if array.shape[0] == 0:
            return
        batch_sum = array.sum(axis=0)
        batch_sum_sq = np.square(array).sum(axis=0)
        if self.sum is None:
            self.sum = batch_sum
            self.sum_sq = batch_sum_sq
        else:
            self.sum += batch_sum
            assert self.sum_sq is not None
            self.sum_sq += batch_sum_sq
        self.count += int(array.shape[0])

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count <= 0 or self.sum is None or self.sum_sq is None:
            raise ValueError("Cannot fit scaler from an empty feature stream.")
        mean = self.sum / float(self.count)
        variance = np.maximum(self.sum_sq / float(self.count) - np.square(mean), 0.0)
        scale = np.sqrt(variance)
        scale[scale < 1e-8] = 1.0
        return mean.astype(np.float32), scale.astype(np.float32)


def configure_deepsense6g_scaler_state(
    dataset,
    *,
    gps_normalize: bool,
    gps_scaler,
    use_gps_bev_xy: bool,
    gps_bev_xy_source: str,
    gps_bev_roi,
    mmwave_normalize: bool,
    mmwave_scaler,
    csi_train_rms: bool,
    csi_rms_normalizer,
    csi_degradation,
) -> None:
    dataset.use_gps = "gps" in dataset.enabled_modalities
    dataset.gps_normalize = gps_normalize
    dataset.gps_scaler = gps_scaler
    dataset.gps_scaler_metadata: dict[str, object] = {}
    dataset._gps_feature_cache: dict[int, np.ndarray] = {}
    dataset._gps_frame_feature_cache: dict[str, np.ndarray] = {}
    dataset.use_gps_bev_xy = bool(use_gps_bev_xy)
    dataset.gps_bev_xy_source = normalize_gps_bev_xy_source(gps_bev_xy_source)
    dataset.gps_bev_roi = tuple(gps_bev_roi or (-60.0, 60.0, -60.0, 60.0))
    dataset._gps_bev_xy_cache: dict[int, np.ndarray] = {}
    dataset.use_mmwave = "mmwave" in dataset.enabled_modalities
    dataset.mmwave_normalize = bool(mmwave_normalize)
    dataset.mmwave_scaler = mmwave_scaler
    dataset.mmwave_scaler_metadata: dict[str, object] = {}
    dataset._mmwave_feature_cache: dict[int, np.ndarray] = {}
    dataset._mmwave_frame_feature_cache: dict[str, np.ndarray] = {}
    dataset.use_csi = "csi" in dataset.enabled_modalities
    dataset.csi_train_rms = bool(csi_train_rms)
    dataset.csi_rms_normalizer = dataset._coerce_csi_rms_normalizer(csi_rms_normalizer)
    dataset.csi_degradation = resolve_csi_degradation_config(csi_degradation)
    dataset._csi_clean_cache: dict[int, np.ndarray] = {}
    dataset._csi_degraded_cache: dict[int, np.ndarray] = {}
    dataset._csi_degradation_diagnostics: dict[int, dict[str, object]] = {}
    dataset._csi_cache = dataset._csi_clean_cache


def prepare_deepsense6g_scalers_and_normalizers(dataset) -> None:
    if dataset.use_gps:
        dataset._prepare_gps_scaler()
    if dataset.position_target_enabled:
        dataset._ensure_position_target_columns()
        dataset._prepare_position_target_scaler()
    if dataset.use_mmwave:
        dataset._prepare_mmwave_scaler()
    if dataset.use_csi:
        dataset._prepare_csi_rms_normalizer()
    if dataset.use_lidar:
        dataset._prepare_lidar_normalizer_from_config()


__all__ = [
    "_StreamingFeatureStats",
    "configure_deepsense6g_scaler_state",
    "prepare_deepsense6g_scalers_and_normalizers",
]
