from __future__ import annotations

from typing import Any

import numpy as np

from kd_sensing.data.transform_ops.gps import (
    PositionTargetStandardScaler,
    build_relative_xy_targets,
    load_relative_xy_target_sequence,
    read_gps_latlon,
)
from kd_sensing.data.transform_ops.mmwave import (
    MMWAVE_POWER_DIM,
    OcclusionTargetStats,
    finite_max_mmwave_power,
    fit_occlusion_threshold_from_paths,
)


def resolve_occlusion_target_config(config: bool | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(config, bool):
        raw: dict[str, Any] = {"enabled": config}
    elif config is None:
        raw = {}
    elif isinstance(config, dict):
        raw = dict(config)
    else:
        raise TypeError("occlusion_target must be a bool, mapping, or None.")
    enabled = bool(raw.get("enabled", False))
    percentile = float(raw.get("threshold_percentile", raw.get("percentile", 20.0)))
    threshold = raw.get("threshold", raw.get("tau"))
    if threshold is not None:
        threshold = float(threshold)
    return {
        "enabled": enabled,
        "threshold_percentile": percentile,
        "threshold": threshold,
    }


def resolve_position_target_config(config: bool | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(config, bool):
        raw: dict[str, Any] = {"enabled": config}
    elif config is None:
        raw = {}
    elif isinstance(config, dict):
        raw = dict(config)
    else:
        raise TypeError("position_target must be a bool, mapping, or None.")
    enabled = bool(raw.get("enabled", False))
    source = str(raw.get("source", raw.get("position_target_source", "future_gps_local_xy")))
    if source not in {"future_gps_local_xy", "last_input_gps_local_xy"}:
        raise ValueError("position_target.source must be one of future_gps_local_xy or last_input_gps_local_xy.")
    normalize = bool(raw.get("normalize", raw.get("standardize", enabled)))
    return {
        "enabled": enabled,
        "source": source,
        "normalize": normalize,
    }


def coerce_occlusion_stats(stats: OcclusionTargetStats | dict[str, Any] | None) -> OcclusionTargetStats | None:
    if stats is None:
        return None
    if isinstance(stats, OcclusionTargetStats):
        return stats
    if isinstance(stats, dict):
        return OcclusionTargetStats.from_dict(stats)
    raise TypeError("occlusion_target_stats must be OcclusionTargetStats, mapping, or None.")


class DeepSense6GTargetProvider:
    def __init__(
        self,
        dataset,
        *,
        occlusion_stats: OcclusionTargetStats | dict[str, Any] | None = None,
        position_scaler: PositionTargetStandardScaler | None = None,
    ):
        self.dataset = dataset
        self.occlusion_target_stats = coerce_occlusion_stats(occlusion_stats)
        self._occlusion_power_cache: dict[str, float | None] = {}
        self.position_target_scaler = position_scaler
        self._position_target_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def prepare_occlusion_target_stats(self) -> None:
        ds = self.dataset
        if self.occlusion_target_stats is not None:
            return
        explicit_threshold = ds.occlusion_target_config.get("threshold")
        percentile = float(ds.occlusion_target_config["threshold_percentile"])
        if explicit_threshold is not None:
            self.occlusion_target_stats = OcclusionTargetStats(
                threshold=float(explicit_threshold),
                threshold_percentile=percentile,
                sample_count=0,
                positive_count=0,
                positive_ratio=0.0,
            )
            return
        if ds.split != "train":
            raise ValueError(
                "Occlusion target generation for non-train split requires train-fitted "
                "occlusion_target_stats or an explicit occlusion_target.threshold."
            )
        future_paths = [
            str(path)
            for paths in ds.samples.future_beam_paths
            for path in paths[: ds.num_pred]
            if ds._valid_resource_path(path)
        ]
        self.occlusion_target_stats = fit_occlusion_threshold_from_paths(
            ds.data_root,
            future_paths,
            threshold_percentile=percentile,
            expected_dim=MMWAVE_POWER_DIM,
            use_finite_max=True,
        )

    def occlusion_targets_for_paths(self, future_beam_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
        ds = self.dataset
        if self.occlusion_target_stats is None:
            raise ValueError("Occlusion target is enabled but occlusion_target_stats are unavailable.")
        labels = np.zeros((ds.num_pred,), dtype=np.float32)
        valid = np.zeros((ds.num_pred,), dtype=bool)
        for horizon, path in enumerate(future_beam_paths[: ds.num_pred]):
            if not ds._valid_resource_path(path):
                continue
            power = self.max_power_for_path(str(path))
            if power is None:
                continue
            labels[horizon] = 1.0 if power < self.occlusion_target_stats.threshold else 0.0
            valid[horizon] = True
        return labels, valid

    def max_power_for_path(self, rel_path: str) -> float | None:
        key = str(rel_path)
        if key not in self._occlusion_power_cache:
            self._occlusion_power_cache[key] = finite_max_mmwave_power(
                self.dataset.data_root,
                key,
                expected_dim=MMWAVE_POWER_DIM,
            )
        return self._occlusion_power_cache[key]

    def ensure_position_target_columns(self) -> None:
        ds = self.dataset
        if ds.position_target_source == "future_gps_local_xy":
            if ds.samples.future_gps_paths is None or ds.samples.future_bs_gps_paths is None:
                raise ValueError(
                    f"Position target source 'future_gps_local_xy' requires future_gps1..future_gpsN "
                    f"and future_bs_gps1..future_bs_gpsN columns in {ds.root_csv}. "
                    "Regenerate sequence CSVs with include_position_targets: true."
                )
            return
        if ds.samples.gps_paths is None or ds.samples.bs_gps_paths is None:
            raise ValueError(
                f"Position target source 'last_input_gps_local_xy' requires gps1..gpsN and bs_gps1..bs_gpsN "
                f"columns in {ds.root_csv}."
            )

    def prepare_position_target_scaler(self) -> None:
        ds = self.dataset
        if not ds.position_target_normalize:
            self.position_target_scaler = None
            return
        if self.position_target_scaler is not None:
            return
        if ds.split != "train":
            raise ValueError(
                "Position target normalization for non-train split requires a train-fitted "
                "position_target_scaler. Use build_dataloaders/evaluate so train statistics can be reused."
            )
        targets = []
        for idx in range(len(ds)):
            position_target, position_valid = self.position_targets_for_index(idx)
            if np.any(position_valid):
                targets.append(position_target[position_valid])
        if not targets:
            raise ValueError("Cannot fit position_target_scaler because no valid position targets were found.")
        self.position_target_scaler = PositionTargetStandardScaler().fit(np.concatenate(targets, axis=0))

    def position_targets_for_index(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        ds = self.dataset
        if idx not in self._position_target_cache:
            if ds.position_target_source == "future_gps_local_xy":
                self._position_target_cache[idx] = self._future_position_targets_for_index(idx)
            elif ds.position_target_source == "last_input_gps_local_xy":
                self._position_target_cache[idx] = self._last_input_position_targets_for_index(idx)
            else:
                raise ValueError(f"Unsupported position_target source '{ds.position_target_source}'.")
        return self._position_target_cache[idx]

    def _future_position_targets_for_index(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        ds = self.dataset
        if ds.samples.future_gps_paths is None or ds.samples.future_bs_gps_paths is None:
            raise ValueError("Future GPS position target paths are unavailable for this dataset.")
        gps_paths = ds.samples.future_gps_paths[idx][: ds.num_pred]
        bs_paths = ds.samples.future_bs_gps_paths[idx][: ds.num_pred]
        targets = np.zeros((ds.num_pred, 2), dtype=np.float32)
        valid = np.zeros((ds.num_pred,), dtype=bool)
        for horizon, (gps_path, bs_path) in enumerate(zip(gps_paths, bs_paths)):
            if not ds._valid_resource_path(gps_path) or not ds._valid_resource_path(bs_path):
                continue
            target = load_relative_xy_target_sequence(
                ds.data_root,
                [str(gps_path)],
                [str(bs_path)],
                num_pred=1,
            )
            targets[horizon] = target[0]
            valid[horizon] = True
        return targets, valid

    def _last_input_position_targets_for_index(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        ds = self.dataset
        if ds.samples.gps_paths is None or ds.samples.bs_gps_paths is None:
            raise ValueError("Input GPS paths are unavailable for fallback position targets.")
        gps_paths = ds.samples.gps_paths[idx][-ds.seq_len :]
        bs_paths = ds.samples.bs_gps_paths[idx][-ds.seq_len :] if ds.samples.bs_gps_paths is not None else []
        if not gps_paths or not bs_paths:
            return np.zeros((ds.num_pred, 2), dtype=np.float32), np.zeros((ds.num_pred,), dtype=bool)
        gps_path = gps_paths[-1]
        bs_path = bs_paths[-1]
        if not ds._valid_resource_path(gps_path) or not ds._valid_resource_path(bs_path):
            return np.zeros((ds.num_pred, 2), dtype=np.float32), np.zeros((ds.num_pred,), dtype=bool)
        ue = np.asarray([read_gps_latlon(ds.data_root, str(gps_path))], dtype=np.float64)
        bs = np.asarray([read_gps_latlon(ds.data_root, str(bs_path))], dtype=np.float64)
        xy = build_relative_xy_targets(ue, bs)[0]
        return (
            np.repeat(xy.reshape(1, 2), ds.num_pred, axis=0).astype(np.float32),
            np.ones((ds.num_pred,), dtype=bool),
        )

    def auxiliary_target_metadata(self) -> dict[str, Any]:
        ds = self.dataset
        metadata: dict[str, Any] = {}
        if ds.occlusion_target_enabled:
            metadata["occlusion_target"] = {
                "enabled": True,
                "threshold_percentile": float(ds.occlusion_target_config["threshold_percentile"]),
                "stats": self.occlusion_target_stats.to_dict() if self.occlusion_target_stats is not None else None,
            }
        if ds.position_target_enabled:
            metadata["position_target"] = {
                "enabled": True,
                "source": ds.position_target_source,
                "normalize": bool(ds.position_target_normalize),
                "scaler": (
                    self.position_target_scaler.to_dict()
                    if self.position_target_scaler is not None and ds.position_target_normalize
                    else None
                ),
            }
        return metadata


__all__ = [
    "DeepSense6GTargetProvider",
    "coerce_occlusion_stats",
    "resolve_occlusion_target_config",
    "resolve_position_target_config",
]
