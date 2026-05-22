from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


LIDAR_NONEMPTY_RATIO_THRESHOLD = 0.05
LIDAR_CHANNEL_STD_THRESHOLD = 1e-6
LIDAR_RAW_ZERO_RATIO_THRESHOLD = 0.95
LIDAR_MODEL_INPUT_ABS_MAX_THRESHOLD = 20.0


@dataclass
class _LidarTensorStats:
    zero_epsilon: float = 1e-8
    sum_: np.ndarray | None = None
    sumsq_: np.ndarray | None = None
    zero_count_: np.ndarray | None = None
    abs_max_: np.ndarray | None = None
    value_count_: int = 0
    frame_count_: int = 0
    nonempty_frame_count_: int = 0

    def update(self, lidar: torch.Tensor | np.ndarray) -> "_LidarTensorStats":
        tensor = torch.as_tensor(lidar).detach().to(dtype=torch.float32, device="cpu")
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim < 5:
            raise ValueError(
                "LiDAR quality expects [B, T, C, ...] or [T, C, ...] with at least 2 spatial dims, "
                f"got {tuple(tensor.shape)}."
            )
        channels = tensor.shape[2]
        channel_first_order = [2, 0, 1, *range(3, tensor.ndim)]
        values = tensor.permute(*channel_first_order).reshape(channels, -1)
        channel_sum = values.sum(dim=1).numpy()
        channel_sumsq = values.square().sum(dim=1).numpy()
        channel_zero = values.abs().le(float(self.zero_epsilon)).sum(dim=1).numpy()
        channel_abs_max = values.abs().amax(dim=1).numpy()
        if self.sum_ is None:
            self.sum_ = np.zeros(channels, dtype=np.float64)
            self.sumsq_ = np.zeros(channels, dtype=np.float64)
            self.zero_count_ = np.zeros(channels, dtype=np.float64)
            self.abs_max_ = np.zeros(channels, dtype=np.float64)
        if self.sum_.shape[0] != channels:
            raise ValueError(f"LiDAR channel count changed from {self.sum_.shape[0]} to {channels}.")
        self.sum_ += channel_sum
        self.sumsq_ += channel_sumsq
        self.zero_count_ += channel_zero
        self.abs_max_ = np.maximum(self.abs_max_, channel_abs_max)
        self.value_count_ += int(values.shape[1])
        frames = tensor.reshape(-1, channels, *tensor.shape[3:])
        self.frame_count_ += int(frames.shape[0])
        self.nonempty_frame_count_ += int(frames.abs().reshape(frames.shape[0], -1).sum(dim=1).gt(self.zero_epsilon).sum().item())
        return self

    def finalize(
        self,
        *,
        split: str | None = None,
        preprocessing: dict[str, Any] | None = None,
        perspective: str,
    ) -> dict[str, Any]:
        if (
            self.sum_ is None
            or self.sumsq_ is None
            or self.zero_count_ is None
            or self.abs_max_ is None
            or self.value_count_ <= 0
        ):
            return {
                "split": split,
                "perspective": perspective,
                "num_frames": 0,
                "nonempty_frames": 0,
                "nonempty_frame_ratio": 0.0,
                "channel_mean": [],
                "channel_std": [],
                "zero_ratio": [],
                "channel_abs_max": [],
                "degradation_risk": True,
                "degradation_reasons": ["no_lidar_frames"],
                "preprocessing": preprocessing or {},
            }
        mean = self.sum_ / self.value_count_
        variance = self.sumsq_ / self.value_count_ - np.square(mean)
        std = np.sqrt(np.maximum(variance, 0.0))
        zero_ratio = self.zero_count_ / self.value_count_
        nonempty_ratio = self.nonempty_frame_count_ / max(self.frame_count_, 1)
        reasons = []
        if nonempty_ratio < LIDAR_NONEMPTY_RATIO_THRESHOLD:
            reasons.append("low_nonempty_frame_ratio")
        if any(float(value) < LIDAR_CHANNEL_STD_THRESHOLD for value in std):
            reasons.append("near_constant_channel")
        if perspective == "raw" and float(np.mean(zero_ratio)) >= LIDAR_RAW_ZERO_RATIO_THRESHOLD:
            reasons.append("raw_extreme_sparsity")
        if perspective == "model_input" and any(
            float(value) > LIDAR_MODEL_INPUT_ABS_MAX_THRESHOLD for value in self.abs_max_
        ):
            reasons.append("model_input_abnormal_amplitude")
        return {
            "split": split,
            "perspective": perspective,
            "num_frames": int(self.frame_count_),
            "nonempty_frames": int(self.nonempty_frame_count_),
            "nonempty_frame_ratio": float(nonempty_ratio),
            "channel_mean": [float(value) for value in mean],
            "channel_std": [float(value) for value in std],
            "zero_ratio": [float(value) for value in zero_ratio],
            "channel_abs_max": [float(value) for value in self.abs_max_],
            "degradation_risk": bool(reasons),
            "degradation_reasons": reasons,
            "preprocessing": preprocessing or {},
        }


@dataclass
class LidarQualityAccumulator:
    zero_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        self.raw_ = _LidarTensorStats(zero_epsilon=self.zero_epsilon)
        self.model_input_ = _LidarTensorStats(zero_epsilon=self.zero_epsilon)

    def update(
        self,
        lidar: torch.Tensor | np.ndarray,
        *,
        raw_lidar: torch.Tensor | np.ndarray | None = None,
    ) -> "LidarQualityAccumulator":
        self.model_input_.update(lidar)
        self.raw_.update(lidar if raw_lidar is None else raw_lidar)
        return self

    def finalize(
        self,
        *,
        split: str | None = None,
        preprocessing: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw = self.raw_.finalize(split=split, preprocessing=preprocessing, perspective="raw")
        model_input = self.model_input_.finalize(
            split=split,
            preprocessing=preprocessing,
            perspective="model_input",
        )
        reasons = sorted(
            set(str(reason) for reason in raw.get("degradation_reasons", []))
            | set(str(reason) for reason in model_input.get("degradation_reasons", []))
        )
        summary = {
            **model_input,
            "raw": raw,
            "model_input": model_input,
            "degradation_risk": bool(reasons),
            "degradation_reasons": reasons,
        }
        return summary


def lidar_preprocessing_metadata_from_dataset(dataset: Any) -> dict[str, Any]:
    if not bool(getattr(dataset, "use_lidar", False)):
        return {}
    return {
        "profile": "bev_streaming_stats" if bool(getattr(dataset, "lidar_normalize", False)) else "bev_raw",
        "bev_size": list(getattr(dataset, "lidar_bev_size", [])),
        "roi": list(getattr(dataset, "lidar_roi", [])),
        "fov_degrees": _optional_list(getattr(dataset, "lidar_fov_degrees", None)),
        "remove_ground": bool(getattr(dataset, "lidar_remove_ground", False)),
        "ground_z_threshold": float(getattr(dataset, "lidar_ground_z_threshold", 0.1)),
        "background_path": getattr(dataset, "lidar_background_path", None),
        "background_distance_threshold": float(getattr(dataset, "lidar_background_distance_threshold", 0.2)),
        "cache_dir": str(getattr(dataset, "lidar_cache_dir", None))
        if getattr(dataset, "lidar_cache_dir", None) is not None
        else None,
        "cache_policy": getattr(dataset, "lidar_cache_policy", None),
        "use_cache": bool(getattr(dataset, "lidar_use_cache", False)),
        "write_cache": bool(getattr(dataset, "lidar_write_cache", False)),
        "normalization": {
            "enabled": bool(getattr(dataset, "lidar_normalize", False)),
            "mode": getattr(dataset, "lidar_normalization_mode", "none"),
            "stats_path": str(getattr(dataset, "lidar_stats_path", None))
            if getattr(dataset, "lidar_stats_path", None) is not None
            else None,
            "recompute": bool(getattr(dataset, "lidar_stats_recompute", False)),
        },
        "augment": {
            "enabled": bool(getattr(dataset, "lidar_augment", False)),
            "point_dropout": float(getattr(dataset, "lidar_point_dropout", 0.0)),
            "jitter_std": float(getattr(dataset, "lidar_jitter_std", 0.0)),
        },
    }


def lidar_preprocessing_metadata_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        return {}
    normalization = dataset_cfg.get("lidar_normalization")
    if not isinstance(normalization, dict):
        normalization = {
            "enabled": bool(dataset_cfg.get("lidar_normalize", False)),
            "mode": "streaming_stats" if bool(dataset_cfg.get("lidar_normalize", False)) else "none",
        }
    return {
        "profile": "bev_streaming_stats" if bool(normalization.get("enabled", False)) else "bev_raw",
        "bev_size": list(dataset_cfg.get("lidar_bev_size", [])),
        "roi": list(dataset_cfg.get("lidar_roi", [])),
        "fov_degrees": _optional_list(dataset_cfg.get("lidar_fov_degrees")),
        "remove_ground": bool(dataset_cfg.get("lidar_remove_ground", False)),
        "ground_z_threshold": float(dataset_cfg.get("lidar_ground_z_threshold", 0.1)),
        "background_path": dataset_cfg.get("lidar_background_path"),
        "background_distance_threshold": float(dataset_cfg.get("lidar_background_distance_threshold", 0.2)),
        "cache_dir": dataset_cfg.get("lidar_cache_dir"),
        "cache_policy": dataset_cfg.get("lidar_cache_policy"),
        "normalization": dict(normalization),
        "augment": {
            "enabled": bool(dataset_cfg.get("lidar_augment", False)),
            "point_dropout": float(dataset_cfg.get("lidar_point_dropout", 0.0)),
            "jitter_std": float(dataset_cfg.get("lidar_jitter_std", 0.0)),
        },
    }


def degradation_baselines_from_labels(
    labels: torch.Tensor,
    *,
    input_beams: torch.Tensor | None = None,
    num_classes: int,
    downsample_ratio: int = 1,
) -> dict[str, Any]:
    labels = labels.detach().to(dtype=torch.long, device="cpu")
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    majority = _majority_baseline(labels)
    result: dict[str, Any] = {"majority_class": majority}
    if input_beams is not None:
        result["last_beam"] = _last_beam_baseline(
            labels,
            input_beams.detach().to(dtype=torch.long, device="cpu"),
            num_classes=int(num_classes),
            downsample_ratio=int(downsample_ratio),
        )
    return result


def lidar_degradation_report(
    metrics: dict[str, Any],
    baselines: dict[str, Any],
    quality_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_top1 = _float_list(metrics.get("topk", {}).get("1", []))
    majority_top1 = _float_list(baselines.get("majority_class", {}).get("top1", []))
    length = min(len(model_top1), len(majority_top1))
    model_avg = _average(model_top1[:length])
    majority_avg = _average(majority_top1[:length])
    exceeds = [model_top1[idx] > majority_top1[idx] for idx in range(length)]
    reasons = []
    if length and model_avg <= majority_avg:
        reasons.append("model_not_above_majority_class")
    if quality_summary and bool(quality_summary.get("degradation_risk", False)):
        reasons.extend(str(reason) for reason in quality_summary.get("degradation_reasons", []))
    report = {
        "risk": bool(reasons),
        "reasons": sorted(set(reasons)),
        "model_top1": model_top1,
        "majority_top1": majority_top1,
        "model_top1_avg": float(model_avg),
        "majority_top1_avg": float(majority_avg),
        "model_exceeds_majority_by_horizon": exceeds,
        "model_exceeds_majority_avg": bool(not length or model_avg > majority_avg),
    }
    if quality_summary:
        report["lidar_preprocessing"] = quality_summary.get("preprocessing", {})
        if "raw" in quality_summary:
            report["lidar_quality_raw"] = quality_summary["raw"]
        if "model_input" in quality_summary:
            report["lidar_quality_model_input"] = quality_summary["model_input"]
    return report


def _majority_baseline(labels: torch.Tensor) -> dict[str, Any]:
    top1 = []
    classes = []
    totals = []
    for horizon in range(labels.shape[1]):
        values = labels[:, horizon]
        valid = values[values.ne(-100)]
        totals.append(int(valid.numel()))
        if valid.numel() == 0:
            top1.append(0.0)
            classes.append(None)
            continue
        unique, counts = torch.unique(valid, return_counts=True)
        winner_idx = int(torch.argmax(counts).item())
        top1.append(float(counts[winner_idx].item() / valid.numel()))
        classes.append(int(unique[winner_idx].item()))
    return {
        "top1": top1,
        "classes": classes,
        "total": totals,
        "avg_top1": _average(top1),
    }


def _last_beam_baseline(
    labels: torch.Tensor,
    input_beams: torch.Tensor,
    *,
    num_classes: int,
    downsample_ratio: int,
) -> dict[str, Any]:
    if input_beams.ndim != 2 or input_beams.shape[0] != labels.shape[0] or input_beams.shape[1] == 0:
        return {"available": False, "top1": [], "top3": [], "avg_top1": 0.0, "avg_top3": 0.0}
    last = torch.floor(input_beams[:, -1].float() / max(downsample_ratio, 1)).to(dtype=torch.long)
    last = torch.remainder(last, max(int(num_classes), 1))
    neighbors = torch.stack(
        [
            torch.remainder(last - 1, max(int(num_classes), 1)),
            last,
            torch.remainder(last + 1, max(int(num_classes), 1)),
        ],
        dim=1,
    )
    top1 = []
    top3 = []
    totals = []
    for horizon in range(labels.shape[1]):
        values = labels[:, horizon]
        valid = values.ne(-100)
        totals.append(int(valid.sum().item()))
        if not bool(valid.any()):
            top1.append(0.0)
            top3.append(0.0)
            continue
        horizon_labels = values[valid]
        top1.append(float(last[valid].eq(horizon_labels).float().mean().item()))
        top3.append(float(neighbors[valid].eq(horizon_labels.unsqueeze(1)).any(dim=1).float().mean().item()))
    return {
        "available": True,
        "top1": top1,
        "top3": top3,
        "total": totals,
        "avg_top1": _average(top1),
        "avg_top3": _average(top3),
        "top3_policy": "last_beam_plus_adjacent_circular",
    }


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _float_list(values: Any) -> list[float]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    return [float(value) for value in (values or [])]


def _optional_list(values: Any) -> list[Any] | None:
    if values is None:
        return None
    return list(values)


__all__ = [
    "LidarQualityAccumulator",
    "degradation_baselines_from_labels",
    "lidar_degradation_report",
    "lidar_preprocessing_metadata_from_config",
    "lidar_preprocessing_metadata_from_dataset",
]
