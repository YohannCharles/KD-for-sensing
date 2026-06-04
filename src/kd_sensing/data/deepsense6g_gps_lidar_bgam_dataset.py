from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from kd_sensing.data.deepsense6g_topk_candidate_manifest import circular_distance
from kd_sensing.data.transform_ops.lidar import (
    build_lidar_bev,
    lidar_bev_grid_metadata,
    load_lidar_bev_metadata,
    read_lidar_point_cloud,
    validate_lidar_bev_metadata,
    write_lidar_bev_metadata,
)
from kd_sensing.data.transform_ops.io import atomic_save_npy


SCALER_FIELDS = ("distance_to_rsu", "candidate_logit")


@dataclass(frozen=True)
class GPSLidarBGAMNormalizer:
    mean: dict[str, float]
    scale: dict[str, float]
    metadata: dict[str, Any]

    def transform(self, field: str, value: float) -> float:
        return (float(value) - float(self.mean.get(field, 0.0))) / max(float(self.scale.get(field, 1.0)), 1e-8)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": dict(self.mean), "scale": dict(self.scale), "metadata": dict(self.metadata)}


def fit_gps_lidar_bgam_normalizer(
    rows: Sequence[Mapping[str, Any]],
    *,
    topk: int = 8,
) -> GPSLidarBGAMNormalizer:
    fit_rows = [row for row in rows if not _is_query_role(row.get("support_query_role") or row.get("split_role"))]
    values: dict[str, list[float]] = {field: [] for field in SCALER_FIELDS}
    for row in fit_rows:
        values["distance_to_rsu"].append(_float(row.get("distance_to_rsu"), _float(row.get("range"), 0.0)))
        for idx in range(int(topk)):
            value = row.get(f"cand{idx}_logit")
            if value not in {None, ""}:
                values["candidate_logit"].append(_float(value, 0.0))
    mean: dict[str, float] = {}
    scale: dict[str, float] = {}
    for field, items in values.items():
        array = np.asarray(items, dtype=np.float64)
        if array.size == 0:
            mean[field] = 0.0
            scale[field] = 1.0
        else:
            mean[field] = float(array.mean())
            std = float(array.std())
            scale[field] = std if std > 1e-8 else 1.0
    return GPSLidarBGAMNormalizer(
        mean=mean,
        scale=scale,
        metadata={
            "fit_sample_count": len(fit_rows),
            "excluded_target_query_count": len(rows) - len(fit_rows),
            "fields": list(SCALER_FIELDS),
            "query_label_used_for_training": False,
        },
    )


class GPSLidarBGAMDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        topk: int = 8,
        num_beams: int = 64,
        data_root: str | Path = "",
        lidar_cfg: Mapping[str, Any] | None = None,
        load_lidar: bool = True,
        lidar_profile: str | None = None,
        missing_lidar_policy: str = "zeros",
        normalizer: GPSLidarBGAMNormalizer | None = None,
        fit_normalizer: bool = True,
        include_query_labels: bool = True,
        training_only: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = _read_csv(self.manifest_path)
        self.topk = int(topk)
        self.num_beams = int(num_beams)
        self.data_root = Path(data_root) if str(data_root) else Path("")
        self.lidar_cfg = dict(lidar_cfg or {})
        self.load_lidar = bool(load_lidar)
        self.lidar_profile = str(lidar_profile or self.lidar_cfg.get("profile", "bev_cache"))
        self.missing_lidar_policy = str(missing_lidar_policy or self.lidar_cfg.get("missing_policy", "zeros"))
        self.include_query_labels = bool(include_query_labels)
        self.training_only = bool(training_only)
        if self.training_only:
            self.rows = [row for row in self.rows if not _is_query_role(row.get("support_query_role") or row.get("split_role"))]
        if normalizer is not None:
            self.normalizer = normalizer
        elif fit_normalizer:
            self.normalizer = fit_gps_lidar_bgam_normalizer(self.rows, topk=self.topk)
        else:
            self.normalizer = GPSLidarBGAMNormalizer(
                mean={field: 0.0 for field in SCALER_FIELDS},
                scale={field: 1.0 for field in SCALER_FIELDS},
                metadata={"fit_sample_count": 0, "query_label_used_for_training": False},
            )
        self.bev_size = tuple(int(value) for value in self.lidar_cfg.get("bev_size", (64, 64)))
        self.roi = tuple(float(value) for value in self.lidar_cfg.get("roi", (-30.0, 30.0, -30.0, 30.0, -3.0, 5.0)))
        self.input_channels = int(self.lidar_cfg.get("input_channels", 3))
        self.expected_bev_metadata = lidar_bev_grid_metadata(
            bev_size=self.bev_size,
            roi=self.roi,
            fov_degrees=self.lidar_cfg.get("fov_degrees"),
            remove_ground=bool(self.lidar_cfg.get("remove_ground", False)),
            ground_z_threshold=float(self.lidar_cfg.get("ground_z_threshold", 0.1)),
            cell_center_convention=str(self.lidar_cfg.get("cell_center_convention", "center")),
            cache_version=str(self.lidar_cfg.get("cache_version", "bgam_bev_v1")),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index)]
        role = str(row.get("support_query_role") or row.get("split_role") or "")
        target = _int(row.get("gt_beam"), _int(row.get("target_label"), -100))
        if role.startswith("query") and not self.include_query_labels:
            target = -100
        candidate_beams = torch.tensor([_int(row.get(f"cand{idx}_beam"), -1) for idx in range(self.topk)], dtype=torch.long)
        candidate_logits = torch.tensor([_float(row.get(f"cand{idx}_logit"), -1e9) for idx in range(self.topk)], dtype=torch.float32)
        candidate_probs = torch.tensor([_float(row.get(f"cand{idx}_prob"), 0.0) for idx in range(self.topk)], dtype=torch.float32)
        if float(candidate_probs.sum()) <= 0.0:
            candidate_probs = torch.softmax(candidate_logits, dim=-1)
        candidate_probs = candidate_probs / candidate_probs.sum().clamp_min(1e-12)
        target_index = _int(row.get("target_candidate_index"), -1)
        if target_index < 0 and target >= 0:
            matches = torch.nonzero(candidate_beams.eq(int(target)), as_tuple=False).reshape(-1)
            target_index = int(matches[0].item()) if matches.numel() else -1
        nearest_index = _int(row.get("nearest_candidate_index"), -1)
        if nearest_index < 0 and target >= 0:
            distances = [circular_distance(int(beam), target, num_beams=self.num_beams) for beam in candidate_beams.tolist()]
            nearest_index = int(np.argmin(np.asarray(distances, dtype=np.int64))) if distances else -1
        theta = _float(row.get("theta_gps"), math.radians(_float(row.get("theta_degrees"), _float(row.get("theta"), 0.0))))
        distance = _float(row.get("distance_to_rsu"), _float(row.get("range"), 0.0))
        item: dict[str, Any] = {
            "row_index": torch.tensor(int(index), dtype=torch.long),
            "sample_id": str(row.get("sample_id") or ""),
            "scene": str(row.get("scene") or ""),
            "scenario_id": str(row.get("scenario_id") or row.get("scene") or ""),
            "support_query_role": role,
            "split_role": str(row.get("split_role") or role),
            "candidate_beams": candidate_beams,
            "candidate_logits": candidate_logits,
            "candidate_probs": candidate_probs,
            "candidate_log_probs": torch.log(candidate_probs.clamp_min(1e-12)),
            "theta_gps": torch.tensor(float(theta), dtype=torch.float32),
            "distance_to_rsu": torch.tensor(float(distance), dtype=torch.float32),
            "distance_to_rsu_norm": torch.tensor(self.normalizer.transform("distance_to_rsu", distance), dtype=torch.float32),
            "gt_beam": torch.tensor(target, dtype=torch.long),
            "target_label": torch.tensor(target, dtype=torch.long),
            "target_in_topk": torch.tensor(target_index >= 0, dtype=torch.bool),
            "target_in_top8": torch.tensor(target_index >= 0, dtype=torch.bool),
            "target_candidate_index": torch.tensor(target_index, dtype=torch.long),
            "nearest_candidate_index": torch.tensor(nearest_index, dtype=torch.long),
            "gps_top1": torch.tensor(_int(row.get("gps_top1"), int(candidate_beams[0].item() if candidate_beams.numel() else -1)), dtype=torch.long),
            "gps_entropy": torch.tensor(_float(row.get("gps_entropy"), 0.0), dtype=torch.float32),
            "beam_angle_source": str(row.get("beam_angle_source") or ""),
            "label_space": str(row.get("label_space") or ""),
            "beam_label_space": str(row.get("beam_label_space") or row.get("history_beam_label_space") or ""),
            "beam_label_mapping_fingerprint": str(
                row.get("beam_label_mapping_fingerprint") or row.get("history_beam_label_mapping_fingerprint") or ""
            ),
            "bgam_metadata": {
                "coordinate_frame": str(row.get("coordinate_frame") or ""),
                "beam_angle_source": str(row.get("beam_angle_source") or ""),
                "lidar_missing_reason": str(row.get("lidar_missing_reason") or ""),
                "label_space": str(row.get("label_space") or ""),
                "beam_label_space": str(row.get("beam_label_space") or row.get("history_beam_label_space") or ""),
                "beam_label_mapping_fingerprint": str(
                    row.get("beam_label_mapping_fingerprint") or row.get("history_beam_label_mapping_fingerprint") or ""
                ),
                "history_pseudo_label_source": str(row.get("history_pseudo_label_source") or ""),
                "history_alignment_policy": str(row.get("history_alignment_policy") or ""),
            },
        }
        history = _history_tensors(row)
        item.update(history)
        gps_probs = _gps_prior_vector(row, prefix="gps_prob_", num_beams=self.num_beams)
        gps_logits = _gps_prior_vector(row, prefix="gps_logit_", num_beams=self.num_beams)
        if gps_probs is not None:
            item["gps_probs"] = torch.tensor(gps_probs, dtype=torch.float32)
        if gps_logits is not None:
            item["gps_logits"] = torch.tensor(gps_logits, dtype=torch.float32)

        if self.load_lidar:
            lidar_payload = self._load_lidar_payload(row)
            item.update(lidar_payload)
        return item

    def _load_lidar_payload(self, row: Mapping[str, Any]) -> dict[str, Any]:
        if self.lidar_profile == "pillar6":
            points = self._load_raw_points(row)
            return {"raw_points": torch.tensor(points, dtype=torch.float32), "lidar_nonempty": torch.tensor(points.size > 0)}
        bev = self._load_bev(row)
        return {"lidar_bev": bev, "lidar_nonempty": torch.tensor(bool(torch.count_nonzero(bev).item() > 0))}

    def _load_bev(self, row: Mapping[str, Any]) -> torch.Tensor:
        path = _resolve_path(row.get("lidar_bev_cache_path"), data_root="", scene="")
        if path is not None and path.exists():
            metadata = load_lidar_bev_metadata(path)
            validate_lidar_bev_metadata(metadata, self.expected_bev_metadata, fields=("roi", "bev_size", "cell_center_convention"))
            array = _load_array(path)
            return _coerce_bev(array, bev_size=self.bev_size, channels=self.input_channels)
        raw_path = _resolve_path(row.get("lidar_path"), data_root=str(self.data_root), scene=str(row.get("scene") or ""))
        if raw_path is not None and raw_path.exists() and bool(self.lidar_cfg.get("rebuild_cache_if_missing", False)):
            array = build_lidar_bev(
                raw_path.parent,
                raw_path.name,
                bev_size=self.bev_size,
                roi=self.roi,
                fov_degrees=self.lidar_cfg.get("fov_degrees"),
                remove_ground=bool(self.lidar_cfg.get("remove_ground", False)),
                ground_z_threshold=float(self.lidar_cfg.get("ground_z_threshold", 0.1)),
            )
            if path is not None:
                atomic_save_npy(path, array.astype(np.float32))
                write_lidar_bev_metadata(path, self.expected_bev_metadata)
            return _coerce_bev(array, bev_size=self.bev_size, channels=self.input_channels)
        if self.missing_lidar_policy in {"zeros", "skip"}:
            return torch.zeros((self.input_channels, *self.bev_size), dtype=torch.float32)
        raise FileNotFoundError(f"Missing LiDAR BEV cache for sample {row.get('sample_id', '')}: {row.get('lidar_bev_cache_path', '')}")

    def _load_raw_points(self, row: Mapping[str, Any]) -> np.ndarray:
        path = _resolve_path(row.get("lidar_path"), data_root=str(self.data_root), scene=str(row.get("scene") or ""))
        if path is not None and path.exists():
            root = path.parent
            rel = path.name
            return read_lidar_point_cloud(root, rel)
        if self.missing_lidar_policy in {"zeros", "skip"}:
            return np.empty((0, 4), dtype=np.float32)
        raise FileNotFoundError(f"Missing raw LiDAR point cloud for sample {row.get('sample_id', '')}: {row.get('lidar_path', '')}")


def build_gps_lidar_bgam_dataloader(
    manifest_path: str | Path,
    *,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    **dataset_kwargs: Any,
) -> DataLoader:
    dataset = GPSLidarBGAMDataset(manifest_path, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=collate_gps_lidar_bgam_batch,
    )


def collate_gps_lidar_bgam_batch(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty GPS+LiDAR BGAM batch.")
    batch: dict[str, Any] = {}
    tensor_keys = {key for item in items for key, value in item.items() if torch.is_tensor(value)}
    for key in sorted(tensor_keys):
        values = [item[key] for item in items if key in item]
        if len(values) != len(items):
            continue
        shapes = {tuple(value.shape) for value in values}
        batch[key] = torch.stack(values) if len(shapes) == 1 else values
    for key in (
        "sample_id",
        "scene",
        "scenario_id",
        "support_query_role",
        "split_role",
        "beam_angle_source",
        "label_space",
        "beam_label_space",
        "beam_label_mapping_fingerprint",
    ):
        batch[key] = [str(item.get(key, "")) for item in items]
    batch["bgam_metadata"] = [dict(item.get("bgam_metadata", {})) for item in items]
    if "raw_points" in tensor_keys:
        batch["raw_points"] = [item.get("raw_points", torch.empty(0, 4)) for item in items]
    return batch


def _load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        payload = np.load(path)
        return np.asarray(payload[payload.files[0]])
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, Mapping):
            for value in payload.values():
                if torch.is_tensor(value) or isinstance(value, np.ndarray):
                    return np.asarray(value)
        return np.asarray(payload)
    raise ValueError(f"Unsupported LiDAR BEV cache suffix: {path}")


def _coerce_bev(array: np.ndarray, *, bev_size: tuple[int, int], channels: int) -> torch.Tensor:
    bev = np.asarray(array, dtype=np.float32)
    if bev.ndim == 2:
        bev = bev[None, ...]
    if bev.ndim != 3:
        raise ValueError(f"LiDAR BEV must have shape [C,H,W], got {bev.shape}.")
    target_h, target_w = int(bev_size[0]), int(bev_size[1])
    if bev.shape[1:] != (target_h, target_w):
        from PIL import Image

        resized = np.zeros((bev.shape[0], target_h, target_w), dtype=np.float32)
        for channel in range(bev.shape[0]):
            resized[channel] = np.asarray(Image.fromarray(bev[channel]).resize((target_w, target_h), resample=Image.BILINEAR))
        bev = resized
    if bev.shape[0] < int(channels):
        padded = np.zeros((int(channels), target_h, target_w), dtype=np.float32)
        padded[: bev.shape[0]] = bev
        bev = padded
    elif bev.shape[0] > int(channels):
        bev = bev[: int(channels)]
    return torch.tensor(bev, dtype=torch.float32)


def _resolve_path(value: Any, *, data_root: str, scene: str) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.exists():
        return path
    candidates = []
    if data_root:
        root = Path(data_root)
        candidates.append(root / raw)
        if scene:
            candidates.append(root / scene / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _gps_prior_vector(row: Mapping[str, Any], *, prefix: str, num_beams: int) -> np.ndarray | None:
    values = [_float(row.get(f"{prefix}{idx}"), math.nan) for idx in range(int(num_beams))]
    if not any(math.isfinite(value) for value in values):
        return None
    array = np.asarray([value if math.isfinite(value) else 0.0 for value in values], dtype=np.float32)
    if prefix == "gps_prob_":
        array = array / np.clip(array.sum(), 1e-12, None)
    return array


def _history_tensors(row: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    beams = _json_int_list(row.get("history_pseudo_beams"))
    probs = _json_float_list(row.get("history_pseudo_probs"))
    entropy = _json_float_list(row.get("history_pseudo_entropy"))
    confidence = _json_float_list(row.get("history_pseudo_confidence"))
    valid = _json_bool_list(row.get("history_valid_mask"))
    length = max(len(beams), len(probs), len(entropy), len(confidence), len(valid))
    if length == 0:
        return {
            "history_pseudo_beams": torch.empty(0, dtype=torch.long),
            "history_pseudo_probs": torch.empty(0, dtype=torch.float32),
            "history_pseudo_log_probs": torch.empty(0, dtype=torch.float32),
            "history_pseudo_entropy": torch.empty(0, dtype=torch.float32),
            "history_pseudo_confidence": torch.empty(0, dtype=torch.float32),
            "history_valid_mask": torch.empty(0, dtype=torch.bool),
        }
    beams = _pad(beams, length, -1)
    probs = _pad(probs, length, 0.0)
    entropy = _pad(entropy, length, 0.0)
    confidence = _pad(confidence, length, 0.0)
    valid = _pad(valid, length, False)
    probs_tensor = torch.tensor(probs, dtype=torch.float32)
    return {
        "history_pseudo_beams": torch.tensor(beams, dtype=torch.long),
        "history_pseudo_probs": probs_tensor,
        "history_pseudo_log_probs": torch.log(probs_tensor.clamp_min(1e-12)),
        "history_pseudo_entropy": torch.tensor(entropy, dtype=torch.float32),
        "history_pseudo_confidence": torch.tensor(confidence, dtype=torch.float32),
        "history_valid_mask": torch.tensor(valid, dtype=torch.bool),
    }


def _json_int_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    try:
        return [int(item) for item in json.loads(str(value or "[]"))]
    except Exception:
        return []


def _json_float_list(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    try:
        return [float(item) for item in json.loads(str(value or "[]"))]
    except Exception:
        return []


def _json_bool_list(value: Any) -> list[bool]:
    if isinstance(value, list):
        return [bool(item) for item in value]
    try:
        return [bool(item) for item in json.loads(str(value or "[]"))]
    except Exception:
        return []


def _pad(values: Sequence[Any], length: int, fill: Any) -> list[Any]:
    out = list(values[: int(length)])
    if len(out) < int(length):
        out.extend([fill for _ in range(int(length) - len(out))])
    return out


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _is_query_role(value: Any) -> bool:
    return str(value or "").startswith("query")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in {None, ""}:
            return int(default)
        return int(float(str(value)))
    except (TypeError, ValueError):
        return int(default)


__all__ = [
    "GPSLidarBGAMDataset",
    "GPSLidarBGAMNormalizer",
    "build_gps_lidar_bgam_dataloader",
    "collate_gps_lidar_bgam_batch",
    "fit_gps_lidar_bgam_normalizer",
]
