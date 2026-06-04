from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from kd_sensing.data.deepsense6g_topk_candidate_manifest import circular_distance


CANDIDATE_FEATURE_NAMES = (
    "beam_sin",
    "beam_cos",
    "rank_norm",
    "logit_norm",
    "prob",
    "log_prob",
    "dist_to_gps_top1_norm",
    "is_gps_top1",
    "is_gps_top3",
    "is_gps_top5",
)
GPS_CONTEXT_FEATURE_NAMES = (
    "E_norm",
    "N_norm",
    "sin_theta",
    "cos_theta",
    "log_range_norm",
    "sin_heading",
    "cos_heading",
    "speed_norm",
    "gps_top1_prob",
    "gps_top2_prob",
    "gps_margin",
    "gps_entropy",
    "gps_pred_beam_sin",
    "gps_pred_beam_cos",
)
SCALER_FIELDS = ("E", "N", "log_range", "speed", "candidate_logit")


@dataclass(frozen=True)
class TopKCandidateNormalizer:
    mean: dict[str, float]
    scale: dict[str, float]
    metadata: dict[str, Any]

    def transform(self, field: str, value: float) -> float:
        return (float(value) - float(self.mean.get(field, 0.0))) / max(float(self.scale.get(field, 1.0)), 1e-8)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": dict(self.mean), "scale": dict(self.scale), "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TopKCandidateNormalizer":
        return cls(
            mean={str(key): float(value) for key, value in _mapping(payload.get("mean")).items()},
            scale={str(key): float(value) for key, value in _mapping(payload.get("scale")).items()},
            metadata=dict(_mapping(payload.get("metadata"))),
        )


def fit_topk_candidate_normalizer(
    rows: Sequence[Mapping[str, Any]],
    *,
    topk: int = 8,
    seed: int = 42,
    fit_split: str = "source_or_support",
) -> TopKCandidateNormalizer:
    fit_rows = [row for row in rows if not str(row.get("support_query_role") or row.get("split_role") or "").startswith("query")]
    values: dict[str, list[float]] = {field: [] for field in SCALER_FIELDS}
    for row in fit_rows:
        easting = _float(row.get("E"), 0.0)
        northing = _float(row.get("N"), 0.0)
        rel_range = _float(row.get("range"), math.sqrt(easting * easting + northing * northing))
        values["E"].append(easting)
        values["N"].append(northing)
        values["log_range"].append(math.log1p(max(rel_range, 0.0)))
        values["speed"].append(_float(row.get("speed"), 0.0))
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
    metadata = {
        "fit_split": fit_split,
        "fit_sample_count": len(fit_rows),
        "excluded_target_query_count": len(rows) - len(fit_rows),
        "fields": list(SCALER_FIELDS),
        "candidate_feature_names": list(CANDIDATE_FEATURE_NAMES),
        "gps_context_feature_names": list(GPS_CONTEXT_FEATURE_NAMES),
        "seed": int(seed),
        "query_label_used_for_training": False,
    }
    return TopKCandidateNormalizer(mean=mean, scale=scale, metadata=metadata)


def save_topk_candidate_normalizer(path: str | Path, normalizer: TopKCandidateNormalizer) -> None:
    candidate = Path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(json.dumps(_json_ready(normalizer.to_dict()), indent=2, sort_keys=True), encoding="utf-8")


def load_topk_candidate_normalizer(path: str | Path) -> TopKCandidateNormalizer:
    return TopKCandidateNormalizer.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class TopKCandidateManifestDataset(Dataset):
    """Dataset backed by Top8 candidate manifest rows."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        topk: int = 8,
        num_beams: int = 64,
        enabled_modalities: Sequence[str] | None = None,
        normalizer: TopKCandidateNormalizer | None = None,
        normalizer_path: str | Path | None = None,
        fit_normalizer: bool = True,
        save_normalizer: bool = False,
        include_query_labels: bool = True,
        training_only: bool = False,
        image_size: int = 64,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = _read_csv(self.manifest_path)
        self.topk = int(topk)
        self.num_beams = int(num_beams)
        self.enabled_modalities = tuple(str(item) for item in (enabled_modalities or ("gps_context",)))
        self.include_query_labels = bool(include_query_labels)
        self.training_only = bool(training_only)
        self.image_size = int(image_size)
        if self.training_only:
            self.rows = [
                row
                for row in self.rows
                if not str(row.get("support_query_role") or row.get("split_role") or "").startswith("query")
            ]
        if normalizer is not None:
            self.normalizer = normalizer
        elif normalizer_path is not None and Path(normalizer_path).exists():
            self.normalizer = load_topk_candidate_normalizer(normalizer_path)
        elif fit_normalizer:
            self.normalizer = fit_topk_candidate_normalizer(self.rows, topk=self.topk)
            if save_normalizer and normalizer_path is not None:
                save_topk_candidate_normalizer(normalizer_path, self.normalizer)
        else:
            self.normalizer = TopKCandidateNormalizer(
                mean={field: 0.0 for field in SCALER_FIELDS},
                scale={field: 1.0 for field in SCALER_FIELDS},
                metadata={"fit_split": "identity", "query_label_used_for_training": False},
            )
        self._array_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index)]
        role = str(row.get("support_query_role") or row.get("split_role") or "")
        target_label = _int(row.get("target_label"), -100)
        if role.startswith("query") and not self.include_query_labels:
            target_label = -100
        candidate_beams = torch.tensor([_int(row.get(f"cand{idx}_beam"), -1) for idx in range(self.topk)], dtype=torch.long)
        candidate_logits = torch.tensor([_float(row.get(f"cand{idx}_logit"), -1e9) for idx in range(self.topk)], dtype=torch.float32)
        candidate_probs = torch.tensor([_float(row.get(f"cand{idx}_prob"), 0.0) for idx in range(self.topk)], dtype=torch.float32)
        if float(candidate_probs.sum()) <= 0.0:
            candidate_probs = torch.softmax(candidate_logits, dim=-1)
        candidate_features = torch.tensor(_candidate_features(row, self.normalizer, topk=self.topk, num_beams=self.num_beams), dtype=torch.float32)
        gps_context = torch.tensor(_gps_context(row, self.normalizer, num_beams=self.num_beams), dtype=torch.float32)
        item: dict[str, Any] = {
            "row_index": torch.tensor(int(index), dtype=torch.long),
            "sample_id": str(row.get("sample_id") or ""),
            "scene": str(row.get("scene") or ""),
            "support_query_role": role,
            "split_role": str(row.get("split_role") or role),
            "candidate_beams": candidate_beams,
            "candidate_logits": candidate_logits,
            "candidate_probs": candidate_probs,
            "candidate_log_probs": torch.log(candidate_probs.clamp_min(1e-12)),
            "candidate_features": candidate_features,
            "gps_context": gps_context,
            "target_label": torch.tensor(target_label, dtype=torch.long),
            "target_in_top8": torch.tensor(_bool(row.get("target_in_top8")), dtype=torch.bool),
            "target_candidate_index": torch.tensor(_int(row.get("target_candidate_index"), -1), dtype=torch.long),
            "nearest_candidate_index": torch.tensor(_int(row.get("nearest_candidate_index"), -1), dtype=torch.long),
            "top8_oracle_error": torch.tensor(_float(row.get("top8_oracle_error"), math.inf), dtype=torch.float32),
            "miss_label": torch.tensor(1.0 if _bool(row.get("top8_miss")) else 0.0, dtype=torch.float32),
            "gps_error": torch.tensor(_float(row.get("gps_circular_error"), math.inf), dtype=torch.float32),
            "image_exists": torch.tensor(_bool(row.get("image_exists")), dtype=torch.bool),
            "camera_ae_feature_available": torch.tensor(_bool(row.get("camera_ae_feature_available")), dtype=torch.bool),
            "lidar_feature_available": torch.tensor(_bool(row.get("lidar_feature_available")), dtype=torch.bool),
            "radar_feature_available": torch.tensor(_bool(row.get("radar_feature_available")), dtype=torch.bool),
        }
        if "camera_ae" in self.enabled_modalities:
            item["camera_ae_feature"] = self._load_indexed_feature(row, "camera_ae_feature_path", "camera_ae_feature_row_index")
        if "image" in self.enabled_modalities and _bool(row.get("image_exists")):
            item["image"] = _load_image_tensor(row.get("image_path"), image_size=self.image_size)
        if "lidar" in self.enabled_modalities:
            item["lidar_feature"] = self._load_path_feature(row.get("lidar_feature_path"))
        if "radar" in self.enabled_modalities:
            item["radar_feature"] = self._load_path_feature(row.get("radar_feature_path"))
        return item

    def _load_indexed_feature(self, row: Mapping[str, Any], path_key: str, index_key: str) -> torch.Tensor:
        path = str(row.get(path_key) or "")
        row_index = _int(row.get(index_key), -1)
        if path and row_index >= 0 and Path(path).exists():
            array = self._cached_array(path)
            if row_index < int(array.shape[0]):
                return torch.tensor(array[row_index], dtype=torch.float32)
        return torch.zeros(len(GPS_CONTEXT_FEATURE_NAMES), dtype=torch.float32)

    def _load_path_feature(self, path_value: Any) -> torch.Tensor:
        path = str(path_value or "")
        if path and Path(path).exists():
            suffix = Path(path).suffix.lower()
            if suffix == ".npy":
                return torch.tensor(np.load(path), dtype=torch.float32).reshape(-1)
            if suffix == ".npz":
                payload = np.load(path)
                first = payload.files[0]
                return torch.tensor(payload[first], dtype=torch.float32).reshape(-1)
            if suffix in {".pt", ".pth"}:
                return torch.as_tensor(torch.load(path, map_location="cpu"), dtype=torch.float32).reshape(-1)
        return torch.zeros(1, dtype=torch.float32)

    def _cached_array(self, path: str) -> np.ndarray:
        if path not in self._array_cache:
            self._array_cache[path] = np.load(path)
        return self._array_cache[path]


def build_topk_candidate_dataloader(
    manifest_path: str | Path,
    *,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    **dataset_kwargs: Any,
) -> DataLoader:
    dataset = TopKCandidateManifestDataset(manifest_path, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=collate_topk_candidate_batch,
    )


def collate_topk_candidate_batch(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty TopK candidate batch.")
    batch: dict[str, Any] = {}
    tensor_keys = {key for item in items for key, value in item.items() if torch.is_tensor(value)}
    for key in sorted(tensor_keys):
        values = [item[key] for item in items if key in item]  # type: ignore[index]
        if len(values) != len(items):
            continue
        shapes = {tuple(value.shape) for value in values}
        batch[key] = torch.stack(values) if len(shapes) == 1 else values
    for key in ("sample_id", "scene", "support_query_role", "split_role"):
        batch[key] = [str(item.get(key, "")) for item in items]
    return batch


def _candidate_features(
    row: Mapping[str, Any],
    normalizer: TopKCandidateNormalizer,
    *,
    topk: int,
    num_beams: int,
) -> list[list[float]]:
    features: list[list[float]] = []
    denom = max(int(topk) - 1, 1)
    max_dist = max(int(num_beams) / 2.0, 1.0)
    for idx in range(int(topk)):
        beam = _int(row.get(f"cand{idx}_beam"), 0) % int(num_beams)
        angle = (float(beam) / float(num_beams)) * 2.0 * math.pi
        rank = _int(row.get(f"cand{idx}_rank"), idx + 1)
        logit = _float(row.get(f"cand{idx}_logit"), 0.0)
        prob = max(_float(row.get(f"cand{idx}_prob"), 0.0), 1e-12)
        dist = _float(row.get(f"cand{idx}_dist_to_gps_top1"), 0.0)
        features.append(
            [
                math.sin(angle),
                math.cos(angle),
                float(max(rank - 1, 0) / denom),
                normalizer.transform("candidate_logit", logit),
                prob,
                math.log(prob),
                float(dist / max_dist),
                1.0 if idx == 0 else 0.0,
                1.0 if idx < 3 else 0.0,
                1.0 if idx < 5 else 0.0,
            ]
        )
    return features


def _gps_context(row: Mapping[str, Any], normalizer: TopKCandidateNormalizer, *, num_beams: int) -> list[float]:
    easting = _float(row.get("E"), 0.0)
    northing = _float(row.get("N"), 0.0)
    rel_range = _float(row.get("range"), math.sqrt(easting * easting + northing * northing))
    theta = math.radians(_float(row.get("theta_degrees") or row.get("theta"), 0.0))
    heading = math.radians(_float(row.get("heading_degrees") or row.get("heading"), 0.0))
    pred = _int(row.get("gps_top1"), _int(row.get("gps_pred_top1"), 0)) % int(num_beams)
    pred_angle = (float(pred) / float(num_beams)) * 2.0 * math.pi
    return [
        normalizer.transform("E", easting),
        normalizer.transform("N", northing),
        math.sin(theta),
        math.cos(theta),
        normalizer.transform("log_range", math.log1p(max(rel_range, 0.0))),
        math.sin(heading),
        math.cos(heading),
        normalizer.transform("speed", _float(row.get("speed"), 0.0)),
        _float(row.get("gps_top1_prob"), 0.0),
        _float(row.get("gps_top2_prob"), 0.0),
        _float(row.get("gps_top1_top2_margin"), 0.0),
        _float(row.get("gps_entropy"), 0.0),
        math.sin(pred_angle),
        math.cos(pred_angle),
    ]


def _load_image_tensor(path_value: Any, *, image_size: int) -> torch.Tensor:
    from PIL import Image

    path = Path(str(path_value or ""))
    image = Image.open(path).convert("RGB").resize((int(image_size), int(image_size)))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - 0.5) / 0.5


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in {None, ""}:
            return int(default)
        return int(float(str(value)))
    except (TypeError, ValueError):
        return int(default)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


__all__ = [
    "CANDIDATE_FEATURE_NAMES",
    "GPS_CONTEXT_FEATURE_NAMES",
    "TopKCandidateManifestDataset",
    "TopKCandidateNormalizer",
    "build_topk_candidate_dataloader",
    "collate_topk_candidate_batch",
    "fit_topk_candidate_normalizer",
    "load_topk_candidate_normalizer",
    "save_topk_candidate_normalizer",
]
