from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from kd_sensing.data.deepsense6g_residual import (
    FALLBACK_PRIOR_SOURCE,
    PREDICTION_LOGIT_NAMES,
    build_residual_manifest,
    circular_gaussian_prior_from_top1,
    load_gps_prior_artifacts,
    ratio_tag,
)
from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.evaluation.metrics import (
    circular_beam_distance,
    delta_class_to_residual,
    residual_delta_class_count,
    residual_to_delta_class,
    signed_circular_residual,
)


CAMERA_MANIFEST_NAME = "camera_residual_manifest.csv"
CAMERA_MANIFEST_WITH_AE_NAME = "camera_residual_manifest_with_ae.csv"
CAMERA_METADATA_NAME = "camera_residual_manifest_metadata.json"
CAMERA_PRIOR_LOGITS_NAME = "camera_gps_prior_logits.npy"
SAVED_LOGITS_PRIOR_SOURCE = "saved_logits"
UNAVAILABLE_PRIOR_SOURCE = "unavailable_support_prior"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def build_camera_residual_manifest(
    cfg: Mapping[str, Any],
    *,
    support_ratio: float | None = None,
    label_space: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build the camera residual manifest from GPS v2 residual inputs."""
    data_cfg = _mapping(cfg.get("data"))
    residual_cfg = _mapping(cfg.get("residual"))
    outputs_cfg = _mapping(cfg.get("outputs"))
    ratio = float(support_ratio if support_ratio is not None else data_cfg.get("support_ratio", 0.15))
    selected_label_space = str(label_space or data_cfg.get("label_space", "mapping_disabled"))
    tag = ratio_tag(ratio)
    out_root = Path(
        output_dir
        or outputs_cfg.get("analysis_root")
        or outputs_cfg.get("root", "outputs/analysis/deepsense6g_camera_residual")
    )
    result_dir = out_root / tag / selected_label_space
    manifest_dir = result_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    base_result = build_residual_manifest(
        cfg,
        support_ratio=ratio,
        label_space=selected_label_space,
        output_dir=out_root,
    )
    base_manifest = Path(str(base_result["manifest_path"]))
    base_rows = _read_csv(base_manifest)

    num_beams = int(data_cfg.get("num_beams", 64))
    radius = int(residual_cfg.get("delta_radius", 8))
    fallback_sigma = float(residual_cfg.get("gps_prior_fallback_sigma", 2.0))
    gps_root = Path(str(data_cfg.get("gps_sweep_root", "outputs/analysis/deepsense6g_gps_adapter_v2_support_sweep")))
    gps_dir = gps_root / tag / selected_label_space
    protocol = str(residual_cfg.get("gps_protocol", "target_adapt_beambench"))
    ablation = _resolve_best_gps_ablation(gps_dir, residual_cfg.get("gps_ablation", "best"))
    prior_records = _safe_load_prior_records(
        gps_dir,
        protocol=protocol,
        ablation=ablation,
        num_beams=num_beams,
        fallback_sigma=fallback_sigma,
    )

    data_root = Path(str(data_cfg.get("data_root", "dataset/DeepSense6G")))
    csv_index = _DeepSenseImageIndex(data_root, data_cfg)
    camera_rows: list[dict[str, Any]] = []
    prior_logits: list[np.ndarray] = []
    warnings: list[str] = list(base_result.get("warnings") or [])

    for row_index, row in enumerate(base_rows):
        scene = str(row.get("scene") or "")
        sample_id = str(row.get("sample_id") or "")
        pred = _int(row.get("gps_pred_top1"), -1)
        target = _int(row.get("target_label"), -100)
        prior = prior_records.get((scene, sample_id))
        logits, prior_source, source_artifact = _prior_logits_for_row(
            prior,
            pred,
            num_beams=num_beams,
            fallback_sigma=fallback_sigma,
        )
        prior_logits.append(logits.astype(np.float32, copy=False))
        probs = _softmax_np(logits.reshape(1, -1)).reshape(-1)
        role = str(row.get("support_query_role") or row.get("split_role") or "")
        split_role = role or ("support" if str(row.get("split")) == "train" else "query_test")
        support_or_query = "support" if split_role == "support" else "query"
        raw = csv_index.lookup(sample_id)
        image_path = str(row.get("image_path") or "")
        discovered = discover_deepsense6g_image_path(
            raw,
            data_root=data_root,
            scene=scene,
            sample_id=sample_id,
            cfg=_mapping(cfg.get("image")),
        )
        if not image_path and discovered:
            image_path = discovered
        image_exists = bool(image_path) and Path(image_path).exists()
        residual_value = _optional_int(row.get("gps_signed_residual"))
        if residual_value is None and pred >= 0 and target >= 0:
            residual_value = int(signed_circular_residual(target, pred, num_beams=num_beams))
        delta_class = (
            int(residual_to_delta_class(residual_value, radius=radius))
            if residual_value is not None
            else int(residual_delta_class_count(radius=radius) - 1)
        )
        overflow_class = int(residual_delta_class_count(radius=radius) - 1)
        gps_error = _optional_float(row.get("gps_circular_error"))
        if gps_error is None and pred >= 0 and target >= 0:
            gps_error = float(circular_beam_distance(pred, target, num_beams=num_beams))

        augmented = dict(row)
        augmented.update(
            {
                "support_or_query": support_or_query,
                "split_role": split_role,
                "gps_pred_top1": pred,
                "gps_error": "" if gps_error is None else float(gps_error),
                "gps_circular_error": "" if gps_error is None else float(gps_error),
                "gps_signed_residual": "" if residual_value is None else int(residual_value),
                "gps_residual_delta_class": delta_class,
                "gps_residual_overflow": bool(delta_class == overflow_class),
                "gps_prior_source": prior_source,
                "gps_prior_source_artifact": source_artifact,
                "gps_prior_logits_path": str(manifest_dir / CAMERA_PRIOR_LOGITS_NAME),
                "gps_logits_row_index": row_index,
                "gps_prior_row_index": row.get("gps_prior_row_index", ""),
                "gps_prior_peak_prob": float(np.max(probs)) if probs.size else 0.0,
                "gps_prior_entropy": float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum()) if probs.size else 0.0,
                "image_path": image_path,
                "image_exists": bool(image_exists),
                "ae_feature_row_index": -1,
                "ae_feature_path": "",
                "ae_feature_available": False,
                "ae_feature_unavailable_reason": "not_extracted" if image_exists else "image_missing",
            }
        )
        camera_rows.append(augmented)

    logits_array = (
        np.stack(prior_logits, axis=0).astype(np.float32)
        if prior_logits
        else np.zeros((0, num_beams), dtype=np.float32)
    )
    prior_logits_path = manifest_dir / CAMERA_PRIOR_LOGITS_NAME
    np.save(prior_logits_path, logits_array)
    manifest_path = manifest_dir / CAMERA_MANIFEST_NAME
    metadata_path = manifest_dir / CAMERA_METADATA_NAME
    _write_csv(manifest_path, camera_rows)

    diagnostics = residual_delta_diagnostics(camera_rows, radius=radius)
    metadata = {
        "workflow": "deepsense6g_camera_residual_manifest",
        "result_dir": str(result_dir),
        "manifest_path": str(manifest_path),
        "base_manifest_path": str(base_manifest),
        "gps_dir": str(gps_dir),
        "gps_protocol": protocol,
        "gps_ablation": ablation,
        "support_ratio": ratio,
        "ratio_tag": tag,
        "label_space": selected_label_space,
        "num_beams": num_beams,
        "delta_radius": radius,
        "prior_logits_path": str(prior_logits_path),
        "prior_shape": list(logits_array.shape),
        "prior_source_counts": _counts(camera_rows, "gps_prior_source"),
        "image_available_count": sum(1 for item in camera_rows if _bool(item.get("image_exists"))),
        "image_missing_count": sum(1 for item in camera_rows if not _bool(item.get("image_exists"))),
        "support_count": sum(1 for item in camera_rows if str(item.get("split_role")) == "support"),
        "query_count": sum(1 for item in camera_rows if str(item.get("split_role")).startswith("query")),
        "query_label_used_for_training": False,
        "diagnostics": diagnostics,
        "warnings": warnings,
    }
    metadata_path.write_text(json.dumps(_json_ready(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "result_dir": str(result_dir),
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "prior_logits_path": str(prior_logits_path),
        "row_count": len(camera_rows),
        "support_count": metadata["support_count"],
        "query_count": metadata["query_count"],
        "image_available_count": metadata["image_available_count"],
        "diagnostics": diagnostics,
        "warnings": warnings,
    }


def discover_deepsense6g_image_path(
    raw: Mapping[str, Any] | None,
    *,
    data_root: str | Path,
    scene: str,
    sample_id: str = "",
    cfg: Mapping[str, Any] | None = None,
) -> str:
    """Resolve common DeepSense6G camera/rgb image path conventions."""
    scene_root = Path(data_root) / str(scene)
    image_cfg = _mapping(cfg)
    columns = tuple(image_cfg.get("path_columns") or ("camera8", "camera", "rgb", "image", "image_path"))
    if raw is not None:
        for column in columns:
            value = str(raw.get(str(column)) or "").strip()
            if not value:
                continue
            path = joined_resource(scene_root, value)
            if path.exists():
                return str(path)
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                return str(path)
    stem = _frame_id_from_sample_id(sample_id)
    if not stem:
        return ""
    dirs = tuple(image_cfg.get("common_dirs") or ("camera", "cameras", "image", "images", "rgb", "RGB"))
    candidates: list[Path] = []
    for directory in dirs:
        base = scene_root / str(directory)
        for suffix in IMAGE_EXTENSIONS:
            candidates.append(base / f"{stem}{suffix}")
            candidates.append(base / f"{stem.zfill(6)}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0]) if candidates else ""


def residual_delta_diagnostics(rows: Sequence[Mapping[str, Any]], *, radius: int = 8) -> dict[str, Any]:
    overflow_class = residual_delta_class_count(radius=radius) - 1
    values = [_optional_int(row.get("gps_residual_delta_class")) for row in rows]
    valid = [value for value in values if value is not None]
    overflow = sum(1 for value in valid if value == overflow_class)
    local = len(valid) - overflow
    gps_errors = [_optional_float(row.get("gps_circular_error") or row.get("gps_error")) for row in rows]
    good = sum(1 for value in gps_errors if value is not None and value < 4.0)
    bad = sum(1 for value in gps_errors if value is not None and value >= 4.0)
    return {
        "delta_radius": int(radius),
        "delta_class_count": int(residual_delta_class_count(radius=radius)),
        "overflow_class": int(overflow_class),
        "residual_overflow_count": int(overflow),
        "local_residual_count": int(local),
        "local_residual_coverage": float(local / len(valid)) if valid else 0.0,
        "gps_good_count": int(good),
        "gps_bad_count": int(bad),
    }


class CameraResidualManifestDataset(Dataset):
    """Dataset backed by camera residual manifest rows."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        stage: str = "residual",
        ablation: str = "camera_ae_residual_gated",
        num_beams: int = 64,
        delta_radius: int = 8,
        image_size: int = 64,
        include_query_labels: bool = True,
        training_only: bool = False,
        use_target_query_unlabeled: bool = False,
        require_image: bool | None = None,
        missing_feature_policy: str = "fallback_gps_context",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.stage = str(stage)
        self.ablation = str(ablation)
        self.num_beams = int(num_beams)
        self.delta_radius = int(delta_radius)
        self.image_size = int(image_size)
        self.include_query_labels = bool(include_query_labels)
        self.training_only = bool(training_only)
        self.use_target_query_unlabeled = bool(use_target_query_unlabeled)
        self.missing_feature_policy = str(missing_feature_policy or "fallback_gps_context")
        self.rows = _read_csv(self.manifest_path)
        if self.stage == "ae_training":
            self.rows = [
                row
                for row in self.rows
                if _bool(row.get("image_exists"))
                and (self.use_target_query_unlabeled or not str(row.get("split_role") or row.get("support_query_role")).startswith("query"))
            ]
            if not self.rows:
                raise ValueError(f"No usable camera images for AE training in manifest: {self.manifest_path}")
        if self.training_only:
            self.rows = [
                row
                for row in self.rows
                if not str(row.get("split_role") or row.get("support_query_role")).startswith("query")
            ]
        self.require_image = bool(require_image) if require_image is not None else self.stage == "ae_training"
        self._array_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index)]
        role = str(row.get("split_role") or row.get("support_query_role") or "")
        target_label = _int(row.get("target_label"), -100)
        if role.startswith("query") and not self.include_query_labels:
            target_label = -100
        item: dict[str, Any] = {
            "sample_id": str(row.get("sample_id") or ""),
            "scene": str(row.get("scene") or ""),
            "split_role": role,
            "support_query_role": role,
            "support_or_query": str(row.get("support_or_query") or ("support" if role == "support" else "query")),
        }
        if self.stage == "ae_training":
            item["image"] = load_image_tensor(row.get("image_path"), image_size=self.image_size)
            return item

        pred = _int(row.get("gps_pred_top1"), -1)
        prior_logits = self._load_prior_logits(row, pred)
        prior_probs = torch.softmax(prior_logits, dim=-1)
        gps_error = _float(row.get("gps_circular_error") or row.get("gps_error"), math.inf)
        residual_class = _int(
            row.get("gps_residual_delta_class"),
            residual_to_delta_class(
                signed_circular_residual(target_label, pred, num_beams=self.num_beams)
                if target_label >= 0 and pred >= 0
                else 10_000,
                radius=self.delta_radius,
            ),
        )
        item.update(
            {
                "target_label": torch.tensor(target_label, dtype=torch.long),
                "gps_pred_top1": torch.tensor(pred, dtype=torch.long),
                "gps_error": torch.tensor(gps_error, dtype=torch.float32),
                "gps_prior_logits": prior_logits,
                "gps_prior_probs": prior_probs,
                "gps_context": torch.tensor(_context_features(row, self.num_beams), dtype=torch.float32),
                "gps_context_features": torch.tensor(_context_features(row, self.num_beams), dtype=torch.float32),
                "residual_delta_class": torch.tensor(residual_class, dtype=torch.long),
                "gate_target": torch.tensor(float(gps_error >= 4.0), dtype=torch.float32),
                "image_exists": torch.tensor(_bool(row.get("image_exists")), dtype=torch.bool),
                "ae_feature_available": torch.tensor(_bool(row.get("ae_feature_available")), dtype=torch.bool),
            }
        )
        if self._needs_camera_feature():
            item["camera_ae_feature"] = self._load_ae_feature(row)
        return item

    def _needs_camera_feature(self) -> bool:
        return "camera_ae" in self.ablation and self.ablation != "gps_context_only_residual"

    def _load_prior_logits(self, row: Mapping[str, Any], pred: int) -> torch.Tensor:
        path = str(row.get("gps_prior_logits_path") or "")
        row_index = _int(row.get("gps_logits_row_index"), -1)
        if path and row_index >= 0 and Path(path).exists():
            array = self._cached_array(path)
            if row_index < int(array.shape[0]):
                return torch.tensor(array[row_index], dtype=torch.float32)
        if pred >= 0:
            logits, _ = circular_gaussian_prior_from_top1(pred, num_beams=self.num_beams)
            return torch.tensor(logits, dtype=torch.float32)
        probs = np.full((self.num_beams,), 1.0 / float(self.num_beams), dtype=np.float32)
        return torch.tensor(np.log(probs), dtype=torch.float32)

    def _load_ae_feature(self, row: Mapping[str, Any]) -> torch.Tensor:
        path = str(row.get("ae_feature_path") or "")
        row_index = _int(row.get("ae_feature_row_index"), -1)
        if path and row_index >= 0 and Path(path).exists():
            array = self._cached_array(path)
            if row_index < int(array.shape[0]):
                return torch.tensor(array[row_index], dtype=torch.float32)
        if self.missing_feature_policy in {"fallback_gps_context", "gps_context_only_residual"}:
            return torch.tensor(_context_features(row, self.num_beams), dtype=torch.float32)
        if self.missing_feature_policy == "zeros":
            return torch.zeros(1, dtype=torch.float32)
        raise FileNotFoundError(f"Camera AE feature is required but unavailable for {row.get('sample_id')}.")

    def _cached_array(self, path: str) -> np.ndarray:
        if path not in self._array_cache:
            self._array_cache[path] = np.load(path)
        return self._array_cache[path]


def collate_camera_residual_batch(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty camera residual batch.")
    batch: dict[str, Any] = {}
    tensor_keys = {key for item in items for key, value in item.items() if torch.is_tensor(value)}
    for key in sorted(tensor_keys):
        values = [item[key] for item in items]  # type: ignore[index]
        shapes = {tuple(value.shape) for value in values}
        if len(shapes) == 1:
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    for key in ("sample_id", "scene", "split_role", "support_query_role", "support_or_query"):
        batch[key] = [str(item.get(key, "")) for item in items]
    return batch


def build_camera_residual_dataloader(
    manifest_path: str | Path,
    *,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    **dataset_kwargs: Any,
) -> DataLoader:
    dataset = CameraResidualManifestDataset(manifest_path, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=collate_camera_residual_batch,
    )


def load_image_tensor(path_value: Any, *, image_size: int = 64) -> torch.Tensor:
    path = Path(str(path_value or ""))
    if not path.exists():
        raise FileNotFoundError(f"Camera image is missing: {path_value}")
    from PIL import Image

    image = Image.open(path).convert("RGB").resize((int(image_size), int(image_size)))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - 0.5) / 0.5


def write_manifest_with_ae_features(
    manifest_path: str | Path,
    *,
    features_path: str | Path,
    features_index_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = Path(manifest_path)
    out = Path(output_path) if output_path is not None else manifest.with_name(CAMERA_MANIFEST_WITH_AE_NAME)
    features = Path(features_path)
    index_rows = _read_csv(features_index_path)
    by_key = {
        (str(row.get("scene") or ""), str(row.get("sample_id") or "")): _int(row.get("row_index"), -1)
        for row in index_rows
    }
    rows = _read_csv(manifest)
    available = 0
    for row in rows:
        key = (str(row.get("scene") or ""), str(row.get("sample_id") or ""))
        row_index = by_key.get(key, -1)
        row["ae_feature_row_index"] = row_index
        row["ae_feature_path"] = str(features) if row_index >= 0 else ""
        row["ae_feature_available"] = bool(row_index >= 0)
        if row_index >= 0:
            row["ae_feature_unavailable_reason"] = ""
            available += 1
        elif not _bool(row.get("image_exists")):
            row["ae_feature_unavailable_reason"] = "image_missing"
        else:
            row["ae_feature_unavailable_reason"] = "feature_not_extracted"
    _write_csv(out, rows)
    return {
        "manifest_with_ae_path": str(out),
        "features_path": str(features),
        "features_index_path": str(features_index_path),
        "row_count": len(rows),
        "feature_available_count": available,
    }


def manifest_fingerprint(manifest_path: str | Path, *, extra: Mapping[str, Any] | None = None) -> str:
    path = Path(manifest_path)
    digest = hashlib.sha256()
    digest.update(str(path.resolve()).encode("utf-8"))
    if path.exists():
        stat = path.stat()
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))
    if extra:
        digest.update(json.dumps(_json_ready(extra), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


class _DeepSenseImageIndex:
    def __init__(self, data_root: Path, data_cfg: Mapping[str, Any]) -> None:
        self.data_root = data_root
        self.data_cfg = data_cfg
        self.cache: dict[tuple[str, str], list[dict[str, str]]] = {}

    def lookup(self, sample_id: str) -> Mapping[str, Any] | None:
        parsed = _parse_sample_id(sample_id)
        if parsed is None:
            return None
        scene, split, row_idx = parsed
        key = (scene, split)
        if key not in self.cache:
            csv_name = str(
                self.data_cfg.get(
                    f"{split}_csv_name",
                    "train_seqs_RA_GPS_LIDAR.csv" if split == "train" else "test_seqs_RA_GPS_LIDAR.csv",
                )
            )
            self.cache[key] = _read_csv(self.data_root / scene / csv_name)
        rows = self.cache.get(key, [])
        if row_idx < 0 or row_idx >= len(rows):
            return None
        return rows[row_idx]


def _safe_load_prior_records(
    gps_dir: Path,
    *,
    protocol: str,
    ablation: str,
    num_beams: int,
    fallback_sigma: float,
):
    try:
        return load_gps_prior_artifacts(
            gps_dir,
            protocol=protocol,
            ablation=ablation,
            num_beams=num_beams,
            fallback_sigma=fallback_sigma,
        )
    except FileNotFoundError:
        return {}


def _prior_logits_for_row(
    prior: Any,
    pred: int,
    *,
    num_beams: int,
    fallback_sigma: float,
) -> tuple[np.ndarray, str, str]:
    if prior is not None:
        source = str(prior.source)
        normalized_source = SAVED_LOGITS_PRIOR_SOURCE if source in PREDICTION_LOGIT_NAMES else source
        return np.asarray(prior.logits, dtype=np.float32), normalized_source, source
    if pred >= 0:
        logits, _ = circular_gaussian_prior_from_top1(pred, num_beams=num_beams, sigma=fallback_sigma)
        return np.asarray(logits, dtype=np.float32), FALLBACK_PRIOR_SOURCE, ""
    probs = np.full((num_beams,), 1.0 / float(num_beams), dtype=np.float32)
    return np.log(probs), UNAVAILABLE_PRIOR_SOURCE, ""


def _resolve_best_gps_ablation(base: Path, requested: Any) -> str:
    value = str(requested or "best")
    if value != "best":
        return value
    rows = [row for row in _read_csv(base / "summary_overall.csv") if str(row.get("protocol")) == "target_adapt_beambench"]
    if not rows:
        return "branch_mixture_circular"
    return str(max(rows, key=lambda row: _float(row.get("DBA"), -1.0)).get("ablation") or "branch_mixture_circular")


def _context_features(row: Mapping[str, Any], num_beams: int) -> list[float]:
    raw = str(row.get("gps_context_features") or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return [float(value) for value in parsed.values()]
        if isinstance(parsed, list):
            return [float(value) for value in parsed]
    pred = _int(row.get("gps_pred_top1"), -1)
    theta = math.radians(_float(row.get("theta_degrees"), 0.0))
    easting = _float(row.get("E"), 0.0)
    northing = _float(row.get("N"), 0.0)
    pred_angle = (float(pred if pred >= 0 else 0) / float(num_beams)) * 2.0 * math.pi
    rel_range = math.sqrt(easting * easting + northing * northing)
    return [
        easting,
        northing,
        math.sin(theta),
        math.cos(theta),
        math.log1p(max(rel_range, 0.0)),
        math.sin(pred_angle),
        math.cos(pred_angle),
        _float(row.get("gps_prior_peak_prob"), 0.0),
        _float(row.get("gps_prior_entropy"), 0.0),
    ]


def _parse_sample_id(sample_id: str) -> tuple[str, str, int] | None:
    parts = str(sample_id).split(":")
    if len(parts) < 3:
        return None
    try:
        return parts[0], parts[1], int(parts[2])
    except ValueError:
        return None


def _frame_id_from_sample_id(sample_id: str) -> str:
    parts = str(sample_id).split(":")
    if len(parts) >= 4:
        return Path(parts[3]).stem
    return parts[-1] if parts else ""


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    return probs / probs.sum(axis=-1, keepdims=True).clip(min=1e-12)


def _counts(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        result[value] = result.get(value, 0) + 1
    return result


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_int(value: Any) -> int | None:
    try:
        if value in {None, ""}:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _int(value: Any, default: int) -> int:
    parsed = _optional_int(value)
    return int(default) if parsed is None else parsed


def _optional_float(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any, default: float) -> float:
    parsed = _optional_float(value)
    return float(default) if parsed is None else parsed


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
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


__all__ = [
    "CAMERA_MANIFEST_NAME",
    "CAMERA_MANIFEST_WITH_AE_NAME",
    "CameraResidualManifestDataset",
    "build_camera_residual_dataloader",
    "build_camera_residual_manifest",
    "collate_camera_residual_batch",
    "delta_class_to_residual",
    "discover_deepsense6g_image_path",
    "load_image_tensor",
    "manifest_fingerprint",
    "residual_delta_diagnostics",
    "residual_to_delta_class",
    "write_manifest_with_ae_features",
]
