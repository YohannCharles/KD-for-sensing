from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from kd_sensing.engine.batch import normalize_batch

IMAGE_ONLY_DISABLED_FIELDS = (
    "gps",
    "lidar",
    "radar",
    "radar_ra",
    "radar_da",
    "mmwave",
    "csi",
    "channel",
    "channel_path",
    "path",
    "path_params",
    "path_descriptor",
    "path_semantic_label",
    "beam_power",
    "beamspace_power_label",
)
IMAGE_ONLY_LEGAL_BATCH_FIELDS = (
    "image",
    "target_beam",
    "metadata",
    "sample_id",
    "domain_metadata",
)
IMAGE_ONLY_SOURCE_VARIANTS = {"image_source_only"}
IMAGE_ONLY_ADAPTATION_VARIANTS = {
    "image_target_linear_probe",
    "image_v8_target_prior_head",
    "image_v9_sector_proto",
}
IMAGE_ONLY_VARIANTS = IMAGE_ONLY_SOURCE_VARIANTS | IMAGE_ONLY_ADAPTATION_VARIANTS


def image_only_protocol_enabled(cfg: Mapping[str, Any] | None) -> bool:
    if not isinstance(cfg, Mapping):
        return False
    hist_cfg = cfg.get("hist_beam") if isinstance(cfg.get("hist_beam"), Mapping) else {}
    protocol = hist_cfg.get("protocol") if isinstance(hist_cfg.get("protocol"), Mapping) else {}
    dataset_cfg = cfg.get("data", {}).get("dataset", {}) if isinstance(cfg.get("data"), Mapping) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), Mapping) else {}
    student_cfg = model_cfg.get("student", {}) if isinstance(model_cfg.get("student"), Mapping) else {}
    modalities = (
        hist_cfg.get("modalities")
        or dataset_cfg.get("enabled_modalities")
        or model_cfg.get("modalities")
        or student_cfg.get("modalities")
        or ()
    )
    normalized = tuple(str(item).strip().lower() for item in modalities)
    return bool(protocol.get("image_only", False)) and normalized == ("image",)


def canonical_image_only_variant(variant: Any) -> str:
    text = str(variant).strip().lower()
    if text in {"image_source_only", "i0_source_only"}:
        return "v0_flat"
    if text in {"image_target_linear_probe", "i1_linear_probe"}:
        return "v8_target_prior_head"
    if text in {"image_v8_target_prior_head", "i2_v8_target_prior"}:
        return "v8_target_prior_head"
    if text in {"image_v9_sector_proto", "i3_v9_sector_proto"}:
        return "v9_input_conditioned_target_adaptation"
    return text


def image_only_stage_consumed_fields(stage: str) -> dict[str, Any]:
    normalized = str(stage).strip().lower()
    if normalized == "source_train":
        return {
            "consumed_input_fields": ["image"],
            "consumed_label_fields": ["source.target_beam"],
            "target_test_label_usage": "not_used",
        }
    if normalized == "target_adaptation":
        return {
            "consumed_input_fields": ["target_support.image"],
            "consumed_label_fields": ["target_support.target_beam"],
            "target_test_label_usage": "not_used",
        }
    if normalized in {"source_only_target_test_eval", "adapted_target_test_eval", "target_test"}:
        return {
            "consumed_input_fields": ["target_test.image"],
            "consumed_label_fields": ["target_test.target_beam:evaluation_only"],
            "target_test_label_usage": "evaluation_only",
        }
    return {
        "consumed_input_fields": ["image"],
        "consumed_label_fields": ["target_beam"],
        "target_test_label_usage": "unknown",
    }


def filter_image_only_batch(
    batch: Any,
    cfg: Mapping[str, Any] | None,
    *,
    stage: str,
) -> dict[str, Any]:
    normalized = normalize_batch(batch)
    if not image_only_protocol_enabled(cfg):
        return normalized
    filtered = {key: value for key, value in normalized.items() if key in IMAGE_ONLY_LEGAL_BATCH_FIELDS}
    if "target_beam" not in filtered and "beam" in normalized:
        filtered["target_beam"] = normalized["beam"]
    metadata = _metadata_with_image_only_diagnostics(
        normalized.get("metadata"),
        available_fields=available_fields_from_batch(normalized),
        stage=stage,
    )
    if metadata is not None:
        filtered["metadata"] = metadata
    return filtered


def available_fields_from_batch(batch: Mapping[str, Any]) -> list[str]:
    fields = {str(key) for key in batch.keys()}
    metadata = batch.get("metadata")
    for row in _metadata_rows(metadata):
        fields.update(str(key) for key, value in row.items() if value not in (None, ""))
    return sorted(fields)


def image_only_run_metadata(
    cfg: Mapping[str, Any],
    run: Mapping[str, Any] | None = None,
    *,
    stage: str | None = None,
    available_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), Mapping) else {}
    disabled = list(hist_cfg.get("disabled_modalities") or ["gps", "lidar", "radar", "mmwave", "csi"])
    excluded = list(hist_cfg.get("excluded_sensitive_fields") or IMAGE_ONLY_DISABLED_FIELDS)
    payload: dict[str, Any] = {
        "image_only_protocol": True,
        "enabled_modalities": ["image"],
        "disabled_modalities": [str(item) for item in disabled],
        "excluded_sensitive_fields": [str(item) for item in excluded],
        "used_target_oracle_fields": [],
        "target_oracle_usage_stage": {},
        "available_fields": sorted(str(item) for item in (available_fields or [])),
        "consumed_fields": {},
    }
    if stage:
        payload["consumed_fields"][stage] = image_only_stage_consumed_fields(stage)
    if run is not None:
        payload.update(
            {
                "source_scenes": list(run.get("source_scenes", [])),
                "target_scene": run.get("target_scene"),
                "seed": run.get("seed"),
                "label_budget": run.get("budget"),
                "probe_mode": run.get("variant"),
            }
        )
    return payload


def image_feature_cache_enabled(cfg: Mapping[str, Any]) -> bool:
    if not image_only_protocol_enabled(cfg):
        return False
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), Mapping) else {}
    cache_cfg = hist_cfg.get("feature_cache", {}) if isinstance(hist_cfg.get("feature_cache"), Mapping) else {}
    return bool(cache_cfg.get("enabled", False))


def feature_cache_dir(cfg: Mapping[str, Any], run_dir: str | Path) -> Path:
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), Mapping) else {}
    cache_cfg = hist_cfg.get("feature_cache", {}) if isinstance(hist_cfg.get("feature_cache"), Mapping) else {}
    configured = cache_cfg.get("dir")
    if configured:
        path = Path(str(configured))
        if path.is_absolute():
            return path
    else:
        path = Path("feature_cache")
    return Path(run_dir) / path


def expected_feature_cache_metadata(
    cfg: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    checkpoint: str | Path | None,
    feature_dim: int | None,
    dtype: str = "float32",
) -> dict[str, Any]:
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), Mapping) else {}
    student_cfg = model_cfg.get("student", {}) if isinstance(model_cfg.get("student"), Mapping) else {}
    image_encoder = student_cfg.get("image_encoder", model_cfg.get("image_encoder"))
    return {
        "version": "image_only_feature_cache_v1",
        "created_at": _utc_now(),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "feature_dim": int(feature_dim) if feature_dim is not None else None,
        "modalities": ["image"],
        "image_encoder": image_encoder,
        "source_scenes": list(run.get("source_scenes", [])),
        "target_scene": run.get("target_scene"),
        "label_budget": int(run.get("budget", 0) or 0),
        "dtype": str(dtype),
    }


def validate_feature_cache_metadata(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    mismatches = []
    for key in (
        "checkpoint",
        "feature_dim",
        "modalities",
        "image_encoder",
        "source_scenes",
        "target_scene",
        "label_budget",
        "dtype",
    ):
        if actual.get(key) != expected.get(key):
            mismatches.append({"field": key, "expected": expected.get(key), "actual": actual.get(key)})
    if mismatches:
        raise ValueError(f"Image feature cache metadata mismatch: {json.dumps(mismatches, sort_keys=True)}")


def write_image_feature_cache(
    path: str | Path,
    *,
    features: torch.Tensor,
    labels: torch.Tensor,
    metadata_rows: list[dict[str, Any]],
    split: str,
    cache_metadata: Mapping[str, Any],
    overwrite: bool = True,
) -> dict[str, Any]:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Feature cache already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    features_cpu = features.detach().cpu().to(torch.float32)
    labels_cpu = labels.detach().cpu().to(torch.long)
    scenes = [str(row.get("scene_slug", row.get("scene_id", row.get("scenario", "")))) for row in metadata_rows]
    sample_ids = [str(row.get("sample_id", index)) for index, row in enumerate(metadata_rows)]
    splits = [str(row.get("split", split)) for row in metadata_rows]
    torch.save(
        {
            "features": features_cpu,
            "labels": labels_cpu,
            "scene": scenes,
            "sample_id": sample_ids,
            "split": splits,
        },
        target,
    )
    meta_path = target.parent / "cache_meta.json"
    existing: dict[str, Any] = {}
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    splits_meta = dict(existing.get("splits", {})) if isinstance(existing.get("splits"), Mapping) else {}
    split_meta = {
        **dict(cache_metadata),
        "path": str(target),
        "split": split,
        "sample_count": int(features_cpu.shape[0]),
        "feature_shape": list(features_cpu.shape),
        "label_shape": list(labels_cpu.shape),
    }
    splits_meta[split] = split_meta
    meta_payload = {**dict(cache_metadata), "splits": splits_meta}
    meta_path.write_text(json.dumps(meta_payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"path": str(target), "cache_meta_path": str(meta_path), "metadata": split_meta}


def load_image_feature_cache(
    path: str | Path,
    *,
    expected_metadata: Mapping[str, Any],
    split: str,
    scope: str,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    split_name = str(split)
    scope_name = str(scope)
    target = Path(path)
    meta_path = target.parent / "cache_meta.json"
    if not target.exists():
        raise FileNotFoundError(f"Image feature cache not found: {target}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Image feature cache metadata not found: {meta_path}")
    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    split_meta = (meta_payload.get("splits") or {}).get(split_name)
    if not isinstance(split_meta, Mapping):
        raise ValueError(f"Image feature cache metadata does not contain split '{split_name}'.")
    validate_feature_cache_metadata(split_meta, expected_metadata)
    payload = torch.load(target, map_location=map_location)
    if scope_name == "adaptation" and split_name == "target_test":
        payload = dict(payload)
        payload.pop("labels", None)
        payload["labels_unavailable_reason"] = "target_test_labels_blocked_for_adaptation_scope"
    return {
        "features": payload.get("features"),
        "labels": payload.get("labels"),
        "scene": payload.get("scene"),
        "sample_id": payload.get("sample_id"),
        "split": payload.get("split"),
        "metadata": dict(split_meta),
        "labels_unavailable_reason": payload.get("labels_unavailable_reason"),
    }


def extract_image_features_for_cache(
    model: Any,
    dataloader: Any,
    cfg: Mapping[str, Any],
    device: torch.device,
    *,
    split: str,
    stage: str,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    from kd_sensing.engine.runtime import run_model_step, transfer_non_blocking

    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), Mapping) else {}
    student_cfg = model_cfg.get("student", model_cfg) if isinstance(model_cfg, Mapping) else {}
    task = cfg.get("experiment", {}).get("task", "fusion") if isinstance(cfg.get("experiment"), Mapping) else "fusion"
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    metadata_rows: list[dict[str, Any]] = []
    was_training = bool(getattr(model, "training", False))
    model.eval()
    with torch.no_grad():
        for raw_batch in dataloader:
            batch = filter_image_only_batch(raw_batch, cfg, stage=stage)
            step = run_model_step(
                model,
                task,
                batch,
                model_cfg=student_cfg,
                seq_length=int(model_cfg.get("seq_length_student", cfg.get("data", {}).get("dataset", {}).get("seq_len", 8))),
                num_pred=int(model_cfg.get("num_pred", cfg.get("data", {}).get("dataset", {}).get("num_pred", 1))),
                device=device,
                downsample_ratio=int(model_cfg.get("downsample_ratio", 1)),
                non_blocking=transfer_non_blocking(dict(cfg)),
            )
            feature_tensor = step.model_output.input_features
            if feature_tensor is None:
                feature_tensor = step.model_output.output_features
            if feature_tensor is None:
                feature_tensor = step.model_output.diagnostics.get("features")
            if not torch.is_tensor(feature_tensor):
                raise RuntimeError("Image-only feature cache requires model output features.")
            if feature_tensor.ndim == 3:
                feature_tensor = feature_tensor[:, 0, :]
            features.append(feature_tensor.detach().cpu())
            if step.labels is None:
                raise RuntimeError("Image-only feature cache requires beam labels.")
            labels.append(step.labels.detach().cpu())
            metadata_rows.extend(_metadata_rows(batch.get("metadata"), fallback_count=int(step.labels.shape[0]), split=split))
    if was_training:
        model.train()
    if not features:
        raise RuntimeError(f"No samples available to write image feature cache for split={split}.")
    return torch.cat(features, dim=0), torch.cat(labels, dim=0), metadata_rows


def _metadata_with_image_only_diagnostics(
    metadata: Any,
    *,
    available_fields: list[str],
    stage: str,
) -> Any:
    rows = _metadata_rows(metadata)
    if not rows:
        return metadata
    consumed = image_only_stage_consumed_fields(stage)
    updated = []
    for row in rows:
        item = dict(row)
        item["image_only_available_fields"] = list(available_fields)
        item["image_only_consumed_input_fields"] = list(consumed["consumed_input_fields"])
        item["image_only_consumed_label_fields"] = list(consumed["consumed_label_fields"])
        updated.append(item)
    if isinstance(metadata, list):
        return updated
    return _rows_to_collated_metadata(updated)


def _metadata_rows(metadata: Any, *, fallback_count: int = 0, split: str | None = None) -> list[dict[str, Any]]:
    if metadata is None:
        rows = [{} for _ in range(int(fallback_count))]
    elif isinstance(metadata, list):
        rows = [dict(item) for item in metadata if isinstance(item, Mapping)]
    elif isinstance(metadata, Mapping):
        length = int(fallback_count)
        for value in metadata.values():
            if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0:
                length = max(length, int(value.shape[0]))
            elif isinstance(value, (list, tuple)):
                length = max(length, len(value))
            else:
                length = max(length, 1)
        rows = []
        for index in range(length):
            row = {}
            for key, value in metadata.items():
                row[str(key)] = _metadata_value_at(value, index)
            rows.append(row)
    else:
        rows = [{} for _ in range(int(fallback_count))]
    if split is not None:
        for row in rows:
            row.setdefault("split", split)
    return rows


def _rows_to_collated_metadata(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    keys = sorted({key for row in rows for key in row})
    return {key: [row.get(key) for row in rows] for key in keys}


def _metadata_value_at(value: Any, index: int) -> Any:
    if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0:
        item = value[index]
        return item.item() if hasattr(item, "item") else item
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    return value


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "IMAGE_ONLY_ADAPTATION_VARIANTS",
    "IMAGE_ONLY_DISABLED_FIELDS",
    "IMAGE_ONLY_SOURCE_VARIANTS",
    "IMAGE_ONLY_VARIANTS",
    "available_fields_from_batch",
    "canonical_image_only_variant",
    "expected_feature_cache_metadata",
    "extract_image_features_for_cache",
    "feature_cache_dir",
    "filter_image_only_batch",
    "image_feature_cache_enabled",
    "image_only_protocol_enabled",
    "image_only_run_metadata",
    "image_only_stage_consumed_fields",
    "load_image_feature_cache",
    "validate_feature_cache_metadata",
    "write_image_feature_cache",
]
