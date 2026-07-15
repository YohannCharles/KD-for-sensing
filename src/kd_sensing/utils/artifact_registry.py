from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import torch

from kd_sensing.data.datasets.deepsense6g_gps_contract import (
    RSU_LOCAL_GPS_FEATURE_MODE,
    normalize_gps_feature_mode,
)
from kd_sensing.engine.modality_resolution import config_uses_gps
from kd_sensing.engine.run_lineage import model_capacity, run_lineage_metadata
from kd_sensing.utils.paths import resolve_path
from kd_sensing.utils.checkpoint import (
    checkpoint_file_digest,
    load_torch_payload,
    publish_checkpoint_copy,
)
from kd_sensing.utils.runtime_output_layout import runtime_output_scope_from_config, runtime_scope_metadata_from_config


DEFAULT_REGISTRY = {
    "enabled": True,
    "dir": "outputs",
    "prefer": True,
    "metric": "val_top1",
    "filename": "{slug}_{role}_acc_{acc}.pth",
}
LEGACY_DEFAULT_REGISTRY_DIR = "outputs/best_checkpoints"
GPS_CHECKPOINT_PROVENANCE_KEYS = ("gps_feature_mode", "gps_angle_frame", "gps_yaw_source")
RSU_YAW_SOURCE = "bs_yaml:sensors.rsu_pose.rotation.yaw"


@dataclass
class CheckpointResolution:
    path: Path | None
    source: str
    metadata: dict[str, Any] | None = None
    registry_dir: Path | None = None
    candidates: list[str] | None = None
    requested: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path) if self.path is not None else None,
            "source": self.source,
            "metadata": self.metadata,
            "registry_dir": str(self.registry_dir) if self.registry_dir is not None else None,
            "candidates": list(self.candidates or []),
            "requested": self.requested,
        }


def registry_config(cfg: dict[str, Any]) -> dict[str, Any]:
    configured = cfg.get("checkpoint", {}).get("registry", {})
    return {**DEFAULT_REGISTRY, **(configured or {})}


def registry_enabled(cfg: dict[str, Any]) -> bool:
    return bool(registry_config(cfg).get("enabled", True))


def registry_preferred(cfg: dict[str, Any]) -> bool:
    settings = registry_config(cfg)
    return bool(settings.get("enabled", True)) and bool(settings.get("prefer", True))


def registry_dir(cfg: dict[str, Any]) -> Path:
    configured_dir = registry_config(cfg).get("dir", DEFAULT_REGISTRY["dir"])
    if configured_dir in {DEFAULT_REGISTRY["dir"], LEGACY_DEFAULT_REGISTRY_DIR}:
        return _default_scene_registry_dir(cfg)
    return resolve_path(configured_dir)


def config_slug(cfg: dict[str, Any]) -> str:
    experiment_name = cfg.get("experiment", {}).get("name")
    run_name = cfg.get("output", {}).get("run_name")
    if experiment_name:
        return sanitize_slug(str(experiment_name))
    if run_name:
        return sanitize_slug(str(run_name))
    task = cfg.get("experiment", {}).get("task", "run")
    return sanitize_slug(f"{task}_{artifact_role(cfg)}")


def artifact_role(cfg: dict[str, Any]) -> str:
    return sanitize_slug(model_capacity(cfg))


def sanitize_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned or "run"


def format_accuracy(value: float) -> str:
    return f"{float(value):.4f}"


def checkpoint_sidecar_path(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    return path.with_suffix(path.suffix + ".json")


def load_checkpoint_metadata(checkpoint_path: str | Path | None) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None
    sidecar = checkpoint_sidecar_path(checkpoint_path)
    if not sidecar.exists():
        return _metadata_from_checkpoint_payload(checkpoint_path)
    for attempt in range(2):
        try:
            metadata = load_sidecar(sidecar)
        except (json.JSONDecodeError, OSError):
            if attempt == 0:
                time.sleep(0.05)
                continue
            return _metadata_from_checkpoint_payload(checkpoint_path)
        if isinstance(metadata, dict):
            return metadata
        return _metadata_from_checkpoint_payload(checkpoint_path)
    return _metadata_from_checkpoint_payload(checkpoint_path)


def load_sidecar(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_sidecar(checkpoint_path: str | Path, metadata: dict[str, Any]) -> Path:
    sidecar = checkpoint_sidecar_path(checkpoint_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{sidecar.name}.", suffix=".tmp", dir=sidecar.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(_json_ready(metadata), f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(sidecar)
    finally:
        tmp_path.unlink(missing_ok=True)
    return sidecar


def gps_checkpoint_provenance(cfg: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg.get("data"), dict):
        return {}
    if not config_uses_gps(cfg):
        return {}
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    mode = normalize_gps_feature_mode(dataset_cfg.get("gps_feature_mode"))
    expected = {
        "gps_feature_mode": mode,
        "gps_angle_frame": "rsu_local" if mode == RSU_LOCAL_GPS_FEATURE_MODE else "world",
        "gps_yaw_source": RSU_YAW_SOURCE if mode == RSU_LOCAL_GPS_FEATURE_MODE else None,
    }
    protocol = cfg.get("mmw_all_weather_protocol")
    if isinstance(protocol, dict):
        for key, value in expected.items():
            if key in protocol and protocol[key] != value:
                raise ValueError(
                    f"mmw_all_weather_protocol.{key}={protocol[key]!r} does not match "
                    f"data.dataset.gps_feature_mode={mode!r} ({value!r})."
                )
    return expected


def validate_evaluation_gps_checkpoint_provenance(
    cfg: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> None:
    expected = gps_checkpoint_provenance(cfg)
    if not expected:
        return
    recorded = metadata if isinstance(metadata, dict) else {}
    recorded_mode = recorded.get("gps_feature_mode")
    if recorded_mode is None:
        if expected["gps_feature_mode"] == RSU_LOCAL_GPS_FEATURE_MODE:
            raise ValueError(
                "rsu_local_relative_polar evaluation requires checkpoint GPS coordinate-frame provenance; "
                "the checkpoint is legacy or missing gps_feature_mode."
            )
        return
    required_keys = GPS_CHECKPOINT_PROVENANCE_KEYS if recorded_mode == RSU_LOCAL_GPS_FEATURE_MODE else ("gps_feature_mode",)
    missing = [key for key in required_keys if key not in recorded]
    if missing:
        raise ValueError(f"Checkpoint GPS coordinate-frame provenance is incomplete; missing {missing}.")
    mismatches = {
        key: {"config": expected[key], "checkpoint": recorded.get(key)}
        for key in GPS_CHECKPOINT_PROVENANCE_KEYS
        if key in recorded and recorded.get(key) != expected[key]
    }
    if mismatches:
        raise ValueError(f"Checkpoint GPS coordinate-frame provenance does not match evaluation config: {mismatches}.")


def archive_best_checkpoint(
    cfg: dict[str, Any],
    *,
    source_checkpoint: str | Path,
    val_top1: float,
    epoch: int,
    run_dir: str | Path,
    split_metadata: dict[str, Any] | None = None,
    normalization_artifacts: dict[str, Any] | None = None,
    objective_metric: dict[str, Any] | None = None,
    task_metrics: dict[str, Any] | None = None,
    selection_metric: str | None = None,
    selection_mode: str | None = None,
    checkpoint_source: str | None = None,
) -> dict[str, Any] | None:
    if not registry_enabled(cfg):
        return None

    source = Path(source_checkpoint)
    if not source.exists():
        raise FileNotFoundError(f"Cannot archive missing checkpoint: {source}")

    target_dir = registry_dir(cfg)
    target_dir.mkdir(parents=True, exist_ok=True)
    slug = config_slug(cfg)
    role = artifact_role(cfg)
    target = target_dir / checkpoint_filename(cfg, slug=slug, role=role, val_top1=val_top1)
    existing = find_registry_checkpoint(cfg, target_slug=slug, role=role)
    if existing.path is not None and existing.metadata is not None:
        existing_metric = float(existing.metadata.get("metric_value", float("-inf")))
        if existing_metric > float(val_top1):
            return {
                **existing.metadata,
                "updated": False,
                "skipped_source_checkpoint": str(source),
                "skipped_metric_value": float(val_top1),
            }
    lineage = run_lineage_metadata(cfg)
    source_digest, source_size = checkpoint_file_digest(source)
    metadata = {
        "path": str(target),
        "source": "registry",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_digest,
        "source_checkpoint_size_bytes": int(source_size),
        "checkpoint_source": checkpoint_source
        or ("top1-checkpoint" if source.name == "best_top1.pth" else "objective-checkpoint"),
        "run_dir": str(run_dir),
        "config_slug": slug,
        "artifact_role": role,
        "training_mode": lineage["training_mode"],
        "model_capacity": lineage["model_capacity"],
        "metric_name": registry_config(cfg).get("metric", "val_top1"),
        "metric_value": float(val_top1),
        "selection_metric": selection_metric
        or ("val_acc_top1" if source.name == "best_top1.pth" else (objective_metric or {}).get("name")),
        "selection_mode": selection_mode
        or ("top1-selection" if source.name == "best_top1.pth" else "objective"),
        "selected_epoch": int(epoch),
        "objective": cfg.get("experiment", {}).get("objective", "beam"),
        "variant": cfg.get("experiment", {}).get("variant"),
        "seq_len": cfg.get("data", {}).get("dataset", {}).get("seq_len"),
        "num_pred": cfg.get("data", {}).get("dataset", {}).get("num_pred"),
        "uses_history_window": bool(int(cfg.get("data", {}).get("dataset", {}).get("seq_len", 0) or 0) > 1),
        "uses_temporal_core": bool(cfg.get("experiment", {}).get("uses_temporal_core", True)),
        "objective_metric": objective_metric or {},
        "task_metrics": task_metrics or {},
        "epoch": int(epoch),
        "task": cfg.get("experiment", {}).get("task"),
        "enabled_modalities": _enabled_modalities_from_cfg(cfg),
        "split_metadata": split_metadata or {},
        "normalization_artifacts": normalization_artifacts or {},
        "updated": True,
        "lineage": lineage,
        **gps_checkpoint_provenance(cfg),
    }
    metadata.update(runtime_scope_metadata_from_config(cfg))
    _, published = publish_checkpoint_copy(source, target, metadata=metadata)
    _remove_old_archives(target_dir, slug=slug, role=role, keep=target, scene_slug=_scope_slug_from_config(cfg))
    return published


def checkpoint_filename(cfg: dict[str, Any], *, slug: str, role: str, val_top1: float) -> str:
    filename_template = str(registry_config(cfg).get("filename", DEFAULT_REGISTRY["filename"]))
    if filename_template == DEFAULT_REGISTRY["filename"]:
        prefix = slug if slug.endswith(role) else f"{slug}_{role}"
        return f"{prefix}_acc_{format_accuracy(val_top1)}.pth"
    return filename_template.format(
        slug=slug,
        role=role,
        acc=format_accuracy(val_top1),
        metric=registry_config(cfg).get("metric", "val_top1"),
    )


def find_registry_checkpoint(
    cfg: dict[str, Any],
    *,
    target_slug: str | None = None,
    role: str | None = None,
) -> CheckpointResolution:
    target_dir = registry_dir(cfg)
    slug = sanitize_slug(target_slug or config_slug(cfg))
    candidates = _registry_candidates(target_dir, slug=slug, role=role, scene_slug=_scope_slug_from_config(cfg))
    if candidates:
        best = max(candidates, key=lambda item: (item["metric_value"], item["mtime"]))
        return CheckpointResolution(
            path=best["path"],
            source="registry",
            metadata=best["metadata"],
            registry_dir=target_dir,
            candidates=[str(item["path"]) for item in candidates],
        )
    return CheckpointResolution(
        path=None,
        source="missing",
        registry_dir=target_dir,
        candidates=[],
    )


def resolve_evaluation_checkpoint(cfg: dict[str, Any], weights: str | None = None) -> CheckpointResolution:
    if weights:
        explicit = resolve_path(weights)
        return CheckpointResolution(
            path=explicit,
            source="explicit",
            metadata=load_checkpoint_metadata(explicit),
            requested=weights,
        )

    configured = cfg.get("evaluation", {}).get("weights")
    if configured and Path(str(configured)).expanduser().is_absolute():
        explicit = Path(str(configured)).expanduser()
        return CheckpointResolution(
            path=explicit,
            source="explicit",
            metadata=load_checkpoint_metadata(explicit),
            requested=str(configured),
        )

    registry_resolution = CheckpointResolution(path=None, source="missing", registry_dir=registry_dir(cfg))
    if registry_preferred(cfg):
        registry_resolution = find_registry_checkpoint(
            cfg,
            target_slug=config_slug(cfg),
            role=artifact_role(cfg),
        )
        if registry_resolution.path is not None:
            registry_resolution.requested = str(configured) if configured else None
            return registry_resolution

    return CheckpointResolution(
        path=None,
        source="missing" if configured else "none",
        registry_dir=registry_resolution.registry_dir,
        candidates=registry_resolution.candidates,
        requested=str(configured) if configured else None,
    )


def _registry_candidates(
    target_dir: Path,
    *,
    slug: str,
    role: str | None,
    scene_slug: str | None = None,
) -> list[dict[str, Any]]:
    if not target_dir.exists():
        return []
    candidates: list[dict[str, Any]] = []
    for checkpoint in target_dir.glob("*.pth"):
        metadata = load_checkpoint_metadata(checkpoint) or {}
        candidate_slug = metadata.get("config_slug") or _slug_from_filename(checkpoint.name)
        candidate_role = metadata.get("artifact_role")
        if candidate_slug != slug:
            continue
        if role is not None and candidate_role not in {None, role}:
            continue
        candidate_scene_slug = metadata.get("scene_slug")
        if scene_slug is not None and candidate_scene_slug is not None and candidate_scene_slug != scene_slug:
            continue
        metric_value = metadata.get("metric_value")
        if metric_value is None:
            metric_value = _accuracy_from_filename(checkpoint.name)
        if metric_value is None:
            continue
        candidates.append(
            {
                "path": checkpoint,
                "metadata": {**metadata, "path": str(checkpoint)},
                "metric_value": float(metric_value),
                "mtime": checkpoint.stat().st_mtime,
            }
        )
    return candidates


def _remove_old_archives(target_dir: Path, *, slug: str, role: str, keep: Path, scene_slug: str | None) -> None:
    for item in _registry_candidates(target_dir, slug=slug, role=role, scene_slug=scene_slug):
        checkpoint = item["path"]
        if checkpoint == keep:
            continue
        sidecar = checkpoint_sidecar_path(checkpoint)
        checkpoint.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)


def _enabled_modalities_from_cfg(cfg: dict[str, Any]) -> list[str]:
    task = cfg.get("experiment", {}).get("task", "image")
    if task == "fusion":
        modalities = (
            cfg.get("model", {}).get("primary", {}).get("modalities")
            or cfg.get("model", {}).get("modalities")
            or ["image", "radar"]
        )
        return [str(name) for name in modalities]
    return [str(task)]


def _accuracy_from_filename(name: str) -> float | None:
    match = re.search(r"_acc_([0-9]+(?:\.[0-9]+)?)\.pth$", name)
    if not match:
        return None
    return float(match.group(1))


def _slug_from_filename(name: str) -> str | None:
    if "_acc_" not in name:
        return None
    return name.split("_acc_", 1)[0]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _default_scene_registry_dir(cfg: dict[str, Any]) -> Path:
    base = resolve_path(cfg.get("output", {}).get("dir", cfg.get("paths", {}).get("output_dir", "outputs")))
    if cfg.get("output", {}).get("group_by_scene", True) is not False:
        scope = runtime_output_scope_from_config(cfg)
        if scope is not None and base.name != scope.slug:
            base = base / scope.slug
    return base / "best_checkpoints"


def _scope_slug_from_config(cfg: dict[str, Any]) -> str | None:
    scope = runtime_output_scope_from_config(cfg)
    return scope.slug if scope is not None else None


def _metadata_from_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any] | None:
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    try:
        payload = load_torch_payload(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    registry_metadata = payload.get("checkpoint_registry")
    metadata = dict(registry_metadata) if isinstance(registry_metadata, dict) else {}
    normalization_artifacts = payload.get("normalization_artifacts")
    if isinstance(normalization_artifacts, dict):
        metadata.setdefault("normalization_artifacts", normalization_artifacts)
    metadata.update({key: payload[key] for key in GPS_CHECKPOINT_PROVENANCE_KEYS if key in payload})
    return metadata or None
