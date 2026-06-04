from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any

import torch

from kd_sensing.data.scenes import scene_metadata_from_config, scene_slug_from_config
from kd_sensing.engine.run_lineage import model_capacity, run_lineage_metadata
from kd_sensing.utils.paths import resolve_path


DEFAULT_REGISTRY = {
    "enabled": True,
    "dir": "outputs/best_checkpoints",
    "prefer": True,
    "metric": "val_top1",
    "filename": "{slug}_{role}_acc_{acc}.pth",
}


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
    if configured_dir == DEFAULT_REGISTRY["dir"]:
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
    return load_sidecar(sidecar)


def load_sidecar(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_sidecar(checkpoint_path: str | Path, metadata: dict[str, Any]) -> Path:
    sidecar = checkpoint_sidecar_path(checkpoint_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(metadata), f, indent=2)
    return sidecar


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
    _remove_old_archives(target_dir, slug=slug, role=role, keep=target, scene_slug=scene_slug_from_config(cfg))
    shutil.copy2(source, target)

    lineage = run_lineage_metadata(cfg)
    metadata = {
        "path": str(target),
        "source": "registry",
        "source_checkpoint": str(source),
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
    }
    metadata.update(scene_metadata_from_config(cfg))
    sidecar = write_sidecar(target, metadata)
    metadata["sidecar_path"] = str(sidecar)
    write_sidecar(target, metadata)
    return metadata


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
    candidates = _registry_candidates(target_dir, slug=slug, role=role, scene_slug=scene_slug_from_config(cfg))
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
        scene_slug = scene_slug_from_config(cfg)
        if scene_slug and base.name != scene_slug:
            base = base / scene_slug
    return base / "best_checkpoints"


def _metadata_from_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any] | None:
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - older torch
            payload = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    registry_metadata = payload.get("checkpoint_registry")
    if isinstance(registry_metadata, dict):
        return registry_metadata
    normalization_artifacts = payload.get("normalization_artifacts")
    if isinstance(normalization_artifacts, dict):
        return {"normalization_artifacts": normalization_artifacts}
    return None
