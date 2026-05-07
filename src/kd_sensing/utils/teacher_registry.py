from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from kd_sensing.modalities import MODALITY_ORDER, normalize_modalities
from kd_sensing.utils.artifact_registry import load_checkpoint_metadata
from kd_sensing.utils.paths import resolve_path


REQUIRED_TEACHER_METRICS = (
    "modality",
    "best_epoch",
    "val_acc_top1",
    "val_acc_top3",
    "val_acc_top5",
    "val_adba",
    "train_acc_top1",
)

SCENE32_MANUAL_PRIOR = {
    "image": 0.20,
    "radar": 0.20,
    "gps": 0.85,
    "lidar": 0.15,
    "mmwave": 0.90,
}

DEFAULT_METRIC_PRIOR_WEIGHTS = {
    "val_acc_top1": 0.6,
    "val_acc_top3": 0.2,
    "val_adba": 0.2,
}


@dataclass(frozen=True)
class TeacherRun:
    modality: str
    run_dir: Path
    metrics: dict[str, Any]
    checkpoint: Path
    checkpoint_metadata: dict[str, Any] | None


def build_teacher_registry(
    *,
    teacher_root: str | Path,
    output_path: str | Path,
    scene: int | str = 32,
    modalities: list[str] | tuple[str, ...] = MODALITY_ORDER,
    prior_mode: str = "manual",
    manual_prior: dict[str, float] | None = None,
    metric_prior_weights: dict[str, float] | None = None,
    prior_min: float = 0.05,
    prior_max: float = 0.95,
) -> dict[str, Any]:
    selected = normalize_modalities(modalities, context="teacher registry modalities")
    root = resolve_path(teacher_root)
    output = resolve_path(output_path)
    if not root.exists():
        raise FileNotFoundError(f"Teacher root not found: {root}")
    runs = [discover_teacher_run(root, modality) for modality in selected]
    priors = _compute_priors(
        runs,
        scene=scene,
        prior_mode=prior_mode,
        manual_prior=manual_prior,
        metric_prior_weights=metric_prior_weights,
        prior_min=prior_min,
        prior_max=prior_max,
    )
    weights = {**DEFAULT_METRIC_PRIOR_WEIGHTS, **(metric_prior_weights or {})}
    registry = {
        "schema_version": 1,
        "scene": f"scene{scene}" if str(scene).isdigit() else str(scene),
        "scene_id": int(scene) if str(scene).isdigit() else scene,
        "teacher_root": str(root),
        "prior_mode": str(prior_mode),
        "prior_min": float(prior_min),
        "prior_max": float(prior_max),
        "metric_prior_weights": weights if str(prior_mode) == "metric" else None,
        "modalities": list(selected),
        "teachers": {},
    }
    for run in runs:
        metric_values = {key: run.metrics[key] for key in REQUIRED_TEACHER_METRICS if key != "modality"}
        registry["teachers"][run.modality] = {
            "modality": run.modality,
            "run_dir": str(run.run_dir),
            "ckpt": str(run.checkpoint),
            "checkpoint": str(run.checkpoint),
            "checkpoint_metadata": run.checkpoint_metadata or {},
            "metrics": metric_values,
            "val_acc_top1": float(run.metrics["val_acc_top1"]),
            "val_acc_top3": float(run.metrics["val_acc_top3"]),
            "val_acc_top5": float(run.metrics["val_acc_top5"]),
            "val_adba": float(run.metrics["val_adba"]),
            "train_acc_top1": float(run.metrics["train_acc_top1"]),
            "best_epoch": int(run.metrics["best_epoch"]),
            "prior": float(priors[run.modality]),
            "prior_mode": str(prior_mode),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    return registry


def discover_teacher_run(teacher_root: str | Path, modality: str) -> TeacherRun:
    root = Path(teacher_root)
    candidates = _teacher_run_candidates(root, modality)
    for run_dir in candidates:
        if not run_dir.exists() or not run_dir.is_dir():
            continue
        metrics = read_teacher_metrics(run_dir, modality)
        checkpoint = find_teacher_checkpoint(run_dir, metrics)
        metadata = load_checkpoint_metadata(checkpoint)
        return TeacherRun(
            modality=modality,
            run_dir=run_dir,
            metrics=metrics,
            checkpoint=checkpoint,
            checkpoint_metadata=metadata,
        )
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find teacher run for modality '{modality}'. Checked: {checked}")


def read_teacher_metrics(run_dir: str | Path, expected_modality: str) -> dict[str, Any]:
    run_path = Path(run_dir)
    for path in _metrics_candidates(run_path):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        metrics = _normalize_metrics_payload(raw, expected_modality)
        if metrics is not None:
            return metrics
        if path.name == "teacher_metrics.json":
            missing = _missing_metric_fields(raw)
            raise ValueError(
                f"Teacher metrics for modality '{expected_modality}' are missing fields: {missing}."
            )
    train_log = run_path / "train_log.json"
    if train_log.exists():
        with train_log.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return metrics_from_train_log(raw, expected_modality)
    raise FileNotFoundError(
        f"Teacher metrics not found for modality '{expected_modality}' in {run_path}. "
        "Expected teacher_metrics.json, metrics.json, or train_log.json."
    )


def metrics_from_train_log(train_log: dict[str, Any], expected_modality: str) -> dict[str, Any]:
    epoch_logs = train_log.get("epoch_logs")
    if isinstance(epoch_logs, list) and epoch_logs:
        best_idx = max(range(len(epoch_logs)), key=lambda idx: float(epoch_logs[idx].get("val_acc", 0.0)))
        best = epoch_logs[best_idx]
        metrics = {
            "modality": _resolve_metric_modality(train_log, expected_modality),
            "best_epoch": int(best.get("epoch", best_idx + 1)),
            "val_acc_top1": float(best.get("val_acc", best.get("val_acc_top1", 0.0))),
            "val_acc_top3": float(best.get("val_atop3", best.get("val_acc_top3", 0.0))),
            "val_acc_top5": float(best.get("val_atop5", best.get("val_acc_top5", 0.0))),
            "val_adba": float(best.get("val_adba", 0.0)),
            "train_acc_top1": float(best.get("train_acc", best.get("train_acc_top1", 0.0))),
        }
        _validate_teacher_metrics(metrics, expected_modality)
        return metrics

    val_acc = train_log.get("val_acc") or []
    if not val_acc:
        raise ValueError(f"train_log for modality '{expected_modality}' does not contain val_acc history.")
    best_idx = max(range(len(val_acc)), key=lambda idx: float(val_acc[idx]))
    metrics = {
        "modality": _resolve_metric_modality(train_log, expected_modality),
        "best_epoch": int(best_idx + 1),
        "val_acc_top1": float(val_acc[best_idx]),
        "val_acc_top3": _history_value(train_log, "val_atop3", best_idx),
        "val_acc_top5": _history_value(train_log, "val_atop5", best_idx),
        "val_adba": _history_value(train_log, "val_adba", best_idx),
        "train_acc_top1": _history_value(train_log, "train_acc", best_idx),
    }
    _validate_teacher_metrics(metrics, expected_modality)
    return metrics


def find_teacher_checkpoint(run_dir: str | Path, metrics: dict[str, Any] | None = None) -> Path:
    run_path = Path(run_dir)
    for key in ("ckpt", "checkpoint", "best_checkpoint", "checkpoint_path"):
        value = (metrics or {}).get(key)
        if value:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = run_path / candidate
            if candidate.exists():
                return candidate
    checkpoint_dir = run_path / "checkpoints"
    for name in ("best_top1.pth", "best.pth", "last.pth"):
        candidate = checkpoint_dir / name
        if candidate.exists():
            return candidate
    candidates = sorted(checkpoint_dir.glob("*.pth")) if checkpoint_dir.exists() else []
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No checkpoint found in teacher run: {run_path}")


def teacher_metrics_from_training(
    cfg: dict[str, Any],
    history: dict[str, Any],
    epoch_logs: list[dict[str, Any]],
    *,
    best_top1_epoch: int | None = None,
) -> dict[str, Any] | None:
    task = str(cfg.get("experiment", {}).get("task", "image"))
    if task == "fusion":
        return None
    if task not in MODALITY_ORDER:
        return None
    if epoch_logs:
        if best_top1_epoch and 1 <= best_top1_epoch <= len(epoch_logs):
            best_idx = int(best_top1_epoch) - 1
        else:
            best_idx = max(range(len(epoch_logs)), key=lambda idx: float(epoch_logs[idx].get("val_acc", 0.0)))
        best = epoch_logs[best_idx]
        return {
            "modality": task,
            "best_epoch": int(best.get("epoch", best_idx + 1)),
            "val_acc_top1": float(best.get("val_acc", 0.0)),
            "val_acc_top3": float(best.get("val_atop3", 0.0)),
            "val_acc_top5": float(best.get("val_atop5", 0.0)),
            "val_adba": float(best.get("val_adba", 0.0)),
            "train_acc_top1": float(best.get("train_acc", 0.0)),
        }
    val_acc = history.get("val_acc") or []
    if not val_acc:
        return None
    best_idx = max(range(len(val_acc)), key=lambda idx: float(val_acc[idx]))
    return {
        "modality": task,
        "best_epoch": best_idx + 1,
        "val_acc_top1": float(val_acc[best_idx]),
        "val_acc_top3": _history_value(history, "val_atop3", best_idx),
        "val_acc_top5": _history_value(history, "val_atop5", best_idx),
        "val_adba": _history_value(history, "val_adba", best_idx),
        "train_acc_top1": _history_value(history, "train_acc", best_idx),
    }


def parse_key_value_floats(items: list[str] | tuple[str, ...] | None) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in items or []:
        raw = str(item).strip()
        if not raw:
            continue
        if raw.startswith("{"):
            parsed = json.loads(raw)
            values.update({str(key): float(value) for key, value in parsed.items()})
            continue
        for part in raw.split(","):
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"Expected key=value item, got '{part}'.")
            key, value = part.split("=", 1)
            values[key.strip()] = float(value)
    return values


def _teacher_run_candidates(root: Path, modality: str) -> list[Path]:
    if root.name in {modality, f"{modality}_teacher_no_kd"}:
        return [root]
    return [
        root / f"{modality}_teacher_no_kd",
        root / modality,
        root / f"{modality}_teacher",
    ]


def _metrics_candidates(run_dir: Path) -> list[Path]:
    return [
        run_dir / "teacher_metrics.json",
        run_dir / "metrics.json",
        run_dir / "artifacts" / "teacher_metrics.json",
    ]


def _normalize_metrics_payload(raw: dict[str, Any], expected_modality: str) -> dict[str, Any] | None:
    if "metrics" in raw and isinstance(raw["metrics"], dict):
        raw = {**raw["metrics"], **{key: value for key, value in raw.items() if key != "metrics"}}
    aliases = {
        "val_acc": "val_acc_top1",
        "val_atop3": "val_acc_top3",
        "val_atop5": "val_acc_top5",
        "train_acc": "train_acc_top1",
        "epoch": "best_epoch",
    }
    metrics = dict(raw)
    for source, target in aliases.items():
        if target not in metrics and source in metrics:
            metrics[target] = metrics[source]
    if "modality" not in metrics:
        metrics["modality"] = expected_modality
    required_without_modality = [key for key in REQUIRED_TEACHER_METRICS if key != "modality"]
    if not all(key in metrics for key in required_without_modality):
        return None
    _validate_teacher_metrics(metrics, expected_modality)
    return {
        "modality": str(metrics["modality"]),
        "best_epoch": int(metrics["best_epoch"]),
        "val_acc_top1": float(metrics["val_acc_top1"]),
        "val_acc_top3": float(metrics["val_acc_top3"]),
        "val_acc_top5": float(metrics["val_acc_top5"]),
        "val_adba": float(metrics["val_adba"]),
        "train_acc_top1": float(metrics["train_acc_top1"]),
        **{key: value for key, value in metrics.items() if key in {"ckpt", "checkpoint", "best_checkpoint", "checkpoint_path"}},
    }


def _missing_metric_fields(raw: dict[str, Any]) -> list[str]:
    payload = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else raw
    aliases = {
        "val_acc_top1": ("val_acc",),
        "val_acc_top3": ("val_atop3",),
        "val_acc_top5": ("val_atop5",),
        "train_acc_top1": ("train_acc",),
        "best_epoch": ("epoch",),
    }
    missing = []
    for key in REQUIRED_TEACHER_METRICS:
        if key in payload:
            continue
        if any(alias in payload for alias in aliases.get(key, ())):
            continue
        missing.append(key)
    return missing


def _validate_teacher_metrics(metrics: dict[str, Any], expected_modality: str) -> None:
    missing = [key for key in REQUIRED_TEACHER_METRICS if key not in metrics]
    if missing:
        raise ValueError(
            f"Teacher metrics for modality '{expected_modality}' are missing fields: {missing}."
        )
    actual = str(metrics["modality"])
    if actual != expected_modality:
        raise ValueError(
            f"Teacher metrics modality mismatch: expected '{expected_modality}', got '{actual}'."
        )


def _resolve_metric_modality(train_log: dict[str, Any], expected_modality: str) -> str:
    runtime = train_log.get("runtime") if isinstance(train_log.get("runtime"), dict) else {}
    splits = runtime.get("splits") if isinstance(runtime.get("splits"), dict) else {}
    train_split = splits.get("train") if isinstance(splits.get("train"), dict) else {}
    enabled = train_split.get("enabled_modalities")
    if isinstance(enabled, list) and len(enabled) == 1:
        return str(enabled[0])
    return expected_modality


def _history_value(history: dict[str, Any], key: str, index: int) -> float:
    values = history.get(key) or []
    if index < len(values):
        return float(values[index])
    return 0.0


def _compute_priors(
    runs: list[TeacherRun],
    *,
    scene: int | str,
    prior_mode: str,
    manual_prior: dict[str, float] | None,
    metric_prior_weights: dict[str, float] | None,
    prior_min: float,
    prior_max: float,
) -> dict[str, float]:
    mode = str(prior_mode)
    if not 0.0 <= float(prior_min) <= float(prior_max) <= 1.0:
        raise ValueError("prior_min and prior_max must satisfy 0 <= prior_min <= prior_max <= 1.")
    if mode == "manual":
        defaults = SCENE32_MANUAL_PRIOR if str(scene) in {"32", "scene32", "scenario32"} else {}
        configured = {**defaults, **(manual_prior or {})}
        missing = [run.modality for run in runs if run.modality not in configured]
        if missing:
            raise ValueError(f"Manual prior missing modalities: {missing}.")
        return {
            run.modality: _clamp(float(configured[run.modality]), prior_min, prior_max)
            for run in runs
        }
    if mode == "metric":
        weights = {**DEFAULT_METRIC_PRIOR_WEIGHTS, **(metric_prior_weights or {})}
        scores = {}
        for run in runs:
            missing = [key for key in weights if key not in run.metrics]
            if missing:
                raise ValueError(
                    f"Teacher metrics for modality '{run.modality}' are missing fields: {missing}."
                )
            scores[run.modality] = sum(float(weights[key]) * float(run.metrics[key]) for key in weights)
        max_score = max(scores.values()) if scores else 0.0
        if max_score <= 0.0:
            return {modality: float(prior_min) for modality in scores}
        return {
            modality: _clamp(score / max_score, prior_min, prior_max)
            for modality, score in scores.items()
        }
    raise ValueError("prior_mode must be 'manual' or 'metric'.")


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))
