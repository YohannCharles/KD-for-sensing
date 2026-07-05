#!/usr/bin/env python3

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CONFIG_NAMES = (
    "final_config.yaml",
    "config.yaml",
    "resolved_config.yaml",
    "runtime_config.yaml",
)
BEST_NAMES = (
    "best_top1.pth",
    "best_top1.pt",
    "best_top1.ckpt",
    "best.pth",
    "best.pt",
    "best.ckpt",
    "best_model.pth",
    "best_model.pt",
    "best_model.ckpt",
    "checkpoint_best.pth",
    "checkpoint_best.pt",
    "checkpoint_best.ckpt",
    "ckpt_best.pth",
    "ckpt_best.pt",
    "model_best.pth",
    "model_best.pt",
    "checkpoints/best_top1.pth",
    "checkpoints/best_top1.pt",
    "checkpoints/best.pth",
    "checkpoints/best.pt",
    "checkpoints/best.ckpt",
    "checkpoints/best_model.pth",
    "checkpoints/best_model.pt",
    "checkpoints/checkpoint_best.pth",
    "checkpoints/checkpoint_best.pt",
    "checkpoints/ckpt_best.pth",
    "checkpoints/ckpt_best.pt",
    "checkpoints/model_best.pth",
    "checkpoints/model_best.pt",
)
LAST_NAMES = (
    "last.pth",
    "last.pt",
    "last.ckpt",
    "latest.pth",
    "latest.pt",
    "checkpoint_last.pth",
    "checkpoint_last.pt",
    "checkpoints/last.pth",
    "checkpoints/last.pt",
    "checkpoints/last.ckpt",
    "checkpoints/latest.pth",
    "checkpoints/latest.pt",
    "checkpoints/checkpoint_last.pth",
    "checkpoints/checkpoint_last.pt",
)
IGNORED_ROOT_CHILDREN = {
    "fresh_eval",
    "fresh_eval_main",
    "p0_fresh_eval",
    "logs",
    "summary",
    "worker_status",
    "checkpoint_selection",
    "best_checkpoints",
}


@dataclass
class CheckpointChoice:
    path: Path | None = None
    checkpoint_used: str = "missing"
    best_epoch: int | None = None
    best_val_acc: float | None = None
    warnings: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)


@dataclass
class RunResolution:
    run_name: str
    searched_paths: list[str]
    run_dir: Path | None = None
    config_path: Path | None = None
    status_json_exists: bool = False
    status_state: str = ""
    checkpoint: CheckpointChoice = field(default_factory=CheckpointChoice)
    last_checkpoint_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def diagnosis(self) -> str:
        if self.run_dir is None:
            return "missing_run_dir"
        if self.status_state and self.status_state != "complete":
            return f"status_{self.status_state}"
        if self.config_path is None:
            return "missing_config"
        if self.checkpoint.path is None:
            return "missing_checkpoint"
        if self.checkpoint.checkpoint_used == "last_fallback":
            return "last_checkpoint_fallback"
        return "ok"


def resolve_run_dir_and_config(
    root: str | Path,
    run_name: str,
    experiment_group: str | None = None,
) -> RunResolution:
    root_path = Path(root)
    dirs = _candidate_run_dirs(root_path, run_name, experiment_group=experiment_group)
    searched = [str(item) for item in dirs]
    infos = [_run_dir_info(path, run_name) for path in dirs]
    existing = [info for info in infos if _candidate_exists(info)]
    if not existing:
        return RunResolution(run_name=run_name, searched_paths=searched)

    def key(info: dict[str, Any]) -> tuple[int, int, int, int, float]:
        return (
            int(info["status_state"] == "complete"),
            int(info["checkpoint"].checkpoint_used == "best"),
            int(info["config_path"] is not None),
            int(info["path"].exists()),
            _mtime(info["path"]),
        )

    ranked = sorted(existing, key=key, reverse=True)
    selected = ranked[0]
    tied = [info for info in ranked if key(info)[:-1] == key(selected)[:-1]]
    warnings: list[str] = []
    if len(tied) > 1:
        warnings.append(
            "multiple candidate run dirs matched; selected latest modified path: "
            + str(selected["path"])
        )
    return RunResolution(
        run_name=run_name,
        searched_paths=searched,
        run_dir=selected["path"],
        config_path=selected["config_path"],
        status_json_exists=selected["status_json_exists"],
        status_state=selected["status_state"],
        checkpoint=selected["checkpoint"],
        last_checkpoint_path=selected["last_checkpoint_path"],
        warnings=[*warnings, *selected["warnings"]],
    )


def resolve_best_checkpoint(run_dir: str | Path | None, config_path: str | Path | None = None) -> CheckpointChoice:
    if run_dir is None:
        return CheckpointChoice(warnings=["missing run_dir"])
    run_path = Path(run_dir)
    run_name = run_path.name
    best_epoch, best_val_acc, metric_warnings = _best_epoch_from_metrics(run_path)
    best = _best_checkpoint_candidates(run_path, run_name, best_epoch=best_epoch)
    if best:
        selected = _select_best_candidate(best, best_epoch)
        meta = _metadata(selected)
        return CheckpointChoice(
            path=selected,
            checkpoint_used="best",
            best_epoch=_metadata_epoch(meta) or best_epoch,
            best_val_acc=_metadata_metric(meta) if _metadata_metric(meta) is not None else best_val_acc,
            warnings=metric_warnings,
            candidates=[str(item) for item in best],
        )

    last = _last_checkpoint_candidates(run_path)
    if last:
        selected = max(last, key=_mtime)
        return CheckpointChoice(
            path=selected,
            checkpoint_used="last_fallback",
            best_epoch=best_epoch,
            best_val_acc=best_val_acc,
            warnings=[
                *metric_warnings,
                "best checkpoint not found; falling back to last checkpoint",
            ],
            candidates=[str(item) for item in [*best, *last]],
        )

    searched = [str(run_path / name) for name in (*BEST_NAMES, *LAST_NAMES)]
    if (run_path.parent / "best_checkpoints").exists():
        searched.append(str(run_path.parent / "best_checkpoints" / f"{run_name}_*"))
    return CheckpointChoice(
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        warnings=[*metric_warnings, "no checkpoint candidates found"],
        candidates=searched,
    )


def complete_run_names(root: str | Path) -> list[str]:
    root_path = Path(root)
    names: set[str] = set()
    for status_path in root_path.rglob("run_status.json"):
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("state") == "complete":
            names.add(status_path.parent.name)
    return sorted(names)


def diagnostics_row(root: str | Path, run_name: str, experiment_group: str | None = None) -> dict[str, Any]:
    resolution = resolve_run_dir_and_config(root, run_name, experiment_group=experiment_group)
    checkpoint = resolution.checkpoint
    return {
        "run_name": run_name,
        "searched_paths": ";".join(resolution.searched_paths),
        "actual_run_dir": str(resolution.run_dir or ""),
        "status_json_exists": resolution.status_json_exists,
        "status_state": resolution.status_state,
        "config_exists": resolution.config_path is not None,
        "config_path": str(resolution.config_path or ""),
        "best_ckpt_exists": checkpoint.path is not None and checkpoint.checkpoint_used == "best",
        "best_ckpt_path": str(checkpoint.path if checkpoint.checkpoint_used == "best" else ""),
        "last_ckpt_exists": resolution.last_checkpoint_path is not None,
        "last_ckpt_path": str(resolution.last_checkpoint_path or ""),
        "checkpoint_used": checkpoint.checkpoint_used,
        "checkpoint_path": str(checkpoint.path or ""),
        "best_epoch": checkpoint.best_epoch if checkpoint.best_epoch is not None else "",
        "best_val_acc": checkpoint.best_val_acc if checkpoint.best_val_acc is not None else "",
        "warnings": ";".join([*resolution.warnings, *checkpoint.warnings]),
        "diagnosis": resolution.diagnosis,
    }


def _candidate_run_dirs(root: Path, run_name: str, experiment_group: str | None = None) -> list[Path]:
    candidates = [root / run_name, root / "scene31" / run_name]
    if experiment_group:
        candidates.append(root / experiment_group / run_name)
    if root.exists():
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name in IGNORED_ROOT_CHILDREN:
                continue
            candidates.append(child / run_name)
    return _unique_paths(candidates)


def _run_dir_info(path: Path, run_name: str) -> dict[str, Any]:
    status_path = path / "run_status.json"
    status_state = ""
    warnings: list[str] = []
    if status_path.exists():
        try:
            status_state = str(json.loads(status_path.read_text(encoding="utf-8")).get("state") or "")
        except json.JSONDecodeError:
            warnings.append(f"invalid run_status.json: {status_path}")
    config_path = _first_existing(path / name for name in CONFIG_NAMES)
    checkpoint = resolve_best_checkpoint(path if path.exists() else None, config_path)
    return {
        "path": path,
        "status_json_exists": status_path.exists(),
        "status_state": status_state,
        "config_path": config_path,
        "checkpoint": checkpoint,
        "last_checkpoint_path": _first_existing(path / name for name in LAST_NAMES),
        "warnings": warnings,
    }


def _candidate_exists(info: dict[str, Any]) -> bool:
    return bool(
        info["path"].exists()
        or info["status_json_exists"]
        or info["config_path"] is not None
        or info["checkpoint"].path is not None
    )


def _best_checkpoint_candidates(run_dir: Path, run_name: str, *, best_epoch: int | None) -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(run_dir / name for name in BEST_NAMES)
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.exists():
        for pattern in ("*best*.pth", "*best*.pt", "*best*.ckpt", "*.pth", "*.pt", "*.ckpt"):
            candidates.extend(checkpoint_dir.glob(pattern))
    if best_epoch is not None:
        candidates.extend(_epoch_checkpoint_names(run_dir, best_epoch))
    registry = run_dir.parent / "best_checkpoints"
    if registry.exists():
        for suffix in ("*.pth", "*.pt", "*.ckpt"):
            candidates.extend(registry.glob(f"{run_name}_{suffix}"))
    return _unique_paths(path for path in candidates if path.exists() and _belongs_to_run(path, run_name))


def _last_checkpoint_candidates(run_dir: Path) -> list[Path]:
    candidates = [run_dir / name for name in LAST_NAMES]
    return _unique_paths(path for path in candidates if path.exists())


def _select_best_candidate(paths: list[Path], best_epoch: int | None) -> Path:
    def key(path: Path) -> tuple[int, float, float]:
        meta = _metadata(path)
        epoch = _metadata_epoch(meta)
        metric = _metadata_metric(meta)
        exact_epoch = int(best_epoch is not None and epoch == best_epoch)
        return (exact_epoch, metric if metric is not None else -math.inf, _mtime(path))

    return max(paths, key=key)


def _epoch_checkpoint_names(run_dir: Path, epoch: int) -> list[Path]:
    names: list[Path] = []
    for value in (str(epoch), f"{epoch:02d}", f"{epoch:03d}"):
        for stem in (
            f"epoch_{value}",
            f"checkpoint_epoch_{value}",
            f"ckpt_epoch_{value}",
            f"checkpoint_{value}",
            f"ckpt_{value}",
        ):
            for suffix in (".pth", ".pt", ".ckpt"):
                names.append(run_dir / f"{stem}{suffix}")
                names.append(run_dir / "checkpoints" / f"{stem}{suffix}")
    return names


def _best_epoch_from_metrics(run_dir: Path) -> tuple[int | None, float | None, list[str]]:
    for name in ("metrics.csv", "history.csv"):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
        except OSError as exc:
            return None, None, [f"failed to read {path}: {exc}"]
        best_epoch: int | None = None
        best_value: float | None = None
        for row in rows:
            value = _first_float(row, "val_acc", "val_beam_top1", "top1_acc", "accuracy/top1", "primary_acc")
            epoch = _int_value(row.get("epoch"))
            if value is None or epoch is None:
                continue
            if best_value is None or value > best_value:
                best_value = value
                best_epoch = epoch
        if best_epoch is not None:
            return best_epoch, best_value, []
        return None, None, [f"{path} has no usable val_acc/top1 rows"]
    return None, None, [f"metrics.csv not found: {run_dir / 'metrics.csv'}"]


def _belongs_to_run(path: Path, run_name: str) -> bool:
    if run_name in path.parts:
        return True
    meta = _metadata(path)
    run_dir = meta.get("run_dir")
    if run_dir and Path(str(run_dir)).name == run_name:
        return True
    for key in ("config_slug", "experiment_name"):
        if meta.get(key) == run_name:
            return True
    return path.stem.startswith(f"{run_name}_")


def _metadata(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _metadata_epoch(meta: dict[str, Any]) -> int | None:
    for key in ("selected_epoch", "epoch", "best_top1_epoch", "best_early_stopping_epoch"):
        value = _int_value(meta.get(key))
        if value is not None:
            return value
    return None


def _metadata_metric(meta: dict[str, Any]) -> float | None:
    value = _first_float(meta, "metric_value", "val_acc", "primary_acc")
    if value is not None:
        return value
    objective = meta.get("objective_metric")
    if isinstance(objective, dict):
        return _first_float(objective, "value")
    task_metrics = meta.get("task_metrics")
    if isinstance(task_metrics, dict):
        return _first_float(task_metrics, "val_acc", "val_beam_top1", "primary_acc")
    return None


def _first_existing(paths: Any) -> Path | None:
    for path in paths:
        if Path(path).exists():
            return Path(path)
    return None


def _unique_paths(paths: Any) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for item in paths:
        path = Path(item)
        key = path.resolve() if path.exists() else path
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _first_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _float(row.get(key))
        if value is not None:
            return value
    return None


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_value(value: Any) -> int | None:
    number = _float(value)
    if number is None:
        return None
    return int(number)


def run_name_sort_key(name: str) -> tuple[str, int]:
    match = re.search(r"_seed(\d+)$", name)
    return (re.sub(r"_seed\d+$", "", name), int(match.group(1)) if match else -1)
