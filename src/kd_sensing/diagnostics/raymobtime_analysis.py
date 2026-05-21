from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from kd_sensing.utils.paths import resolve_path


def analyze_raymobtime_modality_imbalance(
    exp_dirs: list[str] | tuple[str, ...] | None = None,
    exp_dir: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    roots = [exp_dir] if exp_dir else list(exp_dirs or [])
    if not roots:
        raise ValueError("Raymobtime analysis requires exp_dir or exp_dirs.")
    runs = [_load_run(Path(resolve_path(root))) for root in roots]
    out_dir = Path(resolve_path(output_dir or Path(roots[0]) / "raymobtime_analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)
    single_rows = _single_modality_rows(runs)
    gate_rows = _gate_rows(runs)
    drop_rows = _drop_rows(runs)
    los_rows = _los_bucket_rows(runs)
    gradient_rows = _gradient_rows(runs)
    outputs = {
        "single_modality_task_performance": str(out_dir / "single_modality_task_performance.csv"),
        "gate_mean_by_task": str(out_dir / "gate_mean_by_task.csv"),
        "gate_mean_by_task_and_los_bucket": str(out_dir / "gate_mean_by_task_and_los_bucket.csv"),
        "drop_modality_delta": str(out_dir / "drop_modality_delta.csv"),
        "beam_metrics_by_los_bucket": str(out_dir / "beam_metrics_by_los_bucket.csv"),
        "grad_norms_by_task_modality": str(out_dir / "grad_norms_by_task_modality.csv"),
        "summary": str(out_dir / "summary.json"),
    }
    _write_csv(Path(outputs["single_modality_task_performance"]), single_rows)
    _write_csv(Path(outputs["gate_mean_by_task"]), gate_rows)
    _write_csv(Path(outputs["gate_mean_by_task_and_los_bucket"]), los_rows)
    _write_csv(Path(outputs["drop_modality_delta"]), drop_rows)
    _write_csv(Path(outputs["beam_metrics_by_los_bucket"]), los_rows)
    _write_csv(Path(outputs["grad_norms_by_task_modality"]), gradient_rows)
    summary = {
        "runs": runs,
        "outputs": outputs,
        "conditions": sorted({run["condition"] for run in runs}),
    }
    Path(outputs["summary"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _load_run(path: Path) -> dict[str, Any]:
    metrics = _read_json(path / "metrics.json") or _read_json(path / "test_report.json")
    train_log = _read_json(path / "train_log.json")
    final_config = _read_json(path / "final_config.yaml")
    runtime = train_log.get("runtime") or metrics.get("runtime") or final_config.get("runtime") or {}
    setup = (
        runtime.get("prediction_setup")
        or metrics.get("prediction_setup")
        or train_log.get("prediction_setup")
        or final_config.get("runtime", {}).get("prediction_setup")
        or {}
    )
    objective = (
        runtime.get("prediction_objective", {}).get("name")
        or metrics.get("objective", {}).get("name")
        or setup.get("objective")
        or final_config.get("experiment", {}).get("objective")
    )
    modalities = (
        runtime.get("enabled_modalities")
        or setup.get("enabled_modalities")
        or metrics.get("enabled_modalities")
        or final_config.get("model", {}).get("modalities")
        or []
    )
    dataset_type = final_config.get("data", {}).get("dataset", {}).get("type")
    task_semantics = setup.get("task_semantics") or final_config.get("experiment", {}).get("task_semantics")
    if dataset_type not in {None, "raymobtime_s008"}:
        raise ValueError(f"Refusing non-Raymobtime experiment {path}: dataset type is {dataset_type!r}.")
    if objective not in {
        "current_beam_selection",
        "current_los_classification",
        "current_link_quality",
        "selection_multitask",
    }:
        raise ValueError(f"Refusing non-selection experiment {path}: objective is {objective!r}.")
    if task_semantics not in {None, "current_snapshot_beam_selection"}:
        raise ValueError(f"Refusing non-current snapshot experiment {path}: task_semantics={task_semantics!r}.")
    condition = "sensing+ray" if "ray" in modalities else "sensing-only"
    return {
        "path": str(path),
        "dataset_type": dataset_type or "raymobtime_s008",
        "objective": objective,
        "task_semantics": task_semantics or "current_snapshot_beam_selection",
        "enabled_modalities": list(modalities),
        "condition": condition,
        "metrics": metrics,
        "train_log": train_log,
    }


def _single_modality_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        metrics = run["metrics"]
        rows.append(
            {
                "run_dir": run["path"],
                "modalities": "+".join(run["enabled_modalities"]),
                "condition": run["condition"],
                "beam_top1": metrics.get("val_beam_top1", metrics.get("beam_top1")),
                "beam_top3": metrics.get("val_beam_top3", metrics.get("beam_top3")),
                "beam_top5": metrics.get("val_beam_top5", metrics.get("beam_top5")),
                "los_f1": metrics.get("val_los_f1", metrics.get("los_f1")),
                "link_mae": metrics.get("val_link_mae", metrics.get("link_mae")),
            }
        )
    return rows


def _gate_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        epoch_logs = run.get("train_log", {}).get("epoch_logs", [])
        latest = epoch_logs[-1] if epoch_logs else {}
        for key, value in latest.items():
            if key.startswith("gate/") or key.startswith("raymobtime/gate/"):
                parts = key.split("/")
                rows.append({"run_dir": run["path"], "metric": key, "task": parts[-2], "modality": parts[-1], "value": value})
    return rows


def _drop_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        subsets = run["metrics"].get("modality_subsets", {})
        if not isinstance(subsets, dict):
            continue
        full = subsets.get("all", run["metrics"])
        full_top1 = _metric_value(full, "val_beam_top1", "beam_top1")
        full_los = _metric_value(full, "val_los_f1", "los_f1")
        full_link = _metric_value(full, "val_link_mae", "link_mae")
        for name, metrics in subsets.items():
            if not str(name).startswith("drop_") or not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "run_dir": run["path"],
                    "drop_modality": str(name).removeprefix("drop_"),
                    "delta_beam_top1": _metric_value(metrics, "val_beam_top1", "beam_top1") - full_top1,
                    "delta_los_f1": _metric_value(metrics, "val_los_f1", "los_f1") - full_los,
                    "delta_link_mae": _metric_value(metrics, "val_link_mae", "link_mae") - full_link,
                }
            )
    return rows


def _los_bucket_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        buckets = run["metrics"].get("los_buckets", {})
        if isinstance(buckets, dict):
            for bucket, metrics in buckets.items():
                rows.append({"run_dir": run["path"], "los_bucket": bucket, **metrics})
    return rows


def _gradient_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        diagnostics = run.get("train_log", {}).get("epoch_logs", [])
        latest = diagnostics[-1] if diagnostics else {}
        for key, value in latest.items():
            if "grad" in key and any(modality in key for modality in ("coord", "image", "lidar", "ray")):
                rows.append({"run_dir": run["path"], "metric": key, "value": value})
    return rows


def _metric_value(metrics: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml

            return yaml.safe_load(text) or {}
        except Exception:
            return {}
    return json.loads(text)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    if not fieldnames:
        fieldnames = ["run_dir"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


__all__ = ["analyze_raymobtime_modality_imbalance"]
