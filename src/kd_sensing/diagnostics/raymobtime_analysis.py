from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from kd_sensing.utils.paths import resolve_path

_RAYMOBTIME_OBJECTIVES = {
    "current_beam_selection",
    "current_los_classification",
    "current_link_quality",
    "selection_multitask",
}
_SENSING_MODALITIES = ("coord", "image", "lidar")
_BEST_METRICS = (
    ("val_selection_multitask_loss", "min"),
    ("val_beam_top1", "max"),
    ("val_link_mae", "min"),
)
_REEVALUATION_VIEWS = {
    "val_selection_multitask_loss": "best",
    "val_beam_top1": "best_top1",
    "val_link_mae": "best_link_mae",
}
_PAIRED_EVALUATION_SUFFIXES = ("best", "best_top1", "best_link_mae")
_COMPARISON_METRICS = (
    "val_beam_top1",
    "val_beam_top3",
    "val_beam_top5",
    "val_beam_dba",
    "val_los_accuracy",
    "val_los_f1",
    "val_los_auc",
    "val_link_mae",
    "val_link_rmse",
    "val_link_r2",
    "val_selection_multitask_loss",
)


def analyze_raymobtime_modality_imbalance(
    exp_dirs: list[str] | tuple[str, ...] | None = None,
    exp_dir: str | None = None,
    exp_root: str | None = None,
    output_dir: str | None = None,
    matrix_config: str | None = None,
) -> dict[str, Any]:
    roots = _resolve_exp_dirs(exp_dirs=exp_dirs, exp_dir=exp_dir, exp_root=exp_root)
    if not roots:
        raise ValueError("Raymobtime analysis requires exp_dir, exp_dirs, or exp_root.")
    runs = [_load_run(Path(resolve_path(root))) for root in roots]
    runs.sort(key=lambda run: (run["objective"], run["modalities"], run["seed"], run["run_name"]))

    out_dir = Path(resolve_path(output_dir or Path(roots[0]) / "raymobtime_analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = _read_mapping(Path(resolve_path(matrix_config))) if matrix_config else {}
    run_matrix_rows = _run_matrix_rows(runs)
    metric_rows = _metric_rows(runs)
    single_rows = _single_modality_rows(runs)
    gate_rows = _gate_rows(runs)
    gate_los_rows = _gate_los_bucket_rows(runs)
    drop_rows = _drop_rows(runs)
    los_rows = _los_bucket_rows(runs)
    gradient_rows = _gradient_rows(runs, drop_rows)
    checkpoint_rows = _checkpoint_epoch_rows(runs)
    availability_rows = _diagnostic_availability_rows(runs, gate_rows, drop_rows, los_rows, gradient_rows)
    verdict = _diagnosis_verdict(
        runs,
        checkpoint_rows=checkpoint_rows,
        availability_rows=availability_rows,
        matrix=matrix,
    )
    s009_external_validation = _s009_external_validation_summary(verdict)

    outputs = {
        "run_matrix": str(out_dir / "run_matrix.csv"),
        "metric_comparison": str(out_dir / "metric_comparison.csv"),
        "single_modality_task_performance": str(out_dir / "single_modality_task_performance.csv"),
        "gate_mean_by_task": str(out_dir / "gate_mean_by_task.csv"),
        "gate_mean_by_task_and_los_bucket": str(out_dir / "gate_mean_by_task_and_los_bucket.csv"),
        "drop_modality_delta": str(out_dir / "drop_modality_delta.csv"),
        "beam_metrics_by_los_bucket": str(out_dir / "beam_metrics_by_los_bucket.csv"),
        "grad_norms_by_task_modality": str(out_dir / "grad_norms_by_task_modality.csv"),
        "checkpoint_epoch_summary": str(out_dir / "checkpoint_epoch_summary.csv"),
        "diagnostic_availability": str(out_dir / "diagnostic_availability.csv"),
        "summary": str(out_dir / "summary.json"),
        "report": str(out_dir / "diagnosis_report.md"),
    }
    _write_csv(Path(outputs["run_matrix"]), run_matrix_rows)
    _write_csv(Path(outputs["metric_comparison"]), metric_rows)
    _write_csv(Path(outputs["single_modality_task_performance"]), single_rows)
    _write_csv(Path(outputs["gate_mean_by_task"]), gate_rows)
    _write_csv(Path(outputs["gate_mean_by_task_and_los_bucket"]), gate_los_rows)
    _write_csv(Path(outputs["drop_modality_delta"]), drop_rows)
    _write_csv(Path(outputs["beam_metrics_by_los_bucket"]), los_rows)
    _write_csv(Path(outputs["grad_norms_by_task_modality"]), gradient_rows)
    _write_csv(Path(outputs["checkpoint_epoch_summary"]), checkpoint_rows)
    _write_csv(Path(outputs["diagnostic_availability"]), availability_rows)

    summary = {
        "runs": [_public_run(run) for run in runs],
        "outputs": outputs,
        "conditions": sorted({run["condition"] for run in runs}),
        "matrix_config": str(resolve_path(matrix_config)) if matrix_config else None,
        "verdict": verdict,
        "s009_external_validation": s009_external_validation,
        "counts": {
            "runs": len(runs),
            "multitask_runs": sum(1 for run in runs if run["objective"] == "selection_multitask"),
            "sensing_only_runs": sum(1 for run in runs if run["condition"] == "sensing-only"),
            "diagnostic_rows": {
                "gate": len(gate_rows),
                "gate_los_bucket": len(gate_los_rows),
                "drop": len([row for row in drop_rows if row.get("status") == "ok"]),
                "gradient": len(gradient_rows),
                "los_bucket": len(los_rows),
            },
        },
    }
    Path(outputs["summary"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(
        Path(outputs["report"]),
        summary,
        run_matrix_rows,
        metric_rows,
        availability_rows,
        checkpoint_rows,
        los_rows,
        gradient_rows,
    )
    return summary


def _resolve_exp_dirs(
    *,
    exp_dirs: list[str] | tuple[str, ...] | None,
    exp_dir: str | None,
    exp_root: str | None,
) -> list[str]:
    roots = [exp_dir] if exp_dir else list(exp_dirs or [])
    if exp_root:
        root = Path(resolve_path(exp_root))
        discovered = [
            str(path)
            for path in root.iterdir()
            if path.is_dir() and ((path / "metrics.json").exists() or (path / "test_report.json").exists())
        ]
        roots.extend(discovered)
    return sorted(dict.fromkeys(str(root) for root in roots))


def _load_run(path: Path) -> dict[str, Any]:
    metrics = _read_mapping(path / "metrics.json") or _read_mapping(path / "test_report.json")
    train_log = _read_mapping(path / "train_log.json")
    final_config = _read_mapping(path / "final_config.yaml")
    resolved_config = _read_mapping(path / "resolved_config.yaml")
    cfg = final_config or resolved_config
    runtime = train_log.get("runtime") or metrics.get("runtime") or cfg.get("runtime") or {}
    setup = (
        runtime.get("prediction_setup")
        or metrics.get("prediction_setup")
        or train_log.get("prediction_setup")
        or cfg.get("runtime", {}).get("prediction_setup")
        or {}
    )
    objective = (
        runtime.get("prediction_objective", {}).get("name")
        or _nested(metrics, "objective", "name")
        or setup.get("objective")
        or _nested(cfg, "experiment", "objective")
    )
    modalities = (
        runtime.get("enabled_modalities")
        or setup.get("enabled_modalities")
        or metrics.get("enabled_modalities")
        or _nested(cfg, "model", "student", "modalities")
        or _nested(cfg, "model", "modalities")
        or []
    )
    dataset_type = _nested(cfg, "data", "dataset", "type")
    task_semantics = setup.get("task_semantics") or _nested(cfg, "experiment", "task_semantics")
    if dataset_type not in {None, "raymobtime_s008"}:
        raise ValueError(f"Refusing non-Raymobtime experiment {path}: dataset type is {dataset_type!r}.")
    if objective not in _RAYMOBTIME_OBJECTIVES:
        raise ValueError(f"Refusing non-selection experiment {path}: objective is {objective!r}.")
    if task_semantics not in {None, "current_snapshot_beam_selection"}:
        raise ValueError(f"Refusing non-current snapshot experiment {path}: task_semantics={task_semantics!r}.")
    checkpoint_summary = _checkpoint_summary(path, cfg)
    weights = _selection_weights(cfg)
    condition = "sensing+ray" if "ray" in modalities else "sensing-only"
    return {
        "path": str(path),
        "run_name": path.name,
        "config_path": _nested(runtime, "cli_config_path"),
        "resolved_config": str(path / "resolved_config.yaml") if (path / "resolved_config.yaml").exists() else None,
        "final_config": str(path / "final_config.yaml") if (path / "final_config.yaml").exists() else None,
        "dataset_type": dataset_type or "raymobtime_s008",
        "objective": objective,
        "task_semantics": task_semantics or "current_snapshot_beam_selection",
        "enabled_modalities": list(modalities),
        "modalities": "+".join(str(item) for item in modalities),
        "condition": condition,
        "seed": _nested(cfg, "experiment", "seed"),
        "cache_dir": _nested(cfg, "data", "dataset", "cache_dir"),
        "portion": _nested(cfg, "data", "dataset", "portion"),
        "portion_seed": _nested(cfg, "data", "dataset", "portion_seed"),
        "image_size": _nested(cfg, "data", "dataset", "image_size"),
        "split_metadata_path": setup.get("split_metadata_path")
        or _nested(setup, "splits", "test", "split_metadata_path")
        or _nested(cfg, "runtime", "splits", "test", "raymobtime", "split_metadata_path")
        or _nested(cfg, "runtime", "split_metadata", "test", "raymobtime", "split_metadata_path")
        or _nested(cfg, "data", "dataset", "split_metadata_path"),
        "task_combo": _task_combo(weights) if objective == "selection_multitask" else str(objective),
        "loss_weights": weights,
        "checkpoint_registry": checkpoint_summary,
        "evaluations": _paired_evaluations(path),
        "metrics": metrics,
        "train_log": train_log,
    }


def _paired_evaluations(path: Path) -> dict[str, dict[str, Any]]:
    roots = []
    if path.parent.name == "experiments":
        roots.append(path.parent.parent / "evaluations")
    roots.append(path.parent)
    evaluations: dict[str, dict[str, Any]] = {}
    for root in dict.fromkeys(roots):
        for suffix in _PAIRED_EVALUATION_SUFFIXES:
            eval_dir = root / f"{path.name}_{suffix}"
            metrics = _read_mapping(eval_dir / "metrics.json") or _read_mapping(eval_dir / "test_report.json")
            if not metrics:
                continue
            evaluations[suffix] = {
                "run_dir": str(eval_dir),
                "run_name": eval_dir.name,
                "metrics": metrics,
                "test_report": _read_mapping(eval_dir / "test_report.json"),
                "run_status": _read_mapping(eval_dir / "run_status.json"),
            }
    return evaluations


def _run_matrix_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        rows.append(
            {
                "run_dir": run["path"],
                "run_name": run["run_name"],
                "config": run["config_path"],
                "resolved_config": run["resolved_config"],
                "objective": run["objective"],
                "task_combo": run["task_combo"],
                "modalities": run["modalities"],
                "condition": run["condition"],
                "seed": run["seed"],
                "cache_dir": run["cache_dir"],
                "portion": run["portion"],
                "portion_seed": run["portion_seed"],
                "image_size": json.dumps(run["image_size"], ensure_ascii=False),
                "split_metadata_path": run["split_metadata_path"],
                "checkpoint_best": run["checkpoint_registry"]["best_available"],
                "checkpoint_best_top1": run["checkpoint_registry"]["best_top1_available"],
            }
        )
    return rows


def _metric_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        metrics = run["metrics"]
        row = {
            "run_dir": run["path"],
            "run_name": run["run_name"],
            "objective": run["objective"],
            "task_combo": run["task_combo"],
            "modalities": run["modalities"],
            "condition": run["condition"],
            "seed": run["seed"],
            "trained_heads": _trained_heads(run),
            "diagnostic_only_heads": _diagnostic_only_heads(run),
        }
        for metric in _COMPARISON_METRICS:
            row[metric] = _metric_value_or_blank(metrics, metric, metric.removeprefix("val_"))
        rows.append(row)
    return rows


def _single_modality_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        metrics = run["metrics"]
        rows.append(
            {
                "run_dir": run["path"],
                "modalities": run["modalities"],
                "condition": run["condition"],
                "objective": run["objective"],
                "seed": run["seed"],
                "beam_top1": _metric_value_or_blank(metrics, "val_beam_top1", "beam_top1"),
                "beam_top3": _metric_value_or_blank(metrics, "val_beam_top3", "beam_top3"),
                "beam_top5": _metric_value_or_blank(metrics, "val_beam_top5", "beam_top5"),
                "beam_dba_current": _metric_value_or_blank(metrics, "val_beam_dba", "beam_dba_current"),
                "los_f1": _metric_value_or_blank(metrics, "val_los_f1", "los_f1"),
                "link_mae": _metric_value_or_blank(metrics, "val_link_mae", "link_mae"),
            }
        )
    return rows


def _gate_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for key, value in _latest_epoch_log(run).items():
            if not (key.startswith("gate/") or key.startswith("raymobtime/gate/")):
                continue
            parts = key.split("/")
            if len(parts) < 4:
                continue
            rows.append(
                {
                    "run_dir": run["path"],
                    "run_name": run["run_name"],
                    "task": parts[-2],
                    "modality": parts[-1],
                    "sample_count": "",
                    "gate_mean": value,
                    "gate_std": "",
                    "evidence_source": "train_log_epoch_mean",
                    "note": "std unavailable because training currently logs epoch mean scalars only",
                }
            )
    return rows


def _gate_los_bucket_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        buckets = _nested(run["metrics"], "gate_los_buckets")
        if not isinstance(buckets, dict):
            buckets = {}
        for bucket, task_values in buckets.items():
            if not isinstance(task_values, dict):
                continue
            for task, modality_values in task_values.items():
                if not isinstance(modality_values, dict):
                    continue
                for modality, stats in modality_values.items():
                    stats = stats if isinstance(stats, dict) else {"gate_mean": stats}
                    rows.append(
                        {
                            "run_dir": run["path"],
                            "run_name": run["run_name"],
                            "los_bucket": bucket,
                            "task": task,
                            "modality": modality,
                            "sample_count": stats.get("sample_count", ""),
                            "gate_mean": _metric_value_or_blank(stats, "gate_mean", "mean"),
                            "gate_std": _metric_value_or_blank(stats, "gate_std", "std"),
                            "evidence_source": "metrics.gate_los_buckets",
                            "note": "",
                        }
                    )
    return rows


def _drop_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        enabled = set(run["enabled_modalities"])
        if len(enabled.intersection(_SENSING_MODALITIES)) < 2:
            continue
        subsets = run["metrics"].get("modality_subsets", {})
        if not isinstance(subsets, dict):
            subsets = {}
        full = subsets.get("all", run["metrics"])
        full_top1 = _metric_float(full, "val_beam_top1", "beam_top1")
        full_dba = _metric_float(full, "val_beam_dba", "beam_dba_current")
        full_los = _metric_float(full, "val_los_f1", "los_f1")
        full_link = _metric_float(full, "val_link_mae", "link_mae")
        for modality in _SENSING_MODALITIES:
            base = {
                "run_dir": run["path"],
                "run_name": run["run_name"],
                "drop_modality": modality,
            }
            if modality not in enabled:
                rows.append({**base, "status": "unavailable", "reason": "modality_not_enabled"})
                continue
            name = f"drop_{modality}"
            metrics = subsets.get(name)
            if not isinstance(metrics, dict):
                rows.append({**base, "status": "unavailable", "reason": "modality_subset_metrics_missing"})
                continue
            rows.append(
                {
                    **base,
                    "status": "ok",
                    "beam_top1": _metric_value_or_blank(metrics, "val_beam_top1", "beam_top1"),
                    "delta_beam_top1": _delta(_metric_float(metrics, "val_beam_top1", "beam_top1"), full_top1),
                    "beam_dba_current": _metric_value_or_blank(metrics, "val_beam_dba", "beam_dba_current"),
                    "delta_beam_dba_current": _delta(_metric_float(metrics, "val_beam_dba", "beam_dba_current"), full_dba),
                    "los_f1": _metric_value_or_blank(metrics, "val_los_f1", "los_f1"),
                    "delta_los_f1": _delta(_metric_float(metrics, "val_los_f1", "los_f1"), full_los),
                    "link_mae": _metric_value_or_blank(metrics, "val_link_mae", "link_mae"),
                    "delta_link_mae": _delta(_metric_float(metrics, "val_link_mae", "link_mae"), full_link),
                    "reason": "",
                }
            )
    return rows


def _los_bucket_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        _append_los_bucket_rows(rows, run, run["metrics"], evidence_source="run_metrics", checkpoint_view="")
        for view, evaluation in run.get("evaluations", {}).items():
            _append_los_bucket_rows(
                rows,
                run,
                evaluation.get("metrics", {}),
                evidence_source="paired_evaluation",
                checkpoint_view=view,
                evaluation_run_dir=evaluation.get("run_dir", ""),
            )
    return rows


def _append_los_bucket_rows(
    rows: list[dict[str, Any]],
    run: dict[str, Any],
    metrics_source: dict[str, Any],
    *,
    evidence_source: str,
    checkpoint_view: str,
    evaluation_run_dir: str = "",
) -> None:
    buckets = metrics_source.get("los_buckets", {})
    if not isinstance(buckets, dict):
        return
    for bucket, metrics in buckets.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "run_dir": run["path"],
                "run_name": run["run_name"],
                "evaluation_run_dir": evaluation_run_dir,
                "checkpoint_view": checkpoint_view,
                "evidence_source": evidence_source,
                "los_bucket": bucket,
                "los_label": _metric_value_or_blank(metrics, "los_label"),
                "beam_top1": _metric_value_or_blank(metrics, "val_beam_top1", "beam_top1"),
                "beam_top3": _metric_value_or_blank(metrics, "val_beam_top3", "beam_top3"),
                "beam_top5": _metric_value_or_blank(metrics, "val_beam_top5", "beam_top5"),
                "beam_dba_current": _metric_value_or_blank(metrics, "val_beam_dba", "beam_dba_current"),
                "sample_count": _metric_value_or_blank(metrics, "sample_count", "total"),
            }
        )

def _gradient_rows(runs: list[dict[str, Any]], drop_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        latest = _latest_epoch_log(run)
        for key, value in latest.items():
            if "grad" not in key or not any(modality in key for modality in (*_SENSING_MODALITIES, "ray")):
                continue
            parsed = _parse_task_modality_from_metric(key)
            rows.append(
                {
                    "run_dir": run["path"],
                    "run_name": run["run_name"],
                    "diagnostic_type": "gradient_norm",
                    "metric": key,
                    "task": parsed.get("task", ""),
                    "modality": parsed.get("modality", ""),
                    "value": value,
                    "contribution_score": "",
                    "evidence_source": "train_log",
                    "calculation": "logged gradient scalar",
                }
            )
    rows.extend(_contribution_rows_from_drop(runs, drop_rows))
    return rows


def _contribution_rows_from_drop(runs: list[dict[str, Any]], drop_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs_by_path = {run["path"]: run for run in runs}
    metric_specs = (
        ("beam_selection", "beam_top1", "delta_beam_top1", "higher_is_better"),
        ("beam_selection", "beam_dba_current", "delta_beam_dba_current", "higher_is_better"),
        ("los", "los_f1", "delta_los_f1", "higher_is_better"),
        ("link_quality", "link_mae", "delta_link_mae", "lower_is_better"),
    )
    rows = []
    for row in drop_rows:
        if row.get("status") != "ok":
            continue
        run = runs_by_path.get(str(row.get("run_dir", "")), {})
        trained_heads = set(str(run.get("task_combo", "")).split("+")) if run else set()
        for task, metric, delta_key, direction in metric_specs:
            delta = row.get(delta_key)
            if not _finite_number(delta):
                continue
            delta = float(delta)
            contribution = -delta if direction == "higher_is_better" else delta
            rows.append(
                {
                    "run_dir": row.get("run_dir", ""),
                    "run_name": row.get("run_name", ""),
                    "diagnostic_type": "contribution_from_modality_drop",
                    "metric": metric,
                    "task": task,
                    "modality": row.get("drop_modality", ""),
                    "value": "",
                    "contribution_score": contribution,
                    "full_value": _full_value_from_delta(row.get(metric), delta),
                    "drop_value": row.get(metric, ""),
                    "trained_task": _task_was_trained(task, trained_heads),
                    "evidence_source": "test_time_modality_drop_delta",
                    "calculation": "higher-is-better: full - drop; lower-is-better: drop - full",
                }
            )
    return rows


def _full_value_from_delta(drop_value: Any, delta: float) -> Any:
    if not _finite_number(drop_value):
        return ""
    return float(drop_value) - delta


def _task_was_trained(task: str, trained_heads: set[str]) -> bool:
    aliases = {"beam_selection": "beam", "los": "los", "link_quality": "link"}
    return aliases.get(task, task) in trained_heads


def _checkpoint_epoch_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        if run["objective"] != "selection_multitask":
            continue
        epoch_logs = run.get("train_log", {}).get("epoch_logs", [])
        if not isinstance(epoch_logs, list):
            epoch_logs = []
        for metric, mode in _BEST_METRICS:
            best = _best_epoch(epoch_logs, metric, mode)
            view = _REEVALUATION_VIEWS.get(metric, "")
            reevaluation = run.get("evaluations", {}).get(view, {}) if view else {}
            reevaluation_metrics = reevaluation.get("metrics", {}) if isinstance(reevaluation, dict) else {}
            reevaluation_status = "available" if reevaluation_metrics else ("not_found" if view else "not_run")
            rows.append(
                {
                    "run_dir": run["path"],
                    "run_name": run["run_name"],
                    "metric": metric,
                    "mode": mode,
                    "epoch": best.get("epoch", ""),
                    "value": best.get("value", ""),
                    "val_beam_top1": best.get("row", {}).get("val_beam_top1", ""),
                    "val_beam_dba": best.get("row", {}).get("val_beam_dba", ""),
                    "val_los_f1": best.get("row", {}).get("val_los_f1", ""),
                    "val_link_mae": best.get("row", {}).get("val_link_mae", ""),
                    "checkpoint_available": _checkpoint_available_for_metric(run, metric, best.get("epoch")),
                    "reevaluation_status": reevaluation_status,
                    "reevaluation_view": view,
                    "reevaluation_run_dir": reevaluation.get("run_dir", "") if isinstance(reevaluation, dict) else "",
                    "reeval_val_beam_top1": _metric_value_or_blank(reevaluation_metrics, "val_beam_top1", "beam_top1"),
                    "reeval_val_beam_dba": _metric_value_or_blank(
                        reevaluation_metrics,
                        "val_beam_dba",
                        "beam_dba_current",
                    ),
                    "reeval_val_los_f1": _metric_value_or_blank(reevaluation_metrics, "val_los_f1", "los_f1"),
                    "reeval_val_link_mae": _metric_value_or_blank(reevaluation_metrics, "val_link_mae", "link_mae"),
                    "reeval_val_selection_multitask_loss": _metric_value_or_blank(
                        reevaluation_metrics,
                        "val_selection_multitask_loss",
                        "selection_multitask_loss",
                    ),
                    "reevaluation_note": _reevaluation_note(reevaluation_status, view),
                }
            )
    return rows


def _diagnostic_availability_rows(
    runs: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    drop_rows: list[dict[str, Any]],
    los_rows: list[dict[str, Any]],
    gradient_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    gate_runs = {row["run_dir"] for row in gate_rows}
    drop_ok_runs = {row["run_dir"] for row in drop_rows if row.get("status") == "ok"}
    los_runs = {row["run_dir"] for row in los_rows}
    gradient_runs = {row["run_dir"] for row in gradient_rows}
    rows = []
    for run in runs:
        rows.extend(
            [
                _availability(
                    run,
                    "gate",
                    run["path"] in gate_runs,
                    "model diagnostics not exposed or run is not task-aware gated",
                    applicable=run["objective"] == "selection_multitask",
                ),
                _availability(
                    run,
                    "drop_modality",
                    run["path"] in drop_ok_runs,
                    "modality_subsets evaluation not generated",
                    applicable=len(set(run["enabled_modalities"]).intersection(_SENSING_MODALITIES)) >= 2,
                ),
                _availability(
                    run,
                    "gradient",
                    run["path"] in gradient_runs,
                    "gradient/contribution diagnostics not logged by training loop",
                    applicable=run["objective"] == "selection_multitask",
                ),
                _availability(run, "los_bucket", run["path"] in los_runs, "LOS bucket metrics not generated by evaluation pass"),
            ]
        )
    return rows


def _availability(
    run: dict[str, Any],
    diagnostic: str,
    available: bool,
    missing_reason: str,
    *,
    applicable: bool = True,
) -> dict[str, Any]:
    status = "available" if available else "missing"
    if not applicable:
        status = "not_applicable"
    return {
        "run_dir": run["path"],
        "run_name": run["run_name"],
        "objective": run["objective"],
        "modalities": run["modalities"],
        "diagnostic": diagnostic,
        "status": status,
        "missing_reason": "" if status != "missing" else missing_reason,
    }


def _diagnosis_verdict(
    runs: list[dict[str, Any]],
    *,
    checkpoint_rows: list[dict[str, Any]],
    availability_rows: list[dict[str, Any]],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    multitask_runs = [run for run in runs if run["objective"] == "selection_multitask"]
    seeds = sorted({run["seed"] for run in multitask_runs if run["seed"] is not None})
    blocked_categories = []
    for diagnostic in ("drop_modality", "gradient", "los_bucket"):
        applicable = [row for row in availability_rows if row["diagnostic"] == diagnostic and row["status"] != "not_applicable"]
        if applicable and not any(row["status"] == "available" for row in applicable):
            blocked_categories.append(diagnostic)
    missing_matrix = []
    if len(seeds) < 3:
        missing_matrix.append("multitask multi-seed coverage is incomplete")
    if not _has_task_combo_or_loss_weight_matrix(runs, matrix):
        missing_matrix.append("task-combo/loss-weight matrix is incomplete")
    if any(row.get("value", "") == "" for row in checkpoint_rows):
        missing_matrix.append("best-by-metric epoch summary is incomplete for at least one multitask metric")
    recovery = _beam_recovery_evidence(runs)
    if blocked_categories or missing_matrix:
        conclusion = "diagnostics_blocked"
        reason = "; ".join(missing_matrix) if missing_matrix else "diagnostic evidence is incomplete"
    elif recovery["recovered"]:
        conclusion = "likely_parameter_issue"
        reason = recovery["reason"]
    else:
        conclusion = "inconclusive"
        reason = "diagnostics are available but recovery evidence is insufficient for a parameter-issue or confirmed-imbalance verdict"
    return {
        "conclusion": conclusion,
        "reason": reason,
        "multitask_seeds": seeds,
        "blocked_diagnostics": sorted(blocked_categories),
        "beam_recovery": recovery,
    }


def _beam_recovery_evidence(runs: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = _best_single_task_beam(runs)
    multitask_best = _best_multitask_beam_candidate(runs)
    recovered = (
        _finite_number(baseline.get("value"))
        and _finite_number(multitask_best.get("value"))
        and float(multitask_best["value"]) >= float(baseline["value"]) - 0.01
    )
    if recovered:
        reason = (
            "best-by-beam checkpoint or task/loss-weight消融可将 beam 指标恢复到 "
            "beam 单任务 CIL/最佳单模态附近"
        )
    else:
        reason = "beam recovery evidence is not sufficient"
    return {
        "recovered": bool(recovered),
        "baseline": baseline,
        "best_multitask_candidate": multitask_best,
        "reason": reason,
    }


def _best_single_task_beam(runs: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {"value": ""}
    for run in runs:
        if run["objective"] != "current_beam_selection":
            continue
        value = _metric_float(run["metrics"], "val_beam_top1", "beam_top1")
        if value is None:
            continue
        if not _finite_number(best.get("value")) or value > float(best["value"]):
            best = {"run_name": run["run_name"], "modalities": run["modalities"], "value": value}
    return best


def _best_multitask_beam_candidate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {"value": ""}
    for run in runs:
        if run["objective"] != "selection_multitask":
            continue
        candidates = [("run_metrics", run["path"], run["metrics"])]
        candidates.extend(
            (
                f"paired_evaluation:{view}",
                evaluation.get("run_dir", ""),
                evaluation.get("metrics", {}),
            )
            for view, evaluation in run.get("evaluations", {}).items()
        )
        for source, source_dir, metrics in candidates:
            value = _metric_float(metrics, "val_beam_top1", "beam_top1")
            if value is None:
                continue
            if not _finite_number(best.get("value")) or value > float(best["value"]):
                best = {
                    "run_name": run["run_name"],
                    "task_combo": run["task_combo"],
                    "seed": run["seed"],
                    "value": value,
                    "source": source,
                    "source_dir": source_dir,
                }
    return best


def _write_report(
    path: Path,
    summary: dict[str, Any],
    run_matrix_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    availability_rows: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
    los_rows: list[dict[str, Any]],
    gradient_rows: list[dict[str, Any]],
) -> None:
    verdict = summary["verdict"]
    lines = [
        "# Raymobtime s008 Modality Imbalance Diagnosis",
        "",
        f"Conclusion: `{verdict['conclusion']}`",
        "",
        "## Scope",
        "",
        f"- Runs analyzed: {len(run_matrix_rows)}",
        f"- Conditions: {', '.join(summary['conditions'])}",
        "- Training logs, checkpoints, cache files, TensorBoard files, and generated reports remain in ignored output directories.",
        "",
        "## Counter-Evidence Checks",
        "",
        f"- Early stopping / checkpoint selection: {_checkpoint_report_note(checkpoint_rows)}",
        f"- Loss weights: {_beam_recovery_report_note(verdict)}",
        f"- Task-combo conflict localization: {_task_combo_report_note(metric_rows, checkpoint_rows)}",
        f"- Internal diagnostics: {_availability_report_note(availability_rows)}",
        f"- Contribution evidence: {_contribution_report_note(gradient_rows)}",
        f"- LOS bucket evidence: {_los_bucket_report_note(los_rows)}",
        "",
        "## Key Metrics",
        "",
        _markdown_table(
            metric_rows,
            [
                "run_name",
                "objective",
                "task_combo",
                "modalities",
                "seed",
                "val_beam_top1",
                "val_los_f1",
                "val_link_mae",
                "val_selection_multitask_loss",
            ],
            limit=24,
        ),
        "",
        "## Checkpoint Re-evaluation",
        "",
        _markdown_table(
            checkpoint_rows,
            [
                "run_name",
                "metric",
                "epoch",
                "value",
                "reevaluation_status",
                "reevaluation_view",
                "reeval_val_beam_top1",
                "reeval_val_los_f1",
                "reeval_val_link_mae",
            ],
            limit=36,
        ),
        "",
        "## LOS Bucket Beam Metrics",
        "",
        _markdown_table(
            _preferred_los_bucket_rows(los_rows),
            [
                "run_name",
                "checkpoint_view",
                "los_bucket",
                "sample_count",
                "beam_top1",
                "beam_top3",
                "beam_top5",
                "beam_dba_current",
            ],
            limit=36,
        ),
        "",
        "## Contribution Diagnostics",
        "",
        _markdown_table(
            _preferred_contribution_rows(gradient_rows),
            [
                "run_name",
                "task",
                "modality",
                "metric",
                "contribution_score",
                "full_value",
                "drop_value",
                "evidence_source",
            ],
            limit=36,
        ),
        "",
        "## s009 External Validation Appendix",
        "",
        f"- Decision: {summary['s009_external_validation']['decision']}",
        f"- Reason: {summary['s009_external_validation']['reason']}",
        "- Role: s009 is only a second-stage cross-scene validation target and is not used as s008 evidence.",
        f"- Minimum matrix: {summary['s009_external_validation']['minimum_matrix']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _s009_external_validation_summary(verdict: dict[str, Any]) -> dict[str, str]:
    conclusion = str(verdict.get("conclusion", ""))
    minimum_matrix = (
        "lidar and coord+image+lidar beam/LOS/link single-task runs; original multitask; "
        "beam-heavy multitask; most suspicious task-combo"
    )
    if conclusion == "confirmed_imbalance":
        return {
            "decision": "eligible_after_contract_check",
            "reason": "s008 reached confirmed_imbalance, so s009 may be used for cross-scene validation only.",
            "minimum_matrix": minimum_matrix,
        }
    if conclusion == "inconclusive" and not verdict.get("blocked_diagnostics"):
        return {
            "decision": "eligible_if_high_confidence_gap_review_passes",
            "reason": "s008 is inconclusive without blocked diagnostics; review remaining gaps before starting s009.",
            "minimum_matrix": minimum_matrix,
        }
    return {
        "decision": "not_started",
        "reason": (
            f"s008 conclusion is {conclusion}; per the change gate, s009 must not be used while parameter, "
            "checkpoint, or diagnostic gaps still explain the s008 behavior."
        ),
        "minimum_matrix": minimum_matrix,
    }


def _public_run(run: dict[str, Any]) -> dict[str, Any]:
    public = {
        key: value
        for key, value in run.items()
        if key not in {"metrics", "train_log", "evaluations"}
    }
    public["evaluations"] = {
        view: {"run_dir": evaluation.get("run_dir"), "run_name": evaluation.get("run_name")}
        for view, evaluation in run.get("evaluations", {}).items()
    }
    return public


def _checkpoint_report_note(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "no multitask run was available for epoch-level comparison."
    available = sum(1 for row in rows if row.get("value", "") != "")
    reevaluated = sum(1 for row in rows if row.get("reevaluation_status") == "available")
    return (
        f"{available}/{len(rows)} best-by-metric epoch rows were resolved from train_log; "
        f"{reevaluated}/{len(rows)} paired checkpoint re-evaluations were loaded."
    )


def _beam_recovery_report_note(verdict: dict[str, Any]) -> str:
    recovery = verdict.get("beam_recovery", {}) if isinstance(verdict, dict) else {}
    baseline = recovery.get("baseline", {}) if isinstance(recovery, dict) else {}
    candidate = recovery.get("best_multitask_candidate", {}) if isinstance(recovery, dict) else {}
    if recovery.get("recovered"):
        return (
            "beam recovered near or above the best single-task beam baseline "
            f"({candidate.get('run_name', '')} {candidate.get('value', '')} via {candidate.get('source', '')}; "
            f"baseline {baseline.get('run_name', '')} {baseline.get('value', '')})."
        )
    return "beam recovery was not demonstrated by current paired evaluations."


def _task_combo_report_note(metric_rows: list[dict[str, Any]], checkpoint_rows: list[dict[str, Any]]) -> str:
    combo_values = {}
    for row in metric_rows:
        if row.get("objective") == "selection_multitask" and row.get("seed") == 42:
            value = row.get("val_beam_top1")
            if _finite_number(value):
                combo_values[str(row.get("task_combo"))] = float(value)
    for row in checkpoint_rows:
        if row.get("metric") != "val_beam_top1" or not _finite_number(row.get("reeval_val_beam_top1")):
            continue
        combo = _task_combo_from_run_name(str(row.get("run_name", "")))
        combo_values[f"{combo}:best_top1"] = float(row["reeval_val_beam_top1"])
    required = {"beam", "beam+los", "beam+link", "beam+los+link"}
    if not required.issubset({key.split(":")[0] for key in combo_values}):
        return "incomplete until beam-only, beam+los, beam+link, and beam+los+link are all available."
    best_key = max(combo_values, key=lambda key: combo_values[key])
    return f"all core task-combos are available; best beam view is {best_key}={combo_values[best_key]:.4f}."


def _task_combo_from_run_name(run_name: str) -> str:
    if "beam_only" in run_name:
        return "beam"
    if "beam_los" in run_name:
        return "beam+los"
    if "beam_link" in run_name:
        return "beam+link"
    return "beam+los+link"


def _availability_report_note(rows: list[dict[str, Any]]) -> str:
    blocked = []
    partial = []
    for diagnostic in sorted({row["diagnostic"] for row in rows}):
        applicable = [row for row in rows if row["diagnostic"] == diagnostic and row["status"] != "not_applicable"]
        if not applicable:
            continue
        available = [row for row in applicable if row["status"] == "available"]
        missing = [row for row in applicable if row["status"] == "missing"]
        if not available:
            blocked.append(diagnostic)
        elif missing:
            partial.append(f"{diagnostic} partial ({len(available)}/{len(applicable)} runs)")
    if not blocked and not partial:
        return "all requested diagnostic categories have rows."
    notes = []
    if blocked:
        notes.append("blocked categories: " + ", ".join(blocked))
    if partial:
        notes.append("; ".join(partial))
    return "; ".join(notes) + "."


def _contribution_report_note(rows: list[dict[str, Any]]) -> str:
    contribution = [row for row in rows if row.get("diagnostic_type") == "contribution_from_modality_drop"]
    if not contribution:
        return "no gradient or contribution rows were available."
    by_modality: dict[str, list[float]] = {}
    for row in contribution:
        if row.get("task") != "beam_selection" or row.get("metric") != "beam_top1":
            continue
        if _finite_number(row.get("contribution_score")):
            by_modality.setdefault(str(row.get("modality")), []).append(float(row["contribution_score"]))
    if not by_modality:
        return f"{len(contribution)} contribution rows were generated from modality-drop diagnostics."
    means = {
        modality: sum(values) / len(values)
        for modality, values in by_modality.items()
        if values
    }
    top = max(means, key=means.get)
    formatted = ", ".join(f"{name}={value:.4f}" for name, value in sorted(means.items()))
    return f"{len(contribution)} contribution rows generated; beam_top1 mean contribution by dropped modality: {formatted}; strongest={top}."


def _los_bucket_report_note(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "no LOS bucket beam metrics were available."
    views = sorted({str(row.get("checkpoint_view", "run_metrics") or "run_metrics") for row in rows})
    return f"{len(rows)} LOS bucket rows available across views: {', '.join(views)}."


def _preferred_los_bucket_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preferred = [row for row in rows if row.get("checkpoint_view") == "best_top1"]
    return preferred or rows


def _preferred_contribution_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contribution = [row for row in rows if row.get("diagnostic_type") == "contribution_from_modality_drop"]
    beam_rows = [row for row in contribution if row.get("task") == "beam_selection"]
    return beam_rows or contribution or rows


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], *, limit: int) -> str:
    visible = rows[:limit]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in visible:
        body.append("| " + " | ".join(_markdown_cell(row.get(column, "")) for column in columns) + " |")
    if len(rows) > limit:
        omitted = ["..."] + [f"{len(rows) - limit} more rows omitted"] + [""] * max(len(columns) - 2, 0)
        body.append("| " + " | ".join(omitted[: len(columns)]) + " |")
    return "\n".join([header, sep, *body])


def _markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|")


def _latest_epoch_log(run: dict[str, Any]) -> dict[str, Any]:
    logs = run.get("train_log", {}).get("epoch_logs", [])
    if isinstance(logs, list) and logs:
        latest = logs[-1]
        return latest if isinstance(latest, dict) else {}
    return {}


def _best_epoch(epoch_logs: list[Any], metric: str, mode: str) -> dict[str, Any]:
    best_row = None
    best_value = None
    for index, raw_row in enumerate(epoch_logs):
        if not isinstance(raw_row, dict):
            continue
        value = raw_row.get(metric)
        if not _finite_number(value):
            continue
        value = float(value)
        if best_value is None or (mode == "min" and value < best_value) or (mode == "max" and value > best_value):
            best_value = value
            best_row = raw_row
            best_index = index
    if best_row is None:
        return {}
    epoch = best_row.get("epoch", best_index + 1)
    return {"epoch": epoch, "value": best_value, "row": best_row}


def _checkpoint_summary(path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    checkpoint_dir = path / "checkpoints"
    registry = _nested(cfg, "runtime", "checkpoint_registry") or {}
    return {
        "best_available": (checkpoint_dir / "best.pth").exists(),
        "best_top1_available": (checkpoint_dir / "best_top1.pth").exists(),
        "last_available": (checkpoint_dir / "last.pth").exists(),
        "registry_available": bool(registry),
        "selected_epoch": _nested(registry, "selected_epoch"),
        "selection_metric": _nested(registry, "selection_metric"),
    }


def _checkpoint_available_for_metric(run: dict[str, Any], metric: str, epoch: Any) -> bool:
    registry = run["checkpoint_registry"]
    if metric == "val_selection_multitask_loss":
        return bool(registry["best_available"])
    if metric == "val_beam_top1":
        return bool(registry["best_top1_available"])
    return bool(registry["selected_epoch"] == epoch and registry["best_available"])


def _reevaluation_note(status: str, view: str) -> str:
    if status == "available":
        return f"paired evaluation metrics loaded from evaluations/*_{view}"
    if status == "not_found" and view:
        return f"paired evaluations/*_{view} metrics not found; using train_log epoch metrics only"
    return "analysis summarizes train_log epoch metrics"


def _selection_weights(cfg: dict[str, Any]) -> dict[str, float]:
    weights = _nested(cfg, "loss", "objective", "weights") or {}
    if not isinstance(weights, dict):
        weights = {}
    return {
        "beam_selection": float(weights.get("beam_selection", weights.get("beam", 1.0))),
        "los": float(weights.get("los", weights.get("los_weight", 0.5))),
        "link_quality": float(weights.get("link_quality", weights.get("link", 0.2))),
    }


def _task_combo(weights: dict[str, float]) -> str:
    enabled = []
    if weights.get("beam_selection", 0.0) > 0.0:
        enabled.append("beam")
    if weights.get("los", 0.0) > 0.0:
        enabled.append("los")
    if weights.get("link_quality", 0.0) > 0.0:
        enabled.append("link")
    return "+".join(enabled) if enabled else "none"


def _trained_heads(run: dict[str, Any]) -> str:
    if run["objective"] != "selection_multitask":
        return run["objective"]
    return run["task_combo"]


def _diagnostic_only_heads(run: dict[str, Any]) -> str:
    if run["objective"] != "selection_multitask":
        return ""
    trained = set(run["task_combo"].split("+"))
    heads = {"beam", "los", "link"}
    return "+".join(sorted(heads - trained))


def _has_task_combo_or_loss_weight_matrix(runs: list[dict[str, Any]], matrix: dict[str, Any]) -> bool:
    combos = {run["task_combo"] for run in runs if run["objective"] == "selection_multitask"}
    if {"beam", "beam+los", "beam+link", "beam+los+link"}.issubset(combos):
        return True
    planned = matrix.get("runs", []) if isinstance(matrix, dict) else []
    planned_names = {str(item.get("name", "")) for item in planned if isinstance(item, dict)}
    return {"beam_only_multitask_model", "beam_los", "beam_link", "beam_los_link_original"}.issubset(planned_names)


def _parse_task_modality_from_metric(metric: str) -> dict[str, str]:
    parts = metric.split("/")
    task = ""
    modality = ""
    for part in parts:
        if part in {"beam", "beam_selection", "los", "link", "link_quality"}:
            task = part
        if part in {*_SENSING_MODALITIES, "ray"}:
            modality = part
    return {"task": task, "modality": modality}


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _metric_float(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if _finite_number(value):
            return float(value)
    return None


def _metric_value_or_blank(metrics: dict[str, Any], *keys: str) -> Any:
    value = _metric_float(metrics, *keys)
    return "" if value is None else value


def _delta(value: float | None, baseline: float | None) -> Any:
    if value is None or baseline is None:
        return ""
    return value - baseline


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml

            loaded = yaml.safe_load(text) or {}
        except Exception:
            return {}
    else:
        loaded = json.loads(text)
    return loaded if isinstance(loaded, dict) else {}


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
