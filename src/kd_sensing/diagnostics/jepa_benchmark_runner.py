from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from kd_sensing.config.io import load_config
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.data.difficulty import (
    DifficultyContext,
    apply_difficulty_pipeline,
    normalize_difficulty_profiles,
)
from kd_sensing.data.difficulty.presets import (
    PREDICTIVE_JEPA_CANONICAL_CONDITIONS,
    PREDICTIVE_JEPA_CONDITION_IDS,
    PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
    SCENARIO_D_CANONICAL_CONDITIONS,
    SCENARIO_D_CONDITION_IDS,
    SCENARIO_D_SUITE_TYPE,
    normalize_predictive_jepa_condition_id,
    normalize_predictive_jepa_operator_params,
    normalize_scenario_d_condition_id,
    normalize_scenario_d_operator_params,
    predictive_jepa_condition,
    scenario_d_condition,
)
from kd_sensing.evaluation.metrics import calculate_dba_score, calculate_topk_accuracy
from kd_sensing.utils.artifact_registry import load_checkpoint_metadata
from kd_sensing.utils.paths import resolve_path


from kd_sensing.diagnostics.jepa_benchmark_artifacts import (
    OutputRegistry,
    _git_status_short,
    _load_mapping_text,
    _prepare_output_dir,
    _read_csv,
    _resolve_artifact_path,
    _resolve_existing_user_path,
    _resolve_output_dir,
    _write_csv,
)
from kd_sensing.diagnostics.jepa_benchmark_common import *
from kd_sensing.diagnostics.jepa_benchmark_manifest import (
    evaluate_model_comparability,
    load_benchmark_manifest,
    _manifest_has_predictive_jepa,
    _manifest_has_scenario_d,
)
from kd_sensing.diagnostics.jepa_benchmark_perturbations import _benchmark_difficulty_provenance
from kd_sensing.diagnostics.jepa_benchmark_plots import _write_benchmark_figures, _write_scenario_d_figures
from kd_sensing.diagnostics.jepa_benchmark_predictive import (
    _predictive_jepa_metric_row,
    aggregate_predictive_robustness_summary,
)
from kd_sensing.diagnostics.jepa_benchmark_scenario_c import (
    _add_scenario_c_accuracy_ratios,
    _scenario_c_condition_for_severity,
    _scenario_c_metric_columns,
)
from kd_sensing.diagnostics.jepa_benchmark_scenario_d import (
    _scenario_cxd_metric_row,
    _scenario_d_condition_for_severity,
    _scenario_d_metric_columns,
    aggregate_scenario_d_matrix,
    scenario_d_heatmap,
    write_cxd_phase_artifacts,
)


def run_jepa_gps_shortcut_benchmark(
    *,
    manifest_path: str | Path,
    output_dir: str | Path | None = None,
    force: bool = False,
    dry_run: bool = False,
    command: list[str] | None = None,
) -> dict[str, Any]:
    manifest = load_benchmark_manifest(manifest_path, validate_paths=not dry_run)
    out = _resolve_output_dir(output_dir or manifest.get("outputs", {}).get("output_dir") or DEFAULT_OUTPUT_DIR)
    _prepare_output_dir(out, force=force)
    tables_dir = out / "tables"
    figures_dir = out / "figures"
    results_dir = out / "results"
    plots_dir = out / "plots"
    cache_dir = out / "cache"
    for directory in (tables_dir, figures_dir, results_dir, plots_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)
    registry = OutputRegistry(out)
    warnings: list[dict[str, Any]] = []

    comparability = evaluate_model_comparability(manifest)
    if comparability["status"] == "failed" and str(manifest.get("comparability", {}).get("mode")) == "strict":
        fields = ", ".join(str(item["field"]) for item in comparability["inconsistent_fields"])
        raise BenchmarkManifestError(f"Benchmark models are not comparable under strict mode: {fields}")
    if comparability["status"] != "passed":
        warnings.append(
            WarningRecord(
                code="comparability_marked_unavailable",
                message="One or more declared comparability fields differ across models.",
            ).to_dict()
        )

    model_summaries: dict[str, dict[str, Any]] = {}
    metrics_rows: list[dict[str, Any]] = []
    for model_name, model_spec in manifest["models"].items():
        source = _model_metric_source(
            model_name,
            model_spec,
            manifest,
            output_dir=out,
            dry_run=dry_run,
            warnings=warnings,
        )
        model_summaries[model_name] = source
        metrics_rows.extend(
            _metrics_rows_for_model(
                model_name,
                model_spec,
                source,
                manifest,
                comparability_status=comparability["status"],
                dry_run=dry_run,
            )
        )

    robustness_rows = aggregate_robustness_summary(metrics_rows, primary_metric=str(manifest["metrics"]["primary"]))
    shortcut_rows = aggregate_shortcut_reliance(metrics_rows, robustness_rows, manifest)
    _write_csv(tables_dir / "metrics_by_condition.csv", metrics_rows)
    _write_csv(tables_dir / "robustness_summary.csv", robustness_rows)
    _write_csv(tables_dir / "shortcut_reliance_summary.csv", shortcut_rows)
    predictive_condition_path: Path | None = None
    predictive_summary_path: Path | None = None
    predictive_margin_path: Path | None = None
    predictive_warnings_path: Path | None = None
    predictive_summary: list[dict[str, Any]] = []
    if _manifest_has_predictive_jepa(manifest):
        predictive_rows = [
            row for row in metrics_rows if str(row.get("suite_type")) == PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE
        ]
        predictive_summary = aggregate_predictive_robustness_summary(
            metrics_rows,
            manifest,
            primary_metric=str(manifest["metrics"]["primary"]),
        )
        predictive_condition_path = results_dir / "predictive_condition_metrics.csv"
        predictive_summary_path = results_dir / "predictive_regional_summary.json"
        predictive_margin_path = results_dir / "predictive_margin_vs_resnet.json"
        predictive_warnings_path = results_dir / "predictive_warnings.json"
        _write_csv(predictive_condition_path, predictive_rows)
        predictive_summary_path.write_text(
            json.dumps(_json_ready({"summary": predictive_summary}), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        predictive_margin_path.write_text(
            json.dumps(
                _json_ready(
                    {
                        "margins": [
                            {
                                key: row.get(key)
                                for key in (
                                    "model",
                                    "group",
                                    "predictive_dba",
                                    "resnet_predictive_dba",
                                    "margin_vs_resnet_dba",
                                    "claim_pass_5pt",
                                    "claim_status",
                                    "overall_cxd_dba",
                                    "overall_cxd_delta_vs_resnet",
                                )
                            }
                            for row in predictive_summary
                        ]
                    }
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        predictive_warnings_path.write_text(
            json.dumps(_json_ready({"warnings": warnings}), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    scenario_d_rows: list[dict[str, Any]] = []
    heatmap_path: Path | None = None
    scenario_d_results_path: Path | None = None
    cxd_artifacts: dict[str, str] = {}
    if _manifest_has_scenario_d(manifest):
        scenario_d_rows = aggregate_scenario_d_matrix(metrics_rows, primary_metric=str(manifest["metrics"]["primary"]))
        scenario_d_results_path = results_dir / "scenario_d_image_observability.csv"
        _write_csv(scenario_d_results_path, scenario_d_rows)
        heatmap = scenario_d_heatmap(scenario_d_rows, primary_metric=str(manifest["metrics"]["primary"]))
        heatmap_path = results_dir / "heatmap_cx_dy.npy"
        np.save(heatmap_path, heatmap)
        _write_scenario_d_figures(plots_dir, scenario_d_rows, manifest, registry, warnings)
        cxd_artifacts = write_cxd_phase_artifacts(results_dir, plots_dir, metrics_rows, manifest, registry, warnings)
    if bool(manifest.get("figures", {}).get("enabled", manifest.get("figures", {}).get("export", True))):
        _write_benchmark_figures(figures_dir, metrics_rows, manifest, registry, warnings)

    resolved_manifest = _build_runner_manifest(
        manifest,
        output_dir=out,
        command=command,
        dry_run=dry_run,
        model_summaries=model_summaries,
        comparability=comparability,
        warnings=warnings,
        registry=registry,
    )
    manifest_path_out = out / "benchmark_manifest.json"
    resolved_manifest["outputs"] = registry.list_outputs()
    resolved_manifest["outputs"].append(
        {
            "path": "benchmark_manifest.json",
            "kind": "manifest",
            "status": "generated",
            "size_bytes": 0,
        }
    )
    manifest_path_out.write_text(
        json.dumps(_json_ready(resolved_manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    size = int(manifest_path_out.stat().st_size)
    for item in resolved_manifest["outputs"]:
        if item.get("path") == "benchmark_manifest.json":
            item["size_bytes"] = size
    manifest_path_out.write_text(
        json.dumps(_json_ready(resolved_manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(out),
        "manifest": str(manifest_path_out),
        "metrics_by_condition": str(tables_dir / "metrics_by_condition.csv"),
        "robustness_summary": str(tables_dir / "robustness_summary.csv"),
        "shortcut_reliance_summary": str(tables_dir / "shortcut_reliance_summary.csv"),
        "predictive_condition_metrics": str(predictive_condition_path) if predictive_condition_path else "",
        "predictive_regional_summary": str(predictive_summary_path) if predictive_summary_path else "",
        "predictive_margin_vs_resnet": str(predictive_margin_path) if predictive_margin_path else "",
        "predictive_warnings": str(predictive_warnings_path) if predictive_warnings_path else "",
        "scenario_d_results": str(scenario_d_results_path) if scenario_d_results_path else "",
        "scenario_d_heatmap": str(heatmap_path) if heatmap_path else "",
        **cxd_artifacts,
        "models": sorted(manifest["models"]),
        "warnings": warnings,
        "dry_run": bool(dry_run),
    }


def aggregate_robustness_summary(
    metrics_rows: Iterable[Mapping[str, Any]],
    *,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in metrics_rows]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("condition")) == "clean":
            continue
        grouped.setdefault((str(row.get("model")), str(row.get("suite"))), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (model, suite), items in grouped.items():
        items.sort(key=lambda item: (float(item.get("severity") or 0.0), int(item.get("seed") or 0)))
        metric_values = [_float_or_none(item.get("primary_metric")) for item in items]
        severities = [float(item.get("severity") or 0.0) for item in items]
        clean = _float_or_none(items[0].get("clean_primary_metric")) if items else None
        valid_pairs = [(x, y) for x, y in zip(severities, metric_values) if y is not None]
        slope = _collapse_slope(valid_pairs)
        aurc = _area_under_curve(valid_pairs)
        relative_drops = [
            _relative_drop(clean, value)
            for value in metric_values
            if value is not None and clean is not None
        ]
        summaries.append(
            {
                "model": model,
                "suite": suite,
                "suite_type": items[0].get("suite_type", "") if items else "",
                "condition": items[0].get("condition", "") if items else "",
                "split": items[0].get("split", "") if items else "",
                "seed_count": len({str(item.get("seed")) for item in items}),
                "sample_count": items[0].get("sample_count", "") if items else "",
                "primary_metric_name": primary_metric,
                "clean_primary_metric": clean if clean is not None else "",
                "worst_primary_metric": min((value for value in metric_values if value is not None), default=""),
                "max_relative_drop": max(relative_drops, default=0.0),
                "mean_relative_drop": float(np.mean(relative_drops)) if relative_drops else 0.0,
                "collapse_slope": slope,
                "area_under_robustness_curve": aurc,
                "comparability_status": items[0].get("comparability_status", "") if items else "",
                "status": "generated" if items else "empty",
            }
        )
    summaries.sort(key=lambda item: (str(item["model"]), str(item["suite"])))
    return summaries


def aggregate_shortcut_reliance(
    metrics_rows: Iterable[Mapping[str, Any]],
    robustness_rows: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in metrics_rows]
    robustness = [dict(row) for row in robustness_rows]
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row.get("model")), []).append(row)
    output: list[dict[str, Any]] = []
    for model, model_rows in by_model.items():
        clean = next((row for row in model_rows if str(row.get("condition")) == "clean"), None)
        clean_metric = _float_or_none(clean.get("primary_metric")) if clean else None
        drop_gps = _max_drop(model_rows, condition_names={"drop_gps", "gps_missing"})
        drop_image = _max_drop(model_rows, condition_names={"drop_image", "image_occlusion", "image_fog_rain", "image_night"})
        misleading = _max_drop(model_rows, condition_names={"misleading_gps", "gps_distractor"})
        gps_slopes = [
            _float_or_none(row.get("collapse_slope"))
            for row in robustness
            if str(row.get("model")) == model and str(row.get("suite_type")) in GPS_SUITE_TYPES
        ]
        output.append(
            {
                "model": model,
                "group": manifest.get("models", {}).get(model, {}).get("group", ""),
                "clean_primary_metric": clean_metric if clean_metric is not None else "",
                "drop_gps_magnitude": drop_gps,
                "drop_image_magnitude": drop_image,
                "misleading_gps_magnitude": misleading,
                "gps_only_collapse_slope": min((item for item in gps_slopes if item is not None), default=0.0),
                "missing_expression": "zero_fill_with_metadata_warning",
                "counterfactual_intervention": bool(misleading > 0.0),
                "diagnostic_scope": "performance_counterfactual_not_attention_causality",
                "status": "generated",
            }
        )
    output.sort(key=lambda item: str(item["model"]))
    return output


def read_benchmark_analysis_bundle(
    manifest_path: str | Path | None,
    *,
    metrics_by_condition: str | Path | None = None,
    robustness_summary: str | Path | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    manifest: dict[str, Any] = {}
    manifest_digest = None
    manifest_file: Path | None = None
    if manifest_path:
        manifest_file = _resolve_existing_user_path(manifest_path)
        text = manifest_file.read_text(encoding="utf-8")
        manifest = _load_mapping_text(text, path=manifest_file)
        manifest_digest = _sha256_text(text)
    metrics_path = _resolve_artifact_path(
        explicit=metrics_by_condition,
        manifest=manifest,
        manifest_file=manifest_file,
        filename="metrics_by_condition.csv",
        output_key="metrics_by_condition",
    )
    robustness_path = _resolve_artifact_path(
        explicit=robustness_summary,
        manifest=manifest,
        manifest_file=manifest_file,
        filename="robustness_summary.csv",
        output_key="robustness_summary",
    )
    metrics_rows = _read_csv(metrics_path) if metrics_path and metrics_path.exists() else []
    if not metrics_rows:
        warnings.append("benchmark_metrics_unavailable")
    robustness_rows = _read_csv(robustness_path) if robustness_path and robustness_path.exists() else []
    if metrics_rows and not robustness_rows:
        robustness_rows = aggregate_robustness_summary(
            metrics_rows,
            primary_metric=str(manifest.get("metrics", {}).get("primary", DEFAULT_PRIMARY_METRIC)),
        )
    matrix_rows = robustness_matrix_rows(metrics_rows)
    case_rows = select_benchmark_case_studies(metrics_rows)
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_file) if manifest_file is not None else None,
        "manifest_digest": manifest_digest,
        "metrics_path": str(metrics_path) if metrics_path is not None else None,
        "robustness_path": str(robustness_path) if robustness_path is not None else None,
        "metrics_rows": metrics_rows,
        "robustness_rows": robustness_rows,
        "matrix_rows": matrix_rows,
        "case_rows": case_rows,
        "warnings": warnings,
    }


def robustness_matrix_rows(metrics_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in metrics_rows]
    clean_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("condition")) == "clean":
            clean_by_key[(str(row.get("model")), str(row.get("split")), str(row.get("seed")))] = row
    matrix: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("condition")) == "clean":
            continue
        clean = clean_by_key.get((str(row.get("model")), str(row.get("split")), str(row.get("seed"))), {})
        clean_metric = _float_or_none(clean.get("primary_metric"))
        perturbed = _float_or_none(row.get("primary_metric"))
        delta = "" if clean_metric is None or perturbed is None else perturbed - clean_metric
        matrix.append(
            {
                "model": row.get("model", ""),
                "suite": row.get("suite", ""),
                "condition": row.get("condition", ""),
                "severity": row.get("severity", ""),
                "seed": row.get("seed", ""),
                "split": row.get("split", ""),
                "sample_count": row.get("sample_count", ""),
                "clean_metric": clean_metric if clean_metric is not None else row.get("clean_primary_metric", ""),
                "perturbed_metric": perturbed if perturbed is not None else "",
                "delta": delta,
                "relative_drop": row.get("relative_drop", ""),
                "suite_type": row.get("suite_type", ""),
                "metric": row.get("primary_metric_name", DEFAULT_PRIMARY_METRIC),
                "top1": row.get("top1", ""),
                "top3": row.get("top3", ""),
                "top5": row.get("top5", ""),
                "mean_beam_index_error": row.get("mean_beam_index_error", ""),
                "max_delay_steps": row.get("max_delay_steps", ""),
                "gps_stride": row.get("gps_stride", ""),
                "gps_dropout_prob": row.get("gps_dropout_prob", ""),
                "accuracy_c0_ratio": row.get("accuracy_c0_ratio", ""),
                "image_only_missing_gps_slice": row.get("image_only_missing_gps_slice", ""),
            }
        )
    matrix.sort(key=lambda item: (str(item["suite"]), str(item["severity"]), str(item["model"])))
    return matrix


def select_benchmark_case_studies(metrics_rows: Iterable[Mapping[str, Any]], *, seed: int = 42) -> list[dict[str, Any]]:
    rows = [dict(row) for row in metrics_rows if str(row.get("condition")) != "clean"]
    if not rows:
        return []
    rng = np.random.default_rng(int(seed))
    jepa_rows = [row for row in rows if "jepa" in str(row.get("model", "")).lower()]
    gps_rows = [row for row in rows if "gps" in str(row.get("model", "")).lower() and "jepa" not in str(row.get("model", "")).lower()]
    output: list[dict[str, Any]] = []
    if jepa_rows and gps_rows:
        gps_lookup = {
            (str(row.get("suite")), str(row.get("severity")), str(row.get("seed"))): row
            for row in gps_rows
        }
        candidates = []
        for row in jepa_rows:
            other = gps_lookup.get((str(row.get("suite")), str(row.get("severity")), str(row.get("seed"))))
            if other is None:
                continue
            if _float(row.get("relative_drop")) < _float(other.get("relative_drop")):
                candidates.append((row, other))
        if candidates:
            row, other = candidates[int(rng.integers(0, len(candidates)))]
            output.append(_case_row("jepa_recovery", row, other))
    misleading = [row for row in gps_rows if str(row.get("condition")) in {"misleading_gps", "gps_distractor"} or "distractor" in str(row.get("suite_type"))]
    if misleading:
        row = max(misleading, key=lambda item: _float(item.get("relative_drop")))
        output.append(_case_row("gps_shortcut_failure", row, None))
    by_condition: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault((str(row.get("suite")), str(row.get("severity")), str(row.get("seed"))), []).append(row)
    shared = [
        items
        for items in by_condition.values()
        if len(items) >= 2 and all(_float(item.get("relative_drop")) >= 0.25 for item in items)
    ]
    if shared:
        items = shared[int(rng.integers(0, len(shared)))]
        output.append(_case_row("shared_failure", max(items, key=lambda item: _float(item.get("relative_drop"))), None))
    return output


def _model_metric_source(
    model_name: str,
    model_spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    output_dir: Path,
    dry_run: bool,
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    split = str(model_spec.get("split", manifest.get("protocol", {}).get("split", "test")))
    primary = str(manifest.get("metrics", {}).get("primary", DEFAULT_PRIMARY_METRIC))
    if dry_run:
        synthetic = model_spec.get("synthetic_metrics", {}) if isinstance(model_spec.get("synthetic_metrics"), Mapping) else {}
        return _summary_from_metric_mapping(model_name, synthetic, primary=primary, split=split, status="dry_run")
    logits_cache = model_spec.get("logits_cache") or model_spec.get("cache")
    if logits_cache:
        return _summary_from_logits_cache(
            model_name,
            _resolve_existing_user_path(logits_cache),
            primary=primary,
            split=split,
            num_beams=int(model_spec.get("num_beams", 64)),
            dba_delta=float(manifest.get("metrics", {}).get("dba_delta", model_spec.get("dba_delta", 5))),
            distance_mode=str(manifest.get("metrics", {}).get("distance_mode", model_spec.get("distance_mode", "circular"))),
        )
    synthetic_metrics = model_spec.get("synthetic_metrics")
    if isinstance(synthetic_metrics, Mapping):
        return _summary_from_metric_mapping(model_name, synthetic_metrics, primary=primary, split=split, status="synthetic")
    if bool(model_spec.get("delegate_evaluate", False)):
        return _summary_from_delegated_evaluate(
            model_name,
            model_spec,
            output_dir=output_dir,
            primary=primary,
            split=split,
            warnings=warnings,
        )
    if manifest.get("protocol", {}).get("mode") == "train_then_evaluate":
        warnings.append(
            WarningRecord(
                code="train_then_evaluate_planned",
                message=f"Training plan for {model_name} recorded; benchmark runner did not launch training in this invocation.",
            ).to_dict()
        )
        return _summary_from_metric_mapping(model_name, {}, primary=primary, split=split, status="planned_train_then_evaluate")
    raise BenchmarkManifestError(
        f"models.{model_name} needs logits_cache, synthetic_metrics, or delegate_evaluate=true for runner execution."
    )


def _metrics_rows_for_model(
    model_name: str,
    model_spec: Mapping[str, Any],
    source: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    comparability_status: str,
    dry_run: bool,
) -> list[dict[str, Any]]:
    split = str(source.get("split", model_spec.get("split", manifest.get("protocol", {}).get("split", "test"))))
    primary_name = str(manifest.get("metrics", {}).get("primary", DEFAULT_PRIMARY_METRIC))
    clean_primary = _float(source.get("primary_metric"))
    sample_count = int(source.get("sample_count", model_spec.get("sample_count", 0)) or 0)
    rows: list[dict[str, Any]] = []
    for seed in manifest.get("seeds", [42]):
        rows.append(
            {
                "model": model_name,
                "group": model_spec.get("group", ""),
                "suite": "clean",
                "suite_type": "gps_clean",
                "condition": "clean",
                "severity": 0.0,
                "severity_unit": "reference",
                "seed": int(seed),
                "split": split,
                "sample_count": sample_count,
                "primary_metric_name": primary_name,
                "primary_metric": clean_primary,
                "clean_primary_metric": clean_primary,
                "clean_delta": 0.0,
                "relative_drop": 0.0,
                "top1": _metric_or_blank(source, "top1"),
                "top3": _metric_or_blank(source, "top3"),
                "top5": _metric_or_blank(source, "top5"),
                "dba": _metric_or_blank(source, "dba"),
                "mean_beam_index_error": _metric_or_blank(source, "mean_beam_index_error"),
                "consumes_reliability_metadata": _model_consumes_reliability_metadata(model_spec),
                "comparability_status": comparability_status,
                "status": source.get("status", "generated") if not dry_run else "dry_run",
            }
        )
    for suite in manifest.get("perturbation_suites", []):
        suite_type = str(suite.get("type"))
        if suite_type == SCENARIO_C_X_D_SUITE_TYPE:
            for seed in manifest.get("seeds", [42]):
                for gps_condition in suite.get("scenario_c_conditions", []):
                    for image_condition in suite.get("scenario_d_conditions", []):
                        rows.append(
                            _scenario_cxd_metric_row(
                                model_name,
                                model_spec,
                                source,
                                suite,
                                gps_condition=gps_condition,
                                image_condition=image_condition,
                                seed=int(seed),
                                split=split,
                                sample_count=sample_count,
                                primary_name=primary_name,
                                clean_primary=clean_primary,
                                comparability_status=comparability_status,
                                dry_run=dry_run,
                            )
                        )
            continue
        if suite_type == PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE:
            for seed in manifest.get("seeds", [42]):
                for condition in suite.get("predictive_conditions", []):
                    rows.append(
                        _predictive_jepa_metric_row(
                            model_name,
                            model_spec,
                            source,
                            suite,
                            condition=condition,
                            seed=int(seed),
                            split=split,
                            sample_count=sample_count,
                            primary_name=primary_name,
                            clean_primary=clean_primary,
                            comparability_status=comparability_status,
                            dry_run=dry_run,
                        )
                    )
            continue
        suite_max = max([float(value) for value in suite.get("severities", [0.0])] or [0.0] + [1.0])
        for severity in suite.get("severities", [0.0]):
            severity_value = float(severity)
            scenario_c_condition = _scenario_c_condition_for_severity(suite, severity_value) if suite_type == SCENARIO_C_SUITE_TYPE else {}
            scenario_d_condition = _scenario_d_condition_for_severity(suite, severity_value) if suite_type == SCENARIO_D_SUITE_TYPE else {}
            for seed in manifest.get("seeds", [42]):
                metric_value = _perturbed_metric_value(clean_primary, severity_value, suite_type, model_spec, suite_max=suite_max)
                row = {
                    "model": model_name,
                    "group": model_spec.get("group", ""),
                    "suite": suite.get("id", suite_type),
                    "suite_type": suite_type,
                    "condition": scenario_c_condition.get("id", suite.get("condition", _default_condition(suite_type))),
                    "severity": severity_value,
                    "severity_unit": suite.get("severity_unit", _default_severity_unit(suite_type)),
                    "seed": int(seed),
                    "split": split,
                    "sample_count": sample_count,
                    "primary_metric_name": primary_name,
                    "primary_metric": metric_value,
                    "clean_primary_metric": clean_primary,
                    "clean_delta": metric_value - clean_primary,
                    "relative_drop": _relative_drop(clean_primary, metric_value),
                    "top1": _scaled_metric(source, "top1", clean_primary, metric_value),
                    "top3": _scaled_metric(source, "top3", clean_primary, metric_value),
                    "top5": _scaled_metric(source, "top5", clean_primary, metric_value),
                    "dba": metric_value if primary_name == "dba" else _scaled_metric(source, "dba", clean_primary, metric_value),
                    "mean_beam_index_error": _scaled_error_metric(source, "mean_beam_index_error", clean_primary, metric_value),
                    "consumes_reliability_metadata": _model_consumes_reliability_metadata(model_spec),
                    "comparability_status": comparability_status,
                    "status": source.get("status", "generated") if not dry_run else "dry_run",
                }
                if suite_type == SCENARIO_C_SUITE_TYPE:
                    row.update(_scenario_c_metric_columns(scenario_c_condition, model_spec=model_spec))
                if suite_type == SCENARIO_D_SUITE_TYPE:
                    row.update(_scenario_d_metric_columns(scenario_d_condition, seed=int(seed)))
                rows.append(row)
    _add_scenario_c_accuracy_ratios(rows)
    rows.sort(key=lambda item: (str(item["model"]), str(item["suite"]), float(item["severity"]), int(item["seed"])))
    return rows


def _summary_from_logits_cache(
    model_name: str,
    cache_path: Path,
    *,
    primary: str,
    split: str,
    num_beams: int,
    dba_delta: float,
    distance_mode: str,
) -> dict[str, Any]:
    payload = np.load(cache_path, allow_pickle=True)
    logits = np.asarray(payload["logits"], dtype=np.float32)
    labels = np.asarray(payload["labels"], dtype=np.int64)
    if logits.ndim == 3:
        logits = logits[:, 0, :]
    if labels.ndim == 2:
        labels = labels[:, 0]
    outputs = torch.tensor(logits, dtype=torch.float32).unsqueeze(1)
    labels_t = torch.tensor(labels.reshape(-1, 1), dtype=torch.long)
    topk, _ = calculate_topk_accuracy(outputs, labels_t, k_values=(1, 3, 5))
    dba = float(calculate_dba_score(outputs, labels_t, delta=float(dba_delta), distance_mode=distance_mode)[0])
    summary = {
        "model": model_name,
        "source": "logits_cache",
        "path": str(cache_path),
        "split": split,
        "sample_count": int(logits.shape[0]),
        "num_beams": int(num_beams or logits.shape[-1]),
        "top1": _topk_value(topk, 1),
        "top3": _topk_value(topk, 3),
        "top5": _topk_value(topk, 5),
        "dba": dba,
        "status": "generated",
    }
    summary["primary_metric"] = float(summary.get(primary, summary["dba"]))
    pred = np.argmax(logits, axis=-1)
    summary["mean_beam_index_error"] = float(np.mean(np.abs(pred.astype(np.float32) - labels.astype(np.float32))))
    return summary


def _summary_from_metric_mapping(
    model_name: str,
    metrics: Mapping[str, Any],
    *,
    primary: str,
    split: str,
    status: str,
) -> dict[str, Any]:
    sample_count = int(metrics.get("sample_count", metrics.get("n", 0)) or 0)
    summary = {
        "model": model_name,
        "source": status,
        "split": split,
        "sample_count": sample_count,
        "top1": _float(metrics.get("top1", 0.0)),
        "top3": _float(metrics.get("top3", metrics.get("top_3", 0.0))),
        "top5": _float(metrics.get("top5", metrics.get("top_5", 0.0))),
        "dba": _float(metrics.get("dba", metrics.get("DBA", 0.0))),
        "mean_beam_index_error": _float(
            metrics.get("mean_beam_index_error", metrics.get("beam_index_mae", metrics.get("beam_mae", 0.0)))
        ),
        "status": status,
    }
    summary["primary_metric"] = _float(metrics.get(primary, summary.get(primary, summary["dba"])))
    return summary


def _summary_from_delegated_evaluate(
    model_name: str,
    model_spec: Mapping[str, Any],
    *,
    output_dir: Path,
    primary: str,
    split: str,
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    from kd_sensing.engine.evaluator import evaluate

    cfg = load_config(model_spec["config"], overrides=model_spec.get("overrides") or [])
    result = evaluate(cfg, weights=str(model_spec.get("weights")), output_dir=output_dir / "evaluations" / model_name)
    metrics = result.get("metrics", {})
    summary = _summary_from_metric_mapping(model_name, metrics, primary=primary, split=split, status="delegated_evaluate")
    sample_count = (
        result.get("split_metadata", {})
        .get("test", {})
        .get("num_samples", summary.get("sample_count", 0))
    )
    summary["sample_count"] = int(sample_count or summary.get("sample_count", 0))
    summary["checkpoint_load"] = result.get("checkpoint_load")
    warnings.append(
        WarningRecord(
            code="delegate_evaluate_clean_only",
            message=f"{model_name} clean evaluation was delegated; perturbation metrics use deterministic benchmark degradation model.",
        ).to_dict()
    )
    return summary


def _build_runner_manifest(
    manifest: Mapping[str, Any],
    *,
    output_dir: Path,
    command: list[str] | None,
    dry_run: bool,
    model_summaries: Mapping[str, Any],
    comparability: Mapping[str, Any],
    warnings: list[dict[str, Any]],
    registry: OutputRegistry,
) -> dict[str, Any]:
    model_records = {}
    for name, spec in manifest.get("models", {}).items():
        checkpoint_metadata = None
        weights = spec.get("weights") if isinstance(spec, Mapping) else None
        if weights and not bool(spec.get("allow_missing_artifacts", False)):
            try:
                checkpoint_metadata = load_checkpoint_metadata(resolve_path(weights))
            except Exception:
                checkpoint_metadata = None
        model_records[name] = {
            "group": spec.get("group") if isinstance(spec, Mapping) else "",
            "config": spec.get("config") if isinstance(spec, Mapping) else None,
            "weights": weights,
            "logits_cache": spec.get("logits_cache") if isinstance(spec, Mapping) else None,
            "modalities": spec.get("modalities", spec.get("enabled_modalities", [])) if isinstance(spec, Mapping) else [],
            "consumes_reliability_metadata": _model_consumes_reliability_metadata(spec) if isinstance(spec, Mapping) else False,
            "checkpoint_provenance": spec.get("checkpoint_provenance", weights) if isinstance(spec, Mapping) else weights,
            "checkpoint_metadata": checkpoint_metadata,
            "training": spec.get("training") if isinstance(spec, Mapping) else None,
            "summary": model_summaries.get(name),
        }
    return {
        "version": RUNNER_VERSION,
        "benchmark_version": manifest.get("version", BENCHMARK_VERSION),
        "dry_run": bool(dry_run),
        "command": list(command or []),
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "git_status_short": _git_status_short(),
        "input_manifest_path": manifest.get("_manifest_path"),
        "input_manifest_digest": manifest.get("_manifest_digest"),
        "output_dir": str(output_dir),
        "models": model_records,
        "protocol": manifest.get("protocol", {}),
        "perturbation_suites": manifest.get("perturbation_suites", []),
        "scenario_d_model_groups": manifest.get("scenario_d_model_groups", {}),
        "predictive_model_groups": manifest.get("predictive_model_groups", {}),
        "difficulty_provenance": _benchmark_difficulty_provenance(manifest),
        "seeds": manifest.get("seeds", []),
        "metrics": manifest.get("metrics", {}),
        "figures": manifest.get("figures", {}),
        "analysis": manifest.get("analysis", {}),
        "split_metadata": {
            name: {
                "split": spec.get("split", manifest.get("protocol", {}).get("split", "test")),
                "sample_count": model_summaries.get(name, {}).get("sample_count", spec.get("sample_count", "")),
            }
            for name, spec in manifest.get("models", {}).items()
        },
        "comparability": comparability,
        "warnings": warnings,
        "output_files": {
            "metrics_by_condition": "tables/metrics_by_condition.csv",
            "robustness_summary": "tables/robustness_summary.csv",
            "shortcut_reliance_summary": "tables/shortcut_reliance_summary.csv",
            "scenario_d_image_observability": "results/scenario_d_image_observability.csv",
            "heatmap_cx_dy": "results/heatmap_cx_dy.npy",
            "cxd_phase_diagram": "results/cxd_phase_diagram.csv",
            "cxd_phase_heatmap": "results/cxd_phase_heatmap.npy",
            "modality_dominance": "results/modality_dominance.csv",
            "crossing_region_Cx_Dy": "results/crossing_region_Cx_Dy.json",
            "failure_mode_decomposition": "results/failure_mode_decomposition.csv",
            **PREDICTIVE_OUTPUT_FILES,
            "cxd_accuracy_heatmap": "plots/cxd_accuracy_heatmap.png",
            "resnet_jepa_crossover_curve": "plots/resnet_jepa_crossover_curve.png",
            "modality_dominance_heatmap": "plots/modality_dominance_heatmap.png",
            "robustness_surface": "plots/robustness_surface.png",
            "phase_transition_curve": "plots/phase_transition_curve.png",
            "legacy_modality_dominance_plot": "plots/modality_dominance.png",
        },
        "outputs": registry.list_outputs(),
    }


__all__ = [
    "_build_runner_manifest",
    "_metrics_rows_for_model",
    "_model_metric_source",
    "_summary_from_delegated_evaluate",
    "_summary_from_logits_cache",
    "_summary_from_metric_mapping",
    "aggregate_robustness_summary",
    "aggregate_shortcut_reliance",
    "read_benchmark_analysis_bundle",
    "robustness_matrix_rows",
    "run_jepa_gps_shortcut_benchmark",
    "select_benchmark_case_studies",
]
