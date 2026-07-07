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
from kd_sensing.data.difficulty.pipeline import apply_difficulty_pipeline
from kd_sensing.data.difficulty.schema import (
    DifficultyContext,
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
    _prepare_output_dir,
    _resolve_existing_user_path,
    _resolve_output_dir,
    _write_csv,
)
from kd_sensing.diagnostics.jepa_benchmark_aggregation import (
    aggregate_robustness_summary,
    aggregate_shortcut_reliance,
    read_benchmark_analysis_bundle,
    robustness_matrix_rows,
    select_benchmark_case_studies,
)
from kd_sensing.diagnostics.jepa_benchmark_sources import (
    collect_model_metric_rows,
    evaluate_comparability_or_warn,
)
from kd_sensing.diagnostics.jepa_benchmark_suite_dispatch import write_core_benchmark_tables
from kd_sensing.diagnostics.jepa_benchmark_predictive_artifacts import write_predictive_and_fusion_artifacts
from kd_sensing.diagnostics.predictive_gps_query_visualizations import run_predictive_gps_query_visualizations
from kd_sensing.diagnostics.jepa_benchmark_real_forward import (
    PERTURBATION_CACHE_SCHEMA_VERSION,
    iter_perturbation_cache as _iter_perturbation_cache,
    load_perturbation_cache_index as _load_perturbation_cache_index,
    perturbation_cache_config as _perturbation_cache_config,
    perturbation_cache_index_path as _perturbation_cache_index_path,
    perturbation_cache_key as _perturbation_cache_key,
    real_forward_requested as _real_forward_requested,
    real_forward_settings as _real_forward_settings,
    write_perturbation_cache_index as _write_perturbation_cache_index,
    write_perturbation_cache_shard as _write_perturbation_cache_shard,
)
from kd_sensing.diagnostics.jepa_benchmark_common import (
    BENCHMARK_VERSION,
    BenchmarkManifestError,
    CXD_CORE_OUTPUT_FILES,
    CXD_GPS_CONDITION_IDS,
    CXD_IMAGE_CONDITION_IDS,
    CXD_IMAGE_GPS_BASELINE_GROUPS,
    CXD_JEPA_GROUPS,
    CXD_PLOT_OUTPUT_FILES,
    CXD_STRICT_COMPARABILITY_KEYS,
    DEFAULT_COMPARABILITY_KEYS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRIMARY_METRIC,
    GPS_QUERY_ADVANTAGE_CANONICAL_CONDITIONS,
    GPS_QUERY_ADVANTAGE_CXD_GPS_CONDITION_IDS,
    GPS_QUERY_ADVANTAGE_CXD_IMAGE_CONDITION_IDS,
    GPS_QUERY_ADVANTAGE_SLICE_TYPE,
    GPS_SUITE_TYPES,
    IMAGE_SUITE_TYPES,
    MATRIX_SUITE_TYPES,
    PREDICTIVE_GROUP_ALIASES,
    PREDICTIVE_OUTPUT_FILES,
    PREDICTIVE_REQUIRED_MODEL_GROUPS,
    PREDICTIVE_SUITE_TYPES,
    REUSED_WEIGHT_FUSION_DIAGNOSTIC_OUTPUT_FILES,
    RUNNER_VERSION,
    SCENARIO_C_CANONICAL_CONDITIONS,
    SCENARIO_C_SUITE_TYPE,
    SCENARIO_C_X_D_SUITE_TYPE,
    SCENARIO_D_GROUP_ALIASES,
    SCENARIO_D_REQUIRED_MODEL_GROUPS,
    SUITE_ALIASES,
    SUPPORTED_MODEL_GROUPS,
    SUPPORTED_PROTOCOLS,
    SUPPORTED_SUITE_TYPES,
    TEMPORAL_SUITE_TYPES,
    WarningRecord,
    _area_under_curve,
    _batch_size,
    _case_row,
    _collapse_slope,
    _comparable_scalar,
    _condition_digest,
    _condition_index,
    _crossing_condition_rank,
    _csv_scalar,
    _default_condition,
    _default_severity_unit,
    _finite_float,
    _float,
    _float_or_blank,
    _float_or_none,
    _json_ready,
    _max_drop,
    _metadata_rows,
    _metadata_value_at,
    _metric_or_blank,
    _model_consumes_reliability_metadata,
    _non_negative_int,
    _output_kind,
    _positive_int,
    _perturbed_metric_value,
    _predictive_group_category,
    _relative_drop,
    _relative_to_root,
    _sample_ids_from_metadata,
    _scaled_error_metric,
    _scaled_metric,
    _scenario_d_group_category,
    _sha256_text,
    _sorted_modalities,
    _stable_seed,
    _suite_sensitivity,
    _topk_value,
)
from kd_sensing.diagnostics.jepa_benchmark_manifest import (
    load_benchmark_manifest,
    _manifest_has_scenario_d,
)
from kd_sensing.diagnostics.geometry_prior_beam_fusion import (
    GEOMETRY_PRIOR_OUTPUT_FILES,
    aggregate_geometry_prior_diagnostics,
    build_geometry_prior_claim_gate,
    build_geometry_prior_diagnostics_bundle_manifest,
)
from kd_sensing.diagnostics.jepa_benchmark_perturbations import (
    _benchmark_difficulty_provenance,
    apply_benchmark_perturbation,
)
from kd_sensing.diagnostics.jepa_benchmark_plots import _write_benchmark_figures, _write_scenario_d_figures
from kd_sensing.diagnostics.jepa_benchmark_predictive import _predictive_jepa_metric_row
from kd_sensing.diagnostics.jepa_benchmark_predictive_advantage_metrics import (
    _predictive_gps_query_advantage_metric_rows,
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
    predictive_explanatory_visualizations: bool = False,
    predictive_explanatory_output_dir: str | Path | None = None,
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

    comparability = evaluate_comparability_or_warn(manifest, warnings=warnings)
    model_summaries, metrics_rows = collect_model_metric_rows(
        manifest,
        output_dir=out,
        dry_run=dry_run,
        warnings=warnings,
        comparability_status=str(comparability["status"]),
        metric_source_loader=_model_metric_source,
        metric_rows_builder=_metrics_rows_for_model,
    )
    write_core_benchmark_tables(metrics_rows, manifest, tables_dir=tables_dir)
    predictive_artifacts = write_predictive_and_fusion_artifacts(
        metrics_rows,
        manifest,
        results_dir=results_dir,
        warnings=warnings,
    )
    artifact_paths = predictive_artifacts.paths
    scenario_d_rows: list[dict[str, Any]] = []
    heatmap_path: Path | None = None
    scenario_d_results_path: Path | None = None
    cxd_artifacts: dict[str, str] = {}
    geometry_prior_paths: dict[str, Path] = {}
    if _manifest_has_scenario_d(manifest):
        scenario_d_rows = aggregate_scenario_d_matrix(metrics_rows, primary_metric=str(manifest["metrics"]["primary"]))
        scenario_d_results_path = results_dir / "scenario_d_image_observability.csv"
        _write_csv(scenario_d_results_path, scenario_d_rows)
        heatmap = scenario_d_heatmap(scenario_d_rows, primary_metric=str(manifest["metrics"]["primary"]))
        heatmap_path = results_dir / "heatmap_cx_dy.npy"
        np.save(heatmap_path, heatmap)
        _write_scenario_d_figures(plots_dir, scenario_d_rows, manifest, registry, warnings)
        cxd_artifacts = write_cxd_phase_artifacts(results_dir, plots_dir, metrics_rows, manifest, registry, warnings)
    if _manifest_has_geometry_prior(manifest):
        geometry_diagnostics = aggregate_geometry_prior_diagnostics(metrics_rows, manifest=manifest)
        geometry_claim_gate = build_geometry_prior_claim_gate(metrics_rows, manifest)
        geometry_bundle = build_geometry_prior_diagnostics_bundle_manifest(
            manifest,
            diagnostics=geometry_diagnostics,
            claim_gate=geometry_claim_gate,
        )
        geometry_prior_paths = {
            "prior_quality": results_dir / "geometry_prior_quality.csv",
            "branch_weights": results_dir / "geometry_prior_branch_weights.csv",
            "strict_comparison": results_dir / "geometry_prior_strict_comparison.csv",
            "claim_gate": results_dir / "geometry_prior_claim_gate.json",
            "diagnostics_bundle": results_dir / "geometry_prior_diagnostics_bundle_manifest.json",
        }
        _write_csv(geometry_prior_paths["prior_quality"], geometry_diagnostics["prior_quality"])
        _write_csv(geometry_prior_paths["branch_weights"], geometry_diagnostics["branch_weights"])
        _write_csv(geometry_prior_paths["strict_comparison"], geometry_diagnostics["strict_comparison"])
        geometry_prior_paths["claim_gate"].write_text(
            json.dumps(_json_ready({"claim_gate": geometry_claim_gate}), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        geometry_prior_paths["diagnostics_bundle"].write_text(
            json.dumps(_json_ready(geometry_bundle), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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
    explanatory_visualizations: dict[str, Any] = {}
    if predictive_explanatory_visualizations:
        explanatory_visualizations = run_predictive_gps_query_visualizations(
            manifest_path=manifest_path_out,
            output_dir=predictive_explanatory_output_dir or (out / "predictive_explanatory_visualizations"),
            force=force,
        )
    return {
        "output_dir": str(out),
        "manifest": str(manifest_path_out),
        "metrics_by_condition": str(tables_dir / "metrics_by_condition.csv"),
        "robustness_summary": str(tables_dir / "robustness_summary.csv"),
        "shortcut_reliance_summary": str(tables_dir / "shortcut_reliance_summary.csv"),
        "predictive_condition_metrics": str(artifact_paths.get("predictive_condition_metrics") or ""),
        "predictive_regional_summary": str(artifact_paths.get("predictive_regional_summary") or ""),
        "predictive_margin_vs_resnet": str(artifact_paths.get("predictive_margin_vs_resnet") or ""),
        "predictive_warnings": str(artifact_paths.get("predictive_warnings") or ""),
        "predictive_gps_query_advantage_metrics": str(artifact_paths.get("predictive_gps_query_advantage_metrics") or ""),
        "predictive_gps_query_advantage_margins": str(artifact_paths.get("predictive_gps_query_advantage_margins") or ""),
        "predictive_claim_gate": str(artifact_paths.get("predictive_claim_gate") or ""),
        "predictive_diagnostics_bundle_manifest": str(artifact_paths.get("predictive_diagnostics_bundle_manifest") or ""),
        "fusion_diagnostic_condition_metrics": str(artifact_paths.get("fusion_diagnostic_condition_metrics") or ""),
        "fusion_diagnostic_paired_margins": str(artifact_paths.get("fusion_diagnostic_paired_margins") or ""),
        "fusion_diagnostic_summary": str(artifact_paths.get("fusion_diagnostic_summary") or ""),
        "scenario_d_results": str(scenario_d_results_path) if scenario_d_results_path else "",
        "scenario_d_heatmap": str(heatmap_path) if heatmap_path else "",
        "geometry_prior_quality": str(geometry_prior_paths.get("prior_quality", "")),
        "geometry_prior_branch_weights": str(geometry_prior_paths.get("branch_weights", "")),
        "geometry_prior_strict_comparison": str(geometry_prior_paths.get("strict_comparison", "")),
        "geometry_prior_claim_gate": str(geometry_prior_paths.get("claim_gate", "")),
        "geometry_prior_diagnostics_bundle_manifest": str(geometry_prior_paths.get("diagnostics_bundle", "")),
        "predictive_explanatory_visualizations_manifest": str(explanatory_visualizations.get("manifest", "")),
        "predictive_explanatory_visualizations_output_dir": str(explanatory_visualizations.get("output_dir", "")),
        **cxd_artifacts,
        "models": sorted(manifest["models"]),
        "warnings": warnings,
        "dry_run": bool(dry_run),
    }







def _manifest_has_geometry_prior(manifest: Mapping[str, Any]) -> bool:
    if isinstance(manifest.get("geometry_prior_claim_gate"), Mapping) or isinstance(manifest.get("geometry_prior"), Mapping):
        return True
    groups = {
        str(spec.get("group", ""))
        for spec in manifest.get("models", {}).values()
        if isinstance(spec, Mapping)
    }
    return bool(
        groups
        & {
            "geometry_prior",
            "geometry_prior_prior_only",
            "geometry_prior_fusion",
            "geometry_prior_dba_aware",
            "geometry_prior_teacher_guided",
            "geometry_prior_mixed_curriculum",
            "safe_residual_beam_rerank_fusion",
            "safe_residual_rerank_fusion",
            "real_perturbation_residual_rerank_fusion",
        }
    )


# Runner manifest assembly
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
            checkpoint_metadata = load_checkpoint_metadata(resolve_path(weights))
        model_records[name] = {
            "group": spec.get("group") if isinstance(spec, Mapping) else "",
            "config": spec.get("config") if isinstance(spec, Mapping) else None,
            "weights": weights,
            "logits_cache": spec.get("logits_cache") if isinstance(spec, Mapping) else None,
            "modalities": spec.get("modalities", spec.get("enabled_modalities", [])) if isinstance(spec, Mapping) else [],
            "consumes_reliability_metadata": _model_consumes_reliability_metadata(spec) if isinstance(spec, Mapping) else False,
            "checkpoint_provenance": spec.get("checkpoint_provenance", weights) if isinstance(spec, Mapping) else weights,
            "real_benchmark_gate": spec.get("real_benchmark_gate", {}) if isinstance(spec, Mapping) else {},
            "real_benchmark_status": spec.get("real_benchmark_status", "") if isinstance(spec, Mapping) else "",
            "real_benchmark_missing_fields": spec.get("real_benchmark_missing_fields", []) if isinstance(spec, Mapping) else [],
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
        "evaluation": manifest.get("evaluation", {}),
        "perturbation_cache": (
            manifest.get("evaluation", {}).get("real_forward", {}).get("perturbation_cache", {})
            if isinstance(manifest.get("evaluation", {}), Mapping)
            and isinstance(manifest.get("evaluation", {}).get("real_forward", {}), Mapping)
            else {}
        ),
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
            **GEOMETRY_PRIOR_OUTPUT_FILES,
            **PREDICTIVE_OUTPUT_FILES,
            **REUSED_WEIGHT_FUSION_DIAGNOSTIC_OUTPUT_FILES,
            "cxd_accuracy_heatmap": "plots/cxd_accuracy_heatmap.png",
            "resnet_jepa_crossover_curve": "plots/resnet_jepa_crossover_curve.png",
            "modality_dominance_heatmap": "plots/modality_dominance_heatmap.png",
            "robustness_surface": "plots/robustness_surface.png",
            "phase_transition_curve": "plots/phase_transition_curve.png",
            "legacy_modality_dominance_plot": "plots/modality_dominance.png",
        },
        "outputs": registry.list_outputs(),
    }


# Runner metric source ingestion
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
        return _with_real_gate_status(
            model_name,
            model_spec,
            _summary_from_logits_cache(
                model_name,
                _resolve_existing_user_path(logits_cache),
                primary=primary,
                split=split,
                num_beams=int(model_spec.get("num_beams", 64)),
                dba_delta=float(manifest.get("metrics", {}).get("dba_delta", model_spec.get("dba_delta", 5))),
                distance_mode=str(manifest.get("metrics", {}).get("distance_mode", model_spec.get("distance_mode", "circular"))),
            ),
            warnings=warnings,
        )
    if _real_forward_requested(manifest, model_spec):
        return _with_real_gate_status(
            model_name,
            model_spec,
            _summary_from_real_forward(
                model_name,
                model_spec,
                manifest,
                output_dir=output_dir,
                primary=primary,
                split=split,
                warnings=warnings,
            ),
            warnings=warnings,
        )
    synthetic_metrics = model_spec.get("synthetic_metrics")
    if isinstance(synthetic_metrics, Mapping):
        return _summary_from_metric_mapping(model_name, synthetic_metrics, primary=primary, split=split, status="synthetic")
    gate_status = str(model_spec.get("real_benchmark_status") or "")
    if gate_status in {"unavailable", "not_comparable"}:
        warnings.append(
            WarningRecord(
                code=f"real_benchmark_{gate_status}",
                message=f"{model_name} marked {gate_status}: {model_spec.get('real_benchmark_gate', {}).get('reason', '')}",
            ).to_dict()
        )
        return _summary_from_metric_mapping(model_name, {}, primary=primary, split=split, status=gate_status)
    if bool(model_spec.get("delegate_evaluate", False)):
        return _with_real_gate_status(
            model_name,
            model_spec,
            _summary_from_delegated_evaluate(
                model_name,
                model_spec,
                output_dir=output_dir,
                primary=primary,
                split=split,
                warnings=warnings,
            ),
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


def _with_real_gate_status(
    model_name: str,
    model_spec: Mapping[str, Any],
    summary: dict[str, Any],
    *,
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    gate_status = str(model_spec.get("real_benchmark_status") or "")
    if gate_status in {"unavailable", "not_comparable"}:
        summary["status"] = gate_status
        warnings.append(
            WarningRecord(
                code=f"real_benchmark_{gate_status}",
                message=f"{model_name} marked {gate_status}: {model_spec.get('real_benchmark_gate', {}).get('reason', '')}",
            ).to_dict()
        )
    return summary

def _metrics_rows_for_model(
    model_name: str,
    model_spec: Mapping[str, Any],
    source: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    comparability_status: str,
    dry_run: bool,
) -> list[dict[str, Any]]:
    real_forward_rows = source.get("real_forward_rows")
    if isinstance(real_forward_rows, list) and real_forward_rows:
        rows = [dict(row) for row in real_forward_rows if isinstance(row, Mapping)]
        for row in rows:
            row.setdefault("comparability_status", comparability_status)
            row.setdefault("consumes_reliability_metadata", _model_consumes_reliability_metadata(model_spec))
            if dry_run:
                row["status"] = "dry_run"
        _add_scenario_c_accuracy_ratios(rows)
        rows.sort(key=lambda item: (str(item.get("model")), str(item.get("suite")), float(item.get("severity") or 0.0), int(item.get("seed") or 0)))
        return rows
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
                cxd_pairs = suite.get("cxd_pairs")
                if not cxd_pairs:
                    cxd_pairs = [
                        {"gps_condition": gps_condition, "image_condition": image_condition}
                        for gps_condition in suite.get("scenario_c_conditions", [])
                        for image_condition in suite.get("scenario_d_conditions", [])
                    ]
                for pair in cxd_pairs:
                    if isinstance(pair, Mapping):
                        gps_condition = pair.get("gps_condition", {})
                        image_condition = pair.get("image_condition", {})
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
                if bool(suite.get("reused_weight_fusion_diagnostic", False)):
                    rows.extend(
                        _predictive_gps_query_advantage_metric_rows(
                            model_name,
                            model_spec,
                            source,
                            suite,
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
                rows.extend(
                    _predictive_gps_query_advantage_metric_rows(
                        model_name,
                        model_spec,
                        source,
                        suite,
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
    sample_count = int(_float(_first_present(metrics, "sample_count", "n", "total", default=0)))
    top1 = _metric_mapping_topk(metrics, 1)
    top3 = _metric_mapping_topk(metrics, 3)
    top5 = _metric_mapping_topk(metrics, 5)
    dba = _float(_first_present(metrics, "dba", "DBA", "val_adba", "val_beam_dba", default=0.0))
    summary = {
        "model": model_name,
        "source": status,
        "split": split,
        "sample_count": sample_count,
        "top1": top1,
        "top3": top3,
        "top5": top5,
        "dba": dba,
        "mean_beam_index_error": _float(
            metrics.get("mean_beam_index_error", metrics.get("beam_index_mae", metrics.get("beam_mae", 0.0)))
        ),
        "status": status,
    }
    primary_value = _float_or_none(metrics.get(primary))
    if primary_value is None:
        primary_value = _float_or_none(summary.get(primary))
    summary["primary_metric"] = _float(dba if primary_value is None else primary_value)
    return summary


def _first_present(metrics: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = metrics.get(key)
        if value not in (None, ""):
            return value
    return default


def _metric_mapping_topk(metrics: Mapping[str, Any], k: int) -> float:
    topk = metrics.get("topk")
    if isinstance(topk, Mapping):
        value = topk.get(str(k), topk.get(k))
        result = _float_or_none(value)
        if result is not None:
            return result
    return _float(
        _first_present(
            metrics,
            f"top{k}",
            f"top_{k}",
            f"val_top{k}_avg",
            f"val_top{k}_t1",
            "val_acc" if k == 1 else "",
            default=0.0,
        )
    )

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


def _summary_from_real_forward(
    model_name: str,
    model_spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    output_dir: Path,
    primary: str,
    split: str,
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    from kd_sensing.engine.data_factory import shutdown_dataloader_workers
    from kd_sensing.engine.normalization_artifacts import load_normalization_artifacts
    from kd_sensing.engine.optim import build_device, build_model
    from kd_sensing.engine.runtime import configure_torch_runtime_threads, run_model_step, transfer_non_blocking
    from kd_sensing.utils.checkpoint import checkpoint_load_summary, load_model_state
    from kd_sensing.utils.seed import set_seed

    settings = _real_forward_settings(manifest, model_spec)
    cfg = load_config(model_spec["config"], overrides=model_spec.get("overrides") or [])
    cfg.setdefault("data", {}).setdefault("dataloader", {})
    if settings.get("batch_size") is not None:
        cfg["data"]["dataloader"]["test_batch_size"] = int(settings["batch_size"])
    if settings.get("num_workers") is not None:
        cfg["data"]["dataloader"]["test_num_workers"] = int(settings["num_workers"])
    configure_torch_runtime_threads(cfg)
    seed = int(settings.get("seed", cfg.get("experiment", {}).get("seed", 0)) or 0)
    set_seed(seed)
    device = build_device(cfg)
    weights_path = _resolve_optional_checkpoint(model_name, model_spec, settings, warnings)
    checkpoint_metadata = load_checkpoint_metadata(weights_path) if weights_path is not None else None
    model = build_model(cfg["model"]["primary"]).to(device)
    checkpoint_load = None
    if weights_path is not None:
        load_result = load_model_state(
            weights_path,
            model,
            role="real_forward_benchmark",
            map_location=device,
            strict=bool(settings.get("strict_load", cfg.get("checkpoint", {}).get("strict_load", True))),
        )
        checkpoint_load = checkpoint_load_summary(load_result)
    model.eval()

    cache_root = output_dir / "cache" / str(settings.get("cache_subdir", "real_forward"))
    cache_root.mkdir(parents=True, exist_ok=True)
    primary_name = str(primary)
    sample_limit = int(settings.get("sample_count", model_spec.get("sample_count", 0)) or 0)
    dba_delta = float(manifest.get("metrics", {}).get("dba_delta", model_spec.get("dba_delta", 5)))
    distance_mode = str(manifest.get("metrics", {}).get("distance_mode", model_spec.get("distance_mode", "circular")))
    topk_values = tuple(int(k) for k in manifest.get("metrics", {}).get("topk", [1, 3, 5]))
    setup = _real_forward_prediction_setup(cfg)
    condition_specs = _real_forward_condition_specs(manifest)
    perturbation_cache = _perturbation_cache_config(settings, output_dir)
    rows: list[dict[str, Any]] = []
    shard_matrix: list[dict[str, Any]] = []
    clean_by_seed: dict[int, dict[str, Any]] = {}
    dataloader = None

    try:
        for condition in condition_specs:
            condition_seed = int(condition["seed"])
            fingerprint = _real_forward_fingerprint(
                model_name,
                model_spec,
                manifest,
                cfg,
                condition,
                settings=settings,
                weights_path=weights_path,
            )
            cache_path = _real_forward_cache_path(cache_root, model_name, condition, fingerprint=fingerprint)
            summary = _load_real_forward_cache(cache_path, fingerprint=fingerprint) if bool(settings.get("resume", True)) else None
            if summary is None:
                if cache_path.exists():
                    warnings.append(
                        WarningRecord(
                            code="real_forward_cache_stale",
                            message=f"Real-forward cache fingerprint mismatch; recomputing {cache_path.name}.",
                            suite_id=str(condition.get("suite")),
                            condition=str(condition.get("condition")),
                            severity=float(condition.get("severity", 0.0) or 0.0),
                        ).to_dict()
                    )
                cache_key = _perturbation_cache_key(condition, split=split, sample_limit=sample_limit)
                perturbation_index_path = _perturbation_cache_index_path(perturbation_cache, cache_key)
                perturbation_index = (
                    _load_perturbation_cache_index(perturbation_index_path, key=cache_key, cache_cfg=perturbation_cache)
                    if str(perturbation_cache.get("mode")) in {"read", "read_write"}
                    else None
                )
                if perturbation_index is None and str(perturbation_cache.get("mode")) != "read":
                    if dataloader is None:
                        dataset_kwargs = load_normalization_artifacts(checkpoint_metadata)
                        dataloader = _build_real_forward_dataloader(cfg, split, dataset_kwargs)
                summary = _compute_real_forward_condition(
                    model,
                    dataloader,
                    cfg,
                    condition,
                    split=split,
                    device=device,
                    setup=setup,
                    sample_limit=sample_limit,
                    topk_values=topk_values,
                    dba_delta=dba_delta,
                    distance_mode=distance_mode,
                    primary=primary_name,
                    cache_path=cache_path,
                    fingerprint=fingerprint,
                    perturbation_cache=perturbation_cache,
                    perturbation_index=perturbation_index,
                    perturbation_index_path=perturbation_index_path,
                    checkpoint_load=checkpoint_load,
                    checkpoint_metadata=checkpoint_metadata,
                    weights_path=weights_path,
                    non_blocking=transfer_non_blocking(cfg),
                )
            row = _real_forward_metric_row(
                model_name,
                model_spec,
                condition,
                summary,
                primary_name=primary_name,
                clean_primary=None,
            )
            rows.append(row)
            shard_matrix.append(_real_forward_shard_record(model_name, condition, summary, cache_path))
            if str(condition.get("condition")) == "clean":
                clean_by_seed[condition_seed] = row
    finally:
        if dataloader is not None:
            shutdown_dataloader_workers(dataloader)

    for row in rows:
        seed_value = int(row.get("seed", 0) or 0)
        clean_row = clean_by_seed.get(seed_value) or next(iter(clean_by_seed.values()), {})
        clean_primary = _float_or_none(clean_row.get("primary_metric"))
        if clean_primary is None:
            continue
        row["clean_primary_metric"] = clean_primary
        row["clean_delta"] = _float(row.get("primary_metric")) - clean_primary
        row["relative_drop"] = _relative_drop(clean_primary, _float_or_none(row.get("primary_metric")))

    clean_summary = next((row for row in rows if str(row.get("condition")) == "clean"), rows[0] if rows else {})
    summary = {
        "model": model_name,
        "source": "real_forward",
        "split": split,
        "sample_count": int(clean_summary.get("sample_count", 0) or 0),
        "num_beams": int(clean_summary.get("num_beams", model_spec.get("num_beams", 0)) or 0),
        "top1": _float(clean_summary.get("top1")),
        "top3": _float(clean_summary.get("top3")),
        "top5": _float(clean_summary.get("top5")),
        "dba": _float(clean_summary.get("dba")),
        "primary_metric": _float(clean_summary.get("primary_metric")),
        "mean_beam_index_error": _float(clean_summary.get("mean_beam_index_error")),
        "status": "real_forward",
        "evidence_scope": "real_forward",
        "real_forward_rows": rows,
        "shard_matrix": shard_matrix,
        "checkpoint_load": checkpoint_load,
    }
    return summary


def _build_real_forward_dataloader(cfg: dict[str, Any], split: str, dataset_kwargs: dict[str, Any]):
    from kd_sensing.engine.data_factory import build_dataloader, build_split_dataset
    from kd_sensing.engine.data_factory_scalers import normalization_kwargs, prepare_lidar_normalizer

    try:
        dataset = build_split_dataset(cfg, split, **dataset_kwargs)
    except ValueError as exc:
        message = str(exc)
        if split == "train" or "requires a train-fitted" not in message:
            raise
        train_dataset = build_split_dataset(cfg, "train", **dataset_kwargs)
        prepare_lidar_normalizer(cfg, train_dataset)
        resolved_kwargs = {**normalization_kwargs(train_dataset), **dataset_kwargs}
        dataset = build_split_dataset(cfg, split, **resolved_kwargs)
    return build_dataloader(dataset, cfg["data"]["dataloader"], split=split)


def _resolve_optional_checkpoint(
    model_name: str,
    model_spec: Mapping[str, Any],
    settings: Mapping[str, Any],
    warnings: list[dict[str, Any]],
) -> Path | None:
    weights = model_spec.get("weights")
    allow_untrained = bool(settings.get("allow_untrained", False))
    if not weights:
        if allow_untrained:
            warnings.append(
                WarningRecord(
                    code="real_forward_untrained_model",
                    message=f"{model_name} real-forward benchmark is using an untrained model because no weights were declared.",
                ).to_dict()
            )
            return None
        raise BenchmarkManifestError(f"models.{model_name}.weights is required for real-forward evaluation.")
    resolved = resolve_path(str(weights))
    if resolved is not None and resolved.exists():
        return resolved
    if allow_untrained:
        warnings.append(
            WarningRecord(
                code="real_forward_checkpoint_missing_untrained_fallback",
                message=f"{model_name} real-forward checkpoint is missing; using an untrained model.",
            ).to_dict()
        )
        return None
    raise FileNotFoundError(f"Real-forward checkpoint not found for models.{model_name}.weights: {weights}")


def _real_forward_prediction_setup(cfg: Mapping[str, Any]) -> dict[str, Any]:
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), Mapping) else {}
    primary_cfg = model_cfg.get("primary", {}) if isinstance(model_cfg.get("primary"), Mapping) else {}
    data_cfg = cfg.get("data", {}).get("dataset", {}) if isinstance(cfg.get("data"), Mapping) and isinstance(cfg.get("data", {}).get("dataset"), Mapping) else {}
    return {
        "task": str(cfg.get("experiment", {}).get("task", "image")),
        "model_cfg": dict(primary_cfg),
        "seq_length": int(model_cfg.get("seq_length", primary_cfg.get("seq_length", data_cfg.get("seq_len", 1))) or 1),
        "num_pred": int(model_cfg.get("num_pred", primary_cfg.get("num_pred", data_cfg.get("num_pred", 1))) or 1),
        "downsample_ratio": int(model_cfg.get("downsample_ratio", primary_cfg.get("downsample_ratio", 1)) or 1),
    }


def _real_forward_condition_specs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    seeds = [int(seed) for seed in manifest.get("seeds", [42])]
    specs: list[dict[str, Any]] = []
    for seed in seeds:
        specs.append(
            {
                "suite": "clean",
                "suite_type": "gps_clean",
                "condition": "clean",
                "severity": 0.0,
                "severity_unit": "reference",
                "seed": seed,
                "perturbations": [],
                "extra": {},
                "claim_scope": "primary",
            }
        )
        for suite in manifest.get("perturbation_suites", []):
            suite_type = str(suite.get("type"))
            if suite_type == PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE:
                specs.extend(_real_forward_predictive_specs(suite, seed=seed))
            elif suite_type == SCENARIO_C_X_D_SUITE_TYPE:
                specs.extend(_real_forward_cxd_specs(suite, seed=seed))
            else:
                specs.extend(_real_forward_regular_suite_specs(suite, seed=seed))
    return specs


def _real_forward_predictive_specs(suite: Mapping[str, Any], *, seed: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for condition in suite.get("predictive_conditions", []):
        if not isinstance(condition, Mapping):
            continue
        params = condition.get("operator_params", {}) if isinstance(condition.get("operator_params"), Mapping) else {}
        specs.append(
            {
                "suite": suite.get("id", PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE),
                "suite_type": PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
                "condition": condition.get("id", ""),
                "severity": float(condition.get("severity", 0.0) or 0.0),
                "severity_unit": condition.get("severity_unit", suite.get("severity_unit", "stress_severity")),
                "seed": seed,
                "perturbations": [(suite, float(condition.get("severity", 0.0) or 0.0))],
                "operator_params": params,
                "difficulty_digest": _condition_digest(
                    {
                        "suite": suite.get("id"),
                        "stress_suite": condition.get("stress_suite", params.get("stress_suite", "")),
                        "predictive_condition": condition.get("id"),
                        "history_window": suite.get("history_window"),
                        "seed": seed,
                        "params": params,
                    }
                ),
                "claim_scope": condition.get("claim_scope", params.get("claim_scope", "primary")),
                "extra": {
                    "predictive_condition": condition.get("id", ""),
                    "p_severity": float(condition.get("severity", 0.0) or 0.0),
                    "stress_suite": condition.get("stress_suite", params.get("stress_suite", "")),
                    "history_window": int(suite.get("history_window", params.get("history_window", 4)) or 4),
                    "current_frame_missing": bool(params.get("current_frame_missing", False)),
                    "missing_tail_fraction": params.get("missing_tail_fraction", ""),
                    "semantic_occlusion": bool(params.get("semantic_occlusion", False)),
                    "plausible_wrong_gps": bool(params.get("plausible_wrong_gps", False)),
                    "gps_noise_mode": params.get("gps_noise_mode", ""),
                    "gps_jitter_std": params.get("gps_jitter_std", ""),
                    "novel_weather": bool(params.get("novel_weather", False)),
                    "counterfactual_input_intervention": bool(params.get("plausible_wrong_gps", False)),
                },
            }
        )
    advantage = suite.get("gps_query_advantage_slice", {})
    if isinstance(advantage, Mapping) and bool(advantage.get("enabled", False)):
        for condition in advantage.get("conditions", []):
            if isinstance(condition, Mapping):
                specs.append(_real_forward_advantage_spec(suite, condition, seed=seed))
        for condition in advantage.get("combined_conditions", []):
            if isinstance(condition, Mapping):
                specs.append(_real_forward_advantage_cxd_spec(suite, condition, seed=seed))
    return specs


def _real_forward_advantage_spec(
    suite: Mapping[str, Any],
    condition: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    params = condition.get("operator_params", {}) if isinstance(condition.get("operator_params"), Mapping) else {}
    severity = float(condition.get("severity", 0.0) or 0.0)
    predictive_suite = {**dict(suite), "type": PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE, "preset": "canonical"}
    return {
        "suite": f"{suite.get('id', PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE)}:gps_query_advantage",
        "suite_type": GPS_QUERY_ADVANTAGE_SLICE_TYPE,
        "condition": condition.get("id", ""),
        "severity": severity,
        "severity_unit": "gps_query_advantage_level",
        "seed": seed,
        "perturbations": [(predictive_suite, severity)],
        "operator_params": params,
        "difficulty_digest": _condition_digest(
            {
                "suite": suite.get("id"),
                "advantage_condition": condition.get("id"),
                "history_window": suite.get("history_window"),
                "seed": seed,
                "params": params,
            }
        ),
        "claim_scope": "mechanism_diagnostic",
        "extra": {
            "evidence_slice": "gps_query_advantage",
            "advantage_condition": condition.get("id", ""),
            "advantage_family": condition.get("advantage_family", "hard_negative"),
            "a_severity": severity,
            "fallback_count": int(params.get("fallback_count", params.get("expected_fallback_count", 0)) or 0),
            "history_window": int(suite.get("history_window", params.get("history_window", 4)) or 4),
            "history_source_range_policy": "not_applicable",
            "source_history_range_field": "",
            "future_leak_check": "not_applicable",
            "visual_ambiguous_peer": bool(params.get("visual_ambiguous_peer", False)),
            "beam_offset_constrained_wrong_gps": bool(params.get("beam_offset_constrained_wrong_gps", False)),
            "min_beam_offset": params.get("min_beam_offset", params.get("beam_offset_min", "")),
            "scene_constraint": params.get("scene_constraint", ""),
            "counterfactual_input_intervention": bool(params.get("plausible_wrong_gps", False)),
        },
    }


def _real_forward_advantage_cxd_spec(
    suite: Mapping[str, Any],
    condition: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    gps_condition = condition.get("gps_condition", {}) if isinstance(condition.get("gps_condition"), Mapping) else {}
    image_condition = condition.get("image_condition", {}) if isinstance(condition.get("image_condition"), Mapping) else {}
    c_suite = {
        "id": f"{suite.get('id', 'predictive')}:advantage:{condition.get('id', 'cxd')}:scenario_c",
        "type": SCENARIO_C_SUITE_TYPE,
        "conditions": [gps_condition],
        "fallback": gps_condition.get("fallback", "forward_fill"),
    }
    d_suite = {
        "id": f"{suite.get('id', 'predictive')}:advantage:{condition.get('id', 'cxd')}:scenario_d",
        "type": SCENARIO_D_SUITE_TYPE,
        "conditions": [image_condition],
        "fallback": "identity",
    }
    severity = float(image_condition.get("severity", condition.get("severity", 0.0)) or 0.0)
    return {
        "suite": f"{suite.get('id', PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE)}:gps_query_advantage",
        "suite_type": GPS_QUERY_ADVANTAGE_SLICE_TYPE,
        "condition": condition.get("id", ""),
        "severity": severity,
        "severity_unit": "scenario_c_x_d_level",
        "seed": seed,
        "perturbations": [
            (c_suite, float(gps_condition.get("severity", 0.0) or 0.0)),
            (d_suite, severity),
        ],
        "operator_params": {
            "gps_condition": gps_condition,
            "image_operator_params": image_condition.get("operator_params", {}),
        },
        "difficulty_digest": _condition_digest(
            {
                "suite": suite.get("id"),
                "advantage_condition": condition.get("id"),
                "gps_condition": gps_condition.get("id"),
                "image_condition": image_condition.get("id"),
                "seed": seed,
            }
        ),
        "claim_scope": "mechanism_diagnostic",
        "extra": {
            "evidence_slice": "gps_query_advantage",
            "advantage_condition": condition.get("id", ""),
            "advantage_family": condition.get("advantage_family", "combined_cxd"),
            "gps_condition": gps_condition.get("id", ""),
            "image_condition": image_condition.get("id", ""),
            "c_severity": float(gps_condition.get("severity", 0.0) or 0.0),
            "d_severity": severity,
            "fallback_count": 0,
            "history_window": int(suite.get("history_window", 4) or 4),
            "history_source_range_policy": condition.get("history_source_range_policy", "strictly_past"),
            "source_history_range_field": condition.get("source_history_range_field", "gps_source_index"),
            "future_leak_check": "required",
            "counterfactual_input_intervention": True,
        },
    }


def _real_forward_cxd_specs(suite: Mapping[str, Any], *, seed: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    cxd_pairs = suite.get("cxd_pairs")
    if not cxd_pairs:
        cxd_pairs = [
            {"gps_condition": gps_condition, "image_condition": image_condition}
            for gps_condition in suite.get("scenario_c_conditions", [])
            for image_condition in suite.get("scenario_d_conditions", [])
        ]
    for pair in cxd_pairs:
        if not isinstance(pair, Mapping):
            continue
        gps_condition = pair.get("gps_condition", {})
        image_condition = pair.get("image_condition", {})
        if not isinstance(gps_condition, Mapping) or not isinstance(image_condition, Mapping):
            continue
        c_suite = {"id": f"{suite.get('id')}:scenario_c", "type": SCENARIO_C_SUITE_TYPE, "conditions": [gps_condition]}
        d_suite = {"id": f"{suite.get('id')}:scenario_d", "type": SCENARIO_D_SUITE_TYPE, "conditions": [image_condition]}
        condition_id = f"{gps_condition.get('id')}+{image_condition.get('id')}"
        specs.append(
            {
                "suite": suite.get("id", SCENARIO_C_X_D_SUITE_TYPE),
                "suite_type": SCENARIO_C_X_D_SUITE_TYPE,
                "condition": condition_id,
                "severity": float(image_condition.get("severity", 0.0) or 0.0),
                "severity_unit": "scenario_c_x_d_level",
                "seed": seed,
                "perturbations": [
                    (c_suite, float(gps_condition.get("severity", 0.0) or 0.0)),
                    (d_suite, float(image_condition.get("severity", 0.0) or 0.0)),
                ],
                "operator_params": {
                    "gps_condition": gps_condition,
                    "image_operator_params": image_condition.get("operator_params", {}),
                },
                "difficulty_digest": _condition_digest(
                    {
                        "suite": suite.get("id"),
                        "gps_condition": gps_condition.get("id"),
                        "image_condition": image_condition.get("id"),
                        "seed": seed,
                    }
                ),
                "claim_scope": "mechanism_diagnostic" if bool(suite.get("reused_weight_fusion_diagnostic", False)) else "primary",
                "extra": {
                    "gps_condition": gps_condition.get("id", ""),
                    "image_condition": image_condition.get("id", ""),
                    "c_severity": float(gps_condition.get("severity", 0.0) or 0.0),
                    "d_severity": float(image_condition.get("severity", 0.0) or 0.0),
                },
            }
        )
    if bool(suite.get("reused_weight_fusion_diagnostic", False)):
        advantage = suite.get("gps_query_advantage_slice", {})
        if isinstance(advantage, Mapping) and bool(advantage.get("enabled", False)):
            for condition in advantage.get("conditions", []):
                if isinstance(condition, Mapping):
                    specs.append(_real_forward_advantage_spec(suite, condition, seed=seed))
            for condition in advantage.get("combined_conditions", []):
                if isinstance(condition, Mapping):
                    specs.append(_real_forward_advantage_cxd_spec(suite, condition, seed=seed))
    return specs


def _real_forward_regular_suite_specs(suite: Mapping[str, Any], *, seed: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    suite_type = str(suite.get("type"))
    suite_max = max([float(value) for value in suite.get("severities", [0.0])] or [0.0] + [1.0])
    del suite_max
    for severity in suite.get("severities", [0.0]):
        severity_value = float(severity)
        extra: dict[str, Any] = {}
        condition = str(suite.get("condition", _default_condition(suite_type)))
        if suite_type == SCENARIO_C_SUITE_TYPE:
            scenario_c = _scenario_c_condition_for_severity(suite, severity_value)
            condition = str(scenario_c.get("id", condition))
            extra.update(_scenario_c_metric_columns(scenario_c, model_spec={}))
        if suite_type == SCENARIO_D_SUITE_TYPE:
            scenario_d = _scenario_d_condition_for_severity(suite, severity_value)
            condition = str(scenario_d.get("id", condition))
            extra.update(_scenario_d_metric_columns(scenario_d, seed=seed))
        specs.append(
            {
                "suite": suite.get("id", suite_type),
                "suite_type": suite_type,
                "condition": condition,
                "severity": severity_value,
                "severity_unit": suite.get("severity_unit", _default_severity_unit(suite_type)),
                "seed": seed,
                "perturbations": [(suite, severity_value)],
                "operator_params": {},
                "difficulty_digest": _condition_digest(
                    {
                        "suite": suite.get("id"),
                        "suite_type": suite_type,
                        "condition": condition,
                        "severity": severity_value,
                        "seed": seed,
                    }
                ),
                "claim_scope": "primary",
                "extra": extra,
            }
        )
    return specs


def _real_forward_fingerprint(
    model_name: str,
    model_spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
    cfg: Mapping[str, Any],
    condition: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
    weights_path: Path | None,
) -> str:
    config_path = resolve_path(str(model_spec.get("config", "")))
    config_digest = ""
    if config_path is not None and config_path.exists():
        config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    checkpoint_stat = {}
    if weights_path is not None and weights_path.exists():
        stat = weights_path.stat()
        checkpoint_stat = {"path": str(weights_path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    payload = {
        "runner_version": RUNNER_VERSION,
        "mode": "real_forward",
        "model": model_name,
        "group": model_spec.get("group", ""),
        "config_digest": config_digest,
        "checkpoint": checkpoint_stat,
        "split": model_spec.get("split", manifest.get("protocol", {}).get("split", "test")),
        "condition": condition,
        "sample_count": settings.get("sample_count", model_spec.get("sample_count", "")),
        "metrics": manifest.get("metrics", {}),
        "model_primary": cfg.get("model", {}).get("primary", {}) if isinstance(cfg.get("model"), Mapping) else {},
    }
    return _sha256_text(json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":")))[:24]


def _real_forward_cache_path(cache_root: Path, model_name: str, condition: Mapping[str, Any], *, fingerprint: str) -> Path:
    stem = "_".join(
        _safe_filename_part(part)
        for part in (
            model_name,
            condition.get("suite_type", ""),
            condition.get("condition", ""),
            f"s{condition.get('severity', 0)}",
            f"seed{condition.get('seed', 0)}",
            fingerprint[:10],
        )
    )
    return cache_root / f"{stem}.npz"


def _safe_filename_part(value: Any) -> str:
    text = str(value).strip().replace("+", "_plus_").replace(":", "_")
    output = []
    for char in text:
        output.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
    return "".join(output).strip("._") or "item"


def _load_real_forward_cache(path: Path, *, fingerprint: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = np.load(path, allow_pickle=False)
        metadata_json = str(payload["metadata_json"].item())
        metadata = json.loads(metadata_json)
    except Exception:
        return None
    if str(metadata.get("fingerprint")) != str(fingerprint):
        return None
    return _real_forward_summary_from_arrays(
        np.asarray(payload["logits"], dtype=np.float32),
        np.asarray(payload["labels"], dtype=np.int64),
        metadata=metadata,
        primary=str(metadata.get("primary_metric_name", DEFAULT_PRIMARY_METRIC)),
        topk_values=tuple(int(k) for k in metadata.get("topk_values", [1, 3, 5])),
        dba_delta=float(metadata.get("dba_delta", 5)),
        distance_mode=str(metadata.get("distance_mode", "circular")),
        cache_status="hit",
    )


def _compute_real_forward_condition(
    model,
    dataloader,
    cfg: Mapping[str, Any],
    condition: Mapping[str, Any],
    *,
    split: str,
    device: torch.device,
    setup: Mapping[str, Any],
    sample_limit: int,
    topk_values: tuple[int, ...],
    dba_delta: float,
    distance_mode: str,
    primary: str,
    cache_path: Path,
    fingerprint: str,
    perturbation_cache: Mapping[str, Any],
    perturbation_index: Mapping[str, Any] | None,
    perturbation_index_path: Path,
    checkpoint_load: Mapping[str, Any] | None,
    checkpoint_metadata: Mapping[str, Any] | None,
    weights_path: Path | None,
    non_blocking: bool,
) -> dict[str, Any]:
    from kd_sensing.engine.runtime import run_model_step

    logits_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    sample_ids: list[str] = []
    perturbation_warnings: list[dict[str, Any]] = []
    perturbation_cache_status = "off"
    perturbation_cache_shards: list[dict[str, Any]] = []
    diagnostics = _RealForwardDiagnostics()
    seen = 0
    with torch.no_grad():
        if perturbation_index is not None:
            batch_iter = _iter_perturbation_cache(perturbation_index)
            perturbation_cache_status = "hit"
        else:
            if dataloader is None:
                raise BenchmarkManifestError(f"Perturbation cache missing for condition {condition.get('condition')}.")
            batch_iter = ((batch, _batch_sample_ids(batch, offset=seen), []) for batch in dataloader)
            if str(perturbation_cache.get("mode")) in {"write", "read_write"}:
                perturbation_cache_status = "written"
            elif str(perturbation_cache.get("mode")) == "read":
                perturbation_cache_status = "miss"
        for batch, ids, cached_warnings in batch_iter:
            batch_size = _batch_size(batch) or 0
            if sample_limit > 0 and seen >= sample_limit:
                break
            if sample_limit > 0 and batch_size > max(sample_limit - seen, 0):
                batch = _slice_batch(batch, sample_limit - seen)
                batch_size = _batch_size(batch) or 0
                ids = ids[:batch_size]
            if perturbation_index is not None:
                perturbed = batch
                perturbation_warnings.extend(cached_warnings)
            else:
                perturbed = batch
                batch_warnings: list[dict[str, Any]] = []
                for suite, severity in condition.get("perturbations", []):
                    perturbed, suite_warnings = apply_benchmark_perturbation(
                        perturbed,
                        suite,
                        severity=float(severity),
                        seed=int(condition.get("seed", 0)),
                        sample_ids=ids,
                    )
                    batch_warnings.extend(suite_warnings)
                perturbation_warnings.extend(batch_warnings)
                if str(perturbation_cache.get("mode")) in {"write", "read_write"}:
                    perturbation_cache_shards.append(
                        _write_perturbation_cache_shard(
                            perturbation_index_path,
                            shard_index=len(perturbation_cache_shards),
                            key=_perturbation_cache_key(condition, split=split, sample_limit=sample_limit),
                            condition=condition,
                            batch=perturbed,
                            sample_ids=ids,
                            warnings=batch_warnings,
                        )
                    )
            result = run_model_step(
                model,
                str(setup.get("task", cfg.get("experiment", {}).get("task", "image"))),
                perturbed,
                model_cfg=dict(setup.get("model_cfg", {})),
                seq_length=int(setup.get("seq_length", 1)),
                num_pred=int(setup.get("num_pred", 1)),
                device=device,
                downsample_ratio=int(setup.get("downsample_ratio", 1)),
                non_blocking=non_blocking,
            )
            if result.labels is None:
                raise BenchmarkManifestError("Real-forward benchmark requires target labels in every batch.")
            logits = result.logits.detach().cpu()
            labels = result.labels.detach().cpu().to(dtype=torch.long)
            logits_parts.append(logits.numpy().astype(np.float32, copy=False))
            label_parts.append(labels.numpy().astype(np.int64, copy=False))
            sample_ids.extend(ids[: int(labels.shape[0])])
            diagnostics.update(result.model_output.diagnostics, logits=logits, labels=labels)
            seen += int(labels.shape[0])

    if not logits_parts or not label_parts:
        raise BenchmarkManifestError("Real-forward benchmark produced no samples.")
    if perturbation_cache_shards:
        _write_perturbation_cache_index(
            perturbation_index_path,
            key=_perturbation_cache_key(condition, split=split, sample_limit=sample_limit),
            condition=condition,
            split=split,
            sample_limit=sample_limit,
            shards=perturbation_cache_shards,
        )
    logits_np = np.concatenate(logits_parts, axis=0)
    labels_np = np.concatenate(label_parts, axis=0)
    metadata = {
        "fingerprint": fingerprint,
        "evidence_scope": "real_forward",
        "status": "real_forward",
        "primary_metric_name": primary,
        "topk_values": list(topk_values),
        "dba_delta": float(dba_delta),
        "distance_mode": distance_mode,
        "condition": _json_ready(condition),
        "sample_count": int(logits_np.shape[0]),
        "checkpoint": str(weights_path) if weights_path is not None else "",
        "checkpoint_load": _json_ready(checkpoint_load),
        "checkpoint_metadata_digest": _condition_digest(checkpoint_metadata or {}),
        "perturbation_warning_count": len(perturbation_warnings),
        "perturbation_warnings": perturbation_warnings[:50],
        "perturbation_cache": {
            "schema": PERTURBATION_CACHE_SCHEMA_VERSION,
            "mode": perturbation_cache.get("mode", "off"),
            "status": perturbation_cache_status,
            "index": str(perturbation_index_path) if str(perturbation_cache.get("mode")) != "off" else "",
            "shards": len(perturbation_cache_shards)
            if perturbation_cache_shards
            else len(perturbation_index.get("shards", [])) if isinstance(perturbation_index, Mapping) else 0,
        },
        "diagnostics": diagnostics.summary(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        logits=logits_np,
        labels=labels_np,
        sample_ids=np.asarray(sample_ids, dtype="U128"),
        metadata_json=np.asarray(json.dumps(_json_ready(metadata), sort_keys=True)),
    )
    return _real_forward_summary_from_arrays(
        logits_np,
        labels_np,
        metadata=metadata,
        primary=primary,
        topk_values=topk_values,
        dba_delta=dba_delta,
        distance_mode=distance_mode,
        cache_status="computed",
    )


class _RealForwardDiagnostics:
    def __init__(self) -> None:
        self.sample_count = 0
        self.candidate_covered = 0
        self.candidate_count_sum = 0.0
        self.changed_sum = 0.0
        self.fallback_sum = 0.0
        self.gate_sum = 0.0
        self.gate_count = 0
        self.residual_sum = 0.0
        self.residual_count = 0
        self.no_regret_violations = 0
        self.beneficial = 0
        self.neutral = 0
        self.harmful = 0
        self.condition_id_consumed = False
        self.selected_source_counts: dict[str, int] = {}
        self.source_names: list[str] = []

    def update(self, diagnostics: Mapping[str, Any], *, logits: torch.Tensor, labels: torch.Tensor) -> None:
        sample_count = int(labels.numel())
        self.sample_count += sample_count
        if bool(diagnostics.get("condition_id_consumed", False)):
            self.condition_id_consumed = True
        source_names = diagnostics.get("source_names")
        if isinstance(source_names, (list, tuple)) and source_names:
            self.source_names = [str(item) for item in source_names]
        candidate_ids = _diagnostic_time_tensor(diagnostics.get("candidate_ids"), int(labels.shape[1]))
        if candidate_ids is not None:
            candidate_ids = candidate_ids.detach().cpu().to(dtype=torch.long)
            target_in_candidates = candidate_ids.eq(labels.unsqueeze(-1)) & candidate_ids.ge(0)
            self.candidate_covered += int(target_in_candidates.any(dim=-1).sum().item())
            self.candidate_count_sum += float(candidate_ids.ge(0).sum(dim=-1).float().sum().item())
        changed = _diagnostic_time_tensor(diagnostics.get("changed_from_anchor"), int(labels.shape[1]))
        if changed is not None:
            self.changed_sum += float(changed.detach().cpu().to(dtype=torch.float32).sum().item())
        fallback = _diagnostic_time_tensor(diagnostics.get("fallback_to_anchor"), int(labels.shape[1]))
        if fallback is not None:
            self.fallback_sum += float(fallback.detach().cpu().to(dtype=torch.float32).sum().item())
        gate = _diagnostic_time_tensor(diagnostics.get("gate_confidence"), int(labels.shape[1]))
        if gate is not None:
            values = gate.detach().cpu().to(dtype=torch.float32)
            self.gate_sum += float(values.sum().item())
            self.gate_count += int(values.numel())
        residual = _diagnostic_time_tensor(diagnostics.get("residual_magnitude"), int(labels.shape[1]))
        if residual is not None:
            values = residual.detach().cpu().to(dtype=torch.float32)
            self.residual_sum += float(values.sum().item())
            self.residual_count += int(values.numel())
        selected = _diagnostic_time_tensor(diagnostics.get("selected_source"), int(labels.shape[1]))
        if selected is not None:
            for item in selected.detach().cpu().flatten().tolist():
                index = int(item)
                name = self.source_names[index] if 0 <= index < len(self.source_names) else str(index)
                self.selected_source_counts[name] = self.selected_source_counts.get(name, 0) + 1
        anchor_logits = _diagnostic_time_tensor(diagnostics.get("anchor_logits"), int(labels.shape[1]))
        if anchor_logits is not None:
            anchor_logits = anchor_logits.detach().cpu().to(dtype=torch.float32)
            rerank_pred = logits.argmax(dim=-1)
            anchor_pred = anchor_logits.argmax(dim=-1)
            anchor_distance = _class_distance_array(anchor_pred, labels, int(logits.shape[-1]))
            rerank_distance = _class_distance_array(rerank_pred, labels, int(logits.shape[-1]))
            delta = anchor_distance - rerank_distance
            self.beneficial += int((delta > 0).sum().item())
            self.harmful += int((delta < 0).sum().item())
            self.neutral += int((delta == 0).sum().item())
            self.no_regret_violations += int((anchor_distance == 0).logical_and(rerank_distance != 0).sum().item())

    def summary(self) -> dict[str, Any]:
        total = max(self.sample_count, 1)
        return {
            "candidate_recall": float(self.candidate_covered / total),
            "candidate_count_mean": float(self.candidate_count_sum / total),
            "rerank_changed_top1_rate": float(self.changed_sum / total),
            "rerank_fallback_rate": float(self.fallback_sum / total),
            "gate_confidence_mean": float(self.gate_sum / max(self.gate_count, 1)),
            "residual_magnitude_mean": float(self.residual_sum / max(self.residual_count, 1)),
            "no_regret_violation_count": int(self.no_regret_violations),
            "beneficial_rerank_count": int(self.beneficial),
            "neutral_rerank_count": int(self.neutral),
            "harmful_rerank_count": int(self.harmful),
            "condition_id_consumed": bool(self.condition_id_consumed),
            "selected_source_counts": dict(sorted(self.selected_source_counts.items())),
            "source_names": list(self.source_names),
        }


def _diagnostic_time_tensor(value: Any, target_steps: int) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        return None
    if value.ndim < 2:
        return value
    if int(value.shape[1]) == int(target_steps):
        return value
    if int(value.shape[1]) > int(target_steps):
        return value[:, -int(target_steps) :, ...]
    return value


def _class_distance_array(prediction: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    absolute = (prediction.to(dtype=torch.long) - target.to(dtype=torch.long)).abs()
    return torch.minimum(absolute, torch.as_tensor(num_classes, dtype=absolute.dtype) - absolute)


def _slice_batch(batch: Mapping[str, Any], count: int) -> dict[str, Any]:
    return {key: _slice_value(value, count) for key, value in batch.items()}


def _slice_value(value: Any, count: int) -> Any:
    if torch.is_tensor(value) and value.ndim > 0:
        return value[:count]
    if isinstance(value, np.ndarray) and value.ndim > 0:
        return value[:count]
    if isinstance(value, list):
        return value[:count]
    if isinstance(value, tuple):
        return tuple(value[:count])
    if isinstance(value, Mapping):
        return {key: _slice_value(item, count) for key, item in value.items()}
    return value


def _batch_sample_ids(batch: Mapping[str, Any], *, offset: int) -> list[str]:
    size = _batch_size(batch) or 0
    metadata_ids = _sample_ids_from_metadata(_metadata_rows(batch.get("metadata")), batch_size=size)
    if metadata_ids:
        return [str(item) for item in metadata_ids]
    return [str(offset + index) for index in range(size)]


def _real_forward_summary_from_arrays(
    logits_np: np.ndarray,
    labels_np: np.ndarray,
    *,
    metadata: Mapping[str, Any],
    primary: str,
    topk_values: tuple[int, ...],
    dba_delta: float,
    distance_mode: str,
    cache_status: str,
) -> dict[str, Any]:
    logits = torch.tensor(logits_np, dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.long)
    if logits.ndim == 2:
        logits = logits.unsqueeze(1)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    topk, _ = calculate_topk_accuracy(logits, labels, k_values=topk_values)
    dba_values = calculate_dba_score(logits, labels, delta=float(dba_delta), distance_mode=distance_mode)
    top_values = {f"top{int(k)}": _mean_array(topk[k]) for k in topk_values}
    summary = {
        "source": "real_forward",
        "status": "real_forward",
        "evidence_scope": "real_forward",
        "cache_status": cache_status,
        "sample_count": int(logits.shape[0]),
        "num_beams": int(logits.shape[-1]),
        "top1": float(top_values.get("top1", 0.0)),
        "top3": float(top_values.get("top3", top_values.get("top1", 0.0))),
        "top5": float(top_values.get("top5", top_values.get("top3", top_values.get("top1", 0.0)))),
        "dba": _mean_array(dba_values),
        "mean_beam_index_error": _mean_beam_index_error(logits, labels, distance_mode=distance_mode),
        "metadata": dict(metadata),
        **top_values,
    }
    summary["primary_metric"] = float(summary.get(primary, summary["dba"]))
    perturbation_cache = metadata.get("perturbation_cache", {}) if isinstance(metadata.get("perturbation_cache"), Mapping) else {}
    if perturbation_cache:
        summary["perturbation_cache_status"] = perturbation_cache.get("status", "")
        summary["perturbation_cache_index"] = perturbation_cache.get("index", "")
        summary["perturbation_cache_shards"] = perturbation_cache.get("shards", "")
    diagnostics = metadata.get("diagnostics", {}) if isinstance(metadata.get("diagnostics"), Mapping) else {}
    summary.update({key: value for key, value in diagnostics.items() if key not in {"selected_source_counts", "source_names"}})
    selected_source_counts = diagnostics.get("selected_source_counts") if isinstance(diagnostics, Mapping) else None
    if isinstance(selected_source_counts, Mapping):
        total = sum(int(value) for value in selected_source_counts.values()) or 1
        for source, count in selected_source_counts.items():
            summary[f"selected_source_{source}_rate"] = float(int(count) / total)
    return summary


def _mean_array(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64)
    return float(arr.mean()) if arr.size else 0.0


def _mean_beam_index_error(logits: torch.Tensor, labels: torch.Tensor, *, distance_mode: str) -> float:
    pred = logits.argmax(dim=-1)
    valid = labels.ne(-100)
    if not bool(valid.any().item()):
        return 0.0
    if str(distance_mode) == "circular":
        distance = _class_distance_array(pred, labels, int(logits.shape[-1])).to(dtype=torch.float32)
    else:
        distance = (pred.to(dtype=torch.long) - labels.to(dtype=torch.long)).abs().to(dtype=torch.float32)
    return float(distance[valid].mean().item())


def _real_forward_metric_row(
    model_name: str,
    model_spec: Mapping[str, Any],
    condition: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    primary_name: str,
    clean_primary: float | None,
) -> dict[str, Any]:
    primary_value = _float(summary.get("primary_metric"))
    row = {
        "model": model_name,
        "group": model_spec.get("group", ""),
        "suite": condition.get("suite", ""),
        "suite_type": condition.get("suite_type", ""),
        "condition": condition.get("condition", ""),
        "severity": float(condition.get("severity", 0.0) or 0.0),
        "severity_unit": condition.get("severity_unit", ""),
        "seed": int(condition.get("seed", 0) or 0),
        "split": model_spec.get("split", ""),
        "sample_count": int(summary.get("sample_count", 0) or 0),
        "num_beams": int(summary.get("num_beams", model_spec.get("num_beams", 0)) or 0),
        "difficulty_digest": condition.get("difficulty_digest", _condition_digest(condition)),
        "operator_params": condition.get("operator_params", {}),
        "primary_metric_name": primary_name,
        "primary_metric": primary_value,
        "clean_primary_metric": clean_primary if clean_primary is not None else "",
        "clean_delta": "" if clean_primary is None else primary_value - clean_primary,
        "relative_drop": "" if clean_primary is None else _relative_drop(clean_primary, primary_value),
        "top1": _float(summary.get("top1")),
        "top3": _float(summary.get("top3")),
        "top5": _float(summary.get("top5")),
        "dba": _float(summary.get("dba")),
        "mean_beam_index_error": _float(summary.get("mean_beam_index_error")),
        "consumes_reliability_metadata": _model_consumes_reliability_metadata(model_spec),
        "status": "real_forward",
        "evidence_scope": "real_forward",
        "cache_status": summary.get("cache_status", ""),
        "perturbation_cache_status": summary.get("perturbation_cache_status", ""),
        "claim_scope": condition.get("claim_scope", "primary"),
        "condition_id_consumed": bool(summary.get("condition_id_consumed", False)),
        "target_leakage_guard": not bool(summary.get("condition_id_consumed", False)),
    }
    row.update(condition.get("extra", {}) if isinstance(condition.get("extra"), Mapping) else {})
    for key in (
        "candidate_recall",
        "candidate_count_mean",
        "rerank_changed_top1_rate",
        "rerank_fallback_rate",
        "gate_confidence_mean",
        "residual_magnitude_mean",
        "no_regret_violation_count",
        "beneficial_rerank_count",
        "neutral_rerank_count",
        "harmful_rerank_count",
        "selected_source_anchor_rate",
        "selected_source_prior_rate",
        "selected_source_neighborhood_rate",
        "selected_source_teacher_rate",
    ):
        if key in summary:
            row[key] = summary[key]
    return row


def _real_forward_shard_record(
    model_name: str,
    condition: Mapping[str, Any],
    summary: Mapping[str, Any],
    cache_path: Path,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "suite": condition.get("suite", ""),
        "suite_type": condition.get("suite_type", ""),
        "condition": condition.get("condition", ""),
        "severity": condition.get("severity", ""),
        "seed": condition.get("seed", ""),
        "cache_path": str(cache_path),
        "cache_status": summary.get("cache_status", ""),
        "perturbation_cache_status": summary.get("perturbation_cache_status", ""),
        "perturbation_cache_index": summary.get("perturbation_cache_index", ""),
        "evidence_scope": "real_forward",
        "sample_count": int(summary.get("sample_count", 0) or 0),
        "dba": summary.get("dba", ""),
        "top1": summary.get("top1", ""),
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
