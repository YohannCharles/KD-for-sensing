from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np


GEOMETRY_PRIOR_OUTPUT_FILES = {
    "prior_quality": "results/geometry_prior_quality.csv",
    "branch_weights": "results/geometry_prior_branch_weights.csv",
    "strict_comparison": "results/geometry_prior_strict_comparison.csv",
    "claim_gate": "results/geometry_prior_claim_gate.json",
    "diagnostics_bundle": "results/geometry_prior_diagnostics_bundle_manifest.json",
}


def aggregate_geometry_prior_diagnostics(
    rows: Iterable[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    source_rows = [dict(row) for row in rows]
    return {
        "prior_quality": _prior_quality_rows(source_rows),
        "branch_weights": _branch_weight_rows(source_rows),
        "strict_comparison": _strict_comparison_rows(source_rows, manifest=manifest or {}),
    }


def build_geometry_prior_claim_gate(
    rows: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    source_rows = [dict(row) for row in rows]
    cfg = manifest.get("geometry_prior_claim_gate", manifest.get("geometry_prior", {}))
    cfg = cfg if isinstance(cfg, Mapping) else {}
    baseline_group = str(cfg.get("baseline_group", "resnet_image_gps"))
    gps_query_group = str(cfg.get("gps_query_baseline_group", "jepa_gps_query_pool"))
    threshold = float(cfg.get("clean_regression_threshold_dba", 0.02))
    claim_margin = float(cfg.get("claim_margin_dba", 0.0))
    candidate_groups = {
        str(group)
        for group in cfg.get(
            "candidate_groups",
            [
                "geometry_prior_fusion",
                "geometry_prior_dba_aware",
                "geometry_prior_teacher_guided",
                "geometry_prior_mixed_curriculum",
                "safe_residual_beam_rerank_fusion",
                "safe_residual_rerank_fusion",
                "real_perturbation_residual_rerank_fusion",
            ],
        )
    }
    require_real_forward = bool(cfg.get("require_real_forward_perturbations", True))
    clean_rows = [row for row in source_rows if _is_clean_condition(row)]
    baseline_clean = _mean_metric(row for row in clean_rows if str(row.get("group")) == baseline_group)
    gps_query_clean = _mean_metric(row for row in clean_rows if str(row.get("group")) == gps_query_group)
    candidate_clean = {
        group: _mean_metric(row for row in clean_rows if str(row.get("group")) == group)
        for group in sorted(candidate_groups)
    }
    candidate_clean = {group: value for group, value in candidate_clean.items() if value is not None}
    if baseline_clean is None:
        return {
            "claim": "geometry_prior_beam_fusion",
            "claim_status": "unavailable",
            "reason": "missing_clean_baseline",
            "clean_regression_threshold_dba": threshold,
            "baseline_group": baseline_group,
        }
    if not candidate_clean:
        return {
            "claim": "geometry_prior_beam_fusion",
            "claim_status": "pending",
            "reason": "missing_geometry_prior_candidate",
            "clean_regression_threshold_dba": threshold,
            "baseline_group": baseline_group,
            "baseline_clean_dba": baseline_clean,
        }
    candidate_statuses: list[dict[str, Any]] = []
    any_pass = False
    any_failed_clean = False
    for group, clean_value in candidate_clean.items():
        clean_delta = clean_value - baseline_clean
        clean_failed = clean_delta < -threshold
        candidate_rows = [row for row in source_rows if str(row.get("group")) == group]
        baseline_rows = [row for row in source_rows if str(row.get("group")) == baseline_group]
        candidate_overall = _mean_metric(candidate_rows)
        baseline_overall = _mean_metric(baseline_rows)
        overall_delta = (
            None if candidate_overall is None or baseline_overall is None else candidate_overall - baseline_overall
        )
        comparable = all(str(row.get("comparability_status", "passed")) == "passed" for row in candidate_rows)
        perturbation_rows = [row for row in candidate_rows if not _is_clean_condition(row)]
        delegated_clean_only = any(str(row.get("status", "")) == "delegated_evaluate" for row in perturbation_rows)
        real_forward_missing = bool(require_real_forward) and (
            not perturbation_rows or any(not _row_has_real_forward_evidence(row) for row in perturbation_rows)
        )
        status = "pending"
        reason = "needs_strict_margin"
        if clean_failed:
            status = "failed"
            reason = "clean_regression_exceeds_threshold"
            any_failed_clean = True
        elif not comparable:
            status = "pending"
            reason = "comparability_unavailable"
        elif delegated_clean_only or real_forward_missing:
            status = "pending"
            reason = (
                "delegated_clean_only_perturbations_not_real_forward"
                if delegated_clean_only
                else "real_forward_perturbation_evidence_missing"
            )
        elif overall_delta is not None and overall_delta >= claim_margin:
            status = "pass"
            reason = "strict_margin_and_clean_gate_pass"
            any_pass = True
        candidate_statuses.append(
            {
                "group": group,
                "clean_dba": clean_value,
                "baseline_clean_dba": baseline_clean,
                "gps_query_clean_dba": gps_query_clean if gps_query_clean is not None else "",
                "clean_delta_vs_baseline_dba": clean_delta,
                "overall_dba": candidate_overall if candidate_overall is not None else "",
                "baseline_overall_dba": baseline_overall if baseline_overall is not None else "",
                "overall_delta_vs_baseline_dba": overall_delta if overall_delta is not None else "",
                "clean_gate_pass": not clean_failed,
                "real_perturbation_forward": not real_forward_missing,
                "real_forward_required": require_real_forward,
                "claim_status": status,
                "reason": reason,
            }
        )
    if any_failed_clean:
        gate_status = "failed"
    elif any_pass:
        gate_status = "pass"
    else:
        gate_status = "pending"
    return {
        "claim": "geometry_prior_beam_fusion",
        "claim_status": gate_status,
        "baseline_group": baseline_group,
        "gps_query_baseline_group": gps_query_group,
        "clean_regression_threshold_dba": threshold,
        "claim_margin_dba": claim_margin,
        "baseline_clean_dba": baseline_clean,
        "candidate_statuses": candidate_statuses,
        "advantage_only_cannot_upgrade_primary_claim": True,
    }


def build_geometry_prior_diagnostics_bundle_manifest(
    manifest: Mapping[str, Any],
    *,
    diagnostics: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    claim_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    diagnostics = diagnostics or {}
    return {
        "version": "geometry_prior_beam_fusion_diagnostics_bundle_v1",
        "evidence_scope": "strict_geometry_prior_claim_gate",
        "output_files": dict(GEOMETRY_PRIOR_OUTPUT_FILES),
        "diagnostics": {
            "prior_quality_rows": len(list(diagnostics.get("prior_quality", []))),
            "branch_weight_rows": len(list(diagnostics.get("branch_weights", []))),
            "strict_comparison_rows": len(list(diagnostics.get("strict_comparison", []))),
            "missing_fields_are_unavailable": True,
        },
        "claim_gate": dict(claim_gate or {}),
        "comparability": manifest.get("comparability", {}),
    }


def _prior_quality_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append(
            {
                "model": row.get("model", ""),
                "group": row.get("group", ""),
                "condition": row.get("condition", ""),
                "split": row.get("split", ""),
                "seed": row.get("seed", ""),
                "prior_standalone_dba": _field_or_unavailable(row, "prior_standalone_dba", "geometry_prior_dba"),
                "prior_top1": _field_or_unavailable(row, "prior_top1", "geometry_prior_top1"),
                "prior_top3": _field_or_unavailable(row, "prior_top3", "geometry_prior_top3"),
                "prior_top5": _field_or_unavailable(row, "prior_top5", "geometry_prior_top5"),
                "prior_entropy": _field_or_unavailable(row, "prior_entropy", "geometry_prior_entropy"),
                "prior_target_distance": _field_or_unavailable(row, "prior_target_distance"),
                "prior_availability": _field_or_unavailable(row, "prior_availability", "geometry_prior_availability"),
                "candidate_recall": _field_or_unavailable(row, "candidate_recall"),
                "candidate_count_mean": _field_or_unavailable(row, "candidate_count_mean"),
                "no_regret_violation_count": _field_or_unavailable(row, "no_regret_violation_count"),
                "status": _availability_status(
                    row,
                    (
                        "prior_standalone_dba",
                        "geometry_prior_dba",
                        "prior_entropy",
                        "geometry_prior_entropy",
                        "candidate_recall",
                    ),
                ),
            }
        )
    return output


def _branch_weight_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append(
            {
                "model": row.get("model", ""),
                "group": row.get("group", ""),
                "condition": row.get("condition", ""),
                "split": row.get("split", ""),
                "seed": row.get("seed", ""),
                "image_weight_mean": _field_or_unavailable(row, "image_weight_mean", "branch_image_weight"),
                "prior_weight_mean": _field_or_unavailable(row, "prior_weight_mean", "branch_prior_weight"),
                "image_entropy": _field_or_unavailable(row, "image_entropy"),
                "prior_entropy": _field_or_unavailable(row, "prior_entropy", "geometry_prior_entropy"),
                "prior_image_agreement": _field_or_unavailable(row, "prior_image_agreement"),
                "prior_teacher_agreement": _field_or_unavailable(row, "prior_teacher_agreement"),
                "fused_improvement_dba": _field_or_unavailable(row, "fused_improvement_dba"),
                "fused_degradation_dba": _field_or_unavailable(row, "fused_degradation_dba"),
                "rerank_changed_top1_rate": _field_or_unavailable(row, "rerank_changed_top1_rate"),
                "rerank_fallback_rate": _field_or_unavailable(row, "rerank_fallback_rate"),
                "gate_confidence_mean": _field_or_unavailable(row, "gate_confidence_mean"),
                "residual_magnitude_mean": _field_or_unavailable(row, "residual_magnitude_mean"),
                "beneficial_rerank_count": _field_or_unavailable(row, "beneficial_rerank_count"),
                "neutral_rerank_count": _field_or_unavailable(row, "neutral_rerank_count"),
                "harmful_rerank_count": _field_or_unavailable(row, "harmful_rerank_count"),
                "status": _availability_status(
                    row,
                    (
                        "image_weight_mean",
                        "branch_image_weight",
                        "prior_weight_mean",
                        "rerank_changed_top1_rate",
                        "gate_confidence_mean",
                    ),
                ),
            }
        )
    return output


def _strict_comparison_rows(rows: list[dict[str, Any]], *, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("model", "")), str(row.get("group", "")), str(row.get("split", "")))].append(row)
    output = []
    for (model, group, split), items in sorted(grouped.items()):
        output.append(
            {
                "model": model,
                "group": group,
                "split": split,
                "sample_count": _first_available(items, "sample_count"),
                "metric_profile": _first_available(items, "metric_profile"),
                "beam_label_space": _first_available(items, "beam_label_space", "label_space"),
                "seed": _first_available(items, "seed"),
                "difficulty_digest": _first_available(items, "difficulty_digest"),
                "config": _first_available(items, "config"),
                "weights": _first_available(items, "weights"),
                "dba": _mean_metric(items),
                "top1": _mean_field(items, "top1"),
                "top3": _mean_field(items, "top3"),
                "top5": _mean_field(items, "top5"),
                "teacher_provenance": _first_available(items, "teacher_provenance", "checkpoint_provenance"),
                "evidence_scope": _first_available(items, "evidence_scope"),
                "real_perturbation_forward": all(
                    _is_clean_condition(item) or _row_has_real_forward_evidence(item)
                    for item in items
                ),
                "claim_gate_status": _strict_row_status(items, manifest=manifest),
            }
        )
    return output


def _field_or_unavailable(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return "unavailable"


def _availability_status(row: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    return "available" if any(row.get(key) not in (None, "") for key in keys) else "unavailable"


def _strict_row_status(items: list[Mapping[str, Any]], *, manifest: Mapping[str, Any]) -> str:
    if any(str(item.get("comparability_status", "passed")) != "passed" for item in items):
        return "pending"
    required = manifest.get("comparability", {}).get("keys", []) if isinstance(manifest.get("comparability"), Mapping) else []
    if required:
        missing = [key for key in required if _first_available(items, str(key)) == ""]
        if missing:
            return "pending"
    claim_cfg = manifest.get("geometry_prior_claim_gate", manifest.get("geometry_prior", {}))
    require_real_forward = bool(claim_cfg.get("require_real_forward_perturbations", True)) if isinstance(claim_cfg, Mapping) else True
    if require_real_forward and any(not _is_clean_condition(item) and not _row_has_real_forward_evidence(item) for item in items):
        return "pending"
    return "ready"


def _row_has_real_forward_evidence(row: Mapping[str, Any]) -> bool:
    return str(row.get("evidence_scope", "")).lower() == "real_forward" or str(row.get("status", "")).lower() == "real_forward"


def _mean_metric(rows: Iterable[Mapping[str, Any]]) -> float | None:
    return _mean_field(list(rows), "dba", "primary_metric")


def _mean_field(rows: Iterable[Mapping[str, Any]], *keys: str) -> float | None:
    values = []
    for row in rows:
        for key in keys:
            value = _float_or_none(row.get(key))
            if value is not None:
                values.append(value)
                break
    if not values:
        return None
    return float(np.mean(values))


def _first_available(rows: Iterable[Mapping[str, Any]], *keys: str) -> Any:
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                return value
    return ""


def _float_or_none(value: Any) -> float | None:
    if value in (None, "", "unavailable"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _is_clean_condition(row: Mapping[str, Any]) -> bool:
    condition = str(row.get("condition", "")).lower()
    return condition in {"p0", "p0_clean", "p0_clean_current", "clean", "clean_current"} or "clean" in condition


__all__ = [
    "GEOMETRY_PRIOR_OUTPUT_FILES",
    "aggregate_geometry_prior_diagnostics",
    "build_geometry_prior_claim_gate",
    "build_geometry_prior_diagnostics_bundle_manifest",
]
