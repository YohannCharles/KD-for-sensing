from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from kd_sensing.diagnostics.jepa_benchmark_common import (
    DEFAULT_PRIMARY_METRIC,
    GPS_QUERY_ADVANTAGE_SLICE_TYPE,
    PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
    PREDICTIVE_OUTPUT_FILES,
    _condition_digest,
    _float_or_none,
    _model_consumes_reliability_metadata,
    _predictive_group_category,
    _relative_drop,
    _scaled_error_metric,
    _scaled_metric,
)
from kd_sensing.diagnostics.jepa_benchmark_predictive import _predictive_jepa_metric_value


def _predictive_gps_query_advantage_metric_rows(
    model_name: str,
    model_spec: Mapping[str, Any],
    source: Mapping[str, Any],
    suite: Mapping[str, Any],
    *,
    seed: int,
    split: str,
    sample_count: int,
    primary_name: str,
    clean_primary: float,
    comparability_status: str,
    dry_run: bool,
) -> list[dict[str, Any]]:
    slice_cfg = suite.get("gps_query_advantage_slice", {})
    if not isinstance(slice_cfg, Mapping) or not bool(slice_cfg.get("enabled", False)):
        return []
    rows: list[dict[str, Any]] = []
    for condition in slice_cfg.get("conditions", []):
        if isinstance(condition, Mapping):
            rows.append(
                _predictive_gps_query_advantage_metric_row(
                    model_name,
                    model_spec,
                    source,
                    suite,
                    condition=condition,
                    seed=seed,
                    split=split,
                    sample_count=sample_count,
                    primary_name=primary_name,
                    clean_primary=clean_primary,
                    comparability_status=comparability_status,
                    dry_run=dry_run,
                )
            )
    for condition in slice_cfg.get("combined_conditions", []):
        if isinstance(condition, Mapping):
            rows.append(
                _predictive_gps_query_advantage_cxd_metric_row(
                    model_name,
                    model_spec,
                    source,
                    suite,
                    condition=condition,
                    seed=seed,
                    split=split,
                    sample_count=sample_count,
                    primary_name=primary_name,
                    clean_primary=clean_primary,
                    comparability_status=comparability_status,
                    dry_run=dry_run,
                )
            )
    return rows


def _predictive_gps_query_advantage_metric_row(
    model_name: str,
    model_spec: Mapping[str, Any],
    source: Mapping[str, Any],
    suite: Mapping[str, Any],
    *,
    condition: Mapping[str, Any],
    seed: int,
    split: str,
    sample_count: int,
    primary_name: str,
    clean_primary: float,
    comparability_status: str,
    dry_run: bool,
) -> dict[str, Any]:
    metric_value = _predictive_jepa_metric_value(clean_primary, condition, model_spec)
    params = condition.get("operator_params", {}) if isinstance(condition.get("operator_params"), Mapping) else {}
    profile_digest = _condition_digest(
        {
            "suite": suite.get("id"),
            "advantage_condition": condition.get("id"),
            "history_window": suite.get("history_window"),
            "seed": int(seed),
            "params": params,
        }
    )
    fallback_count = int(params.get("fallback_count", params.get("expected_fallback_count", 0)) or 0)
    return {
        "model": model_name,
        "group": model_spec.get("group", ""),
        "suite": f"{suite.get('id', PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE)}:gps_query_advantage",
        "suite_type": GPS_QUERY_ADVANTAGE_SLICE_TYPE,
        "evidence_slice": "gps_query_advantage",
        "claim_scope": "mechanism_diagnostic",
        "condition": condition.get("id", ""),
        "advantage_condition": condition.get("id", ""),
        "advantage_family": condition.get("advantage_family", "hard_negative"),
        "predictive_condition": "",
        "severity": float(condition.get("severity", 0.0) or 0.0),
        "a_severity": float(condition.get("severity", 0.0) or 0.0),
        "severity_unit": "gps_query_advantage_level",
        "seed": int(seed),
        "split": split,
        "sample_count": sample_count,
        "difficulty_digest": profile_digest,
        "operator_params": params,
        "fallback_count": fallback_count,
        "history_window": int(suite.get("history_window", params.get("history_window", 4)) or 4),
        "history_source_range_policy": "not_applicable",
        "source_history_range_field": "",
        "future_leak_check": "not_applicable",
        "visual_ambiguous_peer": bool(params.get("visual_ambiguous_peer", False)),
        "beam_offset_constrained_wrong_gps": bool(params.get("beam_offset_constrained_wrong_gps", False)),
        "min_beam_offset": params.get("min_beam_offset", params.get("beam_offset_min", "")),
        "scene_constraint": params.get("scene_constraint", ""),
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
        "counterfactual_input_intervention": bool(params.get("plausible_wrong_gps", False)),
        "consumes_reliability_metadata": _model_consumes_reliability_metadata(model_spec),
        "comparability_status": comparability_status,
        "status": source.get("status", "generated") if not dry_run else "dry_run",
    }


def _predictive_gps_query_advantage_cxd_metric_row(
    model_name: str,
    model_spec: Mapping[str, Any],
    source: Mapping[str, Any],
    suite: Mapping[str, Any],
    *,
    condition: Mapping[str, Any],
    seed: int,
    split: str,
    sample_count: int,
    primary_name: str,
    clean_primary: float,
    comparability_status: str,
    dry_run: bool,
) -> dict[str, Any]:
    from kd_sensing.diagnostics.jepa_benchmark_scenario_c import _scenario_c_metric_columns
    from kd_sensing.diagnostics.jepa_benchmark_scenario_d import (
        _combined_cxd_metric_value,
        _scenario_d_metric_columns,
    )

    gps_condition = condition.get("gps_condition", {}) if isinstance(condition.get("gps_condition"), Mapping) else {}
    image_condition = condition.get("image_condition", {}) if isinstance(condition.get("image_condition"), Mapping) else {}
    metric_value = _combined_cxd_metric_value(clean_primary, gps_condition, image_condition, model_spec)
    profile_digest = _condition_digest(
        {
            "suite": suite.get("id"),
            "advantage_condition": condition.get("id"),
            "gps_condition": gps_condition.get("id"),
            "image_condition": image_condition.get("id"),
            "seed": int(seed),
        }
    )
    row = {
        "model": model_name,
        "group": model_spec.get("group", ""),
        "suite": f"{suite.get('id', PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE)}:gps_query_advantage",
        "suite_type": GPS_QUERY_ADVANTAGE_SLICE_TYPE,
        "evidence_slice": "gps_query_advantage",
        "claim_scope": "mechanism_diagnostic",
        "condition": condition.get("id", ""),
        "advantage_condition": condition.get("id", ""),
        "advantage_family": condition.get("advantage_family", "combined_cxd"),
        "gps_condition": gps_condition.get("id", ""),
        "image_condition": image_condition.get("id", ""),
        "severity": float(image_condition.get("severity", 0.0) or 0.0),
        "c_severity": float(gps_condition.get("severity", 0.0) or 0.0),
        "d_severity": float(image_condition.get("severity", 0.0) or 0.0),
        "severity_unit": "scenario_c_x_d_level",
        "seed": int(seed),
        "split": split,
        "sample_count": sample_count,
        "difficulty_digest": profile_digest,
        "operator_params": {
            "gps_condition": gps_condition,
            "image_operator_params": image_condition.get("operator_params", {}),
        },
        "fallback_count": 0,
        "history_window": int(suite.get("history_window", 4) or 4),
        "history_source_range_policy": condition.get("history_source_range_policy", "strictly_past"),
        "source_history_range_field": condition.get("source_history_range_field", "gps_source_index"),
        "future_leak_check": "required",
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
        "counterfactual_input_intervention": True,
        "consumes_reliability_metadata": _model_consumes_reliability_metadata(model_spec),
        "comparability_status": comparability_status,
        "status": source.get("status", "generated") if not dry_run else "dry_run",
    }
    row.update(_scenario_c_metric_columns(gps_condition, model_spec=model_spec))
    row.update(_scenario_d_metric_columns(image_condition, seed=seed))
    row["suite_type"] = GPS_QUERY_ADVANTAGE_SLICE_TYPE
    row["evidence_slice"] = "gps_query_advantage"
    row["condition"] = condition.get("id", "")
    row["advantage_condition"] = condition.get("id", "")
    row["difficulty_digest"] = profile_digest
    return row


def aggregate_gps_query_advantage_margins(
    metrics_rows: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in metrics_rows if str(row.get("suite_type")) == GPS_QUERY_ADVANTAGE_SLICE_TYPE]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("condition")), str(row.get("model"))), []).append(row)
    condition_model_rows: list[dict[str, Any]] = []
    for (condition, model), items in grouped.items():
        group = str(items[0].get("group", ""))
        primary = _mean_numeric(item.get("primary_metric") for item in items)
        dba = _mean_numeric(item.get("dba") for item in items)
        top1 = _mean_numeric(item.get("top1") for item in items)
        top3 = _mean_numeric(item.get("top3") for item in items)
        top5 = _mean_numeric(item.get("top5") for item in items)
        condition_model_rows.append(
            {
                "condition": condition,
                "model": model,
                "group": group,
                "predictive_group_category": _predictive_group_category(group),
                "advantage_family": items[0].get("advantage_family", ""),
                "gps_condition": items[0].get("gps_condition", ""),
                "image_condition": items[0].get("image_condition", ""),
                "primary_metric_name": primary_metric,
                "primary_metric": primary if primary is not None else "",
                "dba": dba if dba is not None else "",
                "top1": top1 if top1 is not None else "",
                "top3": top3 if top3 is not None else "",
                "top5": top5 if top5 is not None else "",
                "seed_count": len({str(item.get("seed")) for item in items}),
                "split": items[0].get("split", ""),
                "sample_count": items[0].get("sample_count", ""),
                "difficulty_digest": _condition_digest(
                    {
                        "condition": condition,
                        "model": model,
                        "digests": [item.get("difficulty_digest", "") for item in items],
                    }
                ),
                "fallback_count": int(sum(int(float(item.get("fallback_count", 0) or 0)) for item in items)),
                "comparability_status": items[0].get("comparability_status", ""),
                "row_statuses": sorted({str(item.get("status", "")) for item in items if item.get("status", "")}),
            }
        )

    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in condition_model_rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    threshold = float(
        (
            manifest.get(PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE, {})
            if isinstance(manifest.get(PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE), Mapping)
            else {}
        ).get("claim_margin_dba", 0.05)
    )
    strict = str(manifest.get("comparability", {}).get("mode", "mark")) == "strict"
    output: list[dict[str, Any]] = []
    for condition, items in by_condition.items():
        resnet = next((item for item in items if item["predictive_group_category"] == "resnet_image_gps"), None)
        gps_query = next((item for item in items if item["predictive_group_category"] == "jepa_baseline"), None)
        resnet_value = _float_or_none(resnet.get("dba")) if resnet else None
        gps_query_value = _float_or_none(gps_query.get("dba")) if gps_query else None
        for item in items:
            value = _float_or_none(item.get("dba"))
            margin_resnet = "" if value is None or resnet_value is None else value - resnet_value
            margin_query = "" if value is None or gps_query_value is None else value - gps_query_value
            status = "mechanism_evidence"
            if item.get("comparability_status") != "passed" or not strict:
                status = "not_comparable"
            if resnet_value is None or gps_query_value is None or value is None:
                status = "unavailable"
            if any(row_status in {"synthetic", "dry_run"} or "mock" in row_status for row_status in item.get("row_statuses", [])):
                status = "mock/smoke"
            output.append(
                {
                    **item,
                    "condition": condition,
                    "resnet_image_gps_dba": resnet_value if resnet_value is not None else "",
                    "gps_query_baseline_dba": gps_query_value if gps_query_value is not None else "",
                    "margin_vs_resnet_dba": margin_resnet,
                    "margin_vs_gps_query_dba": margin_query,
                    "advantage_pass_vs_resnet": bool(margin_resnet != "" and float(margin_resnet) >= threshold),
                    "advantage_pass_vs_gps_query": bool(margin_query != "" and float(margin_query) >= threshold),
                    "claim_scope": "mechanism_diagnostic",
                    "claim_status": status,
                }
            )
    output.sort(key=lambda item: (str(item["condition"]), str(item["predictive_group_category"]), str(item["model"])))
    return output


def build_predictive_claim_gate(
    predictive_summary: Iterable[Mapping[str, Any]],
    advantage_margins: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    predictive_rows = [dict(row) for row in predictive_summary]
    advantage_rows = [dict(row) for row in advantage_margins]
    model = next((row for row in predictive_rows if row.get("predictive_group_category") == "jepa_predictive_hybrid"), {})
    model_name = str(model.get("model", ""))
    model_advantage = [
        row
        for row in advantage_rows
        if str(row.get("model")) == model_name
        or (not model_name and str(row.get("predictive_group_category")) == "jepa_predictive_hybrid")
    ]
    p_pass = bool(model.get("claim_pass_5pt", False))
    advantage_vs_resnet = all(bool(row.get("advantage_pass_vs_resnet", False)) for row in model_advantage) if model_advantage else False
    advantage_vs_query = all(bool(row.get("advantage_pass_vs_gps_query", False)) for row in model_advantage) if model_advantage else False
    advantage_available = bool(model_advantage)
    statuses = {str(row.get("claim_status", "")) for row in model_advantage if row.get("claim_status", "")}
    predictive_status = str(model.get("claim_status", "unavailable"))
    if "mock/smoke" in statuses or predictive_status == "mock/smoke":
        gate_status = "mock/smoke"
    elif predictive_status == "not_comparable" or "not_comparable" in statuses:
        gate_status = "not_comparable"
    elif not model or not advantage_available:
        gate_status = "unavailable"
    elif p_pass and advantage_vs_resnet and advantage_vs_query:
        gate_status = "pass"
    elif advantage_vs_resnet and advantage_vs_query and not p_pass:
        gate_status = "mechanism_evidence_pending_primary"
    elif p_pass:
        gate_status = "partial_pending_advantage"
    else:
        gate_status = "pending"
    threshold = float(
        (
            manifest.get(PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE, {})
            if isinstance(manifest.get(PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE), Mapping)
            else {}
        ).get("claim_margin_dba", 0.05)
    )
    return {
        "claim": "predictive_gps_query_plus_plus",
        "primary_suite": PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
        "advantage_slice": GPS_QUERY_ADVANTAGE_SLICE_TYPE,
        "model": model_name,
        "group": model.get("group", ""),
        "threshold_dba": threshold,
        "p_suite_claim_status": predictive_status,
        "p_suite_pass": p_pass,
        "advantage_condition_count": len({str(row.get("condition")) for row in model_advantage}),
        "advantage_pass_vs_resnet": advantage_vs_resnet,
        "advantage_pass_vs_gps_query": advantage_vs_query,
        "claim_status": gate_status,
        "advantage_only_cannot_upgrade_primary_claim": True,
    }


def build_predictive_diagnostics_bundle_manifest(
    manifest: Mapping[str, Any],
    *,
    advantage_margins: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    margins = [dict(row) for row in advantage_margins]
    fallback_by_condition: dict[str, int] = {}
    for row in margins:
        condition = str(row.get("condition", ""))
        fallback_by_condition[condition] = fallback_by_condition.get(condition, 0) + int(float(row.get("fallback_count", 0) or 0))
    return {
        "version": "predictive_gps_query_diagnostics_bundle_v1",
        "evidence_scope": "explanatory_diagnostics_not_primary_claim",
        "primary_claim_source": "strict_metrics_and_provenance",
        "output_files": {
            "predictive_condition_metrics": PREDICTIVE_OUTPUT_FILES["predictive_condition_metrics"],
            "predictive_gps_query_advantage_metrics": PREDICTIVE_OUTPUT_FILES["predictive_gps_query_advantage_metrics"],
            "predictive_gps_query_advantage_margins": PREDICTIVE_OUTPUT_FILES["predictive_gps_query_advantage_margins"],
            "predictive_claim_gate": PREDICTIVE_OUTPUT_FILES["predictive_claim_gate"],
        },
        "diagnostics": {
            "gate_weight_summaries": {"status": "pending_runtime_diagnostics"},
            "branch_availability": {"status": "pending_runtime_diagnostics"},
            "latent_consistency_summaries": {"status": "pending_runtime_diagnostics"},
            "fallback_counts": fallback_by_condition,
            "per_condition_margin_table": PREDICTIVE_OUTPUT_FILES["predictive_gps_query_advantage_margins"],
        },
        "explanatory_figures_do_not_establish_claim": True,
        "comparability": manifest.get("comparability", {}),
    }


def _mean_numeric(values: Iterable[Any]) -> float | None:
    numbers = [_float_or_none(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    if not numbers:
        return None
    return float(np.mean(numbers))


__all__ = [
    "_predictive_gps_query_advantage_metric_rows",
    "aggregate_gps_query_advantage_margins",
    "build_predictive_claim_gate",
    "build_predictive_diagnostics_bundle_manifest",
]
