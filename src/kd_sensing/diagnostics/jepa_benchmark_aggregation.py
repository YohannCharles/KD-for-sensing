from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from kd_sensing.diagnostics.jepa_benchmark_artifacts import (
    _load_mapping_text,
    _read_csv,
    _resolve_artifact_path,
    _resolve_existing_user_path,
)
from kd_sensing.diagnostics.jepa_benchmark_common import (
    DEFAULT_PRIMARY_METRIC,
    GPS_SUITE_TYPES,
    _area_under_curve,
    _case_row,
    _collapse_slope,
    _float,
    _float_or_none,
    _max_drop,
    _relative_drop,
    _sha256_text,
)


def aggregate_robustness_summary(
    metrics_rows: Iterable[Mapping[str, Any]],
    *,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in metrics_rows]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("condition")) in {"clean", "clean_anchor"} or str(row.get("suite")) == "clean_anchor":
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
    misleading = [
        row
        for row in gps_rows
        if str(row.get("condition")) in {"misleading_gps", "gps_distractor"}
        or "distractor" in str(row.get("suite_type"))
    ]
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


__all__ = [
    "aggregate_robustness_summary",
    "aggregate_shortcut_reliance",
    "read_benchmark_analysis_bundle",
    "robustness_matrix_rows",
    "select_benchmark_case_studies",
]
