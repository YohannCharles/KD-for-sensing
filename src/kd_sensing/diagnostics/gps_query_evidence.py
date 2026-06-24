import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


EVIDENCE_VERSION = "gps_query_attention_evidence_v1"
DEFAULT_GPS_QUERY_EVIDENCE_CONFIG = {
    "enabled": False,
    "output_dir": None,
    "model_pairs": [],
    "anchor_baselines": [],
    "metrics": {
        "p0_p5": None,
        "paths": [],
        "benchmark_manifest": None,
        "metrics_by_condition": None,
    },
    "benchmark_manifest": None,
    "forward_cache": None,
    "comparability_keys": [
        "split",
        "scene_set",
        "seed",
        "checkpoint_selection",
        "label_space",
        "metric_profile",
    ],
    "claim_gate": {
        "metric": "dba",
        "min_clean_delta": 0.0,
        "min_mean_delta": 0.0,
        "max_clean_regression": 0.0,
        "min_sample_count": 1,
    },
    "attention": {
        "aggregation": "mean_time_query",
        "max_cases": 32,
    },
    "attention_faithfulness": {
        "enabled": False,
        "patch_ratio": 0.1,
        "patch_count": None,
        "selection_groups": ["top_attention", "low_attention", "random"],
        "occlusion_strategy": "zero",
        "random_seed": 42,
        "max_cases": 32,
        "metric_target": "dba_contribution",
    },
}


def gps_query_evidence_enabled(cfg: Mapping[str, Any]) -> bool:
    evidence = cfg.get("evidence", {}) if isinstance(cfg.get("evidence"), Mapping) else {}
    return bool(evidence.get("enabled", False))


def write_gps_query_evidence_package(
    cfg: Mapping[str, Any],
    *,
    output_dir: Path,
    analyses: Mapping[str, Any],
    comparison_rows: list[dict[str, Any]],
    command: list[str] | None,
    warnings: list[str],
    formats: tuple[str, ...],
    dpi: int,
    attention_faithfulness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evidence_cfg = _evidence_cfg(cfg)
    out = Path(str(evidence_cfg.get("output_dir") or output_dir))
    tables_dir = out / "tables"
    figures_dir = out / "figures"
    cases_dir = out / "cases"
    for directory in (tables_dir, figures_dir, cases_dir):
        directory.mkdir(parents=True, exist_ok=True)

    local_warnings: list[str] = []
    metric_rows = _load_metric_rows(cfg, evidence_cfg, analyses, local_warnings)
    pairs = _pair_records(cfg, evidence_cfg, analyses)
    anchors = _anchor_records(cfg, evidence_cfg)
    paired_delta_rows = _paired_delta_rows(metric_rows, pairs)
    anchor_rows = _anchor_rows(metric_rows, anchors, pairs)
    case_rows = _evidence_case_rows(comparison_rows, cfg, evidence_cfg)
    faithfulness_context = dict(attention_faithfulness or {})
    claim_rows = _claim_gate_rows(paired_delta_rows, pairs, case_rows, analyses, evidence_cfg, faithfulness_context)

    generated: list[Path] = []
    generated.append(_write_csv(tables_dir / "gps_query_metric_rows_long.csv", metric_rows))
    generated.append(_write_csv(tables_dir / "paired_delta_by_condition.csv", paired_delta_rows))
    generated.append(_write_csv(tables_dir / "anchor_comparisons.csv", anchor_rows))
    generated.append(_write_csv(tables_dir / "case_selection.csv", case_rows))
    generated.append(_write_csv(tables_dir / "claim_gate_summary.csv", claim_rows))

    generated.extend(_write_delta_figures(figures_dir, paired_delta_rows, formats, dpi, local_warnings))
    generated.extend(_write_attention_figures(figures_dir, analyses, cfg, evidence_cfg, formats, dpi, local_warnings))
    generated.extend(_write_case_payloads_and_panels(cases_dir, figures_dir, case_rows, analyses, formats, dpi, local_warnings))

    warnings.extend(local_warnings)
    report_lines = _evidence_report_lines(claim_rows, paired_delta_rows, case_rows, local_warnings, faithfulness_context)
    manifest = {
        "version": EVIDENCE_VERSION,
        "command": list(command or []),
        "analysis_config_path": cfg.get("_analysis_config_path"),
        "analysis_config_digest": cfg.get("_analysis_config_digest"),
        "output_dir": str(out),
        "metrics_sources": sorted({str(row.get("source_path", "")) for row in metric_rows if row.get("source_path")}),
        "benchmark_manifest": evidence_cfg.get("benchmark_manifest") or evidence_cfg.get("metrics", {}).get("benchmark_manifest"),
        "forward_cache": evidence_cfg.get("forward_cache"),
        "comparability_keys": list(evidence_cfg.get("comparability_keys") or []),
        "model_pairs": pairs,
        "anchor_baselines": anchors,
        "paired_delta_rows": len(paired_delta_rows),
        "case_rows": len(case_rows),
        "claim_gate": claim_rows,
        "attention_provenance": _attention_provenance(analyses),
        "faithfulness_summary": faithfulness_context,
        "warnings": sorted(set(str(item) for item in local_warnings)),
        "outputs": [_output_record(path, out) for path in generated],
        "caveat": "Attention hotspot 是解释性证据，不是因果证明。",
    }
    manifest_path = out / "evidence_manifest.json"
    manifest_path.write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["outputs"].append(_output_record(manifest_path, out))
    manifest_path.write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "enabled": True,
        "manifest": str(manifest_path),
        "model_pairs": pairs,
        "paired_delta_rows": len(paired_delta_rows),
        "case_rows": len(case_rows),
        "claim_gate": claim_rows,
        "report_lines": report_lines,
        "warnings": local_warnings,
    }


def _evidence_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = cfg.get("evidence", {}) if isinstance(cfg.get("evidence"), Mapping) else {}
    merged = json.loads(json.dumps(DEFAULT_GPS_QUERY_EVIDENCE_CONFIG))
    return _deep_merge(merged, dict(raw))


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _load_metric_rows(
    cfg: Mapping[str, Any],
    evidence_cfg: Mapping[str, Any],
    analyses: Mapping[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    paths = _metric_paths(cfg, evidence_cfg)
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_metric_csv(path, warnings))
    if rows:
        return rows
    return _metric_rows_from_analyses(cfg, analyses)


def _metric_paths(cfg: Mapping[str, Any], evidence_cfg: Mapping[str, Any]) -> list[Path]:
    metrics = evidence_cfg.get("metrics", {}) if isinstance(evidence_cfg.get("metrics"), Mapping) else {}
    raw: list[Any] = []
    for key in ("p0_p5", "metrics_by_condition"):
        value = metrics.get(key)
        if value:
            raw.append(value)
    raw.extend(metrics.get("paths") or [])
    benchmark = cfg.get("benchmark", {}) if isinstance(cfg.get("benchmark"), Mapping) else {}
    if benchmark.get("metrics_by_condition"):
        raw.append(benchmark.get("metrics_by_condition"))
    paths: list[Path] = []
    for item in raw:
        path = Path(str(item))
        if path not in paths:
            paths.append(path)
    return paths


def _read_metric_csv(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        warnings.append(f"evidence_metrics_missing:{path}")
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.extend(_normalize_metric_row(raw, source_path=path))
    return rows


def _normalize_metric_row(raw: Mapping[str, Any], *, source_path: Path) -> list[dict[str, Any]]:
    model = _first(raw, "model", "variant_id", "model_name")
    condition = _first(raw, "condition", "difficulty", "suite", default="P0_clean_current")
    scene_group = _first(raw, "scene_group", "scene_set", "scene", "group", default="all")
    sample_count = _first(raw, "sample_count", "n", "support", default="")
    if raw.get("metric"):
        value = _first(raw, "value", "metric_value", "primary_metric", "dba")
        return [_metric_row(raw, model, condition, scene_group, str(raw.get("metric")), value, sample_count, source_path)]
    metric_keys = [
        "primary_metric",
        "dba",
        "top1",
        "top3",
        "top5",
        "official_top3_dba",
        "final_dba",
        "final_top1",
        "final_top3",
        "mean_beam_index_error",
    ]
    rows = []
    for key in metric_keys:
        if key not in raw or raw.get(key) in (None, ""):
            continue
        metric = str(raw.get("primary_metric_name") or key) if key == "primary_metric" else key
        rows.append(_metric_row(raw, model, condition, scene_group, metric, raw.get(key), sample_count, source_path))
    return rows


def _metric_row(
    raw: Mapping[str, Any],
    model: Any,
    condition: Any,
    scene_group: Any,
    metric: str,
    value: Any,
    sample_count: Any,
    source_path: Path,
) -> dict[str, Any]:
    return {
        "model": str(model),
        "condition": str(condition),
        "scene_group": str(scene_group),
        "metric": str(metric).lower(),
        "value": _float_or_blank(value),
        "sample_count": sample_count,
        "split": _first(raw, "split", default=""),
        "seed": _first(raw, "seed", default=""),
        "checkpoint_selection": _first(raw, "checkpoint_selection", "checkpoint_role", default=""),
        "label_space": _first(raw, "label_space", "beam_label_space", default=""),
        "metric_profile": _first(raw, "metric_profile", "primary_metric_name", default=""),
        "source_path": str(source_path),
    }


def _metric_rows_from_analyses(cfg: Mapping[str, Any], analyses: Mapping[str, Any]) -> list[dict[str, Any]]:
    split_cfg = cfg.get("split", {}) if isinstance(cfg.get("split"), Mapping) else {}
    scene_group = ",".join(str(item) for item in (split_cfg.get("scenes") or ["all"]))
    rows: list[dict[str, Any]] = []
    for name, analysis in analyses.items():
        summary = getattr(analysis, "summary", {}) or {}
        for metric in ("dba", "top1", "top3", "top5"):
            if metric not in summary:
                continue
            rows.append(
                {
                    "model": name,
                    "condition": "P0_clean_current",
                    "scene_group": scene_group,
                    "metric": metric,
                    "value": float(summary.get(metric, 0.0)),
                    "sample_count": summary.get("sample_count", ""),
                    "split": split_cfg.get("evaluation_split", ""),
                    "seed": cfg.get("sampling", {}).get("seed", ""),
                    "checkpoint_selection": "",
                    "label_space": summary.get("num_beams", ""),
                    "metric_profile": summary.get("distance_mode", ""),
                    "source_path": "analysis_cache",
                }
            )
    return rows


def _pair_records(cfg: Mapping[str, Any], evidence_cfg: Mapping[str, Any], analyses: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_pairs = list(evidence_cfg.get("model_pairs") or [])
    if not raw_pairs:
        sampling = cfg.get("sampling", {}) if isinstance(cfg.get("sampling"), Mapping) else {}
        query = sampling.get("query_model")
        baseline = sampling.get("baseline_model")
        if query and baseline:
            raw_pairs.append({"name": f"{query}_vs_{baseline}", "query_model": query, "baseline_model": baseline})
    keys = [str(item) for item in evidence_cfg.get("comparability_keys") or []]
    records = []
    for raw in raw_pairs:
        if not isinstance(raw, Mapping):
            continue
        query = str(raw.get("query_model", raw.get("query", "")))
        baseline = str(raw.get("baseline_model", raw.get("baseline", "")))
        if not query or not baseline:
            continue
        record = {
            "name": str(raw.get("name") or f"{query}_vs_{baseline}"),
            "query_model": query,
            "baseline_model": baseline,
            "role": "paired",
            **_comparability_status(cfg, raw, query, baseline, keys),
        }
        if query not in analyses:
            record.setdefault("warnings", []).append(f"query_analysis_missing:{query}")
        if baseline not in analyses:
            record.setdefault("warnings", []).append(f"baseline_analysis_missing:{baseline}")
        records.append(record)
    return records


def _anchor_records(cfg: Mapping[str, Any], evidence_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for raw in evidence_cfg.get("anchor_baselines") or []:
        if not isinstance(raw, Mapping):
            continue
        model = str(raw.get("model", raw.get("name", "")))
        if not model:
            continue
        records.append(
            {
                "name": str(raw.get("name") or model),
                "model": model,
                "role": "anchor",
                "comparability_status": "anchor_reference_only",
                "provenance": _model_provenance(cfg, model),
            }
        )
    return records


def _comparability_status(
    cfg: Mapping[str, Any],
    pair: Mapping[str, Any],
    query: str,
    baseline: str,
    keys: list[str],
) -> dict[str, Any]:
    q_prov = _model_provenance(cfg, query, pair=pair, prefix="query")
    b_prov = _model_provenance(cfg, baseline, pair=pair, prefix="baseline")
    missing = [key for key in keys if q_prov.get(key) in (None, "") or b_prov.get(key) in (None, "")]
    mismatched = [key for key in keys if key not in missing and str(q_prov.get(key)) != str(b_prov.get(key))]
    status = "strict" if not missing and not mismatched else "not_comparable"
    return {
        "comparability_status": status,
        "missing_fields": missing,
        "mismatched_fields": mismatched,
        "query_provenance": q_prov,
        "baseline_provenance": b_prov,
    }


def _model_provenance(
    cfg: Mapping[str, Any],
    model: str,
    *,
    pair: Mapping[str, Any] | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    model_spec = (cfg.get("models", {}) or {}).get(model, {})
    if not isinstance(model_spec, Mapping):
        model_spec = {}
    explicit = model_spec.get("provenance", {}) if isinstance(model_spec.get("provenance"), Mapping) else {}
    split_cfg = cfg.get("split", {}) if isinstance(cfg.get("split"), Mapping) else {}
    sampling = cfg.get("sampling", {}) if isinstance(cfg.get("sampling"), Mapping) else {}
    pair = pair or {}

    def value(key: str, default: Any = "") -> Any:
        prefixed = f"{prefix}_{key}" if prefix else key
        return (
            pair.get(prefixed)
            or pair.get(key)
            or explicit.get(key)
            or model_spec.get(key)
            or default
        )

    scenes = split_cfg.get("scenes")
    return {
        "split": value("split", split_cfg.get("evaluation_split", "")),
        "scene_set": value("scene_set", ",".join(str(item) for item in scenes) if isinstance(scenes, list) else scenes or ""),
        "seed": value("seed", sampling.get("seed", "")),
        "checkpoint_selection": value("checkpoint_selection", model_spec.get("weights") or model_spec.get("logits_cache") or ""),
        "label_space": value("label_space", model_spec.get("num_beams", "")),
        "metric_profile": value("metric_profile", model_spec.get("distance_mode", "")),
        "config": model_spec.get("config", ""),
        "weights": model_spec.get("weights", ""),
        "logits_cache": model_spec.get("logits_cache", model_spec.get("cache", "")),
    }


def _paired_delta_rows(metric_rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in metric_rows:
        index[(str(row["model"]), str(row["condition"]), str(row["scene_group"]), str(row["metric"]))] = row
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        query = str(pair["query_model"])
        baseline = str(pair["baseline_model"])
        for row in metric_rows:
            if str(row.get("model")) != query:
                continue
            key = (baseline, str(row["condition"]), str(row["scene_group"]), str(row["metric"]))
            base = index.get(key)
            if base is None:
                continue
            qv = _float_or_none(row.get("value"))
            bv = _float_or_none(base.get("value"))
            if qv is None or bv is None:
                continue
            delta = qv - bv
            rows.append(
                {
                    "model_pair": pair["name"],
                    "query_model": query,
                    "baseline_model": baseline,
                    "condition": row["condition"],
                    "scene_group": row["scene_group"],
                    "metric": row["metric"],
                    "query_value": qv,
                    "baseline_value": bv,
                    "absolute_delta": delta,
                    "relative_delta": "" if abs(bv) < 1e-12 else delta / abs(bv),
                    "sample_count": row.get("sample_count") or base.get("sample_count", ""),
                    "source_path": row.get("source_path") or base.get("source_path", ""),
                    "comparability_status": pair.get("comparability_status", "not_comparable"),
                }
            )
    rows.sort(key=lambda item: (str(item["model_pair"]), str(item["scene_group"]), str(item["condition"]), str(item["metric"])))
    return rows


def _anchor_rows(
    metric_rows: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    anchor_models = {row["model"]: row["name"] for row in anchors}
    query_models = {pair["query_model"] for pair in pairs}
    rows = []
    for anchor in metric_rows:
        if anchor.get("model") not in anchor_models:
            continue
        for query in metric_rows:
            if query.get("model") not in query_models:
                continue
            if (anchor.get("condition"), anchor.get("scene_group"), anchor.get("metric")) != (
                query.get("condition"),
                query.get("scene_group"),
                query.get("metric"),
            ):
                continue
            av = _float_or_none(anchor.get("value"))
            qv = _float_or_none(query.get("value"))
            if av is None or qv is None:
                continue
            rows.append(
                {
                    "anchor_name": anchor_models[str(anchor["model"])],
                    "anchor_model": anchor["model"],
                    "query_model": query["model"],
                    "condition": query["condition"],
                    "scene_group": query["scene_group"],
                    "metric": query["metric"],
                    "query_value": qv,
                    "anchor_value": av,
                    "absolute_delta_vs_anchor": qv - av,
                    "comparability_status": "anchor_reference_only",
                }
            )
    return rows


def _evidence_case_rows(
    comparison_rows: list[dict[str, Any]],
    cfg: Mapping[str, Any],
    evidence_cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sampling = cfg.get("sampling", {}) if isinstance(cfg.get("sampling"), Mapping) else {}
    groups = list(sampling.get("case_groups") or ["query_gain", "query_regression", "shared_near_miss", "shared_failure"])
    if "shared_failure" not in groups:
        groups.append("shared_failure")
    per_group = int(sampling.get("cases_per_group", 3))
    seed = int(sampling.get("seed", 42))
    selected: list[dict[str, Any]] = []
    for group in groups:
        candidates = []
        for row in comparison_rows:
            flag = row.get(group)
            if group == "shared_failure" and flag in (None, ""):
                flag = row.get("far_error")
            if int(flag or 0) != 1:
                continue
            query_model = str(row.get("query_model", sampling.get("query_model", "")))
            baseline_model = str(row.get("baseline_model", sampling.get("baseline_model", "")))
            query_error = _float_or_none(row.get(f"{query_model}_top3_min_distance"))
            baseline_error = _float_or_none(row.get(f"{baseline_model}_top3_min_distance"))
            q_dba = _float_or_none(row.get(f"{query_model}_dba_contribution"))
            b_dba = _float_or_none(row.get(f"{baseline_model}_dba_contribution"))
            out = dict(row)
            out.update(
                {
                    "group": group,
                    "case_group": group,
                    "selection_reason": _case_reason(group, baseline_error, query_error),
                    "baseline_error": "" if baseline_error is None else baseline_error,
                    "query_error": "" if query_error is None else query_error,
                    "metric_delta": "" if q_dba is None or b_dba is None else q_dba - b_dba,
                    "selection_seed": seed,
                }
            )
            candidates.append(out)
        candidates.sort(key=lambda item: (str(item.get("scene", "")), str(item.get("sample_id", ""))))
        selected.extend(candidates[:per_group])
    return selected


def _case_reason(group: str, baseline_error: float | None, query_error: float | None) -> str:
    if group == "query_gain":
        return f"baseline_error={baseline_error}, query_error={query_error}"
    if group == "query_regression":
        return f"query_error={query_error}, baseline_error={baseline_error}"
    if group == "shared_failure":
        return "all paired models far from target"
    return "both models near target with limited query improvement"


def _claim_gate_rows(
    paired_delta_rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    analyses: Mapping[str, Any],
    evidence_cfg: Mapping[str, Any],
    faithfulness_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    gate = evidence_cfg.get("claim_gate", {}) if isinstance(evidence_cfg.get("claim_gate"), Mapping) else {}
    metric = str(gate.get("metric", "dba")).lower()
    min_clean = float(gate.get("min_clean_delta", 0.0))
    min_mean = float(gate.get("min_mean_delta", 0.0))
    max_clean_regression = float(gate.get("max_clean_regression", 0.0))
    min_n = int(gate.get("min_sample_count", 1))
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        pair_rows = [row for row in paired_delta_rows if row["model_pair"] == pair["name"] and str(row["metric"]).lower() == metric]
        strict = pair.get("comparability_status") == "strict"
        if not strict:
            status = "blocked"
            reason = "strict comparability failed"
        elif not pair_rows:
            status = "insufficient"
            reason = f"no paired delta rows for metric {metric}"
        else:
            clean_rows = [row for row in pair_rows if _is_clean_condition(str(row.get("condition", "")))]
            clean_delta = min((float(row["absolute_delta"]) for row in clean_rows), default=float("nan"))
            mean_delta = float(np.mean([float(row["absolute_delta"]) for row in pair_rows]))
            min_sample = min((_int_or_zero(row.get("sample_count")) for row in pair_rows), default=0)
            if min_sample < min_n:
                status = "insufficient"
                reason = f"sample_count<{min_n}"
            elif clean_rows and clean_delta < -max_clean_regression:
                status = "blocked"
                reason = "clean regression exceeds gate"
            elif (not clean_rows or clean_delta >= min_clean) and mean_delta >= min_mean:
                status = "supported"
                reason = f"mean_delta={mean_delta:.4f}"
            else:
                status = "exploratory"
                reason = f"mean_delta={mean_delta:.4f}"
        rows.append(
            {
                "claim": "gps_query_paired_effectiveness",
                "model_pair": pair["name"],
                "status": status,
                "reason": reason,
                "evidence_paths": "tables/paired_delta_by_condition.csv",
            }
        )
    paired_supported = any(row.get("claim") == "gps_query_paired_effectiveness" and row.get("status") == "supported" for row in rows)
    attention_count = sum(len(getattr(analysis, "attention_rows", []) or []) for analysis in analyses.values())
    faithfulness = dict(faithfulness_context or {})
    faith_status = str(faithfulness.get("status", "disabled" if not faithfulness.get("enabled") else "insufficient"))
    if not attention_count:
        attention_status = "insufficient"
        attention_reason = "attention_rows=0"
    elif not paired_supported:
        attention_status = "exploratory"
        attention_reason = "paired evidence not supported; attention cannot upgrade claim"
    elif faithfulness.get("enabled") and faith_status == "passed":
        attention_status = "supported"
        attention_reason = "paired evidence supported and attention faithfulness passed"
    elif faithfulness.get("enabled"):
        attention_status = "insufficient"
        attention_reason = f"attention faithfulness {faith_status}"
    else:
        attention_status = "exploratory"
        attention_reason = f"attention_rows={attention_count}; faithfulness disabled"
    rows.append(
        {
            "claim": "attention_hotspot_interpretation",
            "model_pair": "",
            "status": attention_status,
            "reason": attention_reason,
            "evidence_paths": "tables/attention_summary.csv;tables/attention_faithfulness.csv;figures/evidence_attention/",
        }
    )
    groups = {str(row.get("group", row.get("case_group", ""))) for row in case_rows}
    required = {"query_gain", "query_regression", "shared_near_miss", "shared_failure"}
    rows.append(
        {
            "claim": "case_coverage",
            "model_pair": "",
            "status": "supported" if required.issubset(groups) else ("exploratory" if groups else "insufficient"),
            "reason": "covered=" + ",".join(sorted(groups)),
            "evidence_paths": "tables/case_selection.csv;cases/",
        }
    )
    return rows


def _write_delta_figures(
    figures_dir: Path,
    rows: list[dict[str, Any]],
    formats: tuple[str, ...],
    dpi: int,
    warnings: list[str],
) -> list[Path]:
    if not rows or not _matplotlib_available(warnings):
        return []
    metric = str(rows[0].get("metric", "metric"))
    generated: list[Path] = []
    generated.extend(_plot_condition_heatmap(figures_dir, rows, metric, formats, dpi, warnings))
    generated.extend(_plot_scene_group_delta(figures_dir, rows, metric, formats, dpi, warnings))
    return generated


def _plot_condition_heatmap(
    figures_dir: Path,
    rows: list[dict[str, Any]],
    metric: str,
    formats: tuple[str, ...],
    dpi: int,
    warnings: list[str],
) -> list[Path]:
    import matplotlib.pyplot as plt

    filtered = [row for row in rows if str(row.get("metric")) == metric]
    conditions = sorted({str(row["condition"]) for row in filtered})
    pairs = sorted({str(row["model_pair"]) for row in filtered})
    if len(conditions) < 2 or not pairs:
        return []
    matrix = np.full((len(pairs), len(conditions)), np.nan, dtype=np.float64)
    for row in filtered:
        matrix[pairs.index(str(row["model_pair"])), conditions.index(str(row["condition"]))] = float(row["absolute_delta"])
    fig, ax = plt.subplots(figsize=(max(6, len(conditions) * 1.2), max(2.5, len(pairs) * 0.7)))
    im = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(conditions)), labels=conditions, rotation=30, ha="right")
    ax.set_yticks(range(len(pairs)), labels=pairs)
    ax.set_title(f"P0-P5 GPS-query delta heatmap | metric={metric}")
    fig.colorbar(im, ax=ax, fraction=0.046, label="query - baseline")
    fig.tight_layout()
    return _save_figure(fig, figures_dir / "p0_p5_delta_heatmap", formats, dpi)


def _plot_scene_group_delta(
    figures_dir: Path,
    rows: list[dict[str, Any]],
    metric: str,
    formats: tuple[str, ...],
    dpi: int,
    warnings: list[str],
) -> list[Path]:
    import matplotlib.pyplot as plt

    filtered = [row for row in rows if str(row.get("metric")) == metric]
    groups = sorted({str(row["scene_group"]) for row in filtered})
    if len(groups) < 1:
        return []
    means = [float(np.mean([float(row["absolute_delta"]) for row in filtered if str(row["scene_group"]) == group])) for group in groups]
    fig, ax = plt.subplots(figsize=(max(5, len(groups) * 1.1), 3.5))
    ax.bar(groups, means, color="#4477AA")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"Scene-group GPS-query delta | metric={metric}")
    ax.set_ylabel("query - baseline")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return _save_figure(fig, figures_dir / "scene_group_delta", formats, dpi)


def _write_attention_figures(
    figures_dir: Path,
    analyses: Mapping[str, Any],
    cfg: Mapping[str, Any],
    evidence_cfg: Mapping[str, Any],
    formats: tuple[str, ...],
    dpi: int,
    warnings: list[str],
) -> list[Path]:
    if not _matplotlib_available(warnings):
        return []
    import matplotlib.pyplot as plt

    attention_cfg = evidence_cfg.get("attention", {}) if isinstance(evidence_cfg.get("attention"), Mapping) else {}
    limit = int(attention_cfg.get("max_cases") or cfg.get("sampling", {}).get("max_attention_cases", 32) or 32)
    aggregation = str(attention_cfg.get("aggregation", "mean_time_query"))
    out_dir = figures_dir / "evidence_attention"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    written = 0
    for model, analysis in analyses.items():
        sample_rows = {str(row.get("sample_id")): row for row in getattr(analysis, "sample_rows", []) or []}
        for sample_id, grid in (getattr(analysis, "attention_maps", {}) or {}).items():
            if written >= limit:
                return generated
            row = sample_rows.get(str(sample_id), {})
            fig, ax = plt.subplots(figsize=(4.5, 4.0))
            im = ax.imshow(np.asarray(grid), cmap="magma")
            ax.set_title(_attention_title(model, sample_id, row, aggregation), fontsize=9)
            ax.set_xlabel("patch x")
            ax.set_ylabel("patch y")
            fig.colorbar(im, ax=ax, fraction=0.046, label="attention")
            fig.tight_layout()
            generated.extend(_save_figure(fig, out_dir / f"patch_grid_{_safe_slug(model)}_{_safe_slug(sample_id)}", formats, dpi))
            image_path = str(row.get("image_path", "") or row.get("image", ""))
            if image_path:
                generated.extend(_write_overlay_figure(out_dir, model, sample_id, row, grid, image_path, formats, dpi, warnings))
            else:
                warnings.append(f"attention_overlay_unavailable:{model}:{sample_id}:missing_image")
            written += 1
    if written == 0:
        warnings.append("attention_unavailable:evidence:no_attention_maps")
    return generated


def _write_overlay_figure(
    out_dir: Path,
    model: str,
    sample_id: str,
    row: Mapping[str, Any],
    grid: np.ndarray,
    image_path: str,
    formats: tuple[str, ...],
    dpi: int,
    warnings: list[str],
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt

        image = plt.imread(image_path)
    except Exception as exc:
        warnings.append(f"attention_overlay_unavailable:{model}:{sample_id}:{exc}")
        return []
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(image)
    height, width = image.shape[:2]
    im = ax.imshow(np.asarray(grid), cmap="magma", alpha=0.45, extent=(0, width, height, 0))
    ax.set_title(_attention_title(model, sample_id, row, "image_overlay"), fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, label="attention alpha=0.45")
    fig.tight_layout()
    return _save_figure(fig, out_dir / f"overlay_{_safe_slug(model)}_{_safe_slug(sample_id)}", formats, dpi)


def _attention_title(model: str, sample_id: str, row: Mapping[str, Any], aggregation: str) -> str:
    topk = row.get("top3", row.get("top5", ""))
    return (
        f"{model} | sample={sample_id}\n"
        f"scene={row.get('scene', '')} condition={row.get('condition', '')} target={row.get('target', '')}\n"
        f"Top-k={topk} agg={aggregation}"
    )


def _attention_provenance(analyses: Mapping[str, Any]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for name, analysis in analyses.items():
        rows = list(getattr(analysis, "attention_rows", []) or [])
        first = rows[0] if rows else {}
        models[name] = {
            "available": bool(rows),
            "map_semantics": "token_read_map",
            "causal_claim": False,
            "attention_source": first.get("attention_source", "gps_query_pooler" if rows else ""),
            "attention_tensor_shape": first.get("attention_tensor_shape", ""),
            "token_grid": [first.get("token_grid_height", ""), first.get("token_grid_width", "")] if rows else [],
            "aggregation_method": first.get("aggregation_method", "mean_time_query" if rows else ""),
            "normalization_scope": first.get("normalization_scope", "per_sample_shared_minmax" if rows else ""),
            "overlay_image_source": first.get("overlay_image_source", "raw_image_or_model_input_tensor" if rows else ""),
            "cross_sample_comparability": False,
        }
    return {
        "map_semantics": "token_read_map",
        "causal_claim": False,
        "models": models,
    }


def _write_case_payloads_and_panels(
    cases_dir: Path,
    figures_dir: Path,
    rows: list[dict[str, Any]],
    analyses: Mapping[str, Any],
    formats: tuple[str, ...],
    dpi: int,
    warnings: list[str],
) -> list[Path]:
    generated: list[Path] = []
    panel_dir = figures_dir / "evidence_cases"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        payload = _case_payload(row, analyses)
        path = cases_dir / f"{_safe_slug(row.get('group', row.get('case_group', 'case')))}_{_safe_slug(row.get('sample_id'))}.json"
        path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        generated.append(path)
        generated.extend(_write_case_panel(panel_dir, row, payload, analyses, formats, dpi, warnings))
    return generated


def _case_payload(row: Mapping[str, Any], analyses: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = str(row.get("sample_id"))
    payload = {
        "sample_id": sample_id,
        "group": row.get("group", row.get("case_group")),
        "selection_reason": row.get("selection_reason"),
        "target": row.get("target"),
        "scene": row.get("scene"),
        "condition": row.get("condition"),
        "baseline_error": row.get("baseline_error"),
        "query_error": row.get("query_error"),
        "metric_delta": row.get("metric_delta"),
        "models": {},
    }
    for name, analysis in analyses.items():
        match = next((item for item in getattr(analysis, "sample_rows", []) if str(item.get("sample_id")) == sample_id), None)
        if match is None:
            continue
        payload["models"][name] = {
            "top1": match.get("top1"),
            "top3": match.get("top3"),
            "top5": match.get("top5"),
            "target_rank": match.get("target_rank"),
            "top1_error": match.get("top1_error"),
            "top3_min_distance": match.get("top3_min_distance"),
            "dba_contribution": match.get("dba_contribution"),
            "gt_probability": match.get("gt_probability"),
            "attention_available": sample_id in (getattr(analysis, "attention_maps", {}) or {}),
        }
    return payload


def _write_case_panel(
    panel_dir: Path,
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    analyses: Mapping[str, Any],
    formats: tuple[str, ...],
    dpi: int,
    warnings: list[str],
) -> list[Path]:
    if not _matplotlib_available(warnings):
        return []
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].axis("off")
    axes[0].text(
        0.02,
        0.96,
        f"{row.get('group', row.get('case_group'))} | sample {row.get('sample_id')}\n"
        f"target={row.get('target')} baseline_error={row.get('baseline_error')} query_error={row.get('query_error')}\n"
        f"{row.get('selection_reason', '')}",
        va="top",
        fontsize=9,
    )
    names = list(payload.get("models", {}).keys())
    values = [float(payload["models"][name].get("dba_contribution", 0.0) or 0.0) for name in names]
    axes[0].bar(names, values, color="#228833")
    axes[0].set_title("DBA contribution")
    axes[0].tick_params(axis="x", rotation=20)
    sample_id = str(row.get("sample_id"))
    attention_map = None
    attention_model = ""
    for model, analysis in analyses.items():
        maps = getattr(analysis, "attention_maps", {}) or {}
        if sample_id in maps:
            attention_map = maps[sample_id]
            attention_model = model
            break
    if attention_map is None:
        axes[1].axis("off")
        axes[1].text(0.08, 0.55, "attention unavailable", fontsize=10)
    else:
        im = axes[1].imshow(attention_map, cmap="magma")
        axes[1].set_title(f"{attention_model} attention")
        fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    return _save_figure(fig, panel_dir / f"{_safe_slug(row.get('group', row.get('case_group')))}_{_safe_slug(sample_id)}", formats, dpi)


def _evidence_report_lines(
    claim_rows: list[dict[str, Any]],
    paired_delta_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    warnings: list[str],
    faithfulness_context: Mapping[str, Any] | None = None,
) -> list[str]:
    statuses = ", ".join(f"{row['claim']}={row['status']}" for row in claim_rows)
    faithfulness = dict(faithfulness_context or {})
    faith_line = (
        f"- attention_faithfulness.csv: status={faithfulness.get('status')}, "
        f"passed={faithfulness.get('passed_sample_count', 0)}, failed={faithfulness.get('failed_sample_count', 0)}。"
        if faithfulness.get("enabled")
        else "- attention faithfulness 未启用；attention 解释项保持 exploratory。"
    )
    return [
        "",
        "## GPS-query 证据门控",
        f"- claim_gate_summary.csv: {statuses or 'no claims'}。",
        f"- paired_delta_by_condition.csv 记录 {len(paired_delta_rows)} 行同 split/seed/metric/condition delta；anchor_comparisons.csv 只作外部参考。",
        f"- case_selection.csv 与 cases/*.json 覆盖 {len(case_rows)} 个 deterministic case，包含 gain、regression、near-miss/failure 时才适合展示。",
        "- Attention hotspot 语义为 token-read map，只作为解释性证据，不是因果证明。",
        faith_line,
        *( [f"- Evidence warnings: {', '.join(sorted(set(warnings)))}。"] if warnings else [] ),
    ]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _csv_scalar(row.get(key, "")) for key in fieldnames})
    return path


def _save_figure(fig: Any, stem: Path, formats: tuple[str, ...], dpi: int) -> list[Path]:
    import matplotlib.pyplot as plt

    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _matplotlib_available(warnings: list[str]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        return True
    except Exception as exc:
        warnings.append(f"matplotlib_unavailable:{exc}")
        return False


def _output_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": _relative(path, root),
        "kind": _kind(path),
        "status": "generated",
        "size_bytes": int(path.stat().st_size) if path.exists() else 0,
    }


def _kind(path: Path) -> str:
    if path.suffix == ".csv":
        return "table"
    if path.suffix == ".json":
        return "case_payload" if "cases" in path.parts else "manifest"
    if path.suffix.lower() in {".png", ".svg", ".pdf"}:
        return "figure"
    return "artifact"


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _first(raw: Mapping[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        value = raw.get(key)
        if value not in (None, ""):
            return value
    return default


def _float_or_blank(value: Any) -> float | str:
    out = _float_or_none(value)
    return "" if out is None else out


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int_or_zero(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _is_clean_condition(value: str) -> bool:
    lowered = value.lower()
    return "p0" in lowered or "clean" in lowered or value in {"all", "none"}


def _safe_slug(value: Any) -> str:
    text = str(value)
    cleaned = [char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text]
    return "".join(cleaned).strip("._") or "item"


def _csv_scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
