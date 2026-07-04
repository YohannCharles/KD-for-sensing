#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from kd_sensing.eval.missing_buckets import (
    BUCKET_COUNTS,
    bucket_metric_mean,
    missing_bucket_mapping_from_rows,
    write_missing_bucket_mapping,
)
from kd_sensing.eval.missing_patterns import canonical_missing_pattern_name


CORE_PATTERNS = ("full", "missing_gps", "missing_radar", "radar_only", "lidar_only")
TOP1_METRICS = ("full", "avg_missing", "overall_mean", "missing_gps", "missing_radar", "radar_only", "lidar_only", "balanced")
OPTIONAL_INPUT_METRICS = ("top3", "top5", "within_3", "mae")
BEAM_METRICS = ("top1", *OPTIONAL_INPUT_METRICS)
COMPARISON_METHODS = (
    "proto_sampler_uniform_beamsoft_s15_mix05_es40",
    "proto_sampler_uniform_beamsoft_s10_mix025_es40",
    "proto_sampler_uniform_beamsoft_s15_mix025_es40",
)
PROTO_REFERENCE = {
    "full": 0.4128,
    "avg_missing": 0.2752,
    "missing_gps": 0.3082,
    "missing_radar": 0.3412,
    "radar_only": 0.1471,
    "lidar_only": 0.0889,
    "balanced": 0.3176,
}
PROTO_REFERENCE["overall_mean"] = mean(PROTO_REFERENCE[pattern] for pattern in CORE_PATTERNS)
UNIFORM_REFERENCE = {
    "full": 0.4216,
    "avg_missing": 0.2856,
    "overall_mean": 0.2784,
    "balanced": 0.3560,
}
PRIMARY_SORT = ("avg_missing_top1", "miss2_top1", "miss3_top1", "full_top1")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(Path(args.manifest)) if args.manifest else {}
    metric_rows = _load_metric_rows(args)
    bucket_mapping, bucket_warnings = missing_bucket_mapping_from_rows(metric_rows)
    write_missing_bucket_mapping(out_dir / "missing_bucket_mapping.json", bucket_mapping)
    per_run = _per_run_rows(metric_rows, manifest, bucket_mapping)
    method_rows = _method_rows(per_run)
    delta_rows = _delta_rows(method_rows)
    warnings = sorted(dict.fromkeys([*bucket_warnings, *_sanity_warnings(per_run)]))
    conclusion = _conclusion_lines(method_rows)

    prefix = str(args.name_prefix or "")
    _write_csv(out_dir / _named(prefix, "per_run.csv"), per_run, _per_run_fields(per_run))
    _write_csv(out_dir / _named(prefix, "method_mean_std.csv"), method_rows, _method_fields(method_rows))
    _write_csv(out_dir / _named(prefix, "delta_vs_uniform.csv"), delta_rows, _delta_fields(method_rows))
    _write_rank_by_avg_missing(out_dir / _named(prefix, "rank_by_avg_missing_top1.md"), method_rows, warnings)
    for count in BUCKET_COUNTS:
        _write_rank_by_bucket(out_dir / _named(prefix, f"rank_by_miss{count}_top1.md"), method_rows, count, warnings)
    _write_rank_by_beam_proximity(out_dir / _named(prefix, "rank_by_beam_proximity.md"), method_rows, warnings)
    _write_sanity_markdown(out_dir / _named(prefix, "sanity_check.md"), warnings)
    _write_text(out_dir / _named(prefix, "conservative_conclusion.md"), conclusion)
    print("\n".join(conclusion))
    print(f"Wrote Scene31 BC summary to {out_dir}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Scene31 BC fresh eval metrics.")
    parser.add_argument("--root", action="append", default=None, help="Output root. Can be repeated.")
    parser.add_argument("--metrics", action="append", default=[], help="Explicit metrics CSV. Can be repeated.")
    parser.add_argument("--manifest", default="configs/scene31/next_round/experiment_manifest.csv")
    parser.add_argument("--out", default="outputs/scene31_bc_next/summary")
    parser.add_argument("--name-prefix", default="bc", help="Output filename prefix. Empty writes per_run.csv style names.")
    return parser


def _load_metric_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for item in args.metrics:
        path = Path(item)
        seen.add(path.resolve() if path.exists() else path)
        rows.extend(_rows_from_metrics_csv(path))
    roots = [Path(item) for item in (args.root or ["outputs/scene31_bc_next"])]
    for root in roots:
        for pattern in (
            "fresh_eval_main/*/apples_to_apples_metrics.csv",
            "fresh_eval_main/*/*/apples_to_apples_metrics.csv",
            "fresh_eval/*/apples_to_apples_metrics.csv",
            "fresh_eval/*/*/apples_to_apples_metrics.csv",
            "p0_fresh_eval/*/apples_to_apples_metrics.csv",
            "*/apples_to_apples_metrics.csv",
            "*/eval_matrix.csv",
        ):
            for path in sorted(root.glob(pattern)):
                resolved = path.resolve() if path.exists() else path
                if resolved in seen:
                    continue
                seen.add(resolved)
                rows.extend(_rows_from_metrics_csv(path, run_name=path.parent.name))
        for path in sorted(root.rglob("apples_to_apples_metrics.csv")):
            resolved = path.resolve() if path.exists() else path
            if resolved in seen:
                continue
            seen.add(resolved)
            rows.extend(_rows_from_metrics_csv(path))
    return rows


def _rows_from_metrics_csv(path: Path, *, run_name: str | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = _read_csv(path)
    manifest = _sibling_checkpoint_manifest(path)
    max_batches = manifest.get("max_batches", "")
    for row in rows:
        if run_name and not row.get("run_name"):
            row["run_name"] = run_name
        if max_batches != "" and row.get("max_batches") in (None, ""):
            row["max_batches"] = max_batches
        row["metrics_path"] = str(path)
    return rows


def _sibling_checkpoint_manifest(path: Path) -> dict[str, Any]:
    candidate = path.parent / "checkpoint_manifest.json"
    if not candidate.exists():
        return {}
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _per_run_rows(
    rows: list[dict[str, Any]],
    manifest: dict[str, dict[str, str]],
    bucket_mapping: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") not in {"", "ok", None}:
            continue
        run_name = str(row.get("run_name") or "")
        if not run_name:
            continue
        pattern = canonical_missing_pattern_name(str(row.get("pattern") or ""))
        for metric in ("top1", *OPTIONAL_INPUT_METRICS):
            value = _float(row.get(metric))
            if _isnum(value):
                grouped[run_name][pattern][metric].append(value)
        meta.setdefault(run_name, row)

    out: list[dict[str, Any]] = []
    for run_name in sorted(grouped):
        manifest_row = manifest.get(run_name, {})
        top1 = {pattern: _mean(values.get("top1", [])) for pattern, values in grouped[run_name].items()}
        if not _isnum(top1.get("avg_missing")):
            top1["avg_missing"] = _avg_missing(top1)
        row: dict[str, Any] = {
            "run_name": run_name,
            "method": _method_name(run_name),
            "group": manifest_row.get("group", meta.get(run_name, {}).get("group", "")),
            "seed": manifest_row.get("seed", meta.get(run_name, {}).get("seed", _seed_from_name(run_name))),
            "metrics_path": meta.get(run_name, {}).get("metrics_path", ""),
            "max_batches": meta.get(run_name, {}).get("max_batches", ""),
        }
        for metric in TOP1_METRICS:
            value = _derived_top1(metric, top1)
            row[metric] = value
            row[f"{metric}_top1"] = value
        _add_bucket_metric(row, grouped[run_name], bucket_mapping or {}, "top1")
        for metric in TOP1_METRICS:
            row[f"delta_{metric}"] = _delta(row.get(metric), PROTO_REFERENCE.get(metric))
        for metric in UNIFORM_REFERENCE:
            row[f"delta_vs_uniform_{metric}"] = _delta(row.get(metric), UNIFORM_REFERENCE.get(metric))
        for optional in OPTIONAL_INPUT_METRICS:
            _add_optional_metric(row, grouped[run_name], optional)
            if optional in {"within_3", "mae"}:
                _add_bucket_metric(row, grouped[run_name], bucket_mapping or {}, optional)
        out.append(row)
    return sorted(out, key=lambda item: _run_rank_key(item), reverse=True)


def _method_rows(per_run: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_run:
        grouped[str(row["method"])].append(row)
    metrics = [field for field in _numeric_fields(per_run) if field not in {"seed"}]
    out: list[dict[str, Any]] = []
    for method, rows in grouped.items():
        item: dict[str, Any] = {"method": method, "group": rows[0].get("group", ""), "n": len(rows)}
        for metric in metrics:
            values = [_float(row.get(metric)) for row in rows if _isnum(row.get(metric))]
            item[f"{metric}_mean"] = mean(values) if values else float("nan")
            item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else float("nan")
        out.append(item)
    return sorted(out, key=lambda item: _method_rank_key(item), reverse=True)


def _delta_rows(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    uniform = _method_by_name(method_rows).get("proto_sampler_uniform_es40")
    metrics = _comparison_metrics(method_rows)
    out: list[dict[str, Any]] = []
    for row in method_rows:
        item = {"method": row["method"], "n": row.get("n", "")}
        for metric in metrics:
            value = _method_value(row, metric)
            item[metric] = value
            if uniform is not None:
                base = _method_value(uniform, metric)
            else:
                base = UNIFORM_REFERENCE.get(metric)
                if base is None and metric.endswith("_top1"):
                    base = UNIFORM_REFERENCE.get(metric.removesuffix("_top1"))
            item[f"delta_vs_uniform_{metric}"] = _delta(value, base)
        out.append(item)
    return out


def _add_optional_metric(row: dict[str, Any], pattern_values: dict[str, dict[str, list[float]]], metric: str) -> None:
    values = {pattern: _mean(items.get(metric, [])) for pattern, items in pattern_values.items()}
    if any(_isnum(value) for value in values.values()):
        row[f"full_{metric}"] = values.get("full", float("nan"))
        avg_source = values.get("avg_missing", float("nan"))
        row[f"avg_missing_{metric}"] = avg_source if _isnum(avg_source) else _avg_missing(values)
        row[f"overall_mean_{metric}"] = _overall_mean(values)
        if metric == "within_3":
            row["full_within@3"] = row[f"full_{metric}"]
            row["avg_missing_within@3"] = row[f"avg_missing_{metric}"]
            row["overall_mean_within@3"] = row[f"overall_mean_{metric}"]


def _add_bucket_metric(
    row: dict[str, Any],
    pattern_values: dict[str, dict[str, list[float]]],
    bucket_mapping: dict[str, dict[str, Any]],
    metric: str,
) -> None:
    values = {pattern: _mean(items.get(metric, [])) for pattern, items in pattern_values.items()}
    suffix = "top1" if metric == "top1" else metric
    for count in BUCKET_COUNTS:
        value = bucket_metric_mean(values, bucket_mapping, count)
        row[f"miss{count}_{suffix}"] = value
        if metric == "within_3":
            row[f"miss{count}_within@3"] = value


def _derived_top1(metric: str, values: dict[str, float]) -> float:
    if metric == "overall_mean":
        return _overall_mean(values)
    if metric == "balanced":
        return _balanced(values)
    return values.get(metric, float("nan"))


def _avg_missing(values: dict[str, float]) -> float:
    candidates = [
        value
        for pattern, value in values.items()
        if pattern not in {"full", "avg_missing", "overall_mean", "balanced"} and _isnum(value)
    ]
    return _mean(candidates)


def _overall_mean(values: dict[str, float]) -> float:
    nums = [_float(values.get(pattern)) for pattern in CORE_PATTERNS]
    return mean(nums) if all(_isnum(value) for value in nums) else float("nan")


def _balanced(row: dict[str, Any]) -> float:
    avg_missing = _float(row.get("avg_missing"))
    if not _isnum(avg_missing):
        return float("nan")
    score = avg_missing
    score += 0.25 * _zero_nan(row.get("radar_only"))
    score += 0.25 * _zero_nan(row.get("lidar_only"))
    score -= 0.5 * max(0.0, _zero_nan(PROTO_REFERENCE.get("missing_gps")) - _zero_nan(row.get("missing_gps")))
    score -= 0.5 * max(0.0, _zero_nan(PROTO_REFERENCE.get("missing_radar")) - _zero_nan(row.get("missing_radar")))
    score -= 0.25 * max(0.0, _zero_nan(PROTO_REFERENCE.get("full")) - _zero_nan(row.get("full")))
    return score


def _write_rank_by_avg_missing(path: Path, rows: list[dict[str, Any]], warnings: list[str]) -> None:
    lines = ["# Scene31 Rank By Avg Missing Top1", ""]
    columns = [
        "method",
        "n",
        "full_top1",
        "miss1_top1",
        "miss2_top1",
        "miss3_top1",
        "avg_missing_top1",
        "overall_mean_top1",
        "balanced",
        "delta_vs_uniform_avg_missing",
    ]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in sorted(rows, key=lambda item: _method_rank_key(item), reverse=True):
        lines.append(
            "| {method} | {n} | {full} | {miss1} | {miss2} | {miss3} | {avg} | {overall} | {balanced} | {delta} |".format(
                method=row["method"],
                n=row.get("n", ""),
                full=_mean_std(row, "full_top1"),
                miss1=_mean_std(row, "miss1_top1"),
                miss2=_mean_std(row, "miss2_top1"),
                miss3=_mean_std(row, "miss3_top1"),
                avg=_mean_std(row, "avg_missing_top1"),
                overall=_mean_std(row, "overall_mean_top1"),
                balanced=_mean_std(row, "balanced"),
                delta=_fmt(row.get("delta_vs_uniform_avg_missing_mean")),
            )
        )
    _append_warnings(lines, warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_rank_by_bucket(path: Path, rows: list[dict[str, Any]], count: int, warnings: list[str]) -> None:
    metric = f"miss{count}_top1"
    lines = [f"# Scene31 Rank By Miss{count} Top1", ""]
    columns = ["method", "n", metric, "avg_missing_top1", "miss2_top1", "miss3_top1", "full_top1"]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in sorted(rows, key=lambda item: (_zero_nan(item.get(f"{metric}_mean")), *_method_rank_key(item)), reverse=True):
        lines.append(
            "| {method} | {n} | {bucket} | {avg} | {miss2} | {miss3} | {full} |".format(
                method=row["method"],
                n=row.get("n", ""),
                bucket=_mean_std(row, metric),
                avg=_mean_std(row, "avg_missing_top1"),
                miss2=_mean_std(row, "miss2_top1"),
                miss3=_mean_std(row, "miss3_top1"),
                full=_mean_std(row, "full_top1"),
            )
        )
    _append_warnings(lines, warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_rank_by_beam_proximity(path: Path, rows: list[dict[str, Any]], warnings: list[str]) -> None:
    lines = ["# Scene31 Rank By Beam Proximity", ""]
    columns = ["method", "n", "avg_missing_within_3", "avg_missing_mae", "avg_missing_top1"]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in sorted(rows, key=_beam_proximity_key):
        lines.append(
            "| {method} | {n} | {within} | {mae} | {top1} |".format(
                method=row["method"],
                n=row.get("n", ""),
                within=_mean_std(row, "avg_missing_within_3"),
                mae=_mean_std(row, "avg_missing_mae"),
                top1=_mean_std(row, "avg_missing_top1"),
            )
        )
    _append_warnings(lines, warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sanity_markdown(path: Path, warnings: list[str]) -> None:
    lines = ["# Scene31 Summary Sanity Check", ""]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- ok")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_warnings(lines: list[str], warnings: list[str]) -> None:
    if not warnings:
        return
    lines.extend(["", "## Sanity Warnings", ""])
    lines.extend(f"- {warning}" for warning in warnings)


def _per_run_fields(rows: list[dict[str, Any]]) -> list[str]:
    base = ["run_name", "method", "group", "seed", "metrics_path"]
    return base + [field for field in _numeric_fields(rows) if field not in set(base)]


def _method_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields = ["method", "group", "n"]
    for key in _numeric_method_fields(rows):
        fields.extend([key, key.removesuffix("_mean") + "_std"])
    return fields


def _delta_fields(rows: list[dict[str, Any]]) -> list[str]:
    metrics = _comparison_metrics(rows)
    return ["method", "n", *[item for metric in metrics for item in (metric, f"delta_vs_uniform_{metric}")]]


def _numeric_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and _isnum(value):
                fields.append(key)
    return fields


def _numeric_method_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key.endswith("_mean") and key not in fields and _isnum(value):
                fields.append(key)
    return fields


def _run_rank_key(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(_zero_nan(row.get(metric)) for metric in PRIMARY_SORT)


def _method_rank_key(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(_zero_nan(row.get(f"{metric}_mean")) for metric in PRIMARY_SORT)


def _beam_proximity_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        -_zero_nan(row.get("avg_missing_within_3_mean")),
        _nan_as_large(row.get("avg_missing_mae_mean")),
        -_zero_nan(row.get("avg_missing_top1_mean")),
    )


def _comparison_metrics(rows: list[dict[str, Any]]) -> list[str]:
    wanted = [
        "full",
        "avg_missing",
        "overall_mean",
        "balanced",
        "miss1_top1",
        "miss2_top1",
        "miss3_top1",
        "miss1_within_3",
        "miss2_within_3",
        "miss3_within_3",
        "miss1_mae",
        "miss2_mae",
        "miss3_mae",
        *[f"{scope}_{metric}" for metric in BEAM_METRICS for scope in ("full", "avg_missing", "overall_mean")],
    ]
    out: list[str] = []
    for metric in wanted:
        if metric in out:
            continue
        if metric in UNIFORM_REFERENCE or any(_isnum(_method_value(row, metric)) for row in rows):
            out.append(metric)
    return out


def _method_by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("method")): row for row in rows}


def _method_value(row: dict[str, Any] | None, metric: str) -> float:
    if row is None:
        return float("nan")
    return _float(row.get(f"{metric}_mean", row.get(metric)))


def _nan_as_large(value: Any) -> float:
    value_f = _float(value)
    return value_f if math.isfinite(value_f) else float("inf")


def _named(prefix: str, filename: str) -> str:
    if not prefix:
        return filename
    if filename == "rank_by_avg_missing_top1.md":
        filename = "rank_by_avg_missing.md"
    return f"{prefix}_{filename}"


def _sanity_warnings(per_run: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    required_patterns = {"full", "missing_gps", "missing_radar", "radar_only", "lidar_only"}
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in per_run:
        run_name = str(row.get("run_name", ""))
        if not run_name:
            continue
        max_batches = str(row.get("max_batches", "")).strip()
        if max_batches and max_batches.lower() not in {"none", "null"}:
            warnings.append(f"{run_name}: max_batches is recorded as {row.get('max_batches')}")
        for pattern in required_patterns:
            if _isnum(row.get(pattern)):
                grouped[run_name].add(pattern)
        for scope in ("full", "avg_missing", "overall_mean"):
            top1 = _float(row.get(f"{scope}_top1"))
            top3 = _float(row.get(f"{scope}_top3"))
            top5 = _float(row.get(f"{scope}_top5"))
            within = _float(row.get(f"{scope}_within_3"))
            mae = _float(row.get(f"{scope}_mae"))
            if _isnum(top1) and _isnum(top3) and top3 + 1e-12 < top1:
                warnings.append(f"{run_name}: {scope}_top3 < {scope}_top1")
            if _isnum(top3) and _isnum(top5) and top5 + 1e-12 < top3:
                warnings.append(f"{run_name}: {scope}_top5 < {scope}_top3")
            if _isnum(within) and not (0.0 <= within <= 1.0):
                warnings.append(f"{run_name}: {scope}_within_3 outside [0,1]")
            if _isnum(mae) and mae < 0.0:
                warnings.append(f"{run_name}: {scope}_mae < 0")
    for run_name, patterns in grouped.items():
        missing = required_patterns - patterns
        if missing:
            warnings.append(f"{run_name}: missing required patterns {','.join(sorted(missing))}")
    for count in BUCKET_COUNTS:
        if per_run and not any(_isnum(row.get(f"miss{count}_top1")) for row in per_run):
            warnings.append(f"miss{count} bucket has no usable top1 values")
    return sorted(dict.fromkeys(warnings))


def _conclusion_lines(method_rows: list[dict[str, Any]]) -> list[str]:
    by_method = _method_by_name(method_rows)
    exact = _best_method(method_rows, "avg_missing_top1", reverse=True)
    proximity = _best_beam_proximity_method(method_rows)
    lines = [
        "Beam-aware comparison:",
        *_comparison_block(by_method, "proto_sampler_uniform_beamsoft_s15_mix05_es40"),
        "",
        "Current exact-Top1 winner:",
        exact or "unavailable",
        "",
        "Current beam-proximity winner:",
        proximity or "unavailable",
        "",
    ]
    for method in COMPARISON_METHODS:
        lines.extend(_comparison_block(by_method, method, label=f"Uniform vs {method}:"))
        verdict = _beamsoft_verdict(by_method, method)
        if verdict:
            lines.append(verdict)
        lines.append("")
    return lines[:-1] if lines and lines[-1] == "" else lines


def _comparison_block(by_method: dict[str, dict[str, Any]], method: str, *, label: str | None = None) -> list[str]:
    uniform = by_method.get("proto_sampler_uniform_es40")
    candidate = by_method.get(method)
    display = method.replace("proto_sampler_uniform_", "")
    lines = [label or f"- uniform vs {display}"]
    for metric in ("avg_missing_top1", "avg_missing_within_3", "avg_missing_mae"):
        delta = _delta(_method_value(candidate, metric), _method_value(uniform, metric))
        lines.append(f"- {metric}: {_fmt(_method_value(candidate, metric))} (delta {_fmt(delta)})")
    return lines


def _beamsoft_verdict(by_method: dict[str, dict[str, Any]], method: str) -> str:
    uniform = by_method.get("proto_sampler_uniform_es40")
    candidate = by_method.get(method)
    if uniform is None or candidate is None:
        return ""
    top1_delta = _delta(_method_value(candidate, "avg_missing_top1"), _method_value(uniform, "avg_missing_top1"))
    within_delta = _delta(_method_value(candidate, "avg_missing_within_3"), _method_value(uniform, "avg_missing_within_3"))
    mae_delta = _delta(_method_value(candidate, "avg_missing_mae"), _method_value(uniform, "avg_missing_mae"))
    if method in {"proto_sampler_uniform_beamsoft_s10_mix025_es40", "proto_sampler_uniform_beamsoft_s15_mix025_es40"}:
        if _isnum(top1_delta) and top1_delta > 0:
            return "weak beamsoft improves exact missing-modality Top1"
        if (_isnum(within_delta) and within_delta > 0) or (_isnum(mae_delta) and mae_delta < 0):
            return "weak beamsoft improves beam proximity but not exact Top1"
        return "beamsoft does not outperform uniform under current metrics"
    if (_isnum(top1_delta) and top1_delta < 0) and ((_isnum(within_delta) and within_delta > 0) or (_isnum(mae_delta) and mae_delta < 0)):
        return "beamsoft improves beam-proximity metrics but not exact Top1"
    return ""


def _best_method(rows: list[dict[str, Any]], metric: str, *, reverse: bool) -> str:
    valid = [row for row in rows if _isnum(_method_value(row, metric))]
    if not valid:
        return ""
    row = sorted(valid, key=lambda item: _method_value(item, metric), reverse=reverse)[0]
    return str(row.get("method", ""))


def _best_beam_proximity_method(rows: list[dict[str, Any]]) -> str:
    valid = [row for row in rows if _isnum(_method_value(row, "avg_missing_within_3")) or _isnum(_method_value(row, "avg_missing_mae"))]
    if not valid:
        return ""
    return str(sorted(valid, key=_beam_proximity_key)[0].get("method", ""))


def _method_name(run_name: str) -> str:
    return re.sub(r"_seed\d+$", "", run_name)


def _seed_from_name(run_name: str) -> str:
    match = re.search(r"_seed(\d+)$", run_name)
    return match.group(1) if match else ""


def _read_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {row["run_name"]: row for row in _read_csv(path)}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(value) if isinstance(value, float) else value for key, value in row.items()})


def _mean(values: list[float]) -> float:
    return mean(float(value) for value in values) if values else float("nan")


def _mean_std(row: dict[str, Any], metric: str) -> str:
    return f"{_fmt(row.get(metric + '_mean'))}+/-{_fmt(row.get(metric + '_std'))}"


def _delta(value: Any, base: Any) -> float:
    value_f = _float(value)
    base_f = _float(base)
    return value_f - base_f if _isnum(value_f) and _isnum(base_f) else float("nan")


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _isnum(value: Any) -> bool:
    return math.isfinite(_float(value))


def _zero_nan(value: Any) -> float:
    value_f = _float(value)
    return value_f if math.isfinite(value_f) else 0.0


def _fmt(value: Any) -> str:
    value_f = _float(value)
    return f"{value_f:.8g}" if math.isfinite(value_f) else ""


if __name__ == "__main__":
    raise SystemExit(main())
