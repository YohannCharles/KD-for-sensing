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

from kd_sensing.eval.missing_patterns import canonical_missing_pattern_name


CORE_PATTERNS = ("full", "missing_gps", "missing_radar", "radar_only", "lidar_only")
METRICS = ("full", "avg_missing", "missing_gps", "missing_radar", "radar_only", "lidar_only", "overall_mean", "balanced")
DELTA_COLUMNS = {metric: f"delta_{metric}" for metric in METRICS}
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
FILTER_THRESHOLDS = {
    "avg_missing": 0.2900,
    "full": 0.4078,
    "overall_mean": PROTO_REFERENCE["overall_mean"] + 0.0100,
}
EXPECTED_P0_RUNS = (
    "proto_sampler_uniform_es40_seed3",
    "proto_sampler_uniform_es40_seed4",
    "proto_sampler_uniform_es40_seed5",
    "proto_condbtapa_weaksingle_lam005_es40_seed3",
    "proto_condbtapa_weaksingle_lam005_es40_seed4",
    "proto_condbtapa_weaksingle_lam005_es40_seed5",
    "proto_sampler_uniform_condbtapa_weaksingle_lam005_es40_seed1",
    "proto_sampler_uniform_condbtapa_weaksingle_lam005_es40_seed2",
    "proto_sampler_uniform_condbtapa_weaksingle_lam005_es40_seed3",
    "proto_sampler_uniform_condbtapa_weaksingle_lam0025_es40_seed1",
    "proto_sampler_uniform_condbtapa_weaksingle_lam0025_es40_seed2",
    "proto_sampler_uniform_condbtapa_weaksingle_lam0025_es40_seed3",
)
PRIMARY_SORT = ("avg_missing", "full", "overall_mean", "balanced")
OVERALL_SORT = ("overall_mean", "avg_missing", "full")
BALANCED_SORT = ("balanced", "avg_missing", "full", "overall_mean")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_manifest(Path(args.manifest)) if args.manifest else []
    manifest = {row.get("run_name", ""): row for row in manifest_rows}
    metric_rows = _load_metric_rows(args)
    per_run = _per_run_rows(metric_rows, manifest)
    _apply_derived_metrics(per_run, PROTO_REFERENCE)
    method_rows = _method_rows(per_run, PROTO_REFERENCE)
    delta_rows = _delta_rows(method_rows)
    filtered_rows = _filtered_rows(method_rows)
    expected_runs = _expected_runs(manifest_rows)
    warnings = _sanity_warnings(per_run, manifest, expected_runs)

    _write_csv(out_dir / "p0_per_run.csv", per_run, _per_run_fields())
    _write_csv(out_dir / "p0_method_mean_std.csv", method_rows, _method_fields())
    _write_csv(out_dir / "p0_delta_vs_proto.csv", delta_rows, _delta_fields())
    _write_csv(out_dir / "p0_filtered.csv", filtered_rows, _filtered_fields())
    _write_rank_markdown(
        out_dir / "p0_rank_by_avg_missing.md",
        "P0 Rank By Avg Missing",
        _sorted_methods(method_rows, PRIMARY_SORT),
        warnings,
    )
    _write_rank_markdown(
        out_dir / "p0_rank_by_overall.md",
        "P0 Rank By Overall Mean",
        _sorted_methods(method_rows, OVERALL_SORT),
        warnings,
    )
    _write_rank_markdown(
        out_dir / "p0_rank_by_balanced_aux.md",
        "P0 Rank By Balanced Aux",
        _sorted_methods(method_rows, BALANCED_SORT),
        warnings,
    )
    _write_filtered_markdown(out_dir / "p0_filtered.md", filtered_rows)
    _write_sanity_markdown(out_dir / "p0_sanity_check.md", warnings, per_run, expected_runs)

    _print_conclusions(method_rows, warnings)
    print(f"Wrote Scene31 P0 fresh-eval summary to {out_dir}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Scene31 P0 fresh eval metrics.")
    parser.add_argument("--root", action="append", default=None, help="Output root. Can be repeated.")
    parser.add_argument("--run-dir", action="append", default=[], help="Explicit run output directory. Can be repeated.")
    parser.add_argument("--metrics", action="append", default=[], help="Explicit metrics CSV. Can be repeated.")
    parser.add_argument("--manifest", default="configs/scene31/next_round/experiment_manifest.csv")
    parser.add_argument("--out", default="outputs/scene31_next_round/p0_fresh_summary")
    return parser


def _load_metric_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in args.metrics:
        rows.extend(_rows_from_metrics_csv(Path(item)))
    roots = [Path(item) for item in (args.root or ["outputs/scene31_next_round"])]
    for root in roots:
        for pattern in (
            "p0_fresh_eval/*/apples_to_apples_metrics.csv",
            "analysis/p0_fresh_eval/*/apples_to_apples_metrics.csv",
            "analysis/night_grid/fresh_eval/night_grid_metrics.csv",
            "analysis/next_round/fresh_eval/night_grid_metrics.csv",
            "eval/*_missing_patterns.csv",
            "*/eval_matrix.csv",
        ):
            for path in sorted(root.glob(pattern)):
                run_name = None
                if path.name == "apples_to_apples_metrics.csv":
                    run_name = path.parent.name
                elif path.name.endswith("_missing_patterns.csv"):
                    run_name = path.name.removesuffix("_missing_patterns.csv")
                elif path.name == "eval_matrix.csv":
                    run_name = path.parent.name
                rows.extend(_rows_from_metrics_csv(path, run_name=run_name))
    for item in args.run_dir:
        run_dir = Path(item)
        for path in (
            run_dir / "apples_to_apples_metrics.csv",
            run_dir / "eval_matrix.csv",
            run_dir / f"{run_dir.name}_missing_patterns.csv",
        ):
            if path.exists():
                rows.extend(_rows_from_metrics_csv(path, run_name=run_dir.name))
    return rows


def _rows_from_metrics_csv(path: Path, *, run_name: str | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = _read_csv(path)
    manifest = _sibling_checkpoint_manifest(path)
    has_max_batches = (bool(rows) and "max_batches" in rows[0]) or "max_batches" in manifest
    max_batches = manifest.get("max_batches", "")
    run_manifest = manifest.get("runs", {}) if isinstance(manifest.get("runs"), dict) else {}
    for row in rows:
        resolved_run = run_name or str(row.get("run_name") or "")
        if resolved_run and not row.get("run_name"):
            row["run_name"] = resolved_run
        if max_batches != "" and row.get("max_batches") in (None, ""):
            row["max_batches"] = max_batches
        row["_max_batches_recorded"] = "yes" if has_max_batches else "no"
        if resolved_run in run_manifest and isinstance(run_manifest[resolved_run], dict):
            run_info = run_manifest[resolved_run]
            if not row.get("checkpoint_path"):
                row["checkpoint_path"] = run_info.get("checkpoint_path", "")
            if not row.get("checkpoint_epoch"):
                row["checkpoint_epoch"] = run_info.get("checkpoint_epoch", "")
        row["metrics_path"] = str(path)
    return rows


def _sibling_checkpoint_manifest(path: Path) -> dict[str, Any]:
    candidate = path.parent / "checkpoint_manifest.json"
    if not candidate.exists():
        return {}
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _per_run_rows(rows: list[dict[str, Any]], manifest: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    raw_patterns: dict[str, set[str]] = defaultdict(set)
    row_meta: dict[str, dict[str, Any]] = {}
    statuses: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        run_name = str(row.get("run_name") or "")
        if not run_name:
            continue
        pattern = canonical_missing_pattern_name(str(row.get("pattern") or ""))
        status = str(row.get("status") or "")
        statuses[run_name].add(status)
        raw_patterns[run_name].add(pattern)
        if status not in {"", "ok"}:
            row_meta.setdefault(run_name, row)
            continue
        value = _float(row.get("top1"))
        if _isnum(value):
            grouped[run_name][pattern].append(value)
        row_meta.setdefault(run_name, row)

    out: list[dict[str, Any]] = []
    for run_name in sorted(set(grouped) | set(row_meta)):
        meta = manifest.get(run_name, {})
        values = {pattern: _mean(items) for pattern, items in grouped.get(run_name, {}).items() if items}
        avg_from_patterns = _avg_missing_from_patterns(values)
        avg_source = values.get("avg_missing")
        avg_diff = abs(avg_source - avg_from_patterns) if _isnum(avg_source) and _isnum(avg_from_patterns) else float("nan")
        if not _isnum(avg_source) and _isnum(avg_from_patterns):
            values["avg_missing"] = avg_from_patterns
        first = row_meta.get(run_name, {})
        row = {
            "run_name": run_name,
            "method": _method_name(run_name),
            "group": meta.get("group", first.get("group", "")),
            "seed": meta.get("seed", first.get("seed", _seed_from_name(run_name))),
            "checkpoint_path": first.get("checkpoint_path", ""),
            "checkpoint_epoch": first.get("checkpoint_epoch", ""),
            "status": _status(statuses.get(run_name, set())),
            "metrics_path": first.get("metrics_path", ""),
            "max_batches": first.get("max_batches", ""),
            "_max_batches_recorded": first.get("_max_batches_recorded", "no"),
            "_patterns": raw_patterns.get(run_name, set()),
            "_avg_missing_expected": avg_from_patterns,
            "_avg_missing_diff": avg_diff,
        }
        for metric in ("full", "avg_missing", "missing_gps", "missing_radar", "radar_only", "lidar_only"):
            row[metric] = values.get(metric, float("nan"))
        out.append(row)
    return out


def _apply_derived_metrics(rows: list[dict[str, Any]], proto: dict[str, float]) -> None:
    for row in rows:
        row["overall_mean"] = _overall_mean(row)
        row["balanced"] = _balanced(row, proto)
        deltas = []
        for metric in METRICS:
            value = _float(row.get(metric))
            base = _float(proto.get(metric))
            delta = value - base if _isnum(value) and _isnum(base) else float("nan")
            row[DELTA_COLUMNS[metric]] = delta
            if _isnum(delta):
                deltas.append(delta)
        row["sanity_warnings"] = ""


def _method_rows(per_run: list[dict[str, Any]], proto: dict[str, float]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_run:
        grouped[str(row["method"])].append(row)
    out: list[dict[str, Any]] = []
    for method, rows in grouped.items():
        item: dict[str, Any] = {"method": method, "n": len(rows)}
        for metric in METRICS:
            values = [_float(row.get(metric)) for row in rows if _isnum(row.get(metric))]
            item[f"{metric}_mean"] = mean(values) if values else float("nan")
            item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else float("nan")
            base = _float(proto.get(metric))
            value = _float(item[f"{metric}_mean"])
            item[f"{DELTA_COLUMNS[metric]}_mean"] = value - base if _isnum(value) and _isnum(base) else float("nan")
        out.append(item)
    return _sorted_methods(out, PRIMARY_SORT)


def _delta_rows(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in method_rows:
        item = {"method": row["method"], "n": row.get("n", "")}
        for metric in METRICS:
            item[metric] = row.get(f"{metric}_mean")
            item[DELTA_COLUMNS[metric]] = row.get(f"{DELTA_COLUMNS[metric]}_mean")
        out.append(item)
    return out


def _filtered_rows(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in method_rows:
        unmet = _unmet_conditions(row)
        rows.append(
            {
                "method": row["method"],
                "n": row.get("n", ""),
                "full": row.get("full_mean"),
                "avg_missing": row.get("avg_missing_mean"),
                "overall_mean": row.get("overall_mean_mean"),
                "balanced": row.get("balanced_mean"),
                "passed": "yes" if not unmet else "",
                "unmet_conditions": ";".join(unmet),
            }
        )
    passed = [row for row in rows if row["passed"] == "yes"]
    if passed:
        return sorted(passed, key=lambda row: _rank_key(row, PRIMARY_SORT), reverse=True)
    return sorted(rows, key=lambda row: (len(_split_unmet(row)), tuple(-value for value in _rank_key(row, PRIMARY_SORT))))[:10]


def _unmet_conditions(row: dict[str, Any]) -> list[str]:
    unmet = []
    for metric, threshold in FILTER_THRESHOLDS.items():
        value = _float(row.get(f"{metric}_mean"))
        if not _isnum(value) or value < threshold:
            unmet.append(f"{metric}>={threshold:.4f}")
    return unmet


def _sanity_warnings(
    per_run: list[dict[str, Any]],
    manifest: dict[str, dict[str, str]],
    expected_runs: list[str],
) -> list[str]:
    warnings: list[str] = []
    by_run = {row["run_name"]: row for row in per_run}
    for run_name in expected_runs:
        if run_name not in by_run:
            warnings.append(f"{run_name}: missing complete fresh eval metrics")
    required_patterns = {"full", "missing_gps", "missing_radar", "radar_only", "lidar_only"}
    for row in per_run:
        run_name = str(row["run_name"])
        checkpoint = str(row.get("checkpoint_path") or "")
        if not checkpoint:
            warnings.append(f"{run_name}: best checkpoint path is missing")
        elif not Path(checkpoint).exists():
            warnings.append(f"{run_name}: best checkpoint path does not exist: {checkpoint}")
        if row.get("_max_batches_recorded") != "yes":
            warnings.append(f"{run_name}: max_batches metadata is missing; cannot verify full fresh eval")
        elif str(row.get("max_batches") or "") not in {"", "None", "null"}:
            warnings.append(f"{run_name}: fresh eval appears to use max_batches={row.get('max_batches')}")
        patterns = set(row.get("_patterns") or set())
        missing = sorted(required_patterns - patterns)
        if missing:
            warnings.append(f"{run_name}: missing required pattern rows: {','.join(missing)}")
        avg_diff = _float(row.get("_avg_missing_diff"))
        if _isnum(avg_diff) and avg_diff > 1e-6:
            warnings.append(f"{run_name}: avg_missing differs from available missing-pattern mean by {avg_diff:.8g}")
        if not _isnum(row.get("overall_mean")):
            warnings.append(f"{run_name}: overall_mean could not be computed from {','.join(CORE_PATTERNS)}")
        expected_method = re.sub(r"_seed\d+$", "", run_name)
        if re.search(r"_seed\d+$", run_name) and row.get("method") != expected_method:
            warnings.append(f"{run_name}: method merge expected {expected_method}, got {row.get('method')}")
        tags = str(manifest.get(run_name, {}).get("method_tags", ""))
        if "lam0025" in run_name and tags and "lambda_0.025" not in tags:
            warnings.append(f"{run_name}: lam0025 manifest tag is not lambda_0.025")
        if "lam005" in run_name and tags and "lambda_0.05" not in tags:
            warnings.append(f"{run_name}: lam005 manifest tag is not lambda_0.05")
    return warnings


def _write_rank_markdown(path: Path, title: str, rows: list[dict[str, Any]], warnings: list[str]) -> None:
    lines = [f"# {title}", ""]
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings[:20])
        if len(warnings) > 20:
            lines.append(f"- ... {len(warnings) - 20} more warnings; see p0_sanity_check.md")
        lines.append("")
    columns = [
        "method",
        "n",
        "full",
        "avg_missing",
        "overall_mean",
        "missing_gps",
        "missing_radar",
        "radar_only",
        "lidar_only",
        "balanced",
        "main_read",
    ]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append(
            "| {method} | {n} | {full} | {avg} | {overall} | {gps} | {radar} | {radar_only} | {lidar_only} | {balanced} | {read} |".format(
                method=_cell(row["method"]),
                n=row.get("n", ""),
                full=_mean_std(row, "full"),
                avg=_mean_std(row, "avg_missing"),
                overall=_mean_std(row, "overall_mean"),
                gps=_mean_std(row, "missing_gps"),
                radar=_mean_std(row, "missing_radar"),
                radar_only=_mean_std(row, "radar_only"),
                lidar_only=_mean_std(row, "lidar_only"),
                balanced=_mean_std(row, "balanced"),
                read=_cell(_main_read(row)),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_filtered_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# P0 Filtered", ""]
    if rows and all(row.get("passed") != "yes" for row in rows):
        lines.extend(["No method met all thresholds; showing closest top 10.", ""])
    lines.append("| method | n | full | avg_missing | overall_mean | balanced | passed | unmet_conditions |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in rows:
        lines.append(
            f"| {_cell(row['method'])} | {row.get('n', '')} | {_fmt(row.get('full'))} | {_fmt(row.get('avg_missing'))} | "
            f"{_fmt(row.get('overall_mean'))} | {_fmt(row.get('balanced'))} | {row.get('passed', '')} | {_cell(row.get('unmet_conditions', ''))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sanity_markdown(path: Path, warnings: list[str], per_run: list[dict[str, Any]], expected_runs: list[str]) -> None:
    lines = ["# P0 Sanity Check", ""]
    lines.append(f"expected_p0_runs: {len(expected_runs)}")
    lines.append(f"runs_with_metrics: {len(per_run)}")
    lines.append(f"warning_count: {len(warnings)}")
    lines.append("")
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("No sanity warnings.")
    lines.append("")
    lines.extend(
        [
            "## Checks",
            "",
            "- best checkpoint path present and exists",
            "- complete fresh eval metrics present for expected P0 runs",
            "- max_batches is empty",
            "- required patterns include full, missing_gps, missing_radar, radar_only, lidar_only",
            "- avg_missing matches available missing-pattern mean when both are present",
            "- overall_mean uses full, missing_gps, missing_radar, radar_only, lidar_only",
            "- method merge strips trailing _seedN",
            "- lam0025 and lam005 manifest tags match lambda_0.025 and lambda_0.05",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_conclusions(method_rows: list[dict[str, Any]], warnings: list[str]) -> None:
    avg_winner = _best_method(method_rows, PRIMARY_SORT)
    overall_winner = _best_method(method_rows, OVERALL_SORT)
    balanced_winner = _best_method(method_rows, BALANCED_SORT)
    print("Winner selection")
    print(f"Primary winner by avg_missing: {_winner_label(avg_winner, 'avg_missing')}")
    print(f"Primary winner by overall_mean: {_winner_label(overall_winner, 'overall_mean')}")
    print(f"Auxiliary winner by balanced: {_winner_label(balanced_winner, 'balanced')}")
    print("Uniform vs Uniform+condBTAPA lambda=0.025:")
    _print_comparison(
        method_rows,
        "proto_sampler_uniform_condbtapa_weaksingle_lam0025_es40",
        "proto_sampler_uniform_es40",
        ("avg_missing", "full", "overall_mean", "balanced"),
    )
    print("Lambda sensitivity:")
    _print_comparison(
        method_rows,
        "proto_sampler_uniform_condbtapa_weaksingle_lam0025_es40",
        "proto_sampler_uniform_condbtapa_weaksingle_lam005_es40",
        ("avg_missing", "full", "overall_mean", "balanced"),
    )
    if warnings:
        print(f"Sanity warnings: {len(warnings)}; see p0_sanity_check.md")


def _print_comparison(method_rows: list[dict[str, Any]], lhs: str, rhs: str, metrics: tuple[str, ...]) -> None:
    left = _method_lookup(method_rows, lhs)
    right = _method_lookup(method_rows, rhs)
    if left is None or right is None:
        print(f"- {lhs} vs {rhs}: unavailable")
        return
    for metric in metrics:
        lv = _float(left.get(f"{metric}_mean"))
        rv = _float(right.get(f"{metric}_mean"))
        delta = lv - rv if _isnum(lv) and _isnum(rv) else float("nan")
        print(f"- {metric}: {_fmt(lv)} vs {_fmt(rv)}, delta={_fmt(delta)}, exceeds={_yes_no(delta)}")


def _expected_runs(manifest_rows: list[dict[str, str]]) -> list[str]:
    runs = [row["run_name"] for row in manifest_rows if row.get("group") == "p0" and row.get("run_name")]
    return runs or list(EXPECTED_P0_RUNS)


def _method_name(run_name: str) -> str:
    return re.sub(r"_seed\d+$", "", run_name)


def _seed_from_name(run_name: str) -> str:
    match = re.search(r"_seed(\d+)$", run_name)
    return match.group(1) if match else ""


def _avg_missing_from_patterns(values: dict[str, float]) -> float:
    candidates = [
        value
        for pattern, value in values.items()
        if pattern not in {"full", "avg_missing", "overall_mean", "balanced"} and _isnum(value)
    ]
    return _mean(candidates)


def _overall_mean(row: dict[str, Any]) -> float:
    values = [_float(row.get(pattern)) for pattern in CORE_PATTERNS]
    return mean(values) if all(_isnum(value) for value in values) else float("nan")


def _balanced(row: dict[str, Any], proto: dict[str, float]) -> float:
    avg_missing = _float(row.get("avg_missing"))
    if not _isnum(avg_missing):
        return float("nan")
    score = avg_missing
    score += 0.25 * _zero_nan(row.get("radar_only"))
    score += 0.25 * _zero_nan(row.get("lidar_only"))
    score -= 0.5 * max(0.0, _zero_nan(proto.get("missing_gps")) - _zero_nan(row.get("missing_gps")))
    score -= 0.5 * max(0.0, _zero_nan(proto.get("missing_radar")) - _zero_nan(row.get("missing_radar")))
    score -= 0.25 * max(0.0, _zero_nan(proto.get("full")) - _zero_nan(row.get("full")))
    return score


def _sorted_methods(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: _method_rank_key(row, metrics), reverse=True)


def _method_rank_key(row: dict[str, Any], metrics: tuple[str, ...]) -> tuple[float, ...]:
    return tuple(_zero_nan(row.get(f"{metric}_mean")) for metric in metrics)


def _rank_key(row: dict[str, Any], metrics: tuple[str, ...]) -> tuple[float, ...]:
    return tuple(_zero_nan(row.get(metric)) for metric in metrics)


def _split_unmet(row: dict[str, Any]) -> list[str]:
    value = str(row.get("unmet_conditions") or "")
    return [item for item in value.split(";") if item]


def _best_method(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> dict[str, Any] | None:
    valid = [row for row in rows if _isnum(row.get(f"{metrics[0]}_mean"))]
    return max(valid, key=lambda row: _method_rank_key(row, metrics), default=None)


def _method_lookup(rows: list[dict[str, Any]], method: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("method") == method), None)


def _winner_label(row: dict[str, Any] | None, metric: str) -> str:
    if row is None:
        return "unavailable"
    return f"{row['method']} {metric}={_fmt(row.get(metric + '_mean'))}"


def _main_read(row: dict[str, Any]) -> str:
    return (
        f"delta_avg={_fmt(row.get('delta_avg_missing_mean'))}; "
        f"delta_overall={_fmt(row.get('delta_overall_mean_mean'))}; "
        f"delta_full={_fmt(row.get('delta_full_mean'))}"
    )


def _per_run_fields() -> list[str]:
    return [
        "run_name",
        "method",
        "group",
        "seed",
        "checkpoint_path",
        "checkpoint_epoch",
        "status",
        "metrics_path",
        "max_batches",
        *METRICS,
        *[DELTA_COLUMNS[metric] for metric in METRICS],
        "sanity_warnings",
    ]


def _method_fields() -> list[str]:
    fields = ["method", "n"]
    for metric in METRICS:
        fields.extend([f"{metric}_mean", f"{metric}_std"])
    return fields


def _delta_fields() -> list[str]:
    return ["method", "n", *METRICS, *[DELTA_COLUMNS[metric] for metric in METRICS]]


def _filtered_fields() -> list[str]:
    return ["method", "n", "full", "avg_missing", "overall_mean", "balanced", "passed", "unmet_conditions"]


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return _read_csv(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return _fmt(value)
    return value


def _mean_std(row: dict[str, Any], metric: str) -> str:
    return f"{_fmt(row.get(metric + '_mean'))}+/-{_fmt(row.get(metric + '_std'))}"


def _mean(values: list[float]) -> float:
    return mean(values) if values else float("nan")


def _status(statuses: set[str]) -> str:
    clean = {status for status in statuses if status}
    return "ok" if not clean or clean == {"ok"} else ";".join(sorted(clean))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _isnum(value: Any) -> bool:
    return math.isfinite(_float(value))


def _zero_nan(value: Any) -> float:
    number = _float(value)
    return number if math.isfinite(number) else 0.0


def _fmt(value: Any) -> str:
    number = _float(value)
    return f"{number:.8g}" if math.isfinite(number) else ""


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _yes_no(delta: float) -> str:
    return "unavailable" if not _isnum(delta) else "yes" if delta > 0 else "no"


if __name__ == "__main__":
    raise SystemExit(main())
