#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from kd_sensing.eval.missing_patterns import canonical_missing_pattern_name


METRICS = ("full", "avg_missing", "missing_gps", "missing_radar", "radar_only", "lidar_only", "balanced")
BASE_METRICS = METRICS[:-1]
DELTA_COLUMNS = {
    "full": "Δfull",
    "avg_missing": "Δavg_missing",
    "missing_gps": "Δmissing_gps",
    "missing_radar": "Δmissing_radar",
    "radar_only": "Δradar_only",
    "lidar_only": "Δlidar_only",
    "balanced": "Δbalanced",
}
PROTO_REFERENCE = {
    "full": 0.4128,
    "avg_missing": 0.2752,
    "missing_gps": 0.3082,
    "missing_radar": 0.3412,
    "radar_only": 0.1471,
    "lidar_only": 0.0889,
    "balanced": 0.3176,
}
BTAPA_TAU1_REFERENCE = {
    "full": 0.4137,
    "avg_missing": 0.2727,
    "missing_gps": 0.2915,
    "missing_radar": 0.2998,
    "radar_only": 0.1809,
    "lidar_only": 0.1040,
    "balanced": 0.3037,
}
FILTER_THRESHOLDS = {
    "full": 0.4078,
    "avg_missing": 0.2900,
    "radar_only": 0.2000,
    "lidar_only": 0.1150,
    "balanced": 0.3600,
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_manifest(Path(args.manifest)) if args.manifest else {}
    metric_rows = _load_metric_rows(args)
    per_run = _per_run_rows(metric_rows, manifest)
    proto = _proto_reference(per_run)
    _apply_deltas(per_run, proto)
    method_rows = _method_rows(per_run, proto)
    filtered_rows = _filtered_rows(method_rows)

    _write_csv(out_dir / "scene31_next_round_per_run.csv", per_run, _per_run_fields())
    _write_csv(out_dir / "scene31_next_round_method_mean_std.csv", method_rows, _method_fields())
    _write_csv(out_dir / "scene31_next_round_filtered.csv", filtered_rows, _filtered_fields())
    _write_csv(out_dir / "scene31_next_round_references.csv", _reference_rows(proto), ["reference", *METRICS])
    _write_markdown(out_dir / "scene31_next_round_summary.md", method_rows, filtered_rows, proto)
    print(f"Wrote Scene31 next-round summary to {out_dir}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Scene31 next-round fresh eval metrics.")
    parser.add_argument("--root", action="append", default=None, help="Output root. Can be repeated.")
    parser.add_argument("--run-dir", action="append", default=[], help="Explicit run directory. Can be repeated.")
    parser.add_argument("--metrics", action="append", default=[], help="Explicit fresh eval CSV. Can be repeated.")
    parser.add_argument("--manifest", default="configs/scene31/next_round/experiment_manifest.csv")
    parser.add_argument("--out", default="outputs/scene31_next_round/summary")
    return parser


def _load_metric_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in args.metrics:
        rows.extend(_rows_from_metrics_csv(Path(item)))
    roots = [Path(item) for item in (args.root or ["outputs/scene31_next_round"])]
    for root in roots:
        for path in (
            root / "analysis" / "night_grid" / "fresh_eval" / "night_grid_metrics.csv",
            root / "analysis" / "next_round" / "fresh_eval" / "night_grid_metrics.csv",
        ):
            if path.exists():
                rows.extend(_rows_from_metrics_csv(path))
        for path in sorted((root / "eval").glob("*_missing_patterns.csv")):
            run_name = path.name.removesuffix("_missing_patterns.csv")
            rows.extend(_rows_from_metrics_csv(path, run_name=run_name))
        for path in sorted(root.glob("*/eval_matrix.csv")):
            rows.extend(_rows_from_metrics_csv(path, run_name=path.parent.name))
    for item in args.run_dir:
        run_dir = Path(item)
        for path in (run_dir / "eval_matrix.csv", run_dir / f"{run_dir.name}_missing_patterns.csv"):
            if path.exists():
                rows.extend(_rows_from_metrics_csv(path, run_name=run_dir.name))
    return rows


def _rows_from_metrics_csv(path: Path, *, run_name: str | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = _read_csv(path)
    for row in rows:
        if run_name and not row.get("run_name"):
            row["run_name"] = run_name
    return rows


def _per_run_rows(rows: list[dict[str, Any]], manifest: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    row_meta: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("status") not in {"", "ok", None}:
            continue
        run_name = str(row.get("run_name") or "")
        if not run_name:
            continue
        pattern = canonical_missing_pattern_name(str(row.get("pattern") or ""))
        value = _float(row.get("top1"))
        if _isnum(value):
            grouped[run_name][pattern].append(value)
        row_meta.setdefault(run_name, row)

    out: list[dict[str, Any]] = []
    for run_name, patterns in grouped.items():
        meta = manifest.get(run_name, {})
        values = {pattern: _mean(items) for pattern, items in patterns.items() if items}
        if not _isnum(values.get("avg_missing")):
            candidates = [
                value
                for pattern, value in values.items()
                if pattern not in {"full", "avg_missing"} and _isnum(value)
            ]
            values["avg_missing"] = _mean(candidates)
        row = {
            "run_name": run_name,
            "method": _method_name(run_name, meta),
            "group": meta.get("group", row_meta.get(run_name, {}).get("group", "")),
            "seed": meta.get("seed", row_meta.get(run_name, {}).get("seed", "")),
        }
        for metric in BASE_METRICS:
            row[metric] = values.get(metric, float("nan"))
        out.append(row)
    return sorted(out, key=lambda item: (str(item.get("group", "")), str(item.get("run_name", ""))))


def _proto_reference(per_run: list[dict[str, Any]]) -> dict[str, float]:
    candidates = [
        row
        for row in per_run
        if row.get("method") == "proto" or str(row.get("run_name", "")).startswith("main_v3_strong_reliability_proto")
    ]
    if not candidates:
        return dict(PROTO_REFERENCE)
    proto = dict(PROTO_REFERENCE)
    for metric in BASE_METRICS:
        values = [row.get(metric) for row in candidates if _isnum(row.get(metric))]
        if values:
            proto[metric] = mean(float(value) for value in values)
    proto["balanced"] = _balanced(proto, proto)
    return proto


def _apply_deltas(rows: list[dict[str, Any]], proto: dict[str, float]) -> None:
    for row in rows:
        row["balanced"] = _balanced(row, proto)
        for metric in METRICS:
            value = _float(row.get(metric))
            base = _float(proto.get(metric))
            row[DELTA_COLUMNS[metric]] = value - base if _isnum(value) and _isnum(base) else float("nan")


def _method_rows(per_run: list[dict[str, Any]], proto: dict[str, float]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_run:
        grouped[str(row["method"])].append(row)
    out: list[dict[str, Any]] = []
    for method, rows in grouped.items():
        item: dict[str, Any] = {"method": method, "group": rows[0].get("group", ""), "n": len(rows)}
        for metric in METRICS:
            values = [row.get(metric) for row in rows if _isnum(row.get(metric))]
            item[f"{metric}_mean"] = mean(float(value) for value in values) if values else float("nan")
            item[f"{metric}_std"] = stdev(float(value) for value in values) if len(values) > 1 else 0.0 if values else float("nan")
            base = _float(proto.get(metric))
            value = _float(item[f"{metric}_mean"])
            item[f"{DELTA_COLUMNS[metric]}_mean"] = value - base if _isnum(value) and _isnum(base) else float("nan")
        out.append(item)
    return sorted(out, key=lambda item: _zero_nan(item.get("balanced_mean")), reverse=True)


def _filtered_rows(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in method_rows:
        unmet = _unmet_conditions(row)
        rows.append(
            {
                "method": row["method"],
                "group": row.get("group", ""),
                "n": row.get("n", ""),
                "full": row.get("full_mean"),
                "avg_missing": row.get("avg_missing_mean"),
                "radar_only": row.get("radar_only_mean"),
                "lidar_only": row.get("lidar_only_mean"),
                "balanced": row.get("balanced_mean"),
                "passed": "" if unmet else "yes",
                "unmet_conditions": ";".join(unmet),
            }
        )
    passed = [row for row in rows if row["passed"] == "yes"]
    if passed:
        return sorted(passed, key=lambda item: _zero_nan(item.get("balanced")), reverse=True)
    return sorted(
        rows,
        key=lambda item: (len(str(item.get("unmet_conditions", "")).split(";")), -_zero_nan(item.get("balanced"))),
    )[:10]


def _unmet_conditions(row: dict[str, Any]) -> list[str]:
    unmet = []
    for metric, threshold in FILTER_THRESHOLDS.items():
        value = _float(row.get(f"{metric}_mean"))
        if not _isnum(value) or value < threshold:
            unmet.append(f"{metric}>={threshold:g}")
    return unmet


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


def _read_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {row["run_name"]: row for row in _read_csv(path)}


def _reference_rows(proto: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {"reference": "proto", **proto},
        {"reference": "btapa_tau1", **BTAPA_TAU1_REFERENCE},
    ]


def _write_markdown(path: Path, method_rows: list[dict[str, Any]], filtered_rows: list[dict[str, Any]], proto: dict[str, float]) -> None:
    lines = ["# Scene31 Next-Round Summary", ""]
    lines.extend(["## Reference", "", "| reference | full | avg_missing | radar_only | lidar_only | balanced |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in _reference_rows(proto):
        lines.append(
            f"| {row['reference']} | {_fmt(row.get('full'))} | {_fmt(row.get('avg_missing'))} | "
            f"{_fmt(row.get('radar_only'))} | {_fmt(row.get('lidar_only'))} | {_fmt(row.get('balanced'))} |"
        )
    lines.extend(["", "## Method Mean +/- Std", ""])
    columns = ["method", "n", "full", "avg_missing", "radar_only", "lidar_only", "balanced", "Δbalanced"]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in method_rows:
        lines.append(
            "| {method} | {n} | {full} | {avg} | {radar} | {lidar} | {balanced} | {delta} |".format(
                method=row["method"],
                n=row.get("n", ""),
                full=_mean_std(row, "full"),
                avg=_mean_std(row, "avg_missing"),
                radar=_mean_std(row, "radar_only"),
                lidar=_mean_std(row, "lidar_only"),
                balanced=_mean_std(row, "balanced"),
                delta=_fmt(row.get("Δbalanced_mean")),
            )
        )
    lines.extend(["", "## Filtered", ""])
    if filtered_rows and all(row.get("passed") != "yes" for row in filtered_rows):
        lines.append("No method met all thresholds; showing closest top 10.")
        lines.append("")
    lines.append("| method | full | avg_missing | radar_only | lidar_only | balanced | unmet_conditions |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in filtered_rows:
        lines.append(
            f"| {row['method']} | {_fmt(row.get('full'))} | {_fmt(row.get('avg_missing'))} | "
            f"{_fmt(row.get('radar_only'))} | {_fmt(row.get('lidar_only'))} | {_fmt(row.get('balanced'))} | "
            f"{row.get('unmet_conditions', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _method_name(run_name: str, meta: dict[str, str]) -> str:
    if meta.get("group") == "baseline" and meta.get("method_tags"):
        return str(meta["method_tags"])
    return re.sub(r"_seed\d+$", "", run_name)


def _per_run_fields() -> list[str]:
    return ["run_name", "method", "group", "seed", *METRICS, *[DELTA_COLUMNS[metric] for metric in METRICS]]


def _method_fields() -> list[str]:
    fields = ["method", "group", "n"]
    for metric in METRICS:
        fields.extend([f"{metric}_mean", f"{metric}_std", f"{DELTA_COLUMNS[metric]}_mean"])
    return fields


def _filtered_fields() -> list[str]:
    return ["method", "group", "n", "full", "avg_missing", "radar_only", "lidar_only", "balanced", "passed", "unmet_conditions"]


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


if __name__ == "__main__":
    raise SystemExit(main())
