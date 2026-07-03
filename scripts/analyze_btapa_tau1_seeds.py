#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from kd_sensing.eval.missing_patterns import canonical_missing_pattern_name
from kd_sensing.utils.checkpoint_resolver import resolve_checkpoint

PATTERNS = ("full", "avg_missing", "missing_gps", "non_gps_only", "gps_only", "image_only", "radar_only", "lidar_only")
METRICS = ("top1", "top3", "top5", "adba", "mae", "loss")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = {
        "btapa_tau1": list(args.runs),
        "proto": list(args.baseline_runs or []),
        "old_v3": list(args.old_v3_runs or []),
    }
    seed_rows: list[dict[str, str]] = []
    missing_runs: list[str] = []
    for method, runs in groups.items():
        for run in runs:
            checkpoint = resolve_checkpoint(root, run, "best_val_top1")
            source = _read_eval_rows(root / "eval", run)
            if not source:
                print(f"[WARN] missing metrics for run: {run}")
                missing_runs.append(run)
                continue
            by_pattern = _rows_by_pattern(source)
            for pattern in PATTERNS:
                row = by_pattern.get(pattern, {})
                for metric in METRICS:
                    seed_rows.append(
                        {
                            "method": method,
                            "run_name": run,
                            "checkpoint_path": str(checkpoint.path) if checkpoint.path else "",
                            "checkpoint_epoch": str(checkpoint.epoch or ""),
                            "pattern": pattern,
                            "metric": metric,
                            "value": _metric(row, metric),
                        }
                    )

    mean_rows = _mean_std_rows(seed_rows)
    delta_rows = _delta_vs_proto(mean_rows)
    _write_csv(out_dir / "btapa_tau1_seed_metrics.csv", seed_rows)
    _write_csv(out_dir / "btapa_tau1_mean_std.csv", mean_rows)
    _write_csv(out_dir / "btapa_tau1_delta_vs_proto_mean.csv", delta_rows)
    _write_markdown(out_dir / "btapa_tau1_mean_std.md", mean_rows, delta_rows, missing_runs)
    _print_conclusions(mean_rows, delta_rows)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize BTAPA tau1 seeds as mean +/- std.")
    parser.add_argument("--root", default="outputs/scene31")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[
            "main_v3_strong_reliability_btapa_tau1",
            "main_v3_strong_reliability_btapa_tau1_seed2",
            "main_v3_strong_reliability_btapa_tau1_seed3",
        ],
    )
    parser.add_argument("--baseline_runs", "--baseline-runs", nargs="*", default=[])
    parser.add_argument("--old_v3_runs", "--old-v3-runs", nargs="*", default=[])
    parser.add_argument("--out_dir", "--out-dir", default="outputs/scene31/analysis/btapa_tau1_seeds")
    return parser


def _read_eval_rows(eval_dir: Path, run: str) -> list[dict[str, Any]]:
    csv_path = eval_dir / f"{run}_missing_patterns.csv"
    json_path = eval_dir / f"{run}_missing_patterns.json"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    return []


def _rows_by_pattern(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_pattern = {canonical_missing_pattern_name(row.get("pattern", "")): row for row in rows}
    if "avg_missing" not in by_pattern:
        by_pattern["avg_missing"] = _average_missing(by_pattern)
    return by_pattern


def _average_missing(by_pattern: dict[str, dict[str, Any]]) -> dict[str, str]:
    rows = [
        row
        for pattern, row in by_pattern.items()
        if pattern != "full" and (pattern.startswith("missing_") or pattern.endswith("_only") or pattern == "non_gps_only")
    ]
    out: dict[str, str] = {"pattern": "avg_missing"}
    for metric in METRICS:
        values = [_float(row.get(metric)) for row in rows]
        values = [value for value in values if value == value]
        out[metric] = "" if not values else f"{mean(values):.8g}"
    return out


def _mean_std_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        value = _float(row.get("value"))
        if value == value:
            grouped.setdefault((row["method"], row["pattern"], row["metric"]), []).append(value)
    out: list[dict[str, str]] = []
    for key in sorted(grouped):
        values = grouped[key]
        method, pattern, metric = key
        out.append(
            {
                "method": method,
                "pattern": pattern,
                "metric": metric,
                "mean": f"{mean(values):.8g}",
                "std": f"{(stdev(values) if len(values) > 1 else 0.0):.8g}",
                "n": str(len(values)),
            }
        )
    return out


def _delta_vs_proto(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    index = {(row["method"], row["pattern"], row["metric"]): row for row in rows}
    out: list[dict[str, str]] = []
    for pattern in PATTERNS:
        for metric in METRICS:
            btapa = index.get(("btapa_tau1", pattern, metric), {})
            proto = index.get(("proto", pattern, metric), {})
            btapa_mean = _float(btapa.get("mean"))
            proto_mean = _float(proto.get("mean"))
            out.append(
                {
                    "pattern": pattern,
                    "metric": metric,
                    "btapa_mean": btapa.get("mean", ""),
                    "btapa_std": btapa.get("std", ""),
                    "proto_mean": proto.get("mean", ""),
                    "proto_std": proto.get("std", ""),
                    "delta_mean": _fmt(btapa_mean - proto_mean) if btapa_mean == btapa_mean and proto_mean == proto_mean else "",
                }
            )
    return out


def _write_markdown(path: Path, rows: list[dict[str, str]], delta_rows: list[dict[str, str]], missing_runs: list[str]) -> None:
    lines = ["# BTAPA tau1 Seed Mean +/- Std", ""]
    if missing_runs:
        lines.extend(["## Missing Runs", "", *[f"- {run}" for run in missing_runs], ""])
    lines.extend(["## Mean Std", "", "| method | pattern | metric | mean | std | n |", "| --- | --- | --- | ---: | ---: | ---: |"])
    for row in rows:
        if row["pattern"] in PATTERNS and row["metric"] in {"top1", "adba", "loss"}:
            lines.append(f"| {row['method']} | {row['pattern']} | {row['metric']} | {row['mean']} | {row['std']} | {row['n']} |")
    lines.extend(["", "## Delta vs Proto", "", "| pattern | metric | btapa_mean | proto_mean | delta_mean |", "| --- | --- | ---: | ---: | ---: |"])
    for row in delta_rows:
        if row["metric"] in {"top1", "adba"}:
            lines.append(f"| {row['pattern']} | {row['metric']} | {row['btapa_mean']} | {row['proto_mean']} | {row['delta_mean']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_conclusions(rows: list[dict[str, str]], delta_rows: list[dict[str, str]]) -> None:
    btapa_avg = _lookup(rows, "btapa_tau1", "avg_missing", "top1")
    proto_avg = _lookup(rows, "proto", "avg_missing", "top1")
    delta_avg = _delta(delta_rows, "avg_missing", "top1")
    radar = _lookup(rows, "btapa_tau1", "radar_only", "top1")
    print(f"BTAPA tau1 avg_missing top1 mean±std: {_mean_std_text(btapa_avg)}")
    print(f"proto baseline avg_missing top1 mean±std: {_mean_std_text(proto_avg)}")
    print(f"delta mean: {delta_avg.get('delta_mean', '')}")
    print(f"BTAPA tau1 radar_only top1 mean±std: {_mean_std_text(radar)}")
    print(f"BTAPA tau1 mean exceeds proto mean: {_yes_no(_float(btapa_avg.get('mean')), _float(proto_avg.get('mean')))}")
    btapa_std = _float(btapa_avg.get("std"))
    delta_value = abs(_float(delta_avg.get("delta_mean")))
    cautious = delta_value == delta_value and btapa_std == btapa_std and delta_value < btapa_std
    print(f"difference smaller than BTAPA std; use cautious wording: {cautious}")


def _lookup(rows: list[dict[str, str]], method: str, pattern: str, metric: str) -> dict[str, str]:
    return next((row for row in rows if row["method"] == method and row["pattern"] == pattern and row["metric"] == metric), {})


def _delta(rows: list[dict[str, str]], pattern: str, metric: str) -> dict[str, str]:
    return next((row for row in rows if row["pattern"] == pattern and row["metric"] == metric), {})


def _mean_std_text(row: dict[str, str]) -> str:
    if not row:
        return "unavailable"
    return f"{row.get('mean', '')}±{row.get('std', '')} (n={row.get('n', '')})"


def _yes_no(value: float, baseline: float) -> str:
    if value != value or baseline != baseline:
        return "unavailable"
    return "yes" if value > baseline else "no"


def _metric(row: dict[str, Any], key: str) -> str:
    aliases = {"count": ("count", "sample_count", "num_samples")}
    for name in aliases.get(key, (key,)):
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value: float) -> str:
    return f"{value:.8g}" if value == value else ""


if __name__ == "__main__":
    raise SystemExit(main())
