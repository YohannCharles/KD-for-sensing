#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


DEFAULT_RUNS = (
    "main_v3_strong_reliability_btapa",
    "main_v3_strong_reliability_btapa_tau1",
    "main_v3_strong_reliability_btapa_tau4",
    "main_v3_strong_reliability_btapa_adba",
    "main_v3_strong_reliability_btapa_fusiononly",
    "main_v3_strong_reliability_btapa_modw1",
)
PATTERNS = ("full", "avg_missing", "missing_gps", "non_gps_only", "gps_only", "image_only", "radar_only", "lidar_only")
METRICS = ("top1", "top3", "top5", "adba", "mae", "loss")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare BTAPA Scene31 runs against the V3 prototype baseline.")
    parser.add_argument("--root", default="outputs/scene31")
    parser.add_argument("--baseline", default="main_v3_strong_reliability_proto")
    parser.add_argument("--runs", nargs="*", default=list(DEFAULT_RUNS))
    args = parser.parse_args(argv)

    root = Path(args.root)
    rows = _collect(root / "eval", args.baseline, args.runs)
    delta_rows = _delta_vs_baseline(rows, args.baseline)
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "btapa_comparison.csv", rows)
    _write_csv(out_dir / "btapa_delta_vs_v3.csv", delta_rows)
    _write_markdown(out_dir / "btapa_comparison.md", rows, delta_rows, args.baseline)
    _print_conclusions(rows, args.baseline)
    return 0


def _collect(eval_dir: Path, baseline: str, runs: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run in [baseline, *runs]:
        source_rows = _read_eval_rows(eval_dir, run)
        by_pattern = {_canonical_pattern(row.get("pattern", "")): row for row in source_rows}
        for pattern in PATTERNS:
            row = by_pattern.get(pattern, {})
            rows.append(
                {
                    "run": run,
                    "role": "baseline" if run == baseline else "btapa",
                    "pattern": pattern,
                    **{metric: _metric(row, metric) for metric in METRICS},
                    "count": _metric(row, "count"),
                }
            )
    return rows


def _read_eval_rows(eval_dir: Path, run: str) -> list[dict]:
    csv_path = eval_dir / f"{run}_missing_patterns.csv"
    json_path = eval_dir / f"{run}_missing_patterns.json"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    return []


def _metric(row: dict, key: str) -> str:
    aliases = {"count": ("count", "sample_count", "num_samples")}
    for name in aliases.get(key, (key,)):
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def _canonical_pattern(value: str) -> str:
    text = str(value)
    if text.startswith("only_"):
        return f"{text.removeprefix('only_')}_only"
    return text


def _delta_vs_baseline(rows: list[dict[str, str]], baseline: str) -> list[dict[str, str]]:
    base = {(row["pattern"], metric): _float(row.get(metric)) for row in rows if row["run"] == baseline for metric in METRICS}
    out: list[dict[str, str]] = []
    for row in rows:
        if row["run"] == baseline:
            continue
        for metric in METRICS:
            value = _float(row.get(metric))
            base_value = base.get((row["pattern"], metric), float("nan"))
            out.append(
                {
                    "run": row["run"],
                    "pattern": row["pattern"],
                    "metric": metric,
                    "value": row.get(metric, ""),
                    "v3_value": _format(base_value),
                    "delta": _format(value - base_value) if _isnum(value) and _isnum(base_value) else "",
                }
            )
    return out


def _write_markdown(path: Path, rows: list[dict[str, str]], delta_rows: list[dict[str, str]], baseline: str) -> None:
    lines = ["# BTAPA vs V3", "", f"Baseline: `{baseline}`", ""]
    for pattern in ("full", "avg_missing", "missing_gps", "non_gps_only"):
        subset = [row for row in rows if row["pattern"] == pattern]
        lines.extend(
            [
                f"## {pattern}",
                "",
                "| run | top1 | top3 | top5 | adba | mae | loss | count |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        lines.extend(
            f"| {row['run']} | {row['top1']} | {row['top3']} | {row['top5']} | {row['adba']} | {row['mae']} | {row['loss']} | {row['count']} |"
            for row in subset
        )
        lines.append("")
    lines.extend(["## Delta vs V3", "", "| run | pattern | metric | delta |", "| --- | --- | --- | ---: |"])
    for row in delta_rows:
        if row["pattern"] in {"full", "avg_missing", "missing_gps"} and row["metric"] in {"top1", "adba"}:
            lines.append(f"| {row['run']} | {row['pattern']} | {row['metric']} | {row['delta']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_conclusions(rows: list[dict[str, str]], baseline: str) -> None:
    for label, pattern, metric in (
        ("best full top1 run", "full", "top1"),
        ("best avg_missing top1 run", "avg_missing", "top1"),
        ("best missing_gps top1 run", "missing_gps", "top1"),
        ("best avg_missing ADBA run", "avg_missing", "adba"),
    ):
        print(f"{label}: {_best_label(rows, pattern, metric)}")
    base_full = _value(rows, baseline, "full", "top1")
    best_btapa_full = _best([row for row in rows if row["role"] == "btapa"], "full", "top1")
    print(f"BTAPA exceeds V3 full top1: {_yes_no(best_btapa_full, base_full)}")
    tau_runs = [row for row in rows if row["run"].endswith(("btapa", "tau1", "tau4"))]
    print(f"best tau_beam by avg_missing top1: {_best_label(tau_runs, 'avg_missing', 'top1')}")
    print(_delta_line(rows, "ADBA-aware proto effective", "main_v3_strong_reliability_btapa_adba", "main_v3_strong_reliability_btapa", "avg_missing", "adba"))
    print(_delta_line(rows, "modality proto effective", "main_v3_strong_reliability_btapa", "main_v3_strong_reliability_btapa_fusiononly", "missing_gps", "top1"))


def _best_label(rows: list[dict[str, str]], pattern: str, metric: str) -> str:
    row = _best(rows, pattern, metric)
    return "unavailable" if row is None else f"{row['run']} {metric}={row[metric]}"


def _best(rows: list[dict[str, str]], pattern: str, metric: str) -> dict[str, str] | None:
    valid = [row for row in rows if row["pattern"] == pattern and _isnum(_float(row.get(metric)))]
    return max(valid, key=lambda row: _float(row[metric]), default=None)


def _value(rows: list[dict[str, str]], run: str, pattern: str, metric: str) -> float:
    for row in rows:
        if row["run"] == run and row["pattern"] == pattern:
            return _float(row.get(metric))
    return float("nan")


def _yes_no(row: dict[str, str] | None, base: float) -> str:
    if row is None or not _isnum(base):
        return "unavailable"
    return "yes" if _float(row.get("top1")) > base else "no"


def _delta_line(rows: list[dict[str, str]], label: str, run: str, base_run: str, pattern: str, metric: str) -> str:
    value = _value(rows, run, pattern, metric)
    base = _value(rows, base_run, pattern, metric)
    if not _isnum(value) or not _isnum(base):
        return f"{label}: unavailable"
    return f"{label}: delta {pattern} {metric}={value - base:.8g}"


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _isnum(value: float) -> bool:
    return value == value


def _format(value: float) -> str:
    return f"{value:.8g}" if _isnum(value) else ""


if __name__ == "__main__":
    raise SystemExit(main())
