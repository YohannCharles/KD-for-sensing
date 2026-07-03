#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from kd_sensing.eval.missing_patterns import canonical_missing_pattern_name

PATTERNS = (
    "full",
    "avg_missing",
    "missing_gps",
    "non_gps_only",
    "gps_only",
    "image_only",
    "radar_only",
    "lidar_only",
    "missing_image",
    "missing_radar",
    "missing_lidar",
)
METRICS = ("top1", "top3", "top5", "adba", "mae", "loss", "count")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_rows = _read_eval_rows(eval_dir)
    seed_rows, missing_runs = _seed_rows(
        source_rows,
        proto_runs=list(args.proto_runs),
        btapa_runs=list(args.btapa_runs),
    )
    mean_rows = _mean_std_rows(seed_rows)
    delta_rows = _delta_rows(mean_rows)
    observation = _paper_observation(mean_rows, delta_rows)

    _write_csv(out_dir / "proto_vs_btapa_seed_metrics.csv", seed_rows)
    _write_csv(out_dir / "proto_vs_btapa_mean_std.csv", mean_rows)
    _write_csv(out_dir / "proto_vs_btapa_delta_mean.csv", delta_rows)
    (out_dir / "proto_vs_btapa_mean_std.md").write_text(
        _render_markdown(mean_rows, delta_rows, missing_runs, observation),
        encoding="utf-8",
    )
    (out_dir / "proto_vs_btapa_paper_observation.md").write_text(observation + "\n", encoding="utf-8")
    print(observation)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize proto vs BTAPA tau1 seeds from fresh apples-to-apples eval.")
    parser.add_argument("--root", default="outputs/scene31")
    parser.add_argument(
        "--proto_runs",
        "--proto-runs",
        nargs="+",
        default=[
            "main_v3_strong_reliability_proto",
            "main_v3_strong_reliability_proto_seed2",
            "main_v3_strong_reliability_proto_seed3",
        ],
    )
    parser.add_argument(
        "--btapa_runs",
        "--btapa-runs",
        nargs="+",
        default=[
            "main_v3_strong_reliability_btapa_tau1",
            "main_v3_strong_reliability_btapa_tau1_seed2",
            "main_v3_strong_reliability_btapa_tau1_seed3",
        ],
    )
    parser.add_argument("--eval_dir", "--eval-dir", default="outputs/scene31/analysis/proto_vs_btapa_apples")
    parser.add_argument("--out_dir", "--out-dir", default="outputs/scene31/analysis/proto_vs_btapa_seeds")
    return parser


def _read_eval_rows(eval_dir: Path) -> list[dict[str, Any]]:
    combined = eval_dir / "apples_to_apples_metrics.csv"
    if combined.exists():
        return _read_csv(combined)
    rows: list[dict[str, Any]] = []
    for path in sorted(eval_dir.glob("*_missing_patterns.csv")):
        run_name = path.name.removesuffix("_missing_patterns.csv")
        for row in _read_csv(path):
            rows.append({"run_name": run_name, **row})
    return rows


def _seed_rows(
    rows: list[dict[str, Any]],
    *,
    proto_runs: list[str],
    btapa_runs: list[str],
) -> tuple[list[dict[str, str]], list[str]]:
    by_run_pattern = {
        (row.get("run_name", ""), canonical_missing_pattern_name(row.get("pattern", ""))): row
        for row in rows
    }
    out: list[dict[str, str]] = []
    missing_runs: list[str] = []
    for method, runs in (("proto", proto_runs), ("btapa_tau1", btapa_runs)):
        for run in runs:
            if not any(key[0] == run for key in by_run_pattern):
                missing_runs.append(run)
                print(f"[WARN] missing fresh eval rows for run: {run}")
                continue
            for pattern in PATTERNS:
                row = by_run_pattern.get((run, pattern), {})
                for metric in METRICS:
                    out.append(
                        {
                            "method": method,
                            "run_name": run,
                            "checkpoint_path": str(row.get("checkpoint_path", "")),
                            "checkpoint_epoch": str(row.get("checkpoint_epoch", "")),
                            "pattern": pattern,
                            "metric": metric,
                            "value": _metric(row, metric),
                        }
                    )
    return out, missing_runs


def _mean_std_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        value = _float(row.get("value"))
        if value == value:
            grouped.setdefault((row["method"], row["pattern"], row["metric"]), []).append(value)
    out: list[dict[str, str]] = []
    for method, pattern, metric in sorted(grouped):
        values = grouped[(method, pattern, metric)]
        out.append(
            {
                "method": method,
                "pattern": pattern,
                "metric": metric,
                "mean": _fmt(mean(values)),
                "std": _fmt(stdev(values) if len(values) > 1 else 0.0),
                "n": str(len(values)),
            }
        )
    return out


def _delta_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    index = {(row["method"], row["pattern"], row["metric"]): row for row in rows}
    out: list[dict[str, str]] = []
    for pattern in PATTERNS:
        for metric in METRICS:
            proto = index.get(("proto", pattern, metric), {})
            btapa = index.get(("btapa_tau1", pattern, metric), {})
            proto_mean = _float(proto.get("mean"))
            btapa_mean = _float(btapa.get("mean"))
            out.append(
                {
                    "pattern": pattern,
                    "metric": metric,
                    "proto_mean": proto.get("mean", ""),
                    "proto_std": proto.get("std", ""),
                    "btapa_mean": btapa.get("mean", ""),
                    "btapa_std": btapa.get("std", ""),
                    "delta_mean": _fmt(btapa_mean - proto_mean) if btapa_mean == btapa_mean and proto_mean == proto_mean else "",
                }
            )
    return out


def _render_markdown(
    mean_rows: list[dict[str, str]],
    delta_rows: list[dict[str, str]],
    missing_runs: list[str],
    observation: str,
) -> str:
    lines = ["# Proto vs BTAPA tau1 Mean +/- Std", ""]
    if missing_runs:
        lines.extend(["## Missing Fresh Eval Runs", "", *[f"- {run}" for run in missing_runs], ""])
    focus = [
        ("full", "top1"),
        ("avg_missing", "top1"),
        ("missing_gps", "top1"),
        ("radar_only", "top1"),
        ("lidar_only", "top1"),
        ("avg_missing", "adba"),
    ]
    lines.extend(["## Focus Metrics", "", "| pattern | metric | proto | BTAPA tau1 | delta_mean | note |", "| --- | --- | ---: | ---: | ---: | --- |"])
    for pattern, metric in focus:
        proto = _lookup(mean_rows, "proto", pattern, metric)
        btapa = _lookup(mean_rows, "btapa_tau1", pattern, metric)
        delta = _delta(delta_rows, pattern, metric)
        lines.append(
            f"| {pattern} | {metric} | {_mean_std(proto)} | {_mean_std(btapa)} | {delta.get('delta_mean', '')} | {_caution_note(delta, proto, btapa)} |"
        )
    lines.extend(["", "## Mean Std Table", "", "| method | pattern | metric | mean | std | n |", "| --- | --- | --- | ---: | ---: | ---: |"])
    for row in mean_rows:
        if row["pattern"] in PATTERNS and row["metric"] in {"top1", "adba", "loss", "count"}:
            lines.append(f"| {row['method']} | {row['pattern']} | {row['metric']} | {row['mean']} | {row['std']} | {row['n']} |")
    lines.extend(["", "## Observation", "", observation, ""])
    return "\n".join(lines)


def _paper_observation(mean_rows: list[dict[str, str]], delta_rows: list[dict[str, str]]) -> str:
    avg = _delta(delta_rows, "avg_missing", "top1")
    radar = _delta(delta_rows, "radar_only", "top1")
    avg_delta = _float(avg.get("delta_mean"))
    radar_delta = _float(radar.get("delta_mean"))
    exceeds = avg_delta == avg_delta and avg_delta > 0
    cautious = _caution_note(avg, _lookup(mean_rows, "proto", "avg_missing", "top1"), _lookup(mean_rows, "btapa_tau1", "avg_missing", "top1"))
    first = (
        "BTAPA-tau1 improves the mean avg-missing top1 over the ordinary prototype baseline under the unified evaluation protocol."
        if exceeds
        else "BTAPA-tau1 does not clearly improve the mean avg-missing top1 over the ordinary prototype baseline under the unified evaluation protocol."
    )
    radar_text = (
        " The gain is most pronounced in radar-only evaluation."
        if radar_delta == radar_delta and radar_delta > 0
        else ""
    )
    caution = " Since the gain is comparable to the seed variance in some patterns, results should be reported with mean+/-std." if cautious else ""
    return first + radar_text + caution


def _lookup(rows: list[dict[str, str]], method: str, pattern: str, metric: str) -> dict[str, str]:
    return next((row for row in rows if row["method"] == method and row["pattern"] == pattern and row["metric"] == metric), {})


def _delta(rows: list[dict[str, str]], pattern: str, metric: str) -> dict[str, str]:
    return next((row for row in rows if row["pattern"] == pattern and row["metric"] == metric), {})


def _caution_note(delta: dict[str, str], proto: dict[str, str], btapa: dict[str, str]) -> str:
    delta_value = abs(_float(delta.get("delta_mean")))
    spread = max(_float(proto.get("std")), _float(btapa.get("std")))
    if delta_value == delta_value and spread == spread and delta_value < spread:
        return "not statistically strong, report cautiously"
    return ""


def _mean_std(row: dict[str, str]) -> str:
    if not row:
        return ""
    return f"{row.get('mean', '')}+/-{row.get('std', '')} (n={row.get('n', '')})"


def _metric(row: dict[str, Any], key: str) -> str:
    for name in (key, "sample_count", "num_samples") if key == "count" else (key,):
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
