#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from kd_sensing.eval.missing_patterns import canonical_missing_pattern_name


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
    parser.add_argument("--candidate", default="main_v3_strong_reliability_btapa_tau1")
    args = parser.parse_args(argv)

    root = Path(args.root)
    rows = _collect(root / "eval", args.baseline, args.runs)
    delta_rows = _delta_vs_baseline(rows, args.baseline)
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "btapa_comparison.csv", rows)
    _write_csv(out_dir / "btapa_delta_vs_v3.csv", delta_rows)
    _write_markdown(out_dir / "btapa_comparison.md", rows, delta_rows, args.baseline, args.candidate)
    _print_conclusions(rows, args.baseline, args.candidate)
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
    return canonical_missing_pattern_name(value)


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


def _write_markdown(path: Path, rows: list[dict[str, str]], delta_rows: list[dict[str, str]], baseline: str, candidate: str) -> None:
    lines = ["# BTAPA vs V3", "", f"Baseline: `{baseline}`", ""]
    if candidate:
        lines.extend([f"Candidate main: `{candidate}`", ""])
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
    lines.extend(["", "## BTAPA Series Observation", ""])
    lines.extend(_series_observation(rows, baseline, candidate))
    lines.extend(["", "## Paper-ready Observation", "", _paper_observation(rows, baseline, candidate), ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_conclusions(rows: list[dict[str, str]], baseline: str, candidate: str) -> None:
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
    print(f"candidate main: {candidate}")
    for line in _series_observation(rows, baseline, candidate):
        print(line.removeprefix("- "))
    print(f"paper-ready observation: {_paper_observation(rows, baseline, candidate)}")


def _series_observation(rows: list[dict[str, str]], baseline: str, candidate: str) -> list[str]:
    best_avg = _best([row for row in rows if row["role"] == "btapa"], "avg_missing", "top1")
    tau1_best = best_avg is not None and best_avg.get("run") == candidate
    tau4_delta = _delta_value(rows, "main_v3_strong_reliability_btapa_tau4", candidate, "avg_missing", "top1")
    adba_delta = _delta_value(rows, "main_v3_strong_reliability_btapa_adba", "main_v3_strong_reliability_btapa", "avg_missing", "adba")
    fusion_delta = _delta_value(rows, "main_v3_strong_reliability_btapa", "main_v3_strong_reliability_btapa_fusiononly", "missing_gps", "top1")
    modw1_delta = _delta_value(rows, "main_v3_strong_reliability_btapa_modw1", candidate, "avg_missing", "top1")
    candidate_delta = _delta_value(rows, candidate, baseline, "avg_missing", "top1")
    return [
        f"- tau1 是否最佳: {_yes_no_bool(tau1_best)}; candidate vs baseline avg_missing top1 delta={_format(candidate_delta)}",
        f"- tau4 是否过平滑: {_yes_no_bool(_isnum(tau4_delta) and tau4_delta < 0)}; tau4 minus tau1 avg_missing top1={_format(tau4_delta)}",
        f"- ADBA-aware 是否有效: {_yes_no_bool(_isnum(adba_delta) and adba_delta > 0)}; ADBA-aware minus BTAPA avg_missing ADBA={_format(adba_delta)}",
        f"- fusiononly 与 modality alignment 差异: BTAPA minus fusiononly missing_gps top1={_format(fusion_delta)}",
        f"- modw1 是否过强: {_yes_no_bool(_isnum(modw1_delta) and modw1_delta < 0)}; modw1 minus tau1 avg_missing top1={_format(modw1_delta)}",
    ]


def _paper_observation(rows: list[dict[str, str]], baseline: str, candidate: str) -> str:
    full_delta = _delta_value(rows, candidate, baseline, "full", "top1")
    avg_delta = _delta_value(rows, candidate, baseline, "avg_missing", "top1")
    radar_delta = _delta_value(rows, candidate, baseline, "radar_only", "top1")
    if not any(_isnum(value) for value in (full_delta, avg_delta, radar_delta)):
        return "BTAPA tau1 is the candidate main run, but comparable numeric evidence is unavailable in the current CSV inputs."
    parts = []
    if _isnum(full_delta):
        parts.append(f"full-modality Top-1 delta {full_delta:+.4g}")
    if _isnum(avg_delta):
        parts.append(f"avg-missing Top-1 delta {avg_delta:+.4g}")
    if _isnum(radar_delta):
        parts.append(f"radar-only Top-1 delta {radar_delta:+.4g}")
    return (
        "BTAPA with tau_beam=1.0 shows a measured trade-off against the ordinary prototype baseline "
        f"({'; '.join(parts)}). The strongest evidence is in missing-modality and weak-modality slices when "
        "those deltas are positive, suggesting compact beam-neighborhood targets may help weak sensing modalities without changing the main training line."
    )


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
    delta = _delta_value(rows, run, base_run, pattern, metric)
    if not _isnum(delta):
        return f"{label}: unavailable"
    return f"{label}: delta {pattern} {metric}={delta:.8g}"


def _delta_value(rows: list[dict[str, str]], run: str, base_run: str, pattern: str, metric: str) -> float:
    value = _value(rows, run, pattern, metric)
    base = _value(rows, base_run, pattern, metric)
    return value - base if _isnum(value) and _isnum(base) else float("nan")


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


def _yes_no_bool(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    raise SystemExit(main())
