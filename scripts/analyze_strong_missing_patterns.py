#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from kd_sensing.eval.missing_patterns import canonical_missing_pattern_name


EXPERIMENTS = {
    "V0": "strong_amber_style_mask_baseline_fullrun",
    "V1": "strong_weighted_sum_mask",
    "V2": "strong_weighted_sum_reliability",
    "V3": "strong_weighted_sum_reliability_beam_proto",
    "V4": "strong_weighted_sum_reliability_beam_proto_kd",
    "V5": "strong_no_jepa_rbma_proto_kd_fullrun",
}
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
METRICS = ("top1", "top3", "top5", "adba", "loss", "mae", "count")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare strong V0-V5 missing-pattern metrics.")
    parser.add_argument("--eval_dir", default="outputs/scene31/eval")
    parser.add_argument("--out_dir", default="outputs/scene31/analysis")
    args = parser.parse_args(argv)

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    rows = _collect(eval_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "strong_v0_v5_missing_patterns.csv", rows)
    _write_markdown(out_dir / "strong_v0_v5_missing_patterns.md", rows)
    _write_csv(out_dir / "strong_v0_v5_delta_vs_v3.csv", _delta_vs_v3(rows))
    _print_conclusions(rows)
    return 0


def _collect(eval_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for version, exp_name in EXPERIMENTS.items():
        source_rows = _read_eval_rows(eval_dir, exp_name)
        by_pattern = {_canonical_pattern(row.get("pattern", "")): row for row in source_rows}
        for pattern in PATTERNS:
            row = by_pattern.get(pattern, {})
            rows.append(
                {
                    "version": version,
                    "exp_name": exp_name,
                    "pattern": pattern,
                    **{metric: _metric(row, metric) for metric in METRICS},
                }
            )
    return rows


def _read_eval_rows(eval_dir: Path, exp_name: str) -> list[dict]:
    csv_path = eval_dir / f"{exp_name}_missing_patterns.csv"
    json_path = eval_dir / f"{exp_name}_missing_patterns.json"
    if csv_path.exists():
        return _read_csv(csv_path)
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    return []


def _metric(row: dict, key: str) -> str:
    aliases = {"count": ("count", "sample_count", "num_samples")}
    for name in aliases.get(key, (key,)):
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def _canonical_pattern(value: str) -> str:
    return canonical_missing_pattern_name(value)


def _delta_vs_v3(rows: list[dict]) -> list[dict]:
    index = {(row["version"], row["pattern"], metric): _float(row.get(metric)) for row in rows for metric in METRICS[:-1]}
    out = []
    for row in rows:
        if row["version"] == "V3":
            continue
        for metric in METRICS[:-1]:
            value = _float(row.get(metric))
            base = index.get(("V3", row["pattern"], metric), float("nan"))
            out.append(
                {
                    "exp_name": row["exp_name"],
                    "pattern": row["pattern"],
                    "metric": metric,
                    "value": row.get(metric, ""),
                    "v3_value": "" if base != base else f"{base:.8g}",
                    "delta": "" if value != value or base != base else f"{value - base:.8g}",
                }
            )
    return out


def _write_markdown(path: Path, rows: list[dict]) -> None:
    lines = ["# Strong V0-V5 Missing Pattern Comparison", ""]
    for pattern in ("full", "avg_missing", "gps_only", "image_only", "radar_only", "lidar_only"):
        subset = [row for row in rows if row["pattern"] == pattern]
        lines.extend([f"## {pattern}", "", "| version | exp_name | top1 | top3 | top5 | adba | loss | mae | count |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        lines.extend(
            f"| {row['version']} | {row['exp_name']} | {row['top1']} | {row['top3']} | {row['top5']} | {row['adba']} | {row['loss']} | {row['mae']} | {row['count']} |"
            for row in subset
        )
        lines.append("")
    lines.extend(["## Key Deltas", "", "| comparison | pattern | top1_delta | adba_delta |", "| --- | --- | ---: | ---: |"])
    lookup = {(row["version"], row["pattern"]): row for row in rows}
    for other in ("V4", "V5", "V0"):
        for pattern in ("full", "avg_missing"):
            base = lookup.get(("V3", pattern), {})
            row = lookup.get((other, pattern), {})
            lines.append(f"| V3 vs {other} | {pattern} | {_delta(row, base, 'top1')} | {_delta(row, base, 'adba')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_conclusions(rows: list[dict]) -> None:
    for label, pattern, metric in (
        ("best full top1", "full", "top1"),
        ("best avg_missing top1", "avg_missing", "top1"),
        ("best avg_missing ADBA", "avg_missing", "adba"),
    ):
        best = _best(rows, pattern, metric)
        print(f"{label}: {best['version']} {best['exp_name']} {best[metric]}")
    singles = [row for row in rows if row["pattern"] in {"gps_only", "image_only", "radar_only", "lidar_only"}]
    worst = min((row for row in singles if _float(row.get("top1")) == _float(row.get("top1"))), key=lambda r: _float(r["top1"]))
    print(f"worst single modality: {worst['version']} {worst['pattern']} top1={worst['top1']}")
    print(f"whether V3 is best by top1: {_best(rows, 'full', 'top1')['version'] == 'V3'}")
    print(f"whether V4 is best by ADBA: {_best(rows, 'avg_missing', 'adba')['version'] == 'V4'}")


def _best(rows: list[dict], pattern: str, metric: str) -> dict:
    valid = [row for row in rows if row["pattern"] == pattern and _float(row.get(metric)) == _float(row.get(metric))]
    return max(valid, key=lambda row: _float(row[metric]))


def _delta(row: dict, base: dict, metric: str) -> str:
    value = _float(row.get(metric))
    base_value = _float(base.get(metric))
    return "" if value != value or base_value != base_value else f"{value - base_value:.8g}"


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
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


if __name__ == "__main__":
    raise SystemExit(main())
