#!/usr/bin/env python3

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


DEFAULT_ROOT = "outputs/pcpg_radar_balance_v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize PCPG radar-balance v1 local outputs.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern_rows = collect_pattern_rows(root)
    gate_rows = collect_gate_rows(root)
    summary_rows = summarize_runs(pattern_rows)
    _write_csv(out_dir / "pattern_metrics.csv", pattern_rows)
    _write_csv(out_dir / "gate_diagnostics.csv", gate_rows)
    _write_csv(out_dir / "summary.csv", summary_rows)
    (out_dir / "summary.md").write_text(render_markdown(summary_rows), encoding="utf-8")
    print(f"Wrote PCPG radar-balance summary to {out_dir}")
    return 0


def collect_pattern_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.csv")):
        if path.name not in {"eval_matrix.csv", "oracle_eval_matrix.csv", "pattern_metrics.csv"}:
            continue
        for row in _read_csv(path):
            run_name = str(row.get("run_name") or _run_name_from_path(path, root))
            rows.append(
                {
                    **row,
                    "run_name": run_name,
                    "experiment": _experiment_from_run(run_name),
                    "source_path": str(path),
                    "oracle_gate": row.get("oracle_gate", "true" if path.name.startswith("oracle") else "false"),
                }
            )
    return rows


def collect_gate_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.csv")):
        if path.name not in {"reliability_weights_epoch.csv", "pcpg_gate_diagnostics.csv", "gate_diagnostics.csv"}:
            continue
        for row in _read_csv(path):
            run_name = _run_name_from_path(path, root)
            rows.append({**row, "run_name": run_name, "experiment": _experiment_from_run(run_name), "source_path": str(path)})
    return rows


def summarize_runs(pattern_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pattern_rows:
        by_run[str(row.get("run_name") or "")].append(row)
    rows = []
    for run_name, items in sorted(by_run.items()):
        values = {str(row.get("pattern") or row.get("pattern_name")): row for row in items}
        rows.append(
            {
                "run_name": run_name,
                "experiment": _experiment_from_run(run_name),
                "full_top1": _float(_metric(values.get("full"), "top1")),
                "avg_missing_top1": _avg_missing_top1(values),
                "worst_pattern_top1": _worst_pattern_top1(values),
                "radar_only_top1": _float(_metric(values.get("radar_only"), "top1")),
                "lidar_only_top1": _float(_metric(values.get("lidar_only"), "top1")),
                "oracle_gate": str(any(str(row.get("oracle_gate")).lower() == "true" for row in items)).lower(),
            }
        )
    return _attach_experiment_stats(rows)


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# PCPG Radar Balance V1 Summary",
        "",
        "| experiment | n | avg_missing_top1 | worst_pattern_top1 | full_top1 | delta_vs_e1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_exp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_exp[str(row.get("experiment"))].append(row)
    baseline = _mean_metric(by_exp.get("e1_tinyvit_valacc_ckpt", []), "avg_missing_top1")
    for experiment, items in sorted(by_exp.items()):
        avg = _mean_metric(items, "avg_missing_top1")
        delta = avg - baseline if _isnum(avg) and _isnum(baseline) else math.nan
        lines.append(
            f"| {experiment} | {len(items)} | {_fmt(avg)} | {_fmt(_mean_metric(items, 'worst_pattern_top1'))} | "
            f"{_fmt(_mean_metric(items, 'full_top1'))} | {_fmt(delta)} |"
        )
    return "\n".join(lines) + "\n"


def _attach_experiment_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_exp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_exp[str(row.get("experiment"))].append(row)
    out = []
    for row in rows:
        peers = by_exp[str(row.get("experiment"))]
        out.append(
            {
                **row,
                "experiment_n": len(peers),
                "experiment_avg_missing_top1_mean": _mean_metric(peers, "avg_missing_top1"),
                "experiment_avg_missing_top1_std": _std_metric(peers, "avg_missing_top1"),
            }
        )
    return out


def _avg_missing_top1(values: dict[str, dict[str, Any]]) -> float:
    direct = _float(_metric(values.get("avg_missing"), "top1"))
    if _isnum(direct):
        return direct
    candidates = [
        _float(_metric(row, "top1"))
        for pattern, row in values.items()
        if pattern not in {"full", "avg_missing"} and (pattern.startswith("missing_") or pattern.endswith("_only"))
    ]
    valid = [value for value in candidates if _isnum(value)]
    return mean(valid) if valid else math.nan


def _worst_pattern_top1(values: dict[str, dict[str, Any]]) -> float:
    candidates = [
        _float(_metric(row, "top1"))
        for pattern, row in values.items()
        if pattern not in {"full", "avg_missing"} and (pattern.startswith("missing_") or pattern.endswith("_only"))
    ]
    valid = [value for value in candidates if _isnum(value)]
    return min(valid) if valid else math.nan


def _run_name_from_path(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    return parts[0] if parts else path.parent.name


def _experiment_from_run(run_name: str) -> str:
    marker = "_seed"
    return run_name.split(marker, 1)[0] if marker in run_name else run_name


def _metric(row: dict[str, Any] | None, key: str) -> Any:
    return None if row is None else row.get(key)


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [_float(row.get(key)) for row in rows]
    values = [value for value in values if _isnum(value)]
    return mean(values) if values else math.nan


def _std_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [_float(row.get(key)) for row in rows]
    values = [value for value in values if _isnum(value)]
    return pstdev(values) if len(values) > 1 else 0.0


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _isnum(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _fmt(value: Any) -> str:
    numeric = _float(value)
    return "n/a" if not _isnum(numeric) else f"{numeric:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
