#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from typing import Any


DEFAULT_METHODS = ("ours_c2_main", "ours_b4_nonrouter_soft_jepa", "ours_e5_low_lr_pcpg", "amber_full", "rmbp_mm")
MATRIX_FILES = {"top1": "top1_matrix.csv", "within3": "within3_matrix.csv", "mae": "mae_matrix.csv"}
MATRIX_COLUMNS = ["missing_rate", "full", "drop1", "drop2", "drop3"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize H5/P1 temporal matrix v1 evaluation outputs.")
    parser.add_argument("--eval_dir", "--eval-dir", default="outputs/h5_p1_temporal_models_v1/eval_matrix")
    parser.add_argument("--output_dir", "--output-dir", default="outputs/h5_p1_temporal_models_v1/final_summary")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    args = parser.parse_args(argv)
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    summary_rows = []
    all_markdown = {metric: [] for metric in MATRIX_FILES}
    pattern_rows = []
    for method in methods:
        matrices = {}
        for metric, filename in MATRIX_FILES.items():
            rows = _aggregate_method(eval_dir / method, filename)
            matrices[metric] = rows
            _write_csv(out_dir / f"{method}_{metric}_matrix.csv", rows, MATRIX_COLUMNS)
            all_markdown[metric].append(_matrix_markdown(method, rows))
        summary_rows.append(_summary_row(method, matrices))
        for seed_dir in sorted((eval_dir / method).glob("seed*")):
            pattern_path = seed_dir / "pattern_metrics.csv"
            for row in _read_csv(pattern_path):
                row["method"] = method
                row["seed"] = seed_dir.name.removeprefix("seed")
                pattern_rows.append(row)
    _write_csv(out_dir / "summary.csv", summary_rows, _columns(summary_rows))
    _write_csv(out_dir / "pattern_metrics.csv", pattern_rows, _columns(pattern_rows))
    for metric, chunks in all_markdown.items():
        (out_dir / f"all_methods_{metric}_matrices.md").write_text("\n".join(chunks) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(_summary_markdown(summary_rows, all_markdown), encoding="utf-8")
    print(f"wrote {out_dir / 'summary.csv'}")
    print(f"wrote {out_dir / 'summary.md'}")
    return 0


def _aggregate_method(method_dir: Path, filename: str) -> list[dict[str, Any]]:
    by_rate: dict[str, dict[str, list[float]]] = {}
    for path in sorted(method_dir.glob(f"seed*/{filename}")):
        for row in _read_csv(path):
            rate = str(row.get("missing_rate", ""))
            by_rate.setdefault(rate, {column: [] for column in MATRIX_COLUMNS if column != "missing_rate"})
            for column in MATRIX_COLUMNS:
                if column == "missing_rate":
                    continue
                value = _float(row.get(column))
                if value is not None:
                    by_rate[rate][column].append(value)
    rows = []
    for rate in sorted(by_rate, key=lambda item: float(item) if item else -1.0):
        out = {"missing_rate": rate}
        for column, values in by_rate[rate].items():
            out[column] = "" if not values else f"{sum(values) / len(values):.6g}"
            out[f"{column}_std"] = "" if len(values) < 2 else f"{_std(values):.6g}"
        rows.append(out)
    return rows


def _summary_row(method: str, matrices: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    top1 = matrices.get("top1", [])
    within3 = matrices.get("within3", [])
    mae = matrices.get("mae", [])
    return {
        "method": method,
        "mean_top1_all_cells": _matrix_mean(top1),
        "mean_top1_rate20_80": _matrix_mean(top1, min_rate=0.2),
        "mean_top1_drop1_3": _matrix_mean(top1, columns=("drop1", "drop2", "drop3")),
        "mean_top1_severe_cells": _matrix_mean(top1, min_rate=0.6, columns=("drop2", "drop3")),
        "top1_full_0": _cell(top1, 0.0, "full"),
        "top1_drop3_80": _cell(top1, 0.8, "drop3"),
        "within3_mean": _matrix_mean(within3),
        "mae_mean": _matrix_mean(mae),
    }


def _summary_markdown(summary_rows: list[dict[str, Any]], all_markdown: dict[str, list[str]]) -> str:
    lines = ["# H5/P1 Temporal Matrix v1 Summary", ""]
    lines.append("## Top1 Matrices")
    lines.extend(all_markdown["top1"])
    lines.append("## Within@3 Matrices")
    lines.extend(all_markdown["within3"])
    lines.append("## MAE Matrices")
    lines.extend(all_markdown["mae"])
    lines.append("## Method Comparison")
    lines.append(_table(summary_rows, _columns(summary_rows)))
    lines.append("## 自动分析")
    lines.extend(_analysis(summary_rows))
    return "\n".join(lines) + "\n"


def _analysis(rows: list[dict[str, Any]]) -> list[str]:
    best_full = _best(rows, "top1_full_0")
    best_high = _best(rows, "mean_top1_rate20_80")
    best_drop3 = _best(rows, "top1_drop3_80")
    c2 = next((row for row in rows if row.get("method") == "ours_c2_main"), None)
    amber = next((row for row in rows if row.get("method") == "amber_full"), None)
    rmbp = next((row for row in rows if row.get("method") == "rmbp_mm"), None)
    return [
        f"- full 0% 最好: {best_full or '暂无数据'}。",
        f"- high temporal missing 下最好: {best_high or '暂无数据'}。",
        f"- drop3 + 80% 最好: {best_drop3 or '暂无数据'}。",
        f"- AMBER Full vs ours_c2_main gap: {_gap(c2, amber, 'mean_top1_all_cells')}。",
        f"- RMBP-MM vs ours_c2_main gap: {_gap(c2, rmbp, 'mean_top1_all_cells')}。",
        "- 时序缺失是否放大模态缺失: 查看 `mean_top1_rate20_80` 与 `mean_top1_drop1_3`，严重 cell 单独看 `mean_top1_severe_cells`。",
        "- C2 是否保持鲁棒: 优先看 `ours_c2_main` 在 severe cells 和 `top1_drop3_80` 的排名。",
        "- 是否需要 time-aware router: 若 masked_mean 下 drop3/high-rate 明显劣化，应把 time-aware router 作为下一轮 change。",
    ]


def _matrix_markdown(method: str, rows: list[dict[str, Any]]) -> str:
    return f"### {method}\n\n" + _table(rows, MATRIX_COLUMNS)


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_暂无数据_\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _matrix_mean(rows: list[dict[str, Any]], min_rate: float | None = None, columns: tuple[str, ...] = ("full", "drop1", "drop2", "drop3")) -> str:
    values = []
    for row in rows:
        rate = _float(row.get("missing_rate"))
        if min_rate is not None and (rate is None or rate < min_rate):
            continue
        for column in columns:
            value = _float(row.get(column))
            if value is not None:
                values.append(value)
    return "" if not values else f"{sum(values) / len(values):.6g}"


def _cell(rows: list[dict[str, Any]], rate: float, column: str) -> str:
    for row in rows:
        value = _float(row.get("missing_rate"))
        if value is not None and abs(value - rate) < 1e-9:
            return str(row.get(column, ""))
    return ""


def _best(rows: list[dict[str, Any]], key: str) -> str:
    scored = [(row.get("method", ""), _float(row.get(key))) for row in rows]
    scored = [(method, value) for method, value in scored if value is not None]
    if not scored:
        return ""
    method, value = max(scored, key=lambda item: item[1])
    return f"{method} ({value:.6g})"


def _gap(c2: dict[str, Any] | None, other: dict[str, Any] | None, key: str) -> str:
    c2_value = _float((c2 or {}).get(key))
    other_value = _float((other or {}).get(key))
    if c2_value is None or other_value is None:
        return "暂无数据"
    return f"{other_value - c2_value:+.6g}"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


if __name__ == "__main__":
    raise SystemExit(main())
