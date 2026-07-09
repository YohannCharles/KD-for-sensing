#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_h5_p1_temporal_matrix_v1 as h5


DEFAULT_METHODS = (
    "s1_temporalagg_modality_router",
    "s2_pertime_modality_router",
    "s3_two_level_router",
    "s4_global_modality_time_router",
    "amber_full",
    "rmbp_mm",
)
MATRIX_FILES = h5.MATRIX_FILES
MATRIX_COLUMNS = h5.MATRIX_COLUMNS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize S1-S4 temporal router matrix v1 outputs.")
    parser.add_argument("--eval_dir", "--eval-dir", default="outputs/temporal_router_s1_s4_v1/eval_matrix")
    parser.add_argument("--output_dir", "--output-dir", default="outputs/temporal_router_s1_s4_v1/final_summary")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    args = parser.parse_args(argv)
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    summary_rows: list[dict[str, Any]] = []
    all_markdown = {metric: [] for metric in MATRIX_FILES}
    pattern_rows: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []
    for method in methods:
        matrices = {}
        for metric, filename in MATRIX_FILES.items():
            rows = h5._aggregate_method(eval_dir / method, filename)
            matrices[metric] = rows
            h5._write_csv(out_dir / f"{method}_{metric}_matrix.csv", rows, MATRIX_COLUMNS)
            all_markdown[metric].append(h5._matrix_markdown(method, rows))
        summary_rows.append(_summary_row(method, matrices))
        for seed_dir in sorted((eval_dir / method).glob("seed*")):
            for row in h5._read_csv(seed_dir / "pattern_metrics.csv"):
                row["method"] = method
                row["seed"] = seed_dir.name.removeprefix("seed")
                pattern_rows.append(row)
            for row in h5._read_csv(seed_dir / "router_diagnostics.csv"):
                row["method"] = method
                row["seed"] = seed_dir.name.removeprefix("seed")
                router_rows.append(row)
    h5._write_csv(out_dir / "summary.csv", summary_rows, h5._columns(summary_rows))
    h5._write_csv(out_dir / "pattern_metrics.csv", pattern_rows, h5._columns(pattern_rows))
    h5._write_csv(out_dir / "router_diagnostics.csv", router_rows, h5._columns(router_rows))
    for metric, chunks in all_markdown.items():
        (out_dir / f"all_methods_{metric}_matrices.md").write_text("\n".join(chunks) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(_summary_markdown(summary_rows, all_markdown), encoding="utf-8")
    print(f"wrote {out_dir / 'summary.csv'}")
    print(f"wrote {out_dir / 'summary.md'}")
    return 0


def _summary_row(method: str, matrices: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    row = h5._summary_row(method, matrices)
    row["within3_mean_all_cells"] = row.pop("within3_mean", "")
    row["mae_mean_all_cells"] = row.pop("mae_mean", "")
    row["mean_top1_temporal_20_80"] = row.pop("mean_top1_rate20_80", "")
    return row


def _summary_markdown(summary_rows: list[dict[str, Any]], all_markdown: dict[str, list[str]]) -> str:
    lines = ["# Temporal Router S1-S4 v1 Summary", ""]
    lines.append("## 8.1 六个 Top1 矩阵")
    lines.extend(all_markdown["top1"])
    lines.append("## 8.2 六个 Within@3 矩阵")
    lines.extend(all_markdown["within3"])
    lines.append("## 8.3 六个 MAE 矩阵")
    lines.extend(all_markdown["mae"])
    lines.append("## 8.4 方法对比总表")
    lines.append(h5._table(summary_rows, h5._columns(summary_rows)))
    lines.append("## 8.5 路由诊断")
    lines.append("详见 `router_diagnostics.csv`；S1-S4 输出 modality/temporal/global gate 统计，baseline 行可为空。")
    lines.append("## 8.6 自动分析")
    lines.extend(_analysis(summary_rows))
    return "\n".join(lines) + "\n"


def _analysis(rows: list[dict[str, Any]]) -> list[str]:
    best_severe = h5._best(rows, "mean_top1_severe_cells")
    best_drop3 = h5._best(rows, "top1_drop3_80")
    return [
        f"- S1 是否在 temporal missing 下不足: 对比 S1 的 `mean_top1_temporal_20_80` 与其它 S 方法。",
        "- S2 是否优于 S1: 若 S2 的 temporal 20-80 与 severe cells 更高，则 per-time router 有收益。",
        "- S3 是否是最均衡方案: 优先看 `mean_top1_all_cells`、`mean_top1_severe_cells` 和 `mae_mean_all_cells` 的共同排名。",
        "- S4 是否超过 S3: 若 S4 只在个别 cell 提升而均值/MAE 不稳，应视为 AMBER-like 复杂对照。",
        "- AMBER Full 和 RMBP-MM 的差距: 查看 summary.csv 中二者相对 S1-S4 的均值排名。",
        f"- severe cells 最好: {best_severe or '暂无数据'}。",
        f"- drop3+80% 最好: {best_drop3 or '暂无数据'}。",
        "- 是否需要把 S3 作为主方法: 若 S3 在 severe/drop3+80% 接近最好且均值稳定，建议作为主候选。",
        "- 是否需要进一步 time-aware router 消融: 若 S3/S4 明显优于 S1/S2，下一轮应做 time-aware router 消融。",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
