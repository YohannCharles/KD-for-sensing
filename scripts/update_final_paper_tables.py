#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Any

try:
    import scripts.export_scenes31_34_main_paper_tables as base_tables
    from scripts.scene31_34_final_analysis_common import read_csv, write_md_table
except ModuleNotFoundError:
    import export_scenes31_34_main_paper_tables as base_tables
    from scene31_34_final_analysis_common import read_csv, write_md_table


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paper = Path(args.paper_table_root)
    paper.mkdir(parents=True, exist_ok=True)

    base_tables.main(
        [
            "--summary-root",
            args.summary_root,
            "--fig-root",
            "outputs/scenes31_34_main_lmdb/figures",
            "--profile-root",
            args.profile_root,
            "--out",
            args.paper_table_root,
        ]
    )
    _copy_compute(args.profile_root, paper)
    _write_significance(Path(args.statistics_root), paper)
    _write_pattern(Path(args.pattern_root), paper)
    _write_sampling(Path(args.sampling_root), paper)
    _write_notes(paper / "scenes31_34_final_paper_notes.txt", args)
    print(f"Updated final Scene31-34 paper tables in {paper}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update final Scene31-34 paper tables from analysis artifacts.")
    parser.add_argument("--summary-root", default="outputs/scenes31_34_main_lmdb/summary")
    parser.add_argument("--statistics-root", default="outputs/scenes31_34_main_lmdb/statistics")
    parser.add_argument("--pattern-root", default="outputs/scenes31_34_main_lmdb/pattern_analysis")
    parser.add_argument("--profile-root", default="outputs/scenes31_34_main_lmdb/profile")
    parser.add_argument("--cdf-root", default="outputs/scenes31_34_main_lmdb/error_cdf")
    parser.add_argument("--sampling-root", default="outputs/scenes31_34_main_lmdb/sampling_analysis")
    parser.add_argument("--paper-table-root", default="outputs/paper_tables/scenes31_34_main")
    return parser


def _copy_compute(profile_root: str, paper: Path) -> None:
    for name in ("table_compute_cost.csv", "table_compute_cost.md"):
        src = Path(profile_root) / name
        if src.exists():
            (paper / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _write_significance(stat_root: Path, paper: Path) -> None:
    src = stat_root / "significance_summary.md"
    if src.exists():
        (paper / "table_significance_tests.md").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        return
    rows = read_csv(stat_root / "significance_summary.csv")
    visible = [
        {
            "Comparison": row.get("comparison", ""),
            "Metric": row.get("metric", ""),
            "Seed delta": _delta(row, "seed"),
            "Bootstrap mean delta": _delta(row, "bootstrap"),
            "Bootstrap CI": _ci(row),
            "P(delta>0)": row.get("prob_delta_positive", ""),
            "Conclusion": row.get("conclusion", ""),
            "Notes": row.get("notes", ""),
        }
        for row in rows
    ]
    write_md_table(paper / "table_significance_tests.md", visible, ["Comparison", "Metric", "Seed delta", "Bootstrap mean delta", "Bootstrap CI", "P(delta>0)", "Conclusion", "Notes"])


def _write_pattern(pattern_root: Path, paper: Path) -> None:
    rows = read_csv(pattern_root / "pattern_win_count_summary.csv")
    visible = [
        {
            "Method": row.get("method", ""),
            "Metric": row.get("metric", ""),
            "Best patterns": row.get("num_patterns_best", ""),
            "Beats Bernoulli": row.get("num_patterns_beats_bernoulli", ""),
            "Beats classifier subset": row.get("num_patterns_beats_classifier_subset", ""),
            "Mean rank": row.get("mean_rank", ""),
        }
        for row in rows
    ]
    write_md_table(paper / "table_pattern_win_counts.md", visible, ["Method", "Metric", "Best patterns", "Beats Bernoulli", "Beats classifier subset", "Mean rank"])


def _write_sampling(sampling_root: Path, paper: Path) -> None:
    rows = read_csv(sampling_root / "sampling_distribution_by_missing_count.csv")
    visible = [
        {
            "Method": row.get("method", ""),
            "Source": row.get("distribution_source", ""),
            "Missing count": row.get("missing_count", ""),
            "Probability": row.get("probability", ""),
        }
        for row in rows
    ]
    write_md_table(paper / "table_sampling_distribution.md", visible, ["Method", "Source", "Missing count", "Probability"])


def _write_notes(path: Path, args: argparse.Namespace) -> None:
    lines = [
        "Final trusted method: prototype + random non-empty subset exposure.",
        "Do not claim prototype alone is sufficient; proto natural and classifier natural are close.",
        "Random subset exposure is the primary driver.",
        "Prototype head adds extra gain under random subset exposure.",
        "Random subset exposure outperforms Bernoulli randomdrop on Avg-Missing, Miss3, MAE, and degradation curve.",
        "AMR/AMBER-lite are maskfix-clean external baselines but do not challenge the final method.",
        "Random subset exposure has no extra inference-time parameters; latency is measured in table_compute_cost.",
        "No further model-search experiments are recommended.",
        "",
        f"Summary root: {args.summary_root}",
        f"Statistics root: {args.statistics_root}",
        f"Pattern analysis root: {args.pattern_root}",
        f"Profile root: {args.profile_root}",
        f"Error CDF root: {args.cdf_root}",
        f"Sampling root: {args.sampling_root}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _delta(row: dict[str, str], kind: str) -> str:
    metric = row.get("metric", "")
    if metric in {"avg_missing_MAE", "mae_at_75"}:
        key = "seed_mean_delta" if kind == "seed" else "bootstrap_mean_delta"
        return row.get(key, "")
    key = "seed_mean_delta_pp" if kind == "seed" else "bootstrap_mean_delta_pp"
    value = row.get(key, "")
    return f"{value} pp" if value else ""


def _ci(row: dict[str, str]) -> str:
    metric = row.get("metric", "")
    if metric in {"avg_missing_MAE", "mae_at_75"}:
        low = row.get("bootstrap_ci_low", "")
        high = row.get("bootstrap_ci_high", "")
        return f"[{low}, {high}]" if low or high else ""
    low = row.get("bootstrap_ci_low_pp", "")
    high = row.get("bootstrap_ci_high_pp", "")
    return f"[{low}, {high}] pp" if low or high else ""


if __name__ == "__main__":
    raise SystemExit(main())
