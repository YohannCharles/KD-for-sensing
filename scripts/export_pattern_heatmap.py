#!/usr/bin/env python3

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scripts.scene31_34_final_analysis_common import (
        BERNOULLI_METHOD,
        CLASSIFIER_SUBSET_METHOD,
        DEFAULT_ANALYSIS_METHODS,
        REFERENCE_METHOD,
        best_external_method,
        load_pattern_metrics,
        method_label,
        method_rows_from_summary,
        roots_from_args,
        write_csv,
    )
except ModuleNotFoundError:
    from scene31_34_final_analysis_common import (
        BERNOULLI_METHOD,
        CLASSIFIER_SUBSET_METHOD,
        DEFAULT_ANALYSIS_METHODS,
        REFERENCE_METHOD,
        best_external_method,
        load_pattern_metrics,
        method_label,
        method_rows_from_summary,
        roots_from_args,
        write_csv,
    )


WIN_FIELDS = [
    "method",
    "metric",
    "num_patterns_total",
    "num_patterns_best",
    "num_patterns_beats_bernoulli",
    "num_patterns_beats_classifier_subset",
    "num_patterns_beats_natural",
    "num_patterns_beats_uniform",
    "mean_rank",
]
PAPER_LABELS = {
    "scenes31_34_proto_natural_es40": "Proto natural",
    "scenes31_34_proto_sampler_uniform_es40": "Proto uniform",
    "scenes31_34_proto_randomdrop_bernoulli_k075_es40": "Proto Bernoulli",
    "scenes31_34_proto_randomdrop_subset_es40": "Proto subset",
    "scenes31_34_classifier_randomdrop_subset_es40": "Cls subset",
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    roots = roots_from_args([args.root], args.old_root, args.classifier_root, args.external_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    external = best_external_method(method_rows_from_summary(Path(args.root) / "summary"), "amber_lite")
    methods = [*DEFAULT_ANALYSIS_METHODS]
    if external:
        methods.append(external)
    data = load_pattern_metrics(roots, set(methods))
    if data.empty:
        raise SystemExit("No pattern_metrics.csv rows found for requested methods.")

    top1 = _matrix(data, methods, "top1")
    mae = _matrix(data, methods, "mae")
    patterns = _ordered_patterns(data)
    top1 = top1.reindex(index=methods, columns=patterns)
    mae = mae.reindex(index=methods, columns=patterns)

    _write_matrix(out / "pattern_metric_matrix_top1.csv", top1)
    _write_matrix(out / "pattern_metric_matrix_mae.csv", mae)
    _write_delta(out / "pattern_delta_vs_bernoulli_top1.csv", top1, BERNOULLI_METHOD, higher_is_better=True)
    _write_delta(out / "pattern_delta_vs_bernoulli_mae.csv", mae, BERNOULLI_METHOD, higher_is_better=False)
    _write_delta(out / "pattern_delta_vs_classifier_subset_top1.csv", top1, CLASSIFIER_SUBSET_METHOD, higher_is_better=True)
    _write_delta(out / "pattern_delta_vs_classifier_subset_mae.csv", mae, CLASSIFIER_SUBSET_METHOD, higher_is_better=False)
    write_csv(out / "pattern_win_count_summary.csv", _win_rows(top1, mae), WIN_FIELDS)

    _plot_heatmap(top1, out / "fig_pattern_heatmap_top1", "Top-1 accuracy by missing pattern", "Top-1 accuracy", percent=True)
    _plot_heatmap(mae, out / "fig_pattern_heatmap_mae", "MAE by missing pattern", "MAE", percent=False, reverse=True)
    _plot_delta(top1, BERNOULLI_METHOD, out / "fig_pattern_delta_vs_bernoulli_top1", "Top-1 delta vs Bernoulli randomdrop", higher_is_better=True, percent=True)
    _plot_delta(top1, BERNOULLI_METHOD, out / "fig_pattern_delta_vs_bernoulli_top1_paper", "Top-1 delta vs Bernoulli randomdrop", higher_is_better=True, percent=True)
    _plot_delta(mae, BERNOULLI_METHOD, out / "fig_pattern_delta_vs_bernoulli_mae_paper", "MAE delta vs Bernoulli randomdrop", higher_is_better=False, percent=False)
    _plot_delta(top1, CLASSIFIER_SUBSET_METHOD, out / "fig_pattern_delta_vs_classifier_subset_top1", "Top-1 delta vs classifier subset", higher_is_better=True, percent=True)
    _plot_grouped_heatmap(top1, out / "fig_pattern_heatmap_top1_grouped_paper")
    print(f"Wrote Scene31-34 pattern analysis to {out}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Scene31-34 pattern-level heatmaps and win counts.")
    parser.add_argument("--root", default="outputs/scenes31_34_main_lmdb")
    parser.add_argument("--classifier-root", action="append", default=[])
    parser.add_argument("--external-root", action="append", default=[])
    parser.add_argument("--old-root", action="append", default=[])
    parser.add_argument("--out", default="outputs/scenes31_34_main_lmdb/pattern_analysis")
    return parser


def _matrix(data: pd.DataFrame, methods: list[str], metric: str) -> pd.DataFrame:
    frame = data[data["method"].astype(str).isin(methods)].copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    grouped = frame.groupby(["method", "pattern"], dropna=False)[metric].mean().unstack("pattern")
    return grouped


def _ordered_patterns(data: pd.DataFrame) -> list[str]:
    rows = (
        data[["pattern", "missing_count"]]
        .dropna(subset=["pattern"])
        .assign(missing_count=lambda df: pd.to_numeric(df["missing_count"], errors="coerce").fillna(99))
        .drop_duplicates()
        .sort_values(["missing_count", "pattern"])
    )
    return [str(item) for item in rows["pattern"]]


def _write_matrix(path: Path, matrix: pd.DataFrame) -> None:
    out = matrix.copy()
    out.insert(0, "method", [method_label(str(idx)) for idx in out.index])
    out.to_csv(path, index=False)


def _write_delta(path: Path, matrix: pd.DataFrame, baseline: str, *, higher_is_better: bool) -> None:
    rows = []
    if REFERENCE_METHOD not in matrix.index or baseline not in matrix.index:
        write_csv(path, rows, ["pattern", "missing_count", "method", "baseline", "delta"])
        return
    for pattern in matrix.columns:
        method_value = matrix.loc[REFERENCE_METHOD, pattern]
        baseline_value = matrix.loc[baseline, pattern]
        delta = method_value - baseline_value if higher_is_better else baseline_value - method_value
        rows.append(
            {
                "pattern": pattern,
                "missing_count": _missing_count(str(pattern)),
                "method": REFERENCE_METHOD,
                "baseline": baseline,
                "delta": delta,
            }
        )
    write_csv(path, rows, ["pattern", "missing_count", "method", "baseline", "delta"])


def _win_rows(top1: pd.DataFrame, mae: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for metric, matrix, higher in (("top1", top1, True), ("mae", mae, False)):
        ranks = matrix.rank(axis=0, method="average", ascending=not higher)
        best = matrix.eq(matrix.max(axis=0), axis=1) if higher else matrix.eq(matrix.min(axis=0), axis=1)
        total = int(matrix.shape[1])
        for method in matrix.index:
            rows.append(
                {
                    "method": method_label(str(method)),
                    "metric": metric,
                    "num_patterns_total": total,
                    "num_patterns_best": int(best.loc[method].sum()) if method in best.index else 0,
                    "num_patterns_beats_bernoulli": _beats(matrix, method, BERNOULLI_METHOD, higher),
                    "num_patterns_beats_classifier_subset": _beats(matrix, method, CLASSIFIER_SUBSET_METHOD, higher),
                    "num_patterns_beats_natural": _beats(matrix, method, "scenes31_34_proto_natural_es40", higher),
                    "num_patterns_beats_uniform": _beats(matrix, method, "scenes31_34_proto_sampler_uniform_es40", higher),
                    "mean_rank": float(ranks.loc[method].mean()) if method in ranks.index else math.nan,
                }
            )
    return rows


def _beats(matrix: pd.DataFrame, method: str, baseline: str, higher: bool) -> int:
    if method not in matrix.index or baseline not in matrix.index:
        return 0
    if higher:
        return int((matrix.loc[method] > matrix.loc[baseline]).sum())
    return int((matrix.loc[method] < matrix.loc[baseline]).sum())


def _plot_heatmap(matrix: pd.DataFrame, stem: Path, title: str, colorbar: str, *, percent: bool, reverse: bool = False) -> None:
    values = matrix.to_numpy(dtype=float)
    if percent:
        values = values * 100.0
    fig, ax = plt.subplots(figsize=(max(8, matrix.shape[1] * 0.45), max(3.2, matrix.shape[0] * 0.45)))
    cmap = "viridis_r" if reverse else "viridis"
    im = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Missing pattern")
    ax.set_ylabel("Method")
    ax.set_xticks(np.arange(matrix.shape[1]), labels=list(matrix.columns), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(matrix.shape[0]), labels=[method_label(str(item)) for item in matrix.index], fontsize=9)
    fig.colorbar(im, ax=ax, label=colorbar)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=220)
    plt.close(fig)


def _plot_delta(matrix: pd.DataFrame, baseline: str, stem: Path, title: str, *, higher_is_better: bool, percent: bool) -> None:
    if REFERENCE_METHOD not in matrix.index or baseline not in matrix.index:
        return
    delta = matrix.loc[REFERENCE_METHOD] - matrix.loc[baseline]
    if not higher_is_better:
        delta = -delta
    fig, ax = plt.subplots(figsize=(max(8, len(delta) * 0.45), 3.6))
    ax.axhline(0, color="0.3", linewidth=0.8)
    values = delta.to_numpy(dtype=float) * (100.0 if percent else 1.0)
    colors = [_count_color(_missing_count(str(pattern))) for pattern in delta.index]
    ax.bar(np.arange(len(delta)), values, color=colors)
    ax.set_title(title)
    ax.set_xlabel("Missing pattern")
    ax.set_ylabel("Delta Top-1 (percentage points)" if percent else "Delta MAE (beam index)")
    ax.set_xticks(np.arange(len(delta)), labels=list(delta.index), rotation=45, ha="right", fontsize=8)
    _add_group_separators(ax, list(delta.index))
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=220)
    plt.close(fig)


def _plot_grouped_heatmap(matrix: pd.DataFrame, stem: Path) -> None:
    values = matrix.to_numpy(dtype=float) * 100.0
    fig, ax = plt.subplots(figsize=(max(8.5, matrix.shape[1] * 0.5), max(3.2, matrix.shape[0] * 0.5)))
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_title("Top-1 accuracy by missing pattern")
    ax.set_xlabel("Missing pattern")
    ax.set_ylabel("Method")
    ax.set_xticks(np.arange(matrix.shape[1]), labels=list(matrix.columns), rotation=55, ha="right", fontsize=8)
    ax.set_yticks(np.arange(matrix.shape[0]), labels=[_paper_label(str(item)) for item in matrix.index], fontsize=9)
    _add_group_separators(ax, list(matrix.columns))
    fig.colorbar(im, ax=ax, label="Top1 accuracy (%)")
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=220)
    plt.close(fig)


def _add_group_separators(ax: Any, patterns: list[str]) -> None:
    counts = [_missing_count(str(pattern)) for pattern in patterns]
    last = counts[0] if counts else None
    for index, count in enumerate(counts[1:], start=1):
        if count != last:
            ax.axvline(index - 0.5, color="white", linewidth=1.2)
            last = count


def _paper_label(method: str) -> str:
    if "amber_lite" in method:
        return "AMBER-lite"
    return PAPER_LABELS.get(method, method_label(method))


def _count_color(count: int) -> str:
    return {0: "#4C78A8", 1: "#F58518", 2: "#54A24B", 3: "#B279A2"}.get(count, "#9D9D9D")


def _missing_count(pattern: str) -> int:
    if pattern == "full":
        return 0
    if pattern.startswith("missing_") and "_and_" in pattern:
        return len(pattern.removeprefix("missing_").split("_and_"))
    if pattern.startswith("missing_"):
        return 1
    if pattern.endswith("_only"):
        return 3
    if pattern == "non_gps_only":
        return 1
    return -1


if __name__ == "__main__":
    raise SystemExit(main())
