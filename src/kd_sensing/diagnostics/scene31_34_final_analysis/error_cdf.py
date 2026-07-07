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

from kd_sensing.diagnostics.scene31_34_final_analysis.common import (
    DEFAULT_ANALYSIS_METHODS,
    REFERENCE_METHOD,
    best_external_method,
    load_predictions,
    method_label,
    method_rows_from_summary,
    roots_from_args,
    write_csv,
)


FIELDS = ["method", "condition", "abs_error_threshold", "cdf", "num_samples"]
CONDITIONS = {
    "all_missing": lambda df: df["missing_count"] > 0,
    "miss3": lambda df: df["missing_count"] == 3,
    "missing_ratio_75": lambda df: df["missing_count"] == 3,
}
FIGURES = {
    "all_missing": (
        "fig_abs_error_cdf_all_missing",
        "fig_abs_error_cdf_all_missing_paper",
        "fig_abs_error_cdf_all_missing_presentation",
    ),
    "miss3": ("fig_abs_error_cdf_miss3", "fig_abs_error_cdf_miss3_paper", "fig_abs_error_cdf_miss3_presentation"),
    "missing_ratio_75": ("fig_abs_error_cdf_missing_ratio_75",),
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
    data = load_predictions(roots, set(methods))
    if data.empty:
        raise SystemExit("No predictions_by_pattern.csv rows found for requested methods.")
    rows = _cdf_rows(data, methods)
    write_csv(out / "abs_error_cdf_data.csv", rows, FIELDS)
    for condition in CONDITIONS:
        for figure in FIGURES[condition]:
            _plot_condition(rows, methods, condition, out / figure)
    (out / "caption_notes.txt").write_text(
        "CDF curves show beam-index error distribution; higher and left-shifted curves are better.\n",
        encoding="utf-8",
    )
    print(f"Wrote Scene31-34 error CDF analysis to {out}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Scene31-34 absolute beam error CDFs.")
    parser.add_argument("--root", default="outputs/scenes31_34_main_lmdb")
    parser.add_argument("--classifier-root", action="append", default=[])
    parser.add_argument("--external-root", action="append", default=[])
    parser.add_argument("--old-root", action="append", default=[])
    parser.add_argument("--out", default="outputs/scenes31_34_main_lmdb/error_cdf")
    return parser


def _cdf_rows(data: pd.DataFrame, methods: list[str]) -> list[dict[str, Any]]:
    rows = []
    data = data.copy()
    data["abs_error"] = pd.to_numeric(data["abs_error"], errors="coerce")
    for condition, selector in CONDITIONS.items():
        subset = data[selector(data)]
        for method in methods:
            values = subset.loc[subset["method"].astype(str) == method, "abs_error"].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            thresholds = np.arange(0, int(math.ceil(float(np.nanmax(values)))) + 1)
            for threshold in thresholds:
                rows.append(
                    {
                        "method": method,
                        "condition": condition,
                        "abs_error_threshold": int(threshold),
                        "cdf": float((values <= threshold).mean()),
                        "num_samples": int(values.size),
                    }
                )
    return rows


def _plot_condition(rows: list[dict[str, Any]], methods: list[str], condition: str, stem: Path) -> None:
    frame = pd.DataFrame(rows)
    frame = frame[frame["condition"] == condition]
    is_talk = stem.name.endswith("_paper") or stem.name.endswith("_presentation")
    fig, ax = plt.subplots(figsize=(9.2, 4.9) if is_talk else (6.5, 4.2))
    for method in methods:
        part = frame[frame["method"] == method].sort_values("abs_error_threshold")
        if part.empty:
            continue
        ax.plot(
            part["abs_error_threshold"],
            part["cdf"],
            label=method_label(method),
            linewidth=3.0 if method == REFERENCE_METHOD and is_talk else 2.3 if method == REFERENCE_METHOD else 1.5,
        )
    ax.axvline(3, color="0.20", linestyle="--", linewidth=1.2)
    ax.annotate(
        "Within@3",
        xy=(3, 0.98),
        xytext=(4.2, 0.91),
        arrowprops={"arrowstyle": "->", "color": "0.25", "linewidth": 0.9},
        fontsize=11 if is_talk else 9,
        ha="left",
    )
    title = f"Absolute Beam Error CDF ({condition.replace('_', ' ')})" if is_talk else f"Absolute beam error CDF ({condition.replace('_', ' ')})"
    ax.set_title(title)
    ax.set_xlabel("Absolute beam-index error" if is_talk else "absolute beam index error")
    ax.set_ylabel("CDF: P(|error| <= x)" if is_talk else "P(|error| <= x)")
    ax.set_ylim(0, 1.02)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    if is_talk:
        ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
    else:
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.tight_layout(rect=(0, 0, 0.78, 1))
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
