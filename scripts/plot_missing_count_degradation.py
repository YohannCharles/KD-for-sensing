#!/usr/bin/env python3

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = (
    "scenes31_34_proto_natural_es40",
    "scenes31_34_proto_sampler_uniform_es40",
    "scenes31_34_proto_randomdrop_bernoulli_k075_es40",
    "scenes31_34_proto_randomdrop_subset_es40",
)
LABELS = {
    "scenes31_34_proto_natural_es40": "Natural",
    "scenes31_34_proto_sampler_uniform_es40": "Uniform pattern exposure",
    "scenes31_34_proto_randomdrop_bernoulli_k075_es40": "Bernoulli randomdrop",
    "scenes31_34_proto_randomdrop_subset_es40": "Random subset exposure",
}
COLORS = {
    "scenes31_34_proto_natural_es40": "#4C78A8",
    "scenes31_34_proto_sampler_uniform_es40": "#F58518",
    "scenes31_34_proto_randomdrop_bernoulli_k075_es40": "#54A24B",
    "scenes31_34_proto_randomdrop_subset_es40": "#B279A2",
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary_root = Path(args.summary_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    curve = _read_csv(summary_root / "missing_count_curve.csv")
    scene_curve = _read_csv(summary_root / "missing_count_curve_by_scene.csv")
    _plot_metric(
        curve,
        out / "fig_top1_vs_missing_count",
        metric="top1",
        ylabel="Top-1 accuracy (%)",
        title="Top-1 Accuracy vs. Missing Modality Ratio on Scenes31-34",
        percentage=True,
    )
    _plot_metric(
        curve,
        out / "fig_within3_vs_missing_count",
        metric="within3",
        ylabel="Within-3 accuracy (%)",
        title="Within-3 Accuracy vs. Missing Modality Ratio on Scenes31-34",
        percentage=True,
    )
    _plot_metric(
        curve,
        out / "fig_mae_vs_missing_count",
        metric="mae",
        ylabel="Beam index MAE",
        title="Beam MAE vs. Missing Modality Ratio on Scenes31-34",
        percentage=False,
    )
    _plot_scene_top1(scene_curve, out / "fig_top1_vs_missing_ratio_by_scene")
    print(f"Wrote missing-count degradation figures to {out}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Scene31-34 missing-count degradation curves.")
    parser.add_argument("--summary-root", default="outputs/scenes31_34_main_lmdb/summary")
    parser.add_argument("--out", default="outputs/scenes31_34_main_lmdb/figures")
    return parser


def _plot_metric(rows: list[dict[str, str]], base: Path, *, metric: str, ylabel: str, title: str, percentage: bool) -> None:
    plt.figure(figsize=(7.2, 4.6))
    best = _best_method(rows, metric)
    has_any = False
    for method in METHODS:
        method_rows = sorted([row for row in rows if row.get("method") == method], key=lambda row: _float(row.get("missing_count")))
        if not method_rows:
            continue
        xs = [_float(row.get("missing_ratio")) * 100.0 for row in method_rows]
        ys = [_scale(_float(row.get(f"{metric}_mean")), percentage) for row in method_rows]
        if not any(math.isfinite(value) for value in ys):
            continue
        min_n = min((_float(row.get("n")) for row in method_rows if math.isfinite(_float(row.get("n")))), default=0.0)
        yerr = [
            _scale(_float(row.get(f"{metric}_std")), percentage)
            if _float(row.get("n")) >= 3 and math.isfinite(_float(row.get(f"{metric}_std")))
            else 0.0
            for row in method_rows
        ]
        linewidth = 3.0 if method == best else 1.8
        label = LABELS.get(method, method)
        if min_n < 5:
            label = f"{label} (n={int(min_n) if min_n else 0})"
        plt.errorbar(
            xs,
            ys,
            yerr=yerr if any(value > 0 for value in yerr) else None,
            marker="o",
            linewidth=linewidth,
            capsize=3,
            color=COLORS.get(method),
            label=label,
        )
        has_any = True
    _finish_axes(title, ylabel)
    if not has_any:
        plt.text(0.5, 0.5, "No curve data available", ha="center", va="center", transform=plt.gca().transAxes)
    _save(base)


def _plot_scene_top1(rows: list[dict[str, str]], base: Path) -> None:
    scenes = ["Scene31", "Scene32", "Scene33", "Scene34"]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8), sharex=True, sharey=True)
    axes_list = list(axes.ravel())
    for ax, scene in zip(axes_list, scenes, strict=False):
        scene_rows = [row for row in rows if row.get("scene") == scene]
        best = _best_method(scene_rows, "top1", field="top1")
        for method in METHODS:
            method_rows = _aggregate_scene_curve([row for row in scene_rows if row.get("method") == method])
            if not method_rows:
                continue
            xs = [_float(row.get("missing_ratio")) * 100.0 for row in method_rows]
            ys = [_scale(_float(row.get("top1")), True) for row in method_rows]
            if not any(math.isfinite(value) for value in ys):
                continue
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.6 if method == best else 1.5,
                color=COLORS.get(method),
                label=LABELS.get(method, method),
            )
        ax.set_title(scene)
        ax.set_xticks([0, 25, 50, 75])
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    for ax in axes_list[::2]:
        ax.set_ylabel("Top-1 accuracy (%)")
    for ax in axes_list[-2:]:
        ax.set_xlabel("Missing modality ratio")
    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Per-Scene Top-1 Degradation on Scenes31-34")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _save(base, fig=fig)


def _finish_axes(title: str, ylabel: str) -> None:
    plt.title(title)
    plt.xlabel("Missing modality ratio")
    plt.ylabel(ylabel)
    plt.xticks([0, 25, 50, 75], ["0%", "25%", "50%", "75%"])
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    plt.tight_layout(rect=(0, 0.08, 1, 1))


def _save(base: Path, *, fig: Any | None = None) -> None:
    target = fig or plt.gcf()
    target.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    target.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(target)


def _best_method(rows: list[dict[str, str]], metric: str, *, field: str | None = None) -> str:
    grouped: dict[str, list[float]] = defaultdict(list)
    key = field or f"{metric}_mean"
    for row in rows:
        value = _float(row.get(key))
        if math.isfinite(value):
            grouped[str(row.get("method") or "")].append(value)
    if not grouped:
        return "scenes31_34_proto_randomdrop_subset_es40"
    if metric == "mae":
        return min(grouped, key=lambda method: sum(grouped[method]) / len(grouped[method]))
    return max(grouped, key=lambda method: sum(grouped[method]) / len(grouped[method]))


def _aggregate_scene_curve(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        count = int(_float(row.get("missing_count"))) if math.isfinite(_float(row.get("missing_count"))) else -1
        if count >= 0:
            grouped[count].append(row)
    out: list[dict[str, str]] = []
    for count in sorted(grouped):
        items = grouped[count]
        out.append(
            {
                "missing_count": str(count),
                "missing_ratio": str(count / 4),
                "top1": str(_mean(_float(row.get("top1")) for row in items)),
            }
        )
    return out


def _scale(value: float, percentage: bool) -> float:
    if not math.isfinite(value):
        return math.nan
    return value * 100.0 if percentage and value <= 1.5 else value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _mean(values: Any) -> float:
    nums = [value for value in values if math.isfinite(value)]
    return sum(nums) / len(nums) if nums else math.nan


if __name__ == "__main__":
    raise SystemExit(main())
