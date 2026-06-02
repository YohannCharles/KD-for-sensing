#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SCENARIOS = (
    "Town10_crossroad_seed24",
    "Town10_skybridge_seed24",
    "Town10_curvyroad_seed42",
    "Town10_Hroad_seed42",
)
DEFAULT_COLORS = ("#0072bc", "#7ac143", "#ed1c24", "#777777")


@dataclass(frozen=True)
class LabelSeries:
    name: str
    scenario: str
    source_csv: Path
    labels: np.ndarray
    counts: np.ndarray
    pdf: np.ndarray
    smooth_pdf: np.ndarray


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    scenarios = _split_csv_arg(args.scenarios) or list(DEFAULT_SCENARIOS)
    display_names = _split_csv_arg(args.display_names)
    if display_names and len(display_names) != len(scenarios):
        raise ValueError("--display-names must have the same number of entries as --scenarios.")

    series = []
    for index, scenario in enumerate(scenarios):
        csv_path = _resolve_csv_path(
            prepared_root=args.prepared_root,
            scenario=scenario,
            source=args.source,
            split=args.split,
            split_tag=args.split_tag,
        )
        name = display_names[index] if display_names else _default_display_name(scenario)
        series.append(
            load_label_series(
                csv_path,
                name=name,
                scenario=scenario,
                label_column=args.label_column,
                num_classes=args.num_classes,
                smoothing_sigma=args.smoothing_sigma,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / args.figure_name
    summary_path = args.output_dir / args.summary_name
    plot_label_distribution(
        series,
        output_path=figure_path,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        show_bars=not args.no_bars,
        show_small_multiples=not args.no_small_multiples,
    )
    write_summary(series, summary_path)
    print(
        json.dumps(
            {
                "figure": str(figure_path),
                "summary": str(summary_path),
                "scenarios": [item.scenario for item in series],
                "label_column": args.label_column,
                "source": args.source,
            },
            indent=2,
        )
    )
    return 0


def load_label_series(
    csv_path: Path,
    *,
    name: str,
    scenario: str,
    label_column: str,
    num_classes: int | None,
    smoothing_sigma: float,
) -> LabelSeries:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found for {scenario}: {csv_path}")
    frame = pd.read_csv(csv_path)
    column = _resolve_label_column(frame, label_column)
    labels = pd.to_numeric(frame[column], errors="coerce").dropna().astype(int).to_numpy()
    labels = labels[labels >= 0]
    if labels.size == 0:
        raise ValueError(f"No non-negative labels found in {csv_path} column {column!r}.")
    class_count = int(num_classes) if num_classes else int(labels.max()) + 1
    if class_count <= int(labels.max()):
        raise ValueError(f"--num-classes={class_count} is smaller than max label {int(labels.max())}.")
    counts = np.bincount(labels, minlength=class_count).astype(float)
    pdf = counts / counts.sum()
    smooth_pdf = _smooth_distribution(pdf, sigma=float(smoothing_sigma))
    return LabelSeries(
        name=name,
        scenario=scenario,
        source_csv=csv_path,
        labels=labels,
        counts=counts,
        pdf=pdf,
        smooth_pdf=smooth_pdf,
    )


def plot_label_distribution(
    series: Iterable[LabelSeries],
    *,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    show_bars: bool,
    show_small_multiples: bool,
) -> None:
    items = list(series)
    if not items:
        raise ValueError("No series to plot.")
    max_classes = max(len(item.pdf) for item in items)
    x = np.arange(max_classes)

    if show_small_multiples:
        fig = plt.figure(figsize=(11.0, 6.6), constrained_layout=True)
        grid = fig.add_gridspec(3, 4, height_ratios=[1.35, 1.0, 1.0])
        ax_main = fig.add_subplot(grid[0, :])
        small_axes = [
            fig.add_subplot(grid[1, 0:2]),
            fig.add_subplot(grid[1, 2:4]),
            fig.add_subplot(grid[2, 0:2]),
            fig.add_subplot(grid[2, 2:4]),
        ]
    else:
        fig, ax_main = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
        small_axes = []

    colors = _color_cycle(len(items))
    bar_width = min(0.8 / max(len(items), 1), 0.18)
    offsets = (np.arange(len(items)) - (len(items) - 1) / 2.0) * bar_width

    line_handles = []
    for index, item in enumerate(items):
        color = colors[index]
        padded_pdf = _pad(item.pdf, max_classes)
        padded_smooth = _pad(item.smooth_pdf, max_classes)
        if show_bars:
            ax_main.bar(
                x + offsets[index],
                padded_pdf,
                width=bar_width,
                color=color,
                alpha=0.24,
                edgecolor=color,
                linewidth=0.6,
            )
        (line,) = ax_main.plot(x, padded_smooth, color=color, linewidth=2.4, label=item.name)
        line_handles.append(line)

    if title:
        fig.suptitle(title, y=0.97)
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.set_xlim(-1, max_classes)
    ax_main.grid(True, color="#d0d0d0", linewidth=0.8)
    fig.legend(
        handles=line_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=min(4, len(items)),
        frameon=True,
    )

    for ax, item, color in zip(small_axes, items, colors):
        xs = np.arange(len(item.pdf))
        ax.bar(xs, item.pdf, color=color, alpha=0.72, width=0.86)
        ax.plot(xs, item.smooth_pdf, color="#202020", linewidth=1.4)
        ax.set_title(f"{item.name} (n={len(item.labels)})", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-1, max_classes)
        ax.grid(True, axis="y", color="#d8d8d8", linewidth=0.7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(series: Iterable[LabelSeries], output_path: Path) -> None:
    payload = []
    for item in series:
        nonzero = np.flatnonzero(item.counts > 0)
        top_indices = np.argsort(item.counts)[::-1][:10]
        payload.append(
            {
                "name": item.name,
                "scenario": item.scenario,
                "source_csv": str(item.source_csv),
                "sample_count": int(len(item.labels)),
                "num_classes": int(len(item.counts)),
                "unique_label_count": int(len(nonzero)),
                "min_label": int(item.labels.min()),
                "max_label": int(item.labels.max()),
                "top_labels": [
                    {
                        "label": int(index),
                        "count": int(item.counts[index]),
                        "pdf": float(item.pdf[index]),
                    }
                    for index in top_indices
                    if item.counts[index] > 0
                ],
            }
        )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize MMW prepared Town label distributions.")
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("dataset/MMW/sunny/Prepared"),
        help="MMW Prepared directory containing scenario subdirectories.",
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated prepared scenario directories to compare.",
    )
    parser.add_argument(
        "--display-names",
        default="",
        help="Optional comma-separated legend names. Must match --scenarios length.",
    )
    parser.add_argument(
        "--source",
        choices=("manifest", "split"),
        default="manifest",
        help="Read labels from frame manifests or sequence split CSVs.",
    )
    parser.add_argument("--split", choices=("all_sequences", "train", "test"), default="all_sequences")
    parser.add_argument("--split-tag", default="", help="Optional split tag directory under splits/.")
    parser.add_argument("--label-column", default="beam_label", help="Label column to plot.")
    parser.add_argument("--num-classes", type=int, default=64, help="Number of label bins/classes.")
    parser.add_argument("--smoothing-sigma", type=float, default=1.2, help="Gaussian smoothing sigma in label bins.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/mmw_town_label_distribution"))
    parser.add_argument("--figure-name", default="mmw_town_label_distribution.png")
    parser.add_argument("--summary-name", default="mmw_town_label_distribution_summary.json")
    parser.add_argument("--title", default="MMW Town Beam Label Distributions")
    parser.add_argument("--xlabel", default="Beam label")
    parser.add_argument("--ylabel", default="PDF")
    parser.add_argument("--no-bars", action="store_true", help="Hide faint histogram bars in the overlay plot.")
    parser.add_argument("--no-small-multiples", action="store_true", help="Only write the overlay plot.")
    return parser.parse_args(argv)


def _resolve_csv_path(
    *,
    prepared_root: Path,
    scenario: str,
    source: str,
    split: str,
    split_tag: str,
) -> Path:
    scenario_root = prepared_root / scenario
    if source == "manifest":
        return scenario_root / "manifests" / "frame_manifest.csv"
    split_dir = scenario_root / "splits"
    if split_tag:
        split_dir = split_dir / split_tag
    return split_dir / f"{split}.csv"


def _resolve_label_column(frame: pd.DataFrame, requested: str) -> str:
    if requested in frame.columns:
        return requested
    fallback_columns = (
        "beam_label",
        "future_beam_label1",
        "target_beam",
        "coarse_sector",
        "radio_semantic_label",
    )
    for column in fallback_columns:
        if column in frame.columns:
            return column
    raise ValueError(
        f"Label column {requested!r} not found. Available columns: {', '.join(map(str, frame.columns[:40]))}"
    )


def _smooth_distribution(pdf: np.ndarray, *, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return pdf
    radius = max(int(round(sigma * 4.0)), 1)
    taps = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (taps / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(pdf, (radius, radius), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="same")[radius:-radius]
    total = smoothed.sum()
    return smoothed / total if total > 0 else smoothed


def _pad(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) >= length:
        return values
    return np.pad(values, (0, length - len(values)), constant_values=0)


def _split_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _default_display_name(scenario: str) -> str:
    text = scenario
    if text.startswith("Town10_"):
        text = text[len("Town10_") :]
    for suffix in ("_seed24", "_seed42"):
        text = text.replace(suffix, "")
    return text.replace("_", " ")


def _color_cycle(length: int) -> list[str]:
    if length <= len(DEFAULT_COLORS):
        return list(DEFAULT_COLORS[:length])
    cmap = plt.get_cmap("tab10")
    return [DEFAULT_COLORS[index] if index < len(DEFAULT_COLORS) else cmap(index % 10) for index in range(length)]


if __name__ == "__main__":
    raise SystemExit(main())
