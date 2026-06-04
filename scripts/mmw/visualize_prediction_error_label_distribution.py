#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from kd_sensing.data.beam_label_calibration import resolve_beam_label_mapping

DEFAULT_SCENARIOS = (
    "Town10_crossroad_seed24",
    "Town10_skybridge_seed24",
    "Town10_curvyroad_seed42",
    "Town10_Hroad_seed42",
)
DEFAULT_COLORS = ("#0072bc", "#7ac143", "#ed1c24", "#777777")


@dataclass(frozen=True)
class ErrorLabelSeries:
    name: str
    scenario: str
    prediction_csv: Path
    target_scene: str
    source_scenes: tuple[str, ...]
    split_protocol: str
    correct_distance_tolerance: int
    beam_index_mode: str
    beam_label_space: str
    beam_label_mapping_fingerprint: str
    total_count: int
    error_count: int
    labels: np.ndarray
    predicted_labels: np.ndarray
    counts: np.ndarray
    pdf: np.ndarray
    smooth_pdf: np.ndarray


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    scenarios = _split_csv_arg(args.scenarios) or list(DEFAULT_SCENARIOS)
    display_names = _split_csv_arg(args.display_names)
    if display_names and len(display_names) != len(scenarios):
        raise ValueError("--display-names must match --scenarios length.")

    summary = _load_run_summary(args.prediction_root / "summary.json")
    series = []
    for index, scenario in enumerate(scenarios):
        name = display_names[index] if display_names else _default_display_name(scenario)
        series.append(
            load_error_label_series(
                args.prediction_root / scenario / "predictions.csv",
                name=name,
                scenario=scenario,
                num_classes=args.num_classes,
                smoothing_sigma=args.smoothing_sigma,
                correct_distance_tolerance=args.correct_distance_tolerance,
                prediction_column=args.prediction_column,
                beam_index_mode=args.beam_index_mode,
                beam_label_calibration=json.loads(args.beam_label_calibration_json) if args.beam_label_calibration_json else None,
                summary=summary.get(scenario, {}),
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / args.figure_name
    summary_path = args.output_dir / args.summary_name
    plot_error_label_distribution(
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
                "prediction_root": str(args.prediction_root),
                "scenarios": [item.scenario for item in series],
            },
            indent=2,
        )
    )
    return 0


def load_error_label_series(
    prediction_csv: Path,
    *,
    name: str,
    scenario: str,
    num_classes: int,
    smoothing_sigma: float,
    correct_distance_tolerance: int,
    prediction_column: str,
    beam_index_mode: str,
    beam_label_calibration: dict[str, object] | None,
    summary: dict[str, object],
) -> ErrorLabelSeries:
    if not prediction_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found for {scenario}: {prediction_csv}")
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    total_count = 0
    mapping = _resolve_error_label_mapping(
        scenario=scenario,
        num_classes=int(num_classes),
        beam_index_mode=beam_index_mode,
        calibration=beam_label_calibration,
        summary=summary,
    )
    with prediction_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if prediction_column not in (reader.fieldnames or []):
            raise ValueError(
                f"Prediction CSV {prediction_csv} does not contain column '{prediction_column}'. "
                "Use --prediction-column anchor_center_beam only for coarse-anchor diagnostics."
            )
        for row in reader:
            total_count += 1
            truth = _int_value(row.get("true_beam"))
            pred = _int_value(row.get(prediction_column))
            if truth is None or pred is None or truth < 0:
                continue
            truth_mapped = mapping.map_label(int(truth))
            pred_mapped = mapping.map_label(int(pred))
            if _circular_distance(truth_mapped, pred_mapped, num_classes=int(num_classes)) > int(correct_distance_tolerance):
                true_labels.append(truth_mapped)
                predicted_labels.append(pred_mapped)
    labels = np.asarray(true_labels, dtype=np.int64)
    predicted = np.asarray(predicted_labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=int(num_classes)).astype(float) if labels.size else np.zeros(int(num_classes))
    pdf = counts / counts.sum() if counts.sum() > 0 else counts
    smooth_pdf = _smooth_distribution(pdf, sigma=float(smoothing_sigma))
    source_scenes = summary.get("source_scenes", ())
    if not isinstance(source_scenes, (list, tuple)):
        source_scenes = ()
    return ErrorLabelSeries(
        name=name,
        scenario=scenario,
        prediction_csv=prediction_csv,
        target_scene=str(summary.get("target_scene") or scenario),
        source_scenes=tuple(str(item) for item in source_scenes),
        split_protocol=str(summary.get("split_protocol") or ""),
        correct_distance_tolerance=int(correct_distance_tolerance),
        beam_index_mode=str(beam_index_mode),
        beam_label_space=mapping.label_space,
        beam_label_mapping_fingerprint=mapping.fingerprint,
        total_count=int(total_count),
        error_count=int(len(labels)),
        labels=labels,
        predicted_labels=predicted,
        counts=counts,
        pdf=pdf,
        smooth_pdf=smooth_pdf,
    )


def plot_error_label_distribution(
    series: Iterable[ErrorLabelSeries],
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
        fig = plt.figure(figsize=(11.8, 7.1), constrained_layout=True)
        grid = fig.add_gridspec(3, 4, height_ratios=[1.35, 1.05, 1.05])
        ax_main = fig.add_subplot(grid[0, :])
        small_axes = [
            fig.add_subplot(grid[1, 0:2]),
            fig.add_subplot(grid[1, 2:4]),
            fig.add_subplot(grid[2, 0:2]),
            fig.add_subplot(grid[2, 2:4]),
        ]
    else:
        fig, ax_main = plt.subplots(figsize=(11.0, 4.9), constrained_layout=True)
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
        label = f"T:{item.name} err={item.error_count}/{item.total_count}"
        (line,) = ax_main.plot(x, padded_smooth, color=color, linewidth=2.4, label=label)
        line_handles.append(line)

    if title:
        fig.suptitle(title, y=0.985)
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.set_xlim(-1, max_classes)
    ax_main.grid(True, color="#d0d0d0", linewidth=0.8)
    fig.legend(
        handles=line_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=min(2, len(items)),
        frameon=True,
        fontsize=9,
    )

    for ax, item, color in zip(small_axes, items, colors):
        xs = np.arange(len(item.pdf))
        ax.bar(xs, item.pdf, color=color, alpha=0.72, width=0.86)
        ax.plot(xs, item.smooth_pdf, color="#202020", linewidth=1.4)
        sources = ", ".join(_default_display_name(scene) for scene in item.source_scenes) or "none"
        calib = "target support calib" if item.split_protocol == "target_adapt_to_target_test" else item.split_protocol
        ax.set_title(
            f"TARGET {item.name}: wrong true labels (n={item.error_count}/{item.total_count})\n"
            f"LOSO SRC {sources}; {calib}",
            fontsize=9,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-1, max_classes)
        ax.grid(True, axis="y", color="#d8d8d8", linewidth=0.7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(series: Iterable[ErrorLabelSeries], output_path: Path) -> None:
    payload = []
    for item in series:
        top_indices = np.argsort(item.counts)[::-1][:10]
        payload.append(
            {
                "name": item.name,
                "target_scene": item.target_scene,
                "source_scenes": list(item.source_scenes),
                "split_protocol": item.split_protocol,
                "correct_distance_tolerance": int(item.correct_distance_tolerance),
                "beam_index_mode": item.beam_index_mode,
                "beam_label_space": item.beam_label_space,
                "beam_label_mapping_fingerprint": item.beam_label_mapping_fingerprint,
                "prediction_csv": str(item.prediction_csv),
                "total_count": int(item.total_count),
                "error_count": int(item.error_count),
                "error_rate": float(item.error_count / max(item.total_count, 1)),
                "num_classes": int(len(item.counts)),
                "top_wrong_true_labels": [
                    {
                        "label": int(index),
                        "count": int(item.counts[index]),
                        "pdf": float(item.pdf[index]) if item.pdf.sum() > 0 else 0.0,
                    }
                    for index in top_indices
                    if item.counts[index] > 0
                ],
            }
        )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize true-label distributions for wrong MMW beam predictions.")
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("outputs/gps_coarse_anchor/target_adapt_beambench_dba"),
        help="Directory containing summary.json and per-scenario predictions.csv files.",
    )
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--display-names", default="")
    parser.add_argument("--num-classes", type=int, default=64)
    parser.add_argument(
        "--correct-distance-tolerance",
        type=int,
        default=0,
        help="Circular beam distance counted as correct. Use 1 to treat adjacent beams as correct.",
    )
    parser.add_argument("--smoothing-sigma", type=float, default=1.2)
    parser.add_argument(
        "--prediction-column",
        default="predicted_beam",
        help="Prediction CSV column to compare against true_beam.",
    )
    parser.add_argument(
        "--beam-index-mode",
        choices=("raw_label", "calibrated_label"),
        default="raw_label",
        help="Interpret true/predicted labels as raw labels or map them to calibrated label space first.",
    )
    parser.add_argument(
        "--beam-label-calibration-json",
        help="Optional JSON calibration config used when --beam-index-mode=calibrated_label.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/mmw_town_label_distribution"))
    parser.add_argument("--figure-name", default="mmw_town_prediction_error_label_distribution.png")
    parser.add_argument("--summary-name", default="mmw_town_prediction_error_label_distribution_summary.json")
    parser.add_argument("--title", default="MMW Town Wrong Prediction True-Label Distributions")
    parser.add_argument("--xlabel", default="True beam label for wrong predictions")
    parser.add_argument("--ylabel", default="PDF among errors")
    parser.add_argument("--no-bars", action="store_true")
    parser.add_argument("--no-small-multiples", action="store_true")
    return parser.parse_args(argv)


def _resolve_error_label_mapping(
    *,
    scenario: str,
    num_classes: int,
    beam_index_mode: str,
    calibration: dict[str, object] | None,
    summary: dict[str, object],
):
    if beam_index_mode != "calibrated_label":
        return resolve_beam_label_mapping(None, scene=scenario, default_num_classes=int(num_classes))
    payload = dict(calibration or {})
    payload.setdefault("enabled", True)
    payload.setdefault("label_space", "calibrated_gps_angle")
    payload.setdefault("num_classes", int(num_classes))
    if "direction" not in payload and "beam_direction" in summary:
        payload["direction"] = summary.get("beam_direction")
    if "offset" not in payload and "beam_offset" in summary:
        payload["offset"] = summary.get("beam_offset")
    payload.setdefault("fit_source", summary.get("split_protocol", "prediction_error_visualization"))
    return resolve_beam_label_mapping(payload, scene=scenario, default_num_classes=int(num_classes))


def _load_run_summary(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("scene_results", []) if isinstance(payload, dict) else []
    return {str(row.get("target_scene")): row for row in rows if isinstance(row, dict) and row.get("target_scene")}


def _int_value(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _circular_distance(left: int, right: int, *, num_classes: int) -> int:
    diff = abs(int(left) % int(num_classes) - int(right) % int(num_classes))
    return int(min(diff, int(num_classes) - diff))


def _smooth_distribution(pdf: np.ndarray, *, sigma: float) -> np.ndarray:
    if sigma <= 0 or pdf.size == 0 or pdf.sum() <= 0:
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
