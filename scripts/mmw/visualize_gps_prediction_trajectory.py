#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
DBA_TOP_K = 3
DBA_DELTA = 5.0


@dataclass(frozen=True)
class TrajectoryPoint:
    scene: str
    agent: str
    sample_id: str
    seq_index: int
    x: float
    y: float
    true_beam: int
    pred_beam: int
    circular_error: int
    signed_residual: int
    dba: float
    beam_label_space: str
    beam_label_mapping_fingerprint: str


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    scenarios = _split_csv_arg(args.scenarios) or list(DEFAULT_SCENARIOS)
    display_names = _split_csv_arg(args.display_names)
    if display_names and len(display_names) != len(scenarios):
        raise ValueError("--display-names must match --scenarios length.")

    scene_points: list[tuple[str, str, list[TrajectoryPoint]]] = []
    for index, scene in enumerate(scenarios):
        name = display_names[index] if display_names else _default_display_name(scene)
        points = load_scene_points(
            prepared_root=args.prepared_root,
            prediction_root=args.prediction_root,
            split_tag=args.split_tag,
            scene=scene,
            split=args.split,
            num_classes=args.num_classes,
            prediction_column=args.prediction_column,
            beam_label_calibration=json.loads(args.beam_label_calibration_json) if args.beam_label_calibration_json else None,
        )
        scene_points.append((scene, name, points))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spatial_path = args.output_dir / args.spatial_figure_name
    sequence_path = args.output_dir / args.sequence_figure_name
    summary_path = args.output_dir / args.summary_name
    plot_spatial_trajectory(
        scene_points,
        output_path=spatial_path,
        title=args.title,
        label_mode=args.spatial_label_mode,
        num_classes=args.num_classes,
    )
    plot_beam_sequence(scene_points, output_path=sequence_path, title=args.title, label_mode=args.sequence_label_mode)
    write_summary(
        scene_points,
        summary_path,
        spatial_label_mode=args.spatial_label_mode,
        sequence_label_mode=args.sequence_label_mode,
        num_classes=args.num_classes,
    )
    print(
        json.dumps(
            {
                "spatial_figure": str(spatial_path),
                "sequence_figure": str(sequence_path),
                "summary": str(summary_path),
                "prediction_root": str(args.prediction_root),
                "scenarios": scenarios,
            },
            indent=2,
        )
    )
    return 0


def load_scene_points(
    *,
    prepared_root: Path,
    prediction_root: Path,
    split_tag: str,
    scene: str,
    split: str,
    num_classes: int,
    prediction_column: str,
    beam_label_calibration: dict[str, object] | None = None,
) -> list[TrajectoryPoint]:
    prediction_path = prediction_root / scene / "predictions.csv"
    split_path = prepared_root / scene / "splits" / split_tag / f"{split}.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {prediction_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split CSV: {split_path}")

    predictions = _load_predictions(prediction_path, prediction_column=prediction_column)
    mapping = resolve_beam_label_mapping(
        beam_label_calibration,
        scene=scene,
        default_num_classes=int(num_classes),
    )
    points: list[TrajectoryPoint] = []
    with split_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row.get("target_sample_id") or row.get("sample_id") or "")
            pred_row = predictions.get(sample_id)
            if not pred_row:
                continue
            geometry = _last_geometry(row)
            x = _float_value(geometry.get("relative_x"), geometry.get("local_x"))
            y = _float_value(geometry.get("relative_y"), geometry.get("local_y"))
            true_beam = _int_value(row.get("future_beam_label1"), row.get("beam_label"))
            pred_beam = _int_value(pred_row.get(prediction_column))
            if x is None or y is None or true_beam is None or pred_beam is None:
                continue
            true_beam = int(mapping.map_label(int(true_beam))) % int(num_classes)
            pred_beam = int(mapping.map_label(int(pred_beam))) % int(num_classes)
            points.append(
                TrajectoryPoint(
                    scene=scene,
                    agent=str(row.get("agent") or _agent_from_sample_id(sample_id)),
                    sample_id=sample_id,
                    seq_index=int(_int_value(row.get("seq_index")) or len(points)),
                    x=float(x),
                    y=float(y),
                    true_beam=true_beam,
                    pred_beam=pred_beam,
                    circular_error=_circular_beam_distance(pred_beam, true_beam, num_classes=int(num_classes)),
                    signed_residual=_signed_circular_delta(pred_beam, true_beam, num_classes=int(num_classes)),
                    dba=_sample_dba_score(predicted=pred_beam, truth=true_beam, num_classes=int(num_classes)),
                    beam_label_space=mapping.label_space,
                    beam_label_mapping_fingerprint=mapping.fingerprint,
                )
            )
    if not points:
        raise ValueError(f"No matched trajectory points found for {scene}.")
    return sorted(points, key=lambda item: (item.agent, item.seq_index, item.sample_id))


def plot_spatial_trajectory(
    scene_points: Iterable[tuple[str, str, list[TrajectoryPoint]]],
    *,
    output_path: Path,
    title: str,
    label_mode: str = "unwrapped",
    num_classes: int = 64,
) -> None:
    items = list(scene_points)
    rows = len(items)
    fig, axes = plt.subplots(rows, 3, figsize=(11.5, max(3.0, rows * 2.65)), constrained_layout=True)
    axes = np.asarray(axes).reshape(rows, 3)
    scene_arrays = [
        _point_arrays(points, label_mode=label_mode, num_classes=int(num_classes))
        for _, _, points in items
    ]
    if _uses_bounded_label_mode(label_mode):
        beam_vmin = 0.0
        beam_vmax = float(int(num_classes) - 1)
        beam_colorbar_label = "beam label (0-63)"
    else:
        beam_values = np.concatenate(
            [arrays["true_display"] for arrays in scene_arrays]
            + [arrays["pred_display"] for arrays in scene_arrays]
        )
        beam_vmin, beam_vmax = _value_limits(beam_values)
        beam_colorbar_label = "unwrapped calibrated beam index"
    beam_mappable = None
    dba_mappable = None
    for row_idx, ((_, name, points), arrays) in enumerate(zip(items, scene_arrays)):
        x_min, x_max, y_min, y_max = _xy_limits(arrays["x"], arrays["y"])
        configs = (
            ("True Beam", arrays["true_display"], beam_vmin, beam_vmax),
            ("GPS Pred Beam", arrays["pred_display"], beam_vmin, beam_vmax),
            ("DBA", arrays["dba"], 0.0, 1.0),
        )
        for col_idx, (label, values, vmin, vmax) in enumerate(configs):
            ax = axes[row_idx, col_idx]
            _plot_path_background(ax, points)
            scatter = ax.scatter(
                arrays["x"],
                arrays["y"],
                c=values,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                s=18,
                linewidths=0.0,
                alpha=0.9,
                zorder=3,
            )
            if col_idx < 2:
                beam_mappable = scatter
            else:
                dba_mappable = scatter
            ax.scatter([0.0], [0.0], marker="x", color="red", s=48, linewidths=2.2, zorder=4)
            ax.text(0.02, 0.94, "BS", transform=ax.transAxes, fontsize=9, va="top", ha="left")
            ax.set_title(f"{name} {label}", fontsize=10)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.grid(True, color="#d0d0d0", linewidth=0.7)
    fig.suptitle(f"{title} Spatial Trajectory", fontsize=13)
    if beam_mappable is not None:
        cbar = fig.colorbar(beam_mappable, ax=axes[:, :2].ravel().tolist(), shrink=0.82, fraction=0.025, pad=0.015)
        cbar.set_label(beam_colorbar_label)
    if dba_mappable is not None:
        cbar = fig.colorbar(dba_mappable, ax=axes[:, 2].ravel().tolist(), shrink=0.82, fraction=0.035, pad=0.015)
        cbar.set_label("DBA")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_beam_sequence(
    scene_points: Iterable[tuple[str, str, list[TrajectoryPoint]]],
    *,
    output_path: Path,
    title: str,
    label_mode: str = "unwrapped",
) -> None:
    items = list(scene_points)
    rows = len(items)
    fig, axes = plt.subplots(rows, 2, figsize=(11.2, max(3.0, rows * 2.5)), constrained_layout=True)
    axes = np.asarray(axes).reshape(rows, 2)
    for row_idx, (_, name, points) in enumerate(items):
        ax_beam = axes[row_idx, 0]
        ax_error = axes[row_idx, 1]
        for agent, agent_points in _group_by_agent(points).items():
            true_values = np.asarray([point.true_beam for point in agent_points], dtype=float)
            pred_values = np.asarray([point.pred_beam for point in agent_points], dtype=float)
            if _uses_bounded_label_mode(label_mode):
                true_plot = true_values
                pred_plot = pred_values
            else:
                residuals = np.asarray([point.signed_residual for point in agent_points], dtype=float)
                true_plot = _unwrap_beams(true_values, period=64)
                pred_plot = true_plot + residuals
            x_values = np.arange(len(agent_points), dtype=float)
            ax_beam.plot(x_values, true_plot, color="#1f1f1f", linewidth=1.4, alpha=0.85, label="true" if agent == sorted(_group_by_agent(points))[0] else None)
            ax_beam.plot(x_values, pred_plot, color="#d62728", linewidth=1.2, alpha=0.82, label="pred" if agent == sorted(_group_by_agent(points))[0] else None)
            ax_error.plot(
                x_values,
                [point.circular_error for point in agent_points],
                color="#2b6cb0",
                linewidth=1.1,
                alpha=0.82,
            )
            ax_error.scatter(
                x_values,
                [point.circular_error for point in agent_points],
                c=[point.dba for point in agent_points],
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                s=14,
                linewidths=0.0,
                alpha=0.9,
            )
        ax_beam.set_title(f"{name} true vs pred beam trajectory", fontsize=10)
        if _uses_bounded_label_mode(label_mode):
            ax_beam.set_ylim(-2, 65)
            ax_beam.set_yticks([0, 16, 32, 48, 63])
            ax_beam.set_ylabel("beam label")
        else:
            ax_beam.set_ylabel("unwrapped beam index")
        ax_beam.set_xlabel("sequence order within agent")
        ax_beam.grid(True, color="#d0d0d0", linewidth=0.7)
        ax_beam.legend(loc="best", fontsize=8)
        ax_error.set_title(f"{name} circular error, color=DBA", fontsize=10)
        ax_error.set_ylabel("circular beam error")
        ax_error.set_xlabel("sequence order within agent")
        ax_error.set_ylim(-1, 33)
        ax_error.grid(True, color="#d0d0d0", linewidth=0.7)
    fig.suptitle(f"{title} Beam-Index Trajectory", fontsize=13)
    mappable = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0.0, vmax=1.0))
    cbar = fig.colorbar(mappable, ax=axes[:, 1].ravel().tolist(), shrink=0.82, fraction=0.035, pad=0.015)
    cbar.set_label("DBA")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    scene_points: Iterable[tuple[str, str, list[TrajectoryPoint]]],
    output_path: Path,
    *,
    spatial_label_mode: str,
    sequence_label_mode: str,
    num_classes: int,
) -> None:
    payload = []
    for scene, name, points in scene_points:
        dba_values = np.asarray([point.dba for point in points], dtype=float)
        errors = np.asarray([point.circular_error for point in points], dtype=float)
        residuals = [int(point.signed_residual) for point in points]
        zero_points = [point for point in points if point.dba <= 1e-12]
        arrays = _point_arrays(points, label_mode=spatial_label_mode, num_classes=int(num_classes))
        payload.append(
            {
                "scene": scene,
                "name": name,
                "sample_count": len(points),
                "beam_label_space": points[0].beam_label_space if points else "raw",
                "beam_label_mapping_fingerprint": points[0].beam_label_mapping_fingerprint if points else None,
                "spatial_label_mode": spatial_label_mode,
                "sequence_label_mode": sequence_label_mode,
                "spatial_true_beam_range": [float(arrays["true_display"].min()), float(arrays["true_display"].max())],
                "spatial_pred_beam_range": [float(arrays["pred_display"].min()), float(arrays["pred_display"].max())],
                "agent_counts": dict(Counter(point.agent for point in points)),
                "dba_mean": float(np.mean(dba_values)),
                "dba_zero_count": len(zero_points),
                "dba_zero_fraction": float(len(zero_points) / max(len(points), 1)),
                "circular_error_mean": float(np.mean(errors)),
                "circular_error_median": float(np.median(errors)),
                "circular_error_max": int(np.max(errors)),
                "signed_residual_histogram_top10": Counter(residuals).most_common(10),
                "zero_dba_true_label_counts_top10": Counter(point.true_beam for point in zero_points).most_common(10),
                "zero_dba_pred_label_counts_top10": Counter(point.pred_beam for point in zero_points).most_common(10),
            }
        )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_predictions(path: Path, *, prediction_column: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if prediction_column not in (reader.fieldnames or []):
            raise ValueError(
                f"Prediction CSV {path} does not contain column '{prediction_column}'. "
                "Use --prediction-column anchor_center_beam only for coarse-anchor diagnostics."
            )
        for row in reader:
            sample_id = str(row.get("sample_id") or "")
            if sample_id and row.get(prediction_column) is not None:
                result[sample_id] = dict(row)
    return result


def _point_arrays(
    points: list[TrajectoryPoint],
    *,
    label_mode: str = "raw",
    num_classes: int = 64,
) -> dict[str, np.ndarray]:
    true_values = np.asarray([point.true_beam for point in points], dtype=float)
    pred_values = np.asarray([point.pred_beam for point in points], dtype=float)
    true_display = true_values.copy()
    pred_display = pred_values.copy()
    if not _uses_bounded_label_mode(label_mode):
        index_by_id = {id(point): index for index, point in enumerate(points)}
        for agent_points in _group_by_agent(points).values():
            indices = [index_by_id[id(point)] for point in agent_points]
            agent_true = np.asarray([point.true_beam for point in agent_points], dtype=float)
            residuals = np.asarray([point.signed_residual for point in agent_points], dtype=float)
            unwrapped_true = _unwrap_beams(agent_true, period=int(num_classes))
            true_display[indices] = unwrapped_true
            pred_display[indices] = unwrapped_true + residuals
    return {
        "x": np.asarray([point.x for point in points], dtype=float),
        "y": np.asarray([point.y for point in points], dtype=float),
        "true": true_values,
        "pred": pred_values,
        "true_display": true_display,
        "pred_display": pred_display,
        "dba": np.asarray([point.dba for point in points], dtype=float),
    }


def _uses_bounded_label_mode(label_mode: str) -> bool:
    return label_mode in {"label", "raw"}


def _value_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(finite.min())
    vmax = float(finite.max())
    span = max(vmax - vmin, 1.0)
    pad = max(span * 0.04, 1.0)
    return vmin - pad, vmax + pad


def _plot_path_background(ax: Any, points: list[TrajectoryPoint]) -> None:
    for agent_points in _group_by_agent(points).values():
        xs = [point.x for point in agent_points]
        ys = [point.y for point in agent_points]
        ax.plot(xs, ys, color="#8a8a8a", linewidth=0.65, alpha=0.35, zorder=1)


def _group_by_agent(points: list[TrajectoryPoint]) -> dict[str, list[TrajectoryPoint]]:
    grouped: dict[str, list[TrajectoryPoint]] = defaultdict(list)
    for point in points:
        grouped[point.agent].append(point)
    return {agent: sorted(items, key=lambda item: (item.seq_index, item.sample_id)) for agent, items in sorted(grouped.items())}


def _sample_dba_score(*, predicted: int, truth: int, num_classes: int) -> float:
    neighbors = _topk_neighbors(int(predicted), num_classes=int(num_classes), k=DBA_TOP_K)
    distances = [
        min(_circular_beam_distance(int(item), int(truth), num_classes=int(num_classes)) / DBA_DELTA, 1.0)
        for item in neighbors
    ]
    best_so_far = 1.0
    terms = []
    for distance in distances:
        best_so_far = min(best_so_far, float(distance))
        terms.append(1.0 - best_so_far)
    return float(sum(terms) / max(len(terms), 1))


def _topk_neighbors(center_beam: int, *, num_classes: int, k: int) -> tuple[int, ...]:
    beams = list(range(int(num_classes)))
    beams.sort(key=lambda item: (_circular_beam_distance(item, center_beam, num_classes=num_classes), item))
    return tuple(int(item) for item in beams[: max(1, min(int(k), int(num_classes)))])


def _circular_beam_distance(left: int, right: int, *, num_classes: int) -> int:
    diff = abs(int(left) % int(num_classes) - int(right) % int(num_classes))
    return int(min(diff, int(num_classes) - diff))


def _signed_circular_delta(predicted: int, truth: int, *, num_classes: int) -> int:
    value = (int(predicted) - int(truth)) % int(num_classes)
    if value > int(num_classes) // 2:
        value -= int(num_classes)
    return int(value)


def _unwrap_beams(values: np.ndarray, *, period: int) -> np.ndarray:
    if values.size <= 1:
        return values.astype(float)
    return np.unwrap(values.astype(float) * (2.0 * math.pi / float(period))) * float(period) / (2.0 * math.pi)


def _last_geometry(row: dict[str, Any]) -> dict[str, Any]:
    geometries = []
    idx = 1
    while f"geometry{idx}" in row:
        payload = _json_dict(row.get(f"geometry{idx}"))
        if payload:
            geometries.append(payload)
        idx += 1
    return geometries[-1] if geometries else {}


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None or value == "":
        return {}
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _xy_limits(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float]:
    x_values = np.concatenate([xs, np.asarray([0.0])])
    y_values = np.concatenate([ys, np.asarray([0.0])])
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    x_pad = max((x_max - x_min) * 0.08, 1.0)
    y_pad = max((y_max - y_min) * 0.08, 1.0)
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def _float_value(*values: Any) -> float | None:
    for value in values:
        try:
            if value is None or value == "":
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _int_value(*values: Any) -> int | None:
    for value in values:
        try:
            if value is None or value == "":
                continue
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return None


def _agent_from_sample_id(sample_id: str) -> str:
    parts = str(sample_id).split(":")
    return parts[-2] if len(parts) >= 2 else "unknown"


def _split_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _default_display_name(scenario: str) -> str:
    text = scenario
    if text.startswith("Town10_"):
        text = text[len("Town10_") :]
    for suffix in ("_seed24", "_seed42"):
        text = text.replace(suffix, "")
    return text.replace("_", " ")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize true-vs-predicted GPS beam trajectories for MMW Town scenes.")
    parser.add_argument("--prepared-root", type=Path, default=Path("dataset/MMW/sunny/Prepared"))
    parser.add_argument("--prediction-root", type=Path, default=Path("outputs/gps_coarse_anchor/target_adapt_beambench_dba"))
    parser.add_argument("--split-tag", default="l5p3_group_safe")
    parser.add_argument("--split", default="test")
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--display-names", default="")
    parser.add_argument("--num-classes", type=int, default=64)
    parser.add_argument("--prediction-column", default="predicted_beam")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/mmw_town_label_distribution/gps_prediction_trajectory/target_adapt_beambench"),
    )
    parser.add_argument("--spatial-figure-name", default="mmw_town_gps_prediction_spatial_trajectory.png")
    parser.add_argument("--sequence-figure-name", default="mmw_town_gps_prediction_beam_sequence.png")
    parser.add_argument("--summary-name", default="mmw_town_gps_prediction_trajectory_summary.json")
    parser.add_argument("--title", default="MMW Town Target-Adapt GPS Anchor")
    parser.add_argument("--spatial-label-mode", choices=("label", "unwrapped", "raw"), default="label")
    parser.add_argument("--sequence-label-mode", choices=("label", "unwrapped", "raw"), default="label")
    parser.add_argument(
        "--beam-label-calibration-json",
        help="Optional JSON calibration config; when enabled true/pred labels are mapped before plotting.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
