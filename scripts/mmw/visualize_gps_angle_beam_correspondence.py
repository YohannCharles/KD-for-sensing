#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
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
DEFAULT_SPLITS = ("train", "test")
DEFAULT_PREDICTION_ROOT = Path("outputs/analysis/mmw_town_gps_adapter_v2/mapping_enabled")
DBA_TOP_K = 3
DBA_DELTA = 5.0


@dataclass(frozen=True)
class SceneCalibration:
    scenario: str
    source_scenes: tuple[str, ...]
    target_seen_during_calibration: bool
    split_protocol: str
    beam_direction: int
    beam_offset: int
    boresight_angle_degrees: float
    support_fit_sample_count: int
    evaluation_sample_count: int
    dba_avg: float | None


@dataclass(frozen=True)
class SceneSeries:
    name: str
    scenario: str
    calibration: SceneCalibration
    x: np.ndarray
    y: np.ndarray
    calibrated_angle: np.ndarray
    centered_beam_index: np.ndarray
    true_beam: np.ndarray
    plotted_beam: np.ndarray
    dba_score: np.ndarray
    split: np.ndarray
    beam_source: str
    beam_index_mode: str
    beam_label_space: str
    beam_label_mapping_fingerprint: str
    sample_count: int
    num_classes: int


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    scenarios = _split_csv_arg(args.scenarios) or list(DEFAULT_SCENARIOS)
    splits = tuple(_split_csv_arg(args.splits) or list(DEFAULT_SPLITS))
    display_names = _split_csv_arg(args.display_names)
    if display_names and len(display_names) != len(scenarios):
        raise ValueError("--display-names must match --scenarios length.")

    calibrations = _load_calibrations(args.prediction_root / "summary.json")
    beam_label_calibration = json.loads(args.beam_label_calibration_json) if args.beam_label_calibration_json else None
    series = []
    for index, scenario in enumerate(scenarios):
        name = display_names[index] if display_names else _default_display_name(scenario)
        series.append(
            load_scene_series(
                prepared_root=args.prepared_root,
                split_tag=args.split_tag,
                scenario=scenario,
                name=name,
                splits=splits,
                num_classes=args.num_classes,
                calibration=calibrations.get(scenario),
                prediction_root=args.prediction_root,
                beam_source=args.beam_source,
                prediction_column=args.prediction_column,
                beam_index_mode=args.beam_index_mode,
                beam_label_calibration=beam_label_calibration,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / args.figure_name
    summary_path = args.output_dir / args.summary_name
    if args.plot_mode == "dba":
        plot_dba_map(
            series,
            output_path=figure_path,
            title=args.title,
            show_split_marker=not args.no_split_marker,
        )
    else:
        plot_correspondence(
            series,
            output_path=figure_path,
            title=args.title,
            beam_title=args.beam_title,
            show_split_marker=not args.no_split_marker,
            show_dba=not args.no_dba,
        )
    write_summary(series, summary_path)
    print(
        json.dumps(
            {
                "figure": str(figure_path),
                "summary": str(summary_path),
                "prediction_root": str(args.prediction_root),
                "scenarios": [item.scenario for item in series],
                "splits": list(splits),
            },
            indent=2,
        )
    )
    return 0


def load_scene_series(
    *,
    prepared_root: Path,
    split_tag: str,
    scenario: str,
    name: str,
    splits: Iterable[str],
    num_classes: int,
    calibration: SceneCalibration | None,
    prediction_root: Path,
    beam_source: str,
    prediction_column: str,
    beam_index_mode: str,
    beam_label_calibration: dict[str, object] | None = None,
) -> SceneSeries:
    calib = calibration or SceneCalibration(
        scenario=scenario,
        source_scenes=(),
        target_seen_during_calibration=False,
        split_protocol="unknown",
        beam_direction=1,
        beam_offset=0,
        boresight_angle_degrees=0.0,
        support_fit_sample_count=0,
        evaluation_sample_count=0,
        dba_avg=None,
    )
    if beam_label_calibration is not None:
        mapping = resolve_beam_label_mapping(
            beam_label_calibration,
            scene=scenario,
            default_num_classes=int(num_classes),
        )
    else:
        mapping = resolve_beam_label_mapping(
            {
                "enabled": _uses_calibrated_label_space(beam_index_mode),
                "label_space": "calibrated_gps_angle",
                "num_classes": int(num_classes),
                "direction": int(calib.beam_direction),
                "offset": int(calib.beam_offset),
                "fit_source": calib.split_protocol,
            },
            scene=scenario,
        )
    xs: list[float] = []
    ys: list[float] = []
    angles: list[float] = []
    centered_indices: list[int] = []
    labels: list[int] = []
    plotted_beams: list[int] = []
    dba_scores: list[float] = []
    split_names: list[str] = []
    prediction_lookup = _load_prediction_lookup(
        prediction_root / scenario / "predictions.csv",
        column=prediction_column,
    ) if beam_source == "prediction" else {}
    for split in splits:
        csv_path = prepared_root / scenario / "splits" / split_tag / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split CSV for {scenario}: {csv_path}")
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                geometry = _last_geometry(row)
                if not geometry:
                    continue
                x = _float_value(geometry.get("relative_x"), geometry.get("local_x"))
                y = _float_value(geometry.get("relative_y"), geometry.get("local_y"))
                raw_angle = _float_value(geometry.get("relative_azimuth"), None)
                if raw_angle is None and x is not None and y is not None:
                    raw_angle = math.degrees(math.atan2(float(y), float(x)))
                label = _int_value(row.get("future_beam_label1"), row.get("beam_label"))
                if x is None or y is None or raw_angle is None or label is None or label < 0:
                    continue
                beam_value = int(label)
                if beam_source == "prediction":
                    sample_id = str(row.get("target_sample_id") or row.get("sample_id") or "")
                    lookup_key = (sample_id, "target_test" if split == "test" else str(split))
                    predicted = prediction_lookup.get(lookup_key)
                    if predicted is None:
                        predicted = prediction_lookup.get((sample_id, str(split)))
                    if predicted is None:
                        continue
                    beam_value = int(predicted)
                truth_for_metric = mapping.map_label(int(label))
                beam_for_metric = mapping.map_label(int(beam_value))
                dba_score = (
                    _sample_dba_score(predicted=int(beam_for_metric), truth=int(truth_for_metric), num_classes=int(num_classes))
                    if beam_source == "prediction"
                    else float("nan")
                )
                calibrated_angle = _wrap_degrees(float(raw_angle) - float(calib.boresight_angle_degrees))
                centered_index = _centered_beam_index(
                    int(beam_value),
                    num_classes=int(num_classes),
                    offset=int(calib.beam_offset),
                    direction=int(calib.beam_direction),
                )
                xs.append(float(x))
                ys.append(float(y))
                angles.append(calibrated_angle)
                centered_indices.append(centered_index)
                labels.append(int(truth_for_metric) % int(num_classes))
                plotted_beams.append(int(beam_for_metric) % int(num_classes))
                dba_scores.append(float(dba_score))
                split_names.append(str(split))
    if not xs:
        raise ValueError(f"No plottable GPS/beam rows found for {scenario}.")
    angle_array = np.asarray(angles, dtype=float)
    raw_beam_array = np.asarray(plotted_beams, dtype=float)
    centered_array = np.asarray(centered_indices, dtype=float)
    if beam_index_mode == "raw_label":
        index_array = raw_beam_array
    elif beam_index_mode == "calibrated_label":
        index_array = raw_beam_array
    elif beam_index_mode == "centered":
        index_array = centered_array
    elif beam_index_mode == "unwrapped_centered":
        index_array = _unwrap_circular_index(centered_array, angle_array, period=int(num_classes))
    else:
        raise ValueError(f"Unsupported --beam-index-mode={beam_index_mode!r}.")
    return SceneSeries(
        name=name,
        scenario=scenario,
        calibration=calib,
        x=np.asarray(xs, dtype=float),
        y=np.asarray(ys, dtype=float),
        calibrated_angle=angle_array,
        centered_beam_index=index_array,
        true_beam=np.asarray(labels, dtype=int),
        plotted_beam=np.asarray(plotted_beams, dtype=int),
        dba_score=np.asarray(dba_scores, dtype=float),
        split=np.asarray(split_names),
        beam_source=str(beam_source),
        beam_index_mode=beam_index_mode,
        beam_label_space=mapping.label_space,
        beam_label_mapping_fingerprint=mapping.fingerprint,
        sample_count=len(xs),
        num_classes=int(num_classes),
    )


def plot_correspondence(
    series: Iterable[SceneSeries],
    *,
    output_path: Path,
    title: str,
    beam_title: str,
    show_split_marker: bool,
    show_dba: bool,
) -> None:
    items = list(series)
    if not items:
        raise ValueError("No scene series to plot.")
    angle_limit = _symmetric_percentile_limit(np.concatenate([item.calibrated_angle for item in items]), floor=10.0)
    index_values = np.concatenate([item.centered_beam_index for item in items])
    bounded_modes = {"raw_label", "calibrated_label"}
    if all(item.beam_index_mode in bounded_modes for item in items):
        index_vmin = 0.0
        index_vmax = float(max(int(item.num_classes) for item in items) - 1)
    else:
        index_limit = max(1.0, _symmetric_percentile_limit(index_values, floor=8.0))
        index_vmin = -index_limit
        index_vmax = index_limit
    rows = len(items)
    fig, axes = plt.subplots(rows, 2, figsize=(9.4, max(3.0, rows * 2.75)), constrained_layout=True)
    if rows == 1:
        axes = np.asarray([axes])
    angle_mappable = None
    index_mappable = None
    for row_idx, item in enumerate(items):
        ax_angle = axes[row_idx, 0]
        ax_index = axes[row_idx, 1]
        x_min, x_max, y_min, y_max = _xy_limits(item)
        angle_mappable = _scatter_scene(
            ax_angle,
            item,
            values=item.calibrated_angle,
            vmin=-angle_limit,
            vmax=angle_limit,
            show_split_marker=show_split_marker,
        )
        index_mappable = _scatter_scene(
            ax_index,
            item,
            values=item.centered_beam_index,
            vmin=index_vmin,
            vmax=index_vmax,
            show_split_marker=show_split_marker,
        )
        ax_angle.set_title(f"{item.name} Calibrated Angle")
        ax_index.set_title(f"{item.name} {beam_title}")
        for ax in (ax_angle, ax_index):
            ax.scatter([0.0], [0.0], marker="x", color="red", s=48, linewidths=2.2, zorder=4)
            ax.text(0.02, 0.94, "BS", transform=ax.transAxes, fontsize=10, va="top", ha="left")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.grid(True, color="#cfcfcf", linewidth=0.75)
        dba = item.calibration.dba_avg if show_dba else None
        if dba is not None:
            ax_index.text(
                0.98,
                0.98,
                f"DBA={dba:.3f}",
                transform=ax_index.transAxes,
                fontsize=8,
                va="top",
                ha="right",
                zorder=5,
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.82},
            )
    if title:
        fig.suptitle(title, fontsize=13)
    if angle_mappable is not None:
        cbar = fig.colorbar(angle_mappable, ax=axes[:, 0], shrink=0.82, fraction=0.035, pad=0.02)
        cbar.set_label("[deg]")
    if index_mappable is not None:
        cbar = fig.colorbar(index_mappable, ax=axes[:, 1], shrink=0.82, fraction=0.035, pad=0.02)
        index_label = _beam_index_colorbar_label(items)
        cbar.set_label(index_label)
    if show_split_marker:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                linestyle="None",
                label="train",
                markerfacecolor="#777777",
                markersize=5,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="#202020",
                linestyle="None",
                label="test",
                markerfacecolor="#777777",
                markersize=5,
            ),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.01))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dba_map(
    series: Iterable[SceneSeries],
    *,
    output_path: Path,
    title: str,
    show_split_marker: bool,
) -> None:
    items = list(series)
    if not items:
        raise ValueError("No scene series to plot.")
    cols = 2
    rows = int(math.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8.6, max(3.2, rows * 3.0)), constrained_layout=True)
    axes = np.asarray(axes).reshape(rows, cols)
    mappable = None
    for index, item in enumerate(items):
        ax = axes[index // cols, index % cols]
        x_min, x_max, y_min, y_max = _xy_limits(item)
        mappable = _scatter_scene(
            ax,
            item,
            values=item.dba_score,
            vmin=0.0,
            vmax=1.0,
            show_split_marker=show_split_marker,
        )
        mean_dba = float(np.nanmean(item.dba_score)) if item.dba_score.size else 0.0
        summary_dba = item.calibration.dba_avg
        suffix = f"DBA={summary_dba:.3f}" if summary_dba is not None else f"DBA={mean_dba:.3f}"
        source_text = _source_scenes_title(item.calibration.source_scenes)
        title_text = f"{item.name} GPS Prediction ({suffix})"
        if item.calibration.target_seen_during_calibration:
            title_text = f"{title_text}\ncalibration: target train/support"
        elif source_text:
            title_text = f"{title_text}\nsource: {source_text}"
        ax.set_title(title_text, fontsize=10)
        ax.scatter([0.0], [0.0], marker="x", color="red", s=48, linewidths=2.2, zorder=4)
        ax.text(0.02, 0.94, "BS", transform=ax.transAxes, fontsize=10, va="top", ha="left")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, color="#cfcfcf", linewidth=0.75)
    for index in range(len(items), rows * cols):
        axes[index // cols, index % cols].axis("off")
    if title:
        fig.suptitle(title, fontsize=13)
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.82, fraction=0.035, pad=0.02)
        cbar.set_label("DBA")
        cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if show_split_marker:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="#202020",
                linestyle="None",
                label="test",
                markerfacecolor="#777777",
                markersize=5,
            )
        ]
        fig.legend(handles=handles, loc="lower center", ncol=1, frameon=True, bbox_to_anchor=(0.5, -0.01))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(series: Iterable[SceneSeries], output_path: Path) -> None:
    payload = []
    for item in series:
        payload.append(
            {
                "name": item.name,
                "scenario": item.scenario,
                "sample_count": int(item.sample_count),
                "splits": {str(split): int((item.split == split).sum()) for split in sorted(set(item.split.tolist()))},
                "calibration": {
                    "beam_direction": int(item.calibration.beam_direction),
                    "beam_offset": int(item.calibration.beam_offset),
                    "boresight_angle_degrees": float(item.calibration.boresight_angle_degrees),
                    "support_fit_sample_count": int(item.calibration.support_fit_sample_count),
                    "evaluation_sample_count": int(item.calibration.evaluation_sample_count),
                    "dba_avg": item.calibration.dba_avg,
                },
                "x_range": [float(item.x.min()), float(item.x.max())],
                "y_range": [float(item.y.min()), float(item.y.max())],
                "calibrated_angle_range": [float(item.calibrated_angle.min()), float(item.calibrated_angle.max())],
                "beam_source": item.beam_source,
                "beam_index_mode": item.beam_index_mode,
                "beam_label_space": item.beam_label_space,
                "beam_label_mapping_fingerprint": item.beam_label_mapping_fingerprint,
                "source_scenes": list(item.calibration.source_scenes),
                "target_seen_during_calibration": bool(item.calibration.target_seen_during_calibration),
                "split_protocol": item.calibration.split_protocol,
                "plotted_beam_range": [int(item.plotted_beam.min()), int(item.plotted_beam.max())],
                "dba_score_mean": float(np.nanmean(item.dba_score)) if np.isfinite(item.dba_score).any() else None,
                "dba_score_range": [
                    float(np.nanmin(item.dba_score)) if np.isfinite(item.dba_score).any() else None,
                    float(np.nanmax(item.dba_score)) if np.isfinite(item.dba_score).any() else None,
                ],
                "centered_beam_index_range": [
                    float(item.centered_beam_index.min()),
                    float(item.centered_beam_index.max()),
                ],
            }
        )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _scatter_scene(
    ax: Any,
    item: SceneSeries,
    *,
    values: np.ndarray,
    vmin: float,
    vmax: float,
    show_split_marker: bool,
) -> Any:
    if not show_split_marker:
        return ax.scatter(
            item.x,
            item.y,
            c=values,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=13,
            linewidths=0.0,
            alpha=0.88,
        )
    mappable = None
    for split_name, edgecolor, linewidth, alpha, size in (
        ("train", "none", 0.0, 0.58, 12),
        ("test", "#1a1a1a", 0.25, 0.92, 16),
    ):
        mask = item.split == split_name
        if not mask.any():
            continue
        mappable = ax.scatter(
            item.x[mask],
            item.y[mask],
            c=values[mask],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=size,
            edgecolors=edgecolor,
            linewidths=linewidth,
            alpha=alpha,
        )
    if mappable is not None:
        return mappable
    return ax.scatter(item.x, item.y, c=values, cmap="viridis", vmin=vmin, vmax=vmax, s=13, linewidths=0.0)


def _beam_index_colorbar_label(items: Iterable[SceneSeries]) -> str:
    modes = {item.beam_index_mode for item in items}
    if modes == {"raw_label"}:
        return "raw beam label"
    if modes == {"calibrated_label"}:
        return "calibrated beam label"
    if modes == {"unwrapped_centered"}:
        return "unwrapped centered index"
    if modes == {"centered"}:
        return "centered index"
    return "beam index"


def _uses_calibrated_label_space(beam_index_mode: str) -> bool:
    return beam_index_mode in {"calibrated_label", "centered", "unwrapped_centered"}


def _source_scenes_title(source_scenes: tuple[str, ...]) -> str:
    if not source_scenes:
        return ""
    return ", ".join(_default_display_name(scene) for scene in source_scenes)


def _load_calibrations(path: Path) -> dict[str, SceneCalibration]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("scene_results", []) if isinstance(payload, dict) else []
    result: dict[str, SceneCalibration] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        scenario = str(row.get("target_scene") or "")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        if not scenario:
            continue
        result[scenario] = SceneCalibration(
            scenario=scenario,
            source_scenes=tuple(str(item) for item in row.get("source_scenes", []) if item),
            target_seen_during_calibration=bool(row.get("target_seen_during_calibration", False)),
            split_protocol=str(row.get("split_protocol") or ""),
            beam_direction=int(metadata.get("effective_beam_direction", 1)),
            beam_offset=int(metadata.get("effective_beam_offset", 0)),
            boresight_angle_degrees=float(metadata.get("effective_boresight_angle_degrees", 0.0)),
            support_fit_sample_count=int(row.get("support_fit_sample_count", 0) or 0),
            evaluation_sample_count=int(metrics.get("sample_count", 0) or 0),
            dba_avg=float(metrics["dba_avg"]) if metrics.get("dba_avg") is not None else None,
        )
    return result


def _load_prediction_lookup(path: Path, *, column: str) -> dict[tuple[str, str], int]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")
    result: dict[tuple[str, str], int] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if column not in (reader.fieldnames or []):
            raise ValueError(
                f"Prediction CSV {path} does not contain column '{column}'. "
                "Use --prediction-column anchor_center_beam only for coarse-anchor diagnostics."
            )
        for row in reader:
            sample_id = str(row.get("sample_id") or "")
            split = str(row.get("split") or "")
            value = _int_value(row.get(column))
            if sample_id and value is not None:
                result[(sample_id, split)] = int(value)
    return result


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


def _centered_beam_index(label: int, *, num_classes: int, offset: int, direction: int) -> int:
    value = (int(label) - int(offset)) % int(num_classes)
    half = int(num_classes) // 2
    if value > half:
        value -= int(num_classes)
    if int(direction) < 0:
        value = -value
    return int(value)


def _unwrap_circular_index(values: np.ndarray, angles: np.ndarray, *, period: int) -> np.ndarray:
    if values.size <= 1:
        return values.astype(float)
    order = np.argsort(angles, kind="mergesort")
    ordered = values[order].astype(float).copy()
    half = float(period) / 2.0
    for idx in range(1, len(ordered)):
        delta = ordered[idx] - ordered[idx - 1]
        if delta > half:
            ordered[idx:] -= float(period)
        elif delta < -half:
            ordered[idx:] += float(period)
    result = np.empty_like(ordered)
    result[order] = ordered
    return result


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


def _xy_limits(item: SceneSeries) -> tuple[float, float, float, float]:
    xs = np.concatenate([item.x, np.asarray([0.0])])
    ys = np.concatenate([item.y, np.asarray([0.0])])
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_pad = max((x_max - x_min) * 0.08, 1.0)
    y_pad = max((y_max - y_min) * 0.08, 1.0)
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def _symmetric_percentile_limit(values: np.ndarray, *, floor: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(floor)
    return float(max(np.nanpercentile(np.abs(finite), 98), floor))


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


def _wrap_degrees(value: float) -> float:
    return float((float(value) + 180.0) % 360.0 - 180.0)


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
    parser = argparse.ArgumentParser(description="Visualize BeamBench-style GPS angle and beam-index correspondence.")
    parser.add_argument("--prepared-root", type=Path, default=Path("dataset/MMW/sunny/Prepared"))
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--split-tag", default="l5p3_group_safe")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--display-names", default="")
    parser.add_argument("--num-classes", type=int, default=64)
    parser.add_argument("--beam-source", choices=("true_label", "prediction"), default="true_label")
    parser.add_argument("--plot-mode", choices=("correspondence", "dba"), default="correspondence")
    parser.add_argument("--prediction-column", default="predicted_beam")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/mmw_town_label_distribution"))
    parser.add_argument("--figure-name", default="mmw_town_gps_angle_beam_correspondence.png")
    parser.add_argument("--summary-name", default="mmw_town_gps_angle_beam_correspondence_summary.json")
    parser.add_argument("--title", default="MMW Town GPS Angle and Beam Index Correspondence")
    parser.add_argument("--beam-title", default="Beam Index")
    parser.add_argument(
        "--beam-label-calibration-json",
        help="Optional JSON calibration config used for calibrated/centered beam index modes.",
    )
    parser.add_argument("--no-dba", action="store_true")
    parser.add_argument("--no-split-marker", action="store_true")
    parser.add_argument(
        "--beam-index-mode",
        choices=("raw_label", "calibrated_label", "centered", "unwrapped_centered"),
        default="raw_label",
        help="Right-panel color values for correspondence plots.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
