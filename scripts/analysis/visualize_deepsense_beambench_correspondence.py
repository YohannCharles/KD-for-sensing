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


EARTH_RADIUS_M = 6_378_137.0
DEFAULT_SCENES = (31, 32, 33, 34)
PAPER_CENTERED_ANGLES_RAD = {31: -0.72, 32: -0.76, 33: 0.59, 34: -0.51}


@dataclass(frozen=True)
class DeepSenseSceneSeries:
    scene: int
    csv_path: Path
    sample_count: int
    x: np.ndarray
    y: np.ndarray
    bs_x: float
    bs_y: float
    calibrated_angle_deg: np.ndarray
    centered_beam_index: np.ndarray
    beam_index: np.ndarray
    angle_offset_rad: float
    coordinate_origin: tuple[float, float]
    angle_mode: str


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    scenes = tuple(_split_ints(args.scenes) or DEFAULT_SCENES)
    series = [
        load_scene_series(
            data_root=args.data_root,
            scene=scene,
            angle_mode=args.angle_mode,
            use_paper_offset=not args.no_paper_offset,
            beam_center_index=args.beam_center_index,
            max_samples=args.max_samples,
        )
        for scene in scenes
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / args.figure_name
    summary_path = args.output_dir / args.summary_name
    plot_correspondence(series, output_path=figure_path, title=args.title)
    write_summary(series, summary_path)
    if args.write_individual:
        individual_dir = args.output_dir / "per_scene"
        individual_dir.mkdir(parents=True, exist_ok=True)
        for item in series:
            plot_correspondence(
                [item],
                output_path=individual_dir / f"scenario{item.scene}_beambench_correspondence.png",
                title=f"Scene {item.scene} GPS Angle and Beam Index Correspondence",
            )
    print(
        json.dumps(
            {
                "figure": str(figure_path),
                "summary": str(summary_path),
                "scenes": list(scenes),
                "data_root": str(args.data_root),
            },
            indent=2,
        )
    )
    return 0


def load_scene_series(
    *,
    data_root: Path,
    scene: int,
    angle_mode: str,
    use_paper_offset: bool,
    beam_center_index: int,
    max_samples: int | None,
) -> DeepSenseSceneSeries:
    scene_root = data_root / f"scenario{int(scene)}"
    csv_path = scene_root / f"scenario{int(scene)}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"DeepSense scenario CSV not found: {csv_path}")

    raw_rows = _read_rows(csv_path, limit=max_samples)
    if not raw_rows:
        raise ValueError(f"No rows found in {csv_path}.")
    bs_latlon = _read_gps(scene_root / raw_rows[0]["unit1_loc"])
    origin = _plot_origin(scene_root, scene=scene, rows=raw_rows, bs_latlon=bs_latlon)
    bs_x, bs_y = _latlon_to_xy(bs_latlon, origin=origin)
    offset_rad = float(PAPER_CENTERED_ANGLES_RAD.get(int(scene), 0.0)) if use_paper_offset else 0.0

    xs: list[float] = []
    ys: list[float] = []
    angles: list[float] = []
    centered_beams: list[int] = []
    beams: list[int] = []
    for row in raw_rows:
        user_latlon = _read_gps(scene_root / row["unit2_loc"])
        user_x, user_y = _latlon_to_xy(user_latlon, origin=origin)
        rel_x = user_x - bs_x
        rel_y = user_y - bs_y
        angle = _calibrated_angle_deg(
            rel_x=rel_x,
            rel_y=rel_y,
            mode=angle_mode,
            offset_rad=offset_rad,
        )
        beam = int(float(row["unit1_beam"]))
        xs.append(user_x)
        ys.append(user_y)
        angles.append(angle)
        beams.append(beam)
        centered_beams.append(beam - int(beam_center_index))
    return DeepSenseSceneSeries(
        scene=int(scene),
        csv_path=csv_path,
        sample_count=len(xs),
        x=np.asarray(xs, dtype=float),
        y=np.asarray(ys, dtype=float),
        bs_x=float(bs_x),
        bs_y=float(bs_y),
        calibrated_angle_deg=np.asarray(angles, dtype=float),
        centered_beam_index=np.asarray(centered_beams, dtype=float),
        beam_index=np.asarray(beams, dtype=int),
        angle_offset_rad=offset_rad,
        coordinate_origin=origin,
        angle_mode=angle_mode,
    )


def plot_correspondence(
    series: Iterable[DeepSenseSceneSeries],
    *,
    output_path: Path,
    title: str,
) -> None:
    items = list(series)
    if not items:
        raise ValueError("No DeepSense series to plot.")
    rows = len(items)
    fig, axes = plt.subplots(rows, 2, figsize=(8.0, max(2.6, rows * 2.45)), constrained_layout=True)
    if rows == 1:
        axes = np.asarray([axes])
    angle_values = np.concatenate([item.calibrated_angle_deg for item in items])
    beam_values = np.concatenate([item.centered_beam_index for item in items])
    angle_limit = max(float(np.nanpercentile(np.abs(angle_values), 98)), 25.0)
    beam_limit = max(float(np.nanpercentile(np.abs(beam_values), 98)), 30.0)
    angle_mappable = None
    beam_mappable = None
    for idx, item in enumerate(items):
        ax_angle = axes[idx, 0]
        ax_beam = axes[idx, 1]
        x_min, x_max, y_min, y_max = _scene_limits(item)
        angle_mappable = ax_angle.scatter(
            item.x,
            item.y,
            c=item.calibrated_angle_deg,
            cmap="viridis",
            vmin=-angle_limit,
            vmax=angle_limit,
            s=8,
            linewidths=0.0,
            alpha=0.9,
        )
        beam_mappable = ax_beam.scatter(
            item.x,
            item.y,
            c=item.centered_beam_index,
            cmap="viridis",
            vmin=-beam_limit,
            vmax=beam_limit,
            s=8,
            linewidths=0.0,
            alpha=0.9,
        )
        ax_angle.set_title(f"Scene {item.scene} Angle")
        ax_beam.set_title(f"Scene {item.scene} Beam Index")
        for ax in (ax_angle, ax_beam):
            ax.scatter([item.bs_x], [item.bs_y], marker="x", color="red", s=38, linewidths=2.0, zorder=4)
            ax.text(item.bs_x + 0.9, item.bs_y + 1.2, "BS", fontsize=10, ha="left", va="bottom")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.grid(True, color="#c9c9c9", linewidth=0.75)
    if title:
        fig.suptitle(title, fontsize=13)
    if angle_mappable is not None:
        cbar = fig.colorbar(angle_mappable, ax=axes[:, 0], shrink=0.78, fraction=0.04, pad=0.02)
        cbar.set_label("[deg]")
    if beam_mappable is not None:
        cbar = fig.colorbar(beam_mappable, ax=axes[:, 1], shrink=0.78, fraction=0.04, pad=0.02)
        cbar.set_label("index")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(series: Iterable[DeepSenseSceneSeries], output_path: Path) -> None:
    payload = []
    for item in series:
        payload.append(
            {
                "scene": int(item.scene),
                "csv_path": str(item.csv_path),
                "sample_count": int(item.sample_count),
                "angle_mode": item.angle_mode,
                "angle_offset_rad": float(item.angle_offset_rad),
                "angle_offset_degrees": float(math.degrees(item.angle_offset_rad)),
                "coordinate_origin_lat_lon": [float(item.coordinate_origin[0]), float(item.coordinate_origin[1])],
                "bs_xy_m": [float(item.bs_x), float(item.bs_y)],
                "x_range_m": [float(item.x.min()), float(item.x.max())],
                "y_range_m": [float(item.y.min()), float(item.y.max())],
                "calibrated_angle_range_degrees": [
                    float(item.calibrated_angle_deg.min()),
                    float(item.calibrated_angle_deg.max()),
                ],
                "beam_index_range": [int(item.beam_index.min()), int(item.beam_index.max())],
                "centered_beam_index_range": [
                    float(item.centered_beam_index.min()),
                    float(item.centered_beam_index.max()),
                ],
            }
        )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_rows(path: Path, *, limit: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def _read_gps(path: Path) -> tuple[float, float]:
    values = [float(item.strip()) for item in path.read_text(encoding="utf-8").split() if item.strip()]
    if len(values) < 2:
        raise ValueError(f"GPS file must contain latitude and longitude: {path}")
    return float(values[0]), float(values[1])


def _plot_origin(
    scene_root: Path,
    *,
    scene: int,
    rows: list[dict[str, str]],
    bs_latlon: tuple[float, float],
) -> tuple[float, float]:
    zoom_path = scene_root / "resources" / f"scen_{int(scene)}_zoom.txt"
    if zoom_path.exists():
        line = zoom_path.read_text(encoding="utf-8").strip().splitlines()[0]
        lat_text, lon_text = [item.strip() for item in line.split(",", 1)]
        return float(lat_text), float(lon_text)
    user_points = [_read_gps(scene_root / row["unit2_loc"]) for row in rows]
    lat_min = min([bs_latlon[0], *(item[0] for item in user_points)])
    lon_min = min([bs_latlon[1], *(item[1] for item in user_points)])
    return float(lat_min), float(lon_min)


def _latlon_to_xy(latlon: tuple[float, float], *, origin: tuple[float, float]) -> tuple[float, float]:
    lat, lon = latlon
    lat0, lon0 = origin
    x = math.radians(float(lon) - float(lon0)) * EARTH_RADIUS_M * math.cos(math.radians(float(lat0)))
    y = math.radians(float(lat) - float(lat0)) * EARTH_RADIUS_M
    return float(x), float(y)


def _calibrated_angle_deg(*, rel_x: float, rel_y: float, mode: str, offset_rad: float) -> float:
    if mode == "mathematical":
        raw = math.degrees(math.atan2(float(rel_y), float(rel_x)))
    elif mode == "camera_lateral":
        # BeamBench-style visual angle: 0 deg points forward from the BS along the vehicle road,
        # positive values are left-looking beams and negative values are right-looking beams.
        raw = -math.degrees(math.atan2(float(rel_x), -float(rel_y)))
    else:
        raise ValueError(f"Unsupported angle mode: {mode}")
    return _wrap_degrees(raw - math.degrees(float(offset_rad)))


def _scene_limits(item: DeepSenseSceneSeries) -> tuple[float, float, float, float]:
    xs = np.concatenate([item.x, np.asarray([item.bs_x])])
    ys = np.concatenate([item.y, np.asarray([item.bs_y])])
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_pad = max((x_max - x_min) * 0.06, 1.0)
    y_pad = max((y_max - y_min) * 0.06, 1.0)
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def _wrap_degrees(value: float) -> float:
    return float((float(value) + 180.0) % 360.0 - 180.0)


def _split_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value or "").split(",") if item.strip()]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize DeepSense6G BeamBench-style GPS angle/beam correspondence.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset/DeepSense6G"))
    parser.add_argument("--scenes", default=",".join(str(item) for item in DEFAULT_SCENES))
    parser.add_argument("--angle-mode", choices=("camera_lateral", "mathematical"), default="camera_lateral")
    parser.add_argument("--no-paper-offset", action="store_true")
    parser.add_argument("--beam-center-index", type=int, default=31)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/deepsense_beambench_correspondence"))
    parser.add_argument("--figure-name", default="deepsense_s31_s34_beambench_correspondence.png")
    parser.add_argument("--summary-name", default="deepsense_s31_s34_beambench_correspondence_summary.json")
    parser.add_argument("--title", default="DeepSense6G GPS Angle and Beam Index Correspondence")
    parser.add_argument("--write-individual", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
