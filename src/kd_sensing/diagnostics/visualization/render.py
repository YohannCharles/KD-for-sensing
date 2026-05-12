from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from PIL import Image
import torch

from kd_sensing.config.io import dump_config
from kd_sensing.data.samples import _select_portion
from kd_sensing.data.scenes import retarget_deepsense_dataset_config
from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.engine.data_factory import build_dataset, prepare_lidar_normalizer
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.engine.run_metadata import dataset_run_metadata
from kd_sensing.modalities import MODALITY_ORDER, dataset_flags_for_modalities, normalize_modalities
from kd_sensing.utils.paths import resolve_path


from kd_sensing.diagnostics.visualization.config import VisualizationConfig, _json_scalar
from kd_sensing.diagnostics.visualization.sampling import SampleCandidate

def build_sample_record(
    dataset: Any,
    *,
    split: str,
    row: pd.Series,
    candidate: SampleCandidate,
    sample: dict[str, Any],
    statistics: dict[str, Any],
    modalities: tuple[str, ...],
    raw_image_reference_enabled: bool = False,
) -> dict[str, Any]:
    input_beam = _tensor_list(sample.get("input_beam"))
    target_beam = _tensor_list(sample.get("target_beam"))
    paths = {
        "camera": _row_paths(row, "camera"),
        "radar": _row_paths(row, "radar"),
        "gps": _row_paths(row, "gps"),
        "bs_gps": _row_paths(row, "bs_gps"),
        "lidar": _row_paths(row, "lidar"),
        "mmwave": _row_paths(row, "mmwave"),
        "beam": _row_paths(row, "beam"),
        "future_beam": _row_paths(row, "future_beam"),
    }
    return {
        "split": split,
        "scene_id": getattr(dataset, "scene_id", None),
        "scene_slug": getattr(dataset, "scene_slug", None),
        "dataset_index": int(candidate.dataset_index),
        "csv_row_index": int(candidate.csv_row_index),
        "seq_index": candidate.seq_index,
        "future_label": candidate.future_label,
        "input_beam": input_beam,
        "target_beam": target_beam,
        "data_root": str(getattr(dataset, "data_root", "")),
        "csv_path": str(getattr(dataset, "root_csv", "")),
        "paths": paths,
        "enabled_modalities": list(modalities),
        "statistics": statistics,
        "raw_image_reference": {
            "enabled": bool(raw_image_reference_enabled and paths["camera"]),
            "reference_only": bool(raw_image_reference_enabled and paths["camera"]),
        },
    }

def sample_png_path(output_dir: Path, dataset: Any, split: str, candidate: SampleCandidate) -> Path:
    scene_slug = str(getattr(dataset, "scene_slug", "scene"))
    seq_token = _safe_token(candidate.seq_index)
    filename = f"{scene_slug}_{split}_idx{candidate.dataset_index:06d}_seq{seq_token}.png"
    return output_dir / scene_slug / split / filename

def render_sample_overview(
    dataset: Any,
    sample: dict[str, Any],
    record: dict[str, Any],
    path: Path,
    viz: VisualizationConfig,
) -> None:
    panels = _panel_names(sample)
    cols = max(2, int(viz.max_frames_per_sample))
    rows = max(1, len(panels))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.8 * cols, 3.0 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for row_idx, panel in enumerate(panels):
        row_axes = list(axes[row_idx])
        for ax in row_axes:
            ax.axis("off")
        if panel == "image":
            _draw_image_panel(dataset, sample["image"], record, row_axes, viz)
        elif panel == "radar_ra":
            _draw_temporal_heatmaps(sample["radar_ra"], row_axes, "RA")
        elif panel == "radar_da":
            _draw_temporal_heatmaps(sample["radar_da"], row_axes, "DA")
        elif panel == "lidar":
            _draw_lidar_panel(sample["lidar"], row_axes, record)
        elif panel == "gps":
            _draw_gps_panel(sample["gps"], row_axes)
        elif panel == "mmwave":
            _draw_mmwave_panel(sample["mmwave"], row_axes)
        elif panel == "label":
            _draw_label_panel(record, row_axes)

    fig.suptitle(
        f"{record.get('scene_slug')} {record['split']} idx={record['dataset_index']} "
        f"seq={record.get('seq_index')} label={record.get('future_label')}",
        fontsize=12,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)

def _sorted_numbered_columns(columns: Iterable[str], prefix: str) -> list[str]:
    selected = []
    for col in columns:
        if not str(col).startswith(prefix):
            continue
        suffix = str(col)[len(prefix) :]
        if suffix.isdigit():
            selected.append(str(col))
    return sorted(selected, key=lambda name: int(name[len(prefix) :]))

def _row_paths(row: pd.Series, prefix: str) -> list[str]:
    return [str(row[col]) for col in _sorted_numbered_columns(row.index, prefix) if str(row[col]).strip() != "-99"]

def _tensor_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return _as_numpy(value).reshape(-1).astype(int).tolist()

def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)

def _panel_names(sample: dict[str, Any]) -> list[str]:
    panels = []
    if "image" in sample:
        panels.append("image")
    if "radar_ra" in sample:
        panels.append("radar_ra")
    if "radar_da" in sample:
        panels.append("radar_da")
    if "lidar" in sample:
        panels.append("lidar")
    if "gps" in sample:
        panels.append("gps")
    if "mmwave" in sample:
        panels.append("mmwave")
    panels.append("label")
    return panels

def _draw_image_panel(dataset: Any, image: Any, record: dict[str, Any], axes: list[Any], viz: VisualizationConfig) -> None:
    array = _as_numpy(image)
    frames = _last_indices(array.shape[0], len(axes) - 1 if viz.include_raw_image_preview and len(axes) > 1 else len(axes))
    for ax, frame_idx in zip(axes, frames):
        ax.imshow(array[frame_idx], cmap="gray")
        ax.set_title(f"mask t{frame_idx}")
        ax.axis("off")
    if viz.include_raw_image_preview and axes:
        raw_paths = record.get("paths", {}).get("camera", [])
        if raw_paths:
            ax = axes[-1]
            try:
                raw = Image.open(joined_resource(getattr(dataset, "data_root"), raw_paths[-1])).convert("RGB")
                ax.imshow(raw)
                ax.set_title("raw ref")
            except Exception as exc:  # pragma: no cover - visual diagnostic fallback
                ax.text(0.02, 0.95, f"raw ref unavailable:\n{exc}", va="top", fontsize=8)
            ax.axis("off")

def _draw_temporal_heatmaps(value: Any, axes: list[Any], title: str) -> None:
    array = _as_numpy(value)
    frame_indices = _last_indices(array.shape[0], len(axes))
    frames = [array[frame_idx] for frame_idx in frame_indices]
    finite = np.concatenate([frame[np.isfinite(frame)].reshape(-1) for frame in frames]) if frames else np.asarray([])
    vmin = float(np.min(finite)) if finite.size else None
    vmax = float(np.max(finite)) if finite.size else None
    image = None
    used_axes = []
    for ax, frame_idx in zip(axes, frame_indices):
        image = ax.imshow(array[frame_idx], cmap="magma", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title} t{frame_idx}")
        ax.axis("off")
        used_axes.append(ax)
    if image is not None and used_axes:
        used_axes[0].figure.colorbar(image, ax=used_axes, shrink=0.72, fraction=0.035, pad=0.02)

def _draw_lidar_panel(value: Any, axes: list[Any], record: dict[str, Any]) -> None:
    array = _as_numpy(value)
    frame = array[-1] if array.ndim == 4 else array
    if frame.ndim != 3:
        axes[0].text(0.02, 0.95, f"Unexpected LiDAR shape {array.shape}", va="top")
        return
    lidar_stats = record.get("statistics", {}).get("lidar", {})
    nonzero = lidar_stats.get("nonzero_fraction")
    channel_nonzero = lidar_stats.get("channel_nonzero_fraction", [])
    composite = _normalize_channels(frame)
    axes[0].imshow(composite)
    if nonzero is None:
        axes[0].set_title("LiDAR BEV")
    else:
        axes[0].set_title(f"LiDAR BEV nz={float(nonzero):.3f}")
    axes[0].axis("off")
    for channel_idx, ax in enumerate(axes[1:4], start=0):
        if channel_idx >= frame.shape[0]:
            break
        ax.imshow(frame[channel_idx], cmap="viridis")
        nz_text = ""
        if channel_idx < len(channel_nonzero):
            nz_text = f" nz={float(channel_nonzero[channel_idx]):.3f}"
        ax.set_title(f"L{channel_idx}{nz_text}")
        ax.axis("off")

def _draw_gps_panel(value: Any, axes: list[Any]) -> None:
    array = _as_numpy(value)
    ax = axes[0]
    if array.ndim == 2:
        for dim in range(array.shape[1]):
            ax.plot(array[:, dim], marker="o", linewidth=1.3, label=f"d{dim}")
        ax.legend(fontsize=7)
    else:
        ax.plot(array.reshape(-1), marker="o", linewidth=1.3)
    ax.set_title("GPS relative-polar features")
    ax.grid(True, alpha=0.25)

def _draw_mmwave_panel(value: Any, axes: list[Any]) -> None:
    array = _as_numpy(value)
    ax = axes[0]
    ax.imshow(array, cmap="plasma", aspect="auto")
    ax.set_title("mmWave time x beam receive power")
    ax.set_xlabel("beam index")
    ax.set_ylabel("time")

def _draw_label_panel(record: dict[str, Any], axes: list[Any]) -> None:
    ax = axes[0]
    input_beam = record.get("input_beam", [])
    target_beam = record.get("target_beam", [])
    text = "\n".join(
        [
            f"input_beam: {input_beam}",
            f"target_beam: {target_beam}",
            f"future label: {record.get('future_label')}",
            f"csv row: {record.get('csv_row_index')}",
            f"csv: {record.get('csv_path')}",
        ]
    )
    ax.text(0.02, 0.95, text, va="top", ha="left", family="monospace", fontsize=8)
    ax.set_title("beam labels and source")
    ax.axis("off")

def _last_indices(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        return []
    count = min(int(count), int(total))
    return list(range(total - count, total))

def _normalize_channels(frame: np.ndarray) -> np.ndarray:
    channels = []
    for idx in range(min(3, frame.shape[0])):
        channel = frame[idx].astype(np.float32)
        channels.append(_normalize_image(channel))
    while len(channels) < 3:
        channels.append(np.zeros_like(channels[0] if channels else frame[0], dtype=np.float32))
    return np.stack(channels, axis=-1)

def _normalize_image(array: np.ndarray) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    if math.isclose(max_value, min_value):
        return np.zeros_like(array, dtype=np.float32)
    return ((array - min_value) / (max_value - min_value)).astype(np.float32)

def _safe_token(value: Any) -> str:
    token = str(_json_scalar(value)).replace("/", "_").replace("\\", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in token)[:80] or "none"

__all__ = [
    'build_sample_record',
    'render_sample_overview',
    'sample_png_path',
]
