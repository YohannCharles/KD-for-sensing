from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image

try:  # Plotly is an optional viewer dependency.
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots as _make_subplots
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent
    _make_subplots = None

    class _FallbackTrace(dict):
        def __init__(self, trace_type: str, **kwargs: Any) -> None:
            super().__init__(type=trace_type, **kwargs)

    class _FallbackFigure:
        def __init__(self, trace: Any | None = None) -> None:
            self.data: list[Any] = []
            self.layout: dict[str, Any] = {}
            if trace is not None:
                self.data.append(trace)

        def add_trace(self, trace: Any) -> None:
            self.data.append(trace)

        def update_layout(self, *args: Any, **kwargs: Any) -> None:
            if args and isinstance(args[0], dict):
                self.layout.update(args[0])
            self.layout.update(kwargs)

        def update_xaxes(self, **kwargs: Any) -> None:
            self.layout.setdefault("xaxis", {}).update(kwargs)

        def update_yaxes(self, **kwargs: Any) -> None:
            self.layout.setdefault("yaxis", {}).update(kwargs)

        def to_dict(self) -> dict[str, Any]:
            return {"data": self.data, "layout": self.layout}

    class _FallbackGraphObjects:
        Figure = _FallbackFigure

        @staticmethod
        def Scatter(**kwargs: Any) -> _FallbackTrace:
            return _FallbackTrace("scatter", **kwargs)

        @staticmethod
        def Bar(**kwargs: Any) -> _FallbackTrace:
            return _FallbackTrace("bar", **kwargs)

        @staticmethod
        def Heatmap(**kwargs: Any) -> _FallbackTrace:
            return _FallbackTrace("heatmap", **kwargs)

        @staticmethod
        def Image(**kwargs: Any) -> _FallbackTrace:
            return _FallbackTrace("image", **kwargs)

    go = _FallbackGraphObjects()


MODALITIES = ("image", "lidar", "radar", "gps", "mmwave")
DISTRIBUTION_MODALITIES = (*MODALITIES, "fusion")
SHOW_MODES = ("all", "correct only", "wrong only", "low quality only")
LOW_QUALITY_MODALITY_THRESHOLD = 0.4
LOW_QUALITY_MEAN_THRESHOLD = 0.5


def load_manifest(manifest_path: str | Path, project_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Load a viewer manifest from a JSON array, {"samples": [...]}, or JSONL file."""

    path = Path(manifest_path).expanduser()
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    samples: list[dict[str, Any]]
    first = text[0]
    if first in "[{":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
            samples = [item for item in payload["samples"] if isinstance(item, dict)]
        elif isinstance(payload, list):
            samples = [item for item in payload if isinstance(item, dict)]
        else:
            samples = _load_jsonl(text)
    else:
        samples = _load_jsonl(text)

    manifest_dir = path.parent
    root = Path(project_root).expanduser() if project_root is not None else _find_project_root(manifest_dir)
    for index, sample in enumerate(samples):
        sample.setdefault("_manifest_index", index)
        sample.setdefault("_global_index", index)
        sample.setdefault("_manifest_dir", str(manifest_dir))
        if root is not None:
            sample.setdefault("_project_root", str(root))
    return samples


def get_available_scenes(samples: Iterable[dict[str, Any]]) -> list[str]:
    values = []
    for sample in samples:
        scene = sample.get("scene_slug", sample.get("scene_id"))
        if scene is not None and str(scene).strip():
            values.append(str(scene))
    unique = sorted(set(values), key=_natural_key)
    return ["all", *unique] if unique else ["all"]


def get_available_splits(samples: Iterable[dict[str, Any]]) -> list[str]:
    values = [str(sample.get("split")) for sample in samples if sample.get("split") is not None]
    unique = sorted({value for value in values if value.strip()}, key=_natural_key)
    return ["all", *unique] if unique else ["all"]


def filter_samples(
    samples: Iterable[dict[str, Any]],
    scene: str | None = "all",
    split: str | None = "all",
    show_mode: str | None = "all",
) -> list[dict[str, Any]]:
    scene_filter = _none_if_all(scene)
    split_filter = _none_if_all(split)
    mode = str(show_mode or "all").strip().lower()
    if mode not in SHOW_MODES:
        mode = "all"

    filtered = []
    for sample in samples:
        if scene_filter is not None and not _sample_matches_scene(sample, scene_filter):
            continue
        if split_filter is not None and str(sample.get("split")) != split_filter:
            continue
        if mode == "correct only" and safe_get(sample, "prediction.correct") is not True:
            continue
        if mode == "wrong only" and safe_get(sample, "prediction.correct") is not False:
            continue
        if mode == "low quality only" and not _is_low_quality(sample.get("quality")):
            continue
        filtered.append(sample)
    return filtered


def safe_get(data: Any, path: str, default: Any = None) -> Any:
    current = data
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
            continue
        if isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return default
            current = current[index]
            continue
        return default
    return current


def resolve_path(
    path: str | Path | None,
    manifest_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path | None:
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None

    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate

    bases = []
    if manifest_dir is not None:
        bases.append(Path(manifest_dir).expanduser())
    if project_root is not None:
        bases.append(Path(project_root).expanduser())
    for base in bases:
        resolved = base / candidate
        if resolved.exists():
            return resolved
    if bases:
        return bases[0] / candidate
    return candidate


def load_image_safe(
    path: str | Path | None,
    manifest_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Image.Image | None:
    resolved = resolve_path(path, manifest_dir=manifest_dir, project_root=project_root)
    if resolved is None or not resolved.exists() or not resolved.is_file():
        return None
    try:
        with Image.open(resolved) as image:
            return image.convert("RGB").copy()
    except Exception:
        return None


def load_json_safe(
    path: str | Path | dict[str, Any] | list[Any] | None,
    manifest_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Any:
    if isinstance(path, (dict, list)):
        return path
    resolved = resolve_path(path, manifest_dir=manifest_dir, project_root=project_root)
    if resolved is None or not resolved.exists() or not resolved.is_file():
        return None
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        return None


def make_empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": "Missing / Not Available",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 14},
            }
        ],
        margin={"l": 24, "r": 24, "t": 48, "b": 24},
        height=300,
    )
    return fig


def make_image_figure(
    image_path: str | Path | None,
    title: str,
    manifest_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> go.Figure:
    image = load_image_safe(image_path, manifest_dir=manifest_dir, project_root=project_root)
    if image is None:
        return make_empty_figure(title)
    fig = go.Figure(go.Image(z=np.asarray(image)))
    fig.update_layout(title=title, margin={"l": 0, "r": 0, "t": 48, "b": 0}, height=320)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    return fig


def make_gps_figure(
    gps_data: str | Path | dict[str, Any] | list[Any] | None,
    title: str,
    manifest_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> go.Figure:
    data = load_json_safe(gps_data, manifest_dir=manifest_dir, project_root=project_root)
    feature_fig = _gps_feature_figure(data, title)
    if feature_fig is not None:
        return feature_fig
    xy = _gps_xy(data)
    if xy is None:
        return make_empty_figure(title)
    x_values, y_values, x_name, y_name = xy
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            name="trajectory",
            line={"width": 2},
        )
    )
    if x_values and y_values:
        fig.add_trace(
            go.Scatter(
                x=[x_values[-1]],
                y=[y_values[-1]],
                mode="markers",
                name="current",
                marker={"size": 11, "color": "#d62728"},
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=x_name,
        yaxis_title=y_name,
        margin={"l": 48, "r": 16, "t": 48, "b": 48},
        height=300,
        showlegend=True,
    )
    return fig


def make_mmwave_figure(
    mmwave_data: str | Path | dict[str, Any] | list[Any] | None,
    title: str,
    manifest_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> go.Figure:
    if _looks_like_image_path(mmwave_data):
        return make_image_figure(mmwave_data, title, manifest_dir=manifest_dir, project_root=project_root)

    data = load_json_safe(mmwave_data, manifest_dir=manifest_dir, project_root=project_root)
    if isinstance(data, dict):
        title = _title_with_scale(title, data)
        colorbar_title = str(data.get("units") or data.get("scale") or "power")
        if "beam_power_seq" in data:
            array = _numeric_array(data.get("beam_power_seq"))
            if array is None or array.ndim != 2:
                return make_empty_figure(title)
            fig = go.Figure(go.Heatmap(z=array, colorscale="Viridis", colorbar={"title": colorbar_title}))
            fig.update_layout(
                title=title,
                xaxis_title="beam index",
                yaxis_title="time index",
                margin={"l": 56, "r": 16, "t": 48, "b": 48},
                height=320,
            )
            return fig
        if "beam_power" in data:
            return _beam_power_bar(data.get("beam_power"), title, y_title=colorbar_title)
    if isinstance(data, list):
        array = _numeric_array(data)
        if array is None:
            return make_empty_figure(title)
        if array.ndim == 1:
            return _beam_power_bar(array, title)
        if array.ndim == 2:
            fig = go.Figure(go.Heatmap(z=array, colorscale="Viridis", colorbar={"title": "power"}))
            fig.update_layout(title=title, xaxis_title="beam index", yaxis_title="time index", height=320)
            return fig
    return make_empty_figure(title)


def make_score_bar(
    score_dict: dict[str, Any] | None,
    title: str,
    y_range: list[float] | tuple[float, float] = (0.0, 1.0),
) -> go.Figure:
    scores = _numeric_score_items(score_dict)
    if not scores:
        return make_empty_figure(title)
    labels = [item[0] for item in scores]
    values = [item[1] for item in scores]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#4c78a8"))
    fig.update_layout(
        title=title,
        yaxis={"range": list(y_range), "title": "score"},
        xaxis={"title": "modality"},
        margin={"l": 48, "r": 16, "t": 48, "b": 48},
        height=300,
    )
    return fig


def make_single_modality_confidence_figure(
    sample: dict[str, Any] | None,
    title: str = "Single-Modality Confidence (t+1)",
) -> go.Figure:
    return make_score_bar(single_modality_t1_confidence(sample), title)


def single_modality_confidence_dataframe(sample: dict[str, Any] | None) -> pd.DataFrame:
    return dict_to_dataframe(single_modality_t1_confidence(sample), "confidence")


def single_modality_t1_confidence(sample: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(sample, dict):
        return {}
    scores: dict[str, float] = {}
    for modality in DISTRIBUTION_MODALITIES:
        value = _t1_confidence_for_modality(sample, modality)
        if value is not None:
            scores[modality] = value
    for modality in _extra_confidence_modalities(sample):
        if modality in scores:
            continue
        value = _t1_confidence_for_modality(sample, modality)
        if value is not None:
            scores[modality] = value
    return scores


def make_beam_confidence_figure(sample: dict[str, Any] | None, title: str = "Future Beam Label Confidence") -> go.Figure:
    curves = _beam_confidence_curves(sample or {})
    future_labels = _beam_label_values(safe_get(sample, "label.future_beams") if sample else None)
    current_label = _beam_label_values([safe_get(sample, "label.current_beam")] if sample else None)
    if not curves and not future_labels and not current_label:
        return make_empty_figure(title)

    fig = go.Figure()
    max_y = 1.0
    max_x = 0
    for name, x_values, y_values in curves:
        if not x_values or not y_values:
            continue
        max_x = max(max_x, int(max(x_values)))
        max_y = max(max_y, float(max(y_values)))
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=name,
                line={"width": 2},
            )
        )

    shapes: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for index, label in enumerate(future_labels[:1]):
        max_x = max(max_x, label)
        shapes.append(
            {
                "type": "line",
                "x0": label,
                "x1": label,
                "xref": "x",
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#d62728", "width": 2, "dash": "dash"},
            }
        )
        annotations.append(
            {
                "text": f"future t+{index + 1}: {label}",
                "x": label,
                "xref": "x",
                "y": 1.0,
                "yref": "paper",
                "showarrow": False,
                "yshift": 12,
                "font": {"size": 11, "color": "#d62728"},
            }
        )
    for label in current_label[:1]:
        max_x = max(max_x, label)
        shapes.append(
            {
                "type": "line",
                "x0": label,
                "x1": label,
                "xref": "x",
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#666666", "width": 1, "dash": "dot"},
            }
        )

    fig.update_layout(
        title=title,
        xaxis={"title": "beam label", "range": [0, max(63, max_x)]},
        yaxis={"title": "confidence", "range": [0.0, min(1.05, max(1.0, max_y * 1.05))]},
        margin={"l": 56, "r": 24, "t": 64, "b": 48},
        height=360,
        showlegend=True,
        shapes=shapes,
        annotations=annotations,
    )
    if not curves:
        fig.update_layout(
            annotations=[
                *annotations,
                {
                    "text": "No per-label confidence curves in manifest",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 14},
                },
            ]
        )
    return fig


def make_beam_index_trend_figure(
    samples: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    current_sample: dict[str, Any] | None,
    *,
    radius: int = 30,
    title: str = "Future Beam Index Trend (+/-30)",
) -> go.Figure:
    if not current_sample:
        return make_empty_figure(title)

    group = [
        sample
        for sample in samples
        if _same_sequence(sample, current_sample)
        and _first_future_beam(sample) is not None
    ]
    if not group:
        return make_empty_figure(title)
    group = sorted(group, key=_sample_time_sort_key)
    current_position = _current_sample_position(group, current_sample)
    if current_position is None:
        return make_empty_figure(title)

    radius = max(0, int(radius))
    start = max(0, current_position - radius)
    end = min(len(group), current_position + radius + 1)
    window = group[start:end]
    x_values = [index - current_position for index in range(start, end)]
    y_values = [int(_first_future_beam(sample)) for sample in window]
    current_offset = current_position - start
    current_y = y_values[current_offset]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            name="future t+1 beam",
            line={"width": 2, "color": "#4c78a8"},
            marker={"size": 6, "color": "#4c78a8"},
            customdata=[_display_value(sample.get("time_index", sample.get("_manifest_index"))) for sample in window],
            hovertemplate="offset=%{x}<br>time_index=%{customdata}<br>beam=%{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[current_y],
            mode="markers",
            name="current sample",
            marker={"size": 13, "color": "#d62728", "symbol": "circle"},
            hovertemplate="current sample<br>future t+1 beam=%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="relative sample offset",
        yaxis_title="beam index",
        margin={"l": 56, "r": 24, "t": 56, "b": 48},
        height=320,
        showlegend=True,
        shapes=[
            {
                "type": "line",
                "x0": 0,
                "x1": 0,
                "xref": "x",
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#d62728", "width": 1, "dash": "dash"},
            }
        ],
    )
    return fig


def get_future_beams(sample: dict[str, Any] | None) -> list[int]:
    return _beam_label_values(safe_get(sample, "label.future_beams") if sample else None)


def get_horizon_choices(sample: dict[str, Any] | None) -> list[str]:
    return [f"t+{index + 1}" for index, _ in enumerate(get_future_beams(sample))]


def parse_horizon_label(horizon_label: str | None) -> int:
    text = str(horizon_label or "").strip().lower().replace(" ", "")
    if not text:
        return 0
    if text.startswith("t+"):
        text = text[2:]
    elif text.startswith("+"):
        text = text[1:]
    try:
        value = int(text)
    except ValueError:
        return 0
    return max(0, value - 1)


def extract_beam_distribution(sample: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(sample, dict):
        return {}
    for source in (sample.get("beam_distribution"), safe_get(sample, "prediction.beam_distribution")):
        if isinstance(source, dict):
            return source
    return {}


def get_distribution_for_modality(
    sample: dict[str, Any] | None,
    modality: str,
    horizon_index: int,
    view_type: str,
) -> list[float] | None:
    distribution = extract_beam_distribution(sample)
    entry = distribution.get(modality)
    values = _distribution_values(entry, _view_distribution_key(view_type), horizon_index)
    if values is None:
        return None
    return values.astype(float).tolist()


def get_probability_for_modality(
    sample: dict[str, Any] | None,
    modality: str,
    horizon_index: int,
) -> list[float] | None:
    distribution = extract_beam_distribution(sample)
    values = _distribution_values(distribution.get(modality), "prob", horizon_index)
    if values is None:
        return None
    return values.astype(float).tolist()


def compute_rank(values: Any, target_index: int | None) -> int | None:
    if target_index is None:
        return None
    array = _numeric_array(values)
    if array is None:
        return None
    array = array.reshape(-1)
    if target_index < 0 or target_index >= array.size:
        return None
    target_value = float(array[target_index])
    return int(np.sum(array > target_value) + 1)


def compute_entropy(prob_values: Any) -> float | None:
    array = _numeric_array(prob_values)
    if array is None:
        return None
    probs = np.clip(array.reshape(-1).astype(np.float64), 0.0, None)
    total = float(np.sum(probs))
    if total <= 1e-12:
        return None
    if not np.isclose(total, 1.0, rtol=1e-3, atol=1e-6):
        probs = probs / total
    eps = 1e-12
    return float(-np.sum(probs * np.log(np.clip(probs, eps, None))))


def compute_top1_top2_margin(values: Any) -> float | None:
    array = _numeric_array(values)
    if array is None:
        return None
    flat = array.reshape(-1)
    if flat.size < 2:
        return None
    top2 = np.partition(flat, -2)[-2:]
    top2.sort()
    return float(top2[-1] - top2[-2])


def make_future_distribution_summary(
    sample: dict[str, Any] | None,
    horizon_label: str | None,
    view_type: str,
    show_fusion: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    horizon, horizon_index, gt_beam = _resolved_horizon(sample, horizon_label)
    for modality in _distribution_modalities(show_fusion):
        row = _distribution_summary_row(sample, modality, horizon, horizon_index, gt_beam, view_type)
        if row is not None:
            rows.append(row)
            continue
        legacy_row = _legacy_prediction_summary_row(sample, modality, horizon, horizon_index, gt_beam, view_type)
        if legacy_row is not None:
            rows.append(legacy_row)
    return pd.DataFrame(rows, columns=_future_distribution_summary_columns())


def make_future_distribution_heatmap(
    sample: dict[str, Any] | None,
    horizon_label: str | None,
    view_type: str,
    show_fusion: bool = True,
) -> go.Figure:
    horizon, horizon_index, gt_beam = _resolved_horizon(sample, horizon_label)
    rows, values, warnings = _distribution_matrix(sample, horizon_index, view_type, show_fusion)
    if not rows:
        return make_empty_figure(_missing_distribution_title(sample, view_type))

    num_beams = len(values[0])
    x_values = list(range(num_beams))
    heatmap_kwargs: dict[str, Any] = {
        "z": values,
        "x": x_values,
        "y": rows,
        "colorscale": "Viridis",
        "colorbar": {"title": _view_distribution_key(view_type)},
        "hovertemplate": "modality=%{y}<br>beam=%{x}<br>value=%{z:.4f}<extra></extra>",
    }
    if _view_distribution_key(view_type) == "prob":
        heatmap_kwargs.update({"zmin": 0.0, "zmax": 1.0})

    fig = go.Figure(go.Heatmap(**heatmap_kwargs))
    top1_x = [int(np.argmax(np.asarray(row_values, dtype=np.float64))) for row_values in values]
    fig.add_trace(
        go.Scatter(
            x=top1_x,
            y=rows,
            mode="markers",
            name="top1",
            marker={"color": "#1f77b4", "size": 10, "symbol": "circle"},
            hovertemplate="modality=%{y}<br>top1 beam=%{x}<extra></extra>",
        )
    )

    shapes: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    if gt_beam is not None and 0 <= gt_beam < num_beams:
        shapes.append(
            {
                "type": "line",
                "x0": gt_beam,
                "x1": gt_beam,
                "xref": "x",
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#d62728", "width": 2, "dash": "dash"},
            }
        )
        annotations.append(
            {
                "text": f"GT {gt_beam}",
                "x": gt_beam,
                "xref": "x",
                "y": 1.02,
                "yref": "paper",
                "showarrow": False,
                "font": {"color": "#d62728", "size": 12},
            }
        )
    if warnings:
        annotations.append(
            {
                "text": "; ".join(warnings[:2]),
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": -0.18,
                "showarrow": False,
                "xanchor": "left",
                "font": {"size": 11, "color": "#666666"},
            }
        )
    fig.update_layout(
        title=f"Future Beam Distribution | {horizon} | GT Beam = {_display_value(gt_beam)} | {view_type}",
        xaxis_title="beam index",
        yaxis_title="modality",
        margin={"l": 72, "r": 24, "t": 64, "b": 72},
        height=max(320, 120 + 44 * len(rows)),
        shapes=shapes,
        annotations=annotations,
    )
    return fig


def make_future_distribution_per_modality_plot(
    sample: dict[str, Any] | None,
    horizon_label: str | None,
    view_type: str,
    show_fusion: bool = True,
) -> go.Figure:
    if _make_subplots is None:
        return make_empty_figure("Per-Modality Distribution Requires Plotly")
    horizon, horizon_index, gt_beam = _resolved_horizon(sample, horizon_label)
    items = _available_distribution_items(sample, horizon_index, view_type, show_fusion)
    if not items:
        return make_empty_figure(_missing_distribution_title(sample, view_type))

    titles = []
    for modality, values in items:
        prob_values = get_probability_for_modality(sample, modality, horizon_index)
        rank = compute_rank(prob_values if prob_values is not None else values, gt_beam)
        gt_prob = _value_at(prob_values, gt_beam)
        top1 = int(np.argmax(np.asarray(values, dtype=np.float64)))
        titles.append(
            f"{modality} | top1={top1} | GT rank={_display_value(rank)} | P(GT)={_display_float(gt_prob)}"
        )

    fig = _make_subplots(rows=len(items), cols=1, subplot_titles=titles, shared_xaxes=False)
    for row_index, (modality, values) in enumerate(items, start=1):
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        x_values = list(range(array.size))
        top1 = int(np.argmax(array))
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=array.astype(float).tolist(),
                name=modality,
                marker_color="#4c78a8",
                showlegend=False,
            ),
            row=row_index,
            col=1,
        )
        if gt_beam is not None and 0 <= gt_beam < array.size:
            fig.add_trace(
                go.Scatter(
                    x=[gt_beam],
                    y=[float(array[gt_beam])],
                    mode="markers",
                    name=f"{modality} GT",
                    marker={"color": "#d62728", "size": 9},
                    showlegend=False,
                ),
                row=row_index,
                col=1,
            )
            fig.add_vline(x=gt_beam, line_color="#d62728", line_dash="dash", line_width=2, row=row_index, col=1)
        fig.add_vline(x=top1, line_color="#1f77b4", line_dash="dot", line_width=2, row=row_index, col=1)
        fig.update_yaxes(title_text=_view_distribution_key(view_type), row=row_index, col=1)
        fig.update_xaxes(title_text="beam index", row=row_index, col=1)

    fig.update_layout(
        title=f"Future Beam Distribution Per Modality | {horizon} | GT Beam = {_display_value(gt_beam)} | {view_type}",
        margin={"l": 64, "r": 24, "t": 72, "b": 48},
        height=max(320, 230 * len(items)),
        showlegend=False,
    )
    return fig


def make_future_distribution_plot(
    sample: dict[str, Any] | None,
    horizon_label: str | None,
    view_type: str,
    chart_type: str,
    show_fusion: bool = True,
) -> go.Figure:
    if str(chart_type or "").strip().lower() == "per_modality":
        return make_future_distribution_per_modality_plot(sample, horizon_label, view_type, show_fusion)
    return make_future_distribution_heatmap(sample, horizon_label, view_type, show_fusion)


def build_future_distribution_detail(
    sample: dict[str, Any] | None,
    horizon_label: str | None,
    view_type: str,
    show_fusion: bool = True,
) -> dict[str, Any]:
    horizon, horizon_index, gt_beam = _resolved_horizon(sample, horizon_label)
    detail: dict[str, Any] = {
        "horizon": horizon,
        "horizon_index": int(horizon_index),
        "gt_beam": _native_int(gt_beam),
        "view_type": str(view_type or "probability"),
        "modalities": {},
    }
    _, _, warnings = _distribution_matrix(sample, horizon_index, view_type, show_fusion)
    if warnings:
        detail["warnings"] = warnings

    for modality in _distribution_modalities(show_fusion):
        row = _distribution_summary_row(sample, modality, horizon, horizon_index, gt_beam, view_type)
        if row is None:
            row = _legacy_prediction_summary_row(sample, modality, horizon, horizon_index, gt_beam, view_type)
        if row is None:
            continue
        detail["modalities"][modality] = {
            "top1_beam": _native_int(row.get("top1_beam")),
            "top1_value": _native_float(row.get("top1_value")),
            "gt_value": _native_float(row.get("gt_value")),
            "gt_rank": _native_int(row.get("gt_rank")),
            "entropy": _native_float(row.get("entropy")),
            "is_correct": row.get("is_correct"),
            "distance_to_gt": _native_int(row.get("distance_to_gt")),
        }
    if not detail["modalities"]:
        detail["message"] = "Future beam distribution not available"
    return detail


def dict_to_dataframe(score_dict: dict[str, Any] | None, value_name: str) -> pd.DataFrame:
    items = _numeric_score_items(score_dict)
    return pd.DataFrame(items, columns=["modality", value_name])


def build_info(sample: dict[str, Any] | None) -> dict[str, Any]:
    if not sample:
        return {"message": "No samples found"}
    keys = ("sample_id", "scene_id", "scene_slug", "split", "sequence_id", "time_index", "timestamp")
    info = {key: sample.get(key) for key in keys if key in sample}
    for key in ("label", "prediction", "extra"):
        if key in sample:
            info[key] = sample.get(key)
    return info


def _same_sequence(sample: dict[str, Any], current_sample: dict[str, Any]) -> bool:
    return (
        sample.get("scene_id", sample.get("scene_slug")) == current_sample.get("scene_id", current_sample.get("scene_slug"))
        and sample.get("split") == current_sample.get("split")
        and sample.get("sequence_id") == current_sample.get("sequence_id")
    )


def _first_future_beam(sample: dict[str, Any]) -> int | None:
    values = _beam_label_values(safe_get(sample, "label.future_beams"))
    return int(values[0]) if values else None


def _sample_time_sort_key(sample: dict[str, Any]) -> tuple[int, int]:
    return (_sortable_int(sample.get("time_index")), _sortable_int(sample.get("_manifest_index")))


def _current_sample_position(samples: list[dict[str, Any]], current_sample: dict[str, Any]) -> int | None:
    current_identity = (
        current_sample.get("_manifest_dir"),
        current_sample.get("sample_id"),
        current_sample.get("_manifest_index"),
    )
    for index, sample in enumerate(samples):
        identity = (sample.get("_manifest_dir"), sample.get("sample_id"), sample.get("_manifest_index"))
        if identity == current_identity:
            return index
    return None


def _sortable_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def manifest_context(sample: dict[str, Any]) -> tuple[str | None, str | None]:
    return sample.get("_manifest_dir"), sample.get("_project_root")


def clamp_index(index: Any, total: int) -> int:
    if total <= 0:
        return 0
    try:
        value = int(index)
    except (TypeError, ValueError):
        value = 0
    return max(0, min(total - 1, value))


def _load_jsonl(text: str) -> list[dict[str, Any]]:
    samples = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            samples.append(item)
    return samples


def _find_project_root(start: Path) -> Path | None:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
    return None


def _none_if_all(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in {"", "all"} else text


def _sample_matches_scene(sample: dict[str, Any], expected: str) -> bool:
    scene_values = [sample.get("scene_id"), sample.get("scene_slug")]
    return any(str(value) == expected for value in scene_values if value is not None)


def _is_low_quality(quality: Any) -> bool:
    if not isinstance(quality, dict):
        return False
    values = [value for _, value in _numeric_score_items(quality)]
    if not values:
        return False
    return any(value < LOW_QUALITY_MODALITY_THRESHOLD for value in values) or (
        float(np.mean(values)) < LOW_QUALITY_MEAN_THRESHOLD
    )


def _numeric_score_items(score_dict: dict[str, Any] | None) -> list[tuple[str, float]]:
    if not isinstance(score_dict, dict):
        return []
    items = []
    for key, value in score_dict.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            items.append((str(key), number))
    return items


def _extra_confidence_modalities(sample: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for source in (
        sample.get("confidence"),
        sample.get("confidence_curves"),
        sample.get("beam_distribution"),
        sample.get("modality_prediction"),
        sample.get("modality_predictions"),
        safe_get(sample, "prediction.modalities"),
    ):
        if isinstance(source, dict):
            keys.extend(str(key) for key in source.keys())
    order = {modality: index for index, modality in enumerate(DISTRIBUTION_MODALITIES)}
    return sorted(set(keys), key=lambda item: (order.get(item, len(order)), _natural_key(item)))


def _t1_confidence_for_modality(sample: dict[str, Any], modality: str) -> float | None:
    prob_values = get_probability_for_modality(sample, modality, 0)
    if prob_values:
        return _round_float(float(np.max(np.asarray(prob_values, dtype=np.float64))))

    for source in (
        safe_get(sample, f"confidence_curves.{modality}"),
        safe_get(sample, f"prediction.confidence_curves.{modality}"),
        safe_get(sample, f"prediction.per_label_confidence.{modality}"),
        safe_get(sample, f"prediction.probabilities.{modality}"),
        safe_get(sample, f"prediction.probs.{modality}"),
        safe_get(sample, f"prediction.scores.{modality}"),
    ):
        value = _first_curve_max(source)
        if value is not None:
            return _round_float(value)

    for source in (
        safe_get(sample, f"modality_prediction.{modality}"),
        safe_get(sample, f"modality_predictions.{modality}"),
        safe_get(sample, f"prediction.modalities.{modality}"),
    ):
        if not isinstance(source, dict):
            continue
        value = _float_at(
            source.get("top1_confidence", source.get("top1_prob", source.get("top1_probability"))),
            0,
        )
        if value is not None:
            return _round_float(value)

    value = sample.get("confidence", {}).get(modality) if isinstance(sample.get("confidence"), dict) else None
    if isinstance(value, (list, tuple, np.ndarray)):
        curve_value = _first_curve_max(value)
        if curve_value is not None:
            return _round_float(curve_value)
    native = _native_float(value)
    return _round_float(native)


def _first_curve_max(values: Any) -> float | None:
    array = _numeric_array(values)
    if array is None:
        return None
    if array.ndim == 0:
        return float(array)
    if array.ndim == 1:
        return float(np.max(array))
    first = np.asarray(array[0], dtype=np.float64).reshape(-1)
    if first.size == 0:
        return None
    return float(np.max(first))


def _future_distribution_summary_columns() -> list[str]:
    return [
        "modality",
        "horizon",
        "gt_beam",
        "top1_beam",
        "top1_value",
        "gt_value",
        "gt_rank",
        "top1_minus_gt",
        "top1_top2_margin",
        "entropy",
        "is_correct",
        "distance_to_gt",
    ]


def _distribution_modalities(show_fusion: bool) -> tuple[str, ...]:
    if show_fusion:
        return DISTRIBUTION_MODALITIES
    return MODALITIES


def _resolved_horizon(sample: dict[str, Any] | None, horizon_label: str | None) -> tuple[str, int, int | None]:
    future_beams = get_future_beams(sample)
    horizon_index = parse_horizon_label(horizon_label)
    if not future_beams:
        return "t+1", 0, None
    if horizon_index >= len(future_beams):
        horizon_index = 0
    return f"t+{horizon_index + 1}", horizon_index, int(future_beams[horizon_index])


def _view_distribution_key(view_type: str | None) -> str:
    text = str(view_type or "probability").strip().lower()
    return "logit" if text == "logit" else "prob"


def _distribution_values(entry: Any, key: str, horizon_index: int) -> np.ndarray | None:
    if entry is None:
        return None
    raw: Any
    if isinstance(entry, dict):
        raw = None
        aliases = {
            "prob": ("prob", "probs", "probability", "probabilities"),
            "logit": ("logit", "logits"),
        }[key]
        for alias in aliases:
            if alias in entry:
                raw = entry[alias]
                break
        if raw is None:
            return None
    else:
        if key != "prob":
            return None
        raw = entry

    array = _numeric_array(raw)
    if array is None:
        return None
    if array.ndim == 1:
        if horizon_index > 0:
            return None
        return array.reshape(-1)
    if horizon_index < 0 or horizon_index >= array.shape[0]:
        return None
    return np.asarray(array[horizon_index], dtype=np.float64).reshape(-1)


def _available_distribution_items(
    sample: dict[str, Any] | None,
    horizon_index: int,
    view_type: str,
    show_fusion: bool,
) -> list[tuple[str, list[float]]]:
    items = []
    for modality in _distribution_modalities(show_fusion):
        values = get_distribution_for_modality(sample, modality, horizon_index, view_type)
        if values:
            items.append((modality, values))
    return items


def _distribution_matrix(
    sample: dict[str, Any] | None,
    horizon_index: int,
    view_type: str,
    show_fusion: bool,
) -> tuple[list[str], list[list[float]], list[str]]:
    rows: list[str] = []
    values: list[list[float]] = []
    warnings: list[str] = []
    expected_size: int | None = None
    for modality in _distribution_modalities(show_fusion):
        modality_values = get_distribution_for_modality(sample, modality, horizon_index, view_type)
        if not modality_values:
            continue
        if expected_size is None:
            expected_size = len(modality_values)
        elif len(modality_values) != expected_size:
            warnings.append(
                f"Skipped {modality}: num_beams={len(modality_values)} differs from {expected_size}"
            )
            continue
        rows.append(modality)
        values.append([float(value) for value in modality_values])
    return rows, values, warnings


def _missing_distribution_title(sample: dict[str, Any] | None, view_type: str) -> str:
    if _view_distribution_key(view_type) == "logit" and extract_beam_distribution(sample):
        return "Logits not available"
    return "Future Beam Distribution Not Available"


def _distribution_summary_row(
    sample: dict[str, Any] | None,
    modality: str,
    horizon: str,
    horizon_index: int,
    gt_beam: int | None,
    view_type: str,
) -> dict[str, Any] | None:
    values = get_distribution_for_modality(sample, modality, horizon_index, view_type)
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return None
    top1_beam = int(np.argmax(array))
    top1_value = float(array[top1_beam])
    gt_value = _value_at(array, gt_beam)
    prob_values = get_probability_for_modality(sample, modality, horizon_index)
    rank_source = prob_values if prob_values is not None else values
    gt_rank = compute_rank(rank_source, gt_beam)
    entropy = compute_entropy(prob_values)
    margin = compute_top1_top2_margin(values)
    return _summary_row(
        modality=modality,
        horizon=horizon,
        gt_beam=gt_beam,
        top1_beam=top1_beam,
        top1_value=top1_value,
        gt_value=gt_value,
        gt_rank=gt_rank,
        top1_top2_margin=margin,
        entropy=entropy,
    )


def _legacy_prediction_summary_row(
    sample: dict[str, Any] | None,
    modality: str,
    horizon: str,
    horizon_index: int,
    gt_beam: int | None,
    view_type: str,
) -> dict[str, Any] | None:
    if not isinstance(sample, dict):
        return None
    source = None
    for candidate in (
        safe_get(sample, f"modality_prediction.{modality}"),
        safe_get(sample, f"modality_predictions.{modality}"),
        safe_get(sample, f"prediction.modalities.{modality}"),
    ):
        if isinstance(candidate, dict):
            source = candidate
            break
    if source is None:
        return None

    top1_beam = _int_at(source.get("top1", source.get("top1_beam")), horizon_index)
    if top1_beam is None:
        return None
    top1_value = None
    gt_value = None
    if _view_distribution_key(view_type) == "prob":
        top1_value = _float_at(
            source.get("top1_confidence", source.get("top1_prob", source.get("top1_probability"))),
            horizon_index,
        )
        gt_value = _float_at(
            source.get(
                "future_label_confidence",
                source.get("gt_confidence", source.get("gt_probability", source.get("gt_prob"))),
            ),
            horizon_index,
        )
    gt_rank = _int_at(source.get("future_label_rank", source.get("gt_rank")), horizon_index)
    return _summary_row(
        modality=modality,
        horizon=horizon,
        gt_beam=gt_beam,
        top1_beam=top1_beam,
        top1_value=top1_value,
        gt_value=gt_value,
        gt_rank=gt_rank,
        top1_top2_margin=None,
        entropy=None,
    )


def _summary_row(
    *,
    modality: str,
    horizon: str,
    gt_beam: int | None,
    top1_beam: int | None,
    top1_value: float | None,
    gt_value: float | None,
    gt_rank: int | None,
    top1_top2_margin: float | None,
    entropy: float | None,
) -> dict[str, Any]:
    top1_minus_gt = None
    if top1_value is not None and gt_value is not None:
        top1_minus_gt = float(top1_value) - float(gt_value)
    is_correct = None
    distance_to_gt = None
    if top1_beam is not None and gt_beam is not None:
        is_correct = bool(int(top1_beam) == int(gt_beam))
        distance_to_gt = abs(int(top1_beam) - int(gt_beam))
    return {
        "modality": modality,
        "horizon": horizon,
        "gt_beam": _native_int(gt_beam),
        "top1_beam": _native_int(top1_beam),
        "top1_value": _round_float(top1_value),
        "gt_value": _round_float(gt_value),
        "gt_rank": _native_int(gt_rank),
        "top1_minus_gt": _round_float(top1_minus_gt),
        "top1_top2_margin": _round_float(top1_top2_margin),
        "entropy": _round_float(entropy),
        "is_correct": is_correct,
        "distance_to_gt": _native_int(distance_to_gt),
    }


def _value_at(values: Any, index: int | None) -> float | None:
    if index is None:
        return None
    array = _numeric_array(values)
    if array is None:
        return None
    flat = array.reshape(-1)
    if index < 0 or index >= flat.size:
        return None
    return float(flat[index])


def _int_at(values: Any, index: int) -> int | None:
    value = _value_from_sequence(values, index)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(number)


def _float_at(values: Any, index: int) -> float | None:
    value = _value_from_sequence(values, index)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _value_from_sequence(values: Any, index: int) -> Any:
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            return values.item() if index == 0 else None
        values = values.reshape(-1).tolist()
    if isinstance(values, (list, tuple, np.ndarray)):
        if len(values) == 0:
            return None
        if len(values) == 1 and index > 0:
            return values[0]
        if index < len(values):
            return values[index]
        return None
    return values if index == 0 else None


def _round_float(value: Any) -> float | None:
    number = _native_float(value)
    if number is None:
        return None
    return round(number, 4)


def _native_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _native_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(number)


def _display_value(value: Any) -> str:
    native = _native_int(value)
    return "N/A" if native is None else str(native)


def _display_float(value: Any) -> str:
    native = _native_float(value)
    return "N/A" if native is None else f"{native:.4f}"


def _beam_confidence_curves(sample: dict[str, Any]) -> list[tuple[str, list[int], list[float]]]:
    sources = [
        sample.get("beam_distribution"),
        safe_get(sample, "prediction.per_label_confidence"),
        safe_get(sample, "prediction.confidence_curves"),
        safe_get(sample, "prediction.label_confidence"),
        safe_get(sample, "prediction.modality_confidence"),
        safe_get(sample, "prediction.probabilities"),
        safe_get(sample, "prediction.probs"),
        safe_get(sample, "prediction.scores"),
        safe_get(sample, "prediction.logits"),
        sample.get("confidence_curves"),
        sample.get("label_confidence"),
        sample.get("confidence"),
    ]
    curves: list[tuple[str, list[int], list[float]]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for source in sources:
        for name, x_values, y_values in _extract_curve_items(source, "confidence"):
            key = (name, tuple(x_values))
            if key in seen:
                continue
            seen.add(key)
            curves.append((name, x_values, y_values))
    return curves


def _extract_curve_items(data: Any, default_name: str) -> list[tuple[str, list[int], list[float]]]:
    if data is None:
        return []
    if isinstance(data, dict):
        if isinstance(data.get("modalities"), dict):
            return _extract_curve_items(data["modalities"], default_name)
        for key in ("prob", "probability", "probabilities", "probs", "scores", "confidences", "confidence", "values", "logit", "logits"):
            if key in data:
                labels = data.get("labels", data.get("beam_labels"))
                return _curve_entries(default_name, data[key], labels=labels, logits=key in {"logit", "logits"})
        if _dict_has_numeric_labels(data):
            x_values = [int(float(key)) for key in data.keys()]
            y_values = []
            for value in data.values():
                try:
                    y_values.append(float(value))
                except (TypeError, ValueError):
                    return []
            pairs = sorted(zip(x_values, y_values), key=lambda item: item[0])
            return [(default_name, [item[0] for item in pairs], [item[1] for item in pairs])]
        curves: list[tuple[str, list[int], list[float]]] = []
        for key, value in data.items():
            if key in {"labels", "beam_labels"}:
                continue
            curves.extend(_extract_curve_items(value, str(key)))
        return curves
    return _curve_entries(default_name, data)


def _curve_entries(
    name: str,
    values: Any,
    *,
    labels: Any = None,
    logits: bool = False,
) -> list[tuple[str, list[int], list[float]]]:
    array = _numeric_array(values)
    if array is None or array.size <= 1:
        return []
    if logits:
        array = _softmax(array, axis=-1)
    if array.ndim == 1:
        x_values = _label_axis(labels, array.size)
        return [(name, x_values, array.astype(float).tolist())]
    if array.ndim == 2:
        x_values = _label_axis(labels, array.shape[1])
        return [(f"{name} t+1", x_values, array[0].astype(float).tolist())]
    if array.ndim == 3:
        flat = array.reshape((-1, array.shape[-1]))
        x_values = _label_axis(labels, array.shape[-1])
        return [(f"{name} t+1", x_values, flat[0].astype(float).tolist())]
    return []


def _label_axis(labels: Any, size: int) -> list[int]:
    raw = _numeric_array(labels)
    if raw is not None and raw.size == size:
        return raw.reshape(-1).astype(int).tolist()
    return list(range(size))


def _beam_label_values(values: Any) -> list[int]:
    array = _numeric_array(values)
    if array is None:
        return []
    return [int(value) for value in array.reshape(-1).tolist() if np.isfinite(value)]


def _dict_has_numeric_labels(data: dict[str, Any]) -> bool:
    if not data:
        return False
    for key, value in data.items():
        try:
            float(key)
            float(value)
        except (TypeError, ValueError):
            return False
    return True


def _softmax(array: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=axis, keepdims=True)
    return exp / np.clip(denom, 1e-12, None)


def _gps_xy(data: Any) -> tuple[list[float], list[float], str, str] | None:
    if isinstance(data, dict):
        for x_key, y_key in (("x", "y"), ("east", "north"), ("lon", "lat"), ("longitude", "latitude")):
            if x_key in data and y_key in data:
                x_values = _numeric_list(data.get(x_key))
                y_values = _numeric_list(data.get(y_key))
                if x_values and y_values:
                    size = min(len(x_values), len(y_values))
                    return x_values[:size], y_values[:size], x_key, y_key
    if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
        keys = list(data[0].keys())
        for x_key, y_key in (("x", "y"), ("east", "north"), ("lon", "lat"), ("longitude", "latitude")):
            if x_key in keys and y_key in keys:
                x_values = _numeric_list([item.get(x_key) for item in data])
                y_values = _numeric_list([item.get(y_key) for item in data])
                if x_values and y_values:
                    size = min(len(x_values), len(y_values))
                    return x_values[:size], y_values[:size], x_key, y_key
    if isinstance(data, list):
        array = _numeric_array(data)
        if array is not None and array.ndim == 2 and array.shape[1] >= 2:
            return array[:, 0].astype(float).tolist(), array[:, 1].astype(float).tolist(), "x", "y"
    return None


def _gps_feature_figure(data: Any, title: str):
    if not isinstance(data, dict) or "features" not in data:
        return None
    array = _numeric_array(data.get("features"))
    if array is None:
        return make_empty_figure(title)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        return make_empty_figure(title)
    raw_names = data.get("feature_names")
    if isinstance(raw_names, list) and len(raw_names) >= array.shape[1]:
        names = [str(name) for name in raw_names[: array.shape[1]]]
    else:
        names = [f"feature_{idx}" for idx in range(array.shape[1])]
    x_values = list(range(array.shape[0]))
    fig = go.Figure()
    for idx, name in enumerate(names):
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=array[:, idx].astype(float).tolist(),
                mode="lines+markers",
                name=name,
                line={"width": 2},
            )
        )
    fig.update_layout(
        title=_title_with_scale(title, data),
        xaxis_title="time index",
        yaxis_title="feature value",
        margin={"l": 56, "r": 16, "t": 48, "b": 48},
        height=300,
        showlegend=True,
    )
    return fig


def _beam_power_bar(values: Any, title: str, y_title: str = "power") -> go.Figure:
    array = _numeric_array(values)
    if array is None:
        return make_empty_figure(title)
    array = array.reshape(-1)
    fig = go.Figure(go.Bar(x=list(range(array.size)), y=array.astype(float), marker_color="#59a14f"))
    fig.update_layout(
        title=title,
        xaxis_title="beam index",
        yaxis_title=y_title,
        margin={"l": 48, "r": 16, "t": 48, "b": 48},
        height=300,
    )
    return fig


def _title_with_scale(title: str, data: dict[str, Any]) -> str:
    labels = []
    feature_space = data.get("feature_space")
    scale = data.get("scale")
    units = data.get("units")
    if feature_space:
        labels.append(str(feature_space))
    elif scale:
        labels.append(str(scale))
    if units and str(units) not in labels:
        labels.append(str(units))
    if data.get("normalized") is True and "normalized" not in labels:
        labels.append("normalized")
    return f"{title} ({', '.join(labels)})" if labels else title


def _numeric_list(values: Any) -> list[float]:
    array = _numeric_array(values)
    if array is None:
        return []
    return array.reshape(-1).astype(float).tolist()


def _numeric_array(values: Any) -> np.ndarray | None:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.size == 0 or not np.all(np.isfinite(array)):
        return None
    return array


def _looks_like_image_path(value: Any) -> bool:
    if not isinstance(value, (str, Path)):
        return False
    suffix = Path(str(value)).suffix.lower()
    return suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


def _natural_key(value: str) -> tuple[int, str]:
    text = str(value)
    digits = ""
    for char in reversed(text):
        if not char.isdigit():
            break
        digits = char + digits
    if digits:
        return (0, f"{text[: -len(digits)]}{int(digits):012d}")
    return (1, text)
