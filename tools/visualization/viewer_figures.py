from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

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

from tools.visualization.viewer_manifest_io import (
    _looks_like_image_path,
    _numeric_score_items,
    load_image_safe,
    load_json_safe,
    safe_get,
)
from tools.visualization.viewer_prediction_tables import (
    _available_distribution_items,
    _beam_confidence_curves,
    _beam_label_values,
    _current_sample_position,
    _display_float,
    _display_value,
    _distribution_matrix,
    _first_future_beam,
    _missing_distribution_title,
    _numeric_array,
    _numeric_list,
    _resolved_horizon,
    _same_sequence,
    _sample_time_sort_key,
    _value_at,
    _view_distribution_key,
    compute_rank,
    get_probability_for_modality,
    single_modality_t1_confidence,
)

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




__all__ = [
    "go",
    "make_beam_confidence_figure",
    "make_beam_index_trend_figure",
    "make_empty_figure",
    "make_future_distribution_heatmap",
    "make_future_distribution_per_modality_plot",
    "make_future_distribution_plot",
    "make_gps_figure",
    "make_image_figure",
    "make_mmwave_figure",
    "make_score_bar",
    "make_single_modality_confidence_figure",
]
