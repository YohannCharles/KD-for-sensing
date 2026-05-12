#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kd_sensing.cli.common import load_cli_config  # noqa: E402
from kd_sensing.diagnostics.visualization.config import parse_visualization_config  # noqa: E402
from kd_sensing.diagnostics.viewer_manifest import export_viewer_manifest  # noqa: E402
from kd_sensing.diagnostics.viewer_predictions import (  # noqa: E402
    export_viewer_model_predictions,
    parse_key_value_paths,
    parse_modalities,
)

from tools.visualization.viewer_utils import (  # noqa: E402
    SHOW_MODES,
    build_future_distribution_detail,
    build_info,
    clamp_index,
    dict_to_dataframe,
    filter_samples,
    get_available_scenes,
    get_available_splits,
    get_future_beams,
    load_manifest,
    make_beam_confidence_figure,
    make_beam_index_trend_figure,
    make_empty_figure,
    make_future_distribution_plot,
    make_future_distribution_summary,
    make_gps_figure,
    make_mmwave_figure,
    make_score_bar,
    make_single_modality_confidence_figure,
    manifest_context,
    resolve_path,
    safe_get,
    single_modality_confidence_dataframe,
)
from tools.visualization.complementarity_explorer import (  # noqa: E402
    case_detail_payload,
    export_filtered_cases,
    filter_complementarity_cases,
    find_sample_index_for_case,
    load_complementarity_explorer,
    selected_event_row,
)


DEFAULT_MANIFEST = Path("data/visualization/samples.json")
PRELOAD_RADIUS = 5
RENDER_CACHE_MAX_ITEMS = 192
STATIC_PREFIX_COUNT = 15
FUTURE_OUTPUT_COUNT = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the Gradio multimodal temporal sample viewer.")
    parser.add_argument("--config", "-c", help="Dataset/training config. If provided, the viewer processes and caches it first.")
    parser.add_argument("--manifest", help="Path to samples.json or JSONL manifest. With --config, this is the output manifest path.")
    parser.add_argument("--cache-dir", help="Directory for reusable processed viewer cache when --config is used.")
    parser.add_argument(
        "--scenes",
        "--scene",
        dest="scenes",
        help=(
            "Scene ids/aliases to prepare, comma-separated. "
            "Use --scenes 9,32 or --scenes all to populate the Scene dropdown."
        ),
    )
    parser.add_argument("--force-rebuild", action="store_true", help="Reprocess the dataset even if a valid cache exists.")
    parser.add_argument("--sample-limit", type=int, help="Optional cap for quick debugging. Defaults to all samples.")
    parser.add_argument("--predictions", help="Optional prediction JSON to merge, or output path when --run-models is used.")
    parser.add_argument(
        "--run-models",
        action="store_true",
        help="Run single-modality checkpoints first and show per-beam confidence curves in the viewer.",
    )
    parser.add_argument(
        "--no-auto-predictions",
        action="store_true",
        help="Do not auto-merge cached model predictions from --cache-dir/model_predictions when --predictions is omitted.",
    )
    parser.add_argument(
        "--prediction-modalities",
        help="Comma-separated modalities for --run-models. Defaults to diagnostics.visualization.modalities.",
    )
    parser.add_argument(
        "--model-config",
        action="append",
        default=[],
        help="Override a modality model config as modality=path. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--model-checkpoint",
        action="append",
        default=[],
        help="Override a modality checkpoint as modality=path. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--model-devices",
        default="cuda",
        help="Devices for model inference. Defaults to cuda and uses all visible GPUs; use cpu explicitly for CPU.",
    )
    parser.add_argument("--model-workers", type=int, help="Number of modality inference workers. Defaults to parallel.")
    parser.add_argument("--model-batch-size", type=int, default=32, help="Batch size for model prediction export.")
    parser.add_argument("--model-num-workers", type=int, default=0, help="DataLoader workers per model prediction worker.")
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        help="Override config value using dotted key=value syntax when --config is used. Can be repeated.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host/IP for the Gradio server.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio server.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument("--debug", action="store_true", help="Enable Gradio debug mode.")
    parser.add_argument(
        "--profile-render",
        action="store_true",
        help="Log Gradio viewer callback timing and cache statistics for frame-navigation profiling.",
    )
    parser.add_argument(
        "--complementarity-dir",
        help="Optional directory containing complementarity_cases.csv.gz and complementarity_summary.json.",
    )
    parser.add_argument("--project-root", default=str(ROOT), help="Project root used to resolve relative manifest paths.")
    parser.add_argument("--check-only", action="store_true", help=argparse.SUPPRESS)
    return parser


class RenderStats:
    """Small injectable counter/timer for viewer callback tests and profile logs."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.timings_ms: dict[str, float] = {}

    def incr(self, name: str, amount: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + int(amount)

    def add_time(self, name: str, elapsed_seconds: float) -> None:
        self.timings_ms[name] = self.timings_ms.get(name, 0.0) + elapsed_seconds * 1000.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "timings_ms": {key: round(value, 3) for key, value in self.timings_ms.items()},
        }


class FilteredSampleIndex:
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples
        self.filter_calls = 0
        self._indices_by_key: dict[tuple[str, str, str], list[int]] = {}
        self._index_by_identity = {id(sample): index for index, sample in enumerate(samples)}

    def filtered_indices(
        self,
        scene: str | None,
        split: str | None,
        show_mode: str | None,
        *,
        stats: RenderStats | None = None,
    ) -> list[int]:
        key = self._key(scene, split, show_mode)
        cached = self._indices_by_key.get(key)
        if cached is not None:
            if stats is not None:
                stats.incr("filter_cache_hit")
            return cached
        if stats is not None:
            stats.incr("filter_cache_miss")
        self.filter_calls += 1
        filtered = filter_samples(self.samples, scene=scene, split=split, show_mode=show_mode)
        indices = [self._index_by_identity[id(sample)] for sample in filtered if id(sample) in self._index_by_identity]
        self._indices_by_key[key] = indices
        return indices

    def filtered_samples(
        self,
        scene: str | None,
        split: str | None,
        show_mode: str | None,
        *,
        stats: RenderStats | None = None,
    ) -> list[dict[str, Any]]:
        return [self.samples[index] for index in self.filtered_indices(scene, split, show_mode, stats=stats)]

    @staticmethod
    def _key(scene: str | None, split: str | None, show_mode: str | None) -> tuple[str, str, str]:
        return (
            str(scene or "all"),
            str(split or "all"),
            str(show_mode or "all"),
        )


def render_sample(
    samples: list[dict[str, Any]],
    view_index: Any,
    scene: str | None,
    split: str | None,
    show_mode: str | None,
    future_horizon: str | None = "t+1",
    distribution_view: str = "probability",
    distribution_chart: str = "heatmap",
    show_fusion: bool = True,
    render_cache: Any | None = None,
    sample_index: FilteredSampleIndex | None = None,
    stats: RenderStats | None = None,
) -> tuple[Any, ...]:
    filter_started = time.perf_counter()
    if sample_index is not None:
        filtered = sample_index.filtered_samples(scene, split, show_mode, stats=stats)
    else:
        if stats is not None:
            stats.incr("filter_calls")
        filtered = filter_samples(samples, scene=scene, split=split, show_mode=show_mode)
    if stats is not None:
        stats.add_time("filter", time.perf_counter() - filter_started)
    if not filtered:
        return _empty_outputs("No samples found")

    index = clamp_index(view_index, len(filtered))
    sample = filtered[index]
    controls = (future_horizon, distribution_view, distribution_chart, bool(show_fusion))
    if render_cache is not None:
        base_outputs = render_cache.get_or_render(sample, controls, stats=stats)
        render_cache.preload(filtered, index, controls)
    else:
        base_outputs = _render_sample_base(sample, controls, stats=stats)

    beam_index_trend = make_beam_index_trend_figure(filtered, sample)
    sample_id = sample.get("sample_id", sample.get("_manifest_index", index))
    scene_label = sample.get("scene_slug", sample.get("scene_id", ""))
    sample_text = (
        f"Sample {index + 1}/{len(filtered)} | id={sample_id} | "
        f"scene={scene_label} | split={sample.get('split', '')}"
    )
    return (*base_outputs, beam_index_trend, sample_text)


def _render_sample_base(
    sample: dict[str, Any],
    controls: tuple[Any, ...],
    *,
    stats: RenderStats | None = None,
) -> tuple[Any, ...]:
    static_outputs = _render_sample_static(sample, stats=stats)
    future_outputs = _render_sample_future(sample, controls, stats=stats)
    return _compose_base_outputs(static_outputs, future_outputs)


def _render_sample_static(sample: dict[str, Any], *, stats: RenderStats | None = None) -> tuple[Any, ...]:
    started = time.perf_counter()
    try:
        manifest_dir, project_root = manifest_context(sample)

        raw_image = _load_image(sample, "raw.image", manifest_dir, project_root, stats=stats)
        raw_lidar = _load_image(sample, "raw.lidar", manifest_dir, project_root, stats=stats)
        raw_radar = _load_image(sample, "raw.radar", manifest_dir, project_root, stats=stats)
        raw_gps = make_gps_figure(safe_get(sample, "raw.gps"), "Raw GPS", manifest_dir, project_root)
        raw_mmwave = make_mmwave_figure(safe_get(sample, "raw.mmwave"), "Raw mmWave", manifest_dir, project_root)

        proc_image = _load_image(sample, "processed.image", manifest_dir, project_root, stats=stats)
        proc_lidar = _load_image(sample, "processed.lidar", manifest_dir, project_root, stats=stats)
        proc_radar = _load_image(sample, "processed.radar", manifest_dir, project_root, stats=stats)
        proc_gps = make_gps_figure(safe_get(sample, "processed.gps"), "Processed GPS", manifest_dir, project_root)
        proc_mmwave = make_mmwave_figure(
            safe_get(sample, "processed.mmwave"),
            "Processed mmWave",
            manifest_dir,
            project_root,
        )

        quality = sample.get("quality")
        gate = sample.get("gate")
        return (
            raw_image,
            raw_lidar,
            raw_radar,
            raw_gps,
            raw_mmwave,
            proc_image,
            proc_lidar,
            proc_radar,
            proc_gps,
            proc_mmwave,
            build_info(sample),
            make_beam_confidence_figure(sample),
            make_single_modality_confidence_figure(sample),
            make_score_bar(quality, "Modality Quality"),
            make_score_bar(gate, "Gate Weight"),
            single_modality_confidence_dataframe(sample),
            dict_to_dataframe(quality, "quality"),
            dict_to_dataframe(gate, "gate"),
        )
    finally:
        if stats is not None:
            stats.add_time("render_static", time.perf_counter() - started)


def _render_sample_future(
    sample: dict[str, Any],
    controls: tuple[Any, ...],
    *,
    stats: RenderStats | None = None,
) -> tuple[Any, ...]:
    started = time.perf_counter()
    try:
        future_horizon, distribution_view, distribution_chart, show_fusion = controls
        return (
            make_future_distribution_plot(sample, future_horizon, distribution_view, distribution_chart, show_fusion),
            make_future_distribution_summary(sample, future_horizon, distribution_view, show_fusion),
            build_future_distribution_detail(sample, future_horizon, distribution_view, show_fusion),
        )
    finally:
        if stats is not None:
            stats.add_time("render_distribution", time.perf_counter() - started)


def _compose_base_outputs(static_outputs: tuple[Any, ...], future_outputs: tuple[Any, ...]) -> tuple[Any, ...]:
    return (
        *static_outputs[:STATIC_PREFIX_COUNT],
        *future_outputs,
        *static_outputs[STATIC_PREFIX_COUNT:],
    )


def _split_base_outputs(base_outputs: tuple[Any, ...]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    static_outputs = (
        *base_outputs[:STATIC_PREFIX_COUNT],
        *base_outputs[STATIC_PREFIX_COUNT + FUTURE_OUTPUT_COUNT :],
    )
    future_outputs = base_outputs[STATIC_PREFIX_COUNT : STATIC_PREFIX_COUNT + FUTURE_OUTPUT_COUNT]
    return static_outputs, future_outputs


class _SampleRenderCache:
    def __init__(self, *, max_items: int = RENDER_CACHE_MAX_ITEMS, preload_radius: int = PRELOAD_RADIUS) -> None:
        self.max_items = max(1, int(max_items))
        self.preload_radius = max(0, int(preload_radius))
        self._static_items: OrderedDict[tuple[Any, ...], tuple[Any, ...]] = OrderedDict()
        self._future_items: OrderedDict[tuple[Any, ...], tuple[Any, ...]] = OrderedDict()
        self._static_inflight: set[tuple[Any, ...]] = set()
        self._future_inflight: set[tuple[Any, ...]] = set()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="viewer-preload")

    def get_or_render(
        self,
        sample: dict[str, Any],
        controls: tuple[Any, ...],
        *,
        stats: RenderStats | None = None,
    ) -> tuple[Any, ...]:
        static_outputs = self.get_or_render_static(sample, stats=stats)
        future_outputs = self.get_or_render_future(sample, controls, stats=stats)
        return _compose_base_outputs(static_outputs, future_outputs)

    def get_or_render_static(self, sample: dict[str, Any], *, stats: RenderStats | None = None) -> tuple[Any, ...]:
        key = self._sample_key(sample)
        cached = self._get(self._static_items, key)
        if cached is not None:
            if stats is not None:
                stats.incr("static_cache_hit")
            return cached
        if stats is not None:
            stats.incr("static_cache_miss")
        rendered = _render_sample_static(sample, stats=stats)
        self._set(self._static_items, key, rendered)
        return rendered

    def get_or_render_future(
        self,
        sample: dict[str, Any],
        controls: tuple[Any, ...],
        *,
        stats: RenderStats | None = None,
    ) -> tuple[Any, ...]:
        key = (*self._sample_key(sample), *controls)
        cached = self._get(self._future_items, key)
        if cached is not None:
            if stats is not None:
                stats.incr("future_cache_hit")
            return cached
        if stats is not None:
            stats.incr("future_cache_miss")
        rendered = _render_sample_future(sample, controls, stats=stats)
        self._set(self._future_items, key, rendered)
        return rendered

    def preload(self, filtered: list[dict[str, Any]], index: int, controls: tuple[Any, ...]) -> None:
        if self.preload_radius <= 0 or not filtered:
            return
        for neighbor_index in self._neighbor_indices(index, len(filtered)):
            sample = filtered[neighbor_index]
            static_key = self._sample_key(sample)
            if self._mark_inflight(self._static_items, self._static_inflight, static_key):
                self._executor.submit(self._preload_static, sample, static_key)
            future_key = (*static_key, *controls)
            if self._mark_inflight(self._future_items, self._future_inflight, future_key):
                self._executor.submit(self._preload_future, sample, controls, future_key)

    def _preload_static(self, sample: dict[str, Any], key: tuple[Any, ...]) -> None:
        try:
            if self._get(self._static_items, key) is None:
                self._set(self._static_items, key, _render_sample_static(sample))
        except Exception:
            pass
        finally:
            with self._lock:
                self._static_inflight.discard(key)

    def _preload_future(self, sample: dict[str, Any], controls: tuple[Any, ...], key: tuple[Any, ...]) -> None:
        try:
            if self._get(self._future_items, key) is None:
                self._set(self._future_items, key, _render_sample_future(sample, controls))
        except Exception:
            pass
        finally:
            with self._lock:
                self._future_inflight.discard(key)

    def _get(self, items: OrderedDict[tuple[Any, ...], tuple[Any, ...]], key: tuple[Any, ...]) -> tuple[Any, ...] | None:
        with self._lock:
            value = items.get(key)
            if value is None:
                return None
            items.move_to_end(key)
            return value

    def _set(self, items: OrderedDict[tuple[Any, ...], tuple[Any, ...]], key: tuple[Any, ...], value: tuple[Any, ...]) -> None:
        with self._lock:
            items[key] = value
            items.move_to_end(key)
            while len(items) > self.max_items:
                items.popitem(last=False)

    def _mark_inflight(
        self,
        items: OrderedDict[tuple[Any, ...], tuple[Any, ...]],
        inflight: set[tuple[Any, ...]],
        key: tuple[Any, ...],
    ) -> bool:
        with self._lock:
            if key in items or key in inflight:
                return False
            inflight.add(key)
            return True

    def _neighbor_indices(self, index: int, total: int) -> list[int]:
        indices: list[int] = []
        for offset in range(1, self.preload_radius + 1):
            ahead = index + offset
            behind = index - offset
            if ahead < total:
                indices.append(ahead)
            if behind >= 0:
                indices.append(behind)
        return indices

    @staticmethod
    def _sample_key(sample: dict[str, Any]) -> tuple[Any, ...]:
        return (
            sample.get("_manifest_dir"),
            sample.get("sample_id"),
            sample.get("_manifest_index"),
        )


def build_demo(
    samples: list[dict[str, Any]],
    status: str | None = None,
    *,
    profile_render: bool = False,
    complementarity_dir: str | Path | None = None,
):
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - depends on optional package availability
        raise RuntimeError(
            "Gradio is not installed. Install viewer dependencies with: "
            "conda run -n kd_mm_beam python -m pip install -r tools/visualization/requirements_viewer.txt"
        ) from exc

    scenes = get_available_scenes(samples)
    splits = get_available_splits(samples)
    initial_scene = scenes[0]
    initial_split = splits[0]
    initial_mode = SHOW_MODES[0]
    sample_index = FilteredSampleIndex(samples)
    initial_filtered = sample_index.filtered_samples(initial_scene, initial_split, initial_mode)
    slider_max = _slider_max(len(initial_filtered))
    horizon_choices = _horizon_choices_for_samples(samples)
    render_cache = _SampleRenderCache()
    complementarity = load_complementarity_explorer(complementarity_dir)
    complementarity_cases = complementarity["cases"]
    complementarity_choices = complementarity["choices"]
    complementarity_defaults = complementarity_choices["defaults"]
    initial_complementarity = filter_complementarity_cases(
        complementarity_cases,
        scene=complementarity_defaults["scene"],
        horizon=complementarity_defaults["horizon"],
        strong_modality=complementarity_defaults["strong_modality"],
        weak_modality=complementarity_defaults["weak_modality"],
        case_types=complementarity_defaults["case_types"],
        bucket=complementarity_defaults["bucket"],
        sort_by=complementarity_defaults["sort"],
        max_rows=200,
    )

    with gr.Blocks(title="Multimodal Fusion Temporal Sample Viewer") as demo:
        gr.Markdown("# Multimodal Fusion Temporal Sample Viewer")
        if status:
            gr.Markdown(status)
        with gr.Row():
            scene_dropdown = gr.Dropdown(choices=scenes, value=initial_scene, label="Scene")
            split_dropdown = gr.Dropdown(choices=splits, value=initial_split, label="Split")
            show_mode_dropdown = gr.Dropdown(choices=list(SHOW_MODES), value=initial_mode, label="Show Mode")
        with gr.Row():
            sample_slider = gr.Slider(
                minimum=0,
                maximum=slider_max,
                step=1,
                value=0,
                label="Sample Index",
                interactive=True,
            )
            prev_btn = gr.Button("Prev")
            next_btn = gr.Button("Next")
            play_checkbox = gr.Checkbox(label="Auto Play", value=False)
            speed_dropdown = gr.Dropdown(choices=[1, 2, 5], value=1, label="Speed")

        sample_text = gr.Markdown("No samples found" if not samples else "Ready")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Raw Modalities")
                raw_image = gr.Image(label="Raw Image", type="pil")
                raw_lidar = gr.Image(label="Raw LiDAR Points", type="pil")
                raw_radar = gr.Image(label="Raw / Precomputed Radar", type="pil")
                raw_gps = gr.Plot(label="Raw GPS")
                raw_mmwave = gr.Plot(label="Raw mmWave")
            with gr.Column(scale=2):
                gr.Markdown("### Processed Modalities")
                proc_image = gr.Image(label="Processed Image", type="pil")
                proc_lidar = gr.Image(label="Processed LiDAR", type="pil")
                proc_radar = gr.Image(label="Processed Radar", type="pil")
                proc_gps = gr.Plot(label="Processed GPS")
                proc_mmwave = gr.Plot(label="Processed mmWave")
            with gr.Column(scale=2):
                gr.Markdown("### Diagnostics")
                info_json = gr.JSON(label="Sample / Label / Prediction")
                beam_index_trend_plot = gr.Plot(label="Future Beam Index Trend (+/-30)")
                beam_confidence_plot = gr.Plot(label="Future Beam Labels / Confidence Curves")
                confidence_plot = gr.Plot(label="Confidence (t+1)")
                quality_plot = gr.Plot(label="Quality")
                gate_plot = gr.Plot(label="Gate")
                with gr.Accordion("Future Beam Distribution Inspector", open=True):
                    with gr.Row():
                        future_horizon_dropdown = gr.Dropdown(
                            choices=horizon_choices,
                            value=horizon_choices[0],
                            label="Future Horizon",
                        )
                        distribution_view_dropdown = gr.Dropdown(
                            choices=["probability", "logit"],
                            value="probability",
                            label="Distribution View",
                        )
                        distribution_chart_dropdown = gr.Dropdown(
                            choices=["heatmap", "per_modality"],
                            value="heatmap",
                            label="Chart Type",
                        )
                        show_fusion_checkbox = gr.Checkbox(value=True, label="Show Fusion")
                    future_distribution_plot = gr.Plot(label="Future Beam Distribution")
                    future_distribution_summary_df = gr.Dataframe(
                        label="Future Beam Distribution Summary",
                        interactive=False,
                    )
                    future_distribution_detail_json = gr.JSON(label="Selected Horizon Detail")
                confidence_df = gr.Dataframe(label="Confidence Table (t+1)", interactive=False)
                quality_df = gr.Dataframe(label="Quality Table", interactive=False)
                gate_df = gr.Dataframe(label="Gate Table", interactive=False)

        canonical_outputs = [
            raw_image,
            raw_lidar,
            raw_radar,
            raw_gps,
            raw_mmwave,
            proc_image,
            proc_lidar,
            proc_radar,
            proc_gps,
            proc_mmwave,
            info_json,
            beam_confidence_plot,
            confidence_plot,
            quality_plot,
            gate_plot,
            future_distribution_plot,
            future_distribution_summary_df,
            future_distribution_detail_json,
            confidence_df,
            quality_df,
            gate_df,
            beam_index_trend_plot,
            sample_text,
        ]
        all_outputs = canonical_outputs

        with gr.Tabs():
            with gr.Tab("Complementarity Explorer"):
                gr.Markdown("## Complementarity Explorer / 弱模态互补样本分析")
                gr.Markdown(
                    "Potential complementarity: strong-only wrong and weak modality correct. "
                    "Rescue: fusion correct on that subset. Unused complementary: fusion still wrong. "
                    "Negative transfer: strong-only correct but fusion wrong."
                )
                gr.Markdown(complementarity["status"])
                with gr.Row():
                    comp_scene_dropdown = gr.Dropdown(
                        choices=complementarity_choices["scenes"],
                        value=complementarity_defaults["scene"],
                        label="Scene",
                    )
                    comp_horizon_dropdown = gr.Dropdown(
                        choices=complementarity_choices["horizons"],
                        value=complementarity_defaults["horizon"],
                        label="Horizon",
                    )
                    comp_strong_dropdown = gr.Dropdown(
                        choices=complementarity_choices["strong_modalities"],
                        value=complementarity_defaults["strong_modality"],
                        label="Strong Modality",
                    )
                    comp_weak_dropdown = gr.Dropdown(
                        choices=complementarity_choices["weak_modalities"],
                        value=complementarity_defaults["weak_modality"],
                        label="Weak Modality",
                    )
                with gr.Row():
                    comp_case_dropdown = gr.Dropdown(
                        choices=complementarity_choices["case_types"],
                        value=complementarity_defaults["case_types"],
                        multiselect=True,
                        label="Case / Tag",
                    )
                    comp_bucket_dropdown = gr.Dropdown(
                        choices=complementarity_choices["buckets"],
                        value=complementarity_defaults["bucket"],
                        label="Bucket",
                    )
                    comp_min_gain = gr.Number(value=None, label="Min Gain")
                    comp_sort_dropdown = gr.Dropdown(
                        choices=complementarity_choices["sort"],
                        value=complementarity_defaults["sort"],
                        label="Sort",
                    )
                    comp_max_rows = gr.Slider(
                        minimum=10,
                        maximum=1000,
                        step=10,
                        value=200,
                        label="Max Rows",
                    )
                    comp_apply_btn = gr.Button("Apply filters")
                with gr.Row():
                    comp_stats_json = gr.JSON(value=initial_complementarity["stats"], label="Filtered Statistics")
                    comp_case_plot = gr.Plot(value=initial_complementarity["case_type_figure"], label="Case Type Counts")
                    comp_bucket_plot = gr.Plot(value=initial_complementarity["bucket_figure"], label="Bucket Counts")
                comp_table = gr.Dataframe(
                    value=initial_complementarity["table"],
                    label="Complementarity Cases",
                    interactive=False,
                )
                comp_filtered_state = gr.State(initial_complementarity["records"])
                with gr.Row():
                    comp_export_btn = gr.Button("Export filtered CSV")
                    comp_export_file = gr.File(label="Filtered CSV")
                comp_detail_json = gr.JSON(label="Selected Case Detail")
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Raw Modalities")
                        comp_raw_image = gr.Image(label="Raw Image", type="pil")
                        comp_raw_lidar = gr.Image(label="Raw LiDAR Points", type="pil")
                        comp_raw_radar = gr.Image(label="Raw / Precomputed Radar", type="pil")
                        comp_raw_gps = gr.Plot(label="Raw GPS")
                        comp_raw_mmwave = gr.Plot(label="Raw mmWave")
                    with gr.Column(scale=2):
                        gr.Markdown("### Processed Modalities")
                        comp_proc_image = gr.Image(label="Processed Image", type="pil")
                        comp_proc_lidar = gr.Image(label="Processed LiDAR", type="pil")
                        comp_proc_radar = gr.Image(label="Processed Radar", type="pil")
                        comp_proc_gps = gr.Plot(label="Processed GPS")
                        comp_proc_mmwave = gr.Plot(label="Processed mmWave")
                    with gr.Column(scale=2):
                        gr.Markdown("### Diagnostics")
                        comp_info_json = gr.JSON(label="Sample / Label / Prediction")
                        comp_beam_confidence_plot = gr.Plot(label="Future Beam Labels / Confidence Curves")
                        comp_confidence_plot = gr.Plot(label="Confidence (t+1)")
                        comp_quality_plot = gr.Plot(label="Quality")
                        comp_gate_plot = gr.Plot(label="Gate")
                        comp_future_distribution_plot = gr.Plot(label="Future Beam Distribution")
                        comp_future_distribution_summary_df = gr.Dataframe(
                            label="Future Beam Distribution Summary",
                            interactive=False,
                        )
                        comp_future_distribution_detail_json = gr.JSON(label="Selected Horizon Detail")
                        comp_confidence_df = gr.Dataframe(label="Confidence Table (t+1)", interactive=False)
                        comp_quality_df = gr.Dataframe(label="Quality Table", interactive=False)
                        comp_gate_df = gr.Dataframe(label="Gate Table", interactive=False)
                        comp_beam_index_trend_plot = gr.Plot(label="Future Beam Index Trend (+/-30)")
                        comp_sample_text = gr.Markdown("Select a complementarity case")

                complementarity_sample_outputs = [
                    comp_raw_image,
                    comp_raw_lidar,
                    comp_raw_radar,
                    comp_raw_gps,
                    comp_raw_mmwave,
                    comp_proc_image,
                    comp_proc_lidar,
                    comp_proc_radar,
                    comp_proc_gps,
                    comp_proc_mmwave,
                    comp_info_json,
                    comp_beam_confidence_plot,
                    comp_confidence_plot,
                    comp_quality_plot,
                    comp_gate_plot,
                    comp_future_distribution_plot,
                    comp_future_distribution_summary_df,
                    comp_future_distribution_detail_json,
                    comp_confidence_df,
                    comp_quality_df,
                    comp_gate_df,
                    comp_beam_index_trend_plot,
                    comp_sample_text,
                ]

        def render_all(index, scene, split, show_mode, horizon, view_type, chart_type, show_fusion):
            stats = RenderStats()
            started = time.perf_counter()
            outputs = render_sample(
                samples,
                index,
                scene,
                split,
                show_mode,
                horizon,
                view_type,
                chart_type,
                show_fusion,
                render_cache=render_cache,
                sample_index=sample_index,
                stats=stats,
            )
            _log_profile(
                profile_render,
                "load",
                stats,
                started,
                filtered_count=_filtered_count(sample_index, scene, split, show_mode),
                target_index=clamp_index(index, _filtered_count(sample_index, scene, split, show_mode)),
                returned_component_count=_returned_component_count(outputs),
            )
            return outputs

        def update_filter(scene, split, show_mode, horizon, view_type, chart_type, show_fusion):
            stats = RenderStats()
            started = time.perf_counter()
            filtered = sample_index.filtered_samples(scene, split, show_mode, stats=stats)
            maximum = _slider_max(len(filtered))
            outputs = render_sample(
                samples,
                0,
                scene,
                split,
                show_mode,
                horizon,
                view_type,
                chart_type,
                show_fusion,
                render_cache=render_cache,
                sample_index=sample_index,
                stats=stats,
            )
            result = (gr.update(maximum=maximum, value=0), *outputs)
            _log_profile(
                profile_render,
                "filter",
                stats,
                started,
                filtered_count=len(filtered),
                target_index=0,
                returned_component_count=_returned_component_count(result),
            )
            return result

        def go_prev(current_index, scene, split, show_mode, horizon, view_type, chart_type, show_fusion):
            stats = RenderStats()
            started = time.perf_counter()
            filtered = sample_index.filtered_samples(scene, split, show_mode, stats=stats)
            index = max(0, clamp_index(current_index, len(filtered)) - 1)
            outputs = render_sample(
                samples,
                index,
                scene,
                split,
                show_mode,
                horizon,
                view_type,
                chart_type,
                show_fusion,
                render_cache=render_cache,
                sample_index=sample_index,
                stats=stats,
            )
            result = (gr.update(value=index), *outputs)
            _log_profile(
                profile_render,
                "prev",
                stats,
                started,
                filtered_count=len(filtered),
                target_index=index,
                returned_component_count=_returned_component_count(result),
            )
            return result

        def go_next(current_index, scene, split, show_mode, horizon, view_type, chart_type, show_fusion):
            stats = RenderStats()
            started = time.perf_counter()
            filtered = sample_index.filtered_samples(scene, split, show_mode, stats=stats)
            index = min(max(0, len(filtered) - 1), clamp_index(current_index, len(filtered)) + 1)
            outputs = render_sample(
                samples,
                index,
                scene,
                split,
                show_mode,
                horizon,
                view_type,
                chart_type,
                show_fusion,
                render_cache=render_cache,
                sample_index=sample_index,
                stats=stats,
            )
            result = (gr.update(value=index), *outputs)
            _log_profile(
                profile_render,
                "next",
                stats,
                started,
                filtered_count=len(filtered),
                target_index=index,
                returned_component_count=_returned_component_count(result),
            )
            return result

        def autoplay_step(
            current_index,
            play,
            speed,
            scene,
            split,
            show_mode,
            horizon,
            view_type,
            chart_type,
            show_fusion,
        ):
            stats = RenderStats()
            started = time.perf_counter()
            filtered = sample_index.filtered_samples(scene, split, show_mode, stats=stats)
            if not filtered:
                outputs = render_sample(
                    samples,
                    0,
                    scene,
                    split,
                    show_mode,
                    horizon,
                    view_type,
                    chart_type,
                    show_fusion,
                    render_cache=render_cache,
                    sample_index=sample_index,
                    stats=stats,
                )
                result = (gr.update(value=0, maximum=_slider_max(0)), *outputs)
                _log_profile(
                    profile_render,
                    "timer",
                    stats,
                    started,
                    filtered_count=0,
                    target_index=0,
                    returned_component_count=_returned_component_count(result),
                )
                return result
            if not play:
                index = clamp_index(current_index, len(filtered))
            else:
                index = (clamp_index(current_index, len(filtered)) + int(speed or 1)) % len(filtered)
            outputs = render_sample(
                samples,
                index,
                scene,
                split,
                show_mode,
                horizon,
                view_type,
                chart_type,
                show_fusion,
                render_cache=render_cache,
                sample_index=sample_index,
                stats=stats,
            )
            result = (gr.update(value=index, maximum=_slider_max(len(filtered))), *outputs)
            _log_profile(
                profile_render,
                "timer",
                stats,
                started,
                filtered_count=len(filtered),
                target_index=index,
                returned_component_count=_returned_component_count(result),
            )
            return result

        def apply_complementarity_filters(
            comp_scene,
            comp_horizon,
            comp_strong,
            comp_weak,
            comp_cases,
            comp_bucket,
            comp_gain,
            comp_sort,
            comp_limit,
        ):
            result = filter_complementarity_cases(
                complementarity_cases,
                scene=comp_scene,
                horizon=comp_horizon,
                strong_modality=comp_strong,
                weak_modality=comp_weak,
                case_types=comp_cases,
                bucket=comp_bucket,
                min_gain=comp_gain,
                sort_by=comp_sort,
                max_rows=comp_limit,
            )
            return (
                result["stats"],
                result["case_type_figure"],
                result["bucket_figure"],
                result["table"],
                result["records"],
            )

        def export_complementarity_filters(records):
            root = complementarity.get("root")
            output_dir = Path(root) / "exports" if root else None
            return export_filtered_cases(records, output_dir=output_dir)

        def select_complementarity_case(current_table, evt=None):
            row = selected_event_row(evt, current_table)
            if row is None:
                return (case_detail_payload(None), *_empty_outputs("No complementarity case selected"))
            sample_position = find_sample_index_for_case(samples, row)
            if sample_position is None:
                return (case_detail_payload(row, None), *_empty_outputs("Manifest sample not found"))
            sample = samples[sample_position]
            outputs = render_sample(
                samples,
                sample_position,
                "all",
                "all",
                "all",
                row.get("horizon_name", "t+1"),
                "probability",
                "heatmap",
                True,
                render_cache=render_cache,
                sample_index=None,
            )
            return (case_detail_payload(row, sample), *outputs)
        select_data_cls = getattr(gr, "SelectData", None)
        if select_data_cls is not None:
            select_complementarity_case.__annotations__["evt"] = select_data_cls

        future_inputs = [
            future_horizon_dropdown,
            distribution_view_dropdown,
            distribution_chart_dropdown,
            show_fusion_checkbox,
        ]
        render_inputs = [sample_slider, scene_dropdown, split_dropdown, show_mode_dropdown, *future_inputs]
        filter_inputs = [scene_dropdown, split_dropdown, show_mode_dropdown, *future_inputs]
        nav_inputs = [sample_slider, scene_dropdown, split_dropdown, show_mode_dropdown, *future_inputs]
        sample_slider.change(render_all, inputs=render_inputs, outputs=all_outputs)
        scene_dropdown.change(update_filter, inputs=filter_inputs, outputs=[sample_slider, *all_outputs])
        split_dropdown.change(update_filter, inputs=filter_inputs, outputs=[sample_slider, *all_outputs])
        show_mode_dropdown.change(update_filter, inputs=filter_inputs, outputs=[sample_slider, *all_outputs])
        future_horizon_dropdown.change(render_all, inputs=render_inputs, outputs=all_outputs)
        distribution_view_dropdown.change(render_all, inputs=render_inputs, outputs=all_outputs)
        distribution_chart_dropdown.change(render_all, inputs=render_inputs, outputs=all_outputs)
        show_fusion_checkbox.change(render_all, inputs=render_inputs, outputs=all_outputs)
        prev_btn.click(go_prev, inputs=nav_inputs, outputs=[sample_slider, *all_outputs])
        next_btn.click(go_next, inputs=nav_inputs, outputs=[sample_slider, *all_outputs])
        demo.load(render_all, inputs=render_inputs, outputs=all_outputs)

        comp_filter_inputs = [
            comp_scene_dropdown,
            comp_horizon_dropdown,
            comp_strong_dropdown,
            comp_weak_dropdown,
            comp_case_dropdown,
            comp_bucket_dropdown,
            comp_min_gain,
            comp_sort_dropdown,
            comp_max_rows,
        ]
        comp_apply_btn.click(
            apply_complementarity_filters,
            inputs=comp_filter_inputs,
            outputs=[
                comp_stats_json,
                comp_case_plot,
                comp_bucket_plot,
                comp_table,
                comp_filtered_state,
            ],
        )
        comp_export_btn.click(
            export_complementarity_filters,
            inputs=comp_filtered_state,
            outputs=comp_export_file,
        )
        if hasattr(comp_table, "select"):
            comp_table.select(
                select_complementarity_case,
                inputs=comp_table,
                outputs=[comp_detail_json, *complementarity_sample_outputs],
            )

        timer = _make_timer(gr)
        if timer is not None:
            def update_timer_active(play):
                return gr.update(active=bool(play))

            play_checkbox.change(update_timer_active, inputs=play_checkbox, outputs=timer)
            timer.tick(
                autoplay_step,
                inputs=[
                    sample_slider,
                    play_checkbox,
                    speed_dropdown,
                    scene_dropdown,
                    split_dropdown,
                    show_mode_dropdown,
                    *future_inputs,
                ],
                outputs=[sample_slider, *all_outputs],
            )

    return demo


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    manifest_path: str | Path
    prepare_result: dict[str, Any] | None = None
    prediction_result: dict[str, Any] | None = None
    if args.config:
        cfg = load_cli_config(args, unknown)
        predictions = args.predictions
        multi_scene = parse_visualization_config(cfg).compare_scenes is not None
        if args.run_models:
            print(
                "[viewer] Exporting single-modality predictions "
                f"(devices={args.model_devices}, workers={args.model_workers or 'parallel'}, batch_size={args.model_batch_size})",
                flush=True,
            )
            prediction_result = export_viewer_model_predictions(
                cfg,
                output_path=args.predictions,
                cache_dir=args.cache_dir,
                modalities=parse_modalities(args.prediction_modalities),
                model_config_paths=parse_key_value_paths(args.model_config),
                checkpoint_paths=parse_key_value_paths(args.model_checkpoint),
                devices=args.model_devices,
                workers=args.model_workers,
                batch_size=args.model_batch_size,
                num_workers=args.model_num_workers,
                force_rebuild=bool(args.force_rebuild),
                sample_limit=args.sample_limit,
            )
            predictions = prediction_result["prediction_path"]
            print(f"[viewer] Model predictions ready: {predictions}", flush=True)
        elif not predictions and not args.no_auto_predictions and not multi_scene:
            predictions = _latest_cached_predictions(args.cache_dir)
            if predictions is not None:
                prediction_result = _cached_prediction_summary(predictions)
                print(f"[viewer] Reusing cached model predictions: {predictions}", flush=True)
        print("[viewer] Preparing viewer manifest/cache", flush=True)
        prepare_result = export_viewer_manifest(
            cfg,
            output_path=args.manifest,
            cache_dir=args.cache_dir,
            predictions=predictions,
            overwrite=False,
            force_rebuild=bool(args.force_rebuild),
            sample_limit=args.sample_limit,
        )
        manifest_path = prepare_result["manifest_path"]
        print(f"[viewer] Viewer manifest ready: {manifest_path}", flush=True)
    else:
        manifest_path = args.manifest or DEFAULT_MANIFEST
    samples = load_manifest(manifest_path, project_root=args.project_root)
    status = _status_markdown(prepare_result) if prepare_result is not None else f"`manifest`: `{manifest_path}`"
    if prediction_result is not None:
        status = f"{status} | `model predictions`: `{prediction_result.get('prediction_path')}`"
    demo = build_demo(
        samples,
        status=status,
        profile_render=bool(args.profile_render),
        complementarity_dir=args.complementarity_dir,
    )
    result = {
        "status": "complete",
        "manifest": str(manifest_path),
        "sample_count": len(samples),
        "complementarity_dir": args.complementarity_dir,
        "host": args.host,
        "port": int(args.port),
        "share": bool(args.share),
    }
    if prepare_result is not None:
        result.update(
            {
                "cache_hit": prepare_result.get("cache_hit"),
                "cache_dir": prepare_result.get("cache_dir"),
                "meta_path": prepare_result.get("meta_path"),
            }
        )
    if prediction_result is not None:
        result["model_predictions"] = {
            "prediction_path": prediction_result.get("prediction_path"),
            "cache_hit": prediction_result.get("cache_hit"),
            "sample_count": prediction_result.get("sample_count"),
            "modalities": prediction_result.get("modalities"),
            "workers": prediction_result.get("workers"),
            "requested_devices": prediction_result.get("requested_devices"),
            "resolved_devices": prediction_result.get("resolved_devices"),
            "cuda_available": prediction_result.get("cuda_available"),
            "cuda_device_count": prediction_result.get("cuda_device_count"),
        }
    if args.check_only:
        print(json.dumps(result, indent=2))
        return result
    demo.launch(server_name=args.host, server_port=int(args.port), share=bool(args.share), debug=bool(args.debug))
    return result


def _load_image(
    sample: dict[str, Any],
    path_key: str,
    manifest_dir: str | None,
    project_root: str | None,
    *,
    stats: RenderStats | None = None,
) -> Any:
    resolved = resolve_path(safe_get(sample, path_key), manifest_dir=manifest_dir, project_root=project_root)
    if resolved is None or not resolved.exists() or not resolved.is_file():
        if stats is not None:
            stats.incr("image_empty")
        return None
    if stats is not None:
        stats.incr("image_path_output")
    return str(resolved)


def _latest_cached_predictions(cache_dir: str | Path | None) -> str | None:
    if cache_dir is None:
        return None
    root = Path(cache_dir).expanduser() / "model_predictions"
    if not root.exists():
        return None
    candidates = [path for path in root.glob("*/predictions.json") if path.is_file()]
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return str(latest)


def _cached_prediction_summary(prediction_path: str | Path) -> dict[str, Any]:
    path = Path(prediction_path).expanduser()
    summary: dict[str, Any] = {
        "mode": "viewer_model_predictions",
        "cache_hit": True,
        "prediction_path": str(path),
    }
    meta_path = path.with_name("model_predictions_meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        if isinstance(meta, dict):
            summary.update({key: meta.get(key) for key in (
                "sample_count",
                "modalities",
                "workers",
                "requested_devices",
                "resolved_devices",
                "cuda_available",
                "cuda_device_count",
            )})
            summary["meta_path"] = str(meta_path)
    return summary


def _filtered_count(sample_index: FilteredSampleIndex, scene: str | None, split: str | None, show_mode: str | None) -> int:
    return len(sample_index.filtered_indices(scene, split, show_mode))


def _returned_component_count(values: tuple[Any, ...]) -> int:
    return sum(0 if _is_skip_value(value) else 1 for value in values)


def _is_skip_value(value: Any) -> bool:
    return isinstance(value, dict) and value == {"__type__": "update"}


def _log_profile(
    enabled: bool,
    event: str,
    stats: RenderStats,
    started: float,
    *,
    filtered_count: int,
    target_index: int,
    returned_component_count: int,
) -> None:
    if not enabled:
        return
    payload = stats.snapshot()
    payload.update(
        {
            "event": event,
            "filtered_count": int(filtered_count),
            "target_index": int(target_index),
            "callback_total_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "returned_component_count": int(returned_component_count),
        }
    )
    print("[viewer-profile] " + json.dumps(payload, sort_keys=True), flush=True)


def _empty_outputs(message: str) -> tuple[Any, ...]:
    return (
        None,
        None,
        None,
        make_empty_figure("Raw GPS"),
        make_empty_figure("Raw mmWave"),
        None,
        None,
        None,
        make_empty_figure("Processed GPS"),
        make_empty_figure("Processed mmWave"),
        {"message": message},
        make_empty_figure("Future Beam Label Confidence"),
        make_empty_figure("Single-Modality Confidence (t+1)"),
        make_empty_figure("Modality Quality"),
        make_empty_figure("Gate Weight"),
        make_empty_figure("Future Beam Distribution Not Available"),
        make_future_distribution_summary(None, "t+1", "probability"),
        {"message": message},
        dict_to_dataframe(None, "confidence"),
        dict_to_dataframe(None, "quality"),
        dict_to_dataframe(None, "gate"),
        make_empty_figure("Future Beam Index Trend (+/-30)"),
        message,
    )


def _make_timer(gr):
    timer_cls = getattr(gr, "Timer", None)
    if timer_cls is None:
        return None
    return timer_cls(value=1.0, active=False)


def _slider_max(sample_count: int) -> int:
    return max(1, int(sample_count) - 1)


def _horizon_choices_for_samples(samples: list[dict[str, Any]]) -> list[str]:
    max_horizon = 0
    for sample in samples:
        max_horizon = max(max_horizon, len(get_future_beams(sample)))
    if max_horizon <= 0:
        return ["t+1"]
    return [f"t+{index + 1}" for index in range(max_horizon)]


def _status_markdown(result: dict[str, Any]) -> str:
    state = "reused" if result.get("cache_hit") else "processed"
    return (
        f"`dataset cache`: `{state}` | `samples`: `{result.get('sample_count')}` | "
        f"`manifest`: `{result.get('manifest_path')}`"
    )


if __name__ == "__main__":
    main()
