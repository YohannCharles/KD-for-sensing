import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from kd_sensing.config.io import load_config
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.data.difficulty import (
    DifficultyContext,
    apply_difficulty_pipeline,
    normalize_difficulty_profiles,
)
from kd_sensing.data.difficulty.presets import (
    PREDICTIVE_JEPA_CANONICAL_CONDITIONS,
    PREDICTIVE_JEPA_CONDITION_IDS,
    PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
    SCENARIO_D_CANONICAL_CONDITIONS,
    SCENARIO_D_CONDITION_IDS,
    SCENARIO_D_SUITE_TYPE,
    normalize_predictive_jepa_condition_id,
    normalize_predictive_jepa_operator_params,
    normalize_scenario_d_condition_id,
    normalize_scenario_d_operator_params,
    predictive_jepa_condition,
    scenario_d_condition,
)
from kd_sensing.evaluation.metrics import calculate_dba_score, calculate_topk_accuracy
from kd_sensing.utils.artifact_registry import load_checkpoint_metadata
from kd_sensing.utils.paths import resolve_path


from kd_sensing.diagnostics.jepa_benchmark_artifacts import OutputRegistry, _output_formats
from kd_sensing.diagnostics.jepa_benchmark_common import *


def _write_cxd_phase_figures(
    plots_dir: Path,
    phase_rows: list[dict[str, Any]],
    dominance_rows: list[dict[str, Any]],
    crossing: Mapping[str, Any],
    manifest: Mapping[str, Any],
    registry: OutputRegistry,
    warnings: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.append(WarningRecord(code="matplotlib_unavailable", message=str(exc)).to_dict())
        for name in ("cxd_accuracy_heatmap", "resnet_jepa_crossover_curve", "modality_dominance_heatmap"):
            registry.skipped_output(plots_dir / f"{name}.png", reason="matplotlib_unavailable", kind="figure")
        return
    dpi = int(manifest.get("figures", {}).get("dpi", 180)) if isinstance(manifest.get("figures"), Mapping) else 180
    if phase_rows:
        _plot_cxd_accuracy_heatmap(plots_dir / "cxd_accuracy_heatmap.png", phase_rows, dpi=dpi, plt=plt)
    else:
        registry.skipped_output(plots_dir / "cxd_accuracy_heatmap.png", reason="no_cxd_phase_rows", kind="figure")
    conditions = list(crossing.get("conditions", [])) if isinstance(crossing, Mapping) else []
    if conditions:
        _plot_resnet_jepa_crossover_curve(plots_dir / "resnet_jepa_crossover_curve.png", conditions, dpi=dpi, plt=plt)
    else:
        registry.skipped_output(plots_dir / "resnet_jepa_crossover_curve.png", reason="no_crossing_rows", kind="figure")
    if dominance_rows:
        _plot_modality_dominance_heatmap(plots_dir / "modality_dominance_heatmap.png", dominance_rows, dpi=dpi, plt=plt)
    else:
        registry.skipped_output(plots_dir / "modality_dominance_heatmap.png", reason="no_dominance_rows", kind="figure")


def _plot_cxd_accuracy_heatmap(path: Path, rows: list[dict[str, Any]], *, dpi: int, plt: Any) -> None:
    matrix = np.full((len(CXD_GPS_CONDITION_IDS), len(CXD_IMAGE_CONDITION_IDS)), np.nan, dtype=np.float32)
    buckets: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        value = _float_or_none(row.get("primary_metric"))
        if value is None:
            continue
        gps_index = _condition_index(str(row.get("gps_condition")), CXD_GPS_CONDITION_IDS)
        image_index = _condition_index(str(row.get("image_condition")), CXD_IMAGE_CONDITION_IDS)
        if gps_index >= len(CXD_GPS_CONDITION_IDS) or image_index >= len(CXD_IMAGE_CONDITION_IDS):
            continue
        buckets.setdefault((gps_index, image_index), []).append(value)
    for (gps_index, image_index), values in buckets.items():
        matrix[gps_index, image_index] = float(np.mean(values))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(CXD_IMAGE_CONDITION_IDS)))
    ax.set_xticklabels(CXD_IMAGE_CONDITION_IDS, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(CXD_GPS_CONDITION_IDS)))
    ax.set_yticklabels(CXD_GPS_CONDITION_IDS, fontsize=7)
    ax.set_xlabel("image condition")
    ax.set_ylabel("GPS condition")
    ax.set_title("CxD accuracy heatmap")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_resnet_jepa_crossover_curve(path: Path, conditions: list[Mapping[str, Any]], *, dpi: int, plt: Any) -> None:
    materialized = [dict(item) for item in conditions if _float_or_none(item.get("metric_margin")) is not None]
    materialized.sort(key=_crossing_condition_rank)
    x = list(range(len(materialized)))
    y = [_float(item.get("metric_margin")) for item in materialized]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(x, y, marker="o", linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.8)
    labels = [str(item.get("condition_id", "")) for item in materialized]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6)
    ax.set_ylabel("JEPA - Image ResNet metric")
    ax.set_title("Image ResNet/JEPA crossover")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_modality_dominance_heatmap(path: Path, rows: list[dict[str, Any]], *, dpi: int, plt: Any) -> None:
    matrix = np.full((len(CXD_GPS_CONDITION_IDS), len(CXD_IMAGE_CONDITION_IDS)), np.nan, dtype=np.float32)
    buckets: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        value = _float_or_none(row.get("image_contribution_score"))
        if value is None:
            value = _float_or_none(row.get("gps_contribution_score"))
            if value is not None:
                value = 1.0 - value
        if value is None:
            continue
        gps_index = _condition_index(str(row.get("gps_condition")), CXD_GPS_CONDITION_IDS)
        image_index = _condition_index(str(row.get("image_condition")), CXD_IMAGE_CONDITION_IDS)
        if gps_index >= len(CXD_GPS_CONDITION_IDS) or image_index >= len(CXD_IMAGE_CONDITION_IDS):
            continue
        buckets.setdefault((gps_index, image_index), []).append(value)
    for (gps_index, image_index), values in buckets.items():
        matrix[gps_index, image_index] = float(np.mean(values))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(CXD_IMAGE_CONDITION_IDS)))
    ax.set_xticklabels(CXD_IMAGE_CONDITION_IDS, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(CXD_GPS_CONDITION_IDS)))
    ax.set_yticklabels(CXD_GPS_CONDITION_IDS, fontsize=7)
    ax.set_xlabel("image condition")
    ax.set_ylabel("GPS condition")
    ax.set_title("Image contribution diagnostic")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _write_scenario_d_figures(
    plots_dir: Path,
    rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    registry: OutputRegistry,
    warnings: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.append(WarningRecord(code="matplotlib_unavailable", message=str(exc)).to_dict())
        for name in ("robustness_surface", "phase_transition_curve", "modality_dominance"):
            registry.skipped_output(plots_dir / f"{name}.png", reason="matplotlib_unavailable", kind="figure")
        return
    dpi = int(manifest.get("figures", {}).get("dpi", 180)) if isinstance(manifest.get("figures"), Mapping) else 180
    _plot_robustness_surface(plots_dir / "robustness_surface.png", rows, dpi=dpi, plt=plt)
    _plot_phase_transition(plots_dir / "phase_transition_curve.png", rows, dpi=dpi, plt=plt)
    _plot_modality_dominance(plots_dir / "modality_dominance.png", rows, dpi=dpi, plt=plt)


def _plot_robustness_surface(path: Path, rows: list[dict[str, Any]], *, dpi: int, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row.get("model")), []).append(row)
    for model, model_rows in sorted(by_model.items()):
        model_rows.sort(key=lambda item: (float(item.get("c_severity") or 0.0), float(item.get("d_severity") or 0.0)))
        y = [_float(item.get("primary_metric")) for item in model_rows]
        x = list(range(len(y)))
        ax.plot(x, y, marker="o", linewidth=1.2, label=model)
    ax.set_title("Scenario D robustness surface")
    ax.set_xlabel("Cx-Dy condition index")
    ax.set_ylabel("primary metric")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_phase_transition(path: Path, rows: list[dict[str, Any]], *, dpi: int, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row.get("model")), []).append(row)
    for model, model_rows in sorted(by_model.items()):
        model_rows.sort(key=lambda item: float(item.get("d_severity") or 0.0))
        x = [float(row.get("d_severity") or 0.0) for row in model_rows]
        y = [_float(row.get("clean_delta")) for row in model_rows]
        ax.plot(x, y, marker=".", linewidth=1.0, label=model)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Phase transition curve")
    ax.set_xlabel("Scenario D severity")
    ax.set_ylabel("clean delta")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_modality_dominance(path: Path, rows: list[dict[str, Any]], *, dpi: int, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    by_model: dict[str, list[float]] = {}
    for row in rows:
        value = _float_or_none(row.get("modality_dominance_ratio"))
        if value is None:
            continue
        by_model.setdefault(str(row.get("model")), []).append(value)
    labels = sorted(by_model)
    values = [float(np.mean(by_model[label])) if by_model[label] else 0.0 for label in labels]
    ax.bar(labels, values)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Modality dominance")
    ax.set_ylabel("image dominance ratio")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _write_benchmark_figures(
    figures_dir: Path,
    metrics_rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    registry: OutputRegistry,
    warnings: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.append(WarningRecord(code="matplotlib_unavailable", message=str(exc)).to_dict())
        registry.skipped_output(figures_dir / "benchmark_curves.png", reason="matplotlib_unavailable", kind="figure")
        return
    groups = {
        "gps_collapse_curve": lambda row: str(row.get("suite_type")) in GPS_SUITE_TYPES and str(row.get("condition")) != "clean",
        "image_degradation_curve": lambda row: str(row.get("suite_type")) in IMAGE_SUITE_TYPES,
        "temporal_delay_curve": lambda row: str(row.get("suite_type")) in TEMPORAL_SUITE_TYPES,
    }
    formats = _output_formats(manifest)
    for name, predicate in groups.items():
        rows = [row for row in metrics_rows if predicate(row)]
        if not rows:
            registry.skipped_output(figures_dir / f"{name}.png", reason="no_matching_rows", kind="figure")
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        by_model: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_model.setdefault(str(row.get("model")), []).append(row)
        for model, model_rows in sorted(by_model.items()):
            model_rows.sort(key=lambda item: float(item.get("severity") or 0.0))
            x = [float(row.get("severity") or 0.0) for row in model_rows]
            y = [float(row.get("primary_metric") or 0.0) for row in model_rows]
            ax.plot(x, y, marker="o", label=model)
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("severity")
        ax.set_ylabel(str(manifest.get("metrics", {}).get("primary", DEFAULT_PRIMARY_METRIC)))
        ax.legend(fontsize=7)
        fig.tight_layout()
        for fmt in formats:
            fig.savefig(figures_dir / f"{name}.{fmt}", dpi=int(manifest.get("figures", {}).get("dpi", 180)), bbox_inches="tight")
        plt.close(fig)


__all__ = [
    "_plot_cxd_accuracy_heatmap",
    "_plot_modality_dominance",
    "_plot_modality_dominance_heatmap",
    "_plot_phase_transition",
    "_plot_resnet_jepa_crossover_curve",
    "_plot_robustness_surface",
    "_write_benchmark_figures",
    "_write_cxd_phase_figures",
    "_write_scenario_d_figures",
]
