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
from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.engine.data_factory import build_dataset, prepare_lidar_normalizer
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.engine.run_metadata import dataset_run_metadata
from kd_sensing.modalities import MODALITY_ORDER, dataset_flags_for_modalities, normalize_modalities
from kd_sensing.utils.paths import resolve_path


VALID_MODALITIES = MODALITY_ORDER
VALID_SPLITS = ("train", "test")
METADATA_FILE_TEMPLATES = {
    "summary": ("summary", ".json"),
    "samples_jsonl": ("samples", ".jsonl"),
    "samples_csv": ("samples", ".csv"),
    "split_stats": ("split_stats", ".json"),
    "final_config": ("final_config", ".yaml"),
}


@dataclass(frozen=True)
class VisualizationConfig:
    output_dir: Path
    splits: tuple[str, ...]
    sample_count: int
    per_seq_sample_count: int | None
    seed: int
    seq_index: tuple[Any, ...] | None
    labels: tuple[int, ...] | None
    modalities: tuple[str, ...] | None
    compare_scenes: tuple[int, ...] | None
    max_frames_per_sample: int
    include_raw_image_preview: bool
    preserve_existing_outputs: bool

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return _json_ready(payload)


@dataclass(frozen=True)
class SampleCandidate:
    dataset_index: int
    csv_row_index: int
    seq_index: Any
    future_label: int | None


def visualize_modalities(cfg: dict[str, Any]) -> dict[str, Any]:
    """Generate processed-modality diagnostic figures and metadata files."""

    requested_viz = parse_visualization_config(cfg)
    if requested_viz.compare_scenes is not None:
        return visualize_modality_scene_comparison(cfg, requested_viz)

    return _visualize_single_scene(cfg, requested_viz)


def visualize_modality_scene_comparison(cfg: dict[str, Any], requested_viz: VisualizationConfig) -> dict[str, Any]:
    """Run the same visualization diagnostics for multiple DeepSense6G scenes."""

    scenes = requested_viz.compare_scenes or ()
    output_dir = requested_viz.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_results: dict[str, Any] = {}
    output_files: list[str] = []
    for scene_id in scenes:
        scene_cfg = deepcopy(cfg)
        dataset_cfg = scene_cfg.setdefault("data", {}).setdefault("dataset", {})
        if str(dataset_cfg.get("type", "")).strip().lower() in {"scenario9", "scenario32"}:
            dataset_cfg["type"] = "deepsense6g"
        dataset_cfg["scene"] = int(scene_id)
        scene_slug = f"scene{int(scene_id)}"
        scene_viz = replace(requested_viz, output_dir=output_dir / scene_slug, compare_scenes=None)
        result = _visualize_single_scene(scene_cfg, scene_viz)
        scene_results[scene_slug] = {
            "scene_id": int(scene_id),
            "output_dir": result["output_dir"],
            "summary_path": result["summary_path"],
            "split_stats_path": result["split_stats_path"],
            "actual_sample_count": result["actual_sample_count"],
            "enabled_modalities": result["enabled_modalities"],
            "output_files": result.get("output_files", []),
        }
        output_files.extend(result.get("output_files", [result["summary_path"], result["split_stats_path"]]))

    metadata_paths = resolve_metadata_output_paths(
        output_dir,
        preserve_existing=bool(requested_viz.preserve_existing_outputs),
        keys=("summary", "final_config"),
    )
    final_config_path = metadata_paths["final_config"]
    dump_config(final_config_snapshot(cfg, requested_viz), final_config_path)
    output_files.append(str(final_config_path))

    summary_path = metadata_paths["summary"]
    output_files.append(str(summary_path))
    total_sample_count = sum(int(item["actual_sample_count"]) for item in scene_results.values())
    summary = {
        "diagnostics": {
            "visualization": requested_viz.to_json_dict(),
        },
        "compare_scenes": list(scenes),
        "scenes": scene_results,
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "final_config_path": str(final_config_path),
        "actual_sample_count": total_sample_count,
        "output_files": output_files,
    }
    write_json(summary_path, summary)

    return {
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "final_config_path": str(final_config_path),
        "actual_sample_count": total_sample_count,
        "compare_scenes": list(scenes),
        "scenes": scene_results,
        "output_files": output_files,
    }


def _visualize_single_scene(cfg: dict[str, Any], requested_viz: VisualizationConfig) -> dict[str, Any]:
    """Generate processed-modality diagnostic figures and metadata for one scene."""

    effective_cfg = apply_visualization_modalities(cfg, requested_viz.modalities)
    enabled_modalities = resolve_enabled_modalities(effective_cfg)
    viz = requested_viz
    if viz.modalities is None:
        viz = replace(viz, modalities=enabled_modalities)

    output_dir = viz.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_diagnostic_datasets(effective_cfg, viz.splits)
    sample_records: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    output_files: list[str] = []
    metadata_paths = resolve_metadata_output_paths(
        output_dir,
        preserve_existing=bool(viz.preserve_existing_outputs),
        keys=("samples_jsonl", "samples_csv", "split_stats", "final_config", "summary"),
    )

    for split in viz.splits:
        dataset = datasets[split]
        csv_frame = selected_csv_frame_for_dataset(dataset)
        candidates = collect_candidates(dataset, csv_frame)
        selected, sampling_summary = select_sample_candidates(
            candidates,
            sample_count=viz.sample_count,
            per_seq_sample_count=viz.per_seq_sample_count,
            seed=viz.seed,
            seq_index=viz.seq_index,
            labels=viz.labels,
        )

        split_summaries[split] = {
            "dataset": dataset_run_metadata(dataset),
            "sampling": sampling_summary,
            "selected_dataset_indices": [candidate.dataset_index for candidate in selected],
        }

        for candidate in selected:
            row = csv_frame.iloc[candidate.dataset_index]
            sample = dataset[candidate.dataset_index]
            statistics = modality_statistics(sample)
            record = build_sample_record(
                dataset,
                split=split,
                row=row,
                candidate=candidate,
                sample=sample,
                statistics=statistics,
                modalities=enabled_modalities,
                raw_image_reference_enabled=viz.include_raw_image_preview and "image" in sample,
            )
            png_path = sample_png_path(output_dir, dataset, split, candidate)
            render_sample_overview(dataset, sample, record, png_path, viz)
            record["png_path"] = str(png_path)
            sample_records.append(record)
            output_files.append(str(png_path))

    samples_jsonl = metadata_paths["samples_jsonl"]
    samples_csv = metadata_paths["samples_csv"]
    write_samples_jsonl(sample_records, samples_jsonl)
    write_samples_csv(sample_records, samples_csv)
    output_files.extend([str(samples_jsonl), str(samples_csv)])

    scene_metadata = scene_metadata_from_datasets(datasets)
    split_stats_path = metadata_paths["split_stats"]
    split_stats = build_split_stats_report(
        datasets,
        enabled_modalities=enabled_modalities,
        viz=viz,
        scene_metadata=scene_metadata,
    )
    write_json(split_stats_path, split_stats)
    output_files.append(str(split_stats_path))

    snapshot_cfg = final_config_snapshot(effective_cfg, viz)
    final_config_path = metadata_paths["final_config"]
    dump_config(snapshot_cfg, final_config_path)
    output_files.append(str(final_config_path))

    summary_path = metadata_paths["summary"]
    output_files.append(str(summary_path))
    summary = {
        "diagnostics": {
            "visualization": viz.to_json_dict(),
        },
        "scene": scene_metadata,
        "enabled_modalities": list(enabled_modalities),
        "splits": split_summaries,
        "requested_sample_count": viz.sample_count,
        "actual_sample_count": len(sample_records),
        "seed": viz.seed,
        "output_dir": str(output_dir),
        "samples_jsonl": str(samples_jsonl),
        "samples_csv": str(samples_csv),
        "split_stats_path": str(split_stats_path),
        "final_config_path": str(final_config_path),
        "output_files": output_files,
    }
    write_json(summary_path, summary)

    return {
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "samples_jsonl": str(samples_jsonl),
        "samples_csv": str(samples_csv),
        "split_stats_path": str(split_stats_path),
        "final_config_path": str(final_config_path),
        "actual_sample_count": len(sample_records),
        "enabled_modalities": list(enabled_modalities),
        "output_files": output_files,
    }


def parse_visualization_config(cfg: dict[str, Any]) -> VisualizationConfig:
    raw = cfg.get("diagnostics", {}).get("visualization", {}) or {}
    scene_slug = str(cfg.get("data", {}).get("dataset", {}).get("scene_slug") or "scene")
    run_name = cfg.get("output", {}).get("run_name") or cfg.get("experiment", {}).get("name") or "run"
    default_output = Path(str(cfg.get("output", {}).get("dir", "outputs"))) / "diagnostics" / scene_slug / str(run_name)
    output_raw = raw.get("output_dir", default_output)

    splits = _string_tuple(raw.get("splits", ("train", "test")), name="diagnostics.visualization.splits")
    invalid_splits = [split for split in splits if split not in VALID_SPLITS]
    if invalid_splits:
        raise ValueError(f"diagnostics.visualization.splits only supports train/test; got {invalid_splits}.")

    sample_count = int(raw.get("sample_count", 4))
    if sample_count < 0:
        raise ValueError("diagnostics.visualization.sample_count must be non-negative.")
    per_seq_raw = raw.get("per_seq_sample_count")
    per_seq_sample_count = None if per_seq_raw is None else int(per_seq_raw)
    if per_seq_sample_count is not None and per_seq_sample_count < 0:
        raise ValueError("diagnostics.visualization.per_seq_sample_count must be non-negative.")
    max_frames = int(raw.get("max_frames_per_sample", 4))
    if max_frames < 1:
        raise ValueError("diagnostics.visualization.max_frames_per_sample must be positive.")

    return VisualizationConfig(
        output_dir=resolve_path(output_raw),
        splits=splits,
        sample_count=sample_count,
        per_seq_sample_count=per_seq_sample_count,
        seed=int(raw.get("seed", cfg.get("experiment", {}).get("seed", 42))),
        seq_index=_optional_tuple(raw.get("seq_index")),
        labels=_optional_int_tuple(raw.get("labels")),
        modalities=_optional_modalities(raw.get("modalities")),
        compare_scenes=_optional_int_tuple(raw.get("compare_scenes")),
        max_frames_per_sample=max_frames,
        include_raw_image_preview=bool(raw.get("include_raw_image_preview", False)),
        preserve_existing_outputs=bool(raw.get("preserve_existing_outputs", True)),
    )


def apply_visualization_modalities(cfg: dict[str, Any], modalities: tuple[str, ...] | None) -> dict[str, Any]:
    result = deepcopy(cfg)
    if modalities is None:
        return result

    result.setdefault("data", {}).setdefault("dataset", {})
    dataset_cfg = result["data"]["dataset"]
    selected = normalize_modalities(modalities, context="diagnostics.visualization.modalities")
    dataset_cfg.update(dataset_flags_for_modalities(selected))

    result.setdefault("experiment", {})
    model_cfg = result.setdefault("model", {})
    if len(selected) == 1:
        result["experiment"]["task"] = selected[0]
    else:
        result["experiment"]["task"] = "fusion"
        model_cfg["modalities"] = list(selected)
        for role in ("teacher", "student"):
            role_cfg = model_cfg.get(role)
            if isinstance(role_cfg, dict):
                role_cfg.pop("modalities", None)
    return result


def build_diagnostic_datasets(cfg: dict[str, Any], splits: tuple[str, ...]) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    dataset_kwargs: dict[str, Any] = {}
    needs_train = "train" in splits or _needs_train_fit_for_requested_splits(cfg, splits)
    train_dataset = None
    if needs_train:
        try:
            train_dataset = build_dataset(cfg, "train")
            prepare_lidar_normalizer(cfg, train_dataset)
        except Exception as exc:
            raise RuntimeError(
                "Failed to build the train dataset needed for diagnostics. "
                "If only test split is requested, disable GPS/mmWave/LiDAR normalization or provide train data."
            ) from exc
        if getattr(train_dataset, "use_gps", False):
            dataset_kwargs["gps_scaler"] = getattr(train_dataset, "gps_scaler", None)
        if getattr(train_dataset, "use_lidar", False):
            dataset_kwargs["lidar_normalizer"] = getattr(train_dataset, "lidar_normalizer", None)
        if getattr(train_dataset, "use_mmwave", False):
            dataset_kwargs["mmwave_scaler"] = getattr(train_dataset, "mmwave_scaler", None)
        if "train" in splits:
            datasets["train"] = train_dataset

    for split in splits:
        if split == "train":
            continue
        try:
            datasets[split] = build_dataset(cfg, split, **dataset_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build the {split} dataset for diagnostics. "
                "For normalized GPS/LiDAR/mmWave features, diagnostics reuses train-fitted state when possible."
            ) from exc
    return datasets


def selected_csv_frame_for_dataset(dataset: Any) -> pd.DataFrame:
    frame = pd.read_csv(dataset.root_csv, na_values="").fillna(-99)
    metadata = getattr(getattr(dataset, "samples", None), "metadata", {}) or {}
    portion = float(metadata.get("portion", 1.0))
    strategy = str(metadata.get("portion_strategy", "even"))
    seed = int(metadata.get("portion_seed", 42))
    selected, _ = _select_portion(frame, portion=portion, strategy=strategy, seed=seed)
    if len(selected) != len(dataset):
        raise ValueError(
            f"CSV/sample alignment failed for {dataset.root_csv}: selected {len(selected)} rows, "
            f"dataset has {len(dataset)} samples."
        )
    return selected


def collect_candidates(dataset: Any, csv_frame: pd.DataFrame) -> list[SampleCandidate]:
    future_cols = _sorted_numbered_columns(csv_frame.columns, "future_beam")
    candidates = []
    for dataset_index, (csv_row_index, row) in enumerate(csv_frame.iterrows()):
        future_label = None
        if future_cols:
            path = row[future_cols[0]]
            if str(path).strip() != "-99":
                future_label = int(dataset._beam_label(str(path)))
        candidates.append(
            SampleCandidate(
                dataset_index=dataset_index,
                csv_row_index=int(csv_row_index),
                seq_index=_json_scalar(row["seq_index"]) if "seq_index" in row else None,
                future_label=future_label,
            )
        )
    return candidates


def select_sample_candidates(
    candidates: Iterable[SampleCandidate],
    *,
    sample_count: int,
    per_seq_sample_count: int | None = None,
    seed: int,
    seq_index: tuple[Any, ...] | None = None,
    labels: tuple[int, ...] | None = None,
) -> tuple[list[SampleCandidate], dict[str, Any]]:
    all_candidates = list(candidates)
    filtered = filter_sample_candidates(all_candidates, seq_index=seq_index, labels=labels)
    if per_seq_sample_count is not None:
        selected, by_seq_index = _select_per_seq_candidates(
            filtered,
            per_seq_sample_count=int(per_seq_sample_count),
            seed=int(seed),
        )
        requested = int(per_seq_sample_count) * len(by_seq_index)
        return selected, {
            "seed": int(seed),
            "requested_count": requested,
            "per_seq_sample_count": int(per_seq_sample_count),
            "candidate_count": len(filtered),
            "actual_count": len(selected),
            "seq_index_filter": list(seq_index) if seq_index is not None else None,
            "label_filter": list(labels) if labels is not None else None,
            "by_seq_index": by_seq_index,
            "selected_dataset_indices": [candidate.dataset_index for candidate in selected],
            "selected_csv_row_indices": [candidate.csv_row_index for candidate in selected],
        }

    requested = int(sample_count)
    if requested >= len(filtered):
        selected = list(filtered)
    elif requested == 0:
        selected = []
    else:
        rng = np.random.default_rng(int(seed))
        positions = rng.choice(len(filtered), size=requested, replace=False)
        selected = [filtered[int(pos)] for pos in positions]

    return selected, {
        "seed": int(seed),
        "requested_count": requested,
        "per_seq_sample_count": None,
        "candidate_count": len(filtered),
        "actual_count": len(selected),
        "seq_index_filter": list(seq_index) if seq_index is not None else None,
        "label_filter": list(labels) if labels is not None else None,
        "selected_dataset_indices": [candidate.dataset_index for candidate in selected],
        "selected_csv_row_indices": [candidate.csv_row_index for candidate in selected],
    }


def filter_sample_candidates(
    candidates: Iterable[SampleCandidate],
    *,
    seq_index: tuple[Any, ...] | None = None,
    labels: tuple[int, ...] | None = None,
) -> list[SampleCandidate]:
    seq_filter = {_filter_key(value) for value in seq_index} if seq_index is not None else None
    label_filter = {int(value) for value in labels} if labels is not None else None
    return [
        candidate
        for candidate in candidates
        if (seq_filter is None or _filter_key(candidate.seq_index) in seq_filter)
        and (label_filter is None or candidate.future_label in label_filter)
    ]


def _select_per_seq_candidates(
    candidates: list[SampleCandidate],
    *,
    per_seq_sample_count: int,
    seed: int,
) -> tuple[list[SampleCandidate], dict[str, Any]]:
    groups: dict[str, list[SampleCandidate]] = {}
    seq_values: dict[str, Any] = {}
    for candidate in candidates:
        key = _filter_key(candidate.seq_index)
        groups.setdefault(key, []).append(candidate)
        seq_values.setdefault(key, candidate.seq_index)

    rng = np.random.default_rng(int(seed))
    selected: list[SampleCandidate] = []
    summary: dict[str, Any] = {}
    for key, group in groups.items():
        if per_seq_sample_count >= len(group):
            chosen = list(group)
        elif per_seq_sample_count == 0:
            chosen = []
        else:
            positions = rng.choice(len(group), size=per_seq_sample_count, replace=False)
            chosen = [group[int(pos)] for pos in positions]
        selected.extend(chosen)
        summary[key] = {
            "seq_index": seq_values[key],
            "requested_count": int(per_seq_sample_count),
            "candidate_count": len(group),
            "actual_count": len(chosen),
            "selected_dataset_indices": [candidate.dataset_index for candidate in chosen],
            "selected_csv_row_indices": [candidate.csv_row_index for candidate in chosen],
        }
    return selected, summary


def tensor_stats(value: Any) -> dict[str, Any]:
    array = _as_numpy(value)
    finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.number) else np.asarray([])
    stats = {
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "nonzero_fraction": None,
    }
    if array.size == 0 or finite.size == 0:
        return stats
    stats.update(
        {
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "nonzero_fraction": float(np.count_nonzero(array) / array.size),
        }
    )
    return stats


def modality_statistics(sample: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    if "image" in sample:
        image_stats = tensor_stats(sample["image"])
        image_stats["mask_density"] = image_stats["nonzero_fraction"]
        stats["image"] = image_stats
    if "radar_ra" in sample or "radar_da" in sample:
        radar: dict[str, Any] = {}
        if "radar_ra" in sample:
            radar["radar_ra"] = tensor_stats(sample["radar_ra"])
        if "radar_da" in sample:
            radar["radar_da"] = tensor_stats(sample["radar_da"])
        stats["radar"] = radar
    if "lidar" in sample:
        lidar = tensor_stats(sample["lidar"])
        lidar["channel_nonzero_fraction"] = _channel_nonzero_fraction(sample["lidar"])
        stats["lidar"] = lidar
    if "gps" in sample:
        gps = tensor_stats(sample["gps"])
        array = _as_numpy(sample["gps"])
        if array.ndim == 2 and array.size:
            gps["per_dimension_min"] = [float(value) for value in np.min(array, axis=0)]
            gps["per_dimension_max"] = [float(value) for value in np.max(array, axis=0)]
        stats["gps"] = gps
    if "mmwave" in sample:
        mmwave = tensor_stats(sample["mmwave"])
        array = _as_numpy(sample["mmwave"])
        if array.ndim == 2 and array.size:
            mmwave["per_time_mean"] = [float(value) for value in np.mean(array, axis=1)]
            mmwave["per_time_std"] = [float(value) for value in np.std(array, axis=1)]
        stats["mmwave"] = mmwave
    return stats


def build_split_stats_report(
    datasets: dict[str, Any],
    *,
    enabled_modalities: tuple[str, ...],
    viz: VisualizationConfig,
    scene_metadata: dict[str, Any],
) -> dict[str, Any]:
    split_stats: dict[str, Any] = {}
    for split, dataset in datasets.items():
        csv_frame = selected_csv_frame_for_dataset(dataset)
        candidates = collect_candidates(dataset, csv_frame)
        split_stats[split] = build_split_statistics(
            dataset,
            candidates,
            enabled_modalities=enabled_modalities,
            seq_index=viz.seq_index,
            labels=viz.labels,
        )

    report = {
        "scene": scene_metadata,
        "enabled_modalities": list(enabled_modalities),
        "filters": {
            "seq_index": list(viz.seq_index) if viz.seq_index is not None else None,
            "labels": list(viz.labels) if viz.labels is not None else None,
        },
        "splits": split_stats,
    }
    if "train" in split_stats and "test" in split_stats:
        report["train_test"] = {
            "future_label_total_variation_distance": _label_total_variation_distance(
                split_stats["train"].get("future_label_distribution", {}),
                split_stats["test"].get("future_label_distribution", {}),
            )
        }
    return report


def build_split_statistics(
    dataset: Any,
    candidates: Iterable[SampleCandidate],
    *,
    enabled_modalities: tuple[str, ...],
    seq_index: tuple[Any, ...] | None = None,
    labels: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    filtered = filter_sample_candidates(candidates, seq_index=seq_index, labels=labels)
    split_accumulator = _empty_modality_accumulator()
    seq_accumulators: dict[str, dict[str, Any]] = {}
    seq_candidates: dict[str, list[SampleCandidate]] = {}

    for candidate in filtered:
        sample = dataset[candidate.dataset_index]
        statistics = modality_statistics(sample)
        _accumulate_modality_statistics(split_accumulator, statistics)

        key = _filter_key(candidate.seq_index)
        seq_accumulators.setdefault(key, _empty_modality_accumulator())
        seq_candidates.setdefault(key, []).append(candidate)
        _accumulate_modality_statistics(seq_accumulators[key], statistics)

    by_seq_index = {}
    for key, group in seq_candidates.items():
        distribution = _candidate_label_distribution(group)
        by_seq_index[key] = {
            "seq_index": group[0].seq_index,
            "candidate_count": len(group),
            "future_label_distribution": distribution,
            "future_label_top_k": _label_top_k(distribution),
            "majority_baseline": _majority_baseline(distribution),
            "modality_statistics": _finalize_modality_accumulator(seq_accumulators[key], enabled_modalities),
        }

    distribution = _candidate_label_distribution(filtered)
    return {
        "dataset": dataset_run_metadata(dataset),
        "candidate_count": len(filtered),
        "seq_index_count": len(seq_candidates),
        "future_label_distribution": distribution,
        "future_label_top_k": _label_top_k(distribution),
        "majority_baseline": _majority_baseline(distribution),
        "modality_statistics": _finalize_modality_accumulator(split_accumulator, enabled_modalities),
        "by_seq_index": by_seq_index,
    }


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


def write_samples_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_json_ready(record), ensure_ascii=False) + "\n")


def write_samples_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "scene_id",
        "scene_slug",
        "dataset_index",
        "csv_row_index",
        "seq_index",
        "future_label",
        "png_path",
        "enabled_modalities",
        "input_beam",
        "target_beam",
        "paths_json",
        "statistics_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "split": record.get("split"),
                    "scene_id": record.get("scene_id"),
                    "scene_slug": record.get("scene_slug"),
                    "dataset_index": record.get("dataset_index"),
                    "csv_row_index": record.get("csv_row_index"),
                    "seq_index": record.get("seq_index"),
                    "future_label": record.get("future_label"),
                    "png_path": record.get("png_path"),
                    "enabled_modalities": " ".join(record.get("enabled_modalities", [])),
                    "input_beam": json.dumps(record.get("input_beam")),
                    "target_beam": json.dumps(record.get("target_beam")),
                    "paths_json": json.dumps(_json_ready(record.get("paths", {})), ensure_ascii=False),
                    "statistics_json": json.dumps(_json_ready(record.get("statistics", {})), ensure_ascii=False),
                }
            )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2, ensure_ascii=False)


def resolve_metadata_output_paths(
    output_dir: Path,
    *,
    preserve_existing: bool,
    keys: tuple[str, ...],
) -> dict[str, Path]:
    invalid = [key for key in keys if key not in METADATA_FILE_TEMPLATES]
    if invalid:
        raise ValueError(f"Unknown diagnostic metadata output keys: {invalid}.")
    suffix = _metadata_output_suffix(output_dir, preserve_existing=preserve_existing, keys=keys)
    return {key: _metadata_path(output_dir, key, suffix) for key in keys}


def _metadata_output_suffix(output_dir: Path, *, preserve_existing: bool, keys: tuple[str, ...]) -> str:
    if not preserve_existing:
        return ""
    for attempt in range(10000):
        suffix = "" if attempt == 0 else f"_{attempt:03d}"
        paths = [_metadata_path(output_dir, key, suffix) for key in keys]
        if not any(path.exists() for path in paths):
            return suffix
    raise RuntimeError(f"Could not find a free diagnostic metadata suffix in {output_dir}.")


def _metadata_path(output_dir: Path, key: str, suffix: str) -> Path:
    stem, extension = METADATA_FILE_TEMPLATES[key]
    return output_dir / f"{stem}{suffix}{extension}"


def final_config_snapshot(cfg: dict[str, Any], viz: VisualizationConfig) -> dict[str, Any]:
    snapshot = deepcopy(cfg)
    snapshot.setdefault("diagnostics", {})["visualization"] = viz.to_json_dict()
    return snapshot


def scene_metadata_from_datasets(datasets: dict[str, Any]) -> dict[str, Any]:
    if not datasets:
        return {}
    first = next(iter(datasets.values()))
    return {
        "scene_id": getattr(first, "scene_id", None),
        "scene_slug": getattr(first, "scene_slug", None),
    }


def _empty_modality_accumulator() -> dict[str, Any]:
    return {
        "image_mask_density": [],
        "radar_ra_std": [],
        "radar_da_std": [],
        "lidar_nonzero_fraction": [],
        "lidar_channel_nonzero_fraction": [],
    }


def _accumulate_modality_statistics(accumulator: dict[str, Any], statistics: dict[str, Any]) -> None:
    image_stats = statistics.get("image")
    if image_stats and image_stats.get("mask_density") is not None:
        accumulator["image_mask_density"].append(float(image_stats["mask_density"]))

    radar_stats = statistics.get("radar", {})
    radar_ra = radar_stats.get("radar_ra", {})
    radar_da = radar_stats.get("radar_da", {})
    if radar_ra.get("std") is not None:
        accumulator["radar_ra_std"].append(float(radar_ra["std"]))
    if radar_da.get("std") is not None:
        accumulator["radar_da_std"].append(float(radar_da["std"]))

    lidar_stats = statistics.get("lidar")
    if lidar_stats and lidar_stats.get("nonzero_fraction") is not None:
        accumulator["lidar_nonzero_fraction"].append(float(lidar_stats["nonzero_fraction"]))
    if lidar_stats and lidar_stats.get("channel_nonzero_fraction"):
        accumulator["lidar_channel_nonzero_fraction"].append(
            [float(value) for value in lidar_stats["channel_nonzero_fraction"]]
        )


def _finalize_modality_accumulator(
    accumulator: dict[str, Any],
    enabled_modalities: tuple[str, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "image" in enabled_modalities:
        result["image"] = {
            "mask_density": _numeric_summary(accumulator["image_mask_density"]),
        }
    if "radar" in enabled_modalities:
        result["radar"] = {
            "radar_ra_std": _numeric_summary(accumulator["radar_ra_std"]),
            "radar_da_std": _numeric_summary(accumulator["radar_da_std"]),
        }
    if "lidar" in enabled_modalities:
        result["lidar"] = {
            "nonzero_fraction": _numeric_summary(accumulator["lidar_nonzero_fraction"]),
            "channel_nonzero_fraction_mean": _mean_vector(accumulator["lidar_channel_nonzero_fraction"]),
        }
    return result


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _mean_vector(values: list[list[float]]) -> list[float]:
    if not values:
        return []
    max_len = max(len(item) for item in values)
    means = []
    for idx in range(max_len):
        column = [item[idx] for item in values if idx < len(item) and np.isfinite(item[idx])]
        means.append(float(np.mean(column)) if column else 0.0)
    return means


def _candidate_label_distribution(candidates: Iterable[SampleCandidate]) -> dict[str, int]:
    counts: dict[int, int] = {}
    for candidate in candidates:
        if candidate.future_label is None:
            continue
        label = int(candidate.future_label)
        counts[label] = counts.get(label, 0) + 1
    return {str(label): counts[label] for label in sorted(counts)}


def _label_top_k(distribution: dict[str, int], *, k: int = 5) -> list[dict[str, Any]]:
    total = sum(int(count) for count in distribution.values())
    if total == 0:
        return []
    ranked = sorted(distribution.items(), key=lambda item: (-int(item[1]), int(item[0])))
    return [
        {
            "label": int(label),
            "count": int(count),
            "fraction": float(int(count) / total),
        }
        for label, count in ranked[:k]
    ]


def _majority_baseline(distribution: dict[str, int]) -> float | None:
    total = sum(int(count) for count in distribution.values())
    if total == 0:
        return None
    return float(max(int(count) for count in distribution.values()) / total)


def _label_total_variation_distance(left: dict[str, int], right: dict[str, int]) -> float | None:
    left_total = sum(int(count) for count in left.values())
    right_total = sum(int(count) for count in right.values())
    if left_total == 0 or right_total == 0:
        return None
    labels = set(left) | set(right)
    distance = 0.0
    for label in labels:
        left_prob = int(left.get(label, 0)) / left_total
        right_prob = int(right.get(label, 0)) / right_total
        distance += abs(left_prob - right_prob)
    return float(0.5 * distance)


def _needs_train_fit_for_requested_splits(cfg: dict[str, Any], splits: tuple[str, ...]) -> bool:
    if not any(split != "train" for split in splits):
        return False
    modalities = resolve_enabled_modalities(cfg)
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if "gps" in modalities and bool(dataset_cfg.get("gps_normalize", True)):
        return True
    if "mmwave" in modalities and bool(dataset_cfg.get("mmwave_normalize", True)):
        return True
    if "lidar" in modalities and _lidar_normalization_needs_train_fit(dataset_cfg):
        return True
    return False


def _lidar_normalization_needs_train_fit(dataset_cfg: dict[str, Any]) -> bool:
    lidar_norm = dataset_cfg.get("lidar_normalization")
    if isinstance(lidar_norm, dict):
        enabled = bool(lidar_norm.get("enabled", False))
        stats_path = lidar_norm.get("stats_path")
        recompute = bool(lidar_norm.get("recompute", False))
    else:
        enabled = bool(dataset_cfg.get("lidar_normalize", False))
        stats_path = None
        recompute = False
    if not enabled:
        return False
    if stats_path and resolve_path(stats_path).exists() and not recompute:
        return False
    return True


def _optional_tuple(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "all", "none", "null"}:
        return None
    return _tuple_from_value(value)


def _optional_int_tuple(value: Any) -> tuple[int, ...] | None:
    raw = _optional_tuple(value)
    if raw is None:
        return None
    return tuple(int(item) for item in raw)


def _optional_modalities(value: Any) -> tuple[str, ...] | None:
    raw = _optional_tuple(value)
    if raw is None:
        return None
    selected = tuple(str(item).strip().lower() for item in raw)
    return normalize_modalities(selected, context="diagnostics.visualization.modalities")


def _string_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    items = tuple(str(item).strip().lower() for item in _tuple_from_value(value))
    if not items:
        raise ValueError(f"{name} must contain at least one value.")
    return items


def _tuple_from_value(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    if isinstance(value, str):
        text = value.strip()
        if "," in text:
            return tuple(part.strip() for part in text.split(",") if part.strip())
        return (text,)
    return (value,)


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


def _channel_nonzero_fraction(value: Any) -> list[float]:
    array = _as_numpy(value)
    if array.ndim == 4:
        by_channel = np.moveaxis(array, 1, 0)
        return [float(np.count_nonzero(channel) / channel.size) if channel.size else 0.0 for channel in by_channel]
    if array.ndim == 3:
        return [float(np.count_nonzero(channel) / channel.size) if channel.size else 0.0 for channel in array]
    return []


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


def _filter_key(value: Any) -> str:
    return str(_json_scalar(value))


def _safe_token(value: Any) -> str:
    token = str(_json_scalar(value)).replace("/", "_").replace("\\", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in token)[:80] or "none"


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_ready(value.detach().cpu().numpy())
    return value
