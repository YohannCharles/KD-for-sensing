from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

from kd_sensing.config.io import dump_config
from kd_sensing.data.scenes import retarget_deepsense_dataset_config
from kd_sensing.diagnostics.visualization.config import (
    VisualizationConfig,
    apply_visualization_modalities,
    final_config_snapshot,
    parse_visualization_config,
    resolve_metadata_output_paths,
)
from kd_sensing.diagnostics.visualization.datasets import (
    build_diagnostic_datasets,
    scene_metadata_from_datasets,
    selected_csv_frame_for_dataset,
)
from kd_sensing.diagnostics.visualization.render import (
    build_sample_record,
    render_sample_overview,
    sample_png_path,
)
from kd_sensing.diagnostics.visualization.sampling import collect_candidates, select_sample_candidates
from kd_sensing.diagnostics.visualization.stats import build_split_stats_report, modality_statistics
from kd_sensing.diagnostics.visualization.writers import write_json, write_samples_csv, write_samples_jsonl
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.engine.run_metadata import dataset_run_metadata

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
        retarget_deepsense_dataset_config(dataset_cfg, scene_id)
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

__all__ = ["visualize_modalities", "visualize_modality_scene_comparison"]
