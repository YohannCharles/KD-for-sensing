from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from kd_sensing.data.scenes import retarget_deepsense_dataset_config
from kd_sensing.diagnostics.viewer_manifest_cache import (
    _cache_digest,
    _cached_manifest_result,
    _manifest_meta,
    _manifest_output_path,
    _path_stat_dict,
)
from kd_sensing.diagnostics.viewer_manifest_merge import (
    _attach_optional,
    _attach_prediction_bundle,
    _load_external_mapping,
)
from kd_sensing.diagnostics.viewer_manifest_schema import _json_ready, _sample_id
from kd_sensing.diagnostics.viewer_manifest_writer import _manifest_record
from kd_sensing.diagnostics.visualization.config import (
    apply_visualization_modalities,
    parse_visualization_config,
)
from kd_sensing.diagnostics.visualization.datasets import (
    build_diagnostic_datasets,
    selected_csv_frame_for_dataset,
)
from kd_sensing.diagnostics.visualization.sampling import (
    collect_candidates,
    select_sample_candidates,
)
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities

def export_viewer_manifest(
    cfg: dict[str, Any],
    *,
    output_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    predictions: str | Path | dict[str, Any] | list[Any] | None = None,
    quality: str | Path | dict[str, Any] | list[Any] | None = None,
    gate: str | Path | dict[str, Any] | list[Any] | None = None,
    overwrite: bool = False,
    force_rebuild: bool = False,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Prepare a cached Gradio viewer dataset from existing dataset/config metadata."""

    requested_viz = parse_visualization_config(cfg)
    manifest_path = _manifest_output_path(
        output_path=output_path,
        cache_dir=cache_dir,
        default_dir=requested_viz.output_dir,
        cfg=cfg,
        predictions=predictions,
        quality=quality,
        gate=gate,
        sample_limit=sample_limit,
    )
    cache_root = manifest_path.parent
    meta_path = manifest_path.with_name("manifest_meta.json")
    digest = _cache_digest(
        cfg,
        predictions=predictions,
        quality=quality,
        gate=gate,
        sample_limit=sample_limit,
    )
    if not force_rebuild and not overwrite:
        cached = _cached_manifest_result(manifest_path, meta_path, digest)
        if cached is not None:
            return cached

    prediction_map = _load_external_mapping(predictions)
    quality_map = _load_external_mapping(quality)
    gate_map = _load_external_mapping(gate)

    if requested_viz.compare_scenes is not None:
        records: list[dict[str, Any]] = []
        for scene_id in requested_viz.compare_scenes:
            scene_cfg = deepcopy(cfg)
            retarget_deepsense_dataset_config(scene_cfg.setdefault("data", {}).setdefault("dataset", {}), scene_id)
            scene_cfg.setdefault("diagnostics", {}).setdefault("visualization", {})["compare_scenes"] = None
            records.extend(
                _records_for_single_scene(
                    scene_cfg,
                    prediction_map=prediction_map,
                    quality_map=quality_map,
                    gate_map=gate_map,
                    output_dir=cache_root,
                    sample_limit=sample_limit,
                )
            )
    else:
        records = _records_for_single_scene(
            cfg,
            prediction_map=prediction_map,
            quality_map=quality_map,
            gate_map=gate_map,
            output_dir=cache_root,
            sample_limit=sample_limit,
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_json_ready(records), indent=2, ensure_ascii=False), encoding="utf-8")
    meta = _manifest_meta(digest=digest, cfg=cfg, manifest_path=manifest_path, records=records)
    meta_path.write_text(json.dumps(_json_ready(meta), indent=2, ensure_ascii=False), encoding="utf-8")

    viewer_command = (
        "conda run -n kd_mm_beam python tools/visualization/gradio_multimodal_viewer.py "
        f"--manifest {manifest_path}"
    )
    return {
        "mode": "viewer_dataset_cache",
        "message": "Prepared a reusable Gradio viewer dataset cache from the selected dataset config.",
        "cache_hit": False,
        "cache_dir": str(cache_root),
        "manifest_path": str(manifest_path),
        "meta_path": str(meta_path),
        "sample_count": len(records),
        "viewer_command": viewer_command,
    }


def _records_for_single_scene(
    cfg: dict[str, Any],
    *,
    prediction_map: dict[str, Any],
    quality_map: dict[str, Any],
    gate_map: dict[str, Any],
    output_dir: Path,
    sample_limit: int | None,
) -> list[dict[str, Any]]:
    requested_viz = parse_visualization_config(cfg)
    effective_cfg = apply_visualization_modalities(cfg, requested_viz.modalities)
    enabled_modalities = resolve_enabled_modalities(effective_cfg)
    viz = requested_viz
    datasets = build_diagnostic_datasets(effective_cfg, viz.splits)

    records: list[dict[str, Any]] = []
    for split in viz.splits:
        dataset = datasets[split]
        csv_frame = selected_csv_frame_for_dataset(dataset)
        candidates = collect_candidates(dataset, csv_frame)
        selected, _ = select_sample_candidates(
            candidates,
            sample_count=len(candidates),
            per_seq_sample_count=None,
            seed=viz.seed,
            seq_index=viz.seq_index,
            labels=viz.labels,
        )
        if sample_limit is not None:
            selected = selected[: max(0, int(sample_limit))]
        for candidate in selected:
            row = csv_frame.iloc[candidate.dataset_index]
            sample = dataset[candidate.dataset_index]
            sample_id = _sample_id(dataset, split, candidate)
            record = _manifest_record(
                dataset,
                split=split,
                row=row,
                candidate=candidate,
                sample=sample,
                sample_id=sample_id,
                enabled_modalities=enabled_modalities,
                output_dir=output_dir,
            )
            _attach_prediction_bundle(record, prediction_map, sample_id, candidate.dataset_index)
            _attach_optional(record, "quality", quality_map, sample_id, candidate.dataset_index)
            _attach_optional(record, "gate", gate_map, sample_id, candidate.dataset_index)
            records.append(record)
    return records


__all__ = ["export_viewer_manifest"]
