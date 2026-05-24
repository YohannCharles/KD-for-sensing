from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kd_sensing.data.deepverse.label_scene import _jsonable
from kd_sensing.data.deepverse.label_targets import (
    RADAR_FEATURE_NAMES,
    RADAR_FEATURE_SIZE,
    _blockage_metadata,
    _filter_built_by_sample_ids,
    _los_status_source_counts,
)
from kd_sensing.data.deepverse.sanity_check import build_sanity_report
from kd_sensing.data.deepverse.split import assign_splits, make_split_result


def write_label_cache(
    builder: Any,
    output_root: str | Path,
    *,
    split_by: str = "sequence",
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
) -> dict[str, Any]:
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    built = builder.build()
    rows = list(built["rows"])
    split_result = make_split_result(
        rows,
        split_by=split_by,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=builder.seed,
    )
    split = split_result.split
    if split_result.discarded_sample_ids:
        keep_ids = {sample_id for sample_ids in split.values() for sample_id in sample_ids}
        rows = [row for row in rows if str(row["sample_id"]) in keep_ids]
        built = _filter_built_by_sample_ids(built, keep_ids)
    assign_splits(rows, split)

    paths = {
        "metadata": output_path / "metadata.json",
        "samples": output_path / "samples.csv",
        "labels": output_path / "labels.npz",
        "weak_wireless": output_path / "weak_wireless.npz",
        "radar_features": output_path / "radar_features.npz",
        "noisy_position": output_path / "noisy_position.npz",
        "camera_index": output_path / "camera_index.json",
        "lidar_index": output_path / "lidar_index.json",
        "split": output_path / "split.json",
        "sanity_report": output_path / "sanity_report.json",
    }

    pd.DataFrame(rows).to_csv(paths["samples"], index=False)
    np.savez_compressed(paths["labels"], **built["labels"])
    np.savez_compressed(paths["weak_wireless"], **built["weak_wireless"])
    np.savez_compressed(paths["radar_features"], **built["radar_features"])
    np.savez_compressed(paths["noisy_position"], **built["noisy_position"])

    _write_json(paths["camera_index"], _path_index(rows, "camera_paths"))
    _write_json(paths["lidar_index"], _path_index(rows, "lidar_paths"))
    _write_json(paths["split"], split)

    blockage = _blockage_metadata(
        built["labels"],
        min_class_count=builder.blockage_min_class_count,
        min_class_ratio=builder.blockage_min_class_ratio,
    )
    report = build_sanity_report(
        rows=rows,
        labels=built["labels"],
        split=split,
        skip_counts=built["skip_counts"],
        artifact_paths={key: str(path) for key, path in paths.items()},
        radar_features=built["radar_features"].get("radar_feature_history"),
        split_metadata=split_result.metadata,
        blockage=blockage,
    )
    default_inputs = ["camera", "lidar", "weak_wireless", "noisy_position"]
    if builder.enable_radar:
        default_inputs.insert(2, "radar")
    default_objectives = ["beam", "trajectory"]
    if blockage["usable"]:
        default_objectives.append("blockage")
    metadata = {
        "scenario": builder.scenario,
        "seq_len": builder.seq_len,
        "pred_horizon": builder.pred_horizon,
        "num_beams": builder.num_beams,
        "beam_topk": builder.beam_topk,
        "position_noise_std": builder.position_noise_std,
        "seed": builder.seed,
        "split_by": split_result.metadata["effective_split_by"],
        "requested_split_by": split_result.metadata["requested_split_by"],
        "split_protocol": split_result.metadata,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "sample_count": len(rows),
        "skip_counts": built["skip_counts"],
        "los_status_source_counts": _los_status_source_counts(rows),
        "split_counts": {name: len(sample_ids) for name, sample_ids in split.items()},
        "label_distribution": report["label_distribution"],
        "blockage": blockage,
        "default_inputs": default_inputs,
        "default_objectives": default_objectives,
        "radar_feature_size": RADAR_FEATURE_SIZE if builder.enable_radar else 0,
        "radar_feature_names": RADAR_FEATURE_NAMES if builder.enable_radar else [],
        "oracle_only_fields": ["clean_position_history", "beam_gain_future", "los_status_future"],
        "artifacts": {key: str(path) for key, path in paths.items()},
    }
    _write_json(paths["metadata"], metadata)
    _write_json(paths["sanity_report"], report)

    return {
        "paths": {key: str(path) for key, path in paths.items()},
        "metadata": metadata,
        "sanity_report": report,
    }


def _path_index(rows: list[dict[str, Any]], column: str) -> dict[str, Any]:
    return {str(row["sample_id"]): json.loads(row[column]) for row in rows}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


__all__ = ["write_label_cache"]
