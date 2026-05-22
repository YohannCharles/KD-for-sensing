from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


from kd_sensing.diagnostics.visualization.config import _json_ready

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

__all__ = [
    'write_json',
    'write_samples_csv',
    'write_samples_jsonl',
]
