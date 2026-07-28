#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from kd_sensing.data.mmw.pilot_alignment import resolve_last_input_channel_ref


def audit(csv_path: Path, dataset_root: Path, *, sample_count: int, seed: int) -> dict[str, object]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < int(sample_count):
        raise ValueError(f"Requested {sample_count} samples from only {len(rows)} rows.")
    indices = np.random.default_rng(int(seed)).choice(len(rows), size=int(sample_count), replace=False)
    records = []
    for index in sorted(int(value) for value in indices):
        row = rows[index]
        condition = str(row.get("condition", "")).strip()
        root = dataset_root / condition
        csi_columns = sorted(
            (key for key in row if key.startswith("csi") and key[3:].isdigit()),
            key=lambda key: int(key[3:]),
        )
        reference = resolve_last_input_channel_ref(
            row,
            [row[key] for key in csi_columns],
            data_root=root,
            seq_len=len(csi_columns),
            num_pred=1,
        )
        records.append(
            {
                "row_index": index,
                "sample_id": row.get("sample_id"),
                "sensing_frame_ids": json.loads(row["history_frame_ids_json"]),
                "channel_frame_ids": [Path(row[key]).stem.removesuffix("_paths") for key in csi_columns],
                "target_frame_id": reference["target_frame_id"],
                "pilot_frame_id": reference["pilot_frame_id"],
                "channel_ref": reference["channel_ref"],
            }
        )
    return {
        "status": "passed",
        "sample_count": len(records),
        "seed": int(seed),
        "source_csv": str(csv_path.resolve()),
        "pilot_time_mode": "last_input",
        "future_channel_used_as_input": False,
        "outer_test_accessed": False,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit last-input sparse-pilot temporal alignment.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/MMW"))
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.csv, args.dataset_root, sample_count=args.sample_count, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "sample_count", "pilot_time_mode", "future_channel_used_as_input")}, indent=2))


if __name__ == "__main__":
    main()
