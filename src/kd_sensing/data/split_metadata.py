from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any


SPLIT_METADATA_PROTOCOL = "balanced_seq"
SNAPSHOT_NEXT_FRAME_SPLIT_PROTOCOL = "snapshot_next_frame_balanced_seq"
SUPPORTED_SPLIT_METADATA_PROTOCOLS = {
    SPLIT_METADATA_PROTOCOL,
    SNAPSHOT_NEXT_FRAME_SPLIT_PROTOCOL,
}


def default_split_metadata_path(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    stem = path.stem
    for prefix in ("train_seqs", "test_seqs", "val_seqs"):
        if stem.startswith(prefix):
            suffix = stem[len(prefix) :]
            return path.with_name(f"split_metadata{suffix}.json")
    return path.with_name(f"{stem}_split_metadata.json")


def discover_split_metadata_path(csv_path: str | Path) -> Path | None:
    path = Path(csv_path)
    candidates = [
        default_split_metadata_path(path),
        path.with_name(f"{path.stem}_split_metadata.json"),
        path.with_name(f"{path.stem}.split_metadata.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_split_metadata_for_csv(csv_path: str | Path) -> tuple[Path, dict[str, Any]] | None:
    metadata_path = discover_split_metadata_path(csv_path)
    if metadata_path is None:
        return None
    with metadata_path.open("r", encoding="utf-8") as f:
        return metadata_path, json.load(f)


def split_metadata_summary_for_csv(
    csv_path: str | Path,
    *,
    split: str | None = None,
    require_balanced: bool = False,
    warn: bool = False,
) -> dict[str, Any]:
    expected_path = default_split_metadata_path(csv_path)
    loaded = load_split_metadata_for_csv(csv_path)
    if loaded is None:
        summary: dict[str, Any] = {
            "available": False,
            "expected_path": str(expected_path),
        }
        if require_balanced:
            message = (
                f"balanced_seq split metadata sidecar is missing for {Path(csv_path)}; "
                f"expected {expected_path}."
            )
            summary["warning"] = message
            if warn:
                warnings.warn(message, UserWarning, stacklevel=2)
        return summary

    metadata_path, metadata = loaded
    protocol = metadata.get("split_protocol")
    summary = {
        "available": True,
        "path": str(metadata_path),
        "split_protocol": protocol,
        "in_len": metadata.get("in_len"),
        "out_len": metadata.get("out_len"),
        "split_seed": metadata.get("split_seed"),
        "training_set_pct": metadata.get("training_set_pct"),
        "sequence_counts": metadata.get("sequence_counts"),
        "window_counts": metadata.get("window_counts"),
    }
    if protocol not in SUPPORTED_SPLIT_METADATA_PROTOCOLS and require_balanced:
        message = (
            f"split metadata sidecar {metadata_path} uses protocol {protocol!r}; "
            f"expected one of {sorted(SUPPORTED_SPLIT_METADATA_PROTOCOLS)!r}."
        )
        summary["warning"] = message
        if warn:
            warnings.warn(message, UserWarning, stacklevel=2)

    if split:
        split_payload = (metadata.get("splits") or {}).get(split, {})
        if split_payload:
            summary["split"] = split
            summary["split_sequence_count"] = split_payload.get("sequence_count")
            summary["split_num_samples"] = split_payload.get("num_samples")
            summary["split_seq_index"] = split_payload.get("seq_index")
            summary["split_csv_path"] = split_payload.get("csv_path")
    return summary
