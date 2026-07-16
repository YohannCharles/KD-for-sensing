import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


GROUP_SAFE_TIME_BLOCK = "group_safe_time_block"
MMW_SPLIT_PROTOCOL_VERSION = "mmw_sequence_split_v2"
_RESOURCE_REFERENCE_PATTERN = re.compile(r"^(?:camera|radar|gps|bs_gps|lidar|beam|future_beam)\d+$")


@dataclass
class PreparedFrame:
    condition: str = "sunny"
    town: str = ""
    sensor_scenario: str = ""
    channel_scenario: str = ""
    agent: str = ""
    frame_id: str = ""
    sample_id: str = ""
    camera0: str = ""
    lidar: str = ""
    gps: str = ""
    beam_power_path: str = ""
    beam_label: int = 0


def build_sequence_rows(frames: list[PreparedFrame], *, seq_len: int, pred_len: int) -> tuple[list[dict[str, Any]], int]:
    if seq_len <= 0 or pred_len <= 0:
        raise ValueError("MMW seq_len and pred_len must be positive.")
    grouped: dict[tuple[str, str, str, str], list[PreparedFrame]] = defaultdict(list)
    for frame in frames:
        grouped[(frame.condition, frame.town, frame.sensor_scenario, frame.agent)].append(frame)
    rows: list[dict[str, Any]] = []
    breaks = 0
    for key, items in sorted(grouped.items()):
        segment: list[PreparedFrame] = []
        previous: int | None = None
        ordinal = 0
        for frame in sorted(items, key=lambda item: int(item.frame_id)):
            if previous is not None and int(frame.frame_id) != previous + 1:
                breaks += 1
                rows.extend(_segment_rows(segment, seq_len, pred_len, len(rows), _segment_id(key, ordinal)))
                segment, ordinal = [], ordinal + 1
            segment.append(frame)
            previous = int(frame.frame_id)
        rows.extend(_segment_rows(segment, seq_len, pred_len, len(rows), _segment_id(key, ordinal)))
    return rows, breaks


def _segment_rows(segment: list[PreparedFrame], seq_len: int, pred_len: int, start_index: int, segment_id: str) -> list[dict[str, Any]]:
    window = seq_len + pred_len
    rows = []
    for offset in range(max(0, len(segment) - window + 1)):
        history = segment[offset : offset + seq_len]
        future = segment[offset + seq_len : offset + window]
        first, target = history[0], future[0]
        row: dict[str, Any] = {
            "seq_index": start_index + len(rows),
            "condition": first.condition,
            "town": first.town,
            "sensor_scenario": first.sensor_scenario,
            "scene_slug": first.sensor_scenario,
            "channel_scenario": first.channel_scenario,
            "agent": first.agent,
            "contiguous_segment_id": segment_id,
            "start_frame": first.frame_id,
            "end_frame": history[-1].frame_id,
            "future_start_frame": target.frame_id,
            "future_end_frame": future[-1].frame_id,
            "window_start_frame": first.frame_id,
            "window_end_frame": future[-1].frame_id,
            "window_frame_ids_json": json.dumps([frame.frame_id for frame in (*history, *future)]),
            "sample_id": f"{first.sample_id}:seq{start_index + len(rows):06d}",
            "target_sample_id": target.sample_id,
            "target_label": int(target.beam_label),
            "future_label_sequence_json": json.dumps([int(frame.beam_label) for frame in future]),
        }
        for index, frame in enumerate(history, start=1):
            row[f"camera{index}"] = frame.camera0
            row[f"lidar{index}"] = frame.lidar
            row[f"gps{index}"] = frame.gps
            row[f"beam{index}"] = frame.beam_power_path
        for index, frame in enumerate(future, start=1):
            row[f"future_beam{index}"] = frame.beam_power_path
            row[f"future_beam_label{index}"] = int(frame.beam_label)
        rows.append(row)
    return rows


def split_sequence_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    train_ratio: float,
    strategy: str = GROUP_SAFE_TIME_BLOCK,
    seq_len: int | None = None,
    pred_len: int | None = None,
    block_size_frames: int | None = None,
    guard_band_frames: int | None = None,
) -> dict[str, Any]:
    if strategy != GROUP_SAFE_TIME_BLOCK:
        raise ValueError(f"Unsupported MMW split_strategy={strategy!r}; expected {GROUP_SAFE_TIME_BLOCK!r}.")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("MMW train_ratio must be between 0 and 1.")
    window = int(seq_len or _row_window_length(rows) or 1) + int(pred_len or 0)
    guard = max(int(guard_band_frames) if guard_band_frames is not None else window - 1, window - 1)
    block = max(int(block_size_frames) if block_size_frames is not None else window * 2, window)
    assignments = _time_blocks(rows, block_size=block, guard=guard)
    candidates = [item for item in assignments if item["rows"]]
    group_ids = [item["group_id"] for item in candidates]
    rng = np.random.default_rng(int(seed))
    rng.shuffle(group_ids)
    train_count = min(max(1, int(round(len(group_ids) * train_ratio))), max(len(group_ids) - 1, 1))
    train_groups = set(group_ids[:train_count])
    test_groups = set(group_ids[train_count:])
    train_rows = [row for item in candidates if item["group_id"] in train_groups for row in item["rows"]]
    test_rows = [row for item in candidates if item["group_id"] in test_groups for row in item["rows"]]
    diagnostics = compute_split_leakage_diagnostics(train_rows, test_rows, guard_band_frames=guard)
    reasons = []
    if not train_rows or not test_rows:
        reasons.append("insufficient_group_safe_windows")
    if diagnostics["train_test_frame_overlap_count"]:
        reasons.append("train_test_frame_overlap")
    if diagnostics["guard_band_violations"]:
        reasons.append("guard_band_violation")
    return {
        "seed": int(seed),
        "split_seed": int(seed),
        "train_ratio": float(train_ratio),
        "split_strategy": strategy,
        "split_protocol": MMW_SPLIT_PROTOCOL_VERSION,
        "split_protocol_version": MMW_SPLIT_PROTOCOL_VERSION,
        "seq_len": int(seq_len or max(window - int(pred_len or 0), 1)),
        "pred_len": int(pred_len or max(window - int(seq_len or window), 1)),
        "num_pred": int(pred_len or max(window - int(seq_len or window), 1)),
        "block_size_frames": block,
        "guard_band_frames": guard,
        "group_assignments": [{key: value for key, value in item.items() if key != "rows"} for item in assignments],
        "train_groups": sorted(train_groups),
        "test_groups": sorted(test_groups),
        "train_window_count": len(train_rows),
        "test_window_count": len(test_rows),
        "window_counts": {"total": len(rows), "train": len(train_rows), "test": len(test_rows)},
        "beam_label_distribution": _label_histogram(rows),
        "label_distribution": {"all": _label_histogram(rows), "train": _label_histogram(train_rows), "test": _label_histogram(test_rows)},
        "leakage_diagnostics": diagnostics,
        "strict_validation_eligible": not reasons,
        "eligibility_reasons": reasons,
        "train_seq_indices": [int(row["seq_index"]) for row in train_rows],
        "test_seq_indices": [int(row["seq_index"]) for row in test_rows],
        "train_rows": train_rows,
        "test_rows": test_rows,
    }


def _time_blocks(rows: list[dict[str, Any]], *, block_size: int, guard: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("contiguous_segment_id", ""))].append(row)
    assignments = []
    span = block_size + guard
    for segment, items in sorted(grouped.items()):
        start = min(_start(row) for row in items)
        for block_index in range((max(_end(row) for row in items) - start) // span + 1):
            block_start = start + block_index * span
            block_end = block_start + block_size - 1
            block_rows = [row for row in items if _start(row) >= block_start and _end(row) <= block_end]
            assignments.append(
                {
                    "group_id": f"{segment}:block_{block_index:04d}",
                    "contiguous_segment_id": segment,
                    "block_start_frame": block_start,
                    "block_end_frame": block_end,
                    "guard_start_frame": block_end + 1,
                    "guard_end_frame": block_end + guard,
                    "window_count": len(block_rows),
                    "rows": block_rows,
                }
            )
    return assignments


def compute_split_leakage_diagnostics(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    seq_len: int | None = None,  # noqa: ARG001
    pred_len: int | None = None,  # noqa: ARG001
    guard_band_frames: int | None = None,
) -> dict[str, Any]:
    train_frames = {frame for row in train_rows for frame in _frames(row)}
    test_frames = {frame for row in test_rows for frame in _frames(row)}
    overlap = train_frames & test_frames
    guard = int(guard_band_frames or 0)
    violations = sum(
        1
        for train in train_rows
        for test in test_rows
        if train.get("contiguous_segment_id") == test.get("contiguous_segment_id") and _interval_gap((_start(train), _end(train)), (_start(test), _end(test))) < guard
    )
    adjacent = sum(
        1
        for train in train_rows
        for test in test_rows
        if train.get("contiguous_segment_id") == test.get("contiguous_segment_id")
        and _interval_gap((_start(train), _end(train)), (_start(test), _end(test))) <= 0
    )
    return {
        "generated_by": "compute_split_leakage_diagnostics",
        "diagnostics_version": "mmw_split_leakage_v2",
        "guard_band_frames": guard,
        "train_window_count": len(train_rows),
        "test_window_count": len(test_rows),
        "train_frame_count": len(train_frames),
        "test_frame_count": len(test_frames),
        "train_test_frame_overlap_count": len(overlap),
        "train_test_frame_overlap_examples": sorted(overlap)[:10],
        "guard_band_violations": violations,
        "adjacent_window_cross_split_count": adjacent,
        "adjacent_window_cross_split_ratio": 0.0,
    }


def compute_split_identity_audit(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    train_group_ids: list[str] | tuple[str, ...] | set[str] = (),
    test_group_ids: list[str] | tuple[str, ...] | set[str] = (),
) -> dict[str, Any]:
    """Verify stable sample, group, and resource identities do not cross a split."""
    roles = {"train": train_rows, "test": test_rows}
    missing_fields: dict[str, list[str]] = {}
    stable_ids: dict[str, set[str]] = {}
    resource_references: dict[str, set[str]] = {}
    segment_ids: dict[str, set[str]] = {}
    for role, rows in roles.items():
        missing: set[str] = set()
        ids: set[str] = set()
        refs: set[str] = set()
        segments: set[str] = set()
        for row in rows:
            for field in ("sample_id", "target_sample_id", "contiguous_segment_id"):
                value = str(row.get(field, "")).strip()
                if not value:
                    missing.add(field)
                elif field == "contiguous_segment_id":
                    segments.add(value)
                else:
                    ids.add(value)
            row_resource_keys = [key for key in row if _RESOURCE_REFERENCE_PATTERN.fullmatch(str(key))]
            if not row_resource_keys:
                missing.add("resource_reference")
            for key in row_resource_keys:
                value = str(row.get(key, "")).strip()
                if not value or value == "-99":
                    continue
                refs.add(value)
        if missing:
            missing_fields[role] = sorted(missing)
        stable_ids[role] = ids
        resource_references[role] = refs
        segment_ids[role] = segments
    stable_overlap = stable_ids["train"] & stable_ids["test"]
    resource_overlap = resource_references["train"] & resource_references["test"]
    train_groups = {str(value).strip() for value in train_group_ids if str(value).strip()}
    test_groups = {str(value).strip() for value in test_group_ids if str(value).strip()}
    group_overlap = train_groups & test_groups
    reasons = []
    if missing_fields:
        reasons.append("missing_identity_or_resource_fields")
    if stable_overlap:
        reasons.append("stable_sample_identity_overlap")
    if resource_overlap:
        reasons.append("resource_reference_overlap")
    if group_overlap:
        reasons.append("group_assignment_overlap")
    return {
        "generated_by": "compute_split_identity_audit",
        "audit_version": "mmw_split_identity_v1",
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "missing_fields": missing_fields,
        "train_stable_identity_count": len(stable_ids["train"]),
        "test_stable_identity_count": len(stable_ids["test"]),
        "stable_identity_overlap_count": len(stable_overlap),
        "stable_identity_overlap_examples": sorted(stable_overlap)[:10],
        "train_resource_reference_count": len(resource_references["train"]),
        "test_resource_reference_count": len(resource_references["test"]),
        "resource_reference_overlap_count": len(resource_overlap),
        "resource_reference_overlap_examples": sorted(resource_overlap)[:10],
        "train_contiguous_segment_count": len(segment_ids["train"]),
        "test_contiguous_segment_count": len(segment_ids["test"]),
        "train_group_count": len(train_groups),
        "test_group_count": len(test_groups),
        "group_assignment_overlap_count": len(group_overlap),
        "group_assignment_overlap_examples": sorted(group_overlap)[:10],
    }


def build_sequence_splits_from_manifest(
    *,
    data_root: str | Path,
    scene: str,
    seq_len: int,
    pred_len: int,
    split_tag: str = "",
    split_seed: int = 42,
    train_ratio: float = 0.8,
    split_strategy: str = GROUP_SAFE_TIME_BLOCK,
    block_size_frames: int | None = None,
    guard_band_frames: int | None = None,
) -> dict[str, Any]:
    root = Path(data_root)
    prepared_root = root / "Prepared" / str(scene)
    manifest = prepared_root / "manifests" / "frame_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"MMW frame manifest not found: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as handle:
        frames = [_frame_from_manifest(row) for row in csv.DictReader(handle)]
    rows, non_contiguous = build_sequence_rows(frames, seq_len=seq_len, pred_len=pred_len)
    if not rows:
        raise ValueError(f"No MMW sequence windows generated for {scene}.")
    split = split_sequence_rows(
        rows,
        seed=split_seed,
        train_ratio=train_ratio,
        strategy=split_strategy,
        seq_len=seq_len,
        pred_len=pred_len,
        block_size_frames=block_size_frames,
        guard_band_frames=guard_band_frames,
    )
    split_root = prepared_root / "splits" / _safe_tag(split_tag) if _safe_tag(split_tag) else prepared_root / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    paths = {"all_sequences_csv": split_root / "all_sequences.csv", "train_csv": split_root / "train.csv", "test_csv": split_root / "test.csv", "metadata": split_root / "split_metadata.json"}
    for key, values in (("all_sequences_csv", rows), ("train_csv", split["train_rows"]), ("test_csv", split["test_rows"])):
        _write_rows(paths[key], values)
    paths["metadata"].write_text(json.dumps({key: value for key, value in split.items() if not key.endswith("_rows")}, indent=2), encoding="utf-8")
    return {"scene": str(scene), "windows": len(rows), "train_rows": len(split["train_rows"]), "test_rows": len(split["test_rows"]), "non_contiguous_frames": non_contiguous, "strict_validation_eligible": split["strict_validation_eligible"], "outputs": {key: str(value) for key, value in paths.items()}}


def _frame_from_manifest(row: dict[str, str]) -> PreparedFrame:
    return PreparedFrame(
        condition=str(row.get("condition", "sunny")), town=str(row.get("town", "")), sensor_scenario=str(row.get("sensor_scenario", "")), channel_scenario=str(row.get("channel_scenario", "")), agent=str(row.get("agent", "")), frame_id=str(row.get("frame_id", "")), sample_id=str(row.get("sample_id", "")), camera0=str(row.get("camera0", "")), lidar=str(row.get("lidar", "")), gps=str(row.get("gps", "")), beam_power_path=str(row.get("beam_power_path", "")), beam_label=int(float(row.get("beam_label", 0) or 0)),
    )


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _label_histogram(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(int(row.get("target_label", 0)) for row in rows).items())}


def _frames(row: dict[str, Any]) -> list[str]:
    try:
        return [str(value) for value in json.loads(row.get("window_frame_ids_json", "[]"))]
    except json.JSONDecodeError:
        return [str(value) for value in range(_start(row), _end(row) + 1)]


def _start(row: dict[str, Any]) -> int:
    return int(float(row.get("window_start_frame", row.get("start_frame", 0))))


def _end(row: dict[str, Any]) -> int:
    return int(float(row.get("window_end_frame", row.get("future_end_frame", 0))))


def _interval_gap(left: tuple[int, int], right: tuple[int, int]) -> int:
    if left[0] <= right[1] and right[0] <= left[1]:
        return -1
    return right[0] - left[1] - 1 if left[1] < right[0] else left[0] - right[1] - 1


def _row_window_length(rows: list[dict[str, Any]]) -> int:
    return max((_end(row) - _start(row) + 1 for row in rows), default=0)


def _segment_id(key: tuple[str, str, str, str], ordinal: int) -> str:
    return ":".join((*key, f"segment_{ordinal:04d}"))


def _safe_tag(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value).strip())


__all__ = [
    "build_sequence_rows",
    "build_sequence_splits_from_manifest",
    "compute_split_identity_audit",
    "compute_split_leakage_diagnostics",
    "split_sequence_rows",
]
