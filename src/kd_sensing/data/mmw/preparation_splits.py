import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from kd_sensing.data.beam_label_calibration import resolve_beam_label_mapping
from kd_sensing.data.mmw.preparation_config import GROUP_SAFE_TIME_BLOCK, MMW_SPLIT_PROTOCOL_VERSION, SUPPORTED_SEQUENCE_SPLIT_STRATEGIES
from kd_sensing.data.mmw.preparation_beam_power import _beam_histogram
from kd_sensing.data.mmw.preparation_geometry import _azimuth_bin
from kd_sensing.data.mmw.preparation_index import PreparedFrame



def build_sequence_rows(
    frames: list[PreparedFrame],
    *,
    seq_len: int,
    pred_len: int,
) -> tuple[list[dict[str, Any]], int]:
    by_group: dict[tuple[str, str, str, str], list[PreparedFrame]] = defaultdict(list)
    for frame in frames:
        by_group[(frame.condition, frame.town, frame.sensor_scenario, frame.agent)].append(frame)
    rows: list[dict[str, Any]] = []
    seq_index = 0
    non_contiguous_breaks = 0
    for group_key, agent_frames in sorted(by_group.items()):
        ordered = sorted(agent_frames, key=lambda item: int(item.frame_id))
        segment: list[PreparedFrame] = []
        previous: int | None = None
        segment_id = 0
        for frame in ordered:
            current = int(frame.frame_id)
            if previous is not None and current != previous + 1:
                non_contiguous_breaks += 1
                rows.extend(
                    _windows_for_segment(
                        segment,
                        seq_len=seq_len,
                        pred_len=pred_len,
                        start_index=seq_index,
                        contiguous_segment_id=_segment_id(group_key, segment_id),
                        segment_ordinal=segment_id,
                    )
                )
                seq_index = len(rows)
                segment = []
                segment_id += 1
            segment.append(frame)
            previous = current
        rows.extend(
            _windows_for_segment(
                segment,
                seq_len=seq_len,
                pred_len=pred_len,
                start_index=seq_index,
                contiguous_segment_id=_segment_id(group_key, segment_id),
                segment_ordinal=segment_id,
            )
        )
        seq_index = len(rows)
    return rows, non_contiguous_breaks


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
    beam_label_calibration: bool | dict[str, Any] | None = None,
) -> dict[str, Any]:
    strategy = _normalize_split_strategy(strategy)
    return _split_sequence_rows_group_safe(
        rows,
        seed=seed,
        train_ratio=train_ratio,
        seq_len=seq_len,
        pred_len=pred_len,
        block_size_frames=block_size_frames,
        guard_band_frames=guard_band_frames,
        beam_label_calibration=beam_label_calibration,
    )


def _split_sequence_rows_group_safe(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    train_ratio: float,
    seq_len: int | None,
    pred_len: int | None,
    block_size_frames: int | None,
    guard_band_frames: int | None,
    beam_label_calibration: bool | dict[str, Any] | None,
) -> dict[str, Any]:
    window_length = _window_length(rows, seq_len=seq_len, pred_len=pred_len)
    guard = _resolve_guard_band(seq_len=seq_len, pred_len=pred_len, guard_band_frames=guard_band_frames)
    block_size = _resolve_block_size(
        seq_len=seq_len,
        pred_len=pred_len,
        block_size_frames=block_size_frames,
        guard_band_frames=guard,
    )
    assignments = _time_block_assignments(rows, block_size_frames=block_size, guard_band_frames=guard)
    usable = [item for item in assignments if item["split_role"] == "candidate"]
    train_group_ids, test_group_ids = _assign_group_ids(usable, seed=seed, train_ratio=train_ratio)
    train_set = set(train_group_ids)
    test_set = set(test_group_ids)
    train_rows = [row for item in usable if item["group_id"] in train_set for row in item["rows"]]
    test_rows = [row for item in usable if item["group_id"] in test_set for row in item["rows"]]
    diagnostics = compute_split_leakage_diagnostics(
        train_rows,
        test_rows,
        seq_len=seq_len,
        pred_len=pred_len,
        guard_band_frames=guard,
    )
    metadata = _base_split_metadata(
        rows,
        train_rows=train_rows,
        test_rows=test_rows,
        seed=seed,
        train_ratio=train_ratio,
        strategy=GROUP_SAFE_TIME_BLOCK,
        seq_len=seq_len,
        pred_len=pred_len,
        block_size_frames=block_size,
        guard_band_frames=guard,
        group_assignments=assignments,
        train_groups=train_group_ids,
        test_groups=test_group_ids,
        diagnostics=diagnostics,
        beam_label_calibration=beam_label_calibration,
    )
    reasons = list(metadata.get("eligibility_reasons", []))
    if train_rows and test_rows:
        if int(diagnostics.get("train_test_frame_overlap_count", 0)) > 0:
            reasons.append("train_test_frame_overlap")
        if float(diagnostics.get("adjacent_window_cross_split_ratio", 0.0)) > 0.0:
            reasons.append("adjacent_window_cross_split")
        if int(diagnostics.get("guard_band_violations", 0)) > 0:
            reasons.append("guard_band_violation")
    else:
        reasons.append("insufficient_group_safe_windows")
    metadata.update(
        {
            "train_seq_indices": [int(row["seq_index"]) for row in train_rows],
            "test_seq_indices": [int(row["seq_index"]) for row in test_rows],
            "train_rows": train_rows,
            "test_rows": test_rows,
            "strict_validation_eligible": len(_unique_text(reasons)) == 0,
            "eligibility_reasons": _unique_text(reasons),
            "window_length_frames": int(window_length),
        }
    )
    return metadata


def _base_split_metadata(
    rows: list[dict[str, Any]],
    *,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed: int,
    train_ratio: float,
    strategy: str,
    seq_len: int | None,
    pred_len: int | None,
    block_size_frames: int | None,
    guard_band_frames: int | None,
    group_assignments: list[dict[str, Any]],
    train_groups: list[str],
    test_groups: list[str],
    diagnostics: dict[str, Any],
    beam_label_calibration: bool | dict[str, Any] | None = None,
) -> dict[str, Any]:
    label_distribution = {
        "all": _beam_histogram(rows),
        "train": _beam_histogram(train_rows),
        "test": _beam_histogram(test_rows),
    }
    scene = _first_row_value(rows, "scene_slug") or _first_row_value(rows, "sensor_scenario")
    mapping = resolve_beam_label_mapping(beam_label_calibration, scene=str(scene) if scene is not None else None)
    calibrated_label_distribution = {
        "all": _calibrated_beam_histogram(rows, mapping),
        "train": _calibrated_beam_histogram(train_rows, mapping),
        "test": _calibrated_beam_histogram(test_rows, mapping),
    } if mapping.enabled else {}
    public_group_assignments = [_public_group_assignment(item) for item in group_assignments]
    metadata = {
        "seed": int(seed),
        "split_seed": int(seed),
        "train_ratio": float(train_ratio),
        "split_strategy": strategy,
        "split_protocol": MMW_SPLIT_PROTOCOL_VERSION,
        "split_protocol_version": MMW_SPLIT_PROTOCOL_VERSION,
        "strategy_source": "default_group_safe",
        "group_key_fields": ["condition", "town", "sensor_scenario", "agent", "contiguous_segment_id", "time_block_id"],
        "block_size_frames": None if block_size_frames is None else int(block_size_frames),
        "guard_band_frames": None if guard_band_frames is None else int(guard_band_frames),
        "seq_len": None if seq_len is None else int(seq_len),
        "pred_len": None if pred_len is None else int(pred_len),
        "num_pred": None if pred_len is None else int(pred_len),
        "train_window_count": len(train_rows),
        "test_window_count": len(test_rows),
        "window_counts": {"total": len(rows), "train": len(train_rows), "test": len(test_rows)},
        "group_counts": {"total": len(public_group_assignments), "train": len(train_groups), "test": len(test_groups)},
        "train_groups": list(train_groups),
        "test_groups": list(test_groups),
        "group_assignments": public_group_assignments,
        "frame_ranges": {
            "all": _frame_range_summary(rows),
            "train": _frame_range_summary(train_rows),
            "test": _frame_range_summary(test_rows),
        },
        "beam_label_distribution": label_distribution["all"],
        "label_distribution": label_distribution,
        "raw_label_distribution": label_distribution,
        "leakage_diagnostics": diagnostics,
        "diagnostics_version": "mmw_split_leakage_v1",
        "fix_hint": (
            "Resolve structural frame/window/adjacency/guard-band overlap, then regenerate MMW splits "
            "with split_strategy=group_safe_time_block and a fresh strict split tag."
        ),
    }
    metadata.update(mapping.metadata())
    if mapping.enabled:
        metadata["calibrated_label_distribution"] = calibrated_label_distribution
        metadata["label_distribution_by_space"] = {
            "raw": label_distribution,
            mapping.label_space: calibrated_label_distribution,
        }
    return metadata


def _normalize_split_strategy(value: object) -> str:
    strategy = str(value or GROUP_SAFE_TIME_BLOCK).strip()
    if strategy not in SUPPORTED_SEQUENCE_SPLIT_STRATEGIES:
        raise ValueError(
            f"Unsupported MMW split_strategy={strategy!r}; "
            f"expected one of {sorted(SUPPORTED_SEQUENCE_SPLIT_STRATEGIES)}."
        )
    return strategy


def _calibrated_beam_histogram(rows: list[dict[str, Any]], mapping: Any) -> dict[str, int]:
    raw = _beam_histogram(rows)
    counter: Counter[int] = Counter()
    for label, count in raw.items():
        counter[int(mapping.map_label(int(label)))] += int(count)
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _first_row_value(rows: list[dict[str, Any]], key: str) -> Any:
    for row in rows:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _segment_id(group_key: tuple[str, str, str, str], segment_ordinal: int) -> str:
    condition, town, scenario, agent = group_key
    return f"{condition}:{town}:{scenario}:{agent}:segment_{int(segment_ordinal):04d}"


def _resolve_guard_band(
    *,
    seq_len: int | None,
    pred_len: int | None,
    guard_band_frames: int | None,
) -> int:
    minimum = max(_window_length([], seq_len=seq_len, pred_len=pred_len) - 1, 0)
    if guard_band_frames is None:
        return int(minimum)
    return max(int(guard_band_frames), int(minimum))


def _resolve_block_size(
    *,
    seq_len: int | None,
    pred_len: int | None,
    block_size_frames: int | None,
    guard_band_frames: int,
) -> int:
    window_length = _window_length([], seq_len=seq_len, pred_len=pred_len)
    minimum = max(window_length + 1, window_length)
    if block_size_frames is not None:
        return max(int(block_size_frames), int(minimum))
    return int(minimum)


def _window_length(
    rows: list[dict[str, Any]],
    *,
    seq_len: int | None,
    pred_len: int | None,
) -> int:
    if seq_len is not None and pred_len is not None:
        return int(seq_len) + int(pred_len)
    if rows:
        return max((len(_window_frame_ids(row)) for row in rows), default=0)
    return 0


def _time_block_assignments(
    rows: list[dict[str, Any]],
    *,
    block_size_frames: int,
    guard_band_frames: int,
) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_segment_key(row)].append(row)

    span = int(block_size_frames) + int(guard_band_frames)
    for segment_key, segment_rows in sorted(grouped.items(), key=lambda item: item[0]):
        ordered = sorted(segment_rows, key=lambda row: (_row_start_frame(row), int(row.get("seq_index", 0))))
        starts = [_row_start_frame(row) for row in ordered]
        ends = [_row_window_end_frame(row) for row in ordered]
        if not starts or not ends:
            continue
        segment_start = min(starts)
        max_end = max(ends)
        by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in ordered:
            start = _row_start_frame(row)
            end = _row_window_end_frame(row)
            block_index = (start - segment_start) // span if span > 0 else 0
            block_start = segment_start + block_index * span
            block_end = block_start + int(block_size_frames) - 1
            if start >= block_start and end <= block_end:
                by_block[int(block_index)].append(row)

        for block_index in range(0, ((max_end - segment_start) // span) + 1):
            block_start = segment_start + block_index * span
            block_end = block_start + int(block_size_frames) - 1
            guard_start = block_end + 1
            guard_end = block_end + int(guard_band_frames)
            block_rows = by_block.get(block_index, [])
            group_id = f"{':'.join(segment_key)}:block_{block_index:04d}:{block_start:06d}-{block_end:06d}"
            assignments.append(
                {
                    "group_id": group_id,
                    "split_role": "candidate" if block_rows else "discarded_boundary_or_guard",
                    "condition": segment_key[0],
                    "town": segment_key[1],
                    "sensor_scenario": segment_key[2],
                    "agent": segment_key[3],
                    "contiguous_segment_id": segment_key[4],
                    "time_block_id": f"block_{block_index:04d}",
                    "block_start_frame": int(block_start),
                    "block_end_frame": int(block_end),
                    "guard_start_frame": int(guard_start),
                    "guard_end_frame": int(guard_end),
                    "seq_indices": [int(row["seq_index"]) for row in block_rows],
                    "window_count": int(len(block_rows)),
                    "frame_range": _frame_range_summary(block_rows),
                    "beam_label_distribution": _beam_histogram(block_rows),
                    "rows": block_rows,
                }
            )
    return assignments


def _public_group_assignment(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "rows"}


def _assign_group_ids(
    assignments: list[dict[str, Any]],
    *,
    seed: int,
    train_ratio: float,
) -> tuple[list[str], list[str]]:
    group_ids = [str(item["group_id"]) for item in assignments if int(item.get("window_count", 0)) > 0]
    if not group_ids:
        return [], []
    if len(group_ids) == 1:
        return group_ids, []
    rng = np.random.default_rng(int(seed))
    shuffled = list(group_ids)
    rng.shuffle(shuffled)
    train_count = int(round(len(shuffled) * float(train_ratio)))
    train_count = min(max(train_count, 1), len(shuffled) - 1)
    train = sorted(shuffled[:train_count])
    test = sorted(shuffled[train_count:])
    return train, test


def compute_split_leakage_diagnostics(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    seq_len: int | None = None,
    pred_len: int | None = None,
    guard_band_frames: int | None = None,
) -> dict[str, Any]:
    train_frames = _frame_identity_set(train_rows)
    test_frames = _frame_identity_set(test_rows)
    frame_overlap = train_frames & test_frames
    train_windows = [(row, _frame_identity_set([row])) for row in train_rows]
    max_overlaps: list[int] = []
    for test_row in test_rows:
        test_window = _frame_identity_set([test_row])
        max_overlaps.append(max((len(test_window & train_window) for _, train_window in train_windows), default=0))

    adjacent_pairs, adjacent_cross = _adjacent_window_cross_split_counts(train_rows, test_rows)
    train_sequences = {_future_label_sequence_key(row) for row in train_rows}
    train_sequences.discard("")
    reused = sum(1 for row in test_rows if _future_label_sequence_key(row) in train_sequences)
    guard = _resolve_guard_band(seq_len=seq_len, pred_len=pred_len, guard_band_frames=guard_band_frames)
    guard_violations = _guard_band_violation_count(train_rows, test_rows, guard_band_frames=guard)
    return {
        "generated_by": "compute_split_leakage_diagnostics",
        "diagnostics_version": "mmw_split_leakage_v1",
        "window_length_frames": _window_length(train_rows or test_rows, seq_len=seq_len, pred_len=pred_len),
        "guard_band_frames": int(guard),
        "train_window_count": int(len(train_rows)),
        "test_window_count": int(len(test_rows)),
        "train_frame_count": int(len(train_frames)),
        "test_frame_count": int(len(test_frames)),
        "train_test_frame_overlap_count": int(len(frame_overlap)),
        "train_test_frame_overlap_examples": sorted(frame_overlap)[:10],
        "test_window_max_frame_overlap": {
            "max": max(max_overlaps, default=0),
            "histogram": {str(key): int(value) for key, value in sorted(Counter(max_overlaps).items())},
        },
        "adjacent_window_pair_count": int(adjacent_pairs),
        "adjacent_window_cross_split_count": int(adjacent_cross),
        "adjacent_window_cross_split_ratio": float(adjacent_cross / adjacent_pairs) if adjacent_pairs else 0.0,
        "future_label_sequence_reuse_count": int(reused),
        "future_label_sequence_reuse_ratio": float(reused / len(test_rows)) if test_rows else 0.0,
        "future_label_sequence_reuse_role": "label_distribution_diagnostic_only",
        "guard_band_violations": int(guard_violations),
    }


def _segment_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("condition", "")),
        str(row.get("town", "")),
        str(row.get("sensor_scenario", row.get("scene_slug", ""))),
        str(row.get("agent", "")),
        str(row.get("contiguous_segment_id", "")),
    )


def _row_start_frame(row: dict[str, Any]) -> int:
    return _frame_int(row.get("window_start_frame", row.get("start_frame", 0)))


def _row_window_end_frame(row: dict[str, Any]) -> int:
    return _frame_int(row.get("window_end_frame", row.get("future_end_frame", row.get("future_start_frame", row.get("end_frame", 0)))))


def _frame_int(value: object) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _window_frame_ids(row: dict[str, Any]) -> list[str]:
    value = row.get("window_frame_ids_json")
    if value:
        try:
            payload = json.loads(str(value))
            return [str(item) for item in payload]
        except json.JSONDecodeError:
            pass
    start = _row_start_frame(row)
    end = _row_window_end_frame(row)
    return [f"{idx:06d}" for idx in range(start, end + 1)]


def _future_label_sequence_key(row: dict[str, Any]) -> str:
    value = str(row.get("future_label_sequence_key", "") or "").strip()
    if value:
        return value
    payload = row.get("future_label_sequence_json")
    if payload:
        try:
            return ",".join(str(int(item)) for item in json.loads(str(payload)))
        except (json.JSONDecodeError, TypeError, ValueError):
            return ""
    labels = []
    for key, value in row.items():
        if str(key).startswith("future_beam_label"):
            labels.append((str(key), value))
    return ",".join(str(int(value)) for _, value in sorted(labels))


def _frame_identity_set(rows: list[dict[str, Any]]) -> set[str]:
    identities: set[str] = set()
    for row in rows:
        prefix = "|".join(str(item) for item in _segment_key(row)[:4])
        for frame_id in _window_frame_ids(row):
            identities.add(f"{prefix}|{_manifest_frame_id(frame_id)}")
    return identities


def _manifest_frame_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        return f"{int(text):06d}"
    except ValueError:
        return text


def _adjacent_window_cross_split_counts(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> tuple[int, int]:
    marked = [(row, "train") for row in train_rows] + [(row, "test") for row in test_rows]
    grouped: dict[tuple[str, str, str, str, str], list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for row, split in marked:
        grouped[_segment_key(row)].append((row, split))
    pair_count = 0
    cross_count = 0
    for items in grouped.values():
        ordered = sorted(items, key=lambda item: (_row_start_frame(item[0]), int(item[0].get("seq_index", 0))))
        for (left, left_split), (right, right_split) in zip(ordered, ordered[1:]):
            if _row_start_frame(right) - _row_start_frame(left) == 1:
                pair_count += 1
                if left_split != right_split:
                    cross_count += 1
    return pair_count, cross_count


def _guard_band_violation_count(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    guard_band_frames: int,
) -> int:
    if not train_rows or not test_rows or int(guard_band_frames) <= 0:
        return 0
    train_by_segment: dict[tuple[str, str, str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in train_rows:
        train_by_segment[_segment_key(row)].append((_row_start_frame(row), _row_window_end_frame(row)))
    violations = 0
    for test_row in test_rows:
        test_interval = (_row_start_frame(test_row), _row_window_end_frame(test_row))
        for train_interval in train_by_segment.get(_segment_key(test_row), []):
            if _interval_gap(train_interval, test_interval) < int(guard_band_frames):
                violations += 1
                break
    return violations


def _interval_gap(left: tuple[int, int], right: tuple[int, int]) -> int:
    left_start, left_end = left
    right_start, right_end = right
    if left_start <= right_end and right_start <= left_end:
        return -1
    if left_end < right_start:
        return right_start - left_end - 1
    return left_start - right_end - 1


def _frame_range_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"min": None, "max": None}
    starts = [_row_start_frame(row) for row in rows]
    ends = [_row_window_end_frame(row) for row in rows]
    return {"min": int(min(starts)), "max": int(max(ends))}


def _unique_text(values: list[Any]) -> list[str]:
    unique: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in unique:
            unique.append(text)
    return unique


def _prepared_frame_from_manifest_row(row: dict[str, Any]) -> PreparedFrame:
    return PreparedFrame(
        condition=str(row.get("condition", "sunny")),
        town=str(row.get("town", "")),
        sensor_scenario=str(row.get("sensor_scenario", row.get("scenario", ""))),
        channel_scenario=str(row.get("channel_scenario", "")),
        agent=str(row.get("agent", "")),
        channel_agent=str(row.get("channel_agent", "")),
        frame_id=_manifest_frame_id(row.get("frame_id", "")),
        sample_id=str(row.get("sample_id", "")),
        camera0=str(row.get("camera0", "")),
        cameras=_json_manifest_cell(row.get("cameras_json", ""), {}),
        depth_cameras=_json_manifest_cell(row.get("depth_cameras_json", ""), {}),
        lidar=str(row.get("lidar", "")),
        gps=str(row.get("gps", "")),
        radar=str(row.get("radar", "")),
        channel_path=str(row.get("channel_path", "")),
        beam_power_path=str(row.get("beam_power_path", "")),
        beam_label=int(float(row.get("beam_label", 0) or 0)),
        coarse_sector=int(float(row.get("coarse_sector", 0) or 0)),
        radio_semantic_label=_optional_manifest_int(row.get("radio_semantic_label", "")),
        radio_semantic_available=_bool_manifest_cell(row.get("radio_semantic_available", False)),
        radio_semantic_unavailable_reason=str(row.get("radio_semantic_unavailable_reason", "")),
        radio_semantic_metadata=_json_manifest_cell(row.get("radio_semantic_metadata_json", ""), {}),
        modality_availability=_json_manifest_cell(row.get("modality_availability_json", ""), {}),
        relative_geometry=_json_manifest_cell(row.get("relative_geometry_json", ""), {}),
        proxy_features=_json_manifest_cell(row.get("proxy_features_json", ""), {}),
        channel_fields=_json_manifest_cell(row.get("channel_fields_json", ""), {}),
        rsu=_json_manifest_cell(row.get("rsu_json", ""), {}),
    )


def _json_manifest_cell(value: object, default: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return default
    return payload


def _optional_manifest_int(value: object) -> int | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    numeric = int(float(text))
    return numeric if numeric >= 0 else None


def _bool_manifest_cell(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _write_rows(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames or _csv_fieldnames(rows))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({key for row in rows for key in row.keys()})


def _safe_split_tag(value: object) -> str:
    text = str(value or "").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


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
    beam_label_calibration: bool | dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(data_root)
    prepared_root = root / "Prepared" / str(scene)
    manifest_path = prepared_root / "manifests" / "frame_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"MMW frame manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        frames = [_prepared_frame_from_manifest_row(row) for row in csv.DictReader(handle)]
    sequences, non_contiguous = build_sequence_rows(frames, seq_len=int(seq_len), pred_len=int(pred_len))
    if not sequences:
        raise ValueError(
            f"No sequence windows generated for {scene} with seq_len={int(seq_len)}, pred_len={int(pred_len)}; "
            f"manifest rows={len(frames)}."
        )
    split = split_sequence_rows(
        sequences,
        seed=int(split_seed),
        train_ratio=float(train_ratio),
        strategy=split_strategy,
        seq_len=int(seq_len),
        pred_len=int(pred_len),
        block_size_frames=block_size_frames,
        guard_band_frames=guard_band_frames,
        beam_label_calibration=beam_label_calibration,
    )
    safe_tag = _safe_split_tag(split_tag)
    split_dir = prepared_root / "splits"
    if safe_tag:
        split_dir = split_dir / safe_tag
    split_dir.mkdir(parents=True, exist_ok=True)
    all_sequences_path = split_dir / "all_sequences.csv"
    train_path = split_dir / "train.csv"
    test_path = split_dir / "test.csv"
    metadata_path = split_dir / "split_metadata.json"
    sequence_fieldnames = _csv_fieldnames(sequences)
    _write_rows(all_sequences_path, sequences, fieldnames=sequence_fieldnames)
    _write_rows(train_path, split["train_rows"], fieldnames=sequence_fieldnames)
    _write_rows(test_path, split["test_rows"], fieldnames=sequence_fieldnames)
    metadata = {
        key: value
        for key, value in split.items()
        if key not in {"train_rows", "test_rows"}
    }
    metadata.update(
        {
            "type": "mmw_sequence_splits_from_manifest",
            "public_utility": "kd_sensing.data.mmw.preparation.build_sequence_splits_from_manifest",
            "manifest_path": str(manifest_path),
            "data_root": str(root),
            "prepared_root": str(prepared_root),
            "scene": str(scene),
            "scenario": str(scene),
            "condition": root.name,
            "seq_len": int(seq_len),
            "num_pred": int(pred_len),
            "pred_len": int(pred_len),
            "split_tag": safe_tag,
            "split_seed": int(split_seed),
            "train_ratio": float(train_ratio),
            "split_strategy": split["split_strategy"],
            "block_size_frames": split.get("block_size_frames"),
            "guard_band_frames": split.get("guard_band_frames"),
            "manifest_rows": int(len(frames)),
            "window_count": int(len(sequences)),
            "train_rows": int(len(split["train_rows"])),
            "test_rows": int(len(split["test_rows"])),
            "non_contiguous_frames": int(non_contiguous),
            "outputs": {
                "split_dir": str(split_dir),
                "all_sequences_csv": str(all_sequences_path),
                "train_csv": str(train_path),
                "test_csv": str(test_path),
                "metadata": str(metadata_path),
            },
            "repair_command": (
                "conda run -n kd_mm_beam kd-sensing-preprocess --action mmw_sequence_splits_from_manifest "
                f"--data-root {root} --scene {scene} --seq-len {int(seq_len)} --pred-len {int(pred_len)} "
                f"--split-tag {safe_tag or 'default'} --split-seed {int(split_seed)} --train-ratio {float(train_ratio)} "
                f"--split-strategy {split['split_strategy']}"
            ),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "scene": str(scene),
        "seq_len": int(seq_len),
        "pred_len": int(pred_len),
        "split_tag": safe_tag,
        "manifest_rows": int(len(frames)),
        "windows": int(len(sequences)),
        "train_rows": int(len(split["train_rows"])),
        "test_rows": int(len(split["test_rows"])),
        "split_strategy": split["split_strategy"],
        "strict_validation_eligible": bool(split.get("strict_validation_eligible", False)),
        "non_contiguous_frames": int(non_contiguous),
        "split_dir": str(split_dir),
        "metadata_path": str(metadata_path),
        "outputs": metadata["outputs"],
    }


def _windows_for_segment(
    segment: list[PreparedFrame],
    *,
    seq_len: int,
    pred_len: int,
    start_index: int,
    contiguous_segment_id: str,
    segment_ordinal: int,
) -> list[dict[str, Any]]:
    rows = []
    total = int(seq_len) + int(pred_len)
    if len(segment) < total:
        return rows
    for offset in range(0, len(segment) - total + 1):
        history = segment[offset : offset + seq_len]
        future = segment[offset + seq_len : offset + total]
        window_frames = history + future
        history_frame_ids = [frame.frame_id for frame in history]
        future_frame_ids = [frame.frame_id for frame in future]
        window_frame_ids = [frame.frame_id for frame in window_frames]
        future_labels = [int(frame.beam_label) for frame in future]
        row: dict[str, Any] = {
            "seq_index": start_index + len(rows),
            "agent": history[-1].agent,
            "sample_id": history[-1].sample_id,
            "target_sample_id": future[0].sample_id,
            "condition": history[-1].condition,
            "town": history[-1].town,
            "sensor_scenario": history[-1].sensor_scenario,
            "channel_scenario": history[-1].channel_scenario,
            "scene_slug": history[-1].sensor_scenario,
            "start_frame": history[0].frame_id,
            "end_frame": history[-1].frame_id,
            "future_start_frame": future[0].frame_id,
            "future_end_frame": future[-1].frame_id,
            "contiguous_segment_id": contiguous_segment_id,
            "segment_ordinal": int(segment_ordinal),
            "segment_start_frame": segment[0].frame_id,
            "segment_end_frame": segment[-1].frame_id,
            "window_start_frame": window_frames[0].frame_id,
            "window_end_frame": window_frames[-1].frame_id,
            "history_frame_ids_json": json.dumps(history_frame_ids),
            "future_frame_ids_json": json.dumps(future_frame_ids),
            "window_frame_ids_json": json.dumps(window_frame_ids),
            "future_label_sequence_json": json.dumps(future_labels),
            "future_label_sequence_key": ",".join(str(label) for label in future_labels),
            "beam_label": future[0].beam_label,
            "coarse_sector": future[0].coarse_sector,
            "radio_semantic_label": future[0].radio_semantic_label if future[0].radio_semantic_label is not None else -100,
            "radio_semantic_available": future[0].radio_semantic_available,
            "radio_semantic_unavailable_reason": future[0].radio_semantic_unavailable_reason,
            "relative_azimuth": future[0].relative_geometry.get("relative_azimuth"),
            "relative_azimuth_bin": _azimuth_bin(future[0].relative_geometry.get("relative_azimuth")),
        }
        for idx, frame in enumerate(history, start=1):
            row[f"camera{idx}"] = frame.camera0
            row[f"lidar{idx}"] = frame.lidar
            row[f"gps{idx}"] = frame.gps
            row[f"geometry{idx}"] = json.dumps(frame.relative_geometry, sort_keys=True)
            row[f"modality_availability{idx}"] = json.dumps(frame.modality_availability, sort_keys=True)
            row[f"mmwave{idx}"] = frame.beam_power_path
            row[f"csi{idx}"] = frame.channel_path
            row[f"beam{idx}"] = frame.beam_power_path
        for idx, frame in enumerate(future, start=1):
            row[f"future_beam{idx}"] = frame.beam_power_path
            row[f"future_csi{idx}"] = frame.channel_path
            row[f"future_path{idx}"] = frame.channel_path
            row[f"future_beam_label{idx}"] = frame.beam_label
            row[f"future_radio_semantic_label{idx}"] = (
                frame.radio_semantic_label if frame.radio_semantic_label is not None else -100
            )
            row[f"future_radio_semantic_available{idx}"] = frame.radio_semantic_available
            row[f"future_radio_semantic_unavailable_reason{idx}"] = frame.radio_semantic_unavailable_reason
        rows.append(row)
    return rows

__all__ = [
    'build_sequence_rows',
    'split_sequence_rows',
    'compute_split_leakage_diagnostics',
    'build_sequence_splits_from_manifest'
]
