"""Fixed three-state modality-frame panels for Router joint stress."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping

from kd_sensing.data.mmw.twc_evidence import DEFAULT_HISTORY_WINDOW
from kd_sensing.data.temporal_missing import (
    DEFAULT_TEMPORAL_MODALITIES,
    build_random_balanced_modality_frame_masks,
)


PROTOCOL_ID = "mmw_router_joint_stress_v1"
SCHEMA_VERSION = 1
GENERATOR = "seeded_random_kof20_regular_bipartite_three_state_v1"
BALANCE_POLICY = "exact_panel_cell_modality_frame_drop_corrupt_balance_v1"
MASK_SEED = 20260719
CORRUPTION_SEVERITY = 2
JOINT_RATES = (0.2, 0.4, 0.6, 0.8)
MASKS_PER_RATE = 20
STATE_CLEAN = 0
STATE_DROP = 1
STATE_CORRUPT = 2
STATE_NAMES = ("clean", "drop", "corrupt")


def prepare_router_joint_stress_cache(path: str | Path, *, seed: int = MASK_SEED) -> dict[str, Any]:
    """Create or validate the immutable screen cache."""

    target = Path(path).resolve()
    expected = build_router_joint_stress_cache(seed=int(seed))
    if target.is_file():
        existing = load_router_joint_stress_cache(target)
        if existing["checksum"] != expected["checksum"]:
            raise ValueError("Existing Router joint-stress cache differs from the frozen request.")
        return existing
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return expected


def load_router_joint_stress_cache(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_cache(payload)
    return payload


def build_router_joint_stress_cache(*, seed: int) -> dict[str, Any]:
    modalities = tuple(DEFAULT_TEMPORAL_MODALITIES)
    conditions = [_condition([[STATE_CLEAN] * len(modalities) for _ in range(DEFAULT_HISTORY_WINDOW)], 0.0, 0, 0)]
    audits = []
    condition_index = 1
    for rate_index, rate in enumerate(JOINT_RATES):
        stressed = int(round(DEFAULT_HISTORY_WINDOW * len(modalities) * rate))
        if stressed % 2:
            raise ValueError("Joint stress requires equal integer Drop and Corrupt counts.")
        retained_masks = build_random_balanced_modality_frame_masks(
            mask_count=MASKS_PER_RATE,
            missing_rate=rate,
            seed=int(seed) + rate_index * 1009,
            history_window=DEFAULT_HISTORY_WINDOW,
            modality_count=len(modalities),
        )
        stressed_rows = [
            {
                time_index * len(modalities) + modality_index
                for time_index, row in enumerate(matrix)
                for modality_index, retained in enumerate(row)
                if not retained
            }
            for matrix in retained_masks
        ]
        # A regular bipartite graph decomposes into perfect matchings. Half become
        # Drop and half Corrupt, preserving the random K-of-20 stress support.
        matchings = _regular_bipartite_matchings(
            stressed_rows,
            degree=stressed,
            seed=int(seed) + rate_index * 1009 + 1,
        )
        drop_per_mask = stressed // 2
        states = [[[STATE_CLEAN] * len(modalities) for _ in range(DEFAULT_HISTORY_WINDOW)] for _ in range(MASKS_PER_RATE)]
        for matching_index, matching in enumerate(matchings):
            state = STATE_DROP if matching_index < drop_per_mask else STATE_CORRUPT
            for mask_index, cell in enumerate(matching):
                time_index, modality_index = divmod(cell, len(modalities))
                states[mask_index][time_index][modality_index] = state
        rate_conditions = []
        for mask_index, matrix in enumerate(states):
            item = _condition(matrix, rate, mask_index, condition_index)
            conditions.append(item)
            rate_conditions.append(item)
            condition_index += 1
        audits.append(_rate_audit(rate_conditions, rate))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "generator": GENERATOR,
        "balance_policy": BALANCE_POLICY,
        "seed": int(seed),
        "history_window": DEFAULT_HISTORY_WINDOW,
        "modalities": list(modalities),
        "token_count": DEFAULT_HISTORY_WINDOW * len(modalities),
        "states": list(STATE_NAMES),
        "state_codes": {name: index for index, name in enumerate(STATE_NAMES)},
        "joint_rates": list(JOINT_RATES),
        "masks_per_rate": MASKS_PER_RATE,
        "corruption_severity": CORRUPTION_SEVERITY,
        "conditions": conditions,
        "rate_balance_audit": audits,
    }
    payload["checksum"] = _payload_sha256(payload)
    _validate_cache(payload)
    return payload


def _regular_bipartite_matchings(rows: list[set[int]], *, degree: int, seed: int) -> list[list[int]]:
    count = len(rows)
    if count != 20 or any(len(row) != degree for row in rows):
        raise ValueError("Joint stress decomposition requires a regular 20x20 support graph.")
    totals = [sum(cell in row for row in rows) for cell in range(count)]
    if set(totals) != {degree}:
        raise ValueError("Joint stress support is not cell-regular.")
    remaining = [set(row) for row in rows]
    rng = random.Random(int(seed))
    result: list[list[int]] = []
    for _ in range(degree):
        cell_to_row = [-1] * count

        def augment(row_index: int, seen: set[int]) -> bool:
            candidates = list(remaining[row_index])
            rng.shuffle(candidates)
            for cell in candidates:
                if cell in seen:
                    continue
                seen.add(cell)
                previous = cell_to_row[cell]
                if previous < 0 or augment(previous, seen):
                    cell_to_row[cell] = row_index
                    return True
            return False

        order = list(range(count))
        rng.shuffle(order)
        if not all(augment(row_index, set()) for row_index in order):
            raise RuntimeError("Could not decompose the regular joint-stress support graph.")
        row_to_cell = [-1] * count
        for cell, row_index in enumerate(cell_to_row):
            row_to_cell[row_index] = cell
        if any(cell < 0 for cell in row_to_cell):
            raise RuntimeError("Joint-stress matching is incomplete.")
        for row_index, cell in enumerate(row_to_cell):
            remaining[row_index].remove(cell)
        result.append(row_to_cell)
    if any(row for row in remaining):
        raise RuntimeError("Joint-stress matching decomposition left unused edges.")
    return result


def _condition(matrix: list[list[int]], rate: float, mask_index: int, condition_index: int) -> dict[str, Any]:
    modality_count = len(DEFAULT_TEMPORAL_MODALITIES)
    if len(matrix) != DEFAULT_HISTORY_WINDOW or any(
        not isinstance(row, list) or len(row) != modality_count for row in matrix
    ):
        raise ValueError("Router joint-stress state matrix must have shape [5,4].")
    flat = [int(value) for row in matrix for value in row]
    if any(value not in {STATE_CLEAN, STATE_DROP, STATE_CORRUPT} for value in flat):
        raise ValueError("Router joint-stress state matrix contains an invalid state code.")
    counts = [flat.count(state) for state in range(len(STATE_NAMES))]
    per_modality = [
        [sum(int(matrix[time][modality]) == state for time in range(DEFAULT_HISTORY_WINDOW)) for state in range(3)]
        for modality in range(modality_count)
    ]
    per_frame = [[sum(int(value) == state for value in row) for state in range(3)] for row in matrix]
    temporal_mask = [[int(value) != STATE_DROP for value in row] for row in matrix]
    available = [
        name
        for name, modality_counts in zip(DEFAULT_TEMPORAL_MODALITIES, per_modality)
        if modality_counts[STATE_DROP] < DEFAULT_HISTORY_WINDOW
    ]
    rate_name = int(round(rate * 100))
    return {
        "condition_index": int(condition_index),
        "family": "clean" if rate == 0.0 else "joint_drop_corrupt",
        "pattern": "clean" if rate == 0.0 else f"joint_{rate_name:02d}_{mask_index:02d}",
        "requested_stress_rate": float(rate),
        "observed_stress_rate": (counts[STATE_DROP] + counts[STATE_CORRUPT]) / 20.0,
        "drop_rate": counts[STATE_DROP] / 20.0,
        "corrupt_rate": counts[STATE_CORRUPT] / 20.0,
        "state_matrix": matrix,
        "modality_temporal_mask": temporal_mask,
        "state_digest": _payload_sha256({"modalities": list(DEFAULT_TEMPORAL_MODALITIES), "state_matrix": matrix}),
        "state_counts": {name: counts[index] for index, name in enumerate(STATE_NAMES)},
        "per_modality_state_counts": per_modality,
        "per_frame_state_counts": per_frame,
        "available_modalities": available,
        "affected_modality_count": sum((item[STATE_DROP] + item[STATE_CORRUPT]) > 0 for item in per_modality),
        "mask_set_index": int(mask_index),
        "mask_set_size": 1 if rate == 0.0 else MASKS_PER_RATE,
        "balance_policy": BALANCE_POLICY,
    }


def _rate_audit(conditions: list[dict[str, Any]], rate: float) -> dict[str, Any]:
    clean_counts = [0] * 20
    drop_counts = [0] * 20
    corrupt_counts = [0] * 20
    composition: dict[str, int] = defaultdict(int)
    for condition in conditions:
        flat = [value for row in condition["state_matrix"] for value in row]
        for cell, value in enumerate(flat):
            clean_counts[cell] += int(value == STATE_CLEAN)
            drop_counts[cell] += int(value == STATE_DROP)
            corrupt_counts[cell] += int(value == STATE_CORRUPT)
        composition[json.dumps(condition["per_modality_state_counts"], separators=(",", ":"))] += 1
    state_per_mask = int(round(20 * rate / 2.0))
    if set(drop_counts) != {state_per_mask} or set(corrupt_counts) != {state_per_mask}:
        raise ValueError("Router joint-stress panel is not exactly cell-balanced.")
    if set(clean_counts) != {MASKS_PER_RATE - 2 * state_per_mask}:
        raise ValueError("Router joint-stress clean state is not exactly cell-balanced.")
    per_modality_drop = [sum(drop_counts[index::4]) for index in range(4)]
    per_modality_corrupt = [sum(corrupt_counts[index::4]) for index in range(4)]
    per_frame_drop = [sum(drop_counts[index * 4 : (index + 1) * 4]) for index in range(5)]
    per_frame_corrupt = [sum(corrupt_counts[index * 4 : (index + 1) * 4]) for index in range(5)]
    return {
        "requested_stress_rate": float(rate),
        "mask_count": len(conditions),
        "drop_count_per_mask": state_per_mask,
        "corrupt_count_per_mask": state_per_mask,
        "per_cell_clean_counts": clean_counts,
        "per_cell_drop_counts": drop_counts,
        "per_cell_corrupt_counts": corrupt_counts,
        "per_modality_drop_counts": per_modality_drop,
        "per_modality_corrupt_counts": per_modality_corrupt,
        "per_frame_drop_counts": per_frame_drop,
        "per_frame_corrupt_counts": per_frame_corrupt,
        "per_mask_modality_composition_histogram": dict(sorted(composition.items())),
        "balance_status": "exact",
    }


def _validate_cache(cache: Mapping[str, Any]) -> None:
    if cache.get("protocol_id") != PROTOCOL_ID or int(cache.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("Unsupported Router joint-stress cache.")
    if cache.get("generator") != GENERATOR or cache.get("balance_policy") != BALANCE_POLICY:
        raise ValueError("Router joint-stress generator identity mismatch.")
    if list(cache.get("modalities", ())) != list(DEFAULT_TEMPORAL_MODALITIES):
        raise ValueError("Router joint-stress modality order mismatch.")
    if int(cache.get("history_window", -1)) != 5 or int(cache.get("token_count", -1)) != 20:
        raise ValueError("Router joint-stress temporal geometry mismatch.")
    if tuple(float(value) for value in cache.get("joint_rates", ())) != JOINT_RATES:
        raise ValueError("Router joint-stress rate inventory mismatch.")
    if int(cache.get("masks_per_rate", -1)) != MASKS_PER_RATE:
        raise ValueError("Router joint-stress mask count mismatch.")
    recorded = str(cache.get("checksum", ""))
    if recorded != _payload_sha256({key: value for key, value in cache.items() if key != "checksum"}):
        raise ValueError("Router joint-stress cache checksum mismatch.")
    conditions = cache.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != 1 + len(JOINT_RATES) * MASKS_PER_RATE:
        raise ValueError("Router joint-stress condition inventory mismatch.")
    if conditions[0].get("pattern") != "clean" or conditions[0].get("state_counts") != {"clean": 20, "drop": 0, "corrupt": 0}:
        raise ValueError("Router joint-stress clean condition mismatch.")
    seen = set()
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for index, condition in enumerate(conditions):
        matrix = condition.get("state_matrix")
        expected = _condition(matrix, float(condition["requested_stress_rate"]), int(condition["mask_set_index"]), index)
        if condition != expected:
            raise ValueError("Router joint-stress condition metadata mismatch.")
        digest = str(condition["state_digest"])
        if digest in seen:
            raise ValueError("Router joint-stress conditions must be unique.")
        seen.add(digest)
        grouped[float(condition["requested_stress_rate"])].append(condition)
    audits = {float(item["requested_stress_rate"]): item for item in cache.get("rate_balance_audit", ())}
    if set(grouped) != {0.0, *JOINT_RATES} or set(audits) != set(JOINT_RATES):
        raise ValueError("Router joint-stress grouped inventory mismatch.")
    for rate in JOINT_RATES:
        if len(grouped[rate]) != MASKS_PER_RATE or audits[rate] != _rate_audit(grouped[rate], rate):
            raise ValueError(f"Router joint-stress rate={rate:g} audit mismatch.")


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CORRUPTION_SEVERITY",
    "JOINT_RATES",
    "MASKS_PER_RATE",
    "MASK_SEED",
    "PROTOCOL_ID",
    "STATE_CLEAN",
    "STATE_CORRUPT",
    "STATE_DROP",
    "build_router_joint_stress_cache",
    "load_router_joint_stress_cache",
    "prepare_router_joint_stress_cache",
]
