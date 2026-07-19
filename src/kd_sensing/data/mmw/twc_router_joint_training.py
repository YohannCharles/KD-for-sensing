"""Fixed balanced Joint Drop/Corrupt panels for Router calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping

from kd_sensing.data.mmw.twc_router_joint_stress import (
    JOINT_RATES,
    MASKS_PER_RATE as MASKS_PER_RATE_SEVERITY,
    STATE_CLEAN,
    STATE_CORRUPT,
    STATE_DROP,
    build_router_joint_stress_cache,
)
from kd_sensing.data.temporal_missing import DEFAULT_TEMPORAL_MODALITIES


PROTOCOL_ID = "mmw_router_joint_training_v1"
SCHEMA_VERSION = 1
GENERATOR = "three_seeded_balanced_joint_stress_panels_v1"
BALANCE_POLICY = "exact_rate_severity_cell_modality_frame_balance_v1"
PANEL_SEED = 20260719
CORRUPTION_SEVERITIES = (1, 2, 3)
MASKS_PER_RATE = MASKS_PER_RATE_SEVERITY * len(CORRUPTION_SEVERITIES)
PANEL_SIZE = len(JOINT_RATES) * MASKS_PER_RATE


def prepare_router_joint_training_panel(path: str | Path, *, seed: int = PANEL_SEED) -> dict[str, Any]:
    """Create the immutable panel, or validate and reuse an identical one."""

    target = Path(path).resolve()
    expected = build_router_joint_training_panel(seed=int(seed))
    if target.is_file():
        existing = load_router_joint_training_panel(target)
        if existing["checksum"] != expected["checksum"]:
            raise ValueError("Existing Router joint-training panel differs from the frozen request.")
        return existing
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return expected


def load_router_joint_training_panel(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_router_joint_training_panel(payload)
    return payload


def build_router_joint_training_panel(*, seed: int = PANEL_SEED) -> dict[str, Any]:
    raw_conditions: list[dict[str, Any]] = []
    for severity in CORRUPTION_SEVERITIES:
        source = build_router_joint_stress_cache(seed=_derived_seed(seed, "severity", severity))
        for rate in JOINT_RATES:
            rate_conditions = [
                item for item in source["conditions"] if float(item["requested_stress_rate"]) == rate
            ]
            if len(rate_conditions) != MASKS_PER_RATE_SEVERITY:
                raise RuntimeError("Joint-stress source panel has an unexpected rate inventory.")
            raw_conditions.extend(
                _condition(
                    item["state_matrix"],
                    rate=rate,
                    severity=severity,
                    mask_index=int(item["mask_set_index"]),
                    condition_index=-1,
                )
                for item in rate_conditions
            )

    random.Random(_derived_seed(seed, "panel-shuffle")).shuffle(raw_conditions)
    conditions = [
        _condition(
            item["state_matrix"],
            rate=float(item["requested_stress_rate"]),
            severity=int(item["corruption_severity"]),
            mask_index=int(item["mask_set_index"]),
            condition_index=index,
        )
        for index, item in enumerate(raw_conditions)
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "generator": GENERATOR,
        "balance_policy": BALANCE_POLICY,
        "seed": int(seed),
        "history_window": 5,
        "modalities": list(DEFAULT_TEMPORAL_MODALITIES),
        "token_count": 20,
        "states": {"clean": STATE_CLEAN, "drop": STATE_DROP, "corrupt": STATE_CORRUPT},
        "joint_rates": list(JOINT_RATES),
        "corruption_severities": list(CORRUPTION_SEVERITIES),
        "masks_per_rate_severity": MASKS_PER_RATE_SEVERITY,
        "masks_per_rate": MASKS_PER_RATE,
        "panel_size": PANEL_SIZE,
        "conditions": conditions,
        "balance_audit": [_rate_audit(conditions, rate) for rate in JOINT_RATES],
    }
    payload["checksum"] = _payload_sha256(payload)
    validate_router_joint_training_panel(payload)
    return payload


def validate_router_joint_training_panel(panel: Mapping[str, Any]) -> None:
    if panel.get("protocol_id") != PROTOCOL_ID or int(panel.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("Unsupported Router joint-training panel.")
    if panel.get("generator") != GENERATOR or panel.get("balance_policy") != BALANCE_POLICY:
        raise ValueError("Router joint-training generator identity mismatch.")
    if list(panel.get("modalities", ())) != list(DEFAULT_TEMPORAL_MODALITIES):
        raise ValueError("Router joint-training modality order mismatch.")
    if int(panel.get("history_window", -1)) != 5 or int(panel.get("token_count", -1)) != 20:
        raise ValueError("Router joint-training temporal geometry mismatch.")
    if tuple(float(value) for value in panel.get("joint_rates", ())) != JOINT_RATES:
        raise ValueError("Router joint-training rate inventory mismatch.")
    if tuple(int(value) for value in panel.get("corruption_severities", ())) != CORRUPTION_SEVERITIES:
        raise ValueError("Router joint-training severity inventory mismatch.")
    if (
        int(panel.get("masks_per_rate_severity", -1)) != MASKS_PER_RATE_SEVERITY
        or int(panel.get("masks_per_rate", -1)) != MASKS_PER_RATE
        or int(panel.get("panel_size", -1)) != PANEL_SIZE
    ):
        raise ValueError("Router joint-training panel cardinality mismatch.")
    recorded = str(panel.get("checksum", ""))
    expected_checksum = _payload_sha256({key: value for key, value in panel.items() if key != "checksum"})
    if recorded != expected_checksum:
        raise ValueError("Router joint-training panel checksum mismatch.")

    conditions = panel.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != PANEL_SIZE:
        raise ValueError("Router joint-training condition inventory mismatch.")
    patterns: set[str] = set()
    digests: set[str] = set()
    for index, condition in enumerate(conditions):
        expected = _condition(
            condition.get("state_matrix"),
            rate=float(condition.get("requested_stress_rate", -1.0)),
            severity=int(condition.get("corruption_severity", -1)),
            mask_index=int(condition.get("mask_set_index", -1)),
            condition_index=index,
        )
        if condition != expected:
            raise ValueError("Router joint-training condition metadata mismatch.")
        pattern = str(condition["pattern"])
        digest = str(condition["condition_digest"])
        if pattern in patterns or digest in digests:
            raise ValueError("Router joint-training conditions must be unique.")
        patterns.add(pattern)
        digests.add(digest)

    audits = panel.get("balance_audit")
    expected_audits = [_rate_audit(conditions, rate) for rate in JOINT_RATES]
    if audits != expected_audits:
        raise ValueError("Router joint-training balance audit mismatch.")


def condition_for_global_row(panel: Mapping[str, Any], global_row: int) -> Mapping[str, Any]:
    """Select one fixed condition by the caller's deterministic global row."""

    if isinstance(global_row, bool) or int(global_row) != global_row or int(global_row) < 0:
        raise ValueError("Router joint-training global_row must be a non-negative integer.")
    conditions = panel.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != PANEL_SIZE:
        raise ValueError("Router joint-training panel has not been validated.")
    return conditions[int(global_row) % PANEL_SIZE]


def _condition(
    matrix: Any,
    *,
    rate: float,
    severity: int,
    mask_index: int,
    condition_index: int,
) -> dict[str, Any]:
    rows = [[int(value) for value in row] for row in matrix] if isinstance(matrix, list) else []
    if len(rows) != 5 or any(len(row) != 4 for row in rows):
        raise ValueError("Router joint-training state matrix must have shape [5,4].")
    flat = [value for row in rows for value in row]
    if any(value not in {STATE_CLEAN, STATE_DROP, STATE_CORRUPT} for value in flat):
        raise ValueError("Router joint-training state matrix contains an invalid state code.")
    if severity not in CORRUPTION_SEVERITIES:
        raise ValueError("Router joint-training corruption severity must be 1, 2, or 3.")
    if rate not in JOINT_RATES or not 0 <= mask_index < MASKS_PER_RATE_SEVERITY:
        raise ValueError("Router joint-training rate or mask index is invalid.")
    state_per_mask = int(round(20 * rate / 2.0))
    counts = {
        "clean": flat.count(STATE_CLEAN),
        "drop": flat.count(STATE_DROP),
        "corrupt": flat.count(STATE_CORRUPT),
    }
    if counts != {"clean": 20 - 2 * state_per_mask, "drop": state_per_mask, "corrupt": state_per_mask}:
        raise ValueError("Router joint-training state cardinality mismatch.")
    severity_matrix = [
        [severity if value == STATE_CORRUPT else 0 for value in row]
        for row in rows
    ]
    rate_percent = int(round(100 * rate))
    identity = {
        "modalities": list(DEFAULT_TEMPORAL_MODALITIES),
        "state_matrix": rows,
        "corruption_severity": severity,
    }
    return {
        "condition_index": int(condition_index),
        "family": "joint_drop_corrupt",
        "pattern": f"joint_{rate_percent:02d}_s{severity}_{mask_index:02d}",
        "requested_stress_rate": float(rate),
        "observed_stress_rate": (counts["drop"] + counts["corrupt"]) / 20.0,
        "drop_rate": counts["drop"] / 20.0,
        "corrupt_rate": counts["corrupt"] / 20.0,
        "corruption_severity": int(severity),
        "state_matrix": rows,
        "severity_matrix": severity_matrix,
        "modality_temporal_mask": [[value != STATE_DROP for value in row] for row in rows],
        "state_counts": counts,
        "state_digest": _payload_sha256({"modalities": identity["modalities"], "state_matrix": rows}),
        "condition_digest": _payload_sha256(identity),
        "mask_set_index": int(mask_index),
        "rate_mask_index": (severity - 1) * MASKS_PER_RATE_SEVERITY + int(mask_index),
        "mask_set_size": MASKS_PER_RATE,
        "balance_policy": BALANCE_POLICY,
    }


def _rate_audit(conditions: list[Mapping[str, Any]], rate: float) -> dict[str, Any]:
    selected = [item for item in conditions if float(item["requested_stress_rate"]) == rate]
    if len(selected) != MASKS_PER_RATE:
        raise ValueError("Router joint-training rate panel is incomplete.")
    state_per_mask = int(round(20 * rate / 2.0))
    severity_counts: dict[str, int] = {}
    severity_audits = []
    for severity in CORRUPTION_SEVERITIES:
        group = [item for item in selected if int(item["corruption_severity"]) == severity]
        if len(group) != MASKS_PER_RATE_SEVERITY:
            raise ValueError("Router joint-training severity panel is incomplete.")
        severity_counts[str(severity)] = len(group)
        severity_audits.append(_cell_audit(group, severity=severity, state_per_mask=state_per_mask))
    aggregate = _cell_audit(selected, severity=None, state_per_mask=state_per_mask * len(CORRUPTION_SEVERITIES))
    return {
        "requested_stress_rate": float(rate),
        "condition_count": len(selected),
        "severity_condition_counts": severity_counts,
        "per_severity": severity_audits,
        **aggregate,
        "balance_status": "exact",
    }


def _cell_audit(
    conditions: list[Mapping[str, Any]],
    *,
    severity: int | None,
    state_per_mask: int,
) -> dict[str, Any]:
    flat = [[value for row in item["state_matrix"] for value in row] for item in conditions]
    drop = [sum(row[cell] == STATE_DROP for row in flat) for cell in range(20)]
    corrupt = [sum(row[cell] == STATE_CORRUPT for row in flat) for cell in range(20)]
    if set(drop) != {state_per_mask} or set(corrupt) != {state_per_mask}:
        raise ValueError("Router joint-training panel is not exactly cell-balanced.")
    clean = [len(conditions) - drop_count - corrupt_count for drop_count, corrupt_count in zip(drop, corrupt)]
    result = {
        "per_cell_clean_counts": clean,
        "per_cell_drop_counts": drop,
        "per_cell_corrupt_counts": corrupt,
        "per_modality_drop_counts": [sum(drop[index::4]) for index in range(4)],
        "per_modality_corrupt_counts": [sum(corrupt[index::4]) for index in range(4)],
        "per_frame_drop_counts": [sum(drop[index * 4 : (index + 1) * 4]) for index in range(5)],
        "per_frame_corrupt_counts": [sum(corrupt[index * 4 : (index + 1) * 4]) for index in range(5)],
    }
    if severity is not None:
        result = {"corruption_severity": severity, "condition_count": len(conditions), **result}
    return result


def _derived_seed(seed: int, *parts: object) -> int:
    encoded = ":".join((str(int(seed)), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "BALANCE_POLICY",
    "CORRUPTION_SEVERITIES",
    "JOINT_RATES",
    "MASKS_PER_RATE",
    "MASKS_PER_RATE_SEVERITY",
    "PANEL_SEED",
    "PANEL_SIZE",
    "PROTOCOL_ID",
    "build_router_joint_training_panel",
    "condition_for_global_row",
    "load_router_joint_training_panel",
    "prepare_router_joint_training_panel",
    "validate_router_joint_training_panel",
]
