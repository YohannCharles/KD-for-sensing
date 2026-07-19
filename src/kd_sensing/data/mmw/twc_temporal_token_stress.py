"""Immutable balanced modality-frame stress artifacts for MMW TWC evidence."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from kd_sensing.data.mmw.twc_evidence import DEFAULT_HISTORY_WINDOW, PROTOCOL_ID as PARENT_PROTOCOL_ID, load_protocol
from kd_sensing.data.temporal_missing import (
    DEFAULT_TEMPORAL_MODALITIES,
    build_random_balanced_modality_frame_masks,
)


STRESS_PROTOCOL_ID = "mmw_twc_temporal_token_stress_v3"
STRESS_PROTOCOL_VERSION = 2
STRESS_PROTOCOL_KIND = "post_selection_confirmation_evaluation_extension_not_historical_blind_test"
STRESS_GENERATOR = "mmw_twc_seeded_random_kof20_cell_balanced_token_stress_v3"
STRESS_MASK_SEED = 20260720
PRIMARY_STRESS_RATES = (0.2, 0.4, 0.6, 0.8, 0.9)
SINGLE_CELL_RATE = 0.95
STRESS_RATES = (*PRIMARY_STRESS_RATES, SINGLE_CELL_RATE)
MASKS_PER_RATE = 100
SINGLE_CELL_MASK_COUNT = 20
STRESS_BALANCE_POLICY = "seeded_random_kof20_minimum_cell_swap_exact_panel_balance_v3"


def prepare_temporal_token_stress_protocol(
    output_root: str | Path,
    *,
    parent_protocol_manifest: str | Path,
    mask_seed: int = STRESS_MASK_SEED,
) -> dict[str, Any]:
    """Create or validate the immutable evaluation-only stress extension."""

    parent_path = Path(parent_protocol_manifest).resolve()
    parent = load_protocol(parent_path)
    target = Path(output_root).resolve()
    request = {
        "protocol_id": STRESS_PROTOCOL_ID,
        "schema_version": STRESS_PROTOCOL_VERSION,
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "parent_protocol_manifest_path": str(parent_path),
        "parent_protocol_manifest_sha256": str(parent["manifest_sha256"]),
        "mask_seed": int(mask_seed),
        "history_window": DEFAULT_HISTORY_WINDOW,
        "modalities": list(DEFAULT_TEMPORAL_MODALITIES),
        "token_count": DEFAULT_HISTORY_WINDOW * len(DEFAULT_TEMPORAL_MODALITIES),
        "rates": list(STRESS_RATES),
        "mask_type": "modality_frame",
        "masks_per_rate": MASKS_PER_RATE,
        "single_cell_mask_count": SINGLE_CELL_MASK_COUNT,
        "per_rate_mask_counts": _per_rate_mask_counts(),
        "generator": STRESS_GENERATOR,
        "balance_policy": STRESS_BALANCE_POLICY,
        "sampling_semantics": "fixed_seeded_random_kof20_then_minimum_cell_swap_panel_balance",
    }
    request_sha256 = _sha256_payload(request)
    manifest_path = target / "protocol_manifest.json"
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        if str(existing.get("request_sha256", "")) != request_sha256:
            raise ValueError(
                "Existing temporal-token stress request differs; use a different output root instead of mutating fixed evidence."
            )
        return load_temporal_token_stress_protocol(manifest_path)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to create stress protocol in non-empty directory: {target}")

    cache = build_balanced_temporal_token_stress_cache(seed=int(mask_seed))
    cache_path = target / "fixed_mask_cache.json"
    _write_json(cache_path, cache)
    domains = [_outer_domain_record(item) for item in parent["domains"]]
    manifest = {
        "schema_version": STRESS_PROTOCOL_VERSION,
        "protocol_id": STRESS_PROTOCOL_ID,
        "protocol_kind": STRESS_PROTOCOL_KIND,
        "request": request,
        "request_sha256": request_sha256,
        "parent_training_protocol": {
            "protocol_id": PARENT_PROTOCOL_ID,
            "protocol_manifest_path": str(parent_path),
            "protocol_manifest_sha256": str(parent["manifest_sha256"]),
            "fixed_mask_cache_sha256": str(parent["fixed_mask_cache"]["sha256"]),
            "fixed_mask_cache_checksum": str(parent["fixed_mask_cache"]["cache_checksum"]),
        },
        "domains": domains,
        "fixed_mask_cache": {
            "path": str(cache_path.resolve()),
            "sha256": _sha256_file(cache_path),
            "cache_checksum": str(cache["checksum"]),
            "condition_count": len(cache["conditions"]),
        },
    }
    manifest["manifest_sha256"] = _sha256_payload(manifest)
    _write_json(manifest_path, manifest)
    return manifest


def load_temporal_token_stress_protocol(path: str | Path) -> dict[str, Any]:
    """Load a stress extension and bind it to its immutable v1 parent protocol."""

    manifest_path = Path(path).resolve()
    manifest = _read_json(manifest_path)
    _validate_stress_manifest(manifest, manifest_path)
    parent_record = manifest["parent_training_protocol"]
    parent_path = Path(str(parent_record["protocol_manifest_path"])).resolve()
    parent = load_protocol(parent_path)
    if str(parent["manifest_sha256"]) != str(parent_record["protocol_manifest_sha256"]):
        raise ValueError("Temporal-token stress parent protocol SHA256 differs from the immutable extension.")
    if _outer_domain_identities(parent) != _stress_domain_identities(manifest):
        raise ValueError("Temporal-token stress outer split differs from its immutable parent protocol.")
    cache_path = Path(str(manifest["fixed_mask_cache"]["path"])).resolve()
    cache = _read_json(cache_path)
    _validate_stress_cache(cache)
    if _sha256_file(cache_path) != str(manifest["fixed_mask_cache"]["sha256"]):
        raise ValueError("Temporal-token stress cache file SHA256 mismatch.")
    if str(cache["checksum"]) != str(manifest["fixed_mask_cache"]["cache_checksum"]):
        raise ValueError("Temporal-token stress cache checksum mismatch.")
    manifest["parent_protocol"] = parent
    manifest["path"] = str(manifest_path)
    manifest["fixed_mask_cache"]["resolved_path"] = str(cache_path)
    return manifest


def build_balanced_temporal_token_stress_cache(*, seed: int) -> dict[str, Any]:
    """Build fixed random masks with exact cardinality and aggregate cell balance."""

    modalities = tuple(DEFAULT_TEMPORAL_MODALITIES)
    token_count = DEFAULT_HISTORY_WINDOW * len(modalities)
    conditions = [_clean_condition(modalities)]
    rate_audits = []
    for rate in STRESS_RATES:
        mask_count = _mask_count_for_rate(rate)
        masks = build_random_balanced_modality_frame_masks(
            mask_count=mask_count,
            missing_rate=rate,
            seed=int(seed),
            history_window=DEFAULT_HISTORY_WINDOW,
            modality_count=len(modalities),
        )
        rate_conditions = [
            _stress_condition(modalities, matrix=matrix, rate=rate, index=index, set_size=mask_count)
            for index, matrix in enumerate(masks)
        ]
        conditions.extend(rate_conditions)
        rate_audits.append(_rate_balance_audit(rate_conditions, rate=rate, token_count=token_count))
    payload = {
        "schema_version": 2,
        "protocol_id": STRESS_PROTOCOL_ID,
        "generator": STRESS_GENERATOR,
        "seed": int(seed),
        "history_window": DEFAULT_HISTORY_WINDOW,
        "modalities": list(modalities),
        "token_count": token_count,
        "rates": list(STRESS_RATES),
        "mask_type": "modality_frame",
        "masks_per_rate": MASKS_PER_RATE,
        "single_cell_mask_count": SINGLE_CELL_MASK_COUNT,
        "per_rate_mask_counts": _per_rate_mask_counts(),
        "balance_policy": STRESS_BALANCE_POLICY,
        "sampling_semantics": "fixed_seeded_random_kof20_then_minimum_cell_swap_panel_balance",
        "conditions": conditions,
        "rate_balance_audit": rate_audits,
    }
    payload["checksum"] = _sha256_payload(payload)
    _validate_stress_cache(payload)
    return payload


def _clean_condition(modalities: tuple[str, ...]) -> dict[str, Any]:
    matrix = [[True] * len(modalities) for _ in range(DEFAULT_HISTORY_WINDOW)]
    return _condition(
        modalities,
        family="clean",
        pattern="clean",
        rate=0.0,
        matrix=matrix,
        index=0,
        set_size=1,
    )


def _stress_condition(
    modalities: tuple[str, ...],
    *,
    matrix: list[list[bool]],
    rate: float,
    index: int,
    set_size: int,
) -> dict[str, Any]:
    return _condition(
        modalities,
        family="temporal_token_stress",
        pattern=f"balanced_modality_frame_{int(round(rate * 100)):02d}_{index:02d}",
        rate=rate,
        matrix=matrix,
        index=index,
        set_size=set_size,
    )


def _condition(
    modalities: tuple[str, ...],
    *,
    family: str,
    pattern: str,
    rate: float,
    matrix: list[list[bool]],
    index: int,
    set_size: int,
) -> dict[str, Any]:
    token_count = DEFAULT_HISTORY_WINDOW * len(modalities)
    retained_by_modality = [sum(bool(row[column]) for row in matrix) for column in range(len(modalities))]
    retained_by_frame = [sum(bool(value) for value in row) for row in matrix]
    retained = sum(retained_by_modality)
    observed_rate = (token_count - retained) / float(token_count)
    if not _close(observed_rate, rate):
        raise ValueError(f"Temporal-token stress rate={rate:g} is not representable by {token_count} cells.")
    return {
        "family": family,
        "pattern": pattern,
        "mask_type": "modality_frame",
        "requested_missing_rate": float(rate),
        "observed_missing_rate": observed_rate,
        "available_modalities": [name for name, count in zip(modalities, retained_by_modality) if count],
        "drop_count": sum(count == 0 for count in retained_by_modality),
        "modality_temporal_mask": matrix,
        "mask_digest": _matrix_digest(matrix),
        "token_count": token_count,
        "retained_token_count": retained,
        "dropped_token_count": token_count - retained,
        "per_modality_retained_counts": retained_by_modality,
        "per_modality_dropped_counts": [DEFAULT_HISTORY_WINDOW - count for count in retained_by_modality],
        "per_frame_retained_counts": retained_by_frame,
        "per_frame_dropped_counts": [len(modalities) - count for count in retained_by_frame],
        "mask_set_index": int(index),
        "mask_set_size": int(set_size),
        "mask_balance_policy": STRESS_BALANCE_POLICY,
    }


def _rate_balance_audit(
    conditions: list[dict[str, Any]], *, rate: float, token_count: int
) -> dict[str, Any]:
    matrices = [condition["modality_temporal_mask"] for condition in conditions]
    per_cell_retained = [
        sum(bool(matrix[time_index][modality_index]) for matrix in matrices)
        for time_index in range(DEFAULT_HISTORY_WINDOW)
        for modality_index in range(len(DEFAULT_TEMPORAL_MODALITIES))
    ]
    per_modality_retained = [
        sum(bool(matrix[time_index][modality_index]) for matrix in matrices for time_index in range(DEFAULT_HISTORY_WINDOW))
        for modality_index in range(len(DEFAULT_TEMPORAL_MODALITIES))
    ]
    per_frame_retained = [
        sum(bool(matrix[time_index][modality_index]) for matrix in matrices for modality_index in range(len(DEFAULT_TEMPORAL_MODALITIES)))
        for time_index in range(DEFAULT_HISTORY_WINDOW)
    ]
    retained = _retained_token_count(rate, token_count)
    if len(conditions) * retained % token_count:
        raise ValueError("Temporal-token stress mask count cannot exactly balance cell marginals.")
    expected_cell_retained = len(conditions) * retained // token_count
    expected_modality_retained = DEFAULT_HISTORY_WINDOW * expected_cell_retained
    expected_frame_retained = len(DEFAULT_TEMPORAL_MODALITIES) * expected_cell_retained
    if set(per_cell_retained) != {expected_cell_retained}:
        raise ValueError("Temporal-token stress cache is not exactly cell-balanced.")
    if set(per_modality_retained) != {expected_modality_retained}:
        raise ValueError("Temporal-token stress cache is not exactly modality-balanced.")
    if set(per_frame_retained) != {expected_frame_retained}:
        raise ValueError("Temporal-token stress cache is not exactly frame-balanced.")
    denominator_modality = len(conditions) * DEFAULT_HISTORY_WINDOW
    denominator_frame = len(conditions) * len(DEFAULT_TEMPORAL_MODALITIES)
    composition_histogram: dict[str, int] = defaultdict(int)
    for condition in conditions:
        key = ",".join(str(value) for value in condition["per_modality_retained_counts"])
        composition_histogram[key] += 1
    return {
        "requested_missing_rate": float(rate),
        "observed_missing_rate": float(rate),
        "mask_count": len(conditions),
        "token_count": token_count,
        "retained_token_count_per_mask": retained,
        "dropped_token_count_per_mask": token_count - retained,
        "per_cell_retained_counts": per_cell_retained,
        "per_cell_retained_target": expected_cell_retained,
        "per_modality_retained_counts": per_modality_retained,
        "per_modality_dropped_counts": [denominator_modality - value for value in per_modality_retained],
        "per_modality_missing_rates": [
            (denominator_modality - value) / float(denominator_modality) for value in per_modality_retained
        ],
        "per_frame_retained_counts": per_frame_retained,
        "per_frame_dropped_counts": [denominator_frame - value for value in per_frame_retained],
        "per_frame_missing_rates": [
            (denominator_frame - value) / float(denominator_frame) for value in per_frame_retained
        ],
        "per_mask_modality_composition_histogram": dict(sorted(composition_histogram.items())),
        "balance_status": "exact",
    }


def _validate_stress_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    if manifest.get("protocol_id") != STRESS_PROTOCOL_ID or int(manifest.get("schema_version", -1)) != STRESS_PROTOCOL_VERSION:
        raise ValueError("Unsupported temporal-token stress protocol manifest.")
    if manifest.get("protocol_kind") != STRESS_PROTOCOL_KIND:
        raise ValueError("Temporal-token stress protocol kind mismatch.")
    recorded = str(manifest.get("manifest_sha256", ""))
    without_hash = {key: value for key, value in manifest.items() if key not in {"manifest_sha256", "parent_protocol", "path"}}
    if recorded != _sha256_payload(without_hash):
        raise ValueError(f"Temporal-token stress manifest checksum mismatch: {path}")
    parent = manifest.get("parent_training_protocol")
    if not isinstance(parent, Mapping) or str(parent.get("protocol_id", "")) != PARENT_PROTOCOL_ID:
        raise ValueError("Temporal-token stress manifest has no compatible v1 parent protocol.")
    domains = manifest.get("domains")
    if not isinstance(domains, list) or len(domains) != 15:
        raise ValueError("Temporal-token stress manifest requires exactly 15 outer-evidence domains.")
    cache = manifest.get("fixed_mask_cache")
    if not isinstance(cache, Mapping) or not str(cache.get("path", "")):
        raise ValueError("Temporal-token stress manifest has no fixed-mask cache.")


def _validate_stress_cache(cache: Mapping[str, Any]) -> None:
    if cache.get("protocol_id") != STRESS_PROTOCOL_ID or cache.get("generator") != STRESS_GENERATOR:
        raise ValueError("Unsupported temporal-token stress cache.")
    if int(cache.get("history_window", -1)) != DEFAULT_HISTORY_WINDOW:
        raise ValueError("Temporal-token stress cache history window mismatch.")
    if list(cache.get("modalities", ())) != list(DEFAULT_TEMPORAL_MODALITIES):
        raise ValueError("Temporal-token stress cache modality order mismatch.")
    if int(cache.get("token_count", -1)) != DEFAULT_HISTORY_WINDOW * len(DEFAULT_TEMPORAL_MODALITIES):
        raise ValueError("Temporal-token stress cache token cardinality mismatch.")
    if tuple(float(item) for item in cache.get("rates", ())) != STRESS_RATES:
        raise ValueError("Temporal-token stress cache rate inventory mismatch.")
    if (
        cache.get("mask_type") != "modality_frame"
        or int(cache.get("masks_per_rate", -1)) != MASKS_PER_RATE
        or int(cache.get("single_cell_mask_count", -1)) != SINGLE_CELL_MASK_COUNT
        or cache.get("per_rate_mask_counts") != _per_rate_mask_counts()
    ):
        raise ValueError("Temporal-token stress cache mask contract mismatch.")
    recorded = str(cache.get("checksum", ""))
    if recorded != _sha256_payload({key: value for key, value in cache.items() if key != "checksum"}):
        raise ValueError("Temporal-token stress cache checksum mismatch.")
    conditions = cache.get("conditions")
    expected_condition_count = 1 + len(PRIMARY_STRESS_RATES) * MASKS_PER_RATE + SINGLE_CELL_MASK_COUNT
    if not isinstance(conditions, list) or len(conditions) != expected_condition_count:
        raise ValueError("Temporal-token stress cache condition inventory mismatch.")
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for condition in conditions:
        if not isinstance(condition, dict):
            raise ValueError("Temporal-token stress condition must be a mapping.")
        matrix = condition.get("modality_temporal_mask")
        if not isinstance(matrix, list) or len(matrix) != DEFAULT_HISTORY_WINDOW or any(
            not isinstance(row, list) or len(row) != len(DEFAULT_TEMPORAL_MODALITIES) for row in matrix
        ):
            raise ValueError("Temporal-token stress condition matrix must be 5x4.")
        digest = _matrix_digest([[bool(value) for value in row] for row in matrix])
        if digest != str(condition.get("mask_digest", "")) or digest in seen:
            raise ValueError("Temporal-token stress condition identity mismatch.")
        seen.add(digest)
        rate = float(condition.get("requested_missing_rate", -1.0))
        retained = sum(bool(value) for row in matrix for value in row)
        if int(condition.get("retained_token_count", -1)) != retained:
            raise ValueError("Temporal-token stress retained token count mismatch.")
        if int(condition.get("dropped_token_count", -1)) != 20 - retained:
            raise ValueError("Temporal-token stress dropped token count mismatch.")
        expected_set_size = 1 if _close(rate, 0.0) else _mask_count_for_rate(rate)
        if int(condition.get("mask_set_size", -1)) != expected_set_size:
            raise ValueError("Temporal-token stress mask-set size mismatch.")
        observed = (20 - retained) / 20.0
        if not _close(observed, rate) or not _close(observed, float(condition.get("observed_missing_rate", -1.0))):
            raise ValueError("Temporal-token stress requested/observed rate mismatch.")
        grouped[rate].append(condition)
    if set(grouped) != {0.0, *STRESS_RATES} or len(grouped[0.0]) != 1:
        raise ValueError("Temporal-token stress clean/rate condition inventory mismatch.")
    audits = {float(item["requested_missing_rate"]): item for item in cache.get("rate_balance_audit", [])}
    if set(audits) != set(STRESS_RATES):
        raise ValueError("Temporal-token stress balance audit inventory mismatch.")
    for rate in STRESS_RATES:
        rate_conditions = grouped[rate]
        if len(rate_conditions) != _mask_count_for_rate(rate):
            raise ValueError(f"Temporal-token stress rate={rate:g} mask count mismatch.")
        expected = _rate_balance_audit(rate_conditions, rate=rate, token_count=20)
        if audits[rate] != expected:
            raise ValueError(f"Temporal-token stress rate={rate:g} balance audit mismatch.")


def _outer_domain_record(parent_domain: Mapping[str, Any]) -> dict[str, Any]:
    outer = parent_domain.get("split", {}).get("outer_evidence")
    if not isinstance(outer, Mapping):
        raise ValueError("Parent protocol has no outer-evidence split.")
    return {
        "id": str(parent_domain["id"]),
        "condition": str(parent_domain["condition"]),
        "scene": str(parent_domain["scene"]),
        "split": {
            "outer_evidence": {
                "csv": str(outer["csv"]),
                "sha256": str(outer["sha256"]),
                "row_count": int(outer["row_count"]),
            }
        },
    }


def _outer_domain_identities(parent: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [_outer_domain_record(item) for item in parent["domains"]]


def _stress_domain_identities(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in manifest["domains"]]


def _retained_token_count(rate: float, token_count: int) -> int:
    dropped = int(round(float(rate) * token_count))
    if not _close(dropped / float(token_count), rate):
        raise ValueError(f"Rate {rate:g} is not an exact {token_count}-cell cardinality.")
    retained = token_count - dropped
    if retained <= 0:
        raise ValueError("Temporal-token stress must retain at least one modality-frame cell.")
    return retained


def _mask_count_for_rate(rate: float) -> int:
    return SINGLE_CELL_MASK_COUNT if _close(rate, SINGLE_CELL_RATE) else MASKS_PER_RATE


def _per_rate_mask_counts() -> dict[str, int]:
    return {f"{rate:g}": _mask_count_for_rate(rate) for rate in STRESS_RATES}


def _matrix_digest(matrix: list[list[bool]]) -> str:
    return _sha256_payload({"modalities": list(DEFAULT_TEMPORAL_MODALITIES), "modality_temporal_mask": matrix})


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be a mapping: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _close(first: float, second: float, tolerance: float = 1.0e-12) -> bool:
    return abs(float(first) - float(second)) <= tolerance


__all__ = [
    "MASKS_PER_RATE",
    "PRIMARY_STRESS_RATES",
    "SINGLE_CELL_MASK_COUNT",
    "SINGLE_CELL_RATE",
    "STRESS_GENERATOR",
    "STRESS_MASK_SEED",
    "STRESS_PROTOCOL_ID",
    "STRESS_PROTOCOL_KIND",
    "STRESS_RATES",
    "build_balanced_temporal_token_stress_cache",
    "load_temporal_token_stress_protocol",
    "prepare_temporal_token_stress_protocol",
]
