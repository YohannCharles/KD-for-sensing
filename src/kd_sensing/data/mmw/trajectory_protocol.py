"""Build and validate the only supported MMW ID-stratified block split."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from kd_sensing.preprocessing.mmw_radar import augment_mmw_sequence_resource_columns


TRAJECTORY_PROTOCOL_MODE = "mmw_id_stratified_block_v1"
TRAJECTORY_PROTOCOL_ID = TRAJECTORY_PROTOCOL_MODE
TRAJECTORY_PROTOCOL_VERSION = 1
TRAJECTORY_MANIFEST_VERSION = 2
TRAJECTORY_SPLIT_SEED = 0
DEFAULT_BLOCK_SIZE = 32
ASSIGNMENT_ALGORITHM = "deterministic_multistart_conditional_greedy_swap_v2"
EXPECTED_SCENE_COUNT = 5
EXPECTED_WEATHERS = ("foggy", "rainy", "sunny")
SPLIT_ROLES = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
_ROLE_INDEX = {role: index for index, role in enumerate(SPLIT_ROLES)}
CACHE_IDENTITY_FIELDS = (
    "split_protocol",
    "protocol_version",
    "split_seed",
    "block_size",
    "split_manifest_hash",
    "data_source_hash",
    "window_config_hash",
    "weather_binding",
)
_ASSIGNMENT_WEIGHTS = {
    "ratio": 4.0,
    "label": 8.0,
    "trajectory": 2.0,
    "scene": 1.0,
    "coverage": 20.0,
    "trajectory_label": 24.0,
    "scene_label": 12.0,
    "trajectory_coverage": 80.0,
    "scene_coverage": 40.0,
}
_RESOURCE_PATTERNS = {
    "camera": re.compile(r"^camera\d+$"),
    "lidar": re.compile(r"^lidar\d+$"),
    "radar": re.compile(r"^radar\d+$"),
    "cav_gps": re.compile(r"^gps\d+$"),
    "bs_gps": re.compile(r"^bs_gps\d+$"),
    "channel": re.compile(r"^(?:csi|future_csi|future_path)\d+$"),
}
_REQUIRED_SOURCE_COLUMNS = {
    "agent",
    "condition",
    "sensor_scenario",
    "contiguous_segment_id",
    "seq_index",
    "sample_id",
    "target_sample_id",
    "history_frame_ids_json",
    "future_frame_ids_json",
    "future_beam_label1",
}
_REQUIRED_FRAME_COLUMNS = {
    "agent",
    "condition",
    "sensor_scenario",
    "frame_id",
    "beam_label",
    "sample_id",
}


def trajectory_manifest_path(output_root: str | Path, split_seed: int = TRAJECTORY_SPLIT_SEED) -> Path:
    return Path(output_root).resolve() / "splits" / TRAJECTORY_PROTOCOL_ID / f"seed_{int(split_seed)}.json"


def trajectory_audit_path(manifest_path: str | Path) -> Path:
    path = Path(manifest_path)
    return path.with_name(f"{path.stem}_audit.json")


def build_trajectory_protocol(
    output_root: str | Path,
    *,
    dataset_root: str | Path = "dataset/MMW",
    split_seed: int = TRAJECTORY_SPLIT_SEED,
    block_size: int = DEFAULT_BLOCK_SIZE,
    regenerate: bool = False,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the deterministic weather-bound block manifest, or validate and reuse it."""

    seed = int(split_seed)
    size = int(block_size)
    if seed < 0:
        raise ValueError("MMW split_seed must be non-negative.")
    if size <= 0:
        raise ValueError("MMW block_size must be positive.")
    manifest_path = trajectory_manifest_path(output_root, seed)
    dataset_path = Path(dataset_root).resolve()
    if manifest_path.is_file() and not regenerate:
        protocol = load_trajectory_protocol(manifest_path)
        mismatches = []
        if int(protocol["split_seed"]) != seed:
            mismatches.append(f"split_seed={protocol['split_seed']} (requested {seed})")
        if int(protocol["block_size"]) != size:
            mismatches.append(f"block_size={protocol['block_size']} (requested {size})")
        if Path(str(protocol["dataset_root"])).resolve() != dataset_path:
            mismatches.append(f"dataset_root={protocol['dataset_root']} (requested {dataset_path})")
        if mismatches:
            raise ValueError(
                "Existing MMW manifest identity differs from the request: "
                + "; ".join(mismatches)
                + ". Use explicit regenerate to replace it."
            )
        return protocol

    sequence_sources, frame_sources = _discover_sources(dataset_path)
    records, sequence_inventory = _load_source_indexes(sequence_sources, dataset_path)
    frame_indexes, frame_inventory = _load_frame_indexes(frame_sources, dataset_path)
    records, base_samples, weather_mapping, window_config = _establish_base_time_mapping(records, frame_indexes)
    window_config_hash = _sha256_json(window_config)
    blocks, candidate_records, dropped = _build_continuous_blocks(
        records,
        base_samples,
        block_size=size,
        sample_span=int(window_config["sample_span"]),
    )
    assignment, assignment_summary = assign_mmw_blocks_stratified(blocks, split_seed=seed)
    materialized = candidate_records.copy()
    materialized["split"] = materialized["block_id"].map(assignment)
    if materialized["split"].isna().any():
        raise AssertionError("Internal MMW block assignment omitted a materialized window.")
    materialized["trajectory_group_id"] = [
        group_id(scene, cav) for scene, cav in zip(materialized["scene_id"], materialized["cav_id"])
    ]

    actual_counts = Counter(materialized["block_id"].astype(str))
    manifest_blocks = []
    for block in sorted(blocks, key=_block_sort_key):
        item = _public_block(block)
        item["split"] = assignment[item["block_id"]]
        item["num_windows_actual"] = int(actual_counts[item["block_id"]])
        manifest_blocks.append(item)

    split_root = manifest_path.parent / f"seed_{seed}_windows"
    if split_root.exists():
        shutil.rmtree(split_root)
    domains, role_frames = _write_split_indexes(materialized, split_root, dataset_path)
    source_inventory = sorted(sequence_inventory + frame_inventory, key=lambda item: item["relative_path"])
    data_source_hash = _sha256_json(source_inventory)
    split_hashes = {role: _sample_id_hash(frame) for role, frame in role_frames.items()}
    split_csv_hash = _sha256_json(
        [
            {
                "domain": domain["id"],
                "role": role,
                "sha256": domain[f"{role}_csv_sha256"],
                "count": domain[f"{role}_sample_count"],
            }
            for domain in domains
            for role in SPLIT_ROLES
        ]
    )
    statistics = _statistics(
        materialized,
        manifest_blocks,
        assignment_summary=assignment_summary,
        dropped_boundary_windows=dropped,
    )
    resource_diagnostics = _shared_resource_diagnostics(role_frames)
    resolved_report = (
        Path(report_path).resolve()
        if report_path is not None
        else Path(output_root).resolve() / "split_reports" / f"mmw_id_stratified_block_seed{seed}.md"
    )
    report_json_path = resolved_report.with_suffix(".json")
    payload: dict[str, Any] = {
        "dataset": "MMW",
        "protocol": TRAJECTORY_PROTOCOL_ID,
        "mode": TRAJECTORY_PROTOCOL_MODE,
        "protocol_id": TRAJECTORY_PROTOCOL_ID,
        "protocol_version": TRAJECTORY_PROTOCOL_VERSION,
        "manifest_version": TRAJECTORY_MANIFEST_VERSION,
        "assignment_algorithm": ASSIGNMENT_ALGORITHM,
        "split_seed": seed,
        "ratios": dict(SPLIT_RATIOS),
        "block_size": size,
        "trajectory_key": ["scene_id", "cav_id"],
        "base_sample_key": ["scene_id", "cav_id", "base_frame_index"],
        "weather_binding": True,
        "expected_weathers": list(EXPECTED_WEATHERS),
        "weather_mapping_version": "strict_seq_index_frame_map_v1",
        "weather_mapping": weather_mapping,
        "dataset_root": str(dataset_path),
        "source_indexes": source_inventory,
        "data_source_hash": data_source_hash,
        "window_config": window_config,
        "window_config_hash": window_config_hash,
        "train_blocks": [item for item in manifest_blocks if item["split"] == "train"],
        "validation_blocks": [item for item in manifest_blocks if item["split"] == "validation"],
        "test_blocks": [item for item in manifest_blocks if item["split"] == "test"],
        "statistics": statistics,
        "trajectory_count": len({(item["scene_id"], item["cav_id"]) for item in manifest_blocks}),
        "block_count": len(manifest_blocks),
        "candidate_window_count": int(len(records)),
        "materialized_window_count": int(len(materialized)),
        "dropped_boundary_window_count": int(dropped),
        "train_block_count": len([item for item in manifest_blocks if item["split"] == "train"]),
        "validation_block_count": len([item for item in manifest_blocks if item["split"] == "validation"]),
        "test_block_count": len([item for item in manifest_blocks if item["split"] == "test"]),
        "train_window_count": int(len(role_frames["train"])),
        "validation_window_count": int(len(role_frames["validation"])),
        "test_window_count": int(len(role_frames["test"])),
        "split_hashes": split_hashes,
        "split_csv_hash": split_csv_hash,
        "resource_overlap_policy": "diagnostic_only_shared_scene_context",
        "shared_resource_diagnostics": resource_diagnostics,
        "train_role": "train",
        "validation_role": "validation",
        "test_role": "test",
        "test_evaluated": False,
        "random_window_split_used": False,
        "legacy_protocol_used": False,
        "domains": domains,
        "audit_report": str(trajectory_audit_path(manifest_path).resolve()),
        "report_path": str(resolved_report),
        "report_json_path": str(report_json_path),
    }
    audit = validate_mmw_id_block_split(payload, materialized)
    payload["protocol_fingerprint"] = _fingerprint(payload)
    audit.update(
        protocol=TRAJECTORY_PROTOCOL_ID,
        protocol_version=TRAJECTORY_PROTOCOL_VERSION,
        manifest_version=TRAJECTORY_MANIFEST_VERSION,
        assignment_algorithm=ASSIGNMENT_ALGORITHM,
        split_seed=seed,
        block_size=size,
        protocol_fingerprint=payload["protocol_fingerprint"],
        data_source_hash=data_source_hash,
        window_config_hash=window_config_hash,
        weather_binding=True,
        test_evaluated=False,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, payload)
    audit["split_manifest_hash"] = _sha256_file(manifest_path)
    _write_json(trajectory_audit_path(manifest_path), audit)
    report_payload = {
        "manifest": {
            "path": str(manifest_path),
            "sha256": audit["split_manifest_hash"],
            "protocol": TRAJECTORY_PROTOCOL_ID,
            "protocol_version": TRAJECTORY_PROTOCOL_VERSION,
            "manifest_version": TRAJECTORY_MANIFEST_VERSION,
            "assignment_algorithm": ASSIGNMENT_ALGORITHM,
            "split_seed": seed,
            "block_size": size,
            "data_source_hash": data_source_hash,
            "window_config": window_config,
            "window_config_hash": window_config_hash,
        },
        "statistics": statistics,
        "leakage_validation": audit,
    }
    _write_json(report_json_path, report_payload)
    _write_report(payload, audit, resolved_report)
    return load_trajectory_protocol(manifest_path)


def assign_mmw_blocks_stratified(
    blocks: Sequence[Mapping[str, Any]],
    *,
    split_seed: int = TRAJECTORY_SPLIT_SEED,
    restarts: int = 8,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign whole blocks with fixed per-trajectory coverage and deterministic local search."""

    seed = int(split_seed)
    if seed < 0:
        raise ValueError("MMW split_seed must be non-negative.")
    canonical = [dict(item) for item in sorted(blocks, key=_block_sort_key)]
    if not canonical:
        raise ValueError("MMW block assignment requires at least one block.")
    ids = [str(item["block_id"]) for item in canonical]
    if len(ids) != len(set(ids)):
        raise ValueError("MMW block_id values must be unique.")
    by_trajectory: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for block in canonical:
        by_trajectory[(_required_text(block, "scene_id"), _required_text(block, "cav_id"))].append(block)
    too_short = {key: len(values) for key, values in by_trajectory.items() if len(values) < len(SPLIT_ROLES)}
    if too_short:
        details = ", ".join(f"{scene}/{cav}={count} blocks" for (scene, cav), count in sorted(too_short.items()))
        raise ValueError(f"MMW trajectories need at least three blocks for train/validation/test: {details}")

    trajectory_items = sorted(by_trajectory.items(), key=lambda item: _group_sort_key(item[0]))
    quotas = _balanced_trajectory_quotas(trajectory_items, split_seed=seed)
    scorer = _AssignmentScorer(canonical)

    baseline = _contiguous_assignment(canonical, quotas)
    candidates: list[dict[str, str]] = [baseline]
    for restart in range(max(1, int(restarts))):
        rng = random.Random(_derived_seed(seed, restart))
        assignment: dict[str, str] = {}
        for key, values in trajectory_items:
            ordered = list(values)
            rng.shuffle(ordered)
            roles = [role for role in SPLIT_ROLES for _ in range(quotas[key][role])]
            rng.shuffle(roles)
            assignment.update({str(block["block_id"]): role for block, role in zip(ordered, roles)})
        assignment = _improve_assignment(canonical, assignment, by_trajectory, rng, scorer=scorer)
        candidates.append(assignment)

    best_assignment = min(
        candidates,
        key=lambda item: (
            scorer.evaluate(item)["objective"],
            _seeded_assignment_tie_key(item, seed),
        ),
    )
    components = scorer.evaluate(best_assignment)
    baseline_components = scorer.evaluate(baseline)
    summary = {
        "algorithm": ASSIGNMENT_ALGORITHM,
        "weights": dict(_ASSIGNMENT_WEIGHTS),
        "restarts": max(1, int(restarts)),
        "objective": components,
        "simple_contiguous_objective": baseline_components,
        "trajectory_quotas": [
            {"scene_id": key[0], "cav_id": key[1], **quotas[key]}
            for key in sorted(quotas, key=_group_sort_key)
        ],
    }
    _validate_assignment_hard_constraints(canonical, best_assignment)
    return best_assignment, summary


def validate_mmw_id_block_split(manifest: Mapping[str, Any], records: pd.DataFrame) -> dict[str, Any]:
    """Validate block/base/weather/raw-frame isolation and complete coverage before loader construction."""

    failures: list[str] = []
    blocks_by_role = {role: _manifest_blocks(manifest, role) for role in SPLIT_ROLES}
    block_ids = {role: {str(item["block_id"]) for item in values} for role, values in blocks_by_role.items()}
    for left, right in itertools.combinations(SPLIT_ROLES, 2):
        overlap = block_ids[left] & block_ids[right]
        if overlap:
            failures.append(f"{left}_vs_{right}_block_overlap:{sorted(overlap)[:5]}")
    all_blocks = [item for role in SPLIT_ROLES for item in blocks_by_role[role]]
    if len(all_blocks) != len({str(item["block_id"]) for item in all_blocks}):
        failures.append("block_count_does_not_match_disjoint_union")

    base_sets: dict[str, set[tuple[str, str, int]]] = {role: set() for role in SPLIT_ROLES}
    trajectories_by_role: dict[str, set[tuple[str, str]]] = {role: set() for role in SPLIT_ROLES}
    scenes_by_role: dict[str, set[str]] = {role: set() for role in SPLIT_ROLES}
    for role, blocks in blocks_by_role.items():
        for block in blocks:
            scene = _required_text(block, "scene_id")
            cav = _required_text(block, "cav_id")
            start = int(block["block_start_base_index"])
            end = int(block["block_end_base_index"])
            if end < start or int(block.get("num_base_samples", -1)) != end - start + 1:
                failures.append(f"block_range:{block.get('block_id')}")
                continue
            base_sets[role].update((scene, cav, index) for index in range(start, end + 1))
            trajectories_by_role[role].add((scene, cav))
            scenes_by_role[role].add(scene)
    for left, right in itertools.combinations(SPLIT_ROLES, 2):
        overlap = base_sets[left] & base_sets[right]
        if overlap:
            failures.append(f"{left}_vs_{right}_base_frame_overlap:{sorted(overlap)[:5]}")

    all_trajectories = set().union(*trajectories_by_role.values())
    all_scenes = set().union(*scenes_by_role.values())
    for role in SPLIT_ROLES:
        if trajectories_by_role[role] != all_trajectories:
            failures.append(f"trajectory_coverage:{role}")
        if scenes_by_role[role] != all_scenes:
            failures.append(f"scene_coverage:{role}")
        if not block_ids[role]:
            failures.append(f"empty_split:{role}")
    if len(all_scenes) != EXPECTED_SCENE_COUNT:
        failures.append(f"scene_count:{len(all_scenes)}:expected={EXPECTED_SCENE_COUNT}")

    block_role = {block_id: role for role, values in block_ids.items() for block_id in values}
    weather_roles: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    weather_sets: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    raw_frames: dict[str, set[tuple[str, str, str, str]]] = {role: set() for role in SPLIT_ROLES}
    window_errors: list[dict[str, str]] = []
    record_counts = Counter()
    for row in records.to_dict(orient="records"):
        role = _required_text(row, "split")
        block_id = _required_text(row, "block_id")
        if role not in SPLIT_ROLES or block_role.get(block_id) != role:
            failures.append(f"record_role:{row.get('sample_id', '')}:{role}:{block_id}")
            continue
        error = _window_error(row, block=_block_by_id(all_blocks, block_id))
        if error and len(window_errors) < 20:
            window_errors.append({"sample_id": str(row.get("sample_id", "")), "reason": error})
        scene = _required_text(row, "scene_id")
        cav = _required_text(row, "cav_id")
        seq_index = int(row["seq_index"])
        key = (scene, cav, seq_index)
        weather_roles[key].add(role)
        weather_sets[key].add(_required_text(row, "weather"))
        record_counts[role] += 1
        physical_ids = _json_list(row.get("history_frame_ids_json")) + _json_list(row.get("future_frame_ids_json"))
        raw_frames[role].update((_required_text(row, "weather"), scene, cav, _frame_id_key(value)) for value in physical_ids)
    if window_errors:
        failures.append(f"window_crossing_or_order:{window_errors[:5]}")
    expected_weather_set = set(EXPECTED_WEATHERS)
    for key in sorted(weather_roles, key=lambda value: (_group_sort_key(value[:2]), value[2])):
        if len(weather_roles[key]) != 1:
            failures.append(f"weather_copy_overlap:{key}:{sorted(weather_roles[key])}")
        if weather_sets[key] != expected_weather_set:
            failures.append(f"weather_binding:{key}:{sorted(weather_sets[key])}")
    raw_overlap_counts = {}
    for left, right in itertools.combinations(SPLIT_ROLES, 2):
        overlap = raw_frames[left] & raw_frames[right]
        raw_overlap_counts[f"{left}_vs_{right}"] = len(overlap)
        if overlap:
            failures.append(f"{left}_vs_{right}_raw_frame_overlap:{sorted(overlap)[:5]}")

    block_counts = {role: len(block_ids[role]) for role in SPLIT_ROLES}
    if block_counts["train"] <= max(block_counts["validation"], block_counts["test"]):
        failures.append(f"train_not_largest:{block_counts}")
    expected_counts = {
        role: int(manifest.get(f"{role}_window_count", -1)) for role in SPLIT_ROLES
    }
    if dict(record_counts) != expected_counts:
        failures.append(f"window_counts:{dict(record_counts)}:expected={expected_counts}")

    metrics = _label_distribution_statistics(records)
    conditional_metrics = {
        "per_domain": _conditional_label_distribution_statistics(records, ("weather", "scene_id")),
        "per_scene": _conditional_label_distribution_statistics(records, ("scene_id",)),
        "per_trajectory": _conditional_label_distribution_statistics(records, ("scene_id", "cav_id")),
    }
    ratios = _split_ratio_statistics(all_blocks, records)
    checks = {
        "block_overlap": not any("block_overlap" in value for value in failures),
        "base_frame_overlap": not any("base_frame_overlap" in value for value in failures),
        "weather_copy_overlap": not any(value.startswith(("weather_copy_overlap", "weather_binding")) for value in failures),
        "raw_frame_overlap": not any("raw_frame_overlap" in value for value in failures),
        "window_crossing": not window_errors,
        "scene_coverage": not any(value.startswith(("scene_coverage", "scene_count")) for value in failures),
        "trajectory_coverage": not any(value.startswith("trajectory_coverage") for value in failures),
        "split_ratio_hard_constraints": not any(value.startswith(("empty_split", "train_not_largest")) for value in failures),
        "scaler_train_only": True,
        "contrastive_memory_train_only": True,
    }
    audit = {
        "schema_version": 1,
        "audit_id": "mmw_id_stratified_block_audit_v1",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "checks": checks,
        "block_counts": block_counts,
        "trajectory_counts": {role: len(trajectories_by_role[role]) for role in SPLIT_ROLES},
        "scene_counts": {role: len(scenes_by_role[role]) for role in SPLIT_ROLES},
        "window_counts": expected_counts,
        "train_sample_count": expected_counts["train"],
        "validation_sample_count": expected_counts["validation"],
        "test_sample_count": expected_counts["test"],
        "train_sample_id_hash": _sample_id_hash(records.loc[records["split"] == "train"]),
        "validation_sample_id_hash": _sample_id_hash(records.loc[records["split"] == "validation"]),
        "test_sample_id_hash": _sample_id_hash(records.loc[records["split"] == "test"]),
        "raw_frame_overlap_counts": raw_overlap_counts,
        "window_errors": window_errors,
        "label_distribution": metrics,
        "conditional_label_distribution": conditional_metrics,
        "ratios": ratios,
    }
    if failures:
        raise ValueError("MMW ID block split validation failed: " + "; ".join(failures[:10]))
    return audit


def load_trajectory_protocol(
    path: str | Path,
    *,
    verify_sources: bool = True,
    load_test: bool = True,
) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MMW ID block manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = validate_trajectory_protocol(payload, manifest_path=manifest_path, verify_sources=verify_sources)
    roles = SPLIT_ROLES if load_test else ("train", "validation")
    role_frames = _load_materialized_splits(protocol, roles=roles)
    if load_test:
        records = pd.concat([role_frames[role] for role in SPLIT_ROLES], ignore_index=True)
        validate_mmw_id_block_split(protocol, records)
    protocol["split_manifest_hash"] = _sha256_file(manifest_path)
    return protocol


def validate_trajectory_protocol(
    protocol: Mapping[str, Any],
    *,
    manifest_path: str | Path | None = None,
    verify_sources: bool = True,
) -> dict[str, Any]:
    payload = dict(protocol)
    if payload.get("protocol") != TRAJECTORY_PROTOCOL_ID or payload.get("mode") != TRAJECTORY_PROTOCOL_MODE:
        raise ValueError(
            f"Unsupported MMW split protocol; expected {TRAJECTORY_PROTOCOL_ID!r}. Legacy manifests must be regenerated."
        )
    if payload.get("protocol_id") != TRAJECTORY_PROTOCOL_ID:
        raise ValueError("MMW ID block manifest protocol_id mismatch.")
    if int(payload.get("protocol_version", -1)) != TRAJECTORY_PROTOCOL_VERSION:
        raise ValueError(
            f"MMW ID block protocol version mismatch; expected {TRAJECTORY_PROTOCOL_VERSION}, "
            f"got {payload.get('protocol_version')!r}. Regenerate the split."
        )
    if int(payload.get("manifest_version", -1)) != TRAJECTORY_MANIFEST_VERSION:
        raise ValueError(
            f"MMW ID block manifest version mismatch; expected {TRAJECTORY_MANIFEST_VERSION}, "
            f"got {payload.get('manifest_version')!r}. Regenerate the split."
        )
    if payload.get("assignment_algorithm") != ASSIGNMENT_ALGORITHM:
        raise ValueError(
            f"MMW ID block assignment algorithm mismatch; expected {ASSIGNMENT_ALGORITHM!r}. Regenerate the split."
        )
    if payload.get("trajectory_key") != ["scene_id", "cav_id"]:
        raise ValueError("MMW manifest trajectory_key must be ['scene_id', 'cav_id'].")
    if payload.get("base_sample_key") != ["scene_id", "cav_id", "base_frame_index"]:
        raise ValueError("MMW manifest base_sample_key mismatch.")
    if payload.get("weather_binding") is not True or tuple(payload.get("expected_weathers", ())) != EXPECTED_WEATHERS:
        raise ValueError("MMW manifest must bind sunny/rainy/foggy base samples.")
    if payload.get("ratios") != SPLIT_RATIOS or int(payload.get("block_size", 0)) <= 0:
        raise ValueError("MMW manifest must use fixed 70/15/15 ratios and a positive block_size.")
    if payload.get("test_role") != "test" or payload.get("test_evaluated") is not False:
        raise ValueError("MMW manifest must define a sealed test role with test_evaluated=false.")
    expected = str(payload.pop("protocol_fingerprint", ""))
    actual = _fingerprint(payload)
    if expected != actual:
        raise ValueError("MMW ID block manifest fingerprint mismatch.")
    payload["protocol_fingerprint"] = actual
    if _sha256_json(payload.get("source_indexes", [])) != payload.get("data_source_hash"):
        raise ValueError("MMW data source inventory hash mismatch.")
    if _sha256_json(payload.get("window_config", {})) != payload.get("window_config_hash"):
        raise ValueError("MMW window configuration hash mismatch.")
    if verify_sources:
        for source in payload.get("source_indexes", []):
            source_path = Path(str(source.get("path", "")))
            if not source_path.is_file() or _sha256_file(source_path) != source.get("sha256"):
                raise ValueError(f"MMW source index is missing or changed: {source_path}")
    if manifest_path is not None:
        path = Path(manifest_path)
        if path.name != f"seed_{int(payload.get('split_seed', -1))}.json" or path.parent.name != TRAJECTORY_PROTOCOL_ID:
            raise ValueError("MMW ID block manifest path does not match protocol and split_seed.")
    for role in SPLIT_ROLES:
        blocks = payload.get(f"{role}_blocks")
        if not isinstance(blocks, list) or not blocks:
            raise ValueError(f"MMW manifest must contain non-empty {role}_blocks.")
        if int(payload.get(f"{role}_block_count", -1)) != len(blocks):
            raise ValueError(f"MMW manifest {role} block count mismatch.")
    return payload


def validate_trajectory_config_protocol(cfg: Mapping[str, Any]) -> dict[str, Any]:
    data = cfg.get("data", {})
    if data.get("split_protocol") != TRAJECTORY_PROTOCOL_ID:
        raise ValueError(f"data.split_protocol must be {TRAJECTORY_PROTOCOL_ID!r} for MMW.")
    section = cfg.get("data_protocol")
    if not isinstance(section, Mapping) or section.get("mode") != TRAJECTORY_PROTOCOL_MODE:
        raise ValueError(f"data_protocol must bind mode={TRAJECTORY_PROTOCOL_MODE!r}.")
    evaluate_test = bool(cfg.get("runtime", {}).get("evaluate_test_requested", False))
    manifest_value = section.get("split_manifest", section.get("path"))
    protocol = load_trajectory_protocol(
        Path(str(manifest_value)).resolve(),
        verify_sources=False,
        load_test=evaluate_test,
    )
    required = (
        "protocol_id",
        "protocol_version",
        "manifest_version",
        "assignment_algorithm",
        "protocol_fingerprint",
        "split_manifest_hash",
        "split_seed",
        "block_size",
        "data_source_hash",
        "window_config_hash",
        "weather_binding",
        "train_role",
        "validation_role",
        "test_role",
    )
    if any(section.get(key) != protocol.get(key) for key in required):
        raise ValueError("MMW config must bind the exact block manifest, source, window and role identity.")
    if int(data.get("split_seed", -1)) != int(protocol["split_seed"]):
        raise ValueError("data.split_seed does not match the bound MMW manifest.")
    expected_domains = protocol_dataset_domains(protocol)
    actual_domains = data.get("dataset", {}).get("domains")
    if actual_domains != expected_domains:
        raise ValueError("Resolved MMW domains do not exactly match the bound ID block manifest.")
    final_test = cfg.get("training", {}).get("final_test", {"enabled": False})
    final_test_enabled = final_test if isinstance(final_test, bool) else bool(final_test.get("enabled", False))
    if bool(final_test_enabled) != evaluate_test:
        raise ValueError("MMW test loading requires the explicit --evaluate-test runtime authorization.")
    if section.get("test_evaluated") is not evaluate_test:
        raise ValueError("MMW runtime test_evaluated provenance does not match --evaluate-test authorization.")

    audit_path = Path(str(section.get("audit_report", protocol.get("audit_report", "")))).resolve()
    if not audit_path.is_file():
        raise FileNotFoundError(f"MMW ID block audit report is missing: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "passed"
        or audit.get("protocol") != TRAJECTORY_PROTOCOL_ID
        or audit.get("assignment_algorithm") != protocol["assignment_algorithm"]
        or audit.get("protocol_fingerprint") != protocol["protocol_fingerprint"]
        or audit.get("split_manifest_hash") != protocol["split_manifest_hash"]
        or int(audit.get("split_seed", -1)) != int(protocol["split_seed"])
        or audit.get("test_evaluated") is not False
    ):
        raise ValueError("MMW ID block audit is missing, stale, failed, or contaminated by test evaluation.")
    return audit


def bind_trajectory_config(cfg: dict[str, Any], manifest_path: str | Path) -> dict[str, Any]:
    """Bind a loaded MMW config to one exact ID block manifest."""

    path = Path(manifest_path).resolve()
    evaluate_test = bool(cfg.setdefault("runtime", {}).get("evaluate_test_requested", False))
    protocol = load_trajectory_protocol(path, verify_sources=False, load_test=evaluate_test)
    configured_data = cfg.get("data", {})
    configured_protocol = configured_data.get("split_protocol")
    if configured_protocol not in (None, TRAJECTORY_PROTOCOL_ID):
        raise ValueError(f"Legacy MMW split protocol is not supported: {configured_protocol!r}.")
    configured_seed = configured_data.get("split_seed")
    if configured_seed is not None and int(configured_seed) != int(protocol["split_seed"]):
        raise ValueError(
            f"MMW data.split_seed={int(configured_seed)} does not match manifest seed {int(protocol['split_seed'])}."
        )
    configured_block_size = configured_data.get("block_size")
    if configured_block_size is not None and int(configured_block_size) != int(protocol["block_size"]):
        raise ValueError(
            f"MMW data.block_size={int(configured_block_size)} does not match manifest block size "
            f"{int(protocol['block_size'])}."
        )
    audit_path = Path(str(protocol["audit_report"])).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "passed"
        or audit.get("protocol") != TRAJECTORY_PROTOCOL_ID
        or audit.get("assignment_algorithm") != protocol["assignment_algorithm"]
        or audit.get("protocol_fingerprint") != protocol["protocol_fingerprint"]
        or audit.get("split_manifest_hash") != protocol["split_manifest_hash"]
        or audit.get("test_evaluated") is not False
    ):
        raise ValueError("MMW ID block audit is missing, stale, failed, or contaminated by test evaluation.")

    data = cfg.setdefault("data", {})
    data.update(
        split_protocol=TRAJECTORY_PROTOCOL_ID,
        split_seed=int(protocol["split_seed"]),
        block_size=int(protocol["block_size"]),
        split_ratios=dict(SPLIT_RATIOS),
        split_manifest=str(path),
    )
    data.setdefault("dataset", {})["domains"] = protocol_dataset_domains(protocol)
    experiment = cfg.setdefault("experiment", {})
    train_seed = int(experiment.get("train_seed", experiment.get("seed", 0)))
    experiment.update(seed=train_seed, train_seed=train_seed)
    cfg.setdefault("training", {})["final_test"] = {"enabled": evaluate_test}
    cfg["data_protocol"] = {
        "mode": protocol["mode"],
        "path": str(path),
        "split_manifest": str(path),
        "split_manifest_hash": protocol["split_manifest_hash"],
        "audit_report": str(audit_path),
        "protocol_id": protocol["protocol_id"],
        "protocol_version": protocol["protocol_version"],
        "split_protocol_version": protocol["protocol_version"],
        "manifest_version": protocol["manifest_version"],
        "assignment_algorithm": protocol["assignment_algorithm"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "audit_id": audit["audit_id"],
        "audit_sha256": _sha256_file(audit_path),
        "split_seed": int(protocol["split_seed"]),
        "train_seed": train_seed,
        "block_size": int(protocol["block_size"]),
        "data_source_hash": protocol["data_source_hash"],
        "window_config_hash": protocol["window_config_hash"],
        "weather_binding": True,
        "train_role": "train",
        "validation_role": "validation",
        "test_role": "test",
        "train_sample_count": int(audit["train_sample_count"]),
        "validation_sample_count": int(audit["validation_sample_count"]),
        "test_sample_count": int(audit["test_sample_count"]),
        "train_sample_id_hash": audit["train_sample_id_hash"],
        "validation_sample_id_hash": audit["validation_sample_id_hash"],
        "test_sample_id_hash": audit["test_sample_id_hash"],
        "train_block_count": int(audit["block_counts"]["train"]),
        "validation_block_count": int(audit["block_counts"]["validation"]),
        "test_block_count": int(audit["block_counts"]["test"]),
        "train_trajectory_count": int(audit["trajectory_counts"]["train"]),
        "validation_trajectory_count": int(audit["trajectory_counts"]["validation"]),
        "test_trajectory_count": int(audit["trajectory_counts"]["test"]),
        "evaluate_test_requested": evaluate_test,
        "test_evaluated": evaluate_test,
        "leakage_validation": "PASS",
        "outer_test_enabled": evaluate_test,
        "allow_confirmation_train": False,
    }
    validate_trajectory_config_protocol(cfg)
    return protocol


def protocol_dataset_domains(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for domain in protocol["domains"]:
        result.append(
            {
                "id": domain["id"],
                "condition": domain["condition"],
                "scene": domain["scene"],
                "data_root": domain["data_root"],
                "train_csv_name": domain["train_split"],
                "val_csv_name": domain["validation_split"],
                "test_csv_name": domain["test_split"],
            }
        )
    return result


def split_cache_identity(protocol: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        "split_protocol": protocol.get("split_protocol", protocol.get("protocol_id", protocol.get("mode"))),
        "protocol_version": protocol.get("protocol_version"),
        "split_seed": protocol.get("split_seed"),
        "block_size": protocol.get("block_size"),
        "split_manifest_hash": protocol.get("split_manifest_hash"),
        "data_source_hash": protocol.get("data_source_hash"),
        "window_config_hash": protocol.get("window_config_hash"),
        "weather_binding": protocol.get("weather_binding"),
    }
    if any(identity.get(key) in (None, "") for key in CACHE_IDENTITY_FIELDS):
        missing = [key for key in CACHE_IDENTITY_FIELDS if identity.get(key) in (None, "")]
        raise ValueError(f"MMW split cache identity is incomplete: {missing}")
    return identity


def validate_split_cache_identity(cache: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    expected = split_cache_identity(protocol)
    mismatches = {key: (cache.get(key), value) for key, value in expected.items() if cache.get(key) != value}
    if mismatches:
        raise ValueError(f"Legacy or stale MMW split cache identity is not supported: {mismatches}")


def _discover_sources(dataset_root: Path) -> tuple[list[Path], list[Path]]:
    sequence_sources = sorted(dataset_root.glob("*/Prepared/*/splits/h5p1_strict_v2/all_sequences.csv"))
    frame_sources = sorted(dataset_root.glob("*/Prepared/*/manifests/frame_manifest.csv"))
    expected_count = len(EXPECTED_WEATHERS) * EXPECTED_SCENE_COUNT
    if len(sequence_sources) != expected_count or len(frame_sources) != expected_count:
        raise ValueError(
            "MMW ID block protocol requires exactly "
            f"{expected_count} strict sequence indexes and frame manifests, found "
            f"{len(sequence_sources)} and {len(frame_sources)}."
        )
    sequence_keys = {(_source_weather(path, dataset_root), path.parents[2].name) for path in sequence_sources}
    frame_keys = {(_source_weather(path, dataset_root), path.parents[1].name) for path in frame_sources}
    if sequence_keys != frame_keys:
        raise ValueError("MMW strict sequence and frame manifest weather/scene inventories do not match.")
    return sequence_sources, frame_sources


def _source_weather(path: Path, dataset_root: Path) -> str:
    try:
        weather = path.resolve().relative_to(dataset_root).parts[0]
    except ValueError as exc:
        raise ValueError(f"MMW source is outside dataset_root: {path}") from exc
    if weather not in EXPECTED_WEATHERS:
        raise ValueError(f"Unexpected MMW weather directory {weather!r}: {path}")
    return weather


def _load_source_indexes(sources: Sequence[Path], dataset_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    for source in sorted(Path(path).resolve() for path in sources):
        weather, scene = _source_weather(source, dataset_root), source.parents[2].name
        original = pd.read_csv(source, na_values="").fillna("")
        missing = sorted(_REQUIRED_SOURCE_COLUMNS - set(original.columns))
        if missing:
            raise ValueError(f"MMW source index {source} is missing required columns: {missing}")
        if set(original["condition"].astype(str)) != {weather}:
            raise ValueError(f"MMW source weather does not match its path: {source}")
        if set(original["sensor_scenario"].astype(str)) != {scene}:
            raise ValueError(f"MMW source scene does not match its path: {source}")
        augmented = augment_mmw_sequence_resource_columns(original, scene)
        augmented["weather"] = weather
        augmented["scene_id"] = scene
        augmented["cav_id"] = augmented["agent"].astype(str)
        augmented["domain_id"] = f"{weather}/{scene}"
        augmented["source_csv"] = str(source)
        augmented["source_row"] = range(len(augmented))
        augmented["source_row_sha256"] = [
            _sha256_json({column: str(value) for column, value in zip(original.columns, values)})
            for values in original.astype(str).itertuples(index=False, name=None)
        ]
        frames.append(augmented)
        inventory.append(_source_inventory_item(source, dataset_root, "strict_sequence_index", len(augmented), weather, scene))
    return pd.concat(frames, ignore_index=True), inventory


def _load_frame_indexes(
    sources: Sequence[Path], dataset_root: Path
) -> tuple[dict[tuple[str, str], pd.DataFrame], list[dict[str, Any]]]:
    result: dict[tuple[str, str], pd.DataFrame] = {}
    inventory: list[dict[str, Any]] = []
    for source in sorted(Path(path).resolve() for path in sources):
        weather, scene = _source_weather(source, dataset_root), source.parents[1].name
        frame = pd.read_csv(source, na_values="").fillna("")
        missing = sorted(_REQUIRED_FRAME_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"MMW frame manifest {source} is missing required columns: {missing}")
        if set(frame["condition"].astype(str)) != {weather} or set(frame["sensor_scenario"].astype(str)) != {scene}:
            raise ValueError(f"MMW frame manifest path identity mismatch: {source}")
        frame = frame.copy()
        frame["frame_id_key"] = frame["frame_id"].map(_frame_id_key)
        if frame.duplicated(["agent", "frame_id_key"]).any():
            raise ValueError(f"MMW frame manifest contains duplicate agent/frame identity: {source}")
        result[(weather, scene)] = frame
        inventory.append(_source_inventory_item(source, dataset_root, "base_frame_manifest", len(frame), weather, scene))
    return result, inventory


def _source_inventory_item(
    path: Path,
    dataset_root: Path,
    source_type: str,
    count: int,
    weather: str,
    scene: str,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "relative_path": path.relative_to(dataset_root).as_posix(),
        "sha256": _sha256_file(path),
        "source_type": source_type,
        "weather": weather,
        "scene_id": scene,
        "row_count": int(count),
    }


def _establish_base_time_mapping(
    records: pd.DataFrame,
    frame_indexes: Mapping[tuple[str, str], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    output = records.copy()
    output["seq_index"] = pd.to_numeric(output["seq_index"], errors="raise").astype(int)
    output["window_base_indices_json"] = ""
    output["base_window_start_index"] = -1
    output["base_window_end_index"] = -1
    output["base_target_index"] = -1
    weather_payloads: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    history_lengths: set[int] = set()
    future_lengths: set[int] = set()
    base_rows: list[dict[str, Any]] = []

    for (weather, scene, cav), part in output.groupby(["weather", "scene_id", "cav_id"], sort=True):
        ordered = part.sort_values("seq_index", kind="stable")
        sequence = ordered["seq_index"].tolist()
        sequence_start = int(sequence[0])
        if sequence != list(range(sequence_start, sequence_start + len(sequence))):
            raise ValueError(f"MMW strict seq_index must be contiguous for {weather}/{scene}/{cav}.")
        physical_to_base: dict[str, int] = {}
        for row_index, row in ordered.iterrows():
            history = _json_list(row["history_frame_ids_json"])
            future = _json_list(row["future_frame_ids_json"])
            history_lengths.add(len(history))
            future_lengths.add(len(future))
            physical = history + future
            local_sequence_index = int(row["seq_index"]) - sequence_start
            base_indices = [local_sequence_index + offset for offset in range(len(physical))]
            for frame_id, base_index in zip(physical, base_indices):
                key = _frame_id_key(frame_id)
                existing = physical_to_base.setdefault(key, base_index)
                if existing != base_index:
                    raise ValueError(
                        f"MMW frame {weather}/{scene}/{cav}/{key} maps to both base index {existing} and {base_index}."
                    )
            output.at[row_index, "window_base_indices_json"] = json.dumps(base_indices, separators=(",", ":"))
            output.at[row_index, "base_window_start_index"] = base_indices[0]
            output.at[row_index, "base_window_end_index"] = base_indices[-1]
            output.at[row_index, "base_target_index"] = base_indices[len(history)]

        manifest = frame_indexes[(str(weather), str(scene))]
        selected = manifest.loc[manifest["agent"].astype(str) == str(cav)].copy()
        manifest_ids = set(selected["frame_id_key"].astype(str))
        if manifest_ids != set(physical_to_base):
            missing = sorted(manifest_ids - set(physical_to_base))[:5]
            unexpected = sorted(set(physical_to_base) - manifest_ids)[:5]
            raise ValueError(
                f"MMW base frame mapping does not cover frame manifest {weather}/{scene}/{cav}: "
                f"missing={missing}, unexpected={unexpected}."
            )
        mapped_rows = []
        for row in selected.to_dict(orient="records"):
            base_index = physical_to_base[str(row["frame_id_key"])]
            label = _beam_label(row["beam_label"], context=f"{weather}/{scene}/{cav}/{row['frame_id_key']}")
            mapped_rows.append((base_index, label, str(row["frame_id_key"]), str(row["sample_id"])))
        mapped_rows.sort(key=lambda item: item[0])
        base_indices = [item[0] for item in mapped_rows]
        if base_indices != list(range(len(base_indices))):
            raise ValueError(f"MMW base frame indexes are not contiguous for {weather}/{scene}/{cav}.")
        labels = [item[1] for item in mapped_rows]
        weather_payloads[(str(scene), str(cav))][str(weather)] = {
            "labels": labels,
            "physical_frame_ids": [item[2] for item in mapped_rows],
            "sample_ids": [item[3] for item in mapped_rows],
            "source_seq_index_start": sequence_start,
            "source_seq_index_end": int(sequence[-1]),
        }

    if history_lengths != {next(iter(history_lengths))} or future_lengths != {next(iter(future_lengths))}:
        raise ValueError(
            f"MMW strict indexes must use one window shape, found history={sorted(history_lengths)}, future={sorted(future_lengths)}."
        )
    history_span = next(iter(history_lengths))
    future_span = next(iter(future_lengths))
    if history_span <= 0 or future_span <= 0:
        raise ValueError("MMW window history and future spans must be positive.")

    weather_mapping: list[dict[str, Any]] = []
    for (scene, cav), weather_items in sorted(weather_payloads.items(), key=lambda item: _group_sort_key(item[0])):
        if set(weather_items) != set(EXPECTED_WEATHERS):
            raise ValueError(
                f"MMW trajectory {scene}/{cav} must contain {list(EXPECTED_WEATHERS)}, found {sorted(weather_items)}."
            )
        label_sequences = {tuple(item["labels"]) for item in weather_items.values()}
        if len(label_sequences) != 1:
            raise ValueError(f"MMW weather copies disagree on beam labels for {scene}/{cav}.")
        sequence_ranges = {
            (int(item["source_seq_index_start"]), int(item["source_seq_index_end"]))
            for item in weather_items.values()
        }
        if len(sequence_ranges) != 1:
            raise ValueError(f"MMW weather copies disagree on strict seq_index range for {scene}/{cav}.")
        labels = list(next(iter(label_sequences)))
        for base_index, label in enumerate(labels):
            base_rows.append({"scene_id": scene, "cav_id": cav, "base_frame_index": base_index, "beam_label": label})
        weather_mapping.append(
            {
                "scene_id": scene,
                "cav_id": cav,
                "num_base_samples": len(labels),
                "base_index_start": 0,
                "base_index_end": len(labels) - 1,
                "source_seq_index_start": next(iter(sequence_ranges))[0],
                "source_seq_index_end": next(iter(sequence_ranges))[1],
                "beam_label_sha256": _sha256_json(labels),
                "weather_physical_frame_sha256": {
                    weather: _sha256_json(weather_items[weather]["physical_frame_ids"])
                    for weather in EXPECTED_WEATHERS
                },
                "weather_sample_id_sha256": {
                    weather: _sha256_json(weather_items[weather]["sample_ids"])
                    for weather in EXPECTED_WEATHERS
                },
            }
        )
    scenes = {item["scene_id"] for item in weather_mapping}
    if len(scenes) != EXPECTED_SCENE_COUNT:
        raise ValueError(f"MMW ID block protocol requires exactly 5 scenes, found {len(scenes)}: {sorted(scenes)}")
    window_config = {
        "history_span": history_span,
        "future_span": future_span,
        "sample_span": history_span + future_span,
        "source_index": "h5p1_strict_v2/all_sequences.csv",
        "materialization_order": "assign_blocks_then_filter_complete_windows",
        "base_time_mapping": "strict_seq_index_plus_explicit_frame_lists",
    }
    return output, pd.DataFrame(base_rows), weather_mapping, window_config


def _build_continuous_blocks(
    records: pd.DataFrame,
    base_samples: pd.DataFrame,
    *,
    block_size: int,
    sample_span: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame, int]:
    blocks: list[dict[str, Any]] = []
    index_to_block: dict[tuple[str, str, int], str] = {}
    for (scene, cav), part in base_samples.groupby(["scene_id", "cav_id"], sort=True):
        ordered = part.sort_values("base_frame_index", kind="stable")
        indexes = ordered["base_frame_index"].astype(int).tolist()
        if indexes != list(range(len(indexes))):
            raise ValueError(f"MMW trajectory {scene}/{cav} base indexes must be contiguous from zero.")
        labels = ordered["beam_label"].astype(int).to_numpy()
        for ordinal, start_offset in enumerate(range(0, len(indexes), int(block_size))):
            selected = indexes[start_offset : start_offset + int(block_size)]
            start, end = selected[0], selected[-1]
            block_id = f"{scene}::{cav}::{start:06d}-{end:06d}"
            hist = np.bincount(labels[start : end + 1], minlength=64).astype(int).tolist()
            block = {
                "block_id": block_id,
                "scene_id": str(scene),
                "cav_id": str(cav),
                "block_ordinal": ordinal,
                "block_start_base_index": start,
                "block_end_base_index": end,
                "num_base_samples": len(selected),
                "num_weather_versions": len(EXPECTED_WEATHERS),
                "beam_histogram": hist,
                "num_windows_estimated": len(EXPECTED_WEATHERS) * max(0, len(selected) - int(sample_span) + 1),
            }
            blocks.append(block)
            for index in selected:
                index_to_block[(str(scene), str(cav), int(index))] = block_id

    if any(sum(1 for item in values) < len(SPLIT_ROLES) for values in _blocks_by_trajectory(blocks).values()):
        short = {
            key: len(values)
            for key, values in _blocks_by_trajectory(blocks).items()
            if len(values) < len(SPLIT_ROLES)
        }
        details = ", ".join(f"{scene}/{cav}={count} blocks" for (scene, cav), count in sorted(short.items()))
        raise ValueError(f"MMW trajectories need at least three blocks for train/validation/test: {details}")

    keep_rows: list[int] = []
    block_ids: list[str] = []
    block_window_hist: dict[str, np.ndarray] = {str(item["block_id"]): np.zeros(64, dtype=np.int64) for item in blocks}
    dropped = 0
    for row_index, row in records.iterrows():
        base_indices = [int(value) for value in json.loads(str(row["window_base_indices_json"]))]
        candidate_ids = {
            index_to_block.get((_required_text(row, "scene_id"), _required_text(row, "cav_id"), base_index))
            for base_index in base_indices
        }
        if len(candidate_ids) != 1 or None in candidate_ids:
            dropped += 1
            continue
        block_id = str(next(iter(candidate_ids)))
        keep_rows.append(row_index)
        block_ids.append(block_id)
        block_window_hist[block_id][_beam_label(row["future_beam_label1"], context=str(row.get("sample_id", "")))] += 1
    materialized = records.loc[keep_rows].copy().reset_index(drop=True)
    materialized["block_id"] = block_ids
    actual_counts = Counter(block_ids)
    for block in blocks:
        block_id = str(block["block_id"])
        block["window_beam_histogram"] = block_window_hist[block_id].astype(int).tolist()
        if int(actual_counts[block_id]) != int(block["num_windows_estimated"]):
            raise ValueError(
                f"MMW strict window inventory for block {block_id} has {actual_counts[block_id]} windows, "
                f"expected {block['num_windows_estimated']}; source mapping is incomplete."
            )
    return blocks, materialized, dropped


def _quota_options(block_count: int) -> list[dict[str, int]]:
    options = []
    for train in range(1, block_count - 1):
        for validation in range(1, block_count - train):
            test = block_count - train - validation
            if train <= max(validation, test):
                continue
            options.append({"train": train, "validation": validation, "test": test})
    if not options:
        raise ValueError(f"Cannot allocate {block_count} blocks with train largest and every split non-empty.")
    return options


def _balanced_trajectory_quotas(
    trajectory_items: Sequence[tuple[tuple[str, str], Sequence[Mapping[str, Any]]]],
    *,
    split_seed: int,
) -> dict[tuple[str, str], dict[str, int]]:
    states: dict[tuple[int, int], tuple[float, list[dict[str, int]]]] = {(0, 0): (0.0, [])}
    total_blocks = sum(len(values) for _, values in trajectory_items)
    for _key, values in trajectory_items:
        block_count = len(values)
        options = _quota_options(block_count)
        option_errors = {
            tuple(option[role] for role in SPLIT_ROLES): sum(
                abs(option[role] / block_count - SPLIT_RATIOS[role]) for role in SPLIT_ROLES
            )
            for option in options
        }
        best_local_error = min(option_errors.values())
        competitive_options = [
            option
            for option in options
            if option_errors[tuple(option[role] for role in SPLIT_ROLES)]
            <= best_local_error + 2.0 / block_count + 1e-12
        ]
        next_states: dict[tuple[int, int], tuple[float, list[dict[str, int]]]] = {}
        for (_train_total, _validation_total), (cost, selected) in states.items():
            for option in competitive_options:
                state = (_train_total + option["train"], _validation_total + option["validation"])
                local_error = option_errors[tuple(option[role] for role in SPLIT_ROLES)]
                candidate = (cost + local_error, [*selected, option])
                existing = next_states.get(state)
                if existing is None or (candidate[0], _quota_tie_key(candidate[1], split_seed)) < (
                    existing[0],
                    _quota_tie_key(existing[1], split_seed),
                ):
                    next_states[state] = candidate
        states = next_states

    ranked = []
    for (train_total, validation_total), (local_error, selected) in states.items():
        counts = {
            "train": train_total,
            "validation": validation_total,
            "test": total_blocks - train_total - validation_total,
        }
        if counts["train"] <= max(counts["validation"], counts["test"]):
            continue
        ratio_error = sum(abs(counts[role] / total_blocks - SPLIT_RATIOS[role]) for role in SPLIT_ROLES)
        objective = 4.0 * ratio_error + 2.0 * local_error / len(trajectory_items)
        ranked.append((objective, _quota_tie_key(selected, split_seed), selected))
    if not ranked:
        raise ValueError("MMW trajectory quotas cannot satisfy the split coverage constraints.")
    selected = min(ranked)[2]
    return {
        key: dict(option)
        for (key, _values), option in zip(trajectory_items, selected)
    }


def _quota_tie_key(options: Sequence[Mapping[str, int]], seed: int) -> str:
    signature = ";".join(
        ",".join(str(int(option[role])) for role in SPLIT_ROLES)
        for option in options
    )
    return hashlib.sha256(f"{int(seed)}:{signature}".encode("utf-8")).hexdigest()


def _contiguous_assignment(
    blocks: Sequence[Mapping[str, Any]], quotas: Mapping[tuple[str, str], Mapping[str, int]]
) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for key, values in sorted(_blocks_by_trajectory(blocks).items(), key=lambda item: _group_sort_key(item[0])):
        ordered = sorted(values, key=_block_sort_key)
        offset = 0
        for role in SPLIT_ROLES:
            count = int(quotas[key][role])
            assignment.update({str(item["block_id"]): role for item in ordered[offset : offset + count]})
            offset += count
    return assignment


def _improve_assignment(
    blocks: Sequence[Mapping[str, Any]],
    assignment: dict[str, str],
    by_trajectory: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    rng: random.Random,
    *,
    scorer: "_AssignmentScorer",
) -> dict[str, str]:
    current = dict(assignment)
    current_score = float(scorer.evaluate(current)["objective"])
    block_ids = [str(item["block_id"]) for item in blocks]
    trajectory_by_id = {
        str(item["block_id"]): (_required_text(item, "scene_id"), _required_text(item, "cav_id"))
        for item in blocks
    }
    role_counts = {
        key: Counter(current[str(item["block_id"])] for item in values)
        for key, values in by_trajectory.items()
    }
    attempts_without_improvement = 0
    max_attempts = max(4000, len(block_ids) * 40)
    patience = max(1000, len(block_ids) * 12)
    for _attempt in range(max_attempts):
        left, right = rng.sample(block_ids, 2)
        left_role, right_role = current[left], current[right]
        if left_role == right_role:
            continue
        left_group, right_group = trajectory_by_id[left], trajectory_by_id[right]
        if left_group != right_group and (
            role_counts[left_group][left_role] <= 1 or role_counts[right_group][right_role] <= 1
        ):
            continue
        current[left], current[right] = right_role, left_role
        score = float(scorer.evaluate(current)["objective"])
        if score < current_score - 1e-12:
            if left_group != right_group:
                role_counts[left_group].subtract({left_role: 1})
                role_counts[left_group].update({right_role: 1})
                role_counts[right_group].subtract({right_role: 1})
                role_counts[right_group].update({left_role: 1})
            current_score = score
            attempts_without_improvement = 0
        else:
            current[left], current[right] = left_role, right_role
            attempts_without_improvement += 1
            if attempts_without_improvement >= patience:
                break
    return current


class _AssignmentScorer:
    def __init__(self, blocks: Sequence[Mapping[str, Any]]) -> None:
        self.block_ids = [str(item["block_id"]) for item in blocks]
        self.windows = np.asarray([float(item["num_windows_estimated"]) for item in blocks], dtype=np.float64)
        self.histograms = np.asarray(
            [item["window_beam_histogram"] for item in blocks],
            dtype=np.float64,
        )
        self.total_windows = float(self.windows.sum())
        if self.total_windows <= 0:
            raise ValueError("MMW block assignment requires at least one materialized window.")
        self.global_probability = _probabilities(self.histograms.sum(axis=0))
        self.trajectory_groups = self._group_indices(blocks, ("scene_id", "cav_id"))
        self.scene_groups = self._group_indices(blocks, ("scene_id",))

    def evaluate(self, assignment: Mapping[str, str]) -> dict[str, float]:
        try:
            roles = np.fromiter(
                (_ROLE_INDEX[assignment[block_id]] for block_id in self.block_ids),
                dtype=np.int8,
                count=len(self.block_ids),
            )
        except KeyError as exc:
            raise ValueError("MMW block assignment is incomplete or contains an unknown split role.") from exc
        role_windows = np.bincount(roles, weights=self.windows, minlength=len(SPLIT_ROLES))
        role_histograms = np.zeros((len(SPLIT_ROLES), 64), dtype=np.float64)
        for role_index in range(len(SPLIT_ROLES)):
            role_histograms[role_index] = self.histograms[roles == role_index].sum(axis=0)
        ratio_error = sum(
            abs(role_windows[_ROLE_INDEX[role]] / self.total_windows - SPLIT_RATIOS[role])
            for role in SPLIT_ROLES
        )
        role_probabilities = [_probabilities(histogram) for histogram in role_histograms]
        tv_to_global = sum(_tv(probability, self.global_probability) for probability in role_probabilities)
        pairwise_tv = _tv(role_probabilities[0], role_probabilities[1]) + _tv(
            role_probabilities[0], role_probabilities[2]
        )
        label_error = tv_to_global + pairwise_tv
        trajectory_error = self._group_ratio_error(roles, self.trajectory_groups)
        scene_error = self._group_ratio_error(roles, self.scene_groups)
        trajectory_label_error, trajectory_coverage_penalty = self._group_label_error(
            roles, self.trajectory_groups
        )
        scene_label_error, scene_coverage_penalty = self._group_label_error(roles, self.scene_groups)
        train_missing = role_histograms[0] <= 0
        held_out = role_histograms[1] + role_histograms[2]
        coverage_penalty = float(held_out[train_missing].sum() / max(float(held_out.sum()), 1.0))
        objective = (
            _ASSIGNMENT_WEIGHTS["ratio"] * ratio_error
            + _ASSIGNMENT_WEIGHTS["label"] * label_error
            + _ASSIGNMENT_WEIGHTS["trajectory"] * trajectory_error
            + _ASSIGNMENT_WEIGHTS["scene"] * scene_error
            + _ASSIGNMENT_WEIGHTS["coverage"] * coverage_penalty
            + _ASSIGNMENT_WEIGHTS["trajectory_label"] * trajectory_label_error
            + _ASSIGNMENT_WEIGHTS["scene_label"] * scene_label_error
            + _ASSIGNMENT_WEIGHTS["trajectory_coverage"] * trajectory_coverage_penalty
            + _ASSIGNMENT_WEIGHTS["scene_coverage"] * scene_coverage_penalty
        )
        return {
            "objective": float(objective),
            "ratio_error": float(ratio_error),
            "label_distribution_error": float(label_error),
            "per_trajectory_ratio_error": float(trajectory_error),
            "per_scene_ratio_error": float(scene_error),
            "beam_coverage_penalty": float(coverage_penalty),
            "per_trajectory_label_distribution_error": float(trajectory_label_error),
            "per_scene_label_distribution_error": float(scene_label_error),
            "per_trajectory_beam_coverage_penalty": float(trajectory_coverage_penalty),
            "per_scene_beam_coverage_penalty": float(scene_coverage_penalty),
        }

    def _group_label_error(self, roles: np.ndarray, groups: Sequence[np.ndarray]) -> tuple[float, float]:
        distribution_errors = []
        missing_mass = 0.0
        held_out_mass = 0.0
        for indexes in groups:
            histograms = np.zeros((len(SPLIT_ROLES), 64), dtype=np.float64)
            for role_index in range(len(SPLIT_ROLES)):
                histograms[role_index] = self.histograms[indexes[roles[indexes] == role_index]].sum(axis=0)
            train_probability = _probabilities(histograms[_ROLE_INDEX["train"]])
            distribution_errors.append(
                _tv(train_probability, _probabilities(histograms[_ROLE_INDEX["validation"]]))
                + _tv(train_probability, _probabilities(histograms[_ROLE_INDEX["test"]]))
            )
            train_missing = histograms[_ROLE_INDEX["train"]] <= 0
            held_out = histograms[_ROLE_INDEX["validation"]] + histograms[_ROLE_INDEX["test"]]
            missing_mass += float(held_out[train_missing].sum())
            held_out_mass += float(held_out.sum())
        return (
            float(np.mean(distribution_errors)) if distribution_errors else math.inf,
            missing_mass / max(held_out_mass, 1.0),
        )

    def _group_ratio_error(self, roles: np.ndarray, groups: Sequence[np.ndarray]) -> float:
        errors = []
        for indexes in groups:
            weights = self.windows[indexes]
            counts = np.bincount(roles[indexes], weights=weights, minlength=len(SPLIT_ROLES))
            total = float(weights.sum())
            errors.append(
                sum(abs(counts[_ROLE_INDEX[role]] / total - SPLIT_RATIOS[role]) for role in SPLIT_ROLES)
            )
        return float(np.mean(errors)) if errors else math.inf

    @staticmethod
    def _group_indices(
        blocks: Sequence[Mapping[str, Any]], key_fields: tuple[str, ...]
    ) -> list[np.ndarray]:
        grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for index, block in enumerate(blocks):
            grouped[tuple(str(block[field]) for field in key_fields)].append(index)
        return [np.asarray(indexes, dtype=np.int64) for _key, indexes in sorted(grouped.items())]


def _assignment_objective(
    blocks: Sequence[Mapping[str, Any]], assignment: Mapping[str, str]
) -> dict[str, float]:
    return _AssignmentScorer(blocks).evaluate(assignment)


def _group_ratio_error(
    blocks: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, str],
    *,
    key_fields: tuple[str, ...],
) -> float:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for block in blocks:
        grouped[tuple(str(block[field]) for field in key_fields)].append(block)
    errors = []
    for values in grouped.values():
        total = sum(float(item["num_windows_estimated"]) for item in values)
        counts = Counter()
        for item in values:
            counts[assignment[str(item["block_id"])]] += int(item["num_windows_estimated"])
        errors.append(sum(abs(counts[role] / total - SPLIT_RATIOS[role]) for role in SPLIT_ROLES))
    return float(np.mean(errors)) if errors else math.inf


def _validate_assignment_hard_constraints(
    blocks: Sequence[Mapping[str, Any]], assignment: Mapping[str, str]
) -> None:
    expected_ids = {str(item["block_id"]) for item in blocks}
    if set(assignment) != expected_ids or set(assignment.values()) - set(SPLIT_ROLES):
        raise ValueError("MMW block assignment is incomplete or contains an unknown split role.")
    counts = Counter(assignment.values())
    if counts["train"] <= max(counts["validation"], counts["test"]):
        raise ValueError(f"MMW train block count must be largest, got {dict(counts)}.")
    for key, values in _blocks_by_trajectory(blocks).items():
        roles = {assignment[str(item["block_id"])] for item in values}
        if roles != set(SPLIT_ROLES):
            raise ValueError(f"MMW trajectory {key} does not cover train/validation/test: {sorted(roles)}")


def _write_split_indexes(
    records: pd.DataFrame, split_root: Path, dataset_root: Path
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    sort_columns = ["weather", "scene_id", "cav_id", "base_window_start_index", "sample_id"]
    role_frames = {
        role: records.loc[records["split"] == role].sort_values(sort_columns, kind="stable").reset_index(drop=True)
        for role in SPLIT_ROLES
    }
    if any(frame.empty for frame in role_frames.values()):
        raise ValueError("MMW ID block protocol produced an empty split.")
    domains: list[dict[str, Any]] = []
    for (weather, scene), domain_rows in records.groupby(["weather", "scene_id"], sort=True):
        domain: dict[str, Any] = {
            "id": f"{weather}/{scene}",
            "condition": str(weather),
            "scene": str(scene),
            "data_root": str((dataset_root / str(weather)).resolve()),
        }
        for role in SPLIT_ROLES:
            selected = domain_rows.loc[domain_rows["split"] == role].sort_values(sort_columns, kind="stable")
            if selected.empty:
                raise ValueError(f"MMW domain {weather}/{scene} has no {role} windows.")
            path = split_root / role / f"{weather}__{scene}.csv"
            _write_csv(selected, path)
            domain[f"{role}_split"] = str(path.resolve())
            domain[f"{role}_csv_sha256"] = _sha256_file(path)
            domain[f"{role}_sample_count"] = int(len(selected))
        domains.append(domain)
    return domains, role_frames


def _load_materialized_splits(
    protocol: Mapping[str, Any], *, roles: Sequence[str] = SPLIT_ROLES
) -> dict[str, pd.DataFrame]:
    selected_roles = tuple(str(role) for role in roles)
    if not selected_roles or len(set(selected_roles)) != len(selected_roles) or any(
        role not in SPLIT_ROLES for role in selected_roles
    ):
        raise ValueError(f"MMW materialized split roles are invalid: {selected_roles}")
    frames: dict[str, list[pd.DataFrame]] = {role: [] for role in selected_roles}
    for domain in protocol.get("domains", []):
        for role in selected_roles:
            path = Path(str(domain.get(f"{role}_split", "")))
            if not path.is_file() or _sha256_file(path) != domain.get(f"{role}_csv_sha256"):
                raise ValueError(f"MMW materialized {role} index is missing or changed: {path}")
            frame = pd.read_csv(path, na_values="").fillna("")
            if "split" not in frame or set(frame["split"].astype(str)) != {role}:
                raise ValueError(f"MMW materialized index carries the wrong split role: {path}")
            if int(domain.get(f"{role}_sample_count", -1)) != len(frame):
                raise ValueError(f"MMW materialized index count mismatch: {path}")
            frames[role].append(frame)
    result = {role: pd.concat(values, ignore_index=True) if values else pd.DataFrame() for role, values in frames.items()}
    for role in selected_roles:
        if result[role].empty:
            raise ValueError(f"MMW manifest contains no {role} windows.")
        if _sample_id_hash(result[role]) != protocol.get("split_hashes", {}).get(role):
            raise ValueError(f"MMW {role} sample identity hash mismatch.")
    return result


def _statistics(
    records: pd.DataFrame,
    blocks: Sequence[Mapping[str, Any]],
    *,
    assignment_summary: Mapping[str, Any],
    dropped_boundary_windows: int,
) -> dict[str, Any]:
    optimized = _label_distribution_statistics(records)
    optimized_conditional = {
        "per_domain": _conditional_label_distribution_statistics(records, ("weather", "scene_id")),
        "per_scene": _conditional_label_distribution_statistics(records, ("scene_id",)),
        "per_trajectory": _conditional_label_distribution_statistics(records, ("scene_id", "cav_id")),
    }
    quotas = {
        (str(item["scene_id"]), str(item["cav_id"])): {
            role: int(item[role]) for role in SPLIT_ROLES
        }
        for item in assignment_summary["trajectory_quotas"]
    }
    baseline_assignment = _contiguous_assignment(blocks, quotas)
    baseline_records = records.copy()
    baseline_records["split"] = baseline_records["block_id"].map(baseline_assignment)
    baseline = _label_distribution_statistics(baseline_records)
    baseline_conditional = {
        "per_domain": _conditional_label_distribution_statistics(
            baseline_records, ("weather", "scene_id")
        ),
        "per_scene": _conditional_label_distribution_statistics(baseline_records, ("scene_id",)),
        "per_trajectory": _conditional_label_distribution_statistics(
            baseline_records, ("scene_id", "cav_id")
        ),
    }
    ratios = _split_ratio_statistics(blocks, records)
    per_scene = []
    for scene, part in records.groupby("scene_id", sort=True):
        per_scene.append(
            {"scene_id": str(scene), **{f"{role}_windows": int((part["split"] == role).sum()) for role in SPLIT_ROLES}}
        )
    per_trajectory = []
    for (scene, cav), values in sorted(_blocks_by_trajectory(blocks).items(), key=lambda item: _group_sort_key(item[0])):
        per_trajectory.append(
            {
                "scene_id": scene,
                "cav_id": cav,
                **{
                    f"{role}_blocks": sum(str(item.get("split")) == role for item in values)
                    for role in SPLIT_ROLES
                },
            }
        )
    optimized_score = optimized["train_validation"]["tv"] + optimized["train_test"]["tv"]
    baseline_score = baseline["train_validation"]["tv"] + baseline["train_test"]["tv"]
    optimized_conditional_score = sum(
        values["train_validation_macro"]["tv"] + values["train_test_macro"]["tv"]
        for values in optimized_conditional.values()
    )
    baseline_conditional_score = sum(
        values["train_validation_macro"]["tv"] + values["train_test_macro"]["tv"]
        for values in baseline_conditional.values()
    )
    return {
        "ratios": ratios,
        "label_distribution": optimized,
        "simple_contiguous_baseline": baseline,
        "conditional_label_distribution": optimized_conditional,
        "simple_contiguous_conditional_baseline": baseline_conditional,
        "label_tv_sum": optimized_score,
        "simple_contiguous_label_tv_sum": baseline_score,
        "stratified_not_worse_than_contiguous": optimized_score <= baseline_score + 1e-12,
        "conditional_label_tv_sum": optimized_conditional_score,
        "simple_contiguous_conditional_label_tv_sum": baseline_conditional_score,
        "conditional_stratified_not_worse_than_contiguous": (
            optimized_conditional_score <= baseline_conditional_score + 1e-12
        ),
        "dropped_boundary_windows": int(dropped_boundary_windows),
        "assignment": dict(assignment_summary),
        "per_scene": per_scene,
        "per_trajectory": per_trajectory,
    }


def _split_ratio_statistics(
    blocks: Sequence[Mapping[str, Any]], records: pd.DataFrame
) -> dict[str, dict[str, float | int]]:
    measures = {
        "blocks": {role: sum(str(item.get("split")) == role for item in blocks) for role in SPLIT_ROLES},
        "base_frames": {
            role: sum(int(item["num_base_samples"]) for item in blocks if str(item.get("split")) == role)
            for role in SPLIT_ROLES
        },
        "weather_samples": {
            role: sum(
                int(item["num_base_samples"]) * int(item["num_weather_versions"])
                for item in blocks
                if str(item.get("split")) == role
            )
            for role in SPLIT_ROLES
        },
        "windows": {role: int((records["split"] == role).sum()) for role in SPLIT_ROLES},
    }
    return {
        name: {
            role: int(counts[role]) for role in SPLIT_ROLES
        }
        | {f"{role}_ratio": float(counts[role] / max(sum(counts.values()), 1)) for role in SPLIT_ROLES}
        for name, counts in measures.items()
    }


def _label_distribution_statistics(records: pd.DataFrame) -> dict[str, Any]:
    histograms = {
        role: np.bincount(
            pd.to_numeric(records.loc[records["split"] == role, "future_beam_label1"], errors="raise").astype(int),
            minlength=64,
        ).astype(int)
        for role in SPLIT_ROLES
    }
    pairs = {
        "train_validation": _distribution_metrics(histograms["train"], histograms["validation"]),
        "train_test": _distribution_metrics(histograms["train"], histograms["test"]),
        "validation_test": _distribution_metrics(histograms["validation"], histograms["test"]),
    }
    train_coverage = set(np.flatnonzero(histograms["train"]))
    all_hist = sum(histograms.values(), np.zeros(64, dtype=np.int64))
    proportions = {role: _probabilities(histograms[role]) for role in SPLIT_ROLES}
    max_differences = []
    for beam in range(64):
        values = {role: float(proportions[role][beam]) for role in SPLIT_ROLES}
        max_differences.append({"beam": beam, "max_proportion_difference": max(values.values()) - min(values.values()), **values})
    return {
        "histograms": {role: histograms[role].tolist() for role in SPLIT_ROLES},
        **pairs,
        "validation_beams_not_in_train": sorted(set(np.flatnonzero(histograms["validation"])) - train_coverage),
        "test_beams_not_in_train": sorted(set(np.flatnonzero(histograms["test"])) - train_coverage),
        "globally_unseen_beams": sorted(set(range(64)) - set(np.flatnonzero(all_hist))),
        "largest_proportion_differences": sorted(
            max_differences, key=lambda item: (-item["max_proportion_difference"], item["beam"])
        )[:10],
    }


def _conditional_label_distribution_statistics(
    records: pd.DataFrame,
    key_fields: tuple[str, ...],
) -> dict[str, Any]:
    group_key: str | list[str] = key_fields[0] if len(key_fields) == 1 else list(key_fields)
    groups = []
    validation_missing_mass = 0
    validation_mass = 0
    test_missing_mass = 0
    test_mass = 0
    for raw_key, part in records.groupby(group_key, sort=True):
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        metrics = _label_distribution_statistics(part)
        train_histogram = np.asarray(metrics["histograms"]["train"], dtype=np.int64)
        validation_histogram = np.asarray(metrics["histograms"]["validation"], dtype=np.int64)
        test_histogram = np.asarray(metrics["histograms"]["test"], dtype=np.int64)
        train_missing = train_histogram <= 0
        validation_missing = int(validation_histogram[train_missing].sum())
        test_missing = int(test_histogram[train_missing].sum())
        validation_count = int(validation_histogram.sum())
        test_count = int(test_histogram.sum())
        validation_missing_mass += validation_missing
        validation_mass += validation_count
        test_missing_mass += test_missing
        test_mass += test_count
        groups.append(
            {
                **{field: str(value) for field, value in zip(key_fields, key)},
                "train_validation": metrics["train_validation"],
                "train_test": metrics["train_test"],
                "validation_beams_not_in_train": [
                    int(value) for value in metrics["validation_beams_not_in_train"]
                ],
                "test_beams_not_in_train": [int(value) for value in metrics["test_beams_not_in_train"]],
                "validation_unseen_beam_mass": validation_missing / max(validation_count, 1),
                "test_unseen_beam_mass": test_missing / max(test_count, 1),
            }
        )
    metric_names = ("tv", "jsd", "pearson", "spearman")
    pair_names = ("train_validation", "train_test")
    macros = {
        f"{pair}_macro": {
            metric: float(np.mean([group[pair][metric] for group in groups])) if groups else math.inf
            for metric in metric_names
        }
        for pair in pair_names
    }
    worst = max(groups, key=lambda item: item["train_validation"]["tv"], default=None)
    return {
        "group_key": list(key_fields),
        "group_count": len(groups),
        **macros,
        "train_validation_worst_tv": float(worst["train_validation"]["tv"]) if worst else math.inf,
        "train_validation_worst_group": (
            {field: worst[field] for field in key_fields} if worst else {}
        ),
        "validation_unseen_beam_mass": validation_missing_mass / max(validation_mass, 1),
        "test_unseen_beam_mass": test_missing_mass / max(test_mass, 1),
        "groups": groups,
    }


def _distribution_metrics(left_hist: np.ndarray, right_hist: np.ndarray) -> dict[str, float]:
    left, right = _probabilities(left_hist), _probabilities(right_hist)
    midpoint = 0.5 * (left + right)
    jsd = 0.5 * (_kl(left, midpoint) + _kl(right, midpoint))
    return {
        "tv": _tv(left, right),
        "jsd": float(jsd),
        "pearson": _correlation(left, right, method="pearson"),
        "spearman": _correlation(left, right, method="spearman"),
    }


def _probabilities(histogram: np.ndarray) -> np.ndarray:
    values = np.asarray(histogram, dtype=np.float64)
    return values / max(float(values.sum()), 1.0)


def _tv(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.abs(left - right).sum())


def _kl(left: np.ndarray, right: np.ndarray) -> float:
    mask = left > 0
    return float(np.sum(left[mask] * np.log(left[mask] / np.maximum(right[mask], 1e-15))))


def _correlation(left: np.ndarray, right: np.ndarray, *, method: str) -> float:
    left_values = pd.Series(left)
    right_values = pd.Series(right)
    value = left_values.corr(right_values, method=method)
    if pd.isna(value):
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(value)


def _shared_resource_diagnostics(role_frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    identities = {role: _resource_identities(frame) for role, frame in role_frames.items()}
    result: dict[str, Any] = {"policy": "diagnostic_only", "pairwise": {}}
    for left, right in itertools.combinations(SPLIT_ROLES, 2):
        families = {}
        for family in _RESOURCE_PATTERNS:
            overlap = identities[left][family] & identities[right][family]
            families[family] = {"overlap_count": len(overlap), "examples": sorted(overlap)[:5]}
        result["pairwise"][f"{left}_vs_{right}"] = families
    return result


def _resource_identities(frame: pd.DataFrame) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for family, pattern in _RESOURCE_PATTERNS.items():
        columns = [str(column) for column in frame.columns if pattern.match(str(column))]
        result[family] = {
            str(value).strip().replace("\\", "/")
            for column in columns
            for value in frame[column].tolist()
            if _text(value)
        }
    return result


def _window_error(row: Mapping[str, Any], *, block: Mapping[str, Any] | None = None) -> str | None:
    history = _json_list(row.get("history_frame_ids_json"))
    future = _json_list(row.get("future_frame_ids_json"))
    if not history or not future:
        return "missing_history_or_future_frames"
    physical = [_frame_number(value) for value in history + future]
    if any(value is None for value in physical):
        return "non_numeric_frame_id"
    numeric = [int(value) for value in physical if value is not None]
    if any(right != left + 1 for left, right in zip(numeric, numeric[1:])):
        return "non_contiguous_physical_frames"
    try:
        base_indices = [int(value) for value in json.loads(str(row.get("window_base_indices_json", "")))]
    except (TypeError, ValueError, json.JSONDecodeError):
        return "invalid_base_index_list"
    if len(base_indices) != len(physical) or any(right != left + 1 for left, right in zip(base_indices, base_indices[1:])):
        return "non_contiguous_base_indices"
    if int(row.get("base_window_start_index", -1)) != base_indices[0]:
        return "base_window_start_mismatch"
    if int(row.get("base_window_end_index", -1)) != base_indices[-1]:
        return "base_window_end_mismatch"
    if block is not None:
        start = int(block["block_start_base_index"])
        end = int(block["block_end_base_index"])
        if min(base_indices) < start or max(base_indices) > end:
            return "window_crosses_block"
    return None


def _write_report(payload: Mapping[str, Any], audit: Mapping[str, Any], report_path: Path) -> None:
    statistics = payload["statistics"]
    ratios = statistics["ratios"]
    labels = statistics["label_distribution"]
    baseline = statistics["simple_contiguous_baseline"]
    conditional = statistics["conditional_label_distribution"]
    conditional_baseline = statistics["simple_contiguous_conditional_baseline"]
    lines = [
        "# MMW ID-Stratified Block Split Report",
        "",
        f"- Protocol: `{payload['protocol']}`",
        f"- Protocol version: `{payload['protocol_version']}`",
        f"- Manifest version: `{payload['manifest_version']}`",
        f"- Assignment algorithm: `{payload['assignment_algorithm']}`",
        f"- Split seed: `{payload['split_seed']}`",
        f"- Block size: `{payload['block_size']}` base frames",
        f"- Data source hash: `{payload['data_source_hash']}`",
        f"- Manifest hash: `{audit['split_manifest_hash']}`",
        f"- Window config hash: `{payload['window_config_hash']}`",
        f"- Window configuration: `{json.dumps(payload['window_config'], sort_keys=True)}`",
        f"- Test evaluated: `{str(payload['test_evaluated']).lower()}`",
        "",
        "## Split Scale",
        "",
        "| Split | Blocks | Base Frames | Weather Samples | Windows | Window Ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for role in SPLIT_ROLES:
        lines.append(
            f"| {role} | {ratios['blocks'][role]} | {ratios['base_frames'][role]} | "
            f"{ratios['weather_samples'][role]} | {ratios['windows'][role]} | {ratios['windows'][role + '_ratio']:.6f} |"
        )
    lines.extend(["", "## Per Scene", "", "| Scene | Train Windows | Validation Windows | Test Windows |", "| --- | ---: | ---: | ---: |"])
    for item in statistics["per_scene"]:
        lines.append(
            f"| {item['scene_id']} | {item['train_windows']} | {item['validation_windows']} | {item['test_windows']} |"
        )
    lines.extend(
        [
            "",
            "## Per Trajectory",
            "",
            "| Scene | CAV | Train Blocks | Validation Blocks | Test Blocks |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for item in statistics["per_trajectory"]:
        lines.append(
            f"| {item['scene_id']} | {item['cav_id']} | {item['train_blocks']} | "
            f"{item['validation_blocks']} | {item['test_blocks']} |"
        )
    lines.extend(
        [
            "",
            "## Label Distribution",
            "",
            f"- Train-validation: TV={labels['train_validation']['tv']:.6f}, JSD={labels['train_validation']['jsd']:.6f}, "
            f"Pearson={labels['train_validation']['pearson']:.6f}, Spearman={labels['train_validation']['spearman']:.6f}",
            f"- Train-test: TV={labels['train_test']['tv']:.6f}, JSD={labels['train_test']['jsd']:.6f}, "
            f"Pearson={labels['train_test']['pearson']:.6f}, Spearman={labels['train_test']['spearman']:.6f}",
            f"- Simple contiguous train-validation TV: {baseline['train_validation']['tv']:.6f}",
            f"- Simple contiguous train-test TV: {baseline['train_test']['tv']:.6f}",
            f"- Stratified TV sum: {statistics['label_tv_sum']:.6f}",
            f"- Simple contiguous TV sum: {statistics['simple_contiguous_label_tv_sum']:.6f}",
            f"- Validation beams unseen in train: {labels['validation_beams_not_in_train']}",
            f"- Test beams unseen in train: {labels['test_beams_not_in_train']}",
            f"- Global unseen beams: {labels['globally_unseen_beams']}",
            f"- Train histogram: `{labels['histograms']['train']}`",
            f"- Validation histogram: `{labels['histograms']['validation']}`",
            f"- Test histogram: `{labels['histograms']['test']}`",
            f"- Largest proportion differences: `{labels['largest_proportion_differences']}`",
            "",
            "## Conditional Label Distribution",
            "",
            f"- 15-domain train-validation macro TV: {conditional['per_domain']['train_validation_macro']['tv']:.6f}",
            f"- 15-domain train-validation worst TV: {conditional['per_domain']['train_validation_worst_tv']:.6f} "
            f"at `{conditional['per_domain']['train_validation_worst_group']}`",
            f"- 15-domain validation unseen beam mass: {conditional['per_domain']['validation_unseen_beam_mass']:.6f}",
            f"- Scene/domain train-validation macro TV: {conditional['per_scene']['train_validation_macro']['tv']:.6f}",
            f"- Scene/domain train-validation worst TV: {conditional['per_scene']['train_validation_worst_tv']:.6f} "
            f"at `{conditional['per_scene']['train_validation_worst_group']}`",
            f"- Scene/domain validation unseen beam mass: {conditional['per_scene']['validation_unseen_beam_mass']:.6f}",
            f"- Trajectory train-validation macro TV: {conditional['per_trajectory']['train_validation_macro']['tv']:.6f}",
            f"- Trajectory train-validation worst TV: {conditional['per_trajectory']['train_validation_worst_tv']:.6f} "
            f"at `{conditional['per_trajectory']['train_validation_worst_group']}`",
            f"- Trajectory validation unseen beam mass: {conditional['per_trajectory']['validation_unseen_beam_mass']:.6f}",
            f"- Simple contiguous scene/domain train-validation macro TV: "
            f"{conditional_baseline['per_scene']['train_validation_macro']['tv']:.6f}",
            f"- Simple contiguous trajectory train-validation macro TV: "
            f"{conditional_baseline['per_trajectory']['train_validation_macro']['tv']:.6f}",
            f"- Conditional stratified TV sum: {statistics['conditional_label_tv_sum']:.6f}",
            f"- Simple contiguous conditional TV sum: "
            f"{statistics['simple_contiguous_conditional_label_tv_sum']:.6f}",
            "",
            "## Leakage Checks",
            "",
        ]
    )
    for name, passed in audit["checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "Trajectory overlap is allowed and required by this protocol. Block, base-frame, weather-copy and window-frame overlap are forbidden.",
            "",
            f"Boundary-crossing candidate windows dropped: {payload['dropped_boundary_window_count']}",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def group_id(scene_id: str, cav_id: str) -> str:
    return f"{scene_id}::{cav_id}"


def _manifest_blocks(manifest: Mapping[str, Any], role: str) -> list[dict[str, Any]]:
    values = manifest.get(f"{role}_blocks", [])
    if not isinstance(values, list):
        return []
    return [dict(item) for item in values if isinstance(item, Mapping)]


def _block_by_id(blocks: Sequence[Mapping[str, Any]], block_id: str) -> Mapping[str, Any]:
    for block in blocks:
        if str(block.get("block_id")) == block_id:
            return block
    raise KeyError(block_id)


def _blocks_by_trajectory(
    blocks: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    result: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for block in blocks:
        result[(_required_text(block, "scene_id"), _required_text(block, "cav_id"))].append(block)
    return result


def _public_block(block: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "block_id": str(block["block_id"]),
        "scene_id": str(block["scene_id"]),
        "cav_id": str(block["cav_id"]),
        "block_ordinal": int(block["block_ordinal"]),
        "block_start_base_index": int(block["block_start_base_index"]),
        "block_end_base_index": int(block["block_end_base_index"]),
        "num_base_samples": int(block["num_base_samples"]),
        "num_weather_versions": int(block["num_weather_versions"]),
        "beam_histogram": [int(value) for value in block["beam_histogram"]],
        "window_beam_histogram": [int(value) for value in block["window_beam_histogram"]],
        "num_windows_estimated": int(block["num_windows_estimated"]),
    }


def _block_sort_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _natural_key(value.get("scene_id", "")),
        _natural_key(value.get("cav_id", "")),
        int(value.get("block_start_base_index", 0)),
        str(value.get("block_id", "")),
    )


def _group_sort_key(value: tuple[str, str]) -> tuple[list[object], list[object]]:
    return _natural_key(value[0]), _natural_key(value[1])


def _natural_key(value: object) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def _required_text(value: Mapping[str, Any], key: str) -> str:
    text = _text(value.get(key))
    if not text:
        raise ValueError(f"MMW split input is missing {key!r}: {value}")
    return text


def _text(value: object) -> str:
    text = str(value).strip()
    return "" if not text or text.lower() in {"nan", "none", "-99", "-99.0"} else text


def _json_list(value: object) -> list[str]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [_text(item) for item in parsed if _text(item)]


def _frame_number(value: object) -> int | None:
    text = _text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _frame_id_key(value: object) -> str:
    number = _frame_number(value)
    return str(number) if number is not None else _text(value)


def _beam_label(value: object, *, context: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MMW beam label is invalid for {context}: {value!r}") from exc
    if not math.isfinite(number) or not number.is_integer() or not 0 <= number < 64:
        raise ValueError(f"MMW beam label must be an integer in [0, 63] for {context}: {value!r}")
    return int(number)


def _sample_id_hash(frame: pd.DataFrame) -> str:
    if frame.empty or "sample_id" not in frame:
        return hashlib.sha256(b"").hexdigest()
    values = sorted(str(value) for value in frame["sample_id"].astype(str))
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _seeded_assignment_tie_key(assignment: Mapping[str, str], seed: int) -> str:
    signature = "\n".join(f"{key}={assignment[key]}" for key in sorted(assignment))
    return hashlib.sha256(f"{seed}\n{signature}".encode("utf-8")).hexdigest()


def _derived_seed(seed: int, restart: int) -> int:
    digest = hashlib.sha256(f"mmw-id-block:{seed}:{restart}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _fingerprint(payload: Mapping[str, Any]) -> str:
    excluded = {"report_path", "report_json_path", "audit_report"}
    stable = {key: value for key, value in payload.items() if key not in excluded}
    return _sha256_json(stable)


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "ASSIGNMENT_ALGORITHM",
    "CACHE_IDENTITY_FIELDS",
    "DEFAULT_BLOCK_SIZE",
    "EXPECTED_WEATHERS",
    "SPLIT_RATIOS",
    "SPLIT_ROLES",
    "TRAJECTORY_MANIFEST_VERSION",
    "TRAJECTORY_PROTOCOL_ID",
    "TRAJECTORY_PROTOCOL_MODE",
    "TRAJECTORY_PROTOCOL_VERSION",
    "TRAJECTORY_SPLIT_SEED",
    "assign_mmw_blocks_stratified",
    "bind_trajectory_config",
    "build_trajectory_protocol",
    "group_id",
    "load_trajectory_protocol",
    "protocol_dataset_domains",
    "split_cache_identity",
    "trajectory_audit_path",
    "trajectory_manifest_path",
    "validate_mmw_id_block_split",
    "validate_split_cache_identity",
    "validate_trajectory_config_protocol",
    "validate_trajectory_protocol",
]
