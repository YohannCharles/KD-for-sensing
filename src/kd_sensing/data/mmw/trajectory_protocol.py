"""Build and validate the sealed MMW trajectory-disjoint protocol."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from kd_sensing.preprocessing.mmw_radar import augment_mmw_sequence_resource_columns


TRAJECTORY_PROTOCOL_MODE = "trajectory_disjoint_development"
TRAJECTORY_PROTOCOL_ID = "mmw_trajectory_disjoint_v1"
TRAJECTORY_PROTOCOL_VERSION = 1
TRAJECTORY_SPLIT_SEED = 2026
EXPECTED_CANDIDATE_COUNT = 46_860
EXPECTED_DOMAIN_COUNT = 15
SPLIT_ROLES = ("train", "validation", "test")
EXPLICIT_TRAJECTORY_FIELDS = (
    "trajectory_id",
    "route_id",
    "run_id",
    "episode_id",
    "scenario_execution_id",
    "simulation_run_id",
)
RESOURCE_PATTERNS = {
    "camera_resource": re.compile(r"^camera\d+$"),
    "lidar_resource": re.compile(r"^lidar\d+$"),
    "radar_resource": re.compile(r"^radar\d+$"),
    "gps_resource": re.compile(r"^(?:gps|bs_gps)\d+$"),
    "channel_resource": re.compile(r"^(?:csi|future_csi|future_path|channel|path)\d+$"),
}
AUDIT_IDENTITIES = (
    "sample_identity",
    "target_identity",
    "csv_row",
    "dependency_frame",
    "camera_resource",
    "lidar_resource",
    "radar_resource",
    "gps_resource",
    "channel_resource",
    "trajectory_id",
    "trajectory_group_id",
    "scenario_execution_id",
)
HISTORICAL_ROOTS = (
    "outputs/clean_split_recovery",
    "outputs/cache/mmw_twc_outer_v1",
    "outputs/full_pool_capacity",
    "outputs/full_pool_candidate12_search",
    "outputs/full_pool_btma_ablation",
)


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            first, second = sorted((left_root, right_root))
            self.parent[second] = first


def build_trajectory_protocol(
    output_root: str | Path,
    *,
    dataset_root: str | Path = "dataset/MMW",
    historical_roots: Sequence[str | Path] = HISTORICAL_ROOTS,
    force: bool = False,
) -> dict[str, Any]:
    """Build the complete local protocol and fail before training on leakage."""

    root = Path(output_root).resolve()
    protocol_dir = root / "protocol"
    manifest_path = protocol_dir / "split_manifest.json"
    if manifest_path.is_file() and not force:
        return load_trajectory_protocol(manifest_path)
    protocol_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(dataset_root).resolve()
    sources = sorted(dataset_root.glob("*/Prepared/*/splits/h5p1_strict_v2/all_sequences.csv"))
    if len(sources) != EXPECTED_DOMAIN_COUNT:
        raise ValueError(f"Trajectory protocol requires 15 canonical domain CSVs, found {len(sources)}.")

    frame, source_records, inventory = _load_sources(sources, dataset_root)
    _write_csv(inventory, protocol_dir / "source_inventory.csv")
    if len(frame) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(f"Trajectory protocol requires 46,860 candidates, found {len(frame)}.")

    valid, anomalies = _validate_candidates(frame, dataset_root)
    groups, grouped = reconstruct_trajectory_groups(valid)
    cross_group = _cross_group_window_anomalies(grouped)
    anomalies.extend(cross_group)
    invalid_rows = {int(item["source_global_row"]) for item in cross_group}
    if invalid_rows:
        grouped = grouped.loc[~grouped["source_global_row"].isin(invalid_rows)].copy()
        groups, grouped = reconstruct_trajectory_groups(grouped.drop(columns=["trajectory_group_id"], errors="ignore"))

    group_frame = pd.DataFrame(groups)
    _write_csv(group_frame, protocol_dir / "trajectory_groups.csv")
    anomaly_columns = ("source_global_row", "sample_id", "domain", "reason", "action", "detail")
    _write_csv(pd.DataFrame(anomalies, columns=anomaly_columns), protocol_dir / "trajectory_group_anomalies.csv")

    assignments, split_process = assign_trajectory_groups(groups, seed=TRAJECTORY_SPLIT_SEED)
    grouped["split"] = grouped["trajectory_group_id"].map(assignments)
    if grouped["split"].isna().any():
        raise ValueError("Every valid window must belong to exactly one trajectory split.")

    domain_records, role_frames = _write_domain_splits(grouped, protocol_dir, dataset_root)
    split_hashes = _write_split_identities(role_frames, protocol_dir)
    _write_distribution_artifacts(role_frames, groups, assignments, protocol_dir)

    preliminary_audit = audit_trajectory_splits(role_frames)
    group_audit = {
        "schema_version": 1,
        "audit_id": "mmw_trajectory_group_audit_v1",
        "status": "passed" if not anomalies and not preliminary_audit["reasons"] else "failed",
        "candidate_windows": len(frame),
        "valid_windows": len(grouped),
        "cross_trajectory_windows_removed": len(invalid_rows),
        "missing_resource_windows_removed": sum(item["reason"] == "missing_resource" for item in anomalies),
        "trajectory_group_count": len(groups),
        "raw_trajectory_count": int(grouped["raw_trajectory_id"].nunique()),
        "shared_rsu_cross_cav_group_count": sum(len(item["cav_ids"].split("|")) > 1 for item in groups),
        "trajectory_definition": "explicit run metadata when present; otherwise resource-coupled connected components",
        "resource_coupling_rule": "shared Radar/BS-GPS/target/dependency/CSV row or overlapping scenario execution",
        "split_process": split_process,
        "anomaly_count": len(anomalies),
        "anomaly_reasons": dict(Counter(item["reason"] for item in anomalies)),
    }
    _write_json(protocol_dir / "trajectory_group_audit.json", group_audit)
    if group_audit["status"] != "passed":
        raise ValueError(f"Trajectory group audit failed: {group_audit['anomaly_reasons'] or preliminary_audit['reasons']}")

    exposure_rows, exposure_summary = audit_historical_exposure(
        grouped,
        [Path(value).resolve() for value in historical_roots],
    )
    _write_csv(pd.DataFrame(exposure_rows), protocol_dir / "historical_exposure_audit.csv")
    _write_json(protocol_dir / "historical_exposure_summary.json", exposure_summary)

    counts = Counter(assignments.values())
    windows = {role: int(len(role_frames[role])) for role in SPLIT_ROLES}
    payload: dict[str, Any] = {
        "schema_version": 1,
        "mode": TRAJECTORY_PROTOCOL_MODE,
        "protocol_name": "Multimodal-Wireless Original Trajectory Split",
        "protocol_id": TRAJECTORY_PROTOCOL_ID,
        "protocol_version": TRAJECTORY_PROTOCOL_VERSION,
        "split_seed": TRAJECTORY_SPLIT_SEED,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_csvs": source_records,
        "trajectory_definition": group_audit["trajectory_definition"],
        "resource_coupling_rule": group_audit["resource_coupling_rule"],
        "trajectory_group_count": len(groups),
        "train_group_count": counts["train"],
        "validation_group_count": counts["validation"],
        "test_group_count": counts["test"],
        "candidate_window_count": len(frame),
        "valid_window_count": len(grouped),
        "cross_trajectory_windows_removed": len(invalid_rows),
        "missing_resource_windows_removed": group_audit["missing_resource_windows_removed"],
        "train_window_count": windows["train"],
        "validation_window_count": windows["validation"],
        "test_window_count": windows["test"],
        "train_group_ids": sorted(group for group, role in assignments.items() if role == "train"),
        "validation_group_ids": sorted(group for group, role in assignments.items() if role == "validation"),
        "test_group_ids": sorted(group for group, role in assignments.items() if role == "test"),
        "train_role": "train",
        "validation_role": "validation",
        "test_role": "test_sealed",
        "split_hashes": split_hashes,
        "claim_eligible": bool(exposure_summary["claim_eligible"]),
        "legacy_protocol_used": False,
        "chronological_tail_split_used": False,
        "random_window_split_used": False,
        "outer_test_enabled": False,
        "outer_test_accessed": False,
        "allow_confirmation_train": False,
        "test_evaluation_requires_explicit_authorization": True,
        "domains": domain_records,
    }
    payload["protocol_fingerprint"] = _fingerprint(payload)
    _write_json(manifest_path, payload)
    (protocol_dir / "split_manifest.yaml").write_text(yaml.safe_dump(payload, sort_keys=True, allow_unicode=True), encoding="utf-8")

    split_audit = {**preliminary_audit, "protocol_fingerprint": payload["protocol_fingerprint"]}
    _write_json(protocol_dir / "split_audit.json", split_audit)
    _write_split_audit_markdown(split_audit, protocol_dir / "split_audit.md")
    if split_audit["status"] != "passed":
        raise ValueError(f"Trajectory split resource audit failed: {split_audit['reasons']}")
    _write_protocol_summary(payload, group_audit, split_audit, exposure_summary, protocol_dir)
    _write_protocol_comparison(protocol_dir)
    return load_trajectory_protocol(manifest_path)


def _load_sources(sources: Sequence[Path], dataset_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    inventories: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    offset = 0
    for source in sources:
        weather, scene = source.parents[4].name, source.parents[2].name
        domain = f"{weather}/{scene}"
        original = pd.read_csv(source, na_values="").fillna("")
        columns = list(original.columns)
        augmented = augment_mmw_sequence_resource_columns(original, scene)
        augmented.insert(0, "domain_id", domain)
        augmented["source_csv"] = str(source.resolve())
        augmented["source_row"] = np.arange(len(augmented), dtype=np.int64)
        augmented["source_global_row"] = np.arange(offset, offset + len(augmented), dtype=np.int64)
        augmented["raw_trajectory_id"] = _raw_trajectory_ids(augmented, domain)
        explicit_execution = next(
            (
                field
                for field in ("scenario_execution_id", "simulation_run_id", "run_id", "episode_id", "route_id")
                if field in original and original[field].astype(str).str.strip().ne("").all()
            ),
            None,
        )
        augmented["scenario_execution_id"] = (
            [f"{domain}:{explicit_execution}:{value}" for value in original[explicit_execution].astype(str)]
            if explicit_execution
            else domain
        )
        augmented["source_row_sha256"] = [
            _sha256_json({column: str(value) for column, value in zip(columns, values)})
            for values in original.astype(str).itertuples(index=False, name=None)
        ]
        frames.append(augmented)
        offset += len(augmented)
        records.append(
            {
                "path": str(source.resolve()),
                "sha256": _sha256_file(source),
                "domain": domain,
                "weather": weather,
                "scenario": scene,
                "window_count": len(augmented),
            }
        )
        for row in augmented.to_dict(orient="records"):
            history = _json_list(row.get("history_frame_ids_json"))
            future = _json_list(row.get("future_frame_ids_json"))
            inventories.append(
                {
                    "sample_id": _identity(domain, row.get("sample_id")),
                    "domain": domain,
                    "town": str(row.get("town", "")),
                    "scenario": str(row.get("sensor_scenario") or scene),
                    "weather": weather,
                    "trajectory_id": row["raw_trajectory_id"],
                    "run_id": _first_text(row, ("run_id", "route_id", "simulation_run_id")),
                    "episode_id": _first_text(row, ("episode_id",)),
                    "cav_id": str(row.get("agent", "")),
                    "target_frame": json.dumps([_identity(domain, value) for value in future]),
                    "dependency_frames": json.dumps([_identity(domain, value) for value in [*history, *future]]),
                    "camera_resources": json.dumps(_row_resources(row, "camera_resource", domain)),
                    "lidar_resources": json.dumps(_row_resources(row, "lidar_resource", domain)),
                    "radar_resources": json.dumps(_row_resources(row, "radar_resource", domain)),
                    "gps_resources": json.dumps(_row_resources(row, "gps_resource", domain)),
                    "beam_label": row.get("future_beam_label1", row.get("beam_label", "")),
                }
            )
    return pd.concat(frames, ignore_index=True), records, pd.DataFrame(inventories)


def _raw_trajectory_ids(frame: pd.DataFrame, domain: str) -> list[str]:
    explicit = next(
        (field for field in EXPLICIT_TRAJECTORY_FIELDS if field in frame and frame[field].astype(str).str.strip().ne("").all()),
        None,
    )
    if explicit:
        return [f"{domain}:{explicit}:{value}" for value in frame[explicit].astype(str)]
    if "contiguous_segment_id" not in frame:
        raise ValueError("MMW source lacks explicit trajectory metadata and contiguous_segment_id.")
    return [f"{domain}:segment:{value}" for value in frame["contiguous_segment_id"].astype(str)]


def _validate_candidates(frame: pd.DataFrame, dataset_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    anomalies: list[dict[str, Any]] = []
    valid = pd.Series(True, index=frame.index)
    labels = pd.to_numeric(frame.get("future_beam_label1"), errors="coerce")
    invalid_labels = labels.isna() | ~np.isfinite(labels) | labels.mod(1).ne(0) | ~labels.between(0, 63)
    for index in frame.index[invalid_labels]:
        anomalies.append(_anomaly(frame.loc[index], "invalid_beam_label", "excluded", str(frame.at[index, "future_beam_label1"])))
    valid &= ~invalid_labels

    for index, row in frame.iterrows():
        missing: list[str] = []
        condition_root = dataset_root / str(row["condition"])
        for family in ("camera_resource", "lidar_resource", "radar_resource", "gps_resource"):
            for identity in _row_resources(row, family, ""):
                relative = identity.lstrip(":")
                if not (condition_root / relative).is_file():
                    missing.append(relative)
                if family == "radar_resource" and "_RA" in relative and not (condition_root / relative.replace("_RA", "_DA")).is_file():
                    missing.append(relative.replace("_RA", "_DA"))
        if missing:
            valid.at[index] = False
            anomalies.append(_anomaly(row, "missing_resource", "excluded", "|".join(missing[:20])))
    return frame.loc[valid].copy(), anomalies


def reconstruct_trajectory_groups(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Return deterministic connected components and annotate every row."""

    if frame.empty:
        raise ValueError("Trajectory reconstruction requires at least one valid window.")
    nodes = sorted(frame["raw_trajectory_id"].astype(str).unique())
    union = _UnionFind(nodes)
    owners: dict[str, str] = {}
    node_ranges: dict[str, tuple[str, int, int]] = {}
    for node, rows in frame.groupby("raw_trajectory_id", sort=True):
        domain = str(rows["domain_id"].iloc[0])
        executions = sorted(rows["scenario_execution_id"].astype(str).unique())
        if len(executions) != 1:
            raise ValueError(f"Raw trajectory {node} spans multiple scenario executions: {executions}")
        start = int(pd.to_numeric(rows["window_start_frame"]).min())
        end = int(pd.to_numeric(rows["window_end_frame"]).max())
        node_ranges[str(node)] = (executions[0], start, end)
        identities: set[str] = set()
        for _, row in rows.iterrows():
            identities.add(f"csv:{domain}:{row['source_row_sha256']}")
            identities.update(f"target:{value}" for value in _target_frames(row))
            identities.update(f"dependency:{value}" for value in _dependency_frames(row))
            identities.update(f"radar:{value}" for value in _row_resources(row, "radar_resource", domain))
            identities.update(f"gps:{value}" for value in _row_resources(row, "gps_resource", domain) if "/rsu_" in value)
        for identity in identities:
            previous = owners.setdefault(identity, str(node))
            union.union(previous, str(node))

    by_execution: dict[str, list[str]] = defaultdict(list)
    for node, (execution, _, _) in node_ranges.items():
        by_execution[execution].append(node)
    for values in by_execution.values():
        for left, right in itertools.combinations(sorted(values), 2):
            _, left_start, left_end = node_ranges[left]
            _, right_start, right_end = node_ranges[right]
            if max(left_start, right_start) <= min(left_end, right_end):
                union.union(left, right)

    components: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        components[union.find(node)].append(node)
    node_to_group: dict[str, str] = {}
    for members in components.values():
        group_id = "tg_" + hashlib.sha256("\n".join(sorted(members)).encode("utf-8")).hexdigest()[:16]
        node_to_group.update({member: group_id for member in members})
    annotated = frame.copy()
    annotated["trajectory_group_id"] = annotated["raw_trajectory_id"].map(node_to_group)

    groups: list[dict[str, Any]] = []
    for group_id, rows in annotated.groupby("trajectory_group_id", sort=True):
        domains = sorted(rows["domain_id"].astype(str).unique())
        groups.append(
            {
                "trajectory_group_id": group_id,
                "raw_trajectory_ids": "|".join(sorted(rows["raw_trajectory_id"].astype(str).unique())),
                "domains": "|".join(domains),
                "weather": "|".join(sorted(rows["condition"].astype(str).unique())),
                "scenario": "|".join(sorted(rows["sensor_scenario"].astype(str).unique())),
                "cav_ids": "|".join(sorted(rows["agent"].astype(str).unique())),
                "start_frame": int(pd.to_numeric(rows["window_start_frame"]).min()),
                "end_frame": int(pd.to_numeric(rows["window_end_frame"]).max()),
                "window_count": len(rows),
                "unique_target_count": rows["target_sample_id"].astype(str).nunique(),
                "camera_resource_count": len(_frame_resource_set(rows, "camera_resource")),
                "lidar_resource_count": len(_frame_resource_set(rows, "lidar_resource")),
                "radar_resource_count": len(_frame_resource_set(rows, "radar_resource")),
                "gps_resource_count": len(_frame_resource_set(rows, "gps_resource")),
            }
        )
    return groups, annotated


def split_group_counts(total: int) -> tuple[int, int, int]:
    if total < 3:
        raise ValueError("Trajectory protocol requires at least three groups.")
    if total == 50:
        return 40, 5, 5
    candidates = []
    for validation in range(1, total - 1):
        for test in range(1, total - validation):
            train = total - validation - test
            error = sum((value - total * ratio) ** 2 for value, ratio in zip((train, validation, test), (0.8, 0.1, 0.1)))
            candidates.append((error, validation < test, -train, -validation, train, validation, test))
    return min(candidates)[-3:]


def assign_trajectory_groups(
    groups: Sequence[Mapping[str, Any]], *, seed: int = TRAJECTORY_SPLIT_SEED
) -> tuple[dict[str, str], dict[str, Any]]:
    targets = dict(zip(SPLIT_ROLES, split_group_counts(len(groups))))
    values = [dict(group) for group in groups]
    if len(values) <= 20:
        assignment = _exhaustive_assignment(values, targets, seed)
        algorithm = "exact_group_level_stratified_enumeration"
    else:
        assignment = _greedy_assignment(values, targets, seed)
        algorithm = "deterministic_group_level_stratified_greedy"
    coverage = {
        role: {
            "weather": sorted({str(group["weather"]) for group in values if assignment[str(group["trajectory_group_id"])] == role}),
            "scenario": sorted({str(group["scenario"]) for group in values if assignment[str(group["trajectory_group_id"])] == role}),
        }
        for role in SPLIT_ROLES
    }
    all_weather = sorted({str(group["weather"]) for group in values})
    all_scenarios = sorted({str(group["scenario"]) for group in values})
    constraints = []
    for role in SPLIT_ROLES:
        for family, expected in (("weather", all_weather), ("scenario", all_scenarios)):
            missing = sorted(set(expected) - set(coverage[role][family]))
            if missing:
                constraints.append({"split": role, "family": family, "missing": missing, "reason": "group_count_or_integrity_constraint"})
    return assignment, {
        "algorithm": algorithm,
        "seed": seed,
        "target_group_counts": targets,
        "coverage": coverage,
        "unmet_stratification_constraints": constraints,
        "model_results_consulted": False,
    }


def _exhaustive_assignment(groups: list[dict[str, Any]], targets: dict[str, int], seed: int) -> dict[str, str]:
    indices = range(len(groups))
    best: tuple[float, str, dict[str, str]] | None = None
    for test_indices in itertools.combinations(indices, targets["test"]):
        remaining = [index for index in indices if index not in test_indices]
        for validation_indices in itertools.combinations(remaining, targets["validation"]):
            test_set, validation_set = set(test_indices), set(validation_indices)
            assignment = {
                str(group["trajectory_group_id"]): "test" if index in test_set else "validation" if index in validation_set else "train"
                for index, group in enumerate(groups)
            }
            score = _stratification_score(groups, assignment, targets)
            tie = hashlib.sha256(
                f"{seed}:".encode("utf-8") + json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            candidate = (score, tie, assignment)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
    assert best is not None
    return best[2]


def _greedy_assignment(groups: list[dict[str, Any]], targets: dict[str, int], seed: int) -> dict[str, str]:
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}:{group['trajectory_group_id']}".encode("utf-8")).hexdigest(),
    )
    assignment: dict[str, str] = {}
    for group in ordered:
        choices = []
        for role in SPLIT_ROLES:
            if sum(value == role for value in assignment.values()) >= targets[role]:
                continue
            trial = {**assignment, str(group["trajectory_group_id"]): role}
            choices.append((_partial_score(groups, trial, targets), role))
        assignment[str(group["trajectory_group_id"])] = min(choices)[1]
    return assignment


def _stratification_score(groups: Sequence[Mapping[str, Any]], assignment: Mapping[str, str], targets: Mapping[str, int]) -> float:
    score = 0.0
    total_windows = sum(int(group["window_count"]) for group in groups)
    for role in SPLIT_ROLES:
        ratio = targets[role] / len(groups)
        selected = [group for group in groups if assignment[str(group["trajectory_group_id"])] == role]
        score += ((sum(int(group["window_count"]) for group in selected) - total_windows * ratio) / total_windows) ** 2
        for field in ("weather", "scenario"):
            total = Counter(str(group[field]) for group in groups)
            actual = Counter(str(group[field]) for group in selected)
            score += sum((actual[value] - count * ratio) ** 2 for value, count in total.items())
    return score


def _partial_score(groups: Sequence[Mapping[str, Any]], assignment: Mapping[str, str], targets: Mapping[str, int]) -> float:
    assigned = [group for group in groups if str(group["trajectory_group_id"]) in assignment]
    return _stratification_score(
        assigned, assignment, {role: max(1, round(len(assigned) * targets[role] / len(groups))) for role in SPLIT_ROLES}
    )


def _cross_group_window_anomalies(frame: pd.DataFrame) -> list[dict[str, Any]]:
    dependency_groups: dict[str, set[str]] = defaultdict(set)
    for _, row in frame.iterrows():
        for identity in _dependency_frames(row):
            dependency_groups[identity].add(str(row["trajectory_group_id"]))
    anomalies = []
    bad = {identity for identity, groups in dependency_groups.items() if len(groups) > 1}
    if bad:
        for _, row in frame.iterrows():
            shared = sorted(set(_dependency_frames(row)) & bad)
            if shared:
                anomalies.append(_anomaly(row, "cross_trajectory_dependency", "excluded", "|".join(shared[:20])))
    return anomalies


def _write_domain_splits(
    frame: pd.DataFrame, protocol_dir: Path, dataset_root: Path
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    internal = {"source_csv", "source_row", "source_global_row", "raw_trajectory_id"}
    records: list[dict[str, Any]] = []
    role_frames = {role: frame.loc[frame["split"] == role].copy() for role in SPLIT_ROLES}
    for domain, rows in frame.groupby("domain_id", sort=True):
        weather, scene = str(domain).split("/", 1)
        record: dict[str, Any] = {
            "id": domain,
            "condition": weather,
            "scene": scene,
            "data_root": str((dataset_root / weather).resolve()),
        }
        for role in SPLIT_ROLES:
            selected = rows.loc[rows["split"] == role].drop(columns=list(internal), errors="ignore")
            if selected.empty:
                continue
            path = protocol_dir / "splits" / role / f"{weather}__{scene}.csv"
            _write_csv(selected, path)
            record[f"{role}_split"] = str(path)
            record[f"{role}_csv_sha256"] = _sha256_file(path)
            record[f"{role}_sample_count"] = len(selected)
        records.append(record)
    return records, role_frames


def _write_split_identities(role_frames: Mapping[str, pd.DataFrame], protocol_dir: Path) -> dict[str, str]:
    hashes = {}
    for role, frame in role_frames.items():
        sample_ids = sorted(_identity(str(row.domain_id), row.sample_id) for row in frame.itertuples())
        group_ids = sorted(frame["trajectory_group_id"].astype(str).unique())
        sample_text = "\n".join(sample_ids) + "\n"
        group_text = "\n".join(group_ids) + "\n"
        (protocol_dir / f"{role}_sample_ids.txt").write_text(sample_text, encoding="utf-8")
        (protocol_dir / f"{role}_trajectory_groups.txt").write_text(group_text, encoding="utf-8")
        digest = hashlib.sha256(sample_text.encode("utf-8")).hexdigest()
        (protocol_dir / f"{role}_sha256.txt").write_text(digest + "\n", encoding="utf-8")
        hashes[role] = digest
    return hashes


def audit_trajectory_splits(role_frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    identities = {role: _audit_identity_sets(frame) for role, frame in role_frames.items()}
    pairs: dict[str, Any] = {}
    reasons: list[str] = []
    for left, right in itertools.combinations(SPLIT_ROLES, 2):
        overlaps = {}
        for name in AUDIT_IDENTITIES:
            values = sorted(identities[left][name] & identities[right][name])
            overlaps[name] = {"count": len(values), "examples": values[:20]}
            if values:
                reasons.append(f"{left}_vs_{right}:{name}")
        pairs[f"{left}_vs_{right}"] = overlaps
    return {
        "schema_version": 1,
        "audit_id": "mmw_trajectory_split_resource_isolation_v1",
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "pairwise_overlaps": pairs,
        "train_sample_id_hash": _sample_id_hash(role_frames["train"]),
        "validation_sample_id_hash": _sample_id_hash(role_frames["validation"]),
        "test_sample_id_hash": _sample_id_hash(role_frames["test"]),
        "train_sample_count": len(role_frames["train"]),
        "validation_sample_count": len(role_frames["validation"]),
        "test_sample_count": len(role_frames["test"]),
        "outer_test_accessed": False,
    }


def _audit_identity_sets(frame: pd.DataFrame) -> dict[str, set[str]]:
    result = {name: set() for name in AUDIT_IDENTITIES}
    for _, row in frame.iterrows():
        domain = str(row["domain_id"])
        result["sample_identity"].add(_identity(domain, row.get("sample_id")))
        result["target_identity"].add(_identity(domain, row.get("target_sample_id")))
        result["csv_row"].add(f"{domain}:{row.get('source_row_sha256', '')}")
        result["dependency_frame"].update(_dependency_frames(row))
        for family in RESOURCE_PATTERNS:
            result[family].update(_row_resources(row, family, domain))
        result["trajectory_id"].add(str(row["raw_trajectory_id"]))
        result["trajectory_group_id"].add(str(row["trajectory_group_id"]))
        result["scenario_execution_id"].add(str(row["scenario_execution_id"]))
    return result


def audit_historical_exposure(frame: pd.DataFrame, roots: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    known = set(frame["sample_id"].astype(str))
    roles: dict[str, dict[str, set[str]]] = {role: defaultdict(set) for role in SPLIT_ROLES}
    scanned: list[str] = []
    unavailable: list[str] = []
    candidates: set[Path] = set()
    for root in roots:
        if not root.exists():
            unavailable.append(str(root))
            continue
        manifests = {
            path
            for path in root.rglob("*")
            if path.is_file() and "manifest" in path.name.lower() and path.suffix.lower() in {".json", ".yaml", ".yml"}
        }
        candidates.update(manifests)
        for manifest in manifests:
            candidates.update(_manifest_csv_references(manifest))
        protocol_dir = root / "protocol"
        if protocol_dir.is_dir():
            candidates.update(protocol_dir.glob("*train.csv"))
            candidates.update(protocol_dir.glob("*validation.csv"))
            candidates.update(protocol_dir.glob("*test.csv"))

    candidates = {path.resolve() for path in candidates if path.is_file() and "full_pool_capacity/cache/splits" not in path.as_posix()}
    for path in sorted(candidates):
        recovered = _recover_historical_file(path, known)
        if not any(recovered.values()):
            continue
        scanned.append(str(path))
        for role, values in recovered.items():
            for value in values:
                roles[role][value].add(str(path))

    rows: list[dict[str, Any]] = []
    totals = {role: 0 for role in SPLIT_ROLES}
    test_exposed = False
    for (split, group_id), group in frame.groupby(["split", "trajectory_group_id"], sort=True):
        ids = set(group["sample_id"].astype(str))
        flags = {role: ids & set(roles[role]) for role in SPLIT_ROLES}
        totals["train"] += len(flags["train"])
        totals["validation"] += len(flags["validation"])
        totals["test"] += len(flags["test"])
        if split == "test" and (flags["train"] or flags["validation"]):
            test_exposed = True
        sources = sorted({source for role in SPLIT_ROLES for value in flags[role] for source in roles[role][value]})
        rows.append(
            {
                "split": split,
                "trajectory_group_id": group_id,
                "sample_count": len(group),
                "historically_seen_in_train": len(flags["train"]),
                "historically_seen_in_validation": len(flags["validation"]),
                "historically_seen_in_test": len(flags["test"]),
                "never_exposed": len(ids - set().union(*flags.values())),
                "source_manifests": "|".join(sources),
            }
        )
    summary = {
        "schema_version": 1,
        "scanned_files": scanned,
        "unavailable_roots": unavailable,
        "historical_train_exposures": totals["train"],
        "historical_validation_exposures": totals["validation"],
        "historical_test_exposures": totals["test"],
        "test_previously_exposed_for_training_or_selection": test_exposed,
        "claim_eligible": not test_exposed,
        "policy": "false when any sealed test sample was historically used for train or method selection",
    }
    return rows, summary


def _manifest_csv_references(path: Path) -> set[Path]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return set()
    references: set[Path] = set()

    def visit(value: Any, context: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                visit(nested, (*context, str(key).lower()))
        elif isinstance(value, list):
            for nested in value:
                visit(nested, context)
        elif isinstance(value, str) and value.lower().endswith(".csv"):
            context_text = "/".join(context)
            if "source_csv" in context_text or not any(
                token in context_text
                for token in ("split", "inner_train", "inner_validation", "outer", "test", "train_splits", "validation_splits")
            ):
                return
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate).resolve()
            if candidate.is_file():
                references.add(candidate)

    visit(payload)
    return references


def _recover_historical_file(path: Path, known: set[str]) -> dict[str, set[str]]:
    recovered = {role: set() for role in SPLIT_ROLES}
    default_role = _historical_role(path.as_posix())
    try:
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                columns = set(reader.fieldnames or ())
                identity_columns = columns & {"sample_id", "source_sample_id", "target_sample_id", "stable_sample_id"}
                if not identity_columns:
                    return recovered
                for row in reader:
                    row_role = _historical_role(str(row.get("split") or row.get("role") or default_role))
                    for column in identity_columns:
                        value = _canonical_historical_id(row.get(column, ""), known)
                        if value:
                            recovered[row_role].add(value)
            return recovered
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, csv.Error, yaml.YAMLError):
        return recovered

    def visit(value: Any, role: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                visit(nested, _historical_role(str(key), fallback=role))
        elif isinstance(value, list):
            for nested in value:
                visit(nested, role)
        elif isinstance(value, str):
            identity = _canonical_historical_id(value, known)
            if identity:
                recovered[role].add(identity)

    visit(payload, default_role)
    return recovered


def _historical_role(value: str, *, fallback: str = "validation") -> str:
    text = value.lower()
    if "outer" in text or "test" in text:
        return "test"
    if "train" in text or "assignment" in text or "warmup" in text:
        return "train"
    if "validation" in text or "val" in text or "eval" in text or "metric" in text:
        return "validation"
    return fallback if fallback in SPLIT_ROLES else "validation"


def _canonical_historical_id(value: object, known: set[str]) -> str | None:
    text = str(value).strip()
    if text in known:
        return text
    for weather in ("sunny", "rainy", "foggy"):
        marker = f"{weather}:Town03:"
        if marker in text:
            candidate = text[text.index(marker) :]
            candidate = candidate.split("|", 1)[0]
            if candidate in known:
                return candidate
    return None


def load_trajectory_protocol(path: str | Path) -> dict[str, Any]:
    return validate_trajectory_protocol(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_trajectory_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(protocol)
    if payload.get("mode") != TRAJECTORY_PROTOCOL_MODE or payload.get("protocol_id") != TRAJECTORY_PROTOCOL_ID:
        raise ValueError("Invalid MMW trajectory protocol mode or id.")
    if payload.get("outer_test_enabled") is not False or payload.get("outer_test_accessed") is not False:
        raise ValueError("Trajectory protocol must keep test disabled and unaccessed.")
    if payload.get("allow_confirmation_train") is not False:
        raise ValueError("Trajectory protocol must disallow confirmation training.")
    if int(payload.get("candidate_window_count", -1)) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("Trajectory protocol candidate count mismatch.")
    if int(payload.get("trajectory_group_count", 0)) < 3:
        raise ValueError("Trajectory protocol needs at least three groups.")
    if sum(int(payload.get(f"{role}_group_count", -1)) for role in SPLIT_ROLES) != int(payload["trajectory_group_count"]):
        raise ValueError("Trajectory protocol group counts do not add up.")
    for domain in payload.get("domains", []):
        present = 0
        for role in SPLIT_ROLES:
            if not domain.get(f"{role}_split"):
                continue
            present += 1
            path = Path(str(domain[f"{role}_split"]))
            if not path.is_file() or _sha256_file(path) != domain.get(f"{role}_csv_sha256"):
                raise ValueError(f"Trajectory protocol split is missing or changed: {path}")
        if present != 1:
            raise ValueError(f"Trajectory domain {domain.get('id')} must belong to exactly one split.")
    expected = str(payload.pop("protocol_fingerprint", ""))
    actual = _fingerprint(payload)
    if expected != actual:
        raise ValueError("Trajectory protocol fingerprint mismatch.")
    payload["protocol_fingerprint"] = actual
    return payload


def validate_trajectory_config_protocol(cfg: Mapping[str, Any]) -> dict[str, Any]:
    section = cfg.get("data_protocol")
    if not isinstance(section, Mapping) or section.get("mode") != TRAJECTORY_PROTOCOL_MODE:
        raise ValueError(f"data_protocol must declare mode={TRAJECTORY_PROTOCOL_MODE}.")
    protocol = load_trajectory_protocol(Path(str(section.get("path", ""))).resolve())
    for key in ("protocol_id", "protocol_fingerprint", "train_role", "validation_role"):
        if section.get(key) != protocol.get(key):
            raise ValueError("MMW trajectory config must bind the exact protocol identity and roles.")
    if section.get("outer_test_enabled") is not False or section.get("allow_confirmation_train") is not False:
        raise ValueError("MMW trajectory config must disable test and confirmation training.")
    allow_test = section.get("allow_test_evaluation") is True
    expected_domains = protocol_dataset_domains(protocol, allow_test_evaluation=allow_test)
    actual_domains = cfg.get("data", {}).get("dataset", {}).get("domains")
    if actual_domains != expected_domains:
        raise ValueError("Resolved dataset domains do not exactly match the trajectory protocol.")
    if not allow_test and any(domain.get("test_csv_name") for domain in actual_domains):
        raise ValueError("Ordinary trajectory training must not carry test CSV paths.")
    final_test = cfg.get("training", {}).get("final_test")
    enabled = final_test if isinstance(final_test, bool) else (final_test or {}).get("enabled", True)
    if bool(enabled) and not allow_test:
        raise ValueError("Trajectory training must explicitly disable final test.")
    report = json.loads(Path(str(section.get("audit_report", ""))).read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("outer_test_accessed") is not False:
        raise ValueError("Trajectory split audit is missing, failed, or accessed test predictions.")
    if report.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
        raise ValueError("Trajectory split audit fingerprint mismatch.")
    return report


def protocol_dataset_domains(protocol: Mapping[str, Any], *, allow_test_evaluation: bool = False) -> list[dict[str, Any]]:
    result = []
    for domain in protocol["domains"]:
        item: dict[str, Any] = {
            "id": domain["id"],
            "condition": domain["condition"],
            "scene": domain["scene"],
            "data_root": domain["data_root"],
        }
        if domain.get("train_split"):
            item["train_csv_name"] = domain["train_split"]
        if domain.get("validation_split"):
            item["val_csv_name"] = domain["validation_split"]
        if domain.get("test_split") and allow_test_evaluation:
            item["test_csv_name"] = domain["test_split"]
        result.append(item)
    return result


def _write_distribution_artifacts(
    role_frames: Mapping[str, pd.DataFrame],
    groups: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
    protocol_dir: Path,
) -> None:
    for role, frame in role_frames.items():
        counts = Counter(int(value) for value in frame["future_beam_label1"])
        _write_csv(
            pd.DataFrame([{"beam": beam, "window_count": counts.get(beam, 0)} for beam in range(64)]),
            protocol_dir / f"{role}_beam_distribution.csv",
        )
    weather_rows, domain_rows = [], []
    for role, frame in role_frames.items():
        for weather, selected in frame.groupby("condition", sort=True):
            weather_rows.append(_count_row(role, "weather", weather, selected))
        for domain, selected in frame.groupby("domain_id", sort=True):
            domain_rows.append(_count_row(role, "domain", domain, selected))
    _write_csv(pd.DataFrame(weather_rows), protocol_dir / "per_weather_counts.csv")
    _write_csv(pd.DataFrame(domain_rows), protocol_dir / "per_domain_counts.csv")


def _count_row(role: str, field: str, value: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "split": role,
        field: value,
        "trajectory_groups": frame["trajectory_group_id"].nunique(),
        "windows": len(frame),
        "unique_targets": frame["target_sample_id"].nunique(),
        "beam_classes": frame["future_beam_label1"].nunique(),
        "start_frame": int(pd.to_numeric(frame["window_start_frame"]).min()),
        "end_frame": int(pd.to_numeric(frame["window_end_frame"]).max()),
    }


def _write_split_audit_markdown(audit: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Trajectory Split Resource Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        "| Split pair | Identity | Intersection |",
        "| --- | --- | ---: |",
    ]
    for pair, values in audit["pairwise_overlaps"].items():
        for name, result in values.items():
            lines.append(f"| {pair} | {name} | {result['count']} |")
    lines.extend(("", "`outer_test_accessed=false`; channel identities were used only for leakage audit."))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_protocol_summary(
    protocol: Mapping[str, Any],
    group_audit: Mapping[str, Any],
    split_audit: Mapping[str, Any],
    exposure: Mapping[str, Any],
    protocol_dir: Path,
) -> None:
    distributions = {
        role: pd.read_csv(protocol_dir / f"{role}_beam_distribution.csv")["window_count"].to_numpy(dtype=float) for role in SPLIT_ROLES
    }
    lines = [
        "# MMW Trajectory-Disjoint Protocol Summary",
        "",
        f"- Trajectory groups: {protocol['trajectory_group_count']}",
        f"- Split groups: {protocol['train_group_count']}/{protocol['validation_group_count']}/{protocol['test_group_count']}",
        f"- Split windows: {protocol['train_window_count']}/{protocol['validation_window_count']}/{protocol['test_window_count']}",
        f"- Shared-RSU multi-CAV groups: {group_audit['shared_rsu_cross_cav_group_count']}",
        f"- Resource audit: {split_audit['status']}",
        f"- Claim eligible: {str(protocol['claim_eligible']).lower()}",
        f"- Outer test accessed: {str(protocol['outer_test_accessed']).lower()}",
        "",
        "| Pair | Pearson beam correlation | Total variation | Jensen-Shannon distance |",
        "| --- | ---: | ---: | ---: |",
    ]
    for left, right in itertools.combinations(SPLIT_ROLES, 2):
        correlation, tv, js = _distribution_distances(distributions[left], distributions[right])
        lines.append(f"| {left}-{right} | {correlation:.6f} | {tv:.6f} | {js:.6f} |")
    lines.extend(
        (
            "",
            "Test labels were used only to describe the sealed split and were not used for prediction, checkpoint selection, tuning, or seed selection.",
            "",
            f"Historical test exposure for train/method selection: `{str(exposure['test_previously_exposed_for_training_or_selection']).lower()}`.",
        )
    )
    (protocol_dir / "protocol_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_protocol_comparison(protocol_dir: Path) -> None:
    lines = [
        "# Protocol Comparison",
        "",
        "| Protocol | Unit | Generalization question | Comparable model scores |",
        "| --- | --- | --- | --- |",
        "| clean-inner 3,600/900 | Local chronological windows | Local temporal interpolation | No |",
        "| Full-pool chronological | Domain tail windows | Tail extrapolation in the same execution | No |",
        "| trajectory-disjoint | Complete resource-coupled trajectory groups | Generalization to unseen complete executions | No |",
        "",
        "The three protocols differ in both unit and data exposure; their scores cannot be interpreted directly as gains from data scale.",
    ]
    (protocol_dir / "protocol_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _distribution_distances(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    left = left / max(float(left.sum()), 1.0)
    right = right / max(float(right.sum()), 1.0)
    correlation = float(np.corrcoef(left, right)[0, 1]) if np.std(left) and np.std(right) else 0.0
    tv = float(0.5 * np.abs(left - right).sum())
    mean = 0.5 * (left + right)
    left_kl = np.where(left > 0, left * np.log2(left / np.clip(mean, 1e-12, None)), 0.0).sum()
    right_kl = np.where(right > 0, right * np.log2(right / np.clip(mean, 1e-12, None)), 0.0).sum()
    return correlation, tv, float(math.sqrt(max(0.0, 0.5 * (left_kl + right_kl))))


def _frame_resource_set(frame: pd.DataFrame, family: str) -> set[str]:
    values: set[str] = set()
    for _, row in frame.iterrows():
        values.update(_row_resources(row, family, str(row["domain_id"])))
    return values


def _row_resources(row: Mapping[str, Any], family: str, domain: str) -> list[str]:
    pattern = RESOURCE_PATTERNS[family]
    values = []
    for column, value in row.items():
        if pattern.fullmatch(str(column)) and (text := _text(value)):
            values.append(_identity(domain, text) if domain else f":{text}")
    return sorted(set(values))


def _target_frames(row: Mapping[str, Any]) -> list[str]:
    domain = str(row["domain_id"])
    return [_identity(domain, value) for value in _json_list(row.get("future_frame_ids_json"))]


def _dependency_frames(row: Mapping[str, Any]) -> list[str]:
    domain = str(row["domain_id"])
    values = [*_json_list(row.get("history_frame_ids_json")), *_json_list(row.get("future_frame_ids_json"))]
    return [_identity(domain, value) for value in values]


def _json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        decoded = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(item) for item in decoded] if isinstance(decoded, list) else []


def _first_text(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    return next((text for field in fields if (text := _text(row.get(field)))), "")


def _text(value: object) -> str:
    text = str(value).strip()
    return "" if not text or text.lower() in {"nan", "none", "-99", "-99.0"} else text


def _identity(domain: str, value: object) -> str:
    text = _text(value)
    return f"{domain}:{text}" if domain and not text.startswith(f"{domain}:") else text


def _sample_id_hash(frame: pd.DataFrame) -> str:
    values = sorted(_identity(str(row.domain_id), row.sample_id) for row in frame.itertuples())
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _anomaly(row: Mapping[str, Any], reason: str, action: str, detail: str) -> dict[str, Any]:
    return {
        "source_global_row": int(row.get("source_global_row", -1)),
        "sample_id": str(row.get("sample_id", "")),
        "domain": str(row.get("domain_id", "")),
        "reason": reason,
        "action": action,
        "detail": detail,
    }


def _fingerprint(payload: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key not in {"generated_at", "protocol_fingerprint"}}
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
    "AUDIT_IDENTITIES",
    "EXPECTED_CANDIDATE_COUNT",
    "TRAJECTORY_PROTOCOL_ID",
    "TRAJECTORY_PROTOCOL_MODE",
    "TRAJECTORY_SPLIT_SEED",
    "assign_trajectory_groups",
    "audit_historical_exposure",
    "audit_trajectory_splits",
    "build_trajectory_protocol",
    "load_trajectory_protocol",
    "protocol_dataset_domains",
    "reconstruct_trajectory_groups",
    "split_group_counts",
    "validate_trajectory_config_protocol",
    "validate_trajectory_protocol",
]
