"""Fail-closed Full-pool Town3 development protocol construction and validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from kd_sensing.preprocessing.mmw_radar import augment_mmw_sequence_resource_columns


FULL_POOL_PROTOCOL_MODE = "full_pool_development"
FULL_POOL_PROTOCOL_ID = "mmw_full_pool_development_v1"
EXPECTED_CANDIDATE_COUNT = 46_860
EXPECTED_DOMAIN_COUNT = 15
EXPECTED_HISTORICAL_EXCLUSION_COUNT = 588

# Locked counts of the canonical `full_pool_contiguous_time_v1` macro split.
# Every local experiment workflow re-asserts these before training, so they are
# published here as the single source of truth instead of being restated per
# tool.  Changing the protocol MUST change these together with its fingerprint.
FULL_POOL_SPLIT_EXPECTATIONS: dict[str, int] = {
    "candidate_window_count": EXPECTED_CANDIDATE_COUNT,
    "boundary_crossing_excluded_count": 240,
    "historical_removed_from_train_count": 402,
    "historical_protected_count": EXPECTED_HISTORICAL_EXCLUSION_COUNT,
    "train_sample_count": 37_038,
    "validation_sample_count": 9_180,
}
# Derived quantities the audits report alongside the locked counts.
FULL_POOL_RAW_TRAIN_COUNT = (
    FULL_POOL_SPLIT_EXPECTATIONS["train_sample_count"]
    + FULL_POOL_SPLIT_EXPECTATIONS["historical_removed_from_train_count"]
)
FULL_POOL_HISTORICAL_VALIDATION_RETAINED = (
    FULL_POOL_SPLIT_EXPECTATIONS["historical_protected_count"]
    - FULL_POOL_SPLIT_EXPECTATIONS["historical_removed_from_train_count"]
)
FULL_POOL_DEVELOPMENT_WINDOWS = (
    FULL_POOL_SPLIT_EXPECTATIONS["train_sample_count"]
    + FULL_POOL_SPLIT_EXPECTATIONS["validation_sample_count"]
)
# Identity families that MUST have zero train/validation intersection.
FULL_POOL_RESOURCE_INTERSECTION_NAMES: tuple[str, ...] = (
    "sample_id",
    "target_sample_id",
    "full_csv_row",
    "all_frame_dependency",
    "camera_resource",
    "lidar_resource",
    "radar_resource",
    "ue_gps_resource",
    "bs_gps_resource",
    "channel_resource",
)
_RESOURCE_PATTERNS = {
    "camera": re.compile(r"^camera\d+$"),
    "lidar": re.compile(r"^lidar\d+$"),
    "radar": re.compile(r"^radar\d+$"),
    "ue_gps": re.compile(r"^gps\d+$"),
    "bs_gps": re.compile(r"^bs_gps\d+$"),
    "channel": re.compile(r"^(?:csi|beam|mmwave|future_csi|future_beam|future_path)\d+$"),
}


def build_full_pool_protocol(
    output_root: str | Path,
    *,
    dataset_root: str | Path = "dataset/MMW",
    historical_manifest: str | Path = "outputs/cache/mmw_twc_outer_v1/protocol_manifest.json",
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    protocol_dir = root / "protocol"
    cache_dir = root / "cache"
    manifest_path = protocol_dir / "split_manifest.json"
    if manifest_path.exists():
        return load_full_pool_protocol(manifest_path)
    protocol_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(dataset_root).resolve()
    historical_path = Path(historical_manifest).resolve()
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    historical_rows = _recover_historical_exclusions(historical)
    if sum(len(value) for value in historical_rows.values()) != EXPECTED_HISTORICAL_EXCLUSION_COUNT:
        raise ValueError("Historical development exclusion recovery did not produce exactly 588 identities.")

    sources = sorted(dataset_root.glob("*/Prepared/*/splits/h5p1_strict_v2/all_sequences.csv"))
    if len(sources) != EXPECTED_DOMAIN_COUNT:
        raise ValueError(f"Full-pool protocol requires 15 domain sources, found {len(sources)}.")

    created_at = datetime.now(timezone.utc).isoformat()
    all_train: list[pd.DataFrame] = []
    all_validation: list[pd.DataFrame] = []
    exclusion_records: list[dict[str, Any]] = []
    domain_records: list[dict[str, Any]] = []
    total_candidates = total_boundary = total_invalid = total_removed_historical = 0
    invalid_reasons: Counter[str] = Counter()

    for source in sources:
        condition = source.parents[4].name
        scene = source.parents[2].name
        domain_id = f"{condition}/{scene}"
        data_root = dataset_root / condition
        frame = pd.read_csv(source, na_values="").fillna("")
        total_candidates += len(frame)
        source_hash = _sha256_file(source)
        original_columns = list(frame.columns)
        frame = augment_mmw_sequence_resource_columns(frame, scene)
        frame.insert(0, "domain_id", domain_id)
        roles, split_info, invalid = _macro_split(frame)
        total_boundary += split_info["boundary_crossing_count"]
        total_invalid += len(invalid)
        invalid_reasons.update(item["reason"] for item in invalid)

        protected = historical_rows.get(domain_id, {})
        train_indices: list[int] = []
        validation_indices: list[int] = []
        for index, role in roles.items():
            row = frame.loc[index]
            identity = _row_identity(row)
            historical_record = protected.get(identity)
            if historical_record is not None:
                disposition = "removed_from_train" if role == "train" else f"retained_{role}"
                exclusion_records.append(
                    {
                        "domain_id": domain_id,
                        "sample_id": str(row.get("sample_id", "")),
                        "target_sample_id": str(row.get("target_sample_id", "")),
                        "window_start_frame": str(row.get("window_start_frame", "")),
                        "future_end_frame": str(row.get("future_end_frame", "")),
                        "historical_group_id": historical_record["historical_group_id"],
                        "disposition": disposition,
                    }
                )
                if role == "train":
                    total_removed_historical += 1
                    continue
            if role == "train":
                train_indices.append(index)
            elif role == "validation":
                validation_indices.append(index)

        recovered = {item["sample_id"] for item in exclusion_records if item["domain_id"] == domain_id}
        expected = {value["sample_id"] for value in protected.values()}
        if recovered != expected:
            raise ValueError(f"Historical exclusion identities do not map exactly into all_sequences.csv for {domain_id}.")

        train = frame.loc[train_indices].reset_index(drop=True)
        validation = frame.loc[validation_indices].reset_index(drop=True)
        _validate_augmented_resources(data_root, train, validation)
        slug = domain_id.replace("/", "__")
        domain_dir = cache_dir / "splits" / slug
        train_path = domain_dir / "full_pool_train.csv"
        validation_path = domain_dir / "full_pool_inner_validation.csv"
        _write_csv(train, train_path)
        _write_csv(validation, validation_path)
        source_hashes = {
            "source_csv_sha256": source_hash,
            "split_manifest_seed_material": _sha256_json(
                {"domain_id": domain_id, "boundary_frame": split_info["boundary_frame"], "ratio": 0.8}
            ),
            "augmentation_code_sha256": _augmentation_code_hash(),
        }
        cache_metadata = {
            "schema_version": 1,
            "generated_at": created_at,
            "domain_id": domain_id,
            **source_hashes,
            "train_csv": str(train_path),
            "train_csv_sha256": _sha256_file(train_path),
            "validation_csv": str(validation_path),
            "validation_csv_sha256": _sha256_file(validation_path),
        }
        _write_json(domain_dir / "cache_metadata.json", cache_metadata)
        domain_records.append(
            {
                "id": domain_id,
                "condition": condition,
                "scene": scene,
                "data_root": str(data_root),
                "source_csv": str(source),
                **source_hashes,
                "candidate_count": int(len(frame)),
                "train_split": str(train_path),
                "validation_split": str(validation_path),
                "train_csv_sha256": cache_metadata["train_csv_sha256"],
                "validation_csv_sha256": cache_metadata["validation_csv_sha256"],
                "train_sample_count": int(len(train)),
                "validation_sample_count": int(len(validation)),
                "historical_protected_count": len(protected),
                "historical_removed_from_train_count": sum(
                    item["domain_id"] == domain_id and item["disposition"] == "removed_from_train"
                    for item in exclusion_records
                ),
                "beam_support": {
                    "train": _beam_support(train),
                    "validation": _beam_support(validation),
                },
                "split": split_info,
                "source_column_count": len(original_columns),
                "augmented_column_count": len(frame.columns),
            }
        )
        all_train.append(train)
        all_validation.append(validation)

    if total_candidates != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(f"Full-pool candidate count must be 46,860, got {total_candidates}.")
    if len(exclusion_records) != EXPECTED_HISTORICAL_EXCLUSION_COUNT:
        raise ValueError("Not all 588 historical identities were mapped to the candidate pool.")

    global_train = pd.concat(all_train, ignore_index=True)
    global_validation = pd.concat(all_validation, ignore_index=True)
    train_path = protocol_dir / "full_pool_train.csv"
    validation_path = protocol_dir / "full_pool_inner_validation.csv"
    exclusion_path = protocol_dir / "excluded_historical_identities.csv"
    _write_csv(global_train, train_path)
    _write_csv(global_validation, validation_path)
    _write_csv(pd.DataFrame(exclusion_records), exclusion_path)

    payload = {
        "schema_version": 1,
        "mode": FULL_POOL_PROTOCOL_MODE,
        "protocol_id": FULL_POOL_PROTOCOL_ID,
        "split_strategy": "domain_wise_contiguous_macro_split",
        "trajectory_split_rejected_reason": "per-CAV segments share domain-level RSU Radar and BS-GPS frame resources",
        "train_fraction_target": 0.8,
        "train_role": "full_pool_train",
        "validation_role": "full_pool_inner_validation",
        "outer_test_enabled": False,
        "allow_confirmation_train": False,
        "outer_test_accessed": False,
        "generated_at": created_at,
        "historical_manifest": str(historical_path),
        "historical_manifest_sha256": _sha256_file(historical_path),
        "historical_identity_recovery": "source manifest group_assignments with role=excluded_development; no outer split CSV read",
        "candidate_window_count": total_candidates,
        "historical_protected_count": len(exclusion_records),
        "historical_removed_from_train_count": total_removed_historical,
        "boundary_crossing_excluded_count": total_boundary,
        "invalid_row_count": total_invalid,
        "invalid_row_reasons": dict(sorted(invalid_reasons.items())),
        "train_sample_count": int(len(global_train)),
        "validation_sample_count": int(len(global_validation)),
        "full_pool_train_csv": str(train_path),
        "full_pool_validation_csv": str(validation_path),
        "excluded_historical_identities_csv": str(exclusion_path),
        "domains": domain_records,
    }
    payload["protocol_fingerprint"] = _protocol_fingerprint(payload)
    _write_json(manifest_path, payload)

    audit = audit_full_pool_protocol(payload)
    audit_path = protocol_dir / "split_audit.json"
    _write_json(audit_path, audit)
    if audit["status"] != "passed":
        raise ValueError(f"Full-pool resource audit failed: {audit['reasons']}")
    hash_payload = {
        "schema_version": 1,
        "generated_at": created_at,
        "protocol_fingerprint": payload["protocol_fingerprint"],
        "files": {
            path.name: _sha256_file(path)
            for path in (manifest_path, audit_path, exclusion_path, train_path, validation_path)
        },
        "augmentation_code_sha256": _augmentation_code_hash(),
    }
    _write_json(protocol_dir / "protocol_hash.json", hash_payload)
    return load_full_pool_protocol(manifest_path)


def load_full_pool_protocol(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_full_pool_protocol(payload)


def validate_full_pool_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(protocol)
    if payload.get("mode") != FULL_POOL_PROTOCOL_MODE or payload.get("protocol_id") != FULL_POOL_PROTOCOL_ID:
        raise ValueError("Invalid Full-pool protocol mode or id.")
    if payload.get("outer_test_enabled") is not False or payload.get("outer_test_accessed") is not False:
        raise ValueError("Full-pool protocol must keep outer test disabled and unaccessed.")
    if payload.get("allow_confirmation_train") is not False:
        raise ValueError("Full-pool protocol must disallow confirmation training.")
    if int(payload.get("candidate_window_count", -1)) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("Full-pool protocol candidate count mismatch.")
    if int(payload.get("historical_protected_count", -1)) != EXPECTED_HISTORICAL_EXCLUSION_COUNT:
        raise ValueError("Full-pool protocol historical exclusion count mismatch.")
    domains = payload.get("domains")
    if not isinstance(domains, list) or len(domains) != EXPECTED_DOMAIN_COUNT:
        raise ValueError("Full-pool protocol must contain 15 domains.")
    for domain in domains:
        for role in ("train", "validation"):
            path = Path(str(domain[f"{role}_split"]))
            if not path.is_file() or _sha256_file(path) != domain[f"{role}_csv_sha256"]:
                raise ValueError(f"Full-pool {domain['id']} {role} split is missing or has a SHA256 mismatch.")
    expected = str(payload.pop("protocol_fingerprint", ""))
    actual = _protocol_fingerprint(payload)
    if expected != actual:
        raise ValueError("Full-pool protocol fingerprint mismatch.")
    payload["protocol_fingerprint"] = actual
    return payload


def validate_full_pool_config_protocol(cfg: Mapping[str, Any]) -> dict[str, Any]:
    section = cfg.get("data_protocol")
    if not isinstance(section, Mapping) or section.get("mode") != FULL_POOL_PROTOCOL_MODE:
        raise ValueError("data_protocol must declare mode=full_pool_development.")
    path = Path(str(section.get("path", ""))).resolve()
    protocol = load_full_pool_protocol(path)
    if section.get("protocol_id") != protocol["protocol_id"] or section.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
        raise ValueError("MMW Full-pool config must bind the exact protocol identity.")
    if section.get("train_role") != protocol["train_role"] or section.get("validation_role") != protocol["validation_role"]:
        raise ValueError("MMW Full-pool config role binding mismatch.")
    if section.get("outer_test_enabled") is not False or section.get("allow_confirmation_train") is not False:
        raise ValueError("MMW Full-pool config must disable outer test and confirmation training.")
    expected_domains = protocol_dataset_domains(protocol)
    actual_domains = cfg.get("data", {}).get("dataset", {}).get("domains")
    if actual_domains != expected_domains:
        raise ValueError("Resolved dataset domains do not exactly match the Full-pool protocol.")
    if any(domain.get("test_csv_name") for domain in actual_domains):
        raise ValueError("Full-pool config must not carry outer/test CSV paths.")
    final_test = cfg.get("training", {}).get("final_test")
    enabled = final_test if isinstance(final_test, bool) else (final_test or {}).get("enabled", True)
    if bool(enabled):
        raise ValueError("Full-pool config must explicitly disable final test.")
    report_path = Path(str(section.get("audit_report", ""))).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("outer_test_accessed") is not False:
        raise ValueError("Full-pool audit report is missing or failed.")
    if report.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
        raise ValueError("Full-pool audit report protocol fingerprint mismatch.")
    return report


def protocol_dataset_domains(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": domain["id"],
            "condition": domain["condition"],
            "scene": domain["scene"],
            "data_root": domain["data_root"],
            "train_csv_name": domain["train_split"],
            "val_csv_name": domain["validation_split"],
        }
        for domain in protocol["domains"]
    ]


def audit_full_pool_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    train = pd.concat([pd.read_csv(item["train_split"], na_values="").fillna("") for item in protocol["domains"]])
    validation = pd.concat([pd.read_csv(item["validation_split"], na_values="").fillna("") for item in protocol["domains"]])
    train_sets = _audit_identity_sets(train)
    validation_sets = _audit_identity_sets(validation)
    overlaps = {
        name: _overlap(train_sets[name], validation_sets[name])
        for name in train_sets
        if name != "trajectory_session"
    }
    trajectory = _overlap(train_sets["trajectory_session"], validation_sets["trajectory_session"])
    reasons = [f"{name}_overlap" for name, value in overlaps.items() if value["count"]]
    result = {
        "schema_version": 1,
        "audit_id": "mmw_full_pool_resource_isolation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "outer_test_accessed": False,
        "split_strategy": protocol["split_strategy"],
        "trajectory_overlap_required_zero": False,
        "trajectory_session_overlap": trajectory,
        "train_sample_count": int(len(train)),
        "validation_sample_count": int(len(validation)),
        "train_sample_id_hash": _sha256_lines(train_sets["sample_id"]),
        "validation_sample_id_hash": _sha256_lines(validation_sets["sample_id"]),
        "overlaps": overlaps,
        "overlap_counts": {name: value["count"] for name, value in overlaps.items()},
        "beam_support": {"train": _beam_support(train), "validation": _beam_support(validation)},
        "domains": [
            {
                "id": item["id"],
                "train_sample_count": item["train_sample_count"],
                "validation_sample_count": item["validation_sample_count"],
                "beam_support": item["beam_support"],
                "boundary_frame": item["split"]["boundary_frame"],
                "boundary_crossing_count": item["split"]["boundary_crossing_count"],
            }
            for item in protocol["domains"]
        ],
    }
    return result


def _recover_historical_exclusions(manifest: Mapping[str, Any]) -> dict[str, dict[tuple[str, str], dict[str, str]]]:
    recovered: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    for domain in manifest.get("domains", []):
        source = Path(str(domain["source_csv"]))
        if _sha256_file(source) != domain["source_csv_sha256"]:
            raise ValueError(f"Historical exclusion source SHA256 mismatch: {domain['id']}.")
        frame = pd.read_csv(source, na_values="").fillna("")
        blocks = [item for item in domain["partition"]["group_assignments"] if item.get("role") == "excluded_development"]
        values: dict[tuple[str, str], dict[str, str]] = {}
        for _, row in frame.iterrows():
            lo, hi = int(row["window_start_frame"]), int(row["future_end_frame"])
            block = next(
                (
                    item
                    for item in blocks
                    if lo >= int(item["block_start_frame"]) and hi <= int(item["block_end_frame"])
                ),
                None,
            )
            if block is not None:
                identity = _row_identity(row)
                values[identity] = {
                    "sample_id": identity[0],
                    "target_sample_id": identity[1],
                    "historical_group_id": str(block["group_id"]),
                }
        expected = int(domain["exclusion_audit"]["excluded_window_count"])
        if len(values) != expected:
            raise ValueError(f"Historical exclusion count mismatch for {domain['id']}: {len(values)} != {expected}.")
        recovered[str(domain["id"])] = values
    return recovered


def _macro_split(frame: pd.DataFrame) -> tuple[dict[int, str], dict[str, Any], list[dict[str, Any]]]:
    dependencies: dict[int, list[int]] = {}
    invalid: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        try:
            values = _frame_dependencies(row)
            if not str(row.get("sample_id", "")).strip() or not str(row.get("target_sample_id", "")).strip():
                raise ValueError("missing_sample_or_target_id")
            dependencies[index] = values
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            invalid.append({"row_index": int(index), "reason": str(exc) or type(exc).__name__})
    unique_frames = sorted({value for values in dependencies.values() for value in values})
    if len(unique_frames) < 6:
        raise ValueError("A Full-pool domain has too few valid dependency frames.")
    boundary_index = max(0, min(len(unique_frames) - 2, math.ceil(len(unique_frames) * 0.8) - 1))
    boundary = unique_frames[boundary_index]
    roles: dict[int, str] = {}
    boundary_rows = 0
    for index, values in dependencies.items():
        if max(values) <= boundary:
            roles[index] = "train"
        elif min(values) > boundary:
            roles[index] = "validation"
        else:
            roles[index] = "boundary_purged"
            boundary_rows += 1
    return roles, {
        "strategy": "shared_time_axis_single_boundary",
        "target_train_fraction": 0.8,
        "first_dependency_frame": unique_frames[0],
        "last_dependency_frame": unique_frames[-1],
        "unique_dependency_frame_count": len(unique_frames),
        "boundary_frame": boundary,
        "boundary_crossing_count": boundary_rows,
        "raw_train_count": sum(value == "train" for value in roles.values()),
        "raw_validation_count": sum(value == "validation" for value in roles.values()),
    }, invalid


def _frame_dependencies(row: Mapping[str, Any]) -> list[int]:
    history = json.loads(str(row.get("history_frame_ids_json", "[]")))
    future = json.loads(str(row.get("future_frame_ids_json", "[]")))
    values = [int(value) for value in [*history, *future]]
    if len(history) != 5 or len(future) != 1 or len(values) != 6:
        raise ValueError("invalid_history5_prediction1_dependency")
    return values


def _validate_augmented_resources(data_root: Path, *frames: pd.DataFrame) -> None:
    missing: list[dict[str, str]] = []
    checked: set[tuple[str, str]] = set()
    for frame in frames:
        for family, pattern in _RESOURCE_PATTERNS.items():
            columns = [str(column) for column in frame.columns if pattern.fullmatch(str(column))]
            if not columns:
                raise ValueError(f"Augmented Full-pool CSV lacks required {family} resource columns.")
            for column in columns:
                for value in frame[column].astype(str).unique():
                    text = value.strip()
                    if not text or text in {"-99", "-99.0"}:
                        missing.append({"family": family, "column": column, "path": text, "reason": "empty"})
                        continue
                    key = (family, text)
                    if key in checked:
                        continue
                    checked.add(key)
                    path = data_root / text
                    if not path.is_file():
                        missing.append({"family": family, "column": column, "path": str(path), "reason": "missing"})
                    if family == "radar":
                        da = Path(str(path).replace("_RA", "_DA"))
                        if "_RA" not in str(path) or not da.is_file():
                            missing.append({"family": family, "column": column, "path": str(da), "reason": "missing_DA"})
    if missing:
        raise FileNotFoundError(f"Full-pool augmentation has {len(missing)} missing resources; examples={missing[:20]}")


def _audit_identity_sets(frame: pd.DataFrame) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    text_frame = frame.astype(str)
    domains = text_frame["domain_id"]
    result["sample_id"].update((domains + ":" + text_frame["sample_id"]).tolist())
    result["target_sample_id"].update((domains + ":" + text_frame["target_sample_id"]).tolist())
    segments = text_frame["contiguous_segment_id"].str.strip()
    has_segment = segments.ne("")
    result["trajectory_session"].update((domains[has_segment] + ":" + segments[has_segment]).tolist())

    column_names = list(frame.columns)
    positions = {str(column): index for index, column in enumerate(column_names)}
    for values in text_frame.itertuples(index=False, name=None):
        domain = values[positions["domain_id"]]
        result["full_csv_row"].add(_sha256_json(dict(zip(column_names, values))))
        history = json.loads(values[positions["history_frame_ids_json"]])
        future = json.loads(values[positions["future_frame_ids_json"]])
        result["window_frame"].update(f"{domain}:{value}" for value in history)
        result["target_frame"].update(f"{domain}:{value}" for value in future)
        result["all_frame_dependency"].update(f"{domain}:{value}" for value in [*history, *future])

    for family, pattern in _RESOURCE_PATTERNS.items():
        for column in (column for column in frame.columns if pattern.fullmatch(str(column))):
            texts = text_frame[column].str.strip()
            valid = texts.ne("") & ~texts.isin({"-99", "-99.0"})
            result[f"{family}_resource"].update((domains[valid] + ":" + texts[valid]).tolist())
    return dict(result)


def _beam_support(frame: pd.DataFrame) -> dict[str, int]:
    counts = Counter(int(value) for value in frame["future_beam_label1"].tolist())
    return {str(index): int(counts.get(index, 0)) for index in range(64)}


def _row_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("sample_id", "")).strip(), str(row.get("target_sample_id", "")).strip()


def _overlap(left: set[str], right: set[str]) -> dict[str, Any]:
    values = sorted(left & right)
    return {"count": len(values), "examples": values[:20]}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256_lines(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def _protocol_fingerprint(payload: Mapping[str, Any]) -> str:
    return _sha256_json({key: value for key, value in payload.items() if key != "protocol_fingerprint"})


def _augmentation_code_hash() -> str:
    module_path = Path(__file__).resolve()
    radar_path = module_path.parents[2] / "preprocessing" / "mmw_radar.py"
    return _sha256_json({"full_pool_protocol.py": _sha256_file(module_path), "mmw_radar.py": _sha256_file(radar_path)})


__all__ = [
    "FULL_POOL_DEVELOPMENT_WINDOWS",
    "FULL_POOL_HISTORICAL_VALIDATION_RETAINED",
    "FULL_POOL_PROTOCOL_ID",
    "FULL_POOL_PROTOCOL_MODE",
    "FULL_POOL_RAW_TRAIN_COUNT",
    "FULL_POOL_RESOURCE_INTERSECTION_NAMES",
    "FULL_POOL_SPLIT_EXPECTATIONS",
    "audit_full_pool_protocol",
    "build_full_pool_protocol",
    "load_full_pool_protocol",
    "protocol_dataset_domains",
    "validate_full_pool_config_protocol",
    "validate_full_pool_protocol",
]
