"""Immutable local artifacts for the MMW post-selection evidence protocol."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from kd_sensing.data.mmw.preparation_splits import partition_sequence_rows
from kd_sensing.data.temporal_missing import DEFAULT_TEMPORAL_MODALITIES, sample_stratified_modality_temporal_mask


PROTOCOL_ID = "mmw_twc_outer_v1"
PROTOCOL_VERSION = 1
SOURCE_SPLIT_TAG = "h5p1_strict_v2"
SOURCE_CSV_NAME = "train_with_radar_with_bs_gps.csv"
DEFAULT_SPLIT_SEED = 20260716
DEFAULT_MASK_SEED = 20260717
DEFAULT_HISTORY_WINDOW = 5
ROLE_FRACTIONS = {
    "inner_train": 0.64,
    "inner_validation": 0.16,
    "outer_evidence": 0.20,
}
WEATHERS = ("sunny", "rainy", "foggy")
SCENES = (
    "Town03_5wayroad_seed28",
    "Town03_Tjunction_wiz_slope_seed42",
    "Town03_crossroad_wiz_slope_seed42",
    "Town03_gastation_seed40",
    "Town03_roundabout_seed42",
)
TEMPORAL_RATES = (0.2, 0.4, 0.6, 0.8)
TEMPORAL_TYPES = ("modality_frame", "frame_level", "block")
TEMPORAL_MASKS_PER_CELL = 8
JOINT_BLOCK_RATES = (0.4, 0.8)


def default_domains(project_root: str | Path) -> list[dict[str, str]]:
    root = Path(project_root)
    result = []
    for weather in WEATHERS:
        for scene in SCENES:
            source = root / "dataset" / "MMW" / weather / "Prepared" / scene / "splits" / SOURCE_SPLIT_TAG / SOURCE_CSV_NAME
            result.append(
                {
                    "id": f"{weather}/{scene}",
                    "condition": weather,
                    "scene": scene,
                    "data_root": f"dataset/MMW/{weather}",
                    "source_csv": str(source.resolve()),
                }
            )
    return result


def prepare_protocol(
    output_root: str | Path,
    *,
    project_root: str | Path,
    split_seed: int = DEFAULT_SPLIT_SEED,
    mask_seed: int = DEFAULT_MASK_SEED,
    excluded_csvs: Iterable[str | Path] = (),
    domains: Iterable[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Create or validate the immutable local protocol manifest.

    ``excluded_csvs`` contains identities that were observed by prior development
    validation.  It is an explicit experiment input rather than a hidden
    dependency on historical outputs.
    """
    root = Path(project_root).resolve()
    target = Path(output_root).resolve()
    domain_items = [dict(item) for item in (domains or default_domains(root))]
    _validate_domains(domain_items)
    exclusion_items = _exclusion_inputs(excluded_csvs)
    request = _request_identity(
        root=root,
        split_seed=split_seed,
        mask_seed=mask_seed,
        domains=domain_items,
        exclusions=exclusion_items,
    )
    request_sha256 = _sha256_payload(request)
    manifest_path = target / "protocol_manifest.json"
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        if existing.get("request_sha256") != request_sha256:
            raise ValueError(
                f"Existing {PROTOCOL_ID} protocol request does not match {manifest_path}; "
                "use a different output root instead of mutating fixed evidence."
            )
        _validate_existing_manifest(existing, manifest_path)
        return existing
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to create {PROTOCOL_ID} in a non-empty directory: {target}")

    exclusions = _load_exclusion_tokens(exclusion_items)
    domain_records: list[dict[str, Any]] = []
    split_dir = target / "splits"
    for domain in domain_items:
        source_csv = Path(domain["source_csv"]).resolve()
        fieldnames, source_rows = _read_csv(source_csv)
        exclusion_tokens = exclusions.get(domain["id"], set())
        routed_rows = _shared_time_axis_rows(source_rows)
        partition = partition_sequence_rows(
            routed_rows,
            seed=int(split_seed),
            role_fractions=ROLE_FRACTIONS,
            seq_len=DEFAULT_HISTORY_WINDOW,
            pred_len=1,
            block_size_frames=7,
            guard_band_frames=5,
            exclude_row=(
                lambda row, tokens=exclusion_tokens, source=source_rows: bool(
                    _identity_tokens(source[int(row["_twc_source_row_index"])]) & tokens
                )
                if exclusion_tokens
                else None
            ),
        )
        if not bool(partition["strict_validation_eligible"]):
            raise ValueError(f"{domain['id']} evidence partition failed: {partition['eligibility_reasons']}")
        partition["role_rows"] = {
            role: [source_rows[int(row["_twc_source_row_index"])] for row in partition["role_rows"][role]]
            for role in ROLE_FRACTIONS
        }
        destination = split_dir / _safe_name(domain["id"])
        paths = {role: destination / f"{role}.csv" for role in ROLE_FRACTIONS}
        for role, path in paths.items():
            _write_csv(path, fieldnames, partition["role_rows"][role])
        domain_records.append(
            {
                "id": domain["id"],
                "condition": domain["condition"],
                "scene": domain["scene"],
                "data_root": domain["data_root"],
                "source_csv": str(source_csv),
                "source_csv_sha256": _sha256_file(source_csv),
                "source_row_count": len(source_rows),
                "exclusion_audit": {
                    "token_count": len(exclusion_tokens),
                    "excluded_group_count": int(partition["excluded_group_count"]),
                    "excluded_window_count": int(partition["excluded_window_count"]),
                    "remaining_row_count": sum(len(partition["role_rows"][role]) for role in ROLE_FRACTIONS),
                },
                "split": {
                    role: {
                        "csv": str(path.resolve()),
                        "sha256": _sha256_file(path),
                        "row_count": len(partition["role_rows"][role]),
                    }
                    for role, path in paths.items()
                },
                "partition": {
                    key: value
                    for key, value in partition.items()
                    if key not in {"role_rows"}
                },
            }
        )
    if len(domain_records) != len(WEATHERS) * len(SCENES):
        raise ValueError(f"Expected 15 MMW domains, got {len(domain_records)}.")

    cache = build_fixed_mask_cache(seed=int(mask_seed))
    cache_path = target / "fixed_mask_cache.json"
    _write_json(cache_path, cache)
    manifest = {
        "schema_version": PROTOCOL_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_kind": "post_selection_confirmation_not_historical_blind_test",
        "source_split_tag": SOURCE_SPLIT_TAG,
        "source_test_policy": "historical_test_excluded_not_reused",
        "selection_exclusion_policy": "explicit_development_validation_identity_exclusion",
        "request": request,
        "request_sha256": request_sha256,
        "split_seed": int(split_seed),
        "mask_seed": int(mask_seed),
        "role_fractions": ROLE_FRACTIONS,
        "history_window": DEFAULT_HISTORY_WINDOW,
        "domains": domain_records,
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


def build_fixed_mask_cache(*, seed: int) -> dict[str, Any]:
    modalities = tuple(DEFAULT_TEMPORAL_MODALITIES)
    conditions: list[dict[str, Any]] = [_whole_condition(modalities)]
    for available_count in (3, 2, 1):
        for available in itertools.combinations(modalities, available_count):
            conditions.append(_whole_condition(modalities, available=available))
    for rate in TEMPORAL_RATES:
        for mask_type in TEMPORAL_TYPES:
            conditions.extend(
                _temporal_conditions(
                    modalities,
                    rate=rate,
                    mask_type=mask_type,
                    count=TEMPORAL_MASKS_PER_CELL,
                    seed=seed,
                )
            )
    for available_count in (3, 2):
        for available in itertools.combinations(modalities, available_count):
            for rate in JOINT_BLOCK_RATES:
                conditions.append(
                    _joint_condition(
                        modalities,
                        available=available,
                        rate=rate,
                        seed=seed,
                    )
                )
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generator": "mmw_twc_fixed_mask_v1",
        "seed": int(seed),
        "history_window": DEFAULT_HISTORY_WINDOW,
        "modalities": list(modalities),
        "conditions": conditions,
    }
    payload["checksum"] = _sha256_payload(payload)
    return payload


def strict_domains_from_manifest(manifest: Mapping[str, Any], *, role: str) -> list[dict[str, str]]:
    if role not in ROLE_FRACTIONS:
        raise ValueError(f"Unsupported MMW TWC role {role!r}.")
    _validate_manifest_shape(manifest)
    result = []
    for item in manifest["domains"]:
        split = item["split"][role]
        result.append(
            {
                "id": str(item["id"]),
                "condition": str(item["condition"]),
                "scene": str(item["scene"]),
                "data_root": str(item["data_root"]),
                "csv": str(split["csv"]),
                "csv_sha256": str(split["sha256"]),
            }
        )
    return result


def build_confirmation_train_domains(
    protocol_manifest: Mapping[str, Any],
    output_root: str | Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Materialize the frozen confirmation-train union without mutating P0 cache."""
    _validate_manifest_shape(protocol_manifest)
    source_fingerprint = str(protocol_manifest.get("manifest_sha256", ""))
    if not source_fingerprint:
        raise ValueError("MMW TWC protocol manifest is missing manifest_sha256.")
    target = Path(output_root).resolve()
    manifest_path = target / "confirmation_train_splits_manifest.json"
    request = {
        "protocol_id": PROTOCOL_ID,
        "protocol_manifest_sha256": source_fingerprint,
        "roles": ["inner_train", "inner_validation"],
        "output_role": "confirmation_train",
    }
    request_sha256 = _sha256_payload(request)
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        if existing.get("request_sha256") != request_sha256:
            raise ValueError("Existing confirmation-train splits do not match the frozen MMW TWC protocol.")
        domains = _validate_confirmation_train_manifest(existing)
        return domains, existing
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to create confirmation-train splits in non-empty directory: {target}")

    records = []
    domains = []
    for item in protocol_manifest["domains"]:
        split = item["split"]
        inner_train = split["inner_train"]
        inner_validation = split["inner_validation"]
        train_path = Path(str(inner_train["csv"]))
        validation_path = Path(str(inner_validation["csv"]))
        if _sha256_file(train_path) != inner_train["sha256"] or _sha256_file(validation_path) != inner_validation["sha256"]:
            raise ValueError(f"Frozen P0 split input changed for {item['id']}.")
        train_fields, train_rows = _read_csv(train_path)
        validation_fields, validation_rows = _read_csv(validation_path)
        if train_fields != validation_fields:
            raise ValueError(f"Inner train/validation CSV columns differ for {item['id']}.")
        destination = target / "confirmation_train_splits" / _safe_name(str(item["id"])) / "train.csv"
        _write_csv(destination, train_fields, [*train_rows, *validation_rows])
        output = {
            "csv": str(destination.resolve()),
            "sha256": _sha256_file(destination),
            "row_count": len(train_rows) + len(validation_rows),
        }
        records.append(
            {
                "id": item["id"],
                "condition": item["condition"],
                "scene": item["scene"],
                "data_root": item["data_root"],
                "inner_train": inner_train,
                "inner_validation": inner_validation,
                "confirmation_train": output,
                "outer_evidence": split["outer_evidence"],
            }
        )
        domains.append(
            {
                "id": str(item["id"]),
                "condition": str(item["condition"]),
                "scene": str(item["scene"]),
                "data_root": str(item["data_root"]),
                "train_csv_name": output["csv"],
                # The final fixed-epoch run observes this only for engine health
                # metrics; it is never used for selection and outer evidence stays
                # completely out of the training process.
                "val_csv_name": str(inner_validation["csv"]),
                "test_csv_name": str(split["outer_evidence"]["csv"]),
                "confirmation_train_csv_sha256": output["sha256"],
                "inner_validation_csv_sha256": str(inner_validation["sha256"]),
                "outer_evidence_csv_sha256": str(split["outer_evidence"]["sha256"]),
            }
        )
    payload = {
        "schema_version": 1,
        "request": request,
        "request_sha256": request_sha256,
        "domains": records,
    }
    payload["manifest_sha256"] = _sha256_payload(payload)
    _write_json(manifest_path, payload)
    return domains, payload


def load_protocol(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    manifest = _read_json(manifest_path)
    _validate_existing_manifest(manifest, manifest_path)
    return manifest


def _whole_condition(modalities: tuple[str, ...], *, available: tuple[str, ...] | None = None) -> dict[str, Any]:
    enabled = tuple(available or modalities)
    matrix = [[name in enabled for name in modalities] for _ in range(DEFAULT_HISTORY_WINDOW)]
    drop_count = len(modalities) - len(enabled)
    pattern = "clean" if drop_count == 0 else "drop_" + "_".join(name for name in modalities if name not in enabled)
    return _condition(
        family="whole_modality",
        pattern=pattern,
        mask_type="whole_modality",
        rate=0.0,
        available=enabled,
        matrix=matrix,
    )


def _temporal_conditions(
    modalities: tuple[str, ...],
    *,
    rate: float,
    mask_type: str,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempt = 0
    while len(result) < int(count):
        local_seed = _derived_seed(seed, "temporal", rate, mask_type, attempt)
        sampled = sample_stratified_modality_temporal_mask(
            history_window=DEFAULT_HISTORY_WINDOW,
            modalities=modalities,
            fixed_rate=rate,
            fixed_mask_type=mask_type,
            rng=random.Random(local_seed),
        )
        matrix = sampled["modality_temporal_mask"].tolist()
        condition = _condition(
            family="temporal_missing",
            pattern=f"{mask_type}_{int(round(rate * 100)):02d}_{len(result):02d}",
            mask_type=mask_type,
            rate=rate,
            available=modalities,
            matrix=matrix,
        )
        digest = str(condition["mask_digest"])
        if digest not in seen:
            seen.add(digest)
            result.append(condition)
        attempt += 1
        if attempt > max(256, int(count) * 64):
            raise RuntimeError(f"Could not generate {count} unique {mask_type} masks at rate={rate}.")
    return result


def _joint_condition(
    modalities: tuple[str, ...],
    *,
    available: tuple[str, ...],
    rate: float,
    seed: int,
) -> dict[str, Any]:
    dropped = tuple(name for name in modalities if name not in available)
    sampled = sample_stratified_modality_temporal_mask(
        history_window=DEFAULT_HISTORY_WINDOW,
        modalities=modalities,
        fixed_drop_modalities=dropped,
        fixed_rate=rate,
        fixed_mask_type="block",
        rng=random.Random(_derived_seed(seed, "joint", available, rate)),
    )
    return _condition(
        family="joint_missing",
        pattern=f"drop{len(dropped)}_block{int(round(rate * 100)):02d}_" + "_".join(dropped),
        mask_type="block",
        rate=rate,
        available=available,
        matrix=sampled["modality_temporal_mask"].tolist(),
    )


def _condition(
    *,
    family: str,
    pattern: str,
    mask_type: str,
    rate: float,
    available: tuple[str, ...],
    matrix: list[list[bool]],
) -> dict[str, Any]:
    modalities = tuple(DEFAULT_TEMPORAL_MODALITIES)
    canonical = {"modalities": list(modalities), "modality_temporal_mask": matrix}
    digest = _sha256_payload(canonical)
    observed_missing_rate = sum(not bool(value) for row in matrix for value in row) / float(len(matrix) * len(modalities))
    return {
        "family": family,
        "pattern": pattern,
        "mask_type": mask_type,
        "requested_missing_rate": float(rate),
        "available_modalities": list(available),
        "drop_count": len(modalities) - len(available),
        "modality_temporal_mask": matrix,
        "mask_digest": digest,
        "observed_missing_rate": observed_missing_rate,
    }


def _request_identity(
    *,
    root: Path,
    split_seed: int,
    mask_seed: int,
    domains: list[dict[str, str]],
    exclusions: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "schema_version": PROTOCOL_VERSION,
        "project_root": str(root),
        "source_split_tag": SOURCE_SPLIT_TAG,
        "split_seed": int(split_seed),
        "mask_seed": int(mask_seed),
        "role_fractions": ROLE_FRACTIONS,
        "domains": [
            {
                "id": item["id"],
                "source_csv": item["source_csv"],
                "source_csv_sha256": _sha256_file(item["source_csv"]),
            }
            for item in domains
        ],
        "exclusions": exclusions,
    }


def _exclusion_inputs(paths: Iterable[str | Path]) -> list[dict[str, str]]:
    result = []
    for raw in paths:
        path = Path(raw).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Development exclusion CSV is missing: {path}")
        result.append({"path": str(path), "sha256": _sha256_file(path)})
    return sorted(result, key=lambda item: item["path"])


def _load_exclusion_tokens(items: list[dict[str, str]]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for item in items:
        _, rows = _read_csv(Path(item["path"]))
        for row in rows:
            domain_id = _row_domain_id(row)
            if domain_id:
                grouped[domain_id].update(_identity_tokens(row))
    return grouped


def _exclude_observed_rows(rows: list[dict[str, str]], excluded_tokens: set[str]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not excluded_tokens:
        return list(rows), {"excluded_row_count": 0, "remaining_row_count": len(rows), "token_count": 0}
    filtered = [row for row in rows if not (_identity_tokens(row) & excluded_tokens)]
    return filtered, {
        "excluded_row_count": len(rows) - len(filtered),
        "remaining_row_count": len(filtered),
        "token_count": len(excluded_tokens),
    }


def _shared_time_axis_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Group CAV windows by shared simulator time so RSU resources cannot cross roles."""
    routed = []
    for index, row in enumerate(rows):
        item = dict(row)
        item["_twc_source_row_index"] = str(index)
        item["contiguous_segment_id"] = "__mmw_twc_shared_time_axis__"
        routed.append(item)
    return routed


def _identity_tokens(row: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in ("sample_id", "target_sample_id"):
        value = str(row.get(key, "")).strip()
        if value:
            tokens.add(f"{key}:{value}")
    segment = str(row.get("contiguous_segment_id", "")).strip()
    try:
        frames = json.loads(str(row.get("window_frame_ids_json", "[]")))
    except json.JSONDecodeError:
        frames = []
    for frame in frames:
        tokens.add(f"frame:{segment}:{frame}")
    prefixes = ("camera", "radar", "gps", "bs_gps", "lidar", "beam", "future_beam")
    for key, value in row.items():
        if any(key.startswith(prefix) and key[len(prefix) :].isdigit() for prefix in prefixes):
            text = str(value).strip()
            if text and text != "-99":
                tokens.add(f"resource:{text}")
    return tokens


def _row_domain_id(row: Mapping[str, Any]) -> str:
    condition = str(row.get("condition", "")).strip()
    scene = str(row.get("sensor_scenario", row.get("scene_slug", ""))).strip()
    return f"{condition}/{scene}" if condition and scene else ""


def _validate_domains(domains: list[dict[str, str]]) -> None:
    ids = [str(item.get("id", "")) for item in domains]
    if len(domains) != 15 or len(set(ids)) != 15:
        raise ValueError("MMW TWC evidence requires exactly 15 unique weather/scene domains.")
    for item in domains:
        missing = [key for key in ("id", "condition", "scene", "data_root", "source_csv") if not item.get(key)]
        if missing:
            raise ValueError(f"MMW TWC domain is missing {missing}: {item}")
        if not Path(item["source_csv"]).is_file():
            raise FileNotFoundError(f"MMW TWC source CSV is missing: {item['source_csv']}")


def _validate_existing_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    _validate_manifest_shape(manifest)
    recorded = str(manifest.get("manifest_sha256", ""))
    without_hash = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if recorded != _sha256_payload(without_hash):
        raise ValueError(f"MMW TWC manifest checksum mismatch: {path}")
    for item in manifest["domains"]:
        source = Path(str(item["source_csv"]))
        if not source.is_file() or _sha256_file(source) != item["source_csv_sha256"]:
            raise ValueError(f"MMW TWC source CSV changed: {source}")
        for role in ROLE_FRACTIONS:
            split = item["split"][role]
            csv_path = Path(str(split["csv"]))
            if not csv_path.is_file() or _sha256_file(csv_path) != split["sha256"]:
                raise ValueError(f"MMW TWC split artifact changed: {csv_path}")
    cache = manifest["fixed_mask_cache"]
    cache_path = Path(str(cache["path"]))
    if not cache_path.is_file() or _sha256_file(cache_path) != cache["sha256"]:
        raise ValueError(f"MMW TWC fixed-mask cache changed: {cache_path}")
    payload = _read_json(cache_path)
    checksum = str(payload.get("checksum", ""))
    if checksum != str(cache["cache_checksum"]) or checksum != _sha256_payload({key: value for key, value in payload.items() if key != "checksum"}):
        raise ValueError(f"MMW TWC fixed-mask cache checksum mismatch: {cache_path}")


def _validate_manifest_shape(manifest: Mapping[str, Any]) -> None:
    if manifest.get("protocol_id") != PROTOCOL_ID or int(manifest.get("schema_version", -1)) != PROTOCOL_VERSION:
        raise ValueError("Unsupported MMW TWC evidence protocol manifest.")
    domains = manifest.get("domains")
    if not isinstance(domains, list) or len(domains) != 15:
        raise ValueError("MMW TWC manifest requires exactly 15 domains.")


def _validate_confirmation_train_manifest(manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    recorded = str(manifest.get("manifest_sha256", ""))
    without_hash = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if not recorded or recorded != _sha256_payload(without_hash):
        raise ValueError("MMW confirmation-train manifest checksum mismatch.")
    records = manifest.get("domains")
    if not isinstance(records, list) or len(records) != 15:
        raise ValueError("MMW confirmation-train manifest requires exactly 15 domains.")
    result = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("MMW confirmation-train domain record must be a mapping.")
        output = record.get("confirmation_train")
        outer = record.get("outer_evidence")
        if not isinstance(output, Mapping) or not isinstance(outer, Mapping):
            raise ValueError("MMW confirmation-train record is missing output or outer evidence identity.")
        train_path = Path(str(output.get("csv", "")))
        if not train_path.is_file() or _sha256_file(train_path) != output.get("sha256"):
            raise ValueError(f"MMW confirmation-train CSV changed: {train_path}")
        outer_path = Path(str(outer.get("csv", "")))
        if not outer_path.is_file() or _sha256_file(outer_path) != outer.get("sha256"):
            raise ValueError(f"MMW confirmation outer-evidence CSV changed: {outer_path}")
        result.append(
            {
                "id": str(record["id"]),
                "condition": str(record["condition"]),
                "scene": str(record["scene"]),
                "data_root": str(record["data_root"]),
                "train_csv_name": str(output["csv"]),
                "val_csv_name": str(record["inner_validation"]["csv"]),
                "test_csv_name": str(outer["csv"]),
                "confirmation_train_csv_sha256": str(output["sha256"]),
                "inner_validation_csv_sha256": str(record["inner_validation"]["sha256"]),
                "outer_evidence_csv_sha256": str(outer["sha256"]),
            }
        )
    return result


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    if not fieldnames or not rows:
        raise ValueError(f"MMW evidence source CSV is empty or lacks a header: {path}")
    return fieldnames, rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be a mapping: {path}")
    return payload


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _derived_seed(seed: int, *parts: object) -> int:
    text = ":".join((str(int(seed)), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def _safe_name(value: str) -> str:
    return str(value).replace("/", "__").replace(" ", "_")


__all__ = [
    "DEFAULT_MASK_SEED",
    "DEFAULT_SPLIT_SEED",
    "PROTOCOL_ID",
    "ROLE_FRACTIONS",
    "build_fixed_mask_cache",
    "build_confirmation_train_domains",
    "default_domains",
    "load_protocol",
    "prepare_protocol",
    "strict_domains_from_manifest",
]
