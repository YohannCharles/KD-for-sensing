"""Fail-closed clean inner-development protocol and split audit."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


CLEAN_PROTOCOL_MODE = "clean_inner_development"
CLEAN_PROTOCOL_ID = "mmw_clean_inner_development_v1"
FORBIDDEN_TRAIN_PATH_TOKENS = (
    "confirmation_train_splits",
    "train_plus_validation",
    "merged_train_validation",
    "trainval",
)
_RESOURCE_PATTERNS = {
    "image": re.compile(r"^camera\d+$"),
    "lidar": re.compile(r"^lidar\d+$"),
    "radar": re.compile(r"^radar\d+$"),
    "gps": re.compile(r"^(?:gps|bs_gps)\d+$"),
}
_REQUIRED_ZERO_OVERLAPS = (
    "sample_id",
    "target_sample_id",
    "full_csv_row",
    "raw_input_resource",
    "window_frame",
    "target_frame",
)


def build_clean_inner_protocol(source_manifest: str | Path) -> dict[str, Any]:
    """Build the one local protocol consumed by every clean experiment."""
    manifest_path = Path(source_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    project_root = Path(str(manifest.get("request", {}).get("project_root") or Path.cwd())).resolve()
    domains = []
    for source in manifest.get("domains", []):
        splits = source.get("split", {})
        train = _validated_manifest_split(splits, "inner_train")
        validation = _validated_manifest_split(splits, "inner_validation")
        data_root = Path(str(source["data_root"]))
        if not data_root.is_absolute():
            data_root = project_root / data_root
        domains.append(
            {
                "id": str(source["id"]),
                "condition": str(source["condition"]),
                "scene": str(source["scene"]),
                "data_root": str(data_root.resolve()),
                "train_split": str(Path(train["csv"]).resolve()),
                "validation_split": str(Path(validation["csv"]).resolve()),
                "train_csv_sha256": _sha256_file(train["csv"]),
                "validation_csv_sha256": _sha256_file(validation["csv"]),
                "train_sample_count": int(train["row_count"]),
                "validation_sample_count": int(validation["row_count"]),
            }
        )
    if not domains:
        raise ValueError("Clean inner-development protocol requires at least one domain.")
    payload = {
        "schema_version": 1,
        "mode": CLEAN_PROTOCOL_MODE,
        "protocol_id": CLEAN_PROTOCOL_ID,
        "source_protocol_id": str(manifest.get("protocol_id", "")),
        "source_protocol_manifest": str(manifest_path),
        "source_protocol_manifest_sha256": _sha256_file(manifest_path),
        "outer_test_enabled": False,
        "allow_confirmation_train": False,
        "train_role": "inner_train",
        "validation_role": "inner_validation",
        "domains": domains,
    }
    payload["protocol_fingerprint"] = _fingerprint(payload)
    return validate_clean_inner_protocol(payload)


def write_clean_inner_protocol(protocol: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(dict(protocol), sort_keys=False), encoding="utf-8")
    return target


def load_clean_inner_protocol(path: str | Path) -> dict[str, Any]:
    protocol = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(protocol, dict):
        raise ValueError("Clean inner-development protocol must be a mapping.")
    return validate_clean_inner_protocol(protocol)


def validate_clean_inner_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(protocol)
    if payload.get("mode") != CLEAN_PROTOCOL_MODE:
        raise ValueError(f"Clean protocol mode must be {CLEAN_PROTOCOL_MODE!r}.")
    if payload.get("protocol_id") != CLEAN_PROTOCOL_ID:
        raise ValueError(f"Clean protocol id must be {CLEAN_PROTOCOL_ID!r}.")
    if bool(payload.get("outer_test_enabled", True)):
        raise ValueError("Clean inner-development protocol must keep outer test disabled.")
    if bool(payload.get("allow_confirmation_train", True)):
        raise ValueError("Clean inner-development protocol must disallow confirmation train.")
    if payload.get("train_role") != "inner_train" or payload.get("validation_role") != "inner_validation":
        raise ValueError("Clean protocol must use inner_train and inner_validation roles.")
    domains = payload.get("domains")
    if not isinstance(domains, list) or not domains:
        raise ValueError("Clean inner-development protocol domains must be a non-empty list.")
    seen: set[str] = set()
    for domain in domains:
        if not isinstance(domain, dict):
            raise ValueError("Each clean protocol domain must be a mapping.")
        domain_id = str(domain.get("id", "")).strip()
        if not domain_id or domain_id in seen:
            raise ValueError(f"Clean protocol domain id is empty or duplicated: {domain_id!r}.")
        seen.add(domain_id)
        for role in ("train_split", "validation_split"):
            value = str(domain.get(role, "")).strip()
            if not value:
                raise ValueError(f"Clean protocol domain {domain_id} is missing {role}.")
            if not Path(value).is_file():
                raise FileNotFoundError(f"Clean protocol domain {domain_id} {role} is missing: {value}")
        if Path(domain["train_split"]).resolve() == Path(domain["validation_split"]).resolve():
            raise ValueError(f"Clean protocol domain {domain_id} reuses one CSV for train and validation.")
        _reject_forbidden_train_path(str(domain["train_split"]))
        for role in ("train", "validation"):
            actual = _sha256_file(domain[f"{role}_split"])
            expected = str(domain.get(f"{role}_csv_sha256", ""))
            if actual != expected:
                raise ValueError(f"Clean protocol domain {domain_id} {role} CSV SHA256 mismatch.")
    expected_fingerprint = str(payload.pop("protocol_fingerprint", ""))
    actual_fingerprint = _fingerprint(payload)
    if expected_fingerprint and expected_fingerprint != actual_fingerprint:
        raise ValueError("Clean inner-development protocol fingerprint mismatch.")
    payload["protocol_fingerprint"] = actual_fingerprint
    return payload


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


def audit_split_isolation(
    train_csv: str | Path,
    validation_csv: str | Path,
    outer_test_csv: str | Path | None = None,
    fail_closed: bool = True,
) -> dict[str, Any]:
    """Audit one train/validation pair without ever opening outer test."""
    if outer_test_csv is not None:
        raise ValueError("Clean inner-development audit refuses outer-test access.")
    train_path, validation_path = Path(train_csv).resolve(), Path(validation_csv).resolve()
    _reject_forbidden_train_path(str(train_path))
    train_fields, train_rows = _read_csv(train_path)
    validation_fields, validation_rows = _read_csv(validation_path)
    missing = {
        role: [field for field in ("sample_id", "target_sample_id", "window_frame_ids_json") if field not in fields]
        for role, fields in (("train", train_fields), ("validation", validation_fields))
    }
    missing = {role: fields for role, fields in missing.items() if fields}

    train_resources = _resource_identities(train_fields, train_rows)
    validation_resources = _resource_identities(validation_fields, validation_rows)
    unavailable_resources = [
        family
        for family in _RESOURCE_PATTERNS
        if not any(_RESOURCE_PATTERNS[family].fullmatch(field) for field in train_fields)
        or not any(_RESOURCE_PATTERNS[family].fullmatch(field) for field in validation_fields)
    ]
    overlaps = {
        "sample_id": _overlap(_values(train_rows, "sample_id"), _values(validation_rows, "sample_id")),
        "target_sample_id": _overlap(
            _values(train_rows, "target_sample_id"), _values(validation_rows, "target_sample_id")
        ),
        "full_csv_row": _overlap(_row_identities(train_rows), _row_identities(validation_rows)),
        "raw_input_resource": _overlap(
            set().union(*train_resources.values()), set().union(*validation_resources.values())
        ),
        "window_frame": _overlap(_window_frames(train_rows), _window_frames(validation_rows)),
        "target_frame": _overlap(_target_frames(train_rows), _target_frames(validation_rows)),
    }
    resource_overlaps = {
        family: _overlap(train_resources[family], validation_resources[family]) for family in _RESOURCE_PATTERNS
    }
    sequence_checks = {
        field: _optional_identity_check(train_fields, validation_fields, train_rows, validation_rows, field)
        for field in ("sequence_id", "trajectory_id")
    }
    reasons = []
    if missing:
        reasons.append("missing_required_identity_fields")
    if unavailable_resources:
        reasons.append("missing_required_resource_fields")
    reasons.extend(f"{name}_overlap" for name in _REQUIRED_ZERO_OVERLAPS if overlaps[name]["count"])
    result = {
        "schema_version": 1,
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "train_csv": str(train_path),
        "validation_csv": str(validation_path),
        "train_csv_sha256": _sha256_file(train_path),
        "validation_csv_sha256": _sha256_file(validation_path),
        "train_sample_count": len(train_rows),
        "validation_sample_count": len(validation_rows),
        "train_beam_counts": _beam_counts(train_rows),
        "validation_beam_counts": _beam_counts(validation_rows),
        "missing_required_fields": missing,
        "unavailable_resource_families": unavailable_resources,
        "overlaps": overlaps,
        "resource_overlaps": resource_overlaps,
        "sequence_identity_checks": sequence_checks,
    }
    if fail_closed and result["status"] != "passed":
        raise ValueError(f"Train/validation split isolation audit failed: {', '.join(reasons)}")
    return result


def audit_clean_inner_protocol(protocol_path: str | Path, *, fail_closed: bool = True) -> dict[str, Any]:
    path = Path(protocol_path).resolve()
    protocol = load_clean_inner_protocol(path)
    domain_results = []
    pair_results = []
    for train_domain in protocol["domains"]:
        for validation_domain in protocol["domains"]:
            result = audit_split_isolation(
                train_domain["train_split"],
                validation_domain["validation_split"],
                fail_closed=False,
            )
            result["train_domain_id"] = train_domain["id"]
            result["validation_domain_id"] = validation_domain["id"]
            pair_results.append(result)
            if train_domain["id"] == validation_domain["id"]:
                domain_results.append(result)
    overlap_counts = {
        name: sum(int(result["overlaps"][name]["count"]) for result in pair_results)
        for name in _REQUIRED_ZERO_OVERLAPS
    }
    failed_domains = [result["train_domain_id"] for result in domain_results if result["status"] != "passed"]
    failed_pairs = [
        f"{result['train_domain_id']}->{result['validation_domain_id']}"
        for result in pair_results
        if result["status"] != "passed"
    ]
    audit = {
        "schema_version": 1,
        "audit_id": "mmw_clean_inner_split_isolation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not failed_pairs else "failed",
        "protocol_path": str(path),
        "protocol_file_sha256": _sha256_file(path),
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "source_protocol_manifest": protocol["source_protocol_manifest"],
        "source_protocol_manifest_sha256": protocol["source_protocol_manifest_sha256"],
        "outer_test_accessed": False,
        "domain_count": len(domain_results),
        "pair_count": len(pair_results),
        "train_sample_count": sum(int(result["train_sample_count"]) for result in domain_results),
        "validation_sample_count": sum(int(result["validation_sample_count"]) for result in domain_results),
        "train_sample_id_hash": _sample_id_hash(protocol, "train_split"),
        "validation_sample_id_hash": _sample_id_hash(protocol, "validation_split"),
        "overlap_counts": overlap_counts,
        "failed_domains": failed_domains,
        "failed_pairs": failed_pairs,
        "domains": domain_results,
        "pairs": pair_results,
    }
    if fail_closed and audit["status"] != "passed":
        raise ValueError(f"Clean protocol split isolation failed for pairs: {failed_pairs}")
    return audit


def write_clean_split_audit(audit: Mapping[str, Any], json_path: str | Path, markdown_path: str | Path) -> None:
    json_target, markdown_target = Path(json_path), Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(dict(audit), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Clean Split Isolation Audit",
        "",
        f"- Status: `{audit['status']}`",
        f"- Domains: {audit['domain_count']}",
        f"- Train samples: {audit['train_sample_count']}",
        f"- Validation samples: {audit['validation_sample_count']}",
        f"- Outer test accessed: `{str(audit['outer_test_accessed']).lower()}`",
        f"- Protocol manifest SHA256: `{audit['source_protocol_manifest_sha256']}`",
        "",
        "## Required overlap counts",
        "",
        "| Identity | Count |",
        "|---|---:|",
        *[f"| {name} | {count} |" for name, count in audit["overlap_counts"].items()],
        "",
        "## Domains",
        "",
        "| Domain | Train | Validation | Status |",
        "|---|---:|---:|---|",
        *[
            f"| {item['domain_id']} | {item['train_sample_count']} | {item['validation_sample_count']} | {item['status']} |"
            for item in audit["domains"]
        ],
        "",
    ]
    markdown_target.write_text("\n".join(lines), encoding="utf-8")


def validate_clean_config_protocol(cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    dataset = cfg.get("data", {}).get("dataset", {})
    dataset_type = str(dataset.get("type", "")).strip().lower() if isinstance(dataset, Mapping) else ""
    if dataset_type != "mmw":
        return None
    section = cfg.get("data_protocol")
    if section is None:
        raise ValueError("MMW training requires a clean_inner_development data_protocol.")
    if not isinstance(section, dict) or section.get("mode") != CLEAN_PROTOCOL_MODE:
        raise ValueError("data_protocol must declare mode=clean_inner_development.")
    protocol_path = Path(str(section.get("path", ""))).resolve()
    protocol = load_clean_inner_protocol(protocol_path)
    if (
        section.get("protocol_id") != protocol["protocol_id"]
        or section.get("protocol_fingerprint") != protocol["protocol_fingerprint"]
    ):
        raise ValueError("MMW clean config must bind the exact clean protocol identity.")
    if section.get("train_role") != "inner_train" or section.get("validation_role") != "inner_validation":
        raise ValueError("MMW clean config must use inner_train and inner_validation roles.")
    if section.get("outer_test_enabled") is not False or section.get("allow_confirmation_train") is not False:
        raise ValueError("MMW clean config must explicitly disable outer test and confirmation training.")
    expected_domains = protocol_dataset_domains(protocol)
    actual_domains = cfg.get("data", {}).get("dataset", {}).get("domains")
    if not isinstance(actual_domains, list) or actual_domains != expected_domains:
        raise ValueError("Resolved dataset domains do not exactly match the clean protocol.")
    if any(domain.get("test_csv_name") for domain in actual_domains):
        raise ValueError("Clean inner-development config must not carry outer/test CSV paths.")
    final_test = cfg.get("training", {}).get("final_test")
    enabled = final_test if isinstance(final_test, bool) else (final_test or {}).get("enabled", True)
    if bool(enabled):
        raise ValueError("Clean inner-development config must explicitly disable final test.")
    audit = audit_clean_inner_protocol(protocol_path, fail_closed=True)
    report_path = Path(str(section.get("audit_report", ""))).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required_audit_fields = (
        "audit_id",
        "protocol_file_sha256",
        "protocol_fingerprint",
        "train_sample_id_hash",
        "validation_sample_id_hash",
        "pair_count",
        "overlap_counts",
        "failed_pairs",
    )
    if report.get("status") != "passed" or report.get("outer_test_accessed") is not False or any(
        report.get(key) != audit[key] for key in required_audit_fields
    ):
        raise ValueError("Clean split audit report is missing, failed, or does not match the protocol file.")
    return audit


def _validated_manifest_split(splits: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    value = splits.get(role)
    if not isinstance(value, dict) or not value.get("csv"):
        raise ValueError(f"Source protocol manifest is missing {role} split metadata.")
    actual = _sha256_file(value["csv"])
    if actual != str(value.get("sha256", "")):
        raise ValueError(f"Source protocol manifest {role} CSV SHA256 mismatch.")
    return value


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [{key: str(value or "") for key, value in row.items()} for row in reader]
    if not fields or not rows:
        raise ValueError(f"Split CSV must contain a header and at least one row: {path}")
    return fields, rows


def _values(rows: Iterable[Mapping[str, str]], field: str) -> set[str]:
    return {value for row in rows if (value := str(row.get(field, "")).strip())}


def _row_identities(rows: Iterable[Mapping[str, str]]) -> set[str]:
    return {_fingerprint(dict(row)) for row in rows}


def _resource_identities(fields: Iterable[str], rows: Iterable[Mapping[str, str]]) -> dict[str, set[str]]:
    columns = {family: [field for field in fields if pattern.fullmatch(field)] for family, pattern in _RESOURCE_PATTERNS.items()}
    return {
        family: {
            value
            for row in rows
            for field in family_fields
            if (value := str(row.get(field, "")).strip()) and value not in {"-99", "-99.0"}
        }
        for family, family_fields in columns.items()
    }


def _window_frames(rows: Iterable[Mapping[str, str]]) -> set[str]:
    result = set()
    for row in rows:
        segment = str(row.get("contiguous_segment_id", "")).strip()
        for frame in _json_list(row.get("window_frame_ids_json")):
            result.add(f"{segment}:{frame}")
    return result


def _target_frames(rows: Iterable[Mapping[str, str]]) -> set[str]:
    result = set()
    for row in rows:
        segment = str(row.get("contiguous_segment_id", "")).strip()
        frames = _json_list(row.get("future_frame_ids_json"))
        if frames:
            result.update(f"{segment}:{frame}" for frame in frames)
        elif target := str(row.get("target_sample_id", "")).strip():
            result.add(target)
    return result


def _json_list(value: Any) -> list[str]:
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _optional_identity_check(
    train_fields: list[str],
    validation_fields: list[str],
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    field: str,
) -> dict[str, Any]:
    if field not in train_fields or field not in validation_fields:
        return {"status": "unavailable", "overlap_count": None, "overlap_examples": []}
    overlap = _overlap(_values(train_rows, field), _values(validation_rows, field))
    return {"status": "checked", "overlap_count": overlap["count"], "overlap_examples": overlap["examples"]}


def _beam_counts(rows: Iterable[Mapping[str, str]]) -> dict[str, int]:
    counts = Counter(str(row.get("future_beam_label1", row.get("target_label", ""))).strip() for row in rows)
    counts.pop("", None)
    return dict(sorted(counts.items(), key=lambda item: int(float(item[0]))))


def _sample_id_hash(protocol: Mapping[str, Any], role: str) -> str:
    values = []
    for domain in protocol["domains"]:
        _fields, rows = _read_csv(Path(domain[role]))
        values.extend(f"{domain['id']}:{value}" for value in _values(rows, "sample_id"))
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def _overlap(left: set[str], right: set[str]) -> dict[str, Any]:
    values = sorted(left & right)
    return {"count": len(values), "examples": values[:10]}


def _reject_forbidden_train_path(value: str) -> None:
    lowered = value.lower()
    token = next((token for token in FORBIDDEN_TRAIN_PATH_TOKENS if token in lowered), None)
    if token:
        raise ValueError(f"Clean inner-development protocol rejects forbidden path token {token!r}: {value}")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "CLEAN_PROTOCOL_ID",
    "CLEAN_PROTOCOL_MODE",
    "audit_clean_inner_protocol",
    "audit_split_isolation",
    "build_clean_inner_protocol",
    "load_clean_inner_protocol",
    "protocol_dataset_domains",
    "validate_clean_config_protocol",
    "validate_clean_inner_protocol",
    "write_clean_inner_protocol",
    "write_clean_split_audit",
]
