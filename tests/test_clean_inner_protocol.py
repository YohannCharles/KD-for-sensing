import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from kd_sensing.data.mmw.clean_protocol import (
    CLEAN_PROTOCOL_ID,
    audit_clean_inner_protocol,
    audit_split_isolation,
    load_clean_inner_protocol,
    protocol_dataset_domains,
    validate_clean_config_protocol,
    validate_clean_inner_protocol,
)


FIELDS = [
    "sample_id",
    "target_sample_id",
    "contiguous_segment_id",
    "window_frame_ids_json",
    "future_frame_ids_json",
    "camera1",
    "lidar1",
    "radar1",
    "gps1",
    "bs_gps1",
    "future_beam_label1",
]


def _write_split(path: Path, *, prefix: str, frame: int) -> None:
    row = {
        "sample_id": f"{prefix}-sample",
        "target_sample_id": f"{prefix}-target",
        "contiguous_segment_id": f"{prefix}-trajectory",
        "window_frame_ids_json": json.dumps([frame, frame + 1]),
        "future_frame_ids_json": json.dumps([frame + 1]),
        "camera1": f"{prefix}/camera.png",
        "lidar1": f"{prefix}/lidar.pcd",
        "radar1": f"{prefix}/radar.npy",
        "gps1": f"{prefix}/gps.yaml",
        "bs_gps1": f"{prefix}/bs.yaml",
        "future_beam_label1": "3",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)


def _protocol(domains: list[dict[str, str]]) -> dict:
    return {
        "schema_version": 1,
        "mode": "clean_inner_development",
        "protocol_id": CLEAN_PROTOCOL_ID,
        "source_protocol_id": "test-source",
        "source_protocol_manifest": "test-source.json",
        "source_protocol_manifest_sha256": "0" * 64,
        "outer_test_enabled": False,
        "allow_confirmation_train": False,
        "train_role": "inner_train",
        "validation_role": "inner_validation",
        "domains": domains,
    }


def _domain(domain_id: str, train: Path, validation: Path) -> dict[str, str]:
    return {
        "id": domain_id,
        "condition": "sunny",
        "scene": domain_id,
        "data_root": str(train.parent),
        "train_split": str(train),
        "validation_split": str(validation),
        "train_csv_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
        "validation_csv_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
    }


def test_audit_split_isolation_passes_and_marks_unavailable_identity(tmp_path: Path) -> None:
    train, validation = tmp_path / "inner_train.csv", tmp_path / "inner_validation.csv"
    _write_split(train, prefix="train", frame=1)
    _write_split(validation, prefix="validation", frame=20)

    audit = audit_split_isolation(train, validation)

    assert audit["status"] == "passed"
    assert all(item["count"] == 0 for item in audit["overlaps"].values())
    assert audit["sequence_identity_checks"]["sequence_id"]["status"] == "unavailable"


@pytest.mark.parametrize("overlap", ["sample", "target", "row", "resource", "frame"])
def test_audit_split_isolation_fails_closed_on_overlap(tmp_path: Path, overlap: str) -> None:
    train, validation = tmp_path / "inner_train.csv", tmp_path / "inner_validation.csv"
    _write_split(train, prefix="train", frame=1)
    _write_split(validation, prefix="validation", frame=20)
    train_row = next(csv.DictReader(train.open(newline="", encoding="utf-8")))
    validation_row = next(csv.DictReader(validation.open(newline="", encoding="utf-8")))
    if overlap == "sample":
        validation_row["sample_id"] = train_row["sample_id"]
    elif overlap == "target":
        validation_row["target_sample_id"] = train_row["target_sample_id"]
    elif overlap == "row":
        validation_row = train_row
    elif overlap == "resource":
        validation_row["camera1"] = train_row["camera1"]
    else:
        validation_row["window_frame_ids_json"] = train_row["window_frame_ids_json"]
        validation_row["contiguous_segment_id"] = train_row["contiguous_segment_id"]
    with validation.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(validation_row)

    with pytest.raises(ValueError, match="split isolation audit failed"):
        audit_split_isolation(train, validation)


def test_clean_protocol_rejects_confirmation_train(tmp_path: Path) -> None:
    leaked = tmp_path / "confirmation_train_splits" / "train.csv"
    leaked.parent.mkdir()
    validation = tmp_path / "inner_validation.csv"
    _write_split(leaked, prefix="train", frame=1)
    _write_split(validation, prefix="validation", frame=20)

    with pytest.raises(ValueError, match="confirmation_train_splits"):
        validate_clean_inner_protocol(_protocol([_domain("domain", leaked, validation)]))


def test_clean_protocol_audits_cross_domain_train_validation_overlap(tmp_path: Path) -> None:
    train_a, validation_a = tmp_path / "a_train.csv", tmp_path / "a_validation.csv"
    train_b, validation_b = tmp_path / "b_train.csv", tmp_path / "b_validation.csv"
    _write_split(train_a, prefix="shared", frame=1)
    _write_split(validation_a, prefix="a-validation", frame=20)
    _write_split(train_b, prefix="b-train", frame=40)
    _write_split(validation_b, prefix="shared", frame=1)
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(
        yaml.safe_dump(_protocol([_domain("a", train_a, validation_a), _domain("b", train_b, validation_b)])),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pairs"):
        audit_clean_inner_protocol(protocol_path)


def test_clean_config_requires_exact_protocol_identity_and_full_audit(tmp_path: Path) -> None:
    train, validation = tmp_path / "inner_train.csv", tmp_path / "inner_validation.csv"
    _write_split(train, prefix="train", frame=1)
    _write_split(validation, prefix="validation", frame=20)
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(_protocol([_domain("domain", train, validation)])), encoding="utf-8")
    protocol = load_clean_inner_protocol(protocol_path)
    audit = audit_clean_inner_protocol(protocol_path)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    cfg = {
        "data": {"dataset": {"type": "mmw", "domains": protocol_dataset_domains(protocol)}},
        "data_protocol": {
            "mode": "clean_inner_development",
            "path": str(protocol_path),
            "audit_report": str(audit_path),
            "protocol_id": protocol["protocol_id"],
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "train_role": "inner_train",
            "validation_role": "inner_validation",
            "outer_test_enabled": False,
            "allow_confirmation_train": False,
        },
        "training": {"final_test": {"enabled": False}},
    }

    assert validate_clean_config_protocol(cfg)["status"] == "passed"
    cfg["data_protocol"]["protocol_fingerprint"] = "tampered"
    with pytest.raises(ValueError, match="exact clean protocol identity"):
        validate_clean_config_protocol(cfg)
