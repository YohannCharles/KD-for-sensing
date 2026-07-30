import json
from pathlib import Path

import pytest

from tools.run_pcpf import (
    _checkpoint_protocol_fingerprint,
    _load_template,
    _validate_trajectory_audit,
)


TRAJECTORY_FINGERPRINT = "d" * 64


def _protocol() -> dict[str, object]:
    return {
        "protocol_fingerprint": TRAJECTORY_FINGERPRINT,
        "train_window_count": 37_510,
        "validation_window_count": 6_365,
        "test_window_count": 2_985,
    }


def _audit() -> dict[str, object]:
    return {
        "audit_id": "trajectory-audit",
        "status": "passed",
        "outer_test_accessed": False,
        "protocol_fingerprint": TRAJECTORY_FINGERPRINT,
        "train_sample_count": 37_510,
        "validation_sample_count": 6_365,
        "test_sample_count": 2_985,
        "pairwise_overlaps": {
            pair: {"channel_resource": {"count": 0, "examples": []}}
            for pair in ("train_vs_validation", "train_vs_test", "validation_vs_test")
        },
    }


def test_trajectory_audit_requires_exact_counts_and_zero_overlap(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(_audit()), encoding="utf-8")

    validated = _validate_trajectory_audit(path, _protocol())

    assert validated["train_sample_count"] == 37_510
    assert validated["validation_sample_count"] == 6_365
    assert validated["outer_test_accessed"] is False

    failed = _audit()
    failed["pairwise_overlaps"]["train_vs_validation"]["channel_resource"]["count"] = 1
    path.write_text(json.dumps(failed), encoding="utf-8")
    with pytest.raises(ValueError, match="zero resource overlap"):
        _validate_trajectory_audit(path, _protocol())


def test_checkpoint_protocol_fingerprint_supports_formal_and_smoke_payloads() -> None:
    assert _checkpoint_protocol_fingerprint(
        {"resume_contract": {"config": {"data_protocol": {"protocol_fingerprint": TRAJECTORY_FINGERPRINT}}}}
    ) == TRAJECTORY_FINGERPRINT
    assert _checkpoint_protocol_fingerprint(
        {"data_protocol": {"protocol_fingerprint": TRAJECTORY_FINGERPRINT}}
    ) == TRAJECTORY_FINGERPRINT
    assert _checkpoint_protocol_fingerprint({}) is None


def test_sparse_trajectory_stage1_template_is_fresh_start() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = _load_template(root / "tools/configs/pcpf/sparse_csi/stage1.yaml")

    assert cfg["training"]["initialization_checkpoint"] is False
    assert "historical_reference" not in cfg["evaluation"]["pcpf_diagnostics"]
    assert cfg["data"]["dataloader"]["train_batch_size"] == 64
    assert cfg["data"]["dataloader"]["validation_batch_size"] == 64
    assert cfg["data"]["dataloader"]["num_workers"] == 8
