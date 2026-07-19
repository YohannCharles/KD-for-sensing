import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_summary(monkeypatch: pytest.MonkeyPatch):
    path = ROOT / "scripts" / "summarize_mmw_all_weather_matrix.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rows(method: str, *, profile: str = "64_beam_circular_topk_progressive_top3_dba_v1") -> list[dict[str, str]]:
    return [
        {
            "method": method,
            "seed": "1",
            "domain_id": f"domain-{index}",
            "condition": "sunny",
            "scene": f"scene-{index}",
            "sample_csv_sha256": f"samples-{index}",
            "checkpoint_sha256": f"checkpoint-{method}",
            "checkpoint_role": "fixed_epoch_last_pth",
            "training_profile_id": "umask_h4_v1",
            "training_profile_sha256": "profile",
            "router_architecture_profile_id": "umask_router_nopattern_v1",
            "router_architecture_profile_sha256": "router-profile",
            "design_candidate_id": "",
            "design_config_sha256": "",
            "metric_profile": profile,
            "coverage_status": "complete",
            "partial_request": "False",
            "sample_count": "10",
            "expected_sample_count": "10",
            "expected_domain_count": "15",
            "eval_family": "temporal_missing",
            "pattern": "modality_frame",
            "missing_rate": "0.2",
            "available_modalities": "image,radar,gps,lidar",
            "mask_digest": "mask-1",
            "mask_cache_checksum": "cache-1",
        }
        for index in range(15)
    ]


def test_summary_requires_complete_unique_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = _load_summary(monkeypatch)
    rows = _rows("T2")

    summary._validate_complete_evidence(rows)

    partial = deepcopy(rows)
    partial[0]["coverage_status"] = "partial"
    with pytest.raises(ValueError, match="partial evaluation"):
        summary._validate_complete_evidence(partial)

    duplicate = rows + [deepcopy(rows[0])]
    with pytest.raises(ValueError, match="duplicate evidence"):
        summary._validate_complete_evidence(duplicate)


def test_summary_rejects_metric_profile_conflicts(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = _load_summary(monkeypatch)
    rows = [*_rows("T2"), *_rows("S1", profile="64_beam_linear_topk_progressive_top3_dba_v1")]

    with pytest.raises(ValueError, match="conflicting metric profiles"):
        summary._validate_complete_evidence(rows)
